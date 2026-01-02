# -*- coding: utf-8 -*-

import os
import json
import base64
import requests
import re
import datetime
import shutil
import jieba
import numpy as np
from typing import List, Dict, Any, Tuple

# 第三方套件
import fitz  # pip install PyMuPDF
import pymupdf4llm # pip install pymupdf4llm
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from neo4j import GraphDatabase
from duckduckgo_search import DDGS

# ==================== 配置區 (Configuration) ====================

class Config:
    # LLM & VLM API
    TEXT_API_URL = ""
    IMAGE_API_URL = ""
    
    # Paths
    BASE_DIR = "D:\\document"
    IMAGE_DIR = "pdf_images"
    DB_PATH = "rag_final_data.json" # 用於備份的中間檔
    
    # ChromaDB
    CHROMA_HOST = "localhost"
    CHROMA_PORT = 8000
    CHROMA_COLLECTION = "integrated_rag_system_v0"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    # Neo4j
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password123"
    
    # Reranker
    RERANK_MODEL = "BAAI/bge-reranker-v2-m3" # 若無 GPU 可改 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# ==================== 工具類別 (Helpers) ====================

class LLMClient:
    """統一處理 LLM 與 VLM 的呼叫"""
    
    @staticmethod
    def query_text(prompt: str) -> str:
        try:
            resp = requests.post(Config.TEXT_API_URL, json={"question": prompt}, timeout=60)
            resp.raise_for_status()
            return resp.json().get('text', '').strip()
        except Exception as e:
            print(f"⚠️ [LLM Error] {e}")
            return ""

    @staticmethod
    def query_image(prompt: str, image_path: str) -> str:
        try:
            if not os.path.exists(image_path): return ""
            
            # Encode image
            mime_type = "image/png"
            if image_path.lower().endswith(".jpg"): mime_type = "image/jpeg"
            
            with open(image_path, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode('utf-8')
                data_uri = f"data:{mime_type};base64,{b64_str}"
            
            payload = {
                "question": prompt,
                "uploads": [{
                    "data": data_uri,
                    "type": "file",
                    "name": os.path.basename(image_path),
                    "mime": mime_type
                }]
            }
            resp = requests.post(Config.IMAGE_API_URL, json=payload, timeout=90)
            return resp.json().get('text', '').strip()
        except Exception as e:
            print(f"⚠️ [VLM Error] {e}")
            return ""

# ==================== 入庫流程 (Ingestion Pipeline) ====================

class RAGIngestor:
    def __init__(self):
        # 初始化 Chroma
        self.emb_fn = SentenceTransformerEmbeddingFunction(model_name=Config.EMBEDDING_MODEL)
        self.chroma_client = chromadb.HttpClient(host=Config.CHROMA_HOST, port=Config.CHROMA_PORT)
        self.collection = self.chroma_client.get_or_create_collection(
            name=Config.CHROMA_COLLECTION, embedding_function=self.emb_fn
        )
        
        # 初始化 Neo4j
        try:
            self.driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
            self.driver.verify_connectivity()
        except:
            print("⚠️ Neo4j 連線失敗，GraphRAG 功能將停用")
            self.driver = None

    def process_pdf(self, pdf_path: str):
        """Phase 1: PDF 結構化處理 (含 VLM)"""
        print(f"\n[1/4] Processing PDF: {os.path.basename(pdf_path)}")
        
        # 準備圖片目錄
        img_output = pdf_path.replace(".pdf", "_images")
        if not os.path.exists(img_output): os.makedirs(img_output)
        
        # 1. 轉 Markdown (含表格還原)
        md_text = pymupdf4llm.to_markdown(pdf_path, write_images=True, image_path=img_output)
        
        # 2. 圖片語意增強
        lines = md_text.split('\n')
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if line.strip().startswith("![](") and line.strip().endswith(")"):
                img_path = line.strip()[4:-1]
                if os.path.exists(img_path):
                    print(f"   -> Analyzing image: {os.path.basename(img_path)}")
                    caption = LLMClient.query_image("詳細描述這張圖片的內容，包含圖表趨勢或表格數據。", img_path)
                    new_lines.append(f"\n> **[AI Image Analysis]**: {caption}\n")
        
        return "\n".join(new_lines)

    def _semantic_split(self, text: str, breakpoint_percentile=85) -> List[str]:
        """
        [Phase 1 優化] 語意切分器 (Semantic Splitter)
        不使用固定字數，而是根據「語意相似度變化」來決定切分點。
        """
        # 1. 簡單分句 (處理中英文句點)
        single_sentences = re.split(r'(?<=[。！？.!?])\s+', text)
        single_sentences = [s for s in single_sentences if s.strip()]
        
        if len(single_sentences) < 2:
            return [text]

        # 2. 計算每個句子的 Embedding
        try:
            embeddings = self.emb_fn(single_sentences)
        except Exception as e:
            print(f"      [Semantic Error] Embedding 失敗: {e}，回退至純文字")
            return [text]
        
        # 3. 計算相鄰句子的 Cosine Distance
        distances = []
        for i in range(len(embeddings) - 1):
            v1 = np.array(embeddings[i])
            v2 = np.array(embeddings[i+1])
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            distances.append(1 - sim)

        # 4. 決定切分閾值 (使用百分位數)
        if not distances: return [text]
        threshold = np.percentile(distances, breakpoint_percentile)

        # 5. 組合 Chunks
        chunks = []
        current_chunk = ""
        for i, sentence in enumerate(single_sentences):
            current_chunk += sentence
            if i < len(distances) and distances[i] > threshold:
                if len(current_chunk) > 50: # 避免太碎
                    chunks.append(current_chunk)
                    current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def create_chunks(self, md_text: str, source_name: str) -> List[Dict]:
        """Phase 1: Small-to-Big 切分 (升級版：Semantic Child Chunking)"""
        print("[2/4] Chunking (Parent: Structure, Child: Semantic)...")
        
        # Parent Splitting (按章節)
        headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False)
        parents = parent_splitter.split_text(md_text)
        
        chunks_data = []
        for i, p_doc in enumerate(parents):
            p_id = f"{source_name}_p{i}"
            
            # 使用語意切分代替 RecursiveCharacterTextSplitter
            children = self._semantic_split(p_doc.page_content, breakpoint_percentile=85)
            
            for j, c_text in enumerate(children):
                chunks_data.append({
                    "id": f"{p_id}_c{j}",
                    "parent_id": p_id,
                    "text_embedding": c_text,  # 語意聚合的小區塊
                    "text_llm": p_doc.page_content, # 完整章節
                    "metadata": {**p_doc.metadata, "source": source_name, "parent_id": p_id}
                })
        print(f"   -> Generated {len(chunks_data)} semantic chunks from {len(parents)} sections.")
        return chunks_data

    def augment_and_extract_graph(self, chunks: List[Dict]):
        """Phase 2 & 3: Document Augmentation + Graph Extraction"""
        print(f"[3/4] Augmentation & Graph Extraction ({len(chunks)} chunks)...")
        
        for chunk in chunks:
            # Phase 2: Document Augmentation (生成假設性問題)
            # 為了省時，這裡只對前幾個或特定條件的 chunk 做示範，實務上可全做
            prompt_aug = f"針對以下內容，生成3個使用者可能會問的問題：\n{chunk['text_embedding'][:500]}"
            questions = LLMClient.query_text(prompt_aug)
            chunk['text_embedding'] += f"\n\n[Hypothetical Questions]:\n{questions}"
            chunk['metadata']['aug_questions'] = questions

            # Phase 3: Graph Extraction (Neo4j)
            if self.driver:
                prompt_kg = f"提取以下文本中的實體(Entity)與關係(Relation)，輸出JSON格式：{{'entities':[], 'relations':[]}}。\n文本：{chunk['text_embedding'][:1000]}"
                kg_json = LLMClient.query_text(prompt_kg)
                self._save_to_neo4j(kg_json, chunk)

    def _save_to_neo4j(self, kg_text: str, chunk: Dict):
        """解析 JSON 並寫入 Neo4j"""
        try:
            # 嘗試提取 JSON
            match = re.search(r"\{{.*\}}", kg_text, re.DOTALL)
            if not match: return
            data = json.loads(match.group())
            
            with self.driver.session() as session:
                # 建立 Chunk 節點
                session.run("MERGE (c:Chunk {id: $id}) SET c.text = $text", 
                            id=chunk['id'], text=chunk['text_embedding'][:100])
                
                # 建立實體與關係
                for ent in data.get('entities', []):
                    session.run("MERGE (e:Entity {name: $name, type: $type}) MERGE (e)-[:MENTIONED_IN]->(c:Chunk {id: $cid})",
                                name=ent.get('id'), type=ent.get('type', 'Thing'), cid=chunk['id'])
                
                for rel in data.get('relations', []):
                    session.run("""
                        MATCH (a:Entity {name: $src}), (b:Entity {name: $tgt})
                        MERGE (a)-[:RELATED {type: $rtype}]->(b)
                    """, src=rel.get('source'), tgt=rel.get('target'), rtype=rel.get('type'))
        except Exception as e:
            # Graph 提取失敗不應阻擋流程
            pass

    def save_to_db(self, chunks: List[Dict]):
        """寫入 ChromaDB"""
        print("[4/4] Saving to ChromaDB...")
        ids = [c['id'] for c in chunks]
        docs = [c['text_embedding'] for c in chunks] # 包含增強後的問題
        metas = [c['metadata'] for c in chunks]
        
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
        
        # 保存一份完整的映射表 (Parent Content) 到本地，供 Retrieval 階段使用
        # 實務上這部分應該存 Redis 或 SQL，這裡用 JSON 模擬
        parent_map = {c['id']: c['text_llm'] for c in chunks}
        
        # 如果存在舊的映射表，則合併
        if os.path.exists("parent_map.json"):
            with open("parent_map.json", "r", encoding="utf-8") as f:
                old_map = json.load(f)
            parent_map.update(old_map)
            
        with open("parent_map.json", "w", encoding="utf-8") as f:
            json.dump(parent_map, f, ensure_ascii=False, indent=2)
            
        print("✅ Ingestion Complete!")

# ==================== 檢索流程 (Retrieval Engine) ====================

class RAGRetriever:
    def __init__(self):
        # 1. ChromaDB
        self.emb_fn = SentenceTransformerEmbeddingFunction(model_name=Config.EMBEDDING_MODEL)
        self.client = chromadb.HttpClient(host=Config.CHROMA_HOST, port=Config.CHROMA_PORT)
        self.collection = self.client.get_collection(name=Config.CHROMA_COLLECTION, embedding_function=self.emb_fn)
        
        # 2. BM25 (需從 Chroma 撈出所有資料來建立，或讀取暫存檔)
        print("Loading BM25 Index...")
        all_docs = self.collection.get() # 取得所有資料
        self.bm25_docs = all_docs['documents']
        self.bm25_ids = all_docs['ids']
        tokenized_corpus = [list(jieba.cut(doc)) for doc in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 3. Reranker
        print("Loading Reranker...")
        self.reranker = CrossEncoder(Config.RERANK_MODEL)
        
        # 4. Neo4j
        try:
            self.driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
        except: self.driver = None
        
        # 5. Parent Map (Small-to-Big)
        with open("parent_map.json", "r", encoding="utf-8") as f:
            self.parent_map = json.load(f)

    def query_transform(self, query: str) -> List[str]:
        """Phase 2: Query Transformation (多路查詢)"""
        print(f"   [Transform] Generating multi-queries for: {query}")
        prompt = f"請針對問題 '{query}' 生成 3 個不同切入點的搜尋關鍵字或問句，一行一個。"
        res = LLMClient.query_text(prompt)
        new_queries = [q.strip() for q in res.split('\n') if q.strip()]
        return [query] + new_queries[:3] # 包含原問題

    def search_graph(self, query: str) -> List[str]:
        """Phase 3: GraphRAG 檢索 (找出關聯實體)"""
        if not self.driver: return []
        
        # 簡單實作：用關鍵字去 Graph 找 Entity，再找相連的 Chunk
        # 實務上需先對 Query 做 NER
        keywords = list(jieba.cut(query))
        found_ids = []
        with self.driver.session() as session:
            for kw in keywords:
                if len(kw) < 2: continue
                # 找提及該關鍵字的 Chunk
                res = session.run("""
                    MATCH (e:Entity) WHERE e.name CONTAINS $kw
                    MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
                    RETURN c.id LIMIT 5
                """, kw=kw)
                found_ids.extend([record["c.id"] for record in res])
        return list(set(found_ids))
    
    def graph_guided_chunk_ids(self, query: str, limit=50) -> List[str]:
        """
        用 Query → Entity → Graph traversal
        回傳「Graph 認為可能相關的 Chunk IDs」
        """
        if not self.driver:
            return []
    
        # 1. 用 LLM 抽 Entity（最簡版）
        prompt = f"""
        從以下問題中抽取關鍵實體，JSON 輸出：
        {{ "entities": [] }}
        問題：{query}
        """
        res = LLMClient.query_text(prompt)
    
        try:
            data = json.loads(re.search(r"\{.*\}", res, re.S).group())
            entities = data.get("entities", [])
        except:
            return []
    
        if not entities or len(entities) > 10:
            return []
    
        # 2. Neo4j traversal（1~2 hop）
        cypher = """
        MATCH (e:Entity)
        WHERE e.name IN $entities
        MATCH (e)-[:RELATED*1..2]-(o:Entity)
        MATCH (o)-[:MENTIONED_IN]->(c:Chunk)
        RETURN DISTINCT c.id LIMIT $limit
        """
    
        with self.driver.session() as session:
            result = session.run(cypher, entities=entities, limit=limit)
            return [r["c.id"] for r in result]


    def retrieve(self, query: str, top_k=5) -> str:
        """主檢索流程"""
        print(f"\n--- Processing Query: {query} ---")
        
        # 0. Graph-guided candidate pruning
        graph_ids = self.graph_guided_chunk_ids(query)
        
        use_graph_filter = len(graph_ids) >= 5  # 避免 Graph 太小誤傷
        
        # 1. Query Transformation
        queries = self.query_transform(query)
        
        # 2. Hybrid Search (Fusion)
        fused_scores = {}
        
        for q in queries:
            # A. Vector Search
            if use_graph_filter:
                v_res = self.collection.query(
                    query_texts=[q],
                    n_results=20,
                    where={"$id": {"$in": graph_ids}}
                )
            else:
                v_res = self.collection.query(
                    query_texts=[q],
                    n_results=20
                )
            for i, vid in enumerate(v_res['ids'][0]):
                fused_scores[vid] = fused_scores.get(vid, 0) + 1/(60+i)
            
            # B. BM25 Search
            b_docs = self.bm25.get_top_n(list(jieba.cut(q)), self.bm25_docs, n=20)
            # for i, doc in enumerate(b_docs):
            #     # 反查 ID (簡單做)
            #     try:
            #         idx = self.bm25_docs.index(doc)
            #         bid = self.bm25_ids[idx]
            #         fused_scores[bid] = fused_scores.get(bid, 0) + 1/(60+i)
            #     except:
            #         pass
            for i, doc in enumerate(b_docs):
                try:
                    idx = self.bm25_docs.index(doc)
                    bid = self.bm25_ids[idx]
            
                    if use_graph_filter and bid not in graph_ids:
                        continue   # ⭐ 關鍵一行
            
                    fused_scores[bid] = fused_scores.get(bid, 0) + 1/(60+i)
                except:
                    pass
                
        # # C. Graph Search (補充)
        # g_ids = self.search_graph(query)
        # for gid in g_ids:
        #     fused_scores[gid] = fused_scores.get(gid, 0) + 0.05 # Graph 權重加分
            
        # Sort by RRF score
        candidates = sorted(fused_scores.items(), key=lambda x:x[1], reverse=True)[:50]
        candidate_ids = [c[0] for c in candidates]
        
        print(f"   [Retrieval] Found {len(candidate_ids)} candidates from Vector+BM25+Graph")
        
        # 3. Rerank (Small-to-Big)
        # 這裡很關鍵：我們用 Child ID 找到 Parent Content 來做 Rerank
        pairs = []
        valid_ids = []
        seen_parents = set()
        
        for cid in candidate_ids:
            p_text = self.parent_map.get(cid)
            if p_text and p_text not in seen_parents:
                pairs.append([query, p_text])
                valid_ids.append(cid)
                seen_parents.add(p_text)
        
        if not pairs: return "No relevant documents found."
        
        scores = self.reranker.predict(pairs)
        ranked_results = sorted(zip(valid_ids, scores, pairs), key=lambda x:x[1], reverse=True)[:top_k]
        
        # 4. CRAG (Evaluation)
        best_score = ranked_results[0][1]
        print(f"   [Rerank] Top score: {best_score:.4f}")
        
        final_context = ""
        
        # CRAG 閾值判斷 (假設 Cross-Encoder 分數範圍在 -10~10，這裡假設 > 0 為相關)
        # 實務上需根據模型調整 threshold
        if best_score < -2.0: # 信心度過低
            print("   [CRAG] 信心度不足，觸發 Web Search...")
            with DDGS() as ddgs:
                web_res = list(ddgs.text(query, max_results=3))
            web_text = "\n".join([f"Web: {r['body']}" for r in web_res])
            final_context = f"內部資料庫相關性低 (Score: {best_score})。\n\n補充網路搜尋結果：\n{web_text}"
        else:
            final_context = "\n\n".join([f"[Doc Score {s:.2f}]: {p[1]}" for i, s, p in ranked_results])
            
        return final_context

# ==================== 主執行區 (Main) ====================

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Integrated RAG System")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion pipeline on PDFs in the directory")
    parser.add_argument("--search", type=str, help="Run search with the provided query")
    
    args = parser.parse_args()

    # Automation Mode
    if args.ingest:
        print("--- Starting Automated Ingestion ---")
        ingestor = RAGIngestor()
        # 掃描目錄下的 PDF
        for f in os.listdir(Config.BASE_DIR):
            if f.endswith(".pdf"):
                path = os.path.join(Config.BASE_DIR, f)
                # 1. PDF -> MD
                md = ingestor.process_pdf(path)
                # 2. Chunking
                chunks = ingestor.create_chunks(md, f)
                # 3. Augmentation + Graph
                ingestor.augment_and_extract_graph(chunks)
                # 4. Save
                ingestor.save_to_db(chunks)
        print("--- Automated Ingestion Complete ---")

    if args.search:
        print(f"--- Starting Automated Search for: '{args.search}' ---")
        retriever = RAGRetriever()
        q = args.search
        context = retriever.retrieve(q)
        print("\n=== Context for LLM ===")
        print(context)
        print("=======================")
        
        # 最後一步：生成回答 (Optional)
        ans = LLMClient.query_text(f"基於以下參考資料回答問題：{q}\n\n參考資料：\n{context}")
        print(f"\nAI Answer:\n{ans}")
        return # Exit after search if in CLI mode

    # Interactive Mode (Fallback if no args)
    if not args.ingest and not args.search:
        # 模式選擇
        mode = input("Select Mode (1: Ingest PDF, 2: Search): ")
        
        if mode == "1":
            ingestor = RAGIngestor()
            # 掃描目錄下的 PDF
            for f in os.listdir(Config.BASE_DIR):
                if f.endswith(".pdf"):
                    path = os.path.join(Config.BASE_DIR, f)
                    # 1. PDF -> MD
                    md = ingestor.process_pdf(path)
                    # 2. Chunking
                    chunks = ingestor.create_chunks(md, f)
                    # 3. Augmentation + Graph
                    ingestor.augment_and_extract_graph(chunks)
                    # 4. Save
                    ingestor.save_to_db(chunks)
                    
        elif mode == "2":
            retriever = RAGRetriever()
            while True:
                q = input("\nAsk something (q to quit): ")
                if q.lower() == 'q': break
                
                context = retriever.retrieve(q)
                print("\n=== Context for LLM ===")
                print(context)
                print("=======================")
                
                # 最後一步：生成回答 (Optional)
                ans = LLMClient.query_text(f"基於以下參考資料回答問題：{q}\n\n參考資料：\n{context}")
                print(f"\nAI Answer:\n{ans}")

if __name__ == "__main__":
    main()



