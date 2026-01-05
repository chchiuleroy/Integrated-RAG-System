# -*- coding: utf-8 -*-
"""
Integrated RAG System (Phase 1-3 Implementation)
整合：PDF處理(含VLM) + Small-to-Big + Fusion Retrieval + Rerank + GraphRAG + CRAG + Query Transform
"""

import os
import json
import base64
import requests
import re
import datetime
import shutil
import jieba
import numpy as np
import pickle
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
    TEXT_API_URL = "http://10.2.6.150:3000/api/v1/prediction/98317c50-906a-4656-8b23-b94847d02a91"
    IMAGE_API_URL = "http://10.2.6.150:3000/api/v1/prediction/ad31badb-292b-4236-a148-1f1d123c7a3c"
    
    # Paths (Modified for Safety)
    BASE_DIR = os.path.join(os.getcwd(), "documents") # 預設改為當前目錄下的 documents
    if not os.path.exists(BASE_DIR):
        try:
            os.makedirs(BASE_DIR)
        except:
            pass # 權限不足或其他原因忽略
            
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
    
    # Ontology (Lazy Load Logic)
    ONTOLOGY_PATH = "ontology.json"
    ENTITY_TYPES = ["Organization", "Person", "Location", "Event", "Concept", "Product"] # Default fallback
    RELATION_TYPES = ["RELATED", "PART_OF", "BELONGS_TO", "LOCATED_AT", "PARTICIPATED_IN"] # Default fallback

    @classmethod
    def load_ontology(cls):
        """嘗試載入 ontology.json，若失敗則使用預設值"""
        if os.path.exists(cls.ONTOLOGY_PATH):
            try:
                with open(cls.ONTOLOGY_PATH, "r", encoding="utf-8") as f:
                    _onto = json.load(f)
                    cls.ENTITY_TYPES = _onto.get("entity_types", cls.ENTITY_TYPES)
                    cls.RELATION_TYPES = _onto.get("relation_types", cls.RELATION_TYPES)
                    print(f"✅ Loaded Ontology: {cls.ENTITY_TYPES}")
            except Exception as e:
                print(f"⚠️ Failed to load ontology.json: {e}. Using defaults.")
        else:
            print("ℹ️ ontology.json not found. Using default entity types.")

# 執行載入嘗試
Config.load_ontology()


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
    
    @staticmethod
    def clean_and_parse_json(text: str) -> dict:
        """處理包含 Markdown 標籤或多餘說明的 JSON 字串"""
        try:
            # 移除 ```json ... ``` 標籤
            text = re.sub(r"```json|```", "", text).strip()
            # 尋找第一個 { 或 [ 到最後一個 } 或 ]
            match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(text)
        except Exception as e:
            print(f"⚠️ JSON 解析失敗: {e}")
            return {}

# ==================== 圖譜維護 =================================== #

class GraphMaintenance:
    def __init__(self, driver):
        self.driver = driver

    def merge_similar_entities(self, batch_size=50):
        """
        離線批次處理:一次讀取大量實體,分組送交 LLM 判斷相似性
        """
        print("--- 啟動離線實體消解 (Batch Entity Resolution) ---")
        
        with self.driver.session() as session:
            # 按類別分組抓取實體
            for ent_type in Config.ENTITY_TYPES:
                print(f"  > 正在處理類別: {ent_type}")
                query = "MATCH (e:Entity {type: $etype}) RETURN e.name AS name"
                result = session.run(query, etype=ent_type)
                all_names = [r["name"] for r in result]
                
                if len(all_names) < 2: 
                    continue
                
                # 分批次送給 LLM
                for i in range(0, len(all_names), batch_size):
                    batch = all_names[i : i + batch_size]
                    
                    prompt = f"""
                            你是一個專家級數據清洗師。請分析以下類別為 '{ent_type}' 的實體清單。
                            找出「指代同一個對象」的實體(例如:'TSMC' 與 '台積電')。
                            若有相似者,請選出一個最合適的名稱作為主實體。
                            輸出格式 JSON: {{"merges": [ {{"primary": "主名稱", "aliases": ["別名1", "別名2"]}} ]}}
                            
                            待處理清單: {batch}
                            """
                    
                    res_text = LLMClient.query_text(prompt)
                    res_data = LLMClient.clean_and_parse_json(res_text)
                    
                    for merge_task in res_data.get("merges", []):
                        primary = merge_task.get("primary")
                        aliases = merge_task.get("aliases", [])
                        if primary and aliases:
                            self._execute_merge_by_type(primary, aliases)

    def _execute_merge_by_type(self, primary_name: str, aliases: list):
        """
        方案 B-1: 按關係類型分別處理 (保留原始類型)
        缺點: 需要事先知道所有關係類型
        """
        with self.driver.session() as session:
            try:
                # Step 1: 確保主節點存在
                session.run("MERGE (target:Entity {name: $primary})", 
                           primary=primary_name)
                
                # Step 2: 遍歷每個關係類型進行合併
                for rel_type in Config.RELATION_TYPES:
                    # 2a. 轉移該類型的進入關係
                    session.run(f"""
                        MATCH (target:Entity {{name: $primary}})
                        MATCH (alias:Entity) WHERE alias.name IN $aliases
                        MATCH (src)-[r:{rel_type}]->(alias)
                        MERGE (src)-[newR:{rel_type}]->(target)
                        SET newR += properties(r),
                            newR.migrated_at = datetime(),
                            newR.confidence = coalesce(r.confidence, 1) + coalesce(newR.confidence, 0)
                        DELETE r
                    """, primary=primary_name, aliases=aliases)
                    
                    # 2b. 轉移該類型的出去關係
                    session.run(f"""
                        MATCH (target:Entity {{name: $primary}})
                        MATCH (alias:Entity) WHERE alias.name IN $aliases
                        MATCH (alias)-[r:{rel_type}]->(dst)
                        MERGE (target)-[newR:{rel_type}]->(dst)
                        SET newR += properties(r),
                            newR.migrated_at = datetime(),
                            newR.confidence = coalesce(r.confidence, 1) + coalesce(newR.confidence, 0)
                        DELETE r
                    """, primary=primary_name, aliases=aliases)
                
                # Step 3: 處理未知類型的關係 (統一為 RELATED)
                # 3a. 進入關係
                session.run("""
                    MATCH (target:Entity {name: $primary})
                    MATCH (alias:Entity) WHERE alias.name IN $aliases
                    MATCH (src)-[r]->(alias)
                    MERGE (src)-[newR:RELATED]->(target)
                    SET newR = properties(r),
                        newR.original_type = type(r),
                        newR.migrated_at = datetime()
                    DELETE r
                """, primary=primary_name, aliases=aliases)
                
                # 3b. 出去關係
                session.run("""
                    MATCH (target:Entity {name: $primary})
                    MATCH (alias:Entity) WHERE alias.name IN $aliases
                    MATCH (alias)-[r]->(dst)
                    MERGE (target)-[newR:RELATED]->(dst)
                    SET newR = properties(r),
                        newR.original_type = type(r),
                        newR.migrated_at = datetime()
                    DELETE r
                """, primary=primary_name, aliases=aliases)
                
                # Step 4: 合併屬性並刪除別名節點
                result = session.run("""
                    MATCH (target:Entity {name: $primary})
                    MATCH (alias:Entity) WHERE alias.name IN $aliases
                    SET target.aliases = coalesce(target.aliases, []) + collect(alias.name),
                        target.merged_count = coalesce(target.merged_count, 0) + count(alias),
                        target.last_merge_time = datetime()
                    WITH target, collect(alias) AS aliases_to_delete
                    UNWIND aliases_to_delete AS alias
                    DETACH DELETE alias
                    RETURN target.name AS merged_entity, size(aliases_to_delete) AS count
                """, primary=primary_name, aliases=aliases)
                
                record = result.single()
                if record and record['count'] > 0:
                    print(f"   ✅ [Type-Based Merged] {aliases} -> {record['merged_entity']} (合併了 {record['count']} 個節點)")
                else:
                    print(f"   ⚠️ [Merge Warning] 未找到別名節點或已合併: {aliases}")
                    
            except Exception as e:
                print(f"   ❌ [Merge Error] {e}")

    def _execute_merge_unified(self, primary_name: str, aliases: list):
        """
        方案 B-2: 統一關係類型為 RELATED (最簡單但會丟失類型信息)
        優點: 不需要預定義關係類型,程式碼簡潔
        缺點: 原始關係類型會存在 original_type 屬性中
        """
        with self.driver.session() as session:
            try:
                merge_query = """
                // 1. 確保主節點存在
                MERGE (target:Entity {name: $primary})
                
                // 2. 處理所有別名節點
                WITH target
                MATCH (alias:Entity) WHERE alias.name IN $aliases
                
                // 3. 轉移進入關係
                WITH target, alias
                OPTIONAL MATCH (src)-[r_in]->(alias)
                WHERE r_in IS NOT NULL
                WITH target, alias, src, r_in, 
                     type(r_in) AS in_type, 
                     properties(r_in) AS in_props
                MERGE (src)-[new_in:RELATED]->(target)
                SET new_in = in_props,
                    new_in.original_type = in_type,
                    new_in.migrated_at = datetime()
                DELETE r_in
                
                // 4. 轉移出去關係
                WITH target, alias
                OPTIONAL MATCH (alias)-[r_out]->(dst)
                WHERE r_out IS NOT NULL
                WITH target, alias, dst, r_out,
                     type(r_out) AS out_type,
                     properties(r_out) AS out_props
                MERGE (target)-[new_out:RELATED]->(dst)
                SET new_out = out_props,
                    new_out.original_type = out_type,
                    new_out.migrated_at = datetime()
                DELETE r_out
                
                // 5. 合併屬性並刪除
                WITH target, alias
                SET target.aliases = coalesce(target.aliases, []) + alias.name,
                    target.merged_count = coalesce(target.merged_count, 0) + 1
                DETACH DELETE alias
                
                RETURN target.name AS merged_entity, count(DISTINCT alias) AS merged_count
                """
                
                result = session.run(merge_query, primary=primary_name, aliases=aliases)
                record = result.single()
                if record and record['merged_count'] > 0:
                    print(f"   ✅ [Unified Merged] {aliases} -> {record['merged_entity']} (合併了 {record['merged_count']} 個節點)")
                else:
                    print(f"   ⚠️ [Merge Warning] 未找到別名節點: {aliases}")
                    
            except Exception as e:
                print(f"   ❌ [Merge Error] {e}")

    def verify_merge_results(self):
        """驗證合併結果的統計信息"""
        with self.driver.session() as session:
            # 檢查是否有重複實體
            duplicates = session.run("""
                MATCH (e:Entity)
                WITH e.type AS type, e.name AS name, count(*) AS cnt
                WHERE cnt > 1
                RETURN type, name, cnt
                ORDER BY cnt DESC
                LIMIT 10
            """)
            
            dup_list = list(duplicates)
            if dup_list:
                print("\n⚠️ 發現重複實體:")
                for rec in dup_list:
                    print(f"   - {rec['type']}: {rec['name']} (x{rec['cnt']})")
            else:
                print("\n✅ 未發現重複實體")
            
            # 統計合併信息
            stats = session.run("""
                MATCH (e:Entity)
                WHERE e.merged_count IS NOT NULL
                RETURN 
                    count(e) AS merged_entities,
                    sum(e.merged_count) AS total_merged,
                    avg(e.merged_count) AS avg_per_entity
            """).single()
            
            if stats and stats['merged_entities']:
                print("\n📊 合併統計:")
                print(f"   - 主實體數量: {stats['merged_entities']}")
                print(f"   - 總合併節點: {stats['total_merged']}")
                print(f"   - 平均每個主實體合併: {stats['avg_per_entity']:.2f} 個")

# ==================== 入庫流程 (Ingestion Pipeline) ==================== #

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
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        img_output = os.path.join(os.path.dirname(pdf_path), base_name + "_images")
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
            source_id = os.path.splitext(os.path.basename(source_name))[0]
            p_id = f"{source_id}_p{i}"
            
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
        print(f"[3/4] 提取原始圖譜數據 ({len(chunks)} chunks)...")
    
        for chunk in chunks:
            # 1. 僅做假設性問題生成
            prompt_aug = f"針對以下內容，生成3個問題：\n{chunk['text_embedding'][:500]}"
            questions = LLMClient.query_text(prompt_aug)
            chunk['text_embedding'] += f"\n\n[Hypothetical Questions]:\n{questions}"
            chunk['metadata']['aug_questions'] = questions
    
            # 2. 修改：提取時「不」要求 LLM 做消解，只提取原文中的實體名
            if self.driver:
                # 這裡的 Prompt 改為只提取，不歸一化，減少 LLM 推理負擔
                prompt_kg = f"請提取文本中的實體與關係。直接提取原文名稱即可，無需歸一化。格式 JSON: {{'entities':[], 'relations':[]}}\n文本：{chunk['text_embedding'][:1000]}"
                kg_json = LLMClient.query_text(prompt_kg)
                # 呼叫我們之前優化過的批次寫入方法
                self._save_to_neo4j(kg_json, chunk)
    
    def _entity_resolution_llm(self, raw_entities: List[Dict]) -> List[Dict]:
        """
        [優化] 實體對齊 (Entity Linking): 
        將提取出的原始實體送交 LLM 進行歸一化 (Normalization)
        """
        if not raw_entities: return []
        
        prompt = f"""
        你是一個專家級的知識圖譜工程師。請將以下實體清單進行歸一化：
        1. 統一名稱：例如 '台積電', 'TSMC', '台灣積體電路' 應統一為 'TSMC'。
        2. 類別對齊：必須屬於 {Config.ENTITY_TYPES} 之一。
        輸出格式為 JSON: {{"entities": [{{"original_name": "...", "resolved_name": "...", "type": "..."}}]}}
        
        待處理清單：{raw_entities}
        """
        res = LLMClient.query_text(prompt)
        try:
            # 簡單正則提取 JSON
            match = re.search(r"(\{.*\})", res, re.DOTALL)
            return json.loads(match.group()).get('entities', [])
        except:
            return raw_entities # 失敗則回退

    def _save_to_neo4j(self, kg_text: str, chunk: Dict):
        try:
            # 使用更強健的 JSON 提取
            data = LLMClient.clean_and_parse_json(kg_text)
            if not data: return
    
            current_time = datetime.datetime.now().strftime("%Y-%m-%d")
            
            # 準備批次數據
            ent_list = data.get('entities', [])
            rel_list = data.get('relations', [])
    
            with self.driver.session() as session:
                # 0. 在處理 entities 之前先確保 Chunk 節點存在
                session.run("""
                    MERGE (c:Chunk {id: $cid})
                    SET c.source = $source, c.created_at = $time
                """, cid=chunk['id'], source=chunk['metadata'].get('source', 'unknown'), time=current_time)
                # 1. 批次處理實體與 MENTIONED_IN 關係
                session.run("""
                    UNWIND $ents AS ent
                    MERGE (e:Entity {name: ent.name})
                    SET e.type = ent.type, e.last_updated = $time
                    WITH e
                    MATCH (c:Chunk {id: $cid})
                    MERGE (e)-[r:MENTIONED_IN]->(c)
                    SET r.detected_at = $time
                """, ents=ent_list, time=current_time, cid=chunk['id'])
    
                # 2. 批次處理關係 (RELATED)
                session.run("""
                    UNWIND $rels AS rel
                    MATCH (a:Entity {name: rel.source})
                    MATCH (b:Entity {name: rel.target})
                    MERGE (a)-[r:RELATED {type: rel.type}]->(b)
                    ON CREATE SET r.confidence = 1, r.first_seen = $time
                    ON MATCH SET r.confidence = r.confidence + 1, r.last_seen = $time
                """, rels=rel_list, time=current_time)
                
        except Exception as e:
            print(f"❌ Graph Batch Save Error: {e}")

    def save_to_db(self, chunks: List[Dict]):
        """
        [優化版] 將時間維度寫入 ChromaDB Metadata
        """
        print("[4/4] Saving to ChromaDB with temporal metadata...")
        ids = [c['id'] for c in chunks]
        docs = [c['text_embedding'] for c in chunks]
        
        # 準備 Metadata，加入創建時間
        current_time_int = int(datetime.datetime.now().timestamp())
        metas = []
        for c in chunks:
            m = c['metadata'].copy()
            m["created_at"] = current_time_int  # 用於數值過濾
            m["date_string"] = datetime.datetime.now().strftime("%Y-%m-%d")
            metas.append(m)
        
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
        cache_path = "bm25_index.pkl"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
                self.bm25 = cache_data['bm25']
                self.bm25_docs = cache_data['docs']
                self.bm25_ids = cache_data['ids']
        else:
            print("Building BM25 index from scratch...")
            all_data = self.collection.get(include=["documents", "ids"])
            self.bm25_docs = all_data['documents']
            self.bm25_ids = all_data['ids']
            if not self.bm25_docs:
                print("⚠️ Warning: No documents found in ChromaDB. BM25 will be empty.")
                tokenized_corpus = []
            else:
                tokenized_corpus = [list(jieba.cut(doc)) for doc in self.bm25_docs]
            
            self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
            
            with open(cache_path, "wb") as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'docs': self.bm25_docs,
                    'ids': self.bm25_ids
                }, f)
            print("BM25 index built and cached.")
        
        # 3. Reranker
        print("Loading Reranker...")
        self.reranker = CrossEncoder(Config.RERANK_MODEL)
        
        # 4. Neo4j
        try:
            self.driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
        except: self.driver = None
        
        # 5. Parent Map (Small-to-Big) - [Modified for Safety]
        if os.path.exists("parent_map.json"):
            with open("parent_map.json", "r", encoding="utf-8") as f:
                self.parent_map = json.load(f)
        else:
            print("⚠️ parent_map.json not found. Small-to-Big retrieval may fail.")
            self.parent_map = {}
    

    def query_transform(self, query: str) -> List[str]:
        """Phase 2: Query Transformation (多路查詢)"""
        print(f"   [Transform] Generating multi-queries for: {query}")
        prompt = f"請針對問題 '{query}' 生成 3 個不同切入點的搜尋關鍵字或問句，一行一個。"
        res = LLMClient.query_text(prompt)
        new_queries = [q.strip() for q in res.split('\n') if q.strip()]
        return [query] + new_queries[:3] # 包含原問題
    
    def extract_entities(self, query: str) -> List[str]:
        keywords = set(jieba.cut(query))
        entities = []
    
        if not self.driver:
            return []
    
        with self.driver.session() as session:
            for kw in keywords:
                if len(kw) < 2:
                    continue
                res = session.run(
                    "MATCH (e:Entity) WHERE e.name CONTAINS $kw RETURN e.name LIMIT 3",
                    kw=kw
                )
                entities.extend([r["e.name"] for r in res])
    
        return list(set(entities))

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
    
    def graph_guided_chunk_ids(self, query: str, limit=50):
        if not self.driver:
            return [], []
    
        # 1️⃣ 抽取 query entities
        entities = self.extract_entities(query)
        if not entities:
            return [], []
    
        # 2️⃣ Neo4j 查 chunk
        cypher = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
        WHERE e.name IN $entities
        RETURN DISTINCT c.id AS chunk_id
        LIMIT $limit
        """
    
        chunk_ids = []
    
        with self.driver.session() as session:
            result = session.run(cypher, entities=entities, limit=limit)
            for r in result:
                chunk_ids.append(r["chunk_id"])
    
        return chunk_ids, entities
    
    def get_reasoning_paths(self, entities: List[str], limit=5):
        """
        回傳：Entity → Relation → Entity → Chunk 的推理路徑
        """
        if not self.driver:
            return []
    
        cypher = """
        MATCH p=(e:Entity)-[r:RELATED*1..2]-(o:Entity)
        WHERE 
            e.name IN $entities
            AND all(x IN r WHERE x.confidence >= 2)
        MATCH (o)-[:MENTIONED_IN]->(c:Chunk)
        RETURN 
            [n IN nodes(p) | n.name] AS nodes,
            [rel IN relationships(p) | rel.type] AS relations,
            c.id AS chunk_id
        LIMIT $limit
        """
    
        with self.driver.session() as session:
            res = session.run(cypher, entities=entities, limit=limit)
            return [dict(r) for r in res]

    def retrieve(self, query: str, top_k=5) -> str:
        """
        0. Graph-guided -> 1. Query Transform -> 2. Multi-route Retrieval -> 3. RRF Fusion 
        -> 4. Temporal Rerank -> 5. Dynamic Top-K -> 6. CRAG
        """
        print(f"\n--- Processing Query: {query} ---")
        
        # 0. Graph-guided candidate pruning
        graph_ids, entities = self.graph_guided_chunk_ids(query)
        reasoning_paths = []
        if entities:
            # 使用 query entity（你前面已抽過）
            reasoning_paths = self.get_reasoning_paths(entities)

        use_graph_filter = len(graph_ids) >= 5  # 避免 Graph 太小誤傷
        
        # 1. Query Transformation (產生多個搜尋變體)
        queries = self.query_transform(query) # 回傳 [原問題, 變體1, 變體2, 變體3]
        
        # 2. 多路檢索與 RRF 融合 (Fusion)
        fused_scores = {}
        
        for q in queries:
            # A. Vector Search
            if use_graph_filter:
                v_res = self.collection.query(
                    query_texts=[q],
                    n_results=20,
                    ids=graph_ids
                )
            else:
                v_res = self.collection.query(
                    query_texts=[q],
                    n_results=20
                )
            # 防呆：避免 v_res 為空
            if v_res['ids']:
                for i, vid in enumerate(v_res['ids'][0]):
                    fused_scores[vid] = fused_scores.get(vid, 0) + 1/(60+i)
            
            # B. BM25 Search
            if self.bm25:
                b_docs = self.bm25.get_top_n(list(jieba.cut(q)), self.bm25_docs, n=20)
                for i, doc in enumerate(b_docs):
                    try:
                        idx = self.bm25_docs.index(doc)
                        bid = self.bm25_ids[idx]
                        scores = self.bm25.get_scores(list(jieba.cut(q)))  #單次計算，但若loop多，累積O(N)
                        score = scores[idx]
                        if score <= 0:
                            continue
                        
                        if use_graph_filter and bid not in graph_ids:
                            continue
                        
                        fused_scores[bid] = fused_scores.get(bid, 0) + 1/(60+i)
                    except:
                        pass

        # 3. 提取前 50 個候選者進入 Rerank
        candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:50]
        candidate_ids = [c[0] for c in candidates]
        
        # 如果完全沒有候選文檔
        if not candidate_ids:
            print("   [CRAG] 無相關文檔,啟動外部搜索...")
            try:
                with DDGS() as ddgs:
                    web_res = list(ddgs.text(query, max_results=3))
                web_context = "補充外部資訊:\n" + "\n".join([r['body'] for r in web_res])
                return web_context, []
            except:
                return "無法找到相關資料,外部搜索也失敗。", []

        # 4. 獲取 Metadata 與準備 Rerank (Small-to-Big)
        res_meta = self.collection.get(ids=candidate_ids, include=['metadatas'])
        meta_dict = {cid: m for cid, m in zip(res_meta['ids'], res_meta['metadatas'])}
        
        pairs = []
        valid_ids = []
        seen_parents = set()
        for cid in candidate_ids:
            p_text = self.parent_map.get(cid) # 透過 Child ID 找 Parent 完整內容
            if p_text and p_text not in seen_parents:
                pairs.append([query, p_text])
                valid_ids.append(cid)
                seen_parents.add(p_text)
        if not pairs:
            # 直接回退到 fusion 排序結果
            return "內部檢索失敗，無法取得完整上下文。", reasoning_paths

        # 5. 執行 Cross-Encoder Rerank 並加上時間權重
        raw_scores = self.reranker.predict(pairs)
        now_ts = int(datetime.datetime.now().timestamp())
        
        graph_chunk_id_set = set()

        if reasoning_paths:
            graph_chunk_id_set = {
                p["chunk_id"]
                for p in reasoning_paths
                if "chunk_id" in p
            }
        
        scored_results = []
        for i, score in enumerate(raw_scores):
            doc_id = valid_ids[i]
            metadata = meta_dict.get(doc_id, {})
            doc_ts = metadata.get("created_at", now_ts) # 從 metadata 提取時間維度
            
            # 時間衰減 (每隔 180 天扣 0.1 分)
            time_penalty = ((now_ts - doc_ts) / 15552000) * 0.1
            adjusted_score = score - time_penalty
            
            # Graph reasoning boost（推理證據加權）
            if doc_id in graph_chunk_id_set:
                adjusted_score += 0.3
                
            scored_results.append({
                "score": adjusted_score,
                "text": pairs[i][1],
                "date": metadata.get("date_string", "Unknown")
            })
            
        scored_results.sort(key=lambda x: x['score'], reverse=True)

        # 6. 動態 Top-K 篩選
        best_score = scored_results[0]['score']
        dynamic_results = []
        for res in scored_results:
            # 動態過濾：分數差距過大或低於信心門檻則捨棄
            if res['score'] > -1.5 and (best_score - res['score'] < 3.0):
                dynamic_results.append(res)
                if len(dynamic_results) >= top_k: break
        
        # 7. CRAG 門檻判斷 (決定是否啟動 Web Search)
        if not dynamic_results or best_score < -2.5:
            print("   [CRAG] 低信心度,啟動外部搜索...")
            try:
                with DDGS() as ddgs:
                    web_res = list(ddgs.text(query, max_results=3))
                web_context = "補充外部資訊:\n" + "\n".join([r['body'] for r in web_res])
                return web_context, reasoning_paths
            except:
                pass  # 外部搜索失敗,繼續使用現有結果
        
        # 8. 最終輸出
        context_text = "\n\n".join([f"[Date: {r['date']}] {r['text']}" for r in dynamic_results])
        return context_text, reasoning_paths

# ==================== 主執行區 (Main) ====================

import argparse
import sys

# ==================== 執行邏輯封裝 (Workflow Functions) ====================

def run_full_ingestion():
    print("--- 啟動自動化入庫流程 (Ingestion) ---")
    ingestor = RAGIngestor()
    
    # 檢查 documents 目錄是否有 PDF
    pdf_files = [f for f in os.listdir(Config.BASE_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️ Warning: No PDF files found in {Config.BASE_DIR}. Please add PDFs first.")
        return

    # 第一步：只管入庫（此時圖譜中可能有很多重複節點，如 '台積電' 和 'TSMC'）
    for f in pdf_files:
        path = os.path.join(Config.BASE_DIR, f)
        md = ingestor.process_pdf(path)
        chunks = ingestor.create_chunks(md, f)
        ingestor.augment_and_extract_graph(chunks) # 這裡現在很快，因為不消解
        ingestor.save_to_db(chunks)
            
    # 第二步：離線統一清理 (這就是你要求的改動)
    if ingestor.driver:
        maintenance = GraphMaintenance(ingestor.driver)
        # 此處執行優化後的批次消解，極大節省 Token，因為相同實體只會被判斷一次
        maintenance.merge_similar_entities(batch_size=40) 
        
    print("--- 所有流程已完成 ---")

def run_qa_flow(retriever, query):
    """封裝檢索與回答的流程邏輯"""
    context, reasoning_paths = retriever.retrieve(query)
    print("\n=== Context for LLM ===")
    print(context)
    print("=======================")
    
    # 生成回答
    prompt = f"""
                基於以下資料回答問題。
                
                【推理路徑】
                {json.dumps(reasoning_paths, ensure_ascii=False, indent=2)}
                
                【參考文件】
                {context}
                
                問題：{query}
                """
    ans = LLMClient.query_text(prompt)
    print(f"\nAI Answer:\n{ans}")
    return ans

# ==================== 主執行區 (Main) ====================

def main():
    parser = argparse.ArgumentParser(description="Integrated RAG System")
    parser.add_argument("--ingest", action="store_true", help="執行 PDF 入庫流程")
    parser.add_argument("--search", type=str, help="直接執行特定問題搜尋")
    parser.add_argument("--cleanup", action="store_true", help="單獨執行圖譜去重合併")
    args = parser.parse_args()

    # 1. 處理單獨的圖譜清理任務
    if args.cleanup:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
        maintenance = GraphMaintenance(driver)
        maintenance.merge_similar_entities()
        if not args.ingest and not args.search: return

    # 2. 處理入庫任務 (CLI 模式)
    if args.ingest:
        run_full_ingestion()

    # 3. 處理搜尋任務 (CLI 模式)
    if args.search:
        print(f"--- 執行單次搜尋: '{args.search}' ---")
        retriever = RAGRetriever()
        run_qa_flow(retriever, args.search)
        return

    # 4. 互動模式 (當沒有提供任何參數時觸發)
    if not args.ingest and not args.search:
        print("\n=== RAG 系統互動終端 ===")
        print(f"檔案目錄 (Base Dir): {Config.BASE_DIR}")
        print("1: 匯入 PDF (Ingest)")
        print("2: 執行檢索 (Search)")
        print("3: 執行圖譜維護 (Cleanup)")
        mode = input("請選擇模式: ")
        
        if mode == "1":
            run_full_ingestion()
        elif mode == "2":
            retriever = RAGRetriever()
            while True:
                q = input("\n請輸入您的問題 (輸入 q 離開): ")
                if q.lower() == 'q': break
                run_qa_flow(retriever, q)
        elif mode == "3":
            # 複用 Cleanup 邏輯
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
            maintenance = GraphMaintenance(driver)
            maintenance.merge_similar_entities()

if __name__ == "__main__":
    main()