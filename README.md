Integrated RAG System (Phase 1-3 Implementation)
這是一個高度整合的檢索增強生成 (RAG) 系統，結合了 多模態處理、語意切分、混合檢索、圖譜檢索 (GraphRAG) 以及 自我修正機制 (CRAG)。

🌟 核心功能與技術亮點
1. 進階入庫流程 (Ingestion Pipeline)
多模態 PDF 解析：利用 PyMuPDF4LLM 將 PDF 轉為 Markdown，並透過 VLM (視覺語言模型) 對文件中的圖片進行語意描述增強。
語意切分 (Semantic Chunking)：不同於固定長度切分，系統根據向量空間中的語意相似度變化來決定斷句點。
Small-to-Big 架構：存儲細粒度的語意區塊 (Child) 用於檢索，但提供完整的章節內容 (Parent) 給 LLM 閱讀。
知識圖譜構建：自動提取文本中的實體與關係，並寫入 Neo4j 圖資料庫。

2. 強大檢索引擎 (Retrieval Engine)
混合搜尋 (Hybrid Search)：結合向量相似度 (ChromaDB) 與關鍵字過濾 (BM25)。
查詢轉換 (Query Transformation)：自動將單一問題擴展為多維度的搜尋關鍵字，提升召回率。
Graph-Guided Retrieval：利用圖資料庫進行 1~2 hop 的關聯檢索，彌補向量檢索無法處理複雜關聯的問題。

二階段重排序 (Rerank)：使用 Cross-Encoder 對候選清單進行精確打分。

3. 自我修正與增強 (CRAG & Web Search)
CRAG 機制：系統會自動評估檢索內容的信心度。當內部資料相關性不足時，會自動觸發 DuckDuckGo 網路搜尋 來補充背景知識。

🛠️ 技術棧 (Tech Stack)
向量數據庫: ChromaDB
圖數據庫: Neo4j
模型框架: Sentence-Transformers (Embedding & Rerank)
PDF 處理: PyMuPDF, pymupdf4llm
搜尋 API: DuckDuckGo Search

🚀 快速上手

1. 環境設定
確保已安裝必要的套件：
Bash
pip install pymupdf pymupdf4llm chromadb rank_bm25 sentence-transformers neo4j duckduckgo-search jieba requests

3. 配置 Neo4j
在 Config 類別中修改您的 Neo4j 連線資訊：

Python

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"
3. 執行指令
文件入庫 (Ingestion)：將 PDF 放入指定目錄並執行：

Bash

python integrated_rag_system_v1.py --ingest
啟動檢索 (Search)：直接針對問題進行查詢：

Bash
python integrated_rag_system_v1.py --search "您的問題"
📊 系統架構圖 (流程簡述)
PDF -> Markdown + VLM Image Captioning
Chunking -> Semantic Split + Hypothetical Questions
Indexing -> ChromaDB (Vector) + Neo4j (Graph)
Query -> Multi-Query Transform -> Hybrid Search
Rerank -> Confidence Scoring
Evaluate -> If Low Score -> Web Search (CRAG)
Final Answer -> LLM Generation
