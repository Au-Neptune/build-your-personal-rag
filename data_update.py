import os
import glob
import argparse
import shutil
import fitz  # PyMuPDF
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

def clean_and_extract_pdf(pdf_path):
    """
    讀取 PDF 並解決雙欄排版的問題。
    利用 PyMuPDF 的 blocks 進行座標排序，還原正確閱讀順序。
    """
    doc = fitz.open(pdf_path)
    full_text =[]

    for page_num, page in enumerate(doc):
        # 取得文字區塊：格式為 (x0, y0, x1, y1, "text", block_no, block_type)
        blocks = page.get_text("blocks")
        
        # 過濾掉非文字區塊 (block_type != 0)
        text_blocks =[b for b in blocks if b[6] == 0]
        
        # arXiv 雙欄排序啟發式邏輯：先按 Y 座標（容許一定的同一行誤差），再按 X 座標
        # 使用 round(b[1] / 50) 將相近的 Y 座標歸為同一行
        text_blocks.sort(key=lambda b: (round(b[1] / 50), b[0]))
        
        for b in text_blocks:
            text = b[4].replace('\n', ' ').strip()
            # 簡單清理：過濾掉太短的區塊（如單純的頁碼、圖表標題小字）
            if len(text) > 20: 
                full_text.append(text)
                
    return "\n\n".join(full_text)

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    將長文本切分為 Overlapping Chunks。
    依字元數切割，避免英文單字被切斷的簡單實作。
    """
    chunks =[]
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        # 盡量不要切斷單字，找到最近的空白處切斷
        if end < text_length:
            while end > start and text[end] != ' ':
                end -= 1
            if end == start:  # 找不到空白，強制切斷
                end = start + chunk_size
        
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # 忽略過短的無意義 chunk
            chunks.append(chunk)
            
        start = end - overlap
        
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Build or update the RAG Vector DB.")
    parser.add_argument("--rebuild", action="store_true", help="Clear all processed data and rebuild the DB.")
    args = parser.parse_args()

    raw_dir = "data/raw"
    processed_dir = "data/processed"

    # ==========================================
    # 1. 處理 --rebuild 邏輯
    # ==========================================
    if args.rebuild:
        print("🗑️  Rebuilding DB: Clearing processed data and ChromaDB...")
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)
    
    os.makedirs(processed_dir, exist_ok=True)

    # ==========================================
    # 2. 初始化 Embedding 模型與 ChromaDB
    # ==========================================
    print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
    # 自動下載並加載本地開源模型 (第一次執行需要網路下載約 80MB)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # 初始化 ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    # 創建或取得 Collection
    collection = chroma_client.get_or_create_collection(name="ai_art_papers")

    # ==========================================
    # 3. 讀取與處理 PDF (增量更新邏輯)
    # ==========================================
    pdf_files = glob.glob(os.path.join(raw_dir, "*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/raw/")
        return

    print(f"Found {len(pdf_files)} PDF files. Processing...")
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(processed_dir, f"{base_name}.txt")
        
        # [增量更新機制] 如果 txt 已經存在，代表處理過了，跳過文字萃取
        if not os.path.exists(txt_path):
            extracted_text = clean_and_extract_pdf(pdf_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)
        else:
            with open(txt_path, "r", encoding="utf-8") as f:
                extracted_text = f.read()
                
        # 檢查該篇論文是否已經在 DB 中 (透過 metadata 過濾)
        existing_docs = collection.get(where={"source": filename})
        if len(existing_docs["ids"]) > 0:
            continue  # 已經存在於 DB 中，跳過 Embedding 寫入

        # ==========================================
        # 4. Chunking & Embedding 寫入 DB
        # ==========================================
        chunks = chunk_text(extracted_text, chunk_size=1200, overlap=200)
        if not chunks:
            continue
            
        # 產生 ID 與 Metadata
        ids =[f"{base_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas =[{"source": filename, "chunk_index": i} for i in range(len(chunks))]
        
        # 產生向量 (SentenceTransformer 支援直接傳入 list of strings)
        embeddings = embedder.encode(chunks).tolist()
        
        # 寫入 ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

    print("Vector DB update complete!")
    print(f"Total documents in DB: {collection.count()}")

if __name__ == "__main__":
    main()