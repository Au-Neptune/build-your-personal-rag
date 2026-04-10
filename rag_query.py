import os
import argparse
import chromadb
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

def get_retriever():
    """初始化 ChromaDB 與 Embedding 模型"""
    print("正在載入向量模型與資料庫...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = chroma_client.get_collection(name="ai_art_papers")
    return embedder, collection

def get_unique_sources(collection):
    """從 ChromaDB 中撈取所有不重複的文件名稱"""
    all_data = collection.get(include=["metadatas"])
    sources = set()
    for meta in all_data["metadatas"]:
        sources.add(meta["source"])
    return list(sources)

def auto_route_source(query, available_sources, client, model_name):
    """
    Agentic Router: 讓 LLM 自動判斷使用者的問題是否針對特定文件。
    這解決了使用者不知道 metadata 的痛點。
    """
    system_prompt = f"""你是一個精準的文件路由助手 (Query Router)。
以下是我們資料庫中擁有的所有文件清單：
{available_sources}

請分析使用者的問題，判斷他是否在詢問上述清單中的某個特定模型或文件。
- 如果有，請【只】輸出該文件的完整檔名 (例如: seedream2_2025.pdf)。
- 如果沒有，或者問題是廣泛性的，請【只】輸出 "None"。
不要輸出任何其他解釋字眼。"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0 # 保持絕對穩定
        )
        decision = response.choices[0].message.content.strip()
        sources_lower = {s.lower(): s for s in available_sources}
        return sources_lower.get(decision.lower())  # None if not found
    except Exception as e:
        return None

def retrieve_context(query, embedder, collection, top_k=5, source_filter=None):
    """將問題轉為向量，並去資料庫撈取最相關的 Top-K chunks (支援 Metadata 過濾)"""
    query_embedding = embedder.encode([query]).tolist()
    
    # 設定檢索參數
    query_kwargs = {
        "query_embeddings": query_embedding,
        "n_results": top_k
    }
    # 如果有指定來源文件，就加上 where 條件進行精準打擊！
    if source_filter:
        query_kwargs["where"] = {"source": source_filter}
        
    results = collection.query(**query_kwargs)
    
    # 整理檢索結果
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    context_parts = []
    cited_sources =[]
    
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        source_name = meta['source']
        chunk_idx = meta['chunk_index']
        context_parts.append(f"[文件 {i+1}] 來源: {source_name} (Chunk #{chunk_idx})\n內容: {doc}\n")
        cited_sources.append(f"{source_name} (Chunk #{chunk_idx})")
        
    context_string = "\n".join(context_parts)
    return context_string, cited_sources

def generate_answer(client, query, context, chat_history, model_name):
    """透過助教提供的 OpenAI 格式 Endpoint 呼叫 LLM"""

    system_prompt = """你是一個專業的 AI 繪圖技術研究助理。
請根據使用者提供的 [參考資料] 來回答問題。
【重要要求】：
你的回答必須基於參考資料，如果參考資料中沒有相關資訊，請誠實地說「根據目前的資料庫找不到相關資訊」。
"""

    # 將檢索到的 Context 與當前問題打包
    user_prompt = f"[參考資料]\n{context}\n\n[使用者問題]\n{query}"
    
    # 組合對話歷史
    messages =[{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": user_prompt})

    print(f"正在呼叫 LLM ({model_name}) 生成回答...")
    try:
        # 使用官方 OpenAI 的寫法
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            timeout=30,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM 呼叫失敗: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Query the RAG system.")
    parser.add_argument("--query", type=str, default=None, help="單次查詢的問題")
    parser.add_argument("--top-k", type=int, default=10, help="檢索的 chunks 數量")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="使用的 LLM 模型名稱")
    parser.add_argument("--source", type=str, default=None, help="手動鎖定特定的 PDF 檔名進行檢索")
    args = parser.parse_args()

    embedder, collection = get_retriever()
    unique_sources = get_unique_sources(collection)

    # 建立 OpenAI Client
    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL", "https://litellm.netdb.csie.ncku.edu.tw")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    # ==========================================
    # 模式 1：單次查詢模式 (CLI 提供 --query)
    # ==========================================
    if args.query:
        print(f"\n問題: {args.query}")

        target_source = args.source
        if not target_source:
            target_source = auto_route_source(args.query, unique_sources, client, args.model)
            if target_source and target_source != "None":
                print(f"偵測到特定意圖，自動鎖定文獻: {target_source}")
            else:
                target_source = None
                print("判斷為全域問題，進行跨文獻檢索...")

        context, cited_sources = retrieve_context(args.query, embedder, collection, args.top_k, target_source)
        answer = generate_answer(client, args.query, context, [], args.model)
        
        print("\n" + "="*50)
        print("【回答】:")
        print(answer)
        print("\n【引用來源】:")
        for idx, source in enumerate(cited_sources):
            print(f"[文件 {idx+1}] {source}")
        print("="*50 + "\n")
        return

    # ==========================================
    # 模式 2：互動式對話模式
    # ==========================================
    print("\n" + "="*50)
    print("歡迎來到 AI 繪圖知識庫 RAG 系統！(輸入 'exit' 離開)")
    print("="*50)
    
    chat_history =[]
    max_history_turns = 3 
    
    while True:
        try:
            user_input = input("\n你: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("掰掰！")
                break
            if not user_input.strip():
                continue
                
            target_source = args.source
            if not target_source:
                target_source = auto_route_source(user_input, unique_sources, client, args.model)
                if target_source and target_source != "None":
                    print(f"偵測到特定意圖，自動鎖定文獻: {target_source}")
                else:
                    target_source = None
                    print("判斷為全域問題，進行跨文獻檢索...")

            context, cited_sources = retrieve_context(user_input, embedder, collection, args.top_k, target_source)
            answer = generate_answer(client, user_input, context, chat_history, args.model)

            print("\nAI 助理:")
            print(answer)
            print("\n【引用來源】:")
            for idx, source in enumerate(cited_sources):
                print(f"[文件 {idx+1}] {source}")
            print("-" * 50)
            
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": answer})
            
            # 維持歷史長度
            if len(chat_history) > max_history_turns * 2:
                chat_history = chat_history[-max_history_turns * 2:]
                
        except KeyboardInterrupt:
            print("\n掰掰！")
            break

if __name__ == "__main__":
    main()