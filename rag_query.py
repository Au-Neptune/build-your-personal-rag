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

def retrieve_context(query, embedder, collection, top_k=5):
    """將問題轉為向量，並去資料庫撈取最相關的 Top-K chunks"""
    query_embedding = embedder.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
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

def generate_answer(query, context, chat_history, model_name):
    """透過助教提供的 OpenAI 格式 Endpoint 呼叫 LLM"""
    
    # 讀取助教的 Key 和 URL
    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL", "https://litellm.netdb.csie.ncku.edu.tw")
    
    if not api_key:
        return "錯誤: 在 .env 找不到 LITELLM_API_KEY，請確認設定。"

    # 建立 OpenAI Client (指向助教的 LiteLLM 伺服器)
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    system_prompt = """你是一個專業的 AI 繪圖技術研究助理。
請根據使用者提供的 [參考資料] 來回答問題。
【重要要求】：
1. 你的回答必須基於參考資料，如果參考資料中沒有相關資訊，請誠實地說「根據目前的資料庫找不到相關資訊」。
2. 在回答時，如果引用了某個概念，請在句子後面標註引用來源，格式為 [文件 X]。"""

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
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM 呼叫失敗: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Query the RAG system.")
    parser.add_argument("--query", type=str, default=None, help="單次查詢的問題")
    parser.add_argument("--top-k", type=int, default=10, help="檢索的 chunks 數量")
    # 預設使用助教的 Gemini 資源，你也可以在執行時改成 --model gpt-oss:20b 或 gpt-oss-20b
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="使用的 LLM 模型名稱")
    args = parser.parse_args()

    embedder, collection = get_retriever()
    
    # ==========================================
    # 模式 1：單次查詢模式 (CLI 提供 --query)
    # ==========================================
    if args.query:
        print(f"\n問題: {args.query}")
        context, cited_sources = retrieve_context(args.query, embedder, collection, args.top_k)
        answer = generate_answer(args.query, context,[], args.model)
        
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
                
            # 1. 檢索資料
            context, cited_sources = retrieve_context(user_input, embedder, collection, args.top_k)
            
            # 2. 生成回答
            answer = generate_answer(user_input, context, chat_history, args.model)
            
            # 3. 印出結果
            print("\nAI 助理:")
            print(answer)
            print("\n【引用來源】:")
            for idx, source in enumerate(cited_sources):
                print(f"[文件 {idx+1}] {source}")
            print("-" * 50)
            
            # 4. 更新歷史紀錄
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