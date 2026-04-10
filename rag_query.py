import os
import argparse
import chromadb
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")

assert CHROMA_PERSIST_DIR, "Please set CHROMA_PERSIST_DIR in .env"
assert EMBEDDING_MODEL_NAME, "Please set EMBEDDING_MODEL in .env"

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

def translate_to_english(text, client, model_name):
    """簡單的翻譯函式，將非英文問題翻譯成英文，提升檢索效果"""
    system_prompt = "你是一個專業的翻譯助手，請將以下問題翻譯成流暢的英文，請勿添加任何解釋或多餘的文字，直接輸出翻譯結果即可。"
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return text  # 如果翻譯失敗，就回傳原文

def expand_query(original_query, client, model_name):
    """
    Query Expansion / Decomposition:
    讓 LLM 在檢索前，自動將使用者的問題展開成 3 個深入的學術子問題。
    """
    system_prompt = """You are an expert researcher specializing in AI-driven image generation technologies.
Your task is to perform **"Query Expansion."** Please decompose and expand the user's brief or abstract prompt into three specific, in-depth sub-queries to enable more precise vector retrieval within academic paper databases.

Adhere to the following rules:
1. Decompose the prompt into exactly three sub-queries.
2. Ensure each sub-query is concrete and well-defined.
3. Preserve the original semantic intent without diverging into unrelated domains.
4. Ensure sub-queries are complementary (e.g., covering different dimensions such as architecture, components, strengths/weaknesses, application scenarios, or experimental results).

**[Important]** Do not answer the questions. Output only the sub-queries, one per line, without numbering or any additional conversational text."""

    print("[Query Expansion] 正在自動擴展與拆解您的問題...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"使用者原始問題：{original_query}"}
            ],
            temperature=0.7 # 給予一點創造力來產生豐富的關鍵字
        )
        # 將 LLM 輸出的文字依換行符號切分成 list
        expanded_text = response.choices[0].message.content.strip()
        queries =[q.strip() for q in expanded_text.split('\n') if q.strip()]
        
        # 為了保險，把使用者的「原始問題」也加進去，確保原始意圖不遺失
        final_queries = [original_query] + queries[:3]
        return final_queries
    except Exception as e:
        print(f"⚠️ 擴展失敗，退回原始查詢: {e}")
        return [original_query]

def retrieve_context_multi(queries, embedder, collection, top_k_per_query=4, source_filter=None):
    """
    Multi-Query Retrieval:
    接收多個 queries，對每個 query 分別檢索，最後合併並去除重複的知識區塊。
    """
    all_chunks = {}
    cited_sources_set = set()
    
    # 針對每一個擴展出來的問題進行向量搜尋
    for q in queries:
        query_embedding = embedder.encode([q]).tolist()
        
        query_kwargs = {
            "query_embeddings": query_embedding,
            "n_results": top_k_per_query
        }
        if source_filter:
            query_kwargs["where"] = {"source": {"$in": [source_filter]}}
            
        results = collection.query(**query_kwargs)
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # 使用 chunk_id 來去重 (Deduplication)
        for doc, meta in zip(documents, metadatas):
            chunk_id = f"{meta['source']}_chunk_{meta['chunk_index']}"
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = f"[文件] 來源: {meta['source']} (Chunk #{meta['chunk_index']})\n內容: {doc}\n"
                cited_sources_set.add(f"{meta['source']} (Chunk #{meta['chunk_index']})")
                
    context_string = "\n".join(all_chunks.values())
    return context_string, list(cited_sources_set)

def generate_answer(client, query, expanded_queries, context, chat_history, model_name):
    """透過助教提供的 OpenAI 格式 Endpoint 呼叫 LLM"""

    system_prompt = """You are a professional AI image generation research assistant.
Your task is to provide a comprehensive and structured response to the **[User's Core Question]** based on the provided [Reference Materials].
The system also provides a set of **[Expanded Queries]** to help you analyze the problem from various perspectives (e.g., architecture, components, advantages, etc.).

**【Strict Rules】**:
1. Focus exclusively on answering the [User's Core Question].
2. The sub-queries are for analytical support only; do not answer them individually.
3. Synthesize the information covered by the sub-queries into a single, cohesive, and fluid response.
4. Your response must be based entirely on the [Reference Materials].
5. If the provided data is insufficient, clearly state: "Based on the current database, no relevant information could be found."
6. [CRITICAL] You MUST cite the source inline when referencing information, using the format[Chunk X]. (e.g., "The architecture uses DiT [Chunk 1].")

**【Tone and Formatting Requirements】**:

- **Clear Structure**: Use logical sections (e.g., Architecture, Design, Advantages) to organize your response.
- **Synthesized Narrative**: Avoid a Q&A format. The output should read like a refined technical briefing or summary rather than a list of answers.
- [CRITICAL] **Language**: Output your final response in Fluent Traditional Chinese (繁體中文), despite the prompt and context being in English.
"""

    # 將檢索到的 Context 與當前問題打包
    user_prompt = f"""
[User's Core Question]
{query}

[Expanded Queries(for LLM's internal analysis, do not answer separately)]
{chr(10).join(expanded_queries)}

[Reference Materials]
{context}
"""
    
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
    parser.add_argument("--top-k", type=int, default=4, help="檢索的 chunks 數量")
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
        query_english = translate_to_english(args.query, client, args.model)
        expanded_queries = expand_query(query_english, client, args.model)

        target_source = args.source
        if not target_source:
            routing_query = " ".join(expanded_queries)
            target_source = auto_route_source(routing_query, unique_sources, client, args.model)
            if target_source and target_source != "None":
                print(f"偵測到特定意圖，自動鎖定文獻: {target_source}")
            else:
                target_source = None
                print("判斷為全域問題，進行跨文獻檢索...")

        print("[擴展後的搜尋關鍵字]:")
        for eq in expanded_queries:
            print(f"   - {eq}")

        context, cited_sources = retrieve_context_multi(expanded_queries, embedder, collection, args.top_k, target_source)
        answer = generate_answer(client, query_english, expanded_queries, context, [], args.model)
        
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
                
            query_english = translate_to_english(user_input, client, args.model)
            expanded_queries = expand_query(query_english, client, args.model)

            target_source = args.source
            if not target_source:
                routing_query = " ".join(expanded_queries)
                target_source = auto_route_source(routing_query, unique_sources, client, args.model)
                if target_source and target_source != "None":
                    print(f"偵測到特定意圖，自動鎖定文獻: {target_source}")
                else:
                    target_source = None
                    print("判斷為全域問題，進行跨文獻檢索...")

            print("[擴展後的搜尋關鍵字]:")
            for eq in expanded_queries:
                print(f"   - {eq}")

            context, cited_sources = retrieve_context_multi(expanded_queries, embedder, collection, args.top_k, target_source)
            answer = generate_answer(client, query_english, expanded_queries, context, chat_history, args.model)

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