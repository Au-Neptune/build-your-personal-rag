import os
import json
import argparse
import datetime
import openai
from dotenv import load_dotenv

from rag_query import (
    get_retriever,
    get_unique_sources,
    translate_to_english,
    expand_query,
    retrieve_context_multi,
    generate_answer,
)

load_dotenv()

# 10 個領域無關 (domain-agnostic) 的全域問題：
# 不預設系統知道知識庫主題為何，讓 RAG 自行從資料中歸納領域資訊。
GLOBAL_QUESTIONS = [
    "這個知識庫主要涵蓋哪些主題與領域？請說明其範圍、焦點與讀者對象。",
    "此知識庫中最核心、最基礎的概念、理論或技術有哪些？請依重要性說明。",
    "近年來此領域有哪些重要的發展趨勢、典範轉移或演進方向？",
    "此領域中具代表性的關鍵人物、研究機構、公司、產品或模型有哪些？",
    "此領域常用的方法論、系統架構或技術流程 (pipeline) 是什麼？",
    "知識庫中提到哪些代表性的工具、框架、資料集或實際實作方案？",
    "此領域目前面臨的主要挑戰、技術限制或尚未解決的問題是什麼？",
    "此領域中被公認為最佳實踐 (best practices) 或標準做法的有哪些？",
    "不同方法、模型或解決方案之間有哪些差異與取捨 (trade-offs)？",
    "知識庫中提供了哪些具代表性的實驗結果、評估指標 (metrics) 或應用案例？",
]

SYNTHESIZER_SYSTEM_PROMPT = """你是一位資深的技術知識分析師，任務是將多組 RAG 問答結果彙整 (synthesize) 成一份結構完整、專業的 skill.md 知識技能文件。

【嚴格規則】
1. 內容僅能依據【原始問答】彙整，不可自行虛構資訊。
2. 輸出格式為【純 Markdown】，語言為【繁體中文】；遇到專有名詞 (模型名稱、演算法、技術術語) 必須附上英文原文，例如：「擴散模型 (Diffusion Model)」、「注意力機制 (Attention)」。
3. 文件開頭以 `# Skill: <主題名稱>` 作為 H1 標題，主題名稱請依據問答內容自動判斷。
4. 必須嚴格包含以下章節（章節標題保持英文）：
   - ## Overview  （200-300 字簡介此知識領域）
   - ## Core Concepts  （列出 5-15 個核心概念，每項用子標題 + 簡短段落說明）
   - ## Key Trends  （列出 3-10 項近期趨勢，條列並簡述）
   - ## Key Entities  （分類列出關鍵人物 / 組織 / 模型 / 產品）
   - ## Methodology & Best Practices  （整理方法論與最佳實踐）
   - ## Knowledge Gaps & Limitations  （說明限制與未解問題）
   - ## Example Q&A  （從【原始問答】中精選 3-5 組最具代表性者；Q 用原問題全文，A 濃縮為 3-6 行重點）
5. 【不要】輸出 `## Metadata` 章節與 `## Source References` 章節，這兩個章節會由程式自動附加。
6. 直接輸出內容，不要使用「根據提供的資料...」這類開場白，也不要在最外層加 ```markdown 程式碼框。
"""


def answer_single_question(question, embedder, collection, client, model_name, top_k):
    """針對單一全域問題完整跑一次 RAG pipeline，回傳 (answer, cited_sources)。"""
    print(f"\n==> [Q] {question}")
    query_english = translate_to_english(question, client, model_name)
    expanded_queries = expand_query(query_english, client, model_name)
    context, cited = retrieve_context_multi(
        expanded_queries, embedder, collection, top_k_per_query=top_k
    )
    answer = generate_answer(
        client, query_english, expanded_queries, context, [], model_name
    )
    print(f"    引用了 {len(cited)} 個 chunk")
    return answer, cited


def synthesize_skill_md(qa_pairs, client, model_name):
    """把 10 組 Q&A 丟給 LLM 彙整成 skill.md 主體。"""
    qa_blocks = []
    for i, (q, a) in enumerate(qa_pairs, start=1):
        qa_blocks.append(f"### Q{i}: {q}\n\n{a}")
    qa_text = "\n\n---\n\n".join(qa_blocks)

    user_prompt = f"""以下是針對本知識庫進行的 10 組 RAG 問答結果，請依照系統指示的章節結構彙整成一份完整的 skill.md。

【原始問答】
{qa_text}
"""

    print("\n==> 呼叫 LLM 彙整 skill.md ...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        timeout=180,
    )
    return response.choices[0].message.content.strip()


def infer_domain_and_agent(qa_pairs, client, model_name):
    """用一次輕量 LLM call，從前幾組問答中推斷知識領域名稱與適用 Agent 類型。"""
    sample = "\n\n".join(
        [f"Q: {q}\nA: {a[:400]}" for q, a in qa_pairs[:3]]
    )
    system_prompt = """你將閱讀幾組 RAG 問答摘要，請判斷此知識庫的「知識領域」與「適用 Agent 類型」。

【嚴格輸出格式】：只輸出一個 JSON 物件，不要任何額外說明或 ```markdown 框：
{"domain": "<領域名稱>", "agent_type": "<Agent 類型>"}

【欄位規則】：
- domain: 5-25 字的中文領域名稱，專有名詞附英文，例如「AI 影像生成技術 (AI Image Generation)」。
- agent_type: 從以下擇一最合適：研究助手 / 技術顧問 / 領域問答機器人 / 學術研究助理 / 產品分析師。"""

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample},
            ],
            temperature=0.0,
            timeout=60,
        )
        text = resp.choices[0].message.content.strip()
        # 容錯：如果 LLM 不小心包了 code fence，剝掉
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines() if not line.startswith("```")
            )
        data = json.loads(text)
        return (
            data.get("domain", "未指定領域").strip(),
            data.get("agent_type", "領域問答機器人").strip(),
        )
    except Exception as e:
        print(f"⚠️ 領域/Agent 類型推斷失敗，使用預設值: {e}")
        return "未指定領域", "領域問答機器人"


def build_metadata_section(domain, total_sources, agent_type):
    today = datetime.date.today().isoformat()
    return (
        "## Metadata\n"
        f"- **知識領域**：{domain}\n"
        f"- **資料來源數量**：{total_sources} 份文件\n"
        f"- **最後更新時間**：{today}\n"
        f"- **適用 Agent 類型**：{agent_type}\n"
    )


def inject_metadata_after_h1(skill_body, metadata_block):
    """在 LLM 輸出的第一個 H1 之後插入 Metadata 區塊；若找不到 H1 則放在最前面。"""
    lines = skill_body.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("# "):
            return "\n".join(lines[: idx + 1]) + "\n\n" + metadata_block + "\n" + "\n".join(lines[idx + 1 :])
    return metadata_block + "\n" + skill_body


def build_source_references(cited_sources):
    """cited_sources 元素格式為 'filename.pdf (Chunk #3)'，這裡彙整為去重後的檔案清單。"""
    files = sorted({s.split(" (Chunk")[0] for s in cited_sources})
    lines = ["## Source References", ""]
    lines.append(f"本技能文件由 `skill_builder.py` 自動彙整，過程中實際被 RAG 檢索並引用的來源檔案如下（共 {len(files)} 份）：")
    lines.append("")
    for f in files:
        lines.append(f"- `{f}`")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Auto-build skill.md from the RAG knowledge base.")
    parser.add_argument("--output", type=str, default="skill.md", help="輸出 skill.md 的路徑")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="使用的 LLM 模型名稱")
    parser.add_argument("--top-k", type=int, default=4, help="每個擴展子問題檢索的 chunk 數")
    args = parser.parse_args()

    embedder, collection = get_retriever()
    total_sources = len(get_unique_sources(collection))

    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL", "https://litellm.netdb.csie.ncku.edu.tw")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    qa_pairs = []
    all_cited = set()

    print("="*60)
    print(f"開始建構 skill.md，共 {len(GLOBAL_QUESTIONS)} 個全域問題")
    print(f"知識庫中共有 {total_sources} 份不重複文件")
    print("="*60)

    for q in GLOBAL_QUESTIONS:
        answer, cited = answer_single_question(
            q, embedder, collection, client, args.model, args.top_k
        )
        qa_pairs.append((q, answer))
        all_cited.update(cited)

    skill_body = synthesize_skill_md(qa_pairs, client, args.model)

    domain, agent_type = infer_domain_and_agent(qa_pairs, client, args.model)
    metadata_block = build_metadata_section(domain, total_sources, agent_type)
    skill_body_with_meta = inject_metadata_after_h1(skill_body, metadata_block)

    source_section = build_source_references(all_cited)

    final_md = skill_body_with_meta.strip() + "\n\n" + source_section + "\n"

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(final_md)

    unique_files = {s.split(" (Chunk")[0] for s in all_cited}
    print("\n" + "="*60)
    print(f"✅ 已輸出 skill.md 至: {args.output}")
    print(f"   彙整問答：{len(qa_pairs)} 組")
    print(f"   引用 chunk：{len(all_cited)} 個")
    print(f"   涵蓋來源檔案：{len(unique_files)} 份")
    print("="*60)


if __name__ == "__main__":
    main()
