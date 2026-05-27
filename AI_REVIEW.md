# HW3 AI 評分回饋

> 本份回饋由 **Anthropic Claude Opus 4.7** 與 **OpenAI GPT-5.5** 兩個獨立模型，分別針對 Phase 1 與 Phase 2 程式碼與文件進行評分；最終加權分數為兩模型平均後，進行全班性的 Normalize，以及老師加分。

---

## 本次公布的兩個分數

本次 HW3 作業，每位同學會公布**兩個分數**，計算方式為下列三個候選值中**最高的兩個**：

| 候選 | 來源 | 計算 | 值 |
|---|---|---|---:|
| A | Max(Phase 1 兩模型評分) | max(Opus 86.5, GPT 92.5) | 92.50 |
| B | Max(Phase 2 兩模型評分) | max(Opus 85.4, GPT 87.5) | 87.50 |
| C | 最終加權分數 | 詳見下方加權計算 | 92.66 |

### 你的兩個分數

### **92.66**　／　**92.50**

_（C：最終加權分數 = 92.66；A：Max(Phase 1 兩模型評分) = 92.50）_

## 最終加權分數（候選 C）的計算

| 來源 | Phase 1 (40%) | Phase 2 (60%) | 加權平均 |
|---|---:|---:|---:|
| Anthropic Claude Opus 4.7 | 86.5 | 85.4 | 85.82 |
| OpenAI GPT-5.5 | 92.5 | 87.5 | 89.50 |
| **兩模型平均** | **89.5** | **86.4** | **87.66** |
| 老師加分 (professor_bonus) | | | +5 |
| **最終分數** | | | **92.66** |

## 計算公式

```
Phase 1 平均  = (Opus_P1 + GPT_P1) / 2
Phase 2 平均  = (Opus_P2 + GPT_P2) / 2
加權平均      = 0.4 × Phase 1 平均 + 0.6 × Phase 2 平均
Normalize 後  = 依全班分布做校準
最終分數      = min(100, Normalize 後 + 老師加分)
```

## 專案基本資料

- **github_id**：`Au-Neptune-1`
- **領域**（Haiku 抽取）：`ai_image_generation_evolution`
- **領域分類**（taxonomy）：`academic_research_papers`
- **Phase 1 CI**：✓ 通過
- **Phase 2 CI**：✓ 通過

## Phase 1 分項評語（40% 權重）

| 面向 | Opus 分數 | Opus 評語 | GPT 分數 | GPT 評語 |
|---|---:|---|---:|---|
| 資料收集 | 78 | data/raw 收錄 41 份 arXiv PDF，超過 phase1_high_anchor 的 30+ 篇門檻，主題聚焦 2020–2026 AI 影像生成且時間範圍清楚。README 1.2 節以表格列出規模、格式、時間與領域涵蓋。扣分點：格式僅 .pdf 單一類型（雖宣稱支援 .txt 但實際無），且授權合規訊號 missing，未明確列出每篇論文的 arXiv ID 與 license（CC-BY 或 arXiv 條款），低於 high_anchor 對授權揭露的要求。 | 86 | data/raw 有 41 份 arXiv PDF，數量達高標；README 清楚界定 2020–2026 影像生成論文範圍與技術主題。缺點是僅單一 PDF 格式且授權說明不足。 |
| RAG 系統完整度 | 88 | data_update.py 使用 PyMuPDF blocks 加雙欄座標排序解決 arXiv 排版痛點，chunking 800/200 含軟邊界避免切斷英文字，embedding 與 ChromaDB 寫入完整。rag_query.py 實作多層檢索：query rewriting、translation、query expansion 拆 3 子問題、agentic router metadata filter、multi-query top_k=4 去重，並透過 LiteLLM/OpenAI client 整合 gemini-2.5-flash，citation 採 [Chunk X] 行內引用。明顯超越 phase1_high_anchor。未實作 reranking 為小缺。 | 94 | data_update.py 具 PyMuPDF 雙欄抽取、800/200 chunk、SentenceTransformer embedding 與 Chroma 寫入；rag_query.py 有翻譯、擴展、metadata routing、top-k 檢索、引用與 LiteLLM proxy 整合。 |
| 冪等性 | 92 | data_update.py 明確支援 --rebuild 旗標，會 shutil.rmtree processed_dir 與 CHROMA_PERSIST_DIR 從零重建。增量更新採 MD5 file_hash 寫入 chunk metadata，比對既有 hash 相同則 skip、不同則 collection.delete(where=source) 後重建，邏輯完整且優於 mtime 方案。README 3.6 節清楚論述設計理由。屬 phase1_high_anchor 之上水準。 | 96 | data_update.py 支援 --rebuild 清 processed 與 ChromaDB，並以 MD5 hash 比對檔案，未變更則跳過、變更則刪除舊 chunks 後重建，增量更新設計完整。 |

## Phase 2 分項評語（60% 權重）

| 面向 | Opus 分數 | Opus 評語 | GPT 分數 | GPT 評語 |
|---|---:|---|---:|---|
| 資料收集深度 | 80 | data/raw 共 41 份 arXiv PDF，符合 academic_research_papers anchor 的 30+ 篇門檻，時間範圍 2020–2026 涵蓋 DDPM 到 Flux/Sora 脈絡清晰（README §1.2）。但來源僅單一 arXiv PDF 類型，缺乏多樣性，且 license_compliance_signal 為 missing，README 未明確標註各論文授權條款，僅以「arXiv 學術論文」一語帶過，這是與 high anchor 的主要落差。 | 80 | data/raw 有 41 份 arXiv PDF，規模達標且主題聚焦 AI 影像生成演進；但來源類型單一，README §1.2 未清楚列授權、arXiv ID 或逐篇來源表。 |
| skill.md 品質 | 85 | skill.md 9 個必要章節齊全，Core Concepts 列出 8 項核心概念（World Models、Diffusion、GAN、VAE、Transformer、Latent Space、INR、Alignment）皆附英文與簡述；Key Entities 含具體研究者（Goodfellow、Ho、Peebles）、公司產品（Sora、SD3、Flux）；Source References 列出 28 份實際被引用檔案，可溯源。內容深度達 thorough，惟部分 Methodology 段落仍偏survey 式描述，具體數值與比較表較少。 | 84 | skill.md 必要章節齊全，§Core Concepts 有 8 項、§Key Entities 含人物/公司/模型；但部分內容偏泛化，DiT/Flow Matching 等主題未成核心條目，缺段內具體論文引用。 |
| README 設計決策 | 92 | README 21k 字逐節回答 chunking（§3.1 軟邊界＋雙欄 PyMuPDF heuristic）、embedding（§3.2 純英文 vs 多語取捨＋翻譯層）、vectorDB（§3.3 對比 pgvector/Qdrant/FAISS）、retrieval（§3.4 四階段含 Agentic Router）、prompt（§3.5）、idempotency（§3.6 MD5 vs mtime）、skill_builder（§3.7 10 題覆蓋表），每節皆有具體痛點與取捨論證，遠超 high anchor。 | 95 | README §3 詳細回答 chunking、embedding、ChromaDB、retrieval、prompt、idempotency 與 skill_builder 全域問題，且多有替代方案與取捨說明。 |
| skill_builder 品質 | 85 | skill_builder.py 設計 10 個 domain-agnostic 全域問題，覆蓋 Overview/Concepts/Trends/Entities/Methodology/Tools/Gaps/BestPractices/Tradeoffs/Metrics 完整光譜（GLOBAL_QUESTIONS）；每題完整跑 translate→expand→retrieve→generate pipeline 並由 synthesizer LLM 彙整，非機械拼貼；另以輕量 LLM call 推斷 domain/agent_type 注入 Metadata，整合度高。 | 91 | skill_builder.py 設計 10 個全域問題，涵蓋 Overview、Concepts、Trends、Entities、Methodology、Gaps；先多次 RAG 問答再由 LLM synthesize，非單純拼貼。 |

## 整體評語

### 亮點

- 雙欄排版還原啟發式解決arXiv論文結構問題
- 多層檢索策略(翻譯+擴展+路由)提升中文問題精準度
- 完整冪等性設計(MD5+metadata)確保增量更新穩定性
- 系統Prompt嚴格規範融合式生成與行內引用
- skill.md涵蓋9個必需章節且內容深度充分

### 改進建議

- 未導入reranking,依賴query expansion補償
- 翻譯層增加LLM call成本與延遲
- ChromaDB規模限制,擴展至大型知識庫需遷移
- skill.md自動生成依賴LLM品質,無人工審核
- 缺少Docker支援,本地環境依賴管理複雜

---

_本份評分回饋由自動化評分管線（Anthropic + OpenAI 雙模型獨立評分 → 平均 → rescale → 老師加分）產出，僅作為個人學習參考。若對評分有疑問請於課堂或 office hour 提出。_
