### AI 繪圖與視覺生成核心知識庫（41 筆）

#### 一、 基礎與理論體系 (Foundational) - 4 筆
1. **Denoising Diffusion Probabilistic Models (Ho et al., NeurIPS 2020)** [`ddpm_2020.pdf`]：以馬可夫鏈定義前向加噪與反向去噪過程，奠定所有現代擴散模型的數學基礎。
2. **High-Resolution Image Synthesis with Latent Diffusion Models / Stable Diffusion (Rombach et al., CVPR 2022)** [`ldm_stable_diffusion_2022.pdf`]：將擴散過程移至潛在空間，解決高解析度生成算力瓶頸，催生 Stable Diffusion 系列。
3. **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow (Liu et al., ICLR 2023)** [`rectified_flow_2023.pdf`]：以線性插值路徑取代隨機擴散，為 Flux 與 SD3 的 Flow Matching 架構鋪平道路。
4. **DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation (Ruiz et al., CVPR 2023)** [`dreambooth_2022.pdf`]：個人化微調始祖，以少量參考圖像將特定物件嵌入模型權重，定義了後續 LoRA 微調的核心思路。

#### 二、 現代架構主流 (Modern SOTA: 2023 - 2024) - 8 筆
5. **Scalable Diffusion Models with Transformers / DiT (Peebles & Xie, ICCV 2023)** [`dit_2023.pdf`]：以 Transformer 取代 U-Net 作為擴散模型骨幹，奠定 Flux、Sora 等現代大規模生成架構的基礎。
6. **Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models (arXiv:2402.17177, 2024)** [`sora_review_2024.pdf`]：系統性解析 OpenAI Sora 的 Spacetime Patches 技術，涵蓋視頻時空建模與 DiT 架構的實際應用。
7. **Demystifying Flux Architecture (arXiv:2507.09595, 2025)** [`flux_demystified_2025.pdf`]：對 FLUX.1 12B 參數模型進行完整逆向工程解析，詳述 Flow Matching 與雙流 Transformer 的實作細節。
8. **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis / Stable Diffusion 3 (Esser et al., arXiv:2403.03206, 2024)** [`sd3_2024.pdf`]：解析 MM-DiT 架構如何處理文字與圖像的聯合嵌入，引入多模態 Transformer 設計範式。
9. **Kling-Omni Technical Report (Kuaishou/快手, arXiv:2512.16776, 2024)** [`kling_omni_2024.pdf`]：快手可靈的完整技術報告，涵蓋 3D 時空聯合注意力機制與高效視頻生成的工程實踐。
10. **CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer (THUDM/智譜 AI, arXiv:2408.06072, 2024)** [`cogvideox_2024.pdf`]：解析 3D RoPE 旋轉位置編碼在視頻生成中的應用，以及專家 Transformer 如何處理時空一致性。
11. **Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding (Tencent, arXiv:2405.08748, 2024)** [`hunyuan_dit_2024.pdf`]：騰訊開源大規模 DiT 模型，深度優化中文語意理解能力，支援多解析度高品質生成。
12. **Emu3: Next-Token Prediction is All You Need (BAAI, arXiv:2409.18869, 2024)** [`emu3_2024.pdf`]：拋棄擴散範式，以 LLM 的「下一 Token 預測」統一圖像、視頻與文字生成的新路徑。

#### 三、 2025-2026 尖端前沿 (Cutting Edge) - 8 筆
13. **FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space (Black Forest Labs, arXiv:2506.15742, 2025)** [`flux_kontext_2025.pdf`]：探討 Flow Matching 在上下文感知圖像生成與編輯的統一框架，支援 Multi-Reference 多參考圖像輸入。
14. **Kling-Omni Technical Report — Physical-Aware Video Synthesis (Kuaishou, arXiv:2512.16776, 2024)** [`kling_omni_2024.pdf`]：深入解析可靈如何在生成流程中模擬物理規律（重力、流體動力學），實現物理感知的視頻合成。
15. **Seedream 2.0: A Native Chinese-English Bilingual Image Generation Foundation Model (ByteDance/Jimeng, arXiv:2503.07703, 2025)** [`seedream2_2025.pdf`]：解析字節跳動如何透過多輪 RLHF 提升圖像的美學表現與「構圖邏輯」，並整合雙語 LLM 文字編碼器。
16. **Unified Multimodal Understanding and Generation Models: Advances, Challenges, and Opportunities (arXiv:2505.02567, 2025)** [`unified_multimodal_gen_2025.pdf`]：分析原生多模態生成如何統一圖文理解與生成，探討取消獨立 Text-Encoder、轉向統一 Token 流的技術路徑。
17. **FAIL: Flow Matching Adversarial Imitation Learning for Image Generation (arXiv:2602.12155, 2026)** [`fail_flow_alignment_2026.pdf`]：以對抗性模仿學習取代 RLHF/DPO，解決 Flow Matching 模型在複雜場景生成中的分佈偏移問題。
18. **Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification (arXiv:2506.10634, 2025)** [`symmflow_2025.pdf`]：對稱式流量匹配技術，在同一框架內統一處理圖像生成與語意分割，支援雙向一致性學習。
19. **OmniGen: Unified Image Generation (VectorSpaceLab, arXiv:2409.11340, 2024)** [`omnigen_2025.pdf`]：提出統一框架，無需 ControlNet 即可在單一模型中執行文生圖、填充、修圖與風格遷移。
20. **DiTS: Multimodal Diffusion Transformers Are Time Series Forecasters (arXiv:2602.06597, 2026)** [`dits_forecasting_2026.pdf`]：將擴散 Transformer 跨界應用於多模態時間序列預測，探索視覺生成技術在世界模擬中的延伸潛力。

#### 四、 系統綜述與全局視野 (Surveys) - 4 筆
21. **Text-to-Image Diffusion Models in Generative AI: A Comprehensive Survey (arXiv:2308.09116, 2023)** [`t2i_survey_2023.pdf`]：對 DDPM 至 Stable Diffusion 時期的文生圖技術進行完整梳理，涵蓋架構、訓練與評估方法。
22. **Efficient Diffusion Models: A Survey (arXiv:2502.06805, 2025)** [`efficient_dit_survey_2025.pdf`]：系統性綜述 Transformer 取代 U-Net 的演進過程，涵蓋 DiT 系列架構的效率優化與規模擴展策略。
23. **Simulating the Visual World with Artificial Intelligence: A Roadmap (arXiv:2511.08585, 2025)** [`visual_world_simulation_2025.pdf`]：追蹤視覺生成從「像素模擬」轉向「物理世界模擬」的四代演進，描繪世界模型的技術路線圖。
24. **Trustworthy Text-to-Image Diffusion Models: A Timely and Focused Survey (arXiv:2409.18214, 2024)** [`trustworthy_t2i_survey_2024.pdf`]：聚焦六大可信度面向（魯棒性、公平性、安全性、隱私、事實性、可解釋性），對生成模型風險進行系統性評估。

#### 五、 評估指標與數據質量 (Evaluation & Metrics) - 3 筆
25. **ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation (arXiv:2304.05977, 2023)** [`imagereward_2023.pdf`]：透過大規模人類偏好標註訓練評分模型，為 RLHF 在圖像生成中的應用提供可靠的獎勵信號。
26. **GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment (arXiv:2310.11513, 2024)** [`geneval_2024.pdf`]：解決 CLIP Score 無法評估「物體數量與相對位置」的問題，提出基於物件的組合性評估框架。
27. **Rethinking FID: Towards a Better Evaluation Metric for Image Generation (arXiv:2401.09603, 2024)** [`rethinking_fid_2024.pdf`]：系統性分析 FID 的三大缺陷（Inception 特徵偏差、常態假設、樣本複雜度），並提出 CMMD 等替代方案。

#### 六、 操控、微調與實務 (Control & Personalization) - 4 筆
28. **LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., arXiv:2106.09685, 2022)** [`lora_2022.pdf`]：以低秩矩陣分解大幅降低微調參數量，成為 AI 繪圖領域最廣泛使用的個人化微調技術。
29. **IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models (arXiv:2308.06721, 2023)** [`ip_adapter_2023.pdf`]：透過解耦的交叉注意力實現「以圖生圖」，不影響主模型權重即可注入圖像風格與內容控制。
30. **InstantID: Zero-shot Identity-Preserving Generation in Seconds (arXiv:2401.07519, 2024)** [`instantid_2024.pdf`]：無需訓練 LoRA 即可實現人臉身份保持生成，透過 IdentityNet 與 IP-Adapter 的組合達到零樣本角色一致性。
31. **InstantCharacter: Personalize Any Characters with a Scalable Diffusion Transformer Framework (Tencent, arXiv:2504.12395, 2025)** [`instantcharacter_2025.pdf`]：以可擴展的 DiT 適配器實現開放域角色個人化，解決 2025 年跨工作流角色外觀一致性問題。

#### 七、 安全、對齊與商業落地 (Safety & Applications) - 4 筆
32. **A Comprehensive Survey on Concept Erasure in Text-to-Image Diffusion Models (arXiv:2502.14896, 2025)** [`concept_erasure_survey_2025.pdf`]：系統性梳理從模型中精準移除特定概念（版權、有害內容）的各類技術，並分析各方法的魯棒性邊界。
33. **LaWa: Using Latent Space for In-Generation Image Watermarking (arXiv:2408.05868, 2024)** [`lawa_watermarking_2024.pdf`]：在潛在擴散模型的 VAE 潛在空間植入不可見水印，實現生成圖像的來源追蹤與版權歸屬。
34. **Artificial Intelligence in Creative Industries: Advances Prior to 2025 (arXiv:2501.02725, 2025)** [`ai_creative_industries_2025.pdf`]：全面梳理 AI 生成工具（文生圖、視頻、音頻）在設計、廣告與媒體產業中的工作流整合現況與挑戰。
35. **Alignment and Safety of Diffusion Models via Reinforcement Learning and Reward Modeling: A Survey (arXiv:2505.17352, 2025)** [`diffusion_alignment_survey_2025.pdf`]：對比 RLHF 與 DPO 在視覺生成模型中的優劣，分析防範越獄攻擊與提升美學對齊的最新方法。

#### 八、 操控與人機互動 (Prompt Engineering & Editing) - 3 筆
36. **Prompt-to-Prompt Image Editing with Cross Attention Control (Hertz et al., ICLR 2023)** [`prompt_to_prompt_2023.pdf`]：透過操控 Cross-Attention Map 實現語意精確的文字驅動圖像編輯，揭示 Prompt 權重與空間布局的映射關係。
37. **InstructPix2Pix: Learning to Follow Image Editing Instructions (Brooks et al., CVPR 2023)** [`instructpix2pix_2023.pdf`]：以合成訓練數據（Prompt-to-Prompt）訓練模型理解「讓這張圖看起來像冬天」等自然語言指令，開創指令式圖像編輯。
38. **BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion (arXiv:2403.06976, 2024)** [`brushnet_2024.pdf`]：解決傳統遮罩擴散的接縫問題，以雙分支架構實現語意感知的高品質填充與擴圖，保持生成區域與原圖一致性。

#### 九、 範式轉移：從擴散到 Token (The Future Paradigm) - 1 筆
39. **Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs (arXiv:2401.11708, 2024)** [`mastering_t2i_recaptioning_2024.pdf`]：深度對比自回歸與擴散模型的生成路徑差異，解析 LLM 輔助規劃如何提升複雜場景的語意一致性。

#### 十、 數據工程：AI 的燃料 (Data Engineering & Quality) - 2 筆
40. **Improving Image Generation with Better Captions / DALL-E 3 Technical Report (OpenAI, 2023)** [`dalle3_recaptioning_2023.pdf`]：揭示合成標註（Synthetic Captioning）的革命性影響——用 VLM 重新標註訓練數據，是 Flux 等模型大幅提升提示詞遵循度的核心秘密。
41. **Diffusion Model Alignment Using Direct Preference Optimization (Wallace et al., CVPR 2024)** [`diffusion_dpo_alignment_2024.pdf`]：將 DPO 從語言模型遷移至擴散模型，以人類偏好數據直接優化生成策略，無需顯式獎勵模型。
