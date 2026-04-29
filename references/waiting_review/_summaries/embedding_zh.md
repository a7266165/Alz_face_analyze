# Embedding 主題摘要（中文整理）

2026-04-29 由 abstract + 全文 PDF 整理。標籤：`[P]` = 有 PDF，`[T]` = 有 .txt，`[-]` = 只 abstract。

共 5 篇。

> **Backfill 紀錄**：4 篇空 abstract 都已補回 — Alsuhaibani 2023、Grimmer 2021 從 PDF 抽（pymupdf），Alsuhaibani 2024、Okunishi 2025 從 S2 API 抓回。

> **重要觀察**：5 篇裡 3 篇（Alsuhaibani 2023/2024、FF 2024）都用 **I-CONECT 資料集**（NCT02871921，老人視訊對話），形同同一研究的不同 model 變體。

---

## 1. [PT] **Alsuhaibani 2023** — Detection of Mild Cognitive Impairment Using Facial Features in Video Conversations

> *arXiv:2308.15624* (preprint)

MCI 早期偵測有助早期介入減緩 MCI→失智的進展。DL 演算法可達早期非侵入低成本偵測。

**方法**：用 **I-CONECT** 行為介入研究（NCT02871921）的資料 — 社交孤立老人 vs 訪員的 semi-structured 視訊對話。建立框架：
- **空間特徵**：convolutional autoencoder（holistic facial features）
- **時序資訊**：transformer

**結果**：在 I-CONECT 受試者上區分 MCI vs Normal Cognition (NC)。**結合 segment + sequence 資訊達 88% 準確率**（不用時序資訊只有 84%）。

**結論**：時空臉部特徵搭 DL 可作 MCI 偵測。

> 直接相關 — 等於用 face-only video 沒接 audio/language 也能達 88%。

---

## 2. [--] **Alsuhaibani 2024** — Mild Cognitive Impairment Detection from Facial Video Interviews by Applying Spatial-to-Temporal Attention Module

> *Expert Systems with Applications* (Elsevier)

承接 2023 preprint 的擴展版本。同 I-CONECT 資料集。

**新增**：**Spatial-to-Temporal Attention Module (STAM)**，把臉部特徵 + interaction 特徵融合。

**結果**：interaction 特徵讓預測表現比單純臉部特徵好。**綜合方法達 88% 準確率（不用時序 84%）**。

**結論**：時空臉部特徵在 DL 模型下對 MCI 偵測有區別力。

> 與 #1 結果幾乎一樣（都是 88% on I-CONECT），主要差別是 STAM 模組。實際上是 #1 的 published-journal 版本 + 模組改良。

---

## 3. [--] **FF 2024** — A Multimodal Cross-Transformer-based Model to Predict MCI Using Speech, Language and Vision

> *Computers in Biology and Medicine* (Elsevier)

MCI 是失智 / AD 的過渡期，National Institute of Aging 報告 MCI 患者進展失智風險高。AI 可早期偵測，但多數既有研究是 unimodal（只有語音或 prosody），近期研究顯示**多模態**更準。多模態融合方法是大挑戰（early/late fusion 難保留 inter-modal 關係）。

**方法**：提出 **embedding-level co-attention 融合架構**整合三模態：
- **語音**（audio）
- **語言**（transcribed speech）
- **視覺**（facial videos）

用 **I-CONECT 資料集**（75+ 老人 semi-structured 對話）。Cross-Transformer 層內三模態 co-attention。

**結果**：
- **多模態 AUC 85.3%** vs unimodal 60.9% vs bimodal 76.3%
- 三模態顯著超越 baseline

**結論**：embedding-level fusion 抓住多模態互補資訊，提供更準確、可靠的 MCI 預測。

> 跟 #1/#2 同 dataset 但加上 audio + language 模態。3 模態 AUC 85.3% < 純 face 88%（不同 metric，AUC vs acc 不能直接比）。

---

## 4. [PT] **Grimmer 2021** — Deep Face Age Progression: A Survey

> *IEEE Access*

**Face Age Progression (FAP)** = 合成模擬老化效果的臉部影像，預測個人未來外觀。應用範圍：臉部辨識系統、forensic 調查、digital entertainment。深度生成網路顯著提升年齡合成影像品質（visual fidelity / ageing accuracy / identity preservation）。

近年文獻多需要系統性分類整理以識別 taxonomy、加速研究、減少重複。

**Survey 範圍**：比較分析近期 DL FAP 方法（成人 + 兒童老化），分為三種高階概念：
- **Translation-based** FAP
- **Condition-based** FAP
- **Sequence-based** FAP

並完整整理常用效能評估法、跨年齡資料集、未解決挑戰。

> 跟我們 project 較不直接相關 — 是「合成老化臉」的 GAN/conditional 方法 survey，不是「測量年齡」。除非未來想用 face age progression 做 longitudinal 模擬比對，否則 reference value 偏低。

---

## 5. [--] **Okunishi 2025** — Dementia and MCI Detection Based on Comprehensive Facial Expression Analysis From Videos During Conversation

> *IEEE Journal of Biomedical and Health Informatics*

開發失智的成本效益高的 digital biomarker 是迫切需要。多數研究探討用語音/自然語言偵測失智，但**用臉部視訊偵測失智的研究較少**，需要更深入研究。

**方法**：從受試者對話視訊抽**四類臉部表情特徵**：
1. **Action Units (AU)**
2. **emotion categories**（情緒類別）
3. **Valence-Arousal**（V/A）
4. **face embeddings**（臉部嵌入）

每個 frame 抽各類特徵後做統計彙整作為 feature，用**決策樹模型**預測。

**結果**：
- **失智偵測 AUC = 0.933**
- **MCI 偵測 AUC = 0.889**
- 統計分析顯示**失智者比健康者**正向情緒少、負向情緒多、valence + arousal 較低

**結論**：可作為失智、MCI 早期偵測的 **explainable screening tool**。

> 高度相關 — 跟我們 project 4 個面向（embedding/asymmetry/emotion/age）裡的 emotion + embedding 都直接重疊，且四類特徵融合 AUC 0.933 對失智很高。

---

## 摘要 — Embedding 主題分群

**直接相關 + 重要**：
- **Okunishi 2025** — 4 類臉部特徵（含 face embeddings）對失智 AUC 0.933、MCI 0.889，跟我們 4-direction 高度對齊
- **Alsuhaibani 2023/2024** — face-only video DL 對 MCI 88% acc（I-CONECT dataset）
- **FF 2024** — 三模態 cross-transformer fusion AUC 85.3%（同 I-CONECT）

**邊緣相關**：
- **Grimmer 2021** — Face Age Progression GAN survey（合成而非測量，relevance 低）

**Dataset 觀察**：
- **I-CONECT (NCT02871921)** 在 embedding 主題裡是主要 benchmark — 3/5 篇用它，未來引文應 cluster
- **公開 MCI/dementia face video dataset 稀少**，未來投稿可考慮把我們 cohort 與 I-CONECT 對位
