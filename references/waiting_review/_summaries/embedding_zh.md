# Embedding 主題摘要（中文整理）

2026-04-29 由 abstract + 全文 PDF 整理。2026-05-05 套用 5 條硬性需求重新分桶；同日 Phase C lit-monitor 新關鍵字搜出 Umeda-Kameyama 2021/2024 兩篇加入。

標籤：`[P]` = 有 PDF，`[T]` = 有 .txt，`[-]` = 只 abstract。

5 條硬性需求（anchor: Umeda-Kameyama 2021、Cenacchi 2025、Chien 2023、Asgarian 2019、Taati 2019、Matsuda 2025）：
- **R1** — Cohort 直接含 AD/MCI/dementia + 來源 + 具體 n
- **R2** — 純 face-image 模態（無 multimodal fusion）
- **R3** — 具體 AD vs HC 量化結果 + 統計檢定
- **R4** — face feature 來自 passive imaging / 自然觀察 / 既存 dataset frames
- **R5** — research question 是 AD biomarker 或 AD-cohort 上的 face-image methodology audit

排列：本檔按 review status — #1 核心、#2-5 延伸-induction、#6 邊緣。

共 **6 篇**（Grimmer 2021 Face Age Progression survey 已 archive）。

> **Backfill 紀錄**：Alsuhaibani 2023、Grimmer 2021 從 PDF 抽（pymupdf），Alsuhaibani 2024、Okunishi 2025 從 S2 API 抓回。Umeda-Kameyama 2021 manual fetch 從 Aging journal CDN 取得 OA PDF（CC BY 3.0），Umeda-Kameyama 2024 letter 從 Europe PMC 取得 metadata + abstract。

> **重要觀察**：4 篇延伸-induction 裡 3 篇（Alsuhaibani 2023/2024、FF 2024）都用 **I-CONECT 資料集**（NCT02871921，老人視訊對話），形同同一研究的不同 model 變體。

---


## ━━━━━━━━━━━━━━ 核心（1 篇 — 5/5 通過 5 條硬性需求 — 純臉部 + AD cohort + 具體結果 + 無刺激 + AD biomarker/audit）━━━━━━━━━━━━━━

## 1. [PT] **Umeda-Kameyama 2021** — Screening of Alzheimer's Disease by Facial Complexion Using Artificial Intelligence

> *Aging* 13(2):1765-1772（東京大學 Geriatric Medicine + Tokyo Metropolitan Geriatric Hospital；CC BY 3.0 OA）

**背景**：失智篩檢需 simple/non-invasive/inexpensive 工具。Perceived age 已知比 chronological age 更強相關於 cognitive function（Christensen 2009）；AD 病人有特定臉部 complexion（作者觀察）。本研究檢驗 AI 能否從臉部影像直接區分 cognitive impairment vs healthy。

**方法**：

- **Cohort**：**121 cognitive impairment + 117 cognitively sound participants**（Tokyo + Kashiwa cohorts，multi-site recruit；Most diagnosed by psychological tests + family information + lab data + brain MRI/CT + perfusion SPECT）。
- **影像**：**Front-on portrait images, neutral expression**（passive baseline，無 task；類似 Chien 2023 protocol）。Cropped square。共 **484 images**，10-fold split by participant。
- **Models**：5 個 DL 比較 — **Simple CNN、VGG16、ResNet50、SENet50、Xception**；2 optimizers (Adam, SGD)。VGG-Face / ImageNet pretraining。Image augmentation (rotation/shift/flip)。
- **Output**：binary classification probability = "Face AI score"。

**結果**：**Xception + Adam 最佳**：
- **Sensitivity 87.31%, Specificity 94.57%, Accuracy 92.56%**, **AUC 0.9717**
- **Face AI score vs MMSE: r = −0.599, p < 0.0001**（顯著）
- **Face AI score vs chronological age: r = 0.321, p < 0.0001**（弱）
- MMSE 與 Face AI score 的相關**顯著強於**與 chronological age 的相關（p<0.0001）
- 其他 model（SENet50/ResNet50/Simple CNN）loss 無法收斂 — task 比 SPECT 影像分類更難。

**結論**：DL（特別 Xception）能從 portrait 區分 mild dementia vs HC，paving the way 為 facial biomarker for dementia。

**Limitations**：單中心 + 484 images + Japanese front-on neutral only。

> **5/5 通過**：純 2D face portrait（neutral，passive 同 Chien 2023）+ 121 AD + 117 HC named cohort + sens/spec/acc/AUC 全給 + r=-0.599 p<0.0001 跟 MMSE + face-image AD biomarker 主題。**Anchor paper for face-photo + DL AD classification**（與 Chien 3D landmark / Cenacchi micro-dynamics / Asgarian-Taati landmark bias / Matsuda smile-monitoring 形成完整 anchor 集群）。

---


## ━━━━━━━━━━━━━━ 延伸-induction（4 篇 — R4 fail — paper 自行給 stimulus / interview / conversation）━━━━━━━━━━━━━━

## 2. [PT] **Alsuhaibani 2023** — Detection of Mild Cognitive Impairment Using Facial Features in Video Conversations

> *arXiv:2308.15624* (preprint)

MCI 早期偵測有助早期介入減緩 MCI→失智的進展。DL 演算法可達早期非侵入低成本偵測。

**方法**：用 **I-CONECT** 行為介入研究（NCT02871921）的資料 — 社交孤立老人 vs 訪員的 semi-structured 視訊對話。建立框架：
- **空間特徵**：convolutional autoencoder（holistic facial features）
- **時序資訊**：transformer

**結果**：在 I-CONECT 受試者上區分 MCI vs Normal Cognition (NC)。**結合 segment + sequence 資訊達 88% 準確率**（不用時序資訊只有 84%）。

**結論**：時空臉部特徵搭 DL 可作 MCI 偵測。

> R4 fail：I-CONECT semi-structured video chat 是 paper-arranged interview（同 Sun MC-ViViT）→ 延伸-induction。

## 3. [--] **Alsuhaibani 2024** — Mild Cognitive Impairment Detection from Facial Video Interviews by Applying Spatial-to-Temporal Attention Module

> *Expert Systems with Applications* (Elsevier)

承接 2023 preprint 的擴展版本。同 I-CONECT 資料集。

**新增**：**Spatial-to-Temporal Attention Module (STAM)**，把臉部特徵 + interaction 特徵融合。

**結果**：interaction 特徵讓預測表現比單純臉部特徵好。**綜合方法達 88% 準確率（不用時序 84%）**。

**結論**：時空臉部特徵在 DL 模型下對 MCI 偵測有區別力。

> 與 #2 結果幾乎一樣（都是 88% on I-CONECT），主要差別是 STAM 模組。實際上是 #2 的 published-journal 版本 + 模組改良。同 R4 fail。

## 4. [--] **FF 2024** — A Multimodal Cross-Transformer-based Model to Predict MCI Using Speech, Language and Vision

> *Computers in Biology and Medicine* (Elsevier)

MCI 是失智 / AD 的過渡期。AI 可早期偵測，但多數既有研究是 unimodal（只有語音或 prosody），近期研究顯示**多模態**更準。

**方法**：提出 **embedding-level co-attention 融合架構**整合三模態：
- **語音**（audio）
- **語言**（transcribed speech）
- **視覺**（facial videos）

用 **I-CONECT 資料集**（75+ 老人 semi-structured 對話）。Cross-Transformer 層內三模態 co-attention。

**結果**：
- **多模態 AUC 85.3%** vs unimodal 60.9% vs bimodal 76.3%
- 三模態顯著超越 baseline

**結論**：embedding-level fusion 抓住多模態互補資訊。

> R4 fail（I-CONECT chat）+ R2 fail（speech + language + vision multimodal）。Bucket 優先 R4 → 延伸-induction。

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

> R4 fail（conversation video 是 task-elicited）→ 延伸-induction。但 4 類特徵（含 face embeddings）對應到 project 的 4-direction 設計，方法論高度相關。

---


## ━━━━━━━━━━━━━━ 邊緣（1 篇 — R3 negative result，純 abstract，留作對照）━━━━━━━━━━━━━━

## 6. [--] **Umeda-Kameyama 2024** — Investigation of a Model for Evaluating Cognitive Decline from Facial Photographs Using AI

> *Geriatrics & Gerontology International* 24(Suppl 1):393-394（letter to editor，2 pages，OA via PMC11503600）

承接 Umeda-Kameyama 2021 的 follow-up letter。測試 **Microsoft Azure face API**（off-the-shelf "AI perceived age"）能否 replicate 2021 paper 的 cognitive decline screening 效果。

- 同一 121 AD cohort，再加 638 faces 擴充測試。
- **AI Azure age vs chronological age**: r = 0.480, p = 2.53×10⁻⁸（顯著）；**vs human-judged perceived age**: r = 0.791, p = 3.88×10⁻²⁷（強）。
- **但 AI Azure age 失敗未顯著比 chronological age 更強相關於 MMSE / Vitality Index**（不像 human-judged perceived age）。
- Azure sadness emotion 也未能偵測 depression（GDS15 比較失敗）。
- 推測 Azure 在「**老人 + 亞洲人種**」訓練資料偏弱（Azure 把老人 perceived age 低估約 20 歲，max 74 歲）。

**結論**：off-the-shelf face API 不適用於老人 / 非歐美 cohort；需要在 substantial 老人資料訓練的專屬 AI。

> **R3 negative result**（Azure 沒比 chronological age 強）→ 邊緣。但 methodology 有引用值 — 提醒**現成 face API 不適用於老人 / 非歐美 cohort**，是寫 paper 時 framing 用 dataset bias 的關鍵 reference。

---

## 摘要 — Embedding 主題分群

**核心**：
- **Umeda-Kameyama 2021** — 121 AD + 117 HC，front-on neutral portrait + Xception，acc 92.56% / AUC 0.9717 / r=-0.599 with MMSE。Anchor for face-photo DL AD detection。

**延伸-induction**（I-CONECT cluster）：
- **Alsuhaibani 2023/2024** — face-only video DL on I-CONECT 達 88% acc
- **FF 2024** — speech+language+face 三模態 fusion AUC 85.3%（同 I-CONECT）
- **Okunishi 2025** — 4 類臉部特徵（含 face embeddings）對失智 AUC 0.933、MCI 0.889

**邊緣**：
- **Umeda-Kameyama 2024** — Azure face API 在老人/亞洲 cohort 失敗（reference for off-the-shelf API 限制）

**已 archive**：
- **Grimmer 2021** — Face Age Progression GAN survey（影像合成不是測量，off-topic）

**Dataset 觀察**：
- **I-CONECT (NCT02871921)** 在延伸-induction 裡是主要 benchmark — 4 篇有 3 篇用它，引文應 cluster
- **Umeda-Kameyama Tokyo cohort（121 AD + 117 HC）** 是核心 reference — 跟我們 cohort 設計呼應
- 公開 face-image AD/MCI dataset 稀少，未來投稿可考慮把我們 cohort 對位至 Umeda 或 I-CONECT
