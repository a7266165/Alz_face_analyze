# Age 主題摘要（中文整理）

2026-04-29 由 abstract + 全文 PDF 整理；2026-05-06 套用 5 條硬性需求重新分桶（沿用 asymmetry / embedding / emotion 同框架）。

標籤：`[P]` = 有 PDF 全文，`[-]` = 只 abstract。

5 條硬性需求（anchor: Asgarian 2019、Taati 2019、Chien 2023、Cenacchi 2025、Umeda-Kameyama 2021、Matsuda 2025）：
- **R1** — Cohort 直接含 AD/MCI/dementia + 來源 + 具體 n
- **R2** — 純 face-image 模態（無 multimodal fusion）
- **R3** — 具體 AD vs HC 量化結果 + 統計檢定
- **R4** — face feature 來自 passive imaging / 自然觀察 / 既存 dataset frames（MMSE clinical observation 視同 passive，同 Asgarian/Taati）
- **R5** — research question 是 AD biomarker 或 AD-cohort 上的 face-image methodology audit

排列：本檔按 review status — #1-2 核心、#3-8 延伸（任一條 R 不通過）。

共 **8 篇**（Silva 2020「重建外科臉部對稱中線」已 archive；Umeda-Kameyama 2021 移到 `embedding_zh.md` 核心 #1，本檔不再重複收錄）。

> **跨檔關聯**：Umeda-Kameyama 系列共 3 篇橫跨兩個主題 — Y 2020（age 核心 #2）、2021（embedding 核心 #1）、2024 letter（embedding 邊緣 #6）。本檔的 **face-aging line**（perceived age vs cognition、AD-younger paradigm）跟 embedding 主題的 **face-photo + DL classification** 是 anchor 集群的不同 angle，但寫論文時常會被一起引用。

---


## ━━━━━━━━━━━━━━ 核心（2 篇 — 5/5 通過 5 條硬性需求 — face-aging × AD risk biomarker）━━━━━━━━━━━━━━

## 1. [P] **Xu 2024** — Facial Aging, Cognitive Impairment, and Dementia Risk

> *Alzheimer's Research & Therapy* 16:245 (2024)

**背景**：臉部老化、認知障礙、失智都是年齡相關狀況，但臉部年齡與**未來失智風險**的時序關係從未被系統性檢驗。

**目的**：探討臉部年齡（自覺/主觀 + 客觀測量）與認知障礙、失智風險的關係。

**方法**：
- **UK Biobank**：195,329 位 ≥60 歲，**自覺臉部年齡**
- **NHAPC**（中國老化人口營養健康計畫）：612 位 ≥56 歲，**客觀皺紋參數測量**
- UKB 用 Cox proportional hazards 前瞻性分析「自覺臉部年齡 vs 失智風險」HR 與 95% CI，校正年齡、性別、教育、APOE ε4、其他潛在干擾
- 線性 + logistic 迴歸做臉部年齡（主觀/客觀）vs 認知障礙的橫斷面關聯

**結果**：
- 中位追蹤 12.3 年，UKB 出現 5659 例失智
- **高 vs 低自覺臉部年齡的失智 HR = 1.61（95% CI 1.33–1.96, P-trend ≤ 0.001）**（完全校正後）
- UKB 也觀察到自覺臉部年齡 vs 認知障礙的關聯
- NHAPC 三個客觀皺紋參數與認知障礙風險增加相關（P-trend < 0.05）
- 特別是**魚尾紋（crow's feet）數量最高 vs 最低四分位的認知障礙 OR = 2.48（95% CI 1.06–5.78）**

**結論**：高臉部年齡在校正傳統失智風險因子後仍與認知障礙、失智及其亞型相關。臉部老化可能是老人認知衰退與失智風險的指標，有助早期診斷與管理年齡相關狀況。

> **5/5 通過**：UKB 195k 大 cohort + 12.3 yr Cox HR 1.61 + perceived-face-age 與 objective crow's feet 雙路徑 + 純臉部影像 (passive UKB photo) + AD biomarker。**Anchor for face-aging × dementia risk** — 是這條線目前 cohort 最大、follow-up 最長的研究。寫 paper framing「face age 是 AD biomarker」時的首選 reference。

---

## 2. [-] **Umeda-Kameyama 2020** — Cognitive Function Has a Stronger Correlation with Perceived Age than with Chronological Age

**目的**：perceived age（觀察者依臉部外觀判定的年齡）已知為老化 robust biomarker（預測存活、telomere、DNA 甲基化、頸動脈硬化、骨骼狀態）。本研究檢驗失智指標（總體認知、活力、憂鬱、自理）與 perceived age vs chronological age 哪個相關更強。

**方法**：東大附醫老年醫學科 124 位疑似認知衰退病人，做 MMSE、活力指數、GDS-15、IADL、Barthel；5 位老醫師 + 5 位臨床心理師依照片判定 perceived age。

**結果**：10 位評分者 ICC=0.941；Steiger's test 顯示 **perceived age 與 MMSE（女性）+ 活力指數的相關顯著強於 chronological age**，但 GDS-15/IADL/Barthel 無此差異。

**結論**：perceived age 為認知評估的可靠 biomarker。

> **5/5 通過**：124 cognitive decline outpatient cohort + 觀察者 perceived age + Steiger's test 顯著 + clinical photo passive + 認知衰退 biomarker。**Anchor for perceived-age line（Christensen 2009 family）**。是 Umeda-Kameyama 2021（embedding 核心 #1）的 prequel — 同團隊先驗證 perceived age 跟 MMSE 強相關，2021 才用 DL 取代人類觀察者。

---

## ━━━━━━━━━━━━━━ 延伸（6 篇 — 任一 R 不通過 — 留作不同 angle / cross-modality 對照）━━━━━━━━━━━━━━

## 3. [-] **Z 2020** — Do AD Patients Appear Younger than Their Real Age?

**引言**：AD 最主要風險是老化，老化也影響外貌。臨床觀察提示 AD 患者「看起來比實齡年輕」。

**方法**：50 位早期 AD vs 50 位年齡性別配對對照；高解析度正面視訊（清晰光源），MMSE 中自然對話 + 自發微笑 + 靜態影像。年齡估計兩法：（1）電腦深度 CNN 分類器逐 frame 估；（2）人工視覺評估。

**結果**：AD vs 對照人工 ME 無統計顯著（p=0.33），方向偏 AD 較年輕；**電腦版 AD 平均 ME 顯著低於對照（p=0.007）— AI 把 AD 估得比實齡年輕**。

**結論**：人類與 AI 都看到 AD 患者外觀年輕的傾向，機制不明。

> **R4 fail**：MMSE 中錄影 — MMSE = 記憶測驗 = task elicitation，依既定規則 R4 fail。Asgarian/Taati 雖然也是 MMSE 中錄影但他們是 method audit 走 R5 寬鬆通道；本篇是直接做 biomarker hypothesis（"AD 看起來年輕嗎"），無 R5 逃生口。仍是 face-aging × AD 重要 reference paper。
>
> ⚠️ **與本 project 結論方向相反**：本 project 在 sliding-window 分層後仍見 ACS（HC）被估更年輕、P（AD）較接近實齡（delta_AD-HC ≈ −1 to −4 yrs across age windows），即 AD 相對 HC 看起來「老」。最可能源頭：(a) MiVOLO 對 Asian face 的 baseline shift；(b) ACS 是門診篩過的 well-preserved 志願者 vs P 為 frailty 明顯的 AD。**討論章節需要 cite 並 contrast。**

---

## 4. [-] **Zeylan 2020** — Do Alzheimer's Patients Appear Younger than Their Age? A Study with Automatic Face Analysis

臉部年齡估計具挑戰，特別對老年族群。從病人臉部影像估年齡是新議題。本論文（thesis）測假說：**AD 患者比實齡看起來年輕**。先提出訓練 + 正規化方案改進 DL 臉部年齡估計，以 APPA-REAL → UTKFace fine-tune 預訓 ImageNet 模型。對較老臉部預測準確度優於既有研究（FG-NET 60-69 歲組 MAE 8.14 年）。然後在 96 位（AD + 健康對照，64-87 歲）特殊資料集測試主假說。

**結果：殘差年齡估計顯著低估 AD 患者年齡，比健康對照顯著更多**，驗證假說。

> **R4 fail**：同 Z 2020 setup（MMSE 中錄影 = task）。是 Z 2020 的 thesis 延伸（同作者群、結論一致），cite 時可同時引兩篇支撐 paradigm replication。⚠️ 與本 project 結論方向相反，見 §3 註記。

---


## 5. [-] **H 2023** — A Deep Learning Approach to Predict Chronological Age

研究者轉向從臉預測年齡（因臉部辨識應用多）。AD 主要與年齡相關，但臉部年齡估計受形狀/姿勢/尺寸變異影響準確度。本文提出用**眼睛色彩強度 + CNN 集成**做即時年齡預測，先用分割演算法從視訊或影像抽出眼睛。Kaggle 資料 4-59 歲共 27 萬張、約 2GB。**MSE ±8.69 年，準確率 97.29%**。結論：眼睛色彩強度對年齡預測高度有效。

> R1 fail：Kaggle 4-59 yo 大眾資料，無 AD cohort（且 4-59 歲 range 也跟 elderly AD 不重疊）→ 方法論參考。價值：眼睛色彩強度 + CNN ensemble 是 chronological age estimation 的 methodological alternative，可作為對照。

---


## 6. [-] **C 2023** — Detecting Dementia from Face-Related Features with Automated Computational Methods

AD 是隨年齡好發的失智類型，目前無法治癒，全球老化使早期篩檢日益重要。傳統腦掃描或精神測驗成本高、病人壓力大、常延誤介入。研究多探討語言在失智偵測的角色，較少關注臉部特徵。本文用 PROMPT 失智訪談視訊資料集，抽三類特徵：**face mesh、HOG（梯度直方圖）、AU（動作單元）**，訓練 ML 與 DL 模型。**HOG 達 79% 最高準確率，AU 71%，face mesh 66%**。結論：臉部相關特徵是自動失智偵測的潛在關鍵指標。

> R4 fail：PROMPT 失智訪談視訊是 paper-arranged interview（同 I-CONECT cluster）→ 延伸-induction。

---


## 7. [-] **C 2026** — Integrated Analysis of Cerebral Small Vessel Disease and Facial Soft-Tissue Markers in the AD Continuum

**目的**：探討腦小血管病變（CSVD）與臉部軟組織量化測量在 AD 連續譜的整合關係，把周邊肌肉健康作為系統性虛弱 + 神經退化的 biomarker。

**方法**：3T 腦 MRI 67 位（AD=45, MCI=22）；CSVD 用 STRIVE + Fazekas/Potter 量表；臉部軟組織量化咀嚼肌、舌肌、顳肌厚度、脂肪浸潤（Mercuri 量表），透過 T1 半自動分割；CCA 探討中樞-周邊關係。

**結果**：
- AD 組 MMSE 顯著低（23.2 vs 28.2, p<0.0001）
- 中腦 PVS 是 AD 最強預測因子（p=0.003）
- AD 組咀嚼肌體積顯著低（p=0.0273）、脂肪浸潤高（p=0.025），支持局部肌少症
- CSVD 負荷與軟組織狀態多變量正相關（r=0.51, p=0.015）

**結論**：咀嚼肌體積/品質可作為 AD 非侵入性周邊 biomarker，補強傳統影像分層。

> 延伸-multimodal：腦 MRI（CSVD 量表）+ 臉部軟組織 MRI（咀嚼肌/舌肌厚度）+ CCA 中樞-周邊整合 — face 只是其中一支 imaging stream，且都是 T1 MRI 切割（**非 2D 臉部 photo / video**）。R2 fail（multimodal MRI fusion）。「臉部軟組織 sarcopenia 是 AD 周邊 biomarker」這個 angle 補強傳統腦影像，引文時可作為 face-level biomarker 的另一條 support line。

---

## 8. [P] **LM 2025** — Potential of Facial Biomarkers for AD and OSA in Down Syndrome and General Population

唐氏症（DS）三染色體 21 與 AD、阻塞型睡眠呼吸中止（OSA）風險高度相關。傳統診斷（CSF、PSG）侵入性且對 DS 困難。本研究評估**臉部形態作為非侵入 biomarker**：從 MRI 抽 131 位 DS + 216 位歐倍體對照（含 AD、OSA 病例）的 3D 臉部模型，註冊 21 個 landmark 做 Procrustes ANOVA/MANOVA。

**結果**：
- DS vs EU 顯著臉形差異，性別依賴 + DS 老化變化異常（女性更明顯）
- **臉形與 Aβ1-42/Aβ1-40 比值（AD 關鍵 biomarker）相關**
- DS 校正後 AD 組差異不顯著，但 EU 族群可偵測到
- OSA 中臉形與 AHI（呼吸中止指數）相關，DS 嚴重 OSA 者臉形與無 OSA 者不同

**結論**：臉形態是 AD/OSA 偵測管理的非侵入 biomarker 候選。

> 延伸-multimodal：face shape（3D MRI）+ Aβ1-42/Aβ1-40 ratio（生化）+ AHI（睡眠監測）跨多支資料源，face 是其中一支。R2 fail（multimodal fusion）+ DS 校正後 AD 組差異不顯著（R3 partial fail）。但 face shape × Aβ 比值的相關有意義，留作 cross-modality / DS-cohort reference。

---

## 摘要 — Age 主題分群

**核心**（2）— face-aging × AD biomarker anchor：
- **Xu 2024** — UKB 195k face-aging × dementia HR 1.61 / OR 2.48（cohort 最大、follow-up 最長）
- **Umeda-Kameyama 2020** — 124 outpatients perceived age 與 MMSE 相關**顯著強於** chronological age（Christensen line classic）

**延伸**（6）：
- Z 2020（R4 fail：MMSE 中錄影；AD-younger paradigm 開創）
- Zeylan 2020（R4 fail：同 Z 2020 thesis 版）
- H 2023（R1 fail：Kaggle 4-59 yo 無 AD cohort，chronological age method）
- C 2023（R4 fail：PROMPT 失智訪談視訊 = task-driven interview）
- C 2026（R2 fail：腦 MRI + 咀嚼肌 MRI 多模態 fusion，非 2D photo）
- LM 2025（R2 fail：DS 3D MRI face + Aβ + AHI 多模態，且 R3 partial fail — DS 校正後 AD 不顯著）

**已 archive**：Silva 2020（重建外科臉部對稱中線，無 AD cohort）

---

## 跨主題 anchor 集群（共 10 篇 5/5）

| 主題 | Anchor papers | Angle |
|---|---|---|
| asymmetry | Chien 2023, Cenacchi 2025 | 3D landmark 對稱 / in-the-wild micro-dynamics |
| embedding | Umeda-Kameyama 2021 | front-portrait + DL classification |
| emotion | Asgarian 2019, Taati 2019, Matsuda 2025 | landmark bias audit / 真實照護 smile monitoring |
| **age** | **Xu 2024, Umeda-Kameyama 2020** | **face-aging × dementia risk** |

從原本 6 anchor 擴到 **8 anchor**（age 補了 2 篇 face-aging line），完整支撐 paper 的 4 個 angle。Z 2020 / Zeylan 2020 雖然是 face-aging × AD 重要 reference，但因 MMSE 中錄影 = task 不過 R4，列在延伸。
