# Age 主題摘要（中文整理）

2026-04-29 由 abstract + 全文 PDF 整理。標籤：`[P]` = 有 PDF 全文，`[-]` = 只 abstract。

共 9 篇（Silva 2020「重建外科臉部對稱中線」已 archive）。

---

## 1. [-] **C 2023** — Detecting Dementia from Face-Related Features with Automated Computational Methods

AD 是隨年齡好發的失智類型，目前無法治癒，全球老化使早期篩檢日益重要。傳統腦掃描或精神測驗成本高、病人壓力大、常延誤介入。研究多探討語言在失智偵測的角色，較少關注臉部特徵。本文用 PROMPT 失智訪談視訊資料集，抽三類特徵：**face mesh、HOG（梯度直方圖）、AU（動作單元）**，訓練 ML 與 DL 模型。**HOG 達 79% 最高準確率，AU 71%，face mesh 66%**。結論：臉部相關特徵是自動失智偵測的潛在關鍵指標。

---

## 2. [-] **C 2026** — Integrated Analysis of Cerebral Small Vessel Disease and Facial Soft-Tissue Markers in the AD Continuum

**目的**：探討腦小血管病變（CSVD）與臉部軟組織量化測量在 AD 連續譜的整合關係，把周邊肌肉健康作為系統性虛弱 + 神經退化的 biomarker。

**方法**：3T 腦 MRI 67 位（AD=45, MCI=22）；CSVD 用 STRIVE + Fazekas/Potter 量表；臉部軟組織量化咀嚼肌、舌肌、顳肌厚度、脂肪浸潤（Mercuri 量表），透過 T1 半自動分割；CCA 探討中樞-周邊關係。

**結果**：
- AD 組 MMSE 顯著低（23.2 vs 28.2, p<0.0001）
- 中腦 PVS 是 AD 最強預測因子（p=0.003）
- AD 組咀嚼肌體積顯著低（p=0.0273）、脂肪浸潤高（p=0.025），支持局部肌少症
- CSVD 負荷與軟組織狀態多變量正相關（r=0.51, p=0.015）

**結論**：咀嚼肌體積/品質可作為 AD 非侵入性周邊 biomarker，補強傳統影像分層。

---

## 3. [-] **H 2023** — A Deep Learning Approach to Predict Chronological Age

研究者轉向從臉預測年齡（因臉部辨識應用多）。AD 主要與年齡相關，但臉部年齡估計受形狀/姿勢/尺寸變異影響準確度。本文提出用**眼睛色彩強度 + CNN 集成**做即時年齡預測，先用分割演算法從視訊或影像抽出眼睛。Kaggle 資料 4-59 歲共 27 萬張、約 2GB。**MSE ±8.69 年，準確率 97.29%**。結論：眼睛色彩強度對年齡預測高度有效。

> 註：這篇 AD 連結偏弱，主要是 chronological age 預測方法論。

---

## 4. [P] **LM 2025** — Potential of Facial Biomarkers for AD and OSA in Down Syndrome and General Population

唐氏症（DS）三染色體 21 與 AD、阻塞型睡眠呼吸中止（OSA）風險高度相關。傳統診斷（CSF、PSG）侵入性且對 DS 困難。本研究評估**臉部形態作為非侵入 biomarker**：從 MRI 抽 131 位 DS + 216 位歐倍體對照（含 AD、OSA 病例）的 3D 臉部模型，註冊 21 個 landmark 做 Procrustes ANOVA/MANOVA。

**結果**：
- DS vs EU 顯著臉形差異，性別依賴 + DS 老化變化異常（女性更明顯）
- **臉形與 Aβ1-42/Aβ1-40 比值（AD 關鍵 biomarker）相關**
- DS 校正後 AD 組差異不顯著，但 EU 族群可偵測到
- OSA 中臉形與 AHI（呼吸中止指數）相關，DS 嚴重 OSA 者臉形與無 OSA 者不同

**結論**：臉形態是 AD/OSA 偵測管理的非侵入 biomarker 候選。

---

## 5. [P] **Xu 2024** — Facial Aging, Cognitive Impairment, and Dementia Risk

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

---

## 6. [-] **Y 2020** — Cognitive Function Has a Stronger Correlation with Perceived Age than with Chronological Age

**目的**：perceived age（觀察者依臉部外觀判定的年齡）已知為老化 robust biomarker（預測存活、telomere、DNA 甲基化、頸動脈硬化、骨骼狀態）。本研究檢驗失智指標（總體認知、活力、憂鬱、自理）與 perceived age vs chronological age 哪個相關更強。

**方法**：東大附醫老年醫學科 124 位疑似認知衰退病人，做 MMSE、活力指數、GDS-15、IADL、Barthel；5 位老醫師 + 5 位臨床心理師依照片判定 perceived age。

**結果**：10 位評分者 ICC=0.941；Steiger's test 顯示 **perceived age 與 MMSE（女性）+ 活力指數的相關顯著強於 chronological age**，但 GDS-15/IADL/Barthel 無此差異。

**結論**：perceived age 為認知評估的可靠 biomarker。

---

## 7. [-] **Y 2021** — Screening of AD by Facial Complexion Using AI

失智發生率高但簡單、非侵入、低成本篩檢方法仍待開發。本研究測試 **AI 從臉部影像區分 CI vs 正常**。121 位 CI + 117 位健康；測 5 種 DL 模型 × 2 種 optimizer，二元分類為「Face AI score」。

**Xception + Adam 最佳：sens 87.31%、spec 94.57%、acc 92.56%、AUC 0.9717**。

Face AI score 與 MMSE 顯著相關（r=-0.599, p<0.0001），亦與 chronological age 相關（r=0.321），**但 MMSE 比 age 顯著相關更強**（p<0.0001）。

**結論**：DL（如 Xception）能區分輕度失智 vs 健康者臉部，為臉部 dementia biomarker 鋪路。

---

## 8. [-] **Z 2020** — Do AD Patients Appear Younger than Their Real Age?

**引言**：AD 最主要風險是老化，老化也影響外貌。臨床觀察提示 AD 患者「看起來比實齡年輕」。

**方法**：50 位早期 AD vs 50 位年齡性別配對對照；高解析度正面視訊（清晰光源），MMSE 中自然對話 + 自發微笑 + 靜態影像。年齡估計兩法：（1）電腦深度 CNN 分類器逐 frame 估；（2）人工視覺評估。

**結果**：AD vs 對照人工 ME 無統計顯著（p=0.33），方向偏 AD 較年輕；**電腦版 AD 平均 ME 顯著低於對照（p=0.007）— AI 把 AD 估得比實齡年輕**。

**結論**：人類與 AI 都看到 AD 患者外觀年輕的傾向，機制不明。

> ⚠️ **與本 project 結論方向相反**：本 project 在 sliding-window 分層後仍見 ACS（HC）被估更年輕、P（AD）較接近實齡（delta_AD-HC ≈ −1 to −4 yrs across age windows），即 AD 相對 HC 看起來「老」。最可能源頭：(a) MiVOLO 對 Asian face 的 baseline shift；(b) ACS 是門診篩過的 well-preserved 志願者 vs P 為 frailty 明顯的 AD。

---

## 9. [-] **Zeylan 2020** — Do Alzheimer's Patients Appear Younger than Their Age? A Study with Automatic Face Analysis

臉部年齡估計具挑戰，特別對老年族群。從病人臉部影像估年齡是新議題。本論文（thesis）測假說：**AD 患者比實齡看起來年輕**。先提出訓練 + 正規化方案改進 DL 臉部年齡估計，以 APPA-REAL → UTKFace fine-tune 預訓 ImageNet 模型。對較老臉部預測準確度優於既有研究（FG-NET 60-69 歲組 MAE 8.14 年）。然後在 96 位（AD + 健康對照，64-87 歲）特殊資料集測試主假說。

**結果：殘差年齡估計顯著低估 AD 患者年齡，比健康對照顯著更多**，驗證假說。

> 同 Z 2020 的延伸（同作者群、可能是 thesis 版），結論一致 — 與本 project 方向相反，見 §8 註記。
