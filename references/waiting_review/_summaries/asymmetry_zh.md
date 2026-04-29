# Asymmetry 主題摘要（中文整理）

2026-04-29 由 abstract + 全文 PDF 整理。標籤：`[P]` = 有 PDF，`[T]` = 有 .txt 抽取，`[-]` = 只有 abstract（或無）。

共 11 篇（Ko 2021 / Pucciarelli 2018 dupe 已合併；Pucciarelli 2018 craniofacial 重建 thesis 已 archive）。

> **Backfill 紀錄**：Albaker 2023、Yoonesi 2025 的 abstract 為空，從 PDF 抽出文字補回 JSON（Chien 2023 沒 PDF 無法救）。

---

## 1. [PT] **Al-baker 2023** — Accuracy and Reliability of Automated 3D Facial Landmarking in Medical and Biological Studies — A Systematic Review

> *European Journal of Orthodontics* 45(4), 382–395

**背景**：3D 臉部 landmarking 已成臨床/生物應用基礎。手動標記耗時且累積誤差，自動化嘗試多但文獻稀少。

**目的**：評估目前 3D 臉部自動標記方法在醫學/生物研究中的準確性與可靠性，並與手動方法比較。

**方法**：2021 年 4 月電子 + 手動文獻檢索；只納入英文、評估自動 landmark 準確度的研究；QUADAS-2 評估品質；異質性高無法 meta-analysis，做敘述性合成。

**結果**：1002 篇篩選後納入 14 篇；標記數 10-29 個；自動 vs 手動平均差異 **0.67-4.73 mm**，**最佳是 DL 模型**；研究設計與報告品質普遍不佳（reference standard、人口選擇問題），可能 overfitting。

**結論**：跟手動相比，文獻中個別 landmark 的自動定位**仍不足以臨床使用**，需更多研究來開發匹配或超越 gold standard 的系統。

---

## 2. [--] **Al-baker 2025** — The Accuracy of Automated Facial Landmarking — A Comparative Study Between Cliniface Software and Patch-based CNN

承接 2023 review 的後續實證：**patch-based CNN 演算法的自動 landmark 偵測準確度令人滿意**，可用於 3D 臉部影像的臨床評估。**Cliniface 軟體在某些特定 landmark 偵測準確度受限**，限制其臨床應用。

> 同作者群延伸，從 review 走到比較實作。

---

## 3. [--] **Brown 2019** — The Face of Early Cognitive Decline? Shape and Asymmetry Predict Choice Reaction Time Independent of Age, Diet or Exercise

反應時間變慢是認知衰退指標，可早至 24 歲出現。本研究探討 **developmental stability**（用 **fluctuating asymmetry, FA** 衡量）能否獨立於年齡 + 生活型態（飲食、運動）預測認知表現。FA 是臉部相對最大邊的隨機形態偏離，已知與許多物種（含人類）健康、發病率、死亡率相關。

**方法**：88 位大學生，自陳魚類消費 + 運動量；3D 臉部掃描 + 認知測量。

**結果**：意外發現魚類消費**高**反而對應**較慢**反應時間；**臉部不對稱 + 多個臉形變異參數獨立於性別/年齡/飲食/運動預測較慢 choice reaction time**。

**結論**：建議未來縱向介入研究，針對發育期受 ontogenetic stress 影響的脆弱族群減緩早期認知衰退。

---

## 4. [--] **Chien 2023** — Analyzing Facial Asymmetry in Alzheimer's Dementia Using Image-Based Technology

> *Biomedicines* 11(10):2802

**沒有 abstract，沒有 PDF**（無 OA 也無法 backfill）。標題與本 project 直接相關（AD 失智的臉部不對稱影像分析），值得手動找 paper 補回，但目前資訊量為零。

---

## 5. [--] **Heinrich 2026** — AI-Based Angle Map Analysis of Facial Asymmetry in Peripheral Facial Palsy

周邊性面神經麻痺（PFP）造成顯著臉部不對稱 + 功能障礙，需要可靠客觀評估。本研究提出**全自動、無需參考影像**的臉部對稱量化法（用 AI 臉部 landmark 偵測）。

**方法**：198 位 PFP 病人共 **405 筆資料**，每筆 9 種標準臉部表情（靜態 + 動態）。AI 每張影像偵測 **478 landmark**，其中 225 對 paired landmark 計算局部不對稱角度。系統性評估識別出 **91 對高資訊量 landmark**（眼/鼻/口周圍為主），簡化分析又增加區別力，可做 region-specific 評估。

**結果**：Kruskal-Wallis H-test + Spearman 相關（**0.32-0.73, p<0.001**）對應臨床評分；對頭部旋轉強健；直觀全臉角度地圖可**不需參考影像**直接判讀。

**結論**：穩健、客觀、可視化的 PFP 臨床監測 / 嚴重度分級 / 療效評估框架。

> 不是 AD 領域，但 landmark-based 不對稱量化 + paired landmark 設計直接 inform 我們的方法論（特別是 method 是否需要 reference image）。

---

## 6. [PT] **Jafari 2022** — 3D Video Tracking Technology in the Assessment of Orofacial Impairments in Neurological Disorders

神經疾患早期常出現口面部肌肉動作 + 言語細微變化。視訊深度 AI 臉部分析模型有潛力作為客觀非侵入臨床工具。本論文（thesis）用 **V3 框架**（評估 digital biomarker 並落地臨床）來評估**自動視訊臉部分析系統**作為口面部運動的客觀評估工具。

**系統**：3D 攝影機 + AI 演算法，自動從病人做標準口面部任務的視訊**抽客觀、臨床可解釋的運動學特徵**。

**目標**：研究系統的分析效度與臨床效度，用於評估臨床族群口面部障礙的嚴重度。

> Thesis 形式，方法論偏向工程實作 + 臨床落地評估。

---

## 7. [--] **Ko 2021** — Changes in Computer-Analyzed Facial Expressions with Age

臉部表情隨年齡變化已知，但量化特性不明。本研究比較**56 位老人 vs 113 位年輕人**表情強度差異。實驗室拍攝 6 種基本情緒 + 中性的 posed 表情，用 **OpenFace** 量化強度。

**結果**：老人對某些負向情緒和中性表情**強度更強**；做表情時**用更多臉部肌肉**（橫跨各情緒）。

**結論**：助理解老化臉部表情特徵，為其他臉部辨識領域提供實證。

> 跟 emotion 主題很有重疊，但被分到 asymmetry，是 query 重疊導致。實際上是 OpenFace 量化老化 FE 變化的方法論研究。

---

## 8. [--] **Liu 2025** — Genetic Insights of Image-Based Traits: Analysis Pipeline for AI-based Phenotyping, Combined-GWAS, and Federated Learning with Application to the Human Face

影像衍生 phenotype 含豐富形態資訊，了解遺傳基礎對發育機制、複雜視覺特徵的遺傳變異、生醫/演化/法醫應用都重要。但現有方法在影像複雜度 + 遺傳複雜度兩方面都有限制；多 cohort 研究又受隱私法規限制。

**Pipeline**：(i) AI phenotyping 自動抽大量 endophenotype；(ii) **Combined-GWAS（C-GWAS）**找對應遺傳變異；(iii) **federated learning** 跨 cohort 訓練但不分享個別影像；(iv) **可解釋 AI** 視覺化遺傳效應。

**首例應用**：兩個歐洲 cohort（N=7,309）3D 臉部影像 + 基因資料，抽 **200 個臉部 endophenotype**，識別 **43 個顯著臉部相關 loci（含 12 個新發現）**，**70% 在獨立歐洲資料集（N=8,246）複製成功**。AI 視覺化展示這些 loci 對臉部各部位的影響。

**結論**：通用、保護隱私的影像複雜性狀遺傳分析框架。

> 直接相關 — 我們的 ArcFace embedding + landmark asymmetry 也是 image-based phenotype，他們的 federated learning 跟 C-GWAS 框架可能借鏡。

---

## 9. [PT] **Obrochta 2025** — Is the Human Face a Biomarker of Health? — A Scoping Review

普遍假設臉部特徵可作 developmental stability 與 physical/mental 健康的 biomarker，但研究結果混雜。**這是首篇探討臉部特徵（symmetry、averageness、sexual dimorphism）與健康各面向關聯的 scoping review**。

**方法**：Web of Science / MEDLINE / Scopus / Embase 檢索，遵 PRISMA-ScR。

**結果**：702 篇篩選後納入 21 篇，涵蓋臉部特徵 vs 心血管 / 免疫 / 氧化壓力 / cortisol / 生殖 / 認知 / 總體生理健康。

**結論：結果不一致** — 對「臉部特徵能否作 honest health indicator」**沒有定論**。**警告慎用臉部特徵作為個體健康/生物狀態的 biomarker**。

> 對 project 而言是重要的 cautionary review — 我們在臉部 biomarker 上做的任何 strong claim 都應引此 review 並 frame 在現有混雜文獻中。

---

## 10. [PT] **Yang 2025** — Motor Symptoms of Parkinson's Disease: Critical Markers for Early AI-Assisted Diagnosis

> *Frontiers in Aging Neuroscience*

PD 是常見神經退化疾患，早期診斷對減緩進展與優化治療關鍵。AI 提供新機會做早期偵測。研究顯示明顯動作症狀出現前，PD 已有可量化的細微動作異常。

**Review 範圍**：AI 早期 PD 偵測法基於各類動作症狀 — **眼動、臉部表情、言語、書寫、手指敲擊、步態**。整理特徵、現有資料、公開資料集、評估診斷模型效能與限制。

**討論**：研究法 + 挑戰（從實驗到臨床轉譯），提出未來方向以推動 AI 在 PD 診斷的臨床落地。

> 不是 AD-specific，但 PD 跟 AD 在 face-based motor symptom 角度有重疊（hypomimia、口面部 dyskinesia），可借鏡其方法論分類框架。

---

## 11. [PT] **Yoonesi 2025** — Facial Expression Deep Learning Algorithms in the Detection of Neurological Disorders: A Systematic Review and Meta-Analysis

> *BioMedical Engineering OnLine* 24:64

**背景**：神經疾患（從 AD 到 Angelman 症候群）全球負擔重。**臉部表情變化是跨疾患的常見症狀**，可能作為診斷指標。DL（特別 CNN）在偵測這些變化展現潛力。

**目的**：systematic review + meta-analysis 評估 DL 演算法偵測臉部表情變化以診斷神經疾患的效能。

**方法**：PRISMA2020；PubMed/Scopus/Web of Science 至 2024 年 8 月；28 篇納入；JBI checklist 評品質；I² 評異質性；subgroup analysis by disorder。

**結果**：24 篇 2019-2024 研究納入，疾患涵蓋**失智、Bell's palsy、ALS、PD**。
- **整體 pooled accuracy: 89.25% (95% CI 88.75-89.73%)**
- **失智: 99%、Bell's palsy: 93.7%**
- ALS、stroke 較低 (**73.2%**)

**結論**：DL（特別 CNN）有偵測神經疾患臉部表情變化的強大潛力，但需**標準化資料集** + **改進對 motor-related 條件的穩健性**。

> 與本 project 高度相關 — 失智 99% pooled accuracy 是強信號，需要查 24 篇納入研究的具體 accuracy 來源（可能存在發表偏差或單一資料集主導）。
