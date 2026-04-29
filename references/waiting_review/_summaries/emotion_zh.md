# Emotion 主題摘要（中文整理）

2026-04-29 由 abstract + 全文 PDF 整理。

標籤：
- `[P]` = 有 PDF / `[T]` = 有 .txt / `[-]` = 只 abstract
- `★` = 直接 AD/MCI/dementia × 臉部影像分析（高相關）
- `◇` = AD-adjacent（PD/FTD/其他 ND，方法論可借鏡）
- `○` = FER 方法論 / 資料集（generic，但 inform pipeline）
- `△` = 邊緣相關

共 **109 篇**。Backfill 紀錄：6 篇空 abstract 已從 PDF.txt 或 S2 API 補回。

---

## 1. ★ [PT] **A 2024** — Facial Emotion Expressivity in PD and AD

PD 與 AD 都有臉部表情障礙（hypomimia），但兩者比較未有實驗研究。**24 PD + 24 AD + 24 HC**，視訊錄製做 6 種情緒（怒/驚/嫌/喜/悲/恐）+ 中性。比較表情強度量化。

## 2. ◇ [--] **A 2024** — ML + Digital Biomarkers Detect Early ND

AD/PD 等 ND 早期偵測仍是難題。Review ML + digital biomarker（含臉部）的早期偵測方法、現況與挑戰。

## 3. ★ [PT] **Abe 2025** — Estimate Brain Health from Facial Expressions (Neuroaesthetics)

關聯**MRI 灰質體積（GMV）**與 4 種臉部表情（喜怒哀驚），訓練演算法從表情估腦健康並開發 app。113 位中年男女做相關分析：**估算 GMV 與創意活動 + 閱讀習慣相關**（DMN 對應創意，CEN 對應閱讀）。

## 4. ★ [PT] **Agitha 2026** — NeuroSense: Multimodal Early Dementia Screening

整合語音 + EEG + 手寫 + 臉部情緒 + 認知測驗的 AI 多模態系統。每模態抽特徵後 ML 分類。

## 5. ◇ [PT] **Ando 2026** — AI FER 驗證老人化妝療法

兩家養護機構輕度失智長者接受**化妝療法**（一次、約 1 小時），AI 分析療法前後臉部表情變化作為心理評估的替代驗證方法。

## 6. ◇ [--] **AR 2018** — Screening Emotional Expression in bvFTD Pilot

bvFTD 核心症狀 emotional blunting 缺乏實用臨床測量。**3 組各 8 位（bvFTD/AD/HC）**看不同情緒強度的真實情境影片，量化口語（自陳痛苦）+ 視覺（臉部 affect 有/無）反應。

## 7. ○ [PT] **Asgarian 2019** — Limitations of Facial Landmark Detection in Dementia Older Adults

臉部表情分析的前置 landmark 偵測在**老人 + 失智族群**有 systematic bias。實證研究：health status 影響 landmark 偵測效能；方法論層面警告。

## 8. ★ [--] **Bergamasco 2025** — Automatic CI Detection Through Facial Emotion Analysis

**32 位 CI + 28 位 HC**，標準視聽 emotion elicitation 視訊；訓 CNN 追蹤情緒演化作 CI 偵測。非侵入新方法。

## 9. ★ [--] **C 2024** — Multi-Dimensional Emotion Features for AD/MCI Diagnosis

長照 + 社區族群（HC=26 / MCI=23 / AD ≥60 歲）。**多維情緒特徵 + ML** 自動分類框架。

## 10. ★ [PT] **Chen 2025** — Desktop App: Face Recognition + EEG for AD

便捷 AD 早期偵測工具：**MediaPipe 臉部分析 + Muse2 EEG + OpenAI API prompt** 整合的桌面程式。Pilot 級。

## 11. ★ [PT] **Chen 2026** — MECO: Multimodal Dataset for Emotion + Cognitive Understanding in Older Adults

**42 位老人，~38 小時多模態訊號，30,592 同步 segment**。針對既有資料集少見的「老化族群 + 認知衰退影響表情」設計。**新公開 dataset**。

## 12. ◇ [PT] **CR 2018** — Motor Signatures of Emotional Reactivity in FTD

37 位 FTD 病人 vs 21 HC 看動態臉部表情視訊，記錄**臉部 EMG 反應**。神經影像對照 emotion-related motor mimicry 在 FTD 的中斷。

## 13. ★ [--] **Dong 2025** — Facial Micro-expressions for MCI (Clock Drawing Test)

**畫鐘測驗**過程中錄製臉部視訊，分析 micro-expression 與認知功能的關聯，作為 MCI digital biomarker。

## 14. ◇ [--] **E 2024** — Emotional Response to Social Robots in Dementia

失智長者對社交輔助機器人多感官刺激的**即時情緒反應**（用 FER）量化評估 pilot。

## 15. ○ [PT] **El-Hag 2025** — Enhanced FER + Age Estimation (Modified ResNet + SVM)

混合模型：影像增強（白平衡 + adaptive gamma）+ deep feature + ML classification。**邊緣強度 82.235 vs 原始 62.89**。HCI 應用導向。

## 16. ◇ [PT] **F 2019** — Facial Expressiveness + Physiological Arousal in FTD

25 位 bvFTD + HC 看情緒刺激，記錄**臉部 EMG + 皮膚電導 (SCL)**，腦影像看 insula/amygdala/PFC 的萎縮 vs 表情/喚醒對應。

## 17. ★ [PT] **F 2024** — FER in MCI: Exploratory Study

31 MCI + 26 PD-MCI + HC 做臉部情緒辨識（識別 + 記憶中性 vs 情緒表情），探討神經認知 correlates。

## 18. ★ [--] **Fan 2025** — Beyond the Clock: Multimodal Behavior Markers for MCI in CDT

畫鐘測驗時收集**畫線軌跡 + 臉部表情 + 手部動作**三維度，多 head attention CNN 整合做 MCI 偵測。

## 19. ○ [--] **Girard 2025** — Computational Analysis of Expressive Behavior in Clinical Assessment

概念性框架文：CV/語音/NLP 怎麼幫臨床心理評估增強信效度與規模化。

## 20. ○ [PT] **Goel 2025** — AutoML for FER Systems

傳統 FER 需要 expert 手動調參。研究 AutoML 開發 FER 系統的可行性。HCI 導向。

## 21. ◇ [PT] **Gressie 2023** — FER Error Profiles in FTD vs AD

**356 位**（bvFTD=62, SD-left=29, SD-right=14, PNFA=21, AD=76, HC=90）做臉部情緒辨識，分析各 FTD 亞型 vs AD 的錯誤模式差異。

## 22. ◇ [PT] **Hazelton 2023** — Interoception + Cognition in FER (bvFTD/AD/PD)

168 位（52 bvFTD + 41 AD + 24 PD + 51 HC）。檢驗**內感受準確性**（如心跳偵測）+ 認知能力作為情緒辨識的潛在機制。

## 23. ◇ [PT] **Ho 2020** — Face Discrimination + Emotion Recognition in PD-D

24 PD-D + 18 HC 做 morphing 臉部辨別、動態 + 靜態情緒辨識。檢驗 PD-D 三維臉部處理能力 + 認知 / 神經精神評估的對應。

## 24. ◇ [PT] **IG 2023** — Social Cognition: BD vs bvFTD

雙極性疾患 vs bvFTD 鑑別診斷困難（症狀相似）。**20 BD1 + 18 bvFTD + 40 HC** 做 FER + Mini-SEA。社交認知測量區別力評估。

## 25. ★ [--] **J 2021** — Facial Emotion Mimicry in Older Adults With/Without AD

25 位老人（HC + MCI + AD）用 **Kinect 3D** 錄製臉部，比較 6 種基本情緒模仿（mimicking）的適切性，跟自身典型表情比對。

## 26. ◇ [PT] **Jiskoot 2020** — Emotion Recognition of Morphed FE in Pre/Symptomatic FTD + AD

**ERT 工具**（情緒辨識任務）測 4 種強度（40-60-80-100%）下的 6 種情緒。bvFTD=32 + AD（人數見全文）。捕捉 static 任務漏掉的細微缺損。

## 27. ★ [--] **Kanjalkar 2025** — AI Dementia Early Care + Prognosis

整合 LLM（chatbot 認知評估）+ CNN（臉部表情）+ MRI DL（嚴重度分類）的 holistic 失智偵測 + 預後框架。

## 28. ◇ [PT] **KM 2022** — Emotional + Neuropsychiatric Disorders in AD (Review)

AD 在情緒溝通（FE 理解產出、affective prosody、述情障礙）+ 情緒體驗（憂鬱焦慮、agitation、psychosis）兩面都有缺損的回顧。

## 29. ★ [--] **L 2025** — AI-Based Facial Emotion Analysis for Differential Dementia Diagnosis

64 位受試者標準 AV 刺激下的 valence + arousal 抽取，多任務 ML 分類（含 MCI 與不同失智 subtype）。

## 30. ◇ [PT] **L 2025** — Cross-country FER Variance in pre/symptomatic bvFTD (GENFI/ReDLat)

**16 國 159 bvFTD + 521 presymptomatic carrier + 583 控制**。Linear mixed model + VBM 看 FER 跨國差異與神經 correlates。GENFI/ReDLat consortium。

## 31. ◇ [PT] **Liang 2020** — Multi-modal ML for Early Dementia in BSL Users

英國手語使用者老化族群早期失智 + 臉部表情多模態 ML 偵測 toolkit。

## 32. ★ [PT] **Liu 2021** — FER for Sound Intervention Emotional Response in Dementia

養護機構公共空間，**FaceReader** 評估失智長者對 3 種聲音介入（音樂/溪流/鳥鳴）的情緒反應。SAM 量表 vs FER 比較。

## 33. ◇ [--] **M 2024** — Optimized Attention FER for ND Healthcare

CBAM 注意力模組 + 輕量 DLN 在 AD/PD 的 FER 應用。

## 34. ★ [--] **M 2025** — Emotion Recognition in MCI/AD Rehabilitation

**30 位（14 MCI + 16 mild/moderate AD）**做有氧 + 認知訓練 + dual task + 創意活動的團體復健。**Kokoro Sensor** 攝影機分析臉部情緒。

## 35. ★ [--] **MA 2025** — FE Video Analysis for AD Diagnosis (Russian)

**37 HC + 34 NCD（16 mild + 18 major）**做特定診斷流程，**68 keypoints 非接觸捕捉**眉/眼/口運動學。

## 36. ○ [PT] **Mathur 2026** — Math Optimization + NN for FER (Pharma/Clinical)

calculus + transform function + DL 整合的 FER 框架。製藥研究 + mental health drug development 應用導向。

## 37. ◇ [--] **Matić 2026** — PD vs Depression FER Objectivization (Slovenian)

PD 的 hypomimia 跟憂鬱症臨床表型重疊難鑑別。**80 位（30 PD + 25 depression + 25 HC）**做 4 種 FE 任務（自發/自主/模仿/閱讀）。Hi-level Emotion 框架。

## 38. ◇ [PT] **ME 2024** — Social Cognition Differentiates Phenocopy bvFTD from Real bvFTD

**33 phFTD + 95 probable bvFTD** 用社交認知（含 FER）做鑑別診斷。

## 39. ★ [PT] **Montenegro 2018** — AD Diagnosis via VR + Emotion Analysis (Thesis)

PhD thesis：早期 AD 非侵入診斷。整合認知方法（VR 環境）+ 情緒分析。

## 40. ★ [PT] **Mu 2024** — Detect CI + Wellbeing from Remote Conversations

**39 位 NC/MCI 老人遠端視訊對話**，抽臉部 + 聲音 + 語言 + 心血管特徵。**CDR 0.5 vs 0 的 AUC = 0.77**。同時量化社交孤立、神經質、心理福祉。

## 41. ◇ [--] **Nazareth 2019** — Multimodal Emotion Recognition in Dementia (PhD)

PhD 研究計畫 overview：失智者自傳記憶下的多模態情緒辨識（音/視/生理），居家收集對比 normal aging。

## 42. ◇ [--] **Nylander 2025** — FE Metrics as Digital Biomarkers of Neurologic Disease

PD masked facies 等 ND 特徵性臉部表情。提取診間視訊的笑/皺眉/眨眼，評估自動化 FER 作為遠端、客觀診斷工具。

## 43. ○ [PT] **Parte 2026** — Demographic Bias in Facial Landmark Detection

HRI fairness 角度。系統性審計**年齡、性別、種族**對臉部 landmark 偵測的 bias。控制統計法消解干擾。

## 44. ○ [PT] **Pawar 2026** — DL vs ML for FER in Psychiatry

5 種 ML（KNN/SVM/RF/MLP/NN）做 6 種基本情緒分類。psychiatric 應用導向。

## 45. ★ [--] **R 2026** — Audio-Visual Disentangled Representation for Elderly CI Screening

社區規模 AD 篩檢需求。Audio-visual 共同訓 disentangled representation，可擴展非侵入 CI 評估。

## 46. ○ [PT] **Raj 2025** — CNN-FER for HRI

CV + FER 改良 HRI 應用（老人/身障輔助、醫療支援、surveillance 等）。

## 47. ◇ [PT] **Rymaszewski 2025** — ML for Biological Age Estimation Review

**生物年齡 (BA)** 比 chronological age 更能預測壽命/健康。Narrative review：ML 從低侵入資料源（**含臉部影像**、胸 X 光、腦 MRI、血液 biomarker、ECG、心理問卷）估算 BA 的方法。

## 48. ○ [PT] **S. 2026** — Weighted Late Fusion Deep Attention for Multimodal Emotion

audio + video 多模態情緒：CQ chromagram 抽 audio、CNN 抽 video、加權 late fusion + deep attention。

## 49. ○ [PT] **Sharma 2026** — CRBP: 3D-to-2.5D Facial Projection Benchmark

**Bosphorus 3D Face DB** 重建紋理 mesh + 6 個 canonical 2D projection；YOLOv8 偵測，分 raw / face-cropped 兩 subset。3D facial analysis 跨光度標準化 benchmark。

## 50. ★ [PT] **Sooriyaarachchi 2026** — Facial Expressions as Nexus for Health Assessment

評估視角：臉部表情作為**行為 phenotype**，在認知衰退/疼痛等健康障礙下偏離 baseline。Review ML 進展。

## 51. ○ [PT] **Straulino 2025** — Spontaneous Happiness Invariant Kinematic Marker

從喜劇片誘發 spontaneous happiness vs Ekman dataset 的 posed happiness，比較 kinematic 不變性。Methodology 區分自發 vs 擺拍。

## 52. ○ [--] **Sumsion 2025** — Stacking Ensembles + MoE for AU Recognition

AU 識別在不同 AU 表現不均。Stacking 集成 + MoE 做平衡。Average F1 提升。

## 53. ○ [PT] **Sumsion 2026** — ELEGANT: Node + Edge Generation + Landmark MTL for AU

「ELEGANT」框架：**同步生成 graph node + edge** 加 landmark 多任務學習。AU 已用於失智偵測、疼痛偵測等下游，提升 AU 模型直接 inform 這些應用。

## 54. ○ [PT] **Suresh 2025** — Micro-expression + Masked Expression Classification

Micro-expression（無意識、瞬間）+ masked expression（刻意隱藏）分類。NN-based。心理評估、執法應用。

## 55. ○ [PT] **Tandianto 2026** — CNN FER Comparison Across FER-2013/FER+/RAF-DB/AffectNet

5 層 CNN 統一框架做 4 大公開 dataset 跨資料集系統性比較。Hyperparameter 結構化 tuning。

## 56. ◇ [PT] **Velichko 2025** — Multimodal Depression Detection (Semi-auto + Deterministic ML)

半自動標註 + 確定性 ML 多模態憂鬱偵測。語音 + 臉部表情。

## 57. ★ [--] **Y 2023** — ML for Depression/Anxiety/Apathy in MCI (Speech + Facial)

MCI 患者高比例有 depression/anxiety/apathy；這些症狀預測 MCI→失智進展。觀察性研究：**語音 + 臉部表情**作 ML 模型 input。

## 58. ★ [PT] **Z 2022** — Automated Facial Emotion Analysis in CI

400 萬 face 預訓 CV DL 模型分析認知障礙者 passive viewing memory test 的臉部表情。**n=493**（HC + 不同 etiology / severity）。

## 59. ◇ [--] **Öztürk 2026** — PD DBS ON/OFF FER

PD 深部腦刺激 ON/OFF 狀態下臉部表情客觀評估 pilot。**STN-DBS 增加部分臉部動作 amplitude 但未完全恢復 temporal dynamics**；微笑執行延遲持續。

## 60. ○ [PT] **Abosaq 2025** — AI DL Architectures for Robust Emotion Recognition

DL 自動學臉部表情層次特徵。處理 partial occlusion、不一致光照、dataset bias 的穩健架構。

## 61. ◇ [PT] **Ai 2025** — Service Robots Multimodal Emotion (Transfer Learning)

老人特殊情緒表達特性。Transfer learning 跨 AffectNet 等公開資料到老人領域，多模態情緒辨識給社交機器人。

## 62. ○ [PT] **Akiyama 2025** — Reliability of FER Multi-task Cascaded CNN

**MTCNN** vs 主觀情緒判斷一致性。40 位大學生跟 PALRO robot 對話 10 分鐘，13 段 ROI 視訊評估「快樂或快樂+其他組合」。

## 63. ★ [PT] **Alsuhaibani 2024** — Review DL for Non-Invasive CI Detection

Review：語音 + 語言 + 臉部 + 動作的 DL CI 偵測。**語音/語言基本上表現最高**，多模態結合（acoustic + linguistic）效益顯著。

## 64. ★ [--] **Alzahrani 2025** — Facial Cues for CI Detection from In-the-Wild Data

**眨眼率 (EBR)** + 頭部轉動率 (HTR) + 頭部動作統計特徵 (HMSF)。區分 ND/MCI/FMD/HC 的視覺特徵 in-the-wild 分析。

## 65. △ [--] **(anon) 2025** — Group Activities Preference for MCI/Dementia via AI FER

只有重複標題作 abstract，內容無法判斷。可能是 conference poster。

## 66. ○ [PT] **Belharbi 2024** — Guided Interpretable FER via Spatial AU Cues

SOTA FER classifier 缺解釋性。**訓練時引入 AU codebook 顯式對應臉部區域**，實現可解釋深度模型。

## 67. ◇ [PT] **Camuñas 2025** — Affective Computing in Spanish-Speaking Older Adults Video Interviews

老人 + 非英語族群在情感運算缺資料。Spanish-speaking 老人視訊訪談評估 SOTA FER + 文字情感 + 微笑偵測。

## 68. ○ [--] **Chouhayebi 2024** — HOG-HOF + VGG-LSTM Spatio-Temporal FER

DL + 動態紋理混合：HOG-HOF（空間時序）+ VGG-LSTM（深度時序），improving 視訊情緒辨識。

## 69. ○ [PT] **Fabrício 2024** — Brazilian Face DB for Basic Emotion FER

巴西族群：考慮年齡/性別/種族分布的臉部基本情緒資料庫建置 + 驗證。3 階段流程。

## 70. ◇ [PT] **Gabrielli 2024** — Universal Time-Series Model for Stress Monitoring (ND)

ND 病人壓力監測：ECG + actigraphy + 語音 + **臉部分析**。穿戴裝置 HRV 不準的問題；通用時序模型整合多源訊號。

## 71. ◇ [--] **Galanakis 2024** — MediaPipe Holistic for Predicting Aggressive Behavior in Dementia

**MediaPipe Holistic**（手勢 + 姿態 + 臉部）抓 landmark，分類失智者爭吵 vs 非爭吵行為。

## 72. ◇ [PT] **Gaya-Morey 2024** — DL FER for Intellectual Disabilities

智能障礙者 FER 在文獻上未被研究。訓練 12 個 CNN 評估 SOTA DL 在此族群效能。

## 73. ★ [PT] **Gaya-Morey 2025** — DL FER for Elderly Systematic Review

**31 篇近 10 年研究**的 systematic review。老人 FER 應用於 assisted living、心理健康、個人化照護。

## 74. ○ [--] **Gómez-Sirvent 2023** — In-the-Wild Low-Res FER (Voting ResNet)

低解析度（重要實際情境）。改良 ResNet-18 多重 overlapping crop 投票機制。

## 75. ◇ [PT] **Goyal 2025** — IoT Stress Detection for Elderly

CNN FER 把表情分 7 類（怕/怒/嫌/喜/中/悲/驚）。長時間 capture 推估老人**疼痛強度**，協助看護。

## 76. ★ [--] **Hoang 2024** — Subject Harmonization for MCI Detection (Language + FE)

MCI 是 AD 前驅期，數位 marker 經濟可行。語言 marker 已知有效；近年研究顯示**對話中情緒**也是補強信號。**受試者標準化** + 多模態提升 MCI 偵測。

## 77. ○ [PT] **Huang 2023** — Mental States + Personality from Activity + FER

172 位受試者 IPSI 訪談量表（53 題、9 因子）+ CNN 即時 FER + 動作辨識，基於 Russell circumplex model 推估心理狀態。

## 78. ★ [PT] **Jiang 2024** — Contactless Detection Emotion + CI in Elderly Review

加拿大老化族群 ~11% 認知衰退。傳統診斷（MRI/PET/認知測驗）成本高。**Review 非接觸式偵測**（情緒 + CI）方法。

## 79. ◇ [PT] **Jouval 2023** — Emotion Evaluation Across Age Ranges Using Labeled Films

4 個年齡層（YA 20-39 + OA + ...）非病理男女觀看標籤化短片（中/喜/驚/怒/恐/嫌/悲），評估臉部反應。

## 80. ★ [--] **Kadali 2024** — CNN AD Detection from FE + Eye Movements

CNN 結合臉部表情 + 眼動分析做 AD 偵測。Image preprocessing 多種方法。

## 81. ○ [PT] **Kanna 2024** — CNN Face Emotion for Healthcare

CNN FER + transfer learning + augmentation。Healthcare 應用。

## 82. ★ [PT] **Karako 2024** — Predictive DL for Cognitive Risk

從**易取得資料**做 MCI 預測 DL。減少需要病人主動做認知測驗的限制（症狀出現時可能太晚）。

## 83. ○ [PT] **Khare 2023** — Emotion Recognition + AI 2014-2023 Systematic Review

涵蓋物理（臉/語音）+ 生理（EEG/ECG/EDA/PPG）訊號的 AI 情緒辨識方法 systematic review。

## 84. ○ [PT] **Klingner 2023** — Mimik und Emotion (German)

人臉表情表達情緒 + 跨文化共通性 + cerebral network 複雜度的 review（德文）。

## 85. ◇ [--] **Kolosov 2024** — AU + Emotions in Dementia Listening to Music

音樂介入失智照護。AU 與快樂程度的關聯（專家觀察）。**特定 AU 與快樂程度對應**。個人化音樂播放清單應用。

## 86. ○ [PT] **Krishnasamy 2025** — Ensemble DL for Hybrid Facial Datasets + Landmark

98/68 點 landmark + ResNet50 ensemble FER。HCI 應用。

## 87. ★ [PT] **Lee 2024** — MCI Prediction with Multi-stream CNN

MCI 早期診斷成本/時間問題。**Multi-stream CNN** 從多模態資料偵測 MCI/失智。

## 88. ★ [PT] **Maji 2024** — Gamified AI for Early Dementia Detection

整合**健康指標 CNN（1D）+ 臉部影像 CNN** 的遊戲化認知評估。1000 健康指標 + 1800 臉部影像資料訓練。

## 89. ★ [PT] **Matsuda 2025** — Smile Detection in Real-World Dementia Care (QOL Pilot)

QOL 評估通常自評；失智進展後病人無法自評，改 proxy 評估。**真實照護場景的微笑偵測**作為客觀 QOL 評估的 pilot。

## 90. ○ [PT] **Mertens 2024** — FindingEmo: 25k In-the-Wild Emotion Image Dataset

**25k 圖像** annotated for emotion；focus on 多人複雜社交場景（不只單臉）。Valence/Arousal/Emotion label。

## 91. ○ [--] **Munárriz 2024** — Spanish FER Database Minimizing Bias

FER 訓練資料常有 demographic bias 導致歧視。設計上**從一開始就最小化偏差**的西語 FER 資料集。

## 92. ○ [--] **Nomiya 2025** — AI for Sensing Valence + Arousal from Facial Images

維度情緒（V/A）與臉部表情 systematic 對應，但少有 AI 模型直接從臉估算 V/A。RNN-based AI 模型基於實證資料估算主觀 V/A。

## 93. ◇ [PT] **Obayashi 2024** — Smiling Amount Affects Smiling Response in Conversations

對話中對方的微笑量影響自身微笑反應。**40 位（20 女）面對男/女 listener 三分鐘對話**量化雙方笑的強度與頻率。社交動力學量化。

## 94. ○ [--] **Patrick 2024** — Initial Expressed Emotion During Neuropsych Assessment

神經心理測驗中**初始表達情緒**與認知測試表現的關聯（趨近/迴避 motivational dimension）。

## 95. ○ [PT] **Paulchamy 2025** — FER via Transfer Learning (VGG16 + ResNet + AlexNet)

3 個預訓 CNN 整合做 7 類情緒（含 contempt）分類。Transfer learning + multiclass classifier。

## 96. ◇ [PT] **Roßkopf 2023** — Facial Biofeedback on Hypomimia/FER/Affect in PD

PD 的 hypomimia 是次要症狀但影響社交。**EMG 臉部 biofeedback** 增強表情與情緒辨識的 quasi-RCT 短期效果。

## 97. ★ [PT] **Ruoranen 2023** — Memory Disorder Detection via FER

記憶障礙下的表情變化（特別是 apathy 降低表情強度）。FER 程式應用於記憶障礙篩檢的論文（thesis）。

## 98. ○ [PT] **Setiaji 2023** — Emotional Classification via CNN FER

CNN 從臉部影像分類情緒（喜/悲/恐/...）。Smart City 應用。

## 99. ★ [--] **Shigekiyo 2024** — Anxiety Estimation in Dementia (Phrases + FE + Behaviors)

失智者焦慮 → BPSD（agitation 等）。早期 anxiety 偵測減輕 caregiver 負擔。**語句 + 臉部表情 + 行為**整合估算焦慮強度。

## 100. ◇ [PT] **Skaramagkas 2025** — Dual-Stream Transformer for PD Medication State from Facial Videos

Hypomimia 對 levodopa 反應顯著。**183 位 PD 視訊**，dual-stream transformer（frame feature + optical flow）區分 ON/OFF 用藥狀態。

## 101. ○ [PT] **Straulino 2023** — What Is Missing in Emotion Expression Study

紀念 Darwin 1872 著作 150 年。批判 Ekman 經典分類觀（6 prototype），呼籲流動 + 動態方法。Conceptual review。

## 102. ★ [--] **Sun 2023** — MC-ViViT: MCI Detection from Facial Videos (I-CONECT)

**Multi-branch Classifier-ViViT**：抽 spatiotemporal 特徵 + 增強表徵。**用 I-CONECT 資料**。最高準確 90.63%（部分訪談視訊）。

## 103. ◇ [PT] **Takale 2024** — DL ND Patient FER Intensive Care

Hyperparameter tuning + DL FER 在神經疾患加護應用。

## 104. ★ [PT] **Takeshige-Amano 2024** — AD Detection via Smiles + Chatbot Conversations

AD 認知狀態評估迫切。**99 HC + 93 AD/MCI** 與 chatbot 對話、抽微笑影像 + 視聽特徵。臉 + 聲 + 對話內容多 cue。

## 105. ○ [PT] **Ubillús 2023** — FER Algorithms Systematic Review

系統性回顧 FER 演算法。資料集品質 + 模型選擇缺陷。

## 106. ○ [PT] **Veigas 2024** — DL FER for Elderly

老化族群心理健康挑戰。FER 偵測情緒供心理健康 monitoring。

## 107. ◇ [PT] **Xiangyu 2023** — FER for Teleoperated Robot Reminiscence Group Therapy in Dementia

懷舊團體治療（RGT）非藥物失智治療常用法。**OpenCV-based AI FER** 在遠端機器人 RGT 的應用。

## 108. ★ [--] **Zheng 2025** — Augmenting Face Mesh for Dementia Detection

語言 marker 受重視，臉部 marker 相對被忽略。**擴增臉部 mesh** 增強失智偵測 — 直接相關方法論進展。

## 109. ○ [--] **Zhou 2023** — Cross-Attention + Hybrid Feature for Video Emotion Recognition

大規模視訊片段（圖像/視訊）情緒辨識。Cross-attention + hybrid feature weighting NN。心理評估、工作壓力、觀光滿意度應用。

---

## 摘要 — Emotion 主題分群（109 篇）

### ★ 最相關（直接 AD/MCI/dementia × 臉部影像，34 篇）

#1 A 2024、#3 Abe 2025、#4 Agitha 2026、#8 Bergamasco 2025、#9 C 2024、#10 Chen 2025、#11 Chen 2026 (MECO dataset)、#13 Dong 2025、#17 F 2024、#18 Fan 2025、#25 J 2021、#27 Kanjalkar 2025、#29 L 2025（差別診斷）、#32 Liu 2021、#34 M 2025、#35 MA 2025、#39 Montenegro 2018、#40 Mu 2024、#45 R 2026、#50 Sooriyaarachchi 2026、#57 Y 2023、#58 Z 2022、#63 Alsuhaibani 2024、#64 Alzahrani 2025、#73 Gaya-Morey 2025、#76 Hoang 2024、#78 Jiang 2024、#80 Kadali 2024、#82 Karako 2024、#87 Lee 2024、#88 Maji 2024、#89 Matsuda 2025、#97 Ruoranen 2023、#99 Shigekiyo 2024、#102 Sun 2023、#104 Takeshige-Amano 2024、#108 Zheng 2025

→ 這組是 paper discussion / introduction 引用的主力。

### ◇ AD-adjacent（PD/FTD/ND，方法論可借鏡，27 篇）

#2、#5、#6、#12、#14、#16、#21、#22、#23、#24、#26、#28、#30、#33、#37、#38、#41、#42、#47、#56、#59、#61、#67、#70、#71、#72、#75、#79、#85、#93、#96、#100、#103、#107

→ FTD/PD 系列研究在臉部表情病理上常做為 cross-condition reference。

### ○ FER 方法論 / 資料集（generic，30+ 篇）

#7、#15、#19、#20、#36、#43、#44、#46、#48、#49、#51、#52、#53、#54、#55、#60、#62、#66、#68、#69、#74、#77、#81、#83、#84、#86、#90、#91、#92、#94、#95、#98、#101、#105、#106、#109

→ 引述模型、資料集、AU 框架時用。

### △ 邊緣 / 待釐清

#65 (anon) 2025 — 只有重複標題作 abstract，建議手動找原文確認

---

## 重要 dataset / 資源

- **MECO** (Chen 2026) — 42 位老人多模態 emotion+cognition 公開資料集，跟 I-CONECT 相比是新選項
- **CRBP** (Sharma 2026) — Bosphorus 3D-to-2.5D facial benchmark
- **FindingEmo** (Mertens 2024) — 25k in-the-wild emotion images
- **Brazilian face DB** (Fabrício 2024) — 多元族群 FER
- **Spanish FER bias-minimal DB** (Munárriz 2024)

## 最值得深讀的 ~10 篇（排序）

1. **Chen 2026 MECO** — 公開老人多模態 dataset
2. **Sun 2023 MC-ViViT** — I-CONECT 上 90.63%
3. **Takeshige-Amano 2024** — AD chatbot + 微笑（n=192）
4. **Mu 2024** — 4 模態（face/acoustic/linguistic/cardio）AUC 0.77
5. **Hoang 2024** — subject harmonization for MCI
6. **Z 2022** — n=493 大樣本自動 FE 分析
7. **Lee 2024** — multi-stream CNN MCI 預測
8. **Gressie 2023** — 356 位 FTD/AD/HC 大規模
9. **L 2025 GENFI/ReDLat** — 16 國 1263 人跨國 FER 變異
10. **Hazelton 2023** — interoception 機制（168 人）
