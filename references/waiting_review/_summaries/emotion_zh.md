# Emotion 主題摘要（中文整理）

2026-04-29 由 abstract + 全文 PDF 整理。

標籤：
- `[P]` = 有 PDF / `[T]` = 有 .txt / `[-]` = 只 abstract
- `★` = 直接 AD/MCI/dementia × 臉部影像分析（高相關）
- `◇` = AD-adjacent（PD/FTD/其他 ND，方法論可借鏡）
- `○` = FER 方法論 / 資料集（generic，但 inform pipeline）
- `△` = 邊緣相關

排列：本檔按 review status — **#1-8 強保留**、**#9-33 保留**、**#34-65 邊緣**。

共 **65 篇**（44 篇已歸檔到 _archive_rejected/）。

---

---

## ━━━━━━━━━━━━━━ 強保留（8 篇）━━━━━━━━━━━━━━

## 1. ★ [--] **Zheng 2025** — Augmenting Face Mesh for Dementia Detection

語言 marker 受重視，臉部 marker 相對被忽略。**擴增臉部 mesh** 增強失智偵測 — 直接相關方法論進展。

## 2. ★ [PT] **Chen 2026** — MECO: Multimodal Dataset for Emotion + Cognitive Understanding in Older Adults

**42 位老人，~38 小時多模態訊號，30,592 同步 segment**。針對既有資料集少見的「老化族群 + 認知衰退影響表情」設計。**新公開 dataset**。

## 3. ★ [--] **MA 2025** — FE Video Analysis for AD Diagnosis (Russian)

**37 HC + 34 NCD（16 mild + 18 major）**做特定診斷流程，**68 keypoints 非接觸捕捉**眉/眼/口運動學。

## 4. ★ [PT] **Mu 2024** — Detect CI + Wellbeing from Remote Conversations

**39 位 NC/MCI 老人遠端視訊對話**，抽臉部 + 聲音 + 語言 + 心血管特徵。**CDR 0.5 vs 0 的 AUC = 0.77**。同時量化社交孤立、神經質、心理福祉。

## 5. ★ [--] **R 2026** — Audio-Visual Disentangled Representation for Elderly CI Screening

社區規模 AD 篩檢需求。Audio-visual 共同訓 disentangled representation，可擴展非侵入 CI 評估。

## 6. ★ [--] **Alzahrani 2025** — Facial Cues for CI Detection from In-the-Wild Data

**眨眼率 (EBR)** + 頭部轉動率 (HTR) + 頭部動作統計特徵 (HMSF)。區分 ND/MCI/FMD/HC 的視覺特徵 in-the-wild 分析。

## 7. ★ [--] **Hoang 2024** — Subject Harmonization for MCI Detection (Language + FE)

MCI 是 AD 前驅期，數位 marker 經濟可行。語言 marker 已知有效；近年研究顯示**對話中情緒**也是補強信號。**受試者標準化** + 多模態提升 MCI 偵測。

## 8. ★ [PT] **Lee 2024** — MCI Prediction with Multi-stream CNN

MCI 早期診斷成本/時間問題。**Multi-stream CNN** 從多模態資料偵測 MCI/失智。


## ━━━━━━━━━━━━━━ 保留（25 篇）━━━━━━━━━━━━━━

## 9. ★ [PT] **Agitha 2026** — NeuroSense: Multimodal Early Dementia Screening

整合語音 + EEG + 手寫 + 臉部情緒 + 認知測驗的 AI 多模態系統。每模態抽特徵後 ML 分類。

## 10. ○ [PT] **Asgarian 2019** — Limitations of Facial Landmark Detection in Dementia Older Adults

臉部表情分析的前置 landmark 偵測在**老人 + 失智族群**有 systematic bias。實證研究：health status 影響 landmark 偵測效能；方法論層面警告。

## 11. ★ [--] **Bergamasco 2025** — Automatic CI Detection Through Facial Emotion Analysis

**32 位 CI + 28 位 HC**，標準視聽 emotion elicitation 視訊；訓 CNN 追蹤情緒演化作 CI 偵測。非侵入新方法。

## 12. ★ [--] **Dong 2025** — Facial Micro-expressions for MCI (Clock Drawing Test)

**畫鐘測驗**過程中錄製臉部視訊，分析 micro-expression 與認知功能的關聯，作為 MCI digital biomarker。

## 13. ○ [--] **Girard 2025** — Computational Analysis of Expressive Behavior in Clinical Assessment

概念性框架文：CV/語音/NLP 怎麼幫臨床心理評估增強信效度與規模化。

## 14. ★ [--] **J 2021** — Facial Emotion Mimicry in Older Adults With/Without AD

25 位老人（HC + MCI + AD）用 **Kinect 3D** 錄製臉部，比較 6 種基本情緒模仿（mimicking）的適切性，跟自身典型表情比對。

## 15. ◇ [--] **Matić 2026** — PD vs Depression FER Objectivization (Slovenian)

PD 的 hypomimia 跟憂鬱症臨床表型重疊難鑑別。**80 位（30 PD + 25 depression + 25 HC）**做 4 種 FE 任務（自發/自主/模仿/閱讀）。Hi-level Emotion 框架。

## 16. ★ [PT] **Montenegro 2018** — AD Diagnosis via VR + Emotion Analysis (Thesis)

PhD thesis：早期 AD 非侵入診斷。整合認知方法（VR 環境）+ 情緒分析。

## 17. ★ [PT] **Sooriyaarachchi 2026** — Facial Expressions as Nexus for Health Assessment

評估視角：臉部表情作為**行為 phenotype**，在認知衰退/疼痛等健康障礙下偏離 baseline。Review ML 進展。

## 18. ★ [--] **Y 2023** — ML for Depression/Anxiety/Apathy in MCI (Speech + Facial)

MCI 患者高比例有 depression/anxiety/apathy；這些症狀預測 MCI→失智進展。觀察性研究：**語音 + 臉部表情**作 ML 模型 input。

## 19. ★ [PT] **Alsuhaibani 2024** — Review DL for Non-Invasive CI Detection

Review：語音 + 語言 + 臉部 + 動作的 DL CI 偵測。**語音/語言基本上表現最高**，多模態結合（acoustic + linguistic）效益顯著。

## 20. ★ [PT] **Gaya-Morey 2025** — DL FER for Elderly Systematic Review

**31 篇近 10 年研究**的 systematic review。老人 FER 應用於 assisted living、心理健康、個人化照護。

## 21. ★ [PT] **Jiang 2024** — Contactless Detection Emotion + CI in Elderly Review

加拿大老化族群 ~11% 認知衰退。傳統診斷（MRI/PET/認知測驗）成本高。**Review 非接觸式偵測**（情緒 + CI）方法。

## 22. ★ [--] **Kadali 2024** — CNN AD Detection from FE + Eye Movements

CNN 結合臉部表情 + 眼動分析做 AD 偵測。Image preprocessing 多種方法。

## 23. ★ [PT] **Karako 2024** — Predictive DL for Cognitive Risk

從**易取得資料**做 MCI 預測 DL。減少需要病人主動做認知測驗的限制（症狀出現時可能太晚）。

## 24. ★ [PT] **Maji 2024** — Gamified AI for Early Dementia Detection

整合**健康指標 CNN（1D）+ 臉部影像 CNN** 的遊戲化認知評估。1000 健康指標 + 1800 臉部影像資料訓練。

## 25. ★ [PT] **Ruoranen 2023** — Memory Disorder Detection via FER

記憶障礙下的表情變化（特別是 apathy 降低表情強度）。FER 程式應用於記憶障礙篩檢的論文（thesis）。

## 26. ★ [--] **C 2024** — Multi-Dimensional Emotion Features for AD/MCI Diagnosis

長照 + 社區族群（HC=26 / MCI=23 / AD ≥60 歲）。**多維情緒特徵 + ML** 自動分類框架。

## 27. ★ [--] **Fan 2025** — Beyond the Clock: Multimodal Behavior Markers for MCI in CDT

畫鐘測驗時收集**畫線軌跡 + 臉部表情 + 手部動作**三維度，多 head attention CNN 整合做 MCI 偵測。

## 28. ★ [--] **L 2025** — AI-Based Facial Emotion Analysis for Differential Dementia Diagnosis

64 位受試者標準 AV 刺激下的 valence + arousal 抽取，多任務 ML 分類（含 MCI 與不同失智 subtype）。

## 29. ★ [--] **M 2025** — Emotion Recognition in MCI/AD Rehabilitation

**30 位（14 MCI + 16 mild/moderate AD）**做有氧 + 認知訓練 + dual task + 創意活動的團體復健。**Kokoro Sensor** 攝影機分析臉部情緒。

## 30. ★ [PT] **Z 2022** — Automated Facial Emotion Analysis in CI

400 萬 face 預訓 CV DL 模型分析認知障礙者 passive viewing memory test 的臉部表情。**n=493**（HC + 不同 etiology / severity）。

## 31. ★ [PT] **Matsuda 2025** — Smile Detection in Real-World Dementia Care (QOL Pilot)

QOL 評估通常自評；失智進展後病人無法自評，改 proxy 評估。**真實照護場景的微笑偵測**作為客觀 QOL 評估的 pilot。

## 32. ★ [--] **Sun 2023** — MC-ViViT: MCI Detection from Facial Videos (I-CONECT)

**Multi-branch Classifier-ViViT**：抽 spatiotemporal 特徵 + 增強表徵。**用 I-CONECT 資料**。最高準確 90.63%（部分訪談視訊）。

## 33. ★ [PT] **Takeshige-Amano 2024** — AD Detection via Smiles + Chatbot Conversations

AD 認知狀態評估迫切。**99 HC + 93 AD/MCI** 與 chatbot 對話、抽微笑影像 + 視聽特徵。臉 + 聲 + 對話內容多 cue。


## ━━━━━━━━━━━━━━ 邊緣（32 篇）━━━━━━━━━━━━━━

## 34. ★ [PT] **Chen 2025** — Desktop App: Face Recognition + EEG for AD

便捷 AD 早期偵測工具：**MediaPipe 臉部分析 + Muse2 EEG + OpenAI API prompt** 整合的桌面程式。Pilot 級。

## 35. ◇ [--] **E 2024** — Emotional Response to Social Robots in Dementia

失智長者對社交輔助機器人多感官刺激的**即時情緒反應**（用 FER）量化評估 pilot。

## 36. ◇ [PT] **F 2019** — Facial Expressiveness + Physiological Arousal in FTD

25 位 bvFTD + HC 看情緒刺激，記錄**臉部 EMG + 皮膚電導 (SCL)**，腦影像看 insula/amygdala/PFC 的萎縮 vs 表情/喚醒對應。

## 37. ★ [PT] **F 2024** — FER in MCI: Exploratory Study

31 MCI + 26 PD-MCI + HC 做臉部情緒辨識（識別 + 記憶中性 vs 情緒表情），探討神經認知 correlates。

## 38. ★ [--] **Kanjalkar 2025** — AI Dementia Early Care + Prognosis

整合 LLM（chatbot 認知評估）+ CNN（臉部表情）+ MRI DL（嚴重度分類）的 holistic 失智偵測 + 預後框架。

## 39. ★ [PT] **Liu 2021** — FER for Sound Intervention Emotional Response in Dementia

養護機構公共空間，**FaceReader** 評估失智長者對 3 種聲音介入（音樂/溪流/鳥鳴）的情緒反應。SAM 量表 vs FER 比較。

## 40. ◇ [--] **M 2024** — Optimized Attention FER for ND Healthcare

CBAM 注意力模組 + 輕量 DLN 在 AD/PD 的 FER 應用。

## 41. ◇ [--] **Nylander 2025** — FE Metrics as Digital Biomarkers of Neurologic Disease

PD masked facies 等 ND 特徵性臉部表情。提取診間視訊的笑/皺眉/眨眼，評估自動化 FER 作為遠端、客觀診斷工具。

## 42. ○ [PT] **Parte 2026** — Demographic Bias in Facial Landmark Detection

HRI fairness 角度。系統性審計**年齡、性別、種族**對臉部 landmark 偵測的 bias。控制統計法消解干擾。

## 43. ◇ [PT] **Rymaszewski 2025** — ML for Biological Age Estimation Review

**生物年齡 (BA)** 比 chronological age 更能預測壽命/健康。Narrative review：ML 從低侵入資料源（**含臉部影像**、胸 X 光、腦 MRI、血液 biomarker、ECG、心理問卷）估算 BA 的方法。

## 44. ○ [PT] **Sharma 2026** — CRBP: 3D-to-2.5D Facial Projection Benchmark

**Bosphorus 3D Face DB** 重建紋理 mesh + 6 個 canonical 2D projection；YOLOv8 偵測，分 raw / face-cropped 兩 subset。3D facial analysis 跨光度標準化 benchmark。

## 45. ○ [--] **Sumsion 2025** — Stacking Ensembles + MoE for AU Recognition

AU 識別在不同 AU 表現不均。Stacking 集成 + MoE 做平衡。Average F1 提升。

## 46. ○ [PT] **Sumsion 2026** — ELEGANT: Node + Edge Generation + Landmark MTL for AU

「ELEGANT」框架：**同步生成 graph node + edge** 加 landmark 多任務學習。AU 已用於失智偵測、疼痛偵測等下游，提升 AU 模型直接 inform 這些應用。

## 47. ◇ [PT] **Velichko 2025** — Multimodal Depression Detection (Semi-auto + Deterministic ML)

半自動標註 + 確定性 ML 多模態憂鬱偵測。語音 + 臉部表情。

## 48. ◇ [--] **Öztürk 2026** — PD DBS ON/OFF FER

PD 深部腦刺激 ON/OFF 狀態下臉部表情客觀評估 pilot。**STN-DBS 增加部分臉部動作 amplitude 但未完全恢復 temporal dynamics**；微笑執行延遲持續。

## 49. ◇ [PT] **Ai 2025** — Service Robots Multimodal Emotion (Transfer Learning)

老人特殊情緒表達特性。Transfer learning 跨 AffectNet 等公開資料到老人領域，多模態情緒辨識給社交機器人。

## 50. ○ [PT] **Belharbi 2024** — Guided Interpretable FER via Spatial AU Cues

SOTA FER classifier 缺解釋性。**訓練時引入 AU codebook 顯式對應臉部區域**，實現可解釋深度模型。

## 51. ◇ [PT] **Camuñas 2025** — Affective Computing in Spanish-Speaking Older Adults Video Interviews

老人 + 非英語族群在情感運算缺資料。Spanish-speaking 老人視訊訪談評估 SOTA FER + 文字情感 + 微笑偵測。

## 52. ○ [PT] **Fabrício 2024** — Brazilian Face DB for Basic Emotion FER

巴西族群：考慮年齡/性別/種族分布的臉部基本情緒資料庫建置 + 驗證。3 階段流程。

## 53. ◇ [PT] **Gabrielli 2024** — Universal Time-Series Model for Stress Monitoring (ND)

ND 病人壓力監測：ECG + actigraphy + 語音 + **臉部分析**。穿戴裝置 HRV 不準的問題；通用時序模型整合多源訊號。

## 54. ◇ [--] **Galanakis 2024** — MediaPipe Holistic for Predicting Aggressive Behavior in Dementia

**MediaPipe Holistic**（手勢 + 姿態 + 臉部）抓 landmark，分類失智者爭吵 vs 非爭吵行為。

## 55. ◇ [PT] **Goyal 2025** — IoT Stress Detection for Elderly

CNN FER 把表情分 7 類（怕/怒/嫌/喜/中/悲/驚）。長時間 capture 推估老人**疼痛強度**，協助看護。

## 56. ◇ [PT] **Jouval 2023** — Emotion Evaluation Across Age Ranges Using Labeled Films

4 個年齡層（YA 20-39 + OA + ...）非病理男女觀看標籤化短片（中/喜/驚/怒/恐/嫌/悲），評估臉部反應。

## 57. ○ [PT] **Khare 2023** — Emotion Recognition + AI 2014-2023 Systematic Review

涵蓋物理（臉/語音）+ 生理（EEG/ECG/EDA/PPG）訊號的 AI 情緒辨識方法 systematic review。

## 58. ◇ [--] **Kolosov 2024** — AU + Emotions in Dementia Listening to Music

音樂介入失智照護。AU 與快樂程度的關聯（專家觀察）。**特定 AU 與快樂程度對應**。個人化音樂播放清單應用。

## 59. ○ [PT] **Mertens 2024** — FindingEmo: 25k In-the-Wild Emotion Image Dataset

**25k 圖像** annotated for emotion；focus on 多人複雜社交場景（不只單臉）。Valence/Arousal/Emotion label。

## 60. ○ [--] **Patrick 2024** — Initial Expressed Emotion During Neuropsych Assessment

神經心理測驗中**初始表達情緒**與認知測試表現的關聯（趨近/迴避 motivational dimension）。

## 61. ★ [--] **Shigekiyo 2024** — Anxiety Estimation in Dementia (Phrases + FE + Behaviors)

失智者焦慮 → BPSD（agitation 等）。早期 anxiety 偵測減輕 caregiver 負擔。**語句 + 臉部表情 + 行為**整合估算焦慮強度。

## 62. ◇ [PT] **Skaramagkas 2025** — Dual-Stream Transformer for PD Medication State from Facial Videos

Hypomimia 對 levodopa 反應顯著。**183 位 PD 視訊**，dual-stream transformer（frame feature + optical flow）區分 ON/OFF 用藥狀態。

## 63. ◇ [PT] **Takale 2024** — DL ND Patient FER Intensive Care

Hyperparameter tuning + DL FER 在神經疾患加護應用。

## 64. ○ [PT] **Veigas 2024** — DL FER for Elderly

老化族群心理健康挑戰。FER 偵測情緒供心理健康 monitoring。

## 65. ◇ [PT] **Xiangyu 2023** — FER for Teleoperated Robot Reminiscence Group Therapy in Dementia

懷舊團體治療（RGT）非藥物失智治療常用法。**OpenCV-based AI FER** 在遠端機器人 RGT 的應用。
