# Alz_face_analyze

阿茲海默症臉部多模態分析系統（年齡、情緒/AU、不對稱性、頭部旋轉）

## 專案結構

```
Alz_face_analyze/
├── src/
│   ├── config.py                      # 全專案共用配置（路徑常數、PreprocessConfig）
│   │
│   ├── common/                        # 跨模組共用工具
│   │   ├── metrics.py                 # MetricsCalculator
│   │   ├── demographics.py            # DemographicsLoader
│   │   └── mediapipe_utils.py         # 統一 landmark 常數 (midline/bilateral/rotation)
│   │
│   ├── modules/                       # Pipeline 模組
│   │   ├── preprocess/                # 臉部預處理
│   │   │   ├── detector.py            # FaceDetector (MediaPipe)
│   │   │   ├── selector.py            # FaceSelector (選擇最正面)
│   │   │   ├── aligner.py             # FaceAligner (角度校正)
│   │   │   ├── mirror.py              # MirrorGenerator (鏡射影像)
│   │   │   └── base.py                # PreprocessPipeline (Facade)
│   │   │
│   │   ├── embedding/                 # 嵌入向量萃取
│   │   │   ├── base.py                # BaseExtractor (ABC)
│   │   │   ├── feature_extractor.py   # FeatureExtractor (Singleton + Registry)
│   │   │   ├── feature_ops.py         # calculate_differences, add_demographics
│   │   │   ├── dlib_extractor.py      # DlibExtractor (128維)
│   │   │   ├── arcface_extractor.py   # ArcFaceExtractor (512維)
│   │   │   ├── topofr_extractor.py    # TopoFRExtractor (512維)
│   │   │   └── vggface_extractor.py   # VGGFaceExtractor (4096維)
│   │   │
│   │   ├── age/                       # 年齡預測
│   │   │   ├── predictor.py           # MiVOLOPredictor
│   │   │   └── calibration.py         # BootstrapCorrector, MeanCorrector
│   │   │
│   │   ├── emotion/                   # 情緒/AU 分析
│   │   │   ├── extractor/             # AU 提取器
│   │   │   │   ├── au_config.py       # AU 映射與設定
│   │   │   │   └── au/                # OpenFace, LibreFace, Py-Feat, POSTER++, Gaze
│   │   │   └── postprocess/           # 後處理
│   │   │       ├── harmonizer.py      # AUHarmonizer (跨工具標準化)
│   │   │       └── aggregator.py      # TemporalAggregator (時序統計)
│   │   │
│   │   ├── asymmetry/                 # 面部不對稱性
│   │   │   └── landmark_asymmetry.py  # LandmarkAsymmetryAnalyzer
│   │   │
│   │   └── rotation/                  # 頭部旋轉
│   │       ├── angle_calc.py          # VectorAngleCalculator, PnPAngleCalculator
│   │       ├── features.py            # extract_rotation_features()
│   │       └── plotter.py             # AnglePlotter, process_single_folder
│   │
│   ├── analysis/                      # 分析模組
│   │   ├── loader/                    # 資料載入子模組
│   │   │   ├── base.py                # Dataset, DataLoaderProtocol
│   │   │   ├── balancer.py            # DataBalancer (年齡分層平衡)
│   │   │   ├── data_loader.py         # DataLoader (主類別)
│   │   │   └── au_dataset.py          # AUDatasetLoader (AU 特徵載入)
│   │   │
│   │   ├── analyzer/                  # 分析器子模組
│   │   │   ├── __init__.py            # create_analyzer 工廠函數、ANALYZER_REGISTRY
│   │   │   ├── base.py                # BaseAnalyzer (ABC)
│   │   │   ├── xgboost_analyzer.py    # XGBoostAnalyzer
│   │   │   ├── logistic_analyzer.py   # LogisticAnalyzer
│   │   │   └── tabpfn_analyzer.py     # TabPFNAnalyzer
│   │   │
│   │   ├── shap_explainer.py          # AUSHAPExplainer (SHAP 可解釋性)
│   │   ├── cross_tool_comparison.py   # CrossToolComparator (跨工具比較)
│   │   └── plotter.py                 # ResultPlotter (結果視覺化)
│   │
│   ├── meta_analysis/                 # Meta 分析模組
│   │   ├── config.py                  # MetaConfig (dataclass)
│   │   ├── pipeline.py                # MetaPipeline (執行流程)
│   │   ├── data/
│   │   │   ├── dataset.py             # MetaDataset (14 特徵資料結構)
│   │   │   └── loader.py              # MetaDataLoader (資料載入合併)
│   │   └── model/
│   │       ├── trainer.py             # TabPFNMetaTrainer, TrainResult
│   │       └── evaluator.py           # MetaEvaluator (指標計算)
│   │
├── scripts/                           # 執行腳本
│   ├── _paths.py                      # 共用路徑解析
│   ├── _utils.py                      # 共用工具函式
│   │
│   ├── pipeline/                      # 主 Pipeline（依序執行）
│   │   ├── predict_ages.py            #   Stage 0：年齡預測 (MiVOLO)
│   │   ├── prepare_feature.py         #   Stage 1：前處理 + 嵌入萃取
│   │   ├── run_au_pipeline.py         #   Stage 1b：AU 提取 → 標準化 → 聚合
│   │   ├── run_fer_extract.py         #   Stage 1c：FER 情緒提取
│   │   ├── run_new_models_extract.py  #   Stage 1d：DAN/HSEmotion/ViT 提取
│   │   ├── process_angle.py           #   Stage 1e：頭部旋轉角度計算
│   │   ├── run_analyze.py             #   Stage 2：分類器訓練
│   │   └── run_meta_analysis.py       #   Stage 3：Meta 分析
│   │
│   ├── experiments/                   # 探索性 / 替代分析
│   │   ├── compute_emotion_scores.py  #   EmoNet 情緒分數計算
│   │   ├── run_classification.py      #   AU 特徵 XGBoost 分類 + SHAP
│   │   ├── run_au_emo_stats.py        #   AU/情緒統計分析
│   │   ├── run_extended_analysis.py   #   5 模式延伸分析 (CDR/認知/情緒/縱向/性別)
│   │   ├── run_pca_eigenvector.py     #   AU PCA 特徵向量分析
│   │   ├── run_homogeneity_analysis.py #  AU 同質性分析
│   │   ├── AD_binary_analysis.py      #   認知變化二元預測 (VGGFace 4096-d)
│   │   ├── run_xgboost_modules.py     #   M3/M4 獨立 XGBoost
│   │   ├── run_full_features_classifier.py  # 1036-d 直接分類
│   │   ├── run_tabpfn_m1m2_512.py     #   M1+M2 512-d TabPFN
│   │   └── run_m3m4_deep_analysis.py  #   M3+M4 深度分析 & 出圖
│   │
│   ├── visualization/                 # 繪圖 & 統計
│   │   ├── plot_predicted_ages.py     #   年齡預測統計圖
│   │   ├── plot_valence_arousal.py    #   Valence-Arousal 散佈圖
│   │   ├── demographics_statistics.py #   人口學統計
│   │   ├── plot_auc_analysis.py       #   AU 情緒 ROC AUC 分析
│   │   ├── plot_emotion_comparison.py #   跨模型情緒比較圖
│   │   ├── plot_tabpfn_meta_by_n_features.py   # TabPFN Meta 趨勢圖
│   │   └── plot_xgboost_meta_by_n_features.py  # XGBoost Meta 趨勢圖
│   │
│   └── utilities/                     # 一次性處理工具
│       ├── extract_original_features.py       # 原始嵌入提取（不做鏡射）
│       ├── calibrate_age_prediction.py        # 年齡預測校正
│       ├── age_error_bootstrap_correction.py  # Bootstrap 年齡誤差校正
│       ├── age_error_mean_correction.py       # Mean 年齡誤差校正
│       ├── rotation_stat.py                   # 旋轉影像統計
│       └── compare_features.py                # 特徵比對視覺化
│
├── external/                          # 外部模型與權重
│   ├── embedding/                     # 嵌入模型
│   │   ├── dlib/                      # Dlib 人臉辨識模型
│   │   └── TopoFR/                    # TopoFR 拓撲特徵模型
│   └── emotion/                       # 情緒/AU 模型
│       ├── emonet/                    # EmoNet (git submodule)
│       ├── openface/                  # OpenFace 3.0 權重
│       ├── libreface/                 # LibreFace 權重
│       ├── DAN/                       # DAN 模型原始碼
│       ├── dan_weights/               # DAN 權重 (324MB)
│       ├── EmoNeXt/                   # EmoNeXt 模型
│       ├── fer/                       # FER 模型
│       ├── FER-former/                # FER-former 模型
│       ├── POSTER_V2/                 # POSTER V2 模型
│       ├── poster_pp/                 # POSTER++ 權重 (356MB)
│       └── ExpressNet-MoE/            # ExpressNet-MoE 模型
│
├── envs/                              # Conda 環境配置
│   ├── emo_analyze.txt                # 情緒分析主環境
│   ├── libreface.txt                  # LibreFace 環境
│   └── pyfeat.txt                     # Py-Feat 環境
│
├── data/
│   ├── path.txt                       # 原始圖片路徑指向
│   └── demographics/                  # 人口學資料 (ACS.csv, NAD.csv, P.csv)
│
├── references/                        # 參考論文（依模組分類）
│   ├── embedding/                     # dlib, ArcFace, VGGFace, TopoFR
│   ├── preprocessing/                 # MediaPipe
│   ├── age/                           # MiVOLO
│   ├── emotion/                       # 13 篇 FER/AU 論文
│   └── analysis/                      # TabPFN
│
└── workspace/                         # 工作區（輸出）
    ├── predicted_ages.json
    ├── emotion_score.csv
    ├── features/
    └── analysis*/
```

---

## 完整函式調用圖

```
================================================================================
[階段零：年齡預測]
================================================================================

scripts/pipeline/predict_ages.py
│
├── read_raw_path()                         ← 讀取 path.txt
├── scan_subjects()                         ← 掃描 health/ACS, health/NAD, patient/
├── load_images()                           ← 載入每個受試者的前 N 張圖
│
└── src/extractor/features/age/predictor.py
        └── MiVOLOPredictor
            ├── __init__()
            ├── initialize()                ← 載入 MiVOLO v2 模型 (HuggingFace)
            ├── predict()                   ← 批次預測多張圖
            │   └── predict_single()        ← 單張預測
            │       ├── cv2.CascadeClassifier  ← Haar 人臉偵測
            │       └── model 推論
            │               │
            │               ↓
            └──────────────────────────────────── workspace/predicted_ages.json


================================================================================
[階段一：特徵準備]
================================================================================

scripts/pipeline/prepare_feature.py
│
├── FeaturePipeline.__init__()
│   ├── _setup_cpu_limit()                  ← 設定 OMP/MKL/OpenCV 執行緒數
│   └── _setup_output_dirs()                ← 建立 workspace/features/{model}/{type}/
│
└── FeaturePipeline.run()
        │
        ├── _scan_subjects()                ← 掃描 ACS/NAD/P 目錄
        │   └── _has_images()               ← 檢查目錄是否有圖片
        │
        ├── _get_processed_subjects()       ← 斷點續傳：檢查已處理的受試者
        │
        ├── src/extractor/features/embedding/           ← 嵌入向量萃取子模組
        │       ├── feature_extractor.py
        │       │   └── FeatureExtractor    ← Singleton + 類別級 Registry
        │       │       ├── @register()     ← 裝飾器註冊 extractor
        │       │       ├── _get_extractor()← 懶載入
        │       │       └── _report_status()← 報告載入狀態
        │       │
        │       └── feature_ops.py          ← 純函式：特徵後處理
        │           ├── calculate_differences()
        │           └── add_demographics()
        │
        │       底層 extractors (繼承 BaseExtractor):
        │       ├── dlib_extractor.py       → DlibExtractor (128維)
        │       ├── arcface_extractor.py    → ArcFaceExtractor (512維)
        │       ├── topofr_extractor.py     → TopoFRExtractor (512維)
        │       └── vggface_extractor.py    → VGGFaceExtractor (4096維)
        │
        └── 對每個受試者：_process_subject()
                │
                ├── _load_images_from_subject()     ← 載入 jpg/png 圖片
                │
                ├── src/config.py
                │       └── PreprocessConfig        ← 預處理配置 dataclass
                │           ├── n_select = 10
                │           ├── align_face = True
                │           ├── mirror_size = (512, 512)
                │           ├── mirror_method = "flip"
                │           └── steps = ["select", "align", "mirror"]
                │
                ├── src/extractor/preprocess/          ← 臉部預處理子模組
                │       │
                │       ├── detector.py
                │       │   └── FaceDetector        ← MediaPipe 人臉偵測
                │       │       ├── detect()        ← 偵測 468 點 landmarks
                │       │       └── _calculate_vertex_angle_sum()
                │       │               │
                │       │               ↓
                │       │           FaceInfo(landmarks, angle_sum, ...)
                │       │
                │       ├── selector.py
                │       │   └── FaceSelector        ← 選擇最正面的臉
                │       │       └── select()        ← sorted(key=vertex_angle_sum)
                │       │               │
                │       │               ↓
                │       │           List[FaceInfo] (top n)
                │       │
                │       ├── aligner.py
                │       │   └── FaceAligner         ← 角度校正
                │       │       └── align()
                │       │           ├── _build_face_mask()      ← 建立凸包遮罩
                │       │           ├── _calculate_midline_tilt() ← 計算傾斜角度
                │       │           └── cv2.warpAffine()        ← 旋轉影像
                │       │
                │       ├── mirror.py
                │       │   └── MirrorGenerator     ← 生成左右鏡射
                │       │       └── generate()
                │       │           ├── _estimate_midline()     ← PCA 估計中線
                │       │           └── _create_mirror()        ← 鏡射+合成
                │       │               ├── cv2.remap()         ← 映射像素
                │       │               └── alpha 混合 + 裁切置中
                │       │
                │       └── base.py
                │           └── PreprocessPipeline  ← Facade 統一介面
                │               │   (別名: FacePreprocessor)
                │               │
                │               └── process()       ← 主流程
                │                   ├── detector.detect()
                │                   ├── selector.select()
                │                   ├── aligner.align()
                │                   ├── mirror.generate()
                │                   └── _save_results()
                │                           │
                │                           ↓
                │                   ProcessedFace(aligned, left_mirror, right_mirror)
                │
                ├── src/extractor/features/embedding/
                │       └── FeatureExtractor
                │           │
                │           ├── extract_features()                      ← 批次提取特徵
                │           │   └── 對每個 extractor:
                │           │       └── extractor.extract_batch()
                │           │
                │           └── feature_ops.calculate_differences()     ← 計算左右臉差異
                │               ├── "differences"                → left - right
                │               ├── "absolute_differences"       → |left - right|
                │               ├── "averages"                   → (left + right) / 2
                │               ├── "relative_differences"       → diff / norm
                │               └── "absolute_relative_differences"
                │                       │
                │                       ↓
                │               {method: feature_array}
                │
                └── _save_subject_features()
                        │
                        ↓
                workspace/features/{model}/{feature_type}/{subject_id}.npy


================================================================================
[階段二：分析訓練]
================================================================================

scripts/pipeline/run_analyze.py
│
├── AnalysisPipeline.__init__()
│   └── _log_configuration()                ← 記錄設定參數
│
└── AnalysisPipeline.run()
        │
        └── 對每個 min_age 閾值：
                │
                ├── _load_datasets()
                │       │
                │       └── src/meta_analysis/loader/            ← 資料載入子模組
                │               │
                │               ├── base.py
                │               │   └── Dataset             ← 資料集封裝 dataclass
                │               │       (X, y, subject_ids, base_ids, metadata)
                │               │
                │               ├── demographics.py
                │               │   └── DemographicsLoader  ← 人口學資料載入
                │               │       └── load()
                │               │           ├── pd.read_csv("ACS.csv")
                │               │           ├── pd.read_csv("NAD.csv")
                │               │           └── pd.read_csv("P.csv")
                │               │
                │               ├── balancer.py
                │               │   └── DataBalancer        ← 年齡分層平衡
                │               │       └── balance()
                │               │           ├── pd.qcut() 分箱
                │               │           └── 多數類別降採樣
                │               │
                │               └── data_loader.py
                │                   └── DataLoader          ← 主類別
                │                       │
                │                       ├── __init__()
                │                       │   └── _load_predicted_ages()
                │                       │
                │                       └── load_datasets_with_stats()
                │                           │
                │                           ├── _calculate_filter_stats()
                │                           │
                │                           └── 對每個 model × feature_type × cdr_threshold：
                │                               │
                │                               └── _create_dataset()
                │                                   ├── _load_features()
                │                                   ├── _filter_by_predicted_age()
                │                                   ├── _filter_demographics()
                │                                   ├── DataBalancer.balance()
                │                                   └── _align_features_labels()
                │                                           │
                │                                           ↓
                │                                       Dataset(X, y, subject_ids, base_ids)
                │
                ├── _train_models()
                │       │
                │       └── src/meta_analysis/classifier/          ← 分析器子模組
                │               │
                │               ├── __init__.py
                │               │   └── create_analyzer()   ← 工廠函數
                │               │       └── ANALYZER_REGISTRY
                │               │           ├── "xgboost"  → XGBoostAnalyzer
                │               │           ├── "logistic" → LogisticAnalyzer
                │               │           └── "tabpfn"   → TabPFNAnalyzer
                │               │
                │               ├── base.py
                │               │   └── BaseAnalyzer        ← ABC 基底類別
                │               │
                │               └── [xgboost|logistic|tabpfn]_analyzer.py
                │                   └── *Analyzer
                │                       │
                │                       ├── analyze()       ← 分析所有 dataset
                │                       │
                │                       └── _analyze_with_feature_reduction()
                │                               │
                │                               └── while 特徵數 >= 5:
                │                                       │
                │                                       ├── if per_image:
                │                                       │   └── _run_kfold_cv_per_image()
                │                                       │       ├── GroupKFold(base_ids)
                │                                       │       ├── model.fit() / .predict()
                │                                       │       ├── _aggregate_predictions()
                │                                       │       └── _calculate_metrics()
                │                                       │
                │                                       ├── else (averaged):
                │                                       │   └── _run_kfold_cv()
                │                                       │       ├── GroupKFold(base_ids)
                │                                       │       ├── model.fit() / .predict()
                │                                       │       └── _calculate_metrics()
                │                                       │
                │                                       ├── _aggregate_fold_results()
                │                                       ├── _save_result()
                │                                       │
                │                                       └── 捨棄最低重要性的 n_drop_features 個特徵
                │                                               │
                │                                               ↓
                │                                       workspace/analysis/models/
                │                                       workspace/analysis/reports/
                │                                       workspace/analysis/pred_probability/
                │
                ├── _plot_results()
                │       │
                │       └── src/meta_analysis/evaluation/plotter.py
                │               └── ResultPlotter
                │                   └── plot_by_n_features()    ← 按特徵數趨勢
                │                           │
                │                           ↓
                │                   workspace/analysis/plots/
                │
                └── _save_summary()
                        │
                        ↓
                workspace/analysis/training_summary.json


================================================================================
[階段三：Meta 分析]
================================================================================

scripts/pipeline/run_meta_analysis.py
│
├── MetaConfig                              ← 分析設定 dataclass
│   ├── models = ["arcface", ...]
│   ├── asymmetry_method = "absolute_relative_differences"
│   ├── n_folds = 10
│   └── n_features_list = None (自動發現)
│
└── MetaPipeline.__init__()
    │   ├── MetaDataLoader.discover_n_features()  ← 自動掃描可用 n_features
    │   └── _ensure_output_dirs()
    │
    └── MetaPipeline.run()
            │
            └── 對每個 model × n_features：
                    │
                    └── run_single()
                            │
                            ├── src/meta_analysis/data/
                            │       │
                            │       ├── loader.py
                            │       │   └── MetaDataLoader      ← 14 特徵資料載入
                            │       │       ├── _load_lr_predictions("original")
                            │       │       │       → lr_score_original
                            │       │       ├── _load_lr_predictions(asymmetry_method)
                            │       │       │       → lr_score_asymmetry
                            │       │       ├── _load_emotion_scores()
                            │       │       │       → 8 表情 + Valence + Arousal
                            │       │       ├── _load_age_data()
                            │       │       │       → age_error, real_age
                            │       │       ├── _merge_all()
                            │       │       ├── _infer_labels()  ← 從 subject_id 推斷
                            │       │       └── _create_dataset()
                            │       │               │
                            │       │               ↓
                            │       └── dataset.py
                            │           └── MetaDataset(X, y, subject_ids,
                            │                   base_ids, fold_assignments,
                            │                   feature_names)
                            │               14 特徵:
                            │               [age_error, real_age,
                            │                lr_score_original, lr_score_asymmetry,
                            │                Anger, Contempt, Disgust, Fear,
                            │                Happiness, Neutral, Sadness, Surprise,
                            │                Valence, Arousal]
                            │
                            ├── src/meta_analysis/model/
                            │       │
                            │       ├── trainer.py
                            │       │   └── TabPFNMetaTrainer
                            │       │       └── train()
                            │       │           ├── GroupKFold(base_ids)
                            │       │           ├── TabPFNClassifier.fit() / .predict()
                            │       │           ├── MetaEvaluator.calculate()
                            │       │           ├── permutation_importance()
                            │       │           └── MetaEvaluator.aggregate_fold_metrics()
                            │       │                   │
                            │       │                   ↓
                            │       │               TrainResult(test_metrics,
                            │       │                   train_metrics, fold_metrics,
                            │       │                   feature_importance, predictions)
                            │       │
                            │       └── evaluator.py
                            │           └── MetaEvaluator
                            │               ├── calculate()               ← 單 fold 指標
                            │               ├── aggregate_fold_metrics()  ← 多 fold 聚合
                            │               └── format_metrics_report()
                            │
                            └── _save_results()
                                    │
                                    ↓
                            workspace/tabpfn_meta_analysis/
                                ├── reports/         # 報告 + 特徵重要性
                                ├── pred_probability/# 預測分數
                                └── summary.csv      # 彙整結果
```

---

## 模組功能說明

### src/config.py (全專案共用配置)

| 類別 | 功能 |
|------|------|
| `PreprocessConfig` | 預處理配置 (468點、對齊、鏡像參數) |
| `APIConfig` | API 用配置 (暫存、清理) |
| `AnalyzeConfig` | 分析用配置 (儲存中間結果) |
| 路徑常數 | `PROJECT_ROOT`, `RAW_IMAGES_DIR`, `FEATURES_DIR`, `WORKSPACE_DIR` 等 |

### src/common/ (跨模組共用工具)

| 檔案 | 類別 | 功能 |
|------|------|------|
| `demographics.py` | `DemographicsLoader` | 人口學資料載入 |
| `mediapipe_utils.py` | — | 統一 landmark 常數 (midline, bilateral, rotation) |
| `metrics.py` | `MetricsCalculator` | 評估指標計算 |

### src/extractor/

#### src/extractor/preprocess/ (臉部預處理)

| 檔案 | 類別 | 功能 |
|------|------|------|
| `detector.py` | `FaceDetector` | MediaPipe 人臉偵測、landmarks 萃取 |
| `selector.py` | `FaceSelector` | 選擇最正面的 n 張臉 |
| `aligner.py` | `FaceAligner` | 臉部角度校正 (旋轉對齊) |
| `mirror.py` | `MirrorGenerator` | 生成左右鏡射影像 |
| `base.py` | `PreprocessPipeline` | Facade 統一介面 (別名: `FacePreprocessor`) |
| | `FaceInfo` | 單張臉部資訊 dataclass |
| | `ProcessedFace` | 處理後臉部資料 dataclass |

#### src/extractor/features/embedding/ (嵌入向量萃取)

| 檔案 | 類別/函式 | 功能 |
|------|-----------|------|
| `base.py` | `BaseExtractor` | ABC 基底類別 |
| `feature_extractor.py` | `FeatureExtractor` | Singleton + 類別級 Registry、懶載入、批次提取 |
| `dlib_extractor.py` | `DlibExtractor` | Dlib 128維特徵 |
| `arcface_extractor.py` | `ArcFaceExtractor` | ArcFace/InsightFace 512維特徵 |
| `topofr_extractor.py` | `TopoFRExtractor` | TopoFR 512維特徵 |
| `vggface_extractor.py` | `VGGFaceExtractor` | VGGFace 4096維特徵 |

#### src/extractor/features/age/ (年齡預測)

| 檔案 | 類別/函式 | 功能 |
|------|-----------|------|
| `predictor.py` | `MiVOLOPredictor` | MiVOLO v2 年齡預測 (canonical) |
| `calibration.py` | `BootstrapCorrector` | Bootstrap 年齡誤差校正 |
| | `MeanCorrector` | Mean 年齡誤差校正 |
| | `CalibrationModel` | 校正模型基底 |

#### src/extractor/features/emotion/ (情緒/AU 分析)

| 檔案 | 類別/函式 | 功能 |
|------|-----------|------|
| `extractor/au_config.py` | — | AU 映射、模型權重路徑、常數定義 |
| `extractor/au/openface.py` | `OpenFaceExtractor` | OpenFace AU 提取 |
| `extractor/au/libreface.py` | `LibreFaceExtractor` | LibreFace AU 提取 |
| `extractor/au/pyfeat.py` | `PyFeatExtractor` | Py-Feat AU 提取 |
| `extractor/au/poster_pp.py` | `PosterPPExtractor` | POSTER++ AU 提取 |
| `extractor/au/gaze.py` | `GazeFeatureExtractor` | 視線特徵提取 |
| `postprocess/harmonizer.py` | `AUHarmonizer` | 跨工具 AU 標準化 |
| `postprocess/aggregator.py` | `TemporalAggregator` | 時序統計聚合 |

#### src/extractor/features/asymmetry/ (面部不對稱性)

| 檔案 | 類別/函式 | 功能 |
|------|-----------|------|
| `landmark_asymmetry.py` | `LandmarkAsymmetryAnalyzer` | 點/線/三角面積不對稱計算 |
| `feature_ops.py` | `calculate_differences()` | 計算左右特徵差異 (5 種方法) |
| | `add_demographics()` | 加入年齡/性別人口學特徵 |

#### src/extractor/features/rotation/ (頭部旋轉)

| 檔案 | 類別/函式 | 功能 |
|------|-----------|------|
| `angle_calc.py` | `VectorAngleCalculator` | 向量法角度計算 |
| | `PnPAngleCalculator` | PnP 法角度計算 |
| `features.py` | `extract_rotation_features()` | 從角度序列提取統計特徵 |
| `plotter.py` | `AnglePlotter` | 角度訊號圖繪製 |
| | `process_single_folder()` | 雙方法批次處理 |

### src/meta_analysis/ (分析模組)

#### src/meta_analysis/loader/ (資料載入)

| 檔案 | 類別 | 功能 |
|------|------|------|
| `base.py` | `Dataset` | 資料集封裝 dataclass |
| | `DataLoaderProtocol` | Protocol 定義 |
| `balancer.py` | `DataBalancer` | 年齡分層平衡 |
| `embedding.py` | `DataLoader` | Embedding .npy 載入、篩選、平衡 |
| `au_dataset.py` | `AUDatasetLoader` | AU 特徵資料集載入 |
| `meta.py` | `MetaDataLoader` | 載入 LR 分數 + emotion + 年齡 → 合併 |
| `dataset.py` | `MetaDataset` | 14 特徵資料集 dataclass |

#### src/meta_analysis/classifier/ (Base-level 分類器)

| 檔案 | 類別 | 功能 |
|------|------|------|
| `__init__.py` | `create_analyzer()` | 工廠函數，依類型建立分析器 |
| `base.py` | `BaseAnalyzer` | ABC 基底類別 (K-fold CV) |
| `xgboost.py` | `XGBoostAnalyzer` | XGBoost K-fold CV、特徵選擇 |
| `logistic.py` | `LogisticAnalyzer` | Logistic Regression、係數重要性 |
| `tabpfn.py` | `TabPFNAnalyzer` | TabPFN、permutation importance |

#### src/meta_analysis/stacking/ (Meta-level stacking)

| 檔案 | 類別 | 功能 |
|------|------|------|
| `config.py` | `MetaConfig` | Meta 分析設定 (模型、方法、fold 數) |
| `pipeline.py` | `MetaPipeline` | 遍歷 model × n_features 組合執行分析 |
| `trainer.py` | `TabPFNMetaTrainer` | GroupKFold CV 訓練 TabPFN |
| `evaluator.py` | `MetaEvaluator` | 指標計算、fold 聚合、報告格式化 |

#### src/meta_analysis/evaluation/ (評估工具)

| 檔案 | 類別 | 功能 |
|------|------|------|
| `plotter.py` | `ResultPlotter` | 結果視覺化 (按特徵數趨勢圖) |
| `shap_explainer.py` | `AUSHAPExplainer` | SHAP 可解釋性分析 |
| `cross_tool_comparison.py` | `CrossToolComparator` | 跨工具 AU/情緒一致性比較 |

---

## 資料流程

```
原始圖片 (由 data/path.txt 指向)
    │
    ↓ [predict_ages.py]
predicted_ages.json
    │
    ↓ [prepare_feature.py]
    ├── MediaPipe 偵測 468 特徵點
    ├── 選擇最正面 N 張
    ├── 旋轉對齊
    ├── 生成左右鏡像
    └── 萃取 embedding (Dlib/ArcFace/TopoFR)
    │
    ↓
workspace/features/{model}/{type}/{subject}.npy
    │
    ├──────────────────────────────────────────────────┐
    │                                                  │
    ↓ [run_analyze.py]                                 │
    ├── 載入特徵 + 人口學資料                           │
    ├── CDR 篩選、年齡篩選                              │
    ├── 資料平衡（可選）                                │
    ├── XGBoost / Logistic / TabPFN K-fold CV          │
    └── 遞迴特徵消除                                    │
    │                                                  │
    ↓                                                  │
workspace/analysis/                                    │
    ├── models/          # 訓練好的模型                 │
    ├── reports/         # 分析報告                     │
    ├── pred_probability/# 預測分數 ──┐                │
    ├── plots/           # 結果圖表    │                │
    └── training_summary.json          │                │
                                       │                │
    ┌──────────────────────────────────┘                │
    │          ┌───────────────────────────────────────┘
    │          │
    │          ↓ [compute_emotion_scores.py]
    │   workspace/emotion_score.csv
    │          │
    ↓          ↓
    [run_meta_analysis.py]
    ├── 合併 14 特徵:
    │     LR 分數 (original + asymmetry)
    │     + age_error + real_age
    │     + 8 表情 + Valence + Arousal
    ├── TabPFN 10-Fold CV
    └── 特徵重要性 (permutation importance)
    │
    ↓
workspace/tabpfn_meta_analysis/
    ├── reports/         # 報告 + 特徵重要性
    ├── pred_probability/# 預測分數
    └── summary.csv      # 彙整結果
```

---

## 使用方式

```bash
# 0. 年齡預測
python scripts/pipeline/predict_ages.py

# 1. 特徵準備（預處理 + 嵌入提取）
python scripts/pipeline/prepare_feature.py

# 1b. 原始嵌入提取（不做鏡射，用於 original 特徵）
python scripts/utilities/extract_original_features.py

# 1c. 情緒分數計算
python scripts/experiments/compute_emotion_scores.py

# 2. 分析訓練（XGBoost / Logistic / TabPFN）
python scripts/pipeline/run_analyze.py

# 3. Meta 分析（整合 14 特徵 → TabPFN）
python scripts/pipeline/run_meta_analysis.py
```

---

## 支援的模型

### 嵌入模型

| 模型 | 維度 | 來源 |
|------|------|------|
| Dlib | 128 | dlib face_recognition |
| ArcFace | 512 | InsightFace buffalo_l |
| TopoFR | 512 | external/TopoFR |
| VGGFace | 4096 | keras-vggface |

### 分類器

| 分類器 | 用途 | 特徵重要性 |
|--------|------|------------|
| XGBoost | 階段二分析 | 內建 feature importance |
| Logistic Regression | 階段二分析 | 係數絕對值 |
| TabPFN | 階段二分析 / Meta 分析 | permutation importance |

### 其他

| 模型 | 功能 |
|------|------|
| MiVOLO v2 | 年齡預測 (HuggingFace) |
| EmoNet | 情緒分數 (8 表情 + Valence + Arousal) |
