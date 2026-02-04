# Alz_face_analyze

阿茲海默症臉部不對稱性分析系統

## 專案結構

```
Alz_face_analyze/
├── src/
│   ├── core/                       # 核心處理模組
│   │   ├── config.py               # 配置 dataclass
│   │   ├── age_predictor.py        # 年齡預測 (MiVOLO)
│   │   │
│   │   ├── preprocess/             # 臉部預處理子模組
│   │   │   ├── detector.py         # FaceDetector (MediaPipe)
│   │   │   ├── selector.py         # FaceSelector (選擇最正面)
│   │   │   ├── aligner.py          # FaceAligner (角度校正)
│   │   │   ├── mirror.py           # MirrorGenerator (鏡射影像)
│   │   │   └── base.py             # PreprocessPipeline (Facade)
│   │   │
│   │   └── extractor/              # 特徵萃取子模組
│   │       ├── base.py             # BaseExtractor (ABC)
│   │       ├── registry.py         # ExtractorRegistry (Singleton)
│   │       ├── dlib_extractor.py   # DlibExtractor (128維)
│   │       ├── arcface_extractor.py    # ArcFaceExtractor (512維)
│   │       ├── topofr_extractor.py     # TopoFRExtractor (512維)
│   │       └── vggface_extractor.py    # VGGFaceExtractor (4096維)
│   │
│   ├── analysis/                   # 分析模組
│   │   ├── loader/                 # 資料載入子模組
│   │   │   ├── base.py             # Dataset, DataLoaderProtocol
│   │   │   ├── demographics.py     # DemographicsLoader
│   │   │   ├── balancer.py         # DataBalancer (年齡分層平衡)
│   │   │   └── data_loader.py      # DataLoader (主類別)
│   │   │
│   │   ├── analyzer/               # 分析器子模組
│   │   │   ├── base.py             # BaseAnalyzer (ABC)
│   │   │   └── xgboost_analyzer.py # XGBoostAnalyzer
│   │   │
│   │   └── plotter.py              # ResultPlotter (結果視覺化)
│   │
│   └── common/                     # 共用模組
│       ├── types.py                # Protocol 定義、資料型別
│       └── metrics.py              # MetricsCalculator
│
├── scripts/                        # 執行腳本
│   ├── predict_ages.py             # 階段零：年齡預測
│   ├── prepare_feature.py          # 階段一：特徵準備
│   ├── run_analyze.py              # 階段二：分析訓練
│   ├── plot_predicted_ages.py      # 年齡預測統計與視覺化
│   ├── calibrate_age_prediction.py # 年齡預測校正
│   └── demographics_statistics.py  # 人口學統計
│
├── data/
│   ├── images/raw/path.txt         # 原始圖片路徑指向
│   └── demographics/               # 人口學資料 (ACS.csv, NAD.csv, P.csv)
│
├── legacy/                         # 舊版程式碼 (備份參考用)
│
└── workspace/                      # 工作區（輸出）
    ├── predicted_ages.json
    ├── features/
    └── analysis/
```

---

## 完整函式調用圖

```
================================================================================
[階段零：年齡預測]
================================================================================

scripts/predict_ages.py
│
├── read_raw_path()                         ← 讀取 path.txt
├── scan_subjects()                         ← 掃描 health/ACS, health/NAD, patient/
├── load_images()                           ← 載入每個受試者的前 N 張圖
│
└── src/core/age_predictor.py
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

scripts/prepare_feature.py
│
├── FeaturePipeline.__init__()
│   ├── _setup_cpu_limit()                  ← 設定 OMP/MKL/OpenCV 執行緒數
│   ├── _read_path_file()                   ← 讀取 data/images/raw/path.txt
│   └── _setup_output_dirs()                ← 建立 workspace/features/{model}/{type}/
│
└── FeaturePipeline.run()
        │
        ├── _scan_subjects()                ← 掃描 ACS/NAD/P 目錄
        │   └── _has_images()               ← 檢查目錄是否有圖片
        │
        ├── _get_processed_subjects()       ← 斷點續傳：檢查已處理的受試者
        │
        ├── src/core/extractor/             ← 特徵萃取子模組
        │       ├── registry.py
        │       │   └── ExtractorRegistry   ← Singleton 註冊中心
        │       │       ├── register()      ← @register 裝飾器
        │       │       └── get()           ← 依名稱取得 extractor (lazy load)
        │       │
        │       └── FeatureExtractor        ← Facade 包裝類別
        │           ├── __init__()          ← 透過 Registry 載入各 extractor
        │           └── _report_status()    ← 報告載入狀態
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
                ├── src/core/config.py
                │       └── PreprocessConfig        ← 預處理配置 dataclass
                │           ├── n_select = 10
                │           ├── align_face = True
                │           ├── mirror_size = (512, 512)
                │           ├── mirror_method = "midline"
                │           └── steps = ["select", "align", "mirror"]
                │
                ├── src/core/preprocess/            ← 臉部預處理子模組
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
                ├── src/core/extractor/
                │       └── FeatureExtractor
                │           │
                │           ├── extract_features()                      ← 批次提取特徵
                │           │   └── 對每個 extractor:
                │           │       └── extractor.extract()             ← 透過 Registry
                │           │
                │           └── calculate_differences()                 ← 計算左右臉差異
                │               ├── "difference"              → left - right
                │               ├── "absolute_difference"     → |left - right|
                │               ├── "average"                 → (left + right) / 2
                │               ├── "relative_difference"     → diff / norm
                │               └── "absolute_relative_difference"
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

scripts/run_analyze.py
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
                │       └── src/analysis/loader/            ← 資料載入子模組
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
                │       └── src/analysis/analyzer/          ← 分析器子模組
                │               │
                │               ├── base.py
                │               │   └── BaseAnalyzer        ← ABC 基底類別
                │               │
                │               └── xgboost_analyzer.py
                │                   └── XGBoostAnalyzer     ← XGBoost 分析器
                │                       │
                │                       ├── __init__()
                │                       │   └── xgb_params 設定
                │                       │
                │                       └── analyze()       ← 分析所有 dataset
                │                           │
                │                           └── _analyze_with_feature_reduction()
                │                                   │
                │                                   └── while 特徵數 >= 5:
                │                                           │
                │                                           ├── if per_image:
                │                                           │   └── _run_kfold_cv_per_image()
                │                                           │       ├── GroupKFold(base_ids)
                │                                           │       ├── XGBClassifier.fit()
                │                                           │       ├── _aggregate_predictions()
                │                                           │       ├── _calculate_metrics()
                │                                           │       └── _save_predictions()
                │                                           │
                │                                           ├── else (averaged):
                │                                           │   └── _run_kfold_cv()
                │                                           │       ├── GroupKFold(base_ids)
                │                                           │       ├── XGBClassifier.fit()
                │                                           │       ├── _calculate_metrics()
                │                                           │       └── _save_predictions()
                │                                           │
                │                                           ├── _aggregate_fold_results()
                │                                           ├── _save_result()
                │                                           │
                │                                           └── 捨棄最低重要性的 n_drop_features 個特徵
                │                                                   │
                │                                                   ↓
                │                                           workspace/analysis/models/
                │                                           workspace/analysis/reports/
                │                                           workspace/analysis/pred_probability/
                │
                ├── _plot_results()
                │       │
                │       └── src/analysis/plotter.py
                │               └── ResultPlotter
                │                   │
                │                   ├── plot_individual()       ← 每個 dataset 四張圖
                │                   ├── plot_combined()         ← 所有組合合併
                │                   ├── plot_by_model()         ← 按模型分組
                │                   ├── plot_by_n_features()    ← 按特徵數趨勢
                │                   └── plot_filter_stats()     ← 篩選統計圖
                │                           │
                │                           ↓
                │                   workspace/analysis/plots/
                │
                └── _save_summary()
                        │
                        ↓
                workspace/analysis/training_summary.json
```

---

## 模組功能說明

### src/core/

| 檔案/子模組 | 類別 | 功能 |
|-------------|------|------|
| `config.py` | `PreprocessConfig` | 預處理配置 (468點、對齊、鏡像參數) |
| | `APIConfig` | API 用配置 (暫存、清理) |
| | `AnalyzeConfig` | 分析用配置 (儲存中間結果) |
| `age_predictor.py` | `MiVOLOPredictor` | MiVOLO v2 年齡預測 |

#### src/core/preprocess/ (臉部預處理子模組)

| 檔案 | 類別 | 功能 |
|------|------|------|
| `detector.py` | `FaceDetector` | MediaPipe 人臉偵測、landmarks 萃取 |
| `selector.py` | `FaceSelector` | 選擇最正面的 n 張臉 |
| `aligner.py` | `FaceAligner` | 臉部角度校正 (旋轉對齊) |
| `mirror.py` | `MirrorGenerator` | 生成左右鏡射影像 |
| `base.py` | `PreprocessPipeline` | Facade 統一介面 (別名: `FacePreprocessor`) |
| | `FaceInfo` | 單張臉部資訊 dataclass |
| | `ProcessedFace` | 處理後臉部資料 dataclass |

#### src/core/extractor/ (特徵萃取子模組)

| 檔案 | 類別 | 功能 |
|------|------|------|
| `base.py` | `BaseExtractor` | ABC 基底類別 |
| `registry.py` | `ExtractorRegistry` | Singleton 註冊中心、lazy loading |
| `dlib_extractor.py` | `DlibExtractor` | Dlib 128維特徵 |
| `arcface_extractor.py` | `ArcFaceExtractor` | ArcFace/InsightFace 512維特徵 |
| `topofr_extractor.py` | `TopoFRExtractor` | TopoFR 512維特徵 |
| `vggface_extractor.py` | `VGGFaceExtractor` | VGGFace 4096維特徵 |
| `__init__.py` | `FeatureExtractor` | Facade 包裝類別 (向後兼容) |

### src/analysis/

#### src/analysis/loader/ (資料載入子模組)

| 檔案 | 類別 | 功能 |
|------|------|------|
| `base.py` | `Dataset` | 資料集封裝 dataclass |
| | `DataLoaderProtocol` | Protocol 定義 |
| `demographics.py` | `DemographicsLoader` | 人口學資料載入 |
| `balancer.py` | `DataBalancer` | 年齡分層平衡 |
| `data_loader.py` | `DataLoader` | 主類別：載入、篩選、平衡 |

#### src/analysis/analyzer/ (分析器子模組)

| 檔案 | 類別 | 功能 |
|------|------|------|
| `base.py` | `BaseAnalyzer` | ABC 基底類別 |
| `xgboost_analyzer.py` | `XGBoostAnalyzer` | K-fold CV、特徵選擇、指標計算 |

#### src/analysis/plotter.py

| 類別 | 功能 |
|------|------|
| `ResultPlotter` | 結果視覺化 |

### src/common/ (共用模組)

| 檔案 | 類別/函式 | 功能 |
|------|-----------|------|
| `types.py` | Protocol 定義 | `Extractor`, `Analyzer`, `DataLoader` |
| | 資料型別 | `FoldResult`, `TrainingResult`, `DatasetInfo` |
| `metrics.py` | `MetricsCalculator` | 評估指標計算 |

---

## 資料流程

```
原始圖片 (data/images/raw/)
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
    ↓ [run_analyze.py]
    ├── 載入特徵 + 人口學資料
    ├── CDR 篩選、年齡篩選
    ├── 資料平衡
    ├── XGBoost K-fold CV
    └── 遞迴特徵消除
    │
    ↓
workspace/analysis/
    ├── models/          # 訓練好的模型
    ├── reports/         # 分析報告
    ├── plots/           # 結果圖表
    └── training_summary.json
```

---

## 使用方式

```bash
# 1. 年齡預測
python scripts/predict_ages.py

# 2. 特徵準備
python scripts/prepare_feature.py

# 3. 分析訓練
python scripts/run_analyze.py
```

---

## 支援的模型

| 模型 | 維度 | 來源 |
|------|------|------|
| Dlib | 128 | dlib face_recognition |
| ArcFace | 512 | InsightFace buffalo_l |
| TopoFR | 512 | external/TopoFR |
| MiVOLO v2 | - | HuggingFace iitolstykh/mivolo_v2 |
