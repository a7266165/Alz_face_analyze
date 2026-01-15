# Alz_face_analyze

阿茲海默症臉部不對稱性分析系統

## 專案結構

```
Alz_face_analyze/
├── src/
│   ├── core/                   # 核心處理模組
│   │   ├── config.py           # 配置 dataclass
│   │   ├── preprocess.py       # 臉部預處理
│   │   ├── feature_extract.py  # 特徵萃取
│   │   └── age_predictor.py    # 年齡預測
│   │
│   └── analysis/               # 分析模組
│       ├── loader.py           # 資料載入
│       ├── analyzer.py         # XGBoost 分析
│       └── plotter.py          # 結果視覺化
│
├── scripts/                    # 執行腳本
│   ├── predict_ages.py         # 階段零：年齡預測
│   ├── prepare_feature.py      # 階段一：特徵準備
│   └── run_analyze.py          # 階段二：分析訓練
│
├── data/
│   ├── images/raw/path.txt     # 原始圖片路徑指向
│   └── demographics/           # 人口學資料 (ACS.csv, NAD.csv, P.csv)
│
└── workspace/                  # 工作區（輸出）
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
        ├── src/core/feature_extract.py
        │       └── FeatureExtractor.__init__()
        │           ├── _init_dlib()        ← 載入 Dlib 模型 (128維)
        │           ├── _init_arcface()     ← 載入 ArcFace/InsightFace (512維)
        │           ├── _init_topofr()      ← 載入 TopoFR (512維)
        │           └── _report_status()    ← 報告載入狀態
        │
        └── 對每個受試者：_process_subject()
                │
                ├── _load_images_from_subject()     ← 載入 jpg/png 圖片
                │
                ├── src/core/config.py
                │       └── AnalyzeConfig           ← 預處理配置 dataclass
                │           ├── n_select = 10
                │           ├── align_face = True
                │           ├── mirror_size = (512, 512)
                │           └── steps = ["select", "align", "mirror"]
                │
                ├── src/core/preprocess.py
                │       └── FacePreprocessor
                │           │
                │           ├── __init__()
                │           │   ├── mp.solutions.face_mesh.FaceMesh()  ← 初始化 MediaPipe
                │           │   └── _setup_workspace()                  ← 建立子目錄
                │           │
                │           └── process()                               ← 主流程
                │               │
                │               ├── _analyze_all_faces()                ← 分析所有圖片
                │               │   ├── face_mesh.process()             ← MediaPipe 偵測 468 點
                │               │   ├── _landmarks_to_array()           ← 轉換座標格式
                │               │   └── _calculate_vertex_angle_sum()   ← 計算中軸彎曲角度
                │               │           │
                │               │           ↓
                │               │       List[FaceInfo]
                │               │
                │               ├── _select_best_faces()                ← 選擇最正面的 n 張
                │               │   └── sorted(key=vertex_angle_sum)
                │               │           │
                │               │           ↓
                │               │       selected faces
                │               │
                │               └── _process_single_face()              ← 處理單張臉
                │                       │
                │                       ├── _align_face()               ← 角度校正
                │                       │   ├── _build_face_mask()      ← 建立凸包遮罩
                │                       │   ├── _calculate_midline_tilt() ← 計算傾斜角度
                │                       │   └── cv2.warpAffine()        ← 旋轉影像
                │                       │
                │                       ├── _create_mirror_images()     ← 生成左右鏡射
                │                       │   └── _create_midline_mirrors()
                │                       │       ├── _estimate_midline()  ← PCA 估計中線
                │                       │       └── _align_to_canvas_premul() ← 鏡射+合成
                │                       │           ├── 計算反射座標
                │                       │           ├── cv2.remap()      ← 映射像素
                │                       │           └── alpha 混合 + 裁切置中
                │                       │
                │                       ├── _save_selected()            ← 儲存選中影像
                │                       ├── _save_aligned()             ← 儲存對齊影像
                │                       └── _save_mirrors()             ← 儲存鏡射影像
                │                               │
                │                               ↓
                │                       ProcessedFace(aligned, left_mirror, right_mirror)
                │
                ├── src/core/feature_extract.py
                │       └── FeatureExtractor
                │           │
                │           ├── extract_features()                      ← 批次提取特徵
                │           │   ├── _extract_dlib()                     ← Dlib 128維
                │           │   │   ├── dlib_detector()                 ← 人臉偵測
                │           │   │   ├── dlib_predictor()                ← 68 特徵點
                │           │   │   └── dlib_face_rec.compute_face_descriptor()
                │           │   │
                │           │   ├── _extract_arcface()                  ← ArcFace 512維
                │           │   │   └── arcface_app.get()               ← InsightFace API
                │           │   │
                │           │   └── _extract_topofr()                   ← TopoFR 512維
                │           │       ├── cv2.resize(112, 112)
                │           │       └── topofr_model()                  ← PyTorch 推論
                │           │
                │           └── calculate_differences()                 ← 計算左右臉差異
                │               ├── "differences"      → left - right
                │               ├── "absolute_differences" → |left - right|
                │               ├── "averages"         → (left + right) / 2
                │               ├── "relative_differences" → diff / norm
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
                │       └── src/analysis/loader.py
                │               └── DataLoader
                │                   │
                │                   ├── __init__()
                │                   │   └── _load_predicted_ages()      ← 載入 predicted_ages.json
                │                   │
                │                   └── load_datasets_with_stats()
                │                       │
                │                       ├── _calculate_filter_stats()   ← 計算篩選統計
                │                       │
                │                       ├── _load_demographics()        ← 載入人口學資料
                │                       │   ├── pd.read_csv("ACS.csv")
                │                       │   ├── pd.read_csv("NAD.csv")
                │                       │   └── pd.read_csv("P.csv")
                │                       │           │
                │                       │           ↓
                │                       │       pd.DataFrame (合併)
                │                       │
                │                       └── 對每個 model × feature_type × cdr_threshold：
                │                           │
                │                           └── _create_dataset()
                │                               │
                │                               ├── _load_features()        ← 載入 .npy 檔案
                │                               │   └── 偵測格式：averaged / per_image
                │                               │
                │                               ├── _filter_by_predicted_age()  ← 年齡篩選
                │                               │   └── ages[id] >= min_predicted_age
                │                               │
                │                               ├── _filter_demographics()      ← CDR 篩選
                │                               │   ├── assign_label()          ← ACS/NAD→0, P→1
                │                               │   ├── NAD: Global_CDR <= threshold
                │                               │   ├── P: Global_CDR >= threshold
                │                               │   └── _keep_latest_visit()    ← 只保留最新訪視
                │                               │
                │                               ├── _apply_data_balancing()     ← 年齡分層平衡
                │                               │   ├── pd.qcut() 分箱
                │                               │   └── 多數類別降採樣
                │                               │
                │                               └── _align_features_labels()    ← 對齊特徵與標籤
                │                                   └── _extract_base_id()      ← P1-2 → P1
                │                                           │
                │                                           ↓
                │                                       Dataset(X, y, subject_ids, base_ids)
                │
                ├── _train_models()
                │       │
                │       └── src/analysis/analyzer.py
                │               └── XGBoostAnalyzer
                │                   │
                │                   ├── __init__()
                │                   │   └── xgb_params 設定
                │                   │
                │                   └── analyze()                       ← 分析所有 dataset
                │                       │
                │                       └── _analyze_with_feature_reduction()
                │                               │
                │                               └── while 特徵數 >= 5:
                │                                       │
                │                                       ├── if per_image:
                │                                       │   └── _run_kfold_cv_per_image()
                │                                       │       ├── GroupKFold(base_ids)
                │                                       │       ├── XGBClassifier.fit()
                │                                       │       ├── _aggregate_predictions()     ← 個案聚合
                │                                       │       │   └── np.mean(y_prob per subject)
                │                                       │       ├── _aggregate_predictions_with_ids()
                │                                       │       ├── _calculate_metrics()
                │                                       │       ├── _calculate_corrected_metrics_fold()
                │                                       │       └── _save_predictions()
                │                                       │
                │                                       ├── else (averaged):
                │                                       │   └── _run_kfold_cv()
                │                                       │       ├── GroupKFold(base_ids)
                │                                       │       ├── XGBClassifier.fit()
                │                                       │       ├── _calculate_metrics()
                │                                       │       │   ├── confusion_matrix()
                │                                       │       │   ├── accuracy_score()
                │                                       │       │   ├── precision_score()
                │                                       │       │   ├── recall_score()
                │                                       │       │   ├── f1_score()
                │                                       │       │   ├── roc_auc_score()
                │                                       │       │   └── matthews_corrcoef()
                │                                       │       ├── _calculate_corrected_metrics_fold()
                │                                       │       └── _save_predictions()
                │                                       │
                │                                       ├── _aggregate_fold_results()   ← 彙整各 fold
                │                                       │
                │                                       ├── _save_result()
                │                                       │   ├── model.save_model()      → models/
                │                                       │   └── _save_report()          → reports/
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
                │       └── src/analysis/plotter.py
                │               └── ResultPlotter
                │                   │
                │                   ├── __init__()
                │                   │   └── 提取 dataset_keys, ages, n_features_list
                │                   │
                │                   ├── plot_individual()               ← 每個 dataset 四張圖
                │                   │   └── _plot_single_figure()
                │                   │       └── 4 metrics × n_features 條線
                │                   │
                │                   ├── plot_combined()                 ← 所有組合合併
                │                   │   └── _plot_combined_figure()
                │                   │
                │                   ├── plot_by_model()                 ← 按模型分組
                │                   │   └── _plot_by_model_figure()
                │                   │
                │                   ├── plot_by_n_features()            ← 按特徵數趨勢
                │                   │   └── X軸=特徵數, 訓練/測試曲線
                │                   │
                │                   └── plot_filter_stats()             ← 篩選統計圖
                │                           │
                │                           ↓
                │                   workspace/analysis/plots/
                │                       ├── individual/
                │                       ├── combined_all.png
                │                       ├── by_model_{model}.png
                │                       ├── by_n_features/
                │                       └── filter_stats.png
                │
                └── _save_summary()
                        │
                        ↓
                workspace/analysis/training_summary.json
```

---

## 模組功能說明

### src/core/

| 檔案 | 類別/函式 | 功能 |
|------|-----------|------|
| `config.py` | `PreprocessConfig` | 基礎預處理配置 (468點、對齊、鏡像參數) |
| | `APIConfig` | API 用配置 (暫存、清理) |
| | `AnalyzeConfig` | 分析用配置 (儲存中間結果) |
| `preprocess.py` | `FaceInfo` | 單張臉部資訊 dataclass |
| | `ProcessedFace` | 處理後臉部資料 dataclass |
| | `FacePreprocessor` | 臉部預處理器 (偵測→選擇→對齊→鏡像) |
| `feature_extract.py` | `FeatureExtractor` | 多模型特徵萃取 (Dlib/ArcFace/TopoFR) |
| `age_predictor.py` | `MiVOLOPredictor` | MiVOLO v2 年齡預測 |

### src/analysis/

| 檔案 | 類別/函式 | 功能 |
|------|-----------|------|
| `loader.py` | `Dataset` | 資料集封裝 dataclass |
| | `DataLoader` | 資料載入、篩選、平衡 |
| `analyzer.py` | `XGBoostAnalyzer` | K-fold CV、特徵選擇、指標計算 |
| `plotter.py` | `ResultPlotter` | 結果視覺化 |

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
