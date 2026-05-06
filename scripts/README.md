# Scripts

按 `workspace/` 模態組織。每個 modality 資料夾自帶 producer (extract/build) +
analysis (run_*) + visualization (plot_*)。

| 資料夾 | 對應 workspace tree | 內容 |
|---|---|---|
| `preprocess/` | `workspace/preprocess/`、`workspace/preprocess_ABtest/` | 對齊 / 鏡射 / 選圖 (`prepare_feature.py`) |
| `embedding/` | `workspace/embedding/`、`workspace/embedding_ABtest/` | ArcFace / dlib / TopoFR / VGG 特徵 + fwd/rev classifier + sweep + PCA / dropcorr 分析 |
| `asymmetry/` | `workspace/asymmetry/` | 468-landmark asymmetry 抽取 + analysis |
| `age/` | `workspace/age/` | MiVOLO 年齡預測 + age classifier + window classifier + 相關 plots |
| `emo_au/` | `workspace/emo_au/` | FER + AU 抽取 + emotion comparison + valence/arousal |
| `longitudinal/` | `workspace/longitudinal/` | longitudinal patient delta + vector delta builder |
| `rotation/` | `workspace/rotation/` | head pose / vector angle |
| `overview/` | `workspace/overview/` | 跨模態 cohort orchestrators (`run_cohort_pipeline.py` / `run_cross_naive.py` / `run_cross_matched.py` / `run_stat_grid.py`) + cohort overview / deep-dive grid plots |
| `meta/` | `workspace/analysis_<ts>/`、`workspace/tabpfn_meta_analysis_<ts>/` | legacy M1-M4 paper（對應 `src/meta/`），保留至確認無 active caller |
| `external/` | `external/public_face_datasets/` | 公開亞裔人臉資料集整合 (見 `external/README.md`) |
| `literature_monitor/` | `references/` | 文獻自動 monitor (self-contained sub-package) |
| `utilities/` | (cross-cutting libraries, 不對應單一 workspace tree) | `cohort.py` / `feature_loaders.py` / `stats_helpers.py` / `emotion_loader.py` / `facial_landmarks.py` |

## 主要 entry point

跨模態 cohort 分析（最常跑的入口）：
```bash
conda run -n Alz_face_main_analysis python scripts/overview/run_cohort_pipeline.py \
    --cohort-mode {default|p_first_hc_all|p_all_hc_all}
```

Embedding sweep（單獨跑 fwd/rev × 12 reducers × 6 feature types）：
```bash
conda run -n Alz_face_main_analysis python scripts/embedding/run_sweep.py \
    --cohort-mode p_first_hc_all
```

各 modality 內部 entry script 可直接跑（含 `--help`），不必透過 orchestrator。
