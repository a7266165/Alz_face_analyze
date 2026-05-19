# Alz_face_analyze

阿茲海默症臉部多模態分析系統 — 從原始臉部影像抽取 age / emotion / AU / asymmetry / rotation / embedding 特徵，配合 cohort matching + 統計檢定 + classifier sweep 評估與 AD 相關性。

## 專案結構（modality-flat，三軸對齊）

```
Alz_face_analyze/
├── src/                              # Library code
│   ├── config.py                     # 全專案路徑常數 + CohortSpec (V2.2 5-axis)
│   ├── common/                       # 跨模態 helpers (demographics, mediapipe, metrics)
│   ├── meta/                         # 跨模態 modeling layer (loader / classifier / stacking / evaluation)
│   ├── preprocess/                   # 對齊 / 偵測 / 鏡射 / 選圖
│   ├── age/                          # MiVOLO + bootstrap calibration
│   ├── asymmetry/                    # 468-landmark asymmetry
│   ├── embedding/                    # ArcFace / TopoFR / dlib / VGGFace
│   ├── emo_au/                       # FER + AU (OpenFace / LibreFace / Py-Feat / DAN / HSEmotion / ViT / POSTER_V2 / EmoNeXt / FER / EmoNet)
│   └── rotation/                     # head pose / vector angle
│
├── scripts/                          # Entry-point scripts (mirror src/ + workspace/)
│   ├── README.md                     # modality 索引
│   ├── utilities/                    # cohort / feature_loaders / stats_helpers / emotion_loader / facial_landmarks
│   ├── preprocess/                   # prepare_feature.py
│   ├── age/                          # predict_ages, run_classifiers, run_window_classifier, plot_*
│   ├── asymmetry/                    # extract_landmarks, run_analysis
│   ├── embedding/                    # (Phase 2: extract / run_fwd_rev / run_sweep / plot_*)
│   ├── emo_au/                       # extract_au, extract_fer, plot_emotion_comparison, plot_valence_arousal
│   ├── longitudinal/                 # build_dataset, build_hc_and_vectors
│   ├── rotation/                     # process_angle
│   ├── overview/                     # 跨模態 orchestrators (run_cohort_pipeline, run_cross_naive, run_cross_matched, run_stat_grid, plot_*)
│   ├── meta/                         # legacy M1-M4 entry (run_analyze, run_meta_analysis)
│   ├── external/                     # 公開亞裔資料集整合 (EACS)
│   └── literature_monitor/           # 文獻監控 sub-package
│
├── workspace/                        # All artifacts (gitignored)
│   ├── preprocess/                   # aligned / mirrors / selected
│   ├── age/                          # predictions + analysis
│   ├── asymmetry/                    # landmarks (.npy) + analysis
│   ├── embedding/                    # features + analysis (classification / fwd-rev sweeps)
│   ├── embedding_ABtest/             # background-on/off A/B
│   ├── emo_au/                       # features (per tool) + analysis
│   ├── longitudinal/                 # patient_deltas + vector_deltas
│   ├── rotation/                     # PnP / vector angle
│   └── overview/                     # 跨模態 cohort summaries + stat grids
│
├── envs/                             # Conda env spec snapshots + setup README
├── data/                             # demographics CSVs (P / NAD / ACS / EACS)
├── external/                         # 公開亞裔人臉資料集 (raw + filtered/EACS_*)
├── references/                       # literature_monitor 文獻 PDFs / 摘要
├── docs/                             # 額外設計文件
└── paper/                            # 論文草稿
```

## 開發慣例

**核心原則：路徑規劃改動時，下面五邊必須同步：**

- `src/<modality>/`
- `src/config.py`
- `scripts/<modality>/`
- `workspace/<modality>/`
- `memory/`（`.claude/projects/<proj>/memory/`）
