# Alz_face_analyze

阿茲海默症臉部多模態分析系統 — 從原始臉部影像抽取 age / emotion / AU / asymmetry / rotation / embedding 特徵，配合 cohort matching + 統計檢定 + classifier sweep 評估與 AD 相關性。

## 專案結構（modality-flat，三軸對齊）

`src/`、`scripts/`、`workspace/` 三個 tree 都按相同的 modality 鍵組織：

```
Alz_face_analyze/
├── src/                              # Library code
│   ├── config.py                     # 全專案路徑常數 + COHORT_DIRS
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

## 主要 entry point

跨模態 cohort 分析（一鍵跑 5 步：cross_naive、3 種 cross_matched、2 種 hi-lo、stat_grid、age classifiers）：
```bash
conda run -n Alz_face_main_analysis python scripts/overview/run_cohort_pipeline.py \
    --cohort-mode {default | p_first_hc_all | p_all_hc_all}
```

Embedding fwd/rev sweep（PCA grid × feature_type grid）：
```bash
conda run -n Alz_face_main_analysis python scripts/embedding/run_sweep.py \
    --cohort-mode p_first_hc_all
```

每個 modality folder 內的 entry script 也可獨立跑（含 `--help`）。

## Cohort modes

`src/config.py` 內 `cohort_name()` 由 cohort_mode 決定：

| `--cohort-mode` | AD 視作 | HC 視作 | workspace dir |
|---|---|---|---|
| `default` | 首次 visit + strict-HC filter | 首次 visit + strict | `p_first_hc_first/` |
| `p_first_hc_all` | 首次 visit | 全部 visit（HC 不 filter） | `p_first_hc_all/` |
| `p_all_hc_all` | 全部 visit | 全部 visit | `p_all_hc_all/` |

`--hc-source-mode {ACS | ACS_ext | EACS}` 控制 ACS group 來源（內部 / 內部+外部 / 純外部）。

## Conda envs

詳見 [`envs/README.md`](envs/README.md)。一句話總結：

- **Consumer**：`Alz_face_main_analysis`（torch 2.7.1 + tabpfn / xgboost / sklearn / matplotlib，所有 cohort / classifier / sweep / plot 用這個）
- **Producer**：`Alz_face_age` / `Alz_face_asymmetry` / `Alz_face_embedding_{other,vggface}` / `Alz_face_rotation` / `Alz_face_emo_au_{openface,libreface,pyfeat,other}`（依工具版本衝突拆 9 個）
- **Deployment**：`Alz_face_api` / `Alz_face_ui` / `graphviz` / `Alz_face_test_2`（不主動跑 pipeline）

依賴沒有單一 lockfile — 不同 producer envs 各有 cu118 / cu121 + 特殊套件版本衝突，無法用 poetry / uv / pip-tools 統一 lock。各 env 用 `envs/<name>.txt` pip-freeze 快照保留。

## 開發慣例

- `src/` 不主動改（以 [feedback_src_off_limits] 為主）；共用 helper 落在 `scripts/utilities/`
- workspace 結構是 source of truth — 改了要同步更新 `src/config.py` 的 `COHORT_DIRS` 跟相關 README
- 新套件先建 `tmp_env` 試裝，確認穩定再進主 env
- Windows cp950 環境：`conda run` 偶爾吞中文輸出；含中文輸出的腳本改用絕對路徑直跑：
  ```
  "C:/Users/4080/anaconda3/envs/<env>/python.exe" scripts/...
  ```
