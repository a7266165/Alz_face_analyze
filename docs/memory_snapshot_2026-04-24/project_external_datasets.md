---
name: External public face datasets for ACS expansion
description: Sibling folder Alz_face_open_pics/ 蒐集的公開人臉資料集（擴充 ACS/aged HC 候選），現況與未整合狀態。
type: project
originSessionId: 1d472397-f7a8-48e2-8471-10237b08fe97
---
## 位置（2026-04-23 併入主 repo）
`Alz_face_analyze/external/public_face_datasets/`（原 sibling `C:\Users\4080\Desktop\Alz_face_open_pics\` 已搬空，只剩 `.claude/`）

## 動機
補強主 cohort 裡 ACS (Asian Control Subjects) 只有 91 人的問題，尤其 Arm A age gap +21.7y、Arm D × ACS N/A (21 pairs < MIN_CELL_N=25) 都受限於 ACS 人數。收 2026-04-07。

**Why:** ACS 是主專案內部招募的受試者，數量固定且招募成本高；公開資料集可做外部 validation / pretrain / 擴充健康老年影像池。
**How to apply:** 未來若要談 cross-dataset validation、external HC、或 pretrain 相關 embedding，直接從這裡挑資料；主專案 pipeline 尚未整合，要用需自己接 loader。

## 資料夾內容
- `external/public_face_datasets/datasets/` — 13 個已下載公開資料集（gitignored，10+ GB）
- `external/public_face_datasets/filtered/asian_elderly_60plus/` — 3,842 single-image subjects per-folder
- `external/public_face_datasets/filtered/IMDB_60_plus/` — 4,091 subject-visits（1,065 IMDB identities，786 multi-visit ≥2 photo-years）
- `external/public_face_datasets/filtered/manifest.csv` — 7,933 rows metadata
- `external/public_face_datasets/{README.md,DOWNLOAD_STATUS.md,asian_elderly_datasets.csv}` — tracked metadata
- `data/demographics/EACS.csv` — 7,933 rows，schema 同 ACS.csv + `Source` 欄（AFAD/IMDB/...），MMSE/CASI 留空

## Scripts（`scripts/external/`）
- `extract_asian_elderly.py` / `asian_age_stats.py` — 從 raw dataset 過濾亞裔 60+（從 sibling 搬入 + 改路徑從 `src.config` 讀）
- `build_subject_folders.py` — flat pool → per-subject folder + manifest。IMDB 按 (nm_id, photo_year) 分組成 multi-visit；其他每張 = 1 subject
- `build_external_demographics.py` — manifest → `data/demographics/EACS.csv`

## 需申請未拿到
ADReFV、PROMPT、東京大學 AD、YT-DemTalk、DementiaBank、ElderReact、Tsinghua-FED、AgeDB、MORPH II、FG-NET、APPA-REAL、K-FACE、Park Aging Mind 等（Dementia/Clinical 類最重要，但都要申請，本次不做）

**類似 UTKFace 品質的後續推薦**（若要再擴）：
1. **AgeDB** — 16K、年齡 1-101、MiVOLO LAGENDA ckpt 無洩漏、in-the-wild 但 curated（最值得申請）
2. **APPA-REAL** — 7.5K、同時有 real + apparent age、ChaLearn LAP 申請
3. **Tsinghua-FED** — 亞裔老年特化、110 人 × 8 表情、email 申請

## MiVOLO v2 確認無洩漏
HuggingFace `iitolstykh/mivolo_v2` = LAGENDA-trained `mivolov2_d1_384x384` checkpoint（Open Images 標註子集）。對 EACS 全部 7 個 source（AFAD/IMDB/FairFace/UTKFace/MegaAge-Asian/DiverseAsian/SZU-EmoDage）都乾淨。**FairFace 雖在 test set 評估過，但 weight 不是 FairFace 訓練的**。

## 整合狀態（2026-04-24 Phase D 進度）

**已完成 age + emotion 抽取**（prepare_feature 被中止，**embedding / landmark 沒跑**）：
- MiVOLO v2 (LAGENDA ckpt) age prediction：7,933 EACS 全 merged → `predicted_ages.json` 共 12,250 ids
- 8 emotion extractors × 2 pools 全跑完（FER 7016 / 其他 7,933）
- harmonize + aggregate 在 `workspace/emotion/au_features/aggregated/*.csv`
- build_longitudinal_hc_and_vectors.py `_visits_with_features` 放寬：emotion CSV 也算 feature proxy → EACS IMDB 786 multi-visit subjects 進 `eacs_patient_deltas.csv`
- EACS age_error CSV: mean **+7.42**（MiVOLO 低估高齡亞裔）sd 11.56

**UTKFace 清理（2026-04-24）**：
- 原 pred<42 outliers 11 個，`scripts/external/retry_utkface_predictions.py` 用 LibreFace 自動產的 `{name}_aligned.png`（Windows path bug 導致寫進 subject folder 的意外副產品）重跑
- `QUALITY_BLACKLIST` 剔 3 個 raw-only 無 aligned 的 (iStock 浮水印 / 尼泊爾 / 蔥堆奶奶)
- `RACE_BLACKLIST` 剔 2 個誤標亞裔的 (非洲男 / 白人老伯)
- **最終**：UTKFace n=279, r=0.893, MAE=4.64 — 7 個 EACS source 中第二好（僅次 SZU-EmoDage）
- `predicted_ages.json` 從 12,250 → **12,245** ids

**4-arm 新增 CLI flags**（commit 1412861）：
- `--eacs-sources UTKFace`：filter EACS 子集
- `--arms A B`：只跑指定 arms，其他填 n/a
- `--modalities age_only age_error`：只算指定 modality
- 範例 output：`workspace/age_ladder/deep_dive_acs_ext_utkface_AB_age_only_age_error/`
  - Arm A × ACS (extended): n_pos=1045 n_neg=366，age_error d=−0.37 ***
  - Arm B × ACS (age-matched, extended): **n=246 pairs**（原 33），age_error d=−0.40 *

**age-window classifier 新增**（commit f77588c）：
- `--include-eacs-sources UTKFace`：HC 擴充
- `--threshold-mode {fixed, per_window, both}`：per-window threshold 在 test predictions 找最佳（有 optimistic bias，oracle upper-bound reference）
- Output path 改：`{clf}/{feat}feat[_eacs_{sources}]/{visit}/{threshold_mode}/`
- AD_vs_ACS view 把 external 歸類為 E-ACS（紫色 stacked bar）
- AD_vs_NAD view 保持純 internal NAD
- UTKFace 加進 training 對 AD_vs_NAD **小幅拖累** BalAcc（UTKFace noise 讓 HC 分佈變糊）

**Scatter viz**（commit 0d692a4）：
- `scripts/visualization/plot_acs_eacs_predicted_ages.py` — 2×4 per-source grid scatter
- `scripts/visualization/plot_predicted_ages_with_eacs.py` — 2-panel HC + UTKFace (red atop NAD blue + ACS green) | P

**尚未做**：EACS embedding / landmark 特徵（原 prepare_feature 被中止，所以 Arm A/B × ACS 只能跑 age 類；embedding modality cell 會 N/A）。如要跑回，用 `prepare_feature.py --input-root external/public_face_datasets/filtered --input-groups asian_elderly_60plus IMDB_60_plus` 在 `Alz_face_test_2` env，估 1-3 hr。
