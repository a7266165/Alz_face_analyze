---
name: Project details and discoveries
description: 目前狀態、核心發現。
type: project
originSessionId: 91922059-0ce1-41f4-9401-eea335107aee
---
## 現況（2026-04-23，3.5 節 4-arm deep-dive 完成）

**原 3-arm 擴成 4-arm 統計網格（`paper/main_v6_1_3arm.md` 3.5 節）**：
- **Arm A** — Cross-sec naive（無年齡配對）
- **Arm B** — Cross-sec 1:1 age NN matched
- **Arm C** — Longitudinal naive（所有 multi-visit，無年齡配對；新增）
- **Arm D** — Longitudinal 1:1 baseline-age NN matched（原 "Arm C"）

**4 族群比較正交於 4 arm**：`AD vs HC`、`AD vs NAD`、`AD vs ACS`、`AD high-MMSE vs AD low-MMSE`（hi-lo）
→ 16 cells × 14 modality 資料列 = 224 cells
→ **全部 224 cells active**（D:ACS 原 n/a，修復 build_longitudinal feature-filter bug 後 AD pool 從 568→714、ACS 從 57→58，配到 20 pairs 剛好過 MIN_CELL_N=20 gate；D:ACS 全部 ns、q>0.9，屬 exploratory）

## Cohort（完整盤點）

- P 患者 1,095 人 × 3,572 visits；multi-visit with features 714 人（修 feature-filter bug 前只有 568）
- HC 候選：NAD 461 + ACS 91；strict HC = 至少一項認知評估 + (CDR=0 或 MMSE≥26)
- Arm A (naive cross-sec) × HC: 1,045 vs 278（age gap +12.8y）
- Arm A × ACS: 1,045 vs 82（age gap +21.7y，confound 最極端）
- Arm B (age-matched cross-sec): HC 196×2、NAD 183×2、ACS 33×2、hi-lo 346×2
- Arm C (longit naive): 714 multi-visit AD vs 150 HC / 93 NAD / 58 ACS; hi-lo 349/342
- Arm D (longit matched): HC 100×2、NAD 89×2、**ACS 20×2**、hi-lo 269×2 pairs
- 資料缺檔：P44/P100/P1080/P94 整批 visit 遺失（不可救）；386 個 demo rows 無 photo folder（主要 visit-1 entries：P62-1, P75-1, P142-1 等）— `build_longitudinal_hc_and_vectors.py` 已修為先按 ArcFace .npy 存在過濾 demo 再挑 first/last（之前用 `g.iloc[0]` 挑到沒 feature 的 visit 會讓整個 subject 被 drop）

## 統計方法（每 modality 類別）

| Modality 類型 | Test | Effect size |
|---|---|---|
| Scalar (age_only, age_error, 3× asymmetry L2) | Welch t + Mann-Whitney | Cohen's d |
| Low-dim (landmark 4-d per-region L2) | Hotelling T² + permutation p | Mahalanobis D² |
| High-dim embedding mean（arcface 512/dlib 128/topofr 512）| PERMANOVA（**arcface: cosine** / dlib/topofr: Euclidean） | R² |
| High-dim embedding asymmetry full vector（3 model） | PERMANOVA 同上 | R² |
| High-dim landmark 130-d raw xy（去掉 4 area_diff，scale 30× 會淹沒） | PERMANOVA Euclidean | R² |
| emotion_8methods 56-d（8 method × 7 emo mean；去掉 std/range/entropy） | per-method Hotelling T² + Fisher 合併 p | mean T̄² |

每 cell 內 14 modality 跑 BH-FDR；permutation = 1000。

## 核心發現

### Arm A（naive cross-sec）
- age_only AUC 0.86（HC）/ 0.98（ACS）/ 0.81（NAD）— 年齡代理極強
- 所有 embedding mean AUC 在 ACS 衝到 0.9+，經 B/D matching 跌回 0.55-0.65

### Arm B（age-matched cross-sec）
- `age_error` d≈−0.35 (**), AUC 0.59-0.61
- `emotion_8methods` 56-d per-method Fisher：T̄²=13, ***/***（AUC 0.75-0.79）
- `embedding_arcface_mean` 512-d PERMANOVA cosine R²=0.005 (***), AUC 0.62-0.63
- `embedding_*_asymmetry [full vector]` 均 (**) — L2 scalar 只顯 (*)；信息在方向不在總量
- `landmark_asymmetry [130-d raw xy]` 去掉 area_diff 後 R²=0.009 (*) 浮現

### Arm C（longit naive，新訊號）
- `embedding_arcface_mean` full-vector Δ cosine：R²=0.002, q=0.007 / 0.005 (**)（C × HC/NAD）
- 其他多數 null；延續 rate-level invariance

### Arm D（longit matched，本次 TODO (3) 直接收益）
- **`embedding_arcface_mean` full-vector Δ：D:HC R²=0.008 (*)、D:NAD R²=0.008 (*)**（舊 1-d 年化 cosine drift 為 ns）
- 解讀：AD embedding 縱向**漂移方向**攜帶疾病訊號；scalar 壓縮後丟失
- D × ACS：n=20 pairs 全 ns（power 極弱，exploratory）
- D × hi-lo 多 modality ns（MMSE severity 不調制年化 drift rate；hi-lo 從 208→269 pairs 後仍不顯著）

### Pool-wide longitudinal Spearman（supplementary，沿用 v6 L4）
- `emb_cosine_dist` vs ΔCDR-SB: r=+0.276, p<0.0001, n=449
- "concurrent change coupling"（面部與認知同期變化耦合），非 severity→drift predictor

## 關鍵 pipeline 檔案

### Scripts（全在 `scripts/experiments/` 和 `scripts/visualization/`）
- `run_4arm_deep_dive.py` — 4-arm × 16-cell 統計網格 orchestrator；讀 `DEEP_DIVE_N_PERMS` 與 `DEEP_DIVE_VARIANT`（未設→`deep_dive/`，`=v2`→`deep_dive_v2/`）env；`plot_deep_dive_grid.py` 同樣支援 `DEEP_DIVE_VARIANT`
- `build_longitudinal_hc_and_vectors.py` — 產 `ad_patient_deltas.csv` / `hc_patient_deltas.csv` / `vector_deltas.npz`（含 56 emotion Δ、first/last MMSE/CASI/Global_CDR/CDR_SB）
- `plot_deep_dive_grid.py` — 渲染 5-row header HTML + PNG：Row 1 arm / Row 2 比較全寫（hi-lo 的 vs 獨立一行）/ Row 3 per-group `AD n = X visits / Y subject` / Row 4 Age 三行（label / values / p）/ Row 5 hi-lo MMSE+CASI+空行+CDR

### Workspace outputs（`workspace/age_ladder/deep_dive/`）
- `stat_grid_long.csv` — 224 rows，欄含 modality_parent/sub、arm、comparison、test、p、q、effect、effect_type、auc_auc、auc_auc_ci_{low,high}
- `stat_grid_wide.csv` — 10×16 pivot
- `stat_grid.png` / `stat_grid_markdown.md` — 渲染結果
- `feasibility_report.csv` — per-cell n_pos/n_neg/status
- `cell_header_stats.csv` — per-cell n_all_1/0、n_unique_1/0、age/mmse/casi/cdr mean_1/sd_1/mean_0/sd_0/p

### 縱向 Δ 資料
- `workspace/longitudinal/ad_patient_deltas.csv`（748 AD，schema：first/last MMSE/CASI/Global_CDR/CDR_SB + 3-model cosine drift + 3-model embasym Δ + 4 landmark region L2 Δ + 56 emotion Δ，全加 ann_ 年化）
- `workspace/longitudinal/hc_patient_deltas.csv`（234 HC+NAD+ACS，同 schema）
- `workspace/longitudinal/vector_deltas.npz`（785 subjects × 7 vector keys：`emb_drift_vec_{arcface,topofr,dlib}` + `emb_asym_delta_vec_{3 model}` + `lmk_raw_xy_delta`）

## Deep-dive TODO 清單（均已完成於 2026-04-22~23）
- [x] Emotion A/B 縮 224→56-d（去 std/range/entropy）
- [x] Emotion C/D 擴 7→56-d（8 method × 7 emo Δ）
- [x] Embedding mean C/D 改 full 512/128-d vector Δ（取代 1-d cosine drift）
- [x] Dispatch bug fix（C/D 分支讀 test_kind 決定 cosine / Euclidean）
- [x] Header 擴成 5 行（arm / 比較 / n per-group / Age / hi-lo cog）
- [x] CDR 選擇：使用 Global_CDR（5 級離散）配合臨床分期；CDR-SB 已存 CSV 但不渲染

## Cleanup 完成（2026-04-22）
- Phase 1：legacy v5/v6-pre-3arm 腳本（12 個）+ v6 layer/meta_learner 腳本 + 對應 workspace/meta_learner/ 及 embedding/ 舊 analysis snapshot → 全部移至資源回收桶
- Phase 2：舊 `workspace/age_ladder/master_*` + `plot_3arm_master.py` → 資源回收桶
- 舊 paper drafts：`paper/main_v5_chinese.md`、`paper/main_v6_progressive.md` → 資源回收桶
- 僅留 `paper/main_v6_1_3arm.md` 為唯一活躍 paper 草稿
