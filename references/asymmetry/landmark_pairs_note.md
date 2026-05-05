# Landmark-based 區域不對稱性 — 標點筆記

## 概述

用 MediaPipe Face Mesh 468 landmarks 的左右對應點，分 4 區域計算座標差異。
正規化：最左點→X=0, 最頂點→Y=0, 臉寬→500

## 各區域 Landmark Pairs (Right, Left)

### Eye — 16 pairs ✅ 已確認
| # | Right | Left |
|---|-------|------|
| 1 | 33 | 362 |
| 2 | 155 | 382 |
| 3 | 154 | 381 |
| 4 | 153 | 380 |
| 5 | 145 | 374 |
| 6 | 144 | 373 |
| 7 | 163 | 390 |
| 8 | 7 | 249 |
| 9 | 133 | 263 |
| 10 | 246 | 466 |
| 11 | 161 | 388 |
| 12 | 160 | 387 |
| 13 | 159 | 386 |
| 14 | 158 | 385 |
| 15 | 157 | 384 |
| 16 | 173 | 398 |

來源：MediaPipe 標準 eye contour

### Nose — 14 pairs ✅ 已確認
| # | Right | Left |
|---|-------|------|
| 1 | 122 | 351 |
| 2 | 174 | 399 |
| 3 | 198 | 420 |
| 4 | 129 | 358 |
| 5 | 196 | 419 |
| 6 | 3 | 248 |
| 7 | 236 | 456 |
| 8 | 51 | 281 |
| 9 | 134 | 363 |
| 10 | 131 | 360 |
| 11 | 45 | 275 |
| 12 | 220 | 440 |
| 13 | 115 | 344 |
| 14 | 48 | 278 |

### Mouth — 18 pairs ✅（暫定標準唇部輪廓）
| # | Right | Left | 區域 |
|---|-------|------|------|
| 1 | 61 | 291 | Outer upper lip |
| 2 | 185 | 409 | Outer upper lip |
| 3 | 40 | 270 | Outer upper lip |
| 4 | 39 | 269 | Outer upper lip |
| 5 | 37 | 267 | Outer upper lip |
| 6 | 146 | 375 | Outer lower lip |
| 7 | 91 | 321 | Outer lower lip |
| 8 | 181 | 405 | Outer lower lip |
| 9 | 84 | 314 | Outer lower lip |
| 10 | 78 | 308 | Inner upper lip |
| 11 | 191 | 415 | Inner upper lip |
| 12 | 80 | 310 | Inner upper lip |
| 13 | 81 | 311 | Inner upper lip |
| 14 | 82 | 312 | Inner upper lip |
| 15 | 95 | 324 | Inner lower lip |
| 16 | 88 | 318 | Inner lower lip |
| 17 | 178 | 402 | Inner lower lip |
| 18 | 87 | 317 | Inner lower lip |

⚠️ 原始分析有 36 pairs，目前先用標準 18 pairs 跑，之後可能需要補充。

### Face Oval — 17 pairs ✅ 已確認
| # | Right | Left |
|---|-------|------|
| 1 | 109 | 338 |
| 2 | 67 | 297 |
| 3 | 103 | 332 |
| 4 | 54 | 284 |
| 5 | 21 | 251 |
| 6 | 162 | 389 |
| 7 | 127 | 356 |
| 8 | 234 | 454 |
| 9 | 93 | 323 |
| 10 | 132 | 361 |
| 11 | 58 | 288 |
| 12 | 172 | 397 |
| 13 | 136 | 365 |
| 14 | 150 | 379 |
| 15 | 149 | 378 |
| 16 | 148 | 377 |
| 17 | 176 | 400 |

修正紀錄：底部兩組原本是 (148,400) (176,377)，已交換為 (148,377) (176,400)。

## 面積計算

用 Shoelace 公式，以各區域的邊緣 landmarks 組成多邊形，計算左右面積差。

## 特徵總數

| 區域 | Pairs | ×2 (x,y) | Area |
|------|-------|----------|------|
| Eye | 16 | 32 | 1 |
| Nose | 14 | 28 | 1 |
| Mouth | 18 | 36 | 1 |
| Face oval | 17 | 34 | 1 |
| **Total** | **65** | **130** | **4** |

**目前總計：134 維**（mouth 可能需從 18 調回 36）

## 檔案位置

- 模組：`src/extractor/features/asymmetry/regional_landmark.py`
- Landmark .npy：`workspace/asymmetry/landmarks/{subject_id}.npy`（shape: n_images × 468 × 2）
- 特徵 CSV：`workspace/asymmetry/landmark_features.csv`
- 驗證圖：`workspace/asymmetry/landmark_pairs_visualization.png`
- 鼻部候選圖：`workspace/asymmetry/nose_bridge_filtered.png`

## Mouth Pairs 完整調查（2026-04-10）

### 來源
`workspace/asymmetry/facial_landmarks.py` 提供完整 220 組 MediaPipe 左右對應 pairs。
其中 mouth 相關共 43 組，分三層：

| 層級 | Pairs 數 | 說明 |
|------|---------|------|
| 標準唇部輪廓 | 18 | outer/inner upper+lower lip（原始設定） |
| 嘴內補充 | 15 | 口角(38,41,42)、內唇深層(62,72-77,83,85-86,89-90,96) |
| 嘴周肌肉 | 10 | 頰部/鼻唇溝(43,57,92,106,165,167,182-184,186)→**已排除** |

### Refined 33 pairs（最終使用）
原始 18 + 嘴內 15，移除嘴周肌肉 10 組。

新增 15 pairs：
(38,268), (41,271), (42,272), (62,292), (72,302), (73,303), (74,304),
(76,306), (77,307), (83,313), (85,315), (86,316), (89,319), (90,320), (96,325)

### Eye Pairs 修正紀錄
原始 (33,362) 和 (133,263) 在 facial_landmarks.py 中對應為 (33,263) 和 (133,362)。
兩組左右交叉，但僅影響 dx 符號，對 L2 norm 無影響。

---

## 分析結果（2026-04-10）

### 橫斷面嚴重度分類（CDR 0.5 vs 1 vs 2+，P 群體，age≥65）

XGBoost 3-class, GroupKFold(5), n=920 patients

| 特徵集 | BalAcc (first) | BalAcc (latest) |
|--------|---------------|-----------------|
| 全部 65 pairs (134 feat) | 0.355 | 0.351 |
| 全部 220 pairs (440 feat) | 0.349 | 0.365 |
| Eye 16 pairs | 0.369 | 0.313 |
| Nose 14 pairs | 0.346 | 0.335 |
| Mouth 18 pairs | 0.357 | 0.374 |
| Mouth 33 pairs (refined) | 0.365 | 0.352 |
| Mouth 43 pairs (full) | 0.349 | 0.352 |
| Face oval 17 pairs | 0.331 | 0.347 |
| 4 area_diff only | 0.315 | 0.321 |

**結論：全部接近隨機水準 (0.33)，無實質區辨力。與 embedding/emotion/age 結果一致。**

### 縱向分析（首末 visit 變化量 vs 認知退化，n=538）

| 指標 | vs ΔCDR-SB (Spearman r) | p-value |
|------|------------------------|---------|
| **emb_cosine_dist** | **+0.247** | **<0.0001** |
| mouth_combined_l2 (coord+area) | +0.106 | 0.014 * |
| eye_coord_only_l2 | +0.107 | 0.014 * |
| delta_mouth_area | +0.101 | 0.019 * |
| delta_eye_area | +0.090 | 0.038 * |
| mouth_coord_l2 (coord only) | +0.054 | 0.214 |
| mouth_33_l2 (refined) | +0.053 | 0.216 |
| 全臉 220 pairs L2 | +0.076 | 0.078 |
| 全臉 65 pairs L2 | +0.043 | 0.325 |

**信號拆解：**
- 嘴部：信號主要來自 area_diff (r=0.101*)，座標差本身不顯著 (r=0.054)
- 眼部：信號來自座標差 (r=0.107*)，area 較弱 (r=0.090*)
- 嘴唇不管 18/33/43 pairs 結果一致，增加 pairs 無幫助

### 論文角色

兩條路線共同驗證：

1. **古典特徵工程（landmark 不對稱性）**+ **深度學習（face embedding）**→ 橫斷面分類在控制年齡後全部失效 (BalAcc ≈ 0.33)
2. 縱向追蹤：embedding cosine distance (r=0.247) 遠優於 landmark 不對稱性 (r≈0.05-0.10)
3. 認知退化造成的面部變化是 **holistic（整體性）** 的，非局限於左右不對稱性
4. Landmark 不對稱性適合作為 supplementary negative finding 報告

---

## 檔案位置

- 模組：`src/extractor/features/asymmetry/regional_landmark.py`
- 完整 pair 來源：`workspace/asymmetry/facial_landmarks.py`（220 pairs）
- Landmark .npy：`workspace/asymmetry/landmarks/{subject_id}.npy`（shape: n_images × 468 × 2）
- 特徵 CSV：`workspace/asymmetry/landmark_features.csv`（65 pairs, 134 feat）
- 分析結果：`workspace/asymmetry/analysis/`
  - `landmark_pairs_verification.png` — 65 pairs 全臉視覺化
  - `mouth_pairs_refined.png` — Mouth 18 vs 33 vs 43 pairs 比較
  - `cross_sectional_summary.csv` — 橫斷面分類結果
  - `longitudinal_correlations.csv` — 縱向相關性結果
  - `longitudinal_landmark_deltas.csv` — 每位患者 landmark 變化量

## 驗證紀錄

- 視覺化圖：workspace/asymmetry/analysis/ 下各 .png 檔
- 舊結果比對：`C:\Users\4080\Desktop\_temp_save\Alz\對外文件\彰濱\2025結案報告\all_features_diff_1_one_side_tail.csv`
