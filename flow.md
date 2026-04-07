# 年齡預測誤差校正流程

## 目的
MiVOLO 年齡預測模型存在系統性偏差（整體低估年齡，MAE ≈ 6.19 歲）。
本流程使用健康對照組（NAD）建立線性誤差模型，透過 bootstrap 穩定估計後校正全體預測值。

## 資料來源
- `data/demographics/{ACS,NAD,P}.csv` — 受試者基本資料（含真實年齡）
- `workspace/age_prediction/predicted_ages.json` — MiVOLO 預測年齡（ID → predicted_age）

## 流程

### Step 1：載入資料
讀入三組 demographics + predicted_ages.json，計算：
- `error = real_age - predicted_age`（正值 = 低估）
- `age_int = floor(real_age)`

### Step 2：篩選建模資料
從 NAD 中篩出 **real_age ≥ 60** 的子集：
- 763 visits, 437 subjects, 涵蓋 60~92 歲（33 個整數歲）
- 選擇 NAD 而非 ACS 的原因：NAD 年齡分布更接近 Patient 組，校正模型的適用範圍更廣
- 設定 60 歲下限：排除年輕 NAD（其誤差模式可能不同），聚焦於失智症好發年齡段

### Step 3：Bootstrap 建模（1000 次迭代）
每次迭代：
1. 對 NAD 60+ 的每個**整數歲**，隨機抽 **1 位 subject 的 1 筆紀錄** → 33 筆樣本
2. 擬合線性模型：**e = a·y + b**
   - e = 誤差（real_age - predicted_age）
   - y = 真實年齡（real_age）
   - a, b = 線性迴歸係數
3. 用該次 (a, b) 對所有受試者計算校正值：**corrected = yp + (a·y + b)**
4. 記錄校正結果，但 **NAD 中該次被抽到的 subject 不記錄**（避免訓練汙染）
5. ACS 和 P 每次都記錄（從未參與訓練）

每歲只抽 1 人的目的：消除年齡分布不均的影響，使迴歸不被高頻年齡段主導。

### Step 4：平均校正結果
- 每位受試者取所有**有效迭代**的 corrected_predicted_age 平均值
  - ACS / P：1000 次全部有效
  - NAD：約 967 次有效（剔除被抽到的約 33 次）

### Step 5：計算殘差
```
ε = real_age - corrected_predicted_age
```

## 最終係數（1000 次平均）
```
a = -0.0481 ± 0.0918
b =  9.1070 ± 6.9506
```

意義：誤差隨真實年齡略微減小（a < 0），但 intercept 約 9.1 歲代表整體性的低估偏差。

## 校正效果

| 族群 | n | MAE 前→後 | Mean Error 前→後 |
|------|---|----------|-----------------|
| ACS | 223 | 6.86 → 3.69 | 6.14 ± 4.98 → -0.12 ± 4.96 |
| NAD | 794 | 6.70 → 4.51 | 5.73 ± 5.65 →  0.15 ± 5.68 |
| P | 3266 | 6.02 → 5.29 | 3.81 ± 6.51 → -1.43 ± 6.54 |
| All | 4283 | 6.19 → 5.06 | 4.29 ± 6.34 → -1.06 ± 6.34 |

- ACS/NAD 的 Mean Error 接近 0，表示系統性偏差被有效消除
- P 組仍殘留 -1.43 歲偏差，可能反映疾病對面容的影響（患者看起來更老）
- 標準差基本不變，說明校正只移除了系統性偏差，未影響隨機誤差

## 輸出檔案
路徑：`workspace/age_prediction/bootstrap_correction/`

| 檔案 | 內容 |
|------|------|
| `corrected_ages_bootstrap.csv` | 每人校正前後結果 + 有效迭代數 |
| `bootstrap_coefficients.csv` | 1000 次 (a, b) + mean/std |
| `nad_60plus_age_distribution.png` | 建模資料年齡分佈 |
| `error_distribution_before_after.png` | 校正前後誤差直方圖 (ACS/NAD/P) |
| `error_by_age_group.png` | 分年齡層箱型圖 |
| `scatter_before_after.png` | real vs predicted 散佈圖 |
| `bootstrap_coefficients.png` | 係數穩定性 (1000 次) |
| `residual_by_age_{acs,nad,p,all}.png` | 各族群殘差折線圖 (mean ± std) |
| `residual_by_age_combined.png` | 三族群殘差疊合圖 |

## 腳本
`scripts/utilities/age_error_bootstrap_correction.py`
