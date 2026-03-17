# Scripts

## pipeline/ — 主 Pipeline（依序執行）

1. `predict_ages.py` — Stage 0：MiVOLO 年齡預測
2. `prepare_feature.py` — Stage 1：前處理 + 多模型特徵萃取
3. `run_analyze.py` — Stage 2：分類器訓練（XGBoost / Logistic / TabPFN）
4. `run_meta_analysis.py` — Stage 3：14 特徵 Meta-learning

## experiments/ — 探索性 / 替代分析

- `compute_emotion_scores.py` — EmoNet 8-class 情緒分數 + Valence/Arousal
- `run_xgboost_modules.py` — M3/M4 模組各自 XGBoost 訓練
- `run_full_features_classifier.py` — 1036-d 直接分類（M1+M2+M3+M4）
- `run_tabpfn_m1m2_512.py` — M1/M2 512-d TabPFN（無 RFE）
- `run_m3m4_deep_analysis.py` — M3+M4 深度分析 & 出版品質圖表

## visualization/ — 繪圖 & 統計

- `plot_predicted_ages.py` — 年齡預測分佈與 scatter plot
- `plot_valence_arousal.py` — Valence-Arousal 散佈圖
- `demographics_statistics.py` — 人口學統計摘要
- `plot_tabpfn_meta_by_n_features.py` — TabPFN Meta 趨勢圖
- `plot_xgboost_meta_by_n_features.py` — XGBoost Meta 趨勢圖

## utilities/ — 一次性處理工具

- `extract_original_features.py` — 原始嵌入提取（不做鏡射）
- `calibrate_age_prediction.py` — 年齡預測誤差校正
- `compare_features.py` — 五位提取者特徵比對視覺化
