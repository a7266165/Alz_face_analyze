# References

本專案使用的模型及相關論文

---

## 年齡預測

### MiVOLO
- **用途**: 臉部年齡預測
- **模組**: `src/extractor/features/age/predictor.py`
- **論文**: MiVOLO: Multi-input Transformer for Age and Gender Estimation
- **作者**: Maksim Kuprashevich, Irina Tolstykh
- **年份**: 2023
- **連結**: https://arxiv.org/abs/2307.04616
- **GitHub**: https://github.com/WildChlamydia/MiVOLO

---

## 臉部偵測與特徵點

### MediaPipe Face Mesh
- **用途**: 臉部 468 特徵點偵測、中軸線計算
- **模組**: `src/extractor/preprocess/detector.py`
- **論文**: MediaPipe: A Framework for Building Perception Pipelines
- **作者**: Camillo Lugaresi et al. (Google)
- **年份**: 2019
- **連結**: https://arxiv.org/abs/1906.08172
- **文件**: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

---

## 臉部特徵提取

### TopoFR
- **用途**: 臉部拓撲特徵提取
- **模組**: `src/extractor/features/embedding/topofr_extractor.py`
- **論文**: TopoFR: A Closer Look at Topology Alignment on Face Recognition
- **作者**: Jun Dan et al.
- **年份**: 2024 (NeurIPS)
- **連結**: https://arxiv.org/abs/2411.02569
- **GitHub**: https://github.com/modelscope/facechain/tree/main/face_module/TopoFR

### dlib
- **用途**: 臉部特徵提取 (128D embedding)
- **模組**: `src/extractor/features/embedding/dlib_extractor.py`
- **論文**: Dlib-ml: A Machine Learning Toolkit
- **作者**: Davis E. King
- **年份**: 2009
- **連結**: http://jmlr.org/papers/v10/king09a.html
- **GitHub**: https://github.com/davisking/dlib
- **Face Recognition Model**: 基於 ResNet，訓練於 3M 人臉資料集

### ArcFace
- **用途**: 臉部辨識 Loss Function（廣泛用於人臉特徵提取模型訓練）
- **論文**: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
- **作者**: Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou
- **年份**: 2019 (CVPR)
- **連結**: https://arxiv.org/abs/1801.07698
- **GitHub**: https://github.com/deepinsight/insightface

### VGGFace / VGGFace2
- **用途**: 臉部辨識基準模型與資料集
- **論文**:
  1. **VGGFace** (2015): "Deep Face Recognition"
     - 作者: Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman
     - 連結: https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/
  2. **VGGFace2** (2018): "VGGFace2: A dataset for recognising faces across pose and age"
     - 作者: Qiong Cao, Li Shen, Weidi Xie, Omkar M. Parkhi, Andrew Zisserman
     - 連結: https://arxiv.org/abs/1710.08092
- **GitHub**: https://github.com/ox-vgg/vgg_face2

---

## 表情辨識

### EmoNet
- **用途**: 臉部表情辨識（8 類離散情緒 + Valence/Arousal 連續值）
- **模組**: `scripts/compute_emotion_scores.py`
- **論文**: Estimation of continuous valence and arousal levels from faces in naturalistic conditions
- **作者**: Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos, Maja Pantic
- **年份**: 2021 (Nature Machine Intelligence)
- **連結**: https://doi.org/10.1038/s42256-020-00280-0
- **GitHub**: https://github.com/face-analysis/emonet

### OpenFace 3.0
- **用途**: AU 強度提取、視線方向、頭部姿態

- **論文**: OpenFace 3.0: A Lightweight Multitask System for Comprehensive Facial Behavior Analysis
- **年份**: 2025
- **連結**: references/emotion/2025-OpenFace 3.0_A Lightweight Multitask System for Comprehensive Facial Behavior Analysis.pdf

### Py-Feat
- **用途**: AU 機率提取、情緒分類

- **論文**: Py-Feat: Python Facial Expression Analysis Toolbox
- **年份**: 2023
- **連結**: references/emotion/2023-Py-Feat_Python Facial Expression Analysis Toolbox.pdf
- **GitHub**: https://github.com/cosanlab/py-feat

### LibreFace
- **用途**: AU 強度提取、情緒分類

- **論文**: LibreFace: An Open-Source Toolkit for Deep Facial Expression Analysis
- **年份**: 2024
- **連結**: references/emotion/2024-LibreFace-An Open-Source Toolkit for Deep Facial Expression Analysis.pdf
- **GitHub**: https://github.com/ihp-lab/LibreFace

---

## 分類器

### TabPFN
- **用途**: 表格資料 few-shot 分類（Meta 分析階段）
- **模組**: `src/meta_analysis/classifier/tabpfn.py`, `src/meta_analysis/stacking/trainer.py`
- **論文**: TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second
- **作者**: Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter
- **年份**: 2025
- **連結**: references/analysis/Grinsztajn_2025_TabPFN2.5.pdf

---

## 論文檔案

請將 PDF 檔案放置於此資料夾，建議命名格式：
```
{作者}_{年份}_{模型名稱}.pdf
```

例如：
- `Kuprashevich_2023_MiVOLO.pdf`
- `Dan_2024_TopoFR.pdf`

---

## waiting_review/ — 自動文獻監測二篩區

`waiting_review/` 由 `scripts/literature_monitor/` 自動填入候選論文（每天 10
個 slot，覆蓋 embedding / asymmetry / emotion / age 四面向）。落地的 PDF +
JSON metadata 需經人工二篩決定升格或丟棄。

### 結構

```
waiting_review/
├── _state.json                       # 已見 paper ID（去重狀態）
├── _logs/<YYYYMMDD>.log              # 每日 log
├── _digests/<YYYYMMDD>.md            # 每 slot 增量 digest
├── _digests/<YYYYMMDD>_summary.md    # 當日 summary（slot 9 產出）
├── embedding/<YYYYMMDD>/*.pdf, *.json
├── asymmetry/<YYYYMMDD>/*.pdf, *.json
├── emotion/<YYYYMMDD>/*.pdf, *.json
└── age/<YYYYMMDD>/*.pdf, *.json
```

### 二篩流程

1. 每日 review `_digests/<TODAY>_summary.md` 與當日 PDF
2. **保留**：把 PDF + JSON 從 `waiting_review/<topic>/<DATE>/` 搬到 `references/<topic>/`，
   並在本 README 主目錄補一筆條目
3. **丟棄**：搬到 `references/_archive_rejected/<DATE>/`（保留紀錄，避免下輪再被抓）

### 既有論文索引

`_indexed.json` 由 `python -m scripts.literature_monitor.run --rebuild-index`
產生，記錄 `references/<topic>/*.pdf` 的標題列表，給去重邏輯交叉比對用。
新增 / 移除 PDF 後請 rerun 此指令。
