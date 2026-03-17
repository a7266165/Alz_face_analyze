# References

本專案使用的模型及相關論文

---

## 年齡預測

### MiVOLO
- **用途**: 臉部年齡預測
- **模組**: `src/core/age_predictor.py`
- **論文**: MiVOLO: Multi-input Transformer for Age and Gender Estimation
- **作者**: Maksim Kuprashevich, Irina Tolstykh
- **年份**: 2023
- **連結**: https://arxiv.org/abs/2307.04616
- **GitHub**: https://github.com/WildChlamydia/MiVOLO

---

## 臉部偵測與特徵點

### MediaPipe Face Mesh
- **用途**: 臉部 468 特徵點偵測、中軸線計算
- **模組**: `src/core/preprocess/detector.py`
- **論文**: MediaPipe: A Framework for Building Perception Pipelines
- **作者**: Camillo Lugaresi et al. (Google)
- **年份**: 2019
- **連結**: https://arxiv.org/abs/1906.08172
- **文件**: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

---

## 臉部特徵提取

### TopoFR
- **用途**: 臉部拓撲特徵提取
- **模組**: `src/core/extractor/topofr_extractor.py`
- **論文**: TopoFR: A Closer Look at Topology Alignment on Face Recognition
- **作者**: Jun Dan et al.
- **年份**: 2024 (NeurIPS)
- **連結**: https://arxiv.org/abs/2411.02569
- **GitHub**: https://github.com/modelscope/facechain/tree/main/face_module/TopoFR

### dlib
- **用途**: 臉部特徵提取 (128D embedding)
- **模組**: `src/core/extractor/dlib_extractor.py`
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
- **模組**: `src/au_extraction/openface_extractor.py`
- **論文**: OpenFace 3.0: A Lightweight Multitask System for Comprehensive Facial Behavior Analysis
- **年份**: 2025
- **連結**: references/2025-OpenFace 3.0_A Lightweight Multitask System for Comprehensive Facial Behavior Analysis.pdf

### Py-Feat
- **用途**: AU 機率提取、情緒分類
- **模組**: `src/au_extraction/pyfeat_extractor.py`
- **論文**: Py-Feat: Python Facial Expression Analysis Toolbox
- **年份**: 2023
- **連結**: references/2023-Py-Feat_Python Facial Expression Analysis Toolbox.pdf
- **GitHub**: https://github.com/cosanlab/py-feat

### LibreFace
- **用途**: AU 強度提取、情緒分類
- **模組**: `src/au_extraction/libreface_extractor.py`
- **論文**: LibreFace: An Open-Source Toolkit for Deep Facial Expression Analysis
- **年份**: 2024
- **連結**: references/2024-LibreFace-An Open-Source Toolkit for Deep Facial Expression Analysis.pdf
- **GitHub**: https://github.com/ihp-lab/LibreFace

---

## 分類器

### TabPFN
- **用途**: 表格資料 few-shot 分類（Meta 分析階段）
- **模組**: `src/analysis/analyzer/tabpfn_analyzer.py`, `src/meta_analysis/model/trainer.py`
- **論文**: TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second
- **作者**: Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter
- **年份**: 2025
- **連結**: references/Grinsztajn_2025_TabPFN2.5.pdf

---

## 論文檔案

請將 PDF 檔案放置於此資料夾，建議命名格式：
```
{作者}_{年份}_{模型名稱}.pdf
```

例如：
- `Kuprashevich_2023_MiVOLO.pdf`
- `Dan_2024_TopoFR.pdf`
