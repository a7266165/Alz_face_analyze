# Conda Environments

本專案因工具間的依賴衝突，使用三個獨立的 conda 環境。
三者皆基於 Python 3.11 + PyTorch 2.5.1 (CUDA 12.1)。

## 環境說明

| 環境名稱 | 用途 | 關鍵套件 | 對應腳本/模組 |
|----------|------|---------|-------------|
| `Alz_face_analyze_emo` | 分析 + OpenFace 提取 | xgboost, shap, scikit-learn, seaborn, openface3 | `run_classification.py`, `run_pca_eigenvector.py`, `run_au_emo_stats.py`, OpenFace extractor |
| `libreface_env` | LibreFace 提取 | libreface==0.1.1 | `run_au_pipeline.py --tools libreface` |
| `pyfeat_env` | Py-Feat 提取 | py-feat==0.6.2 | `run_au_pipeline.py --tools pyfeat` |

## 建立方式

### 主分析環境
```bash
conda create -n Alz_face_analyze_emo python=3.11 -y
conda activate Alz_face_analyze_emo
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install xgboost shap scikit-learn scipy pandas numpy matplotlib seaborn scikit-image tqdm opencv-python-headless timm huggingface-hub tensorboardx click reportlab pingouin
```

### LibreFace 環境
```bash
conda create -n libreface_env python=3.11 -y
conda activate libreface_env
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install libreface==0.1.1
```

### Py-Feat 環境
```bash
conda create -n pyfeat_env python=3.11 -y
conda activate pyfeat_env
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install py-feat==0.6.2
```

## 使用方式

```bash
# 提取（需在對應環境中執行）
conda activate libreface_env
python scripts/run_au_pipeline.py --tools libreface --steps extract

conda activate pyfeat_env
python scripts/run_au_pipeline.py --tools pyfeat --steps extract

conda activate Alz_face_analyze_emo
python scripts/run_au_pipeline.py --tools openface --steps extract

# 後處理和分析（在主環境中執行）
conda activate Alz_face_analyze_emo
python scripts/run_au_pipeline.py --steps harmonize aggregate
python scripts/run_classification.py
python scripts/run_pca_eigenvector.py
python scripts/run_au_emo_stats.py
```

## 完整依賴清單

各環境的完整 conda export 列表：
- `analyze.txt` — Alz_face_analyze_emo
- `libreface.txt` — libreface_env
- `pyfeat.txt` — pyfeat_env

可用以下指令還原：
```bash
conda create -n <env_name> --file envs/<file>.txt
```
