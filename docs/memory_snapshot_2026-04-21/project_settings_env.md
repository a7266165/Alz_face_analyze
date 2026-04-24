---
name: Project settings and conda environments
description: 專案的 11 個 conda env 用途對照、執行慣例、環境管理規則。合併自原 feedback_env_handling + feedback_env_meta_learner。
type: project
originSessionId: 91922059-0ce1-41f4-9401-eea335107aee
---
## 通則

**不改 `pyproject.toml`、不統一環境。** 各模組（age / emotion / asymmetry / rotation / ui / api）依賴版本衝突嚴重（torch、mediapipe、timm、openface3 等），必須分 env 才能共存。

env 完整 requirement 檔在 `envs/*.txt`（11 份 pip freeze export）。`envs/README.md` 只文件化前三個 tool env，其他靠本備忘錄對照。

## Env 對照表（11 個）

### 📊 分析 / 實驗
| Env | 用途 | 腳本 |
|---|---|---|
| **`Alz_face_test_2`** | **所有 experiment** | `scripts/experiments/run_arm_*.py`、`run_layer*.py`、`run_meta_learner_*.py`、`run_tabpfn_*.py`、visualization、master 整合 |
| `Alz_face_analyze_emo` | 分析 + OpenFace 3 AU/emotion 提取 | `run_au_pipeline.py --tools openface`、原 emo 專案分析腳本 |

### 🧪 特徵提取（module-specific）
| Env | 用途 | 腳本 |
|---|---|---|
| `Alz_face_age` | MiVOLO 年齡預測 | `scripts/pipeline/predict_ages.py` |
| `Alz_face_relation_analysis` | 面部不對稱、embedding 相關分析 | landmark asymmetry、embedding diff 計算 |
| `Alz_face_rotating` | 頭部旋轉（PnP + vector angle） | `scripts/pipeline/process_angle.py` |
| `Alz_face_analyze_only_VGGFace_py311` | VGGFace embedding 提取 | VGGFace-only pipeline |

### 😊 情緒工具（每個方法一個 env）
| Env | 提供什麼 | 腳本 |
|---|---|---|
| `Alz_face_analyze_emo` | OpenFace 3 | `run_au_pipeline.py --tools openface` |
| `libreface_env` | LibreFace 0.1.1 | `run_au_pipeline.py --tools libreface` |
| `pyfeat_env` | Py-Feat 0.6.2 | `run_au_pipeline.py --tools pyfeat` |
| `fer_env` | FER (justinshenk/fer) | `scripts/pipeline/run_fer_extract.py` |
| `fer_models` | DAN / HSEmotion / ViT / POSTER_V2 / emonet / EmoNeXt 等新 emotion DL 模型 | `scripts/pipeline/run_new_models_extract.py` |

### 🌐 部署
| Env | 用途 |
|---|---|
| `Alz_face_api` | API backend |
| `Alz_face_ui` | UI frontend |

## 執行慣例

- **一般情況**：腳本內用 `conda run -n <env> python ...`（現有 docstring 慣例）
- **stdout 含中文 / subprocess 多行參數**：conda run 在 Windows cp950 下會炸；改用**絕對路徑**：
  `"C:/Users/4080/anaconda3/envs/Alz_face_test_2/python.exe" scripts/experiments/xxx.py`
- 檢視時 Python 小腳本建在 `c:/tmp/`、用 Write 工具寫出再執行（避免 `python -c "multi-line"` 導致 conda run NotImplementedError）

## 需要安裝新套件時

**不要改 `Alz_face_test_2` 等既有 env**。遵循 `CLAUDE.md` 全域規則：

1. 若只是**執行**一次性分析，先嘗試自寫實作（例：`bh_fdr` 自寫取代 statsmodels）
2. 若確實需要新套件，建 tmp env：
   ```bash
   conda create -n tmp_env python=3.11 -y
   conda activate tmp_env
   pip install <package>
   python ...
   conda deactivate && conda env remove -n tmp_env -y
   ```
3. 例外：`tmp_graphviz` 作為 `scripts/visualization/plot_pipeline_architecture.py` 的常駐 tmp env

## Why

- 11 個 env 看起來多，但每個都鎖定某版 torch 或關鍵套件，硬統一必衝突（例：OpenFace 3 綁 timm 1.0.x，Py-Feat 綁 0.9.x）
- 各 env 故障隔離；某 extractor 掛掉不會連帶拖倒其他分析
