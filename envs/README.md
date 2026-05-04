# Conda Environments — Alz_face_analyze

按 `workspace/<dir>` 角色重新規劃的 11 個 active env + 2 個保留不動 + `legacy/` 歷史 snapshot。

## Workspace dir → env 對應表

| Workspace dir | Producer env (extract) | Consumer env (analyze) |
|---|---|---|
| `age/` | `Alz_face_age` | `Alz_face_main_analysis` |
| `asymmetry/` | `Alz_face_asymmetry` | `Alz_face_main_analysis` |
| `embedding/` (ArcFace / TopoFR / Dlib) | `Alz_face_embedding_other` | `Alz_face_main_analysis` |
| `embedding/` (VGGFace) | `Alz_face_embedding_vggface` | `Alz_face_main_analysis` |
| `emotion/` (OpenFace 3) | `Alz_face_emo_openface` | `Alz_face_main_analysis` |
| `emotion/` (LibreFace) | `Alz_face_emo_libreface` | `Alz_face_main_analysis` |
| `emotion/` (Py-Feat) | `Alz_face_emo_pyfeat` | `Alz_face_main_analysis` |
| `emotion/` (DAN / HSEmotion / ViT / POSTER_V2 / EmoNeXt / FER) | `Alz_face_emo_other` | `Alz_face_main_analysis` |
| `rotation/` | `Alz_face_rotation` | `Alz_face_main_analysis` |
| `arms_analysis/` | — | `Alz_face_main_analysis` |
| `age_window_classifier/` | — | `Alz_face_main_analysis` |
| `longitudinal/` | `Alz_face_main_analysis` | `Alz_face_main_analysis` |

## Active envs

### Analysis（單一純分析環境）

| env | torch | 主要套件 | 用途 |
|---|---|---|---|
| **`Alz_face_main_analysis`** | 2.7.1+cu118 | tabpfn / xgboost / lightgbm / scikit-learn / feature-engine / statsmodels / pingouin / shap / matplotlib | 所有 `scripts/experiments/*.py`、`scripts/visualization/*.py`、`scripts/utilities/*.py` |

### Extract envs（依工具版本衝突分開）

| env | torch | 主要套件 | 服務目錄 |
|---|---|---|---|
| `Alz_face_age` | 2.7.1+cu118 | mivolo / insightface / ultralytics / deepface / tensorflow 2.20 / onnxruntime-gpu | `workspace/age/` |
| `Alz_face_asymmetry` | — (CPU) | mediapipe / opencv / numpy / scipy | `workspace/asymmetry/` |
| `Alz_face_embedding_other` | 2.7.1+cu118 | dlib 19.24.6 / insightface / onnxruntime-gpu | `workspace/embedding/{arcface,topofr,dlib}/` |
| `Alz_face_embedding_vggface` | — | tensorflow 2.20 / deepface / dlib 19.24.6 | `workspace/embedding/vggface/` |
| `Alz_face_rotation` | — (CPU) | mediapipe / opencv (含 contrib) / scipy | `workspace/rotation/` |
| `Alz_face_emo_openface` | 2.5.1+cu121 | openface-test 0.1.26 / timm 1.0.15 | `workspace/emotion/openface/` |
| `Alz_face_emo_libreface` | 2.5.1+cu121 | libreface 0.1.1 / dlib 19.24.6 / mediapipe | `workspace/emotion/libreface/` |
| `Alz_face_emo_pyfeat` | 2.5.1+cu121 | py-feat 0.6.2 / kornia / scipy<1.12 | `workspace/emotion/pyfeat/` |
| `Alz_face_emo_other` | 2.5.1+cu121 | emotiefflib 1.1.1 / timm / transformers / onnxruntime-gpu | `workspace/emotion/{dan,hsemotion,vit,poster_v2,emonext,fer,emonet}/` |

### Deployment / 雜項（保留不動）

| env | 用途 |
|---|---|
| `Alz_face_api` | backend FastAPI 服務（fastapi/uvicorn + 全 extractor lib） |
| `Alz_face_ui` | UI（PyQt6 + pyrealsense2 + mediapipe） |
| `graphviz` | pipeline diagram 視覺化（純 graphviz Python binding） |
| `Alz_face_test_2` | 留作 sandbox / backup（不再 default 使用） |

## 建立指令（conda + pip）

通則：
- conda 只用來建 python 跟 setuptools 起手；多數套件用 pip
- torch 用 pip + `--index-url` 拉 cu118 / cu121 wheel（conda solve 在這裡常死循環）
- dlib 用 `dlib==19.24.6` 預編譯 wheel（不用 conda-forge，避免 numpy/MKL 衝突）
- mivolo + 老 setup_requires 套件需 `setuptools<65` + `--no-build-isolation`

### `Alz_face_main_analysis`

```bash
conda create -n Alz_face_main_analysis python=3.11 -y
"<env>/python.exe" -m pip install \
    torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
"<env>/python.exe" -m pip install \
    numpy pandas scipy scikit-learn scikit-image \
    xgboost lightgbm matplotlib seaborn \
    statsmodels huggingface-hub joblib tqdm \
    tabpfn feature-engine pingouin shap
```

### `Alz_face_age`（MiVOLO）

```bash
conda create -n Alz_face_age python=3.11 -y
"<env>/python.exe" -m pip install "setuptools<65"
"<env>/python.exe" -m pip install \
    torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
"<env>/python.exe" -m pip install --no-build-isolation \
    git+https://github.com/WildChlamydia/MiVOLO.git
"<env>/python.exe" -m pip install \
    insightface ultralytics deepface tensorflow==2.20.0 \
    onnxruntime-gpu opencv-python-headless \
    timm transformers safetensors huggingface-hub \
    numpy pandas pillow tqdm
```

### `Alz_face_embedding_other`

```bash
conda create -n Alz_face_embedding_other python=3.11 -y
"<env>/python.exe" -m pip install \
    torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
"<env>/python.exe" -m pip install \
    dlib==19.24.6 insightface onnxruntime-gpu \
    opencv-python-headless numpy pandas pillow tqdm
```

### `Alz_face_embedding_vggface`

```bash
conda create -n Alz_face_embedding_vggface python=3.11 -y
"<env>/python.exe" -m pip install \
    tensorflow==2.21.0 tf-keras==2.21.0 \
    deepface==0.0.97 dlib==19.24.6 \
    opencv-python flask flask-cors gunicorn \
    numpy pandas scipy
# tf-keras is required by deepface on TF 2.16+; without it `from deepface import DeepFace` fails.
```

### `Alz_face_asymmetry`

```bash
conda create -n Alz_face_asymmetry python=3.11 -y
"<env>/python.exe" -m pip install \
    "mediapipe==0.10.21" opencv-python-headless numpy pandas scipy
# mediapipe must be ≤0.10.21: 0.10.22+ removed the legacy `mp.solutions.face_mesh`
# API that the project's landmark/rotation code depends on.
```

### `Alz_face_rotation`

```bash
conda create -n Alz_face_rotation python=3.11 -y
"<env>/python.exe" -m pip install \
    "mediapipe==0.10.21" opencv-python opencv-contrib-python \
    numpy scipy matplotlib
# mediapipe pin: same reason as Alz_face_asymmetry — 0.10.22+ dropped legacy API.
```

### `Alz_face_emo_openface`

```bash
conda create -n Alz_face_emo_openface python=3.11 -y
"<env>/python.exe" -m pip install \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
"<env>/python.exe" -m pip install openface-test==0.1.26
```

匯入時是 `import openface`（不是 openface_test）。

### `Alz_face_emo_libreface`

```bash
conda create -n Alz_face_emo_libreface python=3.11 -y
"<env>/python.exe" -m pip install \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
"<env>/python.exe" -m pip install \
    libreface==0.1.1 mediapipe opencv-python
# libreface 0.1.1 會把 torch 降到 2.0.0+cpu，需重新 install 升回 2.5.1+cu121
"<env>/python.exe" -m pip install --upgrade \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

### `Alz_face_emo_pyfeat`

```bash
conda create -n Alz_face_emo_pyfeat python=3.11 -y
"<env>/python.exe" -m pip install \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
"<env>/python.exe" -m pip install h5py
"<env>/python.exe" -m pip install --no-build-isolation \
    matplotlib opencv-python-headless kornia
"<env>/python.exe" -m pip install --no-build-isolation py-feat==0.6.2
"<env>/python.exe" -m pip install "scipy<1.12"  # py-feat 用了 scipy.integrate.simps（已重命名）
```

### `Alz_face_emo_other`（DAN / HSEmotion / ViT / POSTER_V2 / EmoNeXt / FER / emonet）

```bash
conda create -n Alz_face_emo_other python=3.11 -y
"<env>/python.exe" -m pip install \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
"<env>/python.exe" -m pip install \
    emotiefflib==1.1.1 \
    timm transformers safetensors tokenizers \
    huggingface-hub onnxruntime-gpu \
    opencv-python-headless numpy pandas
# 各 model 私有 weights / 私有 pip pkg 按 external/emotion/<model>/requirements.txt 補
```

### `Alz_face_api`（後端）

```bash
conda create -n Alz_face_api python=3.11 -y
"<env>/python.exe" -m pip install "setuptools<65"
"<env>/python.exe" -m pip install \
    torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
"<env>/python.exe" -m pip install --no-build-isolation \
    fastapi uvicorn dlib==19.24.6 insightface ultralytics \
    deepface mediapipe \
    git+https://github.com/WildChlamydia/MiVOLO.git \
    onnxruntime-gpu opencv-python-headless tensorflow==2.20.0 \
    timm transformers safetensors huggingface-hub \
    numpy pandas pillow tqdm
```

## 執行慣例

- **Analysis script**：用 `conda run -n Alz_face_main_analysis python scripts/...py`
- **Pipeline script (extract)**：在對應 extractor env 內 activate 後再跑（pipeline scripts 多半沒寫 conda run 在 docstring）
- **Windows cp950 + conda run**：含中文輸出或多行 subprocess 改用絕對路徑直跑：
  ```
  "C:/Users/4080/anaconda3/envs/<env>/python.exe" scripts/...
  ```
- **不改 `pyproject.toml`**：依專案慣例
- **新套件先建 `tmp_env`**：例外只有 `graphviz` env（pipeline diagram 專用）

## `legacy/` 歷史 snapshot

`envs/legacy/` 含 10 份舊 env 的 pip freeze 快照：

- `Alz_face_age.txt` — rebuild 前的 132-pkg 快照
- `Alz_face_analyze_emo.txt` — decommissioned（取代為 `Alz_face_emo_openface`）
- `Alz_face_analyze_only_VGGFace_py311.txt` — decommissioned（取代為 `Alz_face_embedding_vggface`）
- `Alz_face_api.txt` — rebuild 前的 113-pkg 快照
- `Alz_face_main_analysis.txt` — rebuild 前（原為 `Alz_face_test_2` clone）
- `Alz_face_relation_analysis.txt` — decommissioned（純分析角色已併入 `Alz_face_main_analysis`）
- `Alz_face_rotating.txt` — decommissioned（取代為 `Alz_face_rotation`）
- `emo_analyze.txt` — decommissioned（與 `Alz_face_analyze_emo` 重複）
- `libreface.txt` — decommissioned（取代為 `Alz_face_emo_libreface`）
- `pyfeat.txt` — decommissioned（取代為 `Alz_face_emo_pyfeat`）

要從中還原：
```bash
conda env create -n <name>_restored --file envs/legacy/<name>.txt
```
（注意 pip-installed 套件可能因上游 PyPI 變動而還原失敗；快照僅供參考。）
