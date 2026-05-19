---
name: Conda environments
description: 17 個 conda env 對照表（分析/提取/emo_au/部署/雜項）+ 執行慣例
type: project
originSessionId: 3de60f8b-ee46-4b16-a6aa-7a8eebca3121
---
**不統一環境** — 各模組依賴版本衝突（torch 2.7 vs 2.5、mediapipe、各 emotion lib），必須分 env。
env requirement 檔在 `envs/*.txt`，setup 指令在 `envs/README.md`。

## Env 對照表

| Env | 用途 |
|---|---|
| **`Alz_face_main_analysis`** | 所有分析/plot script（sklearn/xgboost/tabpfn/scipy） |
| `Alz_face_test_2` | sandbox |
| `Alz_face_age` | age 預測（MiVOLO + insightface + tf） |
| `Alz_face_asymmetry` | landmark 不對稱（mediapipe, CPU） |
| `Alz_face_embedding_other` | arcface/topofr/dlib embedding |
| `Alz_face_embedding_vggface` | vggface embedding（tf + deepface） |
| `Alz_face_rotation` | 頭部旋轉（mediapipe, CPU） |
| `Alz_face_emo_au_openface` | OpenFace 3 |
| `Alz_face_emo_au_libreface` | LibreFace |
| `Alz_face_emo_au_pyfeat` | Py-Feat |
| `Alz_face_emo_au_other` | DAN/HSEmotion/ViT/POSTER_V2/EmoNeXt/EmoNet |
| `fer_env` | FER (justinshenk/fer) |
| `Alz_face_api` | API backend（FastAPI） |
| `Alz_face_ui` | UI（PyQt6） |
| `graphviz` | pipeline 圖 |
| `Alz_face_litmonitor` | 文獻 monitor（requests + tenacity） |
| `Alz_face_pdftext` | PDF 文字抽取（pymupdf） |

## 執行慣例

- **Consumer（分析/plot）**：`conda run -n Alz_face_main_analysis python scripts/<mod>/<file>.py`
- **Producer（提取）**：`"C:/Users/4080/anaconda3/envs/<env>/python.exe" scripts/<mod>/<file>.py`
- **安裝新套件**：先建 tmp env 試裝，不污染現有 env
