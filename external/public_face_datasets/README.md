# External Public Face Datasets

用來擴充主專案 ACS（Asian Control Subjects）的公開人臉資料集池。
目的：解 cohort 中 ACS 只有 91 人、Arm A age gap +21.7y、Arm D × ACS
baseline-age NN matching n=21 pairs < MIN_CELL_N 的限制。

> 初次蒐集：2026-04-07；整合進主專案：2026-04-23。

## 目錄結構

```
external/public_face_datasets/
├── README.md                       # 本檔（tracked）
├── DOWNLOAD_STATUS.md              # 下載狀態 + 各資料集條件（tracked）
├── asian_elderly_datasets.csv      # 32 個候選資料集 metadata 表（tracked）
├── datasets/                       # 13 個已下載 raw dataset（gitignored，~10+ GB）
│   ├── AFAD/ AAF/ AFD/ CACD/ FairFace/ GIRAF/ FaceAge/ Face-Age-10K/
│   ├── Diverse Asian Facial Ages/ IMDB-Clean/
│   └── datasets/                   # 舊 sibling 結構沿用
│       └── IMDB-WIKI/ LAOFIW/ MegaAge-Asian/ SZU-EmoDage/ UTKFace/
└── filtered/                       # pipeline-ready per-subject folders（gitignored）
    ├── manifest.csv                # 7,933 subjects metadata（由 build_subject_folders.py 產）
    ├── asian_elderly_60plus/       # 3,842 single-image subjects
    │   ├── EACS_AFAD_00001-1/ EACS_MegaAge_00001-1/ ... 等
    └── IMDB_60_plus/               # 4,091 subject-visits（1,065 identities，786 multi-visit）
        └── EACS_IMDB_{nm_id}-{visit}/
```

## Subject 統計（2026-04-23 整合後）

| Source | subjects | multi-visit? |
|---|---|---|
| IMDB | 4,091 (1,065 identities) | ✓ 786 identities ≥2 visits |
| MegaAge | 2,187 | ✗ single-image |
| FairFace | 932 | ✗ single-image |
| UTKFace | 284 | ✗ single-image |
| SZU-EmoDage | 240 | ✗ single-image |
| AFAD | 154 | ✗ single-image |
| DiverseAsian | 45 | ✗ single-image |
| **Total** | **7,933** | — |

## 整合流程

1. **下載 raw dataset** → `datasets/`（各資料集下載方式見 `DOWNLOAD_STATUS.md`）
2. **過濾亞裔 60+** → `scripts/external/extract_asian_elderly.py`
   輸出：`filtered/asian_elderly_60plus/*.jpg` + `filtered/IMDB_60_plus/*.jpg`（flat）
3. **重組 per-subject folder** → `scripts/external/build_subject_folders.py`
   輸出：`filtered/*/EACS_{SRC}_{...}-{visit}/*.jpg` + `filtered/manifest.csv`
4. **產 demographics** → `scripts/external/build_external_demographics.py`
   輸出：`data/demographics/EACS.csv`（7,933 rows + Source 欄）
5. **Feature extraction**（未做）→ 跑 `prepare_feature.py` 等 pipeline 對 EACS 影像
6. **4-arm deep-dive** → `scripts/experiments/run_4arm_deep_dive.py --hc-source {ACS, ACS_ext, EACS}`

## `--hc-source` 三模式

| mode | 載入 demographics | ACS 群體 |
|---|---|---|
| `ACS`（預設）| `ACS.csv` | 內部 91 人 |
| `ACS_ext` | `ACS.csv + EACS.csv` | 內部 + 外部 = 8,024 人 |
| `EACS` | `EACS.csv` only | 外部 7,933 人（strict HC 全 bypass） |

Strict HC filter 對 `Source != "internal"` 的 rows 自動通過（external 無
MMSE/CDR，視為 age-only control）。結果落 `workspace/age_ladder/deep_dive{_acs_ext,_eacs}/`。

## Non-Goals

- 未解未拿 AGE 相關需申請 dataset（ADReFV / PROMPT / 東大 AD / AgeDB / MORPH II / FG-NET / APPA-REAL / K-FACE / Park Aging Mind）
- 未整合 D-Dementia/Clinical 類（需 AD label，非 HC 擴充用途）
- 未為 EACS 跑 emotion 8-method（多半 single-image，emotion 特徵對單張無意義）
- 未為 EACS 跑 age_error（需有真實 label，IMDB 可做但其他 dataset 沒有 ground-truth age）
