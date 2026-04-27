# scripts/external — Public face dataset integration

外部公開人臉資料集（`external/public_face_datasets/`）的整理工具。所有腳本
讀路徑自 `src.config`（`EXTERNAL_DATASETS_DIR`、`EXTERNAL_FILTERED_DIR`）。

## 執行順序

```
# 1. 從 raw dataset 過濾亞裔 60+（寫入 filtered/{asian_elderly_60plus,IMDB_60_plus}/ flat）
python scripts/external/extract_asian_elderly.py

# 2. 統計各 dataset 年齡分布（可選，report 用）
python scripts/external/asian_age_stats.py

# 3. flat → per-subject folder + 產 manifest.csv
python scripts/external/build_subject_folders.py --dry-run   # 先驗證
python scripts/external/build_subject_folders.py             # 正式

# 4. manifest → data/demographics/EACS.csv
python scripts/external/build_external_demographics.py
```

## 各腳本責任

| 腳本 | 輸入 | 輸出 |
|---|---|---|
| `extract_asian_elderly.py` | `datasets/{AFAD,FairFace,UTKFace,MegaAge-Asian,DiverseAsian,SZU-EmoDage}/` | `filtered/asian_elderly_60plus/*.jpg`（flat） |
| `asian_age_stats.py` | `datasets/*/` | stdout report（無檔案輸出） |
| `build_subject_folders.py` | `filtered/{asian_elderly_60plus,IMDB_60_plus}/*.jpg` | `filtered/*/EACS_{SRC}_{...}-{visit}/*.jpg` + `filtered/manifest.csv` |
| `build_external_demographics.py` | `filtered/manifest.csv` | `data/demographics/EACS.csv` |

## Subject ID 規則

- **Single-image sources** (AFAD/MegaAge/FairFace/UTKFace/DiverseAsian/SZU-EmoDage)：
  `EACS_{SRC}_{seq:05d}-1`，每張圖 = 1 subject，Photo_Session=1
- **IMDB**：`EACS_IMDB_{nm_id}-{visit:02d}`，visit 依 (nm_id, photo_year) 組合升冪
  編號。同年多張落同一 visit folder。Age = photo_year - birth_year

Subject ID 的 base_id / visit 解析：run_4arm_deep_dive.py 用 `^(.+)-\d+$`
regex，通用於 `ACS1-1`、`P123-5`、`EACS_AFAD_00001-1`、`EACS_IMDB_nm0000002-03`。

## 依賴

- Python stdlib + 無其他套件（用 base Anaconda 即可）
