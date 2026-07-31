"""把部署用的 TabPFN(core3) fit 並 pickle 到 Alz_face_api/model/tabpfn_core3.pkl。

（base 兩條 LR 已改為 10 折 GroupKFold 集成，改由 scripts/deploy/export_deploy_folds.py
產出；本腳本只負責 meta 端的 TabPFN。）

對齊 embedding classification combo：
  P_first_HC_all / P_cdrall_HC_cdrall_mmseall / background / arcface /
  differences / all / no_drop / logistic / seed_0 / C_0.001 / fwd

語意（與訓練端一致、使用者確認）：
  - core3 表的兩個 base 分數：用「landed forward OOF」（leakage-free）。
  - core3 母體：keep_nan（complete_case=False）。
  - 標籤：Group=="P" → 1（AD）；NAD/ACS → 0（HC）。

執行（在 Alz_face_analyze 的分析 conda env，需 pandas/tabpfn；不需 cv2/sklearn-embedding）：
  conda activate <你的分析env>
  python scripts/deploy/export_deploy_models.py

注意：本腳本只「讀 workspace 落地的 forward OOF + predicted_ages + demographics」建 core3 表，
不重抽特徵、不重訓 base OOF。若 session_feature_table 報 FileNotFoundError，表示此 cohort 的
base original/differences forward OOF 尚未落地，需先跑 scripts/embedding/classification/run.py。
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np

# 讓 `import src.*` 解析得到（scripts/deploy/ → 上兩層 = repo 根）
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.meta.train import session_feature_table, META_FEATURE_SETS
from src.meta.classifier import make_meta_clf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_deploy_models")

# ============================================================================
# CONFIG（已與使用者逐項確認；改任一項都會改變產出數值）
# ============================================================================
COHORT = ("p_first", "p_cdrall", "hc_all", "hc_cdrall_or_mmseall")
EMB = "arcface"
BG = "background"
PHOTO = "all"            # 每張照片一列（推論端逐張打分再平均）
REDUCER = "no_drop"
LR_C = 0.001             # base OOF 的 LR 正則（用於定位 landed OOF 路徑）
SEED_FOLD = 0            # GroupKFold 折分 seed（定位 landed OOF）
ASYM_VARIANT = "differences"   # core3 asym 分支
TABPFN_SEED = 42         # TabPFNClassifier random_state
DEVICE = "auto"

OUT = Path("C:/Users/4080/Desktop/Alz_face_api/model")


def fit_tabpfn() -> None:
    """在 core3 表（base 分數取自 landed forward OOF）上 fit TabPFN 並 pickle。"""
    logger.info("=" * 70)
    logger.info("Fit TabPFN(core3) → tabpfn_core3.pkl")

    t = session_feature_table(
        COHORT,
        variant=ASYM_VARIANT,
        emb=EMB,
        bg_mode=BG,
        photo_mode=PHOTO,
        reducer=REDUCER,
        base_clf="logistic",
        lr_C=LR_C,
        seed=SEED_FOLD,
        complete_case=False,          # keep_nan：不丟 mmse/casi 缺值列
    )

    cols = META_FEATURE_SETS["core3"]   # [embedding_LR_score, asymmetry_LR_score, age_error]
    X = t[cols].to_numpy(dtype=float)
    y = t["y_true"].to_numpy(dtype=int)
    logger.info(f"  core3 表：sessions={len(t)}  pos(AD)={int(y.sum())}  "
                f"neg(HC)={int((y == 0).sum())}  cols={cols}")
    if np.isnan(X).any():
        logger.warning("  core3 X 含 NaN（理應無；請檢查 OOF / age_error 來源）")

    clf = make_meta_clf("tabpfn_v3", seed=TABPFN_SEED, device=DEVICE)
    clf.fit(X, y)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "tabpfn_core3.pkl", "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"  ✓ dumped {OUT / 'tabpfn_core3.pkl'}")


def main() -> None:
    logger.info(f"cohort={COHORT}  emb={EMB}  bg={BG}  photo={PHOTO}  "
                f"reducer={REDUCER}  C={LR_C}  asym={ASYM_VARIANT}")
    logger.info(f"輸出目錄：{OUT}")
    logger.info("（base 兩條 LR 由 export_deploy_folds.py 產出；本腳本只做 TabPFN）")

    fit_tabpfn()                                        # TabPFN(core3)

    logger.info("=" * 70)
    logger.info("完成。tabpfn_core3.pkl 已寫入 model/，可被 Alz_face_api 載入。")


if __name__ == "__main__":
    main()
