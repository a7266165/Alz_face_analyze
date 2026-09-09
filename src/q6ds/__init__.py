"""6Q-DS(六題失智症篩檢量表)問卷 modality。

原始資料是三份 xlsx(data/q6ds/),彼此高度重疊但各自成表,故三份各建一張建模用
CSV、各跑一輪 nested CV、各存一組模型。特徵固定為 questionaire.jpg 的 q1~q10
加性別、年紀,標籤是 Dx —— 其餘欄位(CASI / MMSE / CDR / AD8 / 教育年數 / 診斷
文字)一律以 ref_ 前綴留在 CSV 供對帳與分層檢視,不進模型。
"""

from .dataset import (
    DATASETS,
    FEATURE_COLS,
    FEATURE_COLS_NOAGE,
    LABEL_COL,
    Q_COLS,
    build_dataset,
    load_dataset,
    reconcile_sources,
    write_dataset,
)
from .model import ARMS, arm_spec
from .evaluate import run_nested_cv, summarize_folds
from .train import fit_final_model, train_dataset

__all__ = [
    "DATASETS",
    "FEATURE_COLS",
    "FEATURE_COLS_NOAGE",
    "LABEL_COL",
    "Q_COLS",
    "build_dataset",
    "load_dataset",
    "reconcile_sources",
    "write_dataset",
    "ARMS",
    "arm_spec",
    "run_nested_cv",
    "summarize_folds",
    "fit_final_model",
    "train_dataset",
]
