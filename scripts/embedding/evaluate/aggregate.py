"""把散落各 cell 的 metrics.csv 彙整成一張 cohort 級長表 —— 取代 legacy build_sweep_metrics.py。

evaluate 寫的 per-cell metrics.csv **只有評估軸**(direction/matched_unit/matching_priority/
eval_unit/contrast/domain + 指標),cell 身份(bg/emb/variant/photo/reducer/clf/clf_param)
編在**路徑**裡。本檔重用 classification/sweep 的 iter_cells + oof_paths_for
**直接從 cell dict 取身份**(免去「路徑反解」),逐檔讀 metrics.csv、補上身份欄、接成長表。

軸旗標與 classification/sweep / evaluate/sweep 一致 → 可彙整全集或任一切片。輸出長表既是
下游繪圖的單一資料源,本身也是可直接看的結果總表(排序好、CSV-friendly)。

註:ROC 等需要原始 y_score 的圖不吃這張表,直接讀 oof_scores.csv(不在此)。
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

import pandas as pd

from src.config import (
    EMBEDDING_CLASSIFICATION_REFACTOR_DIR, cohort_path,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
    DEFAULT_COHORT_TOKENS,
)
from src.embedding.classification import ALL_METHODS, CLASSIFIERS
from scripts.embedding.classification.run import _clf_param_label
from scripts.embedding.classification.sweep import (
    iter_cells, oof_paths_for, EMBEDDINGS, VARIANTS, BG_MODES, PHOTO_MODES, DIRECTIONS,
)

logger = logging.getLogger("aggregate_metrics")

# cell 身份欄(metrics.csv 沒有、編在路徑裡的部分)→ 擺在長表最前面
_IDENT_COLS = ["p_visit", "p_score", "hc_visit", "hc_score",
               "bg", "emb", "variant", "photo", "reducer", "model", "clf_param"]
# 排序鍵(存在才用):身份 → 評估軸
_SORT_KEYS = _IDENT_COLS + ["direction", "contrast", "eval_unit",
                            "matched_unit", "matching_priority", "domain"]


def _read_annotated(metrics_path, cell):
    """讀一個 metrics.csv,前置 cell 身份欄(身份取自 cell dict,非路徑反解)。"""
    df = pd.read_csv(metrics_path)
    clf_param = (_clf_param_label(cell["model"], cell["lr_C"], cell["xgb_params"])
                 if cell["model"] in CLASSIFIERS else None)
    ident = dict(
        p_visit=cell["cohort"][0], p_score=cell["cohort"][1],
        hc_visit=cell["cohort"][2], hc_score=cell["cohort"][3],
        bg=cell["bg"], emb=cell["emb"], variant=cell["variant"],
        photo=cell["photo"], reducer=cell["reducer"], model=cell["model"],
        clf_param=clf_param)
    for k, v in ident.items():
        df[k] = v
    return df[_IDENT_COLS + [c for c in df.columns if c not in _IDENT_COLS]]


def collect_metrics(args):
    """走 iter_cells → oof_paths_for,讀每個存在的 metrics.csv(forward 1、reverse 每
    priority 一個),補身份欄後接成一張長表 DataFrame(無資料則回空表)。"""
    root = args.output_root or EMBEDDING_CLASSIFICATION_REFACTOR_DIR
    frames = []
    for cell in iter_cells(args):
        for oof_path in oof_paths_for(cell, root):
            mpath = oof_path.parent / "metrics.csv"
            if mpath.exists():
                frames.append(_read_annotated(mpath, cell))
    rows = sum(len(f) for f in frames)
    logger.info(f"collected {len(frames)} metrics.csv -> {rows} rows")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # cohort(單一)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[0])
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[1])
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default=DEFAULT_COHORT_TOKENS[2])
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS), default=DEFAULT_COHORT_TOKENS[3])
    # 其餘軸(可多值)—— 與 classification_sweep / evaluate_sweep 一致,決定彙整哪片
    ap.add_argument("--bg-mode", nargs="+", choices=BG_MODES, default=BG_MODES)
    ap.add_argument("--embedding", nargs="+", default=EMBEDDINGS)
    ap.add_argument("--variant", nargs="+", default=VARIANTS)
    ap.add_argument("--photo-mode", nargs="+", choices=PHOTO_MODES, default=PHOTO_MODES)
    ap.add_argument("--model", nargs="+", choices=list(ALL_METHODS), default=list(ALL_METHODS))
    ap.add_argument("--reducer", nargs="+", default=["no_drop"])
    ap.add_argument("--direction", nargs="+", choices=DIRECTIONS, default=DIRECTIONS)
    ap.add_argument("--no-grid-search", dest="grid_search", action="store_false")
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="輸出 csv(預設 <root>/<cohort>/all_metrics.csv)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    root = args.output_root or EMBEDDING_CLASSIFICATION_REFACTOR_DIR
    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    df = collect_metrics(args)
    if df.empty:
        logger.warning("no metrics.csv found for the requested slice; nothing written.")
        return

    sort_keys = [c for c in _SORT_KEYS if c in df.columns]
    df = df.sort_values(sort_keys, na_position="last").reset_index(drop=True)
    out = args.out or (root / cohort_path(*cohort) / "all_metrics.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    logger.info(f"wrote {len(df)} rows x {df.shape[1]} cols -> {out}")


if __name__ == "__main__":
    main()
