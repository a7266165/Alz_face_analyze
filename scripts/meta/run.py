"""Meta producer：對 feature set × asymmetry variant × scorer 窮舉跑 TabPFN v3,落地 leaderboard + OOF。

feature set = full(age, MMSE, CASI, original-OOF, asym-OOF) + 三個 MMSE/CASI 對照
(見 src.meta.FEATURE_SETS)。輸出寫到 META_DIR(workspace/meta)。

用法:
    python scripts/meta/run.py \\
        --p-visit p_first --p-score p_cdrall --hc-visit hc_all --hc-score hc_cdrall_or_mmseall \\
        --emb arcface --bg-mode background --device auto
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    META_DIR, cohort_dirs,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
)
from src.embedding.classification import ALL_METHODS, clf_param_label
from src.meta import sweep

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default="p_first")
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default="p_cdrall")
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default="hc_all")
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS),
                    default="hc_cdrall_or_mmseall")
    ap.add_argument("--emb", default="arcface")
    ap.add_argument("--bg-mode", choices=["background", "no_background"],
                    default="background")
    ap.add_argument("--photo-mode", choices=["mean", "all"], default="mean")
    ap.add_argument("--reducer", default="no_drop")
    # base OOF 身分(須與 embedding 落地時一致;original 用此格,asym 各 scorer)
    ap.add_argument("--base-clf", choices=list(ALL_METHODS), default="logistic",
                    help="original base 模型;定位 workspace 既有 forward OOF")
    ap.add_argument("--base-lr-C", type=float, default=1.0,
                    help="base-clf=logistic 時的 C(定位 C_<value> 落地格)")
    ap.add_argument("--embedding-root", type=Path, default=None,
                    help="embedding OOF 根目錄(預設 EMBEDDING_CLASSIFICATION_DIR)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-root", type=Path, default=META_DIR)
    args = ap.parse_args()

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    visit_dir, cdr_mmse_dir = cohort_dirs(*cohort)
    # 輸出路徑編入 base 模型身分,避免不同 base-clf 結果互相覆蓋。
    base_seg = args.base_clf + (
        f"/{clf_param_label(args.base_clf, args.base_lr_C)}"
        if clf_param_label(args.base_clf, args.base_lr_C) else "")
    out_dir = (args.output_root / visit_dir / cdr_mmse_dir / args.bg_mode
               / args.emb / args.photo_mode / args.reducer / base_seg / "tabpfn_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"cohort={cohort}  emb={args.emb}/{args.bg_mode}  base={base_seg}  out={out_dir}")

    leaderboard, oof_dump = sweep(
        cohort, emb=args.emb, bg_mode=args.bg_mode, photo_mode=args.photo_mode,
        reducer=args.reducer, base_clf=args.base_clf, base_lr_C=args.base_lr_C,
        seed=args.seed, device=args.device, root=args.embedding_root)

    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False, encoding="utf-8")
    for (feature_set, variant, method), oof in oof_dump.items():
        d = out_dir / feature_set
        if variant:
            d = d / variant / method
        d.mkdir(parents=True, exist_ok=True)
        oof.to_csv(d / "oof_scores.csv", index=False, encoding="utf-8")

    logger.info(f"wrote leaderboard ({len(leaderboard)} rows) + {len(oof_dump)} oof_scores.csv")
    top = leaderboard.iloc[0]
    tag = top["feature_set"] + (f"/{top['variant']}/{top['method']}" if top["variant"] else "")
    logger.info(f"best: {tag}  auc={top['auc']:.3f}  mcc={top['mcc']:.3f}")


if __name__ == "__main__":
    main()
