"""Meta producer(單一 session 層級 pipeline)。對統一的 8 個 feature combo 各落地一格自足 cell
(oof_scores.csv + metrics.csv),輸出寫到 META_ANALYSIS_DIR(workspace/meta/analysis)。

- 含影像 OOF 的 combo(core4*):逐 (asym variant, base C) 展開成 sibling cells
  → <feature_set>/<variant>/<base_clf>/C_<C>/tabpfn_v3/。
- 純認知 combo(mmse / casi / mmse_casi):無 OOF → 無 variant/C,只跑一次 → <feature_set>/tabpfn_v3/。
每格 metrics 由 src.common.evaluate 產(eval_by_subject、全 contrast × matched_unit × priority),並補
cell 身份欄供 aggregate.py 彙整。

用法:
    python scripts/meta/run.py --feature-set all \\
        --asym-variant differences absolute_differences relative_differences absolute_relative_differences \\
        --base-lr-C 0.001 0.01 0.1 1.0 10.0 100.0 --photo-mode all
    python scripts/meta/run.py --feature-set core4 --photo-mode all   # 只跑某個 combo
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.common.evaluate import evaluate
from src.config import (
    meta_analysis_path,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
)
from src.embedding.classification import (
    ALL_METHODS, clf_param_label, inner_path, oof_paths,
)
from src.meta import (
    ASYM_VARIANTS, BASELINE_FEATURE_SETS, META_CLASSIFIERS, META_FEATURE_SETS,
    feature_set_needs_oof,
    fold_aligned_shap, inner_feature_table, oof_from_table, session_feature_table,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("shap").setLevel(logging.WARNING)  # 壓掉 KernelExplainer 的 per-instance log
logger = logging.getLogger(__name__)

DEFAULT_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def _precheck_oof(cohort, *, emb, bg_mode, photo_mode, reducer, base_clf,
                  variants, lr_Cs, seed, root):
    """掃描所有需要的 landed embedding 分數,缺的一次列齊報錯。

    每格要兩個檔:oof_scores.csv(外折測試分數 → meta 的測試列)與 inner_scores.csv
    (內折分數 → meta 的訓練列)。後者要 embedding 端加 --inner-folds 5 才會產生。
    """
    needed = [("original", c) for c in lr_Cs] + [(v, c) for v in variants for c in lr_Cs]
    missing = []
    for v, c in needed:
        o = oof_paths(cohort, bg_mode, emb, v, photo_mode, reducer, base_clf,
                      "forward", lr_C=c, seed=seed, root=root)[0]
        i = inner_path(cohort, bg_mode, emb, v, photo_mode, reducer, base_clf,
                       "forward", lr_C=c, seed=seed, root=root)
        missing += [str(p) for p in (o, i) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少以下 base 分數檔:\n  " + "\n  ".join(missing) +
            "\n請跑 scripts/embedding/classification/run.py 產生"
            "(內折表需加 --inner-folds 5)。")


def _ident(cohort, args, *, feature_set, variant, base_clf, clf_param, meta_clf, seed):
    """cell 身份欄(編在路徑裡、metrics.csv 本身沒有的部分)→ 擺長表最前面利於彙整。"""
    return dict(
        p_visit=cohort[0], p_score=cohort[1], hc_visit=cohort[2], hc_score=cohort[3],
        bg=args.bg_mode, emb=args.emb, photo=args.photo_mode, reducer=args.reducer,
        feature_set=feature_set, variant=variant,
        base_clf=base_clf, clf_param=clf_param, meta_clf=meta_clf, seed=seed)


def _write_cell(oof, cohort, out_dir, ident):
    """寫一格:oof_scores.csv → evaluate(write=False) → 補身份欄 → metrics.csv;回 metrics。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_path = out_dir / "oof_scores.csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8")
    metrics = evaluate(oof_path, cohort, direction="forward",
                       eval_units=["eval_by_subject"], write=False)
    for k, v in ident.items():
        metrics[k] = v
    cols = list(ident) + [c for c in metrics.columns if c not in ident]
    metrics[cols].to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8")
    return metrics


def _write_shap(args, table, cols, meta_clf, out_dir, tag, inner=None):
    """--shap 開啟時:對該 cell 算逐個案 SHAP,寫 shap_per_case.csv + shap_importance.csv(與 oof/metrics 並排)。

    沿用同一張 table / 折分(逐折無洩漏);計算重(KernelExplainer × stacker),預設關閉。
    """
    if not args.shap:
        return
    per_case, importance = fold_aligned_shap(
        table, cols, inner=inner, meta_clf=meta_clf, seed=args.seed, device=args.device,
        background=args.shap_background)
    per_case.to_csv(out_dir / "shap_per_case.csv", index=False, encoding="utf-8")
    importance.to_csv(out_dir / "shap_importance.csv", index=False, encoding="utf-8")
    top = "; ".join(f"{r.feature}={r.importance_pct:.0%}" for r in importance.itertuples())
    logger.info(f"[{tag}] shap: per_case({len(per_case)}) + importance | {top}")


def _skip_pair(feature_set, meta_clf):
    """meta_clf=mean 只對 BASELINE_FEATURE_SETS 有意義(見 src/meta/train.py 的說明):
    它不 fit,直接把特徵當 AD 機率用。套到 mmse/casi 會得到 1-AUC,套到含 real_age 的
    combo 分數會落在 [0,1] 外。反過來 baseline feature set 配真的 stacker 是合理的
    (單特徵 LR 是單調轉換,AUC 幾乎一樣),不擋。"""
    return meta_clf == "mean" and feature_set not in BASELINE_FEATURE_SETS


def _log_hl(tag, oof, metrics):
    """印 ad_vs_hc / all 的 AUC·MCC·n 當進度。"""
    hl = metrics[(metrics["contrast"] == "ad_vs_hc") & (metrics["domain"] == "all")]
    msg = f"[{tag}] oof={len(oof)} sessions  metrics={len(metrics)} rows"
    if len(hl):
        r = hl.iloc[0]
        msg += f"  | ad_vs_hc all: auc={r['auc']:.3f} mcc={r['mcc']:.3f} n={int(r['n'])}"
    logger.info(msg)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-set", choices=["all"] + list(META_FEATURE_SETS),
                    default="all", help="跑哪個 combo;all=全部 8 個")
    ap.add_argument("--asym-variant", nargs="+", choices=list(ASYM_VARIANTS),
                    default=list(ASYM_VARIANTS),
                    help="[影像 combo] asymmetry 的 variant;逐個展開成 sibling cell")
    ap.add_argument("--base-lr-C", type=float, nargs="+", default=DEFAULT_C_VALUES,
                    help="[影像 combo] base logistic 的 C;逐個展開成 sibling cell")
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default="p_first")
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default="p_cdrall")
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default="hc_all")
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS),
                    default="hc_cdrall_or_mmseall")
    ap.add_argument("--emb", default="arcface")
    ap.add_argument("--bg-mode", choices=["background", "no_background"],
                    default="background")
    ap.add_argument("--photo-mode", choices=["mean", "all"], default="all")
    ap.add_argument("--reducer", default="no_drop")
    ap.add_argument("--base-clf", choices=list(ALL_METHODS), default="logistic",
                    help="original 與 asymmetry 共用的 base 模型(定位 embedding 既有 forward OOF)")
    ap.add_argument("--meta-clf", nargs="+", choices=list(META_CLASSIFIERS),
                    default=["tabpfn_v3"],
                    help="meta stacker;可多選(tabpfn_v3 / xgb),各落一個 leaf cell")
    ap.add_argument("--embedding-root", type=Path, default=None,
                    help="embedding OOF 根目錄(預設 EMBEDDING_CLASSIFICATION_DIR)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--seed", type=int, default=42, help="meta 估計器 seed(與折分 --fold-seed 無關)")
    ap.add_argument("--fold-seed", type=int, nargs="+", default=[0],
                    help="repeated-CV 折分 seed(路徑 seed_<N>):逐個讀對應 seed 的 base OOF、"
                         "寫對應 seed 的 meta cell;0=現有確定性折。例:--fold-seed 0 1 2 ... 29")
    ap.add_argument("--full-cohort", dest="complete_case", action="store_false",
                    help="保留 mmse/casi 缺值的 session(預設 complete-case 丟掉這些,讓 9 combo 同母體比較)")
    ap.add_argument("--shap", action="store_true",
                    help="同時對每個落地 cell 算逐個案 SHAP(shap_per_case.csv + shap_importance.csv);"
                         "計算重(KernelExplainer × stacker),通常只配特定 feature-set/meta-clf 用")
    ap.add_argument("--shap-background", type=int, default=25,
                    help="[--shap] 每折 KernelExplainer 背景樣本數(kmeans centroid 數)")
    args = ap.parse_args()
    if args.shap and args.feature_set == "all":
        logger.warning("--shap 搭配 --feature-set all:會對整個 grid 的每個 cell 算 SHAP(極慢);"
                       "通常應指定單一 feature-set/meta-clf。")

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    sets = (META_FEATURE_SETS if args.feature_set == "all"
            else {args.feature_set: META_FEATURE_SETS[args.feature_set]})
    cognitive = {k: v for k, v in sets.items() if not feature_set_needs_oof(v)}
    imaging = {k: v for k, v in sets.items() if feature_set_needs_oof(v)}
    logger.info(f"cohort={cohort}  emb={args.emb}/{args.bg_mode}/{args.photo_mode}  "
                f"cognitive={list(cognitive)}  imaging={list(imaging)}  "
                f"variants={args.asym_variant}  C={args.base_lr_C}  meta_clf={args.meta_clf}  "
                f"fold_seed={args.fold_seed}  complete_case={args.complete_case}")

    for fold_seed in args.fold_seed:
        _run_seed(args, cohort, cognitive, imaging, fold_seed)


def _run_seed(args, cohort, cognitive, imaging, fold_seed):
    """單一 fold_seed 的全套產出(讀該 seed 的 base OOF、寫該 seed 的 meta cell)。"""
    logger.info(f"=== fold_seed={fold_seed} ===")
    case_mode = "no_nan" if args.complete_case else "keep_nan"  # meta 母體子樹(丟/保留認知缺值)
    if imaging:
        _precheck_oof(cohort, emb=args.emb, bg_mode=args.bg_mode,
                      photo_mode=args.photo_mode, reducer=args.reducer,
                      base_clf=args.base_clf, variants=args.asym_variant,
                      lr_Cs=args.base_lr_C, seed=fold_seed, root=args.embedding_root)

    common = dict(emb=args.emb, bg_mode=args.bg_mode, photo_mode=args.photo_mode,
                  reducer=args.reducer, base_clf=args.base_clf, seed=fold_seed,
                  complete_case=args.complete_case, root=args.embedding_root)

    # 認知 combo:無 OOF/variant/C,只跑一次(用任一 variant/C 讀表取 fold + 認知欄,皆 invariant)
    #
    # 這些 combo 的特徵不含 base 分數,所以「用內折表當訓練列」與舊做法(用 fold≠k 的
    # 外折列)其實是同一組人、同一組特徵值,結果應完全相同——正因如此它們是乾淨的
    # 負對照:新舊有差就是折分或彙整造成的雜訊,不是方法本身。
    if cognitive:
        ref_variant, ref_C = args.asym_variant[0], args.base_lr_C[0]
        t0 = session_feature_table(cohort, variant=ref_variant, lr_C=ref_C, **common)
        i0 = inner_feature_table(cohort, variant=ref_variant, lr_C=ref_C, **common)
        for fs, cols in cognitive.items():
            for mc in args.meta_clf:
                if _skip_pair(fs, mc):
                    continue
                oof = oof_from_table(t0, cols, inner=i0, meta_clf=mc, seed=args.seed,
                                     device=args.device)
                out_dir = meta_analysis_path(*cohort, args.bg_mode, args.emb, args.photo_mode,
                                             args.reducer, case_mode=case_mode, feature_set=fs,
                                             meta_classifier=mc, seed=fold_seed)
                ident = _ident(cohort, args, feature_set=fs, variant=None,
                               base_clf=None, clf_param=None, meta_clf=mc, seed=fold_seed)
                tag = f"seed_{fold_seed}/{fs}/{mc}"
                _log_hl(tag, oof, _write_cell(oof, cohort, out_dir, ident))
                _write_shap(args, t0, cols, mc, out_dir, tag, inner=i0)

    # 影像 combo:逐 (C, variant) 組一次表,slice 各 combo × 各 meta_clf;同表直接互比
    for c in args.base_lr_C:
        if not imaging:
            break
        clf_param = clf_param_label(args.base_clf, c)
        for variant in args.asym_variant:
            t = session_feature_table(cohort, variant=variant, lr_C=c, **common)
            inner = inner_feature_table(cohort, variant=variant, lr_C=c, **common)
            for fs, cols in imaging.items():
                for mc in args.meta_clf:
                    if _skip_pair(fs, mc):
                        continue
                    oof = oof_from_table(t, cols, inner=inner, meta_clf=mc,
                                         seed=args.seed, device=args.device)
                    out_dir = meta_analysis_path(
                        *cohort, args.bg_mode, args.emb, args.photo_mode, args.reducer,
                        case_mode=case_mode, feature_set=fs, variant=variant,
                        base_classifier=args.base_clf, base_classifier_param=clf_param,
                        meta_classifier=mc, seed=fold_seed)
                    ident = _ident(cohort, args, feature_set=fs, variant=variant,
                                   base_clf=args.base_clf, clf_param=clf_param, meta_clf=mc,
                                   seed=fold_seed)
                    tag = f"seed_{fold_seed}/{fs}/{variant}/{clf_param}/{mc}"
                    _log_hl(tag, oof, _write_cell(oof, cohort, out_dir, ident))
                    _write_shap(args, t, cols, mc, out_dir, tag, inner=inner)


if __name__ == "__main__":
    main()
