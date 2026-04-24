"""
Age-window sliding TabPFN classifier — 2-feature / 9-feature meta-learner.

Pipeline (per pipeline_{2,9}feat.png):
    1. Build feature matrix over the full cohort (P + NAD + ACS) with:
         --features 2  : real_age + age_error
         --features 9  : + 7 emotion category means averaged across 8 extractors
    2. Visit selection (first/latest/all) + Healthy filter (CDR=0 or MMSE≥26)
       on NAD/ACS only (P is kept as-is when Global_CDR is set)
    3. Binary labels: P=1, NAD∪ACS=0
    4. 100 repeated GroupKFold(k=5) runs (same person always in same fold;
       train 80 / test 20 of *subjects*)
    5. TabPFN predicts proba for the test fold in every run
    6. Collect every test prediction (subject_id, group_tag, y_true, y_prob,
       real_age) across runs × folds
    7. Age-window analysis: 10-year sliding window start 40..90, filter
       predictions by real_age ∈ [start, start+10), recompute metrics
    8. Also produce 3 views: AD vs NAD, AD vs ACS, AD vs Healthy (union)

Outputs at workspace/age_window_classifier/{clf}/{feat}feat/:
    predictions.csv                          (all test predictions, flat)
    summary_by_view.csv                      (global + per-window per-view)
    fig_metrics_by_window_{view}.png         (line plot)
    fig_confusion_matrix_{view}.png          (aggregated CM)

Usage (Alz_face_test_2 env has tabpfn/sklearn/xgboost):
    "C:/Users/4080/anaconda3/envs/Alz_face_test_2/python.exe" \\
        scripts/experiments/run_age_window_classifier.py \\
        --features 9 --n-runs 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMOGRAPHICS_DIR = PROJECT_ROOT / "data" / "demographics"
PREDICTED_AGES_FILE = PROJECT_ROOT / "workspace" / "age" / "age_prediction" / "predicted_ages.json"
EMOTION_DIR = PROJECT_ROOT / "workspace" / "emotion" / "au_features" / "aggregated"
OUTPUT_BASE = PROJECT_ROOT / "workspace" / "age_window_classifier"

EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness",
            "surprise", "neutral"]
EMOTION_METHODS = ["dan", "fer", "hsemotion", "vit",
                    "openface", "libreface", "pyfeat", "poster_pp"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Cohort
# ============================================================

def load_demographics():
    frames = []
    for grp in ["P", "NAD", "ACS"]:
        path = DEMOGRAPHICS_DIR / f"{grp}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "ID" not in df.columns:
            for col in df.columns:
                if col in ("ACS", "NAD"):
                    df = df.rename(columns={col: "ID"})
                    break
        df["group"] = grp
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["MMSE"] = pd.to_numeric(df.get("MMSE"), errors="coerce")
        df["Global_CDR"] = pd.to_numeric(df.get("Global_CDR"), errors="coerce")
        frames.append(df)
    demo = pd.concat(frames, ignore_index=True)
    demo["base_id"] = demo["ID"].str.extract(r"^(.+)-\d+$")
    demo["visit"] = pd.to_numeric(
        demo["ID"].str.extract(r"-(\d+)$")[0], errors="coerce")
    return demo


def apply_visit_selection(df, selection):
    df = df[df["Age"].notna()].copy()
    df = df.sort_values(["base_id", "visit"])
    if selection == "all":
        return df
    if selection == "first":
        return df.groupby("base_id", as_index=False).first()
    return df.groupby("base_id", as_index=False).last()


def apply_labels_and_hc_filter(df):
    """P with CDR≥0.5 → label=1; NAD/ACS with strict HC rule → label=0.
    Unlabelled rows dropped."""
    rows = []
    for _, r in df.iterrows():
        g = r["group"]
        cdr = r.get("Global_CDR")
        mmse = r.get("MMSE")
        if g == "P":
            if pd.notna(cdr) and cdr >= 0.5:
                rows.append({**r.to_dict(), "label": 1})
        else:  # NAD / ACS — strict healthy filter
            if (pd.notna(cdr) and cdr == 0) or (
                    pd.isna(cdr) and pd.notna(mmse) and mmse >= 26):
                rows.append({**r.to_dict(), "label": 0})
    return pd.DataFrame(rows)


# ============================================================
# Features
# ============================================================

def attach_age_error(df):
    with open(PREDICTED_AGES_FILE, "r", encoding="utf-8") as f:
        pred = json.load(f)
    df = df.copy()
    df["predicted_age"] = df["ID"].map(pred)
    df = df.dropna(subset=["predicted_age"])
    df["real_age"] = df["Age"].astype(float)
    df["age_error"] = df["real_age"] - df["predicted_age"].astype(float)
    return df


def attach_emotion_means(df):
    """7 emotion category means, averaged across 8 methods."""
    method_tables = []
    for m in EMOTION_METHODS:
        path = EMOTION_DIR / f"{m}_harmonized.csv"
        if not path.exists():
            logger.warning(f"  emotion {m} missing, skipping")
            continue
        tb = pd.read_csv(path)
        cols = [f"{e}_mean" for e in EMOTIONS]
        miss = [c for c in cols if c not in tb.columns]
        if miss:
            logger.warning(f"  {m}: missing {miss}, skipping")
            continue
        tb = tb[["subject_id"] + cols].rename(
            columns={"subject_id": "ID",
                     **{f"{e}_mean": f"{m}__{e}" for e in EMOTIONS}}
        )
        method_tables.append(tb)
    if not method_tables:
        raise RuntimeError("no emotion csv loaded")
    # Outer-merge all method tables on ID
    merged = method_tables[0]
    for t in method_tables[1:]:
        merged = merged.merge(t, on="ID", how="outer")
    # Compute per-emotion mean across methods (NaN-aware)
    for e in EMOTIONS:
        cols = [c for c in merged.columns if c.endswith(f"__{e}")]
        merged[f"emo_{e}"] = merged[cols].mean(axis=1, skipna=True)
    keep = ["ID"] + [f"emo_{e}" for e in EMOTIONS]
    return df.merge(merged[keep], on="ID", how="inner")


# ============================================================
# Classifier
# ============================================================

def get_classifier(name, seed):
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(random_state=seed,
                                  ignore_pretraining_limits=True)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=seed, n_jobs=2, eval_metric="logloss",
            use_label_encoder=False,
        )
    raise ValueError(f"unknown classifier: {name}")


# ============================================================
# Training loop
# ============================================================

def run_repeats(feat_df, feat_cols, classifier, n_runs, n_folds, base_seed=0):
    """100 repeated GroupKFold runs. Returns long DataFrame of test predictions."""
    X = feat_df[feat_cols].to_numpy(dtype=float)
    y = feat_df["label"].to_numpy(dtype=int)
    groups = feat_df["base_id"].to_numpy()
    real_age = feat_df["real_age"].to_numpy(dtype=float)
    ids = feat_df["ID"].to_numpy()
    group_tag = feat_df["group"].to_numpy()

    all_rows = []
    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        # Shuffle rows deterministically by seed so the same group always goes
        # together but the fold assignment changes across runs
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(y))
        X_s, y_s, g_s = X[order], y[order], groups[order]
        age_s = real_age[order]
        ids_s = ids[order]
        tag_s = group_tag[order]

        gkf = GroupKFold(n_splits=n_folds)
        for fold_idx, (tr, te) in enumerate(gkf.split(X_s, y_s, groups=g_s)):
            try:
                clf = get_classifier(classifier, seed)
                clf.fit(X_s[tr], y_s[tr])
                prob = clf.predict_proba(X_s[te])[:, 1]
            except Exception as e:
                logger.warning(f"  run={run_idx} fold={fold_idx} fit failed: {e}")
                continue
            for j, ii in enumerate(te):
                all_rows.append({
                    "run": run_idx,
                    "fold": fold_idx,
                    "ID": ids_s[ii],
                    "group": tag_s[ii],
                    "real_age": age_s[ii],
                    "y_true": int(y_s[ii]),
                    "y_prob": float(prob[j]),
                })
        if (run_idx + 1) % 10 == 0 or run_idx == 0:
            logger.info(f"  run {run_idx+1}/{n_runs} done")
    return pd.DataFrame(all_rows)


# ============================================================
# Metrics filters + windows
# ============================================================

def _score_at_threshold(y_true, y_prob, threshold, metric):
    yh = (y_prob >= threshold).astype(int)
    if metric == "balacc":
        return balanced_accuracy_score(y_true, yh)
    if metric == "mcc":
        return matthews_corrcoef(y_true, yh)
    if metric == "youden":
        cm = confusion_matrix(y_true, yh, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return tpr + tnr - 1.0
    raise ValueError(f"unknown metric: {metric}")


def find_best_threshold(y_true, y_prob, metric="balacc"):
    """Sweep thresholds at unique y_prob values + boundary; return best."""
    if len(set(y_true)) < 2:
        return 0.5
    candidates = np.unique(np.concatenate([
        [0.0, 1.0], np.unique(np.round(y_prob, 4))
    ]))
    best = (-np.inf, 0.5)
    for t in candidates:
        s = _score_at_threshold(y_true, y_prob, t, metric)
        if s > best[0]:
            best = (s, float(t))
    return best[1]


def compute_metrics(pred_df, threshold=0.5):
    """Balanced acc / MCC / AUC + confusion matrix at the given threshold."""
    if pred_df.empty:
        return None
    y = pred_df["y_true"].to_numpy(dtype=int)
    p = pred_df["y_prob"].to_numpy(dtype=float)
    yh = (p >= threshold).astype(int)
    try:
        auc = roc_auc_score(y, p) if len(set(y)) == 2 else np.nan
    except ValueError:
        auc = np.nan
    return {
        "n": len(y),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "balacc": balanced_accuracy_score(y, yh),
        "mcc": matthews_corrcoef(y, yh),
        "auc": auc,
        "cm": confusion_matrix(y, yh, labels=[0, 1]),
        "threshold": threshold,
    }


def metrics_per_run(pred_df, threshold=0.5):
    """Compute metrics per run, then aggregate mean±std."""
    rows = []
    for run, g in pred_df.groupby("run"):
        m = compute_metrics(g, threshold)
        if m is not None:
            rows.append({k: v for k, v in m.items() if k != "cm"})
    if not rows:
        return None
    mdf = pd.DataFrame(rows)
    return {
        "n_runs": len(mdf),
        "threshold": threshold,
        "balacc_mean": mdf["balacc"].mean(),
        "balacc_std": mdf["balacc"].std(),
        "mcc_mean": mdf["mcc"].mean(),
        "mcc_std": mdf["mcc"].std(),
        "auc_mean": mdf["auc"].mean(),
        "auc_std": mdf["auc"].std(),
    }


def filter_view(pred_df, view):
    """view in {'AD_vs_NAD', 'AD_vs_ACS', 'AD_vs_HC'}."""
    if view == "AD_vs_NAD":
        return pred_df[(pred_df["group"] == "P") | (pred_df["group"] == "NAD")]
    if view == "AD_vs_ACS":
        return pred_df[(pred_df["group"] == "P") | (pred_df["group"] == "ACS")]
    if view == "AD_vs_HC":
        return pred_df[pred_df["group"].isin(["P", "NAD", "ACS"])]
    raise ValueError(view)


# ============================================================
# Plotting
# ============================================================

def plot_metrics_by_window(summary_df, out_path, view, group2_label,
                            threshold_mode="fixed"):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = summary_df["window_start"].values
    ax1.plot(x, summary_df["balacc_mean"], "o-", label="BalAcc", color="C0")
    ax1.plot(x, summary_df["mcc_mean"], "s-", label="MCC", color="C1")
    ax1.plot(x, summary_df["auc_mean"], "^-", label="AUC", color="C2")
    ax1.set_xlabel("Age window start (10-year window)")
    ax1.set_ylabel("Metric value")
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    w = 0.35
    ax2.bar(x - w / 2, summary_df["n_neg"], width=w, alpha=0.3,
            label=f"n ({group2_label})", color="C0")
    ax2.bar(x + w / 2, summary_df["n_pos"], width=w, alpha=0.3,
            label="n (Patient)", color="C3")
    ax2.set_ylabel("n samples")
    ax2.legend(loc="upper right")

    # Optional second-axis annotation of per-window threshold
    if "threshold" in summary_df.columns and threshold_mode != "fixed":
        # overlay threshold as small grey dots on ax1 scale
        ax1.plot(x, summary_df["threshold"], ".", color="gray",
                 alpha=0.6, label="threshold")
        ax1.legend(loc="upper left")

    ax1.set_title(f"{view} — Metrics by age window (threshold={threshold_mode})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, out_path, view, classifier, summary, threshold_mode="fixed"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white",
                    fontsize=12)
    labels = view.split("_vs_")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([labels[1], labels[0]])
    ax.set_yticklabels([labels[1], labels[0]])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    thr_str = f"t={summary.get('threshold', 0.5):.2f}" if threshold_mode == "fixed" else "per-window t"
    title = (f"{classifier.upper()} | {view}  ({thr_str})\n"
             f"BalAcc={summary['balacc_mean']:.3f}±{summary['balacc_std']:.3f}  "
             f"MCC={summary['mcc_mean']:.3f}±{summary['mcc_std']:.3f}  "
             f"AUC={summary['auc_mean']:.3f}±{summary['auc_std']:.3f}")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def analyze_predictions(pred_df, out_root, view_order, view_group2,
                         window_start, window_end, window_size,
                         threshold_mode, threshold_metric, min_per_class,
                         classifier_name, n_runs):
    """Compute metrics + plots given an already-populated predictions dataframe.

    threshold_mode:
      fixed        -> always use threshold=0.5
      per_window   -> per age window, find best threshold on pooled test preds
                       of that window maximizing threshold_metric
    """
    sub_tag = "fixed" if threshold_mode == "fixed" else f"per_window_{threshold_metric}"
    out_dir = out_root / sub_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for view in view_order:
        vp = filter_view(pred_df, view)
        if vp.empty:
            logger.info(f"  {view}: empty, skip")
            continue
        # Global — always at 0.5 (reference) unless fixed with non-0.5 (future)
        global_thr = 0.5 if threshold_mode == "fixed" else find_best_threshold(
            vp["y_true"].to_numpy(int), vp["y_prob"].to_numpy(float),
            metric=threshold_metric)
        global_summary = metrics_per_run(vp, global_thr)
        global_cm = compute_metrics(vp, global_thr)["cm"]

        # Per-window
        window_rows = []
        for start in range(window_start, window_end + 1):
            end = start + window_size
            w = vp[(vp["real_age"] >= start) & (vp["real_age"] < end)]
            # Determine window-specific threshold
            if threshold_mode == "fixed":
                wthr = 0.5
            else:
                # Pool ALL predictions in this window (across runs × folds) to
                # pick the best threshold
                if w.empty or len(set(w["y_true"])) < 2:
                    wthr = 0.5
                else:
                    wthr = find_best_threshold(
                        w["y_true"].to_numpy(int),
                        w["y_prob"].to_numpy(float),
                        metric=threshold_metric)
            per_run = []
            cm_acc = np.zeros((2, 2), dtype=int)
            for run, g in w.groupby("run"):
                m = compute_metrics(g, wthr)
                if m is None or m["n_pos"] < min_per_class or \
                        m["n_neg"] < min_per_class:
                    continue
                per_run.append({k: m[k] for k in
                                 ("n", "n_pos", "n_neg", "balacc", "mcc", "auc")})
                cm_acc += m["cm"]
            row = {"view": view, "window_start": start, "window_end": end,
                   "threshold": wthr}
            if not per_run:
                full = compute_metrics(w, wthr) if not w.empty else None
                row.update({
                    "n_pos": full["n_pos"] // max(n_runs, 1) if full else 0,
                    "n_neg": full["n_neg"] // max(n_runs, 1) if full else 0,
                    "balacc_mean": np.nan, "balacc_std": np.nan,
                    "mcc_mean": np.nan, "mcc_std": np.nan,
                    "auc_mean": np.nan, "auc_std": np.nan,
                    "status": "insufficient",
                })
            else:
                pdf = pd.DataFrame(per_run)
                row.update({
                    "n_pos": int(pdf["n_pos"].mean()),
                    "n_neg": int(pdf["n_neg"].mean()),
                    "balacc_mean": pdf["balacc"].mean(),
                    "balacc_std": pdf["balacc"].std(),
                    "mcc_mean": pdf["mcc"].mean(),
                    "mcc_std": pdf["mcc"].std(),
                    "auc_mean": pdf["auc"].mean(),
                    "auc_std": pdf["auc"].std(),
                    "status": "ok",
                })
            window_rows.append(row)
        window_df = pd.DataFrame(window_rows)
        summary_rows.append(window_df)

        ok = window_df[window_df["status"] == "ok"]
        if not ok.empty:
            plot_metrics_by_window(
                ok, out_dir / f"fig_metrics_by_window_{view}.png",
                view, view_group2[view], threshold_mode=threshold_mode)

        plot_confusion(global_cm,
                        out_dir / f"fig_confusion_matrix_{view}.png",
                        view, classifier_name, global_summary,
                        threshold_mode=threshold_mode)

        logger.info(f"  {view} [{sub_tag}]: global "
                    f"BalAcc={global_summary['balacc_mean']:.3f}  "
                    f"MCC={global_summary['mcc_mean']:.3f}  "
                    f"AUC={global_summary['auc_mean']:.3f}  "
                    f"(t={global_thr:.2f})")

    if summary_rows:
        pd.concat(summary_rows, ignore_index=True).to_csv(
            out_dir / "summary_by_view.csv", index=False)
    logger.info(f"  wrote {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=int, choices=[2, 9], default=9)
    ap.add_argument("--visit-selection", default="first",
                    choices=["first", "latest", "all"])
    ap.add_argument("--window-size", type=int, default=10)
    ap.add_argument("--window-start", type=int, default=40)
    ap.add_argument("--window-end", type=int, default=90,
                    help="inclusive last window start")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-runs", type=int, default=100,
                    help="repeated GroupKFold runs per pipeline spec")
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--classifier", default="tabpfn",
                    choices=["tabpfn", "xgboost"])
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--threshold-mode", default="both",
                    choices=["fixed", "per_window", "both"],
                    help="both = produce both sets of outputs in subdirs")
    ap.add_argument("--threshold-metric", default="balacc",
                    choices=["balacc", "mcc", "youden"],
                    help="used for per_window threshold selection")
    ap.add_argument("--reuse-predictions", action="store_true",
                    help="skip training; load existing predictions.csv "
                         "and only recompute metrics + plots")
    args = ap.parse_args()

    out_root = OUTPUT_BASE / args.classifier / f"{args.features}feat"
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== run_age_window_classifier ===")
    logger.info(f"  features={args.features}  clf={args.classifier}  "
                f"n_runs={args.n_runs}  visit={args.visit_selection}")
    logger.info(f"  threshold_mode={args.threshold_mode}  "
                f"metric={args.threshold_metric}")
    logger.info(f"  window=[start, start+{args.window_size}]  "
                f"start={args.window_start}..{args.window_end}")
    logger.info(f"  out_root -> {out_root}")

    pred_path = out_root / "predictions.csv"
    if args.reuse_predictions and pred_path.exists():
        logger.info(f"  REUSE mode: loading {pred_path}")
        pred_df = pd.read_csv(pred_path)
        logger.info(f"  loaded {len(pred_df)} predictions "
                    f"({pred_df['run'].nunique()} runs)")
    else:
        # 1. Cohort
        demo = load_demographics()
        demo = apply_visit_selection(demo, args.visit_selection)
        labeled = apply_labels_and_hc_filter(demo)
        logger.info(f"  labeled: P={int((labeled['label']==1).sum())}  "
                    f"NAD={int(((labeled['label']==0) & (labeled['group']=='NAD')).sum())}  "
                    f"ACS={int(((labeled['label']==0) & (labeled['group']=='ACS')).sum())}")

        # 2. Features
        feat = attach_age_error(labeled)
        feat_cols = ["real_age", "age_error"]
        if args.features == 9:
            feat = attach_emotion_means(feat)
            feat_cols += [f"emo_{e}" for e in EMOTIONS]
        feat = feat.dropna(subset=feat_cols).reset_index(drop=True)
        logger.info(f"  complete-feature rows: {len(feat)}  "
                    f"(P={int((feat['label']==1).sum())} "
                    f"HC={int((feat['label']==0).sum())})")

        # 3. 100 repeated GroupKFold runs
        pred_df = run_repeats(feat, feat_cols, args.classifier,
                               args.n_runs, args.n_folds, args.base_seed)
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"  total test predictions: {len(pred_df)}")

    # 4. Analyze with one or both threshold modes
    view_order = ["AD_vs_NAD", "AD_vs_ACS", "AD_vs_HC"]
    view_group2 = {"AD_vs_NAD": "NAD", "AD_vs_ACS": "ACS", "AD_vs_HC": "NAD+ACS"}
    modes = (["fixed", "per_window"] if args.threshold_mode == "both"
             else [args.threshold_mode])
    for mode in modes:
        analyze_predictions(
            pred_df, out_root, view_order, view_group2,
            args.window_start, args.window_end, args.window_size,
            mode, args.threshold_metric, args.min_per_class,
            args.classifier, args.n_runs,
        )

    logger.info(f"Done.")


if __name__ == "__main__":
    main()
