"""
Pipeline diagrams for the PCA-reduction flow inside the forward/reverse
embedding classifier (run_fwd_rev_embedding.py).

Three diagrams are emitted:
    pipeline_pca_classifier_original.{png,pdf}
        — pre-flag pipeline: visit=first / photo=mean only. One row per
          subject (1280); no visit/photo expansion; no subject-aggregation
          step at eval (1:1 already).
    pipeline_pca_classifier.{png,pdf}
        — current pipeline with --visit-mode / --photo-mode flags. Feature
          matrix can expand to multiple rows per subject; GroupKFold(base_id)
          plus per-fold subject-level aggregation keeps eval leak-free.
    pipeline_pca_eigenvalue.{png,pdf}
        — post-hoc PCA on the full feature pool that produces the
          cumulative eigenvalue / total ratio bottom panel.

Outputs go to paper/figures/.

Usage:
    conda run -n graphviz python scripts/visualization/plot_pipeline_pca.py
"""

from pathlib import Path

import graphviz

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "paper" / "figures"

FONT = "Microsoft JhengHei"
MONO = "Consolas"

# Cluster fill colors (light pastels — same family used in fwd_rev pipeline).
COLOR_INPUT = "#E3F2FD"
COLOR_LOAD = "#FFF3E0"
COLOR_PCA = "#E8F5E9"
COLOR_CLF = "#F3E5F5"
COLOR_EVAL = "#FCE4EC"
COLOR_OUT = "#ECEFF1"


def make_graph(name, title):
    g = graphviz.Digraph(
        name,
        format="png",
        engine="dot",
        graph_attr={
            "rankdir": "TB",
            "fontname": FONT,
            "fontsize": "14",
            "bgcolor": "#FAFAFA",
            "pad": "0.4",
            "nodesep": "0.30",
            "ranksep": "0.45",
            "label": title,
            "labelloc": "t",
            "labeljust": "c",
            "fontcolor": "#212121",
            "newrank": "true",
            "compound": "true",
        },
        node_attr={
            "fontname": FONT,
            "fontsize": "10",
            "shape": "box",
            "style": "rounded,filled",
            "margin": "0.15,0.08",
            "width": "3.5",
            "fixedsize": "false",
        },
        edge_attr={
            "fontname": FONT,
            "fontsize": "8",
            "color": "#616161",
            "fontcolor": "#757575",
        },
    )
    return g




def render(g, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / stem
    g.render(str(out), cleanup=True)
    g.format = "pdf"
    g.render(str(out), cleanup=True)
    print(f"Saved: {out}.png & {out}.pdf")


# ============================================================
# Diagram 1 — pre-flag pipeline (visit=first / photo=mean only)
# ============================================================

def build_classifier_pca_original():
    g = make_graph(
        "pca_classifier_original",
        "Pre-flag PCA pipeline — visit=first / photo=mean (1 row per subject)",
    )

    # Input
    with g.subgraph(name="cluster_in") as c:
        c.attr(label="Input data", style="rounded,filled",
               fillcolor=COLOR_INPUT, color="#90CAF9", fontname=FONT, labeljust="c", labelloc="t")
        c.node("npy",
               "Per-subject embedding\\n"
               "shape (10, D) — 10 photos × D dims\\n"
               "(D=512 for ArcFace/TopoFR; 128 for Dlib)",
               fillcolor="#FFFFFF")
        c.node("cohort",
               "Build first-visit cohort\\n"
               "(ad_vs_hc / ad_vs_nad / ad_vs_acs / mmse_hilo / casi_hilo)\\n"
               "1 row per subject (base_id ≡ ID for first visit)\\n"
               "→ ~1280 rows for ad_vs_hc",
               fillcolor="#FFFFFF")

    # Feature load + mean
    with g.subgraph(name="cluster_load") as c:
        c.attr(label="Feature matrix construction",
               style="rounded,filled", fillcolor=COLOR_LOAD,
               color="#FFB74D", fontname=FONT, labeljust="c", labelloc="t")
        c.node("mean_pool",
               "load_embedding_mean(...)\\n"
               "for each subject ID:\\n"
               "vec = arr.mean(axis=0)   # 10 photos → 1 vec\\n"
               "→ X (N_subj, D), y (N_subj,), base_ids (N_subj,)\\n"
               "(N_subj ≈ 1280 unique rows)",
               fillcolor="#FFFFFF")

    # GroupKFold + per-fold PCA
    with g.subgraph(name="cluster_cv") as c:
        c.attr(label="10-fold GroupKFold(base_id)  ·  per-fold PCA fit",
               style="rounded,filled", fillcolor=COLOR_PCA,
               color="#81C784", fontname=FONT, labeljust="c", labelloc="t",
               margin="40")
        c.node("gkf",
               "GroupKFold(n_splits=10, groups=base_ids)",
               fillcolor="#FFFFFF")
        c.node("scaler",
               "If classifier == logistic:\\n"
               "scaler = StandardScaler().fit(X[train])\\n"
               "X_tr, X_val = scaler.transform(X[train, val])",
               fillcolor="#FFFFFF")
        c.node("pca_fit",
               "_fit_reducer(X_tr)\\n"
               "→ PCA(n_components=K, random_state=0).fit(X_tr)\\n"
               "K is fixed (int) or chosen to retain variance ratio",
               fillcolor="#FFFFFF")
        c.node("pca_apply",
               "_apply_reducer(pca, X)\\n"
               "X_tr ← pca.transform(X_tr)   # (N_tr, K)\\n"
               "X_val ← pca.transform(X_val) # (N_val, K)",
               fillcolor="#FFFFFF")

    # Classifier + OOF
    with g.subgraph(name="cluster_clf") as c:
        c.attr(label="Classifier fit + OOF prediction",
               style="rounded,filled", fillcolor=COLOR_CLF,
               color="#BA68C8", fontname=FONT, labeljust="c", labelloc="t")
        c.node("fit",
               "clf = make_classifier(name, seed)\\n"
               "logistic: LogisticRegression(C=1, class_weight=balanced)\\n"
               "xgb:      XGBClassifier(n_estimators=300, max_depth=6, …)\\n"
               "clf.fit(X_tr, y[train])",
               fillcolor="#FFFFFF")
        c.node("predict",
               "oof[val] = clf.predict_proba(X_val)[:, 1]",
               fillcolor="#FFFFFF")

    # Eval — no aggregate step (1 row per subject already)
    with g.subgraph(name="cluster_eval") as c:
        c.attr(label="Matched-pair evaluation",
               style="rounded,filled", fillcolor=COLOR_EVAL,
               color="#F06292", fontname=FONT, labeljust="c", labelloc="t")
        c.node("metrics",
               "compute_clf_metrics(y, score)\\n"
               "AUC + 95% bootstrap CI / BalAcc / MCC / F1 / Sens / Spec\\n"
               "Wilcoxon paired test on matched pairs (180 / 169 / 29 / …)",
               fillcolor="#FFFFFF")

    # Output
    with g.subgraph(name="cluster_out") as c:
        c.attr(label="Cell-level outputs",
               style="rounded,filled", fillcolor=COLOR_OUT,
               color="#90A4AE", fontname=FONT, labeljust="c", labelloc="t")
        c.node("save",
               "Per-cell artifacts:\\n"
               "forward_matched_metrics.json\\n"
               "forward_oof_scores.csv\\n"
               "forward_cm_full.png · forward_cm_matched.png",
               fillcolor="#FFFFFF")

    g.edge("npy", "cohort", label="ID, label")
    g.edge("cohort", "mean_pool")
    g.edge("mean_pool", "gkf", label="train / val rows")
    g.edge("gkf", "scaler")
    g.edge("scaler", "pca_fit")
    g.edge("pca_fit", "pca_apply")
    g.edge("pca_apply", "fit")
    g.edge("fit", "predict")
    g.edge("predict", "metrics", label="OOF score per subject")
    g.edge("metrics", "save")

    return g


# ============================================================
# Diagram 2 — current pipeline (with --visit-mode / --photo-mode)
# ============================================================

def build_classifier_pca():
    g = make_graph(
        "pca_classifier",
        "Per-fold PCA with --visit-mode / --photo-mode flags "
        "(no train/eval leakage)",
    )

    # Input
    with g.subgraph(name="cluster_in") as c:
        c.attr(label="Input data", style="rounded,filled",
               fillcolor=COLOR_INPUT, color="#90CAF9", fontname=FONT, labeljust="c", labelloc="t")
        c.node("npy",
               "Per-subject embedding\\n"
               "shape (10, D)  — 10 photos × D dims\\n"
               "(D=512 for ArcFace/TopoFR; 128 for Dlib)",
               fillcolor="#FFFFFF")
        c.node("cohort",
               "Build partition cohort\\n"
               "(ad_vs_hc / ad_vs_nad / ad_vs_acs / mmse_hilo / casi_hilo)\\n"
               "label, base_id, group",
               fillcolor="#FFFFFF")

    # Feature load + mean
    with g.subgraph(name="cluster_load") as c:
        c.attr(label="Feature matrix construction",
               style="rounded,filled", fillcolor=COLOR_LOAD,
               color="#FFB74D", fontname=FONT, labeljust="c", labelloc="t")
        c.node("mean_pool",
               "load_embedding(...)\\n"
               "for each subject ID:\\n"
               "vec = arr.mean(axis=0)   # 10 photos → 1 vec\\n"
               "→ X (N_subj, D), y (N_subj,), base_ids (N_subj,)",
               fillcolor="#FFFFFF")

    # GroupKFold + per-fold PCA
    with g.subgraph(name="cluster_cv") as c:
        c.attr(label="10-fold GroupKFold(base_id)  ·  per-fold PCA fit",
               style="rounded,filled", fillcolor=COLOR_PCA,
               color="#81C784", fontname=FONT, labeljust="c", labelloc="t",
               margin="40")
        c.node("gkf",
               "GroupKFold(n_splits=10, groups=base_ids)\\n"
               "(同一 subject 的所有 row 一定在同一 fold)",
               fillcolor="#FFFFFF")
        c.node("scaler",
               "If classifier == logistic:\\n"
               "scaler = StandardScaler().fit(X[train])\\n"
               "X_tr, X_val = scaler.transform(X[train, val])",
               fillcolor="#FFFFFF")
        c.node("pca_fit",
               "_fit_reducer(X_tr)\\n"
               "→ PCA(n_components=K, random_state=0).fit(X_tr)\\n"
               "K is fixed (int) or chosen to retain variance ratio",
               fillcolor="#FFFFFF")
        c.node("pca_apply",
               "_apply_reducer(pca, X)\\n"
               "X_tr ← pca.transform(X_tr)   # (N_tr, K)\\n"
               "X_val ← pca.transform(X_val) # (N_val, K)",
               fillcolor="#FFFFFF")

    # Classifier + OOF
    with g.subgraph(name="cluster_clf") as c:
        c.attr(label="Classifier fit + OOF prediction",
               style="rounded,filled", fillcolor=COLOR_CLF,
               color="#BA68C8", fontname=FONT, labeljust="c", labelloc="t")
        c.node("fit",
               "clf = make_classifier(name, seed)\\n"
               "logistic: LogisticRegression(C=1, class_weight=balanced)\\n"
               "xgb:      XGBClassifier(n_estimators=300, max_depth=6, …)\\n"
               "clf.fit(X_tr, y[train])",
               fillcolor="#FFFFFF")
        c.node("predict",
               "oof[val] = clf.predict_proba(X_val)[:, 1]\\n"
               "(repeat over all 10 folds → OOF score per row)",
               fillcolor="#FFFFFF")

    # Eval
    with g.subgraph(name="cluster_eval") as c:
        c.attr(label="Subject-level aggregation + matched-pair evaluation",
               style="rounded,filled", fillcolor=COLOR_EVAL,
               color="#F06292", fontname=FONT, labeljust="c", labelloc="t")
        c.node("agg",
               "_aggregate_to_subject(oof_df, base_id)\\n"
               "(no-op when 1 row per base_id;\\n"
               " mean of rows when visit/photo expanded)",
               fillcolor="#FFFFFF")
        c.node("metrics",
               "compute_clf_metrics(y_subj, score_subj)\\n"
               "AUC + 95% bootstrap CI / BalAcc / MCC / F1 / Sens / Spec\\n"
               "Wilcoxon paired test on matched pairs",
               fillcolor="#FFFFFF")

    # Output
    with g.subgraph(name="cluster_out") as c:
        c.attr(label="Cell-level outputs",
               style="rounded,filled", fillcolor=COLOR_OUT,
               color="#90A4AE", fontname=FONT, labeljust="c", labelloc="t")
        c.node("save",
               "Per-cell artifacts:\\n"
               "forward_matched_metrics.json\\n"
               "(drop_corr_info includes K kept per fold)\\n"
               "forward_oof_scores.csv\\n"
               "forward_cm_full.png · forward_cm_matched.png",
               fillcolor="#FFFFFF")

    g.edge("npy", "cohort", label="ID, label")
    g.edge("cohort", "mean_pool")
    g.edge("mean_pool", "gkf", label="train / val rows")
    g.edge("gkf", "scaler")
    g.edge("scaler", "pca_fit")
    g.edge("pca_fit", "pca_apply")
    g.edge("pca_apply", "fit")
    g.edge("fit", "predict")
    g.edge("predict", "agg", label="row-level OOF")
    g.edge("agg", "metrics")
    g.edge("metrics", "save")

    return g


# ============================================================
# Diagram 3 — current pipeline: p_first_hc_all cohort × GPU PCA / GPU XGB
# ============================================================

def build_classifier_pca_p_first_hc_all():
    g = make_graph(
        "pca_classifier_p_first_hc_all",
        "Current PCA pipeline — p_first_hc_all cohort",
    )

    # Input
    with g.subgraph(name="cluster_in") as c:
        c.attr(label="Input data", style="rounded,filled",
               fillcolor=COLOR_INPUT, color="#90CAF9", fontname=FONT,
               labeljust="c", labelloc="t")
        c.node("npy",
               "Per-subject embeddings\\n"
               "(10 photos × embedding dim)",
               fillcolor="#FFFFFF")
        c.node("cohort",
               "Build partition cohort\\n"
               "AD: first visit, mild-or-worse dementia\\n"
               "HC: all NAD / ACS visits (no strict filter)\\n"
               "5 partitions: AD vs HC / NAD / ACS, MMSE-hilo, CASI-hilo",
               fillcolor="#FFFFFF")
        c.node("match",
               "1:1 age matching (subject-first)\\n"
               "pass 1: each subject used at most once on both sides\\n"
               "pass 2: visit-level fallback for unmatched\\n"
               "→ matched pairs",
               fillcolor="#FFFFFF")

    # Feature load
    with g.subgraph(name="cluster_load") as c:
        c.attr(label="Feature matrix construction (per visit)",
               style="rounded,filled", fillcolor=COLOR_LOAD,
               color="#FFB74D", fontname=FONT, labeljust="c", labelloc="t")
        c.node("mean_pool",
               "Per visit: mean over 10 photos → 1 embedding vector\\n"
               "Stack visits → feature matrix X with labels y, subject ids",
               fillcolor="#FFFFFF")

    # GroupKFold + PCA
    with g.subgraph(name="cluster_cv") as c:
        c.attr(label="10-fold cross-validation  ·  per-fold PCA",
               style="rounded,filled", fillcolor=COLOR_PCA,
               color="#81C784", fontname=FONT, labeljust="c", labelloc="t",
               margin="40")
        c.node("gkf",
               "GroupKFold (10 splits) grouped by subject id\\n"
               "each fold: ~90 % subjects train · ~10 % subjects val\\n"
               "all visits of one subject stay in the same fold\\n"
               "(no train/val leakage)",
               fillcolor="#FFFFFF")
        c.node("scaler",
               "Standardize features (logistic regression only)",
               fillcolor="#FFFFFF")
        c.node("pca_fit",
               "Fit PCA on training fold\\n"
               "(randomized SVD, n_components clamped to ≤ feature dim)",
               fillcolor="#FFFFFF")
        c.node("pca_apply",
               "Project train and val rows onto fitted components",
               fillcolor="#FFFFFF")

    # Classifier
    with g.subgraph(name="cluster_clf") as c:
        c.attr(label="Classifier fit + out-of-fold (OOF) prediction  ·  repeat × 10 folds",
               style="rounded,filled", fillcolor=COLOR_CLF,
               color="#BA68C8", fontname=FONT, labeljust="c", labelloc="t",
               margin="40")
        c.node("fit",
               "Logistic regression  or  XGBoost\\n"
               "fit on the training fold",
               fillcolor="#FFFFFF", width="5.0")
        c.node("predict",
               "Predict probability on the val fold\\n"
               "→ each row gets exactly 1 OOF score across the 10 folds",
               fillcolor="#FFFFFF", width="5.0")

    # Eval
    with g.subgraph(name="cluster_eval") as c:
        c.attr(label="Subject-level aggregation + matched-pair evaluation",
               style="rounded,filled", fillcolor=COLOR_EVAL,
               color="#F06292", fontname=FONT, labeljust="c", labelloc="t",
               margin="40")
        c.node("agg",
               "Average OOF scores across visits per subject\\n"
               "→ one score per subject",
               fillcolor="#FFFFFF", width="5.0")
        c.node("metrics",
               "Classification metrics: AUC (with 95 % CI),\\n"
               "balanced accuracy, MCC, F1, sensitivity, specificity,\\n"
               "confusion matrix; Wilcoxon paired test on matched pairs",
               fillcolor="#FFFFFF", width="5.0")

    # Output
    with g.subgraph(name="cluster_out") as c:
        c.attr(label="Cell-level outputs",
               style="rounded,filled", fillcolor=COLOR_OUT,
               color="#90A4AE", fontname=FONT, labeljust="c", labelloc="t")
        c.node("save",
               "Metrics summary  ·  OOF scores (visit + subject)\\n"
               "Paired-score scatter  ·  Confusion-matrix plots",
               fillcolor="#FFFFFF")

    g.edge("npy", "cohort")
    g.edge("cohort", "match", label="Arm B 1:1 age match")
    g.edge("match", "mean_pool", style="dashed", label="matched cohort")
    g.edge("mean_pool", "gkf", label="train / val rows")
    g.edge("gkf", "scaler")
    g.edge("scaler", "pca_fit")
    g.edge("pca_fit", "pca_apply")
    g.edge("pca_apply", "fit")
    g.edge("fit", "predict")
    g.edge("predict", "agg", label="visit-level OOF")
    g.edge("agg", "metrics", label="subject-level scores")
    g.edge("metrics", "save")

    return g


# ============================================================
# Diagram 4 — eigenvalue panel (separate, post-hoc)
# ============================================================

def build_eigenvalue_panel():
    g = make_graph(
        "pca_eigenvalue",
        "Cumulative eigenvalue ratio panel "
        "(post-hoc, once per embedding model)",
    )

    with g.subgraph(name="cluster_in") as c:
        c.attr(label="Input pool", style="rounded,filled",
               fillcolor=COLOR_INPUT, color="#90CAF9", fontname=FONT, labeljust="c", labelloc="t")
        c.node("scan",
               "Scan all per-subject embedding files:\\n"
               "arr = np.load(npy, allow_pickle=True)\\n"
               "vec = arr.mean(axis=0)   # 10 photos → 1 vec\\n"
               "→ matrix (N_npy, D)\\n"
               "(現況：N_npy = 4317 subjects；不過濾 cohort，含全部 P/NAD/ACS)",
               fillcolor="#FFFFFF")

    with g.subgraph(name="cluster_pca") as c:
        c.attr(label="PCA fit (no CV split — used only for spectrum, "
                     "not for classifier inputs)",
               style="rounded,filled", fillcolor=COLOR_PCA,
               color="#81C784", fontname=FONT, labeljust="c", labelloc="t")
        c.node("fit",
               "PCA().fit(X)\\n"
               "cum = np.cumsum(pca.explained_variance_ratio_)",
               fillcolor="#FFFFFF")
        c.node("table",
               "DataFrame: embedding × n_components ×\\n"
               "cumulative_variance_ratio",
               fillcolor="#FFFFFF")

    with g.subgraph(name="cluster_out") as c:
        c.attr(label="Plot output",
               style="rounded,filled", fillcolor=COLOR_OUT,
               color="#90A4AE", fontname=FONT, labeljust="c", labelloc="t")
        c.node("plot",
               "Bottom panel of per-partition forward-metrics PNGs\\n"
               "x: PCA n_components (1–512, log)\\n"
               "y: cumulative eigenvalue / total\\n"
               "(0.95 / 0.99 reference lines shown)",
               fillcolor="#FFFFFF")

    g.edge("scan", "fit")
    g.edge("fit", "table")
    g.edge("table", "plot")

    return g


# ============================================================
# Main
# ============================================================

def main():
    g0 = build_classifier_pca_original()
    render(g0, "pipeline_pca_classifier_original")
    g1 = build_classifier_pca()
    render(g1, "pipeline_pca_classifier")
    g2 = build_eigenvalue_panel()
    render(g2, "pipeline_pca_eigenvalue")
    g3 = build_classifier_pca_p_first_hc_all()
    render(g3, "pipeline_pca_classifier_p_first_hc_all")


if __name__ == "__main__":
    main()
