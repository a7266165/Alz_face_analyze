"""
Three pipeline diagrams — old embedding pipeline, forward, reverse.

Renders three SEPARATE PNG + PDF figures (one per pipeline) instead of a single
combined cluster figure. Each diagram is a standalone vertical flow with
symmetric dual-input centering.

Outputs:
    paper/figures/pipeline_old_embedding.{png,pdf}
    paper/figures/pipeline_forward.{png,pdf}
    paper/figures/pipeline_reverse.{png,pdf}

Usage:
    conda run -n graphviz python scripts/visualization/plot_pipeline_fwd_rev.py
"""

from pathlib import Path

import graphviz

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "paper" / "figures"

FONT = "Microsoft JhengHei"
MONO = "Consolas"


# ============================================================
# Shared graph builder
# ============================================================

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
        },
        node_attr={
            "fontname": FONT,
            "fontsize": "10",
            "shape": "box",
            "style": "rounded,filled",
            "margin": "0.15,0.08",
            "width": "2.6",
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
# Pipeline ① — old embedding pipeline (meta_analysis)
# ============================================================

def build_old_embedding():
    g = make_graph(
        "pipeline_old_embedding",
        "① 舊 embedding pipeline\n"
        "512-d asymmetry → Logistic Regression + 遞減特徵篩選 (RFE)",
    )
    g.node("emb",
           "Embedding asymmetry .npy\nworkspace/embedding/features/<model>/<feature_type>/<sid>.npy\n"
           "feature_type ∈ {difference, |difference|, average,\n"
           "relative, |relative|}   ·   per subject 512-d",
           fillcolor="#FFF8E1", color="#F57F17")
    g.node("demo",
           "Demographics\nACS / NAD / P + CDR / Age\n+ predicted_ages.json",
           fillcolor="#C8E6C9", color="#2E7D32")
    g.node("loader",
           "DataLoader\nCDR threshold filter (P=1 vs NAD∪ACS=0)\n"
           "+ age-stratified balancing (qcut bins)\n"
           "+ visit selection  +  predicted-age ≥ 65y filter",
           fillcolor="#FFCDD2", color="#C62828")
    g.node("cv",
           "5-fold GroupKFold (base_id)\n"
           "→ Logistic Regression  (class_weight=balanced, C=1, lbfgs)\n"
           "→ |coef| as feature importance",
           fillcolor="#BBDEFB", color="#1565C0")
    g.node("rfe",
           "Recursive Feature Elimination loop\n"
           "drop bottom N=5 importance / iteration\n"
           "repeat until n_features ≤ 5",
           fillcolor="#FFE082", color="#F57F17")
    g.node("pred",
           "Per-iteration predictions + report\n"
           "n_features × cdr_threshold × model\n"
           "→ pred_probability/<dataset>_test.csv",
           fillcolor="#B2DFDB", color="#00695C")
    g.node("eval",
           "Per-iteration metrics\n"
           "AUC / MCC / F1 / BalAcc / Sens / Spec\n"
           "+ filter-corrected confusion matrix",
           fillcolor="#FFCCBC", color="#BF360C")

    # Pin dual top inputs at same rank for symmetric placement
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("emb")
        s.node("demo")

    g.edge("emb", "loader")
    g.edge("demo", "loader")
    g.edge("loader", "cv")
    g.edge("cv", "rfe", label="importance")
    g.edge("rfe", "cv", label="re-fit on selected", style="dashed",
           constraint="false")
    g.edge("rfe", "pred", label="converged")
    g.edge("pred", "eval")
    return g


# ============================================================
# Pipeline ② — forward strategy
# ============================================================

def build_forward():
    g = make_graph(
        "pipeline_forward",
        "② Forward\n"
        "(train-on-all → OOF score → matched subset → 5 partitions)",
    )
    g.node("emb",
           "Embedding 512-d mean\nArcFace / TopoFR / Dlib\n(.npy averaged over images)",
           fillcolor="#FFF8E1", color="#F57F17")
    g.node("full",
           "Full partition cohort\nad_vs_hc / ad_vs_nad / ad_vs_acs\nmmse_hilo / casi_hilo",
           fillcolor="#BBDEFB", color="#1565C0")
    g.node("feat", "Feature matrix\nX = embedding 512-d  ·  y = partition label",
           fillcolor="#C8E6C9", color="#2E7D32")
    g.node("cv", "10-fold GroupKFold (base_id)\nLR / XGB",
           fillcolor="#C8E6C9", color="#2E7D32")
    g.node("oof", "OOF prediction score\n(per subject)",
           fillcolor="#B2DFDB", color="#00695C")
    g.node("match",
           "Subset to age 1:1 matched\n(arm_b reference: HC=180, NAD=169, ACS=29\n+ MMSE/CASI hi-lo medians)",
           fillcolor="#F8BBD0", color="#AD1457")
    g.node("test",
           "Paired Wilcoxon  (by pair_id)\n+ AUC + 95% CI / BalAcc / MCC / F1 / Sens / Spec",
           fillcolor="#FFCCBC", color="#BF360C")

    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("emb")
        s.node("full")

    g.edge("emb", "feat")
    g.edge("full", "feat")
    g.edge("feat", "cv")
    g.edge("cv", "oof", label="predict_proba")
    g.edge("oof", "match", label="subset by ID")
    g.edge("match", "test")
    return g


# ============================================================
# Pipeline ③ — reverse strategy
# ============================================================

def build_reverse():
    g = make_graph(
        "pipeline_reverse",
        "③ Reverse\n"
        "(train-on-matched → ensemble + single model → apply to unmatched)",
    )
    g.node("emb",
           "Embedding 512-d mean\nArcFace / TopoFR / Dlib",
           fillcolor="#FFF8E1", color="#F57F17")
    g.node("match",
           "Age 1:1 matched cohort\n(same as arm_b)",
           fillcolor="#BBDEFB", color="#1565C0")
    g.node("feat",
           "Feature matrix on matched\nX = embedding 512-d  ·  y = partition label",
           fillcolor="#C8E6C9", color="#2E7D32")
    g.node("cv",
           "10-fold GroupKFold on matched\nLR fold-scaler / XGB\n→ 10 fold-models  +  1 single model on all matched",
           fillcolor="#FFE0B2", color="#E65100")
    g.node("pred",
           "Predict on full cohort\n• ensemble: mean over 10 fold-models\n• single: 1 model trained on all matched",
           fillcolor="#B2DFDB", color="#00695C")
    g.node("eval",
           "Metrics (matched OOF · full ensemble · unmatched-only)\n"
           "AUC + 95% CI / BalAcc / MCC / F1 / Sens / Spec",
           fillcolor="#FFCCBC", color="#BF360C")

    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("emb")
        s.node("match")

    g.edge("emb", "feat")
    g.edge("match", "feat")
    g.edge("feat", "cv")
    g.edge("cv", "pred", label="apply ensemble + single")
    g.edge("pred", "eval")
    return g


# ============================================================
# Main
# ============================================================

def main():
    render(build_old_embedding(), "pipeline_old_embedding")
    render(build_forward(), "pipeline_forward")
    render(build_reverse(), "pipeline_reverse")


if __name__ == "__main__":
    main()
