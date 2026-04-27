"""
四種年齡預測校正方法流程圖

1. 10-Fold (Train 90% / Val 10%)  — ×30 seeds
2. 10-Fold (Train 10% / Val 90%)  — ×30 seeds
3. Bootstrap Correction           — ×1000 iter
4. Mean Correction                — single fit

輸出: workspace/age/age_prediction/age_correction_methods_flowchart.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PROJECT_ROOT

from graphviz import Digraph

OUTPUT_DIR = PROJECT_ROOT / "workspace" / "age" / "age_prediction"

NODE_W = "2.6"


def build_flowchart() -> Digraph:
    g = Digraph(
        "age_correction",
        format="png",
        engine="dot",
        graph_attr={
            "rankdir": "TB",
            "fontname": "Arial",
            "fontsize": "13",
            "bgcolor": "white",
            "dpi": "200",
            "pad": "0.5",
            "nodesep": "0.6",
            "ranksep": "0.6",
            "newrank": "true",
        },
        node_attr={
            "fontname": "Arial",
            "fontsize": "10",
            "style": "filled",
            "penwidth": "1.5",
            "width": NODE_W,
            "fixedsize": "false",
        },
        edge_attr={
            "fontname": "Arial",
            "fontsize": "8",
            "color": "#555555",
        },
    )

    # ── Title ──
    g.node(
        "title",
        label="Age Prediction Correction — Four Methods",
        shape="plaintext",
        fontsize="18",
        fontcolor="#333333",
        fillcolor="transparent",
    )

    # ── Shared input ──
    with g.subgraph(name="cluster_input") as c:
        c.attr(label="Input Data", style="dashed", color="#888888",
               fontcolor="#888888", fontsize="11")
        c.node("raw_pred", "Raw Predicted Ages\n(MiVOLO model)",
               shape="box", fillcolor="#FFF9C4", color="#F9A825")
        c.node("demo", "Demographics\n(ACS / NAD / P)",
               shape="box", fillcolor="#FFF9C4", color="#F9A825")
        c.node("error_def", "error = real_age − predicted_age",
               shape="box", fillcolor="#FFF3E0", color="#FF9800",
               fontsize="9", style="filled,rounded")

    g.edge("title", "raw_pred", style="invis")
    g.edge("raw_pred", "error_def")
    g.edge("demo", "error_def")

    # ═══════════════════════════════════════════════════
    # Method 1: 10-Fold (Train 90% / Val 10%) — BLUE
    # ═══════════════════════════════════════════════════
    with g.subgraph(name="cluster_m1") as c:
        c.attr(label="M1: 10-Fold (90/10)",
               style="filled", color="#E3F2FD", fillcolor="#E3F2FD",
               fontcolor="#1565C0", fontsize="12", penwidth="2")
        c.node("m1_s1", "ACS + NAD\n(n=1,017)",
               shape="box", fillcolor="#BBDEFB", color="#1565C0")
        c.node("m1_s2", "StratifiedKFold(10)\n×30 random seeds",
               shape="box", fillcolor="#BBDEFB", color="#1565C0")
        c.node("m1_s3", "Train 90% subjects\nfit: error = a×pred + b",
               shape="box", fillcolor="#90CAF9", color="#1565C0",
               style="filled,rounded")
        c.node("m1_s4", "Healthy: 1 val fold\nP: 10 folds avg",
               shape="box", fillcolor="#BBDEFB", color="#1565C0")
        c.node("m1_s5", "30 seeds → avg\ncorrected_age",
               shape="box", fillcolor="#E1F5FE", color="#0288D1",
               fontsize="9", style="filled,rounded")

    # ═══════════════════════════════════════════════════
    # Method 2: 10-Fold (Train 10% / Val 90%) — PURPLE
    # ═══════════════════════════════════════════════════
    with g.subgraph(name="cluster_m2") as c:
        c.attr(label="M2: 10-Fold (10/90)",
               style="filled", color="#EDE7F6", fillcolor="#EDE7F6",
               fontcolor="#4527A0", fontsize="12", penwidth="2")
        c.node("m2_s1", "ACS + NAD\n(n=1,017)",
               shape="box", fillcolor="#D1C4E9", color="#4527A0")
        c.node("m2_s2", "StratifiedKFold(10)\n×30 random seeds",
               shape="box", fillcolor="#D1C4E9", color="#4527A0")
        c.node("m2_s3", "Train 10% subjects\nfit: error = a×pred + b",
               shape="box", fillcolor="#B39DDB", color="#4527A0",
               style="filled,rounded")
        c.node("m2_s4", "Healthy: 9 val folds avg\nP: 10 folds avg",
               shape="box", fillcolor="#D1C4E9", color="#4527A0")
        c.node("m2_s5", "30 seeds → avg\ncorrected_age",
               shape="box", fillcolor="#EDE7F6", color="#5E35B1",
               fontsize="9", style="filled,rounded")

    # ═══════════════════════════════════════════════════
    # Method 3: Bootstrap — GREEN
    # ═══════════════════════════════════════════════════
    with g.subgraph(name="cluster_m3") as c:
        c.attr(label="M3: Bootstrap",
               style="filled", color="#E8F5E9", fillcolor="#E8F5E9",
               fontcolor="#2E7D32", fontsize="12", penwidth="2")
        c.node("m3_s1", "NAD only\n(age ≥ 60, n=763)",
               shape="box", fillcolor="#C8E6C9", color="#2E7D32")
        c.node("m3_s2", "Sample 1 subj\nper integer age",
               shape="box", fillcolor="#C8E6C9", color="#2E7D32")
        c.node("m3_s3", "fit: error = a×real + b\n×1000 iterations",
               shape="box", fillcolor="#A5D6A7", color="#2E7D32",
               style="filled,rounded")
        c.node("m3_s4", "NAD: exclude trained\nACS/P: all iters",
               shape="box", fillcolor="#C8E6C9", color="#2E7D32")
        c.node("m3_s5", "1000 iters → avg\ncorrected_age",
               shape="box", fillcolor="#E8F5E9", color="#43A047",
               fontsize="9", style="filled,rounded")

    # ═══════════════════════════════════════════════════
    # Method 4: Mean Correction — ORANGE
    # ═══════════════════════════════════════════════════
    with g.subgraph(name="cluster_m4") as c:
        c.attr(label="M4: Mean Correction",
               style="filled", color="#FFF3E0", fillcolor="#FFF3E0",
               fontcolor="#E65100", fontsize="12", penwidth="2")
        c.node("m4_s1", "NAD only\n(age ≥ 60, n=763)",
               shape="box", fillcolor="#FFE0B2", color="#E65100")
        c.node("m4_s2", "Per integer age:\nmean error (33 bins)",
               shape="box", fillcolor="#FFE0B2", color="#E65100")
        c.node("m4_s3", "fit: mean_err = a×age + b\n(single fit)",
               shape="box", fillcolor="#FFCC80", color="#E65100",
               style="filled,rounded")
        c.node("m4_s4", "Apply to all\n(ACS / NAD / P)",
               shape="box", fillcolor="#FFE0B2", color="#E65100")
        c.node("m4_s5", "corrected =\npred + (a×real + b)",
               shape="box", fillcolor="#FFF3E0", color="#FB8C00",
               fontsize="9", style="filled,rounded")

    # ── Force left-to-right ordering at each row ──
    for step in ["s1", "s2", "s3", "s4", "s5"]:
        with g.subgraph() as s:
            s.attr(rank="same")
            s.node(f"m1_{step}")
            s.node(f"m2_{step}")
            s.node(f"m3_{step}")
            s.node(f"m4_{step}")
        g.edge(f"m1_{step}", f"m2_{step}", style="invis")
        g.edge(f"m2_{step}", f"m3_{step}", style="invis")
        g.edge(f"m3_{step}", f"m4_{step}", style="invis")

    # ── Vertical edges within each method ──
    for prefix in ["m1", "m2", "m3", "m4"]:
        for i in range(1, 5):
            g.edge(f"{prefix}_s{i}", f"{prefix}_s{i+1}")

    # ── Input → four methods ──
    for m in ["m1", "m2", "m3", "m4"]:
        g.edge("error_def", f"{m}_s1")

    # ── Shared output ──
    with g.subgraph(name="cluster_output") as c:
        c.attr(label="Unified Output (per method)", style="dashed",
               color="#888888", fontcolor="#888888", fontsize="11")
        c.node("out_csv", "corrected_ages.csv\nsummary_stats.csv",
               shape="note", fillcolor="#E8EAF6", color="#3F51B5")
        c.node("out_plots",
               "scatter_before_after.png\nerror_distribution.png\nresidual_by_age.png",
               shape="note", fillcolor="#E8EAF6", color="#3F51B5")

    for m in ["m1", "m2", "m3", "m4"]:
        g.edge(f"{m}_s5", "out_csv")
    g.edge("out_csv", "out_plots", style="dashed")

    # ── Comparison table ──
    summary_label = (
        "<<TABLE BORDER='0' CELLBORDER='1' CELLSPACING='0' CELLPADDING='5'>"
        "<TR>"
        "<TD ROWSPAN='2' BGCOLOR='#ECEFF1'><B>Method</B></TD>"
        "<TD ROWSPAN='2' BGCOLOR='#ECEFF1'><B>Training<BR/>Data</B></TD>"
        "<TD ROWSPAN='2' BGCOLOR='#ECEFF1'><B>Regression<BR/>Target</B></TD>"
        "<TD COLSPAN='4' BGCOLOR='#ECEFF1'><B>MAE ↓</B></TD>"
        "<TD COLSPAN='4' BGCOLOR='#ECEFF1'><B>Mean Error</B></TD>"
        "</TR>"
        "<TR>"
        "<TD BGCOLOR='#ECEFF1'><B>All</B></TD>"
        "<TD BGCOLOR='#C8E6C9'><B>ACS</B></TD>"
        "<TD BGCOLOR='#BBDEFB'><B>NAD</B></TD>"
        "<TD BGCOLOR='#FFCDD2'><B>P</B></TD>"
        "<TD BGCOLOR='#ECEFF1'><B>All</B></TD>"
        "<TD BGCOLOR='#C8E6C9'><B>ACS</B></TD>"
        "<TD BGCOLOR='#BBDEFB'><B>NAD</B></TD>"
        "<TD BGCOLOR='#FFCDD2'><B>P</B></TD>"
        "</TR>"
        # M1: 10-Fold 90/10
        "<TR>"
        "<TD BGCOLOR='#E3F2FD'>10-Fold (90/10)</TD>"
        "<TD>ACS + NAD</TD>"
        "<TD>error ~ pred_age</TD>"
        "<TD><B>4.07</B></TD>"
        "<TD>3.74</TD>"
        "<TD><B>3.63</B></TD>"
        "<TD><B>4.20</B></TD>"
        "<TD>0.94</TD>"
        "<TD>−2.75</TD>"
        "<TD>0.77</TD>"
        "<TD>1.24</TD>"
        "</TR>"
        # M2: 10-Fold 10/90
        "<TR>"
        "<TD BGCOLOR='#EDE7F6'>10-Fold (10/90)</TD>"
        "<TD>ACS + NAD</TD>"
        "<TD>error ~ pred_age</TD>"
        "<TD><B>4.07</B></TD>"
        "<TD>3.74</TD>"
        "<TD><B>3.63</B></TD>"
        "<TD><B>4.20</B></TD>"
        "<TD>0.94</TD>"
        "<TD>−2.75</TD>"
        "<TD>0.77</TD>"
        "<TD>1.24</TD>"
        "</TR>"
        # M3: Bootstrap
        "<TR>"
        "<TD BGCOLOR='#E8F5E9'>Bootstrap</TD>"
        "<TD>NAD (≥60)</TD>"
        "<TD>error ~ real_age</TD>"
        "<TD>5.06</TD>"
        "<TD><B>3.69</B></TD>"
        "<TD>4.51</TD>"
        "<TD>5.29</TD>"
        "<TD>−1.06</TD>"
        "<TD>−0.12</TD>"
        "<TD>0.15</TD>"
        "<TD>−1.43</TD>"
        "</TR>"
        # M4: Mean Correction
        "<TR>"
        "<TD BGCOLOR='#FFF3E0'>Mean Corr.</TD>"
        "<TD>NAD (≥60)</TD>"
        "<TD>mean_err ~ age</TD>"
        "<TD>5.05</TD>"
        "<TD><B>3.69</B></TD>"
        "<TD>4.50</TD>"
        "<TD>5.27</TD>"
        "<TD>−0.95</TD>"
        "<TD>−0.03</TD>"
        "<TD>0.26</TD>"
        "<TD>−1.31</TD>"
        "</TR>"
        "</TABLE>>"
    )
    g.node("summary", label=summary_label, shape="plaintext")
    g.edge("out_plots", "summary", style="invis")

    return g


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    g = build_flowchart()
    output_path = str(OUTPUT_DIR / "age_correction_methods_flowchart")
    g.render(output_path, cleanup=True)
    print(f"Flowchart saved: {output_path}.png")


if __name__ == "__main__":
    main()
