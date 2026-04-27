"""
10-Fold (90/10) vs (10/90) 流程對比圖

輸出: workspace/age/age_prediction/calibration/calibration_comparison_flowchart.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PROJECT_ROOT

from graphviz import Digraph

OUTPUT_DIR = PROJECT_ROOT / "workspace" / "age" / "age_prediction" / "calibration"


def build():
    g = Digraph(
        "calib_cmp",
        format="png",
        engine="dot",
        graph_attr={
            "rankdir": "TB",
            "fontname": "Arial",
            "fontsize": "12",
            "bgcolor": "white",
            "dpi": "200",
            "pad": "0.4",
            "nodesep": "0.8",
            "ranksep": "0.55",
            "newrank": "true",
        },
        node_attr={
            "fontname": "Arial",
            "fontsize": "10",
            "style": "filled",
            "penwidth": "1.5",
        },
        edge_attr={
            "fontname": "Arial",
            "fontsize": "9",
            "color": "#555555",
        },
    )

    # ── Title ──
    g.node("title",
           label="10-Fold Calibration: Train 90/10 vs Train 10/90",
           shape="plaintext", fontsize="16", fontcolor="#333333",
           fillcolor="transparent")

    # ── Shared input ──
    g.node("input", "ACS + NAD (552 subjects, 1,017 visits)\n+ P (3,266 visits)",
           shape="box", fillcolor="#FFF9C4", color="#F9A825")
    g.edge("title", "input", style="invis")

    # ── Shared: StratifiedKFold ──
    g.node("skf", "StratifiedKFold(10) × 30 random seeds\n→ 10 folds per seed, 30 seeds total",
           shape="box", fillcolor="#E0E0E0", color="#616161", style="filled,rounded")
    g.edge("input", "skf")

    # ── Shared: Each fold fits regression ──
    g.node("fit", "Each fold: fit error = a × pred_age + b\n(~900 training visits → a ≈ −0.268, b ≈ 23.1)\n10 models per seed, almost identical (a, b)",
           shape="box", fillcolor="#E0E0E0", color="#616161", style="filled,rounded")
    g.edge("skf", "fit")

    # ── Split into two methods ──
    # Left: 90/10
    with g.subgraph(name="cluster_left") as c:
        c.attr(label="Train 90% / Val 10%",
               style="filled", fillcolor="#E3F2FD", color="#E3F2FD",
               fontcolor="#1565C0", fontsize="13", penwidth="2")

        c.node("l_h_fold", "Healthy: each subject\nin val set of 1 fold\n→ 1 corrected value / seed",
               shape="box", fillcolor="#BBDEFB", color="#1565C0")
        c.node("l_p_fold", "P: all 10 folds applied\n→ 10 corrected values / seed",
               shape="box", fillcolor="#BBDEFB", color="#1565C0")
        c.node("l_h_avg", "30 seeds × 1 value\n= 30 values → avg",
               shape="box", fillcolor="#90CAF9", color="#1565C0",
               style="filled,rounded")
        c.node("l_p_avg", "30 seeds × 10 values\n= 300 values → avg",
               shape="box", fillcolor="#90CAF9", color="#1565C0",
               style="filled,rounded")

    # Right: 10/90
    with g.subgraph(name="cluster_right") as c:
        c.attr(label="Train 10% / Val 90%",
               style="filled", fillcolor="#EDE7F6", color="#EDE7F6",
               fontcolor="#4527A0", fontsize="13", penwidth="2")

        c.node("r_h_fold", "Healthy: each subject\nin val set of 9 folds\n→ 9 corrected values / seed",
               shape="box", fillcolor="#D1C4E9", color="#4527A0")
        c.node("r_p_fold", "P: all 10 folds applied\n→ 10 corrected values / seed",
               shape="box", fillcolor="#D1C4E9", color="#4527A0")
        c.node("r_h_avg", "30 seeds × 9 values\n= 270 values → avg",
               shape="box", fillcolor="#B39DDB", color="#4527A0",
               style="filled,rounded")
        c.node("r_p_avg", "30 seeds × 10 values\n= 300 values → avg",
               shape="box", fillcolor="#B39DDB", color="#4527A0",
               style="filled,rounded")

    # Edges
    g.edge("fit", "l_h_fold")
    g.edge("fit", "l_p_fold")
    g.edge("fit", "r_h_fold")
    g.edge("fit", "r_p_fold")
    g.edge("l_h_fold", "l_h_avg")
    g.edge("l_p_fold", "l_p_avg")
    g.edge("r_h_fold", "r_h_avg")
    g.edge("r_p_fold", "r_p_avg")

    # Rank alignment
    for pair in [("l_h_fold", "r_h_fold"), ("l_p_fold", "r_p_fold"),
                 ("l_h_avg", "r_h_avg"), ("l_p_avg", "r_p_avg")]:
        with g.subgraph() as s:
            s.attr(rank="same")
            s.node(pair[0])
            s.node(pair[1])
        g.edge(pair[0], pair[1], style="invis")

    # ── Key insight ──
    g.node("key",
           label=(
               "<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='4' CELLPADDING='6'>"
               "<TR><TD BGCOLOR='#FFF3E0' BORDER='1' STYLE='rounded'>"
               "<B>Key Insight</B><BR/>"
               "All 10 fold models produce nearly identical (a, b)<BR/>"
               "because linear regression on ~900 points is very stable.<BR/><BR/>"
               "avg(30 near-identical values) ≈ avg(270 near-identical values)<BR/>"
               "→ Both methods converge to the same result."
               "</TD></TR>"
               "</TABLE>>"
           ),
           shape="plaintext")

    g.edge("l_h_avg", "key", style="invis")
    g.edge("r_h_avg", "key", style="invis")

    # ── Result ──
    g.node("result",
           label=(
               "<<TABLE BORDER='0' CELLBORDER='1' CELLSPACING='0' CELLPADDING='5'>"
               "<TR>"
               "<TD BGCOLOR='#ECEFF1'><B> </B></TD>"
               "<TD BGCOLOR='#ECEFF1'><B>Healthy values</B></TD>"
               "<TD BGCOLOR='#ECEFF1'><B>P values</B></TD>"
               "<TD BGCOLOR='#ECEFF1'><B>MAE (All)</B></TD>"
               "</TR>"
               "<TR>"
               "<TD BGCOLOR='#E3F2FD'>90/10</TD>"
               "<TD>30</TD>"
               "<TD>300</TD>"
               "<TD><B>4.07</B></TD>"
               "</TR>"
               "<TR>"
               "<TD BGCOLOR='#EDE7F6'>10/90</TD>"
               "<TD>270</TD>"
               "<TD>300</TD>"
               "<TD><B>4.07</B></TD>"
               "</TR>"
               "</TABLE>>"
           ),
           shape="plaintext")

    g.edge("key", "result", style="invis")

    return g


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    g = build()
    out = str(OUTPUT_DIR / "calibration_comparison_flowchart")
    g.render(out, cleanup=True)
    print(f"Saved: {out}.png")


if __name__ == "__main__":
    main()
