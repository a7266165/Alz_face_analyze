"""
Raw age prediction flowchart (no correction).

輸出: workspace/age/raw_age_flowchart.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PROJECT_ROOT  # noqa: F401

from graphviz import Digraph
from src.config import AGE_DIR

OUTPUT_DIR = AGE_DIR


def build_flowchart() -> Digraph:
    g = Digraph(
        "raw_age",
        format="png",
        engine="dot",
        graph_attr={
            "rankdir": "LR",
            "fontname": "Microsoft JhengHei",
            "fontsize": "13",
            "bgcolor": "white",
            "dpi": "200",
            "pad": "0.5",
            "nodesep": "0.4",
            "ranksep": "0.55",
        },
        node_attr={
            "fontname": "Microsoft JhengHei",
            "fontsize": "10",
            "style": "filled,rounded",
            "shape": "box",
            "penwidth": "1.5",
        },
        edge_attr={
            "fontname": "Microsoft JhengHei",
            "fontsize": "8",
            "color": "#555555",
        },
    )

    g.node("photos", "1,200 張照片\n(P + NAD + ACS)",
           fillcolor="#FFF9C4", color="#F9A825")
    g.node("preprocess", "Preprocessing\nDetect → Select → Align",
           fillcolor="#E3F2FD", color="#1565C0")
    g.node("mivolo", "MiVOLO v2\nAge Prediction",
           fillcolor="#FFF3E0", color="#FF9800")
    g.node("agg", "Mean Aggregation\n(10 photos → 1 per subject)",
           fillcolor="#FFF3E0", color="#FF9800")
    g.node("error", "age_error\n= real_age − predicted_age",
           fillcolor="#FFF3E0", color="#FF9800", fontsize="9")
    g.node("out_pred", "predicted_age",
           fillcolor="#E8EAF6", color="#3F51B5")
    g.node("out_err", "age_error",
           fillcolor="#E8EAF6", color="#3F51B5")

    g.edge("photos", "preprocess")
    g.edge("preprocess", "mivolo")
    g.edge("mivolo", "agg")
    g.edge("agg", "error", label="  + Demographics\n  (real_age)")
    g.edge("agg", "out_pred")
    g.edge("error", "out_err")

    return g


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    g = build_flowchart()
    output_path = str(OUTPUT_DIR / "raw_age_flowchart")
    g.render(output_path, cleanup=True)
    print(f"Flowchart saved: {output_path}.png")


if __name__ == "__main__":
    main()
