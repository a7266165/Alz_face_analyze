"""
Mean Correction 單一方法流程圖

輸出: workspace/age/mean_correction_flowchart.png
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
        "mean_correction",
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
            "ranksep": "0.5",
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

    # ── Input ──
    g.node("raw_pred", "Raw Predicted Ages\n(MiVOLO model)",
           fillcolor="#FFF9C4", color="#F9A825")
    g.node("demo", "Demographics\n(ACS / NAD / P)",
           fillcolor="#FFF9C4", color="#F9A825")
    g.node("error_def", "error = real_age − predicted_age",
           fillcolor="#FFF3E0", color="#FF9800", fontsize="9")
    g.edge("raw_pred", "error_def")
    g.edge("demo", "error_def")

    # ── Mean Correction pipeline ──
    with g.subgraph(name="cluster_mc") as c:
        c.attr(label="Mean Correction", style="filled,rounded",
               color="#FFF3E0", fillcolor="#FFF3E0",
               fontcolor="#E65100", fontsize="13", penwidth="2")

        c.node("s1", "篩選 NAD（age ≥ 60）\nn = 763",
               fillcolor="#FFE0B2", color="#E65100")
        c.node("s2", "依整數歲分組\n計算每歲平均 error\n（33 個 age bins）",
               fillcolor="#FFE0B2", color="#E65100")
        c.node("s3", "線性迴歸擬合\nmean_error = a × age + b\n(single fit)",
               fillcolor="#FFCC80", color="#E65100")
        c.node("s4", "套用至全體\n（ACS / NAD / P）",
               fillcolor="#FFE0B2", color="#E65100")
        c.node("s5", "corrected = predicted + (a × real_age + b)",
               fillcolor="#FBE9E7", color="#FB8C00", fontsize="9")

    g.edge("error_def", "s1")
    for i in range(1, 5):
        g.edge(f"s{i}", f"s{i+1}")

    # ── Output ──
    g.node("out", "corrected_age",
           fillcolor="#E8EAF6", color="#3F51B5", shape="box")
    g.edge("s5", "out")

    return g


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    g = build_flowchart()
    output_path = str(OUTPUT_DIR / "mean_correction_flowchart")
    g.render(output_path, cleanup=True)
    print(f"Flowchart saved: {output_path}.png")


if __name__ == "__main__":
    main()
