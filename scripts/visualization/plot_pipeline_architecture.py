"""
Pipeline Architecture Diagram (Graphviz version)

一張圖呈現 Alz_face_analyze 的完整架構與細節。
使用 graphviz 繪製，支援中文字型。

Usage:
    conda run -n tmp_graphviz python scripts/visualization/plot_pipeline_architecture.py
"""

import graphviz

# ── Font ──
FONT = "Microsoft JhengHei"
MONO = "Consolas"


def main():
    g = graphviz.Digraph(
        "pipeline",
        format="png",
        engine="dot",
        graph_attr={
            "rankdir": "TB",
            "fontname": FONT,
            "fontsize": "14",
            "bgcolor": "#FAFAFA",
            "pad": "0.5",
            "nodesep": "0.25",
            "ranksep": "0.5",
            "compound": "true",
            "label": "Alz_face_analyze — Pipeline Architecture\nAlzheimer's Disease Facial Feature Analysis",
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
        },
        edge_attr={
            "fontname": FONT,
            "fontsize": "8",
            "color": "#616161",
            "fontcolor": "#757575",
        },
    )

    # ════════════════════════════════════════
    # DATA SOURCES
    # ════════════════════════════════════════
    with g.subgraph(name="cluster_data") as c:
        c.attr(
            label="Data Sources",
            style="rounded,filled",
            fillcolor="#E3F2FD",
            color="#1565C0",
            fontcolor="#1565C0",
            fontsize="12",
        )
        c.node("raw_img", "Raw Images\n(JPG, 10 photos/session)",
               fillcolor="#BBDEFB", color="#1565C0")
        c.node("demographics", "Demographics\nACS / NAD / P (CSV)",
               fillcolor="#BBDEFB", color="#1565C0")
        c.node("pred_ages", "Predicted Ages\n(calibrated JSON)",
               fillcolor="#BBDEFB", color="#1565C0")
        c.node("emo_scores", "Emotion Scores\nEmoNet (CSV)",
               fillcolor="#BBDEFB", color="#1565C0")

    # ════════════════════════════════════════
    # src/extractor/
    # ════════════════════════════════════════
    with g.subgraph(name="cluster_extractor") as ext:
        ext.attr(
            label="src/extractor/  —  特徵提取層",
            style="rounded,filled",
            fillcolor="#F1F8E9",
            color="#33691E",
            fontcolor="#33691E",
            fontsize="13",
        )

        # -- preprocess/ --
        with ext.subgraph(name="cluster_preprocess") as pre:
            pre.attr(
                label="preprocess/",
                style="rounded,filled",
                fillcolor="#C8E6C9",
                color="#2E7D32",
                fontcolor="#2E7D32",
                fontsize="11",
            )
            pre.node("detector", "FaceDetector\nMediaPipe 468 landmarks",
                     fillcolor="#E8F5E9", color="#2E7D32")
            pre.node("selector", "FaceSelector\n選擇最正面 N 張",
                     fillcolor="#E8F5E9", color="#2E7D32")
            pre.node("aligner", "FaceAligner\n雙眼連線旋轉對齊",
                     fillcolor="#E8F5E9", color="#2E7D32")
            pre.node("mirror", "MirrorGenerator\n左右鏡射影像",
                     fillcolor="#E8F5E9", color="#2E7D32")
            pre.edge("detector", "selector")
            pre.edge("selector", "aligner")
            pre.edge("aligner", "mirror")

        # -- features/ (5 modules) --
        with ext.subgraph(name="cluster_features") as feat:
            feat.attr(
                label="features/",
                style="rounded,filled",
                fillcolor="#DCEDC8",
                color="#558B2F",
                fontcolor="#558B2F",
                fontsize="11",
            )

            feat.node("embedding",
                      "embedding/\nArcFace · TopoFR · Dlib · VGGFace\n→ .npy 128-4096D",
                      fillcolor="#FFF8E1", color="#F57F17", fontcolor="#F57F17")
            feat.node("age",
                      "age/\nMiVOLO v2 + Bootstrap 校正\n→ predicted age JSON",
                      fillcolor="#E8EAF6", color="#283593", fontcolor="#283593")
            feat.node("emotion",
                      "emotion/\nOpenFace · LibreFace · Py-Feat\nHarmonize → Aggregate\n→ 60-124 features CSV",
                      fillcolor="#FBE9E7", color="#BF360C", fontcolor="#BF360C")
            feat.node("asymmetry",
                      "asymmetry/\ncalculate_differences (5 methods)\nLandmark 幾何不對稱\n→ L/R difference vectors",
                      fillcolor="#E0F2F1", color="#004D40", fontcolor="#004D40")
            feat.node("rotation",
                      "rotation/\nVector Angle · PnP\n→ yaw / pitch / roll",
                      fillcolor="#EFEBE9", color="#3E2723", fontcolor="#3E2723")

    # preprocess → features (invisible edges for layout)
    g.edge("mirror", "embedding", style="invis")

    # ════════════════════════════════════════
    # workspace/ (intermediate storage)
    # ════════════════════════════════════════
    g.node("workspace",
           "workspace/\npreprocess/ │ embedding/ │ age/ │ emotion/ │ asymmetry/ │ rotation/",
           shape="folder", fillcolor="#E0F7FA", color="#00695C",
           fontcolor="#00695C", fontsize="9")

    # ════════════════════════════════════════
    # src/meta_analysis/
    # ════════════════════════════════════════
    with g.subgraph(name="cluster_meta") as meta:
        meta.attr(
            label="src/meta_analysis/  —  分析建模層",
            style="rounded,filled",
            fillcolor="#F3E5F5",
            color="#4A148C",
            fontcolor="#4A148C",
            fontsize="13",
        )

        # -- loader/ --
        with meta.subgraph(name="cluster_loader") as ld:
            ld.attr(
                label="loader/",
                style="rounded,filled",
                fillcolor="#D1C4E9",
                color="#311B92",
                fontcolor="#311B92",
                fontsize="11",
            )
            ld.node("ld_embedding", "embedding.py\n載入 .npy + demographics",
                    fillcolor="#EDE7F6", color="#311B92")
            ld.node("ld_au", "au_dataset.py\n載入 AU aggregated CSV",
                    fillcolor="#EDE7F6", color="#311B92")
            ld.node("ld_meta", "meta.py\n載入 fold predictions\n+ age + emotion",
                    fillcolor="#EDE7F6", color="#311B92")
            ld.node("ld_balancer", "balancer.py\nage-stratified balance",
                    fillcolor="#EDE7F6", color="#311B92")

        # -- classifier/ --
        with meta.subgraph(name="cluster_classifier") as cl:
            cl.attr(
                label="classifier/  (Base-level)",
                style="rounded,filled",
                fillcolor="#E1BEE7",
                color="#6A1B9A",
                fontcolor="#6A1B9A",
                fontsize="11",
            )
            cl.node("cl_base", "base.py\n5-fold GroupKFold CV\n遞迴特徵消除",
                    fillcolor="#F3E5F5", color="#6A1B9A")
            cl.node("cl_xgb", "xgboost.py\nfeature importance",
                    fillcolor="#F3E5F5", color="#6A1B9A")
            cl.node("cl_lr", "logistic.py\ncoefficient importance",
                    fillcolor="#F3E5F5", color="#6A1B9A")
            cl.node("cl_tabpfn", "tabpfn.py\npermutation importance",
                    fillcolor="#F3E5F5", color="#6A1B9A")

        # -- stacking/ --
        with meta.subgraph(name="cluster_stacking") as st:
            st.attr(
                label="stacking/  (Meta-level, 14 features)",
                style="rounded,filled",
                fillcolor="#FCE4EC",
                color="#AD1457",
                fontcolor="#AD1457",
                fontsize="11",
            )
            st.node("st_config", "config.py\nMetaConfig",
                    fillcolor="#F8BBD0", color="#AD1457")
            st.node("st_pipeline", "pipeline.py\nmodel × n_features\n組合遍歷",
                    fillcolor="#F8BBD0", color="#AD1457")
            st.node("st_trainer", "trainer.py\nTabPFN Meta-Learner\nfold-aligned stacking",
                    fillcolor="#F8BBD0", color="#AD1457")
            st.node("st_evaluator", "evaluator.py\nAUC · MCC · F1\nSensitivity · Specificity",
                    fillcolor="#F8BBD0", color="#AD1457")

            st.edge("st_config", "st_pipeline")
            st.edge("st_pipeline", "st_trainer")
            st.edge("st_trainer", "st_evaluator")

        # -- evaluation/ --
        with meta.subgraph(name="cluster_eval") as ev:
            ev.attr(
                label="evaluation/",
                style="rounded,filled",
                fillcolor="#ECEFF1",
                color="#37474F",
                fontcolor="#37474F",
                fontsize="11",
            )
            ev.node("ev_shap", "shap_explainer.py\nSHAP 特徵重要性排名",
                    fillcolor="#CFD8DC", color="#37474F")
            ev.node("ev_plotter", "plotter.py\n指標 vs 特徵數趨勢圖",
                    fillcolor="#CFD8DC", color="#37474F")
            ev.node("ev_cross", "cross_tool_comparison.py\n跨工具 AU 一致性\nICC · Pearson",
                    fillcolor="#CFD8DC", color="#37474F")

    # ════════════════════════════════════════
    # 14 Features Box
    # ════════════════════════════════════════
    g.node("feat14",
           "14 Features Integration\n"
           "M1: lr_score_original  ·  M2: lr_score_asymmetry\n"
           "M3: real_age · age_error\n"
           "M4: Anger · Contempt · Disgust · Fear · Happiness\n"
           "      Neutral · Sadness · Surprise · Valence · Arousal",
           shape="note", fillcolor="#FFF9C4", color="#F9A825",
           fontcolor="#F57F17", fontsize="8", fontname=MONO)

    # ════════════════════════════════════════
    # FINAL OUTPUT
    # ════════════════════════════════════════
    g.node("output",
           "Final Output\nreports/ · models/ · plots/\npred_probability/ · training_summary",
           shape="box3d", fillcolor="#B2DFDB", color="#00695C",
           fontcolor="#00695C")

    # ════════════════════════════════════════
    # EDGES — data flow
    # ════════════════════════════════════════

    # Data → extractor
    g.edge("raw_img", "detector", label="JPG")
    g.edge("mirror", "embedding", label="aligned L/R")
    g.edge("mirror", "age", style="dashed", color="#9E9E9E")
    g.edge("mirror", "emotion", style="dashed", color="#9E9E9E")
    g.edge("mirror", "asymmetry", style="dashed", color="#9E9E9E")
    g.edge("mirror", "rotation", style="dashed", color="#9E9E9E")

    # Features → workspace
    g.edge("embedding", "workspace", label=".npy")
    g.edge("age", "workspace", label="JSON")
    g.edge("emotion", "workspace", label="CSV")
    g.edge("asymmetry", "workspace", label="vectors")
    g.edge("rotation", "workspace", label="angles")

    # workspace → loader
    g.edge("workspace", "ld_embedding", label=".npy features")
    g.edge("workspace", "ld_au", label="AU CSV")
    g.edge("demographics", "ld_embedding", label="demographics CSV")
    g.edge("demographics", "ld_meta")

    # loader → classifier
    g.edge("ld_embedding", "cl_base", label="Dataset")
    g.edge("ld_au", "cl_base", label="AU Dataset")
    g.edge("ld_balancer", "cl_base", style="dashed")
    g.edge("cl_base", "cl_xgb", style="dashed", arrowhead="none")
    g.edge("cl_base", "cl_lr", style="dashed", arrowhead="none")
    g.edge("cl_base", "cl_tabpfn", style="dashed", arrowhead="none")

    # classifier → stacking
    g.edge("cl_xgb", "st_pipeline", label="pred_probability\n(fold-aligned CSV)", color="#AD1457")
    g.edge("cl_lr", "st_pipeline", style="dashed", color="#AD1457")

    # External data → stacking
    g.edge("pred_ages", "ld_meta")
    g.edge("emo_scores", "ld_meta")
    g.edge("ld_meta", "st_pipeline", label="14 features")

    # 14 features annotation
    g.edge("feat14", "st_trainer", style="dotted", arrowhead="none", color="#F9A825")

    # stacking → evaluation
    g.edge("st_evaluator", "ev_shap", label="predictions")
    g.edge("st_evaluator", "ev_plotter")

    # classifier → evaluation
    g.edge("cl_xgb", "ev_shap", label="models", style="dashed")
    g.edge("cl_base", "ev_plotter", label="reports", style="dashed")
    g.edge("workspace", "ev_cross", label="AU raw data", style="dashed")

    # → final output
    g.edge("ev_shap", "output")
    g.edge("ev_plotter", "output")
    g.edge("ev_cross", "output")
    g.edge("st_evaluator", "output", label="meta results")

    # ════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════
    g.node("footer",
           "src/common/:  demographics.py · mediapipe_utils.py · metrics.py\n"
           "envs/:  Alz_face_analyze_emo · libreface_env · pyfeat_env\n"
           "external/:  emonet · openface · libreface · DAN · TopoFR · dlib · POSTER_V2 ...",
           shape="plaintext", fontsize="8", fontcolor="#9E9E9E", fontname=MONO)
    g.edge("output", "footer", style="invis")

    # ── Render ──
    out_path = "paper/figures/pipeline_architecture"
    g.render(out_path, cleanup=True)
    # Also render PDF
    g.format = "pdf"
    g.render(out_path, cleanup=True)
    print(f"Saved: {out_path}.png & {out_path}.pdf")


if __name__ == "__main__":
    main()
