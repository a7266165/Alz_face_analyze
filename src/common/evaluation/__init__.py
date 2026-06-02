"""共用評估層 —— 吃 oof_scores.csv + cohort,純算指標(不畫圖、不碰 modality 路徑)。

兩支:
  - metrics.py      純 kernel:compute_clf_metrics / bootstrap_auc_ci / paired_wilcoxon。
  - matched_chain.py 依方向取對的人(fwd: all+1by1 / rev: 1by1+other)→ long-format 指標。

modality-agnostic:embedding / meta / 任何有 [ID,y_true,y_score,fold] 的 OOF 都能用。
"""
from .metrics import bootstrap_auc_ci, compute_clf_metrics, paired_wilcoxon
from .matched_chain import (
    evaluate,
    evaluate_oof_file,
    AD_PARTITIONS,
    MATCH_STRATEGIES,
    EVAL_UNITS,
    PARTITION_KEEP_GROUPS,
)

__all__ = [
    "bootstrap_auc_ci",
    "compute_clf_metrics",
    "paired_wilcoxon",
    "evaluate",
    "evaluate_oof_file",
    "AD_PARTITIONS",
    "MATCH_STRATEGIES",
    "EVAL_UNITS",
    "PARTITION_KEEP_GROUPS",
]
