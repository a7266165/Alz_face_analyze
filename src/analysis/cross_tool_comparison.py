"""
跨工具比較分析

比較 OpenFace、Py-Feat、LibreFace 的 AU 判讀一致性與分類效能差異
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from itertools import combinations

import numpy as np
import pandas as pd

from src.modules.emotion.extractor.au_config import (
    HARMONIZED_AUS,
    HARMONIZED_EMOTIONS,
    AU_HARMONIZED_DIR,
    AU_AGGREGATED_DIR,
)

logger = logging.getLogger(__name__)

TOOLS = ["openface", "pyfeat", "libreface"]


class CrossToolComparator:
    """
    跨工具比較分析器

    計算 AU 一致性（ICC）、情緒一致性（Kappa/Pearson）、分類效能差異（DeLong）
    """

    def __init__(
        self,
        harmonized_dir: Optional[Path] = None,
        aggregated_dir: Optional[Path] = None,
    ):
        self.harmonized_dir = harmonized_dir or AU_HARMONIZED_DIR
        self.aggregated_dir = aggregated_dir or AU_AGGREGATED_DIR

    # ========== AU 一致性分析 ==========

    def compute_au_agreement(
        self, tools: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        計算各 AU 在不同工具間的一致性（ICC）

        對每個 AU，先取各受試者在各工具的 mean 值，
        再計算跨工具的 ICC(2,1)

        Returns:
            DataFrame: AU × (ICC, p-value, tool_pair)
        """
        import pingouin as pg

        if tools is None:
            tools = TOOLS

        tool_dfs = {}
        for tool in tools:
            path = self.aggregated_dir / f"{tool}_harmonized.csv"
            if path.exists():
                tool_dfs[tool] = pd.read_csv(path).set_index("subject_id")
            else:
                logger.warning(f"找不到 {tool} 的 aggregated 檔案")

        if len(tool_dfs) < 2:
            logger.error("需要至少 2 個工具的資料才能比較")
            return pd.DataFrame()

        common_subjects = set.intersection(
            *[set(df.index) for df in tool_dfs.values()]
        )
        logger.info(f"共同受試者: {len(common_subjects)}")

        if not common_subjects:
            logger.error("沒有共同受試者")
            return pd.DataFrame()

        results = []

        for au in HARMONIZED_AUS:
            mean_col = f"{au}_mean"

            available_tools = [
                t for t in tool_dfs
                if mean_col in tool_dfs[t].columns
            ]

            if len(available_tools) < 2:
                continue

            rows = []
            for subject in sorted(common_subjects):
                for tool in available_tools:
                    val = tool_dfs[tool].loc[subject, mean_col]
                    if not np.isnan(val):
                        rows.append({
                            "subject": subject,
                            "rater": tool,
                            "score": val,
                        })

            if not rows:
                continue

            long_df = pd.DataFrame(rows)

            try:
                icc_result = pg.intraclass_corr(
                    data=long_df,
                    targets="subject",
                    raters="rater",
                    ratings="score",
                )
                icc_row = icc_result[icc_result["Type"] == "ICC2"]
                if not icc_row.empty:
                    results.append({
                        "AU": au,
                        "ICC": float(icc_row["ICC"].values[0]),
                        "CI95_low": float(icc_row["CI95%"].values[0][0])
                            if isinstance(icc_row["CI95%"].values[0], (list, tuple))
                            else np.nan,
                        "CI95_high": float(icc_row["CI95%"].values[0][1])
                            if isinstance(icc_row["CI95%"].values[0], (list, tuple))
                            else np.nan,
                        "pval": float(icc_row["pval"].values[0]),
                        "n_tools": len(available_tools),
                        "n_subjects": len(set(long_df["subject"])),
                    })
            except Exception as e:
                logger.warning(f"AU {au} ICC 計算失敗: {e}")

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df = result_df.sort_values("ICC", ascending=False)
        return result_df

    # ========== 情緒一致性分析 ==========

    def compute_emotion_consistency(
        self, tools: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        計算情緒預測在不同工具間的一致性

        使用 Pearson r 比較連續情緒機率值

        Returns:
            DataFrame: emotion × tool_pair × (pearson_r, p_value)
        """
        from scipy.stats import pearsonr

        if tools is None:
            tools = [t for t in TOOLS if t != "openface"]

        tool_dfs = {}
        for tool in tools:
            path = self.aggregated_dir / f"{tool}_harmonized.csv"
            if path.exists():
                tool_dfs[tool] = pd.read_csv(path).set_index("subject_id")

        if len(tool_dfs) < 2:
            logger.error("需要至少 2 個工具的資料")
            return pd.DataFrame()

        common_subjects = set.intersection(
            *[set(df.index) for df in tool_dfs.values()]
        )

        results = []

        for emotion in HARMONIZED_EMOTIONS:
            mean_col = f"{emotion}_mean"

            for tool_a, tool_b in combinations(tool_dfs.keys(), 2):
                if mean_col not in tool_dfs[tool_a].columns:
                    continue
                if mean_col not in tool_dfs[tool_b].columns:
                    continue

                vals_a = []
                vals_b = []
                for subject in common_subjects:
                    va = tool_dfs[tool_a].loc[subject, mean_col]
                    vb = tool_dfs[tool_b].loc[subject, mean_col]
                    if not np.isnan(va) and not np.isnan(vb):
                        vals_a.append(va)
                        vals_b.append(vb)

                if len(vals_a) < 3:
                    continue

                r, p = pearsonr(vals_a, vals_b)
                results.append({
                    "emotion": emotion,
                    "tool_a": tool_a,
                    "tool_b": tool_b,
                    "pearson_r": float(r),
                    "p_value": float(p),
                    "n_subjects": len(vals_a),
                })

        return pd.DataFrame(results)

    # ========== 分類效能比較 ==========

    def compare_classification_auc(
        self, results_dir: Path
    ) -> pd.DataFrame:
        """
        比較各工具的分類 AUC

        從 reports 目錄讀取各工具的分類結果

        Args:
            results_dir: 分類報告目錄

        Returns:
            DataFrame: tool × feature_set × metrics
        """
        comparison = []

        for tool in TOOLS:
            for fs in ["harmonized", "extended"]:
                report_key = f"au_{tool}_{fs}_cdr0"
                report_pattern = f"*/{report_key}_report.txt"
                report_files = list(results_dir.glob(report_pattern))

                if not report_files:
                    continue

                report_file = sorted(report_files)[-1]
                metrics = self._parse_report(report_file)

                if metrics:
                    comparison.append({
                        "tool": tool,
                        "feature_set": fs,
                        **metrics,
                    })

        return pd.DataFrame(comparison)

    def _parse_report(self, report_path: Path) -> Optional[Dict]:
        """從報告文字檔解析指標"""
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()

            metrics = {}
            for line in content.split("\n"):
                line = line.strip()
                for metric in ["accuracy", "mcc", "sensitivity", "specificity", "auc"]:
                    if line.startswith(f"{metric}:"):
                        try:
                            metrics[metric] = float(line.split(":")[1].strip())
                        except ValueError:
                            pass

            return metrics if metrics else None
        except Exception:
            return None

    # ========== 結果輸出 ==========

    def generate_comparison_report(
        self, output_dir: Path
    ) -> Dict:
        """
        生成完整的跨工具比較報告

        Args:
            output_dir: 輸出目錄

        Returns:
            報告結果字典
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {}

        logger.info("計算 AU 一致性 (ICC)...")
        au_agreement = self.compute_au_agreement()
        if not au_agreement.empty:
            au_agreement.to_csv(
                output_dir / "au_agreement_icc.csv",
                index=False,
                encoding="utf-8-sig",
            )
            report["au_agreement"] = au_agreement.to_dict("records")
            logger.info(f"AU ICC 結果:\n{au_agreement.to_string()}")

        logger.info("計算情緒一致性 (Pearson r)...")
        emotion_consistency = self.compute_emotion_consistency()
        if not emotion_consistency.empty:
            emotion_consistency.to_csv(
                output_dir / "emotion_consistency_pearson.csv",
                index=False,
                encoding="utf-8-sig",
            )
            report["emotion_consistency"] = emotion_consistency.to_dict("records")
            logger.info(f"情緒 Pearson r 結果:\n{emotion_consistency.to_string()}")

        with open(output_dir / "cross_tool_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"跨工具比較報告已儲存至: {output_dir}")
        return report
