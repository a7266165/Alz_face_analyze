"""
SHAP 可解釋性分析

對 XGBoost 模型計算 SHAP values，生成特徵重要性排名
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class AUSHAPExplainer:
    """
    AU-based XGBoost 模型的 SHAP 可解釋性分析

    對每個 fold 的模型計算 SHAP values，
    輸出 per-feature mean |SHAP| 和排名
    """

    def explain_fold(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Dict:
        """
        計算單一 fold 的 SHAP values

        Args:
            model: 訓練好的 XGBoost 模型
            X: 特徵矩陣
            feature_names: 特徵名稱列表

        Returns:
            SHAP 分析結果
        """
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # 計算 mean |SHAP| per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # 排名
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        ranking = [
            {
                "feature": feature_names[i],
                "mean_abs_shap": float(mean_abs_shap[i]),
                "rank": rank + 1,
            }
            for rank, i in enumerate(sorted_idx)
        ]

        return {
            "shap_values": shap_values,
            "mean_abs_shap": dict(zip(feature_names, mean_abs_shap.tolist())),
            "ranking": ranking,
        }

    def explain_all_folds(
        self,
        fold_models: list,
        fold_data: list,
        feature_names: List[str],
    ) -> Dict:
        """
        計算所有 fold 的 SHAP values 並聚合

        Args:
            fold_models: 各 fold 的 XGBoost 模型
            fold_data: 各 fold 的 (X_test, y_test) tuple
            feature_names: 特徵名稱列表

        Returns:
            聚合的 SHAP 分析結果
        """
        all_mean_abs = []

        for fold_idx, (model, (X_test, _)) in enumerate(
            zip(fold_models, fold_data)
        ):
            logger.info(f"計算 Fold {fold_idx + 1} SHAP values...")
            fold_result = self.explain_fold(model, X_test, feature_names)
            all_mean_abs.append(
                [fold_result["mean_abs_shap"][fn] for fn in feature_names]
            )

        # 聚合所有 fold 的 mean |SHAP|
        avg_mean_abs = np.mean(all_mean_abs, axis=0)

        sorted_idx = np.argsort(avg_mean_abs)[::-1]
        ranking = [
            {
                "feature": feature_names[i],
                "mean_abs_shap": float(avg_mean_abs[i]),
                "rank": rank + 1,
            }
            for rank, i in enumerate(sorted_idx)
        ]

        return {
            "mean_abs_shap": dict(zip(feature_names, avg_mean_abs.tolist())),
            "ranking": ranking,
            "n_folds": len(fold_models),
        }

    def save_results(
        self, results: Dict, output_path: Path
    ):
        """儲存 SHAP 分析結果（排除 numpy arrays）"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {
            "mean_abs_shap": results["mean_abs_shap"],
            "ranking": results["ranking"],
            "n_folds": results.get("n_folds", 1),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        logger.info(f"SHAP 結果已儲存: {output_path}")
