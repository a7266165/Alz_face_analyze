"""
Meta Analysis Pipeline

整合執行流程：遍歷所有 model × n_features 組合，
訓練 TabPFN 並儲存結果。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.meta_analysis.config import MetaConfig
from src.meta_analysis.data.loader import MetaDataLoader
from src.meta_analysis.model.evaluator import MetaEvaluator
from src.meta_analysis.model.trainer import TabPFNMetaTrainer, TrainResult

logger = logging.getLogger(__name__)


class MetaPipeline:
    """
    Meta 分析 Pipeline

    遍歷所有 model × n_features 組合，
    訓練 TabPFN meta-model 並儲存結果。
    """

    def __init__(
        self,
        predictions_dir: Path,
        emotion_scores_file: Path,
        output_dir: Path,
        config: Optional[MetaConfig] = None,
    ):
        """
        初始化 Pipeline

        Args:
            predictions_dir: LR 預測分數目錄 (pred_probability)
            emotion_scores_file: emotion_score.csv 檔案路徑
            output_dir: 輸出目錄
            config: MetaConfig 設定 (預設使用預設值)
        """
        self.predictions_dir = Path(predictions_dir)
        self.emotion_scores_file = Path(emotion_scores_file)
        self.output_dir = Path(output_dir)
        self.config = config or MetaConfig()

        # 建立輸出目錄結構
        self.reports_dir = self.output_dir / "reports"
        self.pred_prob_dir = self.output_dir / "pred_probability"

        self._ensure_output_dirs()

        # 發現可用的 n_features
        if self.config.n_features_list is None:
            self.n_features_list = MetaDataLoader.discover_n_features(
                self.predictions_dir
            )
            logger.info(f"自動發現 {len(self.n_features_list)} 個 n_features 層級")
        else:
            self.n_features_list = self.config.n_features_list

    def _ensure_output_dirs(self):
        """建立輸出目錄"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_reports:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_predictions:
            self.pred_prob_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """
        執行完整分析

        遍歷所有 model × n_features 組合。

        Returns:
            彙整結果的 DataFrame
        """
        combinations = self._get_combinations()
        total = len(combinations)

        logger.info(f"開始 Meta Analysis (TabPFN): {total} 個組合")
        logger.info(f"n_features 層級: {len(self.n_features_list)} 個")
        logger.info(f"模型: {self.config.models}")
        logger.info(f"不對稱方法: {self.config.asymmetry_method}")
        logger.info(f"CV 折數: 由 base model 預測檔決定（折疊對齊）")

        all_results = []
        success_count = 0
        fail_count = 0

        for i, (model, n_features) in enumerate(combinations, 1):
            try:
                result = self.run_single(model, n_features)
                all_results.append(result)
                success_count += 1

                logger.info(
                    f"[{i}/{total}] {model}, n_features={n_features}: "
                    f"MCC={result['mcc']:.4f}, Acc={result['accuracy']:.4f}"
                )

            except FileNotFoundError as e:
                logger.warning(
                    f"[{i}/{total}] 跳過 {model}, n_features={n_features}: {e}"
                )
                fail_count += 1
            except Exception as e:
                logger.error(
                    f"[{i}/{total}] 失敗 {model}, n_features={n_features}: {e}"
                )
                fail_count += 1

        logger.info(f"分析完成: 成功 {success_count}, 失敗 {fail_count}")

        # 建立彙整 DataFrame
        summary_df = pd.DataFrame(all_results)
        if not summary_df.empty:
            summary_df = summary_df.sort_values("mcc", ascending=False)

            summary_path = self.output_dir / "summary.csv"
            summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
            logger.info(f"彙整結果已儲存: {summary_path}")

        return summary_df

    def run_single(self, model: str, n_features: int) -> Dict[str, Any]:
        """
        執行單一組合的分析

        Args:
            model: 模型名稱
            n_features: n_features 層級

        Returns:
            結果字典
        """
        # 載入資料 (14 特徵)
        loader = MetaDataLoader(
            predictions_dir=self.predictions_dir,
            emotion_scores_file=self.emotion_scores_file,
            demographics_dir=self.config.demographics_dir,
            predicted_ages_file=self.config.predicted_ages_file,
            n_features=n_features,
            model=model,
            asymmetry_method=self.config.asymmetry_method,
            cdr_threshold=self.config.cdr_threshold,
        )
        dataset = loader.load()

        # 訓練 TabPFN（折疊對齊，折數由 dataset 決定）
        trainer = TabPFNMetaTrainer(
            random_seed=self.config.random_seed,
        )
        train_result = trainer.train(dataset)

        # 儲存結果
        dataset_key = f"{model}_cdr{int(self.config.cdr_threshold)}"
        self._save_results(n_features, dataset_key, train_result, dataset)

        return {
            "model": model,
            "n_features": n_features,
            "cdr_threshold": self.config.cdr_threshold,
            "n_samples": dataset.n_samples,
            "accuracy": train_result.test_metrics["accuracy"],
            "accuracy_std": train_result.test_metrics.get("accuracy_std", 0),
            "mcc": train_result.test_metrics["mcc"],
            "mcc_std": train_result.test_metrics.get("mcc_std", 0),
            "sensitivity": train_result.test_metrics["sensitivity"],
            "specificity": train_result.test_metrics["specificity"],
            "precision": train_result.test_metrics["precision"],
            "recall": train_result.test_metrics["recall"],
            "f1": train_result.test_metrics["f1"],
            "auc": train_result.test_metrics.get("auc"),
            "auc_std": train_result.test_metrics.get("auc_std", 0),
        }

    def _get_combinations(self) -> List[Tuple[str, int]]:
        """取得所有 (model, n_features) 組合"""
        combinations = []
        for model in self.config.models:
            for n_features in self.n_features_list:
                combinations.append((model, n_features))
        return combinations

    def _save_results(
        self,
        n_features: int,
        dataset_key: str,
        train_result: TrainResult,
        dataset: "MetaDataset",
    ):
        """儲存訓練結果"""
        n_features_suffix = f"n_features_{n_features}"

        # 儲存特徵重要性
        if self.config.save_reports:
            report_subdir = self.reports_dir / n_features_suffix
            report_subdir.mkdir(parents=True, exist_ok=True)

            importance_path = report_subdir / f"{dataset_key}_importance.json"
            with open(importance_path, "w", encoding="utf-8") as f:
                json.dump(
                    train_result.feature_importance, f,
                    indent=2, ensure_ascii=False,
                )

        # 儲存預測結果
        if self.config.save_predictions:
            pred_subdir = self.pred_prob_dir / n_features_suffix
            pred_subdir.mkdir(parents=True, exist_ok=True)

            pred_df = train_result.predictions
            test_df = pred_df[pred_df["split"] == "test"].copy()
            train_df = pred_df[pred_df["split"] == "train"].copy()

            if not test_df.empty:
                test_output = test_df[["subject_id", "pred_score", "fold"]]
                test_output = test_output.rename(columns={
                    "subject_id": "個案編號",
                    "pred_score": "預測分數",
                })
                test_path = pred_subdir / f"{dataset_key}_test.csv"
                test_output.to_csv(test_path, index=False, encoding="utf-8-sig")

            if not train_df.empty:
                train_output = train_df[["subject_id", "pred_score", "fold"]]
                train_output = train_output.rename(columns={
                    "subject_id": "個案編號",
                    "pred_score": "預測分數",
                })
                train_output = train_output.sort_values(["fold", "個案編號"])
                train_path = pred_subdir / f"{dataset_key}_train.csv"
                train_output.to_csv(train_path, index=False, encoding="utf-8-sig")

        # 儲存報告
        if self.config.save_reports:
            report_subdir = self.reports_dir / n_features_suffix
            report_subdir.mkdir(parents=True, exist_ok=True)

            report_path = report_subdir / f"{dataset_key}_report.txt"
            self._write_report(report_path, train_result, dataset)

    def _write_report(
        self,
        report_path: Path,
        train_result: TrainResult,
        dataset: "MetaDataset",
    ):
        """撰寫文字報告"""
        n_folds = train_result.metadata["n_folds"]

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"TabPFN Meta Analysis 報告 ({n_folds}-Fold CV)\n")
            f.write("=" * 60 + "\n")
            f.write(f"分析時間: {train_result.metadata['timestamp']}\n")
            f.write(f"n_features (LR): {dataset.metadata['n_features']}\n")
            f.write(f"模型: {dataset.metadata['model']}\n")
            f.write(f"不對稱方法: {dataset.metadata['asymmetry_method']}\n")
            f.write(f"CDR 閾值: {dataset.metadata['cdr_threshold']}\n")
            f.write(f"樣本數: {dataset.n_samples}\n")
            f.write(f"特徵數: {dataset.n_features}\n")
            f.write(f"類別分布: {dataset.class_distribution}\n")

            # 測試集指標
            f.write(f"\n測試集效能 ({n_folds}-Fold 平均):\n")
            f.write("-" * 40 + "\n")
            f.write(MetaEvaluator.format_metrics_report(train_result.test_metrics))
            f.write("\n")

            # 混淆矩陣
            cm = train_result.test_metrics["confusion_matrix"]
            f.write(f"\n測試集混淆矩陣 (累計):\n")
            f.write("-" * 40 + "\n")
            f.write("         真實0  真實1\n")
            f.write(f"預測0   {cm[0][0]:5d}  {cm[1][0]:5d}\n")
            f.write(f"預測1   {cm[0][1]:5d}  {cm[1][1]:5d}\n")

            # 特徵重要性 (permutation importance)
            f.write(f"\n特徵重要性 (Permutation Importance):\n")
            f.write("-" * 40 + "\n")
            sorted_importance = sorted(
                train_result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for name, importance in sorted_importance:
                f.write(f"  {name}: {importance:.4f}\n")

            # 訓練集指標
            f.write(f"\n訓練集效能 ({n_folds}-Fold 平均):\n")
            f.write("-" * 40 + "\n")
            f.write(MetaEvaluator.format_metrics_report(train_result.train_metrics))
            f.write("\n")
