"""
Meta Analysis Pipeline (v2)

遍歷 emb_model × meta_classifier 組合，
訓練 meta-model 並儲存結果。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.meta.stacking.config import MetaConfig
from src.meta.loader.meta import MetaDataLoader
from src.meta.stacking.evaluator import MetaEvaluator
from src.meta.stacking.trainer import create_trainer, TrainResult
from src.meta.evaluation.matched_eval import run_matched_eval_chain

logger = logging.getLogger(__name__)


class MetaPipeline:
    """
    Meta 分析 Pipeline (v2)

    遍歷 emb_model × meta_classifier 組合，
    訓練 meta-model 並儲存結果。
    """

    def __init__(
        self,
        output_dir: Path,
        config: Optional[MetaConfig] = None,
        asymmetry_variant: str = "absolute_relative_differences",
    ):
        self.output_dir = Path(output_dir)
        self.config = config or MetaConfig()
        self.asymmetry_variant = asymmetry_variant

        self.reports_dir = self.output_dir / "reports"
        self.pred_prob_dir = self.output_dir / "pred_probability"
        self._ensure_output_dirs()

    def _ensure_output_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_reports:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_predictions:
            self.pred_prob_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        combinations = self._get_combinations()
        total = len(combinations)

        logger.info(f"開始 Meta Analysis: {total} 個組合")
        logger.info(f"模型: {self.config.models}")
        logger.info(f"Classifiers: {self.config.meta_classifiers}")
        logger.info(f"Emotion method: {self.config.emotion_method}")

        all_results = []
        success_count = 0
        fail_count = 0

        for i, (emb_model, meta_clf) in enumerate(combinations, 1):
            try:
                result = self.run_single(emb_model, meta_clf)
                all_results.append(result)
                success_count += 1

                logger.info(
                    f"[{i}/{total}] {emb_model} + {meta_clf}: "
                    f"MCC={result['mcc']:.4f}, Acc={result['accuracy']:.4f}"
                )

            except FileNotFoundError as e:
                logger.warning(f"[{i}/{total}] 跳過 {emb_model} + {meta_clf}: {e}")
                fail_count += 1
            except Exception as e:
                logger.error(f"[{i}/{total}] 失敗 {emb_model} + {meta_clf}: {e}")
                fail_count += 1

        logger.info(f"分析完成: 成功 {success_count}, 失敗 {fail_count}")

        summary_df = pd.DataFrame(all_results)
        if not summary_df.empty:
            summary_df = summary_df.sort_values("mcc", ascending=False)
            summary_path = self.output_dir / "summary.csv"
            summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
            logger.info(f"彙整結果已儲存: {summary_path}")

        return summary_df

    def run_single(self, emb_model: str, meta_clf_name: str) -> Dict[str, Any]:
        loader = MetaDataLoader(
            emb_model=emb_model,
            cohort_mode=self.config.cohort_mode,
            bg_mode=self.config.bg_mode,
            photo_mode=self.config.photo_mode,
            reducer=self.config.reducer,
            base_classifier=self.config.base_classifier,
            base_classifier_param=self.config.base_classifier_param,
            direction=self.config.direction,
            eval_method=self.config.eval_method,
            match_level=self.config.match_level,
            eval_unit=self.config.eval_unit,
            match_strategy=self.config.match_strategy,
            partition=self.config.partition,
            emotion_method=self.config.emotion_method,
            asymmetry_variant=self.asymmetry_variant,
            emotion_features_dir=self.config.emotion_features_dir,
            emotion_schema_file=self.config.emotion_schema_file,
            emonet_csv=self.config.emonet_csv,
            demographics_dir=self.config.demographics_dir,
            predicted_ages_file=self.config.predicted_ages_file,
        )
        dataset = loader.load()

        trainer = create_trainer(meta_clf_name, self.config.random_seed,
                                normalize=self.config.normalize)
        train_result = trainer.train(dataset)

        self._save_results(emb_model, meta_clf_name, train_result, dataset)

        # Run matched eval chain
        if self.config.demographics_dir:
            oof_for_eval = train_result.predictions[
                train_result.predictions["split"] == "test"
            ].copy()
            oof_for_eval = oof_for_eval.rename(columns={"pred_score": "y_score"})
            import re
            oof_for_eval["base_id"] = oof_for_eval["subject_id"].apply(
                lambda s: re.match(r"^([A-Za-z]+\d+)", s).group(1)
                if re.match(r"^([A-Za-z]+\d+)", s) else s
            )
            oof_for_eval["y_true"] = oof_for_eval["base_id"].apply(
                lambda b: 1 if b.startswith("P") else 0
            )

            eval_dir = self.output_dir
            try:
                run_matched_eval_chain(
                    oof_scores=oof_for_eval,
                    demographics_dir=self.config.demographics_dir,
                    cohort_mode=self.config.cohort_mode,
                    output_dir=eval_dir,
                    seed=self.config.random_seed,
                    meta_info={
                        "emb_model": emb_model,
                        "meta_classifier": meta_clf_name,
                    },
                )
                logger.info(f"  Eval chain 完成: {eval_dir}")
            except Exception as e:
                logger.warning(f"  Eval chain 失敗: {e}")

        return {
            "emb_model": emb_model,
            "meta_classifier": meta_clf_name,
            "reducer": self.config.reducer,
            "emotion_method": self.config.emotion_method,
            "n_meta_features": dataset.n_features,
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

    def _get_combinations(self) -> List[Tuple[str, str]]:
        combos = []
        for model in self.config.models:
            for clf in self.config.meta_classifiers:
                combos.append((model, clf))
        return combos

    def _save_results(
        self,
        emb_model: str,
        meta_clf_name: str,
        train_result: TrainResult,
        dataset: "MetaDataset",
    ):
        sub_key = f"{emb_model}_{meta_clf_name}"

        if self.config.save_reports:
            report_subdir = self.reports_dir / sub_key
            report_subdir.mkdir(parents=True, exist_ok=True)

            importance_path = report_subdir / "importance.json"
            with open(importance_path, "w", encoding="utf-8") as f:
                json.dump(train_result.feature_importance, f, indent=2, ensure_ascii=False)

            report_path = report_subdir / "report.txt"
            self._write_report(report_path, train_result, dataset, meta_clf_name)

        if self.config.save_predictions:
            pred_subdir = self.pred_prob_dir / sub_key
            pred_subdir.mkdir(parents=True, exist_ok=True)

            pred_df = train_result.predictions
            test_df = pred_df[pred_df["split"] == "test"].copy()
            train_df = pred_df[pred_df["split"] == "train"].copy()

            if not test_df.empty:
                out = test_df[["subject_id", "pred_score", "fold"]]
                out.to_csv(pred_subdir / "test.csv", index=False, encoding="utf-8-sig")

            if not train_df.empty:
                out = train_df[["subject_id", "pred_score", "fold"]]
                out = out.sort_values(["fold", "subject_id"])
                out.to_csv(pred_subdir / "train.csv", index=False, encoding="utf-8-sig")

    def _write_report(
        self,
        report_path: Path,
        train_result: TrainResult,
        dataset: "MetaDataset",
        meta_clf_name: str,
    ):
        n_folds = train_result.metadata["n_folds"]

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Meta Analysis 報告 ({n_folds}-Fold CV)\n")
            f.write("=" * 60 + "\n")
            f.write(f"分析時間: {train_result.metadata['timestamp']}\n")
            f.write(f"Embedding 模型: {dataset.metadata['emb_model']}\n")
            f.write(f"Meta Classifier: {meta_clf_name}\n")
            f.write(f"Reducer: {dataset.metadata['reducer']}\n")
            f.write(f"Emotion method: {dataset.metadata['emotion_method']}\n")
            f.write(f"Partition: {dataset.metadata['partition']}\n")
            f.write(f"樣本數: {dataset.n_samples}\n")
            f.write(f"Meta 特徵數: {dataset.n_features} ({', '.join(dataset.feature_names)})\n")
            f.write(f"類別分布: {dataset.class_distribution}\n")

            f.write(f"\n測試集效能 ({n_folds}-Fold 平均):\n")
            f.write("-" * 40 + "\n")
            f.write(MetaEvaluator.format_metrics_report(train_result.test_metrics))
            f.write("\n")

            cm = train_result.test_metrics["confusion_matrix"]
            f.write(f"\n測試集混淆矩陣 (累計):\n")
            f.write("-" * 40 + "\n")
            f.write("         真實0  真實1\n")
            f.write(f"預測0   {cm[0][0]:5d}  {cm[1][0]:5d}\n")
            f.write(f"預測1   {cm[0][1]:5d}  {cm[1][1]:5d}\n")

            f.write(f"\n特徵重要性 (Permutation Importance):\n")
            f.write("-" * 40 + "\n")
            sorted_imp = sorted(
                train_result.feature_importance.items(),
                key=lambda x: x[1], reverse=True,
            )
            for name, imp in sorted_imp:
                f.write(f"  {name}: {imp:.4f}\n")

            f.write(f"\n訓練集效能 ({n_folds}-Fold 平均):\n")
            f.write("-" * 40 + "\n")
            f.write(MetaEvaluator.format_metrics_report(train_result.train_metrics))
            f.write("\n")
