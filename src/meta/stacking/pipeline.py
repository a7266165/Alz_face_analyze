"""
Meta Analysis Pipeline (v3)

遍歷 emb_model × meta_classifier 組合，
訓練 meta-model 並儲存結果。
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.meta.stacking.config import MetaConfig
from src.meta.loader.meta import MetaDataLoader
from src.meta.stacking.trainer import create_trainer, TrainResult
from src.meta.evaluation.matched_eval import run_matched_eval_chain

logger = logging.getLogger(__name__)


class MetaPipeline:

    def __init__(
        self,
        output_dir: Path,
        config: Optional[MetaConfig] = None,
        asymmetry_variant: str = "none",
        matching_cache: Optional[dict] = None,
    ):
        self.output_dir = Path(output_dir)
        self.config = config or MetaConfig()
        self.asymmetry_variant = asymmetry_variant
        self.matching_cache = matching_cache
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
            asymmetry_variant=self.asymmetry_variant,
            scoring_method=self.config.scoring_method,
            extra_features=self.config.extra_features,
            emotion_method=self.config.emotion_method,
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

        if self.matching_cache:
            self._run_eval_chain(emb_model, meta_clf_name, train_result)

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

    def _run_eval_chain(self, emb_model, meta_clf_name, train_result):
        oof_for_eval = train_result.predictions[
            train_result.predictions["split"] == "test"
        ].copy()
        oof_for_eval = oof_for_eval.rename(columns={"pred_score": "y_score"})
        oof_for_eval["base_id"] = oof_for_eval["subject_id"].apply(
            lambda s: re.match(r"^([A-Za-z]+\d+)", s).group(1)
            if re.match(r"^([A-Za-z]+\d+)", s) else s
        )
        oof_for_eval["y_true"] = oof_for_eval["base_id"].apply(
            lambda b: 1 if b.startswith("P") else 0
        )
        try:
            run_matched_eval_chain(
                oof_scores=oof_for_eval,
                matching_cache=self.matching_cache,
                output_dir=self.output_dir,
                seed=self.config.random_seed,
                meta_info={
                    "emb_model": emb_model,
                    "meta_classifier": meta_clf_name,
                },
            )
            logger.info(f"  Eval chain 完成: {self.output_dir}")
        except Exception as e:
            logger.warning(f"  Eval chain 失敗: {e}")

    def _get_combinations(self) -> List[Tuple[str, str]]:
        combos = []
        for model in self.config.models:
            for clf in self.config.meta_classifiers:
                combos.append((model, clf))
        return combos
