"""Profile a single meta analysis run — time each step."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PROJECT_ROOT

import logging
logging.basicConfig(level=logging.WARNING)

from src.config import META_ANALYSIS_DIR, PREDICTED_AGES_FILE
from src.meta.stacking.config import MetaConfig
from src.meta.loader.meta import MetaDataLoader
from src.meta.stacking.trainer import create_trainer
from src.meta.evaluation.matched_eval import (
    build_matching_cache, run_matched_eval_chain,
)
import re

COHORT = ("p_first", "p_cdrall", "hc_all", "hc_cdrall_or_mmseall")
DEMO = PROJECT_ROOT / "data" / "demographics"
AGES = PREDICTED_AGES_FILE

def timed(label):
    class Timer:
        def __enter__(self):
            self.t = time.perf_counter()
            return self
        def __exit__(self, *a):
            elapsed = time.perf_counter() - self.t
            print(f"  [{elapsed:6.1f}s] {label}")
    return Timer()


print("=== Profiling single run: background/no_bmi/arcface/none/none/raw/tabpfn ===\n")

with timed("build_matching_cache (10 calls)"):
    mc = build_matching_cache(*COHORT)

with timed("MetaDataLoader init + load"):
    loader = MetaDataLoader(
        emb_model="arcface", p_visit=COHORT[0], p_score=COHORT[1],
        hc_visit=COHORT[2], hc_score=COHORT[3], bg_mode="background",
        photo_mode="mean", reducer="no_drop",
        base_classifier="logistic", base_classifier_param="C_1",
        direction="fwd", eval_method="1by1matched",
        match_level="subject_match", eval_unit="eval_by_subject",
        match_strategy="priority_acs", partition="ad_vs_hc",
        asymmetry_variant="none", scoring_method="none",
        demographics_dir=DEMO, predicted_ages_file=AGES,
    )
    dataset = loader.load()
print(f"    → {dataset.n_samples} samples, {dataset.n_features} features")

for clf_name in ["logistic", "xgboost"]:
    with timed(f"train 10-fold ({clf_name})"):
        trainer = create_trainer(clf_name, random_seed=42)
        result = trainer.train(dataset)
    print(f"    → MCC={result.test_metrics['mcc']:.4f}")

print(f"\n--- Eval chain profiling (using logistic result) ---")
trainer = create_trainer("logistic", random_seed=42)
result = trainer.train(dataset)

oof = result.predictions[result.predictions["split"] == "test"].copy()
oof = oof.rename(columns={"pred_score": "y_score"})
oof["base_id"] = oof["subject_id"].apply(
    lambda s: re.match(r"^([A-Za-z]+\d+)", s).group(1)
    if re.match(r"^([A-Za-z]+\d+)", s) else s
)
oof["y_true"] = oof["base_id"].apply(lambda b: 1 if b.startswith("P") else 0)

out = Path("c:/tmp/meta_profile_test")

with timed("run_matched_eval_chain (1by1matched/subject_match only)"):
    run_matched_eval_chain(
        oof_scores=oof, matching_cache=mc,
        output_dir=out, seed=42,
        meta_info={"emb_model": "arcface", "meta_classifier": "tabpfn"},
    )

print("\n=== Done ===")
