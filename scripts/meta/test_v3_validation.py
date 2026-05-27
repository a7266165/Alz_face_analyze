"""Quick validation: 3 test runs for v3 pipeline."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import PROJECT_ROOT

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from src.config import META_ANALYSIS_DIR, cohort_name, cohort_spec_from_name
from src.meta import MetaConfig, MetaPipeline

COHORT = "p_first_cdrall_hc_all_cdrall_or_mmseall"
spec = cohort_spec_from_name(cohort_name(COHORT))
DEMO = PROJECT_ROOT / "data" / "demographics"
AGES = (PROJECT_ROOT / "workspace" / "age" / "analysis"
        / spec.visit_dir / spec.cdr_mmse_dir
        / "correction" / "calibration" / "predicted_ages_calibrated.json")

tests = [
    ("background", [], "none", "none", "3-feat baseline"),
    ("background", [], "difference", "l2_norm", "4-feat L2 asym"),
    ("background", ["bmi"], "none", "none", "4-feat BMI"),
]

for bg, extra, asym, scoring, label in tests:
    ef_tag = "with_bmi" if "bmi" in extra else "no_bmi"
    out = (META_ANALYSIS_DIR / spec.visit_dir / spec.cdr_mmse_dir
           / bg / ef_tag / "arcface" / asym / scoring
           / "mean" / "no_drop" / "fwd" / "raw" / "tabpfn")
    print(f"\n{'='*60}")
    print(f"  {label} -> {out}")
    print(f"{'='*60}")
    cfg = MetaConfig(
        cohort_mode=COHORT, bg_mode=bg,
        match_strategy="priority_acs",
        scoring_method=scoring, extra_features=extra,
        models=["arcface"], meta_classifiers=["tabpfn"],
        demographics_dir=DEMO, predicted_ages_file=AGES,
    )
    p = MetaPipeline(output_dir=out, config=cfg, asymmetry_variant=asym)
    try:
        df = p.run()
        if not df.empty:
            print(f"  n_features={df['n_meta_features'].iloc[0]}, "
                  f"MCC={df['mcc'].iloc[0]:.4f}, AUC={df['auc'].iloc[0]:.4f}")
        else:
            print("  FAILED: empty result")
    except Exception as e:
        print(f"  ERROR: {e}")
