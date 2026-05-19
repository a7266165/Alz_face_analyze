"""
Backward-compatibility shim.  All cohort logic has moved to ``src.cohort``.

Existing consumers that ``from scripts.utilities.cohort import X`` will
continue to work unchanged.  New code should import from ``src.cohort``
directly.
"""
import sys
from pathlib import Path

# Ensure PROJECT_ROOT is on sys.path so ``src.cohort`` resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cohort import (  # noqa: F401
    # Constants
    DEMOGRAPHICS_DIR,
    EMBEDDING_DIR,
    LANDMARK_FEATURES_CSV,
    LONGITUDINAL_CSV,
    AD_DELTAS_CSV,
    HC_LONGITUDINAL_CSV,
    EACS_DELTAS_CSV,
    VALID_COHORT_MODES,
    VALID_HC_SOURCE_MODES,
    VALID_DESIGNS,
    EMBEDDING_MODELS,
    EMOTION_METHODS,
    ALL_MODALITIES,
    # Filters
    apply_p_cdr_filter,
    apply_hc_strict_filter,
    apply_max_cdr_filter,
    apply_predicted_age_filter,
    filter_pairs_by_predicted_age,
    apply_followup_filter,
    apply_multivisit_filter,
    # Demographics & visit selection
    load_p_demographics,
    select_visit,
    split_by_metric_median,
    # Feature existence gate
    _ids_with_features,
    _pick_first_visit_with_features,
    _keep_visits_with_features,
    visits_with_features,
    # Matching
    match_1to1,
    # Cohort builders
    build_cohort_ad_vs_HCgroup,
    build_cohort_ad_hi_lo,
)
