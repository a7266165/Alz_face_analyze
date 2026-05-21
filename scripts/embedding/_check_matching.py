"""Quick check: P vs ACS direct matching vs current P vs HC-group,
including priority_groups ordering."""
import sys
sys.path.insert(0, ".")
from src.cohort import build_cohort_ad_vs_HCgroup, filter_pairs_by_predicted_age

CM = "p_all_cdrall_hc_all_cdrall_or_mmseall"

def _acs_subjects(pairs):
    ids = set()
    for col in ("minor_id", "major_id"):
        ids |= set(
            pairs[pairs[col].str.startswith("ACS")][col]
            .str.replace(r"-\d+$", "", regex=True)
        )
    return ids

def _run(label, hc_source, priority_groups=None):
    matched, pairs = build_cohort_ad_vs_HCgroup(
        hc_source, design="cross_matched", cohort_mode=CM,
        hc_source_mode="ACS", priority_groups=priority_groups,
    )
    matched, pairs = filter_pairs_by_predicted_age(matched, pairs)
    n_total = pairs["pair_id"].nunique()
    n_acs_subj = len(_acs_subjects(pairs))
    acs_pairs = pairs[
        pairs["minor_id"].str.startswith("ACS")
        | pairs["major_id"].str.startswith("ACS")
    ]
    n_acs_pairs = acs_pairs["pair_id"].nunique()
    print(f"{label}: total pairs={n_total}, "
          f"ACS-involved pairs={n_acs_pairs}, unique ACS subjects={n_acs_subj}")
    if "match_pass" in pairs.columns:
        print("  match_pass (ACS only):",
              acs_pairs["match_pass"].value_counts().to_dict())
    return pairs

print("=" * 60)
_run("Random      (P vs HC-group)", "HC", priority_groups=None)
print()
_run("ACS first   (P vs HC-group)", "HC", priority_groups=["ACS", "NAD"])
print()
_run("NAD first   (P vs HC-group)", "HC", priority_groups=["NAD", "ACS"])
print()
_run("Direct      (P vs ACS only)", "ACS")
print()
_run("Direct      (P vs NAD only)", "NAD")
print("=" * 60)
