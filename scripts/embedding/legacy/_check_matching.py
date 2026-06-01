"""Quick check: P vs ACS direct matching vs current P vs HC-group,
including priority_groups ordering."""
import sys
sys.path.insert(0, ".")
from src.common.matching import match_by_age
from src.config import cohort_spec_from_name

CM = "p_all_cdrall_hc_all_cdrall_or_mmseall"


def _tokens():
    spec = cohort_spec_from_name(CM)
    return (f"p_{spec.p_visit}", f"p_{spec.p_cdr}", f"hc_{spec.hc_visit}",
            "hc_cdr0_or_mmse26" if spec.hc_strict else "hc_cdrall_or_mmseall")


def _run(label, controls, priority_groups=None):
    # match_by_age 回 1:1 index 對齊的兩 list → 第 i 對 = (p_ids[i], hc_ids[i])。
    p_ids, hc_ids = match_by_age(*_tokens(), controls=controls,
                                 priority=priority_groups)
    pairs = list(zip(p_ids, hc_ids))
    acs_pairs = [(p, h) for p, h in pairs if h.startswith("ACS")]
    acs_subj = {h.rsplit("-", 1)[0] for _, h in acs_pairs}
    print(f"{label}: total pairs={len(pairs)}, "
          f"ACS-involved pairs={len(acs_pairs)}, "
          f"unique ACS subjects={len(acs_subj)}")


print("=" * 60)
_run("Random      (P vs HC-group)", None, priority_groups=None)
print()
_run("ACS first   (P vs HC-group)", None, priority_groups=["ACS", "NAD"])
print()
_run("NAD first   (P vs HC-group)", None, priority_groups=["NAD", "ACS"])
print()
_run("Direct      (P vs ACS only)", ["ACS"])
print()
_run("Direct      (P vs NAD only)", ["NAD"])
print("=" * 60)
