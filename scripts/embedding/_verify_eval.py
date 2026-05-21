"""Verify _forward_eval fix: ACS eval pairs should increase significantly."""
import sys
sys.path.insert(0, ".")
import scripts.embedding.run_fwd_rev as mod

mod._MATCH_PRIORITY = ["ACS", "NAD"]
mod._COHORT_MODE = "p_all_cdrall_hc_all_cdrall_or_mmseall"
mod._FEATURE_TYPE = "original_background"

full, matched, pairs, keep_groups = mod.build_partition_cohort("ad_vs_acs")
train = mod._forward_train(full, "arcface", "logistic", 10, 42)
oof_df, m_matched, paired, matched_eval = mod._forward_eval(train, matched, keep_groups, 42)

print(f"ad_vs_acs: n_pairs={paired['n_pairs']}, n={m_matched['n']}, "
      f"AUC={m_matched['auc']:.3f}, p={paired['p']:.4e}")
print(f"  groups: {matched_eval['group'].value_counts().to_dict()}")

full2, matched2, pairs2, kg2 = mod.build_partition_cohort("ad_vs_hc")
oof_df2, m2, paired2, me2 = mod._forward_eval(train, matched2, kg2, 42)
print(f"ad_vs_hc:  n_pairs={paired2['n_pairs']}, n={m2['n']}, AUC={m2['auc']:.3f}")
