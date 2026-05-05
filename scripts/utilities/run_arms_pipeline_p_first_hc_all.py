"""
Run the cross-sectional analyses with the cohort:
    P side : first-visit + Global_CDR>=0.5 + .npy fallback
    HC side: ALL NAD + ALL ACS visits (no strict HC filter)

Outputs fan out across the modality-flat workspace layout:
    age/, emo_au/, asymmetry/, embedding/  — per-modality feature_stat / classification
    overview/                              — cross-modality stat grid (run_4arm_deep_dive)
    overview/<cohort>/{cross_naive, cross_matched, stat_grid}/  — matching artifacts + grid
                                                              + summary_per_modality
                                                              + classifier_summary_all

Sequentially invokes:
    1. run_arm_a_ad_vs_hc.py                      (cross_naive)
    2. run_cross_matched.py --comparison ad_vs_{hc,nad,acs}    (cross_matched)
    3. run_cross_matched.py --comparison {mmse,casi}_hilo      (cross_matched hi-lo)
    4. run_arm_b_auc_supplement.py × {MMSE, CASI}              (auc supplement)
    5. run_4arm_deep_dive.py --arms A B --hc-source ACS        (overview/)
    6. run_arm_age_classifiers.py --arms A B                   (age classifiers)

Then writes cohort_summary.csv + README.md describing the cohort.

Usage:
    conda run -n Alz_face_main_analysis \\
        python scripts/utilities/run_arms_pipeline_p_first_hc_all.py
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    AGE_PRED_ERROR_STAT_DIR, OVERVIEW_DIR, cohort_name,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PYTHON = sys.executable

COHORT_MODE = "p_first_hc_all"
COHORT_DIR = cohort_name(COHORT_MODE)
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "experiments"


def run_step(name, cmd, env=None):
    logger.info(f"\n=== {name} ===")
    logger.info(f"  cmd: {' '.join(str(c) for c in cmd)}")
    if env:
        env_show = {k: env[k] for k in env if k in (
            "HILO_METRIC", "COHORT_MODE", "HC_SOURCE_MODE", "ARMS")}
        if env_show:
            logger.info(f"  env: {env_show}")
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(PROJECT_ROOT),
                   env=full_env)


def write_cohort_summary(output_dir):
    """Compute cohort summary stats from the matched_features.csv files."""
    rows = []

    # cross_naive cohort (was arm_a)
    cn_csv = OVERVIEW_DIR / COHORT_DIR / "cross_naive" / "cohort.csv"
    if cn_csv.exists():
        df = pd.read_csv(cn_csv)
        for label, name in [(1, "cross_naive AD"), (0, "cross_naive HC")]:
            sub = df[df["label"] == label]
            rows.append({
                "stage": "cross_naive/ad_vs_hc",
                "group": name,
                "n_rows": len(sub),
                "n_subjects": sub["base_id"].nunique() if "base_id" in sub.columns else len(sub),
                "age_mean": round(sub["Age"].mean(), 2) if len(sub) else "—",
                "age_std": round(sub["Age"].std(), 2) if len(sub) > 1 else "—",
            })

    # cross_matched per comparison (was arm_b)
    cm_root = OVERVIEW_DIR / COHORT_DIR / "cross_matched"
    for cmp in ("hc", "nad", "acs"):
        f = cm_root / f"ad_vs_{cmp}" / "matched_features.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for label, name in [(1, f"cross_matched AD vs {cmp.upper()} (AD)"),
                              (0, f"cross_matched AD vs {cmp.upper()} (HC)")]:
            sub = df[df["label"] == label]
            rows.append({
                "stage": f"cross_matched/ad_vs_{cmp}",
                "group": name,
                "n_rows": len(sub),
                "n_subjects": sub["base_id"].nunique() if "base_id" in sub.columns else len(sub),
                "age_mean": round(sub["Age"].mean(), 2) if len(sub) else "—",
                "age_std": round(sub["Age"].std(), 2) if len(sub) > 1 else "—",
            })

    # cross_matched hi-lo
    for metric in ("mmse", "casi"):
        f = cm_root / f"{metric}_high_vs_low" / "matched_features.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        gcol = f"{metric}_group"
        if gcol not in df.columns:
            continue
        for grp_val in ("high", "low"):
            sub = df[df[gcol] == grp_val]
            rows.append({
                "stage": f"cross_matched/{metric}_high_vs_low",
                "group": f"{metric.upper()} {grp_val}",
                "n_rows": len(sub),
                "n_subjects": sub["ID"].astype(str).str.extract(
                    r"^(.+)-\d+$").iloc[:, 0].nunique() if len(sub) else 0,
                "age_mean": round(sub["Age"].mean(), 2) if len(sub) else "—",
                "age_std": round(sub["Age"].std(), 2) if len(sub) > 1 else "—",
            })

    df_summary = pd.DataFrame(rows)
    csv = output_dir / "cohort_summary.csv"
    df_summary.to_csv(csv, index=False, encoding="utf-8-sig")
    logger.info(f"Wrote {csv}")
    return df_summary


def write_readme(output_dir, df_summary):
    readme = output_dir / "README.md"
    text = f"""# cohort: first-visit P + ALL NAD/ACS

## Cohort definition

| Side | Filter |
|---|---|
| **P (AD)** | First-visit + `Global_CDR >= 0.5` + `.npy fallback` |
| **NAD / ACS (HC)** | **No strict HC filter** — every visit kept (CDR=0 / MMSE>=26 not required, no first-visit pick per HC subject) |

Differs from the strict cohort `p_first_hc_strict`, which uses strict HC
(CDR=0 OR (CDR=NaN AND MMSE>=26)) AND first-visit per HC subject.

## Cohort summary

{df_summary.to_markdown(index=False)}

## Output fan-out

```
overview/{COHORT_DIR}/
├── README.md                          (this file)
├── cohort_summary.csv                 (per-stage / per-group N + age stats)
├── classifier_summary_all.csv         (run_arm_age_classifiers grand summary)
├── cohort_overview.png                (demographics figure)
├── cross_naive/cohort.csv + per-partition summary_per_modality.csv
├── cross_matched/<partition>/{{matched_features, matched_pairs, matching_report,
│                             summary_stats, fig_*_overview, ...}}
└── stat_grid/<hc_source>/             (run_4arm_deep_dive)

age/analysis/{{pred_error_stat,classification}}/{COHORT_DIR}/<partition>/...
emo_au/analysis/feature_stat/{COHORT_DIR}/<partition>/...
asymmetry/analysis/feature_stat/{COHORT_DIR}/<partition>/...
embedding/analysis/feature_stat/{{original,difference}}/{COHORT_DIR}/<partition>/...
```

## Scope

- cross_naive (was arm_a) / cross_matched (was arm_b) ad_vs_{{hc, nad, acs}}
- cross_matched mmse_hi_lo / casi_hi_lo (within-AD)
- overview/ for arms A,B (HC source = ACS)
- arm_age_classifiers for arms A,B

## Skipped (out of scope)

- longitudinal arms (incompatible with first-visit-only P)
- overview/ acs_ext / eacs (only baseline ACS for now)

## Caveats

1. **HC 含 MCI / 失智 visit**：因為 NAD / ACS 不再做 strict HC 篩，這個 cohort
   的「HC pool」其實混了 CDR>=0.5 的 visit。對 ad_vs_hc effect size 可能造成
   低估（部分 HC 樣本本身就是 dementia）。
2. **多 visit 不獨立**：NAD / ACS 同一個 subject 多 visit 的特徵 correlated；
   GroupKFold 用 base_id 切 fold 仍然成立，但 1:1 age-matched pair 內可能同
   subject 出現多次。
3. **cross_matched pair 數會比 strict cohort 多**：HC pool 大幅擴張後，每個 AD
   都更容易在 caliper 2 年內找到 match。
"""
    readme.write_text(text, encoding="utf-8")
    logger.info(f"Wrote {readme}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-only", action="store_true",
                         help="Skip the 6 producer steps; only re-write "
                              "cohort_summary.csv + README.md from existing "
                              "outputs")
    args = parser.parse_args()

    summary_root = OVERVIEW_DIR / COHORT_DIR
    summary_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cohort summary root: {summary_root}")

    if args.summary_only:
        df_summary = write_cohort_summary(summary_root)
        write_readme(summary_root, df_summary)
        logger.info(f"\n[OK] SUMMARY-ONLY DONE -> {summary_root}")
        return

    # 1. cross_naive (was arm_a)
    run_step(
        "cross_naive (ad_vs_hc)",
        [PYTHON, SCRIPTS_DIR / "run_arm_a_ad_vs_hc.py",
         "--cohort-mode", COHORT_MODE],
    )

    # 2. cross_matched × {hc, nad, acs}
    for cmp in ("ad_vs_hc", "ad_vs_nad", "ad_vs_acs"):
        run_step(
            f"cross_matched ({cmp})",
            [PYTHON, SCRIPTS_DIR / "run_cross_matched.py",
             "--comparison", cmp,
             "--cohort-mode", COHORT_MODE],
        )

    # 3. cross_matched hi-lo × MMSE + CASI
    for cmp in ("mmse_hilo", "casi_hilo"):
        run_step(
            f"cross_matched ({cmp})",
            [PYTHON, SCRIPTS_DIR / "run_cross_matched.py",
             "--comparison", cmp,
             "--cohort-mode", COHORT_MODE],
        )

    # 4. AUC supplement × MMSE + CASI
    for metric in ("MMSE", "CASI"):
        run_step(
            f"arm_b_auc_supplement [HILO_METRIC={metric}]",
            [PYTHON, SCRIPTS_DIR / "run_arm_b_auc_supplement.py",
             "--cohort-mode", COHORT_MODE],
            env={"HILO_METRIC": metric},
        )

    # 5. overview/ stat grid (arms A B, HC=ACS)
    run_step(
        "run_4arm_deep_dive [arms A B, ACS]",
        [PYTHON, SCRIPTS_DIR / "run_4arm_deep_dive.py",
         "--cohort-mode", COHORT_MODE,
         "--arms", "A", "B",
         "--hc-source", "ACS"],
    )

    # 6. Age classifiers (A + B)
    run_step(
        "run_arm_age_classifiers [arms A B]",
        [PYTHON, SCRIPTS_DIR / "run_arm_age_classifiers.py",
         "--cohort-mode", COHORT_MODE,
         "--arms", "A", "B"],
    )

    # 7. Cohort summary + README
    df_summary = write_cohort_summary(summary_root)
    write_readme(summary_root, df_summary)

    logger.info(f"\n[OK] ALL DONE -> {summary_root}")


if __name__ == "__main__":
    main()
