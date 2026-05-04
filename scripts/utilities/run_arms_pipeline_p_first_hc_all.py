"""
Run arms_analysis with the new cohort:
    P side : first-visit + Global_CDR>=0.5 + .npy fallback
    HC side: ALL NAD + ALL ACS visits (no strict HC filter)

Outputs go to workspace/arms_analysis/p_first_hc_all/  (sibling of the existing
per_arm/, grid/), so the original results stay untouched.

Sequentially invokes 6 stages with --cohort-mode p_first_hc_all + --output-dir
overrides:
    1. run_arm_a_ad_vs_hc.py            → per_arm/arm_a/ad_vs_hc/...
    2. run_arm_b_ad_vs_hcgroups.py      → per_arm/arm_b/{ad_vs_hc,nad,acs}/...
    3. run_mmse_hilo_standalone.py × 2  → per_arm/arm_b/{mmse,casi}_high_vs_low/
       (HILO_METRIC=MMSE then CASI)
    4. run_arm_b_auc_supplement.py × 2  → per_arm/arm_b/.../summary_per_modality_auc.csv
    5. run_4arm_deep_dive.py            → grid/acs/...
       (--cohort-mode p_first_hc_all --arms A B --hc-source ACS)
    6. run_arm_age_classifiers.py       → per_arm/arm_{a,b}/.../age/classifier_*

Then writes cohort_summary.csv + README.md describing the folder.

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
    ARMS_P_FIRST_HC_ALL_DIR,
    ARMS_P_FIRST_HC_ALL_PER_ARM,
    ARMS_P_FIRST_HC_ALL_GRID,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PYTHON = sys.executable

ARM_A_DIR = ARMS_P_FIRST_HC_ALL_PER_ARM / "arm_a"
ARM_B_DIR = ARMS_P_FIRST_HC_ALL_PER_ARM / "arm_b"


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

    # Arm A cohort
    arm_a_csv = ARM_A_DIR / "cohort.csv"
    if arm_a_csv.exists():
        df = pd.read_csv(arm_a_csv)
        for label, name in [(1, "Arm A AD"), (0, "Arm A HC")]:
            sub = df[df["label"] == label]
            rows.append({
                "stage": "arm_a/ad_vs_hc",
                "group": name,
                "n_rows": len(sub),
                "n_subjects": sub["base_id"].nunique() if "base_id" in sub.columns else len(sub),
                "age_mean": round(sub["Age"].mean(), 2) if len(sub) else "—",
                "age_std": round(sub["Age"].std(), 2) if len(sub) > 1 else "—",
            })

    # Arm B per comparison
    for cmp in ("hc", "nad", "acs"):
        f = ARM_B_DIR / f"ad_vs_{cmp}" / "matched_features.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for label, name in [(1, f"Arm B AD vs {cmp.upper()} (AD)"),
                              (0, f"Arm B AD vs {cmp.upper()} (HC)")]:
            sub = df[df["label"] == label]
            rows.append({
                "stage": f"arm_b/ad_vs_{cmp}",
                "group": name,
                "n_rows": len(sub),
                "n_subjects": sub["base_id"].nunique() if "base_id" in sub.columns else len(sub),
                "age_mean": round(sub["Age"].mean(), 2) if len(sub) else "—",
                "age_std": round(sub["Age"].std(), 2) if len(sub) > 1 else "—",
            })

    # Arm B hi-lo
    for metric in ("mmse", "casi"):
        f = ARM_B_DIR / f"{metric}_high_vs_low" / "matched_features.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        gcol = f"{metric}_group"
        if gcol not in df.columns:
            continue
        for grp_val in ("high", "low"):
            sub = df[df[gcol] == grp_val]
            rows.append({
                "stage": f"arm_b/{metric}_high_vs_low",
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
    text = f"""# arms_analysis (cohort: first-visit P + ALL NAD/ACS)

## Cohort definition

| Side | Filter |
|---|---|
| **P (AD)** | First-visit + `Global_CDR >= 0.5` + `.npy fallback` |
| **NAD / ACS (HC)** | **No strict HC filter** — every visit kept (CDR=0 / MMSE>=26 not required, no first-visit pick per HC subject) |

Differs from `workspace/arms_analysis/per_arm/`、`grid/` (the original
folders), which use strict HC (CDR=0 OR (CDR=NaN AND MMSE>=26)) AND
first-visit per HC subject.

## Cohort summary

{df_summary.to_markdown(index=False)}

## Folder layout

```
arms_analysis/p_first_hc_all/
├── README.md                     (this file)
├── cohort_summary.csv            (per-stage / per-group N + age stats)
├── per_arm/
│   ├── arm_a/
│   │   └── ad_vs_hc/             (run_arm_a_ad_vs_hc.py)
│   └── arm_b/
│       ├── ad_vs_hc/             (run_arm_b_ad_vs_hcgroups.py × HC)
│       ├── ad_vs_nad/            (× NAD)
│       ├── ad_vs_acs/            (× ACS)
│       ├── mmse_high_vs_low/     (run_mmse_hilo_standalone.py + auc_supplement, HILO_METRIC=MMSE)
│       └── casi_high_vs_low/     (HILO_METRIC=CASI)
├── grid/
│   └── acs/                      (run_4arm_deep_dive.py --arms A B --hc-source ACS)
└── classifier_summary_all.csv    (run_arm_age_classifiers.py)
```

## Scope

- arm_a / arm_b / grid arms A/B
- arm_b mmse_hi_lo / casi_hi_lo (within-AD)
- arm_age_classifiers for arms A,B

## Skipped (out of scope)

- arm_c / arm_d (longitudinal — incompatible with first-visit-only P)
- grid acs_ext / eacs (only baseline ACS for now)

## How to regenerate

```bash
conda run -n Alz_face_main_analysis \\
    python scripts/utilities/run_arms_pipeline_p_first_hc_all.py
```

## Caveats

1. **HC 含 MCI / 失智 visit**：因為 NAD / ACS 不再做 strict HC 篩，這個 cohort
   的「HC pool」其實混了 CDR>=0.5 的 visit。對 ad_vs_hc effect size 可能造成
   低估（部分 HC 樣本本身就是 dementia）。
2. **多 visit 不獨立**：NAD / ACS 同一個 subject 多 visit 的特徵 correlated；
   GroupKFold 用 base_id 切 fold 仍然成立，但 1:1 age-matched pair 內可能同
   subject 出現多次。
3. **Arm B matched pair 數會比舊 cohort 多**：HC pool 大幅擴張後，每個 AD
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

    ARMS_P_FIRST_HC_ALL_DIR.mkdir(parents=True, exist_ok=True)
    ARMS_P_FIRST_HC_ALL_PER_ARM.mkdir(parents=True, exist_ok=True)
    ARMS_P_FIRST_HC_ALL_GRID.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output root: {ARMS_P_FIRST_HC_ALL_DIR}")

    if args.summary_only:
        df_summary = write_cohort_summary(ARMS_P_FIRST_HC_ALL_DIR)
        write_readme(ARMS_P_FIRST_HC_ALL_DIR, df_summary)
        logger.info(f"\n[OK] SUMMARY-ONLY DONE -> {ARMS_P_FIRST_HC_ALL_DIR}")
        return

    # 1. Arm A
    run_step(
        "arm_a (ad_vs_hc)",
        [PYTHON, PROJECT_ROOT / "scripts" / "experiments" / "run_arm_a_ad_vs_hc.py",
         "--cohort-mode", "p_first_hc_all",
         "--output-dir", ARM_A_DIR],
    )

    # 2. Arm B × HC/NAD/ACS
    run_step(
        "arm_b (ad_vs_hc / nad / acs)",
        [PYTHON, PROJECT_ROOT / "scripts" / "experiments" / "run_arm_b_ad_vs_hcgroups.py",
         "--cohort-mode", "p_first_hc_all",
         "--output-dir", ARM_B_DIR],
    )

    # 3. Arm B hi-lo × MMSE + CASI
    for metric in ("MMSE", "CASI"):
        run_step(
            f"mmse_hilo_standalone [HILO_METRIC={metric}]",
            [PYTHON, PROJECT_ROOT / "scripts" / "experiments" / "run_mmse_hilo_standalone.py",
             "--cohort-mode", "p_first_hc_all",
             "--output-dir", ARM_B_DIR],
            env={"HILO_METRIC": metric},
        )

    # 4. Arm B AUC supplement × MMSE + CASI
    for metric in ("MMSE", "CASI"):
        run_step(
            f"arm_b_auc_supplement [HILO_METRIC={metric}]",
            [PYTHON, PROJECT_ROOT / "scripts" / "experiments" / "run_arm_b_auc_supplement.py",
             "--arm-b-dir", ARM_B_DIR],
            env={"HILO_METRIC": metric},
        )

    # 5. Grid arms A/B
    run_step(
        "run_4arm_deep_dive [arms A B, ACS, p_first_hc_all]",
        [PYTHON, PROJECT_ROOT / "scripts" / "experiments" / "run_4arm_deep_dive.py",
         "--cohort-mode", "p_first_hc_all",
         "--arms", "A", "B",
         "--hc-source", "ACS",
         "--output-dir", ARMS_P_FIRST_HC_ALL_GRID / "acs"],
    )

    # 6. Arm age classifiers (A + B)
    run_step(
        "run_arm_age_classifiers [arms A B]",
        [PYTHON, PROJECT_ROOT / "scripts" / "experiments" / "run_arm_age_classifiers.py",
         "--cohort-mode", "p_first_hc_all",
         "--arms", "A", "B",
         "--arms-root", ARMS_P_FIRST_HC_ALL_PER_ARM],
    )

    # 7. Cohort summary + README
    df_summary = write_cohort_summary(ARMS_P_FIRST_HC_ALL_DIR)
    write_readme(ARMS_P_FIRST_HC_ALL_DIR, df_summary)

    logger.info(f"\n[OK] ALL DONE -> {ARMS_P_FIRST_HC_ALL_DIR}")


if __name__ == "__main__":
    main()
