"""
Run the age_prediction analysis with the new cohort:
    P side : first-visit + Global_CDR>=0.5 + .npy fallback
    HC side: ALL NAD + ALL ACS visits (no strict HC filter)

Outputs go to workspace/age/age_prediction_p_first_hc_all/  (sibling of the
existing age_prediction/), so the original results stay untouched.

Sequentially invokes 6 producer scripts with --cohort-mode p_first_hc_all +
--output-dir overrides:
    1. plot_predicted_ages.py            → scatter, sliding-window, top-level
                                            error_by_age_combined.png
    2. calibrate_age_prediction.py       → corrections/calibration/
    3. age_error_mean_correction.py      → corrections/mean_correction/
    4. age_error_bootstrap_correction.py → corrections/bootstrap_correction/
    5. plot_error_by_age_with_utkface.py → error_by_age/ + residual_by_age/
                                            (15 PNG each + sw10/ subdir)
    6. plot_acs_eacs_predicted_ages.py   → scatter/{combined,per_source}.png

Then writes cohort_summary.csv + README.md describing what's in the folder.

Usage:
    conda run -n Alz_face_main_analysis \
        python scripts/utilities/run_age_pipeline_p_first_hc_all.py
"""
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    AGE_PREDICTION_DIR_V2,
    CALIBRATION_DIR_V2,
    MEAN_CORRECTION_DIR_V2,
    BOOTSTRAP_DIR_V2,
    DEMOGRAPHICS_DIR,
    PREDICTED_AGES_FILE,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PYTHON = sys.executable


def run_step(name, script, output_dir=None, extra_args=None,
             include_cohort_mode=True):
    """Helper to call one of the producer scripts.

    include_cohort_mode=False for downstream-only scripts (e.g.,
    plot_error_by_age_with_utkface.py, plot_acs_eacs_predicted_ages.py)
    that don't take --cohort-mode (they read already-cohort-filtered
    upstream outputs).
    """
    cmd = [PYTHON, str(PROJECT_ROOT / script)]
    if include_cohort_mode:
        cmd.extend(["--cohort-mode", "p_first_hc_all"])
    if output_dir is not None:
        cmd.extend(["--output-dir", str(output_dir)])
    if extra_args:
        cmd.extend(extra_args)
    logger.info(f"\n=== {name} ===")
    logger.info(f"  cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def write_cohort_summary(output_dir):
    """Compute cohort summary stats for the new cohort and write CSV."""
    sys.path.insert(0, str(PROJECT_ROOT))
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "calibration",
        PROJECT_ROOT / "src" / "extractor" / "features" / "age" / "calibration.py",
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df_acs, df_nad, df_p = mod.load_demographics_for_calibration(
        DEMOGRAPHICS_DIR, PREDICTED_AGES_FILE, cohort_mode="p_first_hc_all",
    )
    rows = []
    for name, df in [("ACS", df_acs), ("NAD", df_nad), ("P", df_p)]:
        rows.append({
            "group": name,
            "n_rows": len(df),
            "n_subjects": df["subject"].nunique(),
            "real_age_mean": round(df["real_age"].mean(), 2),
            "real_age_std": round(df["real_age"].std(), 2),
            "real_age_min": round(df["real_age"].min(), 1),
            "real_age_max": round(df["real_age"].max(), 1),
            "predicted_age_mean": round(df["predicted_age"].mean(), 2),
            "error_mean": round(df["error"].mean(), 2),
            "error_std": round(df["error"].std(), 2),
            "abs_error_mean": round(df["error"].abs().mean(), 2),
        })
    rows.append({
        "group": "Total",
        "n_rows": len(df_acs) + len(df_nad) + len(df_p),
        "n_subjects": (df_acs["subject"].nunique()
                       + df_nad["subject"].nunique()
                       + df_p["subject"].nunique()),
        "real_age_mean": "—",
        "real_age_std": "—",
        "real_age_min": "—",
        "real_age_max": "—",
        "predicted_age_mean": "—",
        "error_mean": "—",
        "error_std": "—",
        "abs_error_mean": "—",
    })
    df_summary = pd.DataFrame(rows)
    csv = output_dir / "cohort_summary.csv"
    df_summary.to_csv(csv, index=False, encoding="utf-8-sig")
    logger.info(f"Wrote {csv}")
    return df_summary


def write_readme(output_dir, df_summary):
    readme = output_dir / "README.md"
    text = f"""# age_prediction (cohort: first-visit P + ALL NAD/ACS)

## Cohort definition

| Side | Filter |
|---|---|
| **P (AD)** | First-visit + `Global_CDR >= 0.5` + `.npy fallback` |
| **NAD / ACS (HC)** | **No strict HC filter** — every visit with both `predicted_age` and real `Age` is kept |

This differs from `workspace/age/age_prediction/` (the original folder),
which uses every visit per subject for ALL groups (no first-visit pick on
P, no relaxation needed on HC since none was applied either).

## Cohort summary

{df_summary.to_markdown(index=False)}

## Folder layout

```
age_prediction_p_first_hc_all/
├── README.md                      (this file)
├── cohort_summary.csv             (per-group N + age stats)
├── age_error_stat_2.csv
├── age_error_sliding_window.csv
├── error_by_age_combined.png
├── predicted_ages_scatter.png
├── corrections/
│   ├── calibration/
│   │   ├── train90_val10/
│   │   ├── train10_val90/
│   │   └── comparison.csv
│   ├── mean_correction/
│   └── bootstrap_correction/
├── error_by_age/
├── scatter/
└── residual_by_age/
```

## How to regenerate

```bash
conda run -n Alz_face_main_analysis \\
    python scripts/utilities/run_age_pipeline_p_first_hc_all.py
```

## Caveats

1. **HC side含 MCI / 失智 visit**：因為 NAD / ACS 不再做 strict HC 篩，這個
   cohort 的「HC pool」其實混了 CDR≥0.5 的 visit。calibration 學到的
   year-on-year error 修正可能受到這些 dementia visits 影響。對照
   `age_prediction/` 的全 pool calibration 結果可看出差異有多大。
2. **多 visit 不獨立**：NAD / ACS 同一個 subject 多 visit 的 (real_age,
   error) 是 correlated。calibration K-fold 已用 `subject` 切 fold，
   不會跨 fold 洩漏。
3. P 端 N 比舊 folder 少（1007 vs ~1300+ in P.csv）— 篩了 first-visit
   + CDR≥0.5。
"""
    readme.write_text(text, encoding="utf-8")
    logger.info(f"Wrote {readme}")


def main():
    AGE_PREDICTION_DIR_V2.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output root: {AGE_PREDICTION_DIR_V2}")

    # 1. plot_predicted_ages → general stats + sliding window + scatter
    run_step(
        "plot_predicted_ages",
        "scripts/visualization/plot_predicted_ages.py",
        AGE_PREDICTION_DIR_V2,
        extra_args=["--stat-dir", str(CALIBRATION_DIR_V2)],
    )

    # 2. calibrate_age_prediction → K-fold calibration
    run_step(
        "calibrate_age_prediction",
        "scripts/utilities/calibrate_age_prediction.py",
        CALIBRATION_DIR_V2,
    )

    # 3. age_error_mean_correction
    run_step(
        "age_error_mean_correction",
        "scripts/utilities/age_error_mean_correction.py",
        MEAN_CORRECTION_DIR_V2,
    )

    # 4. age_error_bootstrap_correction
    run_step(
        "age_error_bootstrap_correction",
        "scripts/utilities/age_error_bootstrap_correction.py",
        BOOTSTRAP_DIR_V2,
    )

    # 5. plot_error_by_age_with_utkface — uses bootstrap output, writes to
    #    error_by_age/ + residual_by_age/ under the V2 root.
    run_step(
        "plot_error_by_age_with_utkface",
        "scripts/visualization/plot_error_by_age_with_utkface.py",
        output_dir=AGE_PREDICTION_DIR_V2,
        extra_args=["--bootstrap-dir", str(BOOTSTRAP_DIR_V2)],
        include_cohort_mode=False,
    )

    # 6. ACS + EACS scatter (3 PNGs to scatter/ subdir).
    scatter_dir = AGE_PREDICTION_DIR_V2 / "scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    for mode_args, fname in [
        ([], "predicted_ages_scatter_combined.png"),
        (["--mode", "per_source", "--min-pred", "42"],
         "predicted_ages_scatter_per_source_minpred42.png"),
    ]:
        run_step(
            f"plot_acs_eacs_predicted_ages [{fname}]",
            "scripts/visualization/plot_acs_eacs_predicted_ages.py",
            output_dir=None,
            extra_args=mode_args + ["--output", str(scatter_dir / fname)],
            include_cohort_mode=False,
        )

    # 7. cohort summary + README
    df_summary = write_cohort_summary(AGE_PREDICTION_DIR_V2)
    write_readme(AGE_PREDICTION_DIR_V2, df_summary)

    logger.info(f"\n✓ ALL DONE → {AGE_PREDICTION_DIR_V2}")


if __name__ == "__main__":
    main()
