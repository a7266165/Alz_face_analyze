"""
Cohort orchestration pipeline (merge of run_arms_pipeline_p_*.py × 2).

One entry point parametrized by --cohort-mode. Each cohort runs the same 5
producer steps + writes cohort_summary.csv + README.md.

5 steps (post-refactor; AUC supplement is now folded into step 3):
    1. cross_naive (ad_vs_hc)                  → overview/run_cross_naive.py
    2. cross_matched × {ad_vs_hc, ad_vs_nad, ad_vs_acs}  → overview/run_cross_matched.py
    3. cross_matched × {mmse_hilo, casi_hilo}  → overview/run_cross_matched.py
                                                  (auto-includes AUC supplement)
    4. stat grid (cross_naive + cross_matched, HC=ACS)   → overview/run_stat_grid.py
    5. age classifiers (cross_naive + cross_matched)     → age/run_classifiers.py

Cohort modes:
    default          strict-HC + first-visit (legacy main cohort)
    p_first_hc_all   first-visit P + ALL NAD/ACS (HC unfiltered)
    p_all_hc_all     ALL P visits + ALL NAD/ACS (most permissive, multi-visit
                     subject-first matching)

Longitudinal designs are not invoked by default — they need build_longitudinal_*
producers and are out of scope for cross-sectional cohort pipelines.

Usage:
    conda run -n Alz_face_main_analysis python scripts/overview/run_cohort_pipeline.py \\
        --cohort-mode p_first_hc_all
    conda run -n Alz_face_main_analysis python scripts/overview/run_cohort_pipeline.py \\
        --cohort-mode p_all_hc_all --summary-only
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
from src.config import OVERVIEW_DIR, cohort_name

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PYTHON = sys.executable
OVERVIEW_SCRIPTS_DIR = Path(__file__).parent
AGE_SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "age"


# ============================================================
# Per-cohort README templates
# ============================================================

README_TEMPLATES = {
    "default": {
        "title": "cohort: first-visit P + strict-HC NAD/ACS (legacy main cohort)",
        "definition": (
            "| Side | Filter |\n"
            "|---|---|\n"
            "| **P (AD)** | First-visit + `Global_CDR >= 0.5` + `.npy fallback` |\n"
            "| **NAD / ACS (HC)** | Strict HC: `CDR=0` OR (`CDR=NaN` AND `MMSE>=26`) "
            "+ first-visit per HC subject |\n"
        ),
        "intro": (
            "Original main analysis cohort. Conservative on both sides — strict "
            "HC criteria + first-visit only.\n"
        ),
        "caveats": (
            "1. **AD 端 first-visit only**：每 base_id 只用一筆 visit（feature "
            "fallback 規則：若第一 visit 無 embedding+landmark feature，退而求"
            "其次選最早有 feature 的 visit）。\n"
            "2. **HC 嚴篩**：NAD / ACS 必須 `CDR=0` 或 (`CDR=NaN` AND `MMSE>=26`) "
            "才入選。\n"
        ),
    },
    "p_first_hc_all": {
        "title": "cohort: first-visit P + ALL NAD/ACS",
        "definition": (
            "| Side | Filter |\n"
            "|---|---|\n"
            "| **P (AD)** | First-visit + `Global_CDR >= 0.5` + `.npy fallback` |\n"
            "| **NAD / ACS (HC)** | **No strict HC filter** — every visit kept "
            "(CDR=0 / MMSE>=26 not required, no first-visit pick per HC subject) |\n"
        ),
        "intro": (
            "Differs from `default` by relaxing the HC side: keep every NAD/ACS "
            "visit regardless of CDR/MMSE.\n"
        ),
        "caveats": (
            "1. **HC 含 MCI / 失智 visit**：因為 NAD / ACS 不再做 strict HC 篩，"
            "這個 cohort 的「HC pool」其實混了 CDR>=0.5 的 visit。對 ad_vs_hc "
            "effect size 可能造成低估（部分 HC 樣本本身就是 dementia）。\n"
            "2. **多 visit 不獨立**：NAD / ACS 同一個 subject 多 visit 的特徵 "
            "correlated；GroupKFold 用 base_id 切 fold 仍然成立，但 1:1 age-"
            "matched pair 內可能同 subject 出現多次。\n"
            "3. **cross_matched pair 數會比 strict cohort 多**：HC pool 大幅"
            "擴張後，每個 AD 都更容易在 caliper 2 年內找到 match。\n"
        ),
    },
    "p_all_hc_all": {
        "title": "cohort: ALL P visits + ALL NAD/ACS",
        "definition": (
            "| Side | Filter |\n"
            "|---|---|\n"
            "| **P (AD)** | All visits with `Global_CDR>=0.5` AND embedding+landmark "
            ".npy features available |\n"
            "| **NAD / ACS (HC)** | All visits, no strict HC filter |\n"
        ),
        "intro": (
            "Most permissive cohort. AD side has multi-visit records, so 1:1 "
            "age-matching falls back to subject-first two-pass matching (a "
            "base_id can match once at subject level, then unmatched visits "
            "attach to already-used majors at visit-level fallback).\n"
        ),
        "caveats": (
            "1. **AD 端多 visit**：同一 base_id 多 visit 都進 cohort，"
            "subject-first 兩階段配對。\n"
            "2. **HC 端含 dementia visit**：NAD / ACS 不再做 strict HC 篩，"
            "HC pool 混了高 CDR 個案。\n"
            "3. **每個 base_id 配對次數會變多**：HC pool 大幅擴張下，每個 AD "
            "都更容易在 caliper 內找到 match。\n"
        ),
    },
}


# ============================================================
# Step runner
# ============================================================

def run_step(name, cmd, env=None):
    logger.info(f"\n=== {name} ===")
    logger.info(f"  cmd: {' '.join(str(c) for c in cmd)}")
    if env:
        env_show = {k: env[k] for k in env if k in (
            "COHORT_MODE", "HC_SOURCE_MODE")}
        if env_show:
            logger.info(f"  env: {env_show}")
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(PROJECT_ROOT),
                   env=full_env)


# ============================================================
# Cohort summary builder
# ============================================================

def write_cohort_summary(cohort_dir, output_dir):
    """Compute cohort summary from artifacts; works across all cohort modes."""
    rows = []

    # cross_naive (was arm_a)
    cn_csv = OVERVIEW_DIR / cohort_dir / "cross_naive" / "cohort.csv"
    if cn_csv.exists():
        df = pd.read_csv(cn_csv)
        for label, name in [(1, "cross_naive AD"), (0, "cross_naive HC")]:
            sub = df[df["label"] == label]
            rows.append({
                "stage": "cross_naive/ad_vs_hc",
                "group": name,
                "n_rows": len(sub),
                "n_subjects": (sub["base_id"].nunique()
                               if "base_id" in sub.columns else len(sub)),
                "age_mean": round(sub["Age"].mean(), 2) if len(sub) else "—",
                "age_std": round(sub["Age"].std(), 2) if len(sub) > 1 else "—",
            })

    # cross_matched per comparison
    cm_root = OVERVIEW_DIR / cohort_dir / "cross_matched"
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
                "n_subjects": (sub["base_id"].nunique()
                               if "base_id" in sub.columns else len(sub)),
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
                "n_subjects": (sub["ID"].astype(str).str.extract(
                    r"^(.+)-\d+$").iloc[:, 0].nunique() if len(sub) else 0),
                "age_mean": round(sub["Age"].mean(), 2) if len(sub) else "—",
                "age_std": round(sub["Age"].std(), 2) if len(sub) > 1 else "—",
            })

    df_summary = pd.DataFrame(rows)
    csv = output_dir / "cohort_summary.csv"
    df_summary.to_csv(csv, index=False, encoding="utf-8-sig")
    logger.info(f"Wrote {csv}")
    return df_summary


# ============================================================
# README writer
# ============================================================

def write_readme(cohort_mode, cohort_dir, output_dir, df_summary):
    tmpl = README_TEMPLATES[cohort_mode]
    readme = output_dir / "README.md"
    text = f"""# {tmpl['title']}

## Cohort definition

{tmpl['definition']}
{tmpl['intro']}
## Cohort summary

{df_summary.to_markdown(index=False)}

## Output fan-out

```
overview/{cohort_dir}/
├── README.md                          (this file)
├── cohort_summary.csv                 (per-stage / per-group N + age stats)
├── classifier_summary_all.csv         (run_age_classifiers grand summary)
├── cross_naive/cohort.csv + per-partition summary_per_modality.csv
├── cross_matched/<partition>/{{matched_features, matched_pairs, matching_report,
│                             summary_stats, summary_per_modality_auc, ...}}
└── stat_grid/<hc_source>/             (run_stat_grid)

age/analysis/{{pred_error_stat,classification}}/{cohort_dir}/<partition>/...
emo_au/analysis/feature_stat/{cohort_dir}/<partition>/...
asymmetry/analysis/feature_stat/{cohort_dir}/<partition>/...
embedding/analysis/feature_stat/{{original,difference}}/{cohort_dir}/<partition>/...
```

## Scope

- cross_naive / cross_matched ad_vs_{{hc, nad, acs}}
- cross_matched mmse_hi_lo / casi_hi_lo (within-AD; AUC supplement folded in)
- stat_grid for cross_naive + cross_matched (HC source = ACS)
- age_classifiers for cross_naive + cross_matched

## Skipped (out of scope for cross-sec cohort pipeline)

- longitudinal designs (longitudinal_naive / longitudinal_matched) — invoke
  run_stat_grid.py / run_age_classifiers.py with --designs longitudinal_*
  separately; require build_longitudinal_dataset.py outputs.
- stat_grid acs_ext / eacs (only baseline ACS by default)

## Caveats

{tmpl['caveats']}
"""
    readme.write_text(text, encoding="utf-8")
    logger.info(f"Wrote {readme}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort-mode", required=True,
                        choices=list(README_TEMPLATES.keys()))
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip the 5 producer steps; only re-write "
                             "cohort_summary.csv + README.md from existing outputs")
    args = parser.parse_args()

    cohort_dir = cohort_name(args.cohort_mode)
    summary_root = OVERVIEW_DIR / cohort_dir
    summary_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cohort summary root: {summary_root}")

    if args.summary_only:
        df_summary = write_cohort_summary(cohort_dir, summary_root)
        write_readme(args.cohort_mode, cohort_dir, summary_root, df_summary)
        logger.info(f"\n[OK] SUMMARY-ONLY DONE -> {summary_root}")
        return

    # 1. cross_naive (ad_vs_hc)
    run_step(
        "cross_naive (ad_vs_hc)",
        [PYTHON, OVERVIEW_SCRIPTS_DIR / "run_cross_naive.py",
         "--cohort-mode", args.cohort_mode],
    )

    # 2. cross_matched × {hc, nad, acs}
    for cmp in ("ad_vs_hc", "ad_vs_nad", "ad_vs_acs"):
        run_step(
            f"cross_matched ({cmp})",
            [PYTHON, OVERVIEW_SCRIPTS_DIR / "run_cross_matched.py",
             "--comparison", cmp,
             "--cohort-mode", args.cohort_mode],
        )

    # 3. cross_matched hi-lo × {MMSE, CASI} (auto-includes AUC supplement)
    for cmp in ("mmse_hilo", "casi_hilo"):
        run_step(
            f"cross_matched ({cmp})",
            [PYTHON, OVERVIEW_SCRIPTS_DIR / "run_cross_matched.py",
             "--comparison", cmp,
             "--cohort-mode", args.cohort_mode],
        )

    # 4. stat grid (cross_naive + cross_matched, HC=ACS)
    run_step(
        "run_stat_grid [cross_naive + cross_matched, ACS]",
        [PYTHON, OVERVIEW_SCRIPTS_DIR / "run_stat_grid.py",
         "--cohort-mode", args.cohort_mode,
         "--designs", "cross_naive", "cross_matched",
         "--hc-source", "ACS"],
    )

    # 5. age classifiers (cross_naive + cross_matched)
    run_step(
        "run_age_classifiers [cross_naive + cross_matched]",
        [PYTHON, AGE_SCRIPTS_DIR / "run_classifiers.py",
         "--cohort-mode", args.cohort_mode,
         "--designs", "cross_naive", "cross_matched"],
    )

    # 6. Cohort summary + README
    df_summary = write_cohort_summary(cohort_dir, summary_root)
    write_readme(args.cohort_mode, cohort_dir, summary_root, df_summary)

    logger.info(f"\n[OK] ALL DONE -> {summary_root}")


if __name__ == "__main__":
    main()
