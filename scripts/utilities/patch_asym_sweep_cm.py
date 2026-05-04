"""
Patch missing TN/FP/FN/TP columns in
embedding_asymmetry_classification/<reducer>/<variant>/_summary/all_metrics_with_cm.csv
by reading per-cell metrics.json files which contain a `confusion_matrix`
field as [[TN, FP], [FN, TP]].

Why: the per-reducer combined_summary.csv I derived all_metrics_with_cm.csv
from doesn't include CM cells, only AUC/BalAcc/MCC. The CM is in the per-cell
JSONs which we now read back to fill in.

Run:
    conda run -n Alz_face_main_analysis \
        python scripts/utilities/patch_asym_sweep_cm.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def root_for(cohort_mode):
    cohort_dir = "p_first_hc_all" if cohort_mode == "p_first_hc_all" else "p_first_hc_strict"
    return (PROJECT_ROOT / "workspace" / "arms_analysis" /
            cohort_dir / "embedding_asymmetry_classification")


ROOT = root_for("default")  # legacy default; main() resolves args

# CSV scope value → (subdir under cell, json filename, json key for the
# nested metrics block)
SCOPE_MAP = {
    "forward_matched":              ("fwd", "forward_matched_metrics.json", "metrics_matched_subset"),
    "forward_full":                 ("fwd", "forward_matched_metrics.json", "metrics_full_cohort"),
    "reverse_ensemble_matched_oof": ("rev/ensemble", "metrics.json", "metrics_matched_oof"),
    "reverse_ensemble_full":        ("rev/ensemble", "metrics.json", "metrics_full"),
    "reverse_ensemble_unmatched":   ("rev/ensemble", "metrics.json", "metrics_unmatched"),
    "reverse_single_matched_train": ("rev/single",   "metrics.json", "metrics_matched_train"),
    "reverse_single_unmatched":     ("rev/single",   "metrics.json", "metrics_unmatched"),
}


def cm_for_cell(cell_dir, scope):
    rel_subdir, fname, key = SCOPE_MAP[scope]
    json_path = cell_dir / rel_subdir / fname
    if not json_path.exists():
        return None
    try:
        m = json.loads(json_path.read_text())
        block = m.get(key)
        if not block:
            return None
        cm = block.get("confusion_matrix")
        if cm is None or len(cm) != 2 or len(cm[0]) != 2:
            return None
        return int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def patch_file(csv_path, variant_dir):
    df = pd.read_csv(csv_path)
    if not df["TN"].isna().all():
        return 0  # already has CM
    n_filled = 0
    for i, row in df.iterrows():
        scope = row["scope"]
        if scope not in SCOPE_MAP:
            continue
        partition = row["partition"]
        embedding = row["embedding"]
        classifier = row["classifier"]
        cell_dir = variant_dir / partition / embedding / classifier
        if scope.startswith("forward"):
            # fwd cells live under <variant>/fwd/<partition>/<emb>/<clf>/
            cell_dir = variant_dir.parent / variant_dir.name / "fwd" / partition / embedding / classifier
            cm = cm_for_cell(variant_dir.parent / variant_dir.name, scope)
            # Need correct path structure
            json_path = (variant_dir / "fwd" / partition / embedding /
                          classifier / "forward_matched_metrics.json")
        else:
            json_path = None
            if scope.startswith("reverse_ensemble"):
                json_path = (variant_dir / "rev" / partition / embedding /
                              classifier / "ensemble" / "metrics.json")
            elif scope.startswith("reverse_single"):
                json_path = (variant_dir / "rev" / partition / embedding /
                              classifier / "single" / "metrics.json")
        if json_path is None or not json_path.exists():
            continue
        try:
            m = json.loads(json_path.read_text())
            block = m.get(SCOPE_MAP[scope][2])
            if not block:
                continue
            cm = block.get("confusion_matrix")
            if cm is None or len(cm) != 2 or len(cm[0]) != 2:
                continue
            df.at[i, "TN"] = int(cm[0][0])
            df.at[i, "FP"] = int(cm[0][1])
            df.at[i, "FN"] = int(cm[1][0])
            df.at[i, "TP"] = int(cm[1][1])
            n_filled += 1
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    if n_filled > 0:
        df.to_csv(csv_path, index=False)
    return n_filled


def main():
    if not ROOT.exists():
        print(f"ROOT missing: {ROOT}", file=sys.stderr)
        sys.exit(1)

    total_files = 0
    total_rows = 0
    for reducer in sorted(ROOT.iterdir()):
        if not reducer.is_dir() or reducer.name.startswith("_"):
            continue
        if reducer.name == "no_drop" or not reducer.name.startswith(("drop_", "pca_")):
            continue
        for variant_dir in sorted(reducer.iterdir()):
            if not variant_dir.is_dir():
                continue
            csv = variant_dir / "_summary" / "all_metrics_with_cm.csv"
            if not csv.exists():
                continue
            n = patch_file(csv, variant_dir)
            if n > 0:
                total_files += 1
                total_rows += n
                print(f"  {reducer.name}/{variant_dir.name}: filled {n} rows")

    print(f"\nTOTAL: {total_files} files, {total_rows} rows filled")


if __name__ == "__main__":
    main()
