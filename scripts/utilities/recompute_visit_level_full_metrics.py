"""
Recompute visit-level full-cohort metrics from existing OOF score CSVs.

Background: pre-fix versions of run_fwd_rev_embedding.py wrote
metrics_full_cohort (forward) and metrics_full / metrics_unmatched (reverse)
at SUBJECT level. The new convention is VISIT level — count every visit row
that entered training (1 row per visit when photo=mean, 10 when photo=all).

This script avoids re-running the 4-hour sweep by reading the visit-level
OOF CSVs that the pre-fix runs already saved:

    forward cells:
        forward_oof_scores.csv          (visit-level)  ← used to recompute
        forward_oof_scores_subject.csv  (subject-level, kept for matched eval)
        forward_matched_metrics.json    ← metrics_full_cohort updated in place

    reverse cells:
        ensemble/scores.csv      (subject-level only — pre-fix)
        ensemble/scores_visits.csv  (visit-level — only present in re-run cells)
        ensemble/metrics.json    ← metrics_full / metrics_unmatched updated in place
        single/scores.csv         (subject-level only — pre-fix)
        single/scores_visits.csv    (visit-level — only present in re-run cells)
        single/metrics.json      ← metrics_unmatched updated in place

For reverse cells lacking scores_visits.csv (most pre-fix cells), this
script falls back to computing metrics from the subject-level scores.csv
and prints a SKIP_REV note. After a reverse re-run, scores_visits.csv will
be present and visit-level metrics will be recomputed correctly.

Forward matched-pair metrics, paired Wilcoxon, and reverse matched_oof /
matched_train metrics are NOT touched (they are subject-level by design).

Usage:
    conda run -n Alz_face_main_analysis python \
        scripts/utilities/recompute_visit_level_full_metrics.py \
        --cohort-mode p_first_hc_all
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
N_BOOTSTRAP = 1000
SEED = 42


def _safe_write(path, content, retries=3, delay=0.2):
    """Write text via temp + rename. Avoids transient Windows write errors
    (e.g. antivirus locks). Retries up to N times."""
    import os
    import time
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_exc = None
    for _ in range(retries):
        try:
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, path)
            return
        except OSError as e:
            last_exc = e
            time.sleep(delay)
    raise last_exc


def bootstrap_auc_ci(y_true, y_prob, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    aucs = []
    for _ in range(n):
        pb = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        nb = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pb, nb])
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except ValueError:
            continue
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compute_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) < 5 or len(np.unique(y_true)) < 2:
        return None
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    auc = float(roc_auc_score(y_true, y_score))
    ci_low, ci_high = bootstrap_auc_ci(y_true, y_score)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    return {
        "n": int(len(y_true)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auc": auc,
        "auc_ci_low": ci_low,
        "auc_ci_high": ci_high,
        "balacc": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sens": float(cm[1, 1] / max(cm[1, :].sum(), 1)),
        "spec": float(cm[0, 0] / max(cm[0, :].sum(), 1)),
        "confusion_matrix": cm.tolist(),
    }


def update_forward_cell(clf_dir):
    csv = clf_dir / "forward_oof_scores.csv"
    json_path = clf_dir / "forward_matched_metrics.json"
    if not csv.exists() or not json_path.exists():
        return False
    df = pd.read_csv(csv)
    new_full = compute_metrics(df["y_true"], df["y_score"])
    if new_full is None:
        return False
    j = json.loads(json_path.read_text())
    j["metrics_full_cohort"] = new_full
    _safe_write(json_path, json.dumps(j, indent=2))
    return True


def update_reverse_method_cell(method_dir):
    """method_dir = .../rev/<part>/<emb>/<clf>/{ensemble,single}/"""
    json_path = method_dir / "metrics.json"
    visits_csv = method_dir / "scores_visits.csv"
    subj_csv = method_dir / "scores.csv"
    if not json_path.exists():
        return False, "no_metrics_json"
    j = json.loads(json_path.read_text())

    if visits_csv.exists():
        df = pd.read_csv(visits_csv)
        source = "visits"
    else:
        # Fallback: subject-level only — leaves visit-level metrics stale.
        return False, "no_visits_csv"

    # full
    if "metrics_full" in j:
        new_full = compute_metrics(df["y_true"], df["y_score"])
        if new_full is not None:
            j["metrics_full"] = new_full

    # unmatched
    if "metrics_unmatched" in j and "in_matched" in df.columns:
        unmatched = df[df["in_matched"] == 0]
        if len(unmatched) > 0:
            new_un = compute_metrics(unmatched["y_true"], unmatched["y_score"])
            if new_un is not None:
                j["metrics_unmatched"] = new_un

    _safe_write(json_path, json.dumps(j, indent=2))
    return True, source


def cohort_base(cohort_mode):
    cohort_dir = ("p_first_hc_all" if cohort_mode == "p_first_hc_all"
                  else "p_first_hc_strict")
    return PROJECT_ROOT / "workspace" / "arms_analysis" / cohort_dir


def walk_root(root):
    if not root.exists():
        return 0, 0, 0, 0
    n_fwd, n_rev_done, n_rev_skip, n_total = 0, 0, 0, 0
    # Detect layout: <reducer>/<variant>/{fwd,rev}/...  vs <reducer>/{fwd,rev}/...
    for reducer in sorted(root.iterdir()):
        if not reducer.is_dir() or reducer.name.startswith("_"):
            continue
        if (reducer / "fwd").is_dir() or (reducer / "rev").is_dir():
            variant_dirs = [reducer]
        else:
            variant_dirs = [v for v in sorted(reducer.iterdir())
                             if v.is_dir() and ((v / "fwd").is_dir()
                                                 or (v / "rev").is_dir())]
        for vdir in variant_dirs:
            # Forward
            fwd_root = vdir / "fwd"
            if fwd_root.is_dir():
                for partition_d in fwd_root.iterdir():
                    if not partition_d.is_dir():
                        continue
                    for emb_d in partition_d.iterdir():
                        if not emb_d.is_dir():
                            continue
                        for clf_d in emb_d.iterdir():
                            if not clf_d.is_dir():
                                continue
                            n_total += 1
                            if update_forward_cell(clf_d):
                                n_fwd += 1
            # Reverse
            rev_root = vdir / "rev"
            if rev_root.is_dir():
                for partition_d in rev_root.iterdir():
                    if not partition_d.is_dir():
                        continue
                    for emb_d in partition_d.iterdir():
                        if not emb_d.is_dir():
                            continue
                        for clf_d in emb_d.iterdir():
                            if not clf_d.is_dir():
                                continue
                            for sub in ("ensemble", "single"):
                                sub_d = clf_d / sub
                                if not sub_d.is_dir():
                                    continue
                                ok, info = update_reverse_method_cell(sub_d)
                                if ok:
                                    n_rev_done += 1
                                else:
                                    n_rev_skip += 1
    return n_fwd, n_rev_done, n_rev_skip, n_total


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--cohort-mode", default="default",
                    choices=["default", "p_first_hc_all"])
    args = p.parse_args()
    base = cohort_base(args.cohort_mode)
    total_fwd, total_rev_done, total_rev_skip, total_seen = 0, 0, 0, 0
    for sub in ("embedding_classification", "embedding_asymmetry_classification"):
        root = base / sub
        f, r_done, r_skip, t = walk_root(root)
        total_fwd += f
        total_rev_done += r_done
        total_rev_skip += r_skip
        total_seen += t
        print(f"  {sub}: fwd updated={f}, rev updated={r_done}, "
              f"rev skipped (no scores_visits.csv)={r_skip}")
    print(f"\nTOTAL: forward updated={total_fwd}, reverse updated="
          f"{total_rev_done}, reverse skipped={total_rev_skip}")
    if total_rev_skip > 0:
        print(f"\nNOTE: {total_rev_skip} reverse cells lack scores_visits.csv "
              "(pre-fix runs).\nRe-run those reverse cells to get visit-level "
              "full / unmatched metrics.")


if __name__ == "__main__":
    main()
