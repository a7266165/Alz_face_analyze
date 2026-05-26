"""
Embedding-classifier sweep for the p_first_cdr05_hc_all_cdrall_or_mmseall cohort.

Runs run_fwd_rev.py across:
    - feature_type:    original + 5 asymmetry variants (6 total)
    - reducer:         no_drop + PCA grid + dropcorr grid

For each (feature_type, reducer) combo, run_fwd_rev internally
sweeps partition x embedding x classifier x strategy = 60 cells.

GPU PCA + GPU XGB are unconditionally on inside run_fwd_rev.

Reducer naming (a single positional value, parsed):
    "no_drop"      -> no reducer
    int (e.g. 50)  -> PCA n_components fixed (GPU torch.pca_lowrank)
    float<1 (0.95) -> PCA variance ratio (sklearn fallback)
    "drop_<thr>"   -> DropCorrelatedFeatures(threshold=<thr>)

Default settings:
    --cohort-mode p_first_cdr05_hc_all_cdrall_or_mmseall   (P first-visit + HC all visits)
    --photo-mode mean              (mean-pool 10 photos -> 1 vector per visit)

Usage:
    conda run -n Alz_face_main_analysis python \
        scripts/embedding/run_sweep.py
    # subset:
    conda run -n Alz_face_main_analysis python \
        scripts/embedding/run_sweep.py \
        --feature-types original difference \
        --reducers pca_100 drop_0.85
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# Windows-only: avoid subprocess.run spawning a visible console window per
# invocation (TabPFN / xgboost / etc. otherwise pop tabs each spawn).
_SUBPROCESS_KWARGS = {}
if sys.platform == "win32":
    _SUBPROCESS_KWARGS["creationflags"] = getattr(
        subprocess, "CREATE_NO_WINDOW", 0x08000000)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PYTHON = sys.executable

DEFAULT_FEATURE_TYPES = [
    "original",
    "difference", "absolute_difference", "average",
    "relative_differences", "absolute_relative_differences",
]

# Reducer specs as strings. Parser handles None / int / float / drop_X.
DEFAULT_REDUCERS = (
    ["no_drop"]
    + [f"pca_{n}" for n in (1, 2, 5, 10, 20, 50, 100, 200, 400)]
    + [f"pca_{r}" for r in (0.95, 0.99)]
    + [f"drop_{t}" for t in
       (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.825, 0.85,
        0.875, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98)]
)


def parse_reducer(spec):
    """Return ("no_drop"|"pca"|"drop", value) tuple. Value is None for no_drop,
    str/int/float for pca, float for drop."""
    s = spec.strip()
    if s.lower() in ("no_drop", "none", ""):
        return ("no_drop", None)
    if s.startswith("pca_"):
        v = s[len("pca_"):]
        try:
            iv = int(v)
            return ("pca", iv)
        except ValueError:
            try:
                fv = float(v)
                return ("pca", fv)
            except ValueError:
                raise ValueError(f"bad pca spec: {spec!r}")
    if s.startswith("drop_"):
        v = s[len("drop_"):]
        return ("drop", float(v))
    # bare int / float -> PCA
    try:
        return ("pca", int(s))
    except ValueError:
        try:
            return ("pca", float(s))
        except ValueError:
            raise ValueError(f"unrecognized reducer spec: {spec!r}")


def label_for(kind, value):
    if kind == "no_drop":
        return "no_drop"
    if kind == "pca":
        return f"pca_{value}"
    return f"drop_{value}"


def run_one(feat, kind, value, args, progress=""):
    cmd = [PYTHON, str(PROJECT_ROOT / "scripts" / "embedding" /
                        "run_fwd_rev.py"),
           "--cohort-mode", args.cohort_mode,
           "--photo-mode", args.photo_mode,
           "--feature-type", feat]
    if kind == "pca":
        cmd += ["--pca-components", str(value)]
    elif kind == "drop":
        cmd += ["--drop-correlated-threshold", str(value)]
    if args.exclude_classifiers:
        cmd += ["--exclude-classifiers", *args.exclude_classifiers]
    if args.caliper_group != 3.0:
        cmd += ["--caliper-group", str(args.caliper_group)]
    cmd += ["--bg-mode", args.bg_mode]
    if args.match_priority:
        cmd += ["--match-priority", *args.match_priority]
    if args.embedding:
        cmd += ["--embedding", args.embedding]
    if args.grid_search:
        cmd += ["--grid-search"]
    label = label_for(kind, value)
    print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] {progress}{feat} / "
          f"{label}\n{'='*70}", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, **_SUBPROCESS_KWARGS)
    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] {feat} / {label} "
          f"-> exit={proc.returncode} in {dt/60:.1f} min", flush=True)
    return proc.returncode == 0


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--feature-types", nargs="*", default=DEFAULT_FEATURE_TYPES)
    p.add_argument("--reducers", nargs="*", default=None,
                    help="Reducer specs (no_drop, pca_<N>, pca_<ratio>, "
                         "drop_<thr>). Default: full grid (32 reducers).")
    from src.config import VALID_COHORT_CHOICES
    p.add_argument("--cohort-mode",
                    default="p_first_cdr05_hc_all_cdrall_or_mmseall",
                    choices=VALID_COHORT_CHOICES)
    p.add_argument("--photo-mode", default="mean", choices=["mean", "all"])
    p.add_argument("--skip-existing", action="store_true",
                    help="Skip invocations whose output dir already has a "
                         "_summary/combined_summary.csv.")
    p.add_argument("--exclude-classifiers", nargs="*", default=[],
                    choices=["logistic", "xgb", "tabpfn"],
                    help="Forward to run_fwd_rev.py: skip these classifiers "
                         "(e.g. --exclude-classifiers tabpfn to avoid Windows "
                         "subprocess console popups).")
    p.add_argument("--caliper-group", type=float, default=3.0,
                    help="Forward to run_fwd_rev.py: caliper-group window "
                         "(set 0 to skip caliper-group evaluation).")
    p.add_argument("--bg-mode", default="no_background",
                    choices=["background", "no_background"],
                    help="Forward to run_fwd_rev.py: background mode.")
    p.add_argument("--match-priority", nargs="*", default=None,
                    help="Forward to run_fwd_rev.py: HC sub-group matching priority.")
    p.add_argument("--embedding", default=None,
                    help="Forward to run_fwd_rev.py: limit to one embedding model.")
    p.add_argument("--grid-search", action="store_true",
                    help="Forward to run_fwd_rev.py: hyperparameter grid search.")
    args = p.parse_args()

    specs = args.reducers if args.reducers else DEFAULT_REDUCERS
    parsed = [parse_reducer(s) for s in specs]

    print(f"\n{'#'*70}")
    print(f"# SWEEP: {args.embedding or 'ALL'} | {args.bg_mode} | "
          f"{args.cohort_mode}")
    print(f"# Features ({len(args.feature_types)}): {args.feature_types}")
    print(f"# Reducers ({len(parsed)}): "
          f"{[label_for(k, v) for k, v in parsed]}")
    print(f"# Total invocations: {len(args.feature_types) * len(parsed)}")
    print(f"{'#'*70}\n", flush=True)

    t_start = time.time()
    ok, fail, skip = 0, [], 0
    n_feats = len(args.feature_types)
    for feat_i, feat in enumerate(args.feature_types, 1):
        for kind, value in parsed:
            if args.skip_existing:
                # Recreate the expected output dir to test existence.
                from importlib.util import spec_from_file_location, module_from_spec
                if "_imp" not in globals():
                    sp = spec_from_file_location("rfre",
                        PROJECT_ROOT / "scripts" / "embedding" /
                        "run_fwd_rev.py")
                    m = module_from_spec(sp); sp.loader.exec_module(m)
                    globals()["_imp"] = m
                imp = globals()["_imp"]
                pca = value if kind == "pca" else None
                drop = value if kind == "drop" else None
                out = imp.output_dir_for(feat, drop,
                                          args.photo_mode, pca, args.cohort_mode,
                                          embedding="arcface",
                                          bg_mode=args.bg_mode)
                if (out / "_summary" / "combined_summary.csv").exists():
                    skip += 1
                    print(f"SKIP {feat}/{label_for(kind,value)} (exists)")
                    continue
            progress = f"[feat {feat_i}/{n_feats}] "
            if run_one(feat, kind, value, args, progress=progress):
                ok += 1
            else:
                fail.append(f"{feat}/{label_for(kind, value)}")

    total = (time.time() - t_start) / 60
    print(f"\n{'='*70}\nSUMMARY: {ok} ok, {len(fail)} failed, {skip} skipped, "
          f"total {total:.1f} min")
    if fail:
        print("FAILED:")
        for f in fail:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
