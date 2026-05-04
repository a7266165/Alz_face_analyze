"""
Embedding-classifier sweep for the p_first_hc_all cohort.

Runs run_fwd_rev_embedding.py across:
    - feature_type:    original + 5 asymmetry variants (6 total)
    - reducer:         no_drop + PCA grid + dropcorr grid

For each (feature_type, reducer) combo, run_fwd_rev_embedding internally
sweeps partition x embedding x classifier x strategy = 60 cells.

GPU PCA + GPU XGB are unconditionally on inside run_fwd_rev_embedding.

Reducer naming (a single positional value, parsed):
    "no_drop"      -> no reducer
    int (e.g. 50)  -> PCA n_components fixed (GPU torch.pca_lowrank)
    float<1 (0.95) -> PCA variance ratio (sklearn fallback)
    "drop_<thr>"   -> DropCorrelatedFeatures(threshold=<thr>)

Default settings:
    --cohort-mode p_first_hc_all
    --visit-mode all       (use all qualifying HC visits per base_id)
    --photo-mode mean      (mean-pool 10 photos -> 1 vector per visit)

Usage:
    conda run -n Alz_face_main_analysis python \
        scripts/utilities/run_embedding_sweep_p_first_hc_all.py
    # subset:
    conda run -n Alz_face_main_analysis python \
        scripts/utilities/run_embedding_sweep_p_first_hc_all.py \
        --feature-types original difference \
        --reducers pca_100 drop_0.85
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def run_one(feat, kind, value, args):
    cmd = [PYTHON, str(PROJECT_ROOT / "scripts" / "experiments" /
                        "run_fwd_rev_embedding.py"),
           "--cohort-mode", args.cohort_mode,
           "--visit-mode", args.visit_mode,
           "--photo-mode", args.photo_mode,
           "--feature-type", feat]
    if kind == "pca":
        cmd += ["--pca-components", str(value)]
    elif kind == "drop":
        cmd += ["--drop-correlated-threshold", str(value)]
    label = label_for(kind, value)
    print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] {feat} / "
          f"{label}\n{'='*70}", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
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
    p.add_argument("--cohort-mode", default="p_first_hc_all",
                    choices=["default", "p_first_hc_all"])
    p.add_argument("--visit-mode", default="all", choices=["first", "all"])
    p.add_argument("--photo-mode", default="mean", choices=["mean", "all"])
    p.add_argument("--skip-existing", action="store_true",
                    help="Skip invocations whose output dir already has a "
                         "_summary/combined_summary.csv.")
    args = p.parse_args()

    specs = args.reducers if args.reducers else DEFAULT_REDUCERS
    parsed = [parse_reducer(s) for s in specs]

    print(f"Cohort mode:    {args.cohort_mode}")
    print(f"Visit mode:     {args.visit_mode}")
    print(f"Photo mode:     {args.photo_mode}")
    print(f"Feature types:  {args.feature_types}")
    print(f"Reducers ({len(parsed)}): "
          f"{[label_for(k, v) for k, v in parsed]}")
    print(f"Total invocations: {len(args.feature_types) * len(parsed)}")

    t_start = time.time()
    ok, fail, skip = 0, [], 0
    for feat in args.feature_types:
        for kind, value in parsed:
            if args.skip_existing:
                # Recreate the expected output dir to test existence.
                from importlib.util import spec_from_file_location, module_from_spec
                if "_imp" not in globals():
                    sp = spec_from_file_location("rfre",
                        PROJECT_ROOT / "scripts" / "experiments" /
                        "run_fwd_rev_embedding.py")
                    m = module_from_spec(sp); sp.loader.exec_module(m)
                    globals()["_imp"] = m
                imp = globals()["_imp"]
                pca = value if kind == "pca" else None
                drop = value if kind == "drop" else None
                out = imp.output_dir_for(feat, drop, args.visit_mode,
                                          args.photo_mode, pca, args.cohort_mode)
                if (out / "_summary" / "combined_summary.csv").exists():
                    skip += 1
                    print(f"SKIP {feat}/{label_for(kind,value)} (exists)")
                    continue
            if run_one(feat, kind, value, args):
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
