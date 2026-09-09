"""q6ds 進入點。

  python -m src.q6ds.cli build              # 三份 xlsx → CSV + 對帳報告
  python -m src.q6ds.cli train              # nested CV + 最終模型(全部 dataset/arm)
  python -m src.q6ds.cli figures            # 由既有 cv 產出重畫圖(不重訓)
  python -m src.q6ds.cli all                # build → train → figures → 總表

常用選項:--datasets a,b --arms xgb_full,age_only --repeats 20 --jobs 16
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from src.config import Q6DS_DIR, Q6DS_LOG_DIR, Q6DS_SUMMARY_FILE, q6ds_path

from .dataset import DATASETS, load_dataset, reconcile_sources, write_dataset
from .model import ARMS, arm_spec
from .plots import make_dataset_figures, plot_cross_dataset
from .train import (
    load_cv_artifacts,
    load_model,
    train_dataset,
    write_global_summary,
)


class _Log:
    """同時寫 stdout 與 workspace/q6ds/logs/run_<UTC>.log。"""

    def __init__(self):
        Q6DS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = Q6DS_LOG_DIR / f"run_{stamp}.log"
        self.fh = self.path.open("w", encoding="utf-8")

    def __call__(self, msg: str):
        line = f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def cmd_build(args, say):
    Q6DS_DIR.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        rep = write_dataset(ds)
        d = rep["duplicates_full_features"]
        say(f"[build] {ds}: {rep['n_rows']} 列 (Dx=1 {rep['n_pos']} / Dx=0 {rep['n_neg']}), "
            f"唯一特徵組合 {d['n_unique_patterns']}, 同特徵不同標籤 "
            f"{d['n_rows_in_contradictory_groups']} 列, coding={rep['coding_serial_label']}")
    rec = reconcile_sources()
    (Q6DS_DIR / "source_reconciliation.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    for pair, v in rec["pairwise"].items():
        say(f"[build] 重疊 {pair}: {v}")
    say(f"[build] 對帳寫入 {Q6DS_DIR / 'source_reconciliation.json'}")


def cmd_train(args, say):
    for ds in args.datasets:
        train_dataset(ds, arms=args.arms, n_splits=args.splits,
                      n_repeats=args.repeats, seed=args.seed,
                      n_jobs=args.jobs, log=say)
    out = write_global_summary()
    say(f"[train] 總表寫入 {Q6DS_SUMMARY_FILE}(涵蓋磁碟上所有已存在的 arm)")
    return out


def cmd_figures(args, say):
    all_summary, all_folds = [], []
    for ds in args.datasets:
        df = load_dataset(ds)
        folds, oof = load_cv_artifacts(ds, args.arms)
        from .evaluate import summarize_folds
        summary = summarize_folds(folds)
        all_summary.append(summary)

        xgb_models = {}
        for arm in args.arms:
            if arm_spec(arm)["kind"] != "xgb":
                continue
            try:
                model = load_model(ds, arm)
            except FileNotFoundError:
                continue
            xgb_models[arm] = (model, df[arm_spec(arm)["features"]])
        d = make_dataset_figures(ds, df, folds, oof, summary, xgb_models)
        say(f"[figures] {ds} → {d}")
        all_folds.append(folds)

    combined = pd.concat(all_summary, ignore_index=True)
    plot_cross_dataset(combined, Q6DS_DIR / "auroc_by_arm_and_dataset.png")
    say(f"[figures] 跨資料集比較圖 → {Q6DS_DIR / 'auroc_by_arm_and_dataset.png'}")
    _write_report(combined, pd.concat(all_folds, ignore_index=True), say)


def _write_report(combined: pd.DataFrame, folds: pd.DataFrame, say):
    # 折數直接由產出反推,不吃 CLI 參數 —— 單獨重跑 figures 時參數未必與當初訓練一致。
    n_splits, n_repeats = folds["fold"].nunique(), folds["repeat"].nunique()
    s = combined[combined["thr_kind"] == "youden_inner"]
    lines = ["# q6ds 訓練結果總表", "",
             f"- 產生時間(UTC):{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"- 外層 {n_splits}-fold × {n_repeats} repeats;閾值取自內層 OOF 的 Youden J",
             "- ± 為折間標準差,不是母體平均的信賴區間(repeated CV 的折不獨立)", "",
             "| dataset | arm | AUROC | AUPRC | balanced acc | sensitivity | specificity |",
             "|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['arm']} | {r['auroc_mean']:.4f} ± {r['auroc_sd']:.4f} "
            f"| {r['auprc_mean']:.4f} | {r['balanced_accuracy_mean']:.4f} "
            f"| {r['sensitivity_mean']:.4f} | {r['specificity_mean']:.4f} |")
    p = Q6DS_DIR / "REPORT.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"[report] {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="src.q6ds.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "train", "figures", "all"])
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help=f"逗號分隔;預設全部 ({', '.join(DATASETS)})")
    ap.add_argument("--arms", default=",".join(ARMS), help="逗號分隔;預設全部")
    ap.add_argument("--splits", type=int, default=5, help="外層與內層折數")
    ap.add_argument("--repeats", type=int, default=20, help="外層重複次數")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16, help="內層搜尋的平行度")
    args = ap.parse_args(argv)
    args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    for d in args.datasets:
        if d not in DATASETS:
            ap.error(f"unknown dataset {d!r}; 可選:{', '.join(DATASETS)}")
    for a in args.arms:
        if a not in ARMS:
            ap.error(f"unknown arm {a!r}; 可選:{', '.join(ARMS)}")

    say = _Log()
    try:
        say(f"[start] {args.command} datasets={args.datasets} arms={args.arms} "
            f"splits={args.splits} repeats={args.repeats} seed={args.seed} jobs={args.jobs}")
        if args.command in ("build", "all"):
            cmd_build(args, say)
        if args.command in ("train", "all"):
            cmd_train(args, say)
        if args.command in ("figures", "all"):
            cmd_figures(args, say)
        say(f"[done] log → {say.path}")
    finally:
        say.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
