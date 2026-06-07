"""
產出 oof_scores.csv。
"""
from pathlib import Path


def _report_forward(oof, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "oof_scores.csv"
    oof.to_csv(path, index=False, encoding="utf-8")
    return [str(path)]


def _report_reverse(oof_by_ms, output_dir):
    base = Path(output_dir)
    paths = []
    for ms, oof in oof_by_ms.items():
        ms_dir = base / ms
        ms_dir.mkdir(parents=True, exist_ok=True)
        path = ms_dir / "oof_scores.csv"
        oof.to_csv(path, index=False, encoding="utf-8")
        paths.append(str(path))
    return paths


def report(oof, output_dir, direction):
    """把 OOF 落地成 oof_scores.csv,按 direction 分流。回傳寫出的路徑清單。

    Args:
        oof: forward 為單一 DataFrame;reverse 為 dict[match_strategy → DataFrame]。
        output_dir: forward 直接寫此目錄;reverse 在其下開 <match_strategy>/ 子目錄。
    """
    if direction == "forward":
        return _report_forward(oof, output_dir)
    if direction == "reverse":
        return _report_reverse(oof, output_dir)
    raise ValueError(f"unknown direction: {direction!r} (expected 'forward' | 'reverse')")
