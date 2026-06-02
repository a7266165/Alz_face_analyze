"""
Classification 下游 Stage-1 的「落地」—— 把 train 出來的 OOF 寫成 oof_scores.csv。
**對外只有一個 report(dispatcher)。不算指標、不標 label。**

配對評估(年齡配對 × partition × 指標)是**獨立的下游步驟**,只吃 oof_scores.csv +
demographics,與 embedding / score 生產無關,故不在這層做(這層也因此對 src/meta 零依賴)。

  - forward:單一 oof → output_dir/oof_scores.csv
  - reverse:dict[ms → oof] → 每 ms 寫 <ms>/oof_scores.csv

oof_scores.csv schema = [ID, y_true, y_score, fold],1 列/ID,是下游(評估 / Stage-2 meta)入口。
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
