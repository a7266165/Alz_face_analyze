"""
產出 oof_scores.csv,以及(選用)內折表 inner_scores.csv。

兩者都是同一層的 sibling 檔,刻意不開子目錄:下游有數處靠 leaf 目錄名做判斷
(evaluate 用 parent.name 推 reverse 的 matching_priority、evaluate/plot 的
_xgb_segs 會把 seed_0 底下每個子目錄名當超參數解析),多一層資料夾會踩壞它們,
多一個檔案則沒有任何讀取端會受影響。
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


def report(oof, output_dir, direction, inner=None):
    """把 OOF 落地成 oof_scores.csv,按 direction 分流。回傳寫出的路徑清單。

    Args:
        oof: forward 為單一 DataFrame;reverse 為 dict[match_strategy → DataFrame]。
        output_dir: forward 直接寫此目錄;reverse 在其下開 <match_strategy>/ 子目錄。
        inner: 選用的內折表(train 的 n_inner>0 時產生),與 oof_scores.csv 同層寫成
            inner_scores.csv。僅 forward。
    """
    if direction == "forward":
        paths = _report_forward(oof, output_dir)
        if inner is not None:
            path = Path(output_dir) / "inner_scores.csv"
            inner.to_csv(path, index=False, encoding="utf-8")
            paths.append(str(path))
        return paths
    if direction == "reverse":
        return _report_reverse(oof, output_dir)
    raise ValueError(f"unknown direction: {direction!r} (expected 'forward' | 'reverse')")
