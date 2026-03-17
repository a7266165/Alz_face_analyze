"""
Scripts 共用工具函數
"""

from pathlib import Path


def find_latest_dir(workspace: Path, prefix: str) -> Path:
    """
    在 workspace 下尋找最新的符合 prefix 的目錄（依修改時間排序）。

    Args:
        workspace: workspace 目錄路徑
        prefix: 目錄名稱前綴，如 "analysis_"、"tabpfn_meta_analysis_"

    Returns:
        最新的目錄路徑

    Raises:
        FileNotFoundError: 找不到符合條件的目錄
    """
    candidates = sorted(
        [d for d in workspace.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"在 {workspace} 中找不到以 '{prefix}' 開頭的目錄"
        )
    return candidates[0]
