"""Resolve project root for all scripts, regardless of subdirectory depth."""
from pathlib import Path
import sys


def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Cannot find project root (no pyproject.toml found)")


PROJECT_ROOT = _find_project_root()
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

for p in [str(PROJECT_ROOT), str(SCRIPTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)
