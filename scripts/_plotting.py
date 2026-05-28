"""Shared matplotlib setup for all plot scripts — import before pyplot."""
import matplotlib
matplotlib.use("Agg")


def ensure_output_dir(path):
    """Create output directory if it doesn't exist, return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
