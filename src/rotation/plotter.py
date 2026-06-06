"""角度訊號圖繪製。"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from .angle_calc import SequenceResult


class AnglePlotter:
    """角度訊號圖繪製工具。"""

    @staticmethod
    def plot_sequence(
        result: SequenceResult,
        output_path: Path,
        fps: float = 30.0,
        figsize: Tuple[int, int] = (12, 8),
        y_range: Tuple[float, float] = (-50, 50),
    ) -> None:
        """把 yaw/pitch/roll 三軸隨時間（幀數/fps）畫成三宮格並存檔。"""
        times = np.arange(result.length) / fps

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        data = [
            (result.yaw_list, "Yaw", "green"),
            (result.pitch_list, "Pitch", "red"),
            (result.roll_list, "Roll", "blue"),
        ]
        for ax, (values, title, color) in zip(axes, data):
            ax.plot(times, values, color=color, linewidth=0.8)
            ax.set_ylabel(f"{title} (°)")
            ax.set_title(f"{title} Angle Over Time")
            ax.set_ylim(y_range)
            ax.grid(True, alpha=0.3)
            ax.legend([title], loc="upper right")

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{result.folder_name} - {result.method} Method", fontsize=14)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
