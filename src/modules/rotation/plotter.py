"""
角度訊號圖繪製工具與批次處理函數
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.modules.rotation.angle_calc import (
    SequenceResult,
    VectorAngleCalculator,
    PnPAngleCalculator,
)


class AnglePlotter:
    """角度訊號圖繪製工具"""

    @staticmethod
    def plot_sequence(
        result: SequenceResult,
        output_path: Path,
        fps: float = 30.0,
        figsize: Tuple[int, int] = (12, 8),
        y_range: Tuple[float, float] = (-50, 50),
    ) -> None:
        """
        繪製並儲存角度訊號圖

        Args:
            result: 角度計算結果
            output_path: 輸出檔案路徑
            fps: 影片幀率 (用於計算時間軸)
            figsize: 圖片大小
            y_range: Y軸範圍
        """
        times = np.arange(result.length) / fps

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        data = [
            (result.yaw_list, 'Yaw', 'green'),
            (result.pitch_list, 'Pitch', 'red'),
            (result.roll_list, 'Roll', 'blue'),
        ]

        for ax, (values, title, color) in zip(axes, data):
            ax.plot(times, values, color=color, linewidth=0.8)
            ax.set_ylabel(f'{title} (°)')
            ax.set_title(f'{title} Angle Over Time')
            ax.set_ylim(y_range)
            ax.grid(True, alpha=0.3)
            ax.legend([title], loc='upper right')

        axes[-1].set_xlabel('Time (s)')

        fig.suptitle(f'{result.folder_name} - {result.method} Method', fontsize=14)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_comparison(
        result_vector: SequenceResult,
        result_pnp: SequenceResult,
        output_path: Path,
        fps: float = 30.0,
    ) -> None:
        """
        繪製兩種方法的比較圖

        Args:
            result_vector: 向量法結果
            result_pnp: PnP 法結果
            output_path: 輸出檔案路徑
            fps: 影片幀率
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        times_v = np.arange(result_vector.length) / fps
        times_p = np.arange(result_pnp.length) / fps

        data = [
            ('Yaw', result_vector.yaw_list, result_pnp.yaw_list),
            ('Pitch', result_vector.pitch_list, result_pnp.pitch_list),
            ('Roll', result_vector.roll_list, result_pnp.roll_list),
        ]

        for ax, (title, vec_data, pnp_data) in zip(axes, data):
            ax.plot(times_v, vec_data, label='Vector', alpha=0.7)
            ax.plot(times_p, pnp_data, label='PnP', alpha=0.7)
            ax.set_ylabel(f'{title} (°)')
            ax.set_title(f'{title} Comparison')
            ax.set_ylim(-50, 50)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

        axes[-1].set_xlabel('Time (s)')

        fig.suptitle(f'{result_vector.folder_name} - Method Comparison', fontsize=14)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def process_single_folder(
    folder_path: Path,
    output_dir_pnp: Path,
    output_dir_vector: Path,
    verbose: bool = True,
) -> Tuple[SequenceResult, SequenceResult]:
    """
    使用兩種方法處理單一資料夾

    Args:
        folder_path: 影像資料夾路徑
        output_dir_pnp: PnP 結果輸出目錄
        output_dir_vector: Vector 結果輸出目錄
        verbose: 是否顯示進度

    Returns:
        (vector_result, pnp_result)
    """
    folder_name = folder_path.name

    # Vector 方法
    if verbose:
        print(f"  [Vector] ", end="")
    vector_calc = VectorAngleCalculator()
    vector_result = vector_calc.process_folder(folder_path, verbose=verbose)
    vector_calc.close()

    AnglePlotter.plot_sequence(
        vector_result,
        output_dir_vector / f"{folder_name}.png",
    )

    # PnP 方法
    if verbose:
        print(f"  [PnP] ", end="")
    pnp_calc = PnPAngleCalculator()
    pnp_result = pnp_calc.process_folder(folder_path, verbose=verbose)
    pnp_calc.close()

    AnglePlotter.plot_sequence(
        pnp_result,
        output_dir_pnp / f"{folder_name}.png",
    )

    return vector_result, pnp_result
