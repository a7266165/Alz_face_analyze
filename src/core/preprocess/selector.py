"""
臉部選擇器

負責從多張臉部中選擇最正面的 n 張
"""

from typing import List
import logging

from .detector import FaceInfo

logger = logging.getLogger(__name__)


class FaceSelector:
    """
    臉部選擇器

    根據頂點夾角總和選擇最正面的臉部影像
    """

    def __init__(self, n_select: int = 10):
        """
        初始化選擇器

        Args:
            n_select: 要選擇的臉部數量
        """
        self.n_select = n_select

    def select_best(self, face_infos: List[FaceInfo]) -> List[FaceInfo]:
        """
        選擇最正面的 n 張臉

        按頂點夾角總和排序，選擇數值最小的（最正面）

        Args:
            face_infos: 所有臉部資訊

        Returns:
            選中的臉部資訊列表
        """
        if not face_infos:
            return []

        # 按角度絕對值排序（越小越正面）
        sorted_faces = sorted(
            face_infos,
            key=lambda x: abs(x.vertex_angle_sum)
        )

        # 選擇前 n 張
        n_select = min(self.n_select, len(sorted_faces))
        selected = sorted_faces[:n_select]

        # 記錄選擇結果
        for face in selected:
            logger.debug(
                f"選擇: 索引={face.index}, "
                f"角度={face.vertex_angle_sum:.2f}°"
            )

        logger.info(f"從 {len(face_infos)} 張中選擇了 {len(selected)} 張最正面的臉部")
        return selected

    def filter_by_angle(
        self,
        face_infos: List[FaceInfo],
        max_angle: float = 30.0
    ) -> List[FaceInfo]:
        """
        根據角度閾值篩選臉部

        Args:
            face_infos: 臉部資訊列表
            max_angle: 最大允許角度

        Returns:
            符合條件的臉部列表
        """
        return [
            f for f in face_infos
            if abs(f.vertex_angle_sum) <= max_angle
        ]
