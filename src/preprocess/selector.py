"""
臉部選擇（pure function）

依頂點夾角總和（越小越正面）選最正面的 n 張。
"""

import logging
from typing import List

from .detector import FaceInfo

logger = logging.getLogger(__name__)


def select_most_frontal(face_infos: List[FaceInfo], n_select: int = 10) -> List[FaceInfo]:
    """按頂點夾角總和（恆非負，越小越正面）升冪取前 n 張。"""
    if not face_infos:
        return []
    sorted_faces = sorted(face_infos, key=lambda x: x.vertex_angle_sum)
    selected = sorted_faces[:min(n_select, len(sorted_faces))]
    logger.info(f"從 {len(face_infos)} 張中選擇了 {len(selected)} 張最正面的臉部")
    return selected
