"""
預處理 Pipeline（Facade）

整合 Detector、Selector、Aligner 三個子模組
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import shutil

from src.config import PreprocessConfig, SELECTED_DIR, ALIGNED_DIR, ALIGNED_BACKGROUND_DIR
from .detector import FaceDetector, FaceInfo
from .selector import FaceSelector
from .aligner import FaceStraightener

logger = logging.getLogger(__name__)


@dataclass
class ProcessedFace:
    """處理後的臉部資料"""
    aligned: np.ndarray  # 對齊後影像（必要）
    landmarks: np.ndarray  # 對齊後的特徵點 (468, 2)
    original: Optional[np.ndarray] = None  # 原始影像（除錯用）
    metadata: Dict[str, Any] = None  # 元資料


class PreprocessPipeline:
    """
    預處理 Pipeline（Facade）

    整合所有預處理步驟，提供統一的介面
    """

    def __init__(self, config: PreprocessConfig):
        """
        初始化 Pipeline

        Args:
            config: 預處理配置
        """
        self.config = config

        # 初始化各子模組
        self.detector = FaceDetector(
            detection_confidence=config.detection_confidence,
            midline_points=config.midline_points,
        )
        self.selector = FaceSelector(n_select=config.n_select)
        self.straightener = FaceStraightener(midline_points=config.midline_points)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """釋放資源"""
        self.detector.close()

    def cleanup_workspace(self):
        """清理受試者的工作區目錄"""
        if not self.config.subject_id:
            return

        # 只清理該受試者的子目錄，不清理整個全域目錄
        for base_dir in [SELECTED_DIR, ALIGNED_DIR, ALIGNED_BACKGROUND_DIR]:
            subject_dir = base_dir / self.config.subject_id
            if subject_dir.exists():
                shutil.rmtree(subject_dir)
                logger.debug(f"已清理: {subject_dir}")

    def process(
        self,
        images: List[np.ndarray],
        image_paths: Optional[List[Path]] = None,
    ) -> List[ProcessedFace]:
        """
        完整預處理流程

        Args:
            images: 輸入影像列表 (BGR)
            image_paths: 對應的檔案路徑（可選）

        Returns:
            處理後的臉部資料列表
        """
        if not images:
            logger.warning("沒有輸入影像")
            return []

        logger.info(f"開始處理 {len(images)} 張影像")

        # Step 1: 偵測臉部
        face_infos = self.detector.detect_face_batch(images, image_paths)
        logger.info(f"成功偵測 {len(face_infos)} 張臉部")

        if not face_infos:
            logger.warning("沒有偵測到任何臉部")
            return []

        # Step 2: 選擇最正面的 n 張
        if "select" in self.config.steps:
            selected = self.selector.select_most_frontal(face_infos)
            if self.config.save_intermediate:
                self._save_selected(selected)
        else:
            selected = face_infos

        # Step 3: 轉正選中的 n 張相片
        if "align" in self.config.steps:
            # 副路徑（aligned_background/）需要 pre-align 影像 + 對應 landmarks
            need_background = (
                self.config.also_save_aligned_background
                and self.config.save_intermediate
            )
            pre_align_images = (
                [f.image.copy() for f in selected] if need_background else None
            )
            pre_align_landmarks = (
                [f.landmarks.copy() for f in selected] if need_background else None
            )

            # 主路徑：apply_mask=True，沿用既有去背行為
            for face_info in selected:
                face_info.image = self.straightener.straighten_pics(
                    face_info.image, face_info.landmarks, apply_mask=True
                )

            # 重新偵測轉正後的特徵點
            aligned_images = [f.image for f in selected]
            redetected = self.detector.detect_face_batch(aligned_images)
            redetected_map = {r.index: r.landmarks for r in redetected}
            for i, face_info in enumerate(selected):
                if i in redetected_map:
                    face_info.landmarks = redetected_map[i]

            if self.config.save_intermediate:
                self._save_aligned(selected)

            # 副路徑：apply_mask=False，僅落地到 aligned_background/
            if need_background:
                self._save_aligned_background(
                    selected, pre_align_images, pre_align_landmarks
                )

        # 組裝結果
        processed_faces = [
            ProcessedFace(
                aligned=face.image,
                landmarks=face.landmarks,
                original=None,
                metadata={
                    "vertex_angle_sum": float(face.vertex_angle_sum),
                    "confidence": float(face.confidence),
                    "path": str(face.path) if face.path else None,
                    "index": i,
                },
            )
            for i, face in enumerate(selected)
        ]

        logger.info(f"完成處理，共 {len(processed_faces)} 張成功")
        return processed_faces

    # ========== 儲存函數 ==========

    def _save_selected(self, faces: List[FaceInfo]):
        """儲存選中的影像"""
        if not self.config.save_intermediate or not self.config.subject_id:
            return

        save_dir = SELECTED_DIR / self.config.subject_id
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, face in enumerate(faces):
            filename = f"selected_{i:03d}_vas_{face.vertex_angle_sum:.1f}.png"
            path = save_dir / filename
            cv2.imwrite(str(path), face.image)
            logger.debug(f"儲存選中影像: {path}")

    def _save_aligned(self, faces: List[FaceInfo]):
        """儲存對齊後的影像"""
        if not self.config.save_intermediate or not self.config.subject_id:
            return

        save_dir = ALIGNED_DIR / self.config.subject_id
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, face in enumerate(faces):
            filename = (
                f"{face.path.stem}_aligned.png" if face.path
                else f"aligned_{i:03d}.png"
            )
            path = save_dir / filename
            cv2.imwrite(str(path), face.image)
            logger.debug(f"儲存對齊影像: {path}")

    def _save_aligned_background(
        self,
        faces: List[FaceInfo],
        pre_align_images: List[np.ndarray],
        pre_align_landmarks: List[np.ndarray],
    ):
        """以 apply_mask=False 重做對齊，並寫到 ALIGNED_BACKGROUND_DIR"""
        if not self.config.subject_id:
            return

        save_dir = ALIGNED_BACKGROUND_DIR / self.config.subject_id
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, face in enumerate(faces):
            bg_image = self.straightener.straighten_pics(
                pre_align_images[i],
                pre_align_landmarks[i],
                apply_mask=False,
            )
            filename = (
                f"{face.path.stem}_aligned.png" if face.path
                else f"aligned_{i:03d}.png"
            )
            path = save_dir / filename
            cv2.imwrite(str(path), bg_image)
            logger.debug(f"儲存對齊影像（含背景）: {path}")

