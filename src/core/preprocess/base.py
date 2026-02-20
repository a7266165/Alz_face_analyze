"""
預處理 Pipeline（Facade）

整合 Detector、Selector、Aligner、Mirror 四個子模組
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import shutil

from src.config import PreprocessConfig, SELECTED_DIR, ALIGNED_DIR, MIRRORS_DIR
from .detector import FaceDetector, FaceInfo
from .selector import FaceSelector
from .aligner import FaceAligner
from .mirror import MirrorGenerator

logger = logging.getLogger(__name__)


@dataclass
class ProcessedFace:
    """處理後的臉部資料"""
    aligned: np.ndarray  # 對齊後影像（必要）
    left_mirror: Optional[np.ndarray] = None  # 左臉鏡射
    right_mirror: Optional[np.ndarray] = None  # 右臉鏡射
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
        self.aligner = FaceAligner(midline_points=config.midline_points)
        self.mirror_gen = MirrorGenerator(
            method=config.mirror_method,
            mirror_size=config.mirror_size,
            feather_px=config.feather_px,
            margin=config.margin,
            midline_points=config.midline_points,
        )

        # 設定工作區
        self._setup_workspace()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """釋放資源"""
        self.detector.close()

    def _setup_workspace(self):
        """設定工作區目錄結構"""
        if not self.config.save_intermediate or not self.config.subject_id:
            return

        # 目錄會在 _save_* 方法中自動建立
        logger.debug(f"預處理工作區: subject_id={self.config.subject_id}")

    def cleanup_workspace(self):
        """清理受試者的工作區目錄"""
        if not self.config.subject_id:
            return

        # 只清理該受試者的子目錄，不清理整個全域目錄
        for base_dir in [SELECTED_DIR, ALIGNED_DIR, MIRRORS_DIR]:
            subject_dir = base_dir / self.config.subject_id
            if subject_dir.exists():
                shutil.rmtree(subject_dir)
                logger.debug(f"已清理: {subject_dir}")

    def process(
        self,
        images: List[np.ndarray],
        image_paths: Optional[List[Path]] = None,
        histogram_mapping: Optional[np.ndarray] = None,
    ) -> List[ProcessedFace]:
        """
        完整預處理流程

        Args:
            images: 輸入影像列表 (BGR)
            image_paths: 對應的檔案路徑（可選）
            histogram_mapping: 直方圖映射表（可選，未使用）

        Returns:
            處理後的臉部資料列表
        """
        if not images:
            logger.warning("沒有輸入影像")
            return []

        logger.info(f"開始處理 {len(images)} 張影像")

        # Step 1: 偵測臉部
        face_infos = self.detector.detect_batch(images, image_paths)
        logger.info(f"成功偵測 {len(face_infos)} 張臉部")

        if not face_infos:
            logger.warning("沒有偵測到任何臉部")
            return []

        # Step 2: 選擇最正面的 n 張
        if "select" in self.config.steps:
            selected = self.selector.select_best(face_infos)
            if self.config.save_intermediate:
                self._save_selected(selected)
        else:
            selected = face_infos

        # Step 3: 處理選中的影像
        processed_faces = []
        for i, face_info in enumerate(selected):
            try:
                processed = self._process_single_face(face_info, i)
                processed_faces.append(processed)
            except Exception as e:
                logger.error(f"處理第 {i} 張臉部時失敗: {e}")
                continue

        logger.info(f"完成處理，共 {len(processed_faces)} 張成功")
        return processed_faces

    def _process_single_face(
        self,
        face_info: FaceInfo,
        index: int
    ) -> ProcessedFace:
        """處理單張臉部"""
        current_image = face_info.image
        aligned_image = None
        landmarks = face_info.landmarks

        # Step 1: 角度校正
        if "align" in self.config.steps and self.config.align_face:
            aligned_image = self.aligner.align(current_image, landmarks)
            current_image = aligned_image

            if self.config.save_intermediate:
                self._save_aligned(aligned_image, face_info, index)

            # 重新偵測對齊後的特徵點
            new_landmarks = self.detector.redetect_landmarks(current_image)
            if new_landmarks is not None:
                landmarks = new_landmarks

        # Step 2: 生成鏡射
        if "mirror" in self.config.steps:
            left_mirror, right_mirror = self.mirror_gen.generate(
                current_image, landmarks
            )
        else:
            left_mirror = current_image.copy()
            right_mirror = current_image.copy()

        # 儲存鏡射結果
        if self.config.save_intermediate:
            self._save_mirrors(left_mirror, right_mirror, face_info, index)

        return ProcessedFace(
            aligned=aligned_image if aligned_image is not None else current_image,
            left_mirror=left_mirror,
            right_mirror=right_mirror,
            original=face_info.image if self.config.save_intermediate else None,
            metadata={
                "vertex_angle_sum": float(face_info.vertex_angle_sum),
                "confidence": float(face_info.confidence),
                "path": str(face_info.path) if face_info.path else None,
                "index": index,
            },
        )

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

    def _save_aligned(
        self,
        image: np.ndarray,
        face_info: FaceInfo,
        index: int
    ):
        """儲存對齊後的影像"""
        if not self.config.save_intermediate or not self.config.subject_id:
            return

        save_dir = ALIGNED_DIR / self.config.subject_id
        save_dir.mkdir(parents=True, exist_ok=True)

        if face_info.path:
            filename = f"{face_info.path.stem}_aligned.png"
        else:
            filename = f"aligned_{index:03d}.png"

        path = save_dir / filename
        cv2.imwrite(str(path), image)
        logger.debug(f"儲存對齊影像: {path}")

    def _save_mirrors(
        self,
        left_mirror: np.ndarray,
        right_mirror: np.ndarray,
        face_info: FaceInfo,
        index: int
    ):
        """儲存鏡射影像"""
        if not self.config.save_intermediate or not self.config.subject_id:
            return

        save_dir = MIRRORS_DIR / self.config.subject_id
        save_dir.mkdir(parents=True, exist_ok=True)

        if face_info.path:
            base_name = face_info.path.stem
        else:
            base_name = f"face_{index:03d}"

        # 儲存左右鏡射
        left_path = save_dir / f"{base_name}_left.png"
        right_path = save_dir / f"{base_name}_right.png"

        cv2.imwrite(str(left_path), left_mirror)
        cv2.imwrite(str(right_path), right_mirror)
        logger.debug(f"儲存鏡射: {left_path}, {right_path}")
