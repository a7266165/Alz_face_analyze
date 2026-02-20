"""
類型定義與 Protocol

提供專案中使用的介面定義，用於型別檢查和文件化
"""

from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    Protocol,
    runtime_checkable,
    TypedDict,
)
from dataclasses import dataclass, field
import numpy as np


# ========== 資料結構 ==========

@dataclass
class FoldResult:
    """單一 fold 的訓練結果"""
    fold_idx: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None
    train_indices: Optional[np.ndarray] = None
    test_indices: Optional[np.ndarray] = None
    feature_importance: Optional[np.ndarray] = None
    n_train_samples: int = 0
    n_test_samples: int = 0
    n_train_subjects: int = 0
    n_test_subjects: int = 0


@dataclass
class TrainingResult:
    """完整訓練結果"""
    dataset_key: str
    n_features: int
    fold_results: List[FoldResult]
    test_metrics: Dict[str, Any]
    cv_metrics: Optional[Dict[str, Any]] = None
    corrected_metrics: Optional[Dict[str, Any]] = None
    feature_names: Optional[List[str]] = None
    selected_features: Optional[List[str]] = None
    model_path: Optional[str] = None


class DatasetInfo(TypedDict, total=False):
    """資料集資訊"""
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    feature_names: List[str]
    embedding_model: str
    feature_type: str
    cdr_threshold: float


# ========== Protocol 定義 ==========

@runtime_checkable
class Extractor(Protocol):
    """
    特徵提取器協議

    所有特徵提取器（dlib, arcface, topofr 等）都應實現此協議
    """

    @property
    def model_name(self) -> str:
        """模型名稱"""
        ...

    @property
    def feature_dim(self) -> int:
        """特徵維度"""
        ...

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        從影像中提取特徵

        Args:
            image: BGR 格式的影像

        Returns:
            特徵向量，如果提取失敗則返回 None
        """
        ...

    def extract_batch(self, images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        批次提取特徵

        Args:
            images: BGR 格式的影像列表

        Returns:
            特徵向量列表
        """
        ...


@runtime_checkable
class Analyzer(Protocol):
    """
    分析器協議

    所有分析器（XGBoost, Logistic 等）都應實現此協議
    """

    def analyze(
        self,
        datasets: List[DatasetInfo],
        filter_stats: Dict[str, Any]
    ) -> Dict[str, Dict[int, TrainingResult]]:
        """
        執行分析

        Args:
            datasets: 資料集列表
            filter_stats: 篩選統計資訊

        Returns:
            結果字典，格式為 {dataset_key: {n_features: result}}
        """
        ...


@runtime_checkable
class DataLoader(Protocol):
    """
    資料載入器協議
    """

    def load_datasets(self) -> List[DatasetInfo]:
        """
        載入資料集

        Returns:
            資料集列表
        """
        ...

    def load_datasets_with_stats(self) -> Tuple[List[DatasetInfo], Dict[str, Any]]:
        """
        載入資料集及統計資訊

        Returns:
            (資料集列表, 篩選統計)
        """
        ...


# ========== 特徵處理相關類型 ==========

@dataclass
class FaceInfo:
    """臉部資訊"""
    image: np.ndarray
    landmarks: Optional[np.ndarray] = None
    path: Optional[Any] = None  # Path object
    vertex_angle_sum: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    detection_confidence: float = 0.0


@dataclass
class PreprocessResult:
    """預處理結果"""
    selected_faces: List[FaceInfo]
    aligned_faces: List[np.ndarray]
    mirrors: List[Tuple[np.ndarray, np.ndarray]]  # (left_mirror, right_mirror)
    subject_id: Optional[str] = None
