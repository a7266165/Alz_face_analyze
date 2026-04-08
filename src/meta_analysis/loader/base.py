"""
資料載入基礎類別

定義 Dataset 結構和 BaseLoader Protocol
"""

from typing import List, Dict, Optional, Protocol, Tuple, Any, runtime_checkable
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Dataset:
    """資料集封裝"""
    X: np.ndarray  # 特徵矩陣
    y: np.ndarray  # 標籤
    metadata: Dict  # 元資料（模型名稱、CDR閾值等）
    subject_ids: Optional[List[str]] = None  # 受試者ID（如 P1-1, P1-2）
    base_ids: Optional[List[str]] = None  # 個案基底ID（如 P1），用於 K-fold 分組
    sample_groups: Optional[np.ndarray] = None  # 每筆樣本對應的群組索引
    data_format: str = "averaged"  # "averaged" 或 "per_image"

    def __post_init__(self):
        """驗證資料"""
        if len(self.X) != len(self.y):
            raise ValueError(f"X 和 y 長度不一致: {len(self.X)} vs {len(self.y)}")
        if self.sample_groups is not None and len(self.sample_groups) != len(self.X):
            raise ValueError(
                f"sample_groups 長度不一致: {len(self.sample_groups)} vs {len(self.X)}"
            )

    @property
    def n_samples(self) -> int:
        """樣本數"""
        return len(self.X)

    @property
    def n_features(self) -> int:
        """特徵數"""
        return self.X.shape[1] if len(self.X.shape) > 1 else 0

    @property
    def n_positive(self) -> int:
        """正樣本數（患者）"""
        return int(np.sum(self.y == 1))

    @property
    def n_negative(self) -> int:
        """負樣本數（健康）"""
        return int(np.sum(self.y == 0))


@runtime_checkable
class DataLoaderProtocol(Protocol):
    """資料載入器 Protocol"""

    def load_datasets(self) -> List[Dataset]:
        """載入資料集"""
        ...

    def load_datasets_with_stats(self) -> Tuple[List[Dataset], Dict[str, Any]]:
        """載入資料集及統計資訊"""
        ...


@dataclass
class FilterStats:
    """篩選統計"""
    total_subjects: int = 0
    after_cdr_filter: int = 0
    after_age_filter: int = 0
    after_balancing: int = 0
    n_healthy: int = 0
    n_patient: int = 0
    min_predicted_age: float = 65.0
    predicted_ages: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            'total_subjects': self.total_subjects,
            'after_cdr_filter': self.after_cdr_filter,
            'after_age_filter': self.after_age_filter,
            'after_balancing': self.after_balancing,
            'n_healthy': self.n_healthy,
            'n_patient': self.n_patient,
            'min_predicted_age': self.min_predicted_age,
            'predicted_ages': self.predicted_ages,
        }
