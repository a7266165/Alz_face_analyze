"""
資料載入與篩選模組
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pickle

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """資料集封裝"""
    X: np.ndarray  # 特徵矩陣
    y: np.ndarray  # 標籤
    metadata: Dict  # 元資料（模型名稱、CDR閾值等）
    subject_ids: Optional[List[str]] = None  # 受試者ID
    
    def __post_init__(self):
        """驗證資料"""
        if len(self.X) != len(self.y):
            raise ValueError(f"X 和 y 長度不一致: {len(self.X)} vs {len(self.y)}")


class DataLoader:
    """資料載入器"""
    
    def __init__(
        self,
        features_dir: Path,
        demographics_dir: Path,
        embedding_models: List[str] = None,
        feature_types: List[str] = None,
        cdr_thresholds: List[float] = None,
        data_balancing: bool = True,
        use_all_visits: bool = True,
        n_bins: int = 5,
        random_seed: int = 42,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """
        初始化資料載入器
        
        Args:
            features_dir: 特徵目錄（workspace/features/）
            demographics_dir: 人口學資料目錄（data/demographics/）
            embedding_models: 使用的嵌入模型列表
            feature_types: 特徵類型 ["difference", "average", "relative"]
            cdr_thresholds: CDR 篩選閾值列表
            data_balancing: 是否進行資料平衡（年齡）
            use_all_visits: 是否使用所有訪視（False = 只用最新）
            n_bins: 年齡分箱數量
            random_seed: 隨機種子
            use_cache: 是否使用快取
            cache_dir: 快取目錄
        """
        self.features_dir = Path(features_dir)
        self.demographics_dir = Path(demographics_dir)
        
        # 預設值
        self.embedding_models = embedding_models or ["arcface", "dlib", "topofr"]
        self.feature_types = feature_types or ["difference", "average", "relative"]
        self.cdr_thresholds = cdr_thresholds if cdr_thresholds is not None else [0.5, 1.0, 2.0]
        self.data_balancing = data_balancing
        self.use_all_visits = use_all_visits
        self.n_bins = n_bins
        self.random_seed = random_seed
        
        # 快取設定
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir or "workspace/cache")
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 載入的資料
        self._features_cache = {}
        self._demographics_cache = None
    
    # ========== 主要載入方法 ==========
    
    def load_datasets(self) -> List[Dataset]:
        """
        載入所有配置的資料集
        
        Returns:
            資料集列表
        """
        logger.info("開始載入資料集...")
        logger.info(f"嵌入模型: {self.embedding_models}")
        logger.info(f"特徵類型: {self.feature_types}")
        logger.info(f"CDR 閾值: {self.cdr_thresholds}")
        logger.info(f"資料平衡: {self.data_balancing}")
        logger.info(f"使用所有訪視: {self.use_all_visits}")
        
        datasets = []
        
        # 載入人口學資料
        demographics = self._load_demographics()
        
        # 為每種配置創建資料集
        for model in self.embedding_models:
            for feature_type in self.feature_types:
                for cdr_threshold in self.cdr_thresholds:
                    try:
                        dataset = self._create_dataset(
                            model=model,
                            feature_type=feature_type,
                            cdr_threshold=cdr_threshold,
                            demographics=demographics
                        )
                        datasets.append(dataset)
                        
                        logger.info(
                            f"✓ 載入資料集: {model}-{feature_type}-CDR{cdr_threshold} "
                            f"({len(dataset.X)} 樣本)"
                        )
                    except Exception as e:
                        logger.warning(
                            f"✗ 跳過資料集 {model}-{feature_type}-CDR{cdr_threshold}: {e}"
                        )
        
        logger.info(f"成功載入 {len(datasets)} 個資料集")
        return datasets
    
    # ========== 內部載入方法 ==========
    
    def _load_demographics(self) -> pd.DataFrame:
        """載入人口學資料"""
        if self._demographics_cache is not None:
            return self._demographics_cache
        
        # 嘗試從快取載入
        if self.use_cache:
            cache_path = self.cache_dir / "demographics.pkl"
            if cache_path.exists():
                logger.debug("從快取載入人口學資料")
                with open(cache_path, 'rb') as f:
                    self._demographics_cache = pickle.load(f)
                return self._demographics_cache
        
        logger.info("載入人口學資料...")
        
        # 載入三個 CSV 檔案
        dfs = []
        for csv_name in ["ACS.csv", "NAD.csv", "P.csv"]:
            csv_path = self.demographics_dir / csv_name
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['group'] = csv_name.replace('.csv', '')
                dfs.append(df)
                logger.debug(f"載入 {csv_name}: {len(df)} 筆")
            else:
                logger.warning(f"找不到 {csv_path}")
        
        if not dfs:
            raise FileNotFoundError("找不到任何人口學資料")
        
        # 合併
        demographics = pd.concat(dfs, ignore_index=True)
        
        # 處理性別欄位（F=0, M=1）
        if 'Sex' in demographics.columns:
            demographics['Sex'] = demographics['Sex'].map({'F': 0, 'M': 1})
            demographics['Sex'] = pd.to_numeric(demographics['Sex'], errors='coerce')
        
        # 確保 Age 是數值
        demographics['Age'] = pd.to_numeric(demographics['Age'], errors='coerce')
        
        # 儲存快取
        if self.use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(demographics, f)
        
        self._demographics_cache = demographics
        logger.info(f"載入人口學資料完成: {len(demographics)} 筆")
        return demographics
    
    def _load_features(self, model: str, feature_type: str) -> Dict[str, np.ndarray]:
        """
        載入特徵（從逐個受試者的檔案）
        
        Args:
            model: 嵌入模型名稱
            feature_type: 特徵類型
        
        Returns:
            {subject_id: feature_vector}
        """
        cache_key = f"{model}_{feature_type}"
        
        # 檢查記憶體快取
        if cache_key in self._features_cache:
            return self._features_cache[cache_key]
        
        # 嘗試從檔案快取載入
        if self.use_cache:
            cache_path = self.cache_dir / f"features_{cache_key}.pkl"
            if cache_path.exists():
                logger.debug(f"從快取載入特徵: {cache_key}")
                with open(cache_path, 'rb') as f:
                    features = pickle.load(f)
                self._features_cache[cache_key] = features
                return features
        
        # 從目錄載入逐個受試者的特徵
        feature_dir = self.features_dir / model / feature_type
        
        if not feature_dir.exists():
            raise FileNotFoundError(f"找不到特徵目錄: {feature_dir}")
        
        logger.debug(f"載入特徵: {cache_key}")
        
        features = {}
        npy_files = list(feature_dir.glob("*.npy"))
        
        if len(npy_files) == 0:
            raise FileNotFoundError(f"目錄中沒有 .npy 檔案: {feature_dir}")
        
        for npy_file in npy_files:
            subject_id = npy_file.stem  # 檔名即為 subject_id
            try:
                feature_vector = np.load(npy_file)
                features[subject_id] = feature_vector
            except Exception as e:
                logger.warning(f"載入 {npy_file.name} 失敗: {e}")
                continue
        
        if len(features) == 0:
            raise ValueError(f"沒有成功載入任何特徵: {feature_dir}")
        
        logger.debug(f"成功載入 {len(features)} 個受試者的特徵")
        
        # 儲存快取
        if self.use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
            logger.debug(f"特徵已快取: {cache_path}")
        
        self._features_cache[cache_key] = features
        return features
    
    def _create_dataset(
        self,
        model: str,
        feature_type: str,
        cdr_threshold: float,
        demographics: pd.DataFrame
    ) -> Dataset:
        """創建單一資料集"""
        
        # 載入特徵
        features = self._load_features(model, feature_type)
        
        # 篩選人口學資料
        filtered_demo = self._filter_demographics(
            demographics=demographics,
            cdr_threshold=cdr_threshold
        )
        
        # 資料平衡（如果啟用）
        if self.data_balancing:
            filtered_demo = self._apply_data_balancing(filtered_demo)
        
        # 對齊特徵與標籤
        X, y, subject_ids = self._align_features_labels(features, filtered_demo)
        
        # 建立元資料
        metadata = {
            'model': model,
            'feature_type': feature_type,
            'cdr_threshold': cdr_threshold,
            'data_balancing': self.data_balancing,
            'use_all_visits': self.use_all_visits,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'class_distribution': {
                'negative': int(np.sum(y == 0)),
                'positive': int(np.sum(y == 1))
            }
        }
        
        return Dataset(
            X=X,
            y=y,
            metadata=metadata,
            subject_ids=subject_ids
        )
    
    # ========== 資料篩選與平衡 ==========
    
    def _filter_demographics(
        self,
        demographics: pd.DataFrame,
        cdr_threshold: float
    ) -> pd.DataFrame:
        """
        篩選人口學資料
        
        規則：
        1. 基本分類：
           - ACS → label = 0 (健康)
           - NAD → label = 0 (健康)
           - P → label = 1 (病患)
        
        2. CDR 二次篩選（當 cdr_threshold 不為 None 時）：
           - NAD：保留 Global_CDR <= threshold 的個案
           - P：保留 Global_CDR >= threshold 的個案
           - Global_CDR 為空值時默認為 0
           - ACS 不做 CDR 篩選
        
        Args:
            demographics: 原始人口學資料（包含 group 欄位）
            cdr_threshold: CDR 閾值（None 表示不做 CDR 篩選）
        
        Returns:
            篩選後的資料（包含 label 欄位）
        """
        df = demographics.copy()
        
        # 確保 Global_CDR 欄位為數值型態，空值填充為 0
        if 'Global_CDR' in df.columns:
            df['Global_CDR'] = pd.to_numeric(df['Global_CDR'], errors='coerce')
            df['Global_CDR'] = df['Global_CDR'].fillna(0)  # 空值默認為 0
        
        # Step 1: 設定基本標籤
        def assign_label(group):
            if group in ['ACS', 'NAD']:
                return 0  # 健康
            elif group == 'P':
                return 1  # 病患
            else:
                logger.warning(f"未知的 group: {group}")
                return 0
        
        df['label'] = df['group'].apply(assign_label)
        
        original_count = len(df)
        
        # Step 2: CDR 二次篩選（如果需要）
        if cdr_threshold is not None:
            logger.debug(f"應用 CDR 篩選（閾值 = {cdr_threshold}）")
            
            # 為每個 group 建立篩選條件
            mask = pd.Series([True] * len(df), index=df.index)
            
            for group in df['group'].unique():
                group_mask = df['group'] == group
                
                if group == 'ACS':
                    # ACS 不做 CDR 篩選，全部保留
                    continue
                
                elif group == 'NAD':
                    # NAD: 保留 CDR <= threshold 的健康個案
                    if 'Global_CDR' in df.columns:
                        mask[group_mask] = df.loc[group_mask, 'Global_CDR'] <= cdr_threshold
                    # 如果沒有 CDR 欄位，保留所有 NAD
                
                elif group == 'P':
                    # P: 保留 CDR >= threshold 的病患個案
                    if 'Global_CDR' in df.columns:
                        mask[group_mask] = df.loc[group_mask, 'Global_CDR'] >= cdr_threshold
                    # 如果沒有 CDR 欄位，保留所有 P
            
            df = df[mask].copy()
            
            filtered_count = len(df)
            logger.debug(
                f"CDR 篩選: {original_count} → {filtered_count} 筆 "
                f"(保留 {filtered_count/original_count*100:.1f}%)"
            )
        
        # Step 3: 只保留最新訪視（如果設定）
        if not self.use_all_visits:
            df = self._keep_latest_visit(df)
        
        logger.debug(
            f"最終篩選結果: {len(df)} 筆 "
            f"(label=0: {(df['label']==0).sum()}, label=1: {(df['label']==1).sum()})"
        )
        
        return df
    
    def _keep_latest_visit(self, df: pd.DataFrame) -> pd.DataFrame:
        """保留每個受試者的最新訪視"""
        # 假設 ID 格式為 "ACS1-1", "NAD10-2", "P1-3" 等
        df = df.copy()
        df['base_id'] = df['ID'].str.extract(r'^([A-Z]+\d+)', expand=False)
        df['visit'] = df['ID'].str.extract(r'-(\d+)$', expand=False).astype(float)
        
        # 保留每個 base_id 的最大訪視號
        df = df.sort_values('visit', ascending=False).groupby('base_id').first().reset_index(drop=True)
        
        # 清理臨時欄位
        df = df.drop(columns=['base_id', 'visit'], errors='ignore')
        
        return df
    
    def _apply_data_balancing(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """
        執行資料平衡（基於年齡）
        
        策略：
        1. 計算健康組和病患組的整體比例 R
        2. 將年齡分成 n_bins 個箱
        3. 在每個箱內，按比例 R 抽樣（而非取 min）
        4. 保留盡可能多的樣本，同時維持整體年齡分佈平衡
        
        Args:
            demographics: 篩選後的人口學資料
        
        Returns:
            平衡後的資料
        """
        logger.info("執行資料平衡...")
        
        df = demographics.copy()
        
        if len(df) == 0:
            logger.warning("沒有資料可供平衡")
            return df
        
        # 分離健康組和病患組
        health_group = df[df['label'] == 0].copy()
        patient_group = df[df['label'] == 1].copy()
        
        n_health = len(health_group)
        n_patient = len(patient_group)
        
        if n_health == 0 or n_patient == 0:
            logger.warning("健康組或病患組為空，跳過平衡")
            return df
        
        # 計算整體目標比例
        target_ratio = min(n_health, n_patient) / max(n_health, n_patient)
        majority_is_health = n_health > n_patient
        
        logger.debug(
            f"原始樣本數: 健康組 {n_health}, 病患組 {n_patient}, "
            f"目標比例 {target_ratio:.2f}"
        )
        
        # 將所有資料合併以建立統一的年齡分箱
        all_ages = pd.concat([health_group['Age'], patient_group['Age']])
        
        # 建立年齡分箱
        try:
            age_bins = pd.qcut(
                all_ages, 
                q=self.n_bins, 
                duplicates='drop',
                retbins=True
            )[1]
        except ValueError:
            # 如果無法分成指定箱數，自動調整
            n_bins_effective = min(self.n_bins, all_ages.nunique())
            age_bins = pd.qcut(
                all_ages, 
                q=n_bins_effective, 
                duplicates='drop',
                retbins=True
            )[1]
            logger.debug(f"年齡箱數自動調整為 {n_bins_effective}")
        
        # 將分箱應用到兩個組別
        health_group['age_bin'] = pd.cut(health_group['Age'], bins=age_bins, include_lowest=True)
        patient_group['age_bin'] = pd.cut(patient_group['Age'], bins=age_bins, include_lowest=True)
        
        # 在每個年齡箱中進行平衡
        balanced_dfs = []
        rng = np.random.RandomState(self.random_seed)
        
        for bin_val in health_group['age_bin'].cat.categories:
            health_in_bin = health_group[health_group['age_bin'] == bin_val]
            patient_in_bin = patient_group[patient_group['age_bin'] == bin_val]
            
            n_health_bin = len(health_in_bin)
            n_patient_bin = len(patient_in_bin)
            
            if n_health_bin == 0 or n_patient_bin == 0:
                # 如果某個箱內只有一組，全部保留
                if n_health_bin > 0:
                    balanced_dfs.append(health_in_bin)
                if n_patient_bin > 0:
                    balanced_dfs.append(patient_in_bin)
                continue
            
            # 根據整體比例決定抽樣策略
            if majority_is_health:
                # 健康組較多，從健康組抽樣
                target_health = int(n_patient_bin / target_ratio)
                target_health = min(target_health, n_health_bin)  # 不超過實際數量
                
                if n_health_bin > target_health:
                    health_sample = health_in_bin.sample(n=target_health, random_state=rng)
                else:
                    health_sample = health_in_bin
                
                balanced_dfs.append(health_sample)
                balanced_dfs.append(patient_in_bin)  # 病患組全保留
                
            else:
                # 病患組較多，從病患組抽樣
                target_patient = int(n_health_bin / target_ratio)
                target_patient = min(target_patient, n_patient_bin)  # 不超過實際數量
                
                if n_patient_bin > target_patient:
                    patient_sample = patient_in_bin.sample(n=target_patient, random_state=rng)
                else:
                    patient_sample = patient_in_bin
                
                balanced_dfs.append(health_in_bin)  # 健康組全保留
                balanced_dfs.append(patient_sample)
            
            logger.debug(
                f"箱 {bin_val}: "
                f"健康組 {n_health_bin}→{len(balanced_dfs[-2]) if majority_is_health else len(balanced_dfs[-2])}, "
                f"病患組 {n_patient_bin}→{len(balanced_dfs[-1]) if majority_is_health else len(balanced_dfs[-1])}"
            )
        
        if not balanced_dfs:
            logger.warning("沒有任何箱可以配對")
            return pd.DataFrame()
        
        result = pd.concat(balanced_dfs, ignore_index=True)
        
        # 清理臨時欄位
        result = result.drop(columns=['age_bin'], errors='ignore')
        
        final_health = len(result[result['label'] == 0])
        final_patient = len(result[result['label'] == 1])
        
        logger.info(
            f"資料平衡完成: {len(df)} → {len(result)} 筆 "
            f"(健康組 {n_health}→{final_health}, 病患組 {n_patient}→{final_patient})"
        )
        
        return result
    
    def _align_features_labels(
        self,
        features: Dict[str, np.ndarray],
        demographics: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        對齊特徵與標籤
        
        Args:
            features: {subject_id: feature_vector}
            demographics: 人口學資料（包含 label）
        
        Returns:
            X, y, subject_ids
        """
        X_list = []
        y_list = []
        subject_ids = []
        
        for _, row in demographics.iterrows():
            subject_id = str(row['ID'])
            
            if subject_id in features:
                X_list.append(features[subject_id])
                y_list.append(row['label'])
                subject_ids.append(subject_id)
        
        if not X_list:
            raise ValueError("沒有任何樣本的特徵與標籤對齊")
        
        return np.array(X_list), np.array(y_list), subject_ids