"""
scripts/prepare_features.py
完整的特徵準備 pipeline：預處理 → 特徵提取 → 儲存

功能：
1. 從 path.txt 讀取實際影像路徑
2. 使用 core.preprocess 進行預處理
3. 使用 core.feature_extract 提取特徵
4. 逐個受試者儲存特徵（支援斷點續傳）
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
import numpy as np
import cv2
from tqdm import tqdm
import json
from datetime import datetime

# 加入專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.preprocess import FacePreprocessor, ProcessedFace
from src.core.feature_extract import FeatureExtractor
from src.core.config import AnalyzeConfig

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeaturePipeline:
    """特徵準備 Pipeline（支援斷點續傳）"""
    
    def __init__(
        self,
        path_file: Path,
        output_dir: Path,
        embedding_models: List[str] = None,
        feature_types: List[str] = None,
        n_select: int = 10,
        save_intermediate: bool = True,
        max_cpu_cores: Optional[int] = None,
    ):
        """
        初始化 Pipeline
        
        Args:
            path_file: path.txt 檔案路徑
            output_dir: 輸出目錄
            embedding_models: 嵌入模型列表
            feature_types: 特徵類型列表
            n_select: 選擇多少張最正面的相片
            save_intermediate: 是否儲存中間結果
        """
        self.path_file = Path(path_file)
        self.output_dir = Path(output_dir)
        
        # 預設模型和特徵類型
        self.embedding_models = embedding_models or ["arcface", "dlib", "topofr"]
        self.feature_types = feature_types or ["difference", "average", "relative"]

        # CPU 核心數限制（避免過度使用）
        self._setup_cpu_limit(max_cpu_cores)
        
        # 讀取實際影像路徑
        self.raw_images_dir = self._read_path_file()
        
        # 預處理配置
        self.preprocess_config = AnalyzeConfig(
            n_select=n_select,
            save_intermediate=save_intermediate,
            workspace_dir=Path("workspace/preprocessing_only_topofr_no_avg_no_mirror")
        )
        

        # 建立輸出目錄結構
        self._setup_output_dirs()
        
        # 統計資訊
        self.stats = {
            'total_subjects': 0,
            'successful_subjects': 0,
            'failed_subjects': 0,
            'skipped_subjects': 0,
            'total_images': 0,
            'models_extracted': {},
            'start_time': None,
            'end_time': None
        }
    
    def _setup_cpu_limit(self, max_cpu_cores: Optional[int]):
        """
        設定 CPU 核心數限制
        
        Args:
            max_cpu_cores: 最大核心數（None = 不限制）
        """
        if max_cpu_cores is None:
            logger.info(f"CPU 核心數: 不限制")
            return
        
        logger.info(f"CPU 核心數: 限制為 {max_cpu_cores} 核心")
        
        # 設定各種執行緒數環境變數
        os.environ['OMP_NUM_THREADS'] = str(max_cpu_cores)
        os.environ['MKL_NUM_THREADS'] = str(max_cpu_cores)
        os.environ['OPENBLAS_NUM_THREADS'] = str(max_cpu_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(max_cpu_cores)
        
        # 設定 OpenCV 執行緒數
        try:
            import cv2
            cv2.setNumThreads(max_cpu_cores)
        except:
            pass
        
        # 設定 PyTorch 執行緒數
        try:
            import torch
            torch.set_num_threads(max_cpu_cores)
            torch.set_num_interop_threads(max_cpu_cores)
        except:
            pass

    def _setup_output_dirs(self):
        """建立輸出目錄結構"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 為每個模型和特徵類型建立子目錄
        for model in self.embedding_models:
            for ftype in self.feature_types:
                feature_dir = self.output_dir / model / ftype
                feature_dir.mkdir(parents=True, exist_ok=True)
    
    def _read_path_file(self) -> Path:
        """讀取 path.txt 檔案"""
        if not self.path_file.exists():
            raise FileNotFoundError(f"找不到 {self.path_file}")
        
        with open(self.path_file, 'r', encoding='utf-8') as f:
            path_str = f.read().strip()
        
        # 移除可能的引號
        path_str = path_str.strip('"').strip("'")
        
        raw_dir = Path(path_str)
        
        if not raw_dir.exists():
            logger.error(f"無法找到路徑: {raw_dir}")
            raise FileNotFoundError(f"path.txt 中的路徑不存在: {raw_dir}")
        
        logger.info(f"從 path.txt 讀取影像路徑: {raw_dir}")
        return raw_dir
    
    def _get_processed_subjects(self) -> Set[str]:
        """
        掃描輸出目錄，找出已完整處理的受試者
        （所有模型 + 所有特徵類型都有 .npy 才算完成）
        
        Returns:
            已處理的受試者 ID 集合
        """
        # 收集每個模型/特徵類型的受試者
        subject_sets = []
        
        for model in self.embedding_models:
            for ftype in self.feature_types:
                feature_dir = self.output_dir / model / ftype
                if feature_dir.exists():
                    subjects = {f.stem for f in feature_dir.glob("*.npy")}
                    subject_sets.append(subjects)
                else:
                    # 目錄不存在，沒有任何人完成
                    subject_sets.append(set())
        
        if not subject_sets:
            return set()
        
        # 取交集：只有所有組合都有的才算完成
        processed = subject_sets[0]
        for s in subject_sets[1:]:
            processed = processed & s
        
        return processed
    
    def run(self):
        """執行完整 Pipeline（支援斷點續傳）"""
        logger.info("=" * 70)
        logger.info("開始特徵準備 Pipeline（支援斷點續傳）")
        logger.info("=" * 70)
        logger.info(f"原始影像目錄: {self.raw_images_dir}")
        logger.info(f"輸出目錄: {self.output_dir}")
        logger.info(f"嵌入模型: {self.embedding_models}")
        logger.info(f"特徵類型: {self.feature_types}")
        logger.info(f"選擇相片數: {self.preprocess_config.n_select}")
        
        self.stats['start_time'] = datetime.now()
        
        # Step 1: 掃描受試者目錄
        logger.info("\n[Step 1] 掃描受試者目錄...")
        subject_dirs = self._scan_subjects()
        self.stats['total_subjects'] = len(subject_dirs)
        logger.info(f"找到 {len(subject_dirs)} 個受試者")
        
        if len(subject_dirs) == 0:
            logger.error("沒有找到任何受試者目錄")
            return
        

        # Step 2: 檢查斷點
        logger.info("\n[Step 2] 檢查處理進度...")
        processed_subjects = self._get_processed_subjects()
        
        if len(processed_subjects) > 0:
            logger.info(f"發現 {len(processed_subjects)} 個已處理的受試者")
            logger.info(f"將跳過這些受試者，從斷點繼續處理")
            self.stats['skipped_subjects'] = len(processed_subjects)
        else:
            logger.info("未發現已處理的受試者，從頭開始處理")
        
        # 過濾掉已處理的受試者
        remaining_subjects = [
            d for d in subject_dirs 
            if d.name not in processed_subjects
        ]
        
        logger.info(f"待處理受試者數: {len(remaining_subjects)}")
        
        if len(remaining_subjects) == 0:
            logger.info("所有受試者已處理完成！")
            self._print_statistics()
            return
        
        # Step 3: 初始化特徵提取器
        logger.info("\n[Step 3] 初始化特徵提取器...")
        extractor = FeatureExtractor()
        
        # Step 4: 處理每個受試者
        logger.info(f"\n[Step 4] 處理受試者影像（共 {len(remaining_subjects)} 個）...")
        
        with tqdm(remaining_subjects, desc="處理受試者") as pbar:
            for subject_dir in pbar:
                subject_id = subject_dir.name
                pbar.set_description(f"處理 {subject_id}")
                
                try:
                    # 處理單個受試者
                    features = self._process_subject(subject_dir, extractor)
                    
                    if features:
                        # 立即儲存特徵
                        self._save_subject_features(subject_id, features)
                        self.stats['successful_subjects'] += 1
                        logger.info(f"✓ {subject_id}: 特徵已儲存")
                    else:
                        self.stats['failed_subjects'] += 1
                        logger.warning(f"✗ {subject_id}: 處理失敗")
                
                except Exception as e:
                    self.stats['failed_subjects'] += 1
                    logger.error(f"✗ {subject_id}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Step 5: 統計報告
        self.stats['end_time'] = datetime.now()
        self._print_statistics()
        self._save_statistics()
        
        logger.info("\n" + "=" * 70)
        logger.info("特徵準備完成！")
        logger.info("=" * 70)
    
    def _scan_subjects(self) -> List[Path]:
        """掃描受試者目錄"""
        subject_dirs = []
        
        # 掃描三個族群的目錄
        for group_name in ["ACS", "NAD"]:
            group_path = self.raw_images_dir / "health" / group_name
            if group_path.exists():
                for subject_dir in sorted(group_path.iterdir()):
                    if subject_dir.is_dir() and self._has_images(subject_dir):
                        subject_dirs.append(subject_dir)
            else:
                logger.warning(f"找不到目錄: {group_path}")
        
        # 掃描 patient 目錄
        patient_path = self.raw_images_dir / "patient" / "good"
        if patient_path.exists():
            for subject_dir in sorted(patient_path.iterdir()):
                if subject_dir.is_dir() and self._has_images(subject_dir):
                    subject_dirs.append(subject_dir)
        else:
            logger.warning(f"找不到目錄: {patient_path}")
        
        return subject_dirs
    
    def _has_images(self, directory: Path) -> bool:
        """檢查目錄是否包含影像檔案"""
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                return True
        return False
        
    def _process_subject(
        self, 
        subject_dir: Path, 
        extractor: FeatureExtractor
    ) -> Optional[Dict]:
        """處理單個受試者"""
        subject_id = subject_dir.name
        
        images, paths = self._load_images_from_subject(subject_dir)
        if not images:
            logger.warning(f"{subject_id}: 沒有找到影像")
            return None
        
        self.stats['total_images'] += len(images)
        
        # 判斷是否需要鏡射
        need_mirror = any(ft != "origin" for ft in self.feature_types)
        
        # 動態設定處理步驟
        steps = ["select", "align"]
        if need_mirror:
            steps.append("mirror")
        
        subject_config = AnalyzeConfig(
            n_select=self.preprocess_config.n_select,
            save_intermediate=self.preprocess_config.save_intermediate,
            workspace_dir=self.preprocess_config.workspace_dir,
            subject_id=subject_id,
            steps=steps,
        )
        
        # 預處理
        with FacePreprocessor(subject_config) as preprocessor:
            try:
                processed_faces: List[ProcessedFace] = preprocessor.process(images, paths)
            except Exception as e:
                logger.warning(f"{subject_id}: 預處理失敗 - {e}")
                return None
        
        if not processed_faces:
            logger.warning(f"{subject_id}: 預處理後沒有有效臉部")
            return None
        
        # feature_type 到 methods 參數的對應
        FTYPE_TO_METHOD = {
            "difference": "differences",
            "absolute_difference": "absolute_differences",
            "average": "averages",
            "relative_differences": "relative_differences",
            "absolute_relative_differences": "absolute_relative_differences",
        }
        
        # 提取特徵
        subject_features = {}
        
        for model in self.embedding_models:
            if model not in extractor.available_models:
                continue
            
            model_features = {}
            
            try:
                # origin: 直接提取對齊後的圖
                if "origin" in self.feature_types:
                    aligned_images = [face.aligned for face in processed_faces]
                    features_dict = extractor.extract_features(aligned_images, model)
                    
                    if model in features_dict:
                        valid = [f for f in features_dict[model] if f is not None]
                        if valid:
                            model_features["origin"] = np.array(valid, dtype=np.float32)
                
                # 需要鏡射的特徵類型
                mirror_types = [ft for ft in self.feature_types if ft != "origin"]
                
                if mirror_types and need_mirror:
                    # 收集所有左右臉影像
                    left_images = [face.left_mirror for face in processed_faces]
                    right_images = [face.right_mirror for face in processed_faces]
                    
                    # 批次提取特徵
                    left_dict = extractor.extract_features(left_images, model)
                    right_dict = extractor.extract_features(right_images, model)
                    
                    if model not in left_dict or model not in right_dict:
                        logger.warning(f"{subject_id}: {model} 特徵提取失敗")
                        continue
                    
                    # 過濾掉 None（成對過濾）
                    valid_pairs = [
                        (l, r) for l, r in zip(left_dict[model], right_dict[model])
                        if l is not None and r is not None
                    ]
                    
                    if not valid_pairs:
                        logger.warning(f"{subject_id}: {model} 沒有有效特徵")
                        continue
                    
                    left_array = np.array([p[0] for p in valid_pairs])
                    right_array = np.array([p[1] for p in valid_pairs])
                    
                    # 計算不同類型的特徵
                    for ftype in mirror_types:
                        if ftype not in FTYPE_TO_METHOD:
                            logger.warning(f"未知的特徵類型: {ftype}")
                            continue
                        
                        method = FTYPE_TO_METHOD[ftype]
                        result = extractor.calculate_differences(
                            left_array,
                            right_array,
                            methods=[method]
                        )
                        model_features[ftype] = result
                
                if model_features:
                    subject_features[model] = model_features
                    if model not in self.stats['models_extracted']:
                        self.stats['models_extracted'][model] = 0
                    self.stats['models_extracted'][model] += 1
            
            except Exception as e:
                logger.warning(f"{subject_id}: {model} 特徵提取失敗 - {e}")
                continue
        
        return subject_features if subject_features else None
    
    def _load_images_from_subject(self, subject_dir: Path) -> tuple:
        """載入受試者的所有影像（去重）"""
        images = []
        paths = []
        
        # 支援的副檔名（小寫）
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        
        # 遍歷所有檔案，統一用小寫比較副檔名
        for file_path in sorted(subject_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                img = cv2.imread(str(file_path))
                if img is not None:
                    images.append(img)
                    paths.append(file_path)
        
        return images, paths
    
    def _save_subject_features(self, subject_id: str, features: Dict):
        """
        儲存單個受試者的特徵
        
        Args:
            subject_id: 受試者 ID
            features: 特徵字典 {model: {feature_type: feature_vector}}
        """
        for model in self.embedding_models:
            if model not in features:
                continue
            
            for ftype in self.feature_types:
                if ftype not in features[model]:
                    continue
                
                feature_vector = features[model][ftype]
                
                # 儲存路徑
                feature_dir = self.output_dir / model / ftype
                npy_path = feature_dir / f"{subject_id}.npy"
                
                # 儲存 .npy
                np.save(npy_path, feature_vector)
    
    def _print_statistics(self):
        """列印統計資訊"""
        logger.info("\n" + "=" * 70)
        logger.info("處理統計")
        logger.info("=" * 70)
        logger.info(f"總受試者數: {self.stats['total_subjects']}")
        logger.info(f"已跳過（斷點）: {self.stats['skipped_subjects']}")
        logger.info(f"成功處理: {self.stats['successful_subjects']}")
        logger.info(f"處理失敗: {self.stats['failed_subjects']}")
        logger.info(f"總影像數: {self.stats['total_images']}")
        
        logger.info("\n各模型提取統計:")
        for model, count in self.stats['models_extracted'].items():
            logger.info(f"  {model}: {count} 個受試者")
        
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            logger.info(f"\n總耗時: {duration}")
    
    def _save_statistics(self):
        """儲存統計資訊"""
        stats_file = self.output_dir / "feature_extraction_stats.json"
        
        stats_copy = self.stats.copy()
        if stats_copy['start_time']:
            stats_copy['start_time'] = stats_copy['start_time'].isoformat()
        if stats_copy['end_time']:
            stats_copy['end_time'] = stats_copy['end_time'].isoformat()
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_copy, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n統計資訊已儲存: {stats_file}")


def main():
    """主程式"""
    
    # 設定路徑
    path_file = Path("data/images/raw/path.txt")
    output_dir = Path("workspace/features_no_his_only_topofr_no_avg_no_mirror")
    
    # 檢查 path.txt
    if not path_file.exists():
        logger.error(f"找不到 {path_file}")
        logger.info("請在 data/images/raw/path.txt 中指定實際影像目錄路徑")
        return
    
    # 建立 Pipeline
    try:
        pipeline = FeaturePipeline(
            path_file=path_file,
            output_dir=output_dir,
            embedding_models=["topofr"],
            feature_types=["difference", "absolute_difference", "average", "relative_differences", "absolute_relative_differences", "origin"],
            n_select=10,
            save_intermediate=True,
            max_cpu_cores=2,
        )
        
        # 執行
        pipeline.run()
    
    except Exception as e:
        logger.error(f"Pipeline 執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()