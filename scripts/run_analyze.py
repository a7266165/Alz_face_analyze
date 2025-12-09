"""
scripts/run_analyze.py
執行完整的分析流程：載入資料 → 訓練模型 → 生成報告
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# 加入專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.loader import DataLoader
from src.analysis.analyzer import XGBoostAnalyzer

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """完整分析 Pipeline"""
    
    def __init__(
        self,
        features_dir: Path,
        demographics_dir: Path,
        output_dir: Path,
        predicted_ages_file: Path = None,
        embedding_models: list = None,
        feature_types: list = None,
        min_predicted_age: float = 65.0,
        cdr_thresholds: list = None,
        data_balancing: bool = True,
        use_all_visits: bool = False,
        test_size: float = 0.2,
        random_seed: int = 42,
        feature_selection: bool = True,
        importance_ratio: float = 0.8,
        use_cache: bool = True
    ):
        """
        初始化分析 Pipeline
        
        Args:
            features_dir: 特徵目錄
            demographics_dir: 人口學資料目錄
            output_dir: 輸出目錄
            embedding_models: 嵌入模型列表
            feature_types: 特徵類型列表
            cdr_thresholds: CDR 閾值列表
            data_balancing: 是否進行資料平衡
            use_all_visits: 是否使用所有訪視
            test_size: 測試集比例
            random_seed: 隨機種子
            feature_selection: 是否進行特徵選擇
            importance_ratio: 特徵選擇保留比例
            use_cache: 是否使用快取
        """
        self.features_dir = Path(features_dir)
        self.demographics_dir = Path(demographics_dir)
        self.output_dir = Path(output_dir)
        self.predicted_ages_file = Path(predicted_ages_file) if predicted_ages_file else None

        # DataLoader 參數
        self.embedding_models = embedding_models or ["arcface", "dlib", "topofr"]
        self.feature_types = feature_types or ["difference", "average", "relative"]
        self.cdr_thresholds = cdr_thresholds if cdr_thresholds is not None else [0.5, 1.0, 2.0]
        self.min_predicted_age = min_predicted_age
        self.data_balancing = data_balancing
        self.use_all_visits = use_all_visits
        self.use_cache = use_cache
        
        # Analyzer 參數
        self.test_size = test_size
        self.random_seed = random_seed
        self.feature_selection = feature_selection
        self.importance_ratio = importance_ratio
        
        # 建立輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 記錄配置
        self._log_configuration()
    
    def _log_configuration(self):
        """記錄配置資訊"""
        logger.info("=" * 70)
        logger.info("分析 Pipeline 配置")
        logger.info("=" * 70)
        logger.info(f"特徵目錄: {self.features_dir}")
        logger.info(f"人口學目錄: {self.demographics_dir}")
        logger.info(f"輸出目錄: {self.output_dir}")
        logger.info("")
        logger.info("資料載入配置:")
        logger.info(f"  嵌入模型: {self.embedding_models}")
        logger.info(f"  特徵類型: {self.feature_types}")
        logger.info(f"  CDR 閾值: {self.cdr_thresholds}")
        logger.info(f"  資料平衡: {self.data_balancing}")
        logger.info(f"  使用所有訪視: {self.use_all_visits}")
        logger.info(f"  使用快取: {self.use_cache}")
        logger.info("")
        logger.info("模型訓練配置:")
        logger.info(f"  測試集比例: {self.test_size}")
        logger.info(f"  隨機種子: {self.random_seed}")
        logger.info(f"  特徵選擇: {self.feature_selection}")
        logger.info(f"  特徵保留比例: {self.importance_ratio}")
        logger.info("=" * 70)
    
    def run(self):
        """執行完整 Pipeline"""
        start_time = datetime.now()
        
        try:
            # Step 1: 載入資料
            logger.info("\n" + "=" * 70)
            logger.info("[Step 1] 載入資料集")
            logger.info("=" * 70)
            
            datasets = self._load_datasets()
            
            if not datasets:
                logger.error("沒有載入任何資料集，中止執行")
                return
            
            logger.info(f"成功載入 {len(datasets)} 個資料集")
            
            # Step 2: 訓練模型
            logger.info("\n" + "=" * 70)
            logger.info("[Step 2] 訓練模型")
            logger.info("=" * 70)
            
            results = self._train_models(datasets)
            
            if not results:
                logger.error("沒有成功訓練任何模型")
                return
            
            logger.info(f"成功訓練 {len(results)}/{len(datasets)} 個模型")
            
            # Step 3: 顯示總結
            logger.info("\n" + "=" * 70)
            logger.info("[Step 3] 分析總結")
            logger.info("=" * 70)
            
            self._print_summary(results)
            
            # 完成
            elapsed = datetime.now() - start_time
            logger.info("\n" + "=" * 70)
            logger.info("分析完成！")
            logger.info(f"總耗時: {elapsed}")
            logger.info(f"結果已儲存至: {self.output_dir}")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"Pipeline 執行失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_datasets(self):
        """載入資料集"""
        try:
            loader = DataLoader(
                features_dir=self.features_dir,
                demographics_dir=self.demographics_dir,
                predicted_ages_file=self.predicted_ages_file,
                embedding_models=self.embedding_models,
                feature_types=self.feature_types,
                cdr_thresholds=self.cdr_thresholds,
                min_predicted_age=self.min_predicted_age, 
                data_balancing=self.data_balancing,
                use_all_visits=self.use_all_visits,
                n_bins=5,
                random_seed=self.random_seed,
                use_cache=self.use_cache,

            )
            
            datasets = loader.load_datasets()
            return datasets
        
        except Exception as e:
            logger.error(f"載入資料集失敗: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _train_models(self, datasets):
        """訓練模型"""
        try:
            analyzer = XGBoostAnalyzer(
                output_dir=self.output_dir,
                test_size=self.test_size,
                random_seed=self.random_seed,
                feature_selection=self.feature_selection,
                importance_ratio=self.importance_ratio
            )
            
            results = analyzer.analyze(datasets)
            return results
            
        except Exception as e:
            logger.error(f"訓練模型失敗: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _print_summary(self, results):
        """顯示分析總結"""
        if not results:
            logger.warning("沒有結果可以顯示")
            return
        
        # 統計資訊
        logger.info(f"成功訓練的模型數: {len(results)}")
        
        # 找出最佳模型（按 MCC）
        best_mcc_key = max(results.keys(), key=lambda k: results[k]['test_metrics']['mcc'])
        best_mcc = results[best_mcc_key]
        
        logger.info("\n最佳模型（按 MCC）:")
        logger.info(f"  資料集: {best_mcc_key}")
        logger.info(f"  測試 MCC: {best_mcc['test_metrics']['mcc']:.4f}")
        logger.info(f"  測試準確率: {best_mcc['test_metrics']['accuracy']:.4f}")
        logger.info(f"  測試 F1: {best_mcc['test_metrics']['f1']:.4f}")
        if best_mcc['test_metrics'].get('auc'):
            logger.info(f"  測試 AUC: {best_mcc['test_metrics']['auc']:.4f}")
        
        # 找出最佳模型（按準確率）
        best_acc_key = max(results.keys(), key=lambda k: results[k]['test_metrics']['accuracy'])
        if best_acc_key != best_mcc_key:
            best_acc = results[best_acc_key]
            logger.info("\n最佳模型（按準確率）:")
            logger.info(f"  資料集: {best_acc_key}")
            logger.info(f"  測試準確率: {best_acc['test_metrics']['accuracy']:.4f}")
            logger.info(f"  測試 MCC: {best_acc['test_metrics']['mcc']:.4f}")
        
        # 各模型效能統計
        logger.info("\n各嵌入模型平均效能:")
        model_stats = {}
        for key, result in results.items():
            model = result['metadata']['model']
            if model not in model_stats:
                model_stats[model] = {'mcc': [], 'accuracy': []}
            model_stats[model]['mcc'].append(result['test_metrics']['mcc'])
            model_stats[model]['accuracy'].append(result['test_metrics']['accuracy'])
        
        for model, stats in model_stats.items():
            avg_mcc = sum(stats['mcc']) / len(stats['mcc'])
            avg_acc = sum(stats['accuracy']) / len(stats['accuracy'])
            logger.info(f"  {model}: MCC={avg_mcc:.4f}, Accuracy={avg_acc:.4f}")


def main():
    """主程式"""
    
    # ==================== 配置參數 ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 路徑設定
    FEATURES_DIR = "workspace/features"
    DEMOGRAPHICS_DIR = "data/demographics"
    OUTPUT_DIR = f"workspace/analysis_{timestamp}"
    PREDICTED_AGES_FILE = "workspace/predicted_ages.json"
    
    # 資料載入配置
    EMBEDDING_MODELS = ["arcface", "dlib", "topofr"]
    FEATURE_TYPES = ["difference", "average", "relative"]
    MIN_PREDICTED_AGE = 65.0
    CDR_THRESHOLDS = [0.5, 1.0, 2.0]
    DATA_BALANCING = False
    USE_ALL_VISITS = False
    USE_CACHE = False
    
    # 模型訓練配置
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    FEATURE_SELECTION = True
    IMPORTANCE_RATIO = 0.8
    
    # ==================== 執行 Pipeline ====================
    
    pipeline = AnalysisPipeline(
        features_dir=FEATURES_DIR,
        demographics_dir=DEMOGRAPHICS_DIR,
        output_dir=OUTPUT_DIR,
        predicted_ages_file=PREDICTED_AGES_FILE,
        embedding_models=EMBEDDING_MODELS,
        feature_types=FEATURE_TYPES,
        min_predicted_age=MIN_PREDICTED_AGE,
        cdr_thresholds=CDR_THRESHOLDS,
        data_balancing=DATA_BALANCING,
        use_all_visits=USE_ALL_VISITS,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED,
        feature_selection=FEATURE_SELECTION,
        importance_ratio=IMPORTANCE_RATIO,
        use_cache=USE_CACHE
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()