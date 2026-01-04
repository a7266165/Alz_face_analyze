"""
scripts/run_analyze.py
執行完整的分析流程：載入資料 → 訓練模型 → 生成報告
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

# 加入專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.loader import DataLoader
from src.analysis.analyzer import XGBoostAnalyzer
from src.analysis.plotter import ResultPlotter

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
        min_age_range: tuple = (50, 70),
        cdr_thresholds: list = None,
        data_balancing: bool = True,
        use_all_visits: bool = False,
        n_folds: int = 5,                 # 新增
        n_drop_features: int = 5,         # 新增
        # test_size: float = 0.2,
        random_seed: int = 42,
        # feature_selection: bool = True,
        # importance_ratio: float = 0.8,
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
        self.embedding_models = embedding_models
        self.feature_types = feature_types
        self.cdr_thresholds = cdr_thresholds
        self.min_age_start, self.min_age_end = min_age_range
        self.min_ages = list(range(self.min_age_start, self.min_age_end + 1))
        self.data_balancing = data_balancing
        self.use_all_visits = use_all_visits
        self.use_cache = use_cache
        
        # Analyzer 參數
        # self.test_size = test_size
        self.random_seed = random_seed
        # self.feature_selection = feature_selection
        # self.importance_ratio = importance_ratio

        self.n_folds = n_folds
        self.n_drop_features = n_drop_features
        
        # 建立輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # self.all_results: Dict[int, Dict] = {}  # {min_age: {dataset_key: result}}

        self.all_results: Dict[int, Dict[str, Dict[int, Dict]]] = {}  
        # {min_age: {dataset_key: {n_features: result}}}

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
        logger.info(f"年齡篩選範圍: {self.min_age_start} ~ {self.min_age_end}")
        logger.info(f"  資料平衡: {self.data_balancing}")
        logger.info(f"  使用所有訪視: {self.use_all_visits}")
        logger.info(f"  使用快取: {self.use_cache}")
        logger.info("")
        logger.info("模型訓練配置:")
        logger.info(f"  CV 折數: {self.n_folds}")
        logger.info(f"  每次捨棄特徵數: {self.n_drop_features}")
        logger.info(f"  隨機種子: {self.random_seed}")
        logger.info("=" * 70)
    
    def run(self):
        """執行完整 Pipeline"""
        start_time = datetime.now()
        
        total_ages = len(self.min_ages)
        
        for i, min_age in enumerate(self.min_ages, 1):
            logger.info("\n" + "=" * 70)
            logger.info(f"[年齡閾值 {i}/{total_ages}] MIN_PREDICTED_AGE = {min_age}")
            logger.info("=" * 70)

            # 建立子目錄
            age_suffix = f"MIN_PREDICTED_AGE_{min_age}"
            models_dir = self.output_dir / "models" / age_suffix
            reports_dir = self.output_dir / "reports" / age_suffix
            models_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Step 1: 載入資料
                logger.info("\n" + "=" * 70)
                logger.info("[Step 1] 載入資料集")
                logger.info("=" * 70)
                
                datasets, filter_stats = self._load_datasets(min_age)
                
                if not datasets:
                    logger.error("沒有載入任何資料集，中止執行")
                    return
                
                logger.info(f"成功載入 {len(datasets)} 個資料集")
                
                # Step 2: 訓練模型
                logger.info("\n" + "=" * 70)
                logger.info("[Step 2] 訓練模型")
                logger.info("=" * 70)
                
                results = self._train_models(
                    datasets, 
                    filter_stats,
                    models_dir, 
                    reports_dir
                )
                
                # 儲存結果
                self.all_results[min_age] = results

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

        # 繪製圖表
        logger.info("\n" + "=" * 70)
        logger.info("繪製圖表")
        logger.info("=" * 70)
        self._plot_results()
        
        # 儲存總結
        self._save_summary()
        
        elapsed = datetime.now() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("分析完成！")
        logger.info(f"總耗時: {elapsed}")
        logger.info(f"結果已儲存至: {self.output_dir}")
        logger.info("=" * 70)
    
    def _load_datasets(self, min_age: float):
        """載入資料集"""
        try:
            loader = DataLoader(
                features_dir=self.features_dir,
                demographics_dir=self.demographics_dir,
                predicted_ages_file=self.predicted_ages_file,
                embedding_models=self.embedding_models,
                feature_types=self.feature_types,
                cdr_thresholds=self.cdr_thresholds,
                min_predicted_age=min_age, 
                data_balancing=self.data_balancing,
                use_all_visits=self.use_all_visits,
                n_bins=5,
                random_seed=self.random_seed,
                use_cache=self.use_cache,

            )
            
            datasets, filter_stats = loader.load_datasets_with_stats()
            return datasets, filter_stats
        
        except Exception as e:
            logger.error(f"載入資料集失敗: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _train_models(self, datasets, filter_stats, models_dir, reports_dir):
        """訓練模型"""
        try:
            analyzer = XGBoostAnalyzer(
                models_dir=models_dir,
                reports_dir=reports_dir,
                n_folds=self.n_folds,
                n_drop_features=self.n_drop_features,
                random_seed=self.random_seed,
            )
            
            results = analyzer.analyze(datasets, filter_stats)
            return results
            
        except Exception as e:
            logger.error(f"訓練模型失敗: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
    def _save_summary(self):
        """儲存總結報告"""
        import json
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'min_age_range': [self.min_age_start, self.min_age_end],
                'embedding_models': self.embedding_models,
                'feature_types': self.feature_types,
                'cdr_thresholds': self.cdr_thresholds,
                'data_balancing': self.data_balancing,
                'use_all_visits': self.use_all_visits,
                'n_folds': self.n_folds,
                'n_drop_features': self.n_drop_features,
            },
            'results_by_age': {}
        }
        
        for min_age, results_by_dataset in self.all_results.items():
            summary['results_by_age'][min_age] = {}
            for dataset_key, results_by_n_features in results_by_dataset.items():
                summary['results_by_age'][min_age][dataset_key] = {
                    n_feat: {
                        'test_accuracy': r['test_metrics']['accuracy'],
                        'test_mcc': r['test_metrics']['mcc'],
                        'corrected_accuracy': r.get('corrected_metrics', {}).get('accuracy') if r.get('corrected_metrics') else None,
                        'corrected_mcc': r.get('corrected_metrics', {}).get('mcc') if r.get('corrected_metrics') else None,
                    }
                    for n_feat, r in results_by_n_features.items()
                }
        
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"總結報告已儲存: {summary_path}")

    def _print_summary(self, results):
        """顯示分析總結"""
        if not results:
            logger.warning("沒有結果可以顯示")
            return
        
        logger.info(f"成功訓練的資料集數: {len(results)}")
        
        # 找出每個 dataset 的最佳特徵數（按 MCC）
        for dataset_key, results_by_n_features in results.items():
            best_n_feat = max(
                results_by_n_features.keys(),
                key=lambda k: results_by_n_features[k]['test_metrics']['mcc']
            )
            best_result = results_by_n_features[best_n_feat]
            
            logger.info(
                f"  {dataset_key}: 最佳特徵數={best_n_feat}, "
                f"MCC={best_result['test_metrics']['mcc']:.4f}, "
                f"Acc={best_result['test_metrics']['accuracy']:.4f}"
            )

    def _plot_results(self):
        """繪製結果圖表"""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plotter = ResultPlotter(self.all_results, plots_dir)
        
        # 1. 每個 dataset_key 各畫四張圖（按年齡）
        plotter.plot_individual()
        
        # 2. 所有組合在同一張圖（按年齡）
        plotter.plot_combined()
        
        # 3. 按模型分組（按年齡）
        plotter.plot_by_model()
        
        # 4. 按特徵數量（新增）
        plotter.plot_by_n_features()
        
        # 5. 篩選統計
        plotter.plot_filter_stats()
        
        logger.info(f"圖表已儲存至: {plots_dir}")

def main():
    """主程式"""
    
    # ==================== 配置參數 ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 路徑設定
    FEATURES_DIR = "workspace/features_V5"
    DEMOGRAPHICS_DIR = "data/demographics"
    PREDICTED_AGES_FILE = "workspace/predicted_ages.json"
    
    # 資料載入配置
    # EMBEDDING_MODELS = ["arcface", "dlib", "topofr"]
    EMBEDDING_MODELS = ["arcface"]
    FEATURE_TYPES = ["difference", "absolute_difference", "average", "relative_differences", "absolute_relative_differences"]
    MIN_AGE_RANGE = (60, 61)
    CDR_THRESHOLDS = [0]
    DATA_BALANCING = False
    USE_ALL_VISITS = True
    USE_CACHE = False
    
    # 模型訓練配置
    N_FOLDS = 5
    N_DROP_FEATURES = 5
    # TEST_SIZE = 0.2
    RANDOM_SEED = 42
    # FEATURE_SELECTION = True
    # IMPORTANCE_RATIO = 0.8
    
    OUTPUT_DIR = f"workspace/analysis_{timestamp}_balancing_{DATA_BALANCING}_allvisits_{USE_ALL_VISITS}"
    # ==================== 執行 Pipeline ====================
    
    pipeline = AnalysisPipeline(
        features_dir=FEATURES_DIR,
        demographics_dir=DEMOGRAPHICS_DIR,
        output_dir=OUTPUT_DIR,
        predicted_ages_file=PREDICTED_AGES_FILE,
        embedding_models=EMBEDDING_MODELS,
        feature_types=FEATURE_TYPES,
        min_age_range=MIN_AGE_RANGE,
        cdr_thresholds=CDR_THRESHOLDS,
        data_balancing=DATA_BALANCING,
        use_all_visits=USE_ALL_VISITS,
        n_folds=N_FOLDS,                    
        n_drop_features=N_DROP_FEATURES,
        # test_size=TEST_SIZE,
        random_seed=RANDOM_SEED,
        # feature_selection=FEATURE_SELECTION,
        # importance_ratio=IMPORTANCE_RATIO,
        use_cache=USE_CACHE
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()