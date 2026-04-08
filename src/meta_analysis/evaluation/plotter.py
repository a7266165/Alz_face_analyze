"""
src/analysis/plotter.py
結果視覺化
"""

import logging
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ResultPlotter:
    """結果繪圖器"""
    
    METRICS = ['accuracy', 'mcc', 'sensitivity', 'specificity']
    METRIC_LABELS = {
        'accuracy': 'Accuracy',
        'mcc': 'MCC',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity'
    }
    
    def __init__(self, all_results: Dict[int, Dict[str, Dict[int, Dict]]], plots_dir: Path):
        """
        Args:
            all_results: {min_age: {dataset_key: result}}
            plots_dir: 輸出目錄
        """
        self.all_results = all_results
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 提取所有 dataset_keys
        self.dataset_keys = set()
        for results in all_results.values():
            self.dataset_keys.update(results.keys())
        self.dataset_keys = sorted(self.dataset_keys)
        
        # 年齡列表
        self.ages = sorted(all_results.keys())

        # 提取所有特徵數量
        self.n_features_list = set()
        for results_by_dataset in all_results.values():
            for results_by_n_features in results_by_dataset.values():
                self.n_features_list.update(results_by_n_features.keys())
        self.n_features_list = sorted(self.n_features_list, reverse=True)  # 大到小

    def plot_by_n_features(self):
        """按特徵數量繪製趨勢圖（X軸=特徵數，Y軸=指標）"""
        n_features_dir = self.plots_dir / "by_n_features"
        n_features_dir.mkdir(exist_ok=True)
        
        for min_age in self.ages:
            results_by_dataset = self.all_results.get(min_age, {})
            
            for dataset_key in self.dataset_keys:
                results_by_n_features = results_by_dataset.get(dataset_key, {})
                
                if not results_by_n_features:
                    continue
                
                n_features_sorted = sorted(results_by_n_features.keys(), reverse=True)
                
                # 校正前（訓練+測試）
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'{dataset_key} (Age >= {min_age})', fontsize=14)
                
                for idx, metric in enumerate(self.METRICS):
                    ax = axes[idx // 2, idx % 2]
                    
                    # 測試集
                    test_values = [
                        results_by_n_features[n]['test_metrics'].get(metric, np.nan)
                        for n in n_features_sorted
                    ]
                    ax.plot(n_features_sorted, test_values, 'o-', linewidth=2, 
                            markersize=4, label='Test', color='blue')
                    
                    # 訓練集
                    train_values = [
                        results_by_n_features[n].get('train_metrics', {}).get(metric, np.nan)
                        for n in n_features_sorted
                    ]
                    ax.plot(n_features_sorted, train_values, 's--', linewidth=2, 
                            markersize=4, label='Train', color='green', alpha=0.7)
                    
                    ax.set_xlabel('Number of Features')
                    ax.set_ylabel(self.METRIC_LABELS[metric])
                    ax.set_title(self.METRIC_LABELS[metric])
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1)
                    ax.invert_xaxis()
                    ax.legend()
                
                plt.tight_layout()
                plt.savefig(n_features_dir / f"age{min_age}_{dataset_key}.png", dpi=150)
                plt.close()
                
                # 校正後（訓練+測試）
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'{dataset_key} (Age >= {min_age}, Corrected)', fontsize=14)
                
                for idx, metric in enumerate(self.METRICS):
                    ax = axes[idx // 2, idx % 2]
                    
                    # 測試集（校正後）
                    test_values = []
                    for n in n_features_sorted:
                        result = results_by_n_features[n]
                        if result.get('corrected_metrics'):
                            test_values.append(result['corrected_metrics'].get(metric, np.nan))
                        else:
                            test_values.append(np.nan)
                    
                    ax.plot(n_features_sorted, test_values, 'o-', linewidth=2, 
                            markersize=4, label='Test (Corrected)', color='orange')
                    
                    # 訓練集（不校正，因為篩選只影響測試）
                    train_values = [
                        results_by_n_features[n].get('train_metrics', {}).get(metric, np.nan)
                        for n in n_features_sorted
                    ]
                    ax.plot(n_features_sorted, train_values, 's--', linewidth=2, 
                            markersize=4, label='Train', color='green', alpha=0.7)
                    
                    ax.set_xlabel('Number of Features')
                    ax.set_ylabel(self.METRIC_LABELS[metric])
                    ax.set_title(f'{self.METRIC_LABELS[metric]} (Corrected)')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1)
                    ax.invert_xaxis()
                    ax.legend()
                
                plt.tight_layout()
                plt.savefig(n_features_dir / f"age{min_age}_{dataset_key}_corrected.png", dpi=150)
                plt.close()
        
        logger.info(f"特徵數量圖表已儲存: {n_features_dir}")