"""
src/analysis/plotter.py
結果視覺化
"""

import logging
from pathlib import Path
from typing import Dict, List
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
    
    def __init__(self, all_results: Dict[int, Dict], output_dir: Path):
        """
        Args:
            all_results: {min_age: {dataset_key: result}}
            output_dir: 輸出目錄
        """
        self.all_results = all_results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 提取所有 dataset_keys
        self.dataset_keys = set()
        for results in all_results.values():
            self.dataset_keys.update(results.keys())
        self.dataset_keys = sorted(self.dataset_keys)
        
        # 年齡列表
        self.ages = sorted(all_results.keys())
    
    def plot_individual(self):
        """每個 dataset_key 各畫四張圖"""
        individual_dir = self.output_dir / "individual"
        individual_dir.mkdir(exist_ok=True)
        
        for dataset_key in self.dataset_keys:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{dataset_key}', fontsize=14)
            
            for idx, metric in enumerate(self.METRICS):
                ax = axes[idx // 2, idx % 2]
                
                values = []
                for age in self.ages:
                    result = self.all_results.get(age, {}).get(dataset_key)
                    if result:
                        val = result['test_metrics'].get(metric)
                        values.append(val if val is not None else np.nan)
                    else:
                        values.append(np.nan)
                
                ax.plot(self.ages, values, 'o-', linewidth=2, markersize=4)
                ax.set_xlabel('Min Predicted Age')
                ax.set_ylabel(self.METRIC_LABELS[metric])
                ax.set_title(self.METRIC_LABELS[metric])
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(individual_dir / f"{dataset_key}.png", dpi=150)
            plt.close()
        
        logger.info(f"個別圖表已儲存: {individual_dir}")
    
    def plot_combined(self):
        """所有組合在同一張圖"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('All Models Combined', fontsize=14)
        
        for idx, metric in enumerate(self.METRICS):
            ax = axes[idx // 2, idx % 2]
            
            for dataset_key in self.dataset_keys:
                values = []
                for age in self.ages:
                    result = self.all_results.get(age, {}).get(dataset_key)
                    if result:
                        val = result['test_metrics'].get(metric)
                        values.append(val if val is not None else np.nan)
                    else:
                        values.append(np.nan)
                
                ax.plot(self.ages, values, '-', linewidth=1, alpha=0.7, label=dataset_key)
            
            ax.set_xlabel('Min Predicted Age')
            ax.set_ylabel(self.METRIC_LABELS[metric])
            ax.set_title(self.METRIC_LABELS[metric])
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        # 圖例放在外面
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "combined_all.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"合併圖表已儲存: {self.output_dir / 'combined_all.png'}")
    
    def plot_by_model(self):
        """按向量模型分組"""
        models = ['arcface', 'dlib', 'topofr']
        
        for model_name in models:
            # 篩選該模型的 dataset_keys
            model_keys = [k for k in self.dataset_keys if k.startswith(model_name)]
            
            if not model_keys:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{model_name.upper()} Models', fontsize=14)
            
            for idx, metric in enumerate(self.METRICS):
                ax = axes[idx // 2, idx % 2]
                
                for dataset_key in model_keys:
                    # 簡化標籤（去掉模型名稱）
                    label = dataset_key.replace(f"{model_name}_", "")
                    
                    values = []
                    for age in self.ages:
                        result = self.all_results.get(age, {}).get(dataset_key)
                        if result:
                            val = result['test_metrics'].get(metric)
                            values.append(val if val is not None else np.nan)
                        else:
                            values.append(np.nan)
                    
                    ax.plot(self.ages, values, 'o-', linewidth=2, markersize=3, label=label)
                
                ax.set_xlabel('Min Predicted Age')
                ax.set_ylabel(self.METRIC_LABELS[metric])
                ax.set_title(self.METRIC_LABELS[metric])
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                ax.legend(fontsize=8)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"by_model_{model_name}.png", dpi=150)
            plt.close()
        
        logger.info(f"模型分組圖表已儲存: {self.output_dir}")