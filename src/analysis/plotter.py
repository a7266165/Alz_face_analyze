"""
src/analysis/plotter.py
結果視覺化
"""

import logging
from pathlib import Path
from typing import Dict, Optional
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
    
    def _get_metric_value(self, result: Dict, metric: str, corrected: bool) -> float:
        """取得指標值"""
        if corrected:
            metrics = result.get('corrected_metrics', {})
        else:
            metrics = result.get('test_metrics', {})
        
        val = metrics.get(metric) if metrics else None
        return val if val is not None else np.nan
    
    def _get_best_result(self, min_age: int, dataset_key: str, metric: str = 'mcc') -> Optional[Dict]:
        """取得最佳特徵數的結果（用於按年齡繪圖）"""
        results_by_n_features = self.all_results.get(min_age, {}).get(dataset_key, {})
        if not results_by_n_features:
            return None
        
        best_n_feat = max(
            results_by_n_features.keys(),
            key=lambda k: results_by_n_features[k]['test_metrics'].get(metric, 0) or 0
        )
        return results_by_n_features[best_n_feat]

    def _plot_single_figure(self, dataset_key: str, corrected: bool, output_path: Path):
        """繪製單一圖表（四個指標）"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        suffix = " (Corrected)" if corrected else ""
        fig.suptitle(f'{dataset_key}{suffix}', fontsize=14)
        
        for idx, metric in enumerate(self.METRICS):
            ax = axes[idx // 2, idx % 2]
            
            values = []
            for age in self.ages:
                result = self.all_results.get(age, {}).get(dataset_key)
                if result:
                    val = self._get_metric_value(result, metric, corrected)
                    values.append(val)
                else:
                    values.append(np.nan)
            
            ax.plot(self.ages, values, 'o-', linewidth=2, markersize=4)
            ax.set_xlabel('Min Predicted Age')
            ax.set_ylabel(self.METRIC_LABELS[metric])
            ax.set_title(self.METRIC_LABELS[metric])
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    # def plot_individual(self):
    #     """每個 dataset_key 各畫四張圖（校正前 + 校正後）"""
    #     individual_dir = self.plots_dir / "individual"
    #     individual_dir.mkdir(exist_ok=True)
        
    #     for dataset_key in self.dataset_keys:
    #         # 校正前
    #         self._plot_single_figure(
    #             dataset_key, 
    #             corrected=False, 
    #             output_path=individual_dir / f"{dataset_key}.png"
    #         )
    #         # 校正後
    #         self._plot_single_figure(
    #             dataset_key, 
    #             corrected=True, 
    #             output_path=individual_dir / f"{dataset_key}_corrected.png"
    #         )
        
    #     logger.info(f"個別圖表已儲存: {individual_dir}")
    
    def plot_individual(self):
        """每個 dataset_key 各畫四張圖（每條線代表不同特徵數）"""
        individual_dir = self.plots_dir / "individual"
        individual_dir.mkdir(exist_ok=True)
        
        for dataset_key in self.dataset_keys:
            # 收集該 dataset 有哪些特徵數
            n_features_set = set()
            for min_age in self.ages:
                results_by_n_features = self.all_results.get(min_age, {}).get(dataset_key, {})
                n_features_set.update(results_by_n_features.keys())
            n_features_sorted = sorted(n_features_set, reverse=True)
            
            if not n_features_sorted:
                continue
            
            # 校正前
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{dataset_key}', fontsize=14)
            
            for idx, metric in enumerate(self.METRICS):
                ax = axes[idx // 2, idx % 2]
                
                for n_feat in n_features_sorted:
                    values = []
                    for age in self.ages:
                        result = self.all_results.get(age, {}).get(dataset_key, {}).get(n_feat)
                        if result:
                            val = result['test_metrics'].get(metric)
                            values.append(val if val is not None else np.nan)
                        else:
                            values.append(np.nan)
                    
                    ax.plot(self.ages, values, 'o-', linewidth=1, markersize=3, 
                            label=f'n={n_feat}', alpha=0.7)
                
                ax.set_xlabel('Min Predicted Age')
                ax.set_ylabel(self.METRIC_LABELS[metric])
                ax.set_title(self.METRIC_LABELS[metric])
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                ax.legend(fontsize=6, ncol=2)
            
            plt.tight_layout()
            plt.savefig(individual_dir / f"{dataset_key}.png", dpi=150)
            plt.close()
            
            # 校正後
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{dataset_key} (Corrected)', fontsize=14)
            
            for idx, metric in enumerate(self.METRICS):
                ax = axes[idx // 2, idx % 2]
                
                for n_feat in n_features_sorted:
                    values = []
                    for age in self.ages:
                        result = self.all_results.get(age, {}).get(dataset_key, {}).get(n_feat)
                        if result and result.get('corrected_metrics'):
                            val = result['corrected_metrics'].get(metric)
                            values.append(val if val is not None else np.nan)
                        else:
                            values.append(np.nan)
                    
                    ax.plot(self.ages, values, 'o-', linewidth=1, markersize=3, 
                            label=f'n={n_feat}', alpha=0.7)
                
                ax.set_xlabel('Min Predicted Age')
                ax.set_ylabel(self.METRIC_LABELS[metric])
                ax.set_title(f'{self.METRIC_LABELS[metric]} (Corrected)')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                ax.legend(fontsize=6, ncol=2)
            
            plt.tight_layout()
            plt.savefig(individual_dir / f"{dataset_key}_corrected.png", dpi=150)
            plt.close()
        
        logger.info(f"個別圖表已儲存: {individual_dir}")

    def _plot_combined_figure(self, corrected: bool, output_path: Path):
        """繪製合併圖表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        suffix = " (Corrected)" if corrected else ""
        fig.suptitle(f'All Models Combined{suffix}', fontsize=14)
        
        for idx, metric in enumerate(self.METRICS):
            ax = axes[idx // 2, idx % 2]
            
            for dataset_key in self.dataset_keys:
                values = []
                for age in self.ages:
                    result = self.all_results.get(age, {}).get(dataset_key)
                    if result:
                        val = self._get_metric_value(result, metric, corrected)
                        values.append(val)
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
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # def plot_combined(self):
    #     """所有組合在同一張圖（校正前 + 校正後）"""
    #     # 校正前
    #     self._plot_combined_figure(
    #         corrected=False,
    #         output_path=self.plots_dir / "combined_all.png"
    #     )
    #     # 校正後
    #     self._plot_combined_figure(
    #         corrected=True,
    #         output_path=self.plots_dir / "combined_all_corrected.png"
    #     )
        
    #     logger.info(f"合併圖表已儲存: {self.plots_dir}")
    
    def plot_combined(self):
        """所有組合在同一張圖（每條線代表 dataset_key + n_features）"""
        # 校正前
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('All Models Combined', fontsize=14)
        
        for idx, metric in enumerate(self.METRICS):
            ax = axes[idx // 2, idx % 2]
            
            for dataset_key in self.dataset_keys:
                # 收集該 dataset 有哪些特徵數
                n_features_set = set()
                for min_age in self.ages:
                    results_by_n_features = self.all_results.get(min_age, {}).get(dataset_key, {})
                    n_features_set.update(results_by_n_features.keys())
                n_features_sorted = sorted(n_features_set, reverse=True)
                
                for n_feat in n_features_sorted:
                    values = []
                    for age in self.ages:
                        result = self.all_results.get(age, {}).get(dataset_key, {}).get(n_feat)
                        if result:
                            val = result['test_metrics'].get(metric)
                            values.append(val if val is not None else np.nan)
                        else:
                            values.append(np.nan)
                    
                    ax.plot(self.ages, values, '-', linewidth=0.5, alpha=0.5)
            
            ax.set_xlabel('Min Predicted Age')
            ax.set_ylabel(self.METRIC_LABELS[metric])
            ax.set_title(self.METRIC_LABELS[metric])
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "combined_all.png", dpi=150)
        plt.close()
        
        # 校正後（同樣邏輯）
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('All Models Combined (Corrected)', fontsize=14)
        
        for idx, metric in enumerate(self.METRICS):
            ax = axes[idx // 2, idx % 2]
            
            for dataset_key in self.dataset_keys:
                n_features_set = set()
                for min_age in self.ages:
                    results_by_n_features = self.all_results.get(min_age, {}).get(dataset_key, {})
                    n_features_set.update(results_by_n_features.keys())
                n_features_sorted = sorted(n_features_set, reverse=True)
                
                for n_feat in n_features_sorted:
                    values = []
                    for age in self.ages:
                        result = self.all_results.get(age, {}).get(dataset_key, {}).get(n_feat)
                        if result and result.get('corrected_metrics'):
                            val = result['corrected_metrics'].get(metric)
                            values.append(val if val is not None else np.nan)
                        else:
                            values.append(np.nan)
                    
                    ax.plot(self.ages, values, '-', linewidth=0.5, alpha=0.5)
            
            ax.set_xlabel('Min Predicted Age')
            ax.set_ylabel(self.METRIC_LABELS[metric])
            ax.set_title(f'{self.METRIC_LABELS[metric]} (Corrected)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "combined_all_corrected.png", dpi=150)
        plt.close()
        
        logger.info(f"合併圖表已儲存: {self.plots_dir}")

    def _plot_by_model_figure(self, model_name: str, corrected: bool, output_path: Path):
        """繪製單一模型圖表"""
        model_keys = [k for k in self.dataset_keys if k.startswith(model_name)]
        
        if not model_keys:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        suffix = " (Corrected)" if corrected else ""
        fig.suptitle(f'{model_name.upper()} Models{suffix}', fontsize=14)
        
        for idx, metric in enumerate(self.METRICS):
            ax = axes[idx // 2, idx % 2]
            
            for dataset_key in model_keys:
                # 簡化標籤（去掉模型名稱）
                label = dataset_key.replace(f"{model_name}_", "")
                
                values = []
                for age in self.ages:
                    result = self.all_results.get(age, {}).get(dataset_key)
                    if result:
                        val = self._get_metric_value(result, metric, corrected)
                        values.append(val)
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
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    # def plot_by_model(self):
    #     """按向量模型分組（校正前 + 校正後）"""
    #     models = ['arcface', 'dlib', 'topofr']
        
    #     for model_name in models:
    #         # 校正前
    #         self._plot_by_model_figure(
    #             model_name,
    #             corrected=False,
    #             output_path=self.plots_dir / f"by_model_{model_name}.png"
    #         )
    #         # 校正後
    #         self._plot_by_model_figure(
    #             model_name,
    #             corrected=True,
    #             output_path=self.plots_dir / f"by_model_{model_name}_corrected.png"
    #         )
        
    #     logger.info(f"模型分組圖表已儲存: {self.plots_dir}")

    def plot_by_model(self):
        """按向量模型分組（每條線代表 feature_type + cdr + n_features）"""
        models = ['arcface', 'dlib', 'topofr']
        
        for model_name in models:
            model_keys = [k for k in self.dataset_keys if k.startswith(model_name)]
            
            if not model_keys:
                continue
            
            # 校正前
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{model_name.upper()} Models', fontsize=14)
            
            for idx, metric in enumerate(self.METRICS):
                ax = axes[idx // 2, idx % 2]
                
                for dataset_key in model_keys:
                    label_base = dataset_key.replace(f"{model_name}_", "")
                    
                    n_features_set = set()
                    for min_age in self.ages:
                        results_by_n_features = self.all_results.get(min_age, {}).get(dataset_key, {})
                        n_features_set.update(results_by_n_features.keys())
                    n_features_sorted = sorted(n_features_set, reverse=True)
                    
                    for n_feat in n_features_sorted:
                        values = []
                        for age in self.ages:
                            result = self.all_results.get(age, {}).get(dataset_key, {}).get(n_feat)
                            if result:
                                val = result['test_metrics'].get(metric)
                                values.append(val if val is not None else np.nan)
                            else:
                                values.append(np.nan)
                        
                        ax.plot(self.ages, values, '-', linewidth=1, alpha=0.6,
                                label=f'{label_base}_n{n_feat}')
                
                ax.set_xlabel('Min Predicted Age')
                ax.set_ylabel(self.METRIC_LABELS[metric])
                ax.set_title(self.METRIC_LABELS[metric])
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
            
            # 圖例放外面
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=6)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"by_model_{model_name}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # 校正後（同樣邏輯）
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{model_name.upper()} Models (Corrected)', fontsize=14)
            
            for idx, metric in enumerate(self.METRICS):
                ax = axes[idx // 2, idx % 2]
                
                for dataset_key in model_keys:
                    label_base = dataset_key.replace(f"{model_name}_", "")
                    
                    n_features_set = set()
                    for min_age in self.ages:
                        results_by_n_features = self.all_results.get(min_age, {}).get(dataset_key, {})
                        n_features_set.update(results_by_n_features.keys())
                    n_features_sorted = sorted(n_features_set, reverse=True)
                    
                    for n_feat in n_features_sorted:
                        values = []
                        for age in self.ages:
                            result = self.all_results.get(age, {}).get(dataset_key, {}).get(n_feat)
                            if result and result.get('corrected_metrics'):
                                val = result['corrected_metrics'].get(metric)
                                values.append(val if val is not None else np.nan)
                            else:
                                values.append(np.nan)
                        
                        ax.plot(self.ages, values, '-', linewidth=1, alpha=0.6,
                                label=f'{label_base}_n{n_feat}')
                
                ax.set_xlabel('Min Predicted Age')
                ax.set_ylabel(self.METRIC_LABELS[metric])
                ax.set_title(f'{self.METRIC_LABELS[metric]} (Corrected)')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
            
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=6)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"by_model_{model_name}_corrected.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        logger.info(f"模型分組圖表已儲存: {self.plots_dir}")

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

    def plot_filter_stats(self):
        """繪製各年齡閾值的篩選統計"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Filtered Out Subjects by Min Predicted Age', fontsize=14)
        
        # 收集資料（從任一 dataset 取得 filter_stats）
        health_filtered = []
        patient_filtered = []
        health_ratio = []
        patient_ratio = []
        
        for age in self.ages:
            results = self.all_results.get(age, {})
            if results:
                # 取第一個 dataset 的 filter_stats（所有 dataset 的篩選統計相同）
                first_result = next(iter(results.values()), None)
                if first_result and 'filter_stats' in first_result:
                    stats = first_result['filter_stats']
                    health_filtered.append(stats.get('health_filtered_out', 0))
                    patient_filtered.append(stats.get('patient_filtered_out', 0))
                    health_ratio.append(stats.get('health_filtered_out_ratio', 0))
                    patient_ratio.append(stats.get('patient_filtered_out_ratio', 0))
                    continue
            
            health_filtered.append(np.nan)
            patient_filtered.append(np.nan)
            health_ratio.append(np.nan)
            patient_ratio.append(np.nan)
        
        # 左圖：人數
        ax1 = axes[0]
        ax1.plot(self.ages, health_filtered, 'o-', linewidth=2, markersize=4, label='Health (ACS+NAD)', color='blue')
        ax1.plot(self.ages, patient_filtered, 's-', linewidth=2, markersize=4, label='Patient (P)', color='red')
        ax1.set_xlabel('Min Predicted Age')
        ax1.set_ylabel('Filtered Out Count')
        ax1.set_title('Filtered Out Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右圖：比例
        ax2 = axes[1]
        ax2.plot(self.ages, health_ratio, 'o-', linewidth=2, markersize=4, label='Health (ACS+NAD)', color='blue')
        ax2.plot(self.ages, patient_ratio, 's-', linewidth=2, markersize=4, label='Patient (P)', color='red')
        ax2.set_xlabel('Min Predicted Age')
        ax2.set_ylabel('Filtered Out Ratio')
        ax2.set_title('Filtered Out Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "filter_stats.png", dpi=150)
        plt.close()
        
        logger.info(f"篩選統計圖表已儲存: {self.plots_dir / 'filter_stats.png'}")