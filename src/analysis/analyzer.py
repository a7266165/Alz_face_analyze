"""
src/analysis/analyzer.py
XGBoost 分析器（訓練 + 評估 + 報告）
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)

from src.analysis.loader import Dataset

logger = logging.getLogger(__name__)


class XGBoostAnalyzer:
    """
    XGBoost 分析器
    
    完整的分析流程：
    1. 資料分割與特徵選擇
    2. 模型訓練
    3. 模型評估（訓練集 + 測試集）
    4. 結果儲存（模型 + 報告）
    """
    
    def __init__(
        self,
        output_dir: Path,
        test_size: float = 0.2,
        random_seed: int = 42,
        xgb_params: Optional[Dict] = None,
        feature_selection: bool = True,
        importance_ratio: float = 0.8
    ):
        """
        初始化分析器
        
        Args:
            output_dir: 輸出目錄
            test_size: 測試集比例
            random_seed: 隨機種子
            xgb_params: XGBoost 參數（None 使用預設）
            feature_selection: 是否進行特徵選擇
            importance_ratio: 特徵選擇保留比例
        """
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.random_seed = random_seed
        self.feature_selection = feature_selection
        self.importance_ratio = importance_ratio
        
        # XGBoost 預設參數
        self.xgb_params = xgb_params or {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': random_seed,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        # 建立輸出目錄
        self._setup_dirs()
        
        logger.info("XGBoost 分析器初始化完成")
        logger.info(f"輸出目錄: {self.output_dir}")
        logger.info(f"測試集比例: {self.test_size}")
        logger.info(f"隨機種子: {self.random_seed}")
        logger.info(f"特徵選擇: {'啟用' if self.feature_selection else '停用'}")
    
    def _setup_dirs(self):
        """建立輸出目錄結構"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.output_dir / "models"
        self.reports_dir = self.output_dir / "reports"
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
    
    # ========== 主要分析方法 ==========
    
    def analyze(self, datasets: List[Dataset]) -> Dict:
        """
        分析資料集
        
        Args:
            datasets: 資料集列表
        
        Returns:
            訓練結果字典 {dataset_key: result}
        """
        logger.info(f"開始分析 {len(datasets)} 個資料集")
        
        results = {}
        
        for i, dataset in enumerate(datasets, 1):
            # 建立資料集鍵值
            meta = dataset.metadata
            dataset_key = f"{meta['model']}_{meta['feature_type']}_cdr{meta['cdr_threshold']}"
            
            logger.info(f"\n[{i}/{len(datasets)}] 分析: {dataset_key}")
            logger.info("-" * 50)
            
            try:
                X, y = dataset.X, dataset.y
                subject_ids = dataset.subject_ids

                logger.info(f"資料集: {len(X)} 樣本, {X.shape[1]} 特徵")
                logger.info(f"類別分佈: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
                
                # Step 1: 分割資料集
                gss = GroupShuffleSplit(
                    n_splits=1,
                    test_size=self.test_size,
                    random_state=self.random_seed
                )
                train_idx, test_idx = next(gss.split(X, y, groups=subject_ids))
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                logger.info(f"訓練集: {len(X_train)} 樣本, 測試集: {len(X_test)} 樣本")
                # Step 2: 特徵選擇（如果啟用）
                selected_features = None
                original_n_features = X_train.shape[1]
                
                if self.feature_selection:
                    logger.info("執行特徵選擇...")
                    X_train, X_test, selected_features = self._select_features(
                        X_train, y_train, X_test
                    )
                    logger.info(f"特徵選擇: {original_n_features} → {X_train.shape[1]} 維")
                
                # Step 3: 訓練模型
                logger.info("訓練 XGBoost 模型...")
                model = xgb.XGBClassifier(**self.xgb_params)
                model.fit(X_train, y_train)
                
                # Step 4: 預測
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                y_prob_test = model.predict_proba(X_test)[:, 1]
                
                # Step 5: 計算指標
                train_metrics = self._calculate_metrics(y_train, y_pred_train)
                test_metrics = self._calculate_metrics(y_test, y_pred_test, y_prob_test)
                
                # Step 6: 整理結果
                result = {
                    'model': model,
                    'dataset_key': dataset_key,
                    'metadata': dataset.metadata,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'selected_features': selected_features,
                    'feature_importance': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None,
                    'n_train': len(X_train),
                    'n_test': len(X_test),
                    'original_n_features': original_n_features,
                    'selected_n_features': X_train.shape[1],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Step 7: 儲存
                self._save_model(dataset_key, result)
                self._save_report(dataset_key, result)
                
                results[dataset_key] = result
                
                logger.info(
                    f"✓ {dataset_key}: "
                    f"測試準確率 {result['test_metrics']['accuracy']:.3f}, "
                    f"測試 MCC {result['test_metrics']['mcc']:.3f}"
                )
                
            except Exception as e:
                logger.error(f"✗ {dataset_key}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 儲存總結
        self._save_summary(results)
        
        logger.info(f"\n分析完成: {len(results)}/{len(datasets)} 成功")
        return results
    # ========== 特徵選擇 ==========
    
    def _select_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        使用 XGBoost 特徵重要性選擇特徵
        
        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤
            X_test: 測試特徵
        
        Returns:
            (X_train_selected, X_test_selected, selected_indices)
        """
        # 訓練臨時模型
        temp_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=self.random_seed,
            n_jobs=-1
        )
        temp_model.fit(X_train, y_train)
        
        # 獲取特徵重要性
        importance = temp_model.feature_importances_
        
        # 排序並計算累積重要性
        indices = np.argsort(importance)[::-1]
        cumsum = np.cumsum(importance[indices])
        
        # 找出達到閾值的特徵數量
        if cumsum[-1] > 0:
            n_features = np.searchsorted(cumsum, self.importance_ratio * cumsum[-1]) + 1
        else:
            n_features = len(indices)
        
        n_features = max(1, min(n_features, len(indices)))
        
        # 選擇最重要的特徵
        selected_indices = sorted(indices[:n_features].tolist())
        
        return (
            X_train[:, selected_indices],
            X_test[:, selected_indices],
            selected_indices
        )
    
    # ========== 評估指標 ==========
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict:
        """
        計算評估指標
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            y_prob: 預測機率（可選）
        
        Returns:
            指標字典
        """
        # 基本指標
        cm = confusion_matrix(y_true, y_pred)
        
        # 處理混淆矩陣
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
            if cm.shape == (1, 1):
                if y_true[0] == 0:
                    tn = cm[0, 0]
                else:
                    tp = cm[0, 0]
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'mcc': float(matthews_corrcoef(y_true, y_pred)),
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'confusion_matrix': cm.tolist()
        }
        
        # AUC（如果有機率）
        if y_prob is not None:
            try:
                metrics['auc'] = float(roc_auc_score(y_true, y_prob))
            except Exception as e:
                logger.warning(f"無法計算 AUC: {e}")
                metrics['auc'] = None
        
        return metrics
    
    # ========== 儲存 ==========
    
    def _save_model(self, dataset_key: str, result: Dict):
        """儲存模型"""
        model = result['model']
        
        # XGBoost 模型（JSON 格式）
        model_path = self.models_dir / f"{dataset_key}.json"
        model.save_model(str(model_path))
        logger.debug(f"模型已儲存: {model_path}")
        
        # 特徵選擇資訊（如果有）
        if result['selected_features'] is not None:
            import json
            
            feature_info = {
                'selected_indices': result['selected_features'],
                'original_dim': result['original_n_features'],
                'selected_dim': result['selected_n_features'],
                'importance_ratio': self.importance_ratio
            }
            
            feature_path = self.models_dir / f"{dataset_key}_features.json"
            with open(feature_path, 'w', encoding='utf-8') as f:
                json.dump(feature_info, f, indent=2)
            logger.debug(f"特徵資訊已儲存: {feature_path}")
    
    def _save_report(self, dataset_key: str, result: Dict):
        """儲存文字報告"""
        report_path = self.reports_dir / f"{dataset_key}_report.txt"
        
        # 指標順序
        metric_order = ['accuracy', 'mcc', 'sensitivity', 'specificity', 
                        'precision', 'recall', 'f1', 'auc']
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("XGBoost 分析報告\n")
            f.write("=" * 60 + "\n")
            f.write(f"資料集: {dataset_key}\n")
            f.write(f"分析時間: {result['timestamp']}\n")
            f.write(f"訓練集: {result['n_train']} 樣本\n")
            f.write(f"測試集: {result['n_test']} 樣本\n")
            
            # 測試集效能
            f.write("\n測試集效能:\n")
            f.write("-" * 30 + "\n")
            for metric in metric_order:
                if metric in result['test_metrics'] and metric != 'confusion_matrix':
                    value = result['test_metrics'][metric]
                    if value is not None:
                        f.write(f"  {metric}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric}: N/A\n")
            
            # 混淆矩陣
            f.write("\n測試集混淆矩陣:\n")
            f.write("-" * 30 + "\n")
            cm = result['test_metrics']['confusion_matrix']
            f.write("         預測0  預測1\n")
            f.write(f"實際0   {int(cm[0][0]):5d}  {int(cm[0][1]):5d}\n")
            f.write(f"實際1   {int(cm[1][0]):5d}  {int(cm[1][1]):5d}\n")
            
            # 訓練集效能
            f.write("\n訓練集效能:\n")
            f.write("-" * 30 + "\n")
            for metric in metric_order:
                if metric in result['train_metrics'] and metric != 'confusion_matrix':
                    value = result['train_metrics'].get(metric)
                    if value is not None:
                        f.write(f"  {metric}: {value:.4f}\n")
            
            # 特徵選擇
            if result['selected_features'] is not None:
                f.write("\n特徵選擇:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  原始維度: {result['original_n_features']}\n")
                f.write(f"  選擇維度: {result['selected_n_features']}\n")
                compression_ratio = result['selected_n_features'] / result['original_n_features']
                f.write(f"  壓縮比例: {compression_ratio:.1%}\n")
        
        logger.debug(f"報告已儲存: {report_path}")
    
    def _save_summary(self, results: Dict):
        """儲存總結報告"""
        import json
        
        summary_path = self.output_dir / "training_summary.json"
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_datasets': len(results),
            'xgb_params': self.xgb_params,
            'test_size': self.test_size,
            'random_seed': self.random_seed,
            'feature_selection': self.feature_selection,
            'results': {}
        }
        
        # 整理關鍵指標
        for key, result in results.items():
            summary['results'][key] = {
                'train_accuracy': result['train_metrics']['accuracy'],
                'test_accuracy': result['test_metrics']['accuracy'],
                'test_mcc': result['test_metrics']['mcc'],
                'test_f1': result['test_metrics']['f1'],
                'test_auc': result['test_metrics'].get('auc')
            }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"總結報告已儲存: {summary_path}")
        
        # 找出最佳模型
        self._print_best_models(summary['results'])
    
    def _print_best_models(self, results: Dict):
        """顯示最佳模型"""
        if not results:
            return
        
        print("\n" + "=" * 60)
        print("最佳模型")
        print("=" * 60)
        
        # 最佳 MCC
        best_mcc_key = max(results.keys(), key=lambda k: results[k]['test_mcc'])
        best_mcc = results[best_mcc_key]
        
        print(f"\n最佳 MCC: {best_mcc_key}")
        print(f"  MCC: {best_mcc['test_mcc']:.4f}")
        print(f"  準確率: {best_mcc['test_accuracy']:.4f}")
        print(f"  F1: {best_mcc['test_f1']:.4f}")
        if best_mcc['test_auc']:
            print(f"  AUC: {best_mcc['test_auc']:.4f}")
        
        # 最佳準確率
        best_acc_key = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        if best_acc_key != best_mcc_key:
            best_acc = results[best_acc_key]
            print(f"\n最佳準確率: {best_acc_key}")
            print(f"  準確率: {best_acc['test_accuracy']:.4f}")
            print(f"  MCC: {best_acc['test_mcc']:.4f}")
        
        print("=" * 60)