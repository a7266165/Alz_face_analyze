"""
AD Binary Classification Analysis
Using VGGFace features to predict cognitive change (worse/improve)

Features: 4096-dim VGGFace feature vector
Labels: CASI_wrose/improve, MMSE_wrose/improve, CDRSB_wrose/improve
        0 = improve/stable, 1 = worse
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    matthews_corrcoef, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 設定路徑
TABLE_DIR = PROJECT_ROOT / "data" / "demographics"
FEATURES_DIR = PROJECT_ROOT / "workspace" / "features"
OUTPUT_DIR = PROJECT_ROOT / "workspace" / "ad_binary_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 對照表路徑
LOOKUP_CSV = TABLE_DIR / "P.csv"

# 標籤欄位名稱
TARGET_COLS = {
    'CASI': 'CASI_wrose /improve  ',
    'MMSE': 'MMSE_wrose /improve  2',
    'CDRSB': 'CDRSB_wrose /improve  '
}


def load_ad_labels():
    """
    從 Excel 載入 AD 患者標籤資料
    返回含有編號、face日期與三個二元標籤的 DataFrame
    """
    print("=" * 60)
    print("載入 AD 標籤資料")
    print("=" * 60)

    # 從 Excel 讀取 AD_patient_result 工作表
    xlsx_path = TABLE_DIR / "2025高醫不對稱與問卷關聯性分析_2.xlsx"
    df_full = pd.read_excel(xlsx_path, sheet_name='AD_patient_result')

    # 提取需要的欄位: A~G (0-6) 和 N~P (13-15)
    selected_columns = list(df_full.columns[0:7]) + list(df_full.columns[13:16])
    df = df_full[selected_columns]

    # 確保編號為整數
    df['編號'] = df['編號'].astype(int)

    # 轉換 face日期 格式
    df['face日期'] = pd.to_datetime(df['face日期'])

    print(f"載入 {len(df)} 筆標籤資料")
    print(f"欄位: {df.columns.tolist()}")

    # 顯示標籤分佈
    for name, col in TARGET_COLS.items():
        if col in df.columns:
            counts = df[col].value_counts(dropna=False)
            print(f"\n{name} 標籤分佈:")
            print(counts)

    return df


def load_lookup_table():
    """
    載入 P.csv 對照表
    用於根據 Patient + Photo_Date 查找對應的 npy 檔名 (ID)
    """
    print("\n" + "=" * 60)
    print("載入對照表 P.csv")
    print("=" * 60)

    df = pd.read_csv(LOOKUP_CSV)

    # 確保 Patient 為整數
    df['Patient'] = df['Patient'].astype(int)

    # 轉換 Photo_Date 格式
    df['Photo_Date'] = pd.to_datetime(df['Photo_Date'])

    print(f"載入 {len(df)} 筆對照資料")
    print(f"ID 範例: {df['ID'].head(5).tolist()}")

    return df


def load_npy_features_for_ad(labels_df, lookup_df):
    """
    根據標籤資料載入對應的 npy 特徵

    Args:
        labels_df: AD_change_condition.csv 的 DataFrame
        lookup_df: P.csv 對照表的 DataFrame

    Returns:
        展開後的 DataFrame（每張圖片一筆，含 case_id, 標籤, 4096 維特徵）
    """
    print("\n" + "=" * 60)
    print("載入 NPY 特徵向量")
    print("=" * 60)

    all_rows = []
    matched_count = 0
    missing_count = 0

    for idx, row in labels_df.iterrows():
        patient_id = row['編號']
        face_date = row['face日期']

        # 在對照表中查找匹配的 ID
        mask = (lookup_df['Patient'] == patient_id) & (lookup_df['Photo_Date'] == face_date)
        matched = lookup_df[mask]

        if len(matched) == 0:
            missing_count += 1
            continue

        npy_id = matched.iloc[0]['ID']  # 如 P1-3
        npy_path = FEATURES_DIR / f"{npy_id}.npy"

        if not npy_path.exists():
            missing_count += 1
            continue

        # 載入特徵 (10, 4096)
        features = np.load(npy_path)

        # 展開 10 張圖片，每張獨立一筆
        for img_idx in range(features.shape[0]):
            feature_row = {
                'case_id': f"{patient_id}_{face_date.strftime('%Y%m%d')}",
                'patient_id': patient_id,
                'npy_id': npy_id,
                'img_idx': img_idx,
            }

            # 加入標籤
            for name, col in TARGET_COLS.items():
                if col in row.index:
                    feature_row[name] = row[col]

            # 加入 4096 維特徵
            for i in range(features.shape[1]):
                feature_row[f'f_{i}'] = features[img_idx, i]

            all_rows.append(feature_row)

        matched_count += 1

    df = pd.DataFrame(all_rows)

    print(f"成功匹配: {matched_count} 個案")
    print(f"缺失: {missing_count} 個案")
    print(f"展開後總資料筆數: {len(df)} (每個案 10 張圖片)")

    return df


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    計算分類評估指標
    """
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # 計算 Sensitivity 和 Specificity
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity = 0
        specificity = 0

    # 計算 AUC-ROC
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0

    return {
        'accuracy': acc,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc_roc': auc,
        'confusion_matrix': cm
    }


def train_binary_classifier(df, target_name):
    """
    訓練二元分類模型

    Args:
        df: 展開後的 DataFrame
        target_name: 標籤名稱 (CASI/MMSE/CDRSB)

    Returns:
        模型結果字典
    """
    print(f"\n{'=' * 60}")
    print(f"訓練 {target_name} 二元分類模型")
    print("=" * 60)

    # 移除目標標籤的缺失值
    valid_df = df.dropna(subset=[target_name]).copy()
    valid_df[target_name] = valid_df[target_name].astype(int)

    print(f"有效資料筆數: {len(valid_df)}")
    print(f"標籤分佈:\n{valid_df[target_name].value_counts()}")

    if len(valid_df) < 20:
        print(f"警告: {target_name} 資料量過少，跳過訓練")
        return None

    # 準備特徵和標籤
    feature_cols = [c for c in valid_df.columns if c.startswith('f_')]
    X = valid_df[feature_cols].values
    y = valid_df[target_name].values
    case_ids = valid_df['case_id'].values

    # 取得唯一的 case_id 用於分組
    unique_cases = valid_df.drop_duplicates('case_id')[['case_id', target_name]].reset_index(drop=True)

    # 按 case 分割訓練集和測試集（確保同一個案的 10 張圖片在同一組）
    train_cases, test_cases = train_test_split(
        unique_cases, test_size=0.2, random_state=42,
        stratify=unique_cases[target_name]
    )

    train_mask = valid_df['case_id'].isin(train_cases['case_id'])
    test_mask = valid_df['case_id'].isin(test_cases['case_id'])

    X_train = valid_df.loc[train_mask, feature_cols].values
    y_train = valid_df.loc[train_mask, target_name].values
    X_test = valid_df.loc[test_mask, feature_cols].values
    y_test_expanded = valid_df.loc[test_mask, target_name].values
    test_case_ids = valid_df.loc[test_mask, 'case_id'].values

    print(f"訓練集: {len(train_cases)} 個案 ({len(X_train)} 張圖片)")
    print(f"測試集: {len(test_cases)} 個案 ({len(X_test)} 張圖片)")

    # 訓練 XGBoost 模型
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    # 預測每張圖片的機率
    y_prob_expanded = model.predict_proba(X_test)[:, 1]

    # 對同一個案的 10 張圖片取平均機率
    test_df = pd.DataFrame({
        'case_id': test_case_ids,
        'y_true': y_test_expanded,
        'y_prob': y_prob_expanded
    })

    case_results = test_df.groupby('case_id').agg({
        'y_true': 'first',  # 同一個案標籤相同
        'y_prob': 'mean'    # 取平均機率
    }).reset_index()

    # 轉換為最終判斷
    y_true_case = case_results['y_true'].values
    y_prob_case = case_results['y_prob'].values
    y_pred_case = (y_prob_case >= 0.5).astype(int)

    # 計算評估指標
    metrics = calculate_metrics(y_true_case, y_pred_case, y_prob_case)

    print(f"\n評估指標 (以個案為單位):")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  MCC:         {metrics['mcc']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"\n混淆矩陣:\n{metrics['confusion_matrix']}")

    return {
        'model': model,
        'metrics': metrics,
        'y_true': y_true_case,
        'y_pred': y_pred_case,
        'y_prob': y_prob_case,
        'feature_cols': feature_cols,
        'n_train_cases': len(train_cases),
        'n_test_cases': len(test_cases)
    }


def plot_confusion_matrices(results):
    """繪製合併的混淆矩陣 (1x3)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        if result is None:
            ax.text(0.5, 0.5, f'{name}\n無資料', ha='center', va='center', fontsize=14)
            ax.axis('off')
            continue

        cm = result['metrics']['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['改善(0)', '惡化(1)'],
                    yticklabels=['改善(0)', '惡化(1)'])

        acc = result['metrics']['accuracy']
        ax.set_title(f'{name}\nACC: {acc:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('預測標籤')
        ax.set_ylabel('真實標籤')

    plt.suptitle('混淆矩陣 - AD 患者認知變化預測', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"\n已儲存: {OUTPUT_DIR / 'confusion_matrices.png'}")
    plt.close()


def plot_roc_curves(results):
    """繪製合併的 ROC 曲線 (1x3)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        if result is None:
            ax.text(0.5, 0.5, f'{name}\n無資料', ha='center', va='center', fontsize=14)
            ax.axis('off')
            continue

        y_true = result['y_true']
        y_prob = result['y_prob']

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = result['metrics']['auc_roc']

        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('ROC 曲線 - AD 患者認知變化預測', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"已儲存: {OUTPUT_DIR / 'roc_curves.png'}")
    plt.close()


def plot_feature_importance(results):
    """繪製特徵重要性 (Top 20)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        if result is None:
            ax.text(0.5, 0.5, f'{name}\n無資料', ha='center', va='center', fontsize=14)
            ax.axis('off')
            continue

        model = result['model']
        feature_cols = result['feature_cols']

        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(20)

        ax.barh(importance['feature'], importance['importance'], color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title(f'{name} - Top 20 特徵', fontsize=12, fontweight='bold')

    plt.suptitle('特徵重要性 - AD 患者認知變化預測', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"已儲存: {OUTPUT_DIR / 'feature_importance.png'}")
    plt.close()


def save_metrics_summary(results):
    """儲存評估指標彙總表"""
    rows = []
    for name, result in results.items():
        if result is None:
            continue
        m = result['metrics']
        rows.append({
            'Target': name,
            'Accuracy': f"{m['accuracy']:.4f}",
            'MCC': f"{m['mcc']:.4f}",
            'Sensitivity': f"{m['sensitivity']:.4f}",
            'Specificity': f"{m['specificity']:.4f}",
            'AUC-ROC': f"{m['auc_roc']:.4f}",
            'Train_Cases': result['n_train_cases'],
            'Test_Cases': result['n_test_cases']
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / 'metrics_summary.csv', index=False, encoding='utf-8-sig')
    print(f"已儲存: {OUTPUT_DIR / 'metrics_summary.csv'}")

    return df


def main():
    """主程式"""
    print("=" * 60)
    print("AD 患者認知變化二元分類分析")
    print("=" * 60)

    # 1. 載入資料
    labels_df = load_ad_labels()
    lookup_df = load_lookup_table()

    # 2. 載入 NPY 特徵並展開
    features_df = load_npy_features_for_ad(labels_df, lookup_df)

    if len(features_df) == 0:
        print("錯誤: 沒有載入任何特徵資料")
        return

    # 3. 對三個標籤分別訓練模型
    results = {}
    for name in TARGET_COLS.keys():
        results[name] = train_binary_classifier(features_df, name)

    # 4. 繪製視覺化圖表
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    plot_feature_importance(results)

    # 5. 儲存評估指標
    metrics_df = save_metrics_summary(results)
    print(f"\n{'=' * 60}")
    print("評估指標彙總:")
    print("=" * 60)
    print(metrics_df.to_string(index=False))

    print(f"\n{'=' * 60}")
    print("分析完成!")
    print(f"所有輸出已儲存至: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
