"""
測試預處理模組
"""

import cv2
import numpy as np
from pathlib import Path
import logging
import sys

# 加入專案路徑
project_root = Path(__file__).parent.parent  # 從 scripts/ 回到專案根目錄
sys.path.insert(0, str(project_root))

from src.core import PreprocessConfig, FacePreprocessor

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_images(subject_dir: Path, max_images: int = 20) -> tuple:
    """載入單一受試者的測試影像"""
    images = []
    paths = []
    
    # 支援的影像格式
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 收集所有影像檔案
    image_files = []
    seen_files = set()  # 追蹤已經加入的檔案
    
    for ext in extensions:
        for file in subject_dir.glob(f'*{ext}'):
            # 使用小寫檔名來檢查重複
            file_key = file.name.lower()
            if file_key not in seen_files:
                image_files.append(file)
                seen_files.add(file_key)
                
        for file in subject_dir.glob(f'*{ext.upper()}'):
            file_key = file.name.lower()
            if file_key not in seen_files:
                image_files.append(file)
                seen_files.add(file_key)
    
    # 排序並限制數量
    image_files = sorted(image_files)[:max_images]
    
    # 顯示載入的檔案列表
    logger.info(f"找到 {len(image_files)} 張影像在 {subject_dir.name}")
    for i, img_path in enumerate(image_files):
        logger.info(f"  [{i}] {img_path.name}")
    
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
            paths.append(img_path)
    
    return images, paths


def test_with_real_images():
    """使用真實影像測試"""
    print("\n" + "="*60)
    print("真實影像測試")
    print("="*60)
    
    # 嘗試不同的可能路徑
    possible_paths = [
        Path("D:/project/Alz/face/data/datung/raw"),  # 從 path.txt
        Path("data/images/raw"),  # 相對路徑
        Path("../data/images/raw"),  # 可能在上層
    ]
    
    # 找到存在的資料路徑
    data_dir = None
    for path in possible_paths:
        if path.exists():
            data_dir = path
            logger.info(f"使用資料路徑: {data_dir}")
            break
    
    if not data_dir:
        print("⚠ 找不到資料目錄")
        print("請執行以下其中一個步驟：")
        print("  1. 修改腳本中的 possible_paths 變數")
        print("  2. 將測試影像放置於: ./data/images/raw/")
        print("\n嘗試過的路徑：")
        for path in possible_paths:
            print(f"    - {path}")
        return
    
    # 選擇一個測試受試者
    # 優先順序：ACS -> NAD -> P
    test_subject_dir = None
    
    # 嘗試載入 ACS
    acs_dir = data_dir / "health" / "ACS"
    if acs_dir.exists():
        acs_subjects = sorted([d for d in acs_dir.iterdir() if d.is_dir()])
        if acs_subjects:
            test_subject_dir = acs_subjects[0]
            logger.info(f"選擇測試受試者: {test_subject_dir.relative_to(data_dir)}")
    
    # 如果沒有 ACS，嘗試 NAD
    if not test_subject_dir:
        nad_dir = data_dir / "health" / "NAD"
        if nad_dir.exists():
            nad_subjects = sorted([d for d in nad_dir.iterdir() if d.is_dir()])
            if nad_subjects:
                test_subject_dir = nad_subjects[0]
                logger.info(f"選擇測試受試者: {test_subject_dir.relative_to(data_dir)}")
    
    # 如果都沒有，嘗試 patient
    if not test_subject_dir:
        patient_dir = data_dir / "patient"
        if patient_dir.exists():
            patient_subjects = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
            if patient_subjects:
                test_subject_dir = patient_subjects[0]
                logger.info(f"選擇測試受試者: {test_subject_dir.relative_to(data_dir)}")
    
    if not test_subject_dir:
        print("⚠ 找不到任何受試者資料夾")
        return
    
    # 載入影像
    images, paths = load_test_images(test_subject_dir, max_images=20)
    
    if not images:
        print(f"⚠ 無法載入影像從: {test_subject_dir}")
        return
    
    print(f"\n成功載入 {len(images)} 張影像")
    print(f"受試者: {test_subject_dir.name}")
    
    # 測試配置
    config = PreprocessConfig(
        n_select=10,  # 選擇10張最正面的
        save_intermediate=True,
        workspace_dir=Path("workspace") / "test" / test_subject_dir.name,
        detection_confidence=0.5,
        steps=['select', 'align', 'mirror', 'clahe']
    )
    
    print(f"\n配置:")
    print(f"  - 選擇數量: {config.n_select}")
    print(f"  - 儲存中間結果: {config.save_intermediate}")
    print(f"  - 工作目錄: {config.workspace_dir}")
    print(f"  - 處理步驟: {config.steps}")
    
    try:
        with FacePreprocessor(config) as preprocessor:
            print("\n開始處理...")
            results = preprocessor.process(images, paths)
            
            print(f"\n處理完成:")
            print(f"  輸入影像: {len(images)} 張")
            print(f"  成功處理: {len(results)} 張")
            
            if results:
                print(f"\n詳細結果:")
                for i, result in enumerate(results):
                    metadata = result.metadata
                    print(f"  [{i+1}]")
                    print(f"    - 原始角度: {metadata.get('angle', 'N/A'):.2f}°")
                    print(f"    - 信心度: {metadata.get('confidence', 'N/A'):.2f}")
                    print(f"    - 左鏡射尺寸: {result.left_mirror.shape}")
                    print(f"    - 右鏡射尺寸: {result.right_mirror.shape}")
                    
                    # 儲存預覽（左右並排）
                    if config.save_intermediate:
                        preview_dir = config.workspace_dir / "preview"
                        preview_dir.mkdir(parents=True, exist_ok=True)
                        
                        # 建立並排比較圖
                        comparison = np.hstack([result.left_mirror, result.right_mirror])
                        
                        # 加上標籤
                        h, w = comparison.shape[:2]
                        cv2.putText(comparison, "Left Mirror", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(comparison, "Right Mirror", (w//2 + 10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        preview_path = preview_dir / f"comparison_{i:02d}.png"
                        cv2.imwrite(str(preview_path), comparison)
                
                if config.save_intermediate:
                    print(f"\n中間結果已儲存至:")
                    print(f"  {config.workspace_dir}")
                    print(f"  - selected/: 選中的原始影像")
                    print(f"  - aligned/: 角度校正後")
                    print(f"  - mirrors/: 鏡射結果")
                    print(f"  - preview/: 並排預覽")
            else:
                print("\n⚠ 沒有成功處理的影像")
                
    except Exception as e:
        print(f"\n✗ 處理失敗: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主測試函數"""
    print("="*60)
    print("預處理模組測試 - 真實影像")
    print("="*60)
    
    test_with_real_images()
    
    print("\n" + "="*60)
    print("測試完成")
    print("="*60)


if __name__ == "__main__":
    main()