"""
scripts/predict_ages.py
遍歷原始影像目錄，用 MiVOLO 預測年齡並儲存
"""

import sys
import json
import logging
from pathlib import Path
from tqdm import tqdm
import cv2

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

sys.path.insert(0, str(project_root / "src" / "core"))
from age_predictor import MiVOLOPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_raw_path(path_file: Path) -> Path:
    """讀取 path.txt"""
    with open(path_file, 'r', encoding='utf-8') as f:
        return Path(f.read().strip().strip('"').strip("'"))


def scan_subjects(raw_dir: Path) -> list:
    """掃描所有受試者目錄"""
    subjects = []
    
    # ACS, NAD
    for group in ["ACS", "NAD"]:
        group_path = raw_dir / "health" / group
        if group_path.exists():
            for d in sorted(group_path.iterdir()):
                if d.is_dir():
                    subjects.append(d)
    
    # Patient
    patient_path = raw_dir / "patient" / "good"
    if patient_path.exists():
        for d in sorted(patient_path.iterdir()):
            if d.is_dir():
                subjects.append(d)
    
    return subjects


def load_images(subject_dir: Path, max_images: int = 10) -> list:
    """載入前 N 張影像"""
    valid_ext = {'.jpg', '.jpeg', '.png'}
    images = []
    
    for f in sorted(subject_dir.iterdir()):
        if f.suffix.lower() in valid_ext:
            img = cv2.imread(str(f))
            if img is not None:
                images.append(img)
            if len(images) >= max_images:
                break
    
    return images


def main():
    logger.info("=" * 60)
    logger.info("MiVOLO 年齡預測")
    logger.info("=" * 60)
    
    # 路徑設定
    path_file = project_root / "data/images/raw/path.txt"
    output_file = project_root / "workspace/predicted_ages.json"
    
    raw_dir = read_raw_path(path_file)
    logger.info(f"影像目錄: {raw_dir}")
    
    # 初始化模型
    logger.info("初始化 MiVOLO...")
    predictor = MiVOLOPredictor()
    predictor.initialize()
    
    # 掃描受試者
    subjects = scan_subjects(raw_dir)
    logger.info(f"找到 {len(subjects)} 個受試者")
    
    # 預測
    results = {}
    
    for subject_dir in tqdm(subjects, desc="預測年齡"):
        subject_id = subject_dir.name
        images = load_images(subject_dir)
        
        if not images:
            logger.warning(f"{subject_id}: 無影像")
            continue
        
        age = predictor.predict(images)
        
        if age is not None:
            results[subject_id] = round(age, 1)
        else:
            logger.warning(f"{subject_id}: 預測失敗")
    
    # 儲存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ 完成: {len(results)}/{len(subjects)} 個受試者")
    logger.info(f"儲存至: {output_file}")


if __name__ == "__main__":
    main()