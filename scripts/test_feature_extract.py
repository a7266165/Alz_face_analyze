"""
scripts/test_feature_extract.py
測試特徵提取模組 - 使用真實影像與預處理流程
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import logging
import json
import time
from typing import List, Optional, Tuple

# 加入專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.feature_extract import FeatureExtractor
from src.core.preprocess import FacePreprocessor, ProcessedFace
from src.core.config import PreprocessConfig

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractorTester:
    """特徵提取器測試類"""
    
    def __init__(self, test_images_dir: Optional[Path] = None):
        """
        初始化測試器
        
        Args:
            test_images_dir: 測試影像目錄
        """
        self.test_images_dir = test_images_dir
        self.results = {}
        
    def run_all_tests(self):
        """執行所有測試"""
        print("\n" + "=" * 70)
        print("特徵提取模組測試")
        print("=" * 70)
        
        # 1. 初始化測試
        print("\n[測試 1] 初始化特徵提取器")
        print("-" * 40)
        extractor = self.test_initialization()
        
        if not extractor:
            print("✗ 初始化失敗，無法繼續測試")
            return
        
        # 2. 狀態報告
        print("\n[測試 2] 狀態報告")
        print("-" * 40)
        self.test_status_report(extractor)
        
        # 3. 檢查測試影像目錄
        if not self.test_images_dir or not self.test_images_dir.exists():
            print("\n✗ 測試影像目錄不存在，無法繼續測試")
            print("請設定有效的測試影像目錄")
            return
        
        # 3. 完整 Pipeline
        print("\n[測試 3] 完整 Pipeline (預處理 + 特徵提取)")
        print("-" * 40)
        self.test_full_pipeline(extractor)
        
        # 4. 差異計算測試
        print("\n[測試 4] 差異計算")
        print("-" * 40)
        self.test_difference_calculation(extractor)
        
        # 5. 人口學特徵測試
        print("\n[測試 5] 人口學特徵整合")
        print("-" * 40)
        self.test_demographics(extractor)
        
        # 6. 效能測試
        print("\n[測試 6] 效能測試")
        print("-" * 40)
        self.test_performance(extractor)
        
        # 7. 錯誤處理測試
        print("\n[測試 7] 錯誤處理")
        print("-" * 40)
        self.test_error_handling(extractor)
        
        # 總結
        self.print_summary()
    
    # ========== 核心測試方法 ==========
    
    def test_initialization(self) -> Optional[FeatureExtractor]:
        """測試初始化"""
        try:
            extractor = FeatureExtractor()
            print(f"✓ 基本初始化成功")
            
            self.results["initialization"] = "通過"
            return extractor
            
        except Exception as e:
            print(f"✗ 初始化失敗: {e}")
            self.results["initialization"] = f"失敗: {e}"
            return None
    
    def test_status_report(self, extractor: FeatureExtractor):
        """測試狀態報告"""
        try:
            status = extractor.get_status_report()
            
            print(f"可用模型: {status['available_models']}")
            print(f"模型維度: {status['model_dimensions']}")
            print(f"GPU 支援: {status['has_gpu']}")
            
            if status['has_gpu']:
                print(f"GPU 名稱: {status['gpu_name']}")
            
            if not status['available_models']:
                print("⚠ 警告：沒有任何可用模型")
                self.results["status_report"] = "警告：無可用模型"
            else:
                print(f"✓ 狀態報告正常")
                self.results["status_report"] = "通過"
                
        except Exception as e:
            print(f"✗ 狀態報告失敗: {e}")
            self.results["status_report"] = f"失敗: {e}"
    
    # ========== 測試方法 ==========
    
    def test_full_pipeline(self, extractor: FeatureExtractor):
        """測試完整 Pipeline：原始影像 → 預處理 → 特徵提取 → 差異計算"""
        
        # 尋找測試受試者
        subject_dir = self._find_test_subject()
        if not subject_dir:
            print("✗ 找不到測試受試者")
            self.results["full_pipeline"] = "跳過：無測試資料"
            return
        
        print(f"測試受試者: {subject_dir.name}")
        print("\n完整 Pipeline 流程:")
        print("  原始影像 → 預處理 → 特徵提取 → 差異計算 → 人口學整合")
        
        try:
            # Step 1: 載入原始影像
            print("\n[Step 1] 載入原始影像...")
            images, paths = self._load_images_from_subject(subject_dir, max_count=10)
            print(f"  ✓ 載入 {len(images)} 張")
            
            # Step 2: 預處理
            print("\n[Step 2] 預處理...")
            config = PreprocessConfig(
                n_select=min(10, len(images)),
                save_intermediate=True,
                workspace_dir=Path("workspace/test_pipeline"),
                steps=['select', 'align', 'mirror', 'clahe']
            )
            
            with FacePreprocessor(config) as preprocessor:
                processed = preprocessor.process(images, paths)
            
            print(f"  ✓ 產生 {len(processed)} 對鏡射影像")
            
            # Step 3: 特徵提取（多模型同時提取）
            print("\n[Step 3] 特徵提取...")
            
            pipeline_results = {}
            
            try:
                # 收集所有鏡射影像
                all_left = [p.left_mirror for p in processed]
                all_right = [p.right_mirror for p in processed]
                
                # 一次提取所有模型的特徵
                print("  批次提取所有可用模型...")
                left_features_dict = extractor.extract_features(all_left, models=None)
                right_features_dict = extractor.extract_features(all_right, models=None)
                
                # 對每個模型處理
                for model in extractor.get_available_models():
                    print(f"\n  模型: {model}")
                    
                    try:
                        left_features = left_features_dict[model]
                        right_features = right_features_dict[model]
                        
                        # 過濾掉 None
                        valid_pairs = [
                            (l, r) for l, r in zip(left_features, right_features)
                            if l is not None and r is not None
                        ]
                        
                        print(f"    成功提取: {len(valid_pairs)}/{len(processed)} 對")
                        
                        if not valid_pairs:
                            pipeline_results[model] = "失敗：無有效特徵"
                            continue
                        
                        # Step 4: 計算差異
                        print(f"  [Step 4] 計算差異...")
                        
                        all_diffs = []
                        for left, right in valid_pairs:
                            diffs = extractor.calculate_differences(left, right, ["differences", "averages", "relative"])
                            all_diffs.append(diffs['embedding_differences'])
                        
                        # 平均差異
                        avg_diff = np.mean(all_diffs, axis=0)
                        print(f"    平均差異範圍: [{avg_diff.min():.3f}, {avg_diff.max():.3f}]")
                        
                        # Step 5: 人口學整合
                        print(f"  [Step 5] 人口學整合...")
                        age, gender = 70.0, 1.0  # 模擬人口學資料
                        
                        combined_list = extractor.add_demographics([avg_diff], [age], [gender])
                        combined = combined_list[0]  # 取出結果

                        print(f"    原始維度: {len(avg_diff)}")
                        print(f"    整合後維度: {len(combined)}")
                        print(f"    人口學特徵: age={age}, gender={gender}")

                        # 驗證
                        is_valid = extractor.validate_features(combined)
                        print(f"    特徵有效性: {is_valid}")

                        pipeline_results[model] = "通過" if is_valid else "失敗：特徵無效"
                        
                    except Exception as e:
                        print(f"    ✗ {model} 處理失敗: {e}")
                        pipeline_results[model] = f"失敗: {e}"
                
                self.results["full_pipeline"] = pipeline_results
                
            except Exception as e:
                print(f"✗ 特徵提取階段失敗: {e}")
                self.results["full_pipeline"] = f"失敗: {e}"
            
        except Exception as e:
            print(f"✗ Pipeline 測試失敗: {e}")
            self.results["full_pipeline"] = f"失敗: {e}"
    
    def test_difference_calculation(self, extractor: FeatureExtractor):
        """測試差異計算"""
        try:
            left_features = np.random.randn(128).astype(np.float32)
            right_features = np.random.randn(128).astype(np.float32)
            
            # 測試：必須明確指定 methods
            print("測試 1: 未指定 methods（應該報錯）")
            try:
                extractor.calculate_differences(left_features, right_features)
                print("  ✗ 應該報錯但沒有")
                self.results["difference_calculation"] = "失敗：未驗證 methods=None"
                return
            except ValueError as e:
                print(f"  ✓ 正確報錯: {e}")
            
            # 測試：無效方法名稱
            print("\n測試 2: 無效方法名稱（應該報錯）")
            try:
                extractor.calculate_differences(
                    left_features, right_features, ["invalid_method"]
                )
                print("  ✗ 應該報錯但沒有")
                self.results["difference_calculation"] = "失敗：未驗證方法名稱"
                return
            except ValueError as e:
                print(f"  ✓ 正確報錯: {e}")
            
            # 測試：單一方法
            print("\n測試 3: 單一方法")
            for method in ["differences", "averages", "relative"]:
                result = extractor.calculate_differences(
                    left_features, right_features, [method]
                )
                
                # 應該返回字典
                if not isinstance(result, dict):
                    print(f"  ✗ {method}: 應該返回字典")
                    self.results["difference_calculation"] = f"失敗：{method} 返回類型錯誤"
                    return
                
                # 檢查鍵名
                expected_key = f"embedding_{method}" if method != "differences" else "embedding_differences"
                if method == "relative":
                    expected_key = "relative_differences"
                
                if expected_key not in result:
                    print(f"  ✗ {method}: 缺少鍵 {expected_key}")
                    self.results["difference_calculation"] = f"失敗：{method} 鍵名錯誤"
                    return
                
                value = result[expected_key]
                print(f"  ✓ {method}: shape={value.shape}, "
                    f"range=[{value.min():.3f}, {value.max():.3f}]")
            
            # 測試：多種方法
            print("\n測試 4: 多種方法")
            result = extractor.calculate_differences(
                left_features, right_features, 
                ["differences", "averages", "relative"]
            )
            
            expected_keys = {
                "embedding_differences", 
                "embedding_averages", 
                "relative_differences"
            }
            
            if set(result.keys()) != expected_keys:
                print(f"  ✗ 鍵不匹配: 期望 {expected_keys}, 得到 {set(result.keys())}")
                self.results["difference_calculation"] = "失敗：多方法鍵名錯誤"
                return
            
            print("  ✓ 多種方法計算成功")
            for key, value in result.items():
                print(f"    - {key}: shape={value.shape}, "
                    f"range=[{value.min():.3f}, {value.max():.3f}]")
            
            self.results["difference_calculation"] = "通過"
            
        except Exception as e:
            print(f"✗ 差異計算失敗: {e}")
            self.results["difference_calculation"] = f"失敗: {e}"
    
    def test_demographics(self, extractor: FeatureExtractor):
        """測試人口學特徵整合"""
        try:
            # 準備批次測試資料
            features_list = [np.random.randn(128).astype(np.float32) for _ in range(3)]
            ages = [65.0, 70.0, 75.0]
            genders = [1.0, 0.0, 1.0]
            
            # 批次整合
            combined_list = extractor.add_demographics(
                features_list, ages, genders
            )
            
            print(f"批次整合:")
            print(f"  批次大小: {len(combined_list)}")
            print(f"  原始維度: {[len(f) for f in features_list]}")
            print(f"  整合後維度: {[len(c) for c in combined_list]}")
            print(f"  預期維度: {[len(f) + 2 for f in features_list]}")
            
            # 驗證維度
            expected_dims = [len(f) + 2 for f in features_list]
            actual_dims = [len(c) for c in combined_list]
            
            if actual_dims == expected_dims:
                print(f"  ✓ 維度檢查通過")
                self.results["demographics"] = "通過"
            else:
                print(f"  ✗ 維度不符: 預期 {expected_dims}, 實際 {actual_dims}")
                self.results["demographics"] = "失敗：維度不符"
            
        except Exception as e:
            print(f"✗ 人口學特徵整合失敗: {e}")
            self.results["demographics"] = f"失敗: {e}"
    
    def test_performance(self, extractor: FeatureExtractor):
        """測試效能"""
        try:
            # 從測試受試者載入真實影像
            subject_dir = self._find_test_subject()
            if not subject_dir:
                print("✗ 找不到測試影像")
                self.results["performance"] = "跳過：無測試資料"
                return
            
            images, _ = self._load_images_from_subject(subject_dir, max_count=1)
            if not images:
                print("✗ 無法載入影像")
                self.results["performance"] = "失敗：無影像"
                return
            
            test_image = images[0]
            iterations = 10
            
            print(f"執行 {iterations} 次提取測試...")
            
            # 使用新的 extract_features API
            for model in extractor.get_available_models():
                times = []
                
                for _ in range(iterations):
                    start = time.time()
                    _ = extractor.extract_features([test_image], model)
                    elapsed = time.time() - start
                    times.append(elapsed)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                print(f"{model}:")
                print(f"  平均: {avg_time*1000:.2f} ms")
                print(f"  標準差: {std_time*1000:.2f} ms")
                print(f"  FPS: {1/avg_time:.2f}")
            
            self.results["performance"] = "通過"
            
        except Exception as e:
            print(f"✗ 效能測試失敗: {e}")
            self.results["performance"] = f"失敗: {e}"

    def test_error_handling(self, extractor: FeatureExtractor):
        """測試錯誤處理"""
        error_count = 0
        warning_count = 0
        
        test_cases = [
            ("None 輸入", None),
            ("空陣列", np.array([])),
            ("錯誤形狀", np.random.randn(10)),
            ("錯誤類型", "not_an_image"),
            ("全黑影像", np.zeros((224, 224, 3), dtype=np.uint8)),
            ("全白影像", np.ones((224, 224, 3), dtype=np.uint8) * 255)
        ]
        
        for case_name, invalid_input in test_cases:
            print(f"\n測試 {case_name}...")
            
            # 使用新的 extract_features API
            for model in extractor.get_available_models():
                try:
                    features_dict = extractor.extract_features([invalid_input], model)
                    result = features_dict[model][0] if features_dict.get(model) else None
                    
                    if case_name in ["全黑影像", "全白影像"]:
                        if result is not None:
                            print(f"  ⚠ {model}: 成功提取（可能是正常行為）")
                            warning_count += 1
                        else:
                            print(f"  ✓ {model}: 返回 None")
                    else:
                        if result is None:
                            print(f"  ✓ {model}: 正確處理（返回 None）")
                        else:
                            print(f"  ✗ {model}: 意外成功")
                            error_count += 1
                            
                except Exception as e:
                    print(f"  ✗ {model}: 拋出例外: {type(e).__name__}")
                    error_count += 1
        
        # 測試無效模型名稱
        print(f"\n測試無效模型名稱...")
        
        # 載入一張真實測試影像
        subject_dir = self._find_test_subject()
        if subject_dir:
            images, _ = self._load_images_from_subject(subject_dir, max_count=1)
            test_image = images[0] if images else np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        try:
            features_dict = extractor.extract_features([test_image], "invalid_model")
            # 應該返回空字典或不包含該模型
            if not features_dict or "invalid_model" not in features_dict:
                print("  ✓ 正確處理無效模型名稱")
            else:
                error_count += 1
        except Exception:
            # 拋出例外也算正確處理
            print("  ✓ 正確處理無效模型名稱（拋出例外）")
        
        if error_count == 0:
            self.results["error_handling"] = "通過"
        elif warning_count > 0 and error_count == 0:
            self.results["error_handling"] = f"通過（有 {warning_count} 個警告）"
        else:
            self.results["error_handling"] = f"失敗: {error_count} 個錯誤"
    
    # ========== 輔助工具方法 ==========
    
    def _find_test_subject(self) -> Optional[Path]:
        """尋找測試用受試者目錄"""
        if not self.test_images_dir or not self.test_images_dir.exists():
            return None
        
        # 嘗試順序：ACS → NAD → P
        search_paths = [
            self.test_images_dir / "health" / "ACS",
            self.test_images_dir / "health" / "NAD",
            self.test_images_dir / "patient"
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            # 找第一個有影像的受試者
            subjects = sorted([d for d in search_path.iterdir() if d.is_dir()])
            for subject_dir in subjects:
                images = self._load_images_from_subject(subject_dir, max_count=1)
                if images[0]:  # 至少有一張影像
                    return subject_dir
        
        return None
    
    def _load_images_from_subject(
        self, 
        subject_dir: Path, 
        max_count: int = 20
    ) -> Tuple[List[np.ndarray], List[Path]]:
        """
        從受試者目錄載入影像
        
        Returns:
            (影像列表, 路徑列表)
        """
        images = []
        paths = []
        
        # 支援的格式
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # 收集檔案
        image_files = []
        for ext in extensions:
            image_files.extend(subject_dir.glob(f"*{ext}"))
            image_files.extend(subject_dir.glob(f"*{ext.upper()}"))
        
        # 去重並排序
        image_files = sorted(set(image_files))[:max_count]
        
        # 載入
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                paths.append(img_path)
        
        return images, paths
    
    def print_summary(self):
        """列印測試總結"""
        print("\n" + "=" * 70)
        print("測試總結")
        print("=" * 70)
        
        passed = 0
        failed = 0
        warnings = 0
        
        for test_name, result in self.results.items():
            if isinstance(result, dict):
                all_passed = all(
                    "通過" in str(v) or "成功" in str(v) 
                    for v in result.values()
                )
                if all_passed:
                    status = "✓"
                    passed += 1
                else:
                    status = "✗"
                    failed += 1
            elif isinstance(result, str):
                if "通過" in result:
                    status = "✓"
                    passed += 1
                elif "警告" in result:
                    status = "⚠"
                    warnings += 1
                elif "成功" in result:
                    status = "✓"
                    passed += 1
                else:
                    status = "✗"
                    failed += 1
            else:
                status = "?"
                warnings += 1
            
            print(f"{status} {test_name}: {result}")
        
        print("\n" + "-" * 40)
        print(f"通過: {passed}")
        print(f"失敗: {failed}")
        print(f"警告: {warnings}")
        print(f"總計: {len(self.results)}")
        
        if failed == 0:
            print("\n🎉 所有測試通過！")
        else:
            print(f"\n⚠ 有 {failed} 個測試失敗")
        
        self.save_results()
    
    def save_results(self):
        """儲存測試結果"""
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / f"feature_extract_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": self.results,
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n測試報告已儲存: {report_path}")


def main():
    """主測試函數"""
    
    # 設定測試影像目錄
    test_images_dir = None
    
    # 檢查可能的位置
    possible_dirs = [
        Path("D:/project/Alz/face/data/datung/raw"),  # 從 path.txt
        Path("data/images/raw"),  # 相對路徑
        Path("../data/images/raw"),  # 上層
    ]
    
    for dir_path in possible_dirs:
        if dir_path.exists():
            test_images_dir = dir_path
            logger.info(f"使用測試影像目錄: {test_images_dir}")
            break
    
    if not test_images_dir:
        logger.error("✗ 找不到測試影像目錄")
        logger.info("請確保以下任一目錄存在：")
        for dir_path in possible_dirs:
            logger.info(f"  - {dir_path}")
        return
    
    # 執行測試
    tester = FeatureExtractorTester(test_images_dir)
    tester.run_all_tests()


if __name__ == "__main__":
    main()