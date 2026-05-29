"""
年齡預測器

支援模型: MiVOLO, InsightFace, DeepFace, FairFace, OpenCV DNN
"""

import os
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class MiVOLOPredictor:
    """MiVOLO v2 年齡預測器"""

    # Haar 偵測框面積佔整張圖比例低於此值，視為誤判（常誤抓背景/衣物的小區塊），
    # 改用整張圖預測。實測：真臉框佔 12~40%、誤判框佔 0.4~5%，8% 可乾淨區隔。
    MIN_FACE_AREA_FRAC = 0.08

    def __init__(self):
        self.model = None
        self.processor = None
        self.face_detector = None
        self.device = None
    
    def initialize(self):
        """載入模型"""
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self.model = AutoModelForImageClassification.from_pretrained(
                "iitolstykh/mivolo_v2",
                trust_remote_code=True,
                dtype=dtype
            )
            self.processor = AutoImageProcessor.from_pretrained(
                "iitolstykh/mivolo_v2",
                trust_remote_code=True
            )
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.model.eval()
            logger.info(f"✓ MiVOLO 初始化完成 ({self.device.upper()})")
            
        except Exception as e:
            raise RuntimeError(f"MiVOLO 初始化失敗: {e}")
    
    def face_crop(self, image: np.ndarray) -> np.ndarray:
        """回傳實際餵入模型的人臉裁切。

        用 Haar 偵測最大臉框並外擴 30% margin。但 Haar 常在背景/衣物誤抓
        ~1% 的小框（裁出來不是臉，會被估成年輕人），故加最小臉框守門：
        偵測不到臉、或最大框面積佔比 < MIN_FACE_AREA_FRAC 時，一律改用整張圖
        （影像已對齊、臉在中央，整張圖預測仍正確）。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) == 0:
            return image

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        area_frac = (w * h) / (image.shape[0] * image.shape[1])
        if area_frac < self.MIN_FACE_AREA_FRAC:
            return image  # 框過小，視為誤判 → 整張圖

        margin = int(max(w, h) * 0.3)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)
        return image[y1:y2, x1:x2]

    def predict_single(self, image: np.ndarray) -> Optional[float]:
        """預測單張影像的年齡"""
        import torch

        try:
            face_crop = self.face_crop(image)

            # 預處理
            inputs = self.processor(images=[face_crop])["pixel_values"]
            inputs = inputs.to(dtype=self.model.dtype, device=self.model.device)
            
            # 推論
            with torch.no_grad():
                outputs = self.model(faces_input=inputs, body_input=inputs)
            
            if hasattr(outputs, 'age_output'):
                return outputs.age_output[0].item()
                
        except Exception as e:
            logger.debug(f"預測失敗: {e}")
        
        return None

    def predict(self, images: List[np.ndarray]) -> List[float]:
        """
        預測多張影像的年齡
        
        Args:
            images: BGR 影像列表
            
        Returns:
            預測年齡列表（僅包含成功預測的結果）
        """
        ages = []
        for img in images:
            age = self.predict_single(img)
            if age is not None:
                ages.append(age)

        return ages


class InsightFacePredictor:
    """InsightFace (buffalo_l) 年齡預測器"""

    def __init__(self):
        self._app = None

    def initialize(self):
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise RuntimeError("insightface 未安裝")

        try:
            self._app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace 初始化完成")
        except Exception as e:
            raise RuntimeError(f"InsightFace 初始化失敗: {e}")

    def predict_single(self, image: np.ndarray) -> Optional[float]:
        try:
            # InsightFace 期待 BGR：其內部 blobFromImage 已設 swapRB=True
            # （見 model_zoo/attribute.py），會自行轉 RGB。不要在這裡先轉，
            # 否則會雙重交換導致 R/B 通道顛倒。
            faces = self._app.get(image)
            if faces:
                return float(faces[0].age)
        except Exception as e:
            logger.debug(f"InsightFace 預測失敗: {e}")
        return None

    def predict(self, images: List[np.ndarray]) -> List[float]:
        return [a for img in images if (a := self.predict_single(img)) is not None]


class DeepFacePredictor:
    """DeepFace 年齡預測器"""

    def __init__(self):
        self._deepface = None

    def initialize(self):
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
        except ImportError:
            raise RuntimeError("deepface 未安裝")

        try:
            test_img = np.zeros((224, 224, 3), dtype=np.uint8)
            self._deepface.analyze(
                img_path=test_img, actions=['age'], enforce_detection=False,
                silent=True,
            )
            logger.info("DeepFace 初始化完成")
        except Exception as e:
            raise RuntimeError(f"DeepFace 初始化失敗: {e}")

    def predict_single(self, image: np.ndarray) -> Optional[float]:
        try:
            result = self._deepface.analyze(
                img_path=image, actions=['age'], enforce_detection=False,
                silent=True,
            )
            if result:
                return float(result[0]['age'])
        except Exception as e:
            logger.debug(f"DeepFace 預測失敗: {e}")
        return None

    def predict(self, images: List[np.ndarray]) -> List[float]:
        return [a for img in images if (a := self.predict_single(img)) is not None]


class FairFacePredictor:
    """FairFace (ResNet34) 年齡預測器"""

    AGE_BINS = ["0-2", "3-9", "10-19", "20-29", "30-39",
                "40-49", "50-59", "60-69", "70+"]
    AGE_MIDPOINTS = [1.0, 6.0, 14.5, 24.5, 34.5,
                     44.5, 54.5, 64.5, 75.0]
    WEIGHT_FILENAME = "fairface_alldata_20191111.pt"

    def __init__(self):
        self._model = None
        self._transform = None
        self._device = None

    def initialize(self):
        import torch
        from torchvision import transforms, models

        weight_dir = PROJECT_ROOT / "external" / "age" / "fairface"
        weight_path = weight_dir / self.WEIGHT_FILENAME
        # 也接受舊版權重名稱
        if not weight_path.exists():
            alt = weight_dir / "res34_fair_align_multi_7_20190809.pt"
            if alt.exists():
                weight_path = alt
        if not weight_path.exists():
            raise FileNotFoundError(
                f"FairFace 權重不存在: {weight_path}\n"
                f"請從 https://drive.google.com/drive/folders/"
                f"1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu 下載並放到 {weight_dir}/"
            )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # FairFace uses ResNet34 with 18 outputs: 7 race + 2 gender + 9 age
        model = models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 18)
        state = torch.load(str(weight_path), map_location=self._device,
                           weights_only=True)
        model.load_state_dict(state)
        model.to(self._device).eval()
        self._model = model

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        logger.info(f"FairFace 初始化完成 ({self._device.upper()})")

    def predict_single(self, image: np.ndarray) -> Optional[float]:
        import torch

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self._transform(image_rgb).unsqueeze(0).to(self._device)
            with torch.no_grad():
                out = self._model(tensor)
            # age logits are the last 9 outputs
            age_logits = out[0, 9:18]
            probs = torch.softmax(age_logits, dim=0).cpu().numpy()
            midpoints = np.array(self.AGE_MIDPOINTS)
            return float((probs * midpoints).sum())
        except Exception as e:
            logger.debug(f"FairFace 預測失敗: {e}")
        return None

    def predict(self, images: List[np.ndarray]) -> List[float]:
        return [a for img in images if (a := self.predict_single(img)) is not None]


class OpenCVDNNPredictor:
    """OpenCV DNN (Caffe) 年齡預測器"""

    AGE_BINS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
                "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
    AGE_MIDPOINTS = [1.0, 5.0, 10.0, 17.5,
                     28.5, 40.5, 50.5, 80.0]
    MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
    PROTO_URL = ("https://raw.githubusercontent.com/spmallick/"
                 "learnopencv/master/AgeGender/age_deploy.prototxt")
    MODEL_URL = ("https://github.com/spmallick/learnopencv/raw/"
                 "master/AgeGender/age_net.caffemodel")

    def __init__(self):
        self._net = None
        self._face_detector = None

    def initialize(self):
        model_dir = PROJECT_ROOT / "external" / "age" / "opencv_age"
        proto_path = model_dir / "age_deploy.prototxt"
        model_path = model_dir / "age_net.caffemodel"

        if not proto_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"OpenCV DNN 模型不存在: {model_dir}\n"
                f"請下載 age_deploy.prototxt 和 age_net.caffemodel 到 {model_dir}/"
            )

        self._net = cv2.dnn.readNet(str(model_path), str(proto_path))
        self._face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        logger.info("OpenCV DNN 初始化完成")

    def predict_single(self, image: np.ndarray) -> Optional[float]:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._face_detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if len(faces) == 0:
                face_crop = image
            else:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                margin = int(max(w, h) * 0.2)
                x1, y1 = max(0, x - margin), max(0, y - margin)
                x2, y2 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)
                face_crop = image[y1:y2, x1:x2]

            blob = cv2.dnn.blobFromImage(
                face_crop, 1.0, (227, 227), self.MODEL_MEAN, swapRB=False,
            )
            self._net.setInput(blob)
            preds = self._net.forward()
            probs = preds[0]
            midpoints = np.array(self.AGE_MIDPOINTS)
            return float((probs * midpoints).sum())
        except Exception as e:
            logger.debug(f"OpenCV DNN 預測失敗: {e}")
        return None

    def predict(self, images: List[np.ndarray]) -> List[float]:
        return [a for img in images if (a := self.predict_single(img)) is not None]


PREDICTOR_MAP = {
    "mivolo": MiVOLOPredictor,
    "insightface": InsightFacePredictor,
    "deepface": DeepFacePredictor,
    "fairface": FairFacePredictor,
    "opencv_dnn": OpenCVDNNPredictor,
}

BENCHMARK_DIR_NAMES = {
    "mivolo": "1_MiVOLO",
    "insightface": "2_InsightFace",
    "deepface": "3_DeepFace",
    "fairface": "4_FairFace",
    "opencv_dnn": "5_OpenCV_DNN",
}