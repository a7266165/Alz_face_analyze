"""
預處理：五個 pure-function 站 + 一個資源工廠，由 scripts/preprocess/run_preprocess.py 串接：
  detect → select → 去背(mask) → align(rotate) → mirror

mediapipe 的 FaceMesh 由 open_face_mesh() 託管，狀態留在協調者，五站本身皆無狀態。
"""

from .detector import FaceInfo, open_face_mesh, detect_faces
from .selector import select_most_frontal
from .masker import build_mask, apply_mask
from .aligner import calculate_midline_tilt, rotate_to_vertical
from .mirror_generator import generate_mirrors

__all__ = [
    "FaceInfo",
    "open_face_mesh",
    "detect_faces",
    "select_most_frontal",
    "build_mask",
    "apply_mask",
    "calculate_midline_tilt",
    "rotate_to_vertical",
    "generate_mirrors",
]
