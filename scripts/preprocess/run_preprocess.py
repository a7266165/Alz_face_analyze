"""
掃描母資料夾所有個案的人臉相片資料夾，依序進行：
1. 臉部偵測
2. 選擇最正面相片
3. 去背（可選）
4. 轉正
5. 鏡射（可選）

並將中間過程存至workspace
"""

import os
import re
import sys
import json
import math
import shutil
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    RAW_IMAGES_DIR,
    PREPROCESSING_DIR,
    preprocess_dir,
    preprocess_selector_dir,
    PreprocessConfig,
    SELECTION,
    DEFAULT_SELECTION,
    OPENFACE_POSE_DIR,
    OPENFACE2_DIR,
    OPENFACE2_BIN,
    D435I_COLOR_INTRINSICS,
)
from src.preprocess import (
    open_face_mesh,
    detect_faces,
    select_most_frontal,
    select_by_head_pose,
    apply_mask,
    calculate_midline_tilt,
    rotate_to_vertical,
    generate_mirrors,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_SUBTREES = ["health/ACS", "health/NAD", "patient/good"]

# 選幀準則 -> 姿態閘門。key 同時是 workspace 子系統下的 selection 目錄名。
SELECTION_GATES = {
    "openface_p10r5": dict(pitch_max=10.0, roll_max=5.0),
    "openface_p15r10": dict(pitch_max=15.0, roll_max=10.0),
}
CONF_MIN = 0.725          # OpenFace 的 success 就是 confidence > 0.725
MEDFILT_WIN = 5
MIN_GAP_FRAMES = 13       # 0.5 秒 @ 實測約 26 fps
# 資料有兩種拍攝版本：1200 張是約 44 秒的左右轉動，30 張是約 0.6 秒的正面連拍。
# 後者沒有軌跡可言，濾波與局部極小不適用。
TRAJECTORY_MIN_FRAMES = 100


def frame_files(subject_dir: Path) -> dict:
    """{原始幀號: 檔案路徑}。不依賴目錄列舉順序（字典序會讓 image10 排在 image2 前）。

    只取檔名「結尾」的連續數字：NAD104_color_image377 -> 377。
    不可以把整個 stem 的數字串起來 —— 那會把受試者編號一起吃進去
    （"104"+"377"=104377），雖然排序恰好仍正確，但幀距計算會在位數
    變化處失效（frame 9 -> 1049、frame 10 -> 10410，差距變成 9361）。
    """
    out = {}
    for p in subject_dir.iterdir():
        if not (p.is_file() and p.suffix.lower() in SUFFIXES):
            continue
        m = re.search(r"(\d+)$", p.stem)
        if m:
            out[int(m.group(1))] = p
    return out


def stage_hardlinks(subject_dir: Path, staging_root: Path) -> Optional[Path]:
    """建立補零檔名的 hard link 目錄供 OpenFace 讀取。

    OpenFace 的 -fdir 是「字典序」列舉目錄，而原始檔名沒有補零，
    image10 會排在 image2 之前 —— 時序追蹤、中位數濾波、局部極小全部
    在錯亂的順序上運作。實測：修正順序後 yaw 的 lag-1 自相關由 0.80
    升到 0.998，且執行時間減半（追蹤真的能用，不必一直重跑人臉偵測）。

    hard link 在 NTFS 同磁碟區內是零複製的目錄項目，約 54 us/個。
    """
    frames = frame_files(subject_dir)
    if not frames:
        return None
    dst = staging_root / subject_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    for idx, src in frames.items():
        link = dst / f"f{idx:06d}{src.suffix.lower()}"
        if not link.exists():
            try:
                os.link(src, link)
            except OSError as exc:
                logger.error(f"hard link 失敗（需與原始資料同磁碟區）: {exc}")
                return None
    return dst


def load_pose_csv(csv_path: Path, subject_dir: Path):
    """讀 FeatureExtraction 輸出，回傳 (幀號, yaw, pitch, roll, confidence)。

    角度由弧度轉度。CSV 的 frame 欄是 1-based 的處理序，因為暫存目錄已補零，
    處理序等於幀號升冪，所以第 N 列對應 sorted(幀號)[N-1]。
    """
    import csv as _csv

    order = sorted(frame_files(subject_dir))
    idx, yaw, pitch, roll, conf = [], [], [], [], []
    with open(csv_path, newline="") as fh:
        for row in _csv.DictReader(fh):
            row = {k.strip(): v for k, v in row.items()}
            n = int(row["frame"]) - 1
            if n >= len(order):
                continue
            idx.append(order[n])
            pitch.append(math.degrees(float(row["pose_Rx"])))
            yaw.append(math.degrees(float(row["pose_Ry"])))
            roll.append(math.degrees(float(row["pose_Rz"])))
            conf.append(float(row["confidence"]))
    return idx, yaw, pitch, roll, conf


def run_openface(subject_dirs: List[Path], staging_root: Path,
                 pose_root: Path, n_workers: int, batch: int = 100) -> None:
    """對缺 pose 的 visit 批次跑 OpenFace 2.2 FeatureExtraction。

    一個 process 吃多個 -fdir：SequenceCapture::Open 會消耗掉用過的參數，
    外層 while(true) 依序處理每個目錄，所以 430 MB 的模型只載入一次。
    只傳 -pose 會跳過 AU 與 HOG 整段，是最大的加速槓桿。
    """
    todo = [d for d in subject_dirs if not (pose_root / f"{d.name}.csv").is_file()]
    if not todo:
        logger.info("pose 快取齊全，跳過 OpenFace")
        return
    if not OPENFACE2_BIN.is_file():
        raise FileNotFoundError(f"找不到 FeatureExtraction.exe: {OPENFACE2_BIN}")

    logger.info(f"OpenFace：{len(todo)} 個 visit 缺 pose，建立 hard link 暫存樹…")
    pose_root.mkdir(parents=True, exist_ok=True)
    staged = [s for s in (stage_hardlinks(d, staging_root) for d in todo) if s]
    logger.info(f"暫存完成 {len(staged)} 個；分 {n_workers} 片執行 FeatureExtraction")

    K = D435I_COLOR_INTRINSICS
    chunks = [staged[i:i + batch] for i in range(0, len(staged), batch)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}

    def launch(chunk):
        cmd = [str(OPENFACE2_BIN)]
        for d in chunk:
            cmd += ["-fdir", str(d)]
        cmd += ["-pose", "-out_dir", str(pose_root), "-q",
                "-fx", str(K["fx"]), "-fy", str(K["fy"]),
                "-cx", str(K["cx"]), "-cy", str(K["cy"])]
        return subprocess.run(cmd, cwd=str(OPENFACE2_DIR), env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for r in tqdm(pool.map(launch, chunks), total=len(chunks), desc="OpenFace"):
            if r.returncode != 0:
                logger.error(f"FeatureExtraction 失敗: {r.stderr.decode(errors='replace')[:300]}")

    shutil.rmtree(staging_root, ignore_errors=True)
    logger.info(f"pose 完成，已清除暫存樹；csv 落在 {pose_root}")


def limit_cpu(n_cores):
    if n_cores is None:
        return
    logger.info(f"CPU 核心數: 限制為 {n_cores}")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_cores)
    try:
        cv2.setNumThreads(n_cores)
    except Exception:
        pass


def already_done(subject_id: str, variants: List[Tuple[str, bool]],
                 mirror: bool) -> bool:
    """斷點檢查"""
    sel = preprocess_dir("selected") / subject_id
    if not sel.is_dir():
        return False
    k = sum(1 for _ in sel.glob("*.png"))
    if k == 0:
        return False
    for _, is_bg in variants:
        al = preprocess_dir("aligned", background=is_bg) / subject_id
        if sum(1 for _ in al.glob("*.png")) != k:
            return False
        if mirror:
            mr = preprocess_dir("mirrors", background=is_bg) / subject_id
            if sum(1 for _ in mr.glob("*_left.png")) != k:
                return False
    return True


def process_subject(subject_dir: Path, face_mesh, cfg: PreprocessConfig,
                    variants: List[Tuple[str, bool]], mirror: bool,
                    picks: Optional[List[int]] = None) -> bool:
    """picks=None 走原本的 VAS 排名；給了幀號就只讀那幾張（姿態準則已選好）。"""
    subject_id = subject_dir.name

    images, paths = [], []
    if picks is None:
        srcs = [p for p in sorted(subject_dir.iterdir())
                if p.is_file() and p.suffix.lower() in SUFFIXES]
    else:
        # 只載入選中的幀 —— 省掉為了排名而對全部 1200 張跑 FaceMesh 的成本
        if not picks:
            logger.warning(f"{subject_id}: 姿態閘門沒有任何幀通過")
            return False
        by_idx = frame_files(subject_dir)
        srcs = [by_idx[i] for i in picks if i in by_idx]

    for p in srcs:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(img)
            paths.append(p)
    if not images:
        logger.warning(f"{subject_id}: 沒有可讀的影像")
        return False

    faces = detect_faces(face_mesh, images, paths, cfg.midline_points)
    if not faces:
        logger.warning(f"{subject_id}: 未偵測到臉部")
        return False
    if picks is None:
        selected = select_most_frontal(faces, cfg.n_select)
    else:
        selected = faces          # 已由姿態準則選定，這裡不再篩
    if not selected:
        logger.warning(f"{subject_id}: select 後沒有臉部")
        return False

    # 合併兩迴圈 already_done 會失效
    sel_dir = preprocess_dir("selected") / subject_id
    sel_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(selected):
        cv2.imwrite(str(sel_dir / f"selected_{i:03d}_vas_{f.vertex_angle_sum:.1f}.png"),
                    f.image)

    for i, face in enumerate(selected):
        tilt = calculate_midline_tilt(face.landmarks, cfg.midline_points)
        stem = face.path.stem if face.path else None
        for _, is_bg in variants:
            src = face.image if is_bg else apply_mask(face.image, face.landmarks)
            aligned = rotate_to_vertical(src, tilt)

            al_dir = preprocess_dir("aligned", background=is_bg) / subject_id
            al_dir.mkdir(parents=True, exist_ok=True)
            al_name = f"{stem}_aligned.png" if stem else f"aligned_{i:03d}.png"
            cv2.imwrite(str(al_dir / al_name), aligned)

            if mirror:
                if cfg.mirror.mirror_method == "flip":
                    lm = face.landmarks
                else:
                    redet = detect_faces(face_mesh, [aligned],
                                         midline_points=cfg.midline_points)
                    lm = redet[0].landmarks if redet else face.landmarks
                left, right = generate_mirrors(aligned, lm, cfg.mirror)
                mr_dir = preprocess_dir("mirrors", background=is_bg) / subject_id
                mr_dir.mkdir(parents=True, exist_ok=True)
                base = stem if stem else f"face_{i:03d}"
                cv2.imwrite(str(mr_dir / f"{base}_left.png"), left)
                cv2.imwrite(str(mr_dir / f"{base}_right.png"), right)
    return True


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", type=Path, default=None,
                    help="覆寫 RAW_IMAGES_DIR；留空沿用 data/path.txt 設定")
    ap.add_argument("--subtrees", nargs="+", default=None,
                    help=f"要掃描的子樹（相對 input-root），取其直接子目錄為 subject；留空用預設 {DEFAULT_SUBTREES}")
    ap.add_argument("--n-select", type=int, default=10)
    ap.add_argument("--max-cpu-cores", type=int, default=2)
    ap.add_argument("--backgrounds", nargs="+",
                    choices=["no_background", "background"],
                    default=["no_background", "background"],
                    help="要產哪些背景變體（預設兩者都產）")
    ap.add_argument("--no-mirror", action="store_true",
                    help="不產鏡射（只到 aligned 為止）")
    ap.add_argument("--selection", default=None,
                    help="選幀準則，須與環境變數 ALZ_SELECTION 一致（僅作為確認，"
                         f"路徑在 import 時就已綁定）。可選 {DEFAULT_SELECTION} 或 "
                         f"{list(SELECTION_GATES)}")
    ap.add_argument("--staging-root", type=Path, default=None,
                    help="OpenFace 用的 hard link 暫存樹；必須與原始影像同磁碟區。"
                         "留空用 <input-root>/../_of_staging")
    ap.add_argument("--openface-workers", type=int, default=20,
                    help="同時執行的 FeatureExtraction process 數")
    ap.add_argument("--limit", type=int, default=None,
                    help="只處理前 N 個 subject（小規模驗證用）")
    args = ap.parse_args()

    # 路徑在 import src.config 時就依 ALZ_SELECTION 綁定完畢，這裡只能確認不能改
    if args.selection is not None and args.selection != SELECTION:
        logger.error(
            f"--selection={args.selection} 與 ALZ_SELECTION={SELECTION} 不一致。\n"
            f"選幀準則必須在 process 啟動前決定，請改用：\n"
            f"    ALZ_SELECTION={args.selection} python {Path(__file__).name} ...")
        sys.exit(1)
    if SELECTION != DEFAULT_SELECTION and SELECTION not in SELECTION_GATES:
        logger.error(f"未知的選幀準則 ALZ_SELECTION={SELECTION}；"
                     f"可選 {DEFAULT_SELECTION} 或 {list(SELECTION_GATES)}")
        sys.exit(1)

    limit_cpu(args.max_cpu_cores)

    raw_root = args.input_root if args.input_root is not None else RAW_IMAGES_DIR
    variants = [(name, name == "background") for name in args.backgrounds]
    mirror = not args.no_mirror

    # 不用rglob，結構目錄默認 raw_root/DEFAULT_SUBTREES/subject_ID/figs.jpg，
    subjects: List[Path] = []
    for sub in (args.subtrees or DEFAULT_SUBTREES):
        root = raw_root / sub
        if not root.exists():
            logger.warning(f"找不到子樹，略過: {root}")
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and any(f.suffix.lower() in SUFFIXES
                                  for f in d.iterdir() if f.is_file()):
                subjects.append(d)
    if args.limit:
        subjects = subjects[:args.limit]

    gate = SELECTION_GATES.get(SELECTION)
    logger.info("=" * 70)
    logger.info(f"預處理：raw → selected / aligned{' / mirrors' if mirror else ''}")
    logger.info(f"選幀準則: {SELECTION}"
                + (f"  閘門 |pitch|<={gate['pitch_max']} |roll|<={gate['roll_max']} "
                   f"conf>{CONF_MIN}" if gate else "  （MediaPipe 頂角和排名）"))
    logger.info(f"輸出根目錄: {PREPROCESSING_DIR}  選幀層: selector_{SELECTION}")
    logger.info(f"影像來源: {raw_root}  子樹: {args.subtrees or DEFAULT_SUBTREES}")
    logger.info(f"背景變體: {[n for n, _ in variants]}  產鏡射: {mirror}")
    logger.info(f"找到 {len(subjects)} 個受試者")
    logger.info("=" * 70)
    if not subjects:
        logger.error("沒有找到任何受試者目錄")
        return

    # ---- 姿態準則：先確保 pose 快取齊全（兩個閘門共用，只算一次）----
    picks_by_id = {}
    if gate is not None:
        staging = (args.staging_root
                   or raw_root.parent / "_of_staging")
        run_openface(subjects, staging, OPENFACE_POSE_DIR, args.openface_workers)

        n_traj = 0
        for d in tqdm(subjects, desc="挑幀"):
            csv_path = OPENFACE_POSE_DIR / f"{d.name}.csv"
            if not csv_path.is_file():
                picks_by_id[d.name] = []
                continue
            idx, yaw, pitch, roll, conf = load_pose_csv(csv_path, d)
            traj = len(idx) >= TRAJECTORY_MIN_FRAMES
            n_traj += traj
            picks_by_id[d.name] = select_by_head_pose(
                idx, yaw, pitch, roll, conf,
                n_select=args.n_select, conf_min=CONF_MIN,
                medfilt_win=MEDFILT_WIN, min_gap=MIN_GAP_FRAMES,
                trajectory=traj, **gate)

        counts = [len(v) for v in picks_by_id.values()]
        full = sum(c >= args.n_select for c in counts)
        empty = sum(c == 0 for c in counts)
        logger.info(f"挑幀完成：{n_traj} 個轉動版本 / {len(subjects) - n_traj} 個連拍版本；"
                    f"湊滿 {args.n_select} 張 {full}、掛零 {empty}、"
                    f"平均 {sum(counts)/max(len(counts),1):.2f} 張")

        # 產出率明細與閘門設定，供論文引用與後續決策。
        # selector 層現在在去背變體之下，沒有單一的「選幀根目錄」，
        # 所以每個產出的變體各寫一份（兩個檔都只有幾十 KB）。
        config_txt = json.dumps({
            "selection": SELECTION,
            "gate": {**gate, "conf_min": CONF_MIN},
            "medfilt_win": MEDFILT_WIN,
            "min_gap_frames": MIN_GAP_FRAMES,
            "fps_assumed": 26,
            "n_select": args.n_select,
            "trajectory_min_frames": TRAJECTORY_MIN_FRAMES,
            "intrinsics": D435I_COLOR_INTRINSICS,
            "openface": "2.2.0 win_x64, FeatureExtraction -pose",
        }, indent=2, ensure_ascii=False)
        yield_rows = ["visit,n_frames_raw,n_picked,trajectory"]
        for d in subjects:
            p = picks_by_id.get(d.name, [])
            nraw = len(frame_files(d))
            yield_rows.append(
                f"{d.name},{nraw},{len(p)},{int(nraw >= TRAJECTORY_MIN_FRAMES)}")
        for _, is_bg in variants:
            sel_root = preprocess_selector_dir(background=is_bg)
            sel_root.mkdir(parents=True, exist_ok=True)
            (sel_root / "_config.json").write_text(config_txt, encoding="utf-8")
            (sel_root / "_yield.csv").write_text(
                "\n".join(yield_rows) + "\n", encoding="utf-8")
            logger.info(f"閘門設定與產出率已寫入 {sel_root}")

    cfg = PreprocessConfig(n_select=args.n_select)
    start = datetime.now()
    n_ok = n_fail = n_skip = 0
    with open_face_mesh(cfg.detection_confidence) as fm:
        for subject_dir in tqdm(subjects, desc="預處理"):
            subject_id = subject_dir.name
            if already_done(subject_id, variants, mirror):
                n_skip += 1
                continue
            try:
                picks = picks_by_id.get(subject_id) if gate is not None else None
                if process_subject(subject_dir, fm, cfg, variants, mirror, picks):
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as e:
                n_fail += 1
                logger.error(f"✗ {subject_id}: {e}")
                import traceback
                traceback.print_exc()

    logger.info("=" * 70)
    logger.info(f"完成：成功 {n_ok}、失敗 {n_fail}、跳過(斷點) {n_skip}"
                f"、共 {len(subjects)}；耗時 {datetime.now() - start}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
