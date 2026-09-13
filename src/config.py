"""
全專案共用配置

路徑常數、專案級設定、處理參數
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from src.common.mediapipe_utils import MIDLINE_POINTS


# =============================================================================
# 路徑常數
# =============================================================================

# 專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent

# -----------------------------------------------------------------------------
# 外部根目錄由 repo 根的 paths.txt 宣告（KEY=路徑，一行一鍵；gitignore，範本見 paths.example.txt）。
# 缺鍵時依 D:\Alz 佈局相對推導：parents[1] = 子主題根（face\）、parents[2] = 主題根（D:\Alz）。
# 推導出的路徑不存在即報錯，避免在桌面副本等錯誤位置靜默讀寫。
# -----------------------------------------------------------------------------
_PATHS_FILE = PROJECT_ROOT / "paths.txt"
_SUBTHEME_ROOT = PROJECT_ROOT.parents[1]
_ALZ_ROOT = PROJECT_ROOT.parents[2]


def _read_paths() -> dict:
    out = {}
    if _PATHS_FILE.exists():
        for line in _PATHS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


_PATHS = _read_paths()


def _path(key: str, default: Path, must_exist: bool = True) -> Path:
    p = Path(_PATHS[key]) if key in _PATHS else default
    if must_exist and key not in _PATHS and not p.exists():
        raise FileNotFoundError(
            f"paths.txt 未宣告 {key}，且推導路徑不存在: {p}\n請在 {_PATHS_FILE} 加一行 {key}=<路徑>"
        )
    return p


# 受試者主表（去識別）住 D:\Alz\common\demographics，所有 repo 共用
DEMOGRAPHICS_DIR = _path("DEMOGRAPHICS", _ALZ_ROOT / "common" / "demographics")
# 舊名相容：DATA_DIR 曾指 repo 內 data\（已解散）。保留名稱給 src/__init__ 的匯出，勿再使用。
DATA_DIR = DEMOGRAPHICS_DIR.parent
# 單一乾淨人口學表（P/NAD/ACS 合併；已產出之資料檔）。
# 欄位：Group, ID(受試者數字), Photo_Session, Photo_Date, Birth_Date, Sex,
#       Age, BMI, NPT_Date, NPT_Session, Diff_Days, MMSE, CASI, Global_CDR
HOSPITAL_A_CSV = DEMOGRAPHICS_DIR / "hospital_A.csv"

# 分析世代凍結：只納入 Photo_Date < PHOTO_DATE_MAX 的 session。
# hospital_A.csv 於 2026-08-14 更新（P521 sexfix / dedup / 補 21 筆 Photo_Date>=2026-07 的新收案）；
# 那 21 筆雖無臉部特徵（不在 embedding/meta 的 2070 母體），卻會餵進 match_by_age 的最佳年齡配對、
# 擾動 1:1 配對結果，使 1:1 指標與先前落地（paper/PPT，皆 pre-July）對不上。設此上限把配對世代
# 凍在 2026-07 前 → 現行程式即可重現既有結果，且未來 CSV 再增資料也不影響。設 None 解除凍結。
PHOTO_DATE_MAX = "2026-07-01"

# 原始影像目錄（母帶，paths.txt 的 RAW 鍵，延遲讀取）。
# 不在 import 時強制 RAW 存在 —— 純推論 (age/embedding/meta，例如 alz_infer
# 服務) 不碰原始影像,故無需 RAW 也能 import src.config / src.age。
# 真正取用 RAW_IMAGES_DIR（或 `from src.config import RAW_IMAGES_DIR`）時,
# 才透過下方 __getattr__ 取值並在缺鍵時報錯。


def _raw_images_dir() -> Path:
    if "RAW" not in _PATHS:
        raise FileNotFoundError(
            f"paths.txt 未宣告 RAW（原始影像目錄）: {_PATHS_FILE}\n"
            f"請加一行 RAW=<原始影像目錄路徑>"
        )
    return Path(_PATHS["RAW"])


def __getattr__(name: str):
    # PEP 562:延遲提供 RAW_IMAGES_DIR,讓不需要原始影像的 import 不被 path.txt 卡住。
    if name == "RAW_IMAGES_DIR":
        return _raw_images_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# 外部依賴目錄
EXTERNAL_DIR = PROJECT_ROOT / "external"

# 工作區根（子主題層 face\workspace）
WORKSPACE_DIR = _path("WORKSPACE", _SUBTHEME_ROOT / "workspace")

# 文獻：人細篩通過的文獻住 face\paper\refs\<題目>\；AI 下載與初篩的佇列住 face\workspace\literature\
REFERENCES_DIR = _path("REFS", _SUBTHEME_ROOT / "paper" / "refs")
LIT_QUEUE_DIR = _path("LIT_QUEUE", WORKSPACE_DIR / "literature")

# 收案 App 的主表（identity，含個資；只有 scripts/export_hospital_a.py 讀）
INTAKE_MASTER_CSV = _path("INTAKE_MASTER", _ALZ_ROOT / "收案" / "data" / "k" / "outcome_k.csv", must_exist=False)

# -----------------------------------------------------------------------------
# 選幀準則（selection）
#
# 每個準則各自擁有一整棵完整的下游樹：
#     workspace/<subsystem>/<selection>/…
#     workspace/preprocess/…（唯一例外，見下方 Preprocess 段）
#
# 準則的差異只在「10 張怎麼挑」，landmark / 去背 / 轉正 / 鏡射三棵樹都一樣用
# MediaPipe，所以名稱指的是選幀方法而非函式庫。
#
#   mediapipe        現有結果：MediaPipe 中線頂角和（VAS）升冪取前 10
#   openface_p10r5   OpenFace 2.2 頭部姿態，|pitch|<=10, |roll|<=5（PDF 原始規格）
#   openface_p15r10  同上但 |pitch|<=15, |roll|<=10（產出率較高、組間差距較小）
#
# 由環境變數 ALZ_SELECTION 在 process 啟動時決定。**不要**做 runtime setter：
# 全 repo 46 個 `from src.config import X` 都是 module 頂層裸名綁定，import 當下
# 就凍結，runtime 改動只會在少數幾處生效，產生混合 selection 的輸出而不報錯。
# 要同時比較多個準則請一個準則開一個 subprocess。
# -----------------------------------------------------------------------------
DEFAULT_SELECTION = "mediapipe"
SELECTION = os.environ.get("ALZ_SELECTION", DEFAULT_SELECTION)


def subsystem_dir(name: str, selection: Optional[str] = None) -> Path:
    """workspace 子系統根 = WORKSPACE_DIR / <name> / <selection>。

    selection 留空用當前 process 的 SELECTION；傳值則用於跨準則比較
    （唯一支援的跨樹存取方式）。
    """
    return WORKSPACE_DIR / name / (selection or SELECTION)


# -----------------------------------------------------------------------------
# Preprocess
# -----------------------------------------------------------------------------
# preprocess 是唯一「selection 不在子系統根」的子系統。版面照管線的分岔點排：
#
#     讀原圖 → FaceMesh → 選 10 張 ──┬─→ apply_mask → 轉正 → no_background/…/aligned
#                           ↑        └─→ （不遮罩） → 轉正 → background/…/aligned
#                        selected
#
#     workspace/preprocess/selected/selector_<selection>/          分岔前，兩變體共用
#     workspace/preprocess/{no_background|background}/selector_<selection>/{aligned,mirrors}/
#
# bg/no_bg 是影像處理的分支，selector 才是實驗變因，所以 selector 掛在變體之下，
# 同變體的各準則並排可直接比對。其餘子系統一律 subsystem_dir()（selection 在最外層）。
PREPROCESSING_DIR = WORKSPACE_DIR / "preprocess"
_PREPROCESS_STAGES = ("selected", "aligned", "mirrors")
_VARIANT_STAGES = ("aligned", "mirrors")


def preprocess_selector_dir(background: bool = False,
                            selection: Optional[str] = None) -> Path:
    """某個去背變體下、某個選幀準則的根目錄（aligned / mirrors 的上一層）。"""
    variant = "background" if background else "no_background"
    return PREPROCESSING_DIR / variant / f"selector_{selection or SELECTION}"


def preprocess_selected_dir(selection: Optional[str] = None) -> Path:
    """選出來的原始幀（未遮罩、未轉正）。在去背分岔之前，故不掛在任一變體之下。"""
    return PREPROCESSING_DIR / "selected" / f"selector_{selection or SELECTION}"


def preprocess_dir(stage: str, background: bool = False,
                   selection: Optional[str] = None) -> Path:
    """預處理輸出目錄。

    stage      ∈ {selected, aligned, mirrors}
    background  False（預設）→ no_background（去背版）；True → background（保留背景版）
                stage="selected" 時**忽略**此參數：選幀在去背分岔之前，兩個變體
                拿到的是同一批影像，只有一份。
    selection   留空用當前 process 的 SELECTION；傳值則用於跨準則比較

    取代舊的扁平常數（ALIGNED_DIR / ALIGNED_BACKGROUND_DIR / MIRRORS_DIR …），
    bg/no_bg 由參數決定，不再每個葉子各開一個常數。
    """
    if stage not in _PREPROCESS_STAGES:
        raise ValueError(
            f"stage must be one of {_PREPROCESS_STAGES}, got {stage!r}")
    if stage == "selected":
        return preprocess_selected_dir(selection)
    return preprocess_selector_dir(background, selection) / stage

# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------
EMBEDDING_DIR = subsystem_dir("embedding")
EMBEDDING_FEATURES_DIR = EMBEDDING_DIR / "features"
EMBEDDING_ANALYSIS_DIR = EMBEDDING_DIR / "analysis"
EMBEDDING_FEATURE_STAT_DIR = EMBEDDING_ANALYSIS_DIR / "feature_stat"
EMBEDDING_CLASSIFICATION_DIR = EMBEDDING_ANALYSIS_DIR / "classification"

# 2026-06: the refactor sandbox (workspace_refactor/) was folded back into workspace/
# after A/B validation. This name is kept as an alias of the canonical workspace
# classification dir so the embedding downstream scripts' imports stay stable.
EMBEDDING_CLASSIFICATION_REFACTOR_DIR = EMBEDDING_CLASSIFICATION_DIR

# -----------------------------------------------------------------------------
# Age
# -----------------------------------------------------------------------------
AGE_DIR = subsystem_dir("age")
AGE_PREDICTIONS_DIR = AGE_DIR / "predictions"
AGE_BENCHMARK_DIR = AGE_PREDICTIONS_DIR
AGE_ANALYSIS_DIR = AGE_DIR / "analysis"

# 預設指向 DEFAULT_COHORT_TOKENS（analysis 下）。
_AGE_DEFAULT_ANALYSIS = AGE_ANALYSIS_DIR / "P_first_HC_first" / "P_cdr05_HC_cdrall_mmseall"

PREDICTED_AGES_FILE = AGE_PREDICTIONS_DIR / "1_MiVOLO" / "predicted_ages.json"

# 視覺化子樹（直接在 cohort 下）
AGE_SCATTER_DIR = _AGE_DEFAULT_ANALYSIS / "scatter"
AGE_STAT_DIR = _AGE_DEFAULT_ANALYSIS / "stat"
AGE_LINES_DIR = _AGE_DEFAULT_ANALYSIS / "lines"
AGE_HISTOGRAM_DIR = _AGE_DEFAULT_ANALYSIS / "histogram"
AGE_VIOLIN_DIR = _AGE_DEFAULT_ANALYSIS / "violin"

# -----------------------------------------------------------------------------
# BMI
# -----------------------------------------------------------------------------
BMI_DIR = subsystem_dir("bmi")
BMI_MODELS_DIR = BMI_DIR / "models"
BMI_PREDICTIONS_DIR = BMI_DIR / "predictions"
BMI_ANALYSIS_DIR = BMI_DIR / "analysis"

# -----------------------------------------------------------------------------
# Emo_au
# -----------------------------------------------------------------------------
EMO_AU_DIR = subsystem_dir("emo_au")
EMO_AU_FEATURES_DIR = EMO_AU_DIR / "features"
EMO_AU_FEATURES_SCHEMA_FILE = EMO_AU_FEATURES_DIR / "_schema.json"

EMO_AU_ANALYSIS_DIR = EMO_AU_DIR / "analysis"
EMO_AU_FEATURE_STAT_DIR = EMO_AU_ANALYSIS_DIR / "feature_stat"
EMO_AU_CLASSIFICATION_DIR = EMO_AU_ANALYSIS_DIR / "classification"

# -----------------------------------------------------------------------------
# Asymmetry (landmark)
# -----------------------------------------------------------------------------
ASYMMETRY_DIR = subsystem_dir("asymmetry")
ASYMMETRY_FEATURES_DIR = ASYMMETRY_DIR / "features"
ASYMMETRY_LANDMARKS_DIR = ASYMMETRY_FEATURES_DIR / "landmarks"
ASYMMETRY_PAIR_FEATURES_FILE = ASYMMETRY_FEATURES_DIR / "pair_features.csv"
ASYMMETRY_ANALYSIS_DIR = ASYMMETRY_DIR / "analysis"
ASYMMETRY_FEATURE_STAT_DIR = ASYMMETRY_ANALYSIS_DIR / "feature_stat"
ASYMMETRY_CLASSIFICATION_DIR = ASYMMETRY_ANALYSIS_DIR / "classification"

# -----------------------------------------------------------------------------
# Rotation (head pose) — 每法（PnP / vector_angle）各一子目錄，由 producer 串接
# -----------------------------------------------------------------------------
ROTATION_DIR = WORKSPACE_DIR / "rotation"
ROTATION_FIG_DIR = ROTATION_DIR / "fig"
ROTATION_FEATURES_DIR = ROTATION_DIR / "features"

# -----------------------------------------------------------------------------
# Pose (OpenFace 2.2 逐幀頭部姿態) — 選幀的「輸入」，所有 selection 共用，
# 因此不掛 selection 軸。一次算好，換閘門只要重挑不必重算。
# -----------------------------------------------------------------------------
POSE_DIR = WORKSPACE_DIR / "pose"
OPENFACE_POSE_DIR = POSE_DIR / "openface"

# FeatureExtraction.exe 所在目錄（解壓 OpenFace_2.2.0_win_x64.zip 後的根）
OPENFACE2_DIR = EXTERNAL_DIR / "pose" / "OpenFace_2.2.0_win_x64"
OPENFACE2_BIN = OPENFACE2_DIR / "FeatureExtraction.exe"

# D435i 彩色串流內參。
#
# 來源：拍攝端明確設定 enable_stream(color, 1280, 720, bgr8, 30)；D435i 彩色
# HFOV 69.4 度、方形像素 → fx = fy = 640 / tan(34.7°) = 924。以 VFOV 反推
# 2·atan(360/924) = 42.6 度，對上 datasheet 的 42.5 度。
#
# 存檔前經過 cv2.transpose（轉置，行列式 -1，是鏡射不是旋轉），影像變成
# 720x1280 直式，主點跟著交換 → cx=360, cy=640。fx=fy 所以焦距不受影響。
#
# 注意：OpenFace 的四個內參是成對分支的，只傳 -fx 不傳 -fy 會讓 fy 變成 -1。
# 四個要嘛全傳、要嘛全不傳。另外 OpenFace 對 solvePnP 傳入空的畸變矩陣且
# 沒有對應旗標，D435i 的畸變係數無法納入。
D435I_COLOR_INTRINSICS = {"fx": 924.0, "fy": 924.0, "cx": 360.0, "cy": 640.0}

# -----------------------------------------------------------------------------
# Overview — 跨 modality cohort metadata + matching artifacts + per-design summaries
# + cross-modality stat grid（per-cohort × per-hc_source）
# -----------------------------------------------------------------------------
OVERVIEW_DIR = WORKSPACE_DIR / "overview"

# -----------------------------------------------------------------------------
# 6Q-DS(六題失智症篩檢量表)—— 純問卷 modality,不碰影像,故不掛 selection 軸。
#
# 原始 xlsx 三份(dementia / very mild dementia / 20220818 加了 CASI+MMSE),彼此
# 高度重疊但各自成表,因此 workspace 依 dataset id 分樹,每份資料各有自己的
# dataset / cv / model / figures。dataset id 見 src/q6ds/dataset.py:DATASETS。
# -----------------------------------------------------------------------------
Q6DS_RAW_DIR = _path("Q6DS_RAW", _ALZ_ROOT / "q6ds" / "data")
Q6DS_DIR = _path("Q6DS_WORKSPACE", _ALZ_ROOT / "q6ds" / "workspace")
Q6DS_LOG_DIR = Q6DS_DIR / "logs"
Q6DS_SUMMARY_FILE = Q6DS_DIR / "all_metrics.csv"   # 三份資料 × 五個 arm 的總表


def q6ds_dataset_dir(dataset_id: str) -> Path:
    """workspace/q6ds/<dataset_id>/ —— 單一份 xlsx 的全部產出根。"""
    return Q6DS_DIR / dataset_id


def q6ds_path(dataset_id: str, kind: str, arm: Optional[str] = None) -> Path:
    """workspace/q6ds/<dataset_id>/<kind>/[<arm>/]

    kind ∈ {dataset, cv, model, figures};arm ∈ src/q6ds/model.py:ARMS
    (dataset 是所有 arm 共用的建模表,不吃 arm)。
    """
    p = q6ds_dataset_dir(dataset_id) / kind
    return p if arm is None else p / arm


# -----------------------------------------------------------------------------
# Cohort tokens (4-axis — same signature as src.common.cohort.cohort_list)
#
#   p_visit  ∈ {p_first, p_all}
#   p_score  ∈ {p_cdrall, p_cdr05, p_cdr1, p_cdr2}     (Global_CDR >= 0 / .5 / 1 / 2)
#   hc_visit ∈ {hc_first, hc_all}
#   hc_score ∈ {hc_cdrall_or_mmseall, hc_cdr0_or_mmse26}
#
# 一個 cohort = 這 4 個 token 的 tuple,順序同 cohort_list,程式內以 ``*cohort`` 流通。
# 取代舊的 cohort_mode 字串 / 5-axis CohortSpec —— 後者 p_cdr 只有 cdr05/cdrall,表達
# 不出 p_cdr1 / p_cdr2;4-token 是其嚴格超集。輸出路徑沿用舊命名,逐字相容。
# -----------------------------------------------------------------------------

P_VISIT_TOKENS = ("p_first", "p_all")
P_SCORE_TOKENS = ("p_cdrall", "p_cdr05", "p_cdr1", "p_cdr2")
HC_VISIT_TOKENS = ("hc_first", "hc_all")
HC_SCORE_TOKENS = ("hc_cdrall_or_mmseall", "hc_cdr0_or_mmse26")

DEFAULT_COHORT_TOKENS = ("p_first", "p_cdr05", "hc_first", "hc_cdrall_or_mmseall")

# hc_score token → 路徑片段(對齊 legacy cdr_mmse_dir 的 HC 部分:cdr0_mmse26 / cdrall_mmseall)
_HC_SCORE_DIR = {
    "hc_cdrall_or_mmseall": "cdrall_mmseall",
    "hc_cdr0_or_mmse26": "cdr0_mmse26",
}


def validate_cohort_tokens(p_visit, p_score, hc_visit, hc_score) -> None:
    """4 token 各自落在合法字彙;否則 raise。"""
    for tok, vocab in ((p_visit, P_VISIT_TOKENS), (p_score, P_SCORE_TOKENS),
                       (hc_visit, HC_VISIT_TOKENS), (hc_score, HC_SCORE_TOKENS)):
        if tok not in vocab:
            raise ValueError(f"invalid cohort token {tok!r}; expected one of {vocab}")


def cohort_dirs(p_visit, p_score, hc_visit, hc_score) -> Tuple[str, str]:
    """4 token → (visit_dir, cdr_mmse_dir)。逐字相容舊 CohortSpec 命名,並支援新的
    p_cdr1 / p_cdr2(舊 5-axis 表達不出)。"""
    validate_cohort_tokens(p_visit, p_score, hc_visit, hc_score)
    visit_dir = f"P_{p_visit.split('_', 1)[1]}_HC_{hc_visit.split('_', 1)[1]}"
    cdr_mmse_dir = f"P_{p_score.split('_', 1)[1]}_HC_{_HC_SCORE_DIR[hc_score]}"
    return visit_dir, cdr_mmse_dir


def cohort_path(p_visit, p_score, hc_visit, hc_score) -> Path:
    """Two-level cohort 目錄:<visit_dir>/<cdr_mmse_dir>。"""
    visit_dir, cdr_mmse_dir = cohort_dirs(p_visit, p_score, hc_visit, hc_score)
    return Path(visit_dir) / cdr_mmse_dir


# Embedding 正規化軸:載入 embedding 後、算 variant 前，先把每個 z 除以自己的範數。
# PDF §4 要求「ArcFace 若未 L2 正規化則須先正規化」(實測 ‖z‖₂ ≈ 23.7，即未正規化)。
# 路徑落在 emb 與 variant 之間，因 pipeline 順序為「載入 → 正規化 → 算不對稱」。
NO_NORMALIZE = "no_normalize"
NORMALIZE_MODES = (NO_NORMALIZE, "l1_normalize", "l2_normalize")
NORMALIZE_ORD = {"l1_normalize": 1, "l2_normalize": 2}   # np.linalg.norm 的 ord


def embedding_classification_path(
    p_visit: str,
    p_score: str,
    hc_visit: str,
    hc_score: str,
    bg_mode: str,
    emb: str,
    variant: str,
    photo_mode: str = "mean",
    reducer: str = "no_drop",
    normalize: str = "no_normalize",
    clf: Optional[str] = None,
    clf_param: Optional[str] = None,
    direction: Optional[str] = None,
    eval_method: Optional[str] = None,
    match_level: Optional[str] = None,
    eval_unit: Optional[str] = None,
    match_strategy: Optional[str] = None,
    partition: Optional[str] = None,
    seed: Optional[int] = 0,
    fold_kind: str = "group",
    root: Optional[Path] = None,
) -> Path:
    """
    Compose embedding classification output path.

    Layout (follows 10-variable pipeline order):
      classification/<visit>/<cdr_mmse>/<bg_mode>/<emb>/<normalize>/<variant>/<photo>/<reducer>/
        <clf>/seed_<seed>/<clf_param>/<direction>/<eval_method>/<match_level>/<eval_unit>/<match_strategy>/<partition>/

    normalize 在 emb 與 variant 之間,因為 pipeline 的順序是「載入 embedding → 正規化 →
    算不對稱向量」;正規化改變的是特徵本身,故屬 variant 之前的決策(見 §PDF 4)。

    Args:
        p_visit, p_score, hc_visit, hc_score: cohort 4-token(見上方 cohort token 區塊)
        bg_mode: background | no_background
        emb: arcface | topofr | dlib | vggface
        normalize: no_normalize | l1_normalize | l2_normalize
                   (embedding 先各自除以 ‖z‖₁ / ‖z‖₂ 再算 variant;no_normalize = 用原始 z)
        variant: original | differences | absolute_differences |
                 relative_differences | absolute_relative_differences
        photo_mode: mean | all
        reducer: no_drop | pca/n_components_X | drop_feats/pearson_r_X.X
        clf, clf_param, direction, eval_method, match_level, eval_unit,
        match_strategy, partition: 可選。clf_param = classifier 的 hyperparameter 子層
            (grid search 用,如 C_1.0 / ne_300_md_6_lr_0.1);scorer / 非 grid 時為 None。
    """
    visit_dir, cdr_mmse_dir = cohort_dirs(p_visit, p_score, hc_visit, hc_score)
    base = root if root is not None else EMBEDDING_CLASSIFICATION_DIR
    p = (base / visit_dir / cdr_mmse_dir
         / bg_mode / emb / normalize / variant / photo_mode / reducer)
    # clf 之後接一個可選的 hyperparameter 子層(grid search 用,如 logistic/C_1.0、
    # xgb/ne_300_md_6_lr_0.1),再接 direction 起的評估鏈。clf_param 只在 clf 存在時
    # 插入;scorer / 非 grid 時為 None → 退回 clf/direction。
    segs = []
    if clf is not None:
        segs.append(clf)
        if seed is not None:                       # seed_<N> 緊接 classifier(repeated-CV 的折分維度)
            segs.append(f"seed_{seed}")
        # 折分方式(見 src/common/folds.py)。預設 group = GroupKFold,不插段,既有樹不動;
        # 只有非預設(如 stratified_group)才多一層 folds_<kind>,避免兩種折分互相覆蓋。
        if fold_kind and fold_kind != "group":
            segs.append(f"folds_{fold_kind}")
        if clf_param is not None:
            segs.append(clf_param)
        segs += [direction, eval_method, match_level, eval_unit,
                 match_strategy, partition]
    for seg in segs:
        if seg is None:
            break
        p = p / seg
    return p


META_DIR = subsystem_dir("meta")
META_ANALYSIS_DIR = META_DIR / "analysis"

# 部署匯出（fold 模型 + 重現稽核）
DEPLOY_DIR = subsystem_dir("deploy")


def meta_analysis_path(
    p_visit: str,
    p_score: str,
    hc_visit: str,
    hc_score: str,
    bg_mode: str,
    emb_model: str,
    photo_mode: str = "mean",
    reducer: str = "no_drop",
    *,
    case_mode: Optional[str] = None,
    feature_set: Optional[str] = None,
    variant: Optional[str] = None,
    base_classifier: Optional[str] = None,
    base_classifier_param: Optional[str] = None,
    meta_classifier: Optional[str] = None,
    seed: Optional[int] = 0,
    fold_kind: str = "group",
) -> Path:
    """Compose a meta-analysis cell path (single unified session-level pipeline).

    Layout (shares embedding's cohort/bg/emb/photo/reducer prefix, then meta-specific
    axes; **skip-None**——省略不適用的中段,故認知 combo 與影像 combo 共用同一規則):
      meta/analysis/<visit>/<cdr_mmse>/[<case_mode>/]<bg_mode>/<emb>/<photo>/<reducer>/
        <feature_set>/[<variant>/<base_clf>/<base_clf_param>/]<meta_clf>/

    case_mode = meta 母體(no_nan=丟認知缺值/complete-case;keep_nan=保留/full cohort);
    緊接 cohort 之後,讓兩種母體各自完全自足(各有 cells / all_metrics / _summary)。
    路徑只編入會改變數值的軸:含影像 OOF 的 combo 帶 variant(asym variant)+ base_clf
    + base_clf_param(共用的 logistic C);純認知 combo 三者皆 None → 只剩 feature_set/meta_clf。
    """
    p = META_ANALYSIS_DIR / cohort_path(p_visit, p_score, hc_visit, hc_score)
    if case_mode is not None:                 # no_nan / keep_nan;區隔 meta 母體,各自一棵子樹
        p = p / case_mode
    p = p / bg_mode / emb_model / photo_mode / reducer
    seed_seg = f"seed_{seed}" if seed is not None else None  # 影像在 base_clf 後、認知在 feature_set 後(skip-None)
    # 折分方式與 embedding 端同一套慣例(src/common/folds.py):group 不插段(既有樹不動),
    # 非預設才多一層 folds_<kind>,位置也一樣緊接 seed_ 之後,兩種折分不互相覆蓋。
    fold_seg = f"folds_{fold_kind}" if fold_kind and fold_kind != "group" else None
    for seg in (feature_set, variant, base_classifier, seed_seg, fold_seg,
                base_classifier_param, meta_classifier):
        if seg is not None:
            p = p / seg
    return p


def get_raw_images_subdir(group: str) -> Path:
    """
    取得原始影像子目錄

    Args:
        group: "ACS", "NAD", "P" 或 "health/ACS", "health/NAD", "patient"

    Returns:
        完整路徑
    """
    group_mapping = {
        "ACS": "health/ACS",
        "NAD": "health/NAD",
        "P": "patient",
    }
    subdir = group_mapping.get(group, group)
    return _raw_images_dir() / subdir


# =============================================================================
# 處理參數 Dataclass
# =============================================================================

@dataclass
class MirrorConfig:
    """鏡射生成配置"""

    mirror_method: str = "flip"  # "midline" (沿臉部中線) 或 "flip" (水平翻轉)
    mirror_size: Tuple[int, int] = (512, 512)  # 輸出鏡射影像大小
    feather_px: int = 2  # 邊緣羽化像素
    margin: float = 0.08  # 畫布邊緣留白比例
    midline_points: Tuple[int, ...] = MIDLINE_POINTS  # 臉部中軸線特徵點索引


@dataclass
class PreprocessConfig:
    """預處理各站參數（detect / select / align / mirror 共用）。

    去背/鏡射「要不要做」由 run_preprocess.py 的 toggle 控制，
    不再放在 config（昔日的 steps / also_save_aligned_background 已移除）。
    """

    midline_points: Tuple[int, ...] = MIDLINE_POINTS

    # 相片選擇 / 偵測
    n_select: int = 10  # 選擇多少張最正的臉部相片
    detection_confidence: float = 0.5  # MediaPipe 偵測信心度閾值

    # 鏡射參數
    mirror: MirrorConfig = field(default_factory=MirrorConfig)
