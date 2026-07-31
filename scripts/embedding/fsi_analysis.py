"""FSI（臉部不對稱指數）與年齡分析 —— 依 paper/20260714-FSI.pdf 計畫落地。

FSI = 左右對稱臉 ArcFace embedding 的逐元素絕對差「平均」（= absolute_differences 向量對維度
取平均 = l1_norm ÷ d）。每 visit 的 FSI 由 10 張影像的 |L−R| 先對維度、再對影像平均而得——因
平均可交換，等同「逐張算 FSI 再平均」（PDF §6-7）。每位參與者只取第一次符合條件的 visit
（cohort p_first/hc_first；PDF §8），故每人一筆獨立觀測，用普通 OLS + HC3 穩健標準誤，不需
混合效應（PDF §16）。

組別對應：AD=P、SCD=NAD、ACS=ACS；年齡置中於 75（截距/組差即「75 歲時」之值）。
三個模型（PDF §17-19，皆 HC3）：
  §17 非失智共同斜率   FSI ~ age_c + Male + ACS               （SCD+ACS，主檢定 age_c）
  §18 SCD vs ACS 交互  FSI ~ age_c + Male + ACS + age_c:ACS   （β4 = ACS−SCD 斜率差）
  §19 全世代合併       FSI ~ age_c + AD + ACS + Male + age_c:AD + age_c:ACS

每個模型另做兩版反應變數：log FSI = ln(FSI)（FSI 恆正，ln 無定義域問題；斜率可讀成「每年
FSI 的相對變化率」，小係數時 exp(β)−1 ≈ β），以及 FSI_L1 = 未除以 d 的 L1 範數（純常數縮放，
β/CI 同乘 512、p/R² 完全不變，僅供對照單位）。§19 因三組迴歸線在 63-65 歲附近交會，置中點
會改變「@參考年齡」組差 β2/β3 的大小與顯著性，故 log 版同時輸出置中 75（主要）與置中 65
兩版；斜率/交互係數不受置中影響。

另有 FSI_rel（向量層級歸一化，見 _fsi_rel）僅出敘述統計表、不進迴歸：其分母（embedding 自身
L1 長度）與年齡/性別/診斷顯著共變（Male p≈4e-32、Age p≈7e-14），除以它會把分母的結構反向
注入指標，非中性歸一化。

輸出分兩棵平行的樹（<feature_stat>/<cohort>/<bg>/<model>/fsi/<tree>/）：
  no_normalize/   現行做法：直接用原始 ArcFace embedding 算 FSI
  normalize/      PDF §4：每個 embedding 先 L2 正規化成單位向量再算 FSI
兩棵樹結構相同：
  fsi_dataset.csv        參與者級資料（PDF §10；含 FSI / FSI_L1 / log_FSI / FSI_rel(_L2)）
  fsi_report.xlsx        建議報告表（PDF §29；原始 FSI 尺度）
  fsi_models.txt         三模型 statsmodels summary（含 §18 交互項）
  fsi_rel_stat.xlsx      FSI_rel（L1）敘述統計（整體 + 三族群 n / mean ± SD，不做迴歸）
  fsi_rel_l2_stat.xlsx   FSI_rel_L2（L2）敘述統計，同上
  s17_nondemented_common_slope/{s17,s17_log,s17_l1}{.png,_coef.csv}   §17 圖 + 係數表
  s18_nondemented_age_x_acs/{s18,s18_log,s18_l1}{.png,_coef.csv}      §18 圖 + 係數表
  s19_complete_cohort_combined/{s19,s19_log,s19_log_center65,s19_l1}{.png,_coef.csv}
每張圖：依組著色散點 + 擬合線（§17 平行 / §18-19 各自斜率）+ 左上係數註記（女性 Male=0 參考）

注意：normalize/ 只改 FSI 分析；PDF §2 另要求分類向量必須用同一套正規化，而分類 pipeline
仍建在未正規化的 embedding 上，故 normalize/ 目前僅供對照，尚未達成 §2 的一致性要求。

Usage:
    python scripts/embedding/fsi_analysis.py [--model arcface] [--bg-mode background]
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from openpyxl import Workbook
from openpyxl.styles import Font

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401

from src.config import (
    EMBEDDING_FEATURE_STAT_DIR, cohort_path,
    P_VISIT_TOKENS, P_SCORE_TOKENS, HC_VISIT_TOKENS, HC_SCORE_TOKENS,
)
from src.common.cohort import cohort_list, load_demographics, base_id_of, group_of
from src.common.features import load_feature_matrix
# 復用參考表樣式常數（Times New Roman、thin 框線、置中），與 asymmetry_stat/_ancova 一致
from asymmetry_stat import (
    _BORDER, _CENTER, _F_HEADER, _F_BIG, _F_DATA, _ROW_H_HEADER, _ROW_H_DATA,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 對應 PDF 的 first-visit、不做 CDR/MMSE 篩選（要全體 AD/SCD/ACS）= §9 的 1,610 人
DEFAULT_TOKENS = ("p_first", "p_cdrall", "hc_first", "hc_cdrall_or_mmseall")
AGE_CENTER = 75.0                                    # PDF：年齡置中於 75
GROUP_COLOR = {"AD": "#C44E52", "SCD": "#55A868", "ACS": "#4C72B0"}
MODEL_DISPLAY = {"arcface": "ArcFace", "topofr": "TopoFR", "dlib": "dlib", "vggface": "VGGFace"}

# ── 參與者級資料集（PDF §10）────────────────────────────────────────────────

# FSI_rel 的兩種範數版本：欄名 → (np.linalg.norm 的 ord, 顯示下標)
REL_NORMS = {"FSI_rel": (1, "₁"), "FSI_rel_L2": (2, "₂")}


def _fsi_rel(ids, model, bg_mode) -> pd.DataFrame:
    """向量層級的無量綱不對稱指數，每人各一個純量（L1 與 L2 兩種範數）：

        FSI_rel    = ‖XL−XR‖₁ / ((‖XL‖₁ + ‖XR‖₁) / 2)
        FSI_rel_L2 = ‖XL−XR‖₂ / ((‖XL‖₂ + ‖XR‖₂) / 2)

    先算每張照片的比值、再對該 visit 的 10 張取平均（與 FSI「每張算完再平均」一致；
    比值非線性，故平均順序不可交換，這裡固定取「先比值後平均」）。

    分母用左右 embedding 各自的範數長度，屬向量層級歸一化——不是逐維比值。逐維版
    （FA2 / sMAPE 核心）在有號的 ArcFace 維度上，遇左右異號時 |l−r| ≡ |l|+|r|，比值
    恆飽和於上界 2 而不帶強度資訊（實測 8% 的維度如此，佔指標量的 32%），故不採用。
    """
    Xl, id_l = load_feature_matrix(ids, model, "face_left", bg_mode, "all")
    Xr, id_r = load_feature_matrix(ids, model, "face_right", bg_mode, "all")
    if not np.array_equal(id_l, id_r):
        raise RuntimeError("face_left / face_right 的 row_ids 不一致，無法逐張配對")

    cols = {"ID": id_l}
    for col, (ordr, _) in REL_NORMS.items():
        num = np.linalg.norm(Xl - Xr, ord=ordr, axis=1)
        den = (np.linalg.norm(Xl, ord=ordr, axis=1)
               + np.linalg.norm(Xr, ord=ordr, axis=1)) / 2
        cols[col] = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-8)
    return (pd.DataFrame(cols)
            .groupby("ID", as_index=False)[list(REL_NORMS)].mean())



def _fsi_l2_normalized(ids, model, bg_mode) -> pd.DataFrame:
    """PDF §4 版 FSI：每個 embedding 先 L2 正規化成單位向量，再算不對稱。

        ẑ = z / ‖z‖₂ ,  FSI_m = (1/d)·Σₖ |ẑ_R,m,k − ẑ_L,m,k|

    ArcFace 回傳的是未正規化 embedding（實測 ‖z‖₂ ≈ 23.7），故 PDF §4 要求先做這一步；
    §2 另規定分類向量必須用同一套正規化（本函式只影響 FSI 分析，分類 pipeline 未同步）。
    每張照片各自算完 FSI 再對該 visit 的照片平均（§7 規定的順序）。
    """
    Xl, id_l = load_feature_matrix(ids, model, "face_left", bg_mode, "all")
    Xr, id_r = load_feature_matrix(ids, model, "face_right", bg_mode, "all")
    if not np.array_equal(id_l, id_r):
        raise RuntimeError("face_left / face_right 的 row_ids 不一致，無法逐張配對")
    ln = Xl / np.linalg.norm(Xl, axis=1, keepdims=True)
    rn = Xr / np.linalg.norm(Xr, axis=1, keepdims=True)
    a = np.abs(ln - rn)
    return (pd.DataFrame({"ID": id_l, "FSI": a.mean(axis=1), "FSI_L1": a.sum(axis=1)})
            .groupby("ID", as_index=False)[["FSI", "FSI_L1"]].mean())


def build_participant_dataset(cohort, model, bg_mode, normalize=False) -> pd.DataFrame:
    """每位參與者第一次 visit 的 FSI + 共變量。

    復用 cohort_list（first-visit 選取）、load_feature_matrix（|L−R| 向量）、load_demographics
    （性別）。normalize=True 則依 PDF §4 先把每個 embedding L2 正規化再算 FSI。
    回傳欄位：ID, base_id, Dx(AD/SCD/ACS), Age, age_c, Sex, Male, AD, ACS, FSI, FSI_L1, log_FSI,
    FSI_rel, FSI_rel_L2。
    """
    df = cohort_list(*cohort)                                    # 每 base_id 一列（最早 visit）
    ids = df["ID"].tolist()

    if normalize:
        fsi = _fsi_l2_normalized(ids, model, bg_mode)
    else:
        # FSI：absolute_differences（|L−R|）以 photo_mode=mean 對 10 張取平均 → 再對維度取平均（÷d）
        X, row_ids = load_feature_matrix(ids, model, "absolute_differences", bg_mode, "mean")
        if len(X) == 0:
            raise RuntimeError(f"無任何 {model}/{bg_mode} absolute_differences 特徵可載入")
        # FSI_L1 = 未除以 d 的 L1 範數（= FSI × d）。純常數縮放：β/CI 同乘 d，p/t/R² 完全不變。
        fsi = pd.DataFrame({"ID": row_ids, "FSI": X.mean(axis=1), "FSI_L1": X.sum(axis=1)})
    # d 由 FSI_L1/FSI 反推（兩路徑皆成立），避免依賴 normalize=False 分支才有的 X
    d = int(round(fsi["FSI_L1"].iloc[0] / fsi["FSI"].iloc[0]))
    logger.info(f"embedding dim d = {d}；normalize = {normalize}；"
                f"FSI 覆蓋 {len(fsi)}/{len(ids)}")

    sex = load_demographics()[["ID", "Sex"]].drop_duplicates("ID")
    dx_map = {"P": "AD", "NAD": "SCD", "ACS": "ACS"}
    out = (df[["ID", "Age"]]
           .merge(fsi, on="ID", how="inner")
           .merge(_fsi_rel(ids, model, bg_mode), on="ID", how="left")
           .merge(sex, on="ID", how="left"))
    out["base_id"] = out["ID"].map(base_id_of)
    out["Dx"] = out["base_id"].map(lambda b: dx_map[group_of(b)])
    out["age_c"] = out["Age"] - AGE_CENTER
    out["log_FSI"] = np.log(out["FSI"])          # FSI = 平均 |L−R| > 0，ln 恆有定義
    out["Male"] = (out["Sex"] == "M").astype(float)
    out["AD"] = (out["Dx"] == "AD").astype(float)
    out["ACS"] = (out["Dx"] == "ACS").astype(float)

    miss = out[out["Sex"].isna() | out["Age"].isna()]
    if len(miss):                                               # PDF：不插補，complete-case
        logger.warning(f"丟棄 {len(miss)} 筆缺 Sex/Age（complete-case）")
        out = out.drop(miss.index)
    return out.reset_index(drop=True)

# ── 三個模型（PDF §17-19）──────────────────────────────────────────────────

# 每模型的年齡置中點。§18 置中於 65（ACS 偏年輕，把 ACS−SCD offset β3 落在資料範圍內、
# 而非外推到 75）。置中只影響截距與「@參考年齡」的 offset 係數，不動斜率/交互與其檢定。
MODEL_CENTER = {"s17": 65, "s18": 65, "s19": 75, "s19c65": 65}
REPORT_AGE = 75  # 報告表的「at age X」offset 一律在此年齡計算（PDF §29），與各圖置中脫鉤

_S19_RHS = "age_c + AD + ACS + Male + age_c:AD + age_c:ACS"


def fit_models(data: pd.DataFrame, y: str = "FSI") -> dict:
    """擬合 §17/§18/§19，皆 HC3、各依 MODEL_CENTER 置中。y 為反應變數欄名（FSI 或 log_FSI）。

    回 {'s17','s18','s19','s19c65'}；s19c65 與 s19 同式，僅置中 65（供 §19 置中對照圖）。
    """
    nd = data[data["Dx"] != "AD"]                              # 非失智 = SCD + ACS

    def _fit(key, rhs, d):
        d = d.copy()
        d["age_c"] = d["Age"] - MODEL_CENTER[key]
        return smf.ols(f"{y} ~ {rhs}", data=d).fit(cov_type="HC3")

    return {
        "s17": _fit("s17", "age_c + Male + ACS", nd),
        "s18": _fit("s18", "age_c + Male + ACS + age_c:ACS", nd),
        "s19": _fit("s19", _S19_RHS, data),
        "s19c65": _fit("s19c65", _S19_RHS, data),
    }


def _contrast(fit, weights: dict):
    """線性組合 Σ wᵢ·βᵢ 的 (估計, ci_low, ci_high, p)；weights 以 exog 名為鍵。"""
    names = fit.model.exog_names
    unknown = set(weights) - set(names)
    if unknown:
        raise KeyError(f"未知係數 {unknown}；可用: {names}")
    r = np.zeros(len(names))
    for k, w in weights.items():
        r[names.index(k)] = w
    tt = fit.t_test(r)
    lo, hi = tt.conf_int()[0]
    return float(tt.effect[0]), float(lo), float(hi), float(tt.pvalue)


def build_report_rows(fits: dict):
    """PDF §29 報告表列：(來源模型, 說明, 估計, 95%CI, p)。斜率一律換算 per 10 years。

    第 1 列的非失智共同斜率取自 §17；其餘各組斜率、斜率差、75 歲時組差皆取自 §19 合併模型，
    確保內部一致（同一擬合的線性組合，CI/p 由 t_test 導出）。
    """
    s17, s19 = fits["s17"], fits["s19"]
    D = 10.0  # per-decade 縮放
    sh = REPORT_AGE - MODEL_CENTER["s19"]  # 把「@75 的組差」由模型中心平移到 75（sh=0 時退化為純 dummy）
    specs = [
        ("§17", "Non-demented common age slope, per 10 years", s17, {"age_c": D}),
        ("§19", "SCD age slope, per 10 years",                 s19, {"age_c": D}),
        ("§19", "ACS age slope, per 10 years",                 s19, {"age_c": D, "age_c:ACS": D}),
        ("§19", "ACS−SCD slope difference, per 10 years",      s19, {"age_c:ACS": D}),
        ("§19", "AD age slope, per 10 years",                  s19, {"age_c": D, "age_c:AD": D}),
        ("§19", "AD−SCD slope difference, per 10 years",       s19, {"age_c:AD": D}),
        ("§19", "Adjusted AD−SCD FSI difference at age 75",    s19, {"AD": 1.0, "age_c:AD": sh}),
        ("§19", "Adjusted ACS−SCD FSI difference at age 75",   s19, {"ACS": 1.0, "age_c:ACS": sh}),
    ]
    rows = []
    for src_sec, label, fit, w in specs:
        est, lo, hi, p = _contrast(fit, w)
        rows.append((src_sec, label, est, (lo, hi), p))
    # §21 第一步：joint Wald（β5=β6=0），無點估計/CI，只有 F 檢定的 p
    jw = s19.f_test(_joint_R(s19, ("age_c:AD", "age_c:ACS")))
    rows.append(("§21", "Joint slope test (AD & ACS vs SCD)", None,
                 (float(jw.fvalue), None), float(jw.pvalue)))
    return rows

# ── 輸出：xlsx 報告表（PDF §29）─────────────────────────────────────────────

def _pstr(p):
    star = "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{p:.3e}{star}"


def write_report_xlsx(model, bg_mode, fits, rows, out_path):
    """§29 建議報告表 + 各模型 n / R² / adj R² / HC3 註記。"""
    wb = Workbook()
    ws = wb.active
    ws.title = "FSI_report"
    ws.append(["Result", "Model", "Estimate", "95% CI", "p-value"])
    for src_sec, label, est, ci, p in rows:
        if est is None:                    # §21 joint Wald：無點估計，CI 欄放 F 值
            fval = ci[0]
            ws.append([label, src_sec, "—", f"F = {fval:.2f}", _pstr(p)])
        else:
            lo, hi = ci
            ws.append([label, src_sec, f"{est:+.5f}", f"[{lo:+.5f}, {hi:+.5f}]", _pstr(p)])
    last = ws.max_row

    for row in ws.iter_rows(min_row=1, max_row=last, max_col=5):
        for c in row:
            c.alignment = _CENTER
            c.border = _BORDER
            c.font = _F_HEADER if c.row == 1 else _F_DATA

    # footer：每模型的 n / R² / adjR²（HC3）
    ws.append([])
    ws.append(["Model fit (HC3 robust SE)", "n", "R²", "adj R²", ""])
    for key, name in (("s17", "§17 non-demented common-slope"),
                      ("s18", "§18 non-demented age×ACS"),
                      ("s19", "§19 complete-cohort combined")):
        f = fits[key]
        ws.append([name, int(f.nobs), f"{f.rsquared:.4f}", f"{f.rsquared_adj:.4f}", ""])
    for row in ws.iter_rows(min_row=last + 2, max_row=ws.max_row, max_col=5):
        for c in row:
            c.alignment = _CENTER
            c.border = _BORDER
            c.font = _F_BIG if c.row == last + 2 else _F_DATA

    ws.column_dimensions["A"].width = 42
    ws.column_dimensions["B"].width = 8
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 26
    ws.column_dimensions["E"].width = 14
    ws.row_dimensions[1].height = _ROW_H_HEADER
    for rr in range(2, ws.max_row + 1):
        ws.row_dimensions[rr].height = _ROW_H_DATA

    wb.save(out_path)
    logger.info(f"saved {out_path}")


def write_fsi_rel_xlsx(data, col, out_path):
    """FSI_rel（col 指定範數版本）敘述統計表：整體 + AD/SCD/ACS 各組的 n 與 mean ± SD。"""
    sub_p = REL_NORMS[col][1]                                   # 顯示用下標 ₁ / ₂
    wb = Workbook()
    ws = wb.active
    ws.title = col

    ws.cell(1, 1, f"{col} by diagnostic group").font = _F_BIG
    ws.cell(2, 1, f"{col} = ‖XL−XR‖{sub_p} ÷ ((‖XL‖{sub_p} + ‖XR‖{sub_p}) / 2), dimensionless; "
                  "per-photo ratio averaged over the visit's photos · first visit per "
                  "participant · ArcFace, background").font = Font(
                      name="Times New Roman", size=9, italic=True)

    r0 = 4
    for j, h in enumerate(["Group", "n", f"{col} (mean ± SD)"], 1):
        c = ws.cell(r0, j, h)
        c.alignment, c.font, c.border = _CENTER, _F_HEADER, _BORDER

    groups = [("Overall", data)] + [(g, data[data["Dx"] == g]) for g in ("AD", "SCD", "ACS")]
    for i, (name, sub) in enumerate(groups, 1):
        vals = [name, len(sub), f"{sub[col].mean():.4f} ± {sub[col].std():.4f}"]
        for j, v in enumerate(vals, 1):
            c = ws.cell(r0 + i, j, v)
            c.alignment, c.border = _CENTER, _BORDER
            c.font = _F_HEADER if j == 1 else _F_DATA

    for cl, w in {"A": 14, "B": 8, "C": 22}.items():
        ws.column_dimensions[cl].width = w
    ws.row_dimensions[r0].height = _ROW_H_HEADER
    for rr in range(r0 + 1, r0 + 1 + len(groups)):
        ws.row_dimensions[rr].height = _ROW_H_DATA

    wb.save(out_path)
    logger.info(f"saved {out_path}")
    for name, sub in groups:
        logger.info(f"  {col:10s} {name:8s} n={len(sub):>4}  "
                    f"{sub[col].mean():.4f} ± {sub[col].std():.4f}")


def write_models_txt(fits, out_path):
    """三模型完整 statsmodels summary（含 §18 age_c:ACS 交互項），純文字存查。"""
    titles = {"s17": "§17 Primary non-demented common-slope model",
              "s18": "§18 Non-demented age-by-ACS interaction model",
              "s19": "§19 Complete-cohort combined regression model"}
    with open(out_path, "w", encoding="utf-8") as fh:
        for key in ("s17", "s18", "s19"):
            fh.write("=" * 78 + f"\n{titles[key]}\n" + "=" * 78 + "\n")
            fh.write(str(fits[key].summary()) + "\n\n")
    logger.info(f"saved {out_path}")

# ── 每模型一圖（散點 + 擬合線 + 係數註記，女性 Male=0 參考）────────────────────

_YLABEL = "FSI (mean |left−right| ArcFace embedding)"

def response_spec(normalize=False) -> dict:
    """反應變數規格：col=資料欄名、name=公式與標題用的顯示名、ylabel=y 軸標籤。

    normalize 只改 y 軸標籤的措辭（標明 embedding 是否已依 PDF §4 先 L2 正規化）。
    """
    emb = "L2-normalized ArcFace embedding" if normalize else "ArcFace embedding"
    return {
        "FSI":     {"col": "FSI", "name": "FSI",
                    "ylabel": f"FSI (mean |left−right| {emb})"},
        "log_FSI": {"col": "log_FSI", "name": "log FSI",
                    "ylabel": f"log FSI (ln of mean |left−right| {emb})"},
        "FSI_L1":  {"col": "FSI_L1", "name": "FSI_L1",
                    "ylabel": f"FSI_L1 (sum of |left−right| over 512 dims, {emb})"},
    }


def _splice(rows, extra_rows, fit, insert_after):
    """把 extra_rows 插在 insert_after 這個係數之後（None 則接在最末）。"""
    pos = (fit.model.exog_names.index(insert_after) + 1
           if insert_after else len(rows))
    return rows[:pos] + extra_rows + rows[pos:]


def _h0_block(fit, meanings, extras=(), insert_after=None, joint=None):
    """每係數一行（β 順序）：H0: βi = 0   p=…   95% CI […]   (意義)。p / CI 皆 HC3 兩尾。

    extras: 額外的線性組合列 [(顯示標籤, _contrast 權重, 意義)]；估計/CI/p 同樣由該擬合的
            t_test 導出（與係數列同一 HC3 共變異）。insert_after 指定插在哪個係數之後。
    joint:  (顯示標籤, 係數 tuple, 意義)；PDF §21 joint Wald（F 檢定），無純量 CI 故該欄留「—」，
            接在最末列、與其他 H0 行同寬對齊。
    """
    ci = fit.conf_int()                                   # HC3 95% CI（依擬合的 cov）
    rows = [(f"β{i}=0", fit.pvalues[name], *ci.loc[name], meanings.get(name, ""))
            for i, name in enumerate(fit.model.exog_names)]
    extra_rows = []
    for label, weights, meaning in extras:
        _, lo, hi, p = _contrast(fit, weights)
        extra_rows.append((f"{label}=0", p, lo, hi, meaning))
    rows = _splice(rows, extra_rows, fit, insert_after)

    labels = [lb for lb, *_ in rows] + ([joint[0]] if joint else [])
    w = max(len(lb) for lb in labels)                      # 對齊 p= 欄（joint/extras 標籤較長）
    out = [f"H0: {lb:<{w}}   p={_pstr(p):<11} 95% CI [{lo:+.5f}, {hi:+.5f}]   ({mn})"
           for lb, p, lo, hi, mn in rows]
    if joint:
        h0, terms, meaning = joint
        jp = float(fit.f_test(_joint_R(fit, terms)).pvalue)
        out.append(f"H0: {h0:<{w}}   p={_pstr(jp):<11} 95% CI {'—':<19}   ({meaning})")
    return out


def _beta_values(fit, meanings=None, extras=(), insert_after=None):
    """左上框用：每係數點估計 βi = estimate（意義已在標題，這裡不重複）。

    extras / insert_after 同 _h0_block，額外列出線性組合的點估計（如 β1+β5 = AD 斜率）。
    """
    lines = [f"β{i} = {fit.params[name]:+.6f}"
             for i, name in enumerate(fit.model.exog_names)]
    extra_lines = [f"{label} = {_contrast(fit, weights)[0]:+.6f}"
                   for label, weights, _ in extras]
    return "\n".join(_splice(lines, extra_lines, fit, insert_after))


def _fit_stats(fit):
    """左上框末列：模型配適度。併列 adj R²，避免用 R² 直接比較項數不同的模型。"""
    return (f"R² = {fit.rsquared:.4f}   adj R² = {fit.rsquared_adj:.4f}"
            f"   n = {int(fit.nobs)}")


def _joint_R(fit, terms):
    """以係數位置建 joint 檢定的約束矩陣（避開 exog 名含冒號的字串解析問題）。"""
    names = fit.model.exog_names
    R = np.zeros((len(terms), len(names)))
    for i, t in enumerate(terms):
        R[i, names.index(t)] = 1.0
    return R


_JOINT_TERMS = ("age_c:AD", "age_c:ACS")   # §21 joint Wald 檢定的係數（β5, β6）


def _scatter(data, group_order, lines, ages, title, annot, out_path,
             legend_title="diagnosis", annot_fontsize=8.5, annot_family="DejaVu Sans",
             title_kw=None, ycol="FSI", ylabel=_YLABEL):
    """共用繪圖：依組著色散點 + 每組一條擬合線。

    lines: {group: 對應 ages 兩端點的 y 值}；ages: 直線兩端的年齡。
    annot: 左上註記框內容；空字串則不畫框。title_kw: 傳給 set_title 的樣式（loc/fontsize/字體）。
    ycol/ylabel: 反應變數欄名與 y 軸標籤（FSI 或 log_FSI）。
    """
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for g in group_order:
        sub = data[data["Dx"] == g]
        ax.scatter(sub["Age"], sub[ycol], s=10, alpha=0.30,
                   color=GROUP_COLOR[g], label=f"{g} (n={len(sub)})")
    for g in group_order:
        ax.plot(ages, lines[g], color=GROUP_COLOR[g], lw=2)
    if annot:
        ax.text(0.02, 0.98, annot, transform=ax.transAxes, va="top", ha="left",
                fontsize=annot_fontsize, family=annot_family,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, **(title_kw or {}))
    ax.legend(title=legend_title, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"saved {out_path}")


_TITLE_KW = {"loc": "left", "fontfamily": "DejaVu Sans Mono"}


def plot_s17(data, fit, key, resp, out_path):
    """§17 非失智共同斜率：SCD+ACS 散點 + 兩條平行線。標題列 H0/p/CI；左上框列公式 + βi 值。"""
    c = MODEL_CENTER[key]
    nd = data[data["Dx"] != "AD"]
    b = fit.params
    ages = np.array([nd["Age"].min(), nd["Age"].max()]); ac = ages - c
    lines = {"SCD": b["Intercept"] + b["age_c"] * ac,
             "ACS": b["Intercept"] + b["ACS"] + b["age_c"] * ac}
    meanings = {"Intercept": f"{c}-yr female SCD", "age_c": "age slope /yr",
                "Male": "male − female", "ACS": f"ACS − SCD @{c}"}
    title = "\n".join(["whether asymmetry has a relationship with age"]
                      + _h0_block(fit, meanings))
    annot = (f"{resp['name']} = β0 + β1·age_c + β2·Male + β3·ACS   (age_c = age − {c})\n"
             + _beta_values(fit, meanings) + "\n\n" + _fit_stats(fit))
    _scatter(nd, ["SCD", "ACS"], lines, ages, title, annot, out_path,
             legend_title=None, annot_fontsize=8.0, annot_family="DejaVu Sans Mono",
             title_kw={**_TITLE_KW, "fontsize": 8.5},
             ycol=resp["col"], ylabel=resp["ylabel"])


def plot_s18(data, fit, key, resp, out_path):
    """§18 SCD vs ACS 交互：SCD+ACS 散點 + 兩條各自斜率線。標題列 H0/p/CI；左上框列公式 + βi 值。"""
    c = MODEL_CENTER[key]
    nd = data[data["Dx"] != "AD"]
    b = fit.params
    ages = np.array([nd["Age"].min(), nd["Age"].max()]); ac = ages - c
    lines = {"SCD": b["Intercept"] + b["age_c"] * ac,
             "ACS": b["Intercept"] + b["ACS"] + (b["age_c"] + b["age_c:ACS"]) * ac}
    meanings = {"Intercept": f"{c}-yr female SCD", "age_c": "SCD slope /yr",
                "Male": "male − female", "ACS": f"ACS − SCD @{c}",
                "age_c:ACS": "ACS − SCD slope diff"}
    title = "\n".join(["whether the age slope differs between SCD and ACS"]
                      + _h0_block(fit, meanings))
    annot = (f"{resp['name']} = β0 + β1·age_c + β2·Male + β3·ACS + β4·(age_c×ACS)"
             f"   (age_c = age − {c})\n" + _beta_values(fit, meanings)
             + "\n\n" + _fit_stats(fit))
    _scatter(nd, ["SCD", "ACS"], lines, ages, title, annot, out_path,
             legend_title=None, annot_fontsize=8.0, annot_family="DejaVu Sans Mono",
             title_kw={**_TITLE_KW, "fontsize": 8.5},
             ycol=resp["col"], ylabel=resp["ylabel"])


def plot_s19(data, fit, key, resp, out_path):
    """§19 全世代合併：AD+SCD+ACS 散點 + 三條各自斜率線。標題列 H0/p/CI；左上框列公式 + βi 值。"""
    c = MODEL_CENTER[key]
    b = fit.params
    ages = np.array([data["Age"].min(), data["Age"].max()]); ac = ages - c
    lines = {"SCD": b["Intercept"] + b["age_c"] * ac,
             "AD":  b["Intercept"] + b["AD"] + (b["age_c"] + b["age_c:AD"]) * ac,
             "ACS": b["Intercept"] + b["ACS"] + (b["age_c"] + b["age_c:ACS"]) * ac}
    meanings = {"Intercept": f"{c}-yr female SCD", "age_c": "SCD slope /yr",
                "AD": f"AD − SCD @{c}", "ACS": f"ACS − SCD @{c}", "Male": "male − female",
                "age_c:AD": "AD − SCD slope diff", "age_c:ACS": "ACS − SCD slope diff"}
    # §20.2 / §20.3：各組自身的年齡斜率（線性組合，t_test 取估計/CI/p）。
    # 緊接在 β1（SCD 斜率）之後，三組斜率並排好對讀。
    extras = (("β1+β5", {"age_c": 1.0, "age_c:AD": 1.0},  "AD slope /yr"),
              ("β1+β6", {"age_c": 1.0, "age_c:ACS": 1.0}, "ACS slope /yr"))
    # §21 第一步：joint Wald（β5=β6=0），三組斜率整體檢定，當個別斜率差的守門結論
    joint = ("β5=β6=0", _JOINT_TERMS, "all group age slopes equal")
    title = "\n".join(["whether AD differs from SCD in FSI level and age slope"]
                      + _h0_block(fit, meanings, extras, insert_after="age_c", joint=joint))
    annot = (f"{resp['name']} = β0 + β1·age_c + β2·AD + β3·ACS + β4·Male\n"
             f"          + β5·(age_c×AD) + β6·(age_c×ACS)   (age_c = age − {c})\n"
             + _beta_values(fit, meanings, extras, insert_after="age_c")
             + "\n\n" + _fit_stats(fit))
    _scatter(data, ["SCD", "ACS", "AD"], lines, ages, title, annot, out_path,
             legend_title=None, annot_fontsize=7.0, annot_family="DejaVu Sans Mono",
             title_kw={**_TITLE_KW, "fontsize": 8.0},
             ycol=resp["col"], ylabel=resp["ylabel"])


# 每張圖 → (子資料夾, fit 鍵, 繪圖函式, 檔名 stem)
MODEL_DIRS = [
    ("s17_nondemented_common_slope", "s17", plot_s17, "s17"),
    ("s18_nondemented_age_x_acs",    "s18", plot_s18, "s18"),
    ("s19_complete_cohort_combined", "s19", plot_s19, "s19"),
]
# log FSI 版：§19 另出置中 65 一版（三線在 63-65 歲交會，置中點會改變 β2/β3 的顯著性）
LOG_MODEL_DIRS = [
    ("s17_nondemented_common_slope", "s17",    plot_s17, "s17_log"),
    ("s18_nondemented_age_x_acs",    "s18",    plot_s18, "s18_log"),
    ("s19_complete_cohort_combined", "s19",    plot_s19, "s19_log"),
    ("s19_complete_cohort_combined", "s19c65", plot_s19, "s19_log_center65"),
]
# 未除 d 的 L1 版：與 FSI 版僅差常數縮放（β/CI ×512、p/R² 不變），供對照單位用
L1_MODEL_DIRS = [
    ("s17_nondemented_common_slope", "s17", plot_s17, "s17_l1"),
    ("s18_nondemented_age_x_acs",    "s18", plot_s18, "s18_l1"),
    ("s19_complete_cohort_combined", "s19", plot_s19, "s19_l1"),
]

# ── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-visit", choices=list(P_VISIT_TOKENS), default=DEFAULT_TOKENS[0])
    ap.add_argument("--p-score", choices=list(P_SCORE_TOKENS), default=DEFAULT_TOKENS[1])
    ap.add_argument("--hc-visit", choices=list(HC_VISIT_TOKENS), default=DEFAULT_TOKENS[2])
    ap.add_argument("--hc-score", choices=list(HC_SCORE_TOKENS), default=DEFAULT_TOKENS[3])
    ap.add_argument("--model", default="arcface")
    ap.add_argument("--bg-mode", choices=["background", "no_background"], default="background")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="覆寫輸出目錄；留空依 cohort 自動決定")
    args = ap.parse_args()

    cohort = (args.p_visit, args.p_score, args.hc_visit, args.hc_score)
    out_base = args.output_dir or (
        EMBEDDING_FEATURE_STAT_DIR / cohort_path(*cohort) / args.bg_mode / args.model / "fsi")
    out_base.mkdir(parents=True, exist_ok=True)
    logger.info(f"cohort = {cohort}  model = {args.model}  bg = {args.bg_mode}")
    logger.info(f"output-dir = {out_base}")

    # 兩棵平行輸出樹：no_normalize = 現行做法；normalize = PDF §4（先各自 L2 正規化）
    for normalize, tree in ((False, "no_normalize"), (True, "normalize")):
        base = out_base / tree
        base.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 70)
        logger.info(f"── {tree} ── {base}")

        data = build_participant_dataset(cohort, args.model, args.bg_mode, normalize)
        logger.info("participant counts: " + str(data["Dx"].value_counts().to_dict()))
        data.to_csv(base / "fsi_dataset.csv", index=False)

        fits = fit_models(data, y="FSI")
        rows = build_report_rows(fits)
        write_report_xlsx(args.model, args.bg_mode, fits, rows, base / "fsi_report.xlsx")
        write_models_txt(fits, base / "fsi_models.txt")
        write_fsi_rel_xlsx(data, "FSI_rel", base / "fsi_rel_stat.xlsx")
        write_fsi_rel_xlsx(data, "FSI_rel_L2", base / "fsi_rel_l2_stat.xlsx")

        # 每模型一個子資料夾：FSI / log FSI / FSI_L1 三套圖 + 係數表 csv（coef/se/p/CI）
        resp_spec = response_spec(normalize)
        all_fits = {"FSI": fits, "log_FSI": fit_models(data, y="log_FSI"),
                    "FSI_L1": fit_models(data, y="FSI_L1")}
        for resp_key, specs in (("FSI", MODEL_DIRS), ("log_FSI", LOG_MODEL_DIRS),
                                ("FSI_L1", L1_MODEL_DIRS)):
            resp, fits_r = resp_spec[resp_key], all_fits[resp_key]
            for subdir, key, plot_fn, stem in specs:
                d = base / subdir
                d.mkdir(parents=True, exist_ok=True)
                plot_fn(data, fits_r[key], key, resp, d / f"{stem}.png")
                fits_r[key].summary2().tables[1].to_csv(d / f"{stem}_coef.csv")

        logger.info(f"── {tree}: FSI report (per 10 years for slopes) ──")
        for src_sec, label, est, ci, p in rows:
            if est is None:                # §21 joint Wald：F 值 + p
                logger.info(f"  [{src_sec}] {label}: F={ci[0]:.2f}  p={_pstr(p)}")
            else:
                logger.info(f"  [{src_sec}] {label}: {est:+.5f}  "
                            f"95%CI[{ci[0]:+.5f},{ci[1]:+.5f}]  p={_pstr(p)}")


if __name__ == "__main__":
    main()
