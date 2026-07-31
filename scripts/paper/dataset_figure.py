"""論文用「資料集集合圖」：P（患者）vs HC（對照組＝NAD＋ACS）的納入/配對關係。

以巢狀圓呈現（資料圓面積 ∝ 人數、單一比例尺；外框為分組邊界、非等面積，總數標於框外）：
  左  P 全體外框內，配對成功的 596 人拆成「配 SCD 519 + 配 ACS 77」兩圓。
  右  HC 兩個子群（NAD→顯示為 SCD、ACS）各有多少人被抽用當對照，
      以及被抽用的「人數 / 人次」（HC 端每次拜訪可被重複配，故人次 > 人數）。

── 數字來源（全部由真實資料重算，非手填）────────────────────────────────
  cohort/配對設定：p_first · p_cdrall · hc_all · hc_cdrall_or_mmseall
                   priority=["ACS"] · level="visit" · caliper=1.0
  這組設定重現 app 版圖上每一個數字：
    P 1061 → 配上 596（519 配 NAD、77 配 ACS）、未配上 465
    NAD 458 人中 290 人被抽用（519 人次）；ACS 91 人中 31 人被抽用（77 人次）

── 標籤 vs 臨床光譜（CN / SCD / MCI / Dementia）的重要註記 ──────────────
  cohort 代號（P / NAD / ACS）與臨床認知分級不是一對一，subject-level 實測：
    P   n=1061  MMSE≈15.9  CDR: 0.5→272(≈27%,屬 MCI)、1→549、2→145、3→8、0→16
        → 以 Dementia 為主但含相當比例 MCI；標「失智症患者」為簡化。
    NAD n=458   MMSE≈24.7  CDR 多數缺(NaN 307)；有值者 0→42、0.5→94、1→14
        → CN/SCD/MCI 混合，非純 SCD；標「SCD」為簡化。
    ACS n=91    MMSE≈28.3  年齡≈57.9、全無 CDR
        → 較接近 CN（健康年輕對照），與「SCD」不同層級。
  因此本圖沿用你 app 版的標法只是「沿用」，非臨床精確。要改標籤：改下方
  GROUP_LABEL / P_TITLE_LABEL 一處即可，數字與版面會自動不變。

用法：
    python scripts/paper/dataset_figure.py                 # 輸出到預設路徑
    python scripts/paper/dataset_figure.py -o out.png      # 指定輸出
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from _paths import PROJECT_ROOT  # noqa: F401  (亦把專案根/scripts 塞進 sys.path)

from src.common.cohort import base_id_of, load_demographics
from src.common.matching import match_by_age

# ── 可調：顯示標籤（cohort 代號 → 圖上文字）──────────────────────────────────
# 預設沿用 app 版：NAD 顯示為 SCD、ACS 顯示為 ACS。臨床精確性見檔頭註記。
GROUP_LABEL = {"NAD": "SCD", "ACS": "ACS"}
P_TITLE_LABEL = "失智症患者"

DEFAULT_OUTPUT = PROJECT_ROOT / "paper" / "figures" / "資料集.png"

# ── 配色 ─────────────────────────────────────────────────────────────────────
ORANGE, ORANGE_FILL, ORANGE_TEXT = "#DE6B2F", "#F6E7DE", "#8A431C"
GREEN,  GREEN_FILL,  GREEN_TEXT  = "#1F9C78", "#D7EEE5", "#1C7A5E"
PURPLE, PURPLE_FILL, PURPLE_TEXT = "#6A5FD0", "#E4E1F7", "#463FA0"
BLUE = "#2F6FB0"                                     # HC 母體邊界（藍色實線）
TITLE_C, DARK = "#333333", "#222222"

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


# ── 數字：全部由真實資料重算 ─────────────────────────────────────────────────
def compute_counts() -> dict:
    """回傳圖上所有人數/人次，來源＝hospital_A.csv ＋ 年齡配對。"""
    demo = load_demographics()
    demo = demo[demo["Age"].notna()].copy()
    nunq = lambda g: demo[demo["Group"] == g]["base_id"].nunique()

    p_ids, hc_ids = match_by_age(
        "p_first", "p_cdrall", "hc_all", "hc_cdrall_or_mmseall",
        controls=None, caliper=1.0, priority=["ACS"], level="visit", mode="1to1",
    )
    hc = [str(x) for x in hc_ids]
    visits = lambda g: sum(1 for x in hc if x.startswith(g))
    subj = lambda g: len({base_id_of(x) for x in hc if x.startswith(g)})

    p_total = nunq("P")
    return {
        "p_total": p_total,
        "p_matched": len(p_ids),
        "p_unmatched": p_total - len(p_ids),
        "nad_total": nunq("NAD"), "acs_total": nunq("ACS"),
        "nad_visits": visits("NAD"), "acs_visits": visits("ACS"),
        "nad_subj": subj("NAD"), "acs_subj": subj("ACS"),
    }


# ── 幾何：資料圓面積 ∝ 人數（單一比例尺 K）；外框為等大分組邊界 ──────────────
K = 20.0 / (1061 ** 0.5)          # 尺規：sqrt-面積 → 半徑
RING = 22.5                       # 兩族群外框半徑（P、HC 等大；分組邊界、非等面積）
CY = 32                           # 兩外框共用垂直中心


def _r(n: float) -> float:
    return K * (n ** 0.5)


def _circle(ax, cx, cy, r, *, fc, ec, lw=2.0, ls="-", z=1):
    ax.add_patch(Circle((cx, cy), r, facecolor=fc, edgecolor=ec,
                        linewidth=lw, linestyle=ls, zorder=z))


def _txt(ax, x, y, s, *, size, color=DARK, weight="normal", z=5):
    ax.text(x, y, s, ha="center", va="center", fontsize=size,
            color=color, fontweight=weight, zorder=z)


def draw(counts: dict, output: Path) -> Path:
    scd = GROUP_LABEL["NAD"]
    acs = GROUP_LABEL["ACS"]

    fig = plt.figure(figsize=(13.6, 9.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── 標題（僅組別名稱；各資料集總人數改標在實線圈外）──
    _txt(ax, 25, 62, f"P — {P_TITLE_LABEL}", size=23, color=TITLE_C, weight="bold")
    _txt(ax, 73, 62, "HC — 對照組", size=23, color=TITLE_C, weight="bold")

    # ── 左：P（外框＝全體，與 HC 等大；配對 596 拆成 配SCD 519 + 配ACS 77 兩圓）──
    _circle(ax, 25, CY, RING, fc="white", ec=ORANGE, lw=2.2, z=1)
    _txt(ax, 25, CY + RING + 2.2, f"{counts['p_total']:,} 人",          # 總數標於框外
         size=17, color=ORANGE_TEXT, weight="bold")
    # 配 SCD 的 519（1:1 配對，故 = SCD 人次）
    _circle(ax, 20.5, 32, _r(counts["nad_visits"]), fc=ORANGE_FILL, ec=ORANGE, lw=1.2, z=2)
    _txt(ax, 20.5, 33.3, f"{counts['nad_visits']} 人", size=20, color=DARK, weight="bold")
    _txt(ax, 20.5, 30.5, f"配 {scd}", size=14, color=GREEN_TEXT)
    # 配 ACS 的 77（置於 519 正右方，同一水平線）
    _circle(ax, 41, 32, _r(counts["acs_visits"]), fc=ORANGE_FILL, ec=ORANGE, lw=1.2, z=2)
    _txt(ax, 41, 32.9, f"{counts['acs_visits']} 人", size=16, color=DARK, weight="bold")
    _txt(ax, 41, 30.7, f"配 {acs}", size=12, color=PURPLE_TEXT)

    # ── 右：HC（藍色外框＝全體，與 P 等大；SCD/ACS 子群）──
    _circle(ax, 73, CY, RING, fc="none", ec=BLUE, lw=2.0, z=1)
    _txt(ax, 73, CY + RING + 2.2,                                       # HC 總數標於框外
         f"{counts['nad_total'] + counts['acs_total']} 人",
         size=17, color=BLUE, weight="bold")

    # SCD (=NAD)
    rs_out, rs_in = _r(counts["nad_total"]), _r(counts["nad_subj"])
    _circle(ax, 66, 32, rs_out, fc="none", ec=GREEN, lw=2.0, z=2)
    _circle(ax, 66, 32, rs_in, fc=GREEN_FILL, ec=GREEN, lw=1.2, z=3)
    _txt(ax, 66, 32 + rs_out + 1.6, f"{scd} {counts['nad_total']} 人",  # 標於綠圈外
         size=15, color=GREEN_TEXT, weight="bold")
    _txt(ax, 66, 33.2, f"{counts['nad_subj']} 人", size=21, color=GREEN_TEXT, weight="bold")
    _txt(ax, 66, 30.6, f"{counts['nad_visits']} 人次", size=15, color=GREEN_TEXT)

    # ACS
    ra_out, ra_in = _r(counts["acs_total"]), _r(counts["acs_subj"])
    _circle(ax, 88, 31, ra_out, fc="none", ec=PURPLE, lw=2.0, z=2)
    _circle(ax, 88, 31, ra_in, fc=PURPLE_FILL, ec=PURPLE, lw=1.2, z=3)
    _txt(ax, 88, 31 + ra_out + 1.6, f"{acs} {counts['acs_total']} 人",  # 標於紫圈外
         size=15, color=PURPLE_TEXT, weight="bold")
    _txt(ax, 88, 31.9, f"{counts['acs_subj']} 人", size=17, color=PURPLE_TEXT, weight="bold")
    _txt(ax, 88, 29.9, f"{counts['acs_visits']} 人次", size=13, color=PURPLE_TEXT)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    counts = compute_counts()
    print("counts =", counts)
    out = draw(counts, args.output)
    print("saved  ->", out)


if __name__ == "__main__":
    main()
