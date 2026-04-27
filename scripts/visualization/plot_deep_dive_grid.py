"""
Render Section 3.5 deep-dive grid: 14 data rows × 16 cells with 5-row header.

Reads (default GRID_VARIANT=acs):
  workspace/arms_analysis/grid/<GRID_VARIANT>/stat_grid_long.csv
  workspace/arms_analysis/grid/<GRID_VARIANT>/cell_header_stats.csv

Outputs:
  workspace/arms_analysis/grid/<GRID_VARIANT>/stat_grid.png
  workspace/arms_analysis/grid/<GRID_VARIANT>/stat_grid_markdown.md

GRID_VARIANT examples:
  acs                                              (baseline, internal ACS)
  acs_ext / eacs                                   (HC sensitivity)
  subsets/acs_ext_utkface_AB_age_only_age_error    (subset exploration)

Header rows:
  1 — arm group label (A/B/C/D with descriptor)
  2 — comparison full name (e.g. `AD vs HC`, `AD high-MMSE vs AD low-MMSE`)
  3 — n_all (raw visits across cohort base_ids) / n_unique (subjects)
  4 — Age mean±SD per group + Welch t p-value
  5 — hi-lo cells: MMSE / CASI / CDR_SB mean±SD + Welch t p; other cells: —
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
_variant = os.environ.get("GRID_VARIANT", "acs")
DEEP = ROOT / "workspace" / "arms_analysis" / "grid" / _variant
LONG_CSV = DEEP / "stat_grid_long.csv"
HEADER_STATS_CSV = DEEP / "cell_header_stats.csv"

ROW_ORDER = [
    ("age_only", None),
    ("embedding_arcface_mean", None),
    ("embedding_dlib_mean", None),
    ("embedding_topofr_mean", None),
    ("embedding_arcface_asymmetry", "L2 scalar"),
    ("embedding_arcface_asymmetry", "full vector"),
    ("embedding_dlib_asymmetry", "L2 scalar"),
    ("embedding_dlib_asymmetry", "full vector"),
    ("embedding_topofr_asymmetry", "L2 scalar"),
    ("embedding_topofr_asymmetry", "full vector"),
    ("landmark_asymmetry", "4-d per-region L2"),
    ("landmark_asymmetry", "130-d raw xy"),
    ("emotion_8methods", None),
    ("age_error", None),
]

# 5 directions: each row in ROW_ORDER assigned by modality_parent
DIRECTION_MAP = {
    "age_only": "age",
    "age_error": "age",
    "embedding_arcface_mean": "embedding_mean",
    "embedding_dlib_mean": "embedding_mean",
    "embedding_topofr_mean": "embedding_mean",
    "embedding_arcface_asymmetry": "embedding_asymmetry",
    "embedding_dlib_asymmetry": "embedding_asymmetry",
    "embedding_topofr_asymmetry": "embedding_asymmetry",
    "landmark_asymmetry": "landmark_asymmetry",
    "emotion_8methods": "emotion",
}
DIRECTIONS = ["age", "embedding_mean", "embedding_asymmetry",
              "landmark_asymmetry", "emotion"]

ARM_ORDER = ["A", "B", "C", "D"]
ARM_LABELS = {
    "A": "A (cross-sectional naive)",
    "B": "B (cross-sectional matched)",
    "C": "C (longitudinal naive)",
    "D": "D (longitudinal matched)",
}
COMPARE_ORDER = ["HC", "NAD", "ACS", "mmse-hi-lo", "casi-hi-lo"]
# Row-2 label — hi-lo broken into 3 sub-lines (vs on its own line)
COMPARE_FULLNAME_HTML = {
    "HC": "AD vs HC",
    "NAD": "AD vs NAD",
    "ACS": "AD vs ACS",
    "mmse-hi-lo": "AD high-MMSE<br>vs<br>AD low-MMSE",
    "casi-hi-lo": "AD high-CASI<br>vs<br>AD low-CASI",
}
COMPARE_FULLNAME_PLAIN = {
    "HC": "AD vs HC",
    "NAD": "AD vs NAD",
    "ACS": "AD vs ACS",
    "mmse-hi-lo": "AD high-MMSE\nvs\nAD low-MMSE",
    "casi-hi-lo": "AD high-CASI\nvs\nAD low-CASI",
}
# Per-group short label for row-3 n text
GROUP_LABEL = {
    "HC":    ("AD", "HC"),
    "NAD":   ("AD", "NAD"),
    "ACS":   ("AD", "ACS"),
    "mmse-hi-lo": ("AD high", "AD low"),
    "casi-hi-lo": ("AD high", "AD low"),
}


# ---- formatting helpers ----

def _sig_stars(q):
    if pd.isna(q):
        return "ns"
    if q < 0.001: return "***"
    if q < 0.01: return "**"
    if q < 0.05: return "*"
    if q < 0.10: return "."
    return "ns"


def _fmt_p(p):
    if pd.isna(p):
        return "p=n/a"
    if p < 0.001: return "p<.001"
    if p < 0.01:  return "p<.01"
    if p < 0.05:  return f"p={p:.3f}"
    return f"p={p:.2f}"


def _fmt_ms(m, s):
    if pd.isna(m):
        return "—"
    if pd.isna(s):
        return f"{m:.1f}"
    return f"{m:.1f}±{s:.1f}"


def _cell_text(row):
    if pd.isna(row.get("p")) and row.get("skip_reason"):
        return "n/a"
    stat_kind = row.get("effect_type", "")
    effect = row.get("effect", np.nan)
    q = row.get("q", np.nan)
    stars = _sig_stars(q)
    sig_par = f" ({stars})"
    if stat_kind == "cohens_d":
        eff_str = f"d={effect:+.2f}" if not pd.isna(effect) else "d=?"
    elif stat_kind == "mahalanobis_D2":
        eff_str = f"D²={effect:.2f}" if not pd.isna(effect) else "D²=?"
    elif stat_kind == "R2":
        eff_str = f"R²={effect:.3f}" if not pd.isna(effect) else "R²=?"
    elif stat_kind == "mean_T2":
        eff_str = f"T̄²={effect:.1f}" if not pd.isna(effect) else "T̄²=?"
    else:
        eff_str = "?"
    auc = row.get("auc_auc", np.nan)
    auc_lo = row.get("auc_auc_ci_low", np.nan)
    auc_hi = row.get("auc_auc_ci_high", np.nan)
    if not pd.isna(auc):
        auc_str = (f"AUC={auc:.2f}" if pd.isna(auc_lo)
                    else f"AUC={auc:.2f}[{auc_lo:.2f},{auc_hi:.2f}]")
    else:
        auc_str = ""
    return f"{eff_str}{sig_par}\n{auc_str}"


def _row_label(parent, sub):
    if sub is None or (isinstance(sub, float) and pd.isna(sub)):
        return parent
    return f"{parent}\n  [{sub}]"


def _get_hdr(hdr_df, arm, cmp):
    sub = hdr_df[(hdr_df.arm == arm) & (hdr_df.comparison == cmp)]
    return sub.iloc[0] if len(sub) else None


def _header_row3_text(h, cmp, sep="<br>"):
    """Per-group n: 2 lines (one per label group).
      `{grp1} n = X visits / Y subject`
      `{grp2} n = X visits / Y subject`
    """
    if h is None or pd.isna(h.get("n_all_1")):
        return "—"
    g1, g2 = GROUP_LABEL.get(cmp, ("AD", "ctrl"))
    n1v, n1u = int(h["n_all_1"]), int(h["n_unique_1"])
    n2v, n2u = int(h["n_all_0"]), int(h["n_unique_0"])
    return (f"{g1}{sep}n = {n1v:,} visits / {n1u:,} subject{sep}"
            f"{g2}{sep}n = {n2v:,} visits / {n2u:,} subject")


def _header_row4_text(h, sep="<br>"):
    """Age — 3 sub-lines: label, values, p."""
    if h is None or pd.isna(h.get("age_mean_1")):
        return "—"
    g1 = _fmt_ms(h.get("age_mean_1"), h.get("age_sd_1"))
    g2 = _fmt_ms(h.get("age_mean_2"), h.get("age_sd_2"))
    p = _fmt_p(h.get("age_p"))
    return f"Age:{sep}{g1} / {g2} ({p})"


def _header_row5_text(h, cmp, sep="<br>"):
    """hi-lo only — MMSE/CASI/CDR each as 3 sub-lines, blank line between
    CASI and CDR (visual grouping: screening tests vs staging)."""
    if cmp not in ("mmse-hi-lo", "casi-hi-lo") or h is None:
        return "—"
    blocks = []
    for i, (short, pretty) in enumerate(
        [("mmse", "MMSE"), ("casi", "CASI"), ("cdr", "CDR")]):
        m1, s1 = h.get(f"{short}_mean_1"), h.get(f"{short}_sd_1")
        m2, s2 = h.get(f"{short}_mean_2"), h.get(f"{short}_sd_2")
        p = h.get(f"{short}_p")
        if pd.isna(m1):
            continue
        blocks.append((
            f"{pretty}:{sep}{_fmt_ms(m1, s1)} / {_fmt_ms(m2, s2)} "
            f"({_fmt_p(p)})"
        ))
    if not blocks:
        return "—"
    return sep.join(blocks)


# Legacy aliases for the PNG path (identical helpers, \n separator)
_header_row3_text_plain = lambda h, cmp: _header_row3_text(h, cmp, sep="\n")
_header_row4_text_plain = lambda h: _header_row4_text(h, sep="\n")
_header_row5_text_plain = lambda h, cmp: _header_row5_text(h, cmp, sep="\n")


# ---- PNG ----

N_HEADER_ROWS = 5  # arm / compare / n / age / cog
# Heights (row units): halved from prior values for a compact header.
ROW_HEIGHTS = [0.5, 1.1, 1.2, 0.6, 1.9]
DATA_ROW_H = 0.9  # data cells are short (2 lines) — squash vertically


def render_grid(long_df, hdr_df, out_png, out_md, row_order=None):
    if row_order is None:
        row_order = ROW_ORDER
    n_rows = len(row_order)
    n_cols = len(ARM_ORDER) * len(COMPARE_ORDER)

    # --- PNG ---
    header_h_total = sum(ROW_HEIGHTS)
    data_h_total = DATA_ROW_H * n_rows
    fig_w = 3.0 * n_cols + 6
    fig_h = data_h_total + header_h_total + 1
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, data_h_total + header_h_total)
    ax.invert_yaxis()
    ax.axis("off")

    row_y_offsets = [0]
    for h in ROW_HEIGHTS:
        row_y_offsets.append(row_y_offsets[-1] + h)
    # row i spans [row_y_offsets[i], row_y_offsets[i+1]]

    # Row 1: arm label
    y0, y1 = row_y_offsets[0], row_y_offsets[1]
    for ai, arm in enumerate(ARM_ORDER):
        x0 = ai * len(COMPARE_ORDER)
        ax.add_patch(Rectangle((x0, y0), len(COMPARE_ORDER), y1 - y0,
                                 facecolor="#D0E3F0", edgecolor="black",
                                 linewidth=1.0))
        ax.text(x0 + len(COMPARE_ORDER) / 2, (y0 + y1) / 2,
                 ARM_LABELS[arm], ha="center", va="center",
                 fontweight="bold", fontsize=19)

    # Row 2: comparison full name
    y0, y1 = row_y_offsets[1], row_y_offsets[2]
    for ai in range(len(ARM_ORDER)):
        for ci, cmp in enumerate(COMPARE_ORDER):
            x = ai * len(COMPARE_ORDER) + ci
            ax.add_patch(Rectangle((x, y0), 1, y1 - y0,
                                     facecolor="#EEEEEE", edgecolor="black",
                                     linewidth=1.0))
            ax.text(x + 0.5, (y0 + y1) / 2, COMPARE_FULLNAME_PLAIN[cmp],
                     ha="center", va="center", fontsize=14, fontweight="bold",
                     linespacing=1.8)

    # Row 3: per-group n
    y0, y1 = row_y_offsets[2], row_y_offsets[3]
    for ai, arm in enumerate(ARM_ORDER):
        for ci, cmp in enumerate(COMPARE_ORDER):
            x = ai * len(COMPARE_ORDER) + ci
            ax.add_patch(Rectangle((x, y0), 1, y1 - y0,
                                     facecolor="#F5F5F5", edgecolor="black",
                                     linewidth=1.0))
            txt = _header_row3_text_plain(_get_hdr(hdr_df, arm, cmp), cmp)
            ax.text(x + 0.5, (y0 + y1) / 2, txt,
                     ha="center", va="center", fontsize=13,
                     linespacing=1.8)

    # Row 4: Age stats
    y0, y1 = row_y_offsets[3], row_y_offsets[4]
    for ai, arm in enumerate(ARM_ORDER):
        for ci, cmp in enumerate(COMPARE_ORDER):
            x = ai * len(COMPARE_ORDER) + ci
            ax.add_patch(Rectangle((x, y0), 1, y1 - y0,
                                     facecolor="#FAFAFA", edgecolor="black",
                                     linewidth=1.0))
            txt = _header_row4_text_plain(_get_hdr(hdr_df, arm, cmp))
            ax.text(x + 0.5, (y0 + y1) / 2, txt,
                     ha="center", va="center", fontsize=13,
                     linespacing=1.8)

    # Row 5: cog stats (hi-lo only)
    y0, y1 = row_y_offsets[4], row_y_offsets[5]
    for ai, arm in enumerate(ARM_ORDER):
        for ci, cmp in enumerate(COMPARE_ORDER):
            x = ai * len(COMPARE_ORDER) + ci
            ax.add_patch(Rectangle((x, y0), 1, y1 - y0,
                                     facecolor="#FAFAFA", edgecolor="black",
                                     linewidth=1.0))
            txt = _header_row5_text_plain(_get_hdr(hdr_df, arm, cmp), cmp)
            ax.text(x + 0.5, (y0 + y1) / 2, txt,
                     ha="center", va="center", fontsize=13,
                     linespacing=1.8)

    # Row labels (left, drawn outside x=0 via negative x)
    for ri, (parent, sub) in enumerate(row_order):
        y = header_h_total + ri * DATA_ROW_H + DATA_ROW_H / 2
        ax.text(-0.15, y, _row_label(parent, sub),
                 ha="right", va="center", fontsize=14)

    # Data cells
    for ri, (parent, sub) in enumerate(row_order):
        for ai, arm in enumerate(ARM_ORDER):
            for ci, cmp in enumerate(COMPARE_ORDER):
                mask = ((long_df["modality_parent"] == parent) &
                        (long_df["arm"] == arm) &
                        (long_df["comparison"] == cmp))
                if sub is None:
                    mask = mask & long_df["modality_sub"].isna()
                else:
                    mask = mask & (long_df["modality_sub"] == sub)
                r = long_df[mask]
                x = ai * len(COMPARE_ORDER) + ci
                y = header_h_total + ri * DATA_ROW_H
                bg = "white"
                if len(r) == 0:
                    text = "—"
                else:
                    rr = r.iloc[0]
                    text = _cell_text(rr)
                    q = rr.get("q", np.nan)
                    if not pd.isna(q):
                        if q < 0.001: bg = "#FFD8D8"
                        elif q < 0.01: bg = "#FFE6E6"
                        elif q < 0.05: bg = "#FFF2F2"
                        elif q < 0.10: bg = "#FFFAFA"
                ax.add_patch(Rectangle((x, y), 1, DATA_ROW_H,
                                         facecolor=bg, edgecolor="black",
                                         linewidth=1.0))
                ax.text(x + 0.5, y + DATA_ROW_H / 2, text,
                         ha="center", va="center", fontsize=12,
                         linespacing=1.8)

    # Thick outer frame per arm (drawn last so it sits on top of inner edges)
    total_h = header_h_total + DATA_ROW_H * n_rows
    for ai in range(len(ARM_ORDER)):
        x0 = ai * len(COMPARE_ORDER)
        ax.add_patch(Rectangle((x0, 0), len(COMPARE_ORDER), total_h,
                                 facecolor="none", edgecolor="black",
                                 linewidth=3.0))

    plt.title(f"Section 3.5 Deep-dive — {n_rows} rows × {n_cols} cells "
               "(5-row header: arm / comparison / n / Age / cog)",
               fontsize=18)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")

    # --- HTML ---
    center = 'style="text-align:center"'
    arm_sep_head = 'style="text-align:center;border-left:2px solid #555"'
    cell_center = 'style="text-align:center"'
    cell_arm_sep = 'style="text-align:center;border-left:2px solid #555"'

    html = ["<table>", "<thead>"]

    # Row 1: Modality rowspan=5 + arm labels
    html.append("<tr>")
    html.append(f'<th rowspan="5" {center}>Modality</th>')
    for i, arm in enumerate(ARM_ORDER):
        st = arm_sep_head if i > 0 else center
        html.append(f'<th colspan="{len(COMPARE_ORDER)}" {st}>'
                     f'{ARM_LABELS[arm]}</th>')
    html.append("</tr>")

    # Row 2: comparison full name
    html.append("<tr>")
    for ai in range(len(ARM_ORDER)):
        for ci, cmp in enumerate(COMPARE_ORDER):
            st = arm_sep_head if (ai > 0 and ci == 0) else center
            html.append(f"<th {st}>{COMPARE_FULLNAME_HTML[cmp]}</th>")
    html.append("</tr>")

    # Row 3: per-group n (2 lines: AD group / ctrl group)
    html.append("<tr>")
    for ai, arm in enumerate(ARM_ORDER):
        for ci, cmp in enumerate(COMPARE_ORDER):
            st = arm_sep_head if (ai > 0 and ci == 0) else center
            txt = _header_row3_text(_get_hdr(hdr_df, arm, cmp), cmp)
            html.append(f"<th {st}>{txt}</th>")
    html.append("</tr>")

    # Row 4: Age (3 sub-lines: "Age:", values, p)
    html.append("<tr>")
    for ai, arm in enumerate(ARM_ORDER):
        for ci, cmp in enumerate(COMPARE_ORDER):
            st = arm_sep_head if (ai > 0 and ci == 0) else center
            txt = _header_row4_text(_get_hdr(hdr_df, arm, cmp))
            html.append(f"<th {st}>{txt}</th>")
    html.append("</tr>")

    # Row 5: cog stats (hi-lo only) — MMSE/CASI, blank, CDR; each 3 sub-lines
    html.append("<tr>")
    for ai, arm in enumerate(ARM_ORDER):
        for ci, cmp in enumerate(COMPARE_ORDER):
            st = arm_sep_head if (ai > 0 and ci == 0) else center
            txt = _header_row5_text(_get_hdr(hdr_df, arm, cmp), cmp)
            html.append(f"<th {st}>{txt}</th>")
    html.append("</tr>")

    html.append("</thead>")
    html.append("<tbody>")
    for parent, sub in row_order:
        label = parent if sub is None else f"{parent} [{sub}]"
        html.append("<tr>")
        html.append(f"<td>{label}</td>")
        for ai, arm in enumerate(ARM_ORDER):
            for ci, cmp in enumerate(COMPARE_ORDER):
                mask = ((long_df["modality_parent"] == parent) &
                        (long_df["arm"] == arm) &
                        (long_df["comparison"] == cmp))
                if sub is None:
                    mask = mask & long_df["modality_sub"].isna()
                else:
                    mask = mask & (long_df["modality_sub"] == sub)
                r = long_df[mask]
                td_style = cell_arm_sep if (ai > 0 and ci == 0) else cell_center
                if len(r) == 0:
                    html.append(f"<td {td_style}>—</td>")
                else:
                    txt = _cell_text(r.iloc[0]).replace("\n", "<br>")
                    html.append(f"<td {td_style}>{txt}</td>")
        html.append("</tr>")
    html.append("</tbody>")
    html.append("</table>")
    html.append("")
    html.append(
        "**Legend** — `d`=Cohen's d (scalar); `D²`=Mahalanobis (Hotelling); "
        "`R²`=PERMANOVA effect; `T̄²`=mean Hotelling T² across 8 emotion "
        "methods (Fisher-combined p). Significance after BH-FDR within each "
        "cell (14 modalities): `***` q<0.001, `**` q<0.01, `*` q<0.05, `.` "
        "q<0.10. Header rows 3–5 report per-group n (raw visits / unique "
        "subjects), Age mean±SD per group + Welch t p-value, and for AD hi-lo "
        "cells also MMSE / CASI / Global_CDR mean±SD + p (global CDR = 5-level "
        "0/0.5/1/2/3 clinical staging; CDR-SB sum-of-boxes would give finer "
        "resolution but is omitted here for clinical interpretability)."
    )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"Saved {out_md}")


def main():
    long_df = pd.read_csv(LONG_CSV)
    if HEADER_STATS_CSV.exists():
        hdr_df = pd.read_csv(HEADER_STATS_CSV)
    else:
        print(f"WARNING: {HEADER_STATS_CSV} not found — header rows 3-5 will show '—'")
        hdr_df = pd.DataFrame(columns=["arm", "comparison"])

    # Aggregate render at variant root.
    render_grid(long_df, hdr_df, DEEP / "stat_grid.png",
                 DEEP / "stat_grid_markdown.md")

    # Per-direction renders into subfolders. Filter ROW_ORDER + long_df by
    # direction; skip if no rows survive (e.g. subset run with --modalities
    # restricted to one direction).
    for direction in DIRECTIONS:
        d_dir = DEEP / direction
        sub_long_csv = d_dir / "stat_grid_long.csv"
        if not sub_long_csv.exists():
            continue
        sub_rows = [(p, s) for (p, s) in ROW_ORDER
                    if DIRECTION_MAP.get(p) == direction]
        if not sub_rows:
            continue
        sub_long = pd.read_csv(sub_long_csv)
        render_grid(sub_long, hdr_df,
                     d_dir / "stat_grid.png",
                     d_dir / "stat_grid_markdown.md",
                     row_order=sub_rows)


if __name__ == "__main__":
    main()
