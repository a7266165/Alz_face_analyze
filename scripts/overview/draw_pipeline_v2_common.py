"""V2 drawing toolkit — follows the stricter spec in agents/drawer/prompt.md.

Differences vs the original draw_age_emb_pipeline_common:
  - indexed palette P[1..19] (bg = cluster fill, nd = node fill); the named
    aliases below map onto it so diagram code stays readable
  - content-driven sizing: node_w(text) grows width per 5 chars over 10;
    node_h(lines) grows height per line over 2
  - uniform GAP=0.5 / PADDING=0.75, linespacing=1.2
  - figures are written to workspace/overview
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(r"c:\Users\4080\Desktop\Alz_face_analyze\workspace\overview")

# ── constants (prompt.md §2) ──
FONT = "Microsoft JhengHei"
FS = 14
TEXT_COLOR = "#212121"
LINESPACING = 1.2
NODE_H = 1.15
NODE_W = 1.7
GAP = 0.5
SP = 0.4
PADDING = 0.75
NODE_LW = 1.4
CLUSTER_LW = 2.0
EC = "#404040"
LINE_EXTRA_H = 0.25
CHAR_EXTRA_W = 0.5
CHAR_THRESHOLD = 10
CHAR_STEP = 5

# ── palette (prompt.md §3) ──
P = {
    1:  dict(bg='#E8EFF5', nd='#B3CCE4'),   # 藍
    2:  dict(bg='#FDF3E5', nd='#F5D5A0'),   # 橘
    3:  dict(bg='#E4F0ED', nd='#A8D4C4'),   # 綠
    4:  dict(bg='#F5E8E4', nd='#E8BDB0'),   # 粉
    5:  dict(bg='#FBF5E0', nd='#F0D870'),   # 黃
    6:  dict(bg='#F0E6F0', nd='#D4B0D4'),   # 紫
    7:  dict(bg='#E0EEF0', nd='#8DC3CB'),   # 青
    8:  dict(bg='#E8E8FC', nd='#B4B4E8'),   # 薰衣草
    9:  dict(bg='#ECF0E4', nd='#C5D6A8'),   # 萊姆
    10: dict(bg='#FFF3E0', nd='#FFD180'),   # 琥珀
    11: dict(bg='#FCE8E8', nd='#E8B4B4'),   # 珊瑚
    12: dict(bg='#F8ECE4', nd='#E0C0A8'),   # 蜜桃
    13: dict(bg='#E8ECF8', nd='#B0B8E0'),   # 鋼藍
    14: dict(bg='#ECEEF8', nd='#C0C8E4'),   # 灰藍
    15: dict(bg='#F3E8F8', nd='#C8A8E0'),   # 堇
    16: dict(bg='#F0EDE0', nd='#D0C8A0'),   # 米
    17: dict(bg='#EDF0E8', nd='#C0D0A8'),   # 鼠尾草
    18: dict(bg='#F5F0E0', nd='#D8C890'),   # 駝
    19: dict(bg='#E0E0E0', nd='#C0C0C0'),   # 灰 (調暗用)
    # extension set — same pastel two-tone style, fills hues the original 19
    # under-cover (the base set is blue/green/tan-heavy), for diagrams that need
    # more mutually-distinct groups.
    20: dict(bg='#F6E2EF', nd='#D58AC4'),   # 梅紅 (magenta/plum)
    21: dict(bg='#DEF1E4', nd='#7FC698'),   # 翠綠 (emerald — truer green than 萊姆/鼠尾草)
    22: dict(bg='#FBE3E8', nd='#E08494'),   # 玫瑰 (raspberry/rose — deeper than 珊瑚)
    23: dict(bg='#E6E6FA', nd='#8C8FD8'),   # 靛藍 (indigo — deeper than 薰衣草/鋼藍)
    24: dict(bg='#E0EFFB', nd='#7FB8E8'),   # 天藍 (azure — brighter than 藍)
    25: dict(bg='#EFEED6', nd='#C2BE6E'),   # 橄欖 (olive — muted yellow-green)
    26: dict(bg='#F8E6DE', nd='#DD9C7A'),   # 磚紅 (brick/terracotta — warmer than 蜜桃)
}

# ── named aliases onto the palette (same hex as the original diagram) ──
C1 = P[1]; C2 = P[2]; C3 = P[6]; C4 = P[9]; C6 = P[5]; CF = P[3]; CR = P[4]
C_FT = P[11]; C_PA = P[8]; C_PD = P[16]; C_SA = P[13]; C_ES = P[12]
C_ML = P[17]; C_MS = P[18]; C_AGE = P[10]; C_PRE = P[7]; C_AOUT = P[14]
C_ASY = P[15]; C_EMO = P[11]; G = P[19]

AGE_MODELS = ['MiVOLO', 'InsightFace', 'DeepFace', 'FairFace', 'OpenCV\nDNN']
NW_AM = 2.2; GAP_AM = 0.5
TOTAL_AM = 5 * NW_AM + 4 * GAP_AM
X_AGE = 3.0
NW_PRE = 2.2


def node_h(lines=2):
    """Box height: NODE_H for <=2 lines, +LINE_EXTRA_H per extra line."""
    return NODE_H if lines <= 2 else NODE_H + (lines - 2) * LINE_EXTRA_H


def node_w(text=""):
    """Box width: NODE_W for <=10 chars (longest line), +0.5 per extra 5 chars."""
    longest = max((len(s) for s in text.split("\n")), default=0)
    if longest <= CHAR_THRESHOLD:
        return NODE_W
    extra = (longest - CHAR_THRESHOLD + CHAR_STEP - 1) // CHAR_STEP
    return NODE_W + extra * CHAR_EXTRA_W


def node(ax, cx, cy, w, h, text, fc, fs=FS):
    """Filled box with a centred label (zorder 3 box / 4 text)."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="square,pad=0", facecolor=fc, edgecolor=EC,
        linewidth=NODE_LW, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, fontfamily=FONT, color=TEXT_COLOR,
            linespacing=LINESPACING, zorder=4)


def cluster(ax, cx, cy, w, h, fc):
    """Background wrapper box, drawn behind nodes (zorder 0)."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="square,pad=0", facecolor=fc, edgecolor=EC,
        linewidth=CLUSTER_LW, zorder=0))


def line(ax, x1, y1, x2, y2, color=EC, lw=NODE_LW, zorder=2):
    """Straight connector (zorder<0 routes it under cluster fills)."""
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw,
            solid_capstyle="round", zorder=zorder)


def rowx(cx, n, w, gap=GAP):
    """(x-positions, total-width) for n nodes of width w centred on cx."""
    tot = n * w + (n - 1) * gap
    xs = [cx - tot / 2 + w / 2 + i * (w + gap) for i in range(n)]
    return xs, tot


def lay(cx, n, nw, gap=GAP):
    """x-positions only (prompt.md §5)."""
    return rowx(cx, n, nw, gap)[0]


def cluster_w(n, nw, gap=GAP, pad=PADDING):
    """Cluster width = node total + gaps + padding."""
    return n * nw + (n - 1) * gap + pad


def nlines(text):
    """Number of text lines in a label."""
    return str(text).count("\n") + 1


def hgt(text):
    """Content-driven box height for a label (node_h by line count)."""
    return node_h(nlines(text))


def lay_var(cx, widths, gap=GAP):
    """(centres, total) for a row of *variable*-width nodes centred on cx."""
    total = sum(widths) + (len(widths) - 1) * gap
    xs, x = [], cx - total / 2
    for w in widths:
        xs.append(x + w / 2)
        x += w + gap
    return xs, total


def prow(ax, cx, y, labels, colors, gap=GAP):
    """Content-driven row: size each box by node_w(text)/node_h(lines), centre
    the row on cx, draw. `colors` is one nd-colour or a per-label list.
    Returns (centres, widths, total)."""
    labels = list(labels)
    widths = [node_w(l) for l in labels]
    xs, total = lay_var(cx, widths, gap)
    if not isinstance(colors, (list, tuple)):
        colors = [colors] * len(labels)
    for x, w, l, c in zip(xs, widths, labels, colors):
        node(ax, x, y, w, hgt(l), l, c)
    return xs, widths, total
