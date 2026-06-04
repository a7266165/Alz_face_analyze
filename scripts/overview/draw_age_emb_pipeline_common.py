"""Shared constants and helpers for age/embedding pipeline diagrams."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

__all__ = [
    'plt', 'FancyBboxPatch', 'Path',
    'OUT', 'FONT', 'FS', 'FS_HI', 'FS_SM', 'FS_XS', 'EC', 'NODE_H', 'SP',
    'C1', 'C2', 'C3', 'C4', 'C6', 'CF', 'CR',
    'C_FT', 'C_PA', 'C_PD', 'C_SA', 'C_ES', 'C_ML', 'C_MS',
    'C_AGE', 'C_PRE', 'C_EACS', 'C_AOUT', 'C_ASY', 'C_EMO', 'G',
    'AGE_MODELS', 'NW_AM', 'GAP_AM', 'TOTAL_AM',
    'FIG_W', 'X_LIM_LEFT', 'X_LIM_RIGHT', 'CX', 'X_AGE', 'CX_TOP',
    'PRE_LABELS', 'NW_PRE',
    '_box', 'node', 'cluster', 'line',
]

OUT = Path(r"c:\Users\4080\Desktop\Alz_face_analyze\workspace\overview")

FONT = 'Microsoft JhengHei'
FS = 14
FS_HI = 15
FS_SM = 12
FS_XS = 10
EC = '#404040'
NODE_H = 1.15
SP = 0.4

C1 = dict(bg='#E8EFF5', nd='#B3CCE4', hi='#8BB4D9')
C2 = dict(bg='#FDF3E5', nd='#F5D5A0')
C3 = dict(bg='#F0E6F0', nd='#D4B0D4')
C4 = dict(bg='#ECF0E4', nd='#C5D6A8')
C6 = dict(bg='#FBF5E0', nd='#F0D870')
CF = dict(bg='#E4F0ED', nd='#A8D4C4')
CR = dict(bg='#F5E8E4', nd='#E8BDB0')
C_FT = dict(bg='#FCE8E8', nd='#E8B4B4')
C_PA = dict(bg='#E8E8FC', nd='#B4B4E8')
C_PD = dict(bg='#F0EDE0', nd='#D0C8A0')
C_SA = dict(bg='#E8ECF8', nd='#B0B8E0')
C_ES = dict(bg='#F8ECE4', nd='#E0C0A8')
C_ML = dict(bg='#EDF0E8', nd='#C0D0A8')
C_MS = dict(bg='#F5F0E0', nd='#D8C890')
C_AGE = dict(bg='#FFF3E0', nd='#FFD180')
C_PRE = dict(bg='#E0EEF0', nd='#8DC3CB')
C_EACS = dict(bg='#FFFDE8', nd='#F0D870')
C_AOUT = dict(bg='#ECEEF8', nd='#C0C8E4')
C_ASY = dict(bg='#F3E8F8', nd='#C8A8E0')
C_EMO = dict(bg='#FCE8E8', nd='#E8B4B4')
G = dict(bg='#E0E0E0', nd='#C0C0C0')

AGE_MODELS = ['MiVOLO', 'InsightFace', 'DeepFace', 'FairFace', 'OpenCV\nDNN']
NW_AM = 2.2; GAP_AM = 0.5
TOTAL_AM = 5 * NW_AM + 4 * GAP_AM

FIG_W = 30.5
X_LIM_LEFT = -4.5
X_LIM_RIGHT = 28.1
CX = 16.5
X_AGE = 3.0
CX_TOP = 16.5

# Preprocessing: 3 stacked nodes + 2 side-by-side bg nodes
PRE_LABELS = ['Detect', 'Select', 'Align']
NW_PRE = 2.2


def _box(ax, cx, cy, w, h, fc, lw=1.4, zorder=1):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="square,pad=0", facecolor=fc, edgecolor=EC,
        linewidth=lw, zorder=zorder))


def node(ax, cx, cy, w, h, text, fc, fs=FS, fc_text='#212121'):
    _box(ax, cx, cy, w, h, fc, lw=1.4, zorder=3)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, fontfamily=FONT, color=fc_text,
            linespacing=1.35, zorder=4)


def cluster(ax, cx, cy, w, h, fc):
    _box(ax, cx, cy, w, h, fc, lw=2.0, zorder=0)


def line(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=EC, lw=1.4,
            solid_capstyle='round', zorder=2)
