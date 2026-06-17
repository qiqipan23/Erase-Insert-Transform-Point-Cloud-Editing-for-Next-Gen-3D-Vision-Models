"""
Architecture diagram for StructuralSurfaceUNet — clean horizontal U-Net layout.
base_channels=24 (wallfix training run).
Output: figs/fig_structural_unet_arch.png/.pdf
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT = "/rds/general/user/qp23/home/Hypo3D/figs/fig_structural_unet_arch"

C_ENC   = "#1565C0"
C_BOTT  = "#6A1B9A"
C_DEC   = "#2E7D32"
C_HEAD  = "#B71C1C"
C_POOL  = "#37474F"
C_SKIP  = "#E65100"
C_IN    = "#546E7A"

fig, ax = plt.subplots(figsize=(18, 8), facecolor="white")
ax.set_xlim(-0.5, 18.5); ax.set_ylim(-1.0, 8.0)
ax.axis("off")

# ── Helpers ────────────────────────────────────────────────────────────────────
def rect(x, y, w, h, color, label, sub1="", sub2="", fontsize=8):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                       facecolor=color, edgecolor="white", linewidth=2, zorder=3)
    ax.add_patch(p)
    yo = y + h/2
    dy = 0.27 if (sub1 or sub2) else 0
    ax.text(x+w/2, yo+dy, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold", zorder=4)
    if sub1:
        ax.text(x+w/2, yo, sub1, ha="center", va="center",
                fontsize=6, color="white", alpha=0.9, zorder=4)
    if sub2:
        ax.text(x+w/2, yo-0.27, sub2, ha="center", va="center",
                fontsize=6, color="white", alpha=0.9, zorder=4)

def op(x, y, label, color=C_POOL, w=1.1, h=0.45):
    p = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.06",
                       facecolor=color, edgecolor="white", linewidth=1.2, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=6.5, color="white", fontweight="bold", zorder=4)

def arr(x0, y0, x1, y1, color="#555", lw=1.6):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->, head_width=0.2, head_length=0.12",
                                color=color, lw=lw),
                zorder=5)

def skip(ax, x0, ybase, x1, arc_h, label):
    """Curved skip connection arc above the main flow."""
    import matplotlib.patches as mpa
    style = f"arc3,rad=-0.35"
    ax.annotate("",
        xy=(x1, ybase), xytext=(x0, ybase),
        arrowprops=dict(
            arrowstyle="->, head_width=0.25, head_length=0.15",
            color=C_SKIP, lw=2.0, linestyle="dashed",
            connectionstyle=f"arc3,rad=-{arc_h:.2f}"),
        zorder=5)
    ax.text((x0+x1)/2, ybase + arc_h*3.5 + 0.15, label,
            ha="center", va="bottom", fontsize=7, color=C_SKIP,
            style="italic", zorder=6)

# ── Main flow Y positions ─────────────────────────────────────────────────────
# Encoder row (top), decoder row (bottom), connected by vertical lines at bottleneck
EY = 3.5    # encoder & decoder y (same row: horizontal U-Net)
BH = 1.5    # block height
BW = 2.4    # block width
SW = 0.9    # small op width

# X positions for each stage
XS = {
    "in":    0.0,
    "e1":    0.9,
    "p1":    3.6,
    "e2":    4.5,
    "p2":    7.2,
    "bot":   8.1,
    "u2":    10.8,
    "d2":    11.5,
    "u1":    14.2,
    "d1":    14.9,
    "h":     17.6,
}

# ── Input ─────────────────────────────────────────────────────────────────────
rect(XS["in"], EY, 0.8, BH, C_IN, "Input", "6×D", "×H×W", fontsize=7)
ax.text(XS["in"]+0.4, EY-0.55, "6-ch voxel grid\n(TSDF, occ, hole, R, G, B)",
        ha="center", va="top", fontsize=6.5, color="#555")
arr(XS["in"]+0.8, EY+BH/2, XS["e1"], EY+BH/2)

# ── Encoder Block 1 ───────────────────────────────────────────────────────────
rect(XS["e1"], EY, BW, BH, C_ENC, "Encoder Block 1",
     "Conv3(6→24)·BN·ReLU", "Conv3(24→24)·BN·ReLU")
ax.text(XS["e1"]+BW/2, EY+BH+0.12, "24 ch  |  D×H×W",
        ha="center", va="bottom", fontsize=6.5, color="#333")

# pool 1
arr(XS["e1"]+BW, EY+BH/2, XS["p1"]-SW/2, EY+BH/2)
op(XS["p1"], EY+BH/2, "MaxPool3d\n÷2")
arr(XS["p1"]+SW/2, EY+BH/2, XS["e2"], EY+BH/2)

# ── Encoder Block 2 ───────────────────────────────────────────────────────────
rect(XS["e2"], EY, BW, BH, C_ENC, "Encoder Block 2",
     "Conv3(24→48)·BN·ReLU", "Conv3(48→48)·BN·ReLU")
ax.text(XS["e2"]+BW/2, EY+BH+0.12, "48 ch  |  D/2×H/2×W/2",
        ha="center", va="bottom", fontsize=6.5, color="#333")

# pool 2
arr(XS["e2"]+BW, EY+BH/2, XS["p2"]-SW/2, EY+BH/2)
op(XS["p2"], EY+BH/2, "MaxPool3d\n÷2")
arr(XS["p2"]+SW/2, EY+BH/2, XS["bot"], EY+BH/2)

# ── Bottleneck ────────────────────────────────────────────────────────────────
rect(XS["bot"], EY, BW, BH, C_BOTT, "Bottleneck",
     "Conv3(48→96)·BN·ReLU", "Conv3(96→96)·BN·ReLU")
ax.text(XS["bot"]+BW/2, EY+BH+0.12, "96 ch  |  D/4×H/4×W/4",
        ha="center", va="bottom", fontsize=6.5, color="#333")

# up 2
arr(XS["bot"]+BW, EY+BH/2, XS["u2"]-SW/2, EY+BH/2)
op(XS["u2"], EY+BH/2, "ConvT3d\n×2→48ch", color="#00695C")
arr(XS["u2"]+SW/2, EY+BH/2, XS["d2"], EY+BH/2)
# cat label
ax.text(XS["d2"]-0.05, EY+BH+0.12, "cat[up, s2]→96ch",
        ha="left", va="bottom", fontsize=6.5, color=C_SKIP, style="italic")

# ── Decoder Block 2 ───────────────────────────────────────────────────────────
rect(XS["d2"], EY, BW, BH, C_DEC, "Decoder Block 2",
     "Conv3(96→48)·BN·ReLU", "Conv3(48→48)·BN·ReLU")
ax.text(XS["d2"]+BW/2, EY-0.55, "48 ch  |  D/2×H/2×W/2",
        ha="center", va="top", fontsize=6.5, color="#333")

# up 1
arr(XS["d2"]+BW, EY+BH/2, XS["u1"]-SW/2, EY+BH/2)
op(XS["u1"], EY+BH/2, "ConvT3d\n×2→24ch", color="#00695C")
arr(XS["u1"]+SW/2, EY+BH/2, XS["d1"], EY+BH/2)
ax.text(XS["d1"]-0.05, EY+BH+0.12, "cat[up, s1]→48ch",
        ha="left", va="bottom", fontsize=6.5, color=C_SKIP, style="italic")

# ── Decoder Block 1 ───────────────────────────────────────────────────────────
rect(XS["d1"], EY, BW, BH, C_DEC, "Decoder Block 1",
     "Conv3(48→24)·BN·ReLU", "Conv3(24→24)·BN·ReLU")
ax.text(XS["d1"]+BW/2, EY-0.55, "24 ch  |  D×H×W",
        ha="center", va="top", fontsize=6.5, color="#333")

# ── Output heads ──────────────────────────────────────────────────────────────
arr(XS["d1"]+BW, EY+BH*0.7, XS["h"], EY+BH*0.8+0.1)
arr(XS["d1"]+BW, EY+BH*0.3, XS["h"], EY+BH*0.2-0.1)

rect(XS["h"], EY+BH*0.45, 0.9, 0.85, C_HEAD, "Mask", "Head", "1×1×1", fontsize=7)
rect(XS["h"], EY+BH*0.45-1.05, 0.9, 0.85, C_HEAD, "RGB", "Head", "1×1×1", fontsize=7)

ax.text(XS["h"]+0.9+0.15, EY+BH*0.45+0.42, "→ logits (1 ch)", fontsize=7.5,
        va="center", color="#333")
ax.text(XS["h"]+0.9+0.15, EY+BH*0.45-0.63, "→ RGB (3 ch) + sigmoid", fontsize=7.5,
        va="center", color="#333")

# ── Skip connections (arcs above) ────────────────────────────────────────────
# Skip 1: end of enc1 → start of dec1
x_s1_start = XS["e1"] + BW
x_s1_end   = XS["d1"]
skip(ax, x_s1_start, EY+BH, x_s1_end, 0.65, "skip s1  (24 ch, full resolution)")

# Skip 2: end of enc2 → start of dec2
x_s2_start = XS["e2"] + BW
x_s2_end   = XS["d2"]
skip(ax, x_s2_start, EY+BH, x_s2_end, 0.38, "skip s2  (48 ch, ½ resolution)")

# ── Loss box ──────────────────────────────────────────────────────────────────
lx, ly = 0.0, -0.85
p = FancyBboxPatch((lx, ly), 9.5, 0.75, boxstyle="round,pad=0.08",
                   facecolor="#F5F5F5", edgecolor="#BDBDBD", linewidth=1, zorder=2)
ax.add_patch(p)
ax.text(lx+0.18, ly+0.54, "Training & Validation Loss (same formula):",
        fontsize=8, fontweight="bold", color="#333", va="top")
ax.text(lx+0.18, ly+0.26,
        r"$\mathcal{L}$ = BCE(pos_weight=8)  +  0.5·Dice  +  0.5·$\ell_1$RGB (surface voxels only)  +  0.05·$\mathcal{L}_{support}$"
        "    |    Adam, lr=1e-4, 150 epochs, batch=2",
        fontsize=7.5, color="#444", va="top")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title("StructuralSurfaceUNet — 3-Level 3D U-Net Architecture  (base_channels = 24)",
             fontsize=13, fontweight="bold", color="#111", pad=12)

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=C_ENC,  label="Encoder ConvBlock  (Conv3d→BN→ReLU ×2)"),
    mpatches.Patch(color=C_BOTT, label="Bottleneck ConvBlock"),
    mpatches.Patch(color=C_DEC,  label="Decoder ConvBlock  (Conv3d→BN→ReLU ×2)"),
    mpatches.Patch(color=C_HEAD, label="Output head  (1×1×1 Conv3d)"),
    mpatches.Patch(color="#00695C", label="ConvTranspose3d (upsample ×2)"),
    mpatches.Patch(color=C_POOL, label="MaxPool3d (downsample ×2)"),
    mpatches.Patch(color=C_SKIP, label="Skip connection (concat along channel)"),
]
ax.legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.95,
          edgecolor="#ccc", ncol=2, bbox_to_anchor=(1.0, -0.02))

plt.tight_layout(pad=0.5)
for ext in ("png", "pdf"):
    plt.savefig(f"{OUT}.{ext}", dpi=200, facecolor="white", bbox_inches="tight")
    print(f"Saved {OUT}.{ext}")
plt.close()
