"""
Pipeline diagram — single row, compact, matching reference proportions.
Output: figs/fig_pipeline_diagram.png
"""
from __future__ import annotations
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw, ImageFont

THUMBS = Path("/rds/general/user/qp23/home/Hypo3D/figs/pipeline_thumbs")
FIGS   = Path("/rds/general/user/qp23/home/Hypo3D/figs")

STAGES = [
    dict(
        number="1", title="Language Parser",
        technique="GPT-4o, few-shot\nprompted with 5\nin-context examples",
        input_="Free-text natural\nlanguage instruction",
        output="Structured JSON —\nop, target, anchor,\nrelation",
        thumb="stage1_original_scene.png",
        caption='"Place the chair to the right of the desk"',
        border=(141,110,58), move_only=False,
    ),
    dict(
        number="2", title="Instance Lookup",
        technique="ScanNet ground-truth\nsemantic annotations\nno learned model",
        input_="Target label +\nanchor label +\nscene annotation",
        output="Instance cloud I_t\nanchor cloud I_a\nbounding box",
        thumb="stage2_instance_lookup.png",
        caption="chair (blue) + desk (amber) identified",
        border=(0,105,92), move_only=False,
    ),
    dict(
        number="3", title="Object Excision",
        technique="Instance mask removal\n(primary) · AABB\nexcision (fallback)",
        input_="Full scene S +\ninstance mask\nof target I_t",
        output="Background cloud B_t\nwith exposed void",
        thumb="stage3_object_removal.png",
        caption="void where chair stood",
        border=(183,28,28), move_only=False,
    ),
    dict(
        number="4", title="Background Reconstruction",
        technique="NN floor fill ·\nstructural surface fill\n(RANSAC) · ConvONet",
        input_="B_t + void region\nbounding box",
        output="Reconstructed\nbackground B̂_t —\nno visible void",
        thumb="stage4_reconstruction.png",
        caption="floor and wall restored\n(green = filled points)",
        border=(106,27,154), move_only=False,
    ),
    dict(
        number="5a", title="Spatial Relation Resolver",
        technique="Rule-based geometry\n8-direction canonical\nvocabulary",
        input_="Anchor centroid c_a\n+ spatial relation r",
        output="Target centre ĉ_t\ntranslation vector Δ",
        thumb="stage5a_spatial_resolver.png",
        caption="RIGHT_OF resolved ·\nnew position (blue dot)",
        border=(230,81,0), move_only=True,
    ),
    dict(
        number="5b", title="Completion & Placement",
        technique="PBIC (primary) · PointR\n/ PCD-Dreamer (alt.)\ntrained on ShapeNet55",
        input_="Partial instance\ncloud I_t",
        output="Completed and placed\ninstance Ĉ_t at ĉ_t",
        thumb="stage5b_final_scene.png",
        caption="chair completed and placed\nright of desk",
        border=(191,54,12), move_only=True,
    ),
]

# ── Dimensions ────────────────────────────────────────────────────────────────
BOX_W   = 355
BOX_H   = 540
HDR_H   = 68
GAP     = 14
ARR_W   = 38
PAD_X   = 44
PAD_Y   = 100     # top padding (title area)
BOT_PAD = 56      # bottom padding (REMOVE bracket)
N       = len(STAGES)

TOTAL_W = PAD_X*2 + N*BOX_W + (N-1)*(GAP+ARR_W)
TOTAL_H = PAD_Y + BOX_H + BOT_PAD

canvas = Image.new("RGB", (TOTAL_W, TOTAL_H), (255,255,255))
draw   = ImageDraw.Draw(canvas)

# ── Fonts ─────────────────────────────────────────────────────────────────────
def font(size, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

F_MAIN_TITLE = font(36, bold=True)
F_SUBTITLE   = font(17)
F_HDR        = font(17, bold=True)
F_LABEL      = font(14, bold=True)
F_BODY       = font(14)
F_BADGE      = font(11, bold=True)
F_CAP        = font(13)
F_BRACKET    = font(13)

# ── Title ─────────────────────────────────────────────────────────────────────
draw.text((TOTAL_W//2, 14),
          "Language-driven 3D scene editing pipeline",
          font=F_MAIN_TITLE, fill=(20,20,20), anchor="mt")
draw.text((TOTAL_W//2, 54),
          'Illustrated with: scene0039_00 · job180 · "Place the chair to the right of the desk"',
          font=F_SUBTITLE, fill=(120,120,120), anchor="mt")

# ── Helpers ───────────────────────────────────────────────────────────────────
def rr(x0, y0, x1, y1, r, fill, outline, lw=2):
    draw.rounded_rectangle([x0,y0,x1,y1], radius=r,
                           fill=fill, outline=outline, width=lw)

def text_lines(x, y, lines, f, fill, line_gap=3):
    for line in lines:
        bb = draw.textbbox((0,0), line, font=f)
        draw.text((x, y), line, font=f, fill=fill)
        y += (bb[3]-bb[1]) + line_gap
    return y

# ── Boxes ─────────────────────────────────────────────────────────────────────
box_xs = []
for i, s in enumerate(STAGES):
    bx = PAD_X + i*(BOX_W + GAP + ARR_W)
    by = PAD_Y
    box_xs.append(bx)
    border = s["border"]

    # Outer box — white fill, coloured border
    rr(bx, by, bx+BOX_W, by+BOX_H, 12, (255,255,255), border, lw=2)

    # Coloured header strip
    rr(bx, by, bx+BOX_W, by+HDR_H, 12, border, border, lw=0)
    draw.rectangle([bx+2, by+HDR_H//2, bx+BOX_W-2, by+HDR_H], fill=border)

    # Number + title in header
    num_bb = draw.textbbox((0,0), s["number"], font=F_HDR)
    num_w  = num_bb[2]-num_bb[0]
    draw.text((bx+10, by+HDR_H//2-num_bb[3]//2),
              s["number"], font=F_HDR, fill=(255,255,255))
    # vertical line separator
    sep_x = bx+10+num_w+7
    draw.line([(sep_x, by+8),(sep_x, by+HDR_H-8)],
              fill=(255,255,255,120), width=1)
    # title — wrap to fit
    title_x = sep_x + 8
    title_max_w = BOX_W - (sep_x-bx) - 14
    title_lines = textwrap.wrap(s["title"], width=max(1,title_max_w//7))
    lh = 17
    ty0 = by + HDR_H//2 - len(title_lines)*lh//2
    for tl in title_lines:
        draw.text((title_x, ty0), tl, font=F_HDR, fill=(255,255,255))
        ty0 += lh

    # MOVE only badge
    if s["move_only"]:
        bw, bh = 58, 16
        draw.rounded_rectangle(
            [bx+BOX_W-bw-5, by+HDR_H+5, bx+BOX_W-5, by+HDR_H+5+bh],
            radius=4, fill=(255,255,255), outline=border, width=1)
        draw.text((bx+BOX_W-bw//2-5, by+HDR_H+5+bh//2),
                  "MOVE only", font=F_BADGE, fill=border, anchor="mm")

    # Text section
    tx  = bx + 10
    ty  = by + HDR_H + (22 if s["move_only"] else 10)
    col_lbl = tuple(max(0,c-40) for c in border)

    def block(label, body, y):
        bb = draw.textbbox((0,0), label, font=F_LABEL)
        draw.text((tx, y), label, font=F_LABEL, fill=col_lbl)
        y += (bb[3]-bb[1]) + 2
        y = text_lines(tx+2, y, body.split("\n"), F_BODY, (55,55,55), line_gap=2)
        return y + 5

    ty = block("TECHNIQUE:", s["technique"], ty)
    ty = block("INPUT:", s["input_"], ty)
    ty = block("OUTPUT:", s["output"], ty)

    # Divider
    ty += 4
    draw.line([(bx+8, ty),(bx+BOX_W-8, ty)],
              fill=(*border, 80), width=1)
    ty += 6

    # Thumbnail
    tp = THUMBS / s["thumb"]
    if tp.exists():
        img = Image.open(tp).convert("RGB")
        cap_h    = 28
        avail_h  = (by+BOX_H-6) - ty - cap_h
        avail_w  = BOX_W - 16
        scale    = min(avail_w/img.width, avail_h/img.height)
        tw_px    = int(img.width*scale)
        th_px    = int(img.height*scale)
        resized  = img.resize((tw_px, th_px), Image.LANCZOS)
        tx_img   = bx + (BOX_W-tw_px)//2
        canvas.paste(resized, (tx_img, ty))

        cap_y = ty + th_px + 3
        for line in s["caption"].split("\n"):
            draw.text((bx+BOX_W//2, cap_y), line,
                      font=F_CAP, fill=(100,100,100), anchor="mt")
            bb = draw.textbbox((0,0), line, font=F_CAP)
            cap_y += (bb[3]-bb[1]) + 2

# ── Arrows ────────────────────────────────────────────────────────────────────
def arrow(x0, y, x1, dashed=False, col=(50,50,50)):
    if dashed:
        s, g = 8, 8
        x = x0
        while x < x1-12:
            draw.line([(x,y),(min(x+s,x1-12),y)], fill=col, width=2)
            x += s+g
    else:
        draw.line([(x0,y),(x1-12,y)], fill=col, width=2)
    draw.polygon([(x1,y),(x1-12,y-6),(x1-12,y+6)], fill=col)

for i in range(N-1):
    bx   = box_xs[i]
    mid_y = PAD_Y + BOX_H//2
    x0   = bx + BOX_W + 3
    x1   = bx + BOX_W + GAP + ARR_W - 3
    arrow(x0, mid_y, x1, dashed=(i==3))

# ── REMOVE bracket ────────────────────────────────────────────────────────────
r_x0 = box_xs[0]
r_x1 = box_xs[3] + BOX_W
br_y = PAD_Y + BOX_H + 12
tk   = 7
for (sx,sy),(ex,ey) in [
    ((r_x0,br_y-tk),(r_x0,br_y)),
    ((r_x0,br_y),(r_x1,br_y)),
    ((r_x1,br_y),(r_x1,br_y-tk))]:
    seg=8
    if sx==ex:
        y=min(sy,ey)
        while y<max(sy,ey):
            draw.line([(sx,y),(ex,min(y+seg,max(sy,ey)))], fill=(150,150,150), width=1)
            y+=seg*2
    else:
        x=min(sx,ex)
        while x<max(sx,ex):
            draw.line([(x,sy),(min(x+seg,max(sx,ex)),ey)], fill=(150,150,150), width=1)
            x+=seg*2
draw.text(((r_x0+r_x1)//2, br_y+5),
          "REMOVE only: pipeline ends after Stage 4",
          font=F_BRACKET, fill=(150,150,150), anchor="mt")

# ── MOVE bracket ──────────────────────────────────────────────────────────────
m_x0 = box_xs[4]
m_x1 = box_xs[5] + BOX_W
bm_y = PAD_Y - 14
tk2  = 7
for (sx,sy),(ex,ey) in [
    ((m_x0,bm_y+tk2),(m_x0,bm_y)),
    ((m_x0,bm_y),(m_x1,bm_y)),
    ((m_x1,bm_y),(m_x1,bm_y+tk2))]:
    seg=8
    if sx==ex:
        y=min(sy,ey)
        while y<max(sy,ey):
            draw.line([(sx,y),(ex,min(y+seg,max(sy,ey)))], fill=(230,81,0), width=1)
            y+=seg*2
    else:
        x=min(sx,ex)
        while x<max(sx,ex):
            draw.line([(x,sy),(min(x+seg,max(sx,ex)),ey)], fill=(230,81,0), width=1)
            x+=seg*2
draw.text(((m_x0+m_x1)//2, bm_y-5),
          "MOVE only", font=F_BRACKET, fill=(230,81,0), anchor="mb")

# ── Save ──────────────────────────────────────────────────────────────────────
out = FIGS / "fig_pipeline_diagram.png"
canvas.save(out, dpi=(150,150))
print(f"Saved {out}  {canvas.width}×{canvas.height} px")
