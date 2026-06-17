"""Standalone view of the raw ConvONet completion result for job015."""
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HYPO = Path("/rds/general/user/qp23/home/Hypo3D")
COMP = HYPO / "repo/scannet/processed/scene0000_00/completion/scene0000_00_job015"
OUT  = HYPO / "figs"

def load_ply(path):
    data = path.read_bytes()
    hi = data.find(b"end_header"); he = hi+len("end_header")+1
    if data[he-1] != ord('\n'): he += 1
    hdr = data[:he].decode("ascii","ignore")
    n = int(re.search(r"element vertex (\d+)", hdr).group(1))
    binary = "binary_little_endian" in hdr
    props=[]; in_v=False
    for line in hdr.splitlines():
        line=line.strip()
        if line.startswith("element vertex"): in_v=True
        elif line.startswith("element"): in_v=False
        elif in_v and line.startswith("property") and "list" not in line:
            p=line.split()
            if len(p)>=3: props.append((p[1],p[2]))
    TM={"float":"f4","float32":"f4","uchar":"u1","uint8":"u1","int":"i4","uint":"u4","double":"f8"}
    if binary:
        dt=np.dtype([(nm,TM.get(tp,"f4")) for tp,nm in props])
        arr=np.frombuffer(data[he:],dtype=dt,count=n)
        xyz=np.column_stack([arr["x"],arr["y"],arr["z"]]).astype(np.float32)
        rgb=np.column_stack([arr["red"],arr["green"],arr["blue"]]).astype(np.uint8) \
            if "red" in arr.dtype.names else None
    else:
        rows=[list(map(float,l.split())) for l in data[he:].decode("ascii","ignore").splitlines()[:n] if l.strip()]
        a=np.array(rows); xyz=a[:,:3].astype(np.float32)
        rgb=a[:,3:6].astype(np.uint8) if a.shape[1]>=6 else None
    return xyz, rgb

xyz, rgb = load_ply(COMP/"convonet_completed_mesh_colored.ply")
print(f"ConvONet mesh: {xyz.shape[0]} vertices")

plt.rcParams.update({"font.family":"serif","figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight"})
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor="white")
for ax, proj, (xl,yl) in zip(axes, [(0,2),(0,1)], [("X (m)","Z (m)"),("X (m)","Y (m)")]):
    c = rgb.astype(np.float32)/255 if rgb is not None else "#888"
    ax.scatter(xyz[:,proj[0]], xyz[:,proj[1]], s=1.5, c=c, alpha=0.7, linewidths=0, rasterized=True)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect("equal")
    ax.set_facecolor("white"); ax.grid(True,color="#eee",lw=0.4)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axes[0].set_title("Top-down (X–Z)", fontweight="bold")
axes[1].set_title("Front (X–Y)", fontweight="bold")
fig.suptitle(f"Raw ConvONet completion — job015 cabinet  ({xyz.shape[0]:,} vertices, predicted RGB)",
             fontsize=12, fontweight="bold", y=1.0)
for ext in ("png","pdf"):
    plt.savefig(OUT/f"fig_convonet_result.{ext}", facecolor="white")
    print(f"Saved: {OUT}/fig_convonet_result.{ext}")
plt.close()
