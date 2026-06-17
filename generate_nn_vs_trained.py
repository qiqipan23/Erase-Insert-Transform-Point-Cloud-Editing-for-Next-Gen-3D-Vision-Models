"""Compare NN floor fill vs trained structural-surface UNet output — job015 cabinet."""
import json, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

HYPO=Path("/rds/general/user/qp23/home/Hypo3D")
COMP=HYPO/"repo/scannet/processed/scene0000_00/completion/scene0000_00_job015"
EDIT=HYPO/"repo/scannet/processed/scene0000_00/edits"
OUT=HYPO/"figs"
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.titlesize":11,
 "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight"})
VOID_COL="#ff7f0e"

def load_ply(path):
    data=Path(path).read_bytes();hi=data.find(b"end_header");nl=data[hi+len("end_header")]
    he=hi+len("end_header")+(2 if nl==ord('\r') else 1)
    hdr=data[:he].decode("ascii","ignore");n=int(re.search(r"element vertex (\d+)",hdr).group(1))
    bin_="binary_little_endian" in hdr;props=[];in_v=False
    for line in hdr.splitlines():
        line=line.strip()
        if line.startswith("element vertex"):in_v=True
        elif line.startswith("element"):in_v=False
        elif in_v and line.startswith("property") and "list" not in line:
            p=line.split()
            if len(p)>=3:props.append((p[1],p[2]))
    TM={"float":"f4","float32":"f4","double":"f8","uchar":"u1","uint8":"u1","int":"i4","uint":"u4"}
    if bin_:
        dt=np.dtype([(nm,TM.get(tp,"f4")) for tp,nm in props])
        arr=np.frombuffer(data[he:],dtype=dt,count=n)
    else:
        rows=[list(map(float,l.split())) for l in data[he:].decode("ascii","ignore").splitlines()[:n] if l.strip()]
        if not rows:return np.empty((0,3),np.float32),None
        arr=np.rec.fromarrays(np.array(rows).T,names=[nm for _,nm in props])
    xyz=np.column_stack([arr["x"],arr["y"],arr["z"]]).astype(np.float32)
    rgb=None
    for names in (("red","green","blue"),("r","g","b")):
        if all(nm in arr.dtype.names for nm in names):
            r_=np.column_stack([arr[nm] for nm in names]).astype(np.float64)
            rgb=(r_/r_.max()*255 if r_.max()>1.5 else r_*255).astype(np.uint8);break
    return xyz,rgb

def crop(xyz,rgb,c,s,pad=2.4):
    if xyz is None or xyz.shape[0]==0:return xyz,rgb
    mn=c-s/2*pad;mx=c+s/2*pad;m=np.all((xyz>=mn)&(xyz<=mx),axis=1)
    return xyz[m],(rgb[m] if rgb is not None else None)
def sc(ax,xyz,rgb=None,color=None,alpha=0.7,proj=(0,2),ms=None):
    if xyz is None or xyz.shape[0]==0:return
    n=xyz.shape[0];s=ms if ms else (4 if n<20000 else 1.0)
    x,y=xyz[:,proj[0]],xyz[:,proj[1]]
    if color is not None:ax.scatter(x,y,s=s,c=color,alpha=alpha,linewidths=0,rasterized=True)
    elif rgb is not None:ax.scatter(x,y,s=s,c=rgb.astype(np.float32)/255,alpha=alpha,linewidths=0,rasterized=True)
def box(ax,c,s,proj,col=VOID_COL,lw=1.5,ls="--"):
    i,j=proj;ax.add_patch(plt.Rectangle((c[i]-s[i]/2,c[j]-s[j]/2),s[i],s[j],fill=False,edgecolor=col,linewidth=lw,linestyle=ls,zorder=5))
def style(ax,xl,yl):
    ax.set_facecolor("white");ax.set_xlabel(xl,fontsize=8,color="#555");ax.set_ylabel(yl,fontsize=8,color="#555")
    ax.tick_params(labelsize=7,color="#aaa",labelcolor="#666");ax.grid(True,color="#eee",lw=0.5,zorder=0)
    for sp in("top","right"):ax.spines[sp].set_visible(False)
    for sp in("bottom","left"):ax.spines[sp].set_color("#ccc")

mani=json.loads((EDIT/"job015.json").read_text())
center=np.array(mani["remove_center"]);size=np.array(mani["remove_size"])
removed_xyz,removed_rgb=load_ply(HYPO/mani["canonical_edit_file"])
nn_xyz,nn_rgb=load_ply(COMP/"nn_floor_fill_completed_points.ply")
st_xyz,st_rgb=load_ply(COMP/"structural_surface_completed_points.ply")
print(f"void={removed_xyz.shape[0]} nn={nn_xyz.shape[0]} struct={st_xyz.shape[0]}")

PAD=2.4
rem_c,rem_c_rgb=crop(removed_xyz,removed_rgb,center,size,PAD)
nn_c,nn_c_rgb=crop(nn_xyz,nn_rgb,center,size,PAD)
st_c,st_c_rgb=crop(st_xyz,st_rgb,center,size,PAD)

PADL=0.6;cx,cy,cz=center;sx,sy,sz=size
xlim=(cx-sx/2-PADL,cx+sx/2+PADL);ylim=(cy-sy/2-PADL,cy+sy/2+PADL);zlim=(cz-sz/2-PADL,cz+sz/2+PADL)
PROJS=[(0,2),(0,1)];RL=[("X (m)","Z (m)"),("X (m)","Y (m)")];LIMS=[(xlim,zlim),(xlim,ylim)]
titles=["(a) After removal\n(void exposed)",
        "(b) NN floor fill\n(nearest-neighbour, 6903 pts)",
        "(c) Trained structural UNet\n(learned surfaces, 3112 pts)"]
NN_COL="#9C27B0";ST_COL="#2ca02c"

fig=plt.figure(figsize=(12,7.5),facecolor="white")
gs=GridSpec(2,3,figure=fig,hspace=0.38,wspace=0.22,top=0.88,bottom=0.09,left=0.07,right=0.97)
for r,proj in enumerate(PROJS):
    lx,ly=LIMS[r]
    for c in range(3):
        ax=fig.add_subplot(gs[r,c]);style(ax,RL[r][0],RL[r][1])
        if r==0:ax.set_title(titles[c],fontsize=10,pad=6,color="#222")
        if c==0:
            sc(ax,rem_c,rem_c_rgb,proj=proj,alpha=0.55);box(ax,center,size,proj,VOID_COL,1.6)
        elif c==1:
            sc(ax,rem_c,rem_c_rgb,proj=proj,alpha=0.30)
            sc(ax,nn_c,color=NN_COL,proj=proj,alpha=0.8,ms=4);box(ax,center,size,proj,"#888",1.2)
        else:
            sc(ax,rem_c,rem_c_rgb,proj=proj,alpha=0.30)
            sc(ax,st_c,color=ST_COL,proj=proj,alpha=0.85,ms=4);box(ax,center,size,proj,"#888",1.2)
        ax.set_xlim(*lx);ax.set_ylim(*ly);ax.set_aspect("equal")
for r,txt in enumerate(["Top-down view","Front view"]):
    p=fig.get_axes()[r*3].get_position()
    fig.text(0.005,p.y0+p.height/2,txt,va="center",ha="left",fontsize=9,color="#444",rotation=90,fontweight="bold")
handles=[mpatches.Patch(color="#aec7e8",label="Scene geometry (RGB)"),
         mpatches.Patch(color=NN_COL,label="NN floor fill (non-learned)"),
         mpatches.Patch(color=ST_COL,label="Trained structural UNet"),
         mpatches.Patch(facecolor="none",edgecolor=VOID_COL,linestyle="--",linewidth=1.5,label="Removal bounding box")]
fig.legend(handles=handles,loc="lower center",ncol=4,framealpha=0.9,edgecolor="#ccc",fontsize=8.5,bbox_to_anchor=(0.5,0.01))
fig.suptitle("Background Fill: Nearest-Neighbour vs Trained Structural Model — scene0000\\_00 / job015 (cabinet)",
             fontsize=12,color="#111",y=0.95,fontweight="bold")
for ext in ("png","pdf"):
    plt.savefig(OUT/f"fig_nn_vs_trained.{ext}",facecolor="white");print("Saved",OUT/f"fig_nn_vs_trained.{ext}")
plt.close()
