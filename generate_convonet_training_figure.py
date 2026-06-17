"""ConvONet fine-tuning curve (latest retrain run, 739-example dataset)."""
import struct
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGDIR = Path("/rds/general/user/qp23/home/Hypo3D/repo/convolutional_occupancy_networks/out/hypo3d_scannet_remove_retrain/logs")
OUT = Path("/rds/general/user/qp23/home/Hypo3D/figs")

def rv(b,i):
    s=r=0
    while True:
        x=b[i];i+=1;r|=(x&0x7f)<<s
        if not(x&0x80):break
        s+=7
    return r,i
def pe(b):
    i=0;st=None;v=[]
    while i<len(b):
        k,i=rv(b,i);f=k>>3;w=k&7
        if f==2 and w==0:st,i=rv(b,i)
        elif f==5 and w==2:l,i=rv(b,i);v+=ps(b[i:i+l]);i+=l
        elif w==0:_,i=rv(b,i)
        elif w==1:i+=8
        elif w==2:l,i=rv(b,i);i+=l
        elif w==5:i+=4
        else:break
    return st,v
def ps(b):
    i=0;o=[]
    while i<len(b):
        k,i=rv(b,i);f=k>>3;w=k&7
        if f==1 and w==2:l,i=rv(b,i);o+=[pv(b[i:i+l])];i+=l
        elif w==0:_,i=rv(b,i)
        elif w==1:i+=8
        elif w==2:l,i=rv(b,i);i+=l
        elif w==5:i+=4
        else:break
    return [x for x in o if x]
def pv(b):
    i=0;t=None;s=None
    while i<len(b):
        k,i=rv(b,i);f=k>>3;w=k&7
        if f==1 and w==2:l,i=rv(b,i);t=b[i:i+l].decode('utf-8','ignore');i+=l
        elif f==2 and w==5:s=struct.unpack('<f',b[i:i+4])[0];i+=4
        elif w==0:_,i=rv(b,i)
        elif w==1:i+=8
        elif w==2:l,i=rv(b,i);i+=l
        elif w==5:i+=4
        else:break
    return (t,s) if t and s is not None else None
def rd(p):
    d=Path(p).read_bytes();i=0;e=[]
    while i<len(d):
        if i+12>len(d):break
        l=struct.unpack('<Q',d[i:i+8])[0];i+=12
        if i+l+4>len(d):break
        try:e.append(pe(d[i:i+l]))
        except:pass
        i+=l+4
    return e

loss=[];iou=[]
for ev in sorted(LOGDIR.glob("events*")):
    for st,vals in rd(ev):
        if st is None:continue
        for tg,vv in vals:
            if tg=='train/loss' and vv>0:loss.append((st,vv))
            elif tg=='val/iou':iou.append((st,vv))
loss=sorted(set(loss));iou=sorted(set(iou))
ls=np.array([s for s,_ in loss]); lv=np.array([v for _,v in loss])
iu_s=np.array([s for s,_ in iou]); iu_v=np.array([v for _,v in iou])

# smooth loss (moving median)
order=np.argsort(ls); ls=ls[order]; lv=lv[order]
def smooth(x,k=999):
    if len(x)<k: return x
    return np.convolve(x, np.ones(k)/k, mode='same')
lv_s = smooth(lv, 999)

plt.rcParams.update({"font.family":"serif","font.size":11,"figure.dpi":150,
    "savefig.dpi":300,"savefig.bbox":"tight","axes.spines.top":False})
fig, ax1 = plt.subplots(figsize=(9,5), facecolor="white")
ax1.plot(ls, lv, color="#1565C0", alpha=0.18, lw=0.5)
ax1.plot(ls, lv_s, color="#1565C0", lw=2.0, label="Training loss (smoothed)")
ax1.set_xlabel("Training iteration")
ax1.set_ylabel("Occupancy training loss", color="#1565C0")
ax1.tick_params(axis='y', labelcolor="#1565C0")
ax1.set_ylim(0, 350)
ax1.grid(True, color="#eee", lw=0.5)

ax2 = ax1.twinx()
ax2.plot(iu_s, iu_v, color="#E53935", lw=1.6, marker='o', ms=2.5,
         alpha=0.8, label="Validation IoU")
ax2.set_ylabel("Validation IoU", color="#E53935")
ax2.tick_params(axis='y', labelcolor="#E53935")
ax2.set_ylim(0, 0.4)
ax2.spines['top'].set_visible(False)

# annotate peak iou
pk = iu_v.argmax()
ax2.annotate(f"peak IoU {iu_v[pk]:.2f}\n(step {iu_s[pk]:,})",
             xy=(iu_s[pk], iu_v[pk]), xytext=(iu_s[pk]+30000, 0.30),
             fontsize=9, color="#E53935",
             arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.0))

l1,lab1=ax1.get_legend_handles_labels(); l2,lab2=ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lab1+lab2, loc="upper right", fontsize=9, framealpha=0.9)
fig.suptitle("ConvONet in-domain fine-tuning (739-example ScanNet removal dataset)\n"
             "Training loss converges (810→220) but validation IoU peaks at 0.33 then collapses",
             fontsize=10.5, fontweight="bold", y=1.0)
for ext in ("png","pdf"):
    plt.savefig(OUT/f"fig_convonet_training.{ext}", facecolor="white")
    print("Saved", OUT/f"fig_convonet_training.{ext}")
plt.close()
