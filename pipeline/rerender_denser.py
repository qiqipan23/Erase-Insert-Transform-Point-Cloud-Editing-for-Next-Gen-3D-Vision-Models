"""Denser top-down renderer: disk-splat each point so sparse clouds read as solid surfaces."""
import sys, re
from pathlib import Path
import numpy as np
from PIL import Image

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
        dt=np.dtype([(nm,TM.get(tp,"f4")) for tp,nm in props]);arr=np.frombuffer(data[he:],dtype=dt,count=n)
    else:
        rows=[list(map(float,l.split())) for l in data[he:].decode("ascii","ignore").splitlines()[:n] if l.strip()]
        arr=np.rec.fromarrays(np.array(rows).T,names=[nm for _,nm in props])
    xyz=np.column_stack([arr["x"],arr["y"],arr["z"]]).astype(np.float32)
    rgb=None
    for names in (("red","green","blue"),("r","g","b")):
        if all(nm in arr.dtype.names for nm in names):
            r_=np.column_stack([arr[nm] for nm in names]).astype(np.float64)
            rgb=(r_/r_.max()*255 if r_.max()>1.5 else r_*255).astype(np.uint8);break
    if rgb is None: rgb=np.full((len(xyz),3),150,np.uint8)
    return xyz,rgb

def render(xyz,rgb,size=800,pad=0.05,radius=3,bg=(240,240,240)):
    x,y=xyz[:,0],xyz[:,1]
    xmn,xmx=x.min(),x.max();ymn,ymx=y.min(),y.max()
    span=max(xmx-xmn,ymx-ymn,1e-3);p=span*pad
    xmn-=p;xmx+=p;ymn-=p;ymx+=p;spanp=max(xmx-xmn,ymx-ymn)
    z=xyz[:,2];order=np.argsort(z)  # higher z drawn last (on top)
    x,y,rgb=x[order],y[order],rgb[order]
    px=((x-xmn)/spanp*(size-1)).astype(np.int32).clip(0,size-1)
    py=((1.0-(y-ymn)/spanp)*(size-1)).astype(np.int32).clip(0,size-1)
    img=np.full((size,size,3),bg,np.uint8)
    # disk splat
    for dx in range(-radius,radius+1):
        for dy in range(-radius,radius+1):
            if dx*dx+dy*dy>radius*radius: continue
            qx=(px+dx).clip(0,size-1);qy=(py+dy).clip(0,size-1)
            img[qy,qx]=rgb
    return img

if __name__=="__main__":
    src=Path(sys.argv[1]); out=Path(sys.argv[2]); rad=int(sys.argv[3]) if len(sys.argv)>3 else 3
    xyz,rgb=load_ply(src)
    im=render(xyz,rgb,radius=rad)
    bg=(im[:,:,0]>235)&(im[:,:,1]>235)&(im[:,:,2]>235)
    Image.fromarray(im).save(out)
    print(f"{out.name}: coverage {(1-bg.mean())*100:.1f}% (radius={rad})")
