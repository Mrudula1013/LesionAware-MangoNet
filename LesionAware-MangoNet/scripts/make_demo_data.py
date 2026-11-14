#!/usr/bin/env python3
"""Small SYNTHETIC images for software smoke tests ONLY (never publication results)."""
import _bootstrap  # noqa
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from mangonet.config import ALL_DATASETS, CLASSES

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root',default='demo_data')
p.add_argument('--per-class',type=int,default=9)
p.add_argument('--size',type=int,default=224)
a=p.parse_args()
root=Path(a.data_root)
for ds_idx, ds in enumerate(ALL_DATASETS):
    for cls_idx, cls in enumerate(CLASSES):
        folder=root/'raw'/ds/cls; folder.mkdir(parents=True,exist_ok=True)
        for i in range(a.per_class):
            rng=np.random.default_rng(100000*ds_idx+1000*cls_idx+i)
            background=rng.integers(10,90,size=(a.size,a.size,3),dtype=np.uint8)
            image=Image.fromarray(background,'RGB')
            draw=ImageDraw.Draw(image)
            # Random leaf position, shape and distinct lesion distribution per image.
            cx,cy=int(rng.integers(90,135)),int(rng.integers(90,135))
            rw,rh=int(rng.integers(40,60)),int(rng.integers(70,99))
            green=(int(rng.integers(20,65)),int(rng.integers(90,195)),int(rng.integers(10,80)))
            draw.ellipse([cx-rw,cy-rh,cx+rw,cy+rh],fill=green)
            for lesion in range(cls_idx+1):
                x=int(rng.integers(cx-rw//2,cx+rw//2));y=int(rng.integers(cy-rh//2,cy+rh//2))
                r=int(rng.integers(3,10))
                draw.ellipse([x-r,y-r,x+r,y+r],fill=(int(rng.integers(80,200)),60,20))
            image.save(folder/f'leaf_{i:03d}.png')
print(f'Created synthetic test fixture only at {root}; NOT experimental mango disease data.')
