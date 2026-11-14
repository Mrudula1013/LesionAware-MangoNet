#!/usr/bin/env python3
"""Save per-image lesion attention / Grad-CAM / Score-CAM and quantitative faithfulness."""
import _bootstrap  # noqa
import argparse
from pathlib import Path
import pandas as pd
from mangonet.training.trainer import load_trained_model
from mangonet.data.dataset import load_manifest, subset_manifest, MangoImageDataset
from mangonet.evaluation.explainability import generate_explanations, save_explanation_figure
from mangonet.evaluation.evaluator import check_transfer_leakage
from mangonet.utils.reproducibility import get_device,save_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',required=True)
p.add_argument('--data-root',default='data')
p.add_argument('--manifest',default=None)
p.add_argument('--target',choices=['MangoLeafBD','MLD24','MLD-Field'])
p.add_argument('--out',default='outputs/explanations')
p.add_argument('--max-images',type=int,default=8)
p.add_argument('--scorecam-channels',type=int,default=32)
p.add_argument('--skip-faithfulness',action='store_true')
p.add_argument('--device',default='auto')
a=p.parse_args()
model,ckpt=load_trained_model(a.checkpoint,a.device)
frame=load_manifest(a.manifest or str(Path(a.data_root)/'processed/manifest.csv'))
source=ckpt['source']; target=a.target
if target and target!=source:
    check_transfer_leakage(frame,source,target)
    chosen=subset_manifest(frame,source,'test',target=target)
else:
    chosen=subset_manifest(frame,source,'test')
if chosen.empty:raise SystemExit('No test images found')
stats=ckpt['source_stats'];dataset=MangoImageDataset(chosen,a.data_root,stats)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
measures=[]
for idx in range(min(a.max_images,len(dataset))):
    item=dataset[idx];image=item['image'].unsqueeze(0).to(get_device(a.device))
    maps,metrics=generate_explanations(model,image,stats,
                                      top_channels=a.scorecam_channels,
                                      run_faithfulness=not a.skip_faithfulness)
    save_explanation_figure(image,stats,maps,out/f'explanation_{idx:04d}.png')
    metrics['path']=item['path'];metrics['true_class_id']=int(item['label'])
    measures.append(metrics)
    print(f'[{idx+1}/{min(a.max_images,len(dataset))}] {item["path"]}',flush=True)
save_json(measures,out/'faithfulness.json')
flat=[]
for item in measures:
    row={k:v for k,v in item.items() if not isinstance(v,dict)}
    for key in ('lesion','gradcam','scorecam'):
        for sub,value in item.get(key,{}).items():
            if not isinstance(value,list):row[f'{key}_{sub}']=value
    flat.append(row)
pd.DataFrame(flat).to_csv(out/'explainability_summary.csv',index=False)
