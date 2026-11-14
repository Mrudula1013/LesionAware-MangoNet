#!/usr/bin/env python3
"""Test-only 5x5 corruption sweeps; severity regression and reliability."""
import _bootstrap  # noqa
import argparse
from pathlib import Path
from mangonet.training.trainer import load_trained_model
from mangonet.data.dataset import load_manifest,subset_manifest,MangoImageDataset
from mangonet.evaluation.evaluator import check_transfer_leakage
from mangonet.evaluation.calibration import obtain_validation_temperature
from mangonet.evaluation.robustness import evaluate_corruptions
from mangonet.utils.reproducibility import get_device

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',required=True)
p.add_argument('--data-root',default='data')
p.add_argument('--manifest',default=None)
p.add_argument('--target',choices=['MangoLeafBD','MLD24','MLD-Field'],default=None)
p.add_argument('--out',default='outputs/robustness')
p.add_argument('--device',default='auto')
p.add_argument('--max-images',type=int,default=0,help='Subset diagnostic, default all test images')
p.add_argument('--no-tta',action='store_true')
p.add_argument('--temperature',type=float,default=None)
p.add_argument('--types',nargs='*',default=None)
a=p.parse_args()
frame=load_manifest(a.manifest or str(Path(a.data_root)/'processed/manifest.csv'))
model,ckpt=load_trained_model(a.checkpoint,a.device)
source=ckpt['source']; target=a.target
if target and target!=source:
    check_transfer_leakage(frame,source,target)
    df=subset_manifest(frame,source,'test',target=target)
else:
    df=subset_manifest(frame,source,'test')
if df.empty: raise SystemExit('No test images found')
stats=ckpt['source_stats'];device=get_device(a.device)
val=MangoImageDataset(subset_manifest(frame,source,'val'),a.data_root,stats)
T=a.temperature or obtain_validation_temperature(model,val,device)
print(f'Source validation temperature: {T:.5f}')
test=MangoImageDataset(df,a.data_root,stats)
frame=evaluate_corruptions(model,test,stats,device,a.out,T,not a.no_tta,
                           seed=int(ckpt.get('seed',0)),max_images=a.max_images,types=a.types)
print(frame.to_string(index=False))
