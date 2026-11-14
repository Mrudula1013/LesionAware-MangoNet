#!/usr/bin/env python3
"""Source-only temperature calibration followed by independent 8-class testing."""
import _bootstrap  # noqa
import argparse
from pathlib import Path
from mangonet.evaluation.evaluator import evaluate_checkpoint

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',required=True)
p.add_argument('--data-root',default='data')
p.add_argument('--manifest',default=None)
p.add_argument('--target',choices=['MangoLeafBD','MLD24','MLD-Field'],default=None)
p.add_argument('--out',default='outputs/evaluations')
p.add_argument('--batch-size',type=int,default=32)
p.add_argument('--num-workers',type=int,default=0)
p.add_argument('--temperature',type=float,default=None,help='If omitted fit ONLY on source validation data')
p.add_argument('--no-tta',action='store_true')
p.add_argument('--max-images',type=int,default=0,help='SMOKE TEST ONLY')
p.add_argument('--device',default='auto')
a=p.parse_args()
res=evaluate_checkpoint(a.checkpoint,a.manifest or str(Path(a.data_root)/'processed/manifest.csv'),
                        a.data_root,a.out,a.target,a.batch_size,a.num_workers,a.device,
                        a.temperature,not a.no_tta,a.max_images)
print('Evaluation metrics:',res['metrics'])
print('Saved:',a.out)
