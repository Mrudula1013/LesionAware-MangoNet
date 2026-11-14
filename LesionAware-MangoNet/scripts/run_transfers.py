#!/usr/bin/env python3
"""Four locked zero-shot source->target transfers, each across five seeds."""
import _bootstrap  # noqa
import argparse
from pathlib import Path
from mangonet.evaluation.evaluator import evaluate_checkpoint
from mangonet.evaluation.statistics import aggregate_results
from mangonet.utils.reproducibility import save_json

TRANSFERS=[('MangoLeafBD','MLD24'),('MLD24','MangoLeafBD'),
           ('MangoLeafBD','MLD-Field'),('MLD24','MLD-Field')]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root',default='data')
p.add_argument('--manifest',default=None)
p.add_argument('--runs',default='outputs/runs')
p.add_argument('--out',default='outputs/transfers')
p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
p.add_argument('--batch-size',type=int,default=32)
p.add_argument('--device',default='auto')
p.add_argument('--max-images',type=int,default=0)
a=p.parse_args()
manifest=a.manifest or str(Path(a.data_root)/'processed/manifest.csv')
results=[]; missing=[]
for source,target in TRANSFERS:
    for seed in a.seeds:
        ckpt=Path(a.runs)/source/f'seed_{seed}'/'best.pt'
        if not ckpt.exists():
            missing.append(str(ckpt));continue
        output=Path(a.out)/f'{source}_to_{target}'/f'seed_{seed}'
        res=evaluate_checkpoint(ckpt,manifest,a.data_root,output,target,
                                a.batch_size,0,a.device,max_images=a.max_images)
        results.append(res)
        print(f'{source} -> {target} seed={seed}: Macro-F1 {res["metrics"]["macro_f1"]:.4f}', flush=True)
if not results:
    raise SystemExit('No model checkpoints found. Train MangoLeafBD and MLD24 first.')
aggregate_results(results,a.out)
save_json({'missing_checkpoints':missing, 'evaluations_completed':len(results),
           'required_evaluations':4*len(a.seeds)},Path(a.out)/'completion.json')
print(f'Finished {len(results)}/{4*len(a.seeds)} transfers; missing checkpoints: {len(missing)}.')
