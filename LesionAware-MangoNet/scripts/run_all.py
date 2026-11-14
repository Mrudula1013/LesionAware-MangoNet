#!/usr/bin/env python3
"""Orchestrate dataset QC, original training, transfers, evaluation, XAI and compression.

Actual full runs are intentionally computationally intensive; the optional
fast development mode uses a SMALL model and is NOT publication-comparable.
"""
import argparse
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root',default='data')
p.add_argument('--out',default='outputs/full_pipeline')
p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
p.add_argument('--fast-dev-run',action='store_true',help='Tiny smoke test; not manuscript reproduction')
p.add_argument('--skip-qc',action='store_true')
p.add_argument('--skip-baselines',action='store_true')
p.add_argument('--skip-explain',action='store_true')
p.add_argument('--skip-robustness',action='store_true')
p.add_argument('--skip-compression',action='store_true')
p.add_argument('--skip-onnx',action='store_true')
p.add_argument('--device',default='auto')
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
output=Path(a.out)
source_out=output/'runs'
script_dir=root/'scripts'

def run(script, *params):
    cmd=[sys.executable,str(script_dir/script),*map(str,params)]
    print('\nRUNNING:', ' '.join(cmd), flush=True)
    subprocess.run(cmd,check=True,cwd=root)

if not a.skip_qc:
    run('prepare_data.py','--data-root',a.data_root)
if not (Path(a.data_root)/'processed/manifest.csv').exists():
    raise SystemExit('Missing manifest; run prepare_data.py first')
config=root/'configs'/('tiny_model.yaml' if a.fast_dev_run else 'model.yaml')
common=['--data-root',a.data_root,'--device',a.device,'--seeds',*a.seeds]
training=['--epochs','1','--batch-size','4','--num-workers','0',
          '--max-train-batches','1','--max-val-batches','1'] if a.fast_dev_run else []
for source in ('combined','MangoLeafBD','MLD24'):
    run('train.py',*common,'--source',source,'--model-config',config,
        '--out',source_out,*training)
run('run_transfers.py',*common,'--runs',source_out,'--out',output/'transfers',
    *(['--max-images','8'] if a.fast_dev_run else []))
ckpt=source_out/'MangoLeafBD'/f'seed_{a.seeds[0]}'/'best.pt'
if not a.skip_robustness:
    run('run_robustness.py','--checkpoint',ckpt,'--data-root',a.data_root,
        '--target','MLD-Field','--out',output/'robustness',
        *(['--max-images','4'] if a.fast_dev_run else []))
if not a.skip_explain:
    run('explain.py','--checkpoint',ckpt,'--data-root',a.data_root,
        '--target','MLD-Field','--out',output/'explanations',
        '--max-images',2 if a.fast_dev_run else 8,
        *(['--scorecam-channels','4','--skip-faithfulness'] if a.fast_dev_run else []))
if not a.skip_compression:
    run('compress_model.py','--checkpoint',ckpt,'--data-root',a.data_root,
        '--out',output/'compressed',
        '--finetune-epochs',1 if a.fast_dev_run else 15,
        *(['--max-train-batches','1','--batch-size','4'] if a.fast_dev_run else []),
        *(['--skip-onnx'] if a.skip_onnx else []),
        *(['--calibration-images','8','--skip-benchmark'] if a.fast_dev_run else []))
if not a.skip_baselines:
    run('run_baselines.py',*common,'--source','combined','--mode','train',
        '--out',output/'baselines',
        *(['--epochs','1','--batch-size','4','--max-train-batches','1'] if a.fast_dev_run else ['--pretrained']))
print('\nPipeline commands completed; inspect split-QC review flags and metrics before any publication claim.')
