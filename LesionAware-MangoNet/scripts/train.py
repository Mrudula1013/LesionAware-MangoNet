#!/usr/bin/env python3
"""Train original MangoNet for one or all five manuscript seeds."""
import _bootstrap  # noqa
import argparse
from pathlib import Path
from mangonet.config import load_model_config, read_yaml
from mangonet.training.trainer import train_run

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root',default='data')
p.add_argument('--manifest',default=None)
p.add_argument('--source',choices=['combined','MangoLeafBD','MLD24'],default='combined')
p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
p.add_argument('--model-config',default='configs/model.yaml')
p.add_argument('--training-config',default='configs/training.yaml')
p.add_argument('--out',default='outputs/runs')
p.add_argument('--device',default='auto')
p.add_argument('--epochs',type=int,default=None)
p.add_argument('--batch-size',type=int,default=None)
p.add_argument('--num-workers',type=int,default=None)
p.add_argument('--max-train-batches',type=int,default=0,help='SMOKE TEST ONLY')
p.add_argument('--max-val-batches',type=int,default=0,help='SMOKE TEST ONLY')
a=p.parse_args()
manifest=a.manifest or str(Path(a.data_root)/'processed'/'manifest.csv')
for seed in a.seeds:
    path=train_run(a.data_root,manifest,Path(a.out)/a.source/f'seed_{seed}',
                   source=a.source,seed=seed,
                   model_config=load_model_config(a.model_config),
                   train_config=read_yaml(a.training_config),device=a.device,
                   epochs=a.epochs,batch_size=a.batch_size,num_workers=a.num_workers,
                   max_train_batches=a.max_train_batches,max_val_batches=a.max_val_batches)
    print('Saved:',path)
