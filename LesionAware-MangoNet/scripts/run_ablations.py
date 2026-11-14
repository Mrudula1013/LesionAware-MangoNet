#!/usr/bin/env python3
"""Train architectural/loss ablations and evaluate TTA/calibration inference ablations."""
import _bootstrap  # noqa
import argparse
from pathlib import Path
from mangonet.config import load_model_config, read_yaml
from mangonet.training.trainer import train_run
from mangonet.evaluation.evaluator import evaluate_checkpoint
from mangonet.evaluation.statistics import aggregate_results

ARCH_VARIANTS=('full','cnn_only','without_channel','without_spatial','without_lesion','without_consistency')
INFER_VARIANTS=('without_temperature','without_tta')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--variants',nargs='+',choices=ARCH_VARIANTS+INFER_VARIANTS,default=list(ARCH_VARIANTS+INFER_VARIANTS))
p.add_argument('--source',choices=['combined','MangoLeafBD','MLD24'],default='combined')
p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
p.add_argument('--data-root',default='data')
p.add_argument('--manifest',default=None)
p.add_argument('--model-config',default='configs/model.yaml')
p.add_argument('--training-config',default='configs/training.yaml')
p.add_argument('--full-checkpoints',default='outputs/runs')
p.add_argument('--out',default='outputs/ablations')
p.add_argument('--epochs',type=int,default=None)
p.add_argument('--batch-size',type=int,default=None)
p.add_argument('--num-workers',type=int,default=0)
p.add_argument('--max-train-batches',type=int,default=0)
p.add_argument('--max-test-images',type=int,default=0)
p.add_argument('--device',default='auto')
a=p.parse_args()
manifest=a.manifest or str(Path(a.data_root)/'processed/manifest.csv')
results=[]
for variant in a.variants:
    for seed in a.seeds:
        if variant in INFER_VARIANTS:
            ckpt=Path(a.full_checkpoints)/a.source/f'seed_{seed}'/'best.pt'
            if not ckpt.exists():
                raise SystemExit(f'Inference ablation requires a trained full checkpoint: {ckpt}')
        else:
            cfg=load_model_config(a.model_config)
            train_cfg=read_yaml(a.training_config)
            if variant=='cnn_only':cfg.use_transformer=False;cfg.lesion_attention=False
            if variant=='without_channel':cfg.channel_attention=False
            if variant=='without_spatial':cfg.spatial_attention=False
            if variant=='without_lesion':cfg.lesion_attention=False
            if variant in ('without_lesion','cnn_only'):
                train_cfg['lambda_entropy']=0;train_cfg['lambda_consistency']=0
            if variant=='without_consistency':train_cfg['lambda_consistency']=0
            ckpt=train_run(a.data_root,manifest,Path(a.out)/variant/a.source/f'seed_{seed}'/'train',
                           source=a.source,seed=seed,model_config=cfg,train_config=train_cfg,
                           device=a.device,epochs=a.epochs,batch_size=a.batch_size,
                           num_workers=a.num_workers,max_train_batches=a.max_train_batches)
        res=evaluate_checkpoint(ckpt,manifest,a.data_root,
                                Path(a.out)/variant/a.source/f'seed_{seed}'/'test',
                                device=a.device,temperature=1.0 if variant=='without_temperature' else None,
                                tta=variant!='without_tta',max_images=a.max_test_images)
        res['model']=variant
        results.append(res)
if results:aggregate_results(results,a.out,comparator='full' if 'full' in a.variants else None)
