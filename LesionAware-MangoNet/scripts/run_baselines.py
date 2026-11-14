#!/usr/bin/env python3
"""Train or evaluate manuscript comparators under common source splits and augmentation."""
import _bootstrap  # noqa
import argparse
from pathlib import Path
from mangonet.evaluation.baselines import (train_baseline,load_baseline_checkpoint,
                                            TORCHVISION_MODELS,TIMM_MODELS)
from mangonet.evaluation.evaluator import check_transfer_leakage
from mangonet.data.dataset import MangoImageDataset,load_manifest,subset_manifest
from mangonet.evaluation.calibration import predict_probabilities,obtain_validation_temperature
from mangonet.evaluation.metrics import classification_metrics
from mangonet.evaluation.statistics import aggregate_results
from mangonet.training.trainer import make_loader
from mangonet.config import read_yaml
from mangonet.utils.reproducibility import get_device,save_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode',choices=['train','evaluate'],default='train')
p.add_argument('--models',nargs='+',choices=list(TORCHVISION_MODELS)+list(TIMM_MODELS),default=['resnet50','densenet121','mobilenet_v3_large','efficientnet_b0'])
p.add_argument('--source',choices=['combined','MangoLeafBD','MLD24'],default='combined')
p.add_argument('--target',choices=['MangoLeafBD','MLD24','MLD-Field'],default=None)
p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
p.add_argument('--data-root',default='data')
p.add_argument('--manifest',default=None)
p.add_argument('--training-config',default='configs/training.yaml')
p.add_argument('--out',default='outputs/baselines')
p.add_argument('--pretrained',action='store_true',help='Use genuine ImageNet pretrained weights (may need download)')
p.add_argument('--epochs',type=int,default=None)
p.add_argument('--batch-size',type=int,default=32)
p.add_argument('--num-workers',type=int,default=0)
p.add_argument('--max-train-batches',type=int,default=0)
p.add_argument('--max-test-images',type=int,default=0)
p.add_argument('--device',default='auto')
a=p.parse_args()
manifest=a.manifest or str(Path(a.data_root)/'processed/manifest.csv')
frame=load_manifest(manifest)
results=[]
for name in a.models:
    for seed in a.seeds:
        root=Path(a.out)/name/a.source/f'seed_{seed}'
        ckpt=root/'best.pt'
        if a.mode=='train':
            train_baseline(name,a.data_root,manifest,root,source=a.source,seed=seed,
                           config=read_yaml(a.training_config),pretrained=a.pretrained,
                           device=a.device,epochs=a.epochs,batch_size=a.batch_size,
                           num_workers=a.num_workers,max_train_batches=a.max_train_batches)
            continue
        if not ckpt.exists():
            print('Missing baseline checkpoint:',ckpt);continue
        model,meta=load_baseline_checkpoint(ckpt,a.device)
        dev=get_device(a.device);stats=meta['source_stats']
        target=a.target or a.source
        if target!=a.source:
            check_transfer_leakage(frame,a.source,target)
            subset=subset_manifest(frame,a.source,'test',target=target)
        else:
            subset=subset_manifest(frame,a.source,'test')
        if a.max_test_images:subset=subset.head(a.max_test_images)
        if subset.empty:raise SystemExit(f'No test images for {target}')
        val=MangoImageDataset(subset_manifest(frame,a.source,'val'),a.data_root,stats)
        temperature=obtain_validation_temperature(model,val,dev,a.batch_size)
        test=MangoImageDataset(subset,a.data_root,stats)
        loader=make_loader(test,a.batch_size,False,a.num_workers,seed)
        labels=[];probs=[]
        for batch in loader:
            pvals=predict_probabilities(model,batch['image'].to(dev),stats,temperature,True)
            probs.extend(pvals.cpu().tolist());labels.extend(batch['label'].tolist())
        metrics=classification_metrics(labels,probabilities=probs)
        record={'model':name,'source':a.source,'target':target,'seed':seed,
                'temperature':temperature,'metrics':metrics,
                'smoke_test':bool(meta.get('smoke_test') or a.max_test_images)}
        results.append(record)
        save_json(record,root/f'eval_{target}.json')
        print(f'{name} seed {seed}: Macro-F1={metrics["macro_f1"]:.4f}')
if results:aggregate_results(results,Path(a.out)/'comparisons')
