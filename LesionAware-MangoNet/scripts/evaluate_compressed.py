#!/usr/bin/env python3
"""Measure post-compression held-out accuracy, calibration and retention per backend."""
import _bootstrap  # noqa
import argparse
from pathlib import Path
import numpy as np
import torch
from mangonet.training.trainer import load_trained_model,make_loader
from mangonet.data.dataset import (load_manifest,subset_manifest,MangoImageDataset)
from mangonet.evaluation.evaluator import check_transfer_leakage,evaluate_checkpoint
from mangonet.evaluation.calibration import make_tta_views,fit_temperature
from mangonet.evaluation.metrics import classification_metrics
from mangonet.utils.reproducibility import save_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--original',required=True,help='Original full-precision checkpoint')
p.add_argument('--pruned',required=True,help='Pruned and fine-tuned checkpoint')
p.add_argument('--onnx-fp32',default=None)
p.add_argument('--onnx-int8',default=None)
p.add_argument('--data-root',default='data')
p.add_argument('--manifest',default=None)
p.add_argument('--target',choices=['MangoLeafBD','MLD24','MLD-Field'],default=None)
p.add_argument('--out',default='outputs/compression_evaluation')
p.add_argument('--batch-size',type=int,default=16)
p.add_argument('--max-images',type=int,default=0)
p.add_argument('--no-tta',action='store_true')
p.add_argument('--device',default='auto')
a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
manifest=a.manifest or str(Path(a.data_root)/'processed/manifest.csv')
model,meta=load_trained_model(a.pruned,'cpu')
source=meta['source'];target=a.target or source
frame=load_manifest(manifest)
if target!=source:check_transfer_leakage(frame,source,target)
val_frame=subset_manifest(frame,source,'val')
test_frame=(subset_manifest(frame,source,'test',target=target) if target!=source
            else subset_manifest(frame,source,'test'))
if a.max_images:test_frame=test_frame.head(a.max_images)
if test_frame.empty or val_frame.empty:raise SystemExit('Validation/test splits missing')
res=[]
for name,checkpoint in [('original',a.original),('pruned_fp32',a.pruned)]:
    r=evaluate_checkpoint(checkpoint,manifest,a.data_root,out/name,
                          target if target!=source else None,batch_size=a.batch_size,
                          device=a.device,tta=not a.no_tta,max_images=a.max_images)
    res.append({'model':name,**r})

if a.onnx_fp32 or a.onnx_int8:
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("Install ONNX extras: pip install -e '.[export]'")
    stats=meta['source_stats']
    options=ort.SessionOptions();options.intra_op_num_threads=4
    val=MangoImageDataset(val_frame,a.data_root,stats)
    test=MangoImageDataset(test_frame,a.data_root,stats)
    for name,path in [('onnx_fp32',a.onnx_fp32),('onnx_int8',a.onnx_int8)]:
        if not path:continue
        sess=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        val_logits=[];val_labels=[]
        for item in make_loader(val,a.batch_size):
            val_logits.extend(sess.run(None,{'images':item['image'].numpy().astype('float32')})[0])
            val_labels.extend(item['label'].tolist())
        temp=fit_temperature(np.asarray(val_logits),val_labels)
        probs=[];labels=[]
        for batch in make_loader(test,a.batch_size):
            images=batch['image']
            views=make_tta_views(images,stats['mean'],stats['std']) if not a.no_tta else [images]
            per_view=[]
            for view in views:
                logits=sess.run(None,{'images':view.numpy().astype('float32')})[0]
                logits=logits/temp
                p=np.exp(logits-logits.max(axis=1,keepdims=True))
                per_view.append(p/p.sum(axis=1,keepdims=True))
            probs.extend(np.mean(per_view,axis=0).tolist())
            labels.extend(batch['label'].tolist())
        metrics=classification_metrics(labels,probabilities=probs)
        res.append({'model':name,'source':source,'target':target,
                    'temperature':temp,'metrics':metrics,
                    'smoke_test':bool(a.max_images or meta.get('smoke_test'))})
if not res:raise SystemExit('No results')
original_f1=res[0]['metrics']['macro_f1']
for row in res:
    row['macro_f1_retention_pct']=100*row['metrics']['macro_f1']/max(original_f1,1e-12) if original_f1>0 else None
save_json(res,out/'compression_accuracy_comparison.json')
print('Comparison:')
for row in res:print(row['model'],row['metrics']['macro_f1'],row['metrics']['ece'],row['macro_f1_retention_pct'])
