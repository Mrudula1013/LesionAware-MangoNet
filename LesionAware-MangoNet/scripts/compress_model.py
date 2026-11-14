#!/usr/bin/env python3
"""Physical L1 structural pruning -> source-train fine-tune -> ONNX/static INT8 PTQ."""
import _bootstrap  # noqa
import argparse
from pathlib import Path
import copy
import torch
from torch.utils.data import DataLoader
from mangonet.config import ModelConfig
from mangonet.training.trainer import load_trained_model,make_loader,eval_model
from mangonet.training.losses import class_balancing_weights, combined_loss
from mangonet.data.dataset import load_manifest, subset_manifest, MangoImageDataset
from mangonet.compression.pruning import structurally_prune
from mangonet.compression.quantization import (export_onnx, quantize_onnx_static,
                                                benchmark_torch,benchmark_onnx,parity_check,estimate_macs)
from mangonet.utils.reproducibility import get_device,save_json,seed_everything

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',required=True)
p.add_argument('--data-root',default='data')
p.add_argument('--manifest',default=None)
p.add_argument('--out',default='outputs/compressed')
p.add_argument('--fraction',type=float,default=0.30)
p.add_argument('--finetune-epochs',type=int,default=15)
p.add_argument('--finetune-lr',type=float,default=1e-4)
p.add_argument('--batch-size',type=int,default=32)
p.add_argument('--num-workers',type=int,default=0)
p.add_argument('--max-train-batches',type=int,default=0,help='SMOKE TEST ONLY')
p.add_argument('--calibration-images',type=int,default=512)
p.add_argument('--skip-onnx',action='store_true',help='Skip export/PTQ and ONNX benchmarking')
p.add_argument('--skip-benchmark',action='store_true')
p.add_argument('--warmup',type=int,default=50)
p.add_argument('--repeats',type=int,default=500)
p.add_argument('--device',default='auto')
a=p.parse_args()
seed_everything(0)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
original,meta=load_trained_model(a.checkpoint,'cpu')
source=meta['source'];stats=meta['source_stats']
model,pruning_report=structurally_prune(original,a.fraction)
save_json(pruning_report,out/'pruning_manifest.json')
print('Pruning:',pruning_report)
# Evaluate the actual FP32 compute graph before/after structural pruning.
macs_report={'original':estimate_macs(original,original.config.image_size),
             'pruned':estimate_macs(model,model.config.image_size)}
save_json(macs_report,out/'macs_comparison.json')
print('MACs profile:',macs_report)
frame=load_manifest(a.manifest or str(Path(a.data_root)/'processed/manifest.csv'))
trainframe=subset_manifest(frame,source,'train');valframe=subset_manifest(frame,source,'val')
if trainframe.empty or valframe.empty: raise SystemExit('Source training/validation images are required for pruning fine-tune')
train=MangoImageDataset(trainframe,a.data_root,stats,pairs=True)
val=MangoImageDataset(valframe,a.data_root,stats,pairs=False)
trainloader=make_loader(train,a.batch_size,True,a.num_workers,int(meta['seed']))
valloader=make_loader(val,a.batch_size,False,a.num_workers,int(meta['seed']))
alpha=class_balancing_weights(trainframe.label_id.tolist()).to(get_device(a.device))
model.to(get_device(a.device))
device=next(model.parameters()).device
config=meta['training_config'].copy()
config['lambda_entropy']=config.get('lambda_entropy',0.05)
config['lambda_consistency']=config.get('lambda_consistency',0.5)
config['lambda_ce']=config.get('lambda_ce',1.0)
config['lambda_focal']=config.get('lambda_focal',0.5)
config['loss_ramp_epochs']=max(1,config.get('loss_ramp_epochs',10))
config['focal_gamma']=config.get('focal_gamma',2.0)
optim=torch.optim.AdamW(model.parameters(),lr=a.finetune_lr,weight_decay=config.get('weight_decay',0.05))
best=-float('inf');best_state=None;history=[]
for epoch in range(a.finetune_epochs):
    model.train(); losses=[]
    for batch_idx,batch in enumerate(trainloader):
        if a.max_train_batches and batch_idx>=a.max_train_batches:break
        images=batch['views'].to(device)
        labels=batch['label'].to(device)
        optim.zero_grad(set_to_none=True)
        a0=model(images[:,0]);a1=model(images[:,1])
        loss,parts=combined_loss(a0,a1,labels,batch['matrices'].to(device),
                                 batch['valid_views'].to(device),alpha,epoch,config)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step()
        losses.append(parts['total'])
    metrics=eval_model(model,valloader,device)
    history.append({'epoch':epoch+1,'train_loss':sum(losses)/max(1,len(losses)),
                    'val_macro_f1':metrics['macro_f1']})
    if metrics['macro_f1']>best:
        best=metrics['macro_f1'];best_state=copy.deepcopy(model.cpu().state_dict());model.to(device)
    print(f'[pruned] epoch={epoch+1} validation Macro-F1={metrics["macro_f1"]:.4f}',flush=True)
if best_state:
    model.cpu().load_state_dict(best_state)
compressed_meta={**meta,'state_dict':model.state_dict(),
                 'model_config':model.config.to_dict(),
                 'compression':pruning_report,
                 'smoke_test': bool(meta.get('smoke_test') or a.max_train_batches or a.finetune_epochs!=15),
                 'architecture':'LesionAwareMangoNet'}
torch.save(compressed_meta,out/'pruned_finetuned.pt')
save_json({'history':history,'best_validation_macro_f1':best,
           'note':'Quantization may affect classification and explanations; re-evaluate compressed model.'},
          out/'finetune_history.json')
print('Saved:',out/'pruned_finetuned.pt')
if not a.skip_onnx:
    export_onnx(model,out/'pruned_fp32.onnx',model.config.image_size)
    print('FP32 ONNX maximum absolute logit difference:',parity_check(model,out/'pruned_fp32.onnx',model.config.image_size))
    source_cal=MangoImageDataset(trainframe,a.data_root,stats,pairs=False)
    quantize_onnx_static(out/'pruned_fp32.onnx',out/'pruned_int8.onnx',source_cal,a.calibration_images)
    if not a.skip_benchmark:
        benchmarks={'original_torch_cpu_fp32':benchmark_torch(original,original.config.image_size,'cpu',a.warmup,a.repeats),
                    'pruned_torch_cpu_fp32':benchmark_torch(model,model.config.image_size,'cpu',a.warmup,a.repeats),
                    'onnx_cpu_fp32':benchmark_onnx(out/'pruned_fp32.onnx',model.config.image_size,a.warmup,a.repeats),
                    'onnx_cpu_int8':benchmark_onnx(out/'pruned_int8.onnx',model.config.image_size,a.warmup,a.repeats)}
        save_json(benchmarks,out/'inference_benchmark.json')
        print('Benchmarks:',benchmarks)
print('COMPLETED. Always assess compressed accuracy/calibration separately on held-out images.')
