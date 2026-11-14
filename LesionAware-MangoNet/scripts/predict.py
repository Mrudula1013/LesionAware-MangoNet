#!/usr/bin/env python3
"""Eight-class predictions on a user-supplied image using saved model and source normalization."""
import _bootstrap  # noqa
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from mangonet.config import CLASSES
from mangonet.data.dataset import raw_rgb_image
from mangonet.training.trainer import load_trained_model
from mangonet.evaluation.calibration import predict_probabilities
from mangonet.utils.reproducibility import get_device

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--image',required=True)
p.add_argument('--checkpoint',default=None,help='Trained PyTorch .pt checkpoint')
p.add_argument('--onnx',default=None,help='ONNX model, requires --stats')
p.add_argument('--stats',default=None,help='JSON file of source_train normalization statistics')
p.add_argument('--temperature',type=float,default=1.0,help='Fitted source-validation temperature (1=uncalibrated)')
p.add_argument('--no-tta',action='store_true')
p.add_argument('--device',default='auto')
a=p.parse_args()
if bool(a.onnx)==bool(a.checkpoint):raise SystemExit('Specify exactly one of --checkpoint or --onnx')
if a.checkpoint:
    model,meta=load_trained_model(a.checkpoint,a.device)
    stats=meta['source_stats'];dev=get_device(a.device)
    x=raw_rgb_image(a.image,stats['image_size'])
    mean=torch.tensor(stats['mean'])[:,None,None];std=torch.tensor(stats['std'])[:,None,None]
    with torch.inference_mode():
        pval=predict_probabilities(model,((x-mean)/std).unsqueeze(0).to(dev),
                                   stats,a.temperature,not a.no_tta).cpu().numpy()[0]
else:
    if not a.stats:raise SystemExit('--stats is required with --onnx')
    import onnxruntime as ort
    stats=json.loads(Path(a.stats).read_text())
    x=raw_rgb_image(a.image,stats['image_size'])
    mean=np.asarray(stats['mean'])[:,None,None]
    std=np.asarray(stats['std'])[:,None,None]
    sess=ort.InferenceSession(a.onnx,providers=['CPUExecutionProvider'])
    logits=sess.run(None,{'images':((x.numpy()-mean)/std)[None].astype('float32')})[0][0]
    exp=np.exp(logits/a.temperature-np.max(logits/a.temperature))
    pval=exp/exp.sum()
print(json.dumps({'predicted_class':CLASSES[int(pval.argmax())],
                  'confidence':float(pval.max()),
                  'temperature':a.temperature,
                  'calibration_note': 'T=1.0 is uncalibrated unless source validation fitted T happened to be 1.0.',
                  'probabilities':{name:float(p) for name,p in zip(CLASSES,pval)}},indent=2))
