from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from mangonet.config import CLASSES
from mangonet.data.quality_control import infer_label, group_near_duplicates, assign_group_splits, phash_variants, dhash_variants
from mangonet.data.dataset import subset_manifest
from mangonet.evaluation.metrics import classification_metrics,stats_from_seeds,paired_significance,holm_adjust
from mangonet.evaluation.calibration import fit_temperature


def test_label_aliases():
    assert infer_label('folder/Sooty Mold/image001.png')=='Sooty Mould'
    assert infer_label('folder/Dieback/flower.jpg')=='Die Back'
    assert infer_label('folder/Healthy/img.png')=='Healthy'
    assert infer_label('folder/unknown/file.png') is None


def test_group_split_no_leak():
    rows=[]
    for ds in ('MangoLeafBD','MLD24'):
        for c in CLASSES:
            for i in range(16):
                rows.append({'dataset':ds,'label':c,'group_id':f'{ds}/{c}/{i}'})
    for c in CLASSES:
        for i in range(3):rows.append({'dataset':'MLD-Field','label':c,'group_id':f'field/{c}/{i}'})
    assign_group_splits(rows,seed=42)
    df=pd.DataFrame(rows)
    assert set(df[df.dataset=='MLD-Field'].split)=={'test'}
    assert all((df.groupby('group_id').split.nunique()==1))
    for ds in ('MangoLeafBD','MLD24'):
        for c in CLASSES:
            assert set(df[(df.dataset==ds)&(df.label==c)].split)=={'train','val','test'}
    assert len(subset_manifest(df,'MangoLeafBD','train'))>0
    assert len(subset_manifest(df,'MangoLeafBD','test','MLD-Field'))==24


def test_metrics_and_statistics():
    labels=[0,1,2,3,4,5,6,7]
    probs=np.eye(8)*0.85+0.15/8
    probs=probs/probs.sum(axis=1,keepdims=True)
    m=classification_metrics(labels,probabilities=probs)
    assert m['accuracy']==1.0 and m['macro_f1']==1.0
    assert m['ece']>=0 and m['brier']>=0
    s=stats_from_seeds([.7,.8,.75,.77,.76])
    assert s['ci95_lower']<s['mean']<s['ci95_upper']
    assert len(holm_adjust([0.04,0.02,0.01]))==3
    r=paired_significance([.7,.8,.7,.8,.75],[.75,.85,.72,.86,.80])
    assert r['n_pairs']==5


def test_temperature_positive():
    import torch
    logits=torch.tensor([[6.,0.],[4.,1.],[0.,4.],[1.,3.]])
    labels=torch.tensor([0,0,1,1])
    t=fit_temperature(logits,labels,30)
    assert np.isfinite(t) and t>0
