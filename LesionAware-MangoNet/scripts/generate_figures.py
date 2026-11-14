#!/usr/bin/env python3
"""Render 600-dpi figures from ACTUAL result artifacts; never fake metrics."""
import _bootstrap  # noqa
import argparse,json
from pathlib import Path
import pandas as pd
from mangonet.config import CLASSES
from mangonet.utils.visualizations import (training_figure,robustness_figure,confusion_figure,
                                             reliability_figure)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--history',default=None)
p.add_argument('--metrics',default=None,help='Evaluation metrics.json')
p.add_argument('--predictions',default=None,help='Evaluation predictions.csv')
p.add_argument('--robustness',default=None,help='corruptions_by_severity.csv')
p.add_argument('--out',default='outputs/figures')
a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if a.history:
    training_figure(a.history,out/'training_macro_f1.png')
if a.metrics:
    payload=json.loads(Path(a.metrics).read_text())
    confusion_figure(payload['metrics']['confusion_matrix'],CLASSES,out/'confusion_matrix.png')
    confusion_figure(payload['metrics']['confusion_matrix'],CLASSES,out/'confusion_matrix_normalized.png',True)
if a.predictions:
    df=pd.read_csv(a.predictions)
    pcols=[c for c in df.columns if c.startswith('p_')]
    if len(pcols)!=8:raise SystemExit('Expected 8 probability columns in predictions.csv')
    reliability_figure(df['true'].to_numpy(),df[pcols].to_numpy(),out/'reliability.png')
if a.robustness:
    robustness_figure(a.robustness,out/'robustness_by_severity.png')
print('Saved 600-dpi figures to:',out.resolve())
