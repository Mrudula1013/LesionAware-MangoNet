"""600-dpi external-title-ready plots; no titles rendered inside figures."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def confusion_figure(matrix, classes, output, normalize=False):
    values = np.asarray(matrix, dtype=float)
    if normalize:
        values = values / np.maximum(values.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    image = ax.imshow(values, cmap='Blues')
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
    fmt = '.2f' if normalize else '.0f'
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(values[i,j], fmt), ha='center', va='center', fontsize=8,
                    color='white' if values[i,j] > values.max()*0.55 else 'black')
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches='tight'); plt.close(fig)


def reliability_figure(truth, probs, output, bins=15):
    p = np.asarray(probs); y = np.asarray(truth)
    conf = p.max(axis=1); correct = (p.argmax(axis=1) == y).astype(float)
    midpoint, acc, coverage = [], [], []
    for l, r in zip(np.linspace(0,1,bins+1)[:-1], np.linspace(0,1,bins+1)[1:]):
        mask = (conf >= l) & ((conf <= r) if r == 1 else (conf < r))
        midpoint.append((l+r)/2); acc.append(correct[mask].mean() if mask.any() else np.nan)
        coverage.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0,1], [0,1], '--', linewidth=1, label='Ideal calibration')
    valid = np.isfinite(acc)
    ax.plot(np.asarray(midpoint)[valid], np.asarray(acc)[valid], 'o-', label='Observed accuracy')
    ax.set(xlabel='Predicted confidence', ylabel='Empirical accuracy', xlim=(0,1), ylim=(0,1))
    ax.legend(loc='upper left'); fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches='tight'); plt.close(fig)


def training_figure(history_csv, output):
    df = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df.epoch, df.val_macro_f1, label='Validation Macro-F1')
    ax.set(xlabel='Epoch', ylabel='Macro-F1', ylim=(0,1))
    ax.legend(); fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches='tight'); plt.close(fig)


def robustness_figure(csv_file, output):
    df = pd.read_csv(csv_file)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, group in df.groupby('corruption'):
        ax.plot(group.severity, group.macro_f1, marker='o', label=name.replace('_', ' '))
    ax.set(xlabel='Corruption severity', ylabel='Macro-F1', xticks=[1,2,3,4,5], ylim=(0,1))
    ax.legend(loc='best', fontsize=8); fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches='tight'); plt.close(fig)
