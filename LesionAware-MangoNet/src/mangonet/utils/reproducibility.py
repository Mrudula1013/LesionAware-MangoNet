import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (TypeError, RuntimeError):
            pass
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def get_device(arg='auto'):
    if arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(arg)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def checkpoint_load(path, map_location='cpu'):
    # Only load checkpoints from sources you trust; torch checkpoints may contain objects.
    return torch.load(path, map_location=map_location, weights_only=False)
