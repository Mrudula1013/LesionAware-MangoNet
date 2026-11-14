"""Five corruption families, five severity levels, test-only stress."""
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from mangonet.data.dataset import MangoImageDataset
from mangonet.evaluation.calibration import predict_probabilities
from mangonet.evaluation.metrics import classification_metrics
from mangonet.utils.reproducibility import save_json

CORRUPTIONS = {
    'gaussian_noise': [0.04, 0.06, 0.08, 0.10, 0.12],
    'motion_blur': [3, 5, 7, 9, 11],
    'brightness_shift': [0.1, 0.2, 0.3, 0.4, 0.5],
    'jpeg': [80, 60, 40, 25, 15],
    'occlusion': [0.05, 0.10, 0.15, 0.20, 0.25],
}


def corrupt_image(image, kind, severity, rng):
    """Apply corruption to raw RGB float image [3,H,W] in [0,1]."""
    if kind not in CORRUPTIONS or not 1 <= severity <= 5:
        raise ValueError('Unknown corruption or severity; expected severity 1..5')
    x = image.cpu().permute(1, 2, 0).numpy().copy()
    h, w = x.shape[:2]
    value = CORRUPTIONS[kind][severity - 1]
    if kind == 'gaussian_noise':
        x += rng.normal(0, value, size=x.shape).astype(np.float32)
    elif kind == 'motion_blur':
        kernel = np.zeros((value, value), dtype=np.float32)
        kernel[value // 2, :] = 1 / value
        angle = float(rng.uniform(0, 180))
        matrix = cv2.getRotationMatrix2D(((value - 1) / 2, (value - 1) / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, matrix, (value, value))
        kernel /= max(kernel.sum(), 1e-6)
        x = cv2.filter2D(x, -1, kernel)
    elif kind == 'brightness_shift':
        hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
        sign = -1.0 if rng.random() < 0.5 else 1.0
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + sign * value, 0, 1)
        x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    elif kind == 'jpeg':
        array = (np.clip(x, 0, 1) * 255).round().astype(np.uint8)
        buffer = BytesIO()
        Image.fromarray(array, 'RGB').save(buffer, format='JPEG', quality=int(value))
        buffer.seek(0)
        x = np.array(Image.open(buffer).convert('RGB'), dtype=np.float32) / 255
    elif kind == 'occlusion':
        side = max(1, int(round(np.sqrt(value * h * w))))
        y = int(rng.integers(0, max(1, h - side + 1)))
        xx = int(rng.integers(0, max(1, w - side + 1)))
        x[y:y+side, xx:xx+side, :] = 0.5
    return torch.from_numpy(np.clip(x, 0, 1).copy()).permute(2, 0, 1).float()


@torch.no_grad()
def evaluate_corruptions(model, dataset, stats, device, output_dir, temperature=1.0,
                         tta=True, seed=0, max_images=0, types=None):
    """Dataset must contain clean TEST images only; never called for training."""
    model.eval()
    mean = dataset.mean
    std = dataset.std
    limit = min(len(dataset), max_images or len(dataset))
    if not limit:
        raise ValueError('Test dataset empty')
    raw_images, labels = [], []
    for i in range(limit):
        item = dataset[i]
        raw_images.append((item['image'] * std + mean).clamp(0, 1))
        labels.append(int(item['label']))
    clean_preds = []
    for raw in raw_images:
        p = predict_probabilities(model, ((raw-mean)/std).unsqueeze(0).to(device), stats,
                                  temperature, tta)
        clean_preds.append(p.cpu()[0].tolist())
    clean_score = classification_metrics(labels, probabilities=clean_preds)['macro_f1']
    rows = []
    for kind in (types or list(CORRUPTIONS)):
        for severity in range(1, 6):
            probs = []
            for i, raw in enumerate(raw_images):
                rng = np.random.default_rng(seed * 100000 + severity * 10000 + i +
                                            13 * list(CORRUPTIONS).index(kind))
                perturbed = corrupt_image(raw, kind, severity, rng)
                inp = ((perturbed - mean) / std).unsqueeze(0).to(device)
                p = predict_probabilities(model, inp, stats, temperature, tta)
                probs.append(p.cpu()[0].tolist())
            m = classification_metrics(labels, probabilities=probs)
            rows.append({'corruption': kind, 'severity': severity, 'parameter': CORRUPTIONS[kind][severity-1],
                         'n': limit, 'accuracy': m['accuracy'], 'macro_f1': m['macro_f1'],
                         'ece': m['ece'], 'clean_macro_f1': clean_score,
                         'degradation_ratio': (clean_score - m['macro_f1']) / max(clean_score, 1e-8)})
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / 'corruptions_by_severity.csv', index=False)
    slopes = {}
    for name, group in frame.groupby('corruption'):
        x, y = group.severity.to_numpy(dtype=float), group.macro_f1.to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope*x+intercept
        ss_res = float(((y-pred)**2).sum())
        ss_tot = float(((y-y.mean())**2).sum())
        slopes[name] = {'slope_macro_f1_per_level': float(slope),
                        'r2': float(1 - ss_res / ss_tot) if ss_tot else 1.0}
    save_json({'clean_macro_f1': clean_score, 'severity_regressions': slopes,
               'evaluated_test_images': limit, 'seed': seed,
               'warning': 'If max_images is nonzero, this is a subset diagnostic, not manuscript reproduction.'},
              out_dir / 'robustness_summary.json')
    return frame
