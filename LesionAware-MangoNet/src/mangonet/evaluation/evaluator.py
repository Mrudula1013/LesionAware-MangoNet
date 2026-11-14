"""Independent in-dataset and zero-shot source->target evaluation."""
from pathlib import Path
import pandas as pd
import torch
from mangonet.config import CLASSES, HOLDOUT, SOURCES
from mangonet.data.dataset import load_manifest, subset_manifest, MangoImageDataset
from mangonet.training.trainer import load_trained_model, make_loader
from mangonet.evaluation.calibration import (obtain_validation_temperature,
                                               predict_probabilities)
from mangonet.evaluation.metrics import classification_metrics
from mangonet.utils.reproducibility import get_device, save_json


def check_transfer_leakage(frame, source, target):
    """Fail closed if a source TRAIN/VAL leaf may duplicate a target test leaf."""
    if source == 'combined':
        return
    source_groups = set(frame[(frame.dataset == source) & frame.split.isin(['train', 'val'])].group_id)
    target_groups = set(frame[(frame.dataset == target) & (frame.split == 'test')].group_id)
    overlap = source_groups.intersection(target_groups)
    if overlap:
        raise ValueError(f'Potential cross-domain leakage ({len(overlap)} duplicate groups). Inspect duplicate_review.csv; do not use contaminated transfer results.')


def evaluate_checkpoint(checkpoint_path, manifest, data_root, out_dir, target=None,
                        batch_size=32, num_workers=0, device='auto',
                        temperature=None, tta=True, max_images=0):
    device = get_device(device)
    model, ckpt = load_trained_model(checkpoint_path, device)
    frame = load_manifest(manifest)
    source = ckpt['source']
    if target is not None and target != source:
        check_transfer_leakage(frame, source, target)
        test_frame = subset_manifest(frame, source, 'test', target=target)
    else:
        test_frame = subset_manifest(frame, source, 'test')
    if max_images:
        test_frame = test_frame.head(max_images)
    if test_frame.empty:
        raise ValueError(f'No test images for source={source}, target={target}')
    val_frame = subset_manifest(frame, source, 'val')
    stats = ckpt['source_stats']
    if temperature is None:
        if val_frame.empty:
            raise ValueError('Source validation data is required for temperature calibration')
        val_ds = MangoImageDataset(val_frame, data_root, stats)
        temperature = obtain_validation_temperature(model, val_ds, device, batch_size)
    test_ds = MangoImageDataset(test_frame, data_root, stats)
    loader = make_loader(test_ds, batch_size, False, num_workers, ckpt.get('seed', 0))
    labels, predictions, confidences, paths, vectors = [], [], [], [], []
    for batch in loader:
        probs = predict_probabilities(model, batch['image'].to(device), stats, temperature, tta)
        labels.extend(batch['label'].tolist())
        paths.extend(batch['path'])
        preds = probs.argmax(dim=1).cpu().tolist()
        predictions.extend(preds)
        confidences.extend(probs.max(dim=1).values.cpu().tolist())
        vectors.extend(probs.cpu().tolist())
    metrics = classification_metrics(labels, predictions, vectors)
    result = {'source': source, 'target': target or source, 'seed': ckpt.get('seed'),
              'temperature': float(temperature), 'tta': bool(tta),
              'smoke_test': bool(ckpt.get('smoke_test') or max_images), 'metrics': metrics}
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_json(result, output / 'metrics.json')
    records = pd.DataFrame({'path': paths, 'true': labels, 'predicted': predictions,
                            'confidence': confidences, 'true_class': [CLASSES[i] for i in labels],
                            'predicted_class': [CLASSES[i] for i in predictions]})
    for i, name in enumerate(CLASSES):
        records[f'p_{i}_{name.replace(" ", "_")}'] = [v[i] for v in vectors]
    records.to_csv(output / 'predictions.csv', index=False)
    return result
