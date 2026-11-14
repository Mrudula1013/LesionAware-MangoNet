"""Checkpointed, source-only training for LesionAware MangoNet and baselines."""
from collections import Counter
from pathlib import Path
import csv
import math
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from mangonet.config import ModelConfig, read_yaml
from mangonet.data.dataset import load_manifest, subset_manifest, compute_source_stats, MangoImageDataset
from mangonet.models.mango_net import LesionAwareMangoNet
from mangonet.training.losses import combined_loss, class_balancing_weights
from mangonet.evaluation.metrics import classification_metrics
from mangonet.utils.reproducibility import seed_everything, save_json, get_device


def make_loader(dataset, batch_size, shuffle=False, num_workers=0, seed=0):
    generator = torch.Generator().manual_seed(seed)

    def worker_init(worker_id):
        seed_value = torch.initial_seed() % (2 ** 32)
        import numpy as np, random
        np.random.seed(seed_value)
        random.seed(seed_value)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=worker_init, generator=generator)


def eval_model(model, loader, device, max_batches=0):
    model.eval()
    actual, predictions, probs = [], [], []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            logits = model(batch['image'].to(device))['logits']
            p = logits.softmax(dim=1).cpu()
            probs.extend(p.tolist())
            predictions.extend(p.argmax(1).tolist())
            actual.extend(batch['label'].tolist())
    if not actual:
        raise RuntimeError('Validation data loader is empty; cannot select checkpoint')
    return classification_metrics(actual, predictions, probs)


def lr_for_epoch(epoch, epochs, warmup, max_lr, min_lr):
    if epoch < warmup:
        return max_lr * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, epochs - warmup - 1)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def train_run(data_root, manifest, output_dir, source='combined', seed=0, model_config=None,
              train_config=None, device='auto', epochs=None, batch_size=None,
              num_workers=None, max_train_batches=0, max_val_batches=0):
    model_cfg = model_config if isinstance(model_config, ModelConfig) else ModelConfig.from_dict(model_config or {})
    cfg = {'epochs': 100, 'batch_size': 32, 'num_workers': 4, 'learning_rate': 3e-4,
           'min_learning_rate': 1e-6, 'weight_decay': 0.05, 'warmup_epochs': 5,
           'early_stopping_patience': 15, 'focal_gamma': 2.0, 'lambda_ce': 1.0,
           'lambda_focal': 0.5, 'lambda_entropy': 0.05,
           'lambda_consistency': 0.5, 'loss_ramp_epochs': 10, 'mixed_precision': True}
    cfg.update(train_config or {})
    if epochs is not None: cfg['epochs'] = epochs
    if batch_size is not None: cfg['batch_size'] = batch_size
    if num_workers is not None: cfg['num_workers'] = num_workers
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = get_device(device)
    frame = load_manifest(manifest)
    train_frame = subset_manifest(frame, source, 'train')
    val_frame = subset_manifest(frame, source, 'val')
    if train_frame.empty or val_frame.empty:
        raise ValueError(f'Source {source}: training ({len(train_frame)}) or validation ({len(val_frame)}) is empty. Check split groups.')
    stats = compute_source_stats(frame, data_root, source, model_cfg.image_size,
                                 save_to=output_dir / 'source_stats.json')
    train_data = MangoImageDataset(train_frame, data_root, stats, pairs=True)
    val_data = MangoImageDataset(val_frame, data_root, stats, pairs=False)
    train_loader = make_loader(train_data, int(cfg['batch_size']), True, int(cfg['num_workers']), seed)
    val_loader = make_loader(val_data, int(cfg['batch_size']), False, int(cfg['num_workers']), seed)
    model = LesionAwareMangoNet(model_cfg).to(device)
    weights = class_balancing_weights(train_frame.label_id.tolist(), model_cfg.num_classes).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']),
                                  weight_decay=float(cfg['weight_decay']))
    amp = bool(cfg['mixed_precision']) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, best, wait = [], -float('inf'), 0
    for epoch in range(int(cfg['epochs'])):
        start = time.perf_counter()
        lr = lr_for_epoch(epoch, int(cfg['epochs']), int(cfg['warmup_epochs']),
                          float(cfg['learning_rate']), float(cfg['min_learning_rate']))
        for group in optimiser.param_groups:
            group['lr'] = lr
        model.train()
        total_losses = []
        for batch_index, batch in enumerate(train_loader):
            if max_train_batches and batch_index >= max_train_batches:
                break
            views = batch['views'].to(device)
            labels = batch['label'].to(device)
            matrices = batch['matrices'].to(device)
            valid = batch['valid_views'].to(device)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                out0 = model(views[:, 0]); out1 = model(views[:, 1])
                loss, components = combined_loss(out0, out1, labels, matrices, valid, weights, epoch, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
            total_losses.append(components['total'])
        metrics = eval_model(model, val_loader, device, max_batches=max_val_batches)
        score = float(metrics['macro_f1'])
        log = {'epoch': epoch + 1, 'lr': lr,
               'train_loss': float(sum(total_losses) / max(1, len(total_losses))),
               'val_macro_f1': score, 'val_accuracy': metrics['accuracy'],
               'val_ece': metrics['ece'], 'seconds': round(time.perf_counter() - start, 2)}
        history.append(log)
        print(f"[seed {seed}] epoch {epoch+1}/{cfg['epochs']} loss={log['train_loss']:.4f} val_macro_f1={score:.4f} lr={lr:.2e}", flush=True)
        if score > best + 1e-8:
            best, wait = score, 0
            checkpoint = {
                'state_dict': model.state_dict(), 'model_config': model_cfg.to_dict(),
                'training_config': cfg, 'source_stats': stats, 'source': source,
                'seed': seed, 'epoch': epoch + 1, 'validation_metrics': metrics,
                'manifest': str(manifest), 'smoke_test': bool(max_train_batches or max_val_batches),
                'architecture': 'LesionAwareMangoNet',
            }
            torch.save(checkpoint, output_dir / 'best.pt')
        else:
            wait += 1
            if wait >= int(cfg['early_stopping_patience']):
                print(f'Early stop after {epoch+1} epochs', flush=True)
                break
    with (output_dir / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    save_json({'best_val_macro_f1': best, 'epochs_executed': len(history),
               'source': source, 'seed': seed, 'smoke_test': bool(max_train_batches or max_val_batches),
               'warning': 'Training is complete, but test-set results need the separate evaluation step.'},
              output_dir / 'run_summary.json')
    return output_dir / 'best.pt'


def load_trained_model(checkpoint_path, device='cpu'):
    from mangonet.utils.reproducibility import checkpoint_load
    ckpt = checkpoint_load(checkpoint_path, map_location='cpu')
    if ckpt.get('architecture', 'LesionAwareMangoNet') != 'LesionAwareMangoNet':
        raise ValueError('A non-MangoNet checkpoint was provided; use baseline evaluation commands.')
    model = LesionAwareMangoNet(ModelConfig.from_dict(ckpt['model_config']))
    model.load_state_dict(ckpt['state_dict'])
    model.to(get_device(device)).eval()
    return model, ckpt
