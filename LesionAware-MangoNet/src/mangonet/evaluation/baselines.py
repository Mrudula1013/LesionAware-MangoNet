"""Comparable CNN, ViT, Swin and mobile hybrid classifier registry.

Pretrained weights are requested explicitly; no silent architecture or
pretraining substitutions. Generic models share all source-only splits and
augmentation/lr/early-stop settings with the proposed architecture.
"""
from pathlib import Path
import copy
import csv
import torch
from torch import nn
from mangonet.config import ModelConfig
from mangonet.data.dataset import load_manifest, subset_manifest, compute_source_stats, MangoImageDataset
from mangonet.training.trainer import make_loader, eval_model, lr_for_epoch
from mangonet.training.losses import class_balancing_weights, focal_loss
from mangonet.utils.reproducibility import seed_everything, get_device, save_json, checkpoint_load

TORCHVISION_MODELS = {
    'resnet50': 'resnet50', 'densenet121': 'densenet121',
    'mobilenet_v3_large': 'mobilenet_v3_large',
    'efficientnet_b0': 'efficientnet_b0', 'vit_b_16': 'vit_b_16',
    'swin_t': 'swin_t',
}
TIMM_MODELS = {
    'mobilevit_s': 'mobilevit_s', 'convformer_s18': 'convformer_s18',
    'efficientformer_l1': 'efficientformer_l1', 'edgenext_small': 'edgenext_small',
    'vit_base_patch16_224': 'vit_base_patch16_224',
    'swin_tiny_patch4_window7_224': 'swin_tiny_patch4_window7_224',
}


class ClassificationWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        logits = self.backbone(x)
        return {'logits': logits}


def make_baseline(name, num_classes=8, pretrained=False):
    if name in TORCHVISION_MODELS:
        from torchvision import models
        weights = 'DEFAULT' if pretrained else None
        try:
            network = models.get_model(TORCHVISION_MODELS[name], weights=weights)
        except Exception as error:
            raise RuntimeError(f'Could not construct torchvision model {name} (pretrained={pretrained}): {error}') from error
        if name == 'resnet50':
            network.fc = nn.Linear(network.fc.in_features, num_classes)
        elif name == 'densenet121':
            network.classifier = nn.Linear(network.classifier.in_features, num_classes)
        elif name in ('mobilenet_v3_large', 'efficientnet_b0'):
            network.classifier[-1] = nn.Linear(network.classifier[-1].in_features, num_classes)
        elif name == 'vit_b_16':
            network.heads.head = nn.Linear(network.heads.head.in_features, num_classes)
        elif name == 'swin_t':
            network.head = nn.Linear(network.head.in_features, num_classes)
    elif name in TIMM_MODELS:
        try:
            import timm
        except ImportError as error:
            raise ImportError("Timm baselines need: pip install -e '.[baselines]'") from error
        if name not in timm.list_models(name):
            raise RuntimeError(f'Timm model {name} unavailable in this installed timm version. Choose a verified registry name; no silent substitution.')
        network = timm.create_model(TIMM_MODELS[name], pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown baseline {name}; choices: {sorted((*TORCHVISION_MODELS, *TIMM_MODELS))}')
    return ClassificationWrapper(network)


def train_baseline(name, data_root, manifest, output_dir, source='combined', seed=0,
                   config=None, pretrained=False, device='auto', epochs=None,
                   batch_size=None, num_workers=0, max_train_batches=0):
    seed_everything(seed)
    cfg = {'epochs': 100, 'batch_size': 32, 'learning_rate': 3e-4, 'min_learning_rate': 1e-6,
           'weight_decay': 0.05, 'warmup_epochs': 5, 'early_stopping_patience': 15,
           'focal_gamma': 2.0, 'lambda_ce': 1.0, 'lambda_focal': 0.5, 'mixed_precision': True}
    cfg.update(config or {})
    if epochs is not None: cfg['epochs'] = epochs
    if batch_size is not None: cfg['batch_size'] = batch_size
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    frame = load_manifest(manifest)
    train_frame = subset_manifest(frame, source, 'train')
    val_frame = subset_manifest(frame, source, 'val')
    if train_frame.empty or val_frame.empty:
        raise ValueError('Baseline source train or validation split empty')
    stats = compute_source_stats(frame, data_root, source, save_to=out / 'source_stats.json')
    train_ds = MangoImageDataset(train_frame, data_root, stats, pairs=True)
    val_ds = MangoImageDataset(val_frame, data_root, stats)
    train_loader = make_loader(train_ds, cfg['batch_size'], True, num_workers, seed)
    val_loader = make_loader(val_ds, cfg['batch_size'], False, num_workers, seed)
    model = make_baseline(name, 8, pretrained).to(get_device(device))
    device = next(model.parameters()).device
    alpha = class_balancing_weights(train_frame.label_id.tolist()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    amp = cfg['mixed_precision'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best, wait, hist = -float('inf'), 0, []
    for epoch in range(cfg['epochs']):
        lr = lr_for_epoch(epoch, cfg['epochs'], cfg['warmup_epochs'], cfg['learning_rate'], cfg['min_learning_rate'])
        for group in optimizer.param_groups: group['lr'] = lr
        model.train(); losses = []
        for step, batch in enumerate(train_loader):
            if max_train_batches and step >= max_train_batches: break
            x = batch['views'].to(device); labels = batch['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                z0, z1 = model(x[:,0])['logits'], model(x[:,1])['logits']
                ce = (nn.functional.cross_entropy(z0, labels) + nn.functional.cross_entropy(z1, labels))/2
                foc = (focal_loss(z0, labels, alpha) + focal_loss(z1, labels, alpha))/2
                loss = cfg['lambda_ce']*ce + cfg['lambda_focal']*foc
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach()))
        metrics = eval_model(model, val_loader, device)
        score = metrics['macro_f1']
        hist.append({'epoch': epoch+1, 'train_loss': sum(losses)/max(1,len(losses)),
                     'val_macro_f1': score, 'lr': lr})
        print(f'[baseline={name} seed={seed}] epoch={epoch+1} Macro-F1={score:.4f}', flush=True)
        if score > best + 1e-8:
            best, wait = score, 0
            torch.save({'architecture': 'baseline', 'baseline_name': name,
                        'pretrained': pretrained, 'source': source, 'seed': seed,
                        'state_dict': model.state_dict(), 'source_stats': stats,
                        'validation_metrics': metrics, 'training_config': cfg,
                        'smoke_test': bool(max_train_batches)}, out / 'best.pt')
        else:
            wait += 1
            if wait >= cfg['early_stopping_patience']: break
    with (out/'history.csv').open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=hist[0]); wr.writeheader(); wr.writerows(hist)
    return out/'best.pt'


def load_baseline_checkpoint(path, device='cpu'):
    ckpt = checkpoint_load(path)
    if ckpt.get('architecture') != 'baseline':
        raise ValueError('Not a baseline checkpoint')
    model = make_baseline(ckpt['baseline_name'], pretrained=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(get_device(device)).eval()
    return model, ckpt
