"""Source-validation temperature scaling; per-view calibration prior to TTA averaging."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from mangonet.training.trainer import make_loader


def fit_temperature(logits, labels, max_iter=80):
    logits = torch.as_tensor(logits, dtype=torch.float64, device='cpu')
    labels = torch.as_tensor(labels, dtype=torch.long, device='cpu')
    if len(labels) == 0:
        raise ValueError('Temperature fitting needs source validation labels')
    log_temp = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temp.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temp.detach().exp().clamp(0.05, 20.0))


def obtain_validation_temperature(model, dataset, device, batch_size=32, max_samples=0):
    loader = make_loader(dataset, batch_size, False)
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            z = model(batch['image'].to(device))['logits']
            logits.append(z.cpu())
            labels.append(batch['label'])
            if max_samples and sum(len(v) for v in labels) >= max_samples:
                break
    if not logits:
        raise RuntimeError('No source-validation samples for temperature calibration')
    all_logits = torch.cat(logits)[:max_samples or None]
    all_labels = torch.cat(labels)[:max_samples or None]
    return fit_temperature(all_logits, all_labels)


def make_tta_views(images, mean, std):
    """Input normalized; return identity/hflip/vflip/90%-crop/1.1 brightness."""
    mean = torch.as_tensor(mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    raw = (images * std + mean).clamp(0, 1)
    h, w = raw.shape[-2:]
    s = int(round(min(h, w) * 0.9))
    top, left = (h - s) // 2, (w - s) // 2
    crop = F.interpolate(raw[:, :, top:top+s, left:left+s], size=(h, w),
                         mode='bilinear', align_corners=False)
    views = [raw, raw.flip(-1), raw.flip(-2), crop, (raw * 1.1).clamp(0, 1)]
    return [(view - mean) / std for view in views]


@torch.no_grad()
def predict_probabilities(model, images, stats, temperature=1.0, tta=True):
    model.eval()
    views = make_tta_views(images, stats['mean'], stats['std']) if tta else [images]
    probs = [F.softmax(model(view)['logits'] / temperature, dim=1) for view in views]
    return torch.stack(probs).mean(dim=0)
