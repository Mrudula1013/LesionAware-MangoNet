"""Grad-CAM, Score-CAM, lesion attention, agreement and perturbation faithfulness."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from mangonet.evaluation.calibration import predict_probabilities


def normalize_map(x, eps=1e-8):
    x = x - x.amin(dim=(-2, -1), keepdim=True)
    return x / x.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)


def gradcam(model, normalized_image, class_id=None):
    """Uses the final attention-refined 14x14 CNN map, NOT Grad-CAM on logits."""
    model.eval()
    x = normalized_image.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        output = model(x)
        cls = int(output['logits'].argmax(dim=1)[0]) if class_id is None else int(class_id)
        features = output['cnn_features']
        grad = torch.autograd.grad(output['logits'][0, cls], features)[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        heat = (weights * features).sum(dim=1, keepdim=True).relu()
        heat = F.interpolate(heat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return normalize_map(heat).detach(), cls


@torch.no_grad()
def scorecam(model, normalized_image, stats, class_id, top_channels=32):
    """Score-CAM uses original image masked with upsampled activation channels."""
    model.eval()
    x = normalized_image
    output = model(x)
    features = output['cnn_features']
    importance = features.flatten(2).std(dim=-1)[0]
    top = torch.topk(importance, min(top_channels, features.shape[1])).indices
    activations = features[:, top]
    activations = F.interpolate(activations, size=x.shape[-2:], mode='bilinear', align_corners=False)
    masks = normalize_map(activations)
    mean = torch.as_tensor(stats['mean'], device=x.device).view(1, 3, 1, 1)
    std = torch.as_tensor(stats['std'], device=x.device).view(1, 3, 1, 1)
    raw = (x*std+mean).clamp(0, 1)
    masked = raw * masks.transpose(0, 1)
    masked_norm = (masked - mean) / std
    probabilities = []
    for mini_batch in masked_norm.split(8):
        probabilities.append(model(mini_batch)['logits'].softmax(dim=1)[:, class_id])
    scores = torch.cat(probabilities)
    heat = (scores.view(-1, 1, 1) * masks[0]).sum(dim=0, keepdim=True)
    return normalize_map(heat.unsqueeze(0)).detach()


def cosine_agreement(a, b):
    a = torch.as_tensor(a).flatten().float()
    b = torch.as_tensor(b).flatten().float()
    return float(F.cosine_similarity(a[None], b[None]).item())


@torch.no_grad()
def faithfulness(model, x, heatmap, class_id, steps=20, selected_fraction=0.20):
    """Deletion lower AUC is better; insertion higher AUC is better.

    Baseline: Gaussian-blurred input in *normalised model space*, equivalent to
    blurred raw image under a fixed affine per-channel normalisation.
    """
    model.eval()
    blur = TF.gaussian_blur(x, kernel_size=[31, 31], sigma=[10.0, 10.0])
    heat = F.interpolate(heatmap, size=x.shape[-2:], mode='bilinear', align_corners=False)
    indices = heat.flatten().argsort(descending=True)
    n_pixels = indices.numel()
    deletion, insertion = [], []
    original_p = float(model(x)['logits'].softmax(dim=1)[0, class_id])
    for step in range(steps + 1):
        k = round(step / steps * n_pixels)
        flat = torch.zeros(n_pixels, device=x.device, dtype=x.dtype)
        flat[indices[:k]] = 1
        mask = flat.reshape(1, 1, *x.shape[-2:])
        x_deletion = x * (1 - mask) + blur * mask
        x_insertion = x * mask + blur * (1 - mask)
        deletion.append(float(model(x_deletion)['logits'].softmax(1)[0, class_id]))
        insertion.append(float(model(x_insertion)['logits'].softmax(1)[0, class_id]))
    fractions = np.linspace(0, 1, steps+1)
    sel_mask = torch.zeros(n_pixels, device=x.device, dtype=x.dtype)
    sel_mask[indices[:round(n_pixels*selected_fraction)]] = 1
    sel_mask = sel_mask.reshape(1, 1, *x.shape[-2:])
    retained = x * sel_mask + blur * (1 - sel_mask)
    kept_p = float(model(retained)['logits'].softmax(1)[0, class_id])
    return {'deletion_auc_lower_better': float(np.trapezoid(deletion, fractions)),
            'insertion_auc_higher_better': float(np.trapezoid(insertion, fractions)),
            'average_drop_percent_lower_better': 100 * max(0, original_p-kept_p) / max(original_p, 1e-8),
            'average_increase': int(kept_p > original_p),
            'curve_deletion': deletion, 'curve_insertion': insertion}


def generate_explanations(model, normalized_image, stats, top_channels=32, run_faithfulness=True):
    """One 1x3x224x224 normalised tensor; returns heatmaps and numeric scores."""
    model.eval()
    x = normalized_image
    with torch.no_grad():
        output = model(x)
        cls = int(output['logits'].argmax(dim=1)[0])
        lesion = normalize_map(F.interpolate(output['attention'], size=x.shape[-2:],
                                             mode='bilinear', align_corners=False)).detach()
    grad, _ = gradcam(model, x, cls)
    score = scorecam(model, x, stats, cls, top_channels)
    measures = {'class_id': cls, 'attention_gradcam_cosine': cosine_agreement(lesion, grad),
                'attention_scorecam_cosine': cosine_agreement(lesion, score)}
    if run_faithfulness:
        for name, heat in [('lesion', lesion), ('gradcam', grad), ('scorecam', score)]:
            values = faithfulness(model, x, heat, cls)
            measures[name] = values
    return {'lesion': lesion, 'gradcam': grad, 'scorecam': score}, measures


def save_explanation_figure(normalized_image, stats, maps, filename, dpi=600):
    mean = torch.tensor(stats['mean'], device=normalized_image.device).view(1, 3, 1, 1)
    std = torch.tensor(stats['std'], device=normalized_image.device).view(1, 3, 1, 1)
    rgb = (normalized_image * std + mean).clamp(0, 1)[0].detach().cpu().permute(1, 2, 0).numpy()
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    axes[0].imshow(rgb)
    for ax, name in zip(axes[1:], ('lesion', 'gradcam', 'scorecam')):
        ax.imshow(rgb)
        ax.imshow(maps[name][0, 0].cpu().numpy(), cmap='jet', alpha=0.45, vmin=0, vmax=1)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.15)
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
