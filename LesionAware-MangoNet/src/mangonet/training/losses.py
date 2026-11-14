"""Classification + focal + attention entropy + masked aligned consistency."""
import torch
import torch.nn.functional as F
from mangonet.data.augmentation import masked_attention_consistency


def class_balancing_weights(labels, num_classes=8):
    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long), minlength=num_classes).float()
    nonzero = counts > 0
    inv = torch.zeros_like(counts)
    inv[nonzero] = counts[nonzero].sum() / counts[nonzero]
    # Normalise to mean 1 over represented classes; missing classes receive 0.
    inv = inv * (nonzero.sum() / inv.sum().clamp_min(1e-8))
    return inv


def focal_loss(logits, target, alpha=None, gamma=2.0):
    log_probs = F.log_softmax(logits.float(), dim=1)
    log_pt = log_probs.gather(1, target[:, None]).squeeze(1)
    pt = log_pt.exp()
    focal = -((1.0 - pt).clamp_min(0) ** gamma) * log_pt
    if alpha is not None:
        focal = focal * alpha[target]
    return focal.mean()


def attention_entropy(attention):
    p = attention.float().flatten(1).clamp_min(1e-8)
    return -(p * torch.log(p)).sum(dim=1).mean()


def combined_loss(output0, output1, labels, matrices, valid_views, alpha, epoch, config):
    """Both augmented views receive the same image-level classification label."""
    ce = (F.cross_entropy(output0['logits'], labels) +
          F.cross_entropy(output1['logits'], labels)) / 2.0
    focal = (focal_loss(output0['logits'], labels, alpha, config['focal_gamma']) +
             focal_loss(output1['logits'], labels, alpha, config['focal_gamma'])) / 2.0
    if output0.get('attention') is None:
        entropy = ce.new_zeros(())
        consistency = ce.new_zeros(())
    else:
        entropy = (attention_entropy(output0['attention']) + attention_entropy(output1['attention'])) / 2
        stacked = torch.stack((output0['attention'], output1['attention']), dim=1)
        consistency = masked_attention_consistency(stacked, matrices, valid_views)
    ramp = min(1.0, (epoch + 1) / max(1, config['loss_ramp_epochs']))
    total = (config['lambda_ce'] * ce + config['lambda_focal'] * focal +
             ramp * config['lambda_entropy'] * entropy +
             ramp * config['lambda_consistency'] * consistency)
    return total, {k: float(v.detach()) for k, v in
                   [('total', total), ('ce', ce), ('focal', focal),
                    ('entropy', entropy), ('consistency', consistency)]}
