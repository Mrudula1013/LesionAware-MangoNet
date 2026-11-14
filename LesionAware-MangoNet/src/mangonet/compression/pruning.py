"""Actual L1 structural pruning: physically rebuilds CNN and Transformer FFN.

The 256-dimensional attention embedding and four attention heads are preserved
to keep MHSA shape-correct; complete convolutional output filters and FFN hidden
units are physically removed and their connected weights sliced consistently.
"""
from dataclasses import replace
import math
import torch
from mangonet.models.mango_net import LesionAwareMangoNet
from mangonet.utils.reproducibility import count_parameters


def top_l1_filters(weights, fraction):
    k = max(1, math.ceil((1 - fraction) * weights.shape[0]))
    scores = weights.detach().abs().flatten(1).sum(dim=1)
    return torch.topk(scores, k, largest=True).indices.sort().values.cpu()


def _slice(tensor, rows=None, cols=None):
    if rows is not None:
        tensor = tensor.index_select(0, rows.to(tensor.device))
    if cols is not None:
        tensor = tensor.index_select(1, cols.to(tensor.device))
    return tensor.clone()


def structurally_prune(model, fraction=0.30):
    """Return smaller, weight-transferred model and exact pruning manifest."""
    if not isinstance(model, LesionAwareMangoNet):
        raise TypeError('Structured pruning implemented for LesionAwareMangoNet only')
    if not 0 < fraction < 1:
        raise ValueError('fraction must be between 0 and 1')
    cfg = model.config
    old = model.cpu().eval().state_dict()
    old_channels = cfg.cnn_channels
    masks = [top_l1_filters(old['cnn.stem.0.weight'], fraction),
             top_l1_filters(old['cnn.blocks.1.pointwise.weight'], fraction),
             top_l1_filters(old['cnn.blocks.3.pointwise.weight'], fraction),
             top_l1_filters(old['cnn.blocks.5.pointwise.weight'], fraction)]
    new_channels = tuple(int(len(v)) for v in masks)
    branch_masks = [top_l1_filters(old[f'cnn.multiscale.branches.{i}.pointwise.weight'], fraction)
                    for i in range(3)]
    b_new = len(branch_masks[0])
    ffn_new = max(1, math.ceil((1-fraction) * cfg.ffn_dim))
    new_cfg = replace(cfg, cnn_channels=new_channels, multiscale_branch_channels=b_new, ffn_dim=ffn_new)
    reduced = LesionAwareMangoNet(new_cfg)
    state = reduced.state_dict()
    # All non-pruned layers have identical shapes (e.g. MHSA projections, layer norms).
    for name, value in state.items():
        if name in old and old[name].shape == value.shape:
            state[name] = old[name].clone()

    def set_slice(name, rows=None, cols=None):
        state[name] = _slice(old[name], rows, cols)
        if state[name].shape != reduced.state_dict()[name].shape:
            raise RuntimeError(f'Incorrect pruning shape for {name}: {state[name].shape} vs {reduced.state_dict()[name].shape}')

    def bn(prefix, indices):
        for suffix in ('weight', 'bias', 'running_mean', 'running_var'):
            set_slice(f'{prefix}.{suffix}', indices)

    set_slice('cnn.stem.0.weight', masks[0])
    bn('cnn.stem.1', masks[0])
    pair = [(masks[0], masks[0]), (masks[0], masks[1]),
            (masks[1], masks[1]), (masks[1], masks[2]),
            (masks[2], masks[2]), (masks[2], masks[3])]
    for i, (inp, out) in enumerate(pair):
        prefix = f'cnn.blocks.{i}'
        set_slice(f'{prefix}.depthwise.weight', inp)
        set_slice(f'{prefix}.pointwise.weight', out, inp)
        bn(f'{prefix}.bn', out)
    for i, out in enumerate(branch_masks):
        prefix = f'cnn.multiscale.branches.{i}'
        set_slice(f'{prefix}.depthwise.weight', masks[3])
        set_slice(f'{prefix}.pointwise.weight', out, masks[3])
        bn(f'{prefix}.bn', out)
    old_b = cfg.multiscale_branch_channels
    concatenated = torch.cat([idx + i*old_b for i, idx in enumerate(branch_masks)])
    set_slice('cnn.multiscale.project.weight', masks[3], concatenated)
    bn('cnn.multiscale.bn', masks[3])
    if cfg.channel_attention:
        hidden_size = max(1, new_channels[-1] // 8)
        old_hidden = old['attention.channel_mlp.0.weight'].shape[0]
        hidden = torch.topk(old['attention.channel_mlp.0.weight'].abs().sum(1),
                            min(hidden_size, old_hidden), largest=True).indices.sort().values.cpu()
        set_slice('attention.channel_mlp.0.weight', hidden, masks[3])
        set_slice('attention.channel_mlp.0.bias', hidden)
        set_slice('attention.channel_mlp.2.weight', masks[3], hidden)
        set_slice('attention.channel_mlp.2.bias', masks[3])
    set_slice('token_proj.weight', cols=masks[3])
    ffn_masks = {}
    if cfg.use_transformer:
        for i in range(cfg.transformer_layers):
            prefix = f'transformer.layers.{i}'
            chosen = top_l1_filters(old[f'{prefix}.linear1.weight'], fraction)
            ffn_masks[str(i)] = chosen.tolist()
            set_slice(f'{prefix}.linear1.weight', chosen)
            set_slice(f'{prefix}.linear1.bias', chosen)
            set_slice(f'{prefix}.linear2.weight', cols=chosen)
    reduced.load_state_dict(state, strict=True)
    before, after = count_parameters(model), count_parameters(reduced)
    manifest = {'pruning_fraction_threshold': fraction,
                'old_cnn_channels': list(old_channels), 'new_cnn_channels': list(new_channels),
                'old_branch_channels': old_b, 'new_branch_channels': b_new,
                'old_ffn_hidden': cfg.ffn_dim, 'new_ffn_hidden': ffn_new,
                'embedding_dimension_preserved': cfg.embed_dim,
                'old_parameters': before, 'new_parameters': after,
                'parameter_reduction_percent': 100*(before-after)/before,
                'notes': 'Convolutional filters and FFN hidden dimensions physically removed; MHSA embedding unchanged.'}
    return reduced, manifest
