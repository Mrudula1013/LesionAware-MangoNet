"""LesionAware MangoNet: contextualised-token MIL head feeds the classifier."""
import torch
from torch import nn
from mangonet.config import ModelConfig
from .lightweight_cnn import LightweightCNN
from .channel_spatial_attention import ChannelSpatialAttention


class LesionAwareMangoNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if isinstance(config, ModelConfig) else ModelConfig.from_dict(config or {})
        c = self.config
        self.cnn = LightweightCNN(c.cnn_channels, c.multiscale_branch_channels)
        self.attention = ChannelSpatialAttention(c.cnn_channels[-1], c.channel_attention, c.spatial_attention)
        self.token_proj = nn.Linear(c.cnn_channels[-1], c.embed_dim)
        n = (c.image_size // 16) ** 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, c.embed_dim))
        self.position = nn.Parameter(torch.zeros(1, n + 1, c.embed_dim))
        if c.use_transformer:
            encoder = nn.TransformerEncoderLayer(
                d_model=c.embed_dim, nhead=c.attention_heads, dim_feedforward=c.ffn_dim,
                dropout=c.transformer_dropout, activation='gelu', batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder, c.transformer_layers, enable_nested_tensor=False)
        else:
            self.transformer = nn.Identity()
        self.output_norm = nn.LayerNorm(c.embed_dim)
        self.lesion_conv = nn.Conv2d(c.embed_dim, 1, kernel_size=1) if c.lesion_attention else None
        self.dropout = nn.Dropout(c.classifier_dropout)
        self.classifier = nn.Linear(c.embed_dim, c.num_classes)
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x):
        cnn_map = self.attention(self.cnn(x))
        batch, _, height, width = cnn_map.shape
        tokens = self.token_proj(cnn_map.flatten(2).transpose(1, 2))
        cls = self.class_token.expand(batch, -1, -1)
        all_tokens = torch.cat((cls, tokens), dim=1) + self.position
        encoded = self.output_norm(self.transformer(all_tokens))
        class_rep = encoded[:, 0] if self.config.use_transformer else encoded[:, 1:].mean(dim=1)
        spatial_tokens = encoded[:, 1:]
        spatial = spatial_tokens.transpose(1, 2).reshape(batch, -1, height, width)
        if self.lesion_conv is not None:
            raw_attention = torch.sigmoid(self.lesion_conv(spatial))
            attention = raw_attention / raw_attention.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
            lesion_rep = torch.sum(spatial * attention, dim=(2, 3))
            fused = (1.0 - self.config.fusion_weight) * class_rep + self.config.fusion_weight * lesion_rep
        else:
            raw_attention = torch.ones(batch, 1, height, width, device=x.device, dtype=x.dtype)
            attention = raw_attention / (height * width)
            lesion_rep = spatial.mean(dim=(2, 3))
            fused = class_rep
        logits = self.classifier(self.dropout(fused))
        return {
            'logits': logits, 'attention': attention, 'attention_raw': raw_attention,
            'spatial_tokens': spatial, 'cnn_features': cnn_map, 'global_embedding': class_rep,
            'lesion_embedding': lesion_rep,
        }


class LogitsOnly(nn.Module):
    """Tensor-only forward wrapper for ONNX export and Torch profiling."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images)['logits']
