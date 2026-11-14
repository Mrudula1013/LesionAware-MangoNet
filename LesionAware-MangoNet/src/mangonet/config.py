"""Configuration and manuscript-locked class taxonomy."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import yaml

CLASSES = (
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
    'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould',
)
SOURCES = ('MangoLeafBD', 'MLD24')
HOLDOUT = 'MLD-Field'
ALL_DATASETS = (*SOURCES, HOLDOUT)


@dataclass
class ModelConfig:
    image_size: int = 224
    num_classes: int = 8
    cnn_channels: tuple = (32, 64, 96, 128)
    multiscale_branch_channels: int = 64
    embed_dim: int = 256
    transformer_layers: int = 4
    attention_heads: int = 4
    ffn_dim: int = 512
    transformer_dropout: float = 0.1
    classifier_dropout: float = 0.3
    channel_attention: bool = True
    spatial_attention: bool = True
    lesion_attention: bool = True
    use_transformer: bool = True
    fusion_weight: float = 0.5

    def __post_init__(self):
        self.cnn_channels = tuple(map(int, self.cnn_channels))
        assert len(self.cnn_channels) == 4 and all(x > 0 for x in self.cnn_channels)
        assert self.embed_dim % self.attention_heads == 0
        assert self.image_size % 16 == 0
        assert self.num_classes == len(CLASSES)
        assert 0 <= self.fusion_weight <= 1

    @classmethod
    def from_dict(cls, values):
        legal = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(values).items() if k in legal})

    def to_dict(self):
        result = asdict(self)
        result['cnn_channels'] = list(self.cnn_channels)
        return result


def read_yaml(path):
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def load_model_config(path=None):
    if path is None:
        return ModelConfig()
    return ModelConfig.from_dict(read_yaml(path))


def project_root():
    return Path(__file__).resolve().parents[2]
