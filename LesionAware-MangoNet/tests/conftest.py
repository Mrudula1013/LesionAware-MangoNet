import pytest
import torch
from mangonet.config import ModelConfig

@pytest.fixture(scope='session', autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)

@pytest.fixture
def tiny_config():
    return ModelConfig(cnn_channels=(8, 12, 16, 20), multiscale_branch_channels=8,
                       embed_dim=64, transformer_layers=1, attention_heads=2, ffn_dim=128)
