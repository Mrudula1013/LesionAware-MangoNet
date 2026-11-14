import torch
from mangonet.models.mango_net import LesionAwareMangoNet
from mangonet.compression.pruning import structurally_prune
from mangonet.utils.reproducibility import count_parameters


def test_model_shapes_and_attention_probabilities(tiny_config):
    model=LesionAwareMangoNet(tiny_config)
    out=model(torch.randn(2,3,224,224))
    assert out['logits'].shape==(2,8)
    assert out['attention'].shape==(2,1,14,14)
    assert torch.allclose(out['attention'].sum((2,3)),torch.ones(2,1),atol=1e-5)
    out['logits'].mean().backward()
    assert model.lesion_conv.weight.grad is not None
    assert torch.isfinite(model.lesion_conv.weight.grad).all()


def test_model_ablation_cnn_only(tiny_config):
    tiny_config.use_transformer=False
    tiny_config.lesion_attention=False
    model=LesionAwareMangoNet(tiny_config)
    output=model(torch.randn(1,3,224,224))
    assert output['logits'].shape==(1,8)
    assert torch.allclose(output['attention'].sum(),torch.tensor(1.0),atol=1e-5)


def test_structural_pruning_physically_reduces_parameters(tiny_config):
    model=LesionAwareMangoNet(tiny_config)
    original=count_parameters(model)
    small,manifest=structurally_prune(model,0.30)
    assert count_parameters(small)<original
    assert manifest['parameter_reduction_percent']>0
    assert small.config.ffn_dim<model.config.ffn_dim
    out=small(torch.randn(2,3,224,224))
    assert out['logits'].shape==(2,8)
    out['logits'].sum().backward()
    assert small.classifier.weight.grad is not None
