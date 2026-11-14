import random
import torch
from mangonet.data.augmentation import (PairedAugmentor,masked_attention_consistency,
                                         view_to_source,warp_source_to_view,sample_affine)
from mangonet.training.losses import combined_loss
from mangonet.models.mango_net import LesionAwareMangoNet


def test_consistency_zero_identical_maps():
    p=torch.rand(2,1,14,14,requires_grad=True)
    p=p/p.sum((2,3),keepdim=True)
    a=torch.stack([p,p],dim=1)
    m=torch.eye(3).repeat(2,2,1,1)
    v=torch.ones(2,2,1,224,224)
    loss=masked_attention_consistency(a,m,v)
    assert loss.item()<1e-6


def test_flip_alignment_and_gradients():
    p=torch.zeros(1,1,14,14)
    p[:,:,4:7,2:5]=1.0
    p=p/p.sum()
    m=torch.eye(3).unsqueeze(0)
    m[:,0,0]=-1
    target=warp_source_to_view(p,m)
    target.requires_grad_()
    source=warp_source_to_view(p,torch.eye(3).unsqueeze(0))
    amaps=torch.stack([source,target],dim=1)
    transform=torch.stack([torch.eye(3).unsqueeze(0),m],dim=1)
    valid=torch.ones(1,2,1,224,224)
    loss=masked_attention_consistency(amaps,transform,valid)
    assert loss.item()<1e-4
    loss.backward()
    assert target.grad is not None


def test_two_view_training_loss_finite(tiny_config):
    random.seed(123)
    augmentor=PairedAugmentor()
    x=torch.rand(3,224,224)
    views,matrices,valid=augmentor(x)
    assert views.shape==(2,3,224,224)
    assert matrices.shape==(2,3,3)
    assert (valid>=0).all() and (valid<=1).all()
    model=LesionAwareMangoNet(tiny_config)
    out0=model(views[0].unsqueeze(0));out1=model(views[1].unsqueeze(0))
    cfg={'focal_gamma':2.0,'lambda_ce':1.0,'lambda_focal':0.5,'lambda_entropy':0.05,
         'lambda_consistency':0.5,'loss_ramp_epochs':10}
    loss,parts=combined_loss(out0,out1,torch.tensor([3]),matrices.unsqueeze(0),
                             valid.unsqueeze(0),torch.ones(8),1,cfg)
    assert torch.isfinite(loss)
    assert parts['consistency']>=0
    loss.backward()
    assert model.lesion_conv.weight.grad is not None
