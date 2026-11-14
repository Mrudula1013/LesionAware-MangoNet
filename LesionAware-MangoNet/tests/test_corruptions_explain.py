import numpy as np
import torch
from mangonet.models.mango_net import LesionAwareMangoNet
from mangonet.evaluation.robustness import corrupt_image,CORRUPTIONS
from mangonet.evaluation.explainability import generate_explanations


def test_corruptions_preserve_valid_rgb_range():
    image=torch.rand(3,224,224)
    for name in CORRUPTIONS:
        for severity in range(1,6):
            modified=corrupt_image(image,name,severity,np.random.default_rng(12))
            assert modified.shape==image.shape
            assert float(modified.min())>=0 and float(modified.max())<=1


def test_explain_maps_and_internal_agreement(tiny_config):
    model=LesionAwareMangoNet(tiny_config).eval()
    image=torch.rand(1,3,224,224)
    stats={'mean':[0,0,0],'std':[1,1,1],'image_size':224}
    maps,values=generate_explanations(model,image,stats,top_channels=4,run_faithfulness=False)
    assert set(maps)=={'lesion','gradcam','scorecam'}
    assert all(heat.shape==(1,1,224,224) for heat in maps.values())
    assert -1.01<=values['attention_gradcam_cosine']<=1.01
