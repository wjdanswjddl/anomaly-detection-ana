import torch as th

from guided_diffusion.script_util import (
    NUM_CLASSES,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def test_create_model_and_diffusion_forward_shape():
    defaults = model_and_diffusion_defaults()
    defaults["image_size"] = 64
    defaults["num_channels"] = 32
    defaults["num_res_blocks"] = 1
    defaults["attention_resolutions"] = "16"
    defaults["learn_sigma"] = False

    model, diffusion = create_model_and_diffusion(**defaults)
    model.eval()

    x = th.randn(2, 1, 64, 64)
    t = th.zeros(2, dtype=th.long)
    with th.no_grad():
        out = model(x, t)

    # `create_model` sets out_channels=2*in_channels (in_channels=1 for SBND)
    assert out.shape == (2, 2, 64, 64)
    assert diffusion.num_timesteps > 0


def test_create_classifier_forward_shape():
    defaults = classifier_defaults()
    defaults["image_size"] = 64
    defaults["classifier_width"] = 128
    defaults["classifier_depth"] = 1
    defaults["classifier_pool"] = "adaptive"

    classifier = create_classifier(**defaults)
    classifier.eval()

    x = th.randn(2, 1, 64, 64)
    t = th.zeros(2, dtype=th.long)
    with th.no_grad():
        logits = classifier(x, timesteps=t)

    assert logits.shape == (2, NUM_CLASSES)
