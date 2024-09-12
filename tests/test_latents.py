import importlib.resources
import os

import numpy as np

from ct_clip.scripts.run_generate_latents import init_default_model


def test_generate_latent_text():
    resources = importlib.resources.files("tests.resources")
    input_text = (
        resources.joinpath("test_generate_latent_text.input.txt").read_text().strip()
    )
    with resources.joinpath("test_generate_latent_text.expected.npy").open("rb") as f:
        expected_results = np.load(f)

    pretrained_ctclip_model_path = os.environ.get("PRETRAINED_CTCLIP_MODEL_PATH")
    assert (
        pretrained_ctclip_model_path is not None
    ), "Must set PRETRAINED_CTCLIP_MODEL_PATH environment variable, pointing to CT-CLIP zeroshot model"

    ctcliplatents = init_default_model(pretrained_ctclip_model_path)
    latents = ctcliplatents.generate_latents(text=input_text)
    text_latents = latents.texts[0]

    np.testing.assert_allclose(text_latents, expected_results, rtol=1e-2, atol=1e-6)
