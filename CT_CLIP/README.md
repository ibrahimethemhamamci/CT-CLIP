## CT-CLIP

This module provides an implementation of CT-CLIP, leveraging CT-ViT as the image encoder and pretrained medical LLMs as the text encoder.

## Installation

To install CT-CLIP, execute the following command:

```bash
$ pip install -e .
```

This will install the necessary dependencies and enable you to use CT-CLIP.

## Usage

```python
#After defining text and image encoders, define CT-CLIP model

clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 2097152,
    dim_text = 768,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm = False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)

#Load the pretrained weights during inference

clip.load("path_to_pretrained_weights")

```
