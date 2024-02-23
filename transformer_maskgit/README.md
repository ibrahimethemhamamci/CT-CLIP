## CT-Clip Image Encoder (CT-ViT)

This module is an adaptation of the transformer_maskgit module from [GenerateCT](https://github.com/ibrahimethemhamamci/GenerateCT). It introduces modifications to the patch embeddings of the CT-ViT model and enables the return of embedded layers for utilization in CT-CLIP.

## Installation

To install the CT-CLIP image encoder, please run:

```bash
$ pip install -e .
```

This will install the necessary dependencies and allow you to use the CT-CLIP image encoder.

## Usage

```python
#Define CTViT as image encoder

image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 30, #patch size in coronal and sagittal dimensions
    temporal_patch_size = 15, #patch size in axial dimension
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)

```
