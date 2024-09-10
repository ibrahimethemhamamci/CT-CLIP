"""
This module is an adaptation of the transformer_maskgit module from
[GenerateCT](https://github.com/ibrahimethemhamamci/GenerateCT).
It introduces modifications to the patch embeddings of the CT-ViT model and enables the
return of embedded layers for utilization in CT-CLIP.
"""

from ctvit import CTViT
from MaskGITTransformer import MaskGITTransformer, MaskGit, TokenCritic, make_video
from videotextdataset import VideoTextDataset
from ctvit_trainer import CTViTTrainer
