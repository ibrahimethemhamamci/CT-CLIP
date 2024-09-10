"""
This module is an adaptation of the transformer_maskgit module from
[GenerateCT](https://github.com/ibrahimethemhamamci/GenerateCT).
It introduces modifications to the patch embeddings of the CT-ViT model and enables the
return of embedded layers for utilization in CT-CLIP.
"""
from transformer_maskgit.MaskGITTransformer import MaskGITTransformer, CTViT, MaskGit, TokenCritic, make_video
from transformer_maskgit.videotextdataset import VideoTextDataset
from transformer_maskgit.ctvit_trainer import CTViTTrainer