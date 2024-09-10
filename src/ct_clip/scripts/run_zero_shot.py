from transformers import BertTokenizer, BertModel

from ct_clip.ct_clip import CTCLIP
from ct_clip.transformer_maskgit import CTViT
from zero_shot import CTClipInference

tokenizer = BertTokenizer.from_pretrained(
    "microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True
)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

text_encoder.resize_token_embeddings(len(tokenizer))


image_encoder = CTViT(
    dim=512,
    codebook_size=8192,
    image_size=480,
    patch_size=30,
    temporal_patch_size=15,
    spatial_depth=4,
    temporal_depth=4,
    dim_head=32,
    heads=8,
)

clip = CTCLIP(
    image_encoder=image_encoder,
    text_encoder=text_encoder,
    dim_image=2097152,
    dim_text=768,
    dim_latent=512,
    extra_latent_projection=False,  # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds=False,
    use_all_token_embeds=False,
)

clip.load("path_to_pretrained_model")

inference = CTClipInference(
    clip,
    data_folder="path_to_preprocessed_validation_folder",
    reports_file="path_to_validation_reports_csv",
    labels="path_to_validation_labels_csv",
    batch_size=1,
    results_folder="inference_zeroshot/",
    num_train_steps=1,
)

inference.infer()
