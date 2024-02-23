import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, TextTransformer
from CTCLIPTrainer import CTClipTrainer


tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)

text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

print("---------")
print(tokenizer.pad_token_id)
print(tokenizer.mask_token_id)
print("-----------")


image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 30,
    temporal_patch_size = 15,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)


clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 2097152,
    dim_text = 768,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)
trainer = CTClipTrainer(
    clip,
    reports_file_train= "path_to_train_reports_csv",
    reports_file_valid= "path_to_validation_reports_csv",
    data_train= "path_to_preprocessed_train",
    data_valid = "path_to_preprocessed_valid",
    labels = "path_to_validation_labels_csv",
    batch_size = 8,
    results_folder="output_folder",
    num_train_steps = 100001,
    num_workers = 4,
)

trainer.train()
