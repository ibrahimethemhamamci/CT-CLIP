import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from forward_data import CTClipInference
import accelerate

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

text_encoder.resize_token_embeddings(len(tokenizer))


image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 20,
    temporal_patch_size = 10,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)

clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 294912,
    dim_text = 768,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)

clip.load("path_to_pretrained_model")
inference = CTClipInference(
    clip,
    data_folder = '/path_to_data_folder/valid_or_train',
    reports_file= "path_to_validation_or_train_reports.csv",
    labels = "path_to_validation_or_train_predicted_labels.csv",
    batch_size = 1,
    results_folder = "path_to_results_folder/train_or_validation/"
    num_train_steps = 1,
)

inference.infer()
