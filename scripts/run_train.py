from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
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
    patch_size = 20,
    temporal_patch_size = 10,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)
#dim_image = 131072,


clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_text = 768,
    dim_image = 294912,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)
trainer = CTClipTrainer(
    clip,
    reports_file_train= "path_to_train_reports_csv", #TODO: Path to train reports CSV
    reports_file_valid= "path_to_validation_reports_csv", #TODO: Path to validation reports CSV
    data_train= "path_to_preprocessed_train", #TODO: Path to preprocessed train data
    data_valid = "path_to_preprocessed_valid", #TODO: Path to preprocessed validation data
    train_meta_file = "path_to_train_metadata_csv", #TODO: Path to train metadata CSV
    valid_meta_file = "path_to_validation_metadata_csv", #TODO: Path to validation metadata CSV
    labels = "path_to_validation_labels_csv", #TODO: Path to validation labels CSV
    batch_size = 8,
    results_folder="output_folder", #TODO: Path to save output results
    num_train_steps = 100001,
    num_workers = 4,
)

trainer.train()
