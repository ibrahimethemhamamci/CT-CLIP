import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_inference_nii import CTReportDatasetinfer

from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP

import tqdm

from src.args import parse_arguments
from src.models.utils import cosine_lr

class ImageLatentsClassifier(nn.Module):
    def __init__(self, trained_model, latent_dim, num_classes, dropout_prob=0.3):
        super(ImageLatentsClassifier, self).__init__()
        self.trained_model = trained_model
        for param in self.trained_model.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, num_classes)  # Assuming trained_model.image_latents_dim gives the size of the image_latents

    def forward(self, *args, **kwargs):
        kwargs['return_latents'] = True
        _, image_latents, _ = self.trained_model(*args, **kwargs)
        image_latents = self.relu(image_latents)
        image_latents = self.dropout(image_latents)  # Apply dropout on the latents
        return self.classifier(image_latents)

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)
    def load(self, file_path):
        loaded_state_dict = torch.load(file_path)
        self.load_state_dict(loaded_state_dict)

def finetune(args):
    # Initialize BERT tokenizer and text encoder
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialize image encoder and clip model
    image_encoder = CTViT(
        dim=512, codebook_size=8192, image_size=480, patch_size=20,
        temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
        dim_head=32, heads=8
    )

    clip = CTCLIP(
        image_encoder=image_encoder, text_encoder=text_encoder,
        dim_image=294912, dim_text=768, dim_latent=512,
        extra_latent_projection=False, use_mlm=False,
        downsample_image_embeds=False, use_all_token_embeds=False
    )
    clip.load(args.pretrained)

    # Define the number of classes and initialize the image classifier
    num_classes = 18
    image_classifier = ImageLatentsClassifier(clip, 512, num_classes)

    # Load dataset for fine-tuning
    ds = CTReportDatasetinfer(data_folder=args.data_folder, reports_file=args.reports_file, meta_file=args.meta_file, labels = args.labels)
    dl = DataLoader(ds, num_workers=8, batch_size=8, shuffle=True)
    num_batches = len(dl)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Move model to GPU and set it to training mode
    model = image_classifier.cuda()
    devices = list(range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    # Define loss function and optimizer
    weights = torch.tensor([9.211362733, 2.384068466, 8.295479204, 32.8629776, 2.992233613,
                            6.064870808, 3.176470588, 4.187083754, 3.022222222, 1.216071737,
                            1.677849552, 3.152851834, 7.123261694, 18.16629381, 13.8480647,
                            6.335045662, 10.81701149, 13.40695067]).cuda()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # Start training loop
    for epoch in range(args.epochs):
        for i, batch in tqdm.tqdm(enumerate(dl)):
            start_time = time.time()
            step = i + epoch * num_batches

            inputs, _, labels, _ = batch
            labels = labels.float().cuda()
            text_tokens = tokenizer([" "], return_tensors="pt", padding="max_length", truncation=True, max_length=512).to("cuda")

            data_time = time.time() - start_time
            logits = model(text_tokens, inputs, device=torch.device('cuda'), return_latents=True)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler(step)
            batch_time = time.time() - start_time

            if i % args.print_every == 0:
                percent_complete = 100 * i / len(dl)
                print(f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dl)}]\t"
                      f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True)

            if i % args.save_every == 0:
                os.makedirs(args.save, exist_ok=True)

                # Access the underlying model to avoid the 'module.' prefix in state_dict keys
                model_to_save = model.module if hasattr(model, 'module') else model

                model_path = os.path.join(args.save, f'checkpoint_{i}_epoch_{epoch+1}.pt')
                print('Saving model to', model_path)

                # Save the state_dict of the unwrapped model
                torch.save(model_to_save.state_dict(), model_path)

                optim_path = os.path.join(args.save, f'optim_{i}_epoch_{epoch+1}.pt')

                # Save the optimizer state
                torch.save(optimizer.state_dict(), optim_path)

        # Save final model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)

            # Access the underlying model to avoid the 'module.' prefix in state_dict keys
            model_to_save = model.module if hasattr(model, 'module') else model

            model_path = os.path.join(args.save, f'epoch_{epoch+1}.pt')
            print('Saving model to', model_path)

            # Save the state_dict of the unwrapped model
            torch.save(model_to_save.state_dict(), model_path)

            optim_path = os.path.join(args.save, f'epoch_{epoch+1}.pt')

            # Save the optimizer state
            torch.save(optimizer.state_dict(), optim_path)
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments()
    # Start fine-tuning process
    finetune(args)
