from pathlib import Path

from transformers import BertTokenizer

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_inference_nii import CTReportDatasetinfer
import numpy as np
import tqdm

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from ct_clip import CTCLIP


# helpers

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array


class CTClipInference(nn.Module):
    def __init__(
            self,
            CTClip: CTCLIP,
            *,
            data_folder: "external_valid",
            reports_file: "data_reports.xslx",
            meta_file: "meta_data.csv",
            results_folder = './results',
            labels = "labels.csv",
            accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)

        self.register_buffer('steps', torch.Tensor([0]))

        # Load the pre-trained weights
        self.ds = CTReportDatasetinfer(data_folder=data_folder, reports_file=reports_file, meta_file=meta_file, labels=labels)

        # Split dataset into train and validation sets
        self.dl = DataLoader(
            self.ds,
            num_workers=3,
            batch_size=1,
            shuffle = True,
        )

        # prepare with accelerator
        #self.dl_iter=cycle(self.dl)
        self.dl_iter=self.dl
        self.device = self.accelerator.device
        self.CTClip.to(self.device)


        (
            self.dl_iter,
            self.CTClip,
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.CTClip,
        )

        self.result_folder_txt = results_folder
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def infer(self, log_fn=noop):
        device = self.device

        steps = int(self.steps.item())

        with torch.no_grad():

            models_to_evaluate = ((self.CTClip, str(steps)),)

            for model, filename in models_to_evaluate:
                model.eval()
                for batch in tqdm.tqdm(self.dl_iter):
                    valid_data, text, onehotlabels, acc_name = batch
                    
                    text_tokens=self.tokenizer(
                        text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

                    #_, image_latents, enc_image_send = model(text_tokens, valid_data.cuda(),  device=device, return_latents=True)
                    text_embeds, enc_image_send, _ =  model(text_tokens, valid_data.cuda(),  device=device, return_latents=True)
                    # Convert the tensor to a NumPy array
                    #image_latents_np = image_latents.cpu().detach().numpy()
                    text_embeds_np = text_embeds.cpu().detach().numpy()

                    # create directories if they do not exist
                    Path(f'{self.results_folder}/text/').mkdir(parents=True, exist_ok=True)
                    Path(f'{self.results_folder}/image/').mkdir(parents=True, exist_ok=True)

                    # Save the NumPy array as a .npz file
                    np.savez(f'{self.results_folder}/text/{acc_name[0]}.npz', arr=text_embeds_np)

                    # Convert the tensor to a NumPy array
                    enc_image_send_np = enc_image_send.cpu().detach().numpy()

                    # Save the NumPy array as a .npz file
                    np.savez(f'{self.results_folder}/image/{acc_name[0]}.npz', arr=enc_image_send_np)

            self.accelerator.wait_for_everyone()

        self.print('Inference complete')