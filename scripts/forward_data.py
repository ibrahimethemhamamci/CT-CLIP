from pathlib import Path
from shutil import rmtree
from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer, BertModel

from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis

from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from data_inference_nii import CTReportDatasetinfer
#from data_external_valid import CTReportDatasetinfer
import numpy as np
import tqdm
import pandas as pd

from einops import rearrange
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import math
import torch.optim.lr_scheduler as lr_scheduler
from ct_clip import CTCLIP


# helpers

def tensor_to_nifti(tensor, path, affine=np.eye(4)):
    """
    Save tensor as a NIfTI file.

    Args:
        tensor (torch.Tensor): The input tensor with shape (D, H, W) or (C, D, H, W).
        path (str): The path to save the NIfTI file.
        affine (np.ndarray, optional): The affine matrix for the NIfTI file. Defaults to np.eye(4).
    """

    tensor = tensor.cpu()

    if tensor.dim() == 4:
        # Assume single channel data if there are multiple channels
        if tensor.size(0) != 1:
            print("Warning: Saving only the first channel of the input tensor")
        tensor = tensor.squeeze(0)
    tensor=tensor.swapaxes(0,2)
    numpy_data = tensor.detach().numpy().astype(np.float32)
    nifti_img = nib.Nifti1Image(numpy_data, affine)
    nib.save(nifti_img, path)

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


class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_warmup=10000, gamma=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_warmup = T_warmup
        self.gamma = gamma
        self.T_cur = 0
        self.lr_min = 0
        self.iteration = 0

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iteration < self.T_warmup:
            lr = self.eta_max * self.iteration / self.T_warmup
        else:
            self.T_cur = self.iteration - self.T_warmup
            T_i = self.T_0
            while self.T_cur >= T_i:
                self.T_cur -= T_i
                T_i *= self.T_mult
                self.lr_min = self.eta_max * (self.gamma ** self.T_cur)
            lr = self.lr_min + 0.5 * (self.eta_max - self.lr_min) * \
                 (1 + math.cos(math.pi * self.T_cur / T_i))

        self.iteration += 1
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._update_lr()
        self._update_T()

    def _update_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.get_lr()[0]

    def _update_T(self):
        if self.T_cur == self.T_0:
            self.T_cur = 0
            self.lr_min = 0
            self.iteration = 0
            self.T_0 *= self.T_mult
            self.eta_max *= self.gamma

class CTClipInference(nn.Module):
    def __init__(
            self,
            CTClip: CTCLIP,
            *,
            num_train_steps,
            batch_size,
            data_folder: "external_valid",
            reports_file: "data_reports.xslx",
            lr = 1e-4,
            wd = 0.,
            max_grad_norm = 0.5,
            save_results_every = 100,
            save_model_every = 2000,
            results_folder = './results',
            labels = "labels.csv",
            accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        self.results_folder = results_folder
        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        all_parameters = set(CTClip.parameters())

        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        self.max_grad_norm = max_grad_norm
        self.lr=lr
        # Load the pre-trained weights
        self.ds = CTReportDatasetinfer(data_folder=data_folder, csv_file=reports_file,labels=labels)

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
        self.lr_scheduler = CosineAnnealingWarmUpRestarts(self.optim,
                                                          T_0=4000000,    # Maximum number of iterations
                                                          T_warmup=10000, # Number of warmup steps
                                                          eta_max=lr)   # Maximum learning rate


        (
            self.dl_iter,
            self.CTClip,
            self.optim,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.CTClip,
            self.optim,
            self.lr_scheduler
        )

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.result_folder_txt = self.results_folder
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)



    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.CTClip),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        CTClip = self.accelerator.unwrap_model(self.CTClip)
        CTClip.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)


    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())


        # logs
        logs = {}



        if True:
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
                        # Save the NumPy array as a .npz file
                        np.savez(f'{self.results_folder}/text/{acc_name[0]}.npz', arr=text_embeds_np)

                        # Convert the tensor to a NumPy array
                        enc_image_send_np = enc_image_send.cpu().detach().numpy()

                        # Save the NumPy array as a .npz file
                        np.savez(f'{self.results_folder}/image/{acc_name[0]}.npz', arr=enc_image_send_np)

                self.accelerator.wait_for_everyone()
        return True




    def infer(self, log_fn=noop):
        device = next(self.CTClip.parameters()).device
        device=torch.device('cuda')
        while True:
            finish = self.train_step()
            if finish:
                break

        self.print('Inference complete')
