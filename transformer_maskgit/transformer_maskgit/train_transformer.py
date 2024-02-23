from pathlib import Path
from shutil import rmtree
from transformer_maskgit.optimizer import get_optimizer

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from transformer_maskgit import CTViT, MaskGit, MaskGITTransformer
from transformer_maskgit.videotextdataset import VideoTextDataset

from transformer_maskgit.data import tensor_to_nifti

from einops import rearrange
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import math
import torch.optim.lr_scheduler as lr_scheduler


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

class TransformerTrainer(nn.Module):
    def __init__(
        self,
        maskgittransformer: MaskGITTransformer,
        *,
        num_train_steps,
        batch_size,
        pretrained_ctvit_path,
        lr = 3e-5,
        wd = 0.,
        max_grad_norm = 0.5,
        save_results_every = 100,
        save_model_every = 2000,
        results_folder = './results',
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.maskgittransformer = maskgittransformer

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        all_parameters = set(maskgittransformer.parameters())

        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        self.max_grad_norm = max_grad_norm
        self.lr=lr
        # Load the pre-trained weights
        self.ds = VideoTextDataset(data_folder='example_data/ctvit-transformer', xlsx_file='example_data/data_reports.xlsx', num_frames=2)
        # Split dataset into train and validation sets
        valid_frac=0.05
        random_split_seed = 42
        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        list_train=[]
        list_val=[]
        for i in range(len(self.ds)): 
            list_train.append(self.ds.dataset.paths[self.ds.indices[i]])
        for i in range(len(self.valid_ds)): 
            list_val.append(self.valid_ds.dataset.paths[self.valid_ds.indices[i]])

        with open("train_transformer.txt", "w") as f:
            for item in list_train:
                f.write(str(item) + "\n")
        with open("valid_transformer.txt", "w") as f:
            for item in list_val:
                f.write(str(item) + "\n")


        self.dl = DataLoader(
            self.ds,
            num_workers=8,
            batch_size=self.batch_size,
            shuffle = True,
            #batch_sampler=batch_sampler_train,
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            num_workers=8,
            batch_size=self.batch_size,
            shuffle = True,
            #batch_sampler=batch_sampler_val,
        )

        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.valid_dl_iter=cycle(self.valid_dl)
        self.device = self.accelerator.device
        self.maskgittransformer.to(self.device)
        self.lr_scheduler = CosineAnnealingWarmUpRestarts(self.optim,
                                                  T_0=4000000,    # Maximum number of iterations
                                                  T_warmup=10000, # Number of warmup steps
                                                  eta_max=lr)   # Maximum learning rate


        (
 			self.dl_iter,
            self.valid_dl_iter,
            self.maskgittransformer,
            self.optim,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.valid_dl_iter,
            self.maskgittransformer,
            self.optim,
            self.lr_scheduler
        )
        
        


        #self.dl_iter = cycle(self.dl_iter)
        #self.valid_dl_iter = cycle(self.valid_dl_iter)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True) 
        """     
        train_len = int(len(dataset) * 0.8)
        val_len = len(dataset) - train_len
        self.ds, self.valid_ds = random_split(dataset, [train_len, val_len])
        """
        # samplers
        #train_sampler = DistributedSampler(self.ds) if self.accelerator.is_distributed else None
        #val_sampler = DistributedSampler(self.valid_ds) if self.accelerator.is_distributed else None
        #train_sampler=None
        #val_sampler=None
        # dataloaders
        """
        self.dl = DataLoader(self.ds, batch_size=batch_size, sampler=train_sampler)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=batch_size, sampler=val_sampler)
        """
        


    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.maskgittransformer),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        maskgittransformer = self.accelerator.unwrap_model(self.maskgittransformer)
        maskgittransformer.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)


    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.maskgittransformer.train()

        # logs
        logs = {}

        # update maskgittransformer model
        video, text = next(self.dl_iter)
        print(video.shape)
        device=self.device
        video=video.to(device)
        mask = torch.ones((video.shape[0], video.shape[2])).bool().to(device)
        #text = text.to(device)
        text = list(text)
        #video = video
        with self.accelerator.autocast():
            loss = self.maskgittransformer(video, texts=text, video_frame_mask=mask)

        self.accelerator.backward(loss)
        accum_log(logs, {'loss': loss.item()})
        self.max_grad_norm=2.0
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.maskgittransformer.parameters(), self.max_grad_norm)

        self.optim.step()
        self.lr_scheduler.step()
        self.optim.zero_grad()
        self.print(f"{steps}: loss: {logs['loss']}")

       
    
        if self.is_main and not (steps % self.save_results_every):
            with torch.no_grad():

                vaes_to_evaluate = ((self.maskgittransformer, str(steps)),)

                for model, filename in vaes_to_evaluate:
                    model.eval()

                    valid_data, text = next(self.valid_dl_iter)
                    #print(text)
                    #text=text.to(device)
                    valid_data = valid_data.to(device)
                    #text = list(text)
                    is_video = valid_data.ndim == 5
                    #valid_data = valid_data.cuda()

                    if "module" in model.__dict__:
                        model = model.module
                        
                    recons = model.sample(texts =text, num_frames = 201, cond_scale = 5.) # (1, 3, 17, 256, 128)


                    # if is video, save gifs to folder
                    # else save a grid of images

                    sampled_videos_path = self.results_folder / f'samples.{filename}'

                    (sampled_videos_path).mkdir(parents = True, exist_ok = True)
                    i=0
                    for tensor in recons.unbind(dim = 0):
                        print(tensor.shape)
                        print(valid_data[0].shape)
                        tensor_to_nifti(tensor, str(sampled_videos_path / f'{filename}_{i}.nii.gz'))
                        tensor_to_nifti(valid_data[0], str(sampled_videos_path / f'{filename}_{i}_input.nii.gz'))
                        i=i+1
                    filename = str(sampled_videos_path / f'{filename}_{i}.txt')
                    with open(filename, "w", encoding="utf-8") as file:
                        file.write(text[0])

                self.print(f'{steps}: saving to {str(self.results_folder)}')
    
        # save model every so often

        if self.is_main and not (steps % self.save_model_every):
            model = self.accelerator.unwrap_model(self.maskgittransformer)
            state_dict = model.state_dict()
            model_path = str(self.results_folder / f'maskgittransformer.{steps}.pt')

            self.   accelerator.save(state_dict, model_path)


            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs



    def train(self, log_fn=noop):
        device = next(self.maskgittransformer.parameters()).device
        device=torch.device('cuda')
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')
