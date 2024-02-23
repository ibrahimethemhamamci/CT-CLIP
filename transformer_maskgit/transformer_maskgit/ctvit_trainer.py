from math import sqrt
from random import choice
from pathlib import Path
from shutil import rmtree
import tqdm
from beartype import beartype
import nibabel as nib
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from einops import rearrange

from transformer_maskgit.optimizer import get_optimizer

from ema_pytorch import EMA

from transformer_maskgit.ctvit import CTViT
from transformer_maskgit.data import ImageDataset, VideoDataset, tensor_to_nifti

from accelerate import Accelerator

# helpers

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# main trainer class

import numpy as np
from torch.utils.data import BatchSampler

class CustomBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_data()
        dummy_sampler = list(range(len(dataset)))
        super().__init__(dummy_sampler, batch_size, drop_last)


    def __iter__(self):
        indices = []
        for _, group in self.groups.items():
            np.random.shuffle(group)
            while len(group) % self.batch_size != 0:
                group.extend(group[:self.batch_size-(len(group) % self.batch_size)])
            for idx in range(0, len(group), self.batch_size):
                batch = group[idx:idx + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    indices.append(batch)
        np.random.shuffle(indices)
        print(indices)

        # Return a generator that yields the batches
        return (batch for batch in indices)


    def __len__(self):
        if self.drop_last:
            return sum(len(group) // self.batch_size for group in self.groups.values())
        else:
            return sum(-(-len(group) // self.batch_size) for group in self.groups.values())
        
    def get_number_of_slices(self, idx):
        path = self.dataset.dataset.paths[self.dataset.indices[idx]]
        ext = path.suffix
        tensor_shape = nib.load(path).dataobj.shape
        return tensor_shape[2]


    def group_data(self):
        groups = {}
        for idx in tqdm.tqdm(range(len(self.dataset))):
            num_slices = self.get_number_of_slices(idx)
            if num_slices not in groups:
                groups[num_slices] = []
            groups[num_slices].append(idx)
        return groups


@beartype
class CTViTTrainer(nn.Module):
    def __init__(
        self,
        vae: CTViT,
        *,
        num_train_steps,
        batch_size,
        folder,
        train_on_images = False,
        num_frames = 17,
        lr = 3e-5,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        discr_max_grad_norm = None,
        save_results_every = 50,
        save_model_every = 250,
        results_folder = './results',
        valid_frac = 0.05,
        random_split_seed = 42,
        use_ema = True,
        ema_beta = 0.995,
        ema_update_after_step = 0,
        ema_update_every = 1,
        apply_grad_penalty_every = 4,
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        image_size = vae.image_size

        self.accelerator = Accelerator(**accelerate_kwargs)

        self.vae = vae

        self.use_ema = use_ema
        if self.is_main and use_ema:
            self.ema_vae = EMA(vae, update_after_step = ema_update_after_step, update_every = ema_update_every)

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        all_parameters = set(vae.parameters())
        discr_parameters = set(vae.discr.parameters())
        vae_parameters = all_parameters - discr_parameters

        self.vae_parameters = vae_parameters

        self.optim = get_optimizer(vae_parameters, lr = lr, wd = wd)
        self.discr_optim = get_optimizer(discr_parameters, lr = lr*0.01, wd = wd)

        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        # create dataset
        print("This is a test.")
        dataset_klass = ImageDataset if train_on_images else VideoDataset

        if train_on_images:
            self.ds = ImageDataset(folder, image_size)
        else:
            self.ds = VideoDataset(folder, image_size, num_frames = num_frames)

        # split for validation
        self.valid_frac = 0.05
        random_split_seed = 42
        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader
        batch_sampler_train = CustomBatchSampler(self.ds, batch_size=batch_size, drop_last=False)
        batch_sampler_val = CustomBatchSampler(self.valid_ds, batch_size=batch_size, drop_last=False)
        list_train=[]
        list_val=[]
        for i in range(len(self.ds)): 
            list_train.append(self.ds.dataset.paths[self.ds.indices[i]])
        for i in range(len(self.valid_ds)): 
            list_val.append(self.valid_ds.dataset.paths[self.valid_ds.indices[i]])

        with open("train.txt", "w") as f:
            for item in list_train:
                f.write(str(item) + "\n")
        with open("valid.txt", "w") as f:
            for item in list_val:
                f.write(str(item) + "\n")
            
        self.dl = DataLoader(
            self.ds,
            batch_size = batch_size,
            shuffle = True,
            num_workers=1
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size = batch_size,
            shuffle = True,
            num_workers=1
        )

        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.valid_dl_iter=cycle(self.valid_dl)
        (
            self.vae,
            self.optim,
            self.discr_optim,
            self.dl_iter,
            self.valid_dl_iter
        ) = self.accelerator.prepare(
            self.vae,
            self.optim,
            self.discr_optim,
            self.dl_iter,
            self.valid_dl_iter
        )

        #self.dl_iter = cycle(self.dl)
        #self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.apply_grad_penalty_every = apply_grad_penalty_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model = self.accelerator.get_state_dict(self.vae),
            optim = self.optim.state_dict(),
            discr_optim = self.discr_optim.state_dict()
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        vae = self.accelerator.unwrap_model(self.vae)
        vae.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])
        self.discr_optim.load_state_dict(pkg['discr_optim'])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        device = self.device
        device=torch.device('cuda')


        steps = int(self.steps.item())
        apply_grad_penalty = not (steps % self.apply_grad_penalty_every)

        self.vae.train()

        # logs

        logs = {}

        # update vae (generator)
        for i in range(3):
            for _ in range(self.grad_accum_every):
                img = next(self.dl_iter)
                device=torch.device('cuda')
                img = img.to(device)

                with self.accelerator.autocast():
                    loss = self.vae(
                        img,
                        apply_grad_penalty = apply_grad_penalty
                    )

                self.accelerator.backward(loss / self.grad_accum_every)

                accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.vae.parameters(), self.max_grad_norm)

            self.optim.step()
            self.optim.zero_grad()

        # update discriminator

        if exists(self.vae.discr):
            self.discr_optim.zero_grad()

            for _ in range(self.grad_accum_every):
                img = next(self.dl_iter)
                device=torch.device('cuda')
                img = img.to(device)
                with self.accelerator.autocast():

                    loss = self.vae(img, return_discr_loss = True)

                self.accelerator.backward(loss / self.grad_accum_every)

                accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})

            if exists(self.discr_max_grad_norm):
                self.accelerator.clip_grad_norm_(self.vae.discr.parameters(), self.discr_max_grad_norm)

            self.discr_optim.step()

            # log

            self.print(f"{steps}: vae loss: {logs['loss']} - discr loss: {logs['discr_loss']}")

        # update exponential moving averaged generator

        if self.is_main and self.use_ema:
            self.ema_vae.update()

        # sample results every so often

        if self.is_main and not (steps % self.save_results_every):
            vaes_to_evaluate = ((self.vae, str(steps)),)

            if self.use_ema:
                vaes_to_evaluate = ((self.ema_vae.ema_model, f'{steps}.ema'),) + vaes_to_evaluate

            for model, filename in vaes_to_evaluate:
                model.eval()

                valid_data = next(self.valid_dl_iter)

                is_video = valid_data.ndim == 5
                device=torch.device('cuda')
                valid_data = valid_data.to(device)

                recons = model(valid_data, return_recons_only = True)

                # if is video, save gifs to folder
                # else save a grid of images

                if is_video:
                    sampled_videos_path = self.results_folder / f'samples.{filename}'
                    (sampled_videos_path).mkdir(parents = True, exist_ok = True)
                    i=0
                    for tensor in recons.unbind(dim = 0):
                        tensor_to_nifti(tensor, str(sampled_videos_path / f'{filename}_{i}.nii.gz'))
                        i=i+1
                else:
                    imgs_and_recons = torch.stack((valid_data, recons), dim = 0)
                    imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

                    imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                    grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (0, 1))

                    logs['reconstructions'] = grid

                    save_image(grid, str(self.results_folder / f'{filename}.png'))

            self.print(f'{steps}: saving to {str(self.results_folder)}')

        # save model every so often

        if self.is_main and not (steps % self.save_model_every):
            state_dict = self.vae.state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.pt')
            torch.save(state_dict, model_path)

            if self.use_ema:
                ema_state_dict = self.ema_vae.state_dict()
                model_path = str(self.results_folder / f'vae.{steps}.ema.pt')
                torch.save(ema_state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, log_fn = noop):
        device = next(self.vae.parameters()).device
        device=torch.device('cuda')

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')