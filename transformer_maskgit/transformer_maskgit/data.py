from pathlib import Path
import nibabel as nib

import cv2
from PIL import Image
from functools import partial
import json

from typing import Tuple, List
from beartype.door import is_bearable

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader
from torchvision import transforms as T, utils

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f%frames==0:
        return t[:,:-(frames-1)]
    if f%frames==1:
        return t
    else:
        return t[:,:-((f%frames)-1)]
    
def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# image related helpers fnuctions and dataset

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = []

        

        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# tensor of shape (channels, frames, height, width) -> gif

# handle reading and writing gif

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

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

# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(
    tensor,
    path,
    duration = 120,
    loop = 0,
    optimize = True
):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(
    path,
    channels = 3,
    transform = T.ToTensor()
):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

# handle reading and writing mp4

def video_to_tensor(
    path: str,              # Path of the video to be imported
    num_frames = -1,        # Number of frames to be stored in the output tensor
    crop_size = None
) -> torch.Tensor:          # shape (1, channels, frames, height, width)

    video = cv2.VideoCapture(path)

    frames = []
    check = True

    while check:
        check, frame = video.read()

        if not check:
            continue

        if exists(crop_size):
            frame = crop_center(frame, *pair(crop_size))

        frames.append(rearrange(frame, '... -> 1 ...'))

    frames = np.array(np.concatenate(frames[:-1], axis = 0))  # convert list of frames to numpy array
    frames = rearrange(frames, 'f h w c -> c f h w')

    frames_torch = torch.tensor(frames).float()

    return frames_torch[:, :num_frames, :, :]

def tensor_to_video(
    tensor,                # Pytorch video tensor
    path: str,             # Path of the video to be saved
    fps = 25,              # Frames per second for the saved video
    video_format = 'MP4V'
):
    # Import the video and cut it into frames.
    tensor = tensor.cpu()

    num_frames, height, width = tensor.shape[-3:]

    fourcc = cv2.VideoWriter_fourcc(*video_format) # Changes in this line can allow for different video formats.
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))

    frames = []

    for idx in range(num_frames):
        numpy_frame = tensor[:, idx, :, :].numpy()
        numpy_frame = np.uint8(rearrange(numpy_frame, 'c h w -> h w c'))
        video.write(numpy_frame)

    video.release()

    cv2.destroyAllWindows()

    return video

def crop_center(
    img,        # tensor
    cropx,      # Length of the final image in the x direction.
    cropy       # Length of the final image in the y direction.
) -> torch.Tensor:
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:(starty + cropy), startx:(startx + cropx), :]

# video dataset

class VideoDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels = 3,
        num_frames = 17,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['gif', 'mp4', 'nii.gz']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        #self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.paths = []
        self.number_of_slices=[]
        for ext in exts:
            for p in Path(f'{folder}').rglob(f'*.{ext}'):
                if p.is_file():
                    if ext == 'nii.gz':
                        if len(list(nib.load(p).dataobj.shape)) > 2 and (600>nib.load(p).dataobj.shape[2] > 100):
                            self.paths.append(p)
                            self.number_of_slices.append(nib.load(p).dataobj.shape[2])
                    else:
                        self.paths.append(p)
               
        self.transform = T.Compose([
            T.Resize(image_size),
            #T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            #T.CenterCrop(image_size),
            T.ToTensor()
        ])
        print(self.number_of_slices)

        # functions to transform video path to tensor
        self.gif_to_tensor = partial(gif_to_tensor, channels = self.channels, transform = self.transform)
        self.mp4_to_tensor = partial(video_to_tensor, crop_size = self.image_size)
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity
    def nii_img_to_tensor(self, path, transform):
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()
        path_json = str(path).replace(".nii.gz","")+("_metadata.json")
        with open(path_json, 'r') as f:
            json_data = json.load(f)
            slope=int(float(json_data["RescaleSlope"]))
            intercept=int(float(json_data["RescaleIntercept"]))
            manufacturer=json_data["Manufacturer"]
        img_data = slope*img_data + intercept
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = ((img_data / 1000)).astype(np.float32)
        slices = []
        if manufacturer == 'PNMS':
            for i in reversed(range(img_data.shape[2])):
                img_slice = Image.fromarray(img_data[:, :, i], mode='F')
                img_transformed = transform(img_slice)
                slices.append(img_transformed)
            
        else:
            for i in range(img_data.shape[2]):
                img_slice = Image.fromarray(img_data[:, :, i], mode='F')
                img_transformed = transform(img_slice)
                slices.append(img_transformed)
        tensor = torch.stack(slices,dim=1)
        tensor = tensor.unsqueeze(1)
        tensor=F.interpolate(tensor, size=(201, 128, 128), mode='trilinear',align_corners=False)
        tensor = tensor.squeeze(1)
        return tensor

    def __getitem__(self, index):
        path = self.paths[index]
        ext = path.suffix
        if ext == '.gif':
            tensor = self.gif_to_tensor(path)
        elif ext == '.mp4':
            tensor = self.mp4_to_tensor(str(path))
        elif ext == '.gz':
            tensor = self.nii_to_tensor(path)
        else:
            raise ValueError(f'unknown extension {ext}')
        return self.cast_num_frames_fn(tensor)
        
    def __len__(self):
        return len(self.paths)
    def get_n_slices_list(self):
        return self.number_of_slices

# override dataloader to be able to collate strings

def collate_tensors_and_strings(data):
    if is_bearable(data, List[torch.Tensor]):
        return (torch.stack(data, dim = 0),)

    data = zip(*data)
    output = []

    for datum in data:
        if is_bearable(datum, Tuple[torch.Tensor, ...]):
            datum = torch.stack(datum, dim = 0)
        elif is_bearable(datum, Tuple[str, ...]):
            datum = list(datum)
        else:
            raise ValueError('detected invalid type being passed from dataset')

        output.append(datum)

    return tuple(output)

def DataLoader(*args, **kwargs):
    return PytorchDataLoader(*args, collate_fn = collate_tensors_and_strings, **kwargs)
