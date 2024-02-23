import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
import nibabel as nib
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F


def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f%frames==0:
        return t[:,:-(frames-1)]
    if f%frames==1:
        return t
    else:
        return t[:,:-((f%frames)-1)]


class VideoTextDataset(Dataset):
    def __init__(self, data_folder, xlsx_file, min_slices=20, resize_dim=512, num_frames=2, force_num_frames=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.accession_to_text = self.load_accession_text(xlsx_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.resize_dim = resize_dim
        self.resize_transform = transforms.Resize((resize_dim, resize_dim))
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.transform2=transforms.Compose([
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)
        self.lowres_to_tensor=partial(self.get_lowres_image, transform=self.transform2)
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

    def load_accession_text(self, xlsx_file):
        df = pd.read_excel(xlsx_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['AccessionNo']] = row['Impressions']
        return accession_to_text

    def prepare_samples(self):
        samples = []

        for root, dirs, files in os.walk(self.data_folder):
            for file in files:
                print(file)
                if file.endswith(".nii.gz"):
                    nii_gz_files=os.path.join(root, file)
                    samples.append(nii_gz_files)
        return samples

    def __len__(self):
        return len(self.samples)

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
        img_data = (img_data+1)/2
        slices=[]
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
        tensor=F.interpolate(tensor, size=(201, 512, 512), mode='trilinear',align_corners=False)
        tensor = tensor.squeeze(1)
        return tensor

    def get_lowres_image(self,path,transform):
    	nii_img = nib.load(str(path))
    	img_data = nii_img.get_fdata()
    	img_data = (img_data+1)/2
    	tensor = torch.tensor(img_data)
    	tensor=tensor.permute(2,1,0)
    	tensor = tensor.unsqueeze(0)
    	return tensor.float()

    def __getitem__(self, index):
        nii_file = self.samples[index]
        text_path=nii_file.split(".nii.gz")[0]+".txt"
        print(text_path)
        with open(text_path, 'r', encoding="utf-8") as file:
            input_text = file.readline().strip()

        video_tensor = self.lowres_to_tensor(nii_file)

        path_name = text_path.split("/")[-1].split(".")[0].split("_")[0]
        print(path_name)
        #img_data = img_data.transpose(2, 0, 1)  # Rearrange dimensions to (channels, frames, height, width)

        # Resize each frame using the resize_transform
        resized_data = []

        return self.cast_num_frames_fn(video_tensor), input_text, path_name
