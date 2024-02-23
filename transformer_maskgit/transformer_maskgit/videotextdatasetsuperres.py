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
        for patient_folder in glob.glob(os.path.join(self.data_folder, '*')):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                accession_number = os.path.basename(accession_folder)
                if accession_number not in self.accession_to_text:
                    continue

                impression_text = self.accession_to_text[accession_number]

                for nii_file in glob.glob(os.path.join(accession_folder, '*.nii.gz')):
                    nii_img = nib.load(nii_file)
                    nii_lowres=str(nii_file)
                    nii_lowres_img=nii_lowres.split("/")[-1]
                    nii_lowres = nii_lowres.split("/")[-2]
                    nii_lowres="samples."+nii_lowres
                    nii_lowres="example_data/superres/ctvit_outputs/"+nii_lowres+"/"+nii_lowres_img+".nii.gz"
                    

                    if nii_img.shape[-1] < 100 or nii_img.shape[-1] > 600:
                        continue
                    elif not os.path.exists(nii_lowres):
                    	continue
                    else:
                        # Load metadata file
                        metadata_file = os.path.splitext(nii_file)[0][:-4]+ '_metadata.json'
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)

                        # Extract required metadata
                        try:
                            age = metadata['PatientAge'][:-1].zfill(3)
                            age = age[1:]
                        except:
                            age = "None"
                        try:
                            sex = metadata['PatientSex']
                        except:
                            sex="None"
                        if sex.lower() == "m":
                            sex="male"
                        if sex.lower() =="f":
                            sex="female"

                        input_text = f'{age} years old {sex}: {impression_text}'

                        samples.append((nii_file, input_text))
                        self.paths.append(nii_file)

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
        nii_file, input_text = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file)
        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '')  
        nii_lowres=str(nii_file)
        nii_lowres_img=nii_lowres.split("/")[-1]
        nii_lowres = nii_lowres.split("/")[-2]
        nii_lowres="samples."+nii_lowres
        nii_lowres="example_data/superres/ctvit_outputs/"+nii_lowres+"/"+nii_lowres_img+".nii.gz"
        video_lowres=self.lowres_to_tensor(nii_lowres)

        return self.cast_num_frames_fn(video_lowres), self.cast_num_frames_fn(video_tensor), input_text
