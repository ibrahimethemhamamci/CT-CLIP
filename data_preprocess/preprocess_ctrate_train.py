import os
import nibabel as nib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from tqdm import tqdm

def read_nii_files(directory):
    """
    Retrieve paths of all NIfTI files in the given directory.

    Args:
    directory (str): Path to the directory containing NIfTI files.

    Returns:
    list: List of paths to NIfTI files.
    """
    nii_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))
    return nii_files

def read_nii_data(file_path):
    """
    Read NIfTI file data.

    Args:
    file_path (str): Path to the NIfTI file.

    Returns:
    np.ndarray: NIfTI file data.
    """
    try:
        nii_img = nib.load(file_path)
        nii_data = nii_img.get_fdata()
        return nii_data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

def process_file(file_path):
    """
    Process a single NIfTI file.

    Args:
    file_path (str): Path to the NIfTI file.

    Returns:
    None
    """
    img_data = read_nii_data(file_path)
    if img_data is None:
        print(f"Read {file_path} unsuccessful. Passing")
        return

    file_name = os.path.basename(file_path)

    row = df[df['VolumeName'] == file_name]
    slope = float(row["RescaleSlope"].iloc[0])
    intercept = float(row["RescaleIntercept"].iloc[0])
    xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
    z_spacing = float(row["ZSpacing"].iloc[0])

    # Define the target spacing values
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = slope * img_data + intercept
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = ((img_data / 1000)).astype(np.float32)

    img_data = img_data.transpose(2, 0, 1)
    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    resized_array = resize_array(tensor, current, target)
    resized_array = resized_array[0][0]

    save_folder = "train_preprocessed/" #save folder for preprocessed
    folder_path_new = os.path.join(save_folder, "train_" + file_name.split("_")[1], "train_" + file_name.split("_")[1] + file_name.split("_")[2]) #folder name for train or validation
    os.makedirs(folder_path_new, exist_ok=True)
    file_name = file_name.split(".")[0]+".npz"
    save_path = os.path.join(folder_path_new, file_name)
    np.savez(save_path, resized_array)

# Example usage:
if __name__ == "__main__":
    split_to_preprocess = 'train' #select the validation or test split
    nii_files = read_nii_files(split_to_preprocess)

    df = pd.read_csv("train_metadata.csv") #select the metadata

    num_workers = 18  # Number of worker processes

    # Process files using multiprocessing with tqdm progress bar
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_file, nii_files), total=len(nii_files)))
