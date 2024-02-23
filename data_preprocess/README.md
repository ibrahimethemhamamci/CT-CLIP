## CT-Clip Preprocess

This section of the repository contains preprocessing scripts specifically designed for handling CT scans. These scripts perform preprocessing steps, including converting nii.gz files to npz files, normalizing pixel values to Hounsfield units (HU), clipping HU values between (-1000,1000), adjusting volume orientations, and standardizing x, y, and z spacings to 0.75, 0.75, and 1.5, respectively.

## Usage

To utilize these scripts, follow these steps:

1. Navigate to the appropriate subfolder within the dataset directory.
2. Copy the scripts into this subfolder.
3. Run the following commands in the terminal:

```bash
$ python preprocess_ctrate_train.py
$ python preprocess_ctrate_valid.py
```

These commands will execute the preprocessing steps for the training and validation datasets, respectively.
