## Text Classifier

This module encompasses the training and inference scripts for the text classifier model. It serves the purpose of extracting abnormality labels from datasets through partial manual annotations. The manual annotations utilized for training and evaluating the model are accessible in [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

## Installation

Please follow the installation of [CT-CLIP](..).

## Training

Adjust the datast folder and include `train.csv` and `val.csv` for your downloaded dataset. Subsequently, execute the following command:

```bash
$ python train.py --dataset 'path_to/dataset_folder' --augment 0 --scheduler 'None'
# dataset: specify dataset_folder
# augment: random suffle augmentation. 0 or 1
# scheduler: add scheduler. 'cawr' for cosine annealing warmup, 'rlop' for ReduceLrOnPlatau
```

## Inference

Adjust the file paths within the bash command to point to `test_all.csv` (which contains all accessions and reports) and `text_transformer_model.pth` for your downloaded dataset and model. Subsequently, execute the following command:

```bash
$ python infer.py --checkpoint 'path_to/text_transformer_model.pth' --single_data 'path_to/single_csv_file' --save_path 'path_to_save_dir'
```
or 
```bash
$ python infer.py --checkpoint 'path_to/text_transformer_model.pth' --dataset 'path_to/dataset_folder' --save_path 'path_to_save_dir'
```
```bash
# checkpoint: pretrained model weights
# dataset: specify dataset_folder
# single_data: specify single csv file
```

## Evaluation

To evaluate the model, please run the inference in the validation data. After doing that, adjust the ground truth and inferred outputs in the `eval.py` script and run it as:

```bash
$ python eval.py
```

