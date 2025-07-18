
## CT-CLIP: Revolutionizing Abnormality Detection through Chest CT Volumes and Radiology Reports

Before proceeding with training and inference, ensure that you have accurately downloaded the dataset and installed the necessary dependencies as outlined in the [Main Page](..), and preprocess the dataset.

## Training

### Training CT-CLIP

For training the zero-shot CT-CLIP model, adjust the parameters `reports_file_train`, `reports_file_valid`, `data_train`, `data_valid`, `train_meta_file`, `valid_meta_file`, and `labels` in the "run_train.py" script to correspond to the appropriate paths for the downloaded and preprocessed train and validation folder. Then, execute the training script with the following command:

```bash
$ python run_train.py
```

Should you prefer to employ data parallelism with FSDP (Fully Sharded Data Parallelism), initiate the training script with:

```bash
$ accelerate launch --use_fsdp run_train.py
```

Alternatively, for data parallelism without FSDP, initiate the training script with:

```bash
$ accelerate launch run_train.py
```

### Training CT-VocabFine

To train CT-VocabFine, execute the provided script with the following command, ensuring correct paths:

```bash
$ python ct_vocabfine_train.py \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 10 \
    --warmup_length 10000 \
    --save path_to_save_folder \
    --pretrained path_to_pretrained_clip_model \
    --data-folder path_to_preprocessed_train_folder \
    --reports-file path_to_train_reports_csv \
    --meta-file paths_to_train_meta_csv \
    --labels path_to_train_labels_csv
```

### Training CT-LiPro

For training CT-LiPro, utilize the following script with accurate paths:

```bash
$ python ct_lipro_train.py \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 10 \
    --warmup_length 10000 \
    --save path_to_save_folder \
    --pretrained path_to_pretrained_clip_model \
    --data-folder path_to_preprocessed_train_folder \
    --reports-file path_to_train_reports_csv \
    --meta-file paths_to_train_meta_csv \
    --labels path_to_train_labels_csv
```

## Inference

To facilitate comprehensive analysis, inference scripts are designed to store output probabilities from classification models alongside corresponding ground truth values, along with AUROC values. These stored data will be utilized for bootstrapping and calculating additional metrics.

As CT-VocabFine is an open-vocabulary fine-tuning method, the inference of CT-VocabFine is same as inference of zero-shot CT-CLIP.

### Inference of CT-CLIP and CT-VocabFine

For inference of CT-CLIP and CT-VocabFine models, adjust the path for the model script either for CT-CLIP or CT-VocabFine, adjust the parameters `data_folder`, `reports_file`, `meta_file` and `labels` in the "run_zero_shot.py" script to correspond to the appropriate paths for the downloaded and preprocessed validation folder. Then, execute the training script with the following command:

```bash
$ python run_zero_shot.py
```
### Inference of CT-LiPro

For inference of CT-LiPro, utilize the following script with accurate paths:

```bash
$ python ct_lipro_inference.py \
    --save path_to_save_folder \
    --pretrained path_to_trained_lipro_model \
    --data-folder path_to_preprocessed_validation_folder \
    --reports-file path_to_valid_reports_csv \
    --meta-file paths_to_train_meta_csv \
    --labels path_to_valid_labels_csv
```

## Bootstrapping and Calculating Classification Metrics

Following model inference, please ensure to update the path in `bootstrap_values.py` to reflect the correct inference output folder by modifying the `data_dir` variable. Then execute the following command:

```bash
$ python run_zero_shot.py
```

This command initiates bootstrapping with 1000 samples and computes classification scores for each pathology across various bootstrap iterations. Subsequently, the results are saved into CSV files within the designated directory.


## Volume-to-Volume and Report-to-Volume Retrieval

To apply volume-to-volume and report-to-volume retrieval, first save the image and text embeddings of the model. This can be done by modifying the model execution in the `zero_shot.py` script as:

```bash
text_latents, image_latents = model(text_tokens, valid_data.cuda(),  device=device, return_latents=True)
```

The returned latents should be saved into single npz files as `image_latents.npz` and `text_latents.npz`. After that, adjust the paths in the `volume_to_volume.py` and `report_to_volume.py` scripts accordingly. Then they can be executed as:

```bash
$ python volume_to_volume.py
```

and

```bash
$ python report_to_volume.py
```

## T-SNE Plots:

To generate T-SNE plots, first, compute the latent representations for both training and validation data as previously described. Then, update the file paths in `tsne_latents.py` accordingly and execute the script using the following command:

```bash
$ python tsne_latents.py
```


