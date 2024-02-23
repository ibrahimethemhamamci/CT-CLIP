## Text Classifier

This module encompasses the training and inference scripts for the text classifier model. It serves the purpose of extracting pathology labels from datasets through partial manual annotations. The manual annotations utilized for training and evaluating the model are accessible in the [CT Rate dataset](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

## Installation

Please note that the Transformers library version utilized in this model differs from CT-CLIP. To install the required version, execute the following command:

```bash
$ pip install transformers==4.35.2 
```

Ensure the installation of the correct Transformers version (4.30.1) prior to running scripts for CT-CLIP.

## Training

Adjust the file paths within the training script to point to "train_text_classifier.csv" and "valid_text_classifier.csv" for your downloaded dataset. Subsequently, execute the following command:

```bash
$ python train.py
```

## Inference

Adjust the file paths within the infernece script to point to "test_all.csv" (which contains all accessions and reports) and "text_transformer_model.pth" for your downloaded dataset and model. Subsequently, execute the following command:

```bash
$ python inference.py
```
