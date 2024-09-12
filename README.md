# CT-CLIP: A foundation model utilizing chest CT volumes and radiology reports for supervised-level zero-shot detection of abnormalities
A fork of the official repository of CT-CLIP. You can access the dataset and pretrained model weights via the [HuggingFace repository](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

This repo reorganizes the CT-CLIP code into a single coherent python package and
adds APIs to generate latent vectors.

## Setup

0. Install python
1. Clone this repo and cd into the `CT-CLIP` directory
2. Create a virtual environment
```shell
<whatever python you have> -m venv .venv
.venv/bin/activate.sh
```
3. Install the ct_clip package and its dependencies
```shell
python -m pip install -e .
```

After following these steps, your environment should be properly set up with all required packages.

The CT-CLIP model necessitates the use of an A100 GPU with 80GB of VRAM for a batch size of 8 for efficient training, due to the model's considerable size. Inference can be done in smaller GPUs. The patch sizes of the image encoder can be adjusted to make it fit onto smaller GPUs, although this will affect the model performance in smaller pathologies. Batch size can also be lowered, but this is not recommended for CLIP training as it will not learn negative images with lower batch sizes.

## Inference

You will need to download the pretrained CT-CLIP model:

- **CT-CLIP Model**: [Download Here](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/resolve/main/models/CT_CLIP_zeroshot.pt?download=true)

### Generate Latents
The package contains an entrypoint script to generate latent vectors.
```shell
generate-latents --pretrained-model /path/to/CT_CLIP_zeroshot.pt --text "your input text here"
```

Currently it prints the latent vector to stdout. Not the ideal outcome.

### Zero Shot
The package contains an entrypoint script to run the zero-shot inference.
I think it is also currently set up to output latent vectors.
```
usage: run-zero-shot [-h] [--pretrained-model PRETRAINED_MODEL]
                     [--preprocessed-validation-folder PREPROCESSED_VALIDATION_FOLDER]
                     [--validation-reports-csv VALIDATION_REPORTS_CSV]
                     [--validation-labels-csv VALIDATION_LABELS_CSV]
                     [--results-folder RESULTS_FOLDER]

options:
  -h, --help            show this help message and exit
  --pretrained-model PRETRAINED_MODEL
                        Path to pretrained CT-CLIP model file
  --preprocessed-validation-folder PREPROCESSED_VALIDATION_FOLDER
                        Path to preprocessed validation folder
  --validation-reports-csv VALIDATION_REPORTS_CSV
                        Path to validation reports csv
  --validation-labels-csv VALIDATION_LABELS_CSV
                        Path to validation labels csv
  --results-folder RESULTS_FOLDER
                        Path to results folder
```

## Citing Us
If you use CT-RATE or CT-CLIP, the original authors would appreciate your references to [their paper](https://arxiv.org/abs/2403.17834).


## License
The original CT-CLIP project is licensed under [Creative Commons Attribution (CC-BY-NC-SA) license](https://creativecommons.org/licenses/by-nc-sa/4.0/). 
