# CT-CLIP: A foundation model utilizing chest CT volumes and radiology reports for supervised-level zero-shot detection of abnormalities
Welcome to the official repository of CT-CLIP, a pioneering work in 3D medical imaging with a particular focus on chest CT volumes. CT-CLIP provides an open-source codebase, pre-trained models, and a unique dataset (CT-RATE) of chest CT volumes paired with radiology text reports, all freely accessible to researchers.



## Requirements

Before you start, you must install the necessary dependencies. To do so, execute the following commands:

```setup
# Navigate to the 'super_resolution' directory and install the required packages
cd super_resolution
pip install -e .

# Return to the root directory
cd ..

# Navigate to the 'transformer_maskgit' directory and install its required packages
cd transformer_maskgit
pip install -e .

# Return to the root directory
cd ..
```
After following these steps, your environment should be properly set up with all required packages.

The MaskGIT Transformer model necessitates the use of an A100 GPU, with 80G of VRAM, for efficient training and inference operations, due to the model's considerable size.

## Training

Train the CT-ViT model by executing the following command in your terminal:

```train
accelerate launch --use_fsdp train_ctvit.py
```
To train the MaskGIT Transformer model, use the command provided below:

```train
accelerate launch train_transformer.py
```

Lastly, train the Super Resolution Diffusion model using the multi-line command outlined here:

```train
accelerate launch \
    --multi_gpu \
    --mixed_precision=fp16 \
    --num_machines=1 \
    train_superres.py --config superres.yaml --stage 2 --bs 8
```
Remember to replace the respective parameters with the ones relevant to your setup if necessary.


## Inference

To run inference on the CT-ViT model, use the following command:

```eval
python inference_ctvit.py
```

To infer with the MaskGIT Transformer model, execute the command below:

```eval
python inference_transformer.py
```

Lastly, for inference using the Super Resolution Diffusion model, issue this multi-line command:

```eval
accelerate launch \
    --multi_gpu \
    --mixed_precision=fp16 \
    --num_machines=1 \
    inference_superres.py --config superres_inference.yaml --stage 2 --bs 2
```
Remember to adjust the parameters as per your configuration requirements. 

## Sampling Times

Our performance metrics detail the sampling times for generating and upscaling 3D Chest CT volumes. It is important to note that these figures were derived from our tests on an NVIDIA A100 80GB GPU and may vary based on your system's configuration.

- **3D Chest CT Generation:** By leveraging the capabilities of CT-ViT and MaskGIT on an NVIDIA A100 80GB GPU, we are able to generate a low-resolution 3D Chest CT volume (128x128x201) from a given text input. This process takes approximately 30 seconds, making it efficient for real-time applications.

- **Text-Conditional Upsampling:** We utilize a layer-by-layer upsampling technique via the Diffusion Model to enhance the resolution of the generated Chest CT from 128x128x201 to 512x512x201. Each layer's upsampling operation is performed swiftly, taking less than a second on the same GPU. The entire upsampling process for all 201 slices of the CT volume takes around 150 seconds.


## Pretrained Models

For your convenience, we provide access to pretrained models directly. These models have been trained on our paired radiological report and chest CT volume dataset, as elaborated in the paper.

You can download the models from the following links:

- **CT-ViT Model**: [Download Here](https://huggingface.co/generatect/GenerateCT/resolve/main/pretrained_models/ctvit_pretrained.pt)

- **Transformer Model**: [Download Here](https://huggingface.co/generatect/GenerateCT/resolve/main/pretrained_models/transformer_pretrained.pt)

- **Super Resolution Diffusion Model**: [Download Here](https://huggingface.co/generatect/GenerateCT/resolve/main/pretrained_models/superres_pretrained.pt)

By leveraging these pretrained models, you can easily reproduce our results or further extend our work.


## Our Dataset (CT-RATE)

Explore and experiment with our example data, specifically curated for training the CT-ViT, Transformer, and Super Resolution Diffusion networks.

- [Download Example Data](https://huggingface.co/generatect/GenerateCT/resolve/main/example_data.zip)

Feel free to utilize this example data to gain insights into the training process of the components of GenerateCT.


## Generated Data

Explore our generated dataset, consisting of 2286 synthetic CT volumes and their corresponding text prompts. 

- [Download Generated Dataset](https://huggingface.co/generatect/GenerateCT/tree/main/generated_data)

The dataset includes synthetic chest CT volumes, medical language text prompts used in the generation process, and abnormality labels. It was utilized in the supplementary section of our paper to showcase the capabilities of GenerateCT. Feel free to utilize this dataset for research, analysis, or to gain a deeper understanding of the generated CT volumes and their associated text prompts. 


## Evaluation

In our evaluation process, we employed various metrics to assess the performance of our generated CT volumes. 

- **FID and FVD Metrics**: To calculate the Fréchet Inception Distance (FID) and Fréchet Video Distance (FVD), we utilized the evaluation script from the [StyleGAN-V repository](https://github.com/universome/stylegan-v).

- **CLIP Score Metric**: For the CLIP score evaluation, we relied on the [torchmetrics implementation](https://torchmetrics.readthedocs.io/en/stable/multimodal/clip_score.html).

Feel free to explore these metrics to gain a comprehensive understanding of the quality and performance of our generated CT volumes.


## Citing Us
If you use CT-RATE or CT-CLIP, we would appreciate your references to our paper.

## License
We are committed to fostering innovation and collaboration in the research community. To this end, all elements of the CT-RATE dataset are released under a [Creative Commons Attribution (CC-BY-NC-SA) license](https://creativecommons.org/licenses/by-nc-sa/4.0/). This licensing framework ensures that our contributions can be freely used for non-commercial research purposes, while also encouraging contributions and modifications, provided that the original work is properly cited and any derivative works are shared under similar terms.

## Acknowledgements
We would like to express our gratitude to the following repositories for their invaluable contributions to our work: [Phenaki Pytorch by Lucidrains](https://github.com/lucidrains/phenaki-pytorch), [Phenaki by LAION-AI](https://github.com/LAION-AI/phenaki), [Imagen Pytorch by Lucidrains](https://github.com/lucidrains/imagen-pytorch), [StyleGAN-V by universome](https://github.com/universome/stylegan-v), and [CT Net Models by Rachellea](https://github.com/rachellea/ct-net-models). We extend our sincere appreciation to these researchers for their exceptional open-source efforts. If you utilize our models and code, we kindly request that you also consider citing these works to acknowledge their contributions.

