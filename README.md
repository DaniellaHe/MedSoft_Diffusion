# MedSoft_Diffusion
MedSoft-Diffusion: Medical Semantic-Guided Diffusion Model with Soft Mask Conditioning for Vertebral Disease Diagnosis

## Installation

**1. Prepare the environment**

```sh
conda env create -f environment.yaml
conda activate medsoft
```

**2. Data prepocess**

To prepare the dataset for training, we process medical images and their corresponding masks and text descriptions into a structured JSON format. The `ver_openImagesDatasets.py` automatically extracts relevant information and generates an annotation file. Ensure your dataset is organized as follows:

```sh
dataset/
├── sample_001/
│   ├── image.png   # Original medical image
│   ├── mask.png    # Corresponding mask image
│   ├── prompt.txt  # Text description
├── sample_002/
│   ├── image.png
│   ├── mask.png
│   ├── prompt.txt
├── ...
```

Use the following command to process the dataset:

```sh
python ver_openImagesDatasets.py 
--root_dir /path/to/dataset 
--output_file openimages_format.json
```

The processed data is saved in a JSON file, where each entry includes:

```sh
[
    {
        "mask_image": "/path/to/dataset/sample_001/mask.png",
        "label": "Text description",
        "image": "/path/to/dataset/sample_001/image.png",
        "box": [0.15, 0.2, 0.5, 0.6],
        "box_id": "0_box",
        "image_id": "0"
    }
]
```

Even though healthy samples do not contain a disease description, the `prompt.txt` file **must still exist** in each sample folder. For healthy samples, the `prompt.txt` file should contain only `"healthy"`. This ensures consistency in data formatting and compatibility with the processing script.

## Train:

The training process is divided into two stages. 

In the first stage, the Medical Semantic Controller (MSC) and the enhanced cross-attention layers in the U-Net are optimized using feature distillation loss and diffusion loss, respectively. 

In the second stage, the full model MedSoft_Diffusion is finetuned using diffusion loss. Specifically, only the Medical Semantic Controller (MSC) is tuned keeping the other modules frozen in early steps, and in later steps, the enhanced cross-attention layers in the U-Net is unfrozen and tuned as well.

### Stage1:

**1. Convert the weights of the [base model](https://huggingface.co/runwayml/stable-diffusion-inpainting/tree/main) (sd-v1-5-inpainting.ckpt) into customized format:**

```sh
python ./tool/model_convert.py --input_path sd-v1-5-inpainting.ckpt --output_path ckpt_for_stage1.ckpt --config ./models/mldm_v15.yaml
```

**2. Train Medical Semantic Controller (MSC):**

We use the medical multi-model [biomedclip]([microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 · Hugging Face](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)) to train MSC. 

```sh
python ver_train.py --ckpt ckpt_for_stage1.ckpt --config ./models/mldm_v15_stage1.yaml --save_path ./stage1_MSC
```

**3. Train the enhanced cross-attention layers in the U-Net:**

```sh
python ver_train.py --ckpt ckpt_for_stage1.ckpt --config ./models/mldm_v15_unet_only.yaml --save_path ./stage1_Unet
```

### Stage2:

**1. Merge the pre-trained weights from Stage 1:**

```sh
python ./tool/tool_merge_for_stage2.py --stage1_path ./stage1_MSC/last.ckpt --input_path ./stage1_Unet/last.ckpt --output_path ckpt_for_stage2.ckpt --config ./models/mldm_v15.yaml
```

**2. Finetune Medical Semantic Controller (MSC) with diffusion loss:**

```sh
python ver_train.py --ckpt ckpt_for_stage2.ckpt --config ./models/mldm_v15_stage2_1.yaml --save_path ./stage2_1
```

**3. Full finetune:**

```sh
python ver_train.py --ckpt stage2_1/last.ckpt --config ./models/mldm_v15_stage2_1.yaml --save_path ./stage2_2
```

## Inference

**Infer with dataset:**

```sh
python dataset_infer.py --ckpt test.ckpt --output_dir results --config ./models/mldm_v15.yaml
```

## Classification

We use synthetic diseased images generated from the normal images in the training dataset to augment our training set, enhancing the classifier's performance. We use the following code to perform classification:

```sh
python classifier.py args1.json
```
