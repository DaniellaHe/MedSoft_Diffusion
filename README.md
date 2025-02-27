# MedSoft_Diffusion

**MedSoft-Diffusion: Medical Semantic-Guided Diffusion Model with Soft Mask Conditioning for Vertebral Disease Diagnosis**

## Installation

Prepare the environment

```sh
conda env create -f environment.yaml
conda activate medsoft
```

## Data Preprocessing

To prepare the dataset for training, we process medical images and their corresponding masks and text descriptions into a structured JSON format. The `ver_openImagesDatasets.py` script automatically extracts relevant information and generates an annotation file. 

Ensure your dataset is organized as follows:

```plaintext
dataset/
├── sample_001/
│   ├── image.png   # Original medical image
│   ├── mask.png    # Corresponding soft mask image
│   ├── prompt.txt  # Text description
├── sample_002/
│   ├── image.png
│   ├── mask.png
│   ├── prompt.txt
├── ...
```

Process dataset:

```sh
python ver_openImagesDatasets.py \
  --root_dir /path/to/dataset \
  --output_file openimages_format.json
```

JSON format:

```json
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

**Important:** Even though healthy samples do not contain a disease description, the `prompt.txt` file **must still exist** in each sample folder. For healthy samples, the `prompt.txt` file should contain only `"healthy"`. This ensures consistency in data formatting and compatibility with the processing script.

## **Step1: Pretraining Medical Semantic Controller (MSC)**

**1. Convert Stable Diffusion Weights**

Before training, you need to convert the weights of **[Stable Diffusion](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)** (sd-v1-5-inpainting.ckpt) into a customized format:

```sh
python ./tool/model_convert.py \
  --input_path sd-v1-5-inpainting.ckpt \
  --output_path ckpt_for_stage1.ckpt \
  --config ./models/mldm_v15.yaml
```

**2. Train Medical Semantic Controller (MSC)**

We use the medical multi-model **[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)** to train MSC. 

```sh
python ver_train.py \
  --ckpt ckpt_for_stage1.ckpt \
  --config ./models/mldm_v15_stage1.yaml \
  --save_path ./stage1_MSC
```

## **Step 2: Training with Soft Mask Inpainting Strategy (SMIS)**

The **Soft Mask Inpainting Strategy (SMIS)** is designed to improve lesion synthesis realism by:

- Replacing hard lesion masks with **soft masks** (generated from anatomical segmentation masks or bounding boxes).
- Applying **Gaussian blurring** to ensure smooth lesion-background integration.
- Constraining the inpainting process to the **soft-masked lesion region**, preventing changes to the anatomical background.

The following steps train the U-Net's **enhanced cross-attention layers**, which integrate soft mask conditioning and medical semantic features.

**1. Training the U-Net's enhanced cross-attention layers**

```sh
python ver_train.py \
  --ckpt ckpt_for_stage1.ckpt \
  --config ./models/mldm_v15_unet_only.yaml \
  --save_path ./stage1_Unet
```

**2. Merge the Pre-trained Weights from Step 1**

```sh
python ./tool/tool_merge_for_stage2.py \
  --stage1_path ./stage1_MSC/last.ckpt \
  --input_path ./stage1_Unet/last.ckpt \
  --output_path ckpt_for_stage2.ckpt \
  --config ./models/mldm_v15.yaml
```

**3. Fine-tune MSC with Diffusion Loss**

```sh
python ver_train.py \
  --ckpt ckpt_for_stage2.ckpt \
  --config ./models/mldm_v15_stage2_1.yaml \
  --save_path ./stage2_1
```

**4. Fine-tuning the full MedSoft-Diffusion model using soft mask inpainting**

```sh
python ver_train.py \
  --ckpt stage2_1/last.ckpt \
  --config ./models/mldm_v15_stage2_1.yaml \
  --save_path ./stage2_2
```

## Step 3: Inference

During inference, the model uses **soft masks** to confine lesion synthesis to the intended region while preserving anatomical structures.

Infer with dataset:

```sh
python dataset_infer.py \
  --ckpt test.ckpt \
  --output_dir results \
  --config ./models/mldm_v15.yaml
```

## Classification

We use **synthetic diseased images** generated from normal images in the training dataset to augment the training set, thereby enhancing the classifier's performance.

Run Classification:

```sh
python classifier.py args1.json
```

