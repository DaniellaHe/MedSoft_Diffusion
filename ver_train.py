import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import VerOpenImagesDataset
from mldm.logger import ImageLogger
from mldm.model import create_model, load_state_dict
import argparse
import torch


# import torch
# torch.cuda.init()

def main(args):
    # Configurations
    resume_path = args.ckpt  # Path to the checkpoint file
    save_dir = args.save_path  # Directory to save training outputs

    # Create the save directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    batch_size = 16  # Batch size for training MSC
    # batch_size = 1  # Batch size for training Unet
    logger_freq = 400  # Frequency of logging images during training

    # Load model on CPU first; Pytorch Lightning will handle GPU transfer
    model = create_model(args.config).cpu()

    # Handle loading checkpoint weights based on model type
    if args.config == './models/mldm_v15_unet_only.yaml':
        # Load checkpoint and filter out unnecessary parameters
        checkpoint = torch.load(resume_path, map_location="cpu")
        state_dict = checkpoint
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fusion_model.")}
        checkpoint["state_dict"] = filtered_state_dict
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'))

    # Set learning rates for different components
    model.fusion_learning_rate = 1e-4
    model.diffusion_learning_rate = 1e-5

    # Prepare datasets
    train_dataset = VerOpenImagesDataset(json_path="openimages_format_train.json")
    val_dataset = VerOpenImagesDataset(json_path="openimages_format_val.json")

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, num_workers=1, batch_size=batch_size, shuffle=True, prefetch_factor=2
    )
    val_dataloader = DataLoader(
        val_dataset, num_workers=1, batch_size=batch_size, shuffle=True
    )

    # Initialize logger for visualizing generated images
    logger = ImageLogger(batch_frequency=logger_freq)

    # Define checkpoint callback to save the best models
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir,
        filename='{epoch:02d}--{val_loss:6f}',
        save_top_k=2,  # Keep top 2 models with the lowest validation loss
        monitor='val_loss',
        mode='min',
        save_last=True  # Always keep the latest checkpoint
    )

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        gpus=1,  # Use one GPU
        precision=32,  # Use 32-bit precision
        max_epochs=20,  # Train for 2 epochs
        val_check_interval=0.5,  # Validate twice per epoch
        callbacks=[checkpoint_callback]
    )

    # Start training
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Training Script")

    # Arguments for different training stages

    # Train MSC
    parser.add_argument('--ckpt', type=str, required=False, default='ckpt_for_stage1.ckpt', help="Path to checkpoint")
    parser.add_argument('--config', type=str, required=False, default='./models/mldm_v15_stage1.yaml',
                        help="Path to model config file")
    parser.add_argument('--save_path', type=str, required=False, default='./stage1_MSC',
                        help="Directory to save results")

    # Training Unet
    # parser.add_argument('--ckpt', type=str, required=False, default='ckpt_for_stage1.ckpt')
    # parser.add_argument('--config', type=str, required=False, default='./models/mldm_v15_unet_only.yaml')
    # parser.add_argument('--save_path', type=str, required=False, default='./stage1_Unet')

    # Finetuning MSC with diffusion loss
    # parser.add_argument('--ckpt', type=str, required=False, default='ckpt_for_stage2.ckpt')
    # parser.add_argument('--config', type=str, required=False, default='./models/mldm_v15_stage2_1.yaml')
    # parser.add_argument('--save_path', type=str, required=False, default='./stage2_1')

    # Full Finetuning
    # parser.add_argument('--ckpt', type=str, required=False, default='stage2_1/last.ckpt')
    # parser.add_argument('--config', type=str, required=False, default='./models/mldm_v15_stage2_2.yaml')
    # parser.add_argument('--save_path', type=str, required=False, default='./stage2_2')

    args = parser.parse_args()
    main(args)
