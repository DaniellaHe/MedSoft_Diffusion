import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from mldm.logger import ImageLogger
from mldm.model import create_model, load_state_dict
import argparse
from pytorch_lightning import seed_everything
import torch
from dataset import VerOpenImagesDataset, VerGenetrateDataset

def main(args):
    # Configs
    resume_path = args.ckpt
    batch_size = 1
    logger_freq = 300
    eta = 0.0
    scale = 7.5
    ddim_steps = 30
    seed = 0
    seed_everything(seed)

    root_dir = args.output_dir
    for subdir in ["generated_image", "text", "combined"]:
        if not os.path.exists(os.path.join(root_dir, subdir)):
            os.makedirs(os.path.join(root_dir, subdir))

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.config).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.eta = eta
    model.scale = scale
    model.ddim_steps = ddim_steps
    model.batch_size = batch_size
    model.root_dir = root_dir

    # dtype = torch.float16
    # if dtype == torch.float16:
    #     model = model.half()
    #     model.fusion_model.dtype = model.dtype
    #     model.model.diffusion_model.dtype = model.dtype

    # # Misc
    # test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=batch_size, shuffle=False, drop_last=True)
    #
    # trainer = pl.Trainer(gpus=1)
    #
    # trainer.test(model, test_dataloader)

    # # Misc
    # gen_dataset = VerGenetrateDataset(json_path="openimages_format.json", mode='gen')
    # gen_dataloader = DataLoader(gen_dataset, num_workers=1, batch_size=batch_size, shuffle=False, drop_last=True)
    #
    # trainer = pl.Trainer(gpus=1)
    #
    # trainer.test(model, gen_dataloader)

    # Misc
    gen_dataset = VerOpenImagesDataset(json_path="openimages_format_healthy_train.json")
    gen_dataloader = DataLoader(gen_dataset, num_workers=1, batch_size=batch_size, shuffle=False, drop_last=True)

    trainer = pl.Trainer(gpus=1)

    trainer.test(model, gen_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Testing Script")
    parser.add_argument('--ckpt', type=str, required=False, default='./stage2_1/last.ckpt')
    parser.add_argument('--output_dir', type=str, required=False, default='./generated_images/pepper_salt_results_soft')
    parser.add_argument('--config', type=str, required=False, default='./models/mldm_v15.yaml')
    
    args = parser.parse_args()
    main(args)
