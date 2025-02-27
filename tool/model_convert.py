import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import torch
from mldm.model import create_model, load_state_dict

def main(args):
    input_path = args.input_path
    output_path = args.output_path

    assert os.path.exists(input_path), 'Input model does not exist.'
    assert not os.path.exists(output_path), 'Output filename already exists.'
    # assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

    model = create_model(config_path=args.config)

    pretrained_weights = load_state_dict(input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    scratch_dict = model.state_dict()

    target_dict = {}
    for k in scratch_dict.keys():
        if k in pretrained_weights and pretrained_weights[k].shape == scratch_dict[k].shape:
            target_dict[k] = pretrained_weights[k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')

    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), output_path)
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Checkpoint Conversion Script")
    parser.add_argument('--input_path', type=str, required=False,
                        default='../stable-diffusion-inpainting/sd-v1-5-inpainting.ckpt')
    parser.add_argument('--output_path', type=str, required=False, default='ckpt_for_stage1.ckpt')
    parser.add_argument('--config', type=str, required=False, default='./models/mldm_v15.yaml')
    
    args = parser.parse_args()
    main(args)
