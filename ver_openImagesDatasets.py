import os
import json
import argparse
import numpy as np
from PIL import Image
import random


def process_dataset(root_dir, output_file):
    data = []
    image_id = 0  # Unique image identifier

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, 'image.png')
            mask_path = os.path.join(folder_path, 'mask.png')
            prompt_path = os.path.join(folder_path, 'prompt.txt')

            # Skip incomplete folders
            if not os.path.exists(image_path) or not os.path.exists(mask_path) or not os.path.exists(prompt_path):
                print(f"Skipping incomplete folder: {folder_name}")
                continue

            # Load mask image
            mask = Image.open(mask_path).convert("L")
            mask_array = np.array(mask)

            # Compute bounding box
            nonzero = np.argwhere(mask_array > 0)
            if len(nonzero) == 0:
                print(f"No mask area in: {folder_name}")
                continue

            y_min, x_min = nonzero.min(axis=0)
            y_max, x_max = nonzero.max(axis=0)

            # Normalize bounding box coordinates
            mask_width, mask_height = mask.size
            x, y = x_min / mask_width, y_min / mask_height
            w, h = (x_max - x_min) / mask_width, (y_max - y_min) / mask_height

            # Read text description
            with open(prompt_path, 'r') as f:
                prompt = f.read().strip()

            # Create data entry
            item = {
                'mask_image': mask_path,
                'label': prompt,
                'image': image_path,
                'box': [x, y, w, h],
                'box_id': f"{image_id}_box",
                'image_id': f"{image_id}"
            }

            data.append(item)
            image_id += 1

    # Shuffle dataset
    random.shuffle(data)

    # Split dataset (80% train, 20% validation)
    split_index = int(len(data) * 0.8)
    train_data = data[:split_index]
    val_data = data[split_index:]

    # Save train and validation JSON files
    train_output_file = output_file.replace(".json", "_train.json")
    val_output_file = output_file.replace(".json", "_val.json")

    with open(train_output_file, 'w') as f:
        json.dump(train_data, f, indent=4)
    print(f"Train dataset saved to {train_output_file} with {len(train_data)} samples")

    with open(val_output_file, 'w') as f:
        json.dump(val_data, f, indent=4)
    print(f"Validation dataset saved to {val_output_file} with {len(val_data)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset and split into train/val JSON files.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--output_file", type=str, required=False, help="Base name for the output JSON file.", default='openimages_format.json')
    args = parser.parse_args()

    process_dataset(args.root_dir, args.output_file)

