import json
import cv2
import numpy as np
from PIL import Image,ImageOps
from mldm.util import masking,patchify_mask,crop_512
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random
import numpy as np
import os

class VerGenetrateDataset(Dataset):
	def __init__(self, json_path, mode='test'):
		with open(json_path, 'r') as f:
			self.data = json.load(f)
		self.mode = mode
		if json_path:
			with open(json_path, 'r') as f:
				self.data = json.load(f)
		else:
			raise ValueError("A valid json_path must be provided.")
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		item = self.data[idx]
		
		mask_filename = item['mask_image']
		label = item['label']
		image_filename = item['image']
		
		image = Image.open(image_filename).convert("RGB")
		mask = Image.open(mask_filename).convert("L")
		
		image_crop_512 = image.resize((512, 512))
		raw_image = np.array(image.resize((512, 512)))
		mask_crop_512 = mask.resize((512, 512))
		raw_mask = np.array(mask.resize((512, 512)))
		
		available_idx = [i for i in range(len(self.data)) if i != idx]
		random_idx = random.choice(available_idx)
		random_item = self.data[random_idx]
		
		random_mask_filename = random_item['mask_image']
		random_label = random_item['label']
		random_image_filename = random_item['image']
		random_box = random_item["box"]
		random_box_id = random_item["box_id"]
		random_mage_id = random_item["image_id"]
		
		random_image = Image.open(random_image_filename).convert("RGB")
		random_mask = Image.open(random_mask_filename).convert("L")
		
		random_image_crop_512 = random_image.resize((512, 512))
		random_raw_image = np.array(random_image.resize((512, 512)))
		random_mask_crop_512 = random_mask.resize((512, 512))
		random_raw_mask = np.array(random_mask.resize((512, 512)))
		
		return dict(
			jpg=image_crop_512,
			raw_image=raw_image,  #
			raw_mask=raw_mask,  #
			random_raw_image=random_raw_image,  #
			random_raw_mask=random_raw_mask,  #
			txt=label,  #
			random_txt=random_label,  #
			mask_filename=mask_filename,  #
		)
	
	@staticmethod
	def masking(image, mask, return_pil=True):
		image_array = np.array(image)
		mask_array = np.array(mask) > 0
		masked_image = image_array * np.expand_dims(mask_array, axis=-1)
		
		if return_pil:
			return Image.fromarray(masked_image.astype(np.uint8))
		else:
			return masked_image
