import cv2
import numpy as np
import os
from utils.wsi_core.crop import crop_image
from utils.wsi_core.mask_gen import remove_background
from tqdm import tqdm
from glob import glob

image_paths = glob("/workspace/whole_slide_image_LLM/data/train_imgs/*.png")
save_dir = '/workspace/whole_slide_image_LLM/data/image/train/'

for i in tqdm(len(image_paths)):
    image_path = image_paths[i]
    file_name = image_path.split("/")[-1]

    image = cv2.imread(image_path)
    image  = remove_background(image)
    preprocessing_img = crop_image(image)

    save_path = os.path.join(save_dir, file_name)