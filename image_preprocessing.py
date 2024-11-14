import cv2
import numpy as np
import os
from utils.wsi_core.crop import crop_image
from utils.wsi_core.mask_gen import remove_background
from tqdm import tqdm
from glob import glob

image_paths = glob("/workspace/whole_slide_image_LLM/data/test_imgs/*.png")
save_dir = '/workspace/whole_slide_image_LLM/data/image/test/'
#save_dir = '/workspace/whole_slide_image_LLM/data/image/train/'
#image_paths = image_paths[:60]
#image_paths = ['/workspace/whole_slide_image_LLM/data/train_imgs/BC_01_0006.png']

for i in tqdm(range(len(image_paths))):
    image_path = image_paths[i]
    file_name = image_path.split("/")[-1]

    image = cv2.imread(image_path)
    image  = remove_background(image)
    preprocessing_img = crop_image(image)

    save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(save_path, preprocessing_img)