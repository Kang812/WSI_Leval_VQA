import cv2
import os
from glob import glob
from utils.wsi_core.mask_gen import remove_background
from utils.wsi_core.crop import crop_image

mask_paths = glob("/workspace/whole_slide_image_LLM/data/train_masks/*.png")
for mask_path in mask_paths:
    img_path = mask_path.replace("train_masks", "train_imgs")

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)
    mask[mask != 255] = 3
    mask[mask == 255] = 0
    mask[mask == 3] = 1
    
    img_mask = remove_background(img)
    croping_img, crop_mask = crop_image(img, img_mask, mask)

    cv2.imwrite(os.path.join("/workspace/whole_slide_image_LLM/data/classification_dataset/crop_image/", img_path.split("/")[-1]), croping_img)
    cv2.imwrite(os.path.join("/workspace/whole_slide_image_LLM/data/classification_dataset/crop_mask/", img_path.split("/")[-1]), crop_mask)