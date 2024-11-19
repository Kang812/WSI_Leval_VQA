import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

def patch_generate(image_path, mask_path, patch_size, stride, save_dir):
    img = cv2.imread(image_path)
    file_name = image_path.split("/")[-1]

    mask = cv2.imread(mask_path, 0)
    height, width = img.shape[:2]

    for w in range(0, height, patch_size - stride):
        for h in range(0, width, patch_size - stride):
            start_w = w
            end_w = w + 300
            
            if end_w > width:
                end_w = width
                start_w = end_w - 300

            start_h = h
            end_h = h + 300

            if end_h > height:
                end_h = height
                start_h = end_h - 300

            crop_img = img[start_h:end_h, start_w:end_w, :]
            crop_mask = mask[start_h:end_h, start_w:end_w]
            label, label_cnt = np.unique(crop_mask, return_counts = True)
            
            if 1 not in list(label):
                cv2.imwrite(os.path.join(save_dir, 'images', 'not_tumor', f"{start_w}_{start_h}" + file_name), crop_img)
            else:
                cv2.imwrite(os.path.join(save_dir, 'images', 'tumor', f"{start_w}_{start_h}" + file_name), crop_img)

if __name__ == '__main__':

    image_paths = glob("/workspace/whole_slide_image_LLM/data/classification_dataset/crop_image/*.png")
    patch_size = 300
    stride = 150
    save_dir = '/workspace/whole_slide_image_LLM/data/classification_dataset/patches/'

    for i in tqdm(range(len(image_paths))):
        image_path = image_paths[i]
        mask_path = image_path.replace("crop_image", "crop_mask")
        patch_generate(image_path, mask_path, patch_size, stride, save_dir)