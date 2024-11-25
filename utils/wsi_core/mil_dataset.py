import cv2
import torch
import numpy as np
import os
import sys
sys.path.append("/workspace/whole_slide_image_LLM/wsi_level_vqa-main/")
from utils.classifier import infer
from models.classifier_model import model_ckpt_load
from tqdm import tqdm
from glob import glob

def calculate_white_ratio(patch, threshold=240):

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    white_pixels = np.sum(gray > threshold)
    total_pixels = gray.size
    white_pixel_ratio = white_pixels / total_pixels
    return white_pixel_ratio

def mil_dataset_gen(model, image_path, patch_size, stride, tile_n, device, white_threshold, save_dir):
    img = cv2.imread(image_path)
    file_name = image_path.split("/")[-1]
    height, width = img.shape[:2]

    tumor_list = []
    not_tumor_list = []
    tumor_prob_list = []
    tumor_coords = []
    not_tumor_coords = []
    
    for w in range(0, width, patch_size - stride):
        for h in range(0, height, patch_size - stride):
            start_w = w
            end_w = w + patch_size
            
            if end_w > width:
                end_w = width
                start_w = end_w - patch_size

            start_h = h
            end_h = h + patch_size

            if end_h > height:
                end_h = height
                start_h = end_h - patch_size
            
            crop_img = img[start_h:end_h, start_w:end_w, :]
            white_pixel_ratio = calculate_white_ratio(crop_img)
            
            #if white_pixel_ratio > white_threshold:
            #    continue

            predict, prob = infer(model, crop_img, device=device)
            prob = max(prob.detach().cpu().numpy()[0])
            
            if predict == 1:
                if prob > 0.7:
                    if f'{start_w}_{start_h}_{end_w}_{end_h}_' in tumor_coords:
                        continue
                    else:
                        tumor_list.append(crop_img)
                        tumor_coords.append(f'{start_w}_{start_h}_{end_w}_{end_h}_')
                else:
                    if f'{start_w}_{start_h}_{end_w}_{end_h}_' in not_tumor_coords:
                        continue
                    else:
                        not_tumor_list.append(crop_img)
                        not_tumor_coords.append(f'{start_w}_{start_h}_{end_w}_{end_h}_')
            else:
                if f'{start_w}_{start_h}_{end_w}_{end_h}_' in not_tumor_coords:
                    continue
                else:
                    not_tumor_list.append(crop_img)
                    not_tumor_coords.append(f'{start_w}_{start_h}_{end_w}_{end_h}_')

    if len(tumor_list) >= tile_n:
        idxs = np.argsort(np.array(tumor_list).reshape(np.array(tumor_list).shape[0], -1).sum(-1))[:tile_n]
        tiles = np.array(tumor_list)[idxs]
        coords = np.array(tumor_coords)[idxs]
        
        if not os.path.exists(os.path.join(save_dir, file_name.replace(".png",""))):
            os.makedirs(os.path.join(save_dir, file_name.replace(".png","")))
        
        for tile, coord in zip(tiles, coords):
            cv2.imwrite(os.path.join(save_dir, file_name.replace(".png", ""), coord +  file_name), tile)
    elif tumor_list == []:
        idxs = np.argsort(np.array(not_tumor_list).reshape(np.array(not_tumor_list).shape[0], -1).sum(-1))[:tile_n]
        tiles = np.array(not_tumor_list)[idxs]
        coords = np.array(not_tumor_coords)[idxs]
        
        if not os.path.exists(os.path.join(save_dir, file_name.replace(".png",""))):
            os.makedirs(os.path.join(save_dir, file_name.replace(".png","")))

        for tile, coord in zip(tiles, coords):
            cv2.imwrite(os.path.join(save_dir, file_name.replace(".png", ""), coord +  file_name), tile)
    else:
        tumor_idxs = np.argsort(np.array(tumor_list).reshape(np.array(tumor_list).shape[0], -1).sum(-1))
        tumor_tiles = np.array(tumor_list)[tumor_idxs]

        not_tumor_idxs = np.argsort(np.array(not_tumor_list).reshape(np.array(not_tumor_list).shape[0], -1).sum(-1))[:tile_n - len(tumor_idxs)]
        not_tumor_tiles = np.array(not_tumor_list)[not_tumor_idxs]

        if not os.path.exists(os.path.join(save_dir, file_name.replace(".png",""))):
            os.makedirs(os.path.join(save_dir, file_name.replace(".png","")))
        
        tiles = np.array(list(tumor_tiles) + list(not_tumor_tiles))
        coords = list(np.array(tumor_coords)[tumor_idxs]) + list(np.array(not_tumor_coords)[not_tumor_idxs])

        for tile, coord in zip(tiles, coords):
            cv2.imwrite(os.path.join(save_dir, file_name.replace(".png", ""), coord +  file_name), tile)

if __name__ == '__main__':
    model_name = 'efficientnet_b3'
    pretrained = True 
    num_classes = 2
    model_ckpt = '/workspace/whole_slide_image_LLM/data/classification_dataset/patches/save_model/best.pt'
    model = model_ckpt_load(model_name, pretrained, model_ckpt, num_classes)
    model.eval()
    
    image_paths = glob("/workspace/whole_slide_image_LLM/data/train_imgs/*.png")
    #image_paths = ["/workspace/whole_slide_image_LLM/data/train_imgs/BC_01_1231.png"]
    device = torch.device("cuda")
    patch_size = 300
    stride = 0
    tile_n = 16
    white_threshold = 0.8
    save_dir = "/workspace/whole_slide_image_LLM/data/patches_dataset/train/"

    #image_paths = image_paths[17:]
    for i in tqdm(range(len(image_paths))):
        image_path = image_paths[i]
        mil_dataset_gen(model, image_path, patch_size, stride, tile_n, device, white_threshold, save_dir)