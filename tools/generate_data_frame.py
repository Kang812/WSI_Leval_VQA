from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import cv2
import os
import numpy as np

def is_background(patch, threshold=0.8, brightness_threshold=200):
    """
    이미지 패치가 배경인지 여부를 판단합니다.

    Parameters:
    - patch: numpy array, 이미지 패치 (H, W, C)
    - threshold: float, 배경으로 간주되는 픽셀 비율의 기준 (0 ~ 1)
    - brightness_threshold: int, 픽셀이 배경으로 간주되는 밝기 기준 (0 ~ 255)

    Returns:
    - bool, True면 배경, False면 배경 아님
    """
    # 이미지가 컬러가 아닐 경우 그레이스케일로 변환
    if len(patch.shape) == 2:
        gray_patch = patch
    else:
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    
    # 밝기가 기준보다 높은 픽셀을 배경으로 간주
    background_pixels = np.sum(gray_patch > brightness_threshold)
    total_pixels = patch.shape[0] * patch.shape[1]
    
    # 배경 픽셀 비율 계산
    background_ratio = background_pixels / total_pixels

    # 배경 비율이 기준 이상이면 배경으로 판단
    return background_ratio >= threshold


def dataframe_gen(args, background_threshold):
    data_path = args.data_path
    label_df = pd.read_csv(args.label_df)
    zoom_folder = args.zoom_folder
    img_paths = glob(os.path.join(data_path, f"*/{zoom_folder}/*.jpg"))

    new_img_paths = []
    wsis = []
    labels = []
    is_valids = []
    sub_df = label_df[['ID', args.label_col]]
    train, valid = train_test_split(sub_df, test_size=0.2, random_state=42, stratify=sub_df[args.label_col])
    train_list = train['ID'].to_list()
    valid_list = valid['ID'].to_list()

    for i in tqdm(range(len(img_paths))):
        img_path = img_paths[i]
        patch = cv2.imread(img_path)
        if not is_background(patch, threshold=background_threshold, brightness_threshold=200):
            new_img_paths.append(img_path)
            wsi = img_path.split("/")[-3].replace("_files", "")
            wsis.append(wsi)
            if wsi in train_list:
                is_valids.append(0)
            else:
                is_valids.append(1)
            label = label_df[label_df['ID'] == img_path.split("/")[-3].replace("_files", "")][args.label_col].values[0]
            labels.append(label)
    
    df = pd.DataFrame({
        'path' : new_img_paths,
        'wsi' : wsis,
        'is_valid':is_valids,
        'label':labels})
    
    df.to_csv(args.save_dir, index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a patch from the whole slide image')
    parser.add_argument('--data_path', type = str, default="/workspace/whole_slide_image_LLM/data/train_patchs/")
    parser.add_argument('--label_df', type = str, default="/workspace/whole_slide_image_LLM/data/train.csv")
    parser.add_argument('--label_col', type = str, default="N_category")
    parser.add_argument('--zoom_folder', type = int, default=13)
    parser.add_argument('--save_dir', type = str, default='/workspace/whole_slide_image_LLM/data/patches_train.csv')
    args = parser.parse_args()

    dataframe_gen(args, 0.9)

