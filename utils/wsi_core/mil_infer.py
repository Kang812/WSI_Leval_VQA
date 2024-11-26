import cv2
import torch
import numpy as np
import sys
import  albumentations as A
sys.path.append("/workspace/whole_slide_image_LLM/wsi_level_vqa-main/")
from models.classifier_model import model_ckpt_load
from utils.classifier import infer
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from models.C2C import *
from models.C2C.models.resnet import *
import torch.optim as optim
from models.C2C.utils import *

class Wsi_DataSet(Dataset):
    def __init__(self, image_path, patch_size, stride, model_name, model_ckpt, tile_n, transform=None):
        self.image = cv2.imread(image_path)
        self.patch_size = patch_size
        self.stride = stride
        self.model = model_ckpt_load(model_name, True, model_ckpt, 2)
        self.transform = transform
        self.tile_n = tile_n
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.start_coords = self._get_patch_coords(self.height, self.width, self.patch_size, self.stride)

    def _get_patch_coords(self, height, width, patch_size, stride):
        start_coords = []
        for h in range(0, height, patch_size - stride):
            for w in range(0, width, patch_size - stride):
                start_w = min(w, width - patch_size)
                start_h = min(h, height - patch_size)
                start_coords.append([start_w, start_h])
        return start_coords

    def __len__(self):
        return len(self.start_coords)

    def __image_size__(self):
        return self.height, self.width

    def __getitem__(self, index):
        start_coord = self.start_coords[index]
        crop_img = self.image[
            start_coord[1]:start_coord[1] + self.patch_size,
            start_coord[0]:start_coord[0] + self.patch_size,
            :
        ]

        if self.transform:
            crop_img = self.transform(crop_img)

        # Perform inference
        predict, prob = infer(self.model, crop_img, device=torch.device('cpu'))
        prob = max(prob.detach().cpu().numpy()[0])

        return {
            "image": crop_img,
            "coord": start_coord,
            "predict": predict,
            "prob": prob,
        }

    def create_bag(self):
        tumor_tiles = []
        not_tumor_tiles = []

        for coord in self.start_coords:
            crop_img = self.image[
                coord[1]:coord[1] + self.patch_size,
                coord[0]:coord[0] + self.patch_size,
                :
            ]
            predict, prob = infer(self.model, crop_img, device=torch.device('cpu'))
            prob = max(prob.detach().cpu().numpy()[0])

            if predict == 1 and prob > 0.7:
                tumor_tiles.append((crop_img, coord, prob))
            else:
                not_tumor_tiles.append((crop_img, coord, prob))

        # Sort by probability and select tiles
        tumor_tiles = sorted(tumor_tiles, key=lambda x: x[2], reverse=True)[:self.tile_n]
        not_tumor_tiles = sorted(not_tumor_tiles, key=lambda x: x[2], reverse=True)[:max(0, self.tile_n - len(tumor_tiles))]

        bag = tumor_tiles + not_tumor_tiles
        
        if self.transform is not None:
            tiles = [self.transform(image=tile[0])['image'] for tile in bag]
        else:
            tiles = [tile[0] for tile in bag]  # Extract images

        # Convert tiles to tensor
        tiles = np.stack(tiles, axis=0)  # Convert list of images to a single numpy array
        tiles = torch.tensor(tiles, dtype=torch.float32).permute(0, 1, 2, 3,)
        return tiles

def mil_model_infer(model_ckpt, bag):

    model_ft = WSIClassifier(2, bn_track_running_stats=True)
    model_ft.eval()
    
    optimizer = optim.Adam(model_ft.parameters(), lr=1e-4)
    model_ft, optimizer = load_ckp(model_ckpt, model_ft, optimizer)
    outputs, outputs_patch, outputs_attn = model_ft(bag)

    return int(torch.argmax(outputs).numpy())

if __name__ == '__main__':
    image_path = '/workspace/whole_slide_image_LLM/data/test_imgs/BC_01_0011.png'
    model_name = 'efficientnet_b3'
    model_ckpt = '/workspace/whole_slide_image_LLM/data/classification_dataset/patches/save_model/backup/best.pt'
    patch_size = 300
    stride = 0
    tile_n = 16
    
    data_transforms = A.Compose([
        A.Resize(256,256),
        ToTensorV2()]) 
    
    infer_data_set = Wsi_DataSet(image_path, patch_size, stride, model_name, model_ckpt, tile_n, data_transforms)
    bag = infer_data_set.create_bag()