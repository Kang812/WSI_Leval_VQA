import albumentations as A
import cv2
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import AblationCAM, GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from classifier import infer

def get_grad_cam_prob_mask(model, image_path, device):
    
    val_transforms= A.Compose([
        A.LongestMaxSize(512, interpolation=cv2.INTER_NEAREST),
        A.PadIfNeeded(min_height=512, min_width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()])
    
    prediction, prob = infer(model, image_path, device)
    model.to(torch.device("cuda"))

    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32) / 255.0

    input_tensor = val_transforms(image = img)['image']
    input_tensor = input_tensor.unsqueeze(0)

    #target_layers = [model.model.conv_head]
    #target_layers = [model.model.blocks[-1][-1].conv_pwl]
    target_layers = [model.model.conv_head]
    targets = [ClassifierOutputTarget(prediction)]
    
    with AblationCAM(model=model, target_layers=target_layers,) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        #grayscale_cam[grayscale_cam < 0.5] = 0
        
    return grayscale_cam

