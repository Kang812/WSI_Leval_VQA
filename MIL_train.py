import os
import random
import numpy as np
import pandas as pd

import torch
import argparse
import torch.nn as nn
import albumentations as A
import torch.optim as optim
from albumentations.pytorch import ToTensorV2

from models.C2C.models.resnet import *
from models.C2C import train
from models.C2C.loss import KLDLoss

torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)

def main(args):
    device = torch.device(args.device)
    df = pd.read_csv(args.df_path)

    # Initialize Model
    model_ft = WSIClassifier(args.num_classes, bn_track_running_stats=True)
    model_ft = model_ft.to(device)
    
    data_transforms = A.Compose([
        A.Resize(256,256),
        ToTensorV2()])    

    # Cross Entropy Loss 
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kld = KLDLoss()
    criterion_dic = {'CE': criterion_ce, 'KLD': criterion_kld}

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model_ft.parameters(), lr=args.lr)

    model_ft = train.train_model(model_ft,
                                 criterion_dic, 
                                 optimizer, 
                                 df, 
                                 data_transforms=data_transforms,
                                 alpha=1, 
                                 beta=0.01,
                                 gamma=0.01,
                                 num_epochs=args.num_epochs, 
                                 fpath=args.fpath,
                                 topk=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIL model Train')
    parser.add_argument('--device', type = str, default="cuda")
    parser.add_argument('--num_classes', type = int, default=2)
    parser.add_argument('--num_epochs', type = int, default=30)
    parser.add_argument('--df_path', type = str, default="/workspace/whole_slide_image_LLM/data/patches_train.csv")
    parser.add_argument('--lr', type = float, default=1e-4)
    parser.add_argument('--fpath', type = str, default="/workspace/whole_slide_image_LLM/wsi_level_vqa-main/model_save/MIL/checkpoint.pt")

    args = parser.parse_args()
    main(args)