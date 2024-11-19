import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.classifier import trainer
import argparse
import torch
from models.classifier_model import load_model


def main(model, device, args):
    trainer(model, args.train_dataframe_path, args.valid_dataframe_path,
            args.epochs, args.train_batch_size , args.valid_batch_size, args.num_classes,
            args.output_path, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Classifier')
    parser.add_argument('--model_name', type = str, default="efficientnet_b0")
    parser.add_argument('--num_classes', type = int, default=2)
    parser.add_argument('--train_dataframe_path', type = str, default="/workspace/whole_slide_image_LLM/data/image/train.csv")
    parser.add_argument('--valid_dataframe_path', type = str, default="/workspace/whole_slide_image_LLM/data/image/valid.csv")
    parser.add_argument('--epochs', type = int, default=15)
    parser.add_argument('--train_batch_size', type = int, default=8)
    parser.add_argument('--valid_batch_size', type = int, default=4)
    parser.add_argument('--device', type = str, default="cuda")
    parser.add_argument('--output_path', type = str, default="/workspace/whole_slide_image_LLM/data/image/save_model/")
    args = parser.parse_args()

    model = load_model(args.model_name, True, args.num_classes)
    device = torch.device(args.device)
    print(args)
    main(model, device, args)


