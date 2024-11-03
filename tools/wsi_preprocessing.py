import argparse
import os
<<<<<<< HEAD
import sys
sys.path.append("/workspace/whole_slide_image_LLM/wsi_level_vqa-main/")
=======
>>>>>>> origin/main
from utils.get_patchs import convert_png_to_deepzoom
from tqdm import tqdm
from glob import glob


def main(args):
    img_paths = glob(os.path.join(args.data_path, f"*.{args.data_format}"))
    img_paths = img_paths

    for i in tqdm(range(len(img_paths))):
        img_path = img_paths[i]
        filename = img_path.split("/")[-1].split(".")[0]
        output_prefix = os.path.join(args.save_dir, filename)
        convert_png_to_deepzoom(img_path, output_prefix, tile_size=args.patch_size, overlap=args.overlap)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a patch from the whole slide image')
    parser.add_argument('--data_path', type = str, default="/workspace/whole_slide_image_LLM/data/train_imgs/")
    parser.add_argument('--data_format', type =str, default="png")
    parser.add_argument('--patch_size', type = int, default=256)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--save_dir', type = str, default="/workspace/whole_slide_image_LLM/data/train_patchs/")
    args = parser.parse_args()
    
    main(args)