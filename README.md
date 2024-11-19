
# WSI_Leval_VQA

## Environment

```
sudo apt-get update
sudo apt-get install -y libvips-dev
```

```
pip install -r requirement.txt
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```
## Image Preprocessing for Weakly Supervised Learning
- To create a bag by obtaining patches only from tumor regions, a rule-based approach is applied to select tissues located within mask regions among multiple overlapping tissues, followed by cropping.
- Please refer to the image_preprocessing.py file and ./utils/wsi_core for the related code.

![image1](./img/image_preprocessing.png)

- Subsequently, paths were generated to train the classification model. The reference code can be found at ./utils/wsi_core/get_patchs.py.

## Weakly Supervised Learning
- Learning a classification model for weakly supervised learning
- The reason for choosing weakly supervised learning is that there is not enough data with masks available. Instead of using a segmentation approach, this method was chosen to detect tumor regions through weakly supervised learning.

```
./classifier_train.sh

```
- This model is trained to extract only the patches that are crucial for classifying whole slide images during patch extraction
- Hyper parameter
  - model_name : timm model name  
  - num_classes : Number of predicted classes
  - train_dataframe_path : train dataframe path 
  - valid_dataframe_path : valie dataframe path  
  - epochs : Number of training epochs
  - train_batch_size : train batch size
  - valid_batch_size : valid batch size
  - device : cuda or cpu
  - output_path : Training model save location

## MIL Model Train

```
python MIL_train.py --device cuda --num_classes 2 --num_epochs 30 --df_path /workspace/whole_slide_image_LLM/data/patches_train.csv --lr 1e-4 --num_cluster 8 --num_img_per_cluster 8 --fpath /workspace/whole_slide_image_LLM/wsi_level_vqa-main/model_save/MIL/checkpoint.pt --bn_track_running_stats
```
- device : cuda or cpu
- num_classes : Number of classes
- num_epochs : Number of training epochs
- df_path : Training dataset
- lr : Learning rate
- num_img_per_cluster : bag size
- fpath : Model save path 

## Data Generate
- Generate data for training MIL models
  - Create patch data from whole slide image data in png format
```
./wsi_create_patch.sh

```

## Reference
- [WSI-VQA: Interpreting Whole Slide Images by Generative Visual Question Answering](https://arxiv.org/abs/2407.05603)
  -[github](https://github.com/cpystan/WSI-VQA/tree/master?tab=readme-ov-file)
- [MamMIL: Multiple Instance Learning for Whole Slide Images with State Space Models](https://arxiv.org/pdf/2403.05160)
- [TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification](https://arxiv.org/abs/2106.00908)
- [Generalizable Whole Slide Image Classification with Fine-Grained Visual-Semantic Interaction](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Generalizable_Whole_Slide_Image_Classification_with_Fine-Grained_Visual-Semantic_Interaction_CVPR_2024_paper.pdf)
- [ViLa-MIL: Dual-scale Vision-Language Multiple Instance Learning for Whole Slide Image Classification](https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ViLa-MIL_Dual-scale_Vision-Language_Multiple_Instance_Learning_for_Whole_Slide_Image_CVPR_2024_paper.pdf)
- [Cluster-to-Conquer: A Framework for End-to-End Multi-Instance Learning for Whole Slide Image Classification](https://arxiv.org/pdf/2103.10626)

