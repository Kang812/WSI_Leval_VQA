<<<<<<< HEAD
<<<<<<< HEAD
# WSI_Leval_VQA
=======
=======
>>>>>>> origin/main
# wsi_level_vqa

## Environment

```
sudo apt-get update
sudo apt-get install -y libvips-dev
```

```
pip install -r requirement.txt
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

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
<<<<<<< HEAD
- [Cluster-to-Conquer: A Framework for End-to-End Multi-Instance Learning for Whole Slide Image Classification](https://arxiv.org/abs/2103.10626)
>>>>>>> d10a0f6 (wsi_vqa)
=======
- [Cluster-to-Conquer: A Framework for End-to-End Multi-Instance Learning for Whole Slide Image Classification](https://arxiv.org/pdf/2103.10626)
>>>>>>> origin/main
