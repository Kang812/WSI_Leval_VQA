#python MIL_train.py --device cuda --num_classes 2 --num_epochs 30 --df_path /workspace/whole_slide_image_LLM/data/patches_train.csv --lr 1e-4 --num_cluster 8 --num_img_per_cluster 8 --fpath /workspace/whole_slide_image_LLM/wsi_level_vqa-main/model_save/MIL/checkpoint.pt --bn_track_running_stats
python MIL_train.py --device cuda --num_classes 2 --num_epochs 30 --df_path /workspace/whole_slide_image_LLM/data/patches_train.csv --lr 1e-4 --num_cluster 8 --num_img_per_cluster 8 --fpath /workspace/whole_slide_image_LLM/wsi_level_vqa-main/model_save/MIL2/checkpoint.pt
python MIL_train.py --device cuda --num_classes 2 --num_epochs 30 --df_path /workspace/whole_slide_image_LLM/data/patches_train.csv --lr 1e-4 --num_cluster 8 --num_img_per_cluster 8 --fpath /workspace/whole_slide_image_LLM/wsi_level_vqa-main/model_save/MIL3/checkpoint.pt --bn_track_running_stats
