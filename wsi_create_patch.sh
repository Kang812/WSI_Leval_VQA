python tools/wsi_preprocessing.py \
    --data_path /workspace/whole_slide_image_LLM/data/train_imgs/ \
    --data_format png \
    --patch_size 256 \
    --overlap 0 \
    --save_dir /workspace/whole_slide_image_LLM/data/train_patchs/

python tools/wsi_preprocessing.py \
    --data_path /workspace/whole_slide_image_LLM/data/test_imgs/ \
    --data_format png \
    --patch_size 256 \
    --overlap 0 \
    --save_dir /workspace/whole_slide_image_LLM/data/test_patchs/