python tools/classifier_train.py \
    --model_name "efficientnet_b3" \
    --num_classes 2 \
    --train_dataframe_path /workspace/whole_slide_image_LLM/data/classification_dataset/patches/images/train.csv \
    --valid_dataframe_path /workspace/whole_slide_image_LLM/data/classification_dataset/patches/images/val.csv \
    --epochs 20 \
    --train_batch_size 16 \
    --valid_batch_size 4 \
    --device cuda \
    --output_path /workspace/whole_slide_image_LLM/data/save_model/2024_11_21/best.pt
