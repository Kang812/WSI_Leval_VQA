python tools/classifier_train.py \
    --model_name "efficientnet_b3" \
    --num_classes 2 \
    --train_dataframe_path /workspace/whole_slide_image_LLM/data/classification_dataset/patches/images/train.csv \
    --valid_dataframe_path /workspace/whole_slide_image_LLM/data/classification_dataset/patches/images/val.csv \
    --epochs 15 \
    --train_batch_size 8 \
    --valid_batch_size 4 \
    --device cuda \
    --output_path /workspace/whole_slide_image_LLM/data/classification_dataset/patches/save_model/best.pt