python tools/classifier_train.py \
    --model_name "efficientnet_b0" \
    --num_classes 2 \
    --train_dataframe_path /workspace/whole_slide_image_LLM/data/image/train.csv \
    --valid_dataframe_path /workspace/whole_slide_image_LLM/data/image/valid.csv \
    --epochs 30 \
    --train_batch_size 8 \
    --valid_batch_size 4 \
    --device cuda \
    --output_path /workspace/whole_slide_image_LLM/data/image/save_model/best.pt