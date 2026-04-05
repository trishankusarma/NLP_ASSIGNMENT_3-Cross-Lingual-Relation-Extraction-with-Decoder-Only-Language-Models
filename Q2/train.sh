# This script should run the training and save checkpoints in the output/ dir
#!/bin/bash
# Q2/train.sh
OUTPUT_DIR=${1:-"Q2/output"}
echo "Training Task 2 → output: $OUTPUT_DIR"
python -m Q2.train \
    --english_train_file en_sft_dataset/train.jsonl \
    --hindi_train_file sft_dataset/hi_train.jsonl \
    --kanada_train_file sft_dataset/kn_train.jsonl \
    --oria_train_file sft_dataset/or_train.jsonl \
    --tulu_valid_file sft_dataset/tcy_val.jsonl \
    --english_valid_file en_sft_dataset/valid.jsonl \
    --output_dir $OUTPUT_DIR \