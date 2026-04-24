# This script should run the training and save checkpoints in the output/ dir
#!/bin/bash
# Q1/train.sh
OUTPUT_DIR=${1:-"Q1/output"}
echo "Training Task 1 :: output: $OUTPUT_DIR"

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/.."

python train.py \
    --english_train_file ../en_sft_dataset/train.jsonl \
    --hindi_train_file ../sft_dataset/hi_train.jsonl \
    --kanada_train_file ../sft_dataset/kn_train.jsonl \
    --english_valid_file ../en_sft_dataset/valid.jsonl \
    --output_dir $OUTPUT_DIR