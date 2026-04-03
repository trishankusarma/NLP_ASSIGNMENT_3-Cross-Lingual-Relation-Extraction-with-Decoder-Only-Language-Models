# This script should run the training and save checkpoints in the output/ dir
#!/bin/bash
# Q1/train.sh
OUTPUT_DIR=${1:-"Q1/output"}
echo "Training Task 1 :: output: $OUTPUT_DIR"
python -m Q1.train \
    --train_file en_sft_dataset/train.jsonl \
    --valid_file en_sft_dataset/valid.jsonl \
    --output_dir $OUTPUT_DIR