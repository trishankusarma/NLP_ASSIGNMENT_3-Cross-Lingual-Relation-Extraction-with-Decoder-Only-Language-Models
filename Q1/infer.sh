#!/bin/bash
# Q1/infer.sh
# Usage: ./infer.sh <lang> <test_file_path> <output_dir>
LANG=$1
TEST_FILE=$2
OUTPUT_DIR_PRED=${3:-"Q1/pred"}
OUTPUT_DIR_MODEL="Q1/output"
echo "Task 1 Inference | lang=$LANG | test=$TEST_FILE | out=$OUTPUT_DIR_PRED"

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/.."

python infer.py \
    --lang $LANG \
    --test_file $TEST_FILE \
    --output_dir_pred $OUTPUT_DIR_PRED \
    --output_dir_model $OUTPUT_DIR_MODEL