#!/bin/bash
# Q1/infer.sh
# Usage: ./infer.sh <lang> <test_file_path> <output_dir>
LANG=$1
TEST_FILE=$2
OUTPUT_DIR=${3:-"Q1/output"}
echo "Task 1 Inference | lang=$LANG | test=$TEST_FILE | out=$OUTPUT_DIR"
python Q1/infer.py \
    --lang $LANG \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \