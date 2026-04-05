#!/bin/bash
# Q2/infer.sh
LANG=$1
TEST_FILE=$2
OUTPUT_DIR=${3:-"Q2/output"}
echo "Task 2 Inference | lang=$LANG | test=$TEST_FILE | out=$OUTPUT_DIR"
python Q2/infer.py \
    --lang $LANG \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR