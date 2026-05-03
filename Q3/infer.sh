#!/bin/bash
# Q3/infer.sh
LANG=$1
TEST_FILE=$2
OUTPUT_DIR=${3:-"Q3/output"}
echo "Task 3 ICL Inference | lang=$LANG | test=$TEST_FILE | out=$OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES=0

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/.."

python infer.py \
    --lang $LANG \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \