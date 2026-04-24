#!/bin/bash
# Q2/infer.sh
LANG=$1
TEST_FILE=$2
OUTPUT_DIR_FOR_PRED=${3:-"Q2/pred"}
OUTPUT_DIR_FOR_MODEL="Q2/output"
echo "Task 2 Inference | lang=$LANG | test=$TEST_FILE | out=$OUTPUT_DIR_FOR_PRED"

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/.."

python infer.py \
    --lang $LANG \
    --test_file $TEST_FILE \
    --output_dir_pred $OUTPUT_DIR_FOR_PRED \
    --output_dir_model $OUTPUT_DIR_FOR_MODEL