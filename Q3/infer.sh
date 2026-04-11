#!/bin/bash
# Q3/infer.sh
LANG=$1
TEST_FILE=$2
OUTPUT_DIR=${3:-"Q3/output"}
echo "Task 3 ICL Inference | lang=$LANG | test=$TEST_FILE | out=$OUTPUT_DIR"

module load compiler/gcc/11.2.0
export LD_LIBRARY_PATH=/home/scai/msr/aiy247541/.conda/envs/vllm_server_nlp/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

python -m Q3.infer \
    --lang $LANG \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \