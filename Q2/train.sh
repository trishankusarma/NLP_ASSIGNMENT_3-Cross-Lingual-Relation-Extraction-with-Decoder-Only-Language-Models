#!/bin/bash
# Q2/train.sh
OUTPUT_DIR=${1:-"Q2/output"}
echo "Training Task 2 → output: $OUTPUT_DIR"

WIKI_DIR="wikipedia_dumps"
STAGE1_PATH="$OUTPUT_DIR/lora_adapter_stage_1"
CPT_FLAG=""

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/.."

if [ -d "$STAGE1_PATH" ]; then
    echo "Stage 1 adapter found — skipping CPT, loading directly"
    CPT_FLAG=""
elif [ -d "$WIKI_DIR" ] && [ "$(ls -A $WIKI_DIR)" ]; then
    echo "Wikipedia dumps found — running CPT"
    CPT_FLAG="--run_cpt --wiki_dir $WIKI_DIR"
else
    echo "Wikipedia dumps not found — downloading first..."
    python ../unsupervised_corpus/unsupervised_corpus.py

    if [ -d "$WIKI_DIR" ] && [ "$(ls -A $WIKI_DIR)" ]; then
        echo "Download complete — running CPT"
        CPT_FLAG="--run_cpt --wiki_dir $WIKI_DIR"
    else
        echo "Download failed — skipping CPT"
        CPT_FLAG=""
    fi
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
    --english_train_file ../en_sft_dataset/train.jsonl \
    --hindi_train_file   ../sft_dataset/hi_train.jsonl \
    --kanada_train_file  ../sft_dataset/kn_train.jsonl \
    --oria_train_file    ../sft_dataset/or_train.jsonl \
    --tulu_valid_file    ../sft_dataset/tcy_val.jsonl \
    --english_valid_file ../en_sft_dataset/valid.jsonl \
    --output_dir         $OUTPUT_DIR \
    $CPT_FLAG