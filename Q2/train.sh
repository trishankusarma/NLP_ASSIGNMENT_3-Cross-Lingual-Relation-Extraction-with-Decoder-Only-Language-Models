# This script should run the training and save checkpoints in the output/ dir
#!/bin/bash
# Q2/train.sh
OUTPUT_DIR=${1:-"Q2/output"}
echo "Training Task 2 → output: $OUTPUT_DIR"

# Check if Wikipedia dumps already exist
WIKI_DIR="wikipedia_dumps"
CPT_FLAG=""

if [ -d "$WIKI_DIR" ] && [ "$(ls -A $WIKI_DIR)" ]; then
    echo "Wikipedia dumps found at $WIKI_DIR — running with CPT"
    CPT_FLAG="--run_cpt --wiki_dir $WIKI_DIR"
else
    echo "Wikipedia dumps not found — downloading first..."
    python unsupervised_corpus/unsupervised_corpus.py
    
    if [ -d "$WIKI_DIR" ] && [ "$(ls -A $WIKI_DIR)" ]; then
        echo "Download complete — running with CPT"
        CPT_FLAG="--run_cpt --wiki_dir $WIKI_DIR"
    else
        echo "Download failed — running without CPT"
        CPT_FLAG=""
    fi
fi

python -m Q2.train \
    --english_train_file en_sft_dataset/train.jsonl \
    --hindi_train_file sft_dataset/hi_train.jsonl \
    --kanada_train_file sft_dataset/kn_train.jsonl \
    --oria_train_file sft_dataset/or_train.jsonl \
    --tulu_valid_file sft_dataset/tcy_val.jsonl \
    --english_valid_file en_sft_dataset/valid.jsonl \
    --output_dir $OUTPUT_DIR \
    $CPT_FLAG