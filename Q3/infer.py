import os
import json
import re
import argparse
import numpy as np
import random
from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils.logger_class import logging
from utils.utils import load_jsonl, load_lang_map
from .builder import build_icl_prompt, parse_label, reconstruct_output
from .faiss_retriever import FAISSRetriever

MODEL_PATH       = "/home/scai/msr/aiy247541/scratch/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
LABEL2INDEX_PATH = "./label_mapping/label2index.json"

ENGLISH_TRAIN_PATH = "en_sft_dataset/train.jsonl"
ENGLIGH_VALID_PATH = "en_sft_dataset/valid.jsonl"
HINDI_TRAIN_PATH = 'sft_dataset/hi_train.jsonl'
KANADA_TRAIN_PATH = 'sft_dataset/kn_train.jsonl'
ORIA_TRAIN_PATH = 'sft_dataset/or_train.jsonl'
TCY_VALID_PATH = 'sft_dataset/tcy_val.jsonl'

def load_valid_labels(label2index_path=LABEL2INDEX_PATH):
    with open(label2index_path, encoding='utf-8') as f:
        label2index = json.load(f)
    return set(k for k in label2index if k.startswith('/'))

def build_example_pool(lang):
    pool = []

    en_data = load_jsonl(ENGLISH_TRAIN_PATH)
    for sample in en_data:
        for rel in sample['relationMentions']:
            pool.append({
                'sentText' : sample['sentText'],
                'em1Text'  : rel['em1Text'],
                'em2Text'  : rel['em2Text'],
                'label'    : rel['label'],
                'lang'     : 'en',
            })
    print(f"[Pool] English (full): {len(pool)}")

    # ── Sample English down to avoid slow FAISS encoding ─────
    # 94k → 5000 still gives ~200 examples per label
    if len(pool) > 5000:
        pool = random.sample(pool, 5000)
        print(f"[Pool] English (sampled): {len(pool)}")

    lang_files = {
        'hi'  : (HINDI_TRAIN_PATH,  True),
        'kn'  : (KANADA_TRAIN_PATH,  True),
        'or'  : (ORIA_TRAIN_PATH,  False),
        'tcy' : (TCY_VALID_PATH,   False),
    }
    if lang in lang_files:
        fpath, needs_map = lang_files[lang]
        indic_data  = load_jsonl(fpath)
        reverse_map = {}
        if needs_map:
            lmap        = load_lang_map(lang)
            reverse_map = {v: k for k, v in lmap.items()}
            if lang == 'kn':
                # just one mismatch I found
                reverse_map["/ಸ್ಥಳ/ದೇಶ/ರಾಜಧಾನ"] = reverse_map.get(
                    "/ಸ್ಥಳ/ದೇಶ/ರಾಜಧಾನಿ", "/location/country/capital"
                )
        for sample in indic_data:
            for rel in sample['relationMentions']:
                label = reverse_map.get(rel['label'], rel['label'])
                pool.append({
                    'sentText' : sample['sentText'],
                    'em1Text'  : rel['em1Text'],
                    'em2Text'  : rel['em2Text'],
                    'label'    : label,
                    'lang'     : lang,
                })
        print(f"[Pool] {lang}: added {len(indic_data)}")

    print(f"[Pool] Total: {len(pool)}")
    return pool

def main(args):
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    SHOTS_PER_LANG = {'hi': 5, 'kn': 3, 'or': 2, 'tcy': 4}
    num_shots = SHOTS_PER_LANG.get(args.lang, 5)

    # Step 1: Load model
    print(f"Loading model from {MODEL_PATH}...")
    llm = LLM(
        model                  = MODEL_PATH,
        dtype                  = "float16",
        max_model_len          = 4096,
        gpu_memory_utilization = 0.85,
    )
    sampling_params = SamplingParams(
        temperature = 0.0,
        max_tokens  = 50,
        stop        = ["\n\n", "Example"],
    )
    print("Model loaded!")

    # Step 2: Labels + lang map
    valid_labels = load_valid_labels()
    lang_map     = load_lang_map(args.lang) if args.lang in ['hi', 'kn'] else None

    # Step 3: Pool + FAISS
    pool      = build_example_pool(args.lang)
    retriever = FAISSRetriever(pool)

    # Step 4: Test data
    test_data = load_jsonl(args.test_file)
    print(f"Loaded {len(test_data)} test sentences")

    # Step 5: Build all prompts
    all_prompts, all_meta = [], []
    print("Building prompts with FAISS retrieval...")
    for sent_idx, sample in enumerate(tqdm(test_data)):
        for pair_idx, rel in enumerate(sample['relationMentions']):
            query    = f"{sample['sentText']} {rel['em1Text']} {rel['em2Text']}"
            examples = retriever.retrieve(query, k=num_shots)
            prompt   = build_icl_prompt(
                sample['sentText'], rel['em1Text'], rel['em2Text'], examples
            )
            all_prompts.append(prompt)
            all_meta.append((sent_idx, pair_idx))
    print(f"Total prompts: {len(all_prompts)}")

    # Step 6: Batch generate
    print("Running batch generation...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Step 7: Parse + save
    pred_map = {}
    for i, out in enumerate(outputs):
        pred_en            = parse_label(out.outputs[0].text, valid_labels)
        pred_map[all_meta[i]] = pred_en

    result   = reconstruct_output(test_data, pred_map, lang_map)
    out_path = os.path.join(args.output_dir, f"Q3_{args.lang}.jsonl")
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",       type=str, required=True)
    parser.add_argument("--test_file",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="Q3/output")
    args = parser.parse_args()

    logging(s=f"Q3.infer{args.lang}")
    print(f"Q3 Inference | lang={args.lang}")
    main(args)