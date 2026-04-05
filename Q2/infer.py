import os
import json
import re
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from hyper_parameters.config import PartBConfig
from .dataset_wrapper import DatasetWrapper
from utils.utils import load_jsonl, load_lang_map
from utils.logger_class import logging
from .model_class import ModelClass
from .dataset_wrapper import build_prompt

config = PartBConfig()

def load_model(output_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(output_dir, "tokenizer")
    )
    tokenizer.padding_side = 'left'  # left pad for generation

    with open(os.path.join(output_dir, "train_config.json")) as f:
        train_config = json.load(f)
    max_length = train_config["max_length"]

    model = ModelClass(hyper_parameters=config, apply_lora=False)
    model.base_model = PeftModel.from_pretrained(
        model.base_model,
        os.path.join(output_dir, "lora_adapter")
    )
    model.to(device)
    model.eval()

    return {"tokenizer": tokenizer, "model": model, "max_length": max_length}

def flatten_test_data(data):
    """Returns flat list of prompts, keeping track of (sent_idx, pair_idx)."""
    samples = []
    for sent_idx, sample in enumerate(data):
        for pair_idx, rel in enumerate(sample['relationMentions']):
            e1 = rel['em1Text']
            e2 = rel['em2Text']
            samples.append({
                'prompt'    : build_prompt(sample['sentText'], e1, e2),
                'sent_idx'  : sent_idx,
                'pair_idx'  : pair_idx,
            })
    return samples

# Load valid English labels once for post-processing
def load_valid_labels(label2index_path="./label_mapping/label2index.json"):
    with open(label2index_path, encoding='utf-8') as f:
        label2index = json.load(f)
    # Only English labels (they don't contain non-ASCII scripts)
    return set(k for k in label2index if k.startswith('/'))

def parse_label(generated_text: str, valid_labels: set, fallback="NA") -> str:
    """
    Extract a valid relation label from raw model output.
    Tries multiple strategies in order of reliability.
    """
    text = generated_text.strip()

    # Strategy 1: parse JSON {"label": "..."}
    try:
        match = re.search(r'\{[^}]+\}', text)
        if match:
            obj = json.loads(match.group())
            label = obj.get('label', '').strip()
            if label in valid_labels:
                return label
    except (json.JSONDecodeError, KeyError):
        pass

    # Strategy 2: direct label match (longest first to avoid partial matches)
    for label in sorted(valid_labels, key=len, reverse=True):
        if label in text:
            return label

    # Strategy 3: match /word/word/word path pattern
    match = re.search(r'/[\w/]+', text)
    if match:
        candidate = match.group()
        for label in valid_labels:
            if candidate in label:
                return label

    return fallback

def run_inference(model, tokenizer, test_loader, flat_samples,
                  valid_labels, lang_map, device):
    """
    Runs generation batch by batch.
    Returns dict: {(sent_idx, pair_idx): predicted_label_in_target_lang}
    """
    pred_map = {}
    flat_idx = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated = model.base_model.generate(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = 30,      # label is short
                do_sample      = False,   # greedy = deterministic
                pad_token_id   = tokenizer.pad_token_id,
                eos_token_id   = tokenizer.eos_token_id,
            )

            # Decode only the newly generated tokens (after the prompt)
            prompt_len = input_ids.shape[1]
            for i in range(generated.shape[0]):
                new_tokens   = generated[i][prompt_len:]
                decoded      = tokenizer.decode(new_tokens, skip_special_tokens=True)

                # Parse English label
                pred_en      = parse_label(decoded, valid_labels)

                # Map to target language if lang_map provided
                if lang_map and pred_en in lang_map:
                    pred_label = lang_map[pred_en]
                else:
                    pred_label = pred_en

                meta = flat_samples[flat_idx]
                pred_map[(meta['sent_idx'], meta['pair_idx'])] = pred_label
                flat_idx += 1

    return pred_map

def reconstruct_output(test_data, pred_map):
    """Rebuild output JSONL matching input structure with predicted labels."""
    output = []
    for sent_idx, sample in enumerate(test_data):
        out_sample = {
            'articleId'        : sample.get('articleId', ''),
            'sentId'           : sample.get('sentId', ''),
            'sentText'         : sample['sentText'],
            'relationMentions' : [],
        }
        for pair_idx, rel in enumerate(sample['relationMentions']):
            out_sample['relationMentions'].append({
                'em1Text' : rel['em1Text'],
                'em2Text' : rel['em2Text'],
                'label'   : pred_map.get((sent_idx, pair_idx), 'NA'),
            })
        output.append(out_sample)
    return output

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # Load model + tokenizer
    cfg = load_model(args.output_dir, device)
    model     = cfg["model"]
    tokenizer = cfg["tokenizer"]
    max_length = cfg["max_length"]

    # Load valid labels + lang map
    valid_labels = load_valid_labels()
    lang_map = load_lang_map(args.lang) if args.lang != 'en' else None

    # Load + flatten test data
    test_data    = load_jsonl(args.test_file)
    flat_samples = flatten_test_data(test_data)
    print(f"Loaded {len(test_data)} sentences → {len(flat_samples)} pairs")

    # Dataset + loader
    # For inference, DatasetWrapper should NOT add labels field
    # Pass prompt only (no target)
    test_dataset = DatasetWrapper(
        flat_samples, tokenizer, max_length=max_length, inference=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # Run generation
    pred_map = run_inference(
        model, tokenizer, test_loader,
        flat_samples, valid_labels, lang_map, device
    )

    # Reconstruct output
    output = reconstruct_output(test_data, pred_map)

    # Save
    out_path = os.path.join(args.output_dir, f"output_{args.lang}.jsonl")
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(output)} predictions → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",       type=str, required=True)
    parser.add_argument("--test_file",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    logging(s="Q2.infer")
    print(f"Inference on lang={args.lang}")
    main(args)