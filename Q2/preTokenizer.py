# utils/pretokenize.py
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.utils import load_jsonl, load_lang_map

# English has 94222 samples → tokenizing all at once might OOM on CPU RAM
# Batch the tokenization in chunks
def pretokenize_and_save(pairs, tokenizer, max_length, save_path, chunk_size=10000):
    if os.path.exists(save_path):
        print(f"Already exists: {save_path} — skipping")
        return

    print(f"Pre-tokenizing {len(pairs)} pairs → {save_path}")
    
    prompts = [p["prompt"] for p in pairs]
    targets = [p["target"] for p in pairs]

    all_input_ids_list      = []
    all_attention_mask_list = []
    all_labels_list         = []

    for start in tqdm(range(0, len(pairs), chunk_size), desc="Tokenizing chunks"):
        end = min(start + chunk_size, len(pairs))
        
        chunk_prompts = prompts[start:end]
        chunk_targets = targets[start:end]

        full_encs = tokenizer(
            [p + t for p, t in zip(chunk_prompts, chunk_targets)],
            max_length=max_length, truncation=True,
            padding='max_length', return_tensors='pt'
        )
        prompt_encs = tokenizer(
            chunk_prompts, truncation=False,
            padding=False, return_tensors=None
        )

        chunk_labels = full_encs["input_ids"].clone()
        for i in range(len(chunk_prompts)):
            prompt_len = min(len(prompt_encs["input_ids"][i]), max_length)
            chunk_labels[i, :prompt_len] = -100
            chunk_labels[i, full_encs["attention_mask"][i] == 0] = -100

        all_input_ids_list.append(full_encs["input_ids"])
        all_attention_mask_list.append(full_encs["attention_mask"])
        all_labels_list.append(chunk_labels)

    all_input_ids      = torch.cat(all_input_ids_list,      dim=0)
    all_attention_mask = torch.cat(all_attention_mask_list, dim=0)
    all_labels         = torch.cat(all_labels_list,         dim=0)

    valid_mask = (all_labels != -100).any(dim=1)
    n_filtered = (~valid_mask).sum().item()
    if n_filtered > 0:
        print(f"  Filtered {n_filtered} samples with no learning signal")

    torch.save({
        "input_ids"      : all_input_ids[valid_mask],
        "attention_mask" : all_attention_mask[valid_mask],
        "labels"         : all_labels[valid_mask],
    }, save_path)
    print(f"  Saved {valid_mask.sum().item()} samples → {save_path}")