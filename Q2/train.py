import argparse
import random
import numpy as np
import torch
import os
from transformers import AutoTokenizer
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

from utils.logger_class import logging
from hyper_parameters.config import PartBConfig
from utils.utils import load_jsonl, load_lang_map
from .dataset_wrapper import DatasetWrapper, build_prompt, build_target
from .model_class import ModelClass
from .evaluate import evaluate_loss, run_all_f1

config = PartBConfig()

# Per-language max lengths (from token length analysis)
LANG_MAX_LENGTHS = {
    'en'  : 142,    # 99th percentile
    'hi'  : 448,    # 99th percentile
    'kn'  : 714,    # 99th percentile
    'or'  : 1026,   # 99th percentile
    'tcy' : 556,    # 99th percentile (rounded to even)
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def flatten_pairs(data):
    samples = []
    for sample in data:
        sent = sample['sentText']
        for rel in sample['relationMentions']:
            e1 = rel['em1Text']
            e2 = rel['em2Text']
            label = rel.get('label', 'NA')
 
            # If indic data, map label back to English for training consistency
            prompt = build_prompt(sent, e1, e2)
            target = build_target(label)
 
            samples.append({'prompt': prompt, 'target': target})
    return samples

def find_max_length(pairs, tokenizer):
    lengths = []
    
    for pair in pairs:
        
        tokens = tokenizer(pair["prompt"] + pair["target"], truncation=False)
        lengths.append(len(tokens["input_ids"]))
    
    lengths = sorted(lengths)
    print(f"95th percentile: {lengths[int(0.95 * len(lengths))]}")
    print(f"99th percentile: {lengths[int(0.99 * len(lengths))]}")
    print(f"99.5th percentile: {lengths[int(0.995 * len(lengths))]}")
    print(f"99.9th percentile: {lengths[int(0.999 * len(lengths))]}")

    max_length = lengths[int(0.99 * len(lengths))] + lengths[int(0.99 * len(lengths))]%2
    print(f"Max_length : {max_length}")
    return max_length

def collate_fn(batch):
    # filter out samples where all labels are -100
    batch = [b for b in batch if (b["labels"] != -100).sum() > 0]
    if len(batch) == 0:
        return None
    
    return {
        "input_ids":      torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"] for b in batch])
    }

def update_label_to_english(indic_data, lang_map, lang):
    # reverse map: {indic_label: english_label}
    reverse_map = {v: k for k, v in lang_map.items()}

    if lang == "kn":
        reverse_map["/ಸ್ಥಳ/ದೇಶ/ರಾಜಧಾನ"] = reverse_map["/ಸ್ಥಳ/ದೇಶ/ರಾಜಧಾನಿ"]
    
    for sample in indic_data:
        for rel in sample['relationMentions']:
            curr_label = rel["label"]
            updated_label = reverse_map.get(curr_label, None)

            if updated_label is None:
                print(f"Warning: {curr_label} not found in reverse map")
                # keep original as fallback
            else:
                rel["label"] = updated_label

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1 : Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = 'right'  # Something Important for causal LM training

    # Step 2 : Load data
    print("Loading All training data...")
    train_data = load_jsonl(args.english_train_file)
    eng_valid_data = load_jsonl(args.english_valid_file)
    print(f"Added {len(train_data)} and {len(eng_valid_data)} for training and validation from english script")

    lang = ["hi", "kn", "or", "tcy"]

    for index, indic_lang_path in enumerate([args.hindi_train_file, args.kanada_train_file, args.oria_train_file]):
        indic_data = load_jsonl(indic_lang_path)

        if index < 2: # Bcz we only need to do for hindi and english
            lang_map = load_lang_map(lang[index])
            update_label_to_english(indic_data, lang_map, lang[index])

        train_data.extend(indic_data)
        print(f"{indic_lang_path} : Added {len(indic_data)} for training")

    # for tulu just use 3/4 th for training over validation set
    tulu_valid_data = load_jsonl(args.tulu_valid_file)

    tulu_valid_data_used_length = int(np.floor(len(tulu_valid_data) * config.tulu_valid_data_used))
    train_data.extend(tulu_valid_data[:tulu_valid_data_used_length])

    print(f"{args.tulu_valid_file} Added {tulu_valid_data_used_length} data to train out of {len(tulu_valid_data)}")
    print(f"Total train samples: {len(train_data)}")

    # Step 3 : Prepare sample pairs
    training_data_pairs = flatten_pairs(train_data)
    eng_valid_data_pairs = flatten_pairs(eng_valid_data)
    tulu_valid_data_pairs = flatten_pairs(tulu_valid_data)

    # Step 4: Find the max length
    max_length = find_max_length(training_data_pairs, tokenizer)

    # Step 5: Tokenize the dataset using the loaded tokenizer
    train_dataset = DatasetWrapper(training_data_pairs, tokenizer, max_length = max_length)
    eng_valid_dataset = DatasetWrapper(eng_valid_data_pairs, tokenizer, max_length = max_length)
    tulu_valid_dataset = DatasetWrapper(tulu_valid_data_pairs, tokenizer, max_length = max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn,
                              shuffle=True, num_workers=0, pin_memory=True)
    eng_valid_loader = DataLoader(eng_valid_dataset, batch_size=config.batch_size*2, collate_fn=collate_fn,
                              shuffle=False, num_workers=0, pin_memory=True)
    tulu_valid_loader = DataLoader(tulu_valid_dataset, batch_size=config.batch_size*2, collate_fn=collate_fn,
                              shuffle=False, num_workers=0, pin_memory=True)

    config_to_save = {
        "max_length": max_length
    }
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(config_to_save, f)

    # Step 6: Build model using LORA
    print(f"Building model with {config.model_name} and LORA fine tuning")
    model = ModelClass(
        hyper_parameters = config
    )
    model = model.to(device)

    # Step 7: Initialize the optimizer :: for only lora
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * config.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Step 8: Train the model
    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for step, batch in enumerate(pbar):
            if batch is None: 
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels)

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step+1) % 100 == 0 or (step+1) == len(train_loader):
                pbar.set_postfix({
                    "loss": f"{total_loss/(step+1):.4f}"
                })
        
        print(f"Epoch {epoch+1}/{config.epochs} :: Time taken : {time.time()-start_time} | "
                f"Loss: {total_loss/(len(train_loader)):.4f}")

        eng_val_loss = evaluate_loss(model, eng_valid_loader, device)
        tulu_val_loss = evaluate_loss(model, tulu_valid_loader, device)

        print(f"English validation loss : {eng_val_loss/len(eng_valid_loader)} and Tulu Val Loss : {tulu_val_loss/len(tulu_valid_loader)}")
        val_loss = (eng_val_loss + tulu_val_loss) / ( len(eng_valid_loader) + len(tulu_valid_loader))

        print(f"\nOverall Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.base_model.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
            print(f"Saved best model (loss={best_val_loss:.4f})")
        
        run_all_f1(
            model, tokenizer,
            eng_valid_data, tulu_valid_data, tulu_valid_data,
            load_jsonl(args.oria_train_file), tulu_valid_data,
            [], [], device, config, LANG_MAX_LENGTHS,
            tag=f"[Epoch {epoch+1}]"
        )
 
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--english_train_file", type=str, required=True)
    parser.add_argument("--hindi_train_file", type=str, required=True)
    parser.add_argument("--kanada_train_file", type=str, required=True)
    parser.add_argument("--oria_train_file", type=str, required=True)
    parser.add_argument("--tulu_valid_file", type=str, required=True)
    parser.add_argument("--english_valid_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    logging(s = "Q2")
    main(args)