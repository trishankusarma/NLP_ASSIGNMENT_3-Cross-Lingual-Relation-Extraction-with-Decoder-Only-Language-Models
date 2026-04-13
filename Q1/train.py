import argparse
import random
import numpy as np
import torch
import json
from collections import Counter
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
import os
import time
from tqdm import tqdm
import sys
from datetime import datetime

from hyper_parameters.config import PartAConfig
from utils.utils import load_jsonl
# from utils.plot_utils import plot_metrics
from utils.logger_class import logging
from .dataset_wrapper import DatasetWrapper, update_sentence
from .model_class import ModelClass

config = PartAConfig()

LABEL_2_INDEX_PATH = "./label_mapping/label2index.json"
INDEX_2_LABEL_PATH = "./label_mapping/index2label.json"

Q1_LANG_MAX_LENGTHS = {
    'en'  : 142,
    'hi'  : 400,
    'kn'  : 512
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_label_index_mappings(index2labelPath, label2indexPath):
    with open(index2labelPath, 'r', encoding='utf-8') as f:
        index2label = json.load(f)

    with open(label2indexPath, 'r', encoding='utf-8') as f:
        label2index = json.load(f)
    
    return index2label, label2index

def collate_fn(batch):
    """Simple stack — safe because single-language batches have same shape."""
    return {
        "input_ids"      : torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask" : torch.stack([b["attention_mask"]  for b in batch]),
        "entity_map1"    : torch.stack([b["entity_map1"]     for b in batch]),
        "entity_map2"    : torch.stack([b["entity_map2"]     for b in batch]),
        "label"          : torch.stack([b["label"]           for b in batch]),
    }

def make_loader(pairs, tokenizer, special_tokens, max_length, batch_size, shuffle=True):
    dataset = DatasetWrapper(pairs, tokenizer, special_tokens, max_length=max_length)
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=0,
        pin_memory=True, collate_fn=collate_fn,
    )

def flatten_data(train_data, label2index):
    pairs = []
    for sample in train_data:
        for rel in sample["relationMentions"]:
            label = rel["label"]
            label_idx = label2index.get(label, -1)

            # Safe check
            if label_idx == -1:
                print(f"Warning {label} not found in label2idx")
            
            pairs.append({
                'sentText': sample["sentText"], 
                'em1Text': rel["em1Text"], 
                'em2Text': rel["em2Text"], 
                'label': label, 
                'label_id' : label_idx
            })

    return pairs

def round_robin_epoch(loaders):
    """Yield batches in round-robin across loaders until all exhausted."""
    iterators = [iter(l) for l in loaders]
    active    = list(range(len(loaders)))
    while active:
        next_active = []
        for i in active:
            try:
                yield next(iterators[i])
                next_active.append(i)
            except StopIteration:
                pass
        active = next_active

def get_class_weight(pairs, label2index, num_labels):
    label_counter = Counter()

    for pair in pairs:
        label_counter[pair["label"]] += 1

    total_pairs = len(pairs)
    class_weights = torch.zeros(num_labels)
    
    for key, count in label_counter.items():
        class_weights[label2index[key]] = total_pairs / (num_labels * count) # so that freq ones get less weight and rare ones get more weight
    
    return class_weights

def evaluate_task(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            entity_map1 = batch["entity_map1"].to(device)
            entity_map2 = batch["entity_map2"].to(device)
            label = batch["label"].to(device)

            logits, loss = model(input_ids, attention_mask, entity_map1, entity_map2, label)
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(label.cpu().numpy().tolist())

    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Avg validation loss : {total_loss/len(data_loader)} :: Val_F1_Micro : {f1_micro} :: Val_F1_Macro : {f1_macro}")  
    return f1_micro, f1_macro

def main(args):
    # Setting up seed=42 for reproducibility
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # Step 1: lets load the data for both train and validation set
    train_english_data = load_jsonl(args.english_train_file)
    train_hindi_data = load_jsonl(args.hindi_train_file)
    train_kanada_data = load_jsonl(args.kanada_train_file)
    valid_english_data = load_jsonl(args.english_valid_file)
    
    # Step 2: Load the index2label and label2index jsons
    index2label, label2index = load_label_index_mappings(INDEX_2_LABEL_PATH, LABEL_2_INDEX_PATH)
    num_labels = len(index2label)

    # Note there was a typo in the kanada dataset :: to fix
    label2index["/ಸ್ಥಳ/ದೇಶ/ರಾಜಧಾನ"] = label2index["/ಸ್ಥಳ/ದೇಶ/ರಾಜಧಾನಿ"]
    print(f"Number of index2label-keys : {len(index2label)} and label2index-keys : {len(label2index)}")

    # Step 3: Flatten to pairs {'sentText' , 'em1Text', 'em2Text', 'label', 'label_id'} # for each label
    en_pairs = flatten_data(train_english_data, label2index)
    hi_pairs = flatten_data(train_hindi_data, label2index)
    kn_pairs = flatten_data(train_kanada_data, label2index)

    valid_pairs = flatten_data(valid_english_data, label2index)
    print(f"en:{len(en_pairs)} hi:{len(hi_pairs)} kn:{len(kn_pairs)} valid:{len(valid_pairs)}")

    # Step 4: Get class weight for each labeled imbalanced data
    all_train_pairs = en_pairs + hi_pairs + kn_pairs
    class_weights = get_class_weight(all_train_pairs, label2index, num_labels).to(device)

    # Step 5: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    special_tokens = ["[EM1]", "[/EM1]", "[EM2]", "[/EM2]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Step 6: Per-language loaders — each has fixed max_length
    en_loader = make_loader(en_pairs, tokenizer, special_tokens, Q1_LANG_MAX_LENGTHS['en'], config.batch_size)      # 16
    hi_loader = make_loader(hi_pairs, tokenizer, special_tokens, Q1_LANG_MAX_LENGTHS['hi'], config.batch_size // 2) # 8
    kn_loader = make_loader(kn_pairs, tokenizer, special_tokens, Q1_LANG_MAX_LENGTHS['kn'], config.batch_size // 4) # 4

    train_loaders   = [en_loader, hi_loader, kn_loader]
    steps_per_epoch = sum(len(l) for l in train_loaders)
    print(f"Steps per epoch: {steps_per_epoch:,}")

    # Validation — English only (as per assignment: Q1 evaluated on en/hi/kn)
    en_valid_loader = make_loader(valid_pairs, tokenizer, special_tokens,
                               Q1_LANG_MAX_LENGTHS['en'], config.batch_size * 2, shuffle=False)
    # hi_valid_loader = make_loader(hi_pairs, tokenizer, special_tokens,
    #                            Q1_LANG_MAX_LENGTHS['hi'], config.batch_size, shuffle=False)
    # kn_valid_loader = make_loader(kn_pairs, tokenizer, special_tokens,
    #                            Q1_LANG_MAX_LENGTHS['kn'], config.batch_size, shuffle=False)

    # valid_loaders = [en_valid_loader, hi_valid_loader, kn_valid_loader]
    valid_loaders = [en_valid_loader]

    os.makedirs(args.output_dir, exist_ok = True)
    # might be needing this during inference
    # Save config
    config_to_save = {
        "lang_max_lengths" : Q1_LANG_MAX_LENGTHS,
        "num_labels"       : len(index2label)
    }

    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(config_to_save, f)

    # Step 7: Build model using LORA
    print(f"Building model with {config.model_name} and LORA fine tuning")
    model = ModelClass(
        hyper_parameters = config,
        num_labels = num_labels,
        vocab_size = len(tokenizer),
        class_weights = class_weights
    )
    model = model.to(device)
    # Step 8: Initialize the optimizer :: for only lora + classifier
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    total_steps = steps_per_epoch * config.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val_f1 = 0
    
    train_losses = []
    train_accs = []
    val_f1_score_micro = []
    val_f1_score_macro = []

    lang_codes = ['en', 'hi', 'kn']

    # Step 9: Train the model
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        correct_points = 0
        step_count = 0
        total_points = 0
        loss_window = [] # will be maintaining a loss window of size 500
        start_time = time.time()

        pbar = tqdm(
            round_robin_epoch(train_loaders),
            total=steps_per_epoch,
            desc=f"Epoch {epoch+1} Training"
        )
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            entity_map1 = batch["entity_map1"].to(device)
            entity_map2 = batch["entity_map2"].to(device)
            label = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits, loss = model(input_ids, attention_mask, entity_map1, entity_map2, label)
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct_points += (preds == label).sum().item()
            total_points += label.size(0)

            step_count += 1

            loss_window.append(loss.item())
            if len(loss_window) > 500:
                loss_window.pop(0)
        
            if step_count % 100 == 0 or step_count == steps_per_epoch:
                recent_loss = sum(loss_window) / len(loss_window)
                train_losses.append(recent_loss)
                train_accs.append(100 * correct_points / total_points)

                pbar.set_postfix({
                    "loss": f"{recent_loss:.4f}",
                    "acc": f"{100*correct_points/total_points:.2f}%"
                })

        print(f"Epoch {epoch+1}/{config.epochs} :: Time taken : {time.time()-start_time} | "
                   f"Loss: {total_loss/step_count:.4f} | "
                   f"Acc: {100*correct_points/total_points:.2f}%")

        # Step 9.1 : Evaluate model after every epoch
        f1_macro_consider = 0 
        for index, valid_loader in enumerate(valid_loaders):
            f1_micro, f1_macro = evaluate_task(model, valid_loader, device)
            print(f"Lang : {lang_codes[index]} f1_micro : {f1_micro} : f1_macro : {f1_macro}")

            if index == 0:
                val_f1_score_micro.append(f1_micro)
                val_f1_score_macro.append(f1_macro)
                f1_macro_consider = f1_macro

        if f1_macro_consider >= best_val_f1:

            best_val_f1 = f1_macro_consider
            # Now we need to save 3 things :: 1. Fine tuned model weights 2. Classifier weights 3. Tokenizer used
            model.base_model.save_pretrained(
                os.path.join(args.output_dir, "lora_adapter"),
                save_embedding_layers=True
            )
            torch.save(
                model.classifier.state_dict(), 
                os.path.join(args.output_dir, "classifier_head.pt")
            )
            tokenizer.save_pretrained(
                os.path.join(args.output_dir, "tokenizer")
            )
            print(f"Saved the best model {best_val_f1:.4f}")
    
    # plot_metrics(train_losses, train_accs, val_f1_score_micro, val_f1_score_macro, args.output_dir)
    print(f"Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--english_train_file", type=str, required=True)
    parser.add_argument("--hindi_train_file", type=str, required=True)
    parser.add_argument("--kanada_train_file", type=str, required=True)
    parser.add_argument("--english_valid_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    logging(s = "Q1")
    main(args)