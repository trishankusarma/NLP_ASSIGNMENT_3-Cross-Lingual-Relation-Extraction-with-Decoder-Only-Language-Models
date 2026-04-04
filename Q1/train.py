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

from hyper_parameters.config import PartAConfig
from utils.utils import load_jsonl
from .dataset_wrapper import DatasetWrapper, update_sentence
from .model_class import ModelClass

config = PartAConfig()

LABEL_2_INDEX_PATH = "./label_mapping/label2index.json"
INDEX_2_LABEL_PATH = "./label_mapping/index2label.json"

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

def next_power_of_2(x):
    return 1 << (x - 1).bit_length()

def find_max_length(pairs, tokenizer, special_tokens):
    lengths = []
    
    for pair in pairs:

        sentence = update_sentence(pair["sentText"], pair["em1Text"], pair["em2Text"], special_tokens)
        
        tokens = tokenizer(sentence, truncation=False)
        lengths.append(len(tokens["input_ids"]))
    
    lengths = sorted(lengths)
    print(f"95th percentile: {lengths[int(0.95 * len(lengths))]}")
    print(f"99th percentile: {lengths[int(0.99 * len(lengths))]}")

    return next_power_of_2(lengths[int(0.99 * len(lengths))])

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
    train_data = load_jsonl(args.train_file)
    valid_data = load_jsonl(args.valid_file)
    
    # Step 2: Load the index2label and label2index jsons
    index2label, label2index = load_label_index_mappings(INDEX_2_LABEL_PATH, LABEL_2_INDEX_PATH)
    num_labels = len(index2label)
    print(f"Number of index2label-keys : {len(index2label)} and label2index-keys : {len(label2index)}")

    # Step 3: Flatten to pairs {'sentText' , 'em1Text', 'em2Text', 'label', 'label_id'} # for each label
    train_pairs = flatten_data(train_data, label2index)
    valid_pairs = flatten_data(valid_data, label2index)
    print(f"Total number of train-pairs {len(train_pairs)} and valid-pairs {len(valid_pairs)}")

    # Step 4: Get class weight for each labeled imbalanced data
    class_weights = get_class_weight(train_pairs, label2index, num_labels).to(device)

    # Step 5: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    special_tokens = ["[EM1]", "[/EM1]", "[EM2]", "[/EM2]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Step 5.1 :: Think something on how we can handle the tokenizer length
    max_length = find_max_length(train_pairs, tokenizer, special_tokens)

    # Step 6: Tokenize the dataset using the loaded tokenizer
    train_dataset = DatasetWrapper(train_pairs, tokenizer, special_tokens, max_length = max_length)
    valid_dataset = DatasetWrapper(valid_pairs, tokenizer, special_tokens, max_length = max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size*2,
                              shuffle=False, num_workers=0, pin_memory=True)

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
    total_steps = len(train_loader) * config.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val_f1 = 0
    os.makedirs(args.output_dir, exist_ok = True)
    # might be needing this during inference
    config_to_save = {
        "max_length": max_length,
        "num_labels": len(index2label)
    }
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(config_to_save, f)

    # Step 9: Train the model
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        correct_points = 0
        total_points = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for step, batch in enumerate(pbar):
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
        
            if (step+1) % 500 == 0 or (step+1) == len(train_loader):
                pbar.set_postfix({
                    "loss": f"{total_loss/(step+1):.4f}",
                    "acc": f"{100*correct_points/total_points:.2f}%"
                })

        print(f"Epoch {epoch+1}/{config.epochs} :: Time taken : {time.time()-start_time} | "
                   f"Loss: {total_loss/(step+1):.4f} | "
                   f"Acc: {100*correct_points/total_points:.2f}%")

        # Step 9.1 : Evaluate model after every epoch
        f1_micro, f1_macro = evaluate_task(model, valid_loader, device)

        if f1_macro >= best_val_f1:

            best_val_f1 = f1_macro
            # Now we need to save 3 things :: 1. Fine tuned model weights 2. Classifier weights 3. Tokenizer used
            model.base_model.save_pretrained(
                os.path.join(args.output_dir, "lora_adapter")
            )
            torch.save(
                model.classifier.state_dict(), 
                os.path.join(args.output_dir, "classifier_head.pt")
            )
            tokenizer.save_pretrained(
                os.path.join(args.output_dir, "tokenizer")
            )
            print(f"Saved the best model {best_val_f1:.4f}")
    print(f"Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    main(args) 
