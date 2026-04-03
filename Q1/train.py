import argparse
import random
import numpy as np
import torch
import json
from collections import Counter
from transformers import AutoTokenizer, AutoModel, AutoConfig

from hyper_parameters.config import PartAConfig
from utils.utils import load_jsonl
from .dataset_wrapper import DatasetWrapper, update_sentence

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

def get_class_weight(pairs, label2index):
    label_counter = Counter()

    for pair in pairs:
        label_counter[pair["label"]] += 1

    total_pairs = len(pairs)
    num_labels = len(label2index)

    class_weights = torch.zeros(num_labels)
    
    for key, count in label_counter.items():
        class_weights[label2index[key]] = total_pairs / (num_labels * count) # so that freq ones get less weight and rare ones get more weight
    
    return class_weights

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
    print(f"Number of index2label-keys : {len(index2label)} and label2index-keys : {len(label2index)}")

    # Step 3: Flatten to pairs {'sentText' , 'em1Text', 'em2Text', 'label', 'label_id'} # for each label
    train_pairs = flatten_data(train_data, label2index)
    valid_pairs = flatten_data(valid_data, label2index)
    print(f"Total number of train-pairs {len(train_pairs)} and valid-pairs {len(valid_pairs)}")

    # Step 4: Get class weight for each labeled imbalanced data
    class_weights = get_class_weight(train_pairs, label2index).to(device)

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
                              shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size*2,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Step 7: Build model using LORA
    # Step 8: Initialize the optimizer
    # Step 9: Train the model
    # Step 9.1 : Evaluate model after every epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    main(args) 
