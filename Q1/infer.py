import argparse
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

from hyper_parameters.config import PartAConfig
from .dataset_wrapper import DatasetWrapper
from utils.utils import load_jsonl, load_lang_map
from utils.logger_class import logging
from .model_class import ModelClass

LABEL_2_INDEX_PATH = "./label_mapping/label2index.json"
INDEX_2_LABEL_PATH = "./label_mapping/index2label.json"

config = PartAConfig()

def load_model(output_dir, lang, device):
    # Step 1: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(output_dir, "tokenizer")
    )
    vocab_size = len(tokenizer)
    # Step 2 : Loading train_config for num_labels and max_length 
    with open(os.path.join(output_dir, "train_config.json")) as f:
        train_config = json.load(f)
    
    num_labels, max_length = train_config["num_labels"], train_config["lang_max_lengths"][lang]
    # Step 3: Load base model
    model = ModelClass(
        hyper_parameters=config,
        num_labels=num_labels,
        vocab_size=len(tokenizer),
        class_weights=torch.ones(num_labels), # not used during inference
        apply_lora=False
    )
    # Step 4: Load LoRA adapter
    model.base_model = PeftModel.from_pretrained(
        model.base_model,
        os.path.join(output_dir, "lora_adapter")
    )
    # Step 5: Load classifier head
    model.classifier.load_state_dict(
        torch.load(os.path.join(output_dir, "classifier_head.pt"), map_location=device, weights_only=True)
    )

    # Step 6: Move to device + eval mode
    model.to(device)
    model.eval()

    # Step 7: load index2label json also
    with open(INDEX_2_LABEL_PATH, 'r', encoding='utf-8') as f:
        index2label = json.load(f)

    return {
        "tokenizer" : tokenizer,
        "vocab_size" : vocab_size,
        "model" : model,
        "num_labels" : num_labels,
        "max_length" : max_length,
        "index2label" : index2label
    }

def flatten_test_data(test_data):
    pairs = []
    for sample in test_data:
        for rel in sample["relationMentions"]:

            pairs.append({
                "articleId": sample["articleId"],
                "sentId" : sample["sentId"],
                'sentText': sample["sentText"],
                'em1Text': rel["em1Text"],
                'em2Text': rel["em2Text"]
            })

    return pairs

def run_inference(model, data_loader, index2label, lang_map, device):
    # return list of predicted label strings
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            entity_map1 = batch["entity_map1"].to(device)
            entity_map2 = batch["entity_map2"].to(device)

            logits, _ = model(input_ids, attention_mask, entity_map1, entity_map2)
            preds = logits.argmax(dim=-1).cpu().tolist()

            for p in preds:
                english_label = index2label[str(p)]
                # translate if lang_map exists
                if not lang_map:
                    label = english_label
                elif lang_map and english_label in lang_map:
                    label = lang_map[english_label]
                else:
                    print(f"Warning :: {english_label} doesn't exist in the indic literature")
                    label = english_label
                all_preds.append(label)

    return all_preds

def reconstruct_output(test_data, predictions_map):
    outputs = []
    for sample in test_data:    
        rel_mentions = []
        for rel in sample["relationMentions"]:

            pred_label = predictions_map[(
                sample["articleId"],
                sample["sentId"],
                rel["em1Text"],
                rel["em2Text"]
            )]

            rel_mentions.append({
                "em1Text" : rel["em1Text"],
                "em2Text" : rel["em2Text"],
                "label"   : pred_label
            })

        outputs.append({
            "articleId": sample["articleId"],
            "sentId" : sample["sentId"],
            "sentText": sample["sentText"],
            "relationMentions" : rel_mentions
        })
    return outputs

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    model_configurations = load_model(args.output_dir, args.lang, device)
    special_tokens = ["[EM1]", "[/EM1]", "[EM2]", "[/EM2]"]

    test_data = load_jsonl(args.test_file)
    lang_map = load_lang_map(args.lang)
    test_pairs = flatten_test_data(test_data)

    test_dataset = DatasetWrapper(
        test_pairs, 
        model_configurations["tokenizer"], 
        special_tokens, 
        max_length = model_configurations["max_length"]
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    predictions = run_inference(model_configurations["model"], test_loader, model_configurations["index2label"], lang_map, device)

    # Build unique maps for each pair
    all_pred_map = {}
    for index, pair in enumerate(test_pairs):
        all_pred_map[(pair["articleId"], pair["sentId"], pair["em1Text"], pair["em2Text"])] = predictions[index]

    # use that to predict
    outputs = reconstruct_output(test_data, all_pred_map)

    out_path = os.path.join(args.output_dir, f"Q1_{args.lang}.jsonl")
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    logging(s="Q1.infer")
    print(f"Inference on {args.lang}")
    main(args)