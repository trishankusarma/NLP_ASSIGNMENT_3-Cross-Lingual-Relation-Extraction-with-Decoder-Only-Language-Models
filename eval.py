import sys
import json
import re
from sklearn.metrics import f1_score

def load_jsonl(path):
    """Handles both single-line JSONL and pretty-printed JSON objects."""
    data = []
    with open(path, encoding='utf-8') as f:
        content = f.read().strip()
    
    decoder = json.JSONDecoder()
    pos = 0
    content = content.lstrip()
    
    while pos < len(content):
        while pos < len(content) and content[pos] in ' \t\n\r':
            pos += 1
        if pos >= len(content):
            break
        try:
            obj, end_pos = decoder.raw_decode(content, pos)
            data.append(obj)
            pos = end_pos
        except json.JSONDecodeError as e:
            print(f"Parse error at pos {pos}: {e}")
            break
    
    return data

def normalize_entities(data):
    if "entities" in data:
        return data["entities"]
    elif "relationMentions" in data:
        return [
            {
                "em1": r.get("em1Text", ""), 
                "em2": r.get("em2Text", ""), 
                "relation": r.get("label", "")
            } 
            for r in data["relationMentions"]
        ]
    return []

def evaluate_files(pred_file, ref_file):
    all_true = []
    all_pred = []

    try:
        pred_list = load_jsonl(pred_file)
        ref_list  = load_jsonl(ref_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    for line_idx, (pred_data, ref_data) in enumerate(zip(pred_list, ref_list), 1):
        true_rels      = normalize_entities(ref_data)
        predicted_rels = normalize_entities(pred_data)

        pred_dict = {
            (p.get('em1', ''), p.get('em2', '')): p.get('relation', '') 
            for p in predicted_rels
        }

        for t in true_rels:
            em1      = t.get('em1', '')
            em2      = t.get('em2', '')
            true_rel = t.get('relation', '')
            
            all_true.append(true_rel)
            all_pred.append(pred_dict.get((em1, em2), "None"))

    labels = list(set(all_true + all_pred))
    if "None" in labels:
        labels.remove("None")

    if not labels:
        print("No valid relations found to evaluate.")
        sys.exit(0)

    macro_f1 = f1_score(all_true, all_pred, labels=labels, average='macro', zero_division=0)
    micro_f1 = f1_score(all_true, all_pred, labels=labels, average='micro', zero_division=0)

    print(f"Evaluation Results:")
    print(f"-------------------")
    print(f"Total True Relations : {len(all_true)}")
    print(f"Macro F1             : {macro_f1:.4f}")
    print(f"Micro F1             : {micro_f1:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval.py <predictions_jsonl> <reference_jsonl>")
        sys.exit(1)

    evaluate_files(sys.argv[1], sys.argv[2])