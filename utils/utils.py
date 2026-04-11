import json
import os

def load_jsonl(dataset_path):
    """Handles both single-line JSONL (English) and pretty-printed JSON objects (Indic)."""
    data = []
    
    with open(dataset_path, encoding='utf-8') as f:
        content = f.read().strip()
    
    # decoding a stream of JSON objects one by one
    decoder = json.JSONDecoder()
    pos = 0
    content = content.lstrip()
    
    while pos < len(content):
        # Skip whitespace
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

def load_lang_map(lang):
    # English doesn't need mapping
    if lang == "en":
        return None
    
    map_paths = {
        "hi": "label_mapping/hi_map.json",
        "kn": "label_mapping/kn_map.json",
        "or": "label_mapping/or_map.json",
        "tcy": "label_mapping/tcy_map.json"
    }
    
    path = map_paths.get(lang)
    print(f"path of label mapping {path}")
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)  # {english_label: indic_label}
    
    return None

def load_label_index_mappings(index2labelPath, label2indexPath):
    with open(index2labelPath, 'r', encoding='utf-8') as f:
        index2label = json.load(f)

    with open(label2indexPath, 'r', encoding='utf-8') as f:
        label2index = json.load(f)
    
    return index2label, label2index

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