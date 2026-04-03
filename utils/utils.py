import json

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