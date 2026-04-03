import json
from collections import Counter, defaultdict

ENGLISH_DATASET_TRAIN_PATH = "en_sft_dataset/train.jsonl"
ENGLISH_DATASET_VALID_PATH = "en_sft_dataset/valid.jsonl"

HINDI_DATASET_TRAIN_PATH = "sft_dataset/hi_train.jsonl"
KANADA_DATASET_TRAIN_PATH = "sft_dataset/kn_train.jsonl"
ORIA_DATASET_TRAIN_PATH = "sft_dataset/or_train.jsonl"
TCY_DATASET_VALID_PATH = "sft_dataset/tcy_val.jsonl"

datasets = [
    [ "ENGLISH_DATASET_TRAIN", ENGLISH_DATASET_TRAIN_PATH ],
    [ "ENGLISH_DATASET_VAL", ENGLISH_DATASET_VALID_PATH ],
    [ "HINDI_DATASET_TRAIN", HINDI_DATASET_TRAIN_PATH ],
    [ "KANADA_DATASET_TRAIN", KANADA_DATASET_TRAIN_PATH ],
    [ "ORIA_DATASET_TRAIN", ORIA_DATASET_TRAIN_PATH ],
    [ "TCY_DATASET_VAL", TCY_DATASET_VALID_PATH ]
]

def load_jsonl(dataset_path):
    """Handles both single-line JSONL (English) and pretty-printed JSON objects (Indic)."""
    import re
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

def explore(name, dataset_path):
    print(f"{name} : {dataset_path}")
    data = load_jsonl(dataset_path)

    print(f"Length of data : {len(data)}")

    # Get level distribution
    label_counter = Counter()
    total_pairs = sum([len(sample["relationMentions"]) for sample in data ])

    for sample in data:
        for label in sample["relationMentions"]:
            label_counter[label["label"]] += 1
    
    print(f"Number of unique relation pairs : {len(label_counter)} :: Total pairs : {total_pairs}")

    print("Top 30 labels:")
    for key, count in label_counter.most_common(30):
        print(f"  {count:5d}  {key}")
    
    na_count = label_counter.get('NA', label_counter.get('na', 0))
    print(f"\nNA count: {na_count} ({100*na_count/total_pairs:.1f}% of pairs)")

    lengths = [len(s['sentText'].split()) for s in data]
    print(f"\nSentence length (words): min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

for dataset in datasets:
    print(f"\n{'='*60}")
    explore(dataset[0], dataset[1])

# {
#   "sentText": "Massachusetts ASTON MAGNA Great Barrington ; also at Bard College , Annandale-on-Hudson , N.Y. , July 1-Aug .",
#   "articleId": "/m/vinci8/data1/riedel/projects/relation/kb/nyt1/docstore/nyt-2005-2006.backup/1669365.xml.pb",
#   "relationMentions": [
#     {
#       "em1Text": "Annandale-on-Hudson",
#       "em2Text": "Bard College",
#       "label": "/location/location/contains"
#     }
#   ],
#   "entityMentions": [
#     {
#       "start": 1,
#       "label": "ORGANIZATION",
#       "text": "Bard College"
#     },
#     {
#       "start": 2,
#       "label": "LOCATION",
#       "text": "Annandale-on-Hudson"
#     }
#   ],
#   "sentId": "1"
# }