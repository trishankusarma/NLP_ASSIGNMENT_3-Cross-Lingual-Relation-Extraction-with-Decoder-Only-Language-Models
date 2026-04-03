import json
import os
from collections import Counter

from utils.utils import load_jsonl

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

label_mapping_dir = './label_mapping'
os.makedirs(label_mapping_dir, exist_ok=True)

inter_lang_label_map_paths = [
    "sft_dataset/hi_map.json",
    "sft_dataset/kn_map.json",
    "sft_dataset/or_map.json",
    "sft_dataset/tcy_map.json"
]

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

def buildIndexAndLabelMapping():
    # Goal is to map all the index2label and label2index
    # Here comes some complex logic
    # English literature -> we will have all the labels from english literature
    english_data = load_jsonl(ENGLISH_DATASET_TRAIN_PATH)

    english_labels = set()
    for sample in english_data:
        for relationMention in sample["relationMentions"]:
            english_labels.add(relationMention["label"])
    
    english_labels = sorted(list(english_labels))

    index2label = {index: label for index, label in enumerate(english_labels)}
    label2index = {label: index for index, label in enumerate(english_labels)}
    
    # lets get the global mapping done
    for map_path in inter_lang_label_map_paths:

        if not os.path.exists(map_path):
            print(f"Skipping file {map_path} : does not exist")
            continue

        with open(map_path, 'r', encoding='utf-8') as f:
            translation_map = json.load(f)
        
        # Here all we have to do is map the indic_lang label to the corresponding english_lang label index
        for english_label, indic_label in translation_map.items():

            if english_label in english_labels:
                canonical_index = label2index[english_label]
                label2index[indic_label] = canonical_index
            else:
                print(f"Warning {english_label} not in english_labels set")
    
    index2labelFilePath = f"{label_mapping_dir}/index2label.json"
    label2indexFilePath = f"{label_mapping_dir}/label2index.json"
    
    # store index2label and label2index
    with open(index2labelFilePath, 'w', encoding='utf-8') as f:
        json.dump(index2label, f, indent=2, ensure_ascii=False)
    
    with open(label2indexFilePath, 'w', encoding='utf-8') as f:
        json.dump(label2index, f, indent=2, ensure_ascii=False)
    
    print(f"Saved index2label and label2index maps in {index2labelFilePath} and {label2indexFilePath} path")

if __name__ == "__main__":
    for dataset in datasets:
        print(f"\n{'='*60}")
        # this is for mere exploring the datasets given apriorily
        explore(dataset[0], dataset[1])
        
    # Next is the real thing to build the mapping between labels
    buildIndexAndLabelMapping()

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