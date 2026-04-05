import torch
from torch.utils.data import Dataset

def update_sentence(sent, em1, em2, special_tokens):

    curr = sent

    if em1 in sent:
        curr = curr.replace(em1, f"{special_tokens[0]} {em1} {special_tokens[1]}")

    if em2 in sent:
        curr = curr.replace(em2, f"{special_tokens[2]} {em2} {special_tokens[3]}")   

    sentence = f"Sentence: {curr} Entity1: {em1} Entity2 : {em2}"     
    return sentence

class DatasetWrapper(Dataset):
    def __init__(self, pairs, tokenizer, special_tokens, max_length = 256):
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.max_length = max_length
        self.special_tokens = special_tokens # ["[EM1]", "[/EM1]", "[EM2]", "[/EM2]"]
    
    def __len__(self):
        return len(self.pairs)
    
    def built_offset_map_for_entity(self, offset_mapping, en_start, en_end):
        mask = torch.zeros(len(offset_mapping), dtype=torch.long)

        # [0, 5], [6, 10], [11, 14], [15, 20]
        # (6, 14)
        # [0, 1, 1, 0]

        for index, (start, end) in enumerate(offset_mapping):
            if start >= en_start and end <= en_end:
                mask[index] = 1
        
        return mask

    def __getitem__(self, idx):
        
        pair = self.pairs[idx]
        em1Text = pair["em1Text"]
        em2Text = pair["em2Text"]

        sentence = update_sentence(pair["sentText"], em1Text, em2Text, self.special_tokens)

        encoded_token = self.tokenizer(
            sentence,
            max_length = self.max_length,
            padding="max_length", # pad sentence to max length
            truncation=True, # truncate if sentence > max_length
            return_tensors="pt", # return PyTorch tensors directly
            return_offsets_mapping=True  # gives char start/end for each token → helps find entity positions
            # return_tensors="pt" gives shape (1, max_length)
        )
        
        input_ids = encoded_token["input_ids"].squeeze(0) # makes the dimension < max_length , >
        attention_mask = encoded_token["attention_mask"].squeeze(0)
        offset_mapping = encoded_token["offset_mapping"].squeeze(0).tolist()

        # build entity masking too for entity1 and entity2
        en1_start = sentence.find(f"{self.special_tokens[0]} ") + len(f"{self.special_tokens[1]} ")
        en1_end = en1_start + len(em1Text)

        en2_start = sentence.find(f"{self.special_tokens[2]} ") + len(f"{self.special_tokens[3]} ")
        en2_end = en2_start + len(em2Text)

        # for all tokens lying between en1_start and en1_end :: should have 1 for entity_mask1
        entity_map1 = self.built_offset_map_for_entity(offset_mapping, en1_start, en1_end)
        # for all tokens lying between en2_start and en2_end :: should have 1 for entity_mask2
        entity_map2 = self.built_offset_map_for_entity(offset_mapping, en2_start, en2_end)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "entity_map1": entity_map1,
            "entity_map2": entity_map2,
            "label": torch.tensor(pair.get("label_id", -1),  dtype=torch.long)
        }