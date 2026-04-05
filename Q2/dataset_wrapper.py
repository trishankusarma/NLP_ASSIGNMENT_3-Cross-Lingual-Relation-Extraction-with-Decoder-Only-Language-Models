import torch
from torch.utils.data import Dataset

def build_prompt(sent, e1, e2):
    # We now build the input prompt. The model needs to learn this exact format.
    # Will update if necessary
    return (
        f"Task: Extract the relationship between the two entities in the sentence.\n"
        f"Sentence: {sent}\n"
        f"Entity1: {e1}\n"
        f"Entity2: {e2}\n"
        f"Answer:"
    )

def build_target(label):
    # The target the model should generate after 'Answer:
    return f'{{"label": "{label}"}}'

class DatasetWrapper(Dataset):
    # Each item is a (prompt + target) string for causal LM training.
    # We only compute loss on the TARGET tokens (not the prompt).
    def __init__(self, samples, tokenizer, max_length=172, inference=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = samples
        self.inference = inference
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.inference:
            prompt = sample['prompt']
            enc = self.tokenizer(
                prompt,
                max_length     = self.max_length,
                truncation     = True,
                padding        = 'max_length',
                return_tensors = 'pt'
            )
            return {
                "input_ids"      : enc["input_ids"].squeeze(0),
                "attention_mask" : enc["attention_mask"].squeeze(0),
            }

        prompt, target = sample['prompt'], sample['target']
        full_text = prompt + target
 
        # Tokenize full text
        full_text_encode = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
 
        # Tokenize prompt only (to find where target starts)
        prompt_encode = self.tokenizer(
            prompt,
            truncation=False,
            return_tensors='pt'
        )

        input_ids      = full_text_encode["input_ids"].squeeze(0)
        attention_mask = full_text_encode["attention_mask"].squeeze(0)
        labels         = input_ids.clone()

        prompt_len = min(prompt_encode["input_ids"].shape[1], self.max_length)

        # mask prompt and padding
        # don't compute loss on them :: PyTorch CrossEntropyLoss ignores positions where label == -100
        labels[:prompt_len]          = -100
        labels[attention_mask == 0]  = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels
        }