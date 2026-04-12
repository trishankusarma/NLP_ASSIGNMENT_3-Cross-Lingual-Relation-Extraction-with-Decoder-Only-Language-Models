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

# For inference it will still be the same 
class DatasetWrapper(Dataset):
    def __init__(self, samples_or_path, tokenizer=None, 
                 max_length=172, inference=False):
        
        if isinstance(samples_or_path, str):
            # Load pre-tokenized training data
            data = torch.load(samples_or_path, weights_only=True)
            self.input_ids      = data["input_ids"]
            self.attention_mask = data["attention_mask"]
            self.labels         = data["labels"]
            self.pretokenized   = True
        else:
            # Inference mode — tokenize on the fly
            self.samples      = samples_or_path
            self.tokenizer    = tokenizer
            self.max_length   = max_length
            self.inference    = inference
            self.pretokenized = False

    def __len__(self):
        if self.pretokenized:
            return len(self.input_ids)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.pretokenized:
            return {
                "input_ids"      : self.input_ids[idx],
                "attention_mask" : self.attention_mask[idx],
                "labels"         : self.labels[idx],
            }
        
        # Inference path — same as before
        sample = self.samples[idx]
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