import os
import random
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

class CPTDataset(Dataset):
    """Raw text dataset for causal language modeling — no prompts, no labels."""
    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = texts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx],
            max_length     = self.max_length,
            truncation     = True,
            padding        = 'max_length',
            return_tensors = 'pt'
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels         = input_ids.clone()
        labels[attention_mask == 0] = -100   # ignore padding in loss
        return {
            "input_ids"      : input_ids,
            "attention_mask" : attention_mask,
            "labels"         : labels,
        }

def run_cpt(model, tokenizer, device, config, wiki_dir="./wikipedia_dumps"):
    """
    Stage 1: Continued Pre-Training on full Indic Wikipedia text.
    Teaches the model Indic language patterns before SFT.
    
    LR is intentionally lower than SFT (5e-5) to gently adapt
    without overwriting English knowledge.
    """
    # Use full corpus for each language
    sample_sizes = {
        "hi"  : 160000,   # full Hindi Wikipedia
        "kn"  : 30000,    # full Kannada Wikipedia
        "or"  : 17375,    # full Oriya Wikipedia
        "tcy" : 2202,     # full Tulu Wikipedia (tiny)
    }

    all_texts = []
    for lang in ["hi", "kn", "or", "tcy"]:
        path = os.path.join(wiki_dir, f"wiki_{lang}")
        if not os.path.exists(path):
            print(f"[CPT] Skipping {lang} — not found at {path}")
            continue
        ds = load_from_disk(path)
        n  = min(sample_sizes[lang], len(ds))
        sampled = random.sample(range(len(ds)), n)
        for i in sampled:
            all_texts.append(ds[i]['text'])
        print(f"[CPT] Loaded {n} articles from {lang} Wikipedia")
    
    if not all_texts:
        print("[CPT] No Wikipedia data found — skipping CPT stage")
        return

    random.shuffle(all_texts)
    print(f"[CPT] Total CPT articles: {len(all_texts)}")
 
    cpt_dataset = CPTDataset(all_texts, tokenizer, max_length=256)
    cpt_loader  = DataLoader(
        cpt_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
 
    # Separate optimizer for CPT with lower LR
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = config.stage_1_lr,             # lower than SFT lr
        weight_decay = config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=len(cpt_loader))
 
    model.train()
    total_loss = 0
    print(f"[CPT] Starting Stage 1: Continued Pre-Training on {len(all_texts)} articles...")
 
    pbar = tqdm(cpt_loader, desc="CPT Stage 1")
    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
 
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss    = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
 
        total_loss += loss.item()
        if (step + 1) % 200 == 0:
            pbar.set_postfix({"cpt_loss": f"{total_loss/(step+1):.4f}"})
 
    print(f"[CPT] Stage 1 complete. Avg loss: {total_loss/len(cpt_loader):.4f}")