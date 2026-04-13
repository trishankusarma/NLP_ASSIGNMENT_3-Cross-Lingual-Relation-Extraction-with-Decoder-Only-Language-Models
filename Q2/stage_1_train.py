import os
import random
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

def run_cpt(model, tokenizer, device, config, wiki_dir="./wikipedia_dumps"):
    """
    Stage 1: Continued Pre-Training on full Indic Wikipedia text.
    Teaches the model Indic language patterns before SFT.
    
    LR is intentionally lower than SFT (5e-5) to gently adapt
    without overwriting English knowledge.
    """
    # Use full corpus for each language
    sample_sizes = {
        "hi"  : 120000,   # use 3/4 th Hindi Wikipedia
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
            all_texts.append(ds[i]['text'][:1000]) # Taking first 1000 characters
        print(f"[CPT] Loaded {n} articles from {lang} Wikipedia")
    
    if not all_texts:
        print("[CPT] No Wikipedia data found — skipping CPT stage")
        return

    random.shuffle(all_texts)
    print(f"[CPT] Total CPT articles: {len(all_texts)}")

    # Pre-tokenize in chunks
    print("[CPT] Pre-tokenizing...")
    chunk_size = 10000
    all_ids, all_masks, all_labels = [], [], []

    for start in tqdm(range(0, len(all_texts), chunk_size), desc="Pre-tokenizing"):
        chunk = all_texts[start:start + chunk_size]
        enc   = tokenizer(
            chunk,
            max_length     = 256,
            truncation     = True,
            padding        = 'max_length',
            return_tensors = 'pt',
        )
        ids  = enc["input_ids"]
        mask = enc["attention_mask"]
        lbl  = ids.clone()
        lbl[mask == 0] = -100
        all_ids.append(ids)
        all_masks.append(mask)
        all_labels.append(lbl)
    
    all_ids    = torch.cat(all_ids)
    all_masks  = torch.cat(all_masks)
    all_labels = torch.cat(all_labels)
    print(f"[CPT] Pre-tokenized {all_ids.shape[0]} samples")

    # ── DataLoader — same batch_size, no change ───────────────
    cpt_dataset = TensorDataset(all_ids, all_masks, all_labels)
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
    for step, (ids_b, mask_b, lbl_b) in enumerate(pbar):
        ids_b  = ids_b.to(device)
        mask_b = mask_b.to(device)
        lbl_b  = lbl_b.to(device)
 
        optimizer.zero_grad()
        outputs = model(ids_b, mask_b, lbl_b)
        loss    = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
 
        total_loss += loss.item()
        if (step + 1) % 200 == 0:
            pbar.set_postfix({"cpt_loss": f"{total_loss/(step+1):.4f}"})

    print(f"[CPT] Done. Avg loss: {total_loss/len(cpt_loader):.4f}")