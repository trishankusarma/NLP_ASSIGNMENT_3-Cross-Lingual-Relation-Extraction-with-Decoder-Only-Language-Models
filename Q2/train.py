# ── Imports ───────────────────────────────────────────────────
import argparse, random, time, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

from utils.logger_class import logging
from utils.utils import load_jsonl, load_lang_map
from hyper_parameters.config import PartBConfig
from .dataset_wrapper import DatasetWrapper, build_prompt, build_target
from .model_class import ModelClass
from .evaluate import evaluate_loss, run_all_f1
from .stage_1_train import run_cpt
from .infer import load_valid_labels

config = PartBConfig()

LANG_MAX_LENGTHS = {
    'en'  : 142,
    'hi'  : 400,
    'kn'  : 512,
    'or'  : 512,
    'tcy' : 512,
}

LORA_ADAPTER_STAGE_1_DIR = "lora_adapter_stage_1"

# ── Helpers ───────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def flatten_pairs(data):
    """Flatten sentence-level data to pairs — no max_length stored per sample."""
    samples = []
    for sample in data:
        sent = sample['sentText']
        for rel in sample['relationMentions']:
            samples.append({
                'prompt': build_prompt(sent, rel['em1Text'], rel['em2Text']),
                'target': build_target(rel.get('label', 'NA')),
            })
    return samples

def collate_fn(batch):
    """Simple stack — safe because single-language batches have same shape."""
    batch = [b for b in batch if (b["labels"] != -100).sum() > 0]
    if len(batch) == 0:
        return None
    return {
        "input_ids"      : torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask" : torch.stack([b["attention_mask"]  for b in batch]),
        "labels"         : torch.stack([b["labels"]          for b in batch]),
    }

def make_loader(pairs, tokenizer, max_length, batch_size, shuffle=True):
    """max_length stored once at dataset level — not per sample."""
    dataset = DatasetWrapper(pairs, tokenizer, max_length=max_length)
    return DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle, num_workers=0, pin_memory=True
    )

def round_robin_epoch(loaders):
    """Yield batches in round-robin across loaders until all exhausted."""
    iterators = [iter(l) for l in loaders]
    active    = list(range(len(loaders)))
    while active:
        next_active = []
        for i in active:
            try:
                yield next(iterators[i])
                next_active.append(i)
            except StopIteration:
                pass
        active = next_active

def update_label_to_english(indic_data, lang_map, lang):
    reverse_map = {v: k for k, v in lang_map.items()}
    if lang == "kn":
        reverse_map["/ಸ್ಥಳ/ದೇಶ/ರಾಜಧಾನ"] = reverse_map.get("/ಸ್ಥಳ/ದೇಶ/ರಾಜಧಾನಿ", "")
    for sample in indic_data:
        for rel in sample['relationMentions']:
            updated = reverse_map.get(rel["label"], None)
            if updated is not None:
                rel["label"] = updated
    return indic_data

# ── Main ──────────────────────────────────────────────────────

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = 'right'

    # Step 2: Label maps
    valid_labels = load_valid_labels()
    lang_maps = {
        'hi'  : load_lang_map('hi'),
        'kn'  : load_lang_map('kn'),
        'or'  : None,
        'tcy' : None,
    }

    # Step 3: Load + convert labels to English
    print("Loading all data...")
    english_train_data = load_jsonl(args.english_train_file)
    eng_valid_data     = load_jsonl(args.english_valid_file)

    hindi_train_data   = update_label_to_english(
                            load_jsonl(args.hindi_train_file),
                            lang_maps['hi'], 'hi')
    kannada_train_data = update_label_to_english(
                            load_jsonl(args.kanada_train_file),
                            lang_maps['kn'], 'kn')
    oria_train_data    = load_jsonl(args.oria_train_file)
    tulu_valid_data    = load_jsonl(args.tulu_valid_file)

    # Step 4: Flatten — no max_length in samples
    eng_pairs = flatten_pairs(english_train_data)
    hi_pairs  = flatten_pairs(hindi_train_data)
    kn_pairs  = flatten_pairs(kannada_train_data)
    or_pairs  = flatten_pairs(oria_train_data)
    tcy_pairs = flatten_pairs(tulu_valid_data)

    # Upsample Indic 10x
    hi_pairs_up  = hi_pairs  * 1
    kn_pairs_up  = kn_pairs  * 1
    or_pairs_up  = or_pairs  * 1
    tcy_pairs_up = tcy_pairs * 1

    print(f"Pairs — en:{len(eng_pairs)} hi:{len(hi_pairs_up)} "
          f"kn:{len(kn_pairs_up)} or:{len(or_pairs_up)} tcy:{len(tcy_pairs_up)}")

    # Step 5: Per-language DataLoaders — max_length at dataset level
    en_loader  = make_loader(eng_pairs,    tokenizer, LANG_MAX_LENGTHS['en'],  config.batch_size)
    hi_loader  = make_loader(hi_pairs_up,  tokenizer, LANG_MAX_LENGTHS['hi'],  config.batch_size)
    kn_loader  = make_loader(kn_pairs_up,  tokenizer, LANG_MAX_LENGTHS['kn'],  config.batch_size)
    or_loader  = make_loader(or_pairs_up,  tokenizer, LANG_MAX_LENGTHS['or'],  config.batch_size)
    tcy_loader = make_loader(tcy_pairs_up, tokenizer, LANG_MAX_LENGTHS['tcy'], config.batch_size)

    train_loaders = [en_loader, hi_loader, kn_loader, or_loader, tcy_loader]

    # Validation loaders
    eng_valid_loader  = make_loader(
        flatten_pairs(eng_valid_data),  tokenizer,
        LANG_MAX_LENGTHS['en'],  config.batch_size * 4, shuffle=False
    )
    hindi_valid_loader = make_loader(
        flatten_pairs(hindi_train_data), tokenizer,
        LANG_MAX_LENGTHS['hi'], config.batch_size * 4, shuffle=False
    )
    kanada_valid_loader = make_loader(
        flatten_pairs(kannada_train_data), tokenizer,
        LANG_MAX_LENGTHS['kn'], config.batch_size * 4, shuffle=False
    )
    oria_valid_loader = make_loader(
        flatten_pairs(oria_train_data), tokenizer,
        LANG_MAX_LENGTHS['or'], config.batch_size * 4, shuffle=False
    )
    tulu_valid_loader = make_loader(
        flatten_pairs(tulu_valid_data), tokenizer,
        LANG_MAX_LENGTHS['tcy'], config.batch_size * 4, shuffle=False
    )

    valid_loaders = [eng_valid_loader, hindi_valid_loader, kanada_valid_loader, oria_valid_loader, tulu_valid_loader]

    steps_per_epoch = sum(len(l) for l in train_loaders)
    print(f"Steps per epoch: {steps_per_epoch:,}")

    # Save config for inference
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump({"lang_max_lengths": LANG_MAX_LENGTHS}, f, indent=2)

    # Step 6: Build model
    print(f"Building model: {config.model_name} + LoRA")
    model = ModelClass(hyper_parameters=config, apply_lora=False)

    # Step 7.1 : Stage 1: CPT
    if args.run_cpt:
        model = model.to(device)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model.base_model = get_peft_model(model.base_model, lora_config)
        model = model.to(device)
        run_cpt(model, tokenizer, device, config, wiki_dir=args.wiki_dir)
        model.base_model.save_pretrained(
            os.path.join(args.output_dir, LORA_ADAPTER_STAGE_1_DIR)
        )
        print("[CPT] Stage 1 adapter saved")

    # Step 7.2 : Load Stage 1 adapter before SFT if exists
    stage1_path = os.path.join(args.output_dir, LORA_ADAPTER_STAGE_1_DIR)
    if os.path.exists(stage1_path):
        print(f"[SFT] Loading Stage 1 adapter from {stage1_path}")
        model.base_model = PeftModel.from_pretrained(
            model.base_model, stage1_path
        )
        model.base_model.enable_adapter_layers() 
        model.base_model.print_trainable_parameters()  # ← verify params are trainable
    else:
        print("[SFT] No Stage 1 adapter — applying fresh LoRA")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model.base_model = get_peft_model(model.base_model, lora_config)
        model.base_model.print_trainable_parameters()  # ← already trainable

    model.base_model.train()
    model = model.to(device)

    # Step 7.3 : Stage 2: SFT
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=steps_per_epoch * config.epochs
    )
    best_val_loss = float('inf')
    lang_codes = ['en', 'hi', 'kn', 'or', 'tcy']
    # Step 8 : train the model for stage 2
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        step_count = 0
        start_time = time.time()

        pbar = tqdm(
            round_robin_epoch(train_loaders),
            total=steps_per_epoch,
            desc=f"Epoch {epoch+1} Training"
        )
        for batch in pbar:
            if batch is None:
                continue

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            step_count += 1

            if step_count % 100 == 0 or step_count == steps_per_epoch:
                pbar.set_postfix({"loss": f"{total_loss/step_count:.4f}"})

        print(f"Epoch {epoch+1}/{config.epochs} :: "
              f"Time: {time.time()-start_time:.1f}s | "
              f"Loss: {total_loss/step_count:.4f}")

        # Validation
        val_loss = 0
        length_loader = 0
        for index, valid_loader in enumerate(valid_loaders):
            lang_val_loss  = evaluate_loss(model, valid_loader,  device)
            val_loss += lang_val_loss
            length_loader += len(valid_loader)

            print(f"{lang_codes[index]} : Val Loss  : {lang_val_loss/len(valid_loader):.4f}")
            
        val_loss /= length_loader
        print(f"Overall  : {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.base_model.save_pretrained(
                os.path.join(args.output_dir, "lora_adapter")
            )
            tokenizer.save_pretrained(
                os.path.join(args.output_dir, "tokenizer")
            )
            print(f"Saved best model (loss={best_val_loss:.4f})")

        # F1 on all languages — uncomment for debugging
        run_all_f1(
            model, tokenizer,
            eng_valid_data, hindi_train_data, kannada_train_data,
            oria_train_data, tulu_valid_data,
            valid_labels, lang_maps, device, config, LANG_MAX_LENGTHS,
            tag=f"[Epoch {epoch+1}]"
        )

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--english_train_file", type=str, required=True)
    parser.add_argument("--hindi_train_file",   type=str, required=True)
    parser.add_argument("--kanada_train_file",  type=str, required=True)
    parser.add_argument("--oria_train_file",    type=str, required=True)
    parser.add_argument("--tulu_valid_file",    type=str, required=True)
    parser.add_argument("--english_valid_file", type=str, required=True)
    parser.add_argument("--output_dir",         type=str, required=True)
    parser.add_argument("--run_cpt",  action="store_true")
    parser.add_argument("--wiki_dir", type=str, default="./wikipedia_dumps")
    args = parser.parse_args()

    logging(s="Q2")
    main(args)