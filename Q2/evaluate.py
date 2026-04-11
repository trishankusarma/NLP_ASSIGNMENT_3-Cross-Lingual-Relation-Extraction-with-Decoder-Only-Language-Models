import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from .dataset_wrapper import DatasetWrapper, build_prompt
from .infer import parse_label

def evaluate_loss(model, valid_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Evaluating Loss"):
            if batch is None:
                continue
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels
            )
            val_loss += outputs.loss.item()
    model.train()
    return val_loss

def evaluate_f1(model, tokenizer, data, lang_name, valid_labels,
                lang_map, device, config, max_length):
    """
    Generate predictions on a dataset and compute F1 scores.
    Used after every epoch to track cross-lingual progress.
    """
    model.eval()

    # Flatten to pairs
    flat_samples = []
    for sent_idx, sample in enumerate(data):
        for pair_idx, rel in enumerate(sample['relationMentions']):
            flat_samples.append({
                'prompt'   : build_prompt(sample['sentText'], rel['em1Text'], rel['em2Text']),
                'sent_idx' : sent_idx,
                'pair_idx' : pair_idx,
            })

    dataset = DatasetWrapper(
        flat_samples, tokenizer,
        max_length           = max_length,
        inference            = True,
        inference_max_length = max_length,
    )
    loader = DataLoader(
        dataset, batch_size=config.batch_size * 2,
        shuffle=False, num_workers=0
    )

    # Generate predictions
    pred_map = {}
    flat_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"F1 Eval [{lang_name}]"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated = model.base_model.generate(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = 30,
                do_sample      = False,
                pad_token_id   = tokenizer.pad_token_id,
                eos_token_id   = tokenizer.eos_token_id,
            )

            prompt_len = input_ids.shape[1]
            for i in range(generated.shape[0]):
                new_tokens = generated[i][prompt_len:]
                decoded    = tokenizer.decode(new_tokens, skip_special_tokens=True)
                pred_en    = parse_label(decoded, valid_labels)
                pred_map[(flat_samples[flat_idx]['sent_idx'],
                          flat_samples[flat_idx]['pair_idx'])] = pred_en
                flat_idx += 1

    # Build true/pred lists
    all_true, all_pred = [], []
    for sent_idx, sample in enumerate(data):
        for pair_idx, rel in enumerate(sample['relationMentions']):
            true_label = rel['label']
            # Normalize indic label → English for fair comparison
            if lang_map:
                reverse_map = {v: k for k, v in lang_map.items()}
                true_label  = reverse_map.get(true_label, true_label)
            pred_label = pred_map.get((sent_idx, pair_idx), 'NA')
            all_true.append(true_label)
            all_pred.append(pred_label)

    micro_f1 = f1_score(all_true, all_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
    print(f"  [{lang_name}] Micro-F1: {micro_f1:.4f} | Macro-F1: {macro_f1:.4f}")

    model.train()
    return micro_f1, macro_f1

def run_all_f1(model, tokenizer, eng_valid_data, train_hindi_data, train_kanada_data,
               oria_data, tulu_valid_data, valid_labels, lang_maps,
               device, config, lang_max_lengths, tag=""):
    """Convenience wrapper — runs F1 on all 5 languages at once."""
    print(f"\n--- F1 Scores {tag} ---")
    tokenizer.padding_side = 'left'
    evaluate_f1(model, tokenizer, eng_valid_data,   'en',  valid_labels, None,             device, config, lang_max_lengths['en'])
    evaluate_f1(model, tokenizer, train_hindi_data,  'hi',  valid_labels, lang_maps['hi'],  device, config, lang_max_lengths['hi'])
    evaluate_f1(model, tokenizer, train_kanada_data, 'kn',  valid_labels, lang_maps['kn'],  device, config, lang_max_lengths['kn'])
    evaluate_f1(model, tokenizer, oria_data,         'or',  valid_labels, None,             device, config, lang_max_lengths['or'])
    evaluate_f1(model, tokenizer, tulu_valid_data,   'tcy', valid_labels, None,             device, config, lang_max_lengths['tcy'])
    tokenizer.padding_side = 'right'
    print()