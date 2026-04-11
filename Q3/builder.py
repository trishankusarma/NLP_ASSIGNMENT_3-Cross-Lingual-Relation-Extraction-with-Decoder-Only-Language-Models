import re
import json

def build_icl_prompt(test_sent, test_e1, test_e2, examples):
    prompt = (
        "You are an expert in relation extraction. "
        "Given a sentence and two entities, identify the relationship. "
        'Respond ONLY with JSON: {"label": "<relation>"}. '
        "Use NA if no relation.\n\n"
    )
    for i, ex in enumerate(examples):
        prompt += (
            f"Example {i+1}:\n"
            f"Sentence: {ex['sentText']}\n"
            f"Entity1: {ex['em1Text']}\n"
            f"Entity2: {ex['em2Text']}\n"
            f'Answer: {{"label": "{ex["label"]}"}}\n\n'
        )
    prompt += (
        f"Now extract:\n"
        f"Sentence: {test_sent}\n"
        f"Entity1: {test_e1}\n"
        f"Entity2: {test_e2}\n"
        f"Answer:"
    )
    return prompt

def parse_label(text, valid_labels, fallback="NA"):
    text = text.strip()
    try:
        match = re.search(r'\{[^}]+\}', text)
        if match:
            label = json.loads(match.group()).get('label', '').strip()
            if label in valid_labels:
                return label
    except:
        pass
    for label in sorted(valid_labels, key=len, reverse=True):
        if label in text:
            return label
    match = re.search(r'/[\w/]+', text)
    if match:
        for label in valid_labels:
            if match.group() in label:
                return label
    return fallback

def reconstruct_output(test_data, pred_map, lang_map):
    output = []
    for sent_idx, sample in enumerate(test_data):
        out_sample = {
            'articleId'        : sample.get('articleId', ''),
            'sentId'           : sample.get('sentId', ''),
            'sentText'         : sample['sentText'],
            'relationMentions' : [],
        }
        for pair_idx, rel in enumerate(sample['relationMentions']):
            pred_en = pred_map.get((sent_idx, pair_idx), 'NA')
            pred_label = lang_map[pred_en] if (lang_map and pred_en in lang_map) else pred_en
            out_sample['relationMentions'].append({
                'em1Text' : rel['em1Text'],
                'em2Text' : rel['em2Text'],
                'label'   : pred_label,
            })
        output.append(out_sample)
    return output