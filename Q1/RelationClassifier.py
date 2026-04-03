from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn


# Will complete this tommorrow :: gonna go have some sleep now : Good night
class RelationClassifier(nn.Module):
    def __init__(self, model_name, num_labels, lora_r, lora_alpha, vocab_size):
        super().__init__()
        # 1. load base model
        # 2. resize embeddings (because we added special tokens)
        # 3. apply LoRA
        # 4. add classification head: Linear(hidden_size*2, num_labels)
        pass

    def pool_entity(self, hidden_states, entity_mask):
        # entity_mask: (batch, seq_len) with 1s at entity positions
        # hidden_states: (batch, seq_len, hidden_size)
        # return: (batch, hidden_size) — mean of entity token hidden states
        pass

    def forward(self, input_ids, attention_mask, entity_map1, entity_map2):
        # 1. pass through base model → hidden states
        # 2. pool e1 and e2
        # 3. concat and classify
        pass