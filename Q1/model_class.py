from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn as nn

# Lets gooooo
class ModelClass(nn.Module):
    def __init__(self, hyper_parameters, num_labels, vocab_size, class_weights):
        super().__init__()

        # 1. load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            hyper_parameters.model_name,
            dtype=torch.float16
        )
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        self.register_buffer("class_weights", class_weights)

        # 2. resize token embedding for special tokens added
        self.base_model.resize_token_embeddings(vocab_size)
        
        # 3. apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hyper_parameters.lora_r,
            lora_alpha=hyper_parameters.lora_alpha,
            lora_dropout=hyper_parameters.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        self.base_model.print_trainable_parameters()

        # 4. add classifier head: Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(hyper_parameters.dropout)
        # bcz we are concatenating entity1 and entity2 representations -> so hidden_size*2
        self.classifier = nn.Linear(self.hidden_size * 2, num_labels)

    def pool_entity(self, hidden_states, entity_mask):
        # entity_mask: (batch, seq_len) with 1s at entity positions
        entity_mask = entity_mask.unsqueeze(-1).to(hidden_states.dtype) #(B, seq_len, 1)
        # hidden_states: (batch, seq_len, hidden_size)
        masked_entity = (hidden_states * entity_mask) # element-wise multiplication -> (batch, seq_len, hidden_size)
        summed_mask = masked_entity.sum(dim = 1) # summed mask over all tokens of a sequence -> (batch, hidden_size)
        count_entity = entity_mask.sum(dim = 1).clamp(min=1) # count of entities in a sequence -> (batch, 1)
        # (batch, hidden_size) — mean of entity token hidden states
        return summed_mask / count_entity

    def forward(self, input_ids, attention_mask, entity_map1, entity_map2, label = None):
        # All the parameters have shape ( B, seq_len )
        # 1. pass through base model → hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ) 
        # hidden layers  = 29, last layer size  = (16, 128, 1536)
        last_hidden_layer = outputs.hidden_states[-1] #( B, seq_len, hidden_state )
        # 2. pool e1 and e2
        pooled_enitity1 = self.pool_entity(last_hidden_layer, entity_map1) # (B, hidden_state)
        pooled_enitity2 = self.pool_entity(last_hidden_layer, entity_map2) # (B, hidden_state)
        # 3. concat and classify
        concated_entities = torch.cat([pooled_enitity1, pooled_enitity2], dim = -1) # (B, 2*hidden_state)
        concated_entities = self.dropout(concated_entities.float())
        logits = self.classifier(concated_entities) # (B, num_labels)

        loss = None  
        if label is not None: # during inference label might be none
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.float())
            loss = criterion(logits, label)
        return logits, loss