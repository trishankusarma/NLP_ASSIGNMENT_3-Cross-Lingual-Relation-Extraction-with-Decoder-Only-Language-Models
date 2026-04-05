from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn as nn

class ModelClass(nn.Module):
    def __init__(self, hyper_parameters, apply_lora = True):
        super().__init__()

        self.base_model = AutoModelForCausalLM.from_pretrained(
            hyper_parameters.model_name,
            dtype=torch.bfloat16
        )

        if apply_lora:
 
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=hyper_parameters.lora_r,
                lora_alpha=hyper_parameters.lora_alpha,
                lora_dropout=hyper_parameters.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            self.base_model.print_trainable_parameters()
    
    def forward(self, input_ids, attention_mask, labels=None):
        # just pass through to base model
        # HuggingFace CausalLM computes loss automatically if labels provided
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        ) 
        return outputs