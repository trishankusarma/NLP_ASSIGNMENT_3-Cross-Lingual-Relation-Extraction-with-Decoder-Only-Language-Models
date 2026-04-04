from dataclasses import dataclass

@dataclass
class PartAConfig:
    # Part A
    model_name :str = "Qwen/Qwen2.5-1.5B"
    max_length :int = 128
    batch_size :int = 8 
    epochs :int = 5
    lr :float = 1e-4
    lora_r  :int = 32
    lora_alpha  :int = 64
    dropout :float = 0.1
    lora_dropout :float = 0.1
    weight_decay :float = 0.01