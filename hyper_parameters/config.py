from dataclasses import dataclass

@dataclass
class PartAConfig:
    # Part A
    model_name :str = "Qwen/Qwen2.5-1.5B"
    max_length :int = 128
    batch_size :int = 16 
    epochs :int = 3
    lr :int = 2e-4
    lora_r  :int = 16
    lora_alpha  :int = 32
    dropout :float = 0.1
    lora_dropout :float = 0.1