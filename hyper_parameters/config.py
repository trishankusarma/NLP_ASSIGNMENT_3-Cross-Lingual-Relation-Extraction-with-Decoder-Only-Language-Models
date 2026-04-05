from dataclasses import dataclass

@dataclass
class PartAConfig:
    # Part A
    model_name :str = "Qwen/Qwen2.5-1.5B"
    max_length :int = 128
    batch_size :int = 16 
    epochs :int = 5
    lr :float = 1e-4
    lora_r  :int = 16
    lora_alpha  :int = 32
    dropout :float = 0.1
    lora_dropout :float = 0.1
    weight_decay :float = 0.01

@dataclass
class PartBConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B"
    max_input_length: int = 172   # prompt length
    max_new_tokens: int = 32      # label is short
    batch_size: int = 16
    epochs: int = 5
    lr: float = 1e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    dropout: float = 0.1
    weight_decay: float = 0.01

    tulu_valid_data_used: float = 0.75