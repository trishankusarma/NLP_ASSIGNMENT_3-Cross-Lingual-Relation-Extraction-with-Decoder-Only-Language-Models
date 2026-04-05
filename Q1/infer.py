import argparse
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

from hyper_parameters.config import PartAConfig
from .dataset_wrapper import DatasetWrapper
from utils.utils import load_jsonl

LABEL_2_INDEX_PATH = "./label_mapping/label2index.json"
INDEX_2_LABEL_PATH = "./label_mapping/index2label.json"

config = PartAConfig()

def load_model(output_dir, num_labels, vocab_size, device):
    # Step 1: load base model
    # Step 2: resize embeddings
    # Step 3: load LoRA adapter
    # Step 4: load classifier head
    # Step 5: model.eval() + move to device
    pass

def flatten_test_data(test_data):
    # same as training but:
    # - no label_id needed
    # - keep articleId, sentId for reconstruction
    pass

def run_inference(model, data_loader, index2label, device):
    # return list of predicted label strings
    pass

def reconstruct_output(test_data, predictions):
    # group predictions back by sentence
    # return list of output dicts in required format
    pass

def main(args):
    pass