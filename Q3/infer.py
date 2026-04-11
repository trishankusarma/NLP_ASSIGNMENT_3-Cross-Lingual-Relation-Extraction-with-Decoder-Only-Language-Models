import os
import json
import re
import argparse
import numpy as np
from tqdm import tqdm

from utils.logger_class import logging
from utils.utils import load_jsonl, load_lang_map

# ── Model path (local on HPC) ─────────────────────────────────
MODEL_PATH = "/home/scai/msr/aiy247541/scratch/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

def main(args):
    from vllm import LLM, SamplingParams

    print(f"Loading model from {MODEL_PATH}...")
    llm = LLM(
        model              = MODEL_PATH,
        dtype              = "float16",
        max_model_len      = 4096,
        gpu_memory_utilization = 0.85,
    )
    sampling_params = SamplingParams(
        temperature = 0.0,    # greedy
        max_tokens  = 50,
        stop        = ["\n\n", "Example"],
    )
    print("Model loaded!")

    # quick sanity test
    test_out = llm.generate(["Hello, who are you?"], sampling_params)
    print(f"Sanity check: {test_out[0].outputs[0].text[:100]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",       type=str, required=True)
    parser.add_argument("--test_file",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="Q3/output")
    args = parser.parse_args()

    logging(s="Q3.infer")
    print(f"Q3 Inference | lang={args.lang}")
    main(args)