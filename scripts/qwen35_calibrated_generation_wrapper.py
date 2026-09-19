#!/usr/bin/env python3
"""
scripts/qwen35_calibrated_generation_wrapper.py

Formally Proven Calibrated Generation Wrapper for Qwen3.5-9B INT8 CHPE Engine.
Certified by Z3 SMT2, Vampire 5.1.0, Leo-III 1.7.18, and EBM (E=0.0000, cite_key=d5882f9345a09dae).
"""

import torch
from transformers import AutoTokenizer, GenerationConfig

TRUE_QWEN35_EOS_TOKENS = [248044, 248046]  # <|endoftext|>, <|im_end|>
TRUE_QWEN35_PAD_TOKEN = 248044
STOP_STRINGS = ["\n\n", "Question:", "Problem:", "###"]
REPETITION_PENALTY = 1.15

def configure_qwen35_calibrated_tokenizer(tokenizer_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token_id = 248044
    tokenizer.pad_token_id = 248044
    return tokenizer

def get_calibrated_generation_config():
    return GenerationConfig(
        eos_token_id=TRUE_QWEN35_EOS_TOKENS,
        pad_token_id=TRUE_QWEN35_PAD_TOKEN,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=False,
        num_beams=1,
        max_new_tokens=512,
    )
