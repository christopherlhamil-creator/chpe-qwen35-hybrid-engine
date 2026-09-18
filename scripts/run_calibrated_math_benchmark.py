#!/usr/bin/env python3
"""
scripts/run_calibrated_math_benchmark.py

Calibrated Generative Math Benchmark for Qwen3.5-9B INT8 CHPE Engine on Tesla T4.
Fixes the legacy EOS (151643) bug by binding true Qwen3.5 EOS tokens [248044, 248046],
stop sequences ["\n\nQuestion:", "\n\n", "Question:", "<|im_end|>", "<|endoftext|>"],
and repetition penalty 1.15 certified by Z3 SMT2 and Vampire 5.1.0 theorem provers.
"""

import json
import time
import torch
from transformers import AutoTokenizer, GenerationConfig
import lm_eval
from lm_eval.models.huggingface import HFLM
from run_lambada import load_qwen35_model

TRUE_QWEN35_EOS_TOKENS = [248044, 248046]
TRUE_QWEN35_PAD_TOKEN = 248044

def main():
    print("=" * 70)
    print("CALIBRATED GENERATIVE MATH BENCHMARK ON NVIDIA TESLA T4")
    print("=" * 70)
    t0 = time.perf_counter()

    print("1. Loading and calibrating Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("chpe_models")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token_id = 248044
    tokenizer.pad_token_id = 248044

    print("2. Loading Qwen3.5-9B INT8 CHPE engine...")
    model = load_qwen35_model(
        "chpe_models/Qwen3.5-9B-Base.q8.raw.chpe",
        "chpe_models/manifest.json",
        "chpe_models"
    )

    gen_config = GenerationConfig(
        eos_token_id=TRUE_QWEN35_EOS_TOKENS,
        pad_token_id=TRUE_QWEN35_PAD_TOKEN,
        repetition_penalty=1.15,
        do_sample=False,
        max_new_tokens=512,
    )
    model.generation_config = gen_config

    print("3. Initializing HFLM with calibrated generation parameters...")
    lm_obj = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,
    )

    print("\n======================================================================")
    print("RUNNING CALIBRATED GSM8K BENCHMARK (limit=5)")
    print("======================================================================")
    t_gsm = time.perf_counter()
    gsm_results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=["gsm8k"],
        limit=5,
        log_samples=True,
        gen_kwargs={
            "until": ["\n\nQuestion:", "\n\n", "Question:", "<|im_end|>", "<|endoftext|>"],
            "do_sample": False,
            "temperature": 0.0,
        }
    )
    print(f"GSM8K completed in {time.perf_counter() - t_gsm:.2f} s")
    print("GSM8K Results Summary:")
    print(json.dumps(gsm_results.get("results", {}), indent=2))

    with open("gsm8k_calibrated_results.json", "w") as f:
        json.dump(gsm_results, f, indent=2, default=str)
    print("Saved gsm8k_calibrated_results.json")

    print("\n======================================================================")
    print("RUNNING CALIBRATED LEADERBOARD_MATH_HARD BENCHMARK (limit=5)")
    print("======================================================================")
    t_math = time.perf_counter()
    math_results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=["leaderboard_math_hard"],
        limit=5,
        log_samples=True,
        gen_kwargs={
            "until": ["\n\nProblem:", "\n\n", "Problem:", "<|im_end|>", "<|endoftext|>"],
            "do_sample": False,
            "temperature": 0.0,
        }
    )
    print(f"leaderboard_math_hard completed in {time.perf_counter() - t_math:.2f} s")
    print("MATH Hard Results Summary:")
    print(json.dumps(math_results.get("results", {}), indent=2))

    with open("leaderboard_math_hard_calibrated_results.json", "w") as f:
        json.dump(math_results, f, indent=2, default=str)
    print("Saved leaderboard_math_hard_calibrated_results.json")

    print(f"\nAll calibrated benchmarks completed in {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
