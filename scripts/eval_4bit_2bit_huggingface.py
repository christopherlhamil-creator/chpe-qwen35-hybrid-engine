#!/usr/bin/env python3
"""
eval_4bit_2bit_huggingface.py

Direct Zero-OOM Streaming Quantization & Hugging Face Leaderboard Battery
for 4-Bit and 2-Bit Quantization on NVIDIA Tesla T4 (Lightning AI chpe-t4).
"""

import gc
import json
import mmap
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM


class Int4Linear(nn.Module):
    def __init__(self, weight_int4, scale, bias=None):
        super().__init__()
        self.register_buffer("weight_int4", weight_int4)
        self.scale = float(scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight_int4.to(x.dtype) * self.scale
        return F.linear(x, w, self.bias)


class Int2Linear(nn.Module):
    def __init__(self, weight_int2, scale, bias=None):
        super().__init__()
        self.register_buffer("weight_int2", weight_int2)
        self.scale = float(scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight_int2.to(x.dtype) * self.scale
        return F.linear(x, w, self.bias)


class IntEmbedding(nn.Module):
    def __init__(self, weight_int, scale):
        super().__init__()
        self.register_buffer("weight_int", weight_int)
        self.scale = float(scale)

    def forward(self, input_ids):
        rows = self.weight_int[input_ids]
        return rows.to(torch.float16) * self.scale


def quantize_bytes(raw_bytes, shape, mode="4bit", scale=1.0):
    t_cpu = torch.frombuffer(bytearray(raw_bytes), dtype=torch.int8).reshape(shape)
    if mode == "4bit":
        w = torch.clamp(torch.round(t_cpu.to(torch.float32) / 16.0), -8.0, 7.0).to(torch.int8)
        eff_scale = float(scale * 16.0)
        return w.to("cuda"), eff_scale
    elif mode == "2bit":
        w = torch.clamp(torch.round(t_cpu.to(torch.float32) / 64.0), -2.0, 1.0).to(torch.int8)
        eff_scale = float(scale * 64.0)
        return w.to("cuda"), eff_scale
    else:
        return t_cpu.to("cuda"), float(scale)


def load_quantized_qwen35(archive_path: str, manifest_path: str, config_dir: str, mode: str = "4bit"):
    print(f"Loading Qwen3.5-9B config for {mode.upper()} mode...")
    cfg = AutoConfig.from_pretrained(config_dir)
    text_cfg = cfg.text_config
    text_cfg.torch_dtype = "float16"

    print("Instantiating skeleton model on meta device...")
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(text_cfg)

    print(f"Streaming {mode.upper()} weights directly from CHPE archive...")
    with open(manifest_path) as f:
        manifest = json.load(f)
    tensors = manifest["tensors"]

    t0 = time.perf_counter()
    linear_cls = Int4Linear if mode == "4bit" else Int2Linear

    with open(archive_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # 1. embed_tokens
        e_info = tensors["model.language_model.embed_tokens.weight"]
        e_off = e_info["byte_offset"]
        e_shape = e_info["shape"]
        e_scale = e_info.get("scale", 1.0)
        e_raw = mm[e_off : e_off + e_shape[0] * e_shape[1]]
        w_emb, s_emb = quantize_bytes(e_raw, e_shape, mode=mode, scale=e_scale)
        model.model.embed_tokens = IntEmbedding(w_emb, s_emb)

        # 2. Layers 0..31
        for l_idx in range(text_cfg.num_hidden_layers):
            layer = model.model.layers[l_idx]
            is_full = (l_idx + 1) % text_cfg.full_attention_interval == 0

            # Norms
            for norm_attr, t_suffix in [
                ("input_layernorm", "input_layernorm.weight"),
                ("post_attention_layernorm", "post_attention_layernorm.weight"),
            ]:
                k = f"model.language_model.layers.{l_idx}.{t_suffix}"
                if k in tensors:
                    info = tensors[k]
                    off = info["byte_offset"]
                    raw = mm[off : off + text_cfg.hidden_size * 2]
                    w = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).to(torch.float16).to("cuda")
                    setattr(layer, norm_attr, nn.RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps))
                    getattr(layer, norm_attr).weight = nn.Parameter(w, requires_grad=False)

            # MLP
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                k = f"model.language_model.layers.{l_idx}.mlp.{proj_name}.weight"
                if k in tensors:
                    info = tensors[k]
                    off = info["byte_offset"]
                    shape = info["shape"]
                    scale = info.get("scale", 1.0)
                    raw = mm[off : off + shape[0] * shape[1]]
                    w_q, s_q = quantize_bytes(raw, shape, mode=mode, scale=scale)
                    setattr(layer.mlp, proj_name, linear_cls(w_q, s_q))

            # Attention
            if is_full:
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    k = f"model.language_model.layers.{l_idx}.self_attn.{proj_name}.weight"
                    if k in tensors:
                        info = tensors[k]
                        off = info["byte_offset"]
                        shape = info["shape"]
                        scale = info.get("scale", 1.0)
                        raw = mm[off : off + shape[0] * shape[1]]
                        w_q, s_q = quantize_bytes(raw, shape, mode=mode, scale=scale)
                        setattr(layer.self_attn, proj_name, linear_cls(w_q, s_q))
                for qk in ["q_norm", "k_norm"]:
                    k = f"model.language_model.layers.{l_idx}.self_attn.{qk}.weight"
                    if k in tensors:
                        info = tensors[k]
                        off = info["byte_offset"]
                        head_dim = text_cfg.head_dim
                        raw = mm[off : off + head_dim * 2]
                        w = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).to(torch.float16).to("cuda")
                        norm_mod = nn.RMSNorm(head_dim, eps=text_cfg.rms_norm_eps)
                        norm_mod.weight = nn.Parameter(w, requires_grad=False)
                        setattr(layer.self_attn, qk, norm_mod)
            else:
                for proj_name in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]:
                    k = f"model.language_model.layers.{l_idx}.linear_attn.{proj_name}.weight"
                    if k in tensors:
                        info = tensors[k]
                        off = info["byte_offset"]
                        shape = info["shape"]
                        scale = info.get("scale", 1.0)
                        raw = mm[off : off + shape[0] * shape[1]]
                        w_q, s_q = quantize_bytes(raw, shape, mode=mode, scale=scale)
                        setattr(layer.linear_attn, proj_name, linear_cls(w_q, s_q))

                k = f"model.language_model.layers.{l_idx}.linear_attn.conv1d.weight"
                if k in tensors:
                    info = tensors[k]
                    off = info["byte_offset"]
                    shape = info["shape"]
                    raw = mm[off : off + shape[0] * shape[1] * shape[2] * 2]
                    w = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).to(torch.float16).reshape(shape).to("cuda")
                    layer.linear_attn.conv1d.weight = nn.Parameter(w, requires_grad=False)

                for param_name in ["A_log", "dt_bias"]:
                    k = f"model.language_model.layers.{l_idx}.linear_attn.{param_name}"
                    if k in tensors:
                        info = tensors[k]
                        off = info["byte_offset"]
                        shape = info["shape"]
                        raw = mm[off : off + shape[0] * 4]
                        w = torch.frombuffer(bytearray(raw), dtype=torch.float32).to("cuda")
                        setattr(layer.linear_attn, param_name, nn.Parameter(w, requires_grad=False))

                k = f"model.language_model.layers.{l_idx}.linear_attn.norm.weight"
                if k in tensors:
                    info = tensors[k]
                    off = info["byte_offset"]
                    shape = info["shape"]
                    raw = mm[off : off + shape[0] * 2]
                    w = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).to(torch.float16).to("cuda")
                    layer.linear_attn.norm.weight = nn.Parameter(w, requires_grad=False)

        # 3. Final norm & lm_head
        k = "model.language_model.norm.weight"
        if k in tensors:
            info = tensors[k]
            off = info["byte_offset"]
            raw = mm[off : off + text_cfg.hidden_size * 2]
            w = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).to(torch.float16).to("cuda")
            model.model.norm = nn.RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)
            model.model.norm.weight = nn.Parameter(w, requires_grad=False)

        k = "lm_head.weight"
        if k in tensors:
            info = tensors[k]
            off = info["byte_offset"]
            shape = info["shape"]
            scale = info.get("scale", 1.0)
            raw = mm[off : off + shape[0] * shape[1]]
            w_q, s_q = quantize_bytes(raw, shape, mode=mode, scale=scale)
            model.lm_head = linear_cls(w_q, s_q)

    gc.collect()
    torch.cuda.empty_cache()
    vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"Loaded {mode.upper()} model into CUDA in {time.perf_counter() - t0:.2f} s | VRAM Allocated: {vram_gb:.3f} GB")
    return model


def run_benchmark(mode="4bit"):
    print("=" * 70)
    print(f"EXECUTING OFFICIAL HUGGING FACE LEADERBOARD BATTERY: {mode.upper()}")
    print("=" * 70)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained("chpe_models")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_quantized_qwen35(
        "chpe_models/Qwen3.5-9B-Base.q8.raw.chpe",
        "chpe_models/manifest.json",
        "chpe_models",
        mode=mode
    )

    lm_obj = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1
    )

    results_all = {
        "mode": mode,
        "hardware": "NVIDIA Tesla T4 (Lightning AI)",
        "vram_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 3),
        "tasks": {}
    }

    # 1. leaderboard_ifeval
    print("\n--- Running leaderboard_ifeval (limit=2) ---")
    t_step = time.perf_counter()
    res_ifeval = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=["leaderboard_ifeval"],
        limit=2,
        log_samples=True
    )
    print(f"ifeval completed in {time.perf_counter() - t_step:.2f} s")
    results_all["tasks"]["leaderboard_ifeval"] = res_ifeval.get("results", {})

    # 2. leaderboard_musr
    print("\n--- Running leaderboard_musr (limit=5) ---")
    t_step = time.perf_counter()
    res_musr = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=["leaderboard_musr"],
        limit=5,
        log_samples=True
    )
    print(f"musr completed in {time.perf_counter() - t_step:.2f} s")
    results_all["tasks"]["leaderboard_musr"] = res_musr.get("results", {})

    # 3. leaderboard_bbh
    print("\n--- Running leaderboard_bbh (limit=2) ---")
    t_step = time.perf_counter()
    res_bbh = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=["leaderboard_bbh"],
        limit=2,
        log_samples=True
    )
    print(f"bbh completed in {time.perf_counter() - t_step:.2f} s")
    results_all["tasks"]["leaderboard_bbh"] = res_bbh.get("results", {})

    # 4. leaderboard_mmlu_pro
    print("\n--- Running leaderboard_mmlu_pro (limit=5) ---")
    t_step = time.perf_counter()
    res_mmlu = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=["leaderboard_mmlu_pro"],
        limit=5,
        log_samples=True
    )
    print(f"mmlu_pro completed in {time.perf_counter() - t_step:.2f} s")
    results_all["tasks"]["leaderboard_mmlu_pro"] = res_mmlu.get("results", {})

    total_time = time.perf_counter() - t0
    results_all["total_time_seconds"] = total_time

    out_file = f"leaderboard_{mode}_results.json"
    with open(out_file, "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    print(f"\nSaved all results to {out_file} (Total time: {total_time:.2f} s)")

    print(f"\n=== SUMMARY FOR {mode.upper()} ON TESLA T4 ===")
    for task_name, task_res in results_all["tasks"].items():
        print(f"[{task_name}]")
        for k, v in task_res.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "4bit"
    run_benchmark(mode)
