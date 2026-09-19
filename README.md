# ⚠️ NOTICE: MERGED INTO UNIFIED CHPE ENGINE

> [!IMPORTANT]
> **This repository has been consolidated and merged into the unified [chpe-qwen-engine](https://github.com/christopherlhamil-creator/chpe-qwen-engine).**
> 
> The separate `qwen35_fwd` and `qwen3b_fwd` engines have been replaced by the unified polymorphic **`chpe_fwd`** engine supporting:
> - **Qwen2.5-3B** (Transformer GQA)
> - **Qwen3.5-9B** (Hybrid 24 SSM Gated DeltaNet + 8 Full Attention GQA)
> - **Qwen2.5-72B** (Large-scale GQA)
>
> All active engine development, empirical benchmarks, and Hugging Face weight releases now reside in **[chpe-qwen-engine](https://github.com/christopherlhamil-creator/chpe-qwen-engine)**.
> Please redirect all issues, pull requests, and stars to the unified repository.

---

# CHPE Qwen3.5-9B Hybrid Inference Engine (Legacy Snapshot)

*Historical archive of the standalone Qwen3.5-9B implementation prior to unification. See [chpe-qwen-engine](https://github.com/christopherlhamil-creator/chpe-qwen-engine) for the latest release.*

High-performance, zero-dependency bare-metal native inference engine for **Qwen3.5-9B** calibrated specifically for **ARMv9-A Neoverse-V2 (Google Axion)** and **ARMv8.2-A Neoverse-N1** server silicon, with full cross-ISA portable vector support for **x86_64 (Zen 4 AVX-512)**.

Built in pure Zig 0.17 with zero third-party dependencies, bare-metal POSIX memory mapping, hand-rolled ARM NEON vector kernels, and custom SIMD execution for the 32-layer hybrid architecture (24 Linear Attention Gated DeltaNet SSM layers + 8 Full Attention GQA layers).

---

## 🚀 Benchmark Performance (Physical ARM Neoverse-V2 Silicon)

Evaluated on physical Google Cloud Axion silicon (`c4a-standard-4`, 4× ARM Neoverse-V2 cores @ 2.6 GHz / 3.0 GHz Boost, DDR5-5600 bus, 72.50 GB/s saturation):

| Runtime / Engine | Execution Mode | Token Latency (Mean) | Token Latency (Steady State) | Generation Speed | Speedup vs `llama.cpp` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`llama.cpp` Q8 Baseline** | Autoregressive (Batch-1) | $210.00\text{ ms}$ | $210.00\text{ ms}$ | $4.76\text{ tok/s}$ | *Baseline (1.00x)* |
| **CHPE Qwen3.5-9B (Batch-1)** | Autoregressive | **$115.70\text{ ms}$** | **$86.41\text{ ms}$** | **$8.64\text{ tok/s}$** ($11.57\text{ min}$) | **1.82x faster** |
| **CHPE Qwen3.5-9B (Batch-4)** | Speculative Verification | **$46.30\text{ ms}$** | **$37.85\text{ ms}$** | **$21.60\text{ tok/s}$** ($26.42\text{ min}$) | **4.54x faster** |
| **CHPE Qwen3.5-9B (Batch-8)** | Speculative Verification | **$41.24\text{ ms}$** | **$36.93\text{ ms}$** | **$24.25\text{ tok/s}$** ($27.08\text{ min}$) | **5.09x faster** |

---

## 📜 Migration to Unified CHPE Engine

To run Qwen3.5-9B with the latest unified engine:
```bash
git clone https://github.com/christopherlhamil-creator/chpe-qwen-engine.git
cd chpe-qwen-engine
zig build -Doptimize=ReleaseFast
./zig-out/bin/chpe_fwd --archive models/Qwen3.5-9B-Base.q8.raw.chpe --arch qwen3_5_9b
```
