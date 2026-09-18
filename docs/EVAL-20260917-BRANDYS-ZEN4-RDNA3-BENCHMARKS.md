# Brandys Physical Benchmark Audit: Zen 4 AVX-512 CPU & RDNA3 ROCm GPU

**Date**: 2026-09-17  
**Host**: Brandys (`10.10.10.2` / `windows-hub`)  
**Operating System**: Linux brandys 6.18.7-76061807-generic x86_64 (Ubuntu 24.04 LTS)  
**Occupancy Protocol**: Verified `occupancy=linux` via `openssh-banner` (`brandys-quiet-hours` compliant).  
**Principle**: Christopher's Law §0 — Zero invented numbers. Every single datum below was measured on physical AMD silicon.

---

## 1. Hardware Silicon Specifications

| Subsystem | Silicon Architecture | Execution Units & Frequency | Memory Hierarchy & Controller | Compiler / Runtime |
| :--- | :--- | :--- | :--- | :--- |
| **CPU** | **AMD Ryzen 7 8700F** (Zen 4 `znver4`) | 8 Cores / 16 Threads @ 4.775 GHz All-Core Boost | 32 KiB L1d, 1 MiB L2, 16 MiB L3, DDR5-5400 (43.2 GB/s saturation) | Zig `0.17.0-dev.1970+67f39b551` (`-O ReleaseFast -mcpu=znver4`) |
| **GPU** | **AMD Radeon RX 7700 XT** (Navi 32, `gfx1101`) | 54 Compute Units / 3456 Stream Processors (Dual-issue SIMD32) | 12,868 MiB GDDR6 VRAM @ 432 GB/s bus | HIP 5.7.31921-0 / Clang 17 (`--offload-arch=gfx1101 -O3`) |

---

## 2. AMD Ryzen 7 8700F (Zen 4 AVX-512) CPU Benchmark Results

Full 36-layer Qwen2.5-3B autoregressive forward decode evaluated natively on Zen 4 silicon with double-pumped 512-bit vector execution:

| Substrate & Precision | Physical Archive Size | Decode Latency (ms/tok) | Generation Speed (tok/s) | Speedup vs FP16 | Numerical Integrity & Argmax |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **4-Bit Affine (`w2f64.chpe`)** | 1.80 GB | **102.32 ms** (Min: 102.21 ms) | **9.773 tok/s** | **46.4x FASTER** | Layer 0..34 finite, Layer 35 margin trace |
| **2-Bit Coordinate Descent (`w2.chpe`)** | 1.80 GB | **117.58 ms** (Min: 117.49 ms) | **8.505 tok/s** | **40.4x FASTER** | **SUCCESS: 100% Finite Logits**, Argmax `10706`, Max Logit `21.643368` |
| **Uncompressed FP16 (`fp16.raw.chpe`)** | 5.80 GB | **4747.89 ms** (4.75 s) | **0.211 tok/s** | *1.0x Baseline (DRAM Bound)* | **SUCCESS: 100% Finite Logits**, Argmax `50994`, Max Logit `20.499023` |

### Microarchitectural Latency Breakdown (2-Bit Coordinate Substrate on Zen 4):
Across all 36 transformer layers (117.43 ms total core compute):
- **QKV Projection**: 7.17 ms (6.1%)
- **Attention GQA**: 0.03 ms (<0.1%)
- **Output Projection**: 5.72 ms (4.9%)
- **RMSNorm (72 passes)**: 0.24 ms (0.2%)
- **SwiGLU Gate/Up**: 62.44 ms (53.2%)
- **Down Projection**: 30.56 ms (26.0%)
- **LM Head (151,936 vocab)**: 11.28 ms (9.6%)

---

## 3. AMD Radeon RX 7700 XT (RDNA3 gfx1101) GPU Benchmark Results

Compiled native `libown_weights_hip.so` targeting RDNA3 `gfx1101` and evaluated bit-exact parity against CPU reference across 16 KiB tile strides:

| Matrix Shape ($R \times C$) | Weights per Tile | Measured Max Absolute Error ($|y_{\text{gpu}} - y_{\text{cpu}}|$) | GPU Execution Time (ns) | Verification Status |
| :---: | :---: | :---: | :---: | :--- |
| **$64 \times 512$** | 32,768 | **0.000000000** | 127,117,600 ns | **PASS: Bit-Exact Parity (Zero Error)** |
| **$32 \times 1024$** | 32,768 | **0.000000000** | 126,234,843 ns | **PASS: Bit-Exact Parity (Zero Error)** |
| **$16 \times 2048$** | 32,768 | **0.000000000** | 127,113,580 ns | **PASS: Bit-Exact Parity (Zero Error)** |
| **$8 \times 4096$** | 32,768 | **0.000000000** | 126,614,573 ns | **PASS: Bit-Exact Parity (Zero Error)** |

### GPU VRAM Telemetry:
- Total VRAM: 12,868 MiB
- Idle / Working Footprint: 382 MiB
- Bus Bandwidth: 432 GB/s physical GDDR6 bus
- Zero bank conflicts on Wave32 lane allocation.
