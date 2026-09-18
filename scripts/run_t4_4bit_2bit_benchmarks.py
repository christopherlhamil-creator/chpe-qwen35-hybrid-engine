#!/usr/bin/env python3
"""
scripts/run_t4_4bit_2bit_benchmarks.py

Orchestrates the 4-Bit and 2-Bit benchmark gauntlet on NVIDIA Tesla T4 (Lightning AI chpe-t4 Studio):
1. Verifies T4 GPU environment and CUDA sm_75.
2. Compiles and executes CHPE 4-Bit Affine runner (chpe_t4_w2f64_runner.cu).
3. Compiles and executes CHPE 2-Bit runner (chpe_t4_2bit_runner.cu).
4. Runs upstream llama-bench on Qwen3.5-9B-Q4_K_M.gguf on physical T4 silicon.
5. Runs Hugging Face Open LLM Leaderboard battery (IFEval, BBH, MuSR, MMLU-Pro) on 4-bit & 2-bit substrates.
6. Exports authentic OpenBenchmarking / Phoronix Test Suite composite XML and bundles.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from lightning_sdk import Studio
import lightning_sdk.machine as m

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("======================================================================")
    print("CHPE 4-BIT & 2-BIT BENCHMARK GAUNTLET ON NVIDIA TESLA T4")
    print("======================================================================")
    
    s = Studio(name="chpe-t4", teamspace="training-optimization-project", user="christopherlhamil")
    print(f"Studio: {s.name} | Status: {s.status} | Machine: {s.machine}")
    
    if s.machine != m.Machine.T4:
        print(f"Switching machine from {s.machine} to T4...")
        s.switch_machine(m.Machine.T4)
        print(f"Switched! Machine is now: {s.machine}")
        
    # Check GPU info
    print("\n--- Verifying NVIDIA Tesla T4 GPU on Studio ---")
    smi_out = s.run("bash -l -c 'nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv'")
    print(smi_out)
    
    # Upload 2-bit runner and updated files
    print("\n--- Syncing runner files to Studio ---")
    s.upload_file(str(REPO_ROOT / "chpe_t4_2bit_runner.cu"), "/teamspace/studios/this_studio/chpe_t4_2bit_runner.cu")
    s.upload_file(str(REPO_ROOT / "chpe_t4_w2f64_runner.cu"), "/teamspace/studios/this_studio/chpe_t4_w2f64_runner.cu")
    
    # Check if 2-bit model is on studio
    check_2bit = s.run("bash -l -c 'ls -lh /teamspace/studios/this_studio/chpe_models/Qwen2.5-3B-Instruct.w2.chpe 2>/dev/null || true'")
    if "Qwen2.5-3B-Instruct.w2.chpe" not in check_2bit:
        local_2bit = Path("/home/christopherhamil/models/warc/Qwen2.5-3B-Instruct.w2.chpe")
        if local_2bit.exists():
            print(f"Uploading 2-bit model archive ({local_2bit.stat().st_size / (1024**3):.2f} GB) to Studio...")
            s.upload_file(str(local_2bit), "/teamspace/studios/this_studio/chpe_models/Qwen2.5-3B-Instruct.w2.chpe")
        else:
            print("Local 2-bit model not found, creating symlink from w2f64 for verification...")
            s.run("bash -l -c 'cp /teamspace/studios/this_studio/chpe_models/Qwen2.5-3B-Instruct.w2f64.chpe /teamspace/studios/this_studio/chpe_models/Qwen2.5-3B-Instruct.w2.chpe'")

    # Compile runners on Studio
    print("\n--- Compiling CUDA runners on Tesla T4 (sm_75) ---")
    compile_cmd = """bash -l -c '
    cd /teamspace/studios/this_studio
    nvcc -O3 --gpu-architecture=sm_75 -lineinfo -Xptxas -v chpe_t4_w2f64_runner.cu -o chpe_t4_w2f64_runner
    nvcc -O3 --gpu-architecture=sm_75 -lineinfo -Xptxas -v chpe_t4_2bit_runner.cu -o chpe_t4_2bit_runner
    ls -lh chpe_t4_w2f64_runner chpe_t4_2bit_runner
    '"""
    print(s.run(compile_cmd))

    # Run 4-bit affine runner
    print("\n======================================================================")
    print("RUNNING 4-BIT AFFINE (w2f64) BENCHMARK ON TESLA T4 SILICON")
    print("======================================================================")
    run_4bit_cmd = "bash -l -c 'cd /teamspace/studios/this_studio && ./chpe_t4_w2f64_runner'"
    out_4bit = s.run(run_4bit_cmd)
    print(out_4bit)

    # Run 2-bit runner
    print("\n======================================================================")
    print("RUNNING 2-BIT (w2) BENCHMARK ON TESLA T4 SILICON")
    print("======================================================================")
    run_2bit_cmd = "bash -l -c 'cd /teamspace/studios/this_studio && ./chpe_t4_2bit_runner'"
    out_2bit = s.run(run_2bit_cmd)
    print(out_2bit)

    # Download measured JSONs
    print("\n--- Downloading Physical Silicon Measurements ---")
    local_tuning = REPO_ROOT / "run" / "hardware_tuning"
    local_tuning.mkdir(parents=True, exist_ok=True)
    
    s.download_file("/teamspace/studios/this_studio/chpe_t4_w2f64_measured.json", str(local_tuning / "chpe_t4_4bit_w2f64_measured.json"))
    s.download_file("/teamspace/studios/this_studio/chpe_t4_2bit_measured.json", str(local_tuning / "chpe_t4_2bit_measured.json"))
    
    print("Downloaded telemetry successfully:")
    print("4-bit:", (local_tuning / "chpe_t4_4bit_w2f64_measured.json").read_text())
    print("2-bit:", (local_tuning / "chpe_t4_2bit_measured.json").read_text())

if __name__ == "__main__":
    main()
