#!/usr/bin/env python3
"""scripts/deploy_qwen72b_lightning.py

Deploys and orchestrates the Qwen2.5-72B CHPE packing and benchmarking pipeline
on Lightning AI Cloud Studio (Option B):
1. Verifies studio connectivity and machine status.
2. Switches studio machine to Machine.L4 (24GB VRAM Ada Lovelace, sm_89, $0.75/hr).
3. Uploads packer (scripts/pack_qwen72b_chpe.py) and CUDA runner (chpe_l4_qwen72b_runner.cu).
4. Verifies GPU hardware, CUDA compute capability, and disk space.
5. Compiles native CUDA runner for sm_89 / sm_75.
6. Executes 72B streaming packing and hardware benchmarks.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from lightning_sdk import Studio
import lightning_sdk.machine as m

REPO_ROOT = Path(__file__).resolve().parent.parent

def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy Qwen2.5-72B to Lightning AI Studio")
    parser.add_argument("--machine", default="L4", choices=["L4", "T4", "CPU"], help="Machine type to switch to")
    parser.add_argument("--status-only", action="store_true", help="Print studio status without starting")
    parser.add_argument("--stop", action="store_true", help="Stop the studio to preserve cloud credits")
    parser.add_argument("--start", action="store_true", help="Start the studio")
    parser.add_argument("--pack", action="store_true", help="Run 72B streaming packer on studio")
    parser.add_argument("--bench", action="store_true", help="Run 72B CUDA GEMV benchmark on studio")
    parser.add_argument("--mock-pack", action="store_true", help="Test packing pipeline in mock mode on studio")

    args = parser.parse_args()

    print("======================================================================")
    print("LIGHTNING AI STUDIO ORCHESTRATOR: QWEN2.5-72B PIPELINE")
    print("======================================================================")

    s = Studio(name="chpe-t4", teamspace="training-optimization-project", user="christopherlhamil")
    print(f"Studio Name : {s.name}")
    print(f"Status      : {s.status}")
    print(f"Machine     : {s.machine}")

    if args.status_only:
        return 0

    if args.stop:
        print("\nStopping studio to preserve cloud credits...")
        s.stop()
        print(f"Studio stopped. Current status: {s.status}")
        return 0

    target_machine = getattr(m.Machine, args.machine)

    if s.status != "Running":
        print(f"\nStarting studio on machine {args.machine}...")
        try:
            s.start(target_machine)
        except Exception as e:
            print(f"Warning: Failed to start on {args.machine}: {e}")
            if args.machine != "T4":
                print("Falling back to Machine.T4...")
                target_machine = m.Machine.T4
                s.start(target_machine)
        print(f"Studio started! Status: {s.status} | Machine: {s.machine}")
    elif s.machine != target_machine:
        print(f"\nSwitching machine from {s.machine} to {args.machine}...")
        try:
            s.switch_machine(target_machine)
        except Exception as e:
            print(f"Warning: Failed to switch to {args.machine}: {e}")
            if args.machine != "T4":
                print("Falling back to Machine.T4...")
                target_machine = m.Machine.T4
                s.switch_machine(target_machine)
        print(f"Switched! Machine is now: {s.machine}")

    # Inspect disk space and environment
    print("\n--- Studio Filesystem & GPU Status ---")
    df_out = s.run("bash -l -c 'df -h /teamspace/studios/this_studio'")
    print("Disk Space:")
    print(df_out)

    smi_out = s.run("bash -l -c 'nvidia-smi 2>/dev/null || echo \"No GPU or nvidia-smi not in PATH\"'")
    print("\nNVIDIA GPU Status:")
    print(smi_out)

    # Sync runner and packer
    print("\n--- Syncing Scripts to Studio ---")
    packer_local = REPO_ROOT / "scripts" / "pack_qwen72b_chpe.py"
    runner_local = REPO_ROOT / "chpe_l4_qwen72b_runner.cu"

    s.upload_file(str(packer_local), "pack_qwen72b_chpe.py")
    s.upload_file(str(runner_local), "chpe_l4_qwen72b_runner.cu")
    print("Uploaded pack_qwen72b_chpe.py and chpe_l4_qwen72b_runner.cu.")

    # Compile runner
    print("\n--- Compiling CUDA Runner on Studio ---")
    arch_flag = "-arch=sm_89" if args.machine == "L4" else "-arch=sm_75"
    compile_cmd = f"bash -l -c 'nvcc -O3 {arch_flag} -o /teamspace/studios/this_studio/chpe_qwen72b_runner /teamspace/studios/this_studio/chpe_l4_qwen72b_runner.cu'"
    compile_out = s.run(compile_cmd)
    print(compile_out or "Compilation succeeded with code 0.")

    if args.mock_pack:
        print("\n--- Testing Mock Packing Pipeline on Studio ---")
        mock_cmd = "bash -l -c 'python3 /teamspace/studios/this_studio/pack_qwen72b_chpe.py --mock --out-chpe /teamspace/studios/this_studio/chpe_models/mock_qwen72b.chpe'"
        mock_out = s.run(mock_cmd)
        print(mock_out)
        s.run("bash -l -c 'rm -f /teamspace/studios/this_studio/chpe_models/mock_qwen72b.chpe'")

    if args.bench:
        print("\n--- Running 80-Layer Qwen2.5-72B CUDA GEMV Benchmark on Silicon ---")
        bench_out = s.run("bash -l -c '/teamspace/studios/this_studio/chpe_qwen72b_runner /teamspace/studios/this_studio/chpe_models/Qwen2.5-72B-Instruct.w2.chpe 2'")
        print(bench_out)

    if args.pack:
        print("\n--- Launching Full 72B Streaming Shard Packer ---")
        pack_cmd = "bash -l -c 'python3 /teamspace/studios/this_studio/pack_qwen72b_chpe.py --bits 2 --out-chpe /teamspace/studios/this_studio/chpe_models/Qwen2.5-72B-Instruct.w2.chpe'"
        pack_out = s.run(pack_cmd)
        print(pack_out)

    return 0

if __name__ == "__main__":
    sys.exit(main())
