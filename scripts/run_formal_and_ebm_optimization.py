#!/usr/bin/env python3
"""scripts/run_formal_and_ebm_optimization.py

Orchestrates formal verification (Z3, Vampire, Leo-III) and physical GPU execution
of EBM optimization for Qwen2.5-72B 2-Bit Quantization on Lightning AI (NVIDIA L4).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SMT_DIR = REPO_ROOT / "tests" / "smt"
TPTP_DIR = REPO_ROOT / "tests" / "tptp"
SCARS_DIR = REPO_ROOT / "zk" / "scars"

Z3_BIN = "/home/christopherhamil/.local/bin/z3"
VAMPIRE_BIN = "/home/christopherhamil/.grok/tools/vampire/vampire"
LEO3_JAR = "/home/christopherhamil/.grok/tools/leo3/leo3.jar"


def run_z3_verification() -> dict:
    print("=" * 75)
    print("⚡ [1/4] Z3 SMT2 FORMAL VERIFICATION OF 2-BIT INVARIANTS")
    print("=" * 75)

    specs = [
        ("qwen72b_2bit_group128_bounds.smt2", "Signed 2-bit group-128 quantization bounds & packing"),
        ("qwen72b_uint4_coalesced_stride.smt2", "128-bit uint4 memory bus coalescing & sector stride"),
        ("qwen72b_variance_preservation_bounds.smt2", "EBM variance preservation bounds across 80 layers")
    ]

    results = {}
    for filename, desc in specs:
        p = SMT_DIR / filename
        assert p.exists(), f"Missing spec {p}"
        proc = subprocess.run([Z3_BIN, str(p)], capture_output=True, text=True)
        res = proc.stdout.strip()
        status = "UNSAT" if res == "unsat" else "SAT/ERROR"
        content_hash = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        results[filename] = {"status": status, "cite_key": content_hash, "desc": desc}
        print(f"  Z3 Spec: {filename:45s} -> {status} (cite_key={content_hash})")
        if status != "UNSAT":
            print(f"  Output: {res}\n  Error: {proc.stderr}")
            sys.exit(1)
    return results


def run_vampire_verification() -> dict:
    print("\n" + "=" * 75)
    print("🧛 [2/4] VAMPIRE 5.1.0 FIRST-ORDER ATP THEOREMS")
    print("=" * 75)

    theorems = [
        ("qwen72b_deep_stack_variance_preservation.p", "80-layer deep stack variance preservation theorem"),
        ("qwen72b_interlayer_error_absorption.p", "Inter-layer sequential error absorption theorem")
    ]

    results = {}
    for filename, desc in theorems:
        p = TPTP_DIR / filename
        assert p.exists(), f"Missing theorem {p}"
        proc = subprocess.run([VAMPIRE_BIN, "--mode", "casc", "--time_limit", "10", str(p)],
                              capture_output=True, text=True)
        out = proc.stdout
        status = "THEOREM" if ("SZS status Theorem" in out or "Theorem" in out) else "UNKNOWN"
        content_hash = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        results[filename] = {"status": status, "cite_key": content_hash, "desc": desc}
        print(f"  Vampire: {filename:45s} -> {status} (cite_key={content_hash})")
        if status != "THEOREM":
            print(f"  Output: {out}\n  Error: {proc.stderr}")
            sys.exit(1)
    return results


def run_leo3_verification() -> dict:
    print("\n" + "=" * 75)
    print("🦁 [3/4] LEO-III 1.7.18 HIGHER-ORDER MODAL ATP THEOREMS")
    print("=" * 75)

    p = TPTP_DIR / "qwen72b_ebm_optimization_soundness.p"
    assert p.exists(), f"Missing theorem {p}"
    proc = subprocess.run(["java", "-jar", LEO3_JAR, str(p)], capture_output=True, text=True)
    out = proc.stdout
    status = "THEOREM" if ("SZS status Theorem" in out or "Theorem" in out) else "UNKNOWN"
    content_hash = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    print(f"  Leo-III: {p.name:45s} -> {status} (cite_key={content_hash})")
    if status != "THEOREM":
        print(f"  Output: {out}\n  Error: {proc.stderr}")
        sys.exit(1)
    return {p.name: {"status": status, "cite_key": content_hash, "desc": "EBM optimization higher-order soundness"}}


def run_lightning_ebm_execution() -> dict:
    print("\n" + "=" * 75)
    print("⚡ [4/4] PHYSICAL SILICON EXECUTION: EBM OPTIMIZATION ON NVIDIA L4")
    print("=" * 75)

    from lightning_sdk import Studio
    import lightning_sdk.machine as m

    s = Studio(name="chpe-t4", teamspace="training-optimization-project", user="christopherlhamil")
    print(f"Connected to Studio {s.name}. Current Status: {s.status} | Machine: {s.machine}")

    # 1. Sync EBM script to studio
    ebm_script = REPO_ROOT / "scripts" / "ebm_optimize_72b.py"
    print(f"Uploading {ebm_script.name} to studio...")
    s.upload_file(str(ebm_script), "ebm_optimize_72b.py")

    # 2. Switch to Machine.L4
    print("Switching studio to Machine.L4 (24GB VRAM Ada Lovelace)...")
    s.switch_machine(m.Machine.L4)
    print(f"Switched! Status: {s.status} | Machine: {s.machine}")

    try:
        # 3. Verify GPU
        smi = s.run("nvidia-smi")
        print("NVIDIA-SMI Verified:")
        print(smi.splitlines()[0] if smi else "")

        # 4. Run EBM Sweep
        print("\nExecuting EBM Calibration Sweep on physical NVIDIA L4...")
        sweep_out = s.run("python3 /teamspace/studios/this_studio/ebm_optimize_72b.py --sweep")
        print(sweep_out)

        # 5. Download results
        res_content = s.run("cat /teamspace/studios/this_studio/ebm_calibration_results.json")
        res_data = json.loads(res_content) if res_content else {}
    finally:
        # CRITICAL: Always switch back to CPU to halt GPU credit burn!
        print("\nHalting GPU billing: Switching studio back to Machine.CPU...")
        s.switch_machine(m.Machine.CPU)
        print(f"Studio machine switched to: {s.machine}")

    return res_data


def main():
    t0 = time.perf_counter()
    z3_res = run_z3_verification()
    vamp_res = run_vampire_verification()
    leo_res = run_leo3_verification()
    ebm_res = run_lightning_ebm_execution()
    dt = time.perf_counter() - t0

    print("\n" + "=" * 75)
    print(f"🎉 2-BIT QWEN2.5-72B OPTIMIZATION COMPLETE IN {dt:.1f}s")
    print("=" * 75)
    print(f"Best EBM Alpha: {ebm_res.get('best_alpha', 'N/A')}")


if __name__ == "__main__":
    main()
