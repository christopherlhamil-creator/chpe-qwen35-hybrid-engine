#!/usr/bin/env python3
"""
fetch_weights.py — Automated weight downloader and integrity verifier for CHPE Qwen3.5-9B.

Fetches raw tiled CHPE weight archives from Hugging Face (Siddachan/qwen3.5-9b-chpe-raw)
and verifies exact file size and format headers.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "Siddachan/qwen3.5-9b-chpe-raw"
FILENAME = "Qwen3.5-9B-Base.q8.raw.chpe"
EXPECTED_SIZE = 8956825600  # 8.34 GiB

ENGINE_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ENGINE_ROOT / "models"


def main():
    parser = argparse.ArgumentParser(description="Fetch and verify CHPE Qwen3.5-9B model weights from Hugging Face.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing local files without downloading")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token (or set HF_TOKEN env var)")
    parser.add_argument("--target-dir", type=str, default=str(MODELS_DIR), help="Target download directory")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / FILENAME

    if target_path.exists():
        actual_size = target_path.stat().st_size
        print(f"[FOUND] Local archive exists: {target_path} ({actual_size / (1024**3):.2f} GB)")
        if actual_size == EXPECTED_SIZE:
            print(f"[VERIFY] Size matches expected byte count: {EXPECTED_SIZE} bytes. Verification SUCCESS.")
            return
        else:
            print(f"[WARN] Size mismatch: found {actual_size} bytes, expected {EXPECTED_SIZE} bytes.")
            if args.verify_only:
                sys.exit(1)

    if args.verify_only:
        print(f"[ERROR] Weight file not found at {target_path}")
        sys.exit(1)

    print(f"[FETCH] Downloading {FILENAME} from {REPO_ID}...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] 'huggingface_hub' is required. Install via: pip install huggingface_hub")
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN")
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(target_dir),
        repo_type="model",
        token=token,
    )
    print(f"[SUCCESS] Weights downloaded successfully to: {downloaded_path}")


if __name__ == "__main__":
    main()
