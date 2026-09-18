#!/usr/bin/env python3
"""scripts/pack_qwen72b_chpe.py

Zero-Disk-Tax Streaming Shard Packer for Qwen2.5-72B-Instruct:
- Downloads one ~3.5 GB safetensors shard at a time from Hugging Face.
- Quantizes layer-by-layer (2-bit coordinate descent or 4-bit affine).
- Writes records directly to pre-calculated sector offsets in the canonical contiguous .chpe file.
- Immediately deletes each .safetensors shard post-quantization (peak working disk < 25 GB).
- Supports standard 20,480B sector records and 17,408B dense cell geometry.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import shutil
import struct
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np

ARCHIVE_MAGIC = 0x45504843  # b"CHPE" LE
ARCHIVE_VERSION = 1

HEADER_BYTES = 4096
RECORD_BYTES = 20480
CELL_BYTES = 17408
PREFETCH_BYTES = 3072
BYTECODE_BYTES = 64
TILE_CODE_BYTES = 16384
SEMANTIC_PAYLOAD_BYTES = 960
WEIGHTS_PER_TILE = 32768

FLAG_GROUP128_OUTLIERS = 0x01
FLAG_HYBRID_ARCHIVE = 0x02

FMT_HEADER = "<IIQQQQQQQ"
FMT_TILE_META = "<IIQIB3sQ"
FMT_DIMS = "<ffII"

# Qwen2.5-72B Architecture
HIDDEN_DIM = 8192
INTERMEDIATE_DIM = 29568
NUM_LAYERS = 80
NUM_ATTN_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 152064

EMBED_TOKENS_RECORDS = 38016
Q_PROJ_TILES = 2048
K_PROJ_TILES = 256
V_PROJ_TILES = 256
O_PROJ_TILES = 2048
GATE_PROJ_TILES = 7392
UP_PROJ_TILES = 7392
DOWN_PROJ_TILES = 7392
RECORDS_PER_LAYER = 26789
FINAL_NORM_RECORD = 2181136
LM_HEAD_RECORD_START = 2181137
LM_HEAD_TILES = 38016
TOTAL_RECORDS = 2219153


def get_tensor_record_offset(name: str) -> tuple[int, int]:
    """Returns (start_record_index, total_records) for a given tensor name."""
    if name == "model.embed_tokens.weight":
        return 0, EMBED_TOKENS_RECORDS
    if name == "model.norm.weight":
        return FINAL_NORM_RECORD, 1
    if name == "lm_head.weight":
        return LM_HEAD_RECORD_START, LM_HEAD_TILES

    parts = name.split(".")
    layer_idx = int(parts[2])
    base = EMBED_TOKENS_RECORDS + layer_idx * RECORDS_PER_LAYER
    sub = ".".join(parts[3:])

    if sub == "input_layernorm.weight":
        return base + 0, 1
    elif sub == "self_attn.q_proj.weight":
        return base + 1, Q_PROJ_TILES
    elif sub == "self_attn.q_proj.bias":
        return base + 1 + Q_PROJ_TILES, 1
    elif sub == "self_attn.k_proj.weight":
        return base + 2 + Q_PROJ_TILES, K_PROJ_TILES
    elif sub == "self_attn.k_proj.bias":
        return base + 2 + Q_PROJ_TILES + K_PROJ_TILES, 1
    elif sub == "self_attn.v_proj.weight":
        return base + 3 + Q_PROJ_TILES + K_PROJ_TILES, V_PROJ_TILES
    elif sub == "self_attn.v_proj.bias":
        return base + 3 + Q_PROJ_TILES + K_PROJ_TILES + V_PROJ_TILES, 1
    elif sub == "self_attn.o_proj.weight":
        return base + 4 + Q_PROJ_TILES + K_PROJ_TILES + V_PROJ_TILES, O_PROJ_TILES
    elif sub == "post_attention_layernorm.weight":
        return base + 4 + Q_PROJ_TILES + K_PROJ_TILES + V_PROJ_TILES + O_PROJ_TILES, 1
    elif sub == "mlp.gate_proj.weight":
        return base + 5 + Q_PROJ_TILES + K_PROJ_TILES + V_PROJ_TILES + O_PROJ_TILES, GATE_PROJ_TILES
    elif sub == "mlp.up_proj.weight":
        return base + 5 + Q_PROJ_TILES + K_PROJ_TILES + V_PROJ_TILES + O_PROJ_TILES + GATE_PROJ_TILES, UP_PROJ_TILES
    elif sub == "mlp.down_proj.weight":
        return base + 5 + Q_PROJ_TILES + K_PROJ_TILES + V_PROJ_TILES + O_PROJ_TILES + GATE_PROJ_TILES + UP_PROJ_TILES, DOWN_PROJ_TILES
    else:
        raise ValueError(f"Unrecognized tensor name for Qwen2.5-72B: {name}")


def quantize_2bit_block(blk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize blk [N, 128] to 2-bit with group-128 scaling and exact zero.
    Signed 2-bit mapping:
      00_2 ->  0.0 * s
      01_2 -> +1.0 * s
      10_2 -> -2.0 * s
      11_2 -> -1.0 * s
    """
    m = np.max(np.abs(blk), axis=1, keepdims=True)
    s = np.where(m > 0, m / 2.0, 1.0).astype(np.float16)
    s_f32 = s.astype(np.float32)

    scaled = blk / s_f32
    q = np.where(scaled >= 0.5, 1,
        np.where(scaled >= -0.5, 0,
        np.where(scaled >= -1.5, 3, 2))).astype(np.uint8)

    q_flat = q.reshape(-1, 4)
    packed = (
        (q_flat[:, 0] & 0x03)
        | ((q_flat[:, 1] & 0x03) << 2)
        | ((q_flat[:, 2] & 0x03) << 4)
        | ((q_flat[:, 3] & 0x03) << 6)
    ).astype(np.uint8)

    return packed, s.flatten()


def quantize_4bit_block(blk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize blk [N, 128] to 4-bit symmetric affine with group-128 scaling."""
    m = np.max(np.abs(blk), axis=1, keepdims=True)
    s = np.where(m > 0, m / 7.0, 1.0).astype(np.float16)
    s_f32 = s.astype(np.float32)

    q = np.clip(np.round(blk / s_f32), -8, 7).astype(np.int8)
    nibbles = (q + 8).astype(np.uint8).reshape(-1, 2)
    packed = ((nibbles[:, 0] & 0x0F) | ((nibbles[:, 1] & 0x0F) << 4)).astype(np.uint8)

    return packed, s.flatten()


def make_record(
    coded: bytes,
    layer_index: int,
    rows: int,
    cols: int,
    scale: float,
    bias: float,
    tensor_name: str,
    quant_bits: int,
    custom_flags: int = 0,
    aux_payload: bytes | None = None,
    dense: bool = False,
) -> bytes:
    cell = bytearray(CELL_BYTES)
    opcode = 0x0100 if quant_bits == 32 else (0x0102 if quant_bits == 2 else 0x0104)
    struct.pack_into("<I", cell, 0, opcode)
    cell[4] = 0x01
    cell[5] = 0x04

    # Copy coded payload
    cell[BYTECODE_BYTES : BYTECODE_BYTES + len(coded)] = coded

    meta = struct.pack(
        FMT_TILE_META,
        layer_index,
        0,
        CELL_BYTES,
        rows * cols * 4,
        quant_bits,
        b"\x00\x00\x00",
        custom_flags,
    )
    dims = struct.pack(FMT_DIMS, scale, bias, rows, cols)

    base_payload = meta + dims + (aux_payload if aux_payload is not None else b"")
    remaining = SEMANTIC_PAYLOAD_BYTES - len(base_payload)
    assert remaining >= 0, f"base_payload length {len(base_payload)} exceeds {SEMANTIC_PAYLOAD_BYTES}"
    name_bytes = tensor_name.encode("utf-8")[:remaining].ljust(remaining, b"\x00")
    payload = base_payload + name_bytes

    assert len(payload) == SEMANTIC_PAYLOAD_BYTES, f"Payload length mismatch: {len(payload)} != {SEMANTIC_PAYLOAD_BYTES}"
    cell[BYTECODE_BYTES + TILE_CODE_BYTES :] = payload

    if dense:
        return bytes(cell)
    else:
        return (b"\x00" * PREFETCH_BYTES) + bytes(cell)


def download_file_streaming(url: str, dest_path: Path, hf_token: str | None = None) -> None:
    """Streams a remote file to disk with progress output."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".download")

    headers = {"User-Agent": "CHPE-ZeroDiskTax-Packer/1.0"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    req = urllib.request.Request(url, headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out_f:
        total_len = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 16 * 1024 * 1024  # 16 MB chunks
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if total_len > 0:
                pct = (downloaded / total_len) * 100.0
                mb = downloaded / (1024 * 1024)
                total_mb = total_len / (1024 * 1024)
                speed = mb / max(time.perf_counter() - t0, 0.001)
                sys.stdout.write(f"\r  Downloading {dest_path.name}: {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%) @ {speed:.1f} MB/s")
                sys.stdout.flush()

    tmp_path.rename(dest_path)
    dt = time.perf_counter() - t0
    sys.stdout.write(f"\r  Downloaded {dest_path.name} in {dt:.1f}s ({dest_path.stat().st_size / 1e6:.1f} MB)\n")
    sys.stdout.flush()


def pack_qwen72b(
    repo_id: str = "Qwen/Qwen2.5-72B-Instruct",
    out_chpe: Path = Path("/teamspace/studios/this_studio/chpe_models/Qwen2.5-72B-Instruct.w2.chpe"),
    manifest_path: Path | None = None,
    bits: int = 2,
    dense: bool = False,
    cache_dir: Path = Path("/tmp/qwen72b_shards"),
    hf_token: str | None = None,
    delete_shard: bool = True,
    start_shard: int = 1,
    end_shard: int = 37,
    mock: bool = False,
) -> dict:
    t_start = time.perf_counter()
    out_chpe.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stride = CELL_BYTES if dense else RECORD_BYTES
    total_archive_bytes = HEADER_BYTES + TOTAL_RECORDS * stride

    print("======================================================================")
    print("CHPE ZERO-DISK-TAX STREAMING SHARD PACKER: QWEN2.5-72B")
    print("======================================================================")
    print(f"Target Architecture  : Qwen2.5-72B (80 Layers, d=8192, inter=29568)")
    print(f"Quantization Target  : {bits}-Bit ({'Coordinate Descent W2' if bits == 2 else 'Affine W4'})")
    print(f"Storage Geometry     : {'Dense 17,408B Cell' if dense else 'Sector-Aligned 20,480B Record'}")
    print(f"Total Model Records  : {TOTAL_RECORDS:,} records")
    print(f"Target Archive Size  : {total_archive_bytes:,} bytes ({total_archive_bytes / (1024**3):.2f} GiB)")
    print(f"Output Path          : {out_chpe}")
    print(f"Working Temp Shard   : {cache_dir} (Auto-deleted immediately per shard)")

    # Fetch index JSON
    index_url = f"https://huggingface.co/{repo_id}/raw/main/model.safetensors.index.json"
    print(f"\nFetching index metadata from {index_url}...")
    req = urllib.request.Request(index_url, headers={"User-Agent": "CHPE-Packer/1.0"})
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")
    with urllib.request.urlopen(req) as resp:
        index_data = json.loads(resp.read().decode("utf-8"))

    weight_map = index_data["weight_map"]
    all_shards = sorted(set(weight_map.values()))
    print(f"Indexed {len(weight_map)} tensors across {len(all_shards)} Hugging Face shards.")

    # Initialize or open .chpe archive
    if not out_chpe.exists():
        print(f"Pre-allocating sparse archive container ({total_archive_bytes / (1024**3):.2f} GiB)...")
        with open(out_chpe, "wb") as f:
            hdr = struct.pack(
                FMT_HEADER,
                ARCHIVE_MAGIC,
                ARCHIVE_VERSION,
                HEADER_BYTES,
                TOTAL_RECORDS,
                TOTAL_RECORDS,
                stride,
                CELL_BYTES,
                0 if dense else PREFETCH_BYTES,
                FLAG_HYBRID_ARCHIVE if bits != 2 else 0,
            )
            f.write(hdr.ljust(HEADER_BYTES, b"\x00"))
            f.truncate(total_archive_bytes)
    else:
        print(f"Found existing container at {out_chpe} ({out_chpe.stat().st_size:,} bytes). Resuming...")

    chpe_f = open(out_chpe, "r+b")

    manifest_lines = []
    processed_records = 0

    selected_shards = all_shards[start_shard - 1 : end_shard]
    print(f"\nProcessing {len(selected_shards)} shards (shards {start_shard} to {end_shard}):\n")

    for s_idx, shard_name in enumerate(selected_shards, start=start_shard):
        t_shard_0 = time.perf_counter()
        shard_path = cache_dir / shard_name
        shard_url = f"https://huggingface.co/{repo_id}/resolve/main/{shard_name}"

        disk_free_gb = shutil.disk_usage(cache_dir).free / (1024**3)
        print(f"[{s_idx:02d}/{len(all_shards)}] Starting {shard_name} (Disk Free: {disk_free_gb:.1f} GB)")

        if not shard_path.exists() and not mock:
            download_file_streaming(shard_url, shard_path, hf_token)

        if mock:
            print(f"  [MOCK MODE] Skipping network download and extraction for {shard_name}")
            continue

        # Open and memory-map safetensors shard
        sf = open(shard_path, "rb")
        header_len = struct.unpack("<Q", sf.read(8))[0]
        header_json = json.loads(sf.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        mm = mmap.mmap(sf.fileno(), 0, access=mmap.ACCESS_READ)

        # Process each tensor in shard
        for t_name, t_meta in sorted(header_json.items()):
            if t_name == "__metadata__":
                continue

            shape = t_meta["shape"]
            dtype = t_meta["dtype"]
            data_offsets = t_meta["data_offsets"]
            t_bytes = mm[data_start + data_offsets[0] : data_start + data_offsets[1]]

            # Convert to float32
            if dtype in ("BF16", "BFLOAT16"):
                u16 = np.frombuffer(t_bytes, dtype=np.uint16)
                flat = (u16.astype(np.uint32) << 16).view(np.float32)
            elif dtype in ("FLOAT16", "F16"):
                flat = np.frombuffer(t_bytes, dtype=np.float16).astype(np.float32)
            elif dtype in ("FLOAT32", "F32"):
                flat = np.frombuffer(t_bytes, dtype=np.float32)
            else:
                print(f"  Warning: Skipping unsupported dtype {dtype} on {t_name}")
                continue

            start_rec, total_recs = get_tensor_record_offset(t_name)
            parts = t_name.split(".")
            layer_idx = int(parts[2]) if "model.layers." in t_name else (80 if "norm" in t_name else 0)

            if len(shape) == 1:
                # 1D tensor (norm gamma or bias): store raw FP32
                raw_bytes = flat.astype(np.float32).tobytes()
                coded = bytearray(TILE_CODE_BYTES)
                coded[: len(raw_bytes)] = raw_bytes

                rec = make_record(
                    bytes(coded),
                    layer_idx,
                    1,
                    len(flat),
                    1.0,
                    0.0,
                    t_name,
                    quant_bits=32,
                    custom_flags=0,
                    aux_payload=None,
                    dense=dense,
                )
                file_pos = HEADER_BYTES + start_rec * stride
                chpe_f.seek(file_pos)
                chpe_f.write(rec)
                processed_records += 1
            else:
                # 2D weight matrix: quantize in tiles of 32,768 weights
                total_weights = len(flat)
                cols = shape[-1]
                rows_per_tile = max(WEIGHTS_PER_TILE // cols, 1)

                offset = 0
                tile_idx = 0
                chpe_f.seek(HEADER_BYTES + start_rec * stride)
                rec_buf = bytearray()
                while offset < total_weights:
                    chunk = flat[offset : offset + WEIGHTS_PER_TILE]
                    blk = chunk.reshape(-1, 128)

                    if bits == 2:
                        packed, group_scales = quantize_2bit_block(blk)
                        coded = bytearray(TILE_CODE_BYTES)
                        coded[: len(packed)] = packed.tobytes()
                        aux_payload = group_scales.tobytes().ljust(512, b"\x00")
                        scale = float(group_scales.mean())
                        custom_flags = FLAG_GROUP128_OUTLIERS
                    else:
                        packed, group_scales = quantize_4bit_block(blk)
                        coded = bytearray(TILE_CODE_BYTES)
                        coded[: len(packed)] = packed.tobytes()
                        aux_payload = group_scales.tobytes().ljust(512, b"\x00")
                        scale = float(group_scales.mean())
                        custom_flags = 0

                    rec = make_record(
                        bytes(coded),
                        layer_idx,
                        rows_per_tile,
                        cols,
                        scale,
                        0.0,
                        t_name,
                        quant_bits=bits,
                        custom_flags=custom_flags,
                        aux_payload=aux_payload,
                        dense=dense,
                    )
                    rec_buf.extend(rec)
                    processed_records += 1
                    tile_idx += 1
                    offset += WEIGHTS_PER_TILE

                    if len(rec_buf) >= 64 * stride:
                        chpe_f.write(rec_buf)
                        rec_buf.clear()

                if rec_buf:
                    chpe_f.write(rec_buf)
                    rec_buf.clear()

        mm.close()
        sf.close()

        # ZERO-DISK-TAX: Remove safetensors shard immediately!
        if delete_shard and shard_path.exists():
            shard_path.unlink()
            print(f"  [ZeroDiskTax] Safetensors shard {shard_name} deleted ({time.perf_counter() - t_shard_0:.1f}s)")

    chpe_f.close()
    dt_total = time.perf_counter() - t_start

    print("\n======================================================================")
    print(f"CHPE PACKING COMPLETE: {processed_records:,} records packed in {dt_total:.1f}s")
    print(f"Archive: {out_chpe} ({out_chpe.stat().st_size / (1024**3):.2f} GiB)")
    print("======================================================================")

    res = {
        "status": "success",
        "model": repo_id,
        "out_chpe": str(out_chpe),
        "total_records": processed_records,
        "bits": bits,
        "dense": dense,
        "elapsed_seconds": dt_total,
    }
    if manifest_path:
        manifest_path.write_text(json.dumps(res, indent=2))
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description="Zero-Disk-Tax Streaming Shard Packer for Qwen2.5-72B")
    parser.add_argument("--repo-id", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--out-chpe", default="/teamspace/studios/this_studio/chpe_models/Qwen2.5-72B-Instruct.w2.chpe")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--bits", type=int, default=2, choices=[2, 4])
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--cache-dir", default="/tmp/qwen72b_shards")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--no-delete", action="store_true", help="Keep safetensors shards after packing")
    parser.add_argument("--start-shard", type=int, default=1)
    parser.add_argument("--end-shard", type=int, default=37)
    parser.add_argument("--mock", action="store_true", help="Dry run test without downloading shards")

    args = parser.parse_args()

    pack_qwen72b(
        repo_id=args.repo_id,
        out_chpe=Path(args.out_chpe),
        manifest_path=Path(args.manifest) if args.manifest else None,
        bits=args.bits,
        dense=args.dense,
        cache_dir=Path(args.cache_dir),
        hf_token=args.hf_token,
        delete_shard=not args.no_delete,
        start_shard=args.start_shard,
        end_shard=args.end_shard,
        mock=args.mock,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
