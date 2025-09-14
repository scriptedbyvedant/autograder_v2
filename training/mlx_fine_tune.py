#!/usr/bin/env python3
"""
mlx_lora_finetune.py

One-shot MLX workflow on Apple Silicon:
- Convert HF model -> MLX (q4/q8)
- LoRA finetune on JSONL {"text": "..."}
- Optional: test generation
- Optional: fuse LoRA adapters

USAGE:
  python mlx_lora_finetune.py \
    --hf-model microsoft/phi-2 \
    --data ./data/train.jsonl \
    --workdir ./runs/phi2_run \
    --qbits 4 \
    --seq-len 512 \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 8 \
    --lora-r 64 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --test-prompt "### Instruction:\nSay hi.\n\n### Response:\n" \
    --fuse

Notes:
- Ensure: pip install -U mlx mlx-lm datasets tokenizers (and optionally tiktoken)
- JSONL must contain a "text" field per line.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

def run(cmd: str):
    print(f"\n$ {cmd}")
    # Using shell=True for this script as it involves complex commands
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"ERROR: Command failed with exit code {proc.returncode}")
        raise SystemExit(proc.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", required=True, help="Hugging Face model id or local path (e.g., microsoft/phi-2)")
    ap.add_argument("--data", required=True, help="Path to JSONL train file with a 'text' field")
    ap.add_argument("--workdir", required=True, help="Base directory for outputs (models, lora, logs)")
    ap.add_argument("--qbits", type=int, default=4, choices=[4,8], help="Quantization bits for MLX conversion")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--test-prompt", type=str, default="", help="If provided, run a quick generation after training")
    ap.add_argument("--fuse", action="store_true", help="If set, fuse LoRA adapters into the base model at the end")
    args = ap.parse_args()

    # Paths
    workdir = Path(args.workdir).resolve()
    models_dir = workdir / "models"
    lora_dir = workdir / "lora"
    logs_dir = workdir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    lora_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Verify dataset
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        sys.exit(1)
    with data_path.open() as f:
        try:
            obj = json.loads(f.readline())
        except (json.JSONDecodeError, StopIteration):
            obj = {}
    if "text" not in obj:
        print("ERROR: JSONL must contain a 'text' field per line.")
        sys.exit(1)

    # Model paths
    mlx_model_path = models_dir / f"mlx_model_q{args.qbits}"
    lora_out_path  = lora_dir / "adapters"
    fused_out_path = models_dir / f"mlx_model_q{args.qbits}_merged"

    # Constructing commands using lists of arguments is safer
    def build_and_run(cmd_parts):
        # Use shlex.quote for safety on each part, then join
        # This is safer than just passing a raw string to shell=True
        cmd = ' '.join([shlex.quote(part) for part in cmd_parts])
        run(cmd)

    # 1) Convert HF -> MLX (quantized)
    if not mlx_model_path.exists():
        cmd_convert = [
            sys.executable, "-m", "mlx_lm.convert",
            "--hf-path", args.hf_model,
            "--mlx-path", str(mlx_model_path),
            "-q",
            "--qbits", str(args.qbits),
        ]
        build_and_run(cmd_convert)
    else:
        print(f"Skip convert: {mlx_model_path} already exists.")

    # 2) LoRA finetune
    cmd_ft = [
        sys.executable, "-m", "mlx_lm.finetune",
        "--model", str(mlx_model_path),
        "--train", str(data_path),
        "--lora",
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation", str(args.grad_accum),
        "--epochs", str(args.epochs),
        "--seq-len", str(args.seq_len),
        "--lr", str(args.lr),
        "--output-dir", str(lora_out_path),
    ]
    build_and_run(cmd_ft)

    # 3) Optional quick generation
    if args.test_prompt:
        cmd_gen = [
            sys.executable, "-m", "mlx_lm.generate",
            "--model", str(mlx_model_path),
            "--lora-path", str(lora_out_path),
            "--prompt", args.test_prompt,
            "--max-tokens", "128",
            "--temp", "0.7",
            "--top-p", "0.95",
        ]
        build_and_run(cmd_gen)

    # 4) Optional fuse
    if args.fuse:
        cmd_fuse = [
            sys.executable, "-m", "mlx_lm.fuse_lora",
            "--model", str(mlx_model_path),
            "--lora-path", str(lora_out_path),
            "--save-path", str(fused_out_path),
        ]
        build_and_run(cmd_fuse)
        print(f"Fused model saved to: {fused_out_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
