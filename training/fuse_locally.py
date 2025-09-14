#!/usr/bin/env python3
"""
Fuses the downloaded LoRA adapters with the local base model.

Instructions:
1. Ensure you have run the training in Colab and downloaded the 'trained_adapters.npz' file.
2. Place the downloaded file at 'training/mlx_runs/lora/adapters/adapters.npz'.
   (You may need to rename 'trained_adapters.npz' to 'adapters.npz').
3. Run this script from your terminal.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run(cmd: str):
    print(f"\n$ {cmd}")
    # Use shell=True for simplicity, capture output for debugging
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if proc.stdout:
        print("--- STDOUT ---")
        print(proc.stdout)
    if proc.stderr:
        print("--- STDERR ---")
        print(proc.stderr)
    if proc.returncode != 0:
        print(f"ERROR: Command failed with exit code {proc.returncode}")
        raise SystemExit(proc.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="training/mlx_runs", help="Base directory where models and adapters are stored")
    ap.add_argument("--qbits", type=int, default=4, help="Quantization of the base model")
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    models_dir = workdir / "models"
    lora_dir = workdir / "lora"

    # --- Define Paths ---
    base_model_path = models_dir / f"mlx_model_q{args.qbits}"
    adapter_path = lora_dir / "adapters"
    fused_model_path = models_dir / f"mlx_model_q{args.qbits}_fused"

    # --- Pre-flight Checks ---
    if not base_model_path.exists():
        print(f"ERROR: Base model not found at {base_model_path}. Make sure you have run the download/conversion process first.")
        sys.exit(1)
    
    # Important: The file from Colab is named 'trained_adapters.npz'. 
    # The fuse command expects 'adapters.npz'. We handle this by just pointing to the directory.
    downloaded_adapter_file = adapter_path / "trained_adapters.npz"
    expected_adapter_file = adapter_path / "adapters.npz"
    if not downloaded_adapter_file.exists() and not expected_adapter_file.exists():
        print(f"ERROR: No adapter file found! Please place your downloaded 'trained_adapters.npz' in the '{adapter_path}' directory.")
        sys.exit(1)
    
    # If the user downloaded it as trained_adapters.npz, rename it for the fuse script
    if downloaded_adapter_file.exists() and not expected_adapter_file.exists():
        print(f"Renaming '{downloaded_adapter_file.name}' to '{expected_adapter_file.name}' for compatibility.")
        downloaded_adapter_file.rename(expected_adapter_file)

    # --- Run Fuse Command ---
    print("Starting the model fusion process...")
    cmd_fuse = (
        f'"{sys.executable}" -m mlx_lm.fuse '
        f'--model "{base_model_path}" '
        f'--adapter-path "{adapter_path}" '
        f'--save-path "{fused_model_path}"'
    )
    run(cmd_fuse)

    print(f"\n--- Fusion Complete! ---")
    print(f"Your new, fine-tuned model has been saved to:")
    print(fused_model_path)

if __name__ == "__main__":
    main()
