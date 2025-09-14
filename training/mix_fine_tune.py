import argparse
import os
import json
import subprocess
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

def run_command(cmd_str):
    """Executes a command and streams its output."""
    print(f"\n$ {cmd_str}", flush=True)
    # Use shell=True to handle complex commands and paths with spaces
    process = subprocess.Popen(
        cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    if process.returncode != 0:
        print(f"\nERROR: Command failed with exit code {process.returncode}", flush=True)
        # Exit with the same code as the failed process
        sys.exit(process.returncode)

def prepare_dataset(data_jsonl, workdir):
    """Splits the JSONL file into train and validation sets."""
    dataset_dir = Path(workdir) / "dataset_mlxlm"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_path = dataset_dir / "train.jsonl"
    valid_path = dataset_dir / "valid.jsonl"

    try:
        with open(data_jsonl, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print("Dataset file is empty.", flush=True)
            sys.exit(1)

        if len(lines) < 4: # Need at least a few samples to split
            train_set, valid_set = lines, []
        else:
            train_set, valid_set = train_test_split(lines, test_size=0.1, random_state=42)

        if not train_set:
             print("Training set is empty after split. Check your data.", flush=True)
             sys.exit(1)

        with open(train_path, 'w') as f:
            f.writelines(train_set)
        if valid_set:
            with open(valid_path, 'w') as f:
                f.writelines(valid_set)

        print(f"Prepared dataset dir at {dataset_dir} (train={len(train_set)}, valid={len(valid_set)})", flush=True)
        return str(dataset_dir), len(train_set)

    except FileNotFoundError:
        print(f"Dataset path not found at {data_jsonl}", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"Error preparing dataset: {e}", flush=True)
        sys.exit(1)

def main(args):
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_base_path = workdir / "models"
    model_path = model_base_path / f"mlx_model_q{args.qbits}"

    # 1. Convert Model
    if not model_path.exists():
        model_base_path.mkdir(parents=True, exist_ok=True)
        convert_cmd = (
            f'{sys.executable} -m mlx_lm.convert --hf-path {args.hf_model} '
            f'--mlx-path {model_path} -q --q-bits {args.qbits}'
        )
        run_command(convert_cmd)
    else:
        print(f"Skip convert: {model_path} already exists.", flush=True)

    # 2. Prepare Dataset
    dataset_dir, num_train_samples = prepare_dataset(args.data_jsonl, workdir)

    if num_train_samples == 0:
        print("No training samples found. Exiting.", flush=True)
        sys.exit(0)

    # 3. Fine-tune using LoRA
    adapter_path = workdir / "lora" / "adapters"
    # Correctly calculate iterations based on gradient accumulation
    iters = (num_train_samples * args.epochs) // (args.batch_size * args.grad_accum)
    if iters == 0: iters = 1 # ensure at least one iteration

    finetune_cmd = (
        f'{sys.executable} -m mlx_lm.lora --model {model_path} '
        f'--data {dataset_dir} --train --lora-layers 4 '
        f'--batch-size {args.batch_size} --iters {iters} '
        f'--lr {args.lr} --max-seq-len {args.seq_len} '
        f'--adapter-path {adapter_path} --save-every {iters} --steps-per-report 1 '
        f'--lora-r {args.lora_r} --lora-alpha {args.lora_alpha} --lora-dropout {args.lora_dropout} '
        f'{'--grad-checkpoint' if args.grad_checkpoint else ''}'
    )
    run_command(finetune_cmd)

    # 4. Fuse adapters
    if args.fuse:
        fused_path = model_base_path / f"mlx_model_q{args.qbits}_fused"
        fuse_cmd = (
            f'{sys.executable} -m mlx_lm.fuse --model {model_path} '
            f'--adapter-path {adapter_path} --save-path {fused_path}'
        )
        run_command(fuse_cmd)

    print("\nFine-tuning process completed successfully.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with MLX.")
    # Required Arguments
    parser.add_argument("--hf-model", required=True, help="Hugging Face model ID.")
    parser.add_argument("--workdir", required=True, help="Working directory for models and data.")
    parser.add_argument("--data-jsonl", required=True, help="Path to the JSONL dataset file.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--seq-len", type=int, default=512, help="Maximum sequence length.")

    # Model and Quantization
    parser.add_argument("--qbits", type=int, default=4, choices=[4, 8], help="Quantization bits.")

    # LoRA Specific Arguments for Memory Saving
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA r parameter (rank). Default: 8")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha parameter. Default: 16")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    
    # Flags
    parser.add_argument("--fuse", action="store_true", help="Fuse the LoRA adapter after training.")
    parser.add_argument("--grad-checkpoint", action='store_true', default=True, help="Enable gradient checkpointing to save memory.")

    args = parser.parse_args()
    main(args)
