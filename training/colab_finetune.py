#!/usr/bin/env python3
"""
Run this script in a single Google Colab cell.

Instructions:
1. Create a new Colab Notebook.
2. Change the runtime to GPU (Runtime -> Change runtime type -> T4 GPU).
3. Copy and paste the entire content of this file into a single Colab cell.
4. Upload your 'training_dataset.jsonl' file to the Colab session.
5. Run the cell.
6. After it finishes, download the 'trained_adapters.npz' file from the Colab files panel.
"""

# Step 1: Install the library
print("⏳ Installing MLX dependencies...")
import subprocess
import sys
# Run pip install in a subprocess to ensure it's clean
subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "mlx-lm", "--quiet"])
print("✅ Installation complete.")


# Step 2: Manually find and load the `libmlx.so` shared library
import os
import ctypes
from pathlib import Path

try:
    print("⏳ Locating 'libmlx.so' using pip's file list...")

    # Use 'pip show -f' to find all files installed by the mlx package.
    # This is the most reliable way to find the library file.
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "-f", "mlx-lm"],
        capture_output=True,
        text=True,
        check=True
    )

    # Search for the line that points to the shared library.
    # We search for a path ending in 'lib/libmlx.so' to be specific.
    libmlx_so_line = next((line for line in result.stdout.splitlines() if 'lib/libmlx.so' in line.replace('\\', '/')), None)

    if not libmlx_so_line:
        raise FileNotFoundError(
            "'lib/libmlx.so' not found in the output of 'pip show -f mlx-lm'. "
            "The installation may be corrupt or in an unexpected layout."
        )

    # The line from pip is usually indented. Strip whitespace to get the path.
    libmlx_so_path = Path(libmlx_so_line.strip())

    if not libmlx_so_path.exists():
         raise FileNotFoundError(f"File path from pip does not exist: '{libmlx_so_path}'")

    # The directory containing libmlx.so is what we need.
    mlx_lib_path = libmlx_so_path.parent
    print(f"✅ Found MLX library directory: {mlx_lib_path}")

    # 1. Add the path to LD_LIBRARY_PATH for any subprocesses to inherit.
    os.environ['LD_LIBRARY_PATH'] = f"{str(mlx_lib_path)}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    print("✅ Environment path configured for subprocesses.")

    # 2. Manually load the shared library for the *current* process.
    ctypes.CDLL(str(libmlx_so_path))
    print("✅ Manually loaded libmlx.so into the current process.")

    # 3. Verify that the import now works as expected.
    import mlx.core
    print("✅ Verification complete: mlx.core imported successfully.")

except (subprocess.CalledProcessError, StopIteration, FileNotFoundError, ImportError, OSError) as e:
    print(f"❌ ERROR: A critical error occurred while configuring the MLX environment: {e}")
    print("This can happen in some environments. Please try restarting the session and running the cell again.")
    sys.exit(1)


# Step 3: Run the rest of the fine-tuning process
import json
import random

# --- 3a. Verify and split data ---
print("\n⏳ Preparing dataset...")
DATA_SOURCE_FILE = "training_dataset.jsonl"
DATA_DIR = Path("./prepared_data")
DATA_DIR.mkdir(exist_ok=True)

if not Path(DATA_SOURCE_FILE).exists():
    print(f"❌ ERROR: Cannot find '{DATA_SOURCE_FILE}'. Please upload it again using the file browser.")
    sys.exit(1)

with open(DATA_SOURCE_FILE, 'r') as f:
    all_data = [json.loads(line) for line in f if line.strip()]

if len(all_data) < 3:
    raise ValueError(f"Dataset must have at least 3 entries. Found {len(all_data)}.")

random.shuffle(all_data)
valid_data, test_data, train_data = all_data[:1], all_data[1:2], all_data[2:]

def write_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data: f.write(json.dumps(item) + '\n')

write_jsonl(train_data, DATA_DIR / "train.jsonl")
write_jsonl(valid_data, DATA_DIR / "valid.jsonl")
write_jsonl(test_data, DATA_DIR / "test.jsonl")
print(f"✅ Dataset prepared: Train={len(train_data)}, Valid={len(valid_data)}, Test={len(test_data)}")

# --- 3b. Define parameters and run training ---
print("\n🚀 Starting fine-tuning process...")
HUGGING_FACE_MODEL = "mlx-community/Mistral-7B-v0.1-hf-4bit-mlx"
ADAPTER_PATH = Path("./lora_adapters")
ITERATIONS = 500
BATCH_SIZE = 1
NUM_LAYERS = 16

cmd = (
    f'python -m mlx_lm.lora '
    f'--model {HUGGING_FACE_MODEL} --data "{DATA_DIR}" --train '
    f'--batch-size {BATCH_SIZE} --iters {ITERATIONS} --num-layers {NUM_LAYERS} '
    f'--adapter-path "{ADAPTER_PATH}"'
)

print(f"Executing command:\n{cmd}")

# Run the training, printing output live
try:
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end='')
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
except subprocess.CalledProcessError as e:
    print(f"\n--- ❌ ERROR: The training process failed with return code {e.returncode}. ---")
    sys.exit(1)

# --- 3c. Prepare final file for download ---
print("\n--- ✅ Fine-tuning complete! ---")
final_adapter_file = ADAPTER_PATH / "adapters.npz"
if final_adapter_file.exists():
    renamed_file = Path("./trained_adapters.npz")
    final_adapter_file.rename(renamed_file)
    print(f"Successfully created: {renamed_file}")
    print("\nACTION REQUIRED: Please download 'trained_adapters.npz' from the file browser on the left.")
else:
    print("❌ ERROR: Could not find the expected 'adapters.npz' file after training.")
