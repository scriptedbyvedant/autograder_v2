# File: pages/3_fine_tuning.py

import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

# Add project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from database.postgres_handler import PostgresHandler
from training.export_data import PROMPT_TEMPLATE
from utils.logger import logger

# --- CONFIGURATION ---
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TRAINING_DIR = PROJECT_ROOT / "training"
MLX_RUN_DIR = TRAINING_DIR / "mlx_runs"
TRAINING_DATASET_PATH = TRAINING_DIR / "training_dataset.jsonl"
FUSED_MODEL_PATH = MLX_RUN_DIR / "models" / "mlx_model_q4_merged"

# --- HELPER FUNCTIONS ---

def count_training_examples() -> int:
    """Counts the number of human-corrected examples in the database."""
    try:
        pg_handler = PostgresHandler()
        query = """
        SELECT COUNT(*)
        FROM grading_results
        WHERE new_feedback IS NOT NULL
          AND new_feedback != ''
          AND new_feedback != old_feedback;
        """
        result = pg_handler.execute_query(query, fetch="one")
        return result['count'] if result else 0
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        st.error(f"Database connection failed: {e}")
        return 0

def run_script_and_stream_output(command: str, output_placeholder):
    """Executes a command and streams its output to the UI."""
    logger.info(f"Running command:\n{command}")
    output_placeholder.info(f"Executing command...")
    log_output = ""
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )

        for line in iter(process.stdout.readline, ''):
            log_output += line
            logger.info(line.strip())
            output_placeholder.code(log_output, language="bash")
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("Command finished successfully.")
            output_placeholder.success("Process finished successfully.")
        else:
            logger.error(f"Command failed with return code {return_code}.")
            output_placeholder.error(f"Process failed with return code {return_code}. Check logs above.")
            
    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}\n{log_output}")
        output_placeholder.error(f"An unexpected error occurred: {e}\n{log_output}")

# --- PAGE UI ---
st.set_page_config(page_title="⚙️ Fine-Tuning (MLX)", layout="wide")
st.title("⚙️ Fine-Tuning Control Panel for Apple Silicon (MLX)")

if "logged_in_prof" not in st.session_state:
    st.warning("Please login first to access this page.", icon="🔒"); st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current Status")
    num_examples = count_training_examples()
    is_finetuned = FUSED_MODEL_PATH.is_dir()

    st.metric("Corrected Examples Ready for Training", num_examples)
    if is_finetuned:
        st.success("**Status:** Currently using a **Fine-Tuned Model**.")
    else:
        st.info("**Status:** Currently using the **Base Model**.")

    st.markdown("--- ")
    st.subheader("How It Works")
    st.markdown("""
    This page uses **Apple's MLX framework** to fine-tune a model directly on your Mac's Apple Silicon GPU. This is much more reliable than the previous PyTorch-based method.

    1.  **Export Data**: Gathers your corrections from the database into a `training_dataset.jsonl` file.
    
    2.  **Start MLX Fine-Tuning**: This kicks off a multi-step process that:
        - Converts the base Hugging Face model to a quantized MLX format.
        - Fine-tunes it on your data using LoRA.
        - Fuses the LoRA adapters to create a final, ready-to-use model.
    """)

with col2:
    st.subheader("Controls")

    # --- 1. DATA EXPORT ---
    st.markdown("#### Step 1: Export Data for Training")
    if st.button("Export Data", help="Gathers your corrections into a training file."):
        export_script_path = TRAINING_DIR / "export_data.py"
        cmd_str = f'{sys.executable} "{export_script_path}"'
        output_box = st.empty()
        run_script_and_stream_output(cmd_str, output_box)

    if TRAINING_DATASET_PATH.exists():
        st.success(f"Training dataset found at `{TRAINING_DATASET_PATH}`.")
    else:
        st.warning(f"Training dataset not yet created.")

    st.markdown("--- ")

    # --- 2. MLX FINE-TUNING ---
    st.markdown("#### Step 2: Start MLX Fine-Tuning")
    st.info("This process is optimized for Apple Silicon and should be much more stable.")
    if st.button("Start MLX Fine-Tuning", disabled=not TRAINING_DATASET_PATH.exists(), help="This uses the MLX framework for stable fine-tuning on Apple Silicon."):
        # Correcting the filename from the typo 'mlx_fine_tune.py' to the actual 'mix_fine_tune.py'
        mlx_script_path = TRAINING_DIR / "mix_fine_tune.py"
        
        command_parts = [
            f'{sys.executable}',
            f'"{mlx_script_path}"',
            f'--hf-model "{BASE_MODEL_NAME}"',
            f'--data "{TRAINING_DATASET_PATH}"',
            f'--workdir "{MLX_RUN_DIR}"',
            '--qbits 4',
            '--epochs 1',
            '--batch-size 1',
            '--grad-accum 8',
            '--fuse',
        ]
        command = ' '.join(command_parts)

        output_box = st.empty()
        run_script_and_stream_output(command, output_box)

with st.expander("View Prompt Template"):
    st.code(PROMPT_TEMPLATE, language="text")
