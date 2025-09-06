# File: pages/3_fine_tuning.py

import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

# Add project root to the Python path to allow importing from other directories
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from database.postgres_handler import PostgresHandler
from training.export_data import PROMPT_TEMPLATE

# --- CONFIGURATION ---
ADAPTER_PATH = PROJECT_ROOT / "training" / "results" / "final_model"
TRAINING_DATASET_PATH = PROJECT_ROOT / "training" / "training_dataset.jsonl"

# --- HELPER FUNCTIONS ---

def count_training_examples() -> int:
    """Counts the number of human-corrected examples in the database."""
    try:
        pg_handler = PostgresHandler()
        # Query for examples where the professor has provided distinct, non-empty feedback.
        query = """
        SELECT COUNT(*)
        FROM grading_results
        WHERE new_feedback IS NOT NULL
          AND new_feedback != ''
          AND new_feedback != old_feedback;
        """
        count_df = pg_handler.execute_query(query, fetch="one")
        return count_df['count'].iloc[0] if count_df is not None and not count_df.empty else 0
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return 0

def run_script(script_path: str, output_placeholder):
    """Executes a script as a subprocess and streams its output to the UI."""
    output_placeholder.info(f"Running {script_path}...")
    log_output = ""
    try:
        # Start the subprocess
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )

        # Stream the output in real-time
        for line in iter(process.stdout.readline, ''):
            log_output += line
            output_placeholder.code(log_output, language="bash")
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            output_placeholder.success(f"Script finished successfully.")
        else:
            output_placeholder.error(f"Script failed with return code {return_code}. Check logs above.")
            
    except FileNotFoundError:
        output_placeholder.error(f"Error: Script not found at {script_path}")
    except Exception as e:
        output_placeholder.error(f"An unexpected error occurred: {e}\n{log_output}")

# --- PAGE UI ---
st.set_page_config(page_title="⚙️ Fine-Tuning", layout="wide")
st.title("⚙️ Fine-Tuning Control Panel")

if "logged_in_prof" not in st.session_state:
    st.warning("Please login first to access this page.", icon="🔒"); st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current Status")
    num_examples = count_training_examples()
    is_finetuned = ADAPTER_PATH.is_dir()

    st.metric("Corrected Examples Ready for Training", num_examples)
    if is_finetuned:
        st.success("**Status:** Currently using a **Fine-Tuned Model**.")
    else:
        st.info("**Status:** Currently using the **Base Model**.")

    st.markdown("--- ")
    st.subheader("How It Works")
    st.markdown("""
    This page allows you to improve the AI's grading accuracy by training it on your own corrections. The process involves two main steps:

    1.  **Export Data**: First, you must export the human-corrected grading data from the database. This script gathers all the examples where you have edited the AI's feedback and formats them into a training file.
    
    2.  **Start Fine-Tuning**: Once the data is exported, you can begin the fine-tuning process. This will train a new version of the model on your examples. **This requires a powerful GPU.**
    """)

with col2:
    st.subheader("Controls")

    # --- 1. DATA EXPORT ---
    st.markdown("#### Step 1: Export Data for Training")
    if st.button("Export Data", help="Gathers your corrections into a training file."):
        script_path = str(PROJECT_ROOT / "training" / "export_data.py")
        output_box = st.empty()
        run_script(script_path, output_box)

    if TRAINING_DATASET_PATH.exists():
        st.success(f"Training dataset found at `{TRAINING_DATASET_PATH}`.")
    else:
        st.warning(f"Training dataset not yet created.")

    st.markdown("--- ")

    # --- 2. FINE-TUNING ---
    st.markdown("#### Step 2: Start Fine-Tuning")
    st.warning("**Warning:** This process is computationally intensive and requires a compatible GPU (e.g., NVIDIA T4, V100, A100). It may take a long time.")
    if st.button("Start Fine-Tuning", disabled=not TRAINING_DATASET_PATH.exists(), help="This will fail if you do not have a compatible GPU and the required libraries."):
        script_path = str(PROJECT_ROOT / "training" / "fine_tune.py")
        output_box = st.empty()
        run_script(script_path, output_box)

with st.expander("View Prompt Template"):
    st.code(PROMPT_TEMPLATE, language="text")
