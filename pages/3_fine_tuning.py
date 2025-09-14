
# File: pages/3_fine_tuning.py

import streamlit as st
import json
import sys
from pathlib import Path

# Add project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from database.postgres_handler import PostgresHandler
from utils.logger import logger

# --- CONFIGURATION & CONSTANTS ---

# This is the prompt template the fine-tuning job expects.
# It's crucial that the data generated matches this structure.
PROMPT_TEMPLATE = """
[INST]
You are an expert teaching assistant. Given a question, a student's answer,
and a grading rubric, provide a score and constructive feedback.

### Question:
{question}

### Student's Answer:
{student_answer}

### Ideal Answer & Rubric:
{ideal_answer}

[/INST]
{model_response}
"""

# The content of the Colab script to be displayed on the page.
# Sourced from training/colab_finetune.py
COLAB_SCRIPT_CONTENT = """
#!/usr/bin/env python3
# Run this script in a single Google Colab cell.
# Instructions:
# 1. Create a new Colab Notebook.
# 2. Change the runtime to GPU (Runtime -> Change runtime type -> T4 GPU).
# 3. Copy and paste the entire content of this file into a single Colab cell.
# 4. Upload your 'training_dataset.jsonl' file to the Colab session.
# 5. Run the cell.
# 6. After it finishes, download the 'trained_adapters.npz' file.

# Step 1: Install the library
print("⏳ Installing MLX dependencies...")
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "mlx-lm", "--quiet"])
print("✅ Installation complete.")

# Step 2: Configure MLX environment
import os
import ctypes
from pathlib import Path
try:
    print("⏳ Locating 'libmlx.so'...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "-f", "mlx-lm"],
        capture_output=True, text=True, check=True
    )
    libmlx_so_line = next((line for line in result.stdout.splitlines() if 'lib/libmlx.so' in line.replace('\\', '/')), None)
    if not libmlx_so_line: raise FileNotFoundError("'lib/libmlx.so' not found.")
    libmlx_so_path = Path(libmlx_so_line.strip())
    if not libmlx_so_path.exists(): raise FileNotFoundError(f"File path from pip does not exist: '{libmlx_so_path}'")
    mlx_lib_path = libmlx_so_path.parent
    os.environ['LD_LIBRARY_PATH'] = f"{str(mlx_lib_path)}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    ctypes.CDLL(str(libmlx_so_path))
    import mlx.core
    print("✅ MLX environment configured successfully.")
except Exception as e:
    print(f"❌ ERROR: A critical error occurred while configuring the MLX environment: {e}")
    sys.exit(1)

# Step 3: Run the fine-tuning process
import json, random
print("\n⏳ Preparing dataset...")
DATA_SOURCE_FILE = "training_dataset.jsonl"
DATA_DIR = Path("./prepared_data")
DATA_DIR.mkdir(exist_ok=True)

if not Path(DATA_SOURCE_FILE).exists():
    print(f"❌ ERROR: Cannot find '{DATA_SOURCE_FILE}'. Please upload it first.")
    sys.exit(1)

with open(DATA_SOURCE_FILE, 'r') as f: all_data = [json.loads(line) for line in f if line.strip()]

if len(all_data) < 3:
    raise ValueError(f"Dataset must have at least 3 entries to create train/test/validation splits. Found {len(all_data)}.")

random.shuffle(all_data)
# Split data: 1 for validation, 1 for test, rest for training
valid_data, test_data, train_data = all_data[:1], all_data[1:2], all_data[2:]

def write_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data: f.write(json.dumps(item) + '\n')

write_jsonl(train_data, DATA_DIR / "train.jsonl")
write_jsonl(valid_data, DATA_DIR / "valid.jsonl")
write_jsonl(test_data, DATA_DIR / "test.jsonl")
print(f"✅ Dataset prepared: Train={len(train_data)}, Valid={len(valid_data)}, Test={len(test_data)}")

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

try:
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout: print(line, end='')
    if proc.returncode != 0: raise subprocess.CalledProcessError(proc.returncode, cmd)
except subprocess.CalledProcessError as e:
    print(f"\n--- ❌ ERROR: The training process failed with return code {e.returncode}. ---")
    sys.exit(1)

print("\n--- ✅ Fine-tuning complete! ---")
final_adapter_file = ADAPTER_PATH / "adapters.npz"
if final_adapter_file.exists():
    renamed_file = Path("./trained_adapters.npz")
    final_adapter_file.rename(renamed_file)
    print(f"Successfully created: {renamed_file}")
    print("\nACTION REQUIRED: Please download 'trained_adapters.npz' from the file browser on the left.")
else:
    print("❌ ERROR: Could not find the expected 'adapters.npz' file after training.")
"""

# --- HELPER FUNCTIONS ---

def generate_training_data():
    """
    Fetches corrected grading results from the database and formats them
    into a JSONL string for fine-tuning.
    """
    try:
        pg_handler = PostgresHandler()
        # This query selects only the records that have been manually corrected by a human.
        # It now joins on course and assignment_no for improved accuracy.
        query = """
        SELECT
            gr.question,
            gr.student_answer,
            gr.new_feedback AS corrected_feedback,
            pd.ideal_answer,
            pd.rubric
        FROM
            grading_results AS gr
        JOIN
            (
                SELECT DISTINCT course, assignment_no, question, ideal_answer, rubric
                FROM prof_data
            ) AS pd
        ON
            gr.question = pd.question
            AND gr.course = pd.course
            AND gr.assignment_no = pd.assignment_no
        WHERE
            gr.new_feedback IS NOT NULL
            AND gr.new_feedback != ''
            AND gr.new_feedback != gr.old_feedback;
        """
        corrected_examples = pg_handler.execute_query(query, fetch="all")
        logger.info(f"Found {len(corrected_examples)} corrected examples in the database.")

        if not corrected_examples:
            return None, 0

        # Convert the data to the JSONL format with the required prompt structure.
        jsonl_output = ""
        for ex in corrected_examples:
            # Reconstruct the "ideal answer" to include the rubric, as expected by the model.
            ideal_answer_with_rubric = f"Ideal Answer: {ex['ideal_answer']}\nRubric: {json.dumps(ex['rubric'])}"
            
            # The student answer is stored as a JSON string of content blocks.
            # We need to parse it and extract the text.
            try:
                answer_blocks = json.loads(ex['student_answer'])
                student_text = next((block['content'] for block in answer_blocks if block['type'] == 'text'), "")
            except (json.JSONDecodeError, TypeError):
                student_text = "No valid answer content found."

            # Format the final text prompt
            text_prompt = PROMPT_TEMPLATE.format(
                question=ex['question'],
                student_answer=student_text,
                ideal_answer=ideal_answer_with_rubric,
                model_response=ex['corrected_feedback']
            )
            # Create a JSONL entry. The training script expects a 'text' key.
            jsonl_output += json.dumps({"text": text_prompt}) + "\n"
            
        return jsonl_output, len(corrected_examples)

    except Exception as e:
        logger.error(f"Error generating training data: {e}")
        st.error(f"An error occurred while generating training data: {e}")
        return None, 0

# --- PAGE UI ---

st.set_page_config(page_title="🚀 Model Finetuning Assistant", layout="wide")
st.title("🚀 Model Finetuning Assistant")

if "logged_in_prof" not in st.session_state:
    st.warning("Please login first to access this page.", icon="🔒"); st.stop()

st.info(
    "**Objective:** Improve the AI's grading accuracy by training it on your past corrections. "
    "This page automates the data preparation and provides a step-by-step guide for fine-tuning."
)

# --- STEP 1: GENERATE DATA ---
st.header("Step 1: Generate & Download Training Data")
st.markdown(
    "Click the button below to gather all the grading adjustments you've made "
    "and package them into a `training_dataset.jsonl` file. This file is required for the next step."
)

if st.button("📦 Generate Training Data", key="generate_data"):
    with st.spinner("Querying database and preparing data..."):
        jsonl_data, count = generate_training_data()
        if jsonl_data and count > 0:
            st.session_state['generated_training_data'] = jsonl_data
            st.session_state['generated_training_count'] = count
            st.success(f"Successfully generated a training file with {count} corrected examples.")
        else:
            st.warning("No corrected examples found in the database. Please grade some answers and make corrections before trying to fine-tune.")

if 'generated_training_data' in st.session_state:
    st.download_button(
        label="📥 Download training_dataset.jsonl",
        data=st.session_state['generated_training_data'],
        file_name="training_dataset.jsonl",
        mime="application/jsonl"
    )

st.markdown("---")


# --- STEP 2: TRAIN IN COLAB ---
st.header("Step 2: Train the Model in Google Colab")
st.markdown("""
Follow these instructions carefully:
1.  **Open Google Colab:** [Click here to open a new Colab notebook.](https://colab.research.google.com/)
2.  **Set GPU Runtime:** In Colab, go to `Runtime` > `Change runtime type` and select `T4 GPU`.
3.  **Copy the Code:** Click the "Copy" button on the top right of the code block below and paste it into a single cell in your Colab notebook.
4.  **Upload Data:** In the Colab file browser (the folder icon on the left), upload the `training_dataset.jsonl` file you downloaded in Step 1.
5.  **Run the Cell:** Execute the Colab cell. The training process will take some time (approx. 15-20 minutes).
""")

st.code(COLAB_SCRIPT_CONTENT, language="python")

st.markdown("---")


# --- STEP 3: DEPLOY THE NEW MODEL ---
st.header("Step 3: Deploy the Fine-Tuned Model")
st.markdown("""
1.  **Download the Result:** Once the Colab script finishes, a file named `trained_adapters.npz` will appear in the file browser. Download this file to your computer.
2.  **Place the File:** Move the `trained_adapters.npz` file into the `training/` directory within this project.
3.  **Restart the Application:** Stop and restart the Streamlit application. The system will automatically detect and use your new fine-tuned model for all future grading tasks.
""")

st.warning("⚠️ **Important:** If you ever want to revert to the original base model, simply delete the `trained_adapters.npz` file from the `training/` directory and restart the application.")
