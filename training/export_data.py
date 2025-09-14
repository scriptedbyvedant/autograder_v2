# File: training/export_data.py

import os
import json
from dotenv import load_dotenv
import sys

# Add project root to path to allow importing other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.postgres_handler import PostgresHandler
from utils.logger import logger

load_dotenv()

# --- CONFIGURATION ---
OUTPUT_FILE = "training_dataset.jsonl"
# This is the instruction template for the language model.
PROMPT_TEMPLATE = """### Instruction:
You are an AI assistant helping a professor grade. Based on the question and the student's answer, provide constructive feedback. The feedback should explain what the student did well and where they can improve, aligning with a standard academic rubric.

### Input:
Question: {question}
Student Answer: {student_answer}

### Response:
{new_feedback}"""

def create_finetuning_dataset():
    """
    Connects to the database, fetches human-corrected grading results,
    formats them into instruction-following prompts, and saves them to a
    JSON Lines file.
    """
    logger.info("Starting data export process...")
    try:
        pg_handler = PostgresHandler()
        logger.info("Database connection successful.")
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        return

    # Fetch all results where the professor has provided corrected feedback.
    query = """
    SELECT question, student_answer, old_feedback, new_feedback
    FROM grading_results
    WHERE new_feedback IS NOT NULL
      AND new_feedback != ''
      AND new_feedback != old_feedback;
    """
    logger.info("Executing query to fetch corrected examples...")
    results = pg_handler.execute_query(query, fetch="all")
    
    if not results:
        logger.warning("No corrected grading examples found in the database. Fine-tuning requires data.")
        logger.warning("To generate data, review and save edits on the 'Grading Results' page in the app.")
        return

    logger.info(f"Found {len(results)} corrected examples for training.")

    # Create the training data
    training_data = []
    for row in results:
        # Format the data into the prompt structure
        formatted_text = PROMPT_TEMPLATE.format(
            question=row['question'],
            student_answer=row['student_answer'],
            new_feedback=row['new_feedback']
        )
        training_data.append({"text": formatted_text})

    # Save to a JSON Lines file
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Successfully created dataset at: {output_path}")
        logger.info(f"This file contains {len(training_data)} training examples.")
    except IOError as e:
        logger.error(f"Error writing to file: {e}")

if __name__ == "__main__":
    create_finetuning_dataset()
