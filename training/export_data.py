
import os
import json
import pandas as pd
from dotenv import load_dotenv
import sys

# Add project root to path to allow importing database handler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.postgres_handler import PostgresHandler

load_dotenv()

# --- CONFIGURATION ---
OUTPUT_FILE = "training_dataset.jsonl"
# This is the instruction template for the language model.
# It's a "zero-shot" prompt because we are not providing examples in the prompt itself.
# The model will learn from the many examples in the training dataset.
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
    print("Connecting to database...")
    try:
        pg_handler = PostgresHandler()
        print("Database connection successful.")
    except Exception as e:
        print(f"Failed to connect to the database: {e}")
        return

    # Fetch all results where the professor has provided corrected feedback.
    # We select rows where the new_feedback is different from the old one
    # and is not empty.
    query = """
    SELECT question, student_answer, old_feedback, new_feedback
    FROM grading_results
    WHERE new_feedback IS NOT NULL
      AND new_feedback != ''
      AND new_feedback != old_feedback;
    """
    print("Executing query to fetch corrected examples...")
    results_df = pg_handler.execute_query(query, fetch="all")
    if results_df is None or results_df.empty:
        print("No corrected grading examples found in the database. Fine-tuning requires data.")
        print("To generate data, review and save edits on the 'Grading Results' page in the app.")
        return

    print(f"Found {len(results_df)} corrected examples for training.")

    # Create the training data
    training_data = []
    for index, row in results_df.iterrows():
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
        print(f"Successfully created dataset at: {output_path}")
        print(f"This file contains {len(training_data)} training examples.")
    except IOError as e:
        print(f"Error writing to file: {e}")


if __name__ == "__main__":
    create_finetuning_dataset()
