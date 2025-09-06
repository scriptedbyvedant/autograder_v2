
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# --- CONFIGURATION ---

# The base model to be fine-tuned. We use a 7B parameter model as a good starting point.
# This requires a capable GPU (e.g., NVIDIA T4, V100, A100 with at least 16GB VRAM).
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# The dataset file created by the export_data.py script.
DATASET_FILE = os.path.join(os.path.dirname(__file__), "training_dataset.jsonl")

# The directory where the fine-tuned model adapters will be saved.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_finetuning():
    """
    Loads the dataset, configures the model for LoRA-based fine-tuning,
    and runs the training process.
    """
    # 1. Load the dataset
    print(f"Loading dataset from {DATASET_FILE}...")
    if not os.path.exists(DATASET_FILE):
        print("Dataset file not found. Please run 'python training/export_data.py' first.")
        return
    
    dataset = load_dataset('json', data_files=DATASET_FILE, split="train")
    print(f"Dataset loaded with {len(dataset)} examples.")

    # 2. Configure Quantization (for memory efficiency)
    # This allows us to load a large model on a smaller GPU by using 4-bit precision.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # 3. Load the pre-trained model
    print(f"Loading base model: {BASE_MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically place layers on available GPUs
        trust_remote_code=True
    )
    model.config.use_cache = False # Recommended for training

    # 4. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    # Set a padding token if the model doesn't have one.
    tokenizer.pad_token = tokenizer.eos_token

    # 5. Configure LoRA (Low-Rank Adaptation)
    # This freezes the base model and only trains a small number of adapter layers,
    # which is much faster and more memory-efficient.
    lora_config = LoraConfig(
        lora_alpha=16,          # The scaling factor for the LoRA matrices
        lora_dropout=0.1,       # Dropout for regularization
        r=64,                   # The rank of the LoRA matrices
        bias="none",
        task_type="CAUSAL_LM",
        # Apply LoRA to all linear layers of the attention blocks
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )
    model = get_peft_model(model, lora_config)
    print("LoRA configured. Trainable parameters:")
    model.print_trainable_parameters()

    # 6. Set up Training Arguments
    # These arguments control the training process, such as batch size, learning rate, etc.
    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,    # Batch size per GPU
        gradient_accumulation_steps=1,  # Number of steps to accumulate gradients
        optim="paged_adamw_32bit",
        save_steps=100,                   # Save a checkpoint every 100 steps
        logging_steps=10,                 # Log training metrics every 10 steps
        learning_rate=2e-4,
        fp16=True,                        # Use 16-bit precision for training
        max_grad_norm=0.3,
        max_steps=-1,                     # If -1, train for `num_train_epochs`
        num_train_epochs=1,               # Number of times to iterate over the dataset
        warmup_ratio=0.03,
        group_by_length=True,             # Group sequences of similar length for efficiency
        lr_scheduler_type="constant",
    )

    # 7. Initialize the Trainer
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                     'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                     'labels': torch.stack([f['input_ids'] for f in data])},
    )

    # Tokenize the dataset
    def tokenize_function(examples):
        # We need to tokenize the text field from the dataset
        tokenized_batch = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        return tokenized_batch

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    trainer.train_dataset = tokenized_dataset

    # 8. Run the Fine-Tuning
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")

    # 9. Save the trained LoRA adapters
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.model.save_pretrained(final_model_path)
    print(f"Fine-tuned model adapters saved to {final_model_path}")


if __name__ == "__main__":
    run_finetuning()
