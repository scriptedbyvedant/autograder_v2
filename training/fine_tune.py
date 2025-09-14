
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# --- CONFIGURATION ---
PROJECT_ROOT = os.getcwd()
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")
DATASET_FILE = os.path.join(TRAINING_DIR, "training_dataset.jsonl")
OUTPUT_DIR = os.path.join(TRAINING_DIR, "results")

def run_finetuning():
    """
    Loads the dataset, configures the model for LoRA-based fine-tuning,
    and runs the training process.
    """
    # 1. Load the dataset
    print(f"Loading dataset from {DATASET_FILE}...")
    if not os.path.exists(DATASET_FILE):
        print(f"Dataset file not found at {DATASET_FILE}. Please run the data export step first.")
        return
    
    dataset = load_dataset('json', data_files=DATASET_FILE, split="train")
    print(f"Dataset loaded with {len(dataset)} examples.")

    # 2. Load the pre-trained model
    print(f"Loading base model: {BASE_MODEL_NAME}...")
    # The definitive fix for the "meta tensor" error in memory-constrained environments:
    # - low_cpu_mem_usage=True: Creates the model as an empty "shell" on the meta device.
    # - .to('cpu'): Immediately materializes the model by loading the weights into the shell on the CPU.
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,    # Keep memory usage low
        low_cpu_mem_usage=True,       # Prevents memory spike and disk offloading
        trust_remote_code=True
    )
    print("Materializing model on CPU...")
    model = model.to('cpu')

    model.config.use_cache = False # Recommended for training

    # 3. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Configure LoRA (Low-Rank Adaptation)
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    print("LoRA configured. Trainable parameters:")
    model.print_trainable_parameters()

    # 5. Set up Training Arguments
    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4, # Reduced for stability on CPU
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=False, # Must be False for CPU-only training
        max_grad_norm=0.3,
        max_steps=-1,
        num_train_epochs=1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
    )

    # 6. Initialize the Trainer
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
        tokenized_batch = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        return tokenized_batch

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    trainer.train_dataset = tokenized_dataset

    # 7. Run the Fine-Tuning
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")

    # 8. Save the trained LoRA adapters
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.model.save_pretrained(final_model_path)
    print(f"Fine-tuned model adapters saved to {final_model_path}")

if __name__ == "__main__":
    run_finetuning()
