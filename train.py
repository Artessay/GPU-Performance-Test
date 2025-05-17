import os
import json
import time
import torch
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from utils import seed_everything, load_model_and_tokenizer

model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
data_path = "data/alpaca_zh_demo.json"

assert torch.cuda.is_available()
result_path = f"data/{torch.cuda.get_device_name(0)}_train_{torch.cuda.device_count()}.json".replace(" ", "_")

def train():
    seed_everything(42)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Read dataset
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Prepare dataset
    def format_instruction(item):
        # Construct instruction and input messages
        messages = [
            {"role": "system", "content": item["instruction"]},
            {"role": "user", "content": item["input"]},
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {"text": text}  # Fixed the dictionary syntax here

    dataset = Dataset.from_list([format_instruction(d) for d in data])

    # Process dataset
    def tokenize_function(item):
        return tokenizer(
            item["text"],
            return_tensors="pt",
            padding=True,  # Add padding to support batch processing
            truncation=True,  # Add truncation to handle long inputs
            max_length=1024,  # Set the maximum length of input sequences
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training parameter adjustment (key modification points)
    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=3e-5,
        gradient_accumulation_steps=4,

        # Disable checkpoint saving
        save_strategy="no",  # Turn off the save strategy
        logging_steps=1,  # Log every 1 step
        logging_dir="./logs",  # Set the logging directory
        output_dir="./output",  # Set the output directory
    )

    # Create Trainer
    start_time = time.time()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    elapsed = time.time() - start_time

    # Start training
    trainer.train()
    print("Training completed")
    print(f"Total time: {elapsed:.2f}s")
    
    # Save results
    with open(result_path, "w") as f:
        json.dump({
            "model": model_path,
            "time": elapsed,
        }, f, indent=4)

if __name__ == "__main__":
    train()