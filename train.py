import os
import json
import time
import torch
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
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
            {"role": "assistant", "content": item["output"]}, # Add assistant response
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {"text": text}

    dataset = Dataset.from_list([format_instruction(d) for d in data])

    # Process dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Split dataset into training and validation sets
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)
    
    # Generate training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Create Trainer
    start_time = time.time()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()
    
    # # Save the fine-tuned model
    # model_save_path = "./fine_tuned_model"
    # model.save_pretrained(model_save_path)
    # tokenizer.save_pretrained(model_save_path)

    elapsed = time.time() - start_time

    print("Training completed")
    print(f"Total time: {elapsed:.2f}s")
    
    # Save results
    with open(result_path, "w") as f:
        json.dump({
            "model": model_path,
            "time": elapsed,
            # "save_path": model_save_path,
        }, f, indent=4)

if __name__ == "__main__":
    train()