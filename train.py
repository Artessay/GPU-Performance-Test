import os
import json
import time
import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from utils import seed_everything, load_model_and_tokenizer

model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
data_path = "data/alpaca_zh_demo.json"

assert torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
device_name = torch.cuda.get_device_name(0)
result_path = f"data/{device_name}_train_{num_gpus}.json".replace(" ", "_")

def train():
    seed_everything(42)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)

    def process_data(examples):
        texts = []
        for instruction, input_text, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output},
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            texts.append(text)

        tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=1024)
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    dataset = load_dataset("json", data_files=data_path)
    tokenized_datasets = dataset.map(process_data, batched=True)
    
    # Disable evaluation, use only training set
    train_dataset = tokenized_datasets
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
    )
    
    # Disable evaluation and checkpoint saving
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        logging_strategy="steps",
        eval_strategy="no",  # Disable evaluation
        save_strategy="no",  # Disable checkpoint saving
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        data_collator=data_collator,
    )
    
    start_time = time.time()

    trainer.train()

    elapsed = time.time() - start_time

    print("Training completed")
    print(f"Total time: {elapsed:.2f}s")
    
    with open(result_path, "w") as f:
        json.dump({
            "model": model_path,
            "time": elapsed,
            "num_gpus": num_gpus,
            "device_name": device_name,
        }, f, indent=4)

if __name__ == "__main__":
    train()