import os
import json
import time
import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)

def load_model_and_tokenizer(model_name_or_path):
    """Load model and tokenizer without specifying device mapping"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True  # For ZeRO-3 compatibility
    )
    return model, tokenizer

def process_data(examples, tokenizer):
    """Process dataset to convert dialogues into model training format"""
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
    
    tokenized = tokenizer(texts, padding=True, truncation=True, max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def train_model(model_path, data_path, deepspeed_config):
    """Main function for model training"""
    # Environment setup
    assert torch.cuda.is_available(), "GPU not available, please check environment configuration"
    num_gpus = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    result_path = f"data/{device_name}_train_{num_gpus}.json".replace(" ", "_")
    
    print(f"Starting training: {model_path} on {device_name} x {num_gpus} GPUs")
    
    # Set random seed
    set_seed(42)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Load and process dataset
    dataset = load_dataset("json", data_files=data_path)
    tokenized_datasets = dataset.map(
        lambda examples: process_data(examples, tokenizer),
        batched=True
    )
    train_dataset = tokenized_datasets["train"]
    
    # Configure training parameters
    training_args = TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_dir="./logs",
        logging_steps=10,
        logging_strategy="steps",
        eval_strategy="no",
        save_strategy="no",
        remove_unused_columns=True,
        report_to="none",
        deepspeed=deepspeed_config,
    )
    
    # Create data collator and trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    
    # Save training results
    if trainer.is_world_process_zero():
        print(f"Training completed - Total time: {elapsed:.2f} seconds")
        with open(result_path, "w") as f:
            json.dump({
                "model": model_path,
                "time": elapsed,
                "num_gpus": num_gpus,
                "device_name": device_name,
            }, f, indent=4)

if __name__ == "__main__":
    # Set environment variables and paths
    model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-3B-Instruct")
    data_path = "data/alpaca_zh_demo.json"
    deepspeed_config = "data/ds_config.json"
    
    # Start training
    train_model(model_path, data_path, deepspeed_config)