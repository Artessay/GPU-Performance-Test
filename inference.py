import os
import json
import time
import torch

from utils import seed_everything, load_model_and_tokenizer

# Configuration parameters
model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
data_path = "data/alpaca_zh_demo.json"
result_path = f"data/{torch.cuda.get_device_name(0)}_infer.json".replace(" ", "_")

def inference():
    seed_everything(42)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Read dataset
    with open(data_path, 'r') as f:
        dataset = json.load(f)

    # Lists to store inference times and token counts
    times = []
    token_counts = []

    for i, item in enumerate(dataset):
        # Construct instruction and input messages
        messages = [
            {"role": "system", "content": item["instruction"]},
            {"role": "user", "content": item["input"]},
        ]

        # Encode input and move to CUDA
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Perform inference
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,  # Corrected parameter value
            )
        elapsed = time.time() - start_time

        # Decode output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Count generated tokens
        token_count = len(generated_ids[0])
        token_counts.append(token_count)

        # Record time and print results
        times.append(elapsed)
        print(f"Question {i+1}/{len(dataset)}\n"
              f"Generated {token_count} tokens in {elapsed:.2f}s ({elapsed/token_count:.4f}s per token)\n"
              f"Problem: {item['instruction']}\nResponse: {response}\n{'-'*40}")

    # Output time statistics
    total_tokens = sum(token_counts)
    total_time = sum(times)
    print(f"\nTotal samples: {len(times)}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per sample: {total_time/len(times):.2f}s")
    print(f"Average time per token: {total_time/total_tokens:.4f}s")

    # Save results
    with open(result_path, 'w') as f:
        json.dump({
            "samples": len(times),
            "model_name": model_path.split('/')[-1],
            "avg_time_per_sample": total_time/len(times),
            "avg_time_per_token": total_time/total_tokens,
        }, f, indent=4)

if __name__ == "__main__":
    inference()