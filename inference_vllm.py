import os
import json
import time
import torch
from vllm import LLM, SamplingParams
from transformers import set_seed, AutoTokenizer

# Configuration parameters
model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
data_path = "data/alpaca_zh_demo.json"

assert torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
result_path = f"data/{torch.cuda.get_device_name(0)}_vllm_{num_gpus}.json".replace(" ", "_")

def inference():
    set_seed(42)

    # Load model with vLLM
    model = LLM(model=model_path, tensor_parallel_size=num_gpus)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Read dataset
    with open(data_path, 'r') as f:
        dataset = json.load(f)

    prompts = []
    for i, item in enumerate(dataset):
        # Construct instruction and input messages
        messages = [
            {"role": "system", "content": item["instruction"]},
            {"role": "user", "content": item["input"]},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompts.append(text)

    # Configure sampling parameters
    sampling_params = SamplingParams(
        max_tokens=1024,
        seed=42,
        temperature=0.0,
    )

    # Perform inference
    start_time = time.time()
    outputs = model.generate(prompts, sampling_params)
    total_time = time.time() - start_time

    # Extract response and token count
    total_tokens = sum([len(output.outputs[0].token_ids) for output in outputs])

    num_samples = len(dataset)
    print(f"\nTotal samples: {num_samples}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per sample: {total_time/num_samples:.2f}s")
    print(f"Average time per token: {total_time/total_tokens:.4f}s")

    # Save results
    with open(result_path, 'w') as f:
        json.dump({
            "samples": len(dataset),
            "num_tokens": total_tokens,
            "model_name": model_path.split('/')[-1],
            "avg_time_per_sample": total_time/num_samples,
            "avg_time_per_token": total_time/total_tokens,
        }, f, indent=4)

if __name__ == "__main__":
    inference()