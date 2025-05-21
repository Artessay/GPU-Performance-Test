import json
import time
import torch

def test_gpu_performance(device_id):
    device = torch.device(f'cuda:{device_id}')

    # Create two large matrices
    size = 10000 # 36000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm-up
    for _ in range(100):
        c = torch.matmul(a, b)

    # Start timing
    start_time = time.time()
    for _ in range(1000):
        c = torch.matmul(a, b)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"GPU {device_id} took {elapsed_time:.4f} seconds to run 1000 matrix multiplications.")
    return elapsed_time


if __name__ == "__main__":
    # Check if there are available GPUs
    assert torch.cuda.is_available(), "No available GPUs were found."
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} available GPUs.")
    
    total_time = 0.0
    for i in range(num_gpus):
        total_time += test_gpu_performance(i)
    print(f"Average time taken by all GPUs: {total_time / num_gpus:.4f} seconds.")

    result_path = f"data/{torch.cuda.get_device_name(0)}_matrix_{torch.cuda.device_count()}.json".replace(" ", "_")
    with open(result_path, 'w') as f:
        json.dump({
            "samples": num_gpus,
            "avg_time_per_sample": total_time/num_gpus,
        }, f, indent=4)