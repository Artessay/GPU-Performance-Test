# GPU Performance Test for Large Language Models

Benchmark inference speed of large language models on GPU devices using Hugging Face Transformers.

Measures tokens per second (TPS) and latency for common LLM inference workloads.

## Prerequisites

* NVIDIA GPU with CUDA support
* CUDA Toolkit installed
* Conda package manager

## Installation

### Clone Repository

Clone GitHub repository to your local folder.

```bash
git clone https://github.com/Artessay/GPU-Performance-Test.git
cd GPU-Performance-Test
```

### Creating Conda Environment (Optional)

```bash
conda create -n gpu_test python=3.12 -y
conda activate gpu_test
```

### Install Dependencies

```bash
bash ./env_install.sh
```

## Usage

### Download Model

```
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /data/Qwen/Qwen2.5-7B-Instruct
```

### Set Model Path

Define the LLM path via environment variable.

```bash
# Example: Use Qwen model  
export MODEL_PATH="/data/Qwen/Qwen2.5-7B-Instruct"
```

### Run Benchmark

Test matrix multiplication performance.

```bash
python matrix_multiplication.py
``` 

Test train performance under DeepSpeed framework.

```bash
deepspeed --num_gpus=8 train_deepspeed.py 
```

Test inference performance under vLLM framework.

```bash
python inference_vllm.py
```

Test inference performance under transformers framework.

```bash
python inference.py
```

Let's improve LLM performance together! 🚀