# GPU Performance Test for Large Language Models

Benchmark inference speed of large language models on GPU devices using Hugging Face Transformers.

Measures tokens per second (TPS) and latency for common LLM inference workloads.

## Prerequisites

* NVIDIA GPU with CUDA support
* CUDA Toolkit installed
* Conda package manager

## Installation

### Creating Conda Environment

```bash
conda create -n llm python=3.12 -y
conda activate llm
```

### Install Dependencies

```bash
pip install torch
pip install deepspeed datasets transformers
```

## Usage

### Clone Repository

Clone GitHub repository to your local folder

```bash
git clone https://github.com/Artessay/GPU-Performance-Test.git
cd GPU-Performance-Test
```

### Set Model Path

Define the LLM path via environment variable:

```bash
# Example: Use Qwen model  
export MODEL_PATH="/data/Qwen/Qwen2.5-7B-Instruct"
```

### Run Benchmark

Test train performance.

```bash
deepspeed --num_gpus=8 train.py 
```

Test inference performance.

```bash
python inference.py
```

Let's improve LLM performance together! 🚀