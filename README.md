# GPU Performance Test for Large Language Models

Benchmark train and inference speed of large language models on GPU devices using Hugging Face Transformers.

Measures tokens per second (TPS) and latency for common LLM inference workloads.

## Prerequisites

* NVIDIA GPU with CUDA support
* CUDA Toolkit installed
* Conda package manager

## Installation

### Create Conda Environment

```bash
conda create -n llm python=3.12 -y
conda activate llm
```

### Install Dependencies

```bash
pip install torch
pip install accelerate datasets transformers
```

## Usage

### Set Model Path

Define the LLM path via environment variable:

```bash
# Example: Use Qwen model  
export MODEL_PATH="/data/Qwen/Qwen2.5-7B-Instruct"  
```

### Run Benchmark

Test train performance.

```bash
accelerate launch train.py
```

Test inference performance.

```bash
python inference.py
```

Let's improve LLM performance together! 🚀