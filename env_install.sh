#!/bin/bash

echo "1. install inference frameworks and pytorch"
pip install "vllm==0.8.4" "torch==2.6.0"


echo "2. install basic packages for train"
pip install "transformers>=4.51.0" accelerate deepspeed datasets


# echo "3. install FlashAttention and FlashInfer"
# # Install flash-attn-2.7.4.post1 (cxx11abi=False)
# wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && \
#     pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# # Install flashinfer-0.2.2.post1+cu124 (cxx11abi=False)
# wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl && \
#     pip install flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl


echo "Successfully installed all packages"