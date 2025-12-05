#!/bin/bash
#SBATCH --job-name=vllm_qwen_serve
#SBATCH --output=vllm_qwen_serve_%j.log
#SBATCH --error=vllm_qwen_serve_%j.err
#SBATCH --partition=ml
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

PORT=8000

apptainer exec \
    --nv \
    --env HF_TOKEN=$YOUR_HF_TOKEN \
    --bind $HOME/.cache/huggingface:/root/.cache/huggingface \
    vllm.sif \
    python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --quantization bitsandbytes \
        --max_model_len 218416 \
        --port $PORT \
        --host 0.0.0.0