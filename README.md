# Setup scripts
## Setting up VLLM with docker / apptainer
To run on SLURM, you need to install `apptainer`.
After that, generate the `.sif` fils starting from a docker image:
```
apptainer build vllm.sif docker://vllm/vllm-openai
```
Note: since the head note on the `fep` queue is very slow, it is recommented to run the above command on your local 
machine.

## Serving a VLLM model through the OpenAI API
After the `.sif` file is generated, a job file is needed to start running a VLLM model on a compute node.
A sample file is described below (same as in `run_vllm.sbatch`). Make sure to replace `YOUR_HF_TOKEN` with your HuggingFace token.
```aiignore
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
```
To start the model, run the following command:
```
sbatch run_vllm.sbatch
```
For now, the model is only available on the compute node. Therefore, to call the API, you have to connect to that node.
