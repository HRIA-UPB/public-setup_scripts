#!/bin/bash
#SBATCH --job-name=verl_gsm8k_ppo
#SBATCH --output=verl_gsm8k_%j.log
#SBATCH --error=verl_gsm8k_%j.err
#SBATCH --partition=ml
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ========================
# Environment setup
# ========================
export PYTHONUNBUFFERED=1
export PYTHONPATH=/export/home/acs/prof/andrei.mihalea/workspace/verl
#export CUDA_VISIBLE_DEVICES=0
unset ROCR_VISIBLE_DEVICES
export WANDB_API_KEY=<insert-w&b-api-key>

# ========================
# Paths
# ========================
IMG_PATH=$HOME/verl.sif
WORKDIR=$HOME/workspace/verl
DATADIR=$HOME/data/gsm8k
YOUR_PROJECT_NAME=test_verl
YOUR_RUN_NAME=run1

# Make sure data dir exists
mkdir -p $DATADIR

# ========================
# Step 1: Preprocess GSM8K dataset
# ========================
# singularity exec --nv \
#   --bind $HOME:/workspace \
#   $IMG_PATH \
#   python3 workspace/verl/examples/data_preprocess/gsm8k.py \
#   --local_save_dir /workspace/data/gsm8k

# ========================
# Step 2: Download base model
# ========================
# singularity exec --nv \
#   --bind $HOME:/workspace \
#   $IMG_PATH \
#   python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"
#
ssh -N -R 8265:localhost:8265 $USER@$(hostname -f) &

# ========================
# Step 3: PPO training
# ========================
singularity exec --nv \
  --bind $HOME:/workspace/verl \
  $IMG_PATH \
  python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15  \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$YOUR_PROJECT_NAME \
    trainer.experiment_name=$YOUR_RUN_NAME | 2>&1 tee verl_demo.log \
