#!/bin/bash
#SBATCH --job-name=mt_dapo_0.5b_3tenant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --account=infra01
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --environment=env
#SBATCH --output=logs/multi-tenant/0.5b/mt_dapo_0.5b_3tenant_%j.out


REPO_ROOT="/users/${USER}/scratch/verl"

# Tenant definitions: name:train_file:val_file:lr (comma-separated)
export TENANTS=${TENANTS:-"alice:${HOME}/data/gsm8k/train.parquet:${HOME}/data/gsm8k/test.parquet:1e-5,bob:${HOME}/data/gsm8k/train.parquet:${HOME}/data/gsm8k/test.parquet:5e-5,carol:${HOME}/data/gsm8k/train.parquet:${HOME}/data/gsm8k/test.parquet:1e-5"}

# LoRA config (same for all tenants)
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-32}
export EXP_NAME=${EXP_NAME:-'MT-gsm8k-0.5b-3tenants-rr'}
export STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.1}
export TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-2}
export MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
bash "${REPO_ROOT}/multi_tenant_run.sh" "$@"
