#!/bin/bash
#SBATCH --job-name=st_dapo_7b_1tenant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --account=infra01
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --environment=env
#SBATCH --output=logs/single-tenant/7b/st_dapo_7b_1tenant_%j.out


REPO_ROOT="/users/${USER}/scratch/verl"

# Tenant definitions: name:train_file:val_file (comma-separated)
# export TENANTS="alice:${HOME}/data/alice_train.parquet:${HOME}/data/alice_val.parquet,bob:${HOME}/data/bob_train.parquet:${HOME}/data/bob_val.parquet"

# LoRA config (same for all tenants)
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-32}
export EXP_NAME=${EXP_NAME:-'ST-dapo-7b-1tenants'}
export STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.0}
export TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-1}

export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export UPDATE_WEIGHTS_BUCKET_MEGABYTES=2048


TRAIN_FILE=${TRAIN_FILE:-"${HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${HOME}/data/aime-2024.parquet"}

# change these with dapo math files for quick testing
export TENANTS=${TENANTS:-"alice:${HOME}/data/dapo-math-17k.parquet:${HOME}/data/aime-2024.parquet:1e-5"}

bash "${REPO_ROOT}/multi_tenant_run.sh" "$@"
