#!/bin/bash
#SBATCH --job-name=st_dapo_0.5b_1tenant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --account=infra01
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --environment=env
#SBATCH --output=logs/multi-tenant/0.5b/st_dapo_0.5b_1tenant_%j.out

REPO_ROOT="/users/${USER}/scratch/verl"

export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-32}
export EXP_NAME=${EXP_NAME:-'ST-gsm8k-0.5b-1tenant'}
export STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.1}
export TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-2}
export MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"

# Single tenant: alice only
export TENANTS="alice:${HOME}/data/gsm8k/train.parquet:${HOME}/data/gsm8k/test.parquet:1e-5"

bash "${REPO_ROOT}/multi_tenant_run.sh" "$@"
