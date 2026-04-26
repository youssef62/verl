# Multi-Tenant LoRA Training — Design Document

## Overview

Extend `verl.experimental.fully_async_policy` to support multi-tenant training where multiple users each train their own LoRA adapter on top of a shared base model, within a single SLURM job.

## Architecture Decisions

### Rollout Engine
- **Single vLLM engine with multi-LoRA serving**
- All tenant LoRA adapters are loaded simultaneously in the vLLM engine
- Base model is loaded once and shared across all tenants
- Rollout requests are tagged with `tenant_id` to select the correct adapter

### Training Workers
- **Time-sliced across tenants** on a shared pool of training GPUs
- Whichever tenant's queue has enough samples ready trains next
- LoRA adapter swapping happens on the training GPUs between tenants
- **All adapters kept in GPU memory** (no CPU offload) — rank-32 LoRA on 7B ≈ ~50MB per adapter, negligible overhead

### Message Queues
- **Per-tenant MessageQueue** (one Ray actor per tenant)
- Each tenant has its own independent queue
- Trainer polls all queues and picks the first tenant with enough samples (`require_batches` threshold)

### Weight Sync
- **Per-tenant adapter sync** — only LoRA weights are synced (already the default behavior)
- Base model synced once at startup, never again
- Syncing tenant A's adapter does not block tenant B's rollout
- Uses existing `TensorLoRARequest` mechanism in vLLM

### Staleness
- **Global staleness tracking** — shared across all tenants
- Single `staleness_threshold` applies to the system as a whole

### Hyperparameters & Reward
- **Same hyperparameters** across all tenants (LoRA rank, learning rate, etc.)
- **Same reward function** (`dapo`) across all tenants
- Per-tenant config may be added later

### Datasets
- **Different datasets per tenant** — each tenant specifies their own `train_files` and `val_files`

### Tenants
- **Fixed at launch** — tenant set defined in the sbatch/shell script, cannot change mid-training
- Arbitrary number of tenants supported, starting with 2 for initial implementation

### Tenant Specification Format
Tenants are defined as a comma-separated list in the shell script:
```bash
TENANTS="alice:~/data/alice_train.parquet:~/data/alice_val.parquet,bob:~/data/bob_train.parquet:~/data/bob_val.parquet"
```
Format: `<tenant_name>:<train_file>:<val_file>` separated by commas.

### Checkpointing
- **Per-tenant adapter saves** — each tenant's LoRA adapter checkpointed independently
- Saved to `<CKPTS_DIR>/<tenant_name>/global_step_<version>/`
- Simplest implementation: save after each training step per tenant

### Infrastructure
- **Single SLURM job** launches the multi-tenant coordinator
- One sbatch file, one run script
- Same cluster setup as existing experiments (SCITAS, 4 GPUs, etc.)

## Components to Modify

1. **`fully_async_main.py`** — Parse tenant list, create per-tenant queues and data loaders, pass tenant config to trainer and rollouter
2. **`fully_async_trainer.py`** — Poll multiple queues, swap active LoRA adapter per tenant, per-tenant checkpointing
3. **`fully_async_rollouter.py`** — Tag rollout requests with tenant_id, feed samples to correct per-tenant queue
4. **`message_queue.py`** — No structural change needed (one instance per tenant)
5. **`agent_loop/agent_loop.py`** — Pass tenant_id / LoRA adapter identifier in generation requests
6. **New: experiment shell script (.sh) and sbatch file** — Multi-tenant launch configuration

## Data Flow

```
Tenant A data ──► Rollouter ──► vLLM (adapter A) ──► MessageQueue A ──┐
                                                                       ├──► Trainer (time-sliced)
Tenant B data ──► Rollouter ──► vLLM (adapter B) ──► MessageQueue B ──┘
                                                                          │
                                                              ┌───────────┤
                                                              ▼           ▼
                                                     Sync adapter A   Sync adapter B
                                                     to vLLM          to vLLM
```

## Metrics

Metrics are split into two categories: **system-level** (shared resources) and **per-tenant** (training quality).

### System-Level Metrics (logged every fit step at `total_fit_steps`)

These measure shared infrastructure performance — the trainer and rollouter serve all tenants, so their timing is inherently global.

| Metric | What it measures |
|--------|-----------------|
| `timing_s/gen` | Wall-clock time the trainer waited in `_fit_generate()` polling tenant queues until *any* tenant had enough samples. This is trainer idle time. |
| `timing_s/step` | Total wall-clock time for one `fit_step` (gen + compute + update). |
| `timing_s/ref` | Time computing reference policy log probs. |
| `timing_s/values` | Time computing critic values. |
| `timing_s/adv` | Time computing advantages. |
| `timing_s/update_critic` | Time for critic gradient update. |
| `timing_s/update_actor` | Time for actor (LoRA) gradient update. |
| `timing_s/param_sync` | Time syncing updated LoRA weights to vLLM (only on param_sync steps). |
| `fully_async/trainer/idle_ratio` | `timing_s/gen / timing_s/step` — fraction of step time spent waiting for samples. |
| `fully_async/total_wait_time` | Cumulative queue wait time within sample collection. |
| `perf/throughput` | Tokens/sec/GPU across all tenants. |

**X-axis:** `total_fit_steps` — a monotonically increasing counter shared across all tenants (step 1 might be tenant A, step 2 tenant B, etc.).

### Rollouter Metrics (logged at `current_param_version` on param_sync)

The rollouter is a single shared actor generating for all tenants interleaved. Its time cannot be split per-tenant.

| Metric | What it measures |
|--------|-----------------|
| `fully_async/rollouter/active_time` | Time the rollouter spent actively generating (for all tenants) since the last `reset_staleness` call. If the rollouter never paused, this equals `version_time`. |
| `fully_async/rollouter/version_time` | Wall-clock time since the last `reset_staleness` call. |
| `fully_async/rollouter/idle_ratio` | `1 - active_time / version_time` — fraction of time the rollouter was paused waiting for staleness to be reset. |

### Per-Tenant Data Metrics (logged at `{tenant}/...` at that tenant's `global_steps`)

These measure training quality for each tenant independently. They are accumulated in a per-tenant `MetricsAggregator` and flushed when that tenant hits `param_sync`.

| Metric pattern | What it measures |
|----------------|-----------------|
| `{tenant}/critic/score/{mean,max,min}` | Reward model scores for this tenant's batches. |
| `{tenant}/critic/rewards/{mean,max,min}` | Shaped rewards. |
| `{tenant}/critic/advantages/{mean,max,min}` | GAE advantages. |
| `{tenant}/actor/entropy` | Policy entropy. |
| `{tenant}/actor/pg_loss` | Policy gradient loss. |
| `{tenant}/actor/pg_clipfrac` | PPO clip fraction. |
| `{tenant}/response_length/{mean,max,min}` | Generated response lengths. |
| `{tenant}/training/global_step` | This tenant's own step counter. |
| `{tenant}/active_tenant_lora_id` | LoRA adapter ID used for this tenant. |
| `{tenant}/fully_async/count/stale_trajectory_processed` | Stale samples processed for this tenant. |

**X-axis:** `tenant_global_steps[tenant_name]` — each tenant has its own independent step counter, so tenant A at step 10 means 10 training steps were performed on tenant A's data, regardless of how many steps tenant B took.
