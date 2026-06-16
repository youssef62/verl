# Multi-Tenant LoRA RL Training

Self-contained experimental module for **multi-tenant** reinforcement learning: multiple tenants each
train their own **LoRA adapter** on top of a single shared base model, concurrently, within one job.

It is a fully decoupled fork of the fully-async (decoupled Trainer / Rollouter / MessageQueue)
architecture — see `verl/experimental/fully_async_policy` for the single-tenant original. This module
vendors the base trainer/rollouter/message-queue/agent-loop pieces it needs so it has **no dependency
on `fully_async_policy`** (it still imports `verl` core, including `verl.experimental.separation` and
`verl.experimental.agent_loop`).

## Architecture

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

- **Rollout:** a single vLLM engine with multi-LoRA serving. The base model is loaded once and shared;
  all tenant adapters are loaded simultaneously. Generation requests are tagged with `tenant_id` (mapped
  to a unique `lora_int_id`) to select the right adapter.
- **Training:** time-sliced across tenants on a shared pool of training GPUs. Whichever tenant's queue
  has enough samples ready (`require_batches` threshold) trains next; the LoRA adapter + optimizer state
  is swapped between tenants on the training GPUs. Optimizer state is preserved per tenant across swaps.
- **Message queues:** one `MessageQueue` Ray actor per tenant; the trainer polls all of them.
- **Weight sync:** only LoRA adapter weights are synced (per tenant, via the NCCL checkpoint engine path).
  The base model is synced once at startup. Syncing tenant A's adapter does not block tenant B's rollout.
- **Scheduling:** the rollouter interleaves tenants `round_robin` or fills one tenant at a time (`burst`).
- **Metrics:** per-tenant metrics/progress bars in wandb; system-level timing/perf metrics
  (see `is_system_metric` in `detach_utils.py`) are logged once per step, not attributed to a tenant.

## Module layout

| File | Role |
|---|---|
| `multi_tenant_main.py` | Hydra entry point + `MultiTenantTaskRunner` orchestrator |
| `multi_tenant_trainer.py` | `MultiTenantTrainer` — per-tenant queues, adapter swapping, per-tenant sync |
| `multi_tenant_rollouter.py` | `MultiTenantRollouter` — per-tenant dataloaders, tenant-tagged generation, per-tenant queues |
| `base_trainer.py` / `base_rollouter.py` | vendored base classes (`DecoupledTrainerBase`, `DecoupledRollouterBase`) |
| `message_queue.py` | vendored FIFO `MessageQueue` / `MessageQueueClient` |
| `detach_utils.py` | batch assembly, metrics aggregation, **`TenantConfig` / `parse_tenants` / `is_system_metric`** |
| `agent_loop/` | vendored `DecoupledAgentLoopManager` (resumable generation) |
| `config/multi_tenant_ppo_trainer.yaml` | Hydra config (extends `ppo_trainer`, adds `async_training` + `multi_tenant` sections) |
| `shell/` | example launch scripts (`multi_tenant_run.sh`, `*.sbatch`) |

## Tenant configuration

A tenant is defined by `name`, `train_file`, `val_file`, and an auto-assigned `lora_int_id` (1-based),
with optional per-tenant `learning_rate` and `max_response_length` overrides. Supply tenants in any of
three ways (parsed by `parse_tenants` in `detach_utils.py`):

**Inline YAML list** (in the config or an override):
```yaml
multi_tenant:
  tenants:
    - name: alice
      train_file: /data/alice_train.parquet
      val_file: /data/alice_val.parquet
      learning_rate: 1e-5        # optional
      max_response_length: 4096  # optional
    - name: bob
      train_file: /data/bob_train.parquet
      val_file: /data/bob_val.parquet
```

**External YAML file:** `+multi_tenant.config_path=/path/to/tenants.yaml` (file has a top-level `tenants:` list).

**Legacy colon-string** (used by `shell/multi_tenant_run.sh`):
```
+multi_tenant.tenants='alice:train.parquet:val.parquet:1e-5,bob:train.parquet:val.parquet:5e-5'
```
Format per entry: `name:train_file:val_file[:lr[:max_response_length]]`, comma-separated.

`scheduling` (`round_robin` | `burst`) controls how the rollouter interleaves tenants.

## Launch

```bash
TENANTS="alice:${HOME}/data/gsm8k/train.parquet:${HOME}/data/gsm8k/test.parquet:1e-5,\
bob:${HOME}/data/gsm8k/train.parquet:${HOME}/data/gsm8k/test.parquet:5e-5" \
  bash verl/experimental/multi_tenant/shell/multi_tenant_run.sh
```

which runs `python -m verl.experimental.multi_tenant.multi_tenant_main` with
`config_name=multi_tenant_ppo_trainer`. See `shell/` for FSDP and SLURM (`.sbatch`) examples, including a
single-tenant baseline (`single_tenant_run_0.5b.sbatch`) and burst-scheduling variants.
