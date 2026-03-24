# Multi-Tenant LoRA Training — Implementation Progress

## Status: Core implementation complete, needs testing

## New Files Created

| File | Purpose |
|------|---------|
| `multi_tenant_run.sh` | Shell script for multi-tenant launch (mirrors `dapo_7b_math_vllm_2_2_lora_run.sh`) |
| `multi_tenant_run.sbatch` | SLURM batch script (mirrors `dapo_7b_vllm_lora_2_2_ra32_st0_ts1.sbatch`) |
| `verl/experimental/fully_async_policy/multi_tenant_main.py` | Entry point — parses tenants, creates per-tenant queues, orchestrates components |
| `verl/experimental/fully_async_policy/multi_tenant_rollouter.py` | Extends `FullyAsyncRollouter` — per-tenant dataloaders, interleaved sample feeding, per-tenant queue routing |
| `verl/experimental/fully_async_policy/multi_tenant_trainer.py` | Extends `FullyAsyncTrainer` — polls per-tenant queues, swaps LoRA adapters, per-tenant weight sync to vLLM |

## Existing Files Modified

| File | Change |
|------|--------|
| `verl/experimental/fully_async_policy/detach_utils.py` | Added `TenantConfig` dataclass, `parse_tenants()` function, `tenant_id` field to `RolloutSample` |
| `verl/experimental/separation/engine_workers.py` | Added `get_lora_adapter_weights()` method to `DetachActorWorker` for extracting LoRA state dict |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | (1) Added `TensorLoRARequest` import; (2) Added `add_tenant_lora()` method to `vLLMHttpServer` for loading per-tenant adapters; (3) Modified `generate()` to support `_lora_int_id` in sampling_params for multi-LoRA routing |
| `verl/experimental/agent_loop/agent_loop.py` | Added `_lora_int_id` passthrough from DataProto `meta_info` to `sampling_params` in `AgentLoopWorker.generate_sequences()` |

## Architecture

```
                    ┌─────────────────────┐
                    │  MultiTenantMain    │
                    │  (parse tenants,    │
                    │   create queues)    │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌────────────┐  ┌──────────────┐  ┌──────────────┐
     │  Tenant A  │  │   Tenant B   │  │   Tenant N   │
     │  Queue     │  │   Queue      │  │   Queue      │
     └─────┬──────┘  └──────┬───────┘  └──────┬───────┘
           │                │                  │
     ┌─────┴────────────────┴──────────────────┴─────┐
     │              MultiTenantRollouter             │
     │  - Per-tenant dataloaders (round-robin)       │
     │  - Tags samples with tenant_id + lora_int_id  │
     │  - Routes to correct tenant queue             │
     │  - vLLM generates with tenant's LoRA adapter  │
     └───────────────────────────────────────────────┘
           │                │                  │
     ┌─────┴────────────────┴──────────────────┴─────┐
     │              MultiTenantTrainer               │
     │  - Polls all tenant queues                    │
     │  - Trains whichever tenant is ready first     │
     │  - Swaps LoRA adapter via save/restore CPU    │
     │  - Syncs updated adapter to vLLM directly     │
     └───────────────────────────────────────────────┘
```

## Key Design Decisions

### Tenant adapter swapping (training side)
Uses existing `save_model_to_cpu` / `restore_model_from_cpu` from `DetachActorWorker`. Each tenant gets a unique version key (`100000 + tenant_index`). The full FSDP2 sharded state is saved/restored per worker — efficient because it's per-shard, not full-gather.

### LoRA adapter sync to vLLM
Bypasses the NCCL checkpoint engine entirely. Instead:
1. Extracts LoRA adapter weights via `get_lora_adapter_weights()` (new method on `DetachActorWorker`)
2. Sends directly to vLLM server handles via Ray RPC `add_tenant_lora()`
3. vLLM creates `TensorLoRARequest` with per-tenant `lora_int_id`

This is efficient because LoRA adapters are tiny (~50MB for rank-32 on 7B).

### Multi-LoRA generation
`_lora_int_id` is injected into `DataProto.meta_info` by the rollouter, flows through `AgentLoopWorker` into `sampling_params`, and is popped by the vLLM server to create the correct `LoRARequest`.

## TODOs / Known Limitations

- [ ] **Testing**: No tests written yet — needs end-to-end test on cluster
- [ ] **Checkpoint resume**: Multi-tenant checkpoint loading not implemented (only saving)
- [ ] **Validation**: Per-tenant validation not implemented (currently uses same validation as single-tenant)
- [ ] **Per-tenant hyperparameters**: Currently all tenants share the same config
- [ ] **Dynamic tenants**: Tenants are fixed at launch; no join/leave support
- [ ] **vLLM `add_lora` via `collective_rpc`**: The `add_tenant_lora` method uses `collective_rpc("add_lora", ...)` which may need testing with vLLM v1 — the API for dynamic LoRA loading may differ

## 2026-03-22 - Hydra CLI tenant parsing fix

- Fixed launch override formatting in [multi_tenant_run.sh](multi_tenant_run.sh) so `multi_tenant.tenants` is passed to Hydra as a single string value.
- Root cause: `multi_tenant.tenants="${TENANTS}"` lets shell consume quotes; Hydra then sees unquoted commas and raises: `Ambiguous value for argument ...`.
- Patch: changed to `multi_tenant.tenants="'${TENANTS}'"` so literal single quotes reach Hydra, preventing comma-based sweep/list parsing.
- Note: the `multistorageclient` "Profile \"\" not found" stack traces are emitted by an optional Hydra config source plugin and are noisy but non-fatal in this run; the fatal stop was the ambiguous `multi_tenant.tenants` override.

## 2026-03-22 - Hydra struct key fix for `multi_tenant`

- New run logs showed a second Hydra failure after quote handling was fixed:
      - `Could not override 'multi_tenant.tenants'`
      - `Key 'multi_tenant' is not in struct`
- Root cause: `fully_async_ppo_trainer.yaml` does not predefine a `multi_tenant` node, and Hydra struct mode rejects creating it via plain override syntax.
- Patch: switched launcher override in [multi_tenant_run.sh](multi_tenant_run.sh) to append syntax:
      - from: `multi_tenant.tenants="'${TENANTS}'"`
      - to: `+multi_tenant.tenants="'${TENANTS}'"`
- Result: Hydra can now create `multi_tenant` dynamically while still treating the comma-separated tenant list as a single string.

## 2026-03-22 - vLLM tenant LoRA load fix (`list` vs `lora_int_id`)

- New cluster run progressed into trainer/rollouter startup and failed during initial tenant adapter load on vLLM:
      - `AttributeError: 'list' object has no attribute 'lora_int_id'`
      - Failure originates in vLLM worker `add_lora` path when processing `collective_rpc` payload.
- Root cause: in some vLLM v1 builds, `collective_rpc("add_lora", ...)` argument conversion can deliver a list-shaped object to worker-side `add_lora`, instead of a LoRARequest-like object.
- Patch in [verl/workers/rollout/vllm_rollout/vllm_async_server.py](verl/workers/rollout/vllm_rollout/vllm_async_server.py):
      - prefer `await self.engine.add_lora(lora_request)` and `await self.engine.remove_lora(lora_int_id)` when available;
      - keep `collective_rpc` as fallback for compatibility.
- This avoids msgspec argument shape conversion issues in the tenant LoRA sync path.

## 2026-03-22 - Switched to debug instrumentation for LoRA RPC

- Per request, replaced the provisional native `engine.add_lora/remove_lora` workaround with explicit debug logging in [verl/workers/rollout/vllm_rollout/vllm_async_server.py](verl/workers/rollout/vllm_rollout/vllm_async_server.py).
- `add_tenant_lora()` now logs:
      - `loaded_loras` runtime type/value,
      - exact `collective_rpc` args shape for `remove_lora` and `add_lora`,
      - `TensorLoRARequest` runtime type and whether `lora_int_id` is present before the RPC call,
      - exception context if `collective_rpc("add_lora", ...)` fails.
- Goal: verify whether request object is transformed into a list before reaching vLLM worker `add_lora`.

## 2026-03-22 - Confirmed list-conversion and added worker shim

- Debug run confirmed `TensorLoRARequest` is correct before RPC in [verl/workers/rollout/vllm_rollout/vllm_async_server.py](verl/workers/rollout/vllm_rollout/vllm_async_server.py), but worker receives a list-like object (`'list' object has no attribute 'lora_int_id'`).
- Added compatibility shim in [verl/workers/rollout/vllm_rollout/utils.py](verl/workers/rollout/vllm_rollout/utils.py):
      - `vLLMColocateWorkerExtension.add_lora()` now unwraps single-item list/tuple payloads before forwarding to `model_runner.add_lora`.
      - `vLLMColocateWorkerExtension.remove_lora()` now does the same for id payloads.
- This keeps `collective_rpc` path and handles vLLM argument conversion behavior in this environment.

## 2026-03-22 - vLLM multi-LoRA sync runtime fix

- New run reached trainer/rollouter startup and failed during initial tenant LoRA load on vLLM with:
      - `AttributeError: 'list' object has no attribute 'lora_int_id'`
      - stack rooted at `vLLMHttpServer.add_tenant_lora()` -> `engine.collective_rpc("add_lora", ...)`
- Root cause 1: `collective_rpc` argument packing for `add_lora/remove_lora` was incompatible with this vLLM runtime path (worker received a list instead of a LoRA request object).
- Patch 1 ([verl/workers/rollout/vllm_rollout/vllm_async_server.py](verl/workers/rollout/vllm_rollout/vllm_async_server.py)):
      - `remove_lora`: switched from kwargs form to positional args form: `args=(lora_int_id,)`
      - `add_lora`: switched from kwargs form to positional args form: `args=(lora_request,)`
- Root cause 2 (next likely blocker): vLLM server args showed `--max_loras 1` while running 2 tenants.
- Patch 2:
      - [verl/workers/rollout/vllm_rollout/vllm_async_server.py](verl/workers/rollout/vllm_rollout/vllm_async_server.py): `max_loras` now reads from `model_config.lora.max_loras` (default `1`).
      - [multi_tenant_run.sh](multi_tenant_run.sh): derives tenant count from `TENANTS` and passes `+actor_rollout_ref.model.lora.max_loras=${max_loras}`.

## 2026-03-22 - Minimal confirmed fix in `vllm_async_server.py`

- Confirmed effective change for
      - `AttributeError: 'list' object has no attribute 'lora_int_id'`
- In [verl/workers/rollout/vllm_rollout/vllm_async_server.py](verl/workers/rollout/vllm_rollout/vllm_async_server.py), `add_tenant_lora()` was simplified to call native engine APIs directly:
      - `self.engine.remove_lora(lora_int_id)`
      - `self.engine.add_lora(lora_request)`
      - with `inspect.isawaitable(...)` guard so both sync/async return styles are handled.
- Why this works: it bypasses the `collective_rpc("add_lora", ...)` argument-conversion path where the request was transformed into a list on worker side.
- Kept complementary capacity fix: `max_loras` is now configurable from `model_config.lora.max_loras` instead of hardcoded `1`.

## 2026-03-23 - Fix TensorLoRARequest downcast via staged tensor cache

- Root cause identified by investigation agent: `engine.add_lora(TensorLoRARequest)` passes through vLLM's internal zmq/msgspec transport (AsyncLLM → EngineCore → workers), which serializes the struct back to plain `LoRARequest` and drops `peft_config`/`lora_tensors`. The hijack then falls back to file loading on `simon_lora_path` → failure.
- Fix uses a two-step approach:
  1. **Stage tensors on workers** before `engine.add_lora()`: call `engine.collective_rpc("stage_lora_tensors", args=(lora_int_id, peft_config, lora_tensors))`. This passes plain `(int, dict, dict)` through pickle (no LoRARequest serialization). Each worker stores the tensors in a module-level dict `_staged_lora_tensors` in `verl/utils/vllm/utils.py`.
  2. **Call `engine.add_lora(plain LoRARequest)`** for engine-level LoRA tracking (needed so `list_loras()` can validate generation requests).
- Changes:
  - [verl/utils/vllm/utils.py](verl/utils/vllm/utils.py): Added `_staged_lora_tensors` dict; hijack now checks it as a fallback when `lora_request` is not `TensorLoRARequest`; replaced second `isinstance` check with `lora_tensors is not None`.
  - [verl/workers/rollout/vllm_rollout/utils.py](verl/workers/rollout/vllm_rollout/utils.py): Added `stage_lora_tensors()` method to `vLLMColocateWorkerExtension`.
  - [verl/workers/rollout/vllm_rollout/vllm_async_server.py](verl/workers/rollout/vllm_rollout/vllm_async_server.py): `add_tenant_lora()` now stages tensors via `collective_rpc` then calls `engine.add_lora(LoRARequest)` (plain, not TensorLoRARequest).

## 2026-03-23 - Fix tensor list-conversion in staged cache

- New run hit: `AttributeError: 'list' object has no attribute 'to'` in `lora_model.py:101 from_lora_tensors` → `loras[module_name].lora_a = tensor.to(device, dtype)`.
- Root cause: `engine.collective_rpc` uses zmq+msgspec for IPC. msgspec doesn't support `torch.Tensor` and serializes them via Python's iteration protocol → tensors arrive in the worker as nested Python lists.
- The staging mechanism itself works (hijack is reached, cache is populated), but tensor values are corrupted.
- Fix: serialize `lora_tensors` dict to bytes via `cloudpickle.dumps()` before passing to `collective_rpc`; deserialize on the worker side with `cloudpickle.loads()`. `bytes` is a native msgspec type, so it passes through the zmq transport correctly.
- Changes:
  - [verl/workers/rollout/vllm_rollout/vllm_async_server.py](verl/workers/rollout/vllm_rollout/vllm_async_server.py): `add_tenant_lora()` serializes `lora_tensors` with `cloudpickle.dumps()` before staging.
  - [verl/workers/rollout/vllm_rollout/utils.py](verl/workers/rollout/vllm_rollout/utils.py): `stage_lora_tensors()` accepts `lora_tensors_bytes: bytes` and deserializes with `cloudpickle.loads()`.

## 2026-03-23 - Fix `TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'`

- Run logs showed all LoRA adapters loaded, then immediately crashed with `TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'` in `detach_utils.py:197` inside `assemble_batch_from_rollout_samples`.
- Root cause: `vllm_async_server.py` initializes `self.global_steps = None` and only updates it via `set_global_steps()` (called from `checkpoint_manager.update_weights`). In single-tenant fully async, the pre-fit `_fit_update_weights()` call sets `global_steps = 0` in vLLM before any generation. In multi-tenant, `MultiTenantTrainer._fit_update_weights()` has an extra guard `if self.active_tenant is None: return` — but `active_tenant` is still `None` at that point (it is set later by `init_tenant_adapters_on_rollout` → `_init_tenant_model_states`). So the initial base-model weight sync is skipped entirely, `global_steps` stays `None` in vLLM, and the first generated samples carry `min_global_steps = max_global_steps = None` in `non_tensor_batch`, causing `abs(None - None)` to crash.
- Fix: in [verl/experimental/fully_async_policy/multi_tenant_trainer.py](verl/experimental/fully_async_policy/multi_tenant_trainer.py), when `active_tenant is None`, still call `checkpoint_manager.update_weights(global_steps=self.current_param_version)` to initialize vLLM's `global_steps`, then return (skipping the LoRA-specific steps that require an active tenant).

## 2026-03-23 - Fix `fit()` guard: MessageQueue client not set

- Run logs showed both tenant LoRAs loaded successfully, then immediately crashed with `ValueError: MessageQueue client not set. Call set_message_queue_client() first.` at `fully_async_trainer.py:397`.
- Root cause: `FullyAsyncTrainerBase.fit()` checks `self.message_queue_client is None`. `MultiTenantTrainer` overrides `set_message_queue_client()` to a no-op and uses `tenant_queue_clients` instead, but never populates `message_queue_client`, so it stays `None`.
- Fix: set `self.message_queue_client = True` at the end of `set_tenant_queue_clients()` to satisfy the guard. The base `_get_samples_from_queue()` (which actually uses `message_queue_client`) is fully overridden by multi-tenant, so the sentinel is never accessed.
- Change: [verl/experimental/fully_async_policy/multi_tenant_trainer.py](verl/experimental/fully_async_policy/multi_tenant_trainer.py): `set_tenant_queue_clients()` sets `self.message_queue_client = True`.

## 2026-03-24 - Per-tenant current_param_version

- `current_param_version` is now per-tenant: loaded from `tenant_param_versions[tenant_name]` at the top of `_fit_update_local_step` and saved back after the update — same pattern as `local_trigger_step` and `global_steps`.
- Removed the redundant `self.tenant_param_versions[tenant_name] = self.current_param_version` write from `_fit_update_weights` (now handled by `_fit_update_local_step`).
- Removed the `tenant_step` indirection in `_fit_update_weights`; all logging uses `self.current_param_version` directly since it is already per-tenant.
- Change: [verl/experimental/fully_async_policy/multi_tenant_trainer.py](verl/experimental/fully_async_policy/multi_tenant_trainer.py).

## 2026-03-24 - Per-tenant staleness limit

- Added per-tenant staleness tracking (`tenant_staleness_samples` dict) in `MultiTenantRollouter`.
- Rewrote `_feed_samples` with per-tenant staleness and queue-full gates checked **before** advancing the dataloader — no data waste when a tenant is skipped. Scheduling is delegated to `_schedule_next_tenants()` (new overrideable hook, default round-robin), fully decoupled from the gating logic.
- When all tenants are gated in a round, `_feed_samples` waits on the condition variable; `reset_staleness` notifies to unblock.
- `_should_pause_generation` now pauses the processor only when **all** tenant queues are full (previously paused if any single queue was full). Per-tenant staleness gating lives entirely in `_feed_samples`.
- `reset_staleness(tenant_id=None)` accepts an optional `tenant_id`; when provided, only that tenant's counter is reset (to its current queue size). Trainer now passes `tenant_name` on each adapter sync.
- Changes: [verl/experimental/fully_async_policy/multi_tenant_rollouter.py](verl/experimental/fully_async_policy/multi_tenant_rollouter.py), [verl/experimental/fully_async_policy/multi_tenant_trainer.py](verl/experimental/fully_async_policy/multi_tenant_trainer.py).

## 2026-03-24 - Per-tenant progress bars and metrics

- Added per-tenant `tqdm` progress bars (one per tenant, `position=i+1`; position 0 is the shared global bar which is no longer ticked in multi-tenant mode to avoid overshooting).
- Each tenant bar has `total=total_training_steps` — tenants train **independently** for the full budget, not sharing it.
- Added per-tenant `MetricsAggregator` instances created in `set_total_train_steps`.
- `_fit_postprocess_step` feeds the per-tenant aggregator with tenant-prefixed keys (e.g. `alice/actor/loss`, `bob/actor/loss`), keeping each tenant's curves independent in the tracker.
- `_fit_update_weights` logs from the per-tenant aggregator at `tenant_global_steps` (per-tenant x-axis), not `current_param_version`.
- `fit()` closes per-tenant bars on training completion.
- Change: [verl/experimental/fully_async_policy/multi_tenant_trainer.py](verl/experimental/fully_async_policy/multi_tenant_trainer.py).

## 2026-03-24 - Per-tenant step counters in log

- `global_steps` and `local_trigger_step` were shared across all tenants in the base class — both incremented on every `fit_step()` regardless of tenant, so with 2 tenants they doubled at the same rate.
- Added `tenant_global_steps` and `tenant_local_trigger_steps` dicts (initialized to `1` per tenant) in `MultiTenantTrainer.__init__`.
- Overrode `_fit_update_local_step()`: restores the current tenant's counters into the shared base-class fields before logging and advancing them; prints `[FullyAsyncTrainer][tenant=<name>]` prefix so each log line is attributed to the correct tenant.
- Overrode `_fit_postprocess_step()`: increments per-tenant `global_steps` rather than the shared counter.
- Result: log now shows `[FullyAsyncTrainer][tenant=alice] global_steps: 1 local_trigger_step: 1 ...` and `[FullyAsyncTrainer][tenant=bob] global_steps: 1 local_trigger_step: 1 ...` independently.
- Change: [verl/experimental/fully_async_policy/multi_tenant_trainer.py](verl/experimental/fully_async_policy/multi_tenant_trainer.py).
