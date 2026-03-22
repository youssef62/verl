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
