"""Multi-tenant rollouter that extends FullyAsyncRollouter.

Creates per-tenant dataloaders, interleaves samples across tenants,
and routes generated samples to per-tenant MessageQueues.
"""

import asyncio
import time
from pprint import pformat

import numpy as np
import ray

from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    TenantConfig,
    prepare_single_generation_data,
    safe_create_task,
)
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRolllouterBase
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType


@ray.remote(num_cpus=10, max_concurrency=100)
class MultiTenantRollouter(FullyAsyncRolllouterBase):
    """Multi-tenant rollouter: per-tenant data loading, generation with tenant LoRA, per-tenant queues."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
        tenant_configs: list[TenantConfig] = None,
    ):
        if tenant_configs is None or len(tenant_configs) == 0:
            raise ValueError("tenant_configs must be provided and non-empty")

        self.tenant_configs = tenant_configs
        self.tenant_lora_map = {tc.name: tc.lora_int_id for tc in tenant_configs}

        # Use first tenant's data files for parent class init (it creates default dataloaders).
        # We'll override with per-tenant dataloaders below.
        from omegaconf import open_dict

        with open_dict(config):
            config.data.train_files = tenant_configs[0].train_file
            config.data.val_files = tenant_configs[0].val_file

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            device_name=device_name,
        )

        # Create per-tenant dataloaders, overriding the parent's single dataloader
        self._create_tenant_dataloaders(config, tokenizer, processor)

        # Per-tenant queue clients (set later by main)
        self.tenant_queue_clients: dict[str, MessageQueueClient] = {}

        # Per-tenant staleness counters (independent of scheduling strategy)
        self.tenant_staleness_samples: dict[str, int] = {tc.name: 0 for tc in tenant_configs}

    def _create_tenant_dataloaders(self, config, tokenizer, processor):
        """Create per-tenant train dataloaders and validation datasets."""
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        self.tenant_dataloaders = {}
        self.tenant_val_datasets = {}

        for tc in self.tenant_configs:
            train_dataset = create_rl_dataset(
                tc.train_file,
                config.data,
                tokenizer,
                processor,
                max_samples=config.data.get("train_max_samples", -1),
            )
            val_dataset = create_rl_dataset(
                tc.val_file,
                config.data,
                tokenizer,
                processor,
                max_samples=config.data.get("val_max_samples", -1),
            )
            train_sampler = create_rl_sampler(config.data, train_dataset)

            # Create dataloader for this tenant
            from torchdata.stateful_dataloader import StatefulDataLoader

            train_dataloader = StatefulDataLoader(
                dataset=train_dataset,
                batch_size=config.data.gen_batch_size,
                num_workers=config.data.get("dataloader_num_workers", 0),
                sampler=train_sampler,
                drop_last=True,
                collate_fn=collate_fn,
            )

            self.tenant_dataloaders[tc.name] = train_dataloader
            self.tenant_val_datasets[tc.name] = val_dataset

            print(
                f"[MTRollouter] Created dataloader for tenant '{tc.name}': "
                f"train={len(train_dataset)} samples, val={len(val_dataset)} samples"
            )

        # Recalculate total rollout steps based on smallest tenant dataset
        min_steps = min(len(dl) for dl in self.tenant_dataloaders.values())
        # Total steps = per-tenant steps * num_tenants * epochs
        self.total_rollout_steps = min_steps * len(self.tenant_configs) * config.trainer.total_epochs
        if config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[MTRollouter] Total rollout steps (all tenants): {self.total_rollout_steps}")

    async def set_tenant_queue_clients(self, tenant_queue_clients: dict[str, MessageQueueClient]):
        """Set per-tenant message queue clients."""
        async with self.lock:
            self.tenant_queue_clients = tenant_queue_clients

    # Override: use the first tenant's queue as the default (for compatibility)
    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        pass  # Not used in multi-tenant mode

    def _schedule_next_tenants(self, active_tenants: list[TenantConfig]) -> list[TenantConfig]:
        """Scheduling strategy: return ordered list of tenants to try this round.

        Default is round-robin (preserves order of active_tenants).
        Override this method to change the scheduling strategy (priority-based,
        weighted, etc.) without touching staleness or queue-gating logic.
        """
        return list(active_tenants)

    async def _feed_samples(self):
        """Override: feed samples with per-tenant staleness and queue-fullness gating.

        Key properties:
        - Scheduling is delegated to _schedule_next_tenants() (overrideable)
        - Staleness and queue-full checks are per-tenant and independent
        - Dataloader is only advanced when a tenant passes all gates (no data waste)
        - When all tenants are gated, waits on condition variable for reset/drain signal
        """
        for epoch in range(self.config.trainer.total_epochs):
            tenant_iters = {tc.name: iter(self.tenant_dataloaders[tc.name]) for tc in self.tenant_configs}
            active_tenants = list(self.tenant_configs)

            while active_tenants:
                scheduled = self._schedule_next_tenants(active_tenants)
                made_progress = False
                next_active = []

                for tc in scheduled:
                    # Per-tenant staleness gate: skip without advancing the dataloader
                    if (self.max_required_samples is not None and
                            self.tenant_staleness_samples.get(tc.name, 0) >= self.max_required_samples):
                        next_active.append(tc)
                        continue

                    # Per-tenant queue-full gate: skip without advancing the dataloader
                    if self.max_queue_size is not None and self.tenant_queue_clients:
                        queue_size = self.tenant_queue_clients[tc.name].get_statistics_sync()["queue_size"]
                        if queue_size >= self.max_queue_size:
                            next_active.append(tc)
                            continue

                    # All gates passed — pull next batch (dataloader only advanced here)
                    try:
                        batch_dict = next(tenant_iters[tc.name])
                    except StopIteration:
                        print(f"[MTRollouter] Tenant '{tc.name}' exhausted in epoch {epoch}")
                        continue  # exhausted: not added to next_active

                    next_active.append(tc)

                    full_batch = prepare_single_generation_data(batch_dict, self.config)
                    full_batch.meta_info["_lora_int_id"] = tc.lora_int_id
                    sample_id = f"sample_{tc.name}_{epoch}_{self.global_steps}"
                    rollout_sample = RolloutSample(
                        full_batch=full_batch,
                        sample_id=sample_id,
                        epoch=epoch,
                        rollout_status={},
                        tenant_id=tc.name,
                    )

                    await self.pending_queue.put(rollout_sample)
                    self.tenant_staleness_samples[tc.name] = self.tenant_staleness_samples.get(tc.name, 0) + 1
                    made_progress = True

                    if self.global_steps >= self.total_rollout_steps:
                        print(
                            f"[MTRollouter][Feed] Maximum count reached, stopping: "
                            f"{self.global_steps} >= {self.total_rollout_steps}"
                        )
                        await self.pending_queue.put(None)
                        print(f"[MTRollouter][Feed] Sample addition complete, {self.global_steps} samples added")
                        return

                    self.global_steps += 1

                active_tenants = next_active

                # No tenant made progress this round — all are gated.
                # Wait for a staleness reset or queue drain signal.
                if not made_progress and active_tenants:
                    async with self.lock:
                        print(
                            f"[MTRollouter][Feed] All active tenants gated this round, waiting. "
                            f"staleness={dict(self.tenant_staleness_samples)}"
                        )
                        await self.condition.wait()

        # All epochs exhausted
        await self.pending_queue.put(None)
        print(f"[MTRollouter][Feed] Sample addition complete, {self.global_steps} samples added")

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Override: generate with tenant LoRA and route to tenant's queue."""
        tenant_id = rollout_sample.tenant_id

        # Generate — the DataProto already has _lora_int_id in meta_info
        ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
        rollout_sample.full_batch = ret
        rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
            [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
        )
        rollout_sample.rollout_status = await self.get_statistics()

        # Route to the correct tenant's queue
        queue_client = self.tenant_queue_clients[tenant_id]
        success = await queue_client.put_sample(
            sample=ray.cloudpickle.dumps(rollout_sample),
        )
        if success:
            self.total_generated_samples += 1
        else:
            self.dropped_stale_samples += 1
        self.processed_sample_count += 1

    async def fit(self):
        """Override: check tenant queues are set before starting."""
        print("[MTRollouter] Starting MultiTenantRollouter...")

        if not self.tenant_queue_clients:
            raise ValueError("Tenant queue clients not set. Call set_tenant_queue_clients() first.")

        # Set running state
        async with self.lock:
            self.paused = False
            self.running = True

        # Reuse parent's streaming generation logic
        generation_task = safe_create_task(self._streaming_generation_main(), name="generation_task")
        monitor_task = safe_create_task(self._async_monitor_loop(), name="monitor_task")

        try:
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[MTRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[MTRollouter] MultiTenantRollouter fit completed")

    async def _streaming_generation_main(self):
        """Override: send termination signal to ALL tenant queues on completion."""
        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        print(f"[MTRollouter] Start streaming mode, max concurrent samples: {self.max_concurrent_samples}")

        self.feed_task = safe_create_task(self._feed_samples(), name="feed_task")
        self.processor_task = safe_create_task(self._processor_worker(), name="processor_task")

        try:
            done, pending = await asyncio.wait(
                [self.feed_task, self.processor_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.exception():
                    raise task.exception()

            if self.feed_task not in done:
                raise RuntimeError("Processor task exited prematurely")

            print("[MTRollouter] Sample feed completed")
            await self.processor_task
            print("[MTRollouter] Streaming process completed")
            await self.pending_queue.join()
            print("[MTRollouter] pending_queue joined")

        except Exception as e:
            print(f"[MTRollouter] Streaming process exception: {e}")
            raise e

        finally:
            if self.feed_task and not self.feed_task.done():
                self.feed_task.cancel()
                await asyncio.gather(self.feed_task, return_exceptions=True)

            if self.processor_task and not self.processor_task.done():
                self.processor_task.cancel()
                await asyncio.gather(self.processor_task, return_exceptions=True)

            self.feed_task = None
            self.processor_task = None

            # Send finish signal to ALL tenant queues
            for name, client in self.tenant_queue_clients.items():
                await client.put_sample(sample=None)
                print(f"[MTRollouter] Sent termination signal to tenant '{name}' queue")

        async with self.lock:
            self.running = False

    async def _should_pause_generation(self) -> bool:
        """Override: pause the processor only when ALL tenant queues are full.

        Per-tenant staleness and per-tenant queue-fullness are handled upstream in
        _feed_samples (gates before pulling from the dataloader).  Pausing here only
        when *every* queue is full avoids wasting generation capacity on samples that
        would be dropped, while letting tenants with room continue unimpeded.
        """
        if not self.tenant_queue_clients or self.max_queue_size is None:
            return False

        all_full = all(
            client.get_statistics_sync()["queue_size"] >= self.max_queue_size
            for client in self.tenant_queue_clients.values()
        )
        if all_full:
            if not self.paused:
                print(
                    f"[MTRollouter][ShouldPause] All tenant queues full "
                    f"(max={self.max_queue_size}), pausing processor"
                )
            return True

        return False

    async def reset_staleness(self, tenant_id: str | None = None):
        """Override: reset per-tenant staleness counter(s) and update global counter.

        Args:
            tenant_id: If given, reset only that tenant's counter to its current queue
                       size (called after the trainer syncs that tenant's parameters).
                       If None, reset all tenants (backward-compatible).

        The condition is notified so any _feed_samples coroutine waiting on this tenant
        wakes up and re-evaluates whether it can proceed.
        """
        import time

        async with self.lock:
            self.paused = False
            self.condition.notify_all()

            if tenant_id is not None:
                # Reset only the specified tenant's per-tenant counter
                client = self.tenant_queue_clients[tenant_id]
                queue_size = client.get_statistics_sync()["queue_size"]
                self.tenant_staleness_samples[tenant_id] = queue_size
            else:
                # Reset all tenants (backward-compatible path)
                for tc in self.tenant_configs:
                    client = self.tenant_queue_clients[tc.name]
                    queue_size = client.get_statistics_sync()["queue_size"]
                    self.tenant_staleness_samples[tc.name] = queue_size

            # Keep global staleness_samples consistent
            total_queue_size = sum(
                client.get_statistics_sync()["queue_size"] for client in self.tenant_queue_clients.values()
            )
            self.staleness_samples = len(self.active_tasks) + total_queue_size

            timing_raw = {}
            rollout_version_time = time.time() - self.step_start_time
            # If idle_start_time < step_start_time, the rollouter never paused
            # in this window — it was active the entire time.
            if self.idle_start_time >= self.step_start_time:
                rollout_active_time = self.idle_start_time - self.step_start_time
            else:
                rollout_active_time = rollout_version_time
            idle_ratio = 1 - rollout_active_time / rollout_version_time if rollout_version_time > 0 else 0.0
            timing_raw["fully_async/rollouter/active_time"] = rollout_active_time
            timing_raw["fully_async/rollouter/version_time"] = rollout_version_time
            timing_raw["fully_async/rollouter/idle_ratio"] = idle_ratio

            print(
                f"[MTRollouter][Public][reset_staleness] tenant_id={tenant_id!r} "
                f"reset staleness_samples to: {self.staleness_samples} "
                f"tenant_staleness={dict(self.tenant_staleness_samples)} "
                f"idle_ratio: {timing_raw['fully_async/rollouter/idle_ratio']:.4f}"
            )
            self.step_start_time = time.time()
        return timing_raw


    async def get_statistics(self) -> dict:
        """Override: include per-tenant queue stats."""
        stats = {
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            "count/total_generated_samples": self.total_generated_samples,
            "count/staleness_samples": self.staleness_samples,
            "count/dropped_stale_samples": self.dropped_stale_samples,
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_queue_size": self.max_queue_size,
            "static/max_concurrent_samples": self.max_concurrent_samples,
        }

        for name, client in self.tenant_queue_clients.items():
            queue_stats = client.get_statistics_sync()
            stats[f"monitor/queue/{name}_queue_size"] = queue_stats["queue_size"]

        for tc in self.tenant_configs:
            stats[f"count/staleness_{tc.name}"] = self.tenant_staleness_samples.get(tc.name, 0)

        return stats
