# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import torch
from omegaconf import DictConfig

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import (
    get_device_name,
)
from verl.workers.engine_workers import ActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["DetachActorWorker"]


class DetachActorWorker(ActorRolloutRefWorker):
    """
    A worker class that extends ActorRolloutRefWorker to support detaching and restoring the actor model.

    This worker facilitates saving the model state to CPU and restoring it, enabling efficient
    resource management and checkpointing in distributed training. It currently supports
    FSDP, FSDP2, and Megatron strategies.
    """

    def __init__(self, config: DictConfig, role: str):
        """
        Initialize the DetachActorWorker.

        Args:
            config: Configuration dictionary.
            role: The role of the worker (e.g., 'actor', 'rollout', 'ref').
        """
        ActorRolloutRefWorker.__init__(self, config, role)
        self._strategy_handlers = None
        self.copy_handler, self.restore_handler = self._get_strategy_handlers()

    def _get_strategy_handlers(self):
        """
        Get the strategy-specific handlers for saving and restoring the model.

        Returns:
            tuple: A tuple containing (save_handler, restore_handler).

        Raises:
            NotImplementedError: If the strategy is not supported.
        """
        if self._strategy_handlers is not None:
            return self._strategy_handlers

        strategy = self.config.actor.strategy

        if strategy in ["fsdp", "fsdp2"]:
            from verl.utils.fsdp_utils import (
                fsdp2_sharded_load_from_cpu,
                fsdp2_sharded_save_to_cpu,
            )

            self._strategy_handlers = (fsdp2_sharded_save_to_cpu, fsdp2_sharded_load_from_cpu)
        elif strategy == "megatron":
            from verl.utils.megatron_utils import (
                copy_megatron_model_to_cpu,
                restore_megatron_model_from_cpu,
            )

            self._strategy_handlers = (copy_megatron_model_to_cpu, restore_megatron_model_from_cpu)
        else:
            raise NotImplementedError(f"Unsupported strategy: {strategy}")

        return self._strategy_handlers

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_model_to_cpu(self, n):
        """
        Save the current model state to CPU memory.

        Args:
            n: Identifier/Key for the saved model state.
        """
        if not hasattr(self, "cpu_saved_models"):
            self.cpu_saved_models = {}

        self.cpu_saved_models[n] = self.copy_handler(self.actor.engine.module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_model_from_cpu(self, n):
        """
        Restore the model state from CPU memory.

        Args:
            n: Identifier/Key for the saved model state to restore.
        """
        if n in self.cpu_saved_models:
            strategy = self.config.actor.strategy

            if strategy in ["fsdp", "fsdp2"]:
                cpu_sharded_state, global_spec = self.cpu_saved_models[n]
                self.restore_handler(self.actor.engine.module, cpu_sharded_state, global_spec)
            else:
                self.restore_handler(self.actor.engine.module, self.cpu_saved_models[n])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_optimizer_to_cpu(self, n):
        """Save the current optimizer and LR scheduler state to CPU memory.

        Each param's state (exp_avg, exp_avg_sq for Adam) is deep-copied to CPU.
        For FSDP2 DTensor params, we extract the local shard via _local_tensor.

        Args:
            n: Identifier/Key for the saved state.
        """
        import copy

        if not hasattr(self, "cpu_saved_optimizers"):
            self.cpu_saved_optimizers = {}

        optimizer = self.actor.engine.optimizer
        if optimizer is None:
            return

        # Deep-copy optimizer state tensors to CPU (sharded — each rank saves its own shard)
        cpu_state = {}
        for param, state in optimizer.state.items():
            saved = {}
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    # For DTensor (FSDP2), extract the local shard
                    raw = v._local_tensor if hasattr(v, "_local_tensor") else v
                    saved[k] = raw.detach().cpu().clone()
                else:
                    saved[k] = copy.deepcopy(v)
            cpu_state[param] = saved

        # Save LR scheduler state
        lr_scheduler = self.actor.engine.lr_scheduler
        lr_state = lr_scheduler.state_dict() if lr_scheduler is not None else None

        # Save per-param-group learning rates (lr and initial_lr). These live on
        # optimizer.param_groups and are NOT part of optimizer.state, so they
        # need to be tracked separately for per-tenant LR support.
        param_group_lrs = [
            {"lr": g.get("lr"), "initial_lr": g.get("initial_lr")} for g in optimizer.param_groups
        ]

        self.cpu_saved_optimizers[n] = (cpu_state, lr_state, param_group_lrs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_optimizer_from_cpu(self, n):
        """Restore optimizer and LR scheduler state from CPU memory.

        For FSDP2 DTensor params, copies data into the existing _local_tensor in-place.

        Args:
            n: Identifier/Key for the saved state to restore.
        """
        if not hasattr(self, "cpu_saved_optimizers") or n not in self.cpu_saved_optimizers:
            return

        optimizer = self.actor.engine.optimizer
        if optimizer is None:
            return

        saved = self.cpu_saved_optimizers[n]
        # Back-compat: old saves stored 2-tuples (cpu_state, lr_state).
        if len(saved) == 3:
            cpu_state, lr_state, param_group_lrs = saved
        else:
            cpu_state, lr_state = saved
            param_group_lrs = None

        # Restore optimizer state tensors to GPU
        for param, state in cpu_state.items():
            if param not in optimizer.state:
                optimizer.state[param] = {}
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    live_v = optimizer.state[param].get(k)
                    if live_v is not None and hasattr(live_v, "_local_tensor"):
                        # DTensor: copy into the existing local shard in-place
                        live_v._local_tensor.copy_(v.to(live_v._local_tensor.device))
                    elif live_v is not None:
                        live_v.copy_(v.to(live_v.device))
                    else:
                        target_device = param._local_tensor.device if hasattr(param, "_local_tensor") else param.device
                        optimizer.state[param][k] = v.to(target_device, non_blocking=True)
                else:
                    optimizer.state[param][k] = v

        # Restore LR scheduler state first (so base_lrs / last_epoch are in
        # place), then apply the saved per-param-group lr / initial_lr on top —
        # these carry the per-tenant learning rate the next optimizer.step()
        # will actually use.
        if lr_state is not None:
            lr_scheduler = self.actor.engine.lr_scheduler
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(lr_state)

        if param_group_lrs is not None:
            for g, saved_g in zip(optimizer.param_groups, param_group_lrs, strict=False):
                if saved_g.get("lr") is not None:
                    g["lr"] = saved_g["lr"]
                if saved_g.get("initial_lr") is not None:
                    g["initial_lr"] = saved_g["initial_lr"]

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_learning_rate(self, lr: float):
        """Override the optimizer's learning rate (and scheduler base_lrs).

        Used in multi-tenant training to give each tenant its own LR. Affects
        all param groups and updates scheduler.base_lrs so that subsequent
        scheduler.step() calls decay/warmup off the new base LR.
        """
        # dispatch_one_to_all wraps scalar args as [val]*world_size; unwrap.
        if isinstance(lr, (list, tuple)):
            lr = lr[0]
        lr = float(lr)

        optimizer = self.actor.engine.optimizer
        if optimizer is None:
            return

        for g in optimizer.param_groups:
            g["lr"] = lr
            g["initial_lr"] = lr

        lr_scheduler = self.actor.engine.lr_scheduler
        if lr_scheduler is not None:
            if hasattr(lr_scheduler, "base_lrs"):
                lr_scheduler.base_lrs = [lr for _ in lr_scheduler.base_lrs]
            if hasattr(lr_scheduler, "_last_lr"):
                lr_scheduler._last_lr = [lr for _ in lr_scheduler._last_lr]

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_cpu_model(self, n):
        """
        Clear the saved model state from CPU memory.

        Args:
            n: Identifier/Key for the saved model state to remove.
        """
        if n in self.cpu_saved_models:
            del self.cpu_saved_models[n]

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_lora_adapter_weights(self):
        """Extract LoRA adapter weights and peft_config from the current model.

        Returns only the LoRA adapter parameters (not the base model), suitable
        for loading into vLLM via TensorLoRARequest.

        Returns:
            tuple: (lora_state_dict, peft_config) where lora_state_dict is a dict
                   of {name: cpu_tensor} and peft_config is the PEFT configuration dict.
                   Returns (None, None) if the model has no LoRA adapter.
        """
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            base_sync_done=True,
        )
        if peft_config is None:
            return None, None
        # PEFTHelper.from_dict (used in hijack__load_adapter) expects a plain dict,
        # but get_per_tensor_param returns the raw LoraConfig object.
        if hasattr(peft_config, "to_dict"):
            peft_config = peft_config.to_dict()
        # Collect the generator into a dict of CPU tensors
        state_dict = {}
        for name, tensor in per_tensor_param:
            if hasattr(tensor, "full_tensor"):
                tensor = tensor.full_tensor()
            state_dict[name] = tensor.detach().cpu()
        return state_dict, peft_config
