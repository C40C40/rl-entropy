# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Main entry for SCEM trainer.
"""

import os
# Assume placed in verl/trainer/ppo
import sys

import hydra
import ray
from omegaconf import DictConfig

from verl.trainer.ppo.ray_trainer import (  # Adjust imports as needed
    ResourcePoolManager, Role)

from .scem_trainer import RaySCEMTrainer, scem_reward_fn, scem_val_reward_fn


@hydra.main(config_path="conf", config_name="scem_trainer")
def main(config: DictConfig):
    from omegaconf import OmegaConf
    ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    default_runtime_env = {
    "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
    }
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    print(f"ray init kwargs: {ray_init_kwargs}")
    ray.init(**OmegaConf.to_container(ray_init_kwargs))
    
    # Initialize tokenizer, etc. (similar to main_ppo)
    from transformers import AutoTokenizer
    model_path = os.path.expanduser(config.actor_rollout_ref.model.path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Role worker mapping (照搬DAPO/PPO)
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        # Add Critic, Ref if needed
    }

    # Resource pool
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RaySCEMTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        # Other args like reward_fn if needed
        reward_fn=scem_reward_fn,
        val_reward_fn=scem_val_reward_fn
    )

    trainer.init_workers()
    trainer.fit(tokenizer=tokenizer)

if __name__ == "__main__":
    main()