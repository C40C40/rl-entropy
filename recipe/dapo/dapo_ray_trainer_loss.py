# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import numpy as np
import torch
from tqdm import tqdm
import os

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator, RayPPOTrainer, apply_kl_penalty,
    compute_advantage, compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


# === 熵预算相关 ===
LOGIC_KEYWORDS = ["因为", "所以", "因此", "∴", "=>", "="]
ANSWER_PATTERNS = ["Answer:", "答案", "最终结果", "答"]
FORMAT_WORDS = ["首先", "然后", "综上", "最后"]

def assign_entropy_weight(token: str, cfg) -> float:
    kw_cfg = cfg.algorithm.entropy_budget.keyword_weight
    if any(pat in token for pat in LOGIC_KEYWORDS):
        return kw_cfg.logic
    elif any(pat in token for pat in ANSWER_PATTERNS):
        return kw_cfg.answer
    elif any(pat in token for pat in FORMAT_WORDS):
        return kw_cfg.format
    else:
        return kw_cfg.normal

def apply_entropy_budget_loss(batch, tokenizer, cfg):
    if not cfg.algorithm.entropy_budget.enable:
        return torch.tensor(0.0, device=batch.batch["input_ids"].device)

    entropys = batch.batch["entropys"]
    input_ids = batch.batch["input_ids"]
    response_mask = batch.batch["response_mask"]

    budget_total = cfg.algorithm.entropy_budget.budget_total
    weights = []
    for row in input_ids.tolist():
        row_weights = [assign_entropy_weight(tokenizer.decode([tok]), cfg) for tok in row]
        weights.append(row_weights)
    weights = torch.tensor(weights, device=entropys.device, dtype=entropys.dtype)

    actual_budget = torch.sum(entropys * weights * response_mask)
    loss = ((actual_budget - budget_total) ** 2) / (budget_total + 1e-8)
    return loss


class RayDAPOTrainer(RayPPOTrainer):
    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                # === rollout generation ===
                gen_batch = new_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"],
                                          non_tensor_batch_keys=["raw_prompt_ids"])
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                new_batch = new_batch.union(gen_batch_output)

                # === compute rewards ===
                reward_result = self.reward_fn(new_batch, return_dict=True)
                reward_tensor = reward_result["reward_tensor"]
                new_batch.batch["token_level_scores"] = reward_tensor
                new_batch.batch["token_level_rewards"] = reward_tensor

                batch = new_batch

                # === response mask & log_probs ===
                batch.batch["response_mask"] = compute_response_mask(batch)
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                batch = batch.union(old_log_prob)

                # === 动态熵预算 ===
                entropy_budget_loss = apply_entropy_budget_loss(batch, self.tokenizer, self.config)
                metrics["actor/entropy_budget_loss"] = float(entropy_budget_loss.detach().cpu().item())
                batch.batch["entropy_budget_loss"] = entropy_budget_loss

                # === advantage, critic, actor update ===
                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                )

                if self.use_critic:
                    critic_output = self.critic_wg.update_critic(batch)
                    metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                if self.config.trainer.critic_warmup <= self.global_steps:
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
                if self.global_steps >= self.total_training_steps:
                    progress_bar.close()
                    return
