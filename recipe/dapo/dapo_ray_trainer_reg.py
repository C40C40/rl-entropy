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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

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
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip

# === 定义关键字和权重分配 (中英文混合) ===
LOGIC_KEYWORDS = [
    # 中文逻辑
    "因为", "所以", "因此", "∴", "=>", "=", "即", "推出", "则", "得",
    # 英文逻辑
    "because", "since", "as", "therefore", "thus", "hence", "so", "consequently",
            "if", "then", "provided", "assuming", "implies", "follows",
            "however", "but", "nevertheless", "nonetheless", "yet", "although", "though", "despite", "whereas",
            "and", "or", "also", "furthermore", "moreover", "next", "subsequently",
            "let", "assume", "consider", "suppose", "equals", "is", "are", "given", "known",
            "step", "first", "second"
]

ANSWER_PATTERNS = [
    # 中文答案
    "Answer:", "答案", "最终结果", "答",
    # 英文答案
    "answer", "final result"
]

FORMAT_WORDS = [
    # 中文结构词
    "首先", "然后", "综上", "最后", "故", "由此",
    # 英文结构词
    "first", "next", "then", "finally", "in summary", "conclusion"
]


def assign_entropy_weight(token: str, cfg) -> float:
    """根据token内容分配熵权重"""
    kw_cfg = cfg.algorithm.entropy_budget.keyword_weight
    t = token.lower().strip()  # 🔑 统一小写，避免大小写不匹配

    if any(pat in t for pat in LOGIC_KEYWORDS):
        return kw_cfg.logic
    elif any(pat in t for pat in ANSWER_PATTERNS):
        return kw_cfg.answer
    elif any(pat in t for pat in FORMAT_WORDS):
        return kw_cfg.format
    else:
        return kw_cfg.normal


def apply_entropy_budget_loss(batch, tokenizer, cfg):
    if not cfg.algorithm.entropy_budget.enable:
        return torch.tensor(0.0, device=batch.batch["input_ids"].device)

    entropys = batch.batch["entropys"]          # [bsz, seq_len_ent]
    input_ids = batch.batch["input_ids"]        # [bsz, seq_len_in]
    response_mask = batch.batch["response_mask"]

    seq_len = entropys.size(1)
    input_ids = input_ids[:, :seq_len]
    response_mask = response_mask[:, :seq_len]

    # decode & assign weights
    weights = []
    for row in input_ids.tolist():
        row_weights = [assign_entropy_weight(tokenizer.decode([tok]), cfg) for tok in row]
        weights.append(row_weights)
    weights = torch.tensor(weights, device=entropys.device, dtype=entropys.dtype)

    h_baseline = (entropys * response_mask).sum() / (response_mask.sum() + 1e-8)
    
    lam = cfg.algorithm.entropy_budget.get("lam", 1.0)
    beta = cfg.algorithm.entropy_budget.get("beta", 0.05)
    low_entropy_penalty = torch.sum(weights * torch.relu(h_baseline - entropys) * response_mask) / (response_mask.sum() + 1e-8)
    high_entropy_penalty = torch.sum(weights * torch.relu(entropys - h_baseline) * response_mask) / (response_mask.sum() + 1e-8)
    loss = lam * low_entropy_penalty + beta * high_entropy_penalty

    return loss

class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
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

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                is_last_step = self.gen_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]

                        # === 先算熵预算正则 ===
                        tmp_batch = batch.union(old_log_prob)   # 临时合并，保证有 entropys
                        entropy_budget_loss = apply_entropy_budget_loss(tmp_batch, self.tokenizer, self.config)
                        metrics["actor/entropy_budget_loss"] = float(entropy_budget_loss.detach().cpu().item())
                        batch.meta_info["entropy_budget_loss"] = entropy_budget_loss

                        # === 再做聚合熵 logging ===
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)

                        # === 轻量级关键 token logging ===
                        key_token_info = []
                        input_ids = batch.batch["input_ids"].tolist()
                        entropy_vals = entropys.tolist()
                        response_masks = response_masks.tolist()

                        for row_ids, row_ent, row_mask in zip(input_ids, entropy_vals, response_masks):
                            row_tokens = [self.tokenizer.decode([tid], skip_special_tokens=True).lower().strip()
                                        for tid in row_ids]
                            for tok, ent, m in zip(row_tokens, row_ent, row_mask):
                                if m == 0:   # 只记录 response 部分
                                    continue
                                category = None
                                if any(kw in tok for kw in LOGIC_KEYWORDS):
                                    category = "logic"
                                elif any(kw in tok for kw in ANSWER_PATTERNS):
                                    category = "answer"
                                elif any(kw in tok for kw in FORMAT_WORDS):
                                    category = "format"
                                if category:
                                    key_token_info.append({
                                        "token": tok,
                                        "entropy": float(ent),
                                        "category": category
                                    })

                        if key_token_info:
                            metrics["key_tokens"] = key_token_info
                            metrics["key_tokens_avg_entropy"] = float(
                                np.mean([kt["entropy"] for kt in key_token_info])
                            )

                        # === 最后再 pop 掉，避免污染 batch ===
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                             # === 熵预算正则安全融合 ===
                            if "entropy_budget_loss" in batch.meta_info:
                                # 记录到 metrics
                                actor_output.meta_info["metrics"]["actor/entropy_budget_loss"] = float(
                                    batch.meta_info["entropy_budget_loss"].detach().cpu().item()
                                )

                                # 安全地加到 loss 上
                                base_loss = actor_output.meta_info.get("loss", 0.0)
                                actor_output.meta_info["loss"] = base_loss + batch.meta_info["entropy_budget_loss"]

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing
                
                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
    
                safe_metrics = {}
                for k, v in metrics.items():
                    if torch.is_tensor(v):
                        if v.numel() == 1:  # 标量 tensor
                            safe_metrics[k] = v.detach().cpu().item()
                    elif isinstance(v, (int, float, np.floating, np.integer)):
                        safe_metrics[k] = float(v)
                    # 其他 (list/dict/np.array) 不写入 TensorBoard

                logger.log(data=safe_metrics, step=self.global_steps)

                import hashlib
                import json
                import os

                log_dir = os.path.join(self.config.trainer.default_local_dir, "json_logs")
                os.makedirs(log_dir, exist_ok=True)
                log_file_path = os.path.join(log_dir, "all_steps.json")

                if batch is None or len(batch) == 0:
                    print(f"Warning: no valid batch to log at step {self.global_steps}")
                else:
                    if self.global_steps % 10 == 0 or self.global_steps == 1:
                        # 读取已有记录
                        if os.path.exists(log_file_path):
                            with open(log_file_path, "r", encoding="utf-8") as f:
                                all_records = json.load(f)
                        else:
                            all_records = []

                        # 初始化去重集合
                        if not hasattr(self, "logged_prompts"):
                            self.logged_prompts = set()

                        input_ids_list = batch.batch["input_ids"].tolist()
                        response_mask_list = batch.batch["response_mask"].tolist()
                        entropys_list = batch.batch.get("entropys", torch.zeros_like(batch.batch["input_ids"])).tolist()

                        for idx, (input_ids, response_mask, entropys_row) in enumerate(
                            zip(input_ids_list, response_mask_list, entropys_list)
                        ):
                            prompt_tokens = [tid for tid, m in zip(input_ids, response_mask) if m == 0]
                            resp_tokens = [tid for tid, m in zip(input_ids, response_mask) if m == 1]

                            prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                            response_text = self.tokenizer.decode(resp_tokens, skip_special_tokens=True)

                            prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()
                            if prompt_hash in self.logged_prompts:
                                continue
                            self.logged_prompts.add(prompt_hash)

                            # key token 提取
                            key_token_info = []
                            row_tokens = [self.tokenizer.decode([tid], skip_special_tokens=True).lower().strip() for tid in input_ids]
                            for tok, ent, m in zip(row_tokens, entropys_row, response_mask):
                                if m == 0:   # 只记录 response 部分
                                    continue
                                category = None
                                if any(kw in tok for kw in LOGIC_KEYWORDS):
                                    category = "logic"
                                elif any(kw in tok for kw in ANSWER_PATTERNS):
                                    category = "answer"
                                elif any(kw in tok for kw in FORMAT_WORDS):
                                    category = "format"
                                if category:
                                    key_token_info.append({
                                        "token": tok,
                                        "entropy": float(ent),
                                        "category": category
                                    })

                            record = {
                                "step": self.global_steps,
                                "prompt": prompt_text,
                                "response": response_text,
                                "metrics": metrics,
                                "key_tokens": key_token_info
                            }
                            all_records.append(record)

                        # 写回 JSON（覆盖原文件，但内容追加）
                        with open(log_file_path, "w", encoding="utf-8") as f:
                            json.dump(all_records, f, ensure_ascii=False, indent=2)
                            
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
