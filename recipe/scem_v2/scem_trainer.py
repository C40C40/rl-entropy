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
SCEM Trainer (Self-Consistency Entropy Modification) with Ray-based single controller.
This is an unsupervised RL variant based on DAPO/PPO, modifying rewards based on majority vote and entropy.
"""

import os
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           reduce_metrics)
from verl.trainer.ppo.ray_trainer import (AdvantageEstimator, RayPPOTrainer,
                                          apply_kl_penalty, compute_advantage,
                                          compute_response_mask)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


class RaySCEMTrainer(RayPPOTrainer):
    """
    SCEM Trainer: Unsupervised RL based on majority vote and entropy modification.
    Inherits from RayPPOTrainer, with custom reward and gradient masking in fit.
    """

    def fit(self, tokenizer=None):
        """
        The training loop for SCEM (modified from PPO/DAPO).
        Implements majority vote, entropy-based reward modification, and gradient masking.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking
        print(f"logs in {self.config.trainer.logger}")
        if "wandb" in self.config.trainer.logger:
            self.config.trainer.logger = ['console', 'tensorboard']

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
                    # NOTE: This usually changes the order of the `batch`,
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
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # # SCEM custom logic: Compute majority vote and modify rewards based on entropy
                    # # Group by UID
                    # prompt_uid2responses = defaultdict(list)
                    # prompt_uid2entropys = defaultdict(list)
                    # prompt_uid2rewards = defaultdict(list)
                    # for uid, response, entropy, reward in zip(
                    #     batch.non_tensor_batch["uid"],
                    #     batch.batch["responses"].tolist(),
                    #     entropys.tolist(),
                    #     batch.batch["token_level_rewards"].tolist(),
                    # ):
                    #     prompt_uid2responses[uid].append(tuple(response))  # Use tuple for hashable comparison
                    #     prompt_uid2entropys[uid].append(np.mean(entropy))  # Mean entropy per response
                    #     prompt_uid2rewards[uid].append(reward)  # Token-level rewards

                    # # For each group, find majority vote
                    # majority_entropy_mean = []
                    # minority_entropy_mean = []
                    # majority_ratio = []
                    # optimized_reward_mean = []
                    # for uid in prompt_uid2responses:
                    #     responses = prompt_uid2responses[uid]
                    #     vote_count = Counter(responses)
                    #     majority_resp = vote_count.most_common(1)[0][0]
                    #     majority_count = vote_count[majority_resp]
                    #     ratio = majority_count / len(responses)
                    #     majority_ratio.append(ratio)

                    #     # Indices of majority and minority
                    #     maj_indices = [i for i, r in enumerate(responses) if r == majority_resp]
                    #     min_indices = [i for i in range(len(responses)) if i not in maj_indices]

                    #     # Entropies
                    #     maj_entropies = [prompt_uid2entropys[uid][i] for i in maj_indices]
                    #     min_entropies = [prompt_uid2entropys[uid][i] for i in min_indices]
                    #     majority_entropy_mean.append(np.mean(maj_entropies) if maj_entropies else 0)
                    #     minority_entropy_mean.append(np.mean(min_entropies) if min_entropies else 0)

                    #     # Modify rewards
                    #     lambda_entropy = self.config.algorithm.scem.lambda_entropy  # From config
                    #     for i in maj_indices:
                    #         mean_entropy = np.mean(prompt_uid2entropys[uid][i])
                    #         batch.batch["token_level_rewards"][i] += lambda_entropy * mean_entropy  # Raise for majority
                    #     for i in min_indices:
                    #         mean_entropy = np.mean(prompt_uid2entropys[uid][i])
                    #         batch.batch["token_level_rewards"][i] -= lambda_entropy * mean_entropy  # Lower for minority

                    #     # Optimized reward mean
                    #     optimized_rewards = [batch.batch["token_level_rewards"][j].mean().item() for j in range(len(responses))]
                    #     optimized_reward_mean.append(np.mean(optimized_rewards))
                    
                    # SCEM custom logic: Compute majority vote and modify rewards based on entropy
                    # Group by UID
                    prompt_uid2responses = defaultdict(list)
                    prompt_uid2entropys = defaultdict(list)
                    prompt_uid2rewards = defaultdict(list)
                    import re

                    # Assume tokenizer is accessible; adjust based on your setup (e.g., self.actor_rollout_wg.actor.tokenizer)
                    if tokenizer is None:
                        # Initialize tokenizer, etc. (similar to main_ppo)
                        from transformers import AutoTokenizer
                        model_path = os.path.expanduser(self.config.actor_rollout_ref.model.path)
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        
                    for uid, response, entropy, reward in zip(
                        batch.non_tensor_batch["uid"],
                        batch.batch["responses"].tolist(),
                        entropys.tolist(),
                        batch.batch["token_level_rewards"].tolist(),
                    ):
                        # Decode response to text
                        text = tokenizer.decode(response, skip_special_tokens=True)
                        # Extract final answer using regex for \boxed{...}
                        match = re.search(r'\\boxed\{(.*?)\}', text)
                        final_answer = match.group(1) if match else "NO_ANSWER"  # Fallback if no boxed answer found
                        
                        prompt_uid2responses[uid].append(final_answer)  # Use final answer for comparison
                        prompt_uid2entropys[uid].append(np.mean(entropy))  # Mean entropy per response
                        prompt_uid2rewards[uid].append(reward)  # Token-level rewards

                    # For each group, find majority vote
                    majority_entropy_mean = []
                    minority_entropy_mean = []
                    majority_ratio = []
                    optimized_reward_mean = []
                    for uid in prompt_uid2responses:
                        responses = prompt_uid2responses[uid]  # Now list of final_answers
                        vote_count = Counter(responses)
                        majority_resp = vote_count.most_common(1)[0][0]
                        majority_count = vote_count[majority_resp]
                        ratio = majority_count / len(responses)
                        majority_ratio.append(ratio)

                        # Indices of majority and minority
                        maj_indices = [i for i, r in enumerate(responses) if r == majority_resp]
                        min_indices = [i for i in range(len(responses)) if i not in maj_indices]

                        # Entropies
                        maj_entropies = [prompt_uid2entropys[uid][i] for i in maj_indices]
                        min_entropies = [prompt_uid2entropys[uid][i] for i in min_indices]
                        majority_entropy_mean.append(np.mean(maj_entropies) if maj_entropies else 0)
                        minority_entropy_mean.append(np.mean(min_entropies) if min_entropies else 0)

                        # Modify rewards
                        lambda_entropy = self.config.algorithm.scem.lambda_entropy  # From config
                        for i in maj_indices:
                            mean_entropy = prompt_uid2entropys[uid][i]  # Already mean
                            prompt_uid2rewards[uid][i] = [r + lambda_entropy * mean_entropy for r in prompt_uid2rewards[uid][i]]  # Raise for majority
                        for i in min_indices:
                            mean_entropy = prompt_uid2entropys[uid][i]
                            prompt_uid2rewards[uid][i] = [r - lambda_entropy * mean_entropy for r in prompt_uid2rewards[uid][i]]  # Lower for minority

                        # Update batch with modified rewards
                        for i in range(len(responses)):
                            batch.batch["token_level_rewards"][i] = torch.tensor(prompt_uid2rewards[uid][i])

                        # Optimized reward mean
                        optimized_rewards = [torch.mean(batch.batch["token_level_rewards"][j]).item() for j in range(len(responses))]
                        optimized_reward_mean.append(np.mean(optimized_rewards))

                    metrics["scem/majority_entropy_mean"] = np.mean(majority_entropy_mean)
                    metrics["scem/minority_entropy_mean"] = np.mean(minority_entropy_mean)
                    metrics["scem/majority_ratio"] = np.mean(majority_ratio)
                    metrics["scem/optimized_reward_mean"] = np.mean(optimized_reward_mean)

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

                    # SCEM custom logic: Mask advantages where entropy <=1
                    entropy_mask = (entropys > 1).float()  # Token-level mask
                    batch.batch["advantages"] = batch.batch["advantages"] * entropy_mask

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
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        # Compute gradient norm after update
                        # grad_norm = 0
                        # for p in self.actor_rollout_wg.actor.parameters():  # 或 self.actor_rollout_wg.actor.model.parameters()，根据你的结构调整
                        #     if p.grad is not None:
                        #         grad_norm += p.grad.data.norm(2).item() ** 2
                        # grad_norm = grad_norm ** 0.5 if grad_norm > 0 else 0  # 避免 NaN
                        # metrics["scem/grad_norm"] = grad_norm
                        # Compute gradient norm (and optionally clip)
                        # actor = self.get_model("actor")  # 获取 actor 模型
                        # grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)  # max_norm 来自你的 config (e.g., actor.grad_clip, 默认1.0)
                        # 或不 clip，只计算：
                        # grad_norm = 0
                        # for p in actor.parameters():
                        #     if p.grad is not None:
                        #         grad_norm += p.grad.data.norm(2).item() ** 2
                        # grad_norm = grad_norm ** 0.5 if grad_norm > 0 else 0

                        # metrics["scem/grad_norm"] = grad_norm.item()  # 转为 scalar，如果是 tensor
                        

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

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

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

def scem_reward_fn(batch: DataProto, return_dict: bool = False):
    """Custom reward function for SCEM: Modify rewards based on majority vote and entropy."""
    # Assume entropys already computed (from old_log_prob in fit)
    # But for reward_fn, we need to compute or assume it's available
    # In practice, reward_fn is called before old_log_prob, so move logic to fit or adjust
    # For now, placeholder: assume base reward is zero or from rule, then modify in fit as before
    # Base reward (unsupervised, perhaps length or something, but for math, assume 0)
    token_level_rewards = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)  # Placeholder base reward

    if return_dict:
        return {"reward_tensor": token_level_rewards, "reward_extra_info": {}}
    return token_level_rewards

def scem_val_reward_fn(batch: DataProto, return_dict: bool = False):
    """Custom validation reward function for SCEM: Similar to reward_fn but for val, no modification."""
    # For validation, compute base rewards or metrics without modification
    token_level_rewards = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)  # Placeholder

    extra_info = {}  # Add val-specific metrics if needed

    if return_dict:
        return {"reward_tensor": token_level_rewards, "reward_extra_info": extra_info}
    return token_level_rewards