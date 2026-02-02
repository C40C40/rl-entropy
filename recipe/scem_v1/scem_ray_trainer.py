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
SCEM PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
Customized for SCEM algorithm based on self-consistency and entropy-based rewards.
"""

import logging
import os
import re
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from pprint import pformat, pprint

import numpy as np
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
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    Customized for SCEM: computes old_log_prob (including entropys) before reward computation.
    Uses a custom reward function based on self-consistency and entropy.
    """

    def _create_scem_reward_fn(self):
        def scem_reward_fn(batch: DataProto, return_dict: bool = False):
            """
            Custom reward function for SCEM.
            Groups by uid, extracts answers via self-consistency, assigns rewards as -entropy for majority, +entropy for minority.
            Assumes 'entropys' and 'response_mask' are in batch.
            """
            if "entropys" not in batch.batch:
                raise ValueError("Batch must contain 'entropys' for SCEM reward computation.")

            uids = batch.non_tensor_batch["uid"]
            unique_uids = np.unique(uids)
            token_level_scores = torch.zeros_like(batch.batch["entropys"])

            for uid in unique_uids:
                idxs = np.where(uids == uid)[0]
                group_batch = batch[idxs]
                num_responses = len(group_batch)

                # Decode responses (only response parts)
                responses = []
                for i in range(num_responses):
                    response_mask_i = group_batch.batch["response_mask"][i]
                    response_start = len(group_batch.batch["input_ids"][i]) - len(response_mask_i)
                    full_resp_ids = group_batch.batch["input_ids"][i][response_start:]
                    full_resp_att_mask = group_batch.batch["attention_mask"][i][response_start:] == 1
                    effective_resp_ids = full_resp_ids[full_resp_att_mask]
                    effective_resp_mask = response_mask_i[full_resp_att_mask] == 1
                    resp_ids = effective_resp_ids[effective_resp_mask]
                    resp_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                    responses.append(resp_text)

                # Extract boxed answers
                answers = []
                for resp in responses:
                    match = re.search(r"\\boxed\{(.*?)\}", resp)
                    if match:
                        ans = match.group(1).strip()
                        try:
                            ans = float(ans)  # Assume numerical for math problems
                        except ValueError:
                            pass
                    else:
                        ans = None
                    answers.append(ans)

                # Find majority via self-consistency
                valid_answers = [a for a in answers if a is not None]
                majority = None
                if valid_answers:
                    counts = Counter(valid_answers)
                    majority_candidates = counts.most_common()
                    majority = majority_candidates[0][0]  # Take the first most common; ties handled by count

                # Assign signs and compute token-level rewards
                for j, ans in enumerate(answers):
                    if ans == majority:
                        sign = -1.0
                    else:
                        sign = 1.0
                    token_level_scores[idxs[j]] = sign * group_batch.batch["entropys"][j]

            reward_extra_infos_dict = {}  # No extra info for now

            if return_dict:
                return {"reward_tensor": token_level_scores, "reward_extra_info": reward_extra_infos_dict}
            else:
                return token_level_scores

        return scem_reward_fn

    def _validate(self):
        from collections import defaultdict

        from verl.trainer.ppo.metric_utils import (compute_data_metrics,
                                                   reduce_metrics)

        metrics = {}
        timing_raw = defaultdict(float)
        num_batches = 0

        # Assuming tqdm for progress, as in fit
        progress_bar = tqdm(total=len(self.val_dataloader), desc="Validation Progress")

        for batch_dict in self.val_dataloader:
            num_batches += 1
            new_batch: DataProto = DataProto.from_single_dict(batch_dict)

            answers_gt = batch_dict.get('Answer', [None] * len(new_batch))  # 获取 GT 列表
            new_batch.non_tensor_batch["Answer"] = np.array(answers_gt, dtype=object)  # 添加到 non_tensor_batch
            
            # Pop keys for generation (copied from fit)
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
            gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # Generate sequences
            with marked_timer("gen_val", timing_raw, "red"):
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                gen_batch_output.meta_info.pop("timing", None)

            # Union generated outputs
            new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # SCEM-specific: Compute log_probs to add 'entropys' (required for reward)
            
            # Compute response_mask before reward (required for decoding in __call__)
            new_batch.batch["response_mask"] = compute_response_mask(new_batch)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
            
            with marked_timer("old_log_prob_val", timing_raw, "blue"):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(new_batch)
                # Optional: Compute entropy metrics if needed, but mainly for 'entropys'
                if "entropys" in old_log_prob.batch:
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = new_batch.batch.get("response_mask")
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    metrics.update({"val/actor/entropy": entropy_agg.detach().item()})
                    # old_log_prob.batch.pop("entropys")  # Pop if not needed post-reward
                new_batch = new_batch.union(old_log_prob)

            # Now compute reward (SCEM will use 'entropys')
            with marked_timer("reward_val", timing_raw, "yellow"):
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                    new_batch = new_batch.union(reward_tensor)

                reward_result = self.val_reward_fn(new_batch, return_dict=True)
                reward_tensor = reward_result["reward_tensor"]
                reward_extra_infos_dict = reward_result.get("reward_extra_info", {})

                new_batch.batch["token_level_scores"] = reward_tensor

                if reward_extra_infos_dict:
                    new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                # Validation typically doesn't apply KL in reward, but add if configured
                if self.config.algorithm.use_kl_in_reward:
                    new_batch, kl_metrics = apply_kl_penalty(
                        new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

            # Compute validation metrics (copied from fit)
            # metrics.update(compute_data_metrics(batch=new_batch, use_critic=self.use_critic))
            # Add other metrics if needed (e.g., timing, throughput)
            acc_list = []
            for i in range(len(new_batch)):
                # 提取生成的响应文本
                response_mask_i = new_batch.batch["response_mask"][i]
                response_start = len(new_batch.batch["input_ids"][i]) - len(response_mask_i)
                full_resp_ids = new_batch.batch["input_ids"][i][response_start:]
                full_resp_att_mask = new_batch.batch["attention_mask"][i][response_start:] == 1
                effective_resp_ids = full_resp_ids[full_resp_att_mask]
                effective_resp_mask = response_mask_i[full_resp_att_mask] == 1
                resp_ids = effective_resp_ids[effective_resp_mask]
                generated_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                
                # 提取生成的 \boxed{答案}
                match = re.search(r"\\boxed\{(.*?)\}", generated_text)
                generated_ans = match.group(1).strip() if match else None
                
                # 获取 ground truth 
                gt_ans = new_batch.non_tensor_batch.get("Answer", [None])[i]
                
                # 比较 (AIME 答案通常为整数，字符串匹配或数值比较)
                if generated_ans is not None and gt_ans is not None:
                    try:
                        acc = 1.0 if int(generated_ans) == int(gt_ans) else 0.0
                    except ValueError:
                        acc = 1.0 if generated_ans == gt_ans else 0.0
                else:
                    acc = 0.0
                acc_list.append(acc)

            metrics["val/accuracy"] = np.mean(acc_list) if acc_list else 0.0

            progress_bar.update(1)

        progress_bar.close()

        # Reduce and return metrics
        val_metrics = reduce_metrics(metrics)
        val_metrics["val/num_batches"] = num_batches

        return val_metrics

    def fit(self):
        """
        The training loop of PPO for SCEM.
        Modified to compute old_log_prob (with entropys) before reward computation.
        Uses custom SCEM reward_fn.
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
            logging.info(f"Initial validation metrics: {pformat(val_metrics)}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # Override reward_fn with custom SCEM reward
        self.reward_fn = self._create_scem_reward_fn()

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
                answers_gt = batch_dict.get('Answer', [None] * len(new_batch))  # 获取 GT 列表
                new_batch.non_tensor_batch["Answer"] = np.array(answers_gt, dtype=object)  # 添加到 non_tensor_batch

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

                    # SCEM modification: Compute old_log_prob (including entropys) before reward
                    new_batch.batch["response_mask"] = compute_response_mask(new_batch)
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(new_batch)
                        # Keep entropys in the batch for reward computation
                        batch_entropys = old_log_prob.batch["entropys"]
                        entropy_agg = agg_loss(loss_mat=batch_entropys, loss_mask=new_batch.batch.get("response_mask"), loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        # Union with entropys included
                        new_batch = new_batch.union(old_log_prob)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores using custom SCEM reward_fn
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            logging.error(f"Error in reward_fn: {e}")
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

                    # Pop entropys after reward computation if not needed elsewhere
                    new_batch.batch.pop("entropys", None)
                    
                    if self.config.algorithm.filter_groups.metric == "acc":
                        acc_list = []
                        for k in range(len(new_batch)):
                            # 提取生成的响应文本 (复用解码逻辑)
                            response_mask_k = new_batch.batch["response_mask"][k]
                            response_start = len(new_batch.batch["input_ids"][k]) - len(response_mask_k)
                            full_resp_ids = new_batch.batch["input_ids"][k][response_start:]
                            full_resp_att_mask = new_batch.batch["attention_mask"][k][response_start:] == 1
                            effective_resp_ids = full_resp_ids[full_resp_att_mask]
                            effective_resp_mask = response_mask_k[full_resp_att_mask] == 1
                            resp_ids = effective_resp_ids[effective_resp_mask]
                            generated_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                            
                            # 提取生成的 \boxed{答案}
                            match = re.search(r"\\boxed\{(.*?)\}", generated_text)
                            generated_ans = match.group(1).strip() if match else None
                            
                            # 获取 ground truth 
                            gt_ans = new_batch.non_tensor_batch.get("Answer", [None])[k]
                            
                            # 比较 (AIME/DAPO-math 答案通常为整数/字符串)
                            if generated_ans is not None and gt_ans is not None:
                                try:
                                    acc = 1.0 if int(generated_ans) == int(gt_ans) else 0.0
                                except ValueError:
                                    acc = 1.0 if generated_ans == gt_ans else 0.0
                            else:
                                acc = 0.0
                            acc_list.append(acc)
                        new_batch.non_tensor_batch["acc"] = np.array(acc_list)

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
                            logging.info(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                logging.info(f"{num_gen_batches=}. Keep generating...")
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

                    # old_log_prob already computed, but for reference policy if needed
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

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    logging.info(f"Final validation metrics: {pformat(last_val_metrics)}")
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