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
This trainer supports model-agnostic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import json
import os
import numpy as np
import torch
from tqdm import tqdm

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

        # Initialize JSON log file
        json_log_path = os.path.join(self.config.trainer.default_local_dir, "training_log.json")
        if not os.path.exists(json_log_path):
            with open(json_log_path, 'w') as f:
                json.dump([], f)

        # Entropy regularization parameters
        entropy_reg_config = self.config.algorithm.get("entropy_reg", {})
        lambda_reg = entropy_reg_config.get("lambda", 0.005)
        beta_reg = entropy_reg_config.get("beta", 0.001)
        logic_words = [
            "because", "since", "as", "therefore", "thus", "hence", "so", "consequently",
            "if", "then", "provided", "assuming", "implies", "follows",
            "however", "but", "nevertheless", "nonetheless", "yet", "although", "though", "despite", "whereas",
            "and", "or", "also", "furthermore", "moreover", "next", "subsequently",
            "let", "assume", "consider", "suppose", "equals", "is", "are", "given", "known",
            "step", "first", "second"
        ]
        logic_words_set = set(word.lower() for word in logic_words)
        answer_words = ["answer", "final", "result", "solution"]
        answer_words_set = set(word.lower() for word in answer_words)

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
                    # Restore raw_prompt_ids to new_batch after union
                    new_batch.non_tensor_batch["raw_prompt_ids"] = np.repeat(
                        gen_batch.non_tensor_batch["raw_prompt_ids"], self.config.actor_rollout_ref.rollout.n, axis=0
                    )

                    with marked_timer("reward", timing_raw, "yellow"):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                            print(f"reward_extra_infos_dict: {reward_extra_infos_dict}")
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}
                        new_batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    # Debug: Print new_batch sizes before filtering
                    print(f"Before filtering: num_prompt_in_batch: {num_prompt_in_batch}, "
                          f"uid_length: {len(new_batch.non_tensor_batch['uid'])}, "
                          f"raw_prompt_ids_length: {len(new_batch.non_tensor_batch['raw_prompt_ids'])}")

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)
                        prompt_uid2metric_std = {}
                        for uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[uid] = np.std(metric_vals)
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
                        # Debug: Print filtering info
                        print(f"After filtering: num_prompt_in_batch: {num_prompt_in_batch}, "
                              f"kept_traj_idxs: {len(kept_traj_idxs)}, "
                              f"uid_length: {len(new_batch.non_tensor_batch['uid'])}, "
                              f"raw_prompt_ids_length: {len(new_batch.non_tensor_batch['raw_prompt_ids'])}")
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
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]
                            num_prompt_in_batch = len(set(batch.non_tensor_batch["uid"]))
                            # Debug: Print batch info after concatenation
                            print(f"After concatenation: batch_size: {batch.batch.batch_size[0]}, "
                                  f"raw_prompt_ids_length: {len(batch.non_tensor_batch.get('raw_prompt_ids', []))}")

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

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

                        # Extract and classify tokens for l_budget and logging
                        key_tokens_list = []
                        token_weights = []
                        high_entropy_tokens = []
                        low_entropy_tokens = []
                        high_entropy_count = 0
                        low_entropy_count = 0
                        if "responses" not in batch.batch:
                            raise KeyError("Expected 'responses' in batch.batch, but not found. Available keys: {}".format(list(batch.batch.keys())))
                        response_sequences = batch.batch["responses"]
                        max_response_length = self.config.data.max_response_length  # 2048

                        # Compute h_baseline dynamically
                        h_baseline = (entropys * response_masks).sum() / (response_masks.sum() + 1e-8)
                        print(f"Computed h_baseline: {h_baseline.item()}")

                        for i in range(len(response_sequences)):
                            response_tokens = response_sequences[i]
                            response_entropy = entropys[i] * response_masks[i]
                            response_text = self.tokenizer.batch_decode(response_tokens.unsqueeze(0))[0].split()
                            # Remove <|endoftext|> from response_text
                            response_text = [t for t in response_text if t != "<|endoftext|>"]
                            key_tokens = []
                            weights = []
                            # Align weights to max_response_length
                            for j in range(max_response_length):
                                if j < len(response_text):
                                    token = response_text[j]
                                    token_lower = token.lower()
                                    entropy = response_entropy[j].item() if j < len(response_entropy) else 0.0
                                    if token_lower in logic_words_set:
                                        category = "logic"
                                        w_t = 1.0
                                        key_tokens.append({"token": token, "entropy": entropy, "category": category})
                                        weights.append(w_t)
                                        if entropy > h_baseline:
                                            high_entropy_tokens.append(entropy)
                                            high_entropy_count += 1
                                        elif entropy < h_baseline:
                                            low_entropy_tokens.append(entropy)
                                            low_entropy_count += 1
                                    elif token_lower in answer_words_set:
                                        category = "answer"
                                        w_t = 0.5
                                        key_tokens.append({"token": token, "entropy": entropy, "category": category})
                                        weights.append(w_t)
                                        if entropy > h_baseline:
                                            high_entropy_tokens.append(entropy)
                                            high_entropy_count += 1
                                        elif entropy < h_baseline:
                                            low_entropy_tokens.append(entropy)
                                            low_entropy_count += 1
                                    else:
                                        weights.append(0.0)
                                else:
                                    weights.append(0.0)  # Pad with 0.0 for shorter sequences
                            key_tokens_list.append(key_tokens)
                            token_weights.append(weights)
                        batch.non_tensor_batch["key_tokens"] = np.array(key_tokens_list, dtype=object)
                        batch.batch["token_weights"] = torch.tensor(token_weights, device=entropys.device)

                        # Compute l_budget per sample
                        token_weights = batch.batch["token_weights"]
                        entropys_masked = entropys * response_masks
                        low_entropy_penalty = torch.sum(token_weights * torch.relu(torch.tensor(h_baseline, device=entropys.device) - entropys_masked), dim=1) / (torch.sum(token_weights, dim=1) + 1e-8)
                        high_entropy_penalty = torch.sum(token_weights * torch.relu(entropys_masked - torch.tensor(h_baseline, device=entropys.device)), dim=1) / (torch.sum(token_weights, dim=1) + 1e-8)
                        l_budget = lambda_reg * low_entropy_penalty - beta_reg * high_entropy_penalty
                        metrics["train/regulation_term_value"] = l_budget.detach().mean().item()
                        metrics["train/h_baseline"] = h_baseline.item()
                        metrics["train/high_entropy_avg"] = np.mean(high_entropy_tokens) if high_entropy_tokens else 0.0
                        metrics["train/low_entropy_avg"] = np.mean(low_entropy_tokens) if low_entropy_tokens else 0.0
                        metrics["train/high_entropy_count"] = high_entropy_count
                        metrics["train/low_entropy_count"] = low_entropy_count
                        batch.batch["l_budget"] = l_budget
                        print(f"High entropy tokens: count={high_entropy_count}, avg={np.mean(high_entropy_tokens) if high_entropy_tokens else 0.0:.4f}")
                        print(f"Low entropy tokens: count={low_entropy_count}, avg={np.mean(low_entropy_tokens) if low_entropy_tokens else 0.0:.4f}")

                        # Debug: Print shapes for verification
                        print(f"entropys shape: {entropys.shape}")
                        print(f"response_masks shape: {response_masks.shape}")
                        print(f"token_weights shape: {batch.batch['token_weights'].shape}")
                        print(f"l_budget shape: {l_budget.shape}")
                        print(f"non_tensor_batch keys: {list(batch.non_tensor_batch.keys())}")
                        print(f"key_tokens type: {type(batch.non_tensor_batch['key_tokens'])}")
                        print(f"key_tokens shape: {batch.non_tensor_batch['key_tokens'].shape if isinstance(batch.non_tensor_batch['key_tokens'], np.ndarray) else 'N/A'}")

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )
                        # Debug: Print advantages shape
                        print(f"advantages shape: {batch.batch['advantages'].shape}")

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, "red"):
                            # Debug: Print old_log_probs shape
                            print(f"old_log_probs shape: {batch.batch['old_log_probs'].shape}")
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Validate (exact copy from original code)
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

                    # Debug: Print non_tensor_batch keys before JSON logging
                    print(f"non_tensor_batch keys before JSON logging: {list(batch.non_tensor_batch.keys())}")

                    # Log to JSON: Only log 10 random samples
                    unique_prompts = set(tuple(list(p)) for p in batch.non_tensor_batch.get("raw_prompt_ids", []))
                    prompt_text = self.tokenizer.batch_decode(list(unique_prompts)[0])[0] if unique_prompts else ""
                    json_log_entry = {
                        "step": self.global_steps,
                        "prompt": prompt_text,
                        "h_baseline": h_baseline.item(),
                        "responses": []
                    }
                    # Randomly select 10 indices
                    num_samples = len(response_sequences)
                    sample_indices = np.random.choice(num_samples, size=min(10, num_samples), replace=False)
                    print(f"Logging {len(sample_indices)} samples to JSON for step {self.global_steps}")
                    for i in sample_indices:
                        response_text = self.tokenizer.batch_decode(response_sequences[i].unsqueeze(0))[0]
                        # Clean response_text for JSON logging
                        response_text = response_text.replace("<|endoftext|>", "").rstrip()
                        print(f"Response {i} raw text: {response_text[:100]}...")
                        key_tokens = batch.non_tensor_batch["key_tokens"][i]
                        key_count = len(key_tokens)
                        avg_entropy = np.mean([t["entropy"] for t in key_tokens]) if key_count > 0 else 0.0
                        low_penalty_per = low_entropy_penalty[i].item() / (key_count or 1)
                        high_penalty_per = high_entropy_penalty[i].item() / (key_count or 1)
                        key_stats = {
                            "regulation_term_value": l_budget[i].item(),
                            "h_baseline": h_baseline.item(),
                            "key_tokens_count": key_count,
                            "key_tokens_avg_entropy": avg_entropy,
                            "low_entropy_penalty_per_response": low_penalty_per,
                            "high_entropy_penalty_per_response": high_penalty_per
                        }
                        json_log_entry["responses"].append({
                            "response_index": int(i),
                            "sentence": response_text,
                            "key_tokens": key_tokens,
                            "key_stats": key_stats
                        })
                    with open(json_log_path, 'r+') as f:
                        json_data = json.load(f)
                        json_data.append(json_log_entry)
                        f.seek(0)
                        json.dump(json_data, f, indent=2)

                    # Add metrics for TensorBoard (based on all samples)
                    key_counts = [len(tokens) for tokens in batch.non_tensor_batch["key_tokens"]]
                    metrics["train/key_tokens_count"] = np.mean(key_counts) if key_counts else 0.0
                    avg_entropies = [np.mean([t["entropy"] for t in tokens]) for tokens in batch.non_tensor_batch["key_tokens"] if tokens]
                    metrics["train/key_tokens_avg_entropy"] = np.mean(avg_entropies) if avg_entropies else 0.0
                    metrics["train/h_baseline"] = h_baseline.item()

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

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)
                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)