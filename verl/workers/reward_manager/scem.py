# verl/workers/reward_manager/scem.py

import re
from collections import Counter, defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score

from .abstract import AbstractRewardManager
from .registry import register


@register("scem")
class SCEMRewardManager(AbstractRewardManager):
    """SCEM Reward Manager based on self-consistency and entropy."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
    
    def __call__(self, batch: DataProto, return_dict: bool = False):
        """
        Compute SCEM rewards: Group by uid, find majority answer via self-consistency,
        assign -entropy for majority, +entropy for minority.
        Assumes 'entropys' is in batch from prior log_prob computation.
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
                majority = majority_candidates[0][0]  # Take the first most common

            # Assign signs and compute token-level rewards
            for j, ans in enumerate(answers):
                if ans == majority:
                    sign = -1.0
                else:
                    sign = 1.0
                token_level_scores[idxs[j]] = sign * group_batch.batch["entropys"][j]

        reward_extra_infos_dict = {}  # Add any extra info if needed

        if return_dict:
            return {"reward_tensor": token_level_scores, "reward_extra_info": reward_extra_infos_dict}
        else:
            return token_level_scores