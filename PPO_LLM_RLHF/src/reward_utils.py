import os
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"


class RewardModel:
    """
    Thin wrapper around the frozen reward model. Loaded once, reused across
    the entire training run. Always in eval() mode, always under
    torch.no_grad() during scoring -- no gradients should ever flow through
    the RM.
    """

    def __init__(
        self,
        cache_dir: str = "./hf_cache",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        max_length: int = 512,
    ):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            REWARD_MODEL_NAME, cache_dir=cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL_NAME, cache_dir=cache_dir, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

        # Running statistics across all scoring calls -- useful for
        # diagnosing reward scale issues (e.g. reward hacking showing up as
        # a sudden shift in mean/std over outer iterations).
        self.running_count = 0
        self.running_mean = 0.0
        self.running_m2 = 0.0  # for online variance (Welford's algorithm)

    @torch.no_grad()
    def score_batch(self, questions: list[str], answers: list[str]) -> torch.Tensor:
        """
        questions: list of prompts (e.g. TL;DR post+"TL;DR:" or HH-RLHF
                   extracted prompt ending in "\n\nAssistant:")
        answers:   list of policy-generated completions, same length

        Returns: 1D float tensor of raw RM logit scores, one per pair,
                 on CPU (so it composes easily with the rest of the PPO
                 pipeline which may be on a different device/dtype).
        """
        assert len(questions) == len(answers), (
            f"questions/answers length mismatch: {len(questions)} vs {len(answers)}"
        )

        inputs = self.tokenizer(
            questions,
            answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        logits = self.model(**inputs).logits.squeeze(-1)  # shape: (batch,)
        scores = logits.float().cpu()

        self._update_running_stats(scores)
        return scores

    def _update_running_stats(self, scores: torch.Tensor):
        for s in scores.tolist():
            self.running_count += 1
            delta = s - self.running_mean
            self.running_mean += delta / self.running_count
            delta2 = s - self.running_mean
            self.running_m2 += delta * delta2

    @property
    def running_std(self) -> float:
        if self.running_count < 2:
            return 0.0
        return (self.running_m2 / (self.running_count - 1)) ** 0.5

    def get_stats_dict(self) -> dict:
        return {
            "reward_model/running_mean": self.running_mean,
            "reward_model/running_std": self.running_std,
            "reward_model/running_count": self.running_count,
        }


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    rm = RewardModel()

    questions = [
        "What is the capital of France?",
        "What is the capital of France?",
    ]
    answers = [
        "The capital of France is Paris.",
        "Bananas are yellow.",
    ]
    scores = rm.score_batch(questions, answers)
    print("Scores:", scores.tolist())
    print("Expected: first score notably higher than second (matches earlier "
          "inspect_data.py sanity check: ~6.97 vs ~-3.43)")
    print()
    print("Running stats:", rm.get_stats_dict())

    """
Wraps OpenAssistant/reward-model-deberta-v3-large-v2 -- the SAME reward
model used for both tasks (TL;DR and HH-RLHF), per the agreed experimental
design: using one shared RM isolates the variable of interest (the task's
reward landscape) from differences between reward models.

This is a sequence-classification model, NOT a causal LM. Confirmed usage
(scripts/inspect_data.py sanity check):
    inputs = tokenizer(question, answer, return_tensors="pt")
    score  = model(**inputs).logits[0]

Confirmed scale is uncalibrated logits (e.g. ~6.97 for a good answer,
~-3.43 for an irrelevant one) -- NOT a bounded/normalized score. We track
running statistics so reward magnitudes can be monitored/clipped if they
destabilize PPO training, which is a separate concern from the Nesterov
experiment itself but a known general RLHF pitfall worth guarding against.
"""