# One-time download/cache script. Run this on the LOGIN NODE 
'''
Usage:
    source .venv/bin/activate
    python scripts/download_models.py

Everything lands in ./hf_cache. SLURM jobs later set:
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
Setup:
- Policy / SFT base : TinyLlama-1.1B-Chat-v1.0 (same for both tasks)
- Reward model      : OpenAssistant/reward-model-deberta-v3-large-v2
                      (SAME reward model for both tasks)
- Task 1 (controlled)   : CarperAI/openai_summarize_tldr      (TL;DR)
- Task 2 (stress test)  : Anthropic/hh-rlhf                   (helpfulness)
'''

import os

CACHE_DIR = os.path.abspath("./hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"Caching everything to: {CACHE_DIR}\n")

# SFT base model: TinyLlama-1.1B-Chat (shared across both tasks)
print("Downloading TinyLlama/TinyLlama-1.1B-Chat-v1.0 (policy / SFT base)")

from transformers import AutoModelForCausalLM, AutoTokenizer

POLICY_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

AutoTokenizer.from_pretrained(POLICY_MODEL, cache_dir=CACHE_DIR)
AutoModelForCausalLM.from_pretrained(POLICY_MODEL, cache_dir=CACHE_DIR)
print(f"OK: {POLICY_MODEL}\n")

# Reward model: OpenAssistant DeBERTa-v3-large-v2 
print("Downloading OpenAssistant/reward-model-deberta-v3-large-v2 (reward model, shared)")

from transformers import AutoModelForSequenceClassification

REWARD_MODEL = "OpenAssistant/reward-model-deberta-v3-large-v2"

AutoTokenizer.from_pretrained(REWARD_MODEL, cache_dir=CACHE_DIR)
AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL, cache_dir=CACHE_DIR)
print(f"OK: {REWARD_MODEL}\n")

# Task 1 dataset (controlled): TL;DR summarization
#  Fields: "prompt" (Reddit post + "TL;DR:" suffix), "label" (human summary)
print("Downloading CarperAI/openai_summarize_tldr (Task 1: TL;DR summarization)")

from datasets import load_dataset

ds_tldr = load_dataset("CarperAI/openai_summarize_tldr", cache_dir=CACHE_DIR)
print(f"OK: CarperAI/openai_summarize_tldr  splits={list(ds_tldr.keys())}  "
      f"train_size={len(ds_tldr['train'])}\n")

# Task 2 dataset (stress test): Anthropic HH-RLHF
# We only need the human prompt turn as the generation seed; the chosen/rejected completions are discarded since the policy generates its own responses and the reward model scores them directly.
print("Downloading Anthropic/hh-rlhf (Task 2: helpfulness, stress test)")

ds_hh = load_dataset("Anthropic/hh-rlhf", cache_dir=CACHE_DIR)
print(f"OK: Anthropic/hh-rlhf  splits={list(ds_hh.keys())}  "
      f"train_size={len(ds_hh['train'])}\n")

print(f"Cache directory: {CACHE_DIR}")

#Remember to set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 in your SLURM scripts so compute nodes never try to reach the network