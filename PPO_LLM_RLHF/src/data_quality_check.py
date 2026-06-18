from transformers import AutoTokenizer
from datasets import load_dataset
from data_utils import format_tldr_prompt
import os

os.environ["HF_HUB_OFFLINE"] = "1"
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir="./hf_cache")
ds = load_dataset("CarperAI/openai_summarize_tldr", cache_dir="./hf_cache")
lengths = [len(tok(format_tldr_prompt(ex))["input_ids"]) for ex in ds["train"].select(range(200))]
print(f"min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")
already_has_cue = sum(
    1 for ex in ds["train"].select(range(2000))
    if ex["prompt"].strip()[-10:].strip().lower().rstrip(":").strip() == "tl;dr"
)
print(f"{already_has_cue}/2000 prompts already end with TL;DR cue")