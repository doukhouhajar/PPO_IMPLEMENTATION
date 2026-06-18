from typing import Optional


# Task 1: TL;DR summarization

TLDR_SUFFIX = "\nTL;DR:"


def format_tldr_prompt(example: dict, max_chars: int = 2000) -> str:
    """
    Build the actual prompt fed to the policy for TL;DR summarization.

    Most raw 'prompt' fields have no completion cue, so we append
    "TL;DR:" ourselves -- this is the standard cue used in the
    OpenAI/CarperAI summarization RLHF setup and is what the policy
    needs to know it should produce a summary now, not continue the
    Reddit post.

    Some raw posts already end with a "TL;DR" cue of their own (observed
    in practice, e.g. "...opinions. TL;DR:") -- if so, we strip the
    existing cue and re-append our canonical TLDR_SUFFIX, so every
    returned prompt ends in an identical cue string. This avoids both
    a duplicated "TL;DR: TL;DR:" suffix (which would confuse the policy
    about where to actually start generating) and avoids leaving
    inconsistent cue formatting (whitespace/casing) across examples,
    which would otherwise be an unintended confound at exactly the
    position that matters most for the policy's generation behavior.

    max_chars truncates very long posts, cutting at the END (right side)
    on a word boundary to keep tokenized length manageable; the subreddit
    + title (most informative part) is at the start and is preserved.
    """
    post = example["prompt"].strip()
    if len(post) > max_chars:
        post = post[:max_chars].rsplit(" ", 1)[0]  # avoid cutting mid-word

    # Strip ANY number of trailing "tl;dr"-style cues already present in the
    # raw data (observed: some posts have one, some apparently have it
    # doubled already in the source field), tolerating arbitrary trailing
    # punctuation/whitespace around each cue (not just a colon). We loop
    # rather than strip once, since a single pass can leave a second cue
    # exposed. After stripping all native cues, we re-append our canonical
    # TLDR_SUFFIX exactly once, so every returned prompt ends in an
    # identical cue string regardless of how many cues (if any) the raw
    # post originally had or how they were punctuated.
    while True:
        stripped = post.rstrip()
        # Peel off trailing punctuation/whitespace (colons, periods, etc.)
        stripped = stripped.rstrip(":.;, ").rstrip()
        if stripped.lower().endswith("tl;dr"):
            post = stripped[: -len("tl;dr")].rstrip()
            continue
        break

    return post + TLDR_SUFFIX


def get_tldr_reference_summary(example: dict) -> str:
    """
    The human-written reference summary. NOT used in PPO training.
    Only useful for optionally checking that the reward model scores
    human references sensibly (a basic RM sanity check), e.g.:
        rm_score(format_tldr_prompt(ex), get_tldr_reference_summary(ex))
    should generally be positive / high relative to a random completion.
    """
    return example["label"].strip()


# Task 2: HH-RLHF helpfulness

ASSISTANT_MARKER = "\n\nAssistant:"
HUMAN_MARKER = "\n\nHuman:"


def extract_hh_prompt(example: dict, source_field: str = "chosen") -> Optional[str]:
    """
    Extract the prompt (full conversation context up to and including the
    last "Assistant:" marker) from an HH-RLHF example.

    'chosen' and 'rejected' share identical context except for the final
    Assistant turn, so source_field doesn't matter -- default to 'chosen'
    for clarity, but either works.

    Returns None if the expected marker isn't found (defensive -- a small
    fraction of HH-RLHF examples have unusual formatting; filter these out
    rather than silently producing a malformed prompt).
    """
    text = example[source_field]

    last_idx = text.rfind(ASSISTANT_MARKER)
    if last_idx == -1:
        return None

    prompt = text[: last_idx + len(ASSISTANT_MARKER)]

    # Defensive check: prompt should contain at least one Human turn.
    if HUMAN_MARKER not in prompt:
        return None

    return prompt


def get_hh_dataset_completions(example: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (chosen_completion, rejected_completion) -- the text AFTER the
    last Assistant marker in each field. NOT used for PPO training (the
    policy generates its own completions), but useful for:
      - validating the reward model: RM(prompt, chosen) should generally
        score higher than RM(prompt, rejected)
      - debugging / sanity checks during setup
    """
    chosen_prompt = extract_hh_prompt(example, "chosen")
    rejected_prompt = extract_hh_prompt(example, "rejected")
    if chosen_prompt is None or rejected_prompt is None:
        return None, None
    chosen_completion = example["chosen"][len(chosen_prompt):].strip()
    rejected_completion = example["rejected"][len(rejected_prompt):].strip()
    return chosen_completion, rejected_completion


# Dataset -> list of prompts (used by both training scripts)

def build_tldr_prompts(dataset_split, max_chars: int = 2000) -> list[str]:
    """dataset_split: a HF Dataset split, e.g. ds['train']"""
    return [format_tldr_prompt(ex, max_chars=max_chars) for ex in dataset_split]


def build_hh_prompts(dataset_split) -> list[str]:
    """
    dataset_split: a HF Dataset split, e.g. ds['train']
    Filters out the small fraction of malformed examples (no Assistant
    marker found) rather than crashing or silently keeping bad prompts.
    """
    prompts = []
    n_skipped = 0
    for ex in dataset_split:
        p = extract_hh_prompt(ex)
        if p is None:
            n_skipped += 1
            continue
        prompts.append(p)
    if n_skipped > 0:
        print(f"[build_hh_prompts] skipped {n_skipped} malformed examples "
              f"out of {len(dataset_split)}")
    return prompts


if __name__ == "__main__":
    # Quick self-test against the cached datasets, run on login node:
    #   python src/data_utils.py
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from datasets import load_dataset

    CACHE_DIR = os.path.abspath("./hf_cache")

    print("TL;DR")
    ds_tldr = load_dataset("CarperAI/openai_summarize_tldr", cache_dir=CACHE_DIR)
    ex = ds_tldr["train"][0]
    prompt = format_tldr_prompt(ex)
    print("Formatted prompt (last 200 chars):")
    print(prompt[-200:])
    print()
    print("Reference summary:", get_tldr_reference_summary(ex))
    print()

    print("HH-RLHF")
    ds_hh = load_dataset("Anthropic/hh-rlhf", cache_dir=CACHE_DIR)
    ex = ds_hh["train"][0]
    prompt = extract_hh_prompt(ex)
    print("Extracted prompt:")
    print(prompt)
    print()
    chosen_c, rejected_c = get_hh_dataset_completions(ex)
    print("Chosen completion:", chosen_c)
    print("Rejected completion:", rejected_c)
    print()

    print("Filtering check on first 500 HH examples")
    sample_prompts = build_hh_prompts(ds_hh["train"].select(range(500)))
    print(f"Got {len(sample_prompts)} valid prompts out of 500")

"""
Prompt extraction for both tasks, written against the CONFIRMED real
dataset formats (verified via scripts/inspect_data.py):

TL;DR (CarperAI/openai_summarize_tldr):
    Features: {'prompt': str, 'label': str}
    'prompt' = full Reddit post (subreddit + title + body), NO "TL;DR:" cue.
    'label'  = human reference summary — NEVER used during PPO training,
               only useful for optionally sanity-checking the reward model.

HH-RLHF (Anthropic/hh-rlhf):
    Features: {'chosen': str, 'rejected': str}
    Each field is the FULL multi-turn transcript as one string with
    literal "\n\nHuman:" / "\n\nAssistant:" markers. 'chosen' and
    'rejected' are identical except for the final Assistant turn.
    We discard both completions entirely — the policy generates its own
    response. We only need the prompt: everything up to and including
    the LAST "\n\nAssistant:" marker.
"""