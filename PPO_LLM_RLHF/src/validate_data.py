import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from datasets import load_dataset
from data_utils import format_tldr_prompt, build_hh_prompts

CACHE_DIR = os.path.abspath("./hf_cache")


def validate_tldr():
    print("TL;DR validation")
    ds = load_dataset("CarperAI/openai_summarize_tldr", cache_dir=CACHE_DIR)

    for split_name in ds.keys():
        split = ds[split_name]
        n = len(split)
        n_native_cue = 0
        n_duplicated = 0
        lengths = []

        for ex in split:
            raw = ex["prompt"].strip()
            stripped = raw.rstrip().rstrip(":").rstrip()
            if stripped.lower().endswith("tl;dr"):
                n_native_cue += 1

            formatted = format_tldr_prompt(ex)
            # The only invariant that matters: formatted must end with exactly
            # our canonical TLDR_SUFFIX ("\nTL;DR:"), and the text immediately
            # before that suffix must not itself end in another TL;DR cue
            # (which would indicate a true duplicate, not a body-text mention).
            # A simple count("tl;dr") > 1 produces false positives for posts
            # that legitimately mention "tl;dr" somewhere in the body text.
            from data_utils import TLDR_SUFFIX
            suffix_ok = formatted.endswith(TLDR_SUFFIX)
            body = formatted[: -len(TLDR_SUFFIX)].rstrip().rstrip(":.;, ").rstrip()
            trailing_dup = body.lower().endswith("tl;dr")
            if not suffix_ok or trailing_dup:
                n_duplicated += 1
            lengths.append(len(formatted))

        print(f"\nSplit: {split_name} (n={n})")
        print(f"  native cue present in raw prompt: {n_native_cue} ({100*n_native_cue/n:.1f}%)")
        print(f"  duplicated cues after formatting: {n_duplicated}  <-- must be 0")
        print(f"  formatted length: min={min(lengths)}, max={max(lengths)}, "
              f"mean={sum(lengths)/n:.1f}")

        if n_duplicated > 0:
            print(f"  *** STILL BROKEN on split '{split_name}' -- do not proceed ***")


def validate_hh():
    print("HH-RLHF validation")
    ds = load_dataset("Anthropic/hh-rlhf", cache_dir=CACHE_DIR)

    for split_name in ds.keys():
        split = ds[split_name]
        n = len(split)
        prompts = build_hh_prompts(split)
        n_valid = len(prompts)
        lengths = [len(p) for p in prompts]

        print(f"\nSplit: {split_name} (n={n})")
        print(f"  valid prompts: {n_valid} ({100*n_valid/n:.1f}%)")
        print(f"  skipped (malformed): {n - n_valid}")
        if lengths:
            lengths_sorted = sorted(lengths)
            p99 = lengths_sorted[int(0.99 * len(lengths_sorted))]
            print(f"  length: min={min(lengths)}, max={max(lengths)}, "
                  f"mean={sum(lengths)/n_valid:.1f}, p99={p99}")


if __name__ == "__main__":
    validate_tldr()
    validate_hh()
    print("\nDone")