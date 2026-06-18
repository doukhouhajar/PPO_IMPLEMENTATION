import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["BNB_CUDA_VERSION"] = "124" 

import argparse
import sys
import time
import random
from distutils.util import strtobool
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "src"))
from data_utils import build_tldr_prompts, build_hh_prompts
from reward_utils import RewardModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO-RLHF with outer-loop Nesterov momentum"
    )
    # Experiment identity
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--task", type=str, choices=["tldr", "hh"],required=True, help="tldr = summarization, hh = helpfulness")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="ppo-rlhf-nesterov")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None, help="group name for sweep organisation")
    # Paths
    parser.add_argument("--hf-cache-dir", type=str, default="./hf_cache")
    parser.add_argument("--policy-model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--reward-model", type=str, default="OpenAssistant/reward-model-deberta-v3-large-v2")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    # LoRA config
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (effective scale = alpha/r = 2)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="comma-separated list of target modules")
    # PPO / training
    parser.add_argument("--num-outer-iterations", type=int, default=200, help="number of PPO outer updates")
    parser.add_argument("--rollout-batch-size", type=int, default=128, help="prompts per outer iteration")
    parser.add_argument("--inner-epochs", type=int, default=2, help="Adam epochs over each rollout batch")
    parser.add_argument("--mini-batch-size", type=int, default=16, help="minibatch size for inner loop gradient steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip coefficient epsilon")
    parser.add_argument("--kl-coef", type=float, default=0.05, help="KL penalty coefficient in reward shaping")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="entropy bonus coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.1, help="value function loss coefficient")
    parser.add_argument("--target-kl", type=float, default=0.05, help="inner-loop early stopping KL threshold")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--whiten-rewards", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="normalize raw RM scores before PPO update")
    # Generation
    parser.add_argument("--max-prompt-len", type=int, default=512, help="max tokens for prompt tokenization")
    parser.add_argument("--max-gen-len", type=int, default=128, help="max new tokens to generate per prompt ")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    # Nesterov
    parser.add_argument("--nesterov-alpha", type=float, default=0.0, help="0.0 = standard PPO; >0 = Nesterov lookahead ")
    parser.add_argument("--nesterov-adaptive", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="gate lookahead on clipped surrogate improvement")

    # Misc
    parser.add_argument("--save-every", type=int, default=50, help="save LoRA checkpoint every N outer iterations")
    parser.add_argument("--qual-check-every", type=int, default=50, help="log qualitative sample completions every N iters")
    parser.add_argument("--qual-check-n", type=int, default=10, help="number of prompts to generate for qualitative check")

    args = parser.parse_args()

    # Per-task generation length defaults
    if args.max_gen_len == 128 and args.task == "hh":
        args.max_gen_len = 256

    # Derived
    args.lora_target_modules = [m.strip()
                                 for m in args.lora_target_modules.split(",")]
    return args


# LoRA parameter helpers

def clone_lora_params(model) -> dict:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_lora_params(model, param_dict: dict):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and name in param_dict:
                param.copy_(param_dict[name])


def interpolate_lora_params(base: dict, delta_start: dict,
                             delta_end: dict, alpha: float) -> dict:
    """
    Compute θ_ref_candidate = θ_{k+1} + alpha * (θ_{k+1} - θ_k).
    base = theta_kp1  (θ_{k+1})
    delta_start = theta_k  (θ_k)
    delta_end   = theta_kp1 again (passed separately for clarity)
    Returns a new dict, does not modify inputs.
    """
    return {
        name: base[name] + alpha * (base[name] - delta_start[name])
        for name in base
    }


# Rollout collection

@torch.no_grad()
def collect_rollouts(
    policy,
    tokenizer,
    prompts: list[str],
    sft_model,
    reward_model,
    args,
    device,
) -> dict:
    """
    Generate completions for a batch of prompts, score them with the RM,
    compute KL from SFT, and return everything needed for the PPO update.

    Returns a dict of tensors:
        prompt_ids, prompt_mask      : tokenized prompts [B, L_p]
        response_ids, response_mask  : generated tokens  [B, L_r]
        ref_logprobs                 : log π_{θ_ref}(a|s) per token [B, L_r]
        sft_logprobs                 : log π_sft(a|s) per token      [B, L_r]
        rewards                      : scalar RM score per prompt     [B]
        kl_from_sft                  : per-token KL, summed over seq  [B]
        shaped_rewards               : rewards - kl_coef * kl_from_sft [B]
    """
    policy.eval()

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_len,
    ).to(device)

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_gen_len,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Generate completions
    with torch.no_grad():
        output_ids = policy.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            generation_config=gen_cfg,
        )

    prompt_len = enc["input_ids"].shape[1]
    response_ids = output_ids[:, prompt_len:]  # [B, L_r]

    # Build response attention mask (1 until EOS, 0 after)
    B, L_r = response_ids.shape
    response_mask = torch.ones_like(response_ids, dtype=torch.float)
    for i in range(B):
        eos_positions = (response_ids[i] == tokenizer.eos_token_id).nonzero()
        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            response_mask[i, first_eos + 1:] = 0.0

    # Compute reference log-probs (log π_{θ_ref}, i.e. current policy used
    # for this rollout). We compute them *now*, before any parameter updates,
    # so they correctly serve as the denominator in the PPO ratio.
    ref_logprobs = compute_token_logprobs(
        policy, enc["input_ids"], enc["attention_mask"],
        response_ids, response_mask, device
    )  # [B, L_r]

    # Compute SFT log-probs for KL-from-SFT monitoring
    sft_logprobs = compute_token_logprobs(
        sft_model, enc["input_ids"], enc["attention_mask"],
        response_ids, response_mask, device
    )  # [B, L_r]

    # KL(π_{θ_ref} || π_sft) per sequence = sum_t (ref_lp - sft_lp)
    # (approximation valid when policies are close; exact for categorical)
    kl_from_sft = ((ref_logprobs - sft_logprobs) * response_mask).sum(dim=1)  # [B]

    # Decode completions for reward model scoring
    completions = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

    # RM scores
    rm_scores = reward_model.score_batch(prompts, completions)
    rewards = rm_scores.to(dtype=torch.float32, device=device)

    # Reward shaping: subtract KL penalty to discourage reward hacking
    # shaped_r = RM(s,a) - kl_coef * KL(π_ref || π_sft)
    shaped_rewards = rewards - args.kl_coef * kl_from_sft.clamp(min=0)

    policy.train()

    return {
        "prompt_ids":      enc["input_ids"],
        "prompt_mask":     enc["attention_mask"],
        "response_ids":    response_ids,
        "response_mask":   response_mask,
        "ref_logprobs":    ref_logprobs,
        "sft_logprobs":    sft_logprobs,
        "rewards":         rewards,
        "kl_from_sft":     kl_from_sft,
        "shaped_rewards":  shaped_rewards,
        "completions":     completions,
    }


@torch.no_grad()
def compute_token_logprobs(
    model,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    device,
) -> torch.Tensor:
    """
    Compute per-token log-probs of response_ids given prompt_ids.
    Returns tensor of shape [B, L_r].
    """
    B, L_p = prompt_ids.shape
    L_r = response_ids.shape[1]

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)         # [B, L_p+L_r]
    full_mask = torch.cat([
        prompt_mask,
        response_mask.long()
    ], dim=1)                                                         # [B, L_p+L_r]

    with torch.no_grad():
        logits = model(
            input_ids=full_ids,
            attention_mask=full_mask,
        ).logits  # [B, L_p+L_r, V]

    # Shift: logits[t] predicts token at position t+1
    # Response positions: L_p-1 .. L_p+L_r-2  (logits) -> L_p .. L_p+L_r-1 (labels)
    response_logits = logits[:, L_p - 1: L_p + L_r - 1, :]  # [B, L_r, V]
    log_probs = F.log_softmax(response_logits, dim=-1)        # [B, L_r, V]

    token_logprobs = log_probs.gather(
        dim=2,
        index=response_ids.unsqueeze(-1)
    ).squeeze(-1)  # [B, L_r]

    return token_logprobs * response_mask  # zero out padding positions


# Value head (simple linear on top of frozen LM hidden states)

class ValueHead(torch.nn.Module):
    """
    Scalar value head: takes the last non-padding hidden state and
    maps it to a scalar value estimate V(s).
    Projects from the policy's hidden size to 1.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, 1)
        torch.nn.init.orthogonal_(self.linear.weight, gain=1.0)
        torch.nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, hidden_states: torch.Tensor,
                response_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, L, H]  (response token hidden states only)
        response_mask: [B, L]
        Returns: [B] scalar values (last non-padding token)
        """
        # Find last non-padding position per sequence
        lengths = response_mask.sum(dim=1).long() - 1  # [B]
        lengths = lengths.clamp(min=0)
        last_hidden = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            lengths
        ]  # [B, H]
        return self.linear(last_hidden).squeeze(-1)  # [B]


# PPO inner loop

def ppo_inner_loop(
    policy,
    value_head,
    optimizer,
    rollout: dict,
    args,
    device,
) -> dict:
    """
    Run args.inner_epochs passes of PPO clip + value loss over the rollout
    batch. Returns a dict of scalar metrics averaged over all minibatches.

    The ratio is always π_θ / π_{θ_ref}, where π_{θ_ref} is fixed as the
    rollout-collection policy (stored in rollout["ref_logprobs"]).

    For Nesterov runs, the caller ensures that on entry to this function,
    the model holds θ_k (not θ_ref), so that gradient updates accumulate
    on the correct starting point. After the inner loop, the model holds
    θ_{k+1}.
    """
    B = rollout["prompt_ids"].shape[0]

    # Flatten everything we need for minibatching
    prompt_ids    = rollout["prompt_ids"]     # [B, L_p]
    prompt_mask   = rollout["prompt_mask"]    # [B, L_p]
    response_ids  = rollout["response_ids"]   # [B, L_r]
    response_mask = rollout["response_mask"]  # [B, L_r]
    b_ref_logprobs = rollout["ref_logprobs"].detach()   # [B, L_r]
    b_shaped_rewards = rollout["shaped_rewards"].detach()  # [B]

    # Advantage = shaped reward - value baseline (computed once before loop)
    with torch.no_grad():
        full_ids  = torch.cat([prompt_ids, response_ids], dim=1)
        full_mask = torch.cat([prompt_mask, response_mask.long()], dim=1)
        hidden = policy(
            input_ids=full_ids,
            attention_mask=full_mask,
            output_hidden_states=True,
        ).hidden_states[-1]  # [B, L_p+L_r, H]
        resp_hidden = hidden[:, prompt_ids.shape[1]:, :]  # [B, L_r, H]
        b_values = value_head(resp_hidden, response_mask).detach()  # [B]

    b_advantages = b_shaped_rewards - b_values  # [B]
    if args.norm_adv and b_advantages.std() > 1e-8:
        b_advantages = (b_advantages - b_advantages.mean()) / \
                       (b_advantages.std() + 1e-8)

    b_returns = b_shaped_rewards  # [B] (no GAE needed for bandit setting)

    # Metrics accumulators
    metrics = {
        "pg_loss": [], "vf_loss": [], "entropy": [],
        "approx_kl": [], "clipfrac": [], "early_stop_epoch": args.inner_epochs,
    }

    b_inds = torch.arange(B, device=device)

    for epoch in range(args.inner_epochs):
        perm = b_inds[torch.randperm(B, device=device)]

        for start in range(0, B, args.mini_batch_size):
            mb_inds = perm[start: start + args.mini_batch_size]

            mb_prompt_ids    = prompt_ids[mb_inds]
            mb_prompt_mask   = prompt_mask[mb_inds]
            mb_response_ids  = response_ids[mb_inds]
            mb_response_mask = response_mask[mb_inds]
            mb_ref_lp        = b_ref_logprobs[mb_inds]       # [mb, L_r]
            mb_adv           = b_advantages[mb_inds]         # [mb]
            mb_ret           = b_returns[mb_inds]            # [mb]
            mb_old_val       = b_values[mb_inds]             # [mb]

            # Forward pass
            mb_full_ids  = torch.cat([mb_prompt_ids,  mb_response_ids],  dim=1)
            mb_full_mask = torch.cat([mb_prompt_mask, mb_response_mask.long()], dim=1)

            output = policy(
                input_ids=mb_full_ids,
                attention_mask=mb_full_mask,
                output_hidden_states=True,
            )
            logits = output.logits  # [mb, L_p+L_r, V]
            L_p = mb_prompt_ids.shape[1]
            L_r = mb_response_ids.shape[1]

            resp_logits = logits[:, L_p - 1: L_p + L_r - 1, :]  # [mb, L_r, V]
            log_probs = F.log_softmax(resp_logits, dim=-1)
            new_token_logprobs = log_probs.gather(
                dim=2, index=mb_response_ids.unsqueeze(-1)
            ).squeeze(-1) * mb_response_mask  # [mb, L_r]

            # Sequence-level log-prob = sum over response tokens
            new_logprob = new_token_logprobs.sum(dim=1)  # [mb]
            ref_logprob = (mb_ref_lp * mb_response_mask).sum(dim=1)  # [mb]

            logratio = new_logprob - ref_logprob
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean().item()
                clipfrac = ((ratio - 1).abs() > args.clip_coef).float().mean().item()

            # Policy loss (PPO clip)
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio,
                                              1 - args.clip_coef,
                                              1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            resp_hidden_mb = output.hidden_states[-1][:, L_p:, :]  # [mb, L_r, H]
            new_val = value_head(resp_hidden_mb, mb_response_mask)  # [mb]
            vf_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()

            # Entropy bonus (token-level, averaged over non-padding positions)
            n_tokens = mb_response_mask.sum() + 1e-8
            probs = F.softmax(resp_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)  # [mb, L_r]
            entropy = (entropy * mb_response_mask).sum() / n_tokens

            loss = pg_loss - args.ent_coef * entropy + args.vf_coef * vf_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_head.parameters()),
                args.max_grad_norm,
            )
            optimizer.step()

            metrics["pg_loss"].append(pg_loss.item())
            metrics["vf_loss"].append(vf_loss.item())
            metrics["entropy"].append(entropy.item())
            metrics["approx_kl"].append(approx_kl)
            metrics["clipfrac"].append(clipfrac)

        # Early stopping on KL (per epoch, using last minibatch's KL)
        if args.target_kl is not None and approx_kl > args.target_kl:
            metrics["early_stop_epoch"] = epoch + 1
            break

    return {k: np.mean(v) if isinstance(v, list) else v
            for k, v in metrics.items()}


# Nesterov surrogate evaluation (for adaptive gate)

@torch.no_grad()
def eval_clipped_surrogate(
    model,
    prompt_ids, prompt_mask, response_ids, response_mask,
    ref_logprobs, advantages, clip_coef,
    device,
) -> float:
    """
    Evaluate the clipped PPO surrogate objective at the model's current
    parameters (full batch, no minibatching, for a stable scalar estimate).
    Used by the adaptive Nesterov gate to compare θ_{k+1} vs θ_ref_candidate.
    """
    B   = prompt_ids.shape[0]
    L_p = prompt_ids.shape[1]
    L_r = response_ids.shape[1]

    full_ids  = torch.cat([prompt_ids, response_ids], dim=1)
    full_mask = torch.cat([prompt_mask, response_mask.long()], dim=1)

    logits = model(input_ids=full_ids, attention_mask=full_mask).logits
    resp_logits = logits[:, L_p - 1: L_p + L_r - 1, :]
    log_probs   = F.log_softmax(resp_logits, dim=-1)
    new_lp = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
    new_lp = (new_lp * response_mask).sum(dim=1)  # [B]

    ref_lp = (ref_logprobs * response_mask).sum(dim=1)  # [B]
    ratio  = (new_lp - ref_lp).exp()

    adv = advantages
    surr = torch.min(
        ratio * adv,
        torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * adv,
    ).mean().item()
    return surr


# Qualitative check logger

@torch.no_grad()
def log_qualitative_samples(
    policy, tokenizer, prompts, args, device, iteration, wandb_run
):
    """
    Generate and log a small set of sample completions for qualitative
    inspection. Logged as a W&B Table so you can eyeball degeneration
    at checkpoints.
    """
    policy.eval()
    sample_prompts = prompts[:args.qual_check_n]
    enc = tokenizer(
        sample_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_len,
    ).to(device)

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_gen_len,
        do_sample=False,         # greedy for reproducibility
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    output_ids = policy.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        generation_config=gen_cfg,
    )
    prompt_len = enc["input_ids"].shape[1]
    completions = tokenizer.batch_decode(
        output_ids[:, prompt_len:], skip_special_tokens=True
    )

    table = wandb.Table(columns=["iteration", "prompt_tail", "completion"])
    for p, c in zip(sample_prompts, completions):
        table.add_data(iteration, p[-200:], c)

    wandb_run.log({"qualitative/samples": table}, step=iteration)
    policy.train()


# Main training loop

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # W&B init
    run_name = (
        f"{args.task}__"
        f"alpha{args.nesterov_alpha}__"
        f"{'adaptive' if args.nesterov_adaptive else 'always_on'}__"
        f"seed{args.seed}__"
        f"{int(time.time())}"
    )
    if args.track:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group or f"{args.task}_nesterov_sweep",
            name=run_name,
            config=vars(args),
            save_code=False,
        )
    else:
        wandb_run = wandb.init(mode="disabled")

    # load dataset
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print(f"Loading dataset for task: {args.task}")
    if args.task == "tldr":
        ds = load_dataset("CarperAI/openai_summarize_tldr",
                          cache_dir=args.hf_cache_dir)
        train_prompts = build_tldr_prompts(ds["train"])
        val_prompts   = build_tldr_prompts(ds["valid"])
    else:
        ds = load_dataset("Anthropic/hh-rlhf", cache_dir=args.hf_cache_dir)
        train_prompts = build_hh_prompts(ds["train"])
        val_prompts   = build_hh_prompts(ds["test"])

    print(f"Train prompts: {len(train_prompts)}, Val prompts: {len(val_prompts)}")

    # load reward model
    print(f"Loading reward model: {args.reward_model}")
    reward_model = RewardModel(cache_dir=args.hf_cache_dir, device=device)

    # load policy (TinyLlama)
    print(f"Loading policy: {args.policy_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.policy_model,
        cache_dir=args.hf_cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for decoder-only generation

    base_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        cache_dir=args.hf_cache_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
    )
    policy = get_peft_model(base_model, lora_config)
    policy.print_trainable_parameters()

    # SFT reference model: same base, no LoRA, frozen
    # We reload the base weights so it's a true frozen SFT copy.
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        cache_dir=args.hf_cache_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    sft_model.eval()
    for p in sft_model.parameters():
        p.requires_grad_(False)

    # Value head (trained alongside policy LoRA)
    hidden_size = base_model.config.hidden_size
    value_head = torch.nn.Linear(hidden_size, 1).to(device)
    torch.nn.init.orthogonal_(value_head.weight, gain=1.0)
    torch.nn.init.constant_(value_head.bias, 0.0)

    # Build a small wrapper so value_head integrates with policy's hidden states
    class ValueHeadWrapper(torch.nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear

        def forward(self, hidden_states, response_mask):
            lengths = response_mask.sum(dim=1).long() - 1
            lengths = lengths.clamp(min=0)
            last_h = hidden_states[
                torch.arange(hidden_states.size(0), device=hidden_states.device),
                lengths,
            ]
            return self.linear(last_h).squeeze(-1)

    value_head = ValueHeadWrapper(value_head).to(device)

    # Optimizer (LoRA params + value head)
    trainable_params = (
        [p for p in policy.parameters() if p.requires_grad]
        + list(value_head.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate,
                                 eps=1e-5)

    # Nesterov state initialisation
    use_nesterov = args.nesterov_alpha > 0.0
    if use_nesterov:
        theta_k   = clone_lora_params(policy)   # θ_{k-1} at first iteration
        theta_ref = clone_lora_params(policy)   # θ_ref = θ_0
        # policy currently holds θ_0 = θ_ref; rollout collected with this
    else:
        theta_k   = None
        theta_ref = None

    # Prompt iterator (cycle through train set)
    rng = np.random.default_rng(args.seed)
    prompt_indices = np.arange(len(train_prompts))

    os.makedirs(args.output_dir, exist_ok=True)
    policy.train()

    # outer training loop
    for iteration in range(1, args.num_outer_iterations + 1):

        t_start = time.time()

        # learning rate annealing 
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / args.num_outer_iterations
            for pg in optimizer.param_groups:
                pg["lr"] = frac * args.learning_rate

        #  sample prompts for this rollout 
        idx = rng.choice(prompt_indices, size=args.rollout_batch_size,
                         replace=False)
        batch_prompts = [train_prompts[i] for i in idx]

        #  collect rollout under θ_ref (policy currently holds θ_ref) 
        rollout = collect_rollouts(
            policy, tokenizer, batch_prompts,
            sft_model, reward_model, args, device,
        )

        #  KL from SFT stats for reward-hacking monitoring 
        kl_mean = rollout["kl_from_sft"].mean().item()
        kl_max  = rollout["kl_from_sft"].max().item()

        #  reward whitening ─
        if args.whiten_rewards:
            r = rollout["shaped_rewards"]
            rollout["shaped_rewards"] = (r - r.mean()) / (r.std() + 1e-8)

        #  for Nesterov: restore θ_k so inner loop starts from θ_k ─
        if use_nesterov:
            load_lora_params(policy, theta_k)

        #  PPO inner loop ─
        inner_metrics = ppo_inner_loop(
            policy, value_head, optimizer, rollout, args, device
        )
        # model now holds θ_{k+1}

        #  Nesterov outer-loop update 
        nesterov_metrics = {}
        if use_nesterov:
            theta_kp1 = clone_lora_params(policy)  # θ_{k+1}

            # θ_ref_candidate = θ_{k+1} + α·(θ_{k+1} - θ_k)
            theta_ref_candidate = interpolate_lora_params(
                base=theta_kp1,
                delta_start=theta_k,
                delta_end=theta_kp1,
                alpha=args.nesterov_alpha,
            )

            if args.nesterov_adaptive:
                # Evaluate clipped surrogate at θ_{k+1} (current model state)
                surr_kp1 = eval_clipped_surrogate(
                    policy,
                    rollout["prompt_ids"], rollout["prompt_mask"],
                    rollout["response_ids"], rollout["response_mask"],
                    rollout["ref_logprobs"], rollout["shaped_rewards"],
                    args.clip_coef, device,
                )
                # Evaluate at θ_ref_candidate
                load_lora_params(policy, theta_ref_candidate)
                surr_cand = eval_clipped_surrogate(
                    policy,
                    rollout["prompt_ids"], rollout["prompt_mask"],
                    rollout["response_ids"], rollout["response_mask"],
                    rollout["ref_logprobs"], rollout["shaped_rewards"],
                    args.clip_coef, device,
                )
                # Restore θ_{k+1} before accept/reject
                load_lora_params(policy, theta_kp1)

                accepted = surr_cand > surr_kp1
                theta_ref_next = theta_ref_candidate if accepted else theta_kp1

                nesterov_metrics.update({
                    "nesterov/surr_kp1":      surr_kp1,
                    "nesterov/surr_candidate": surr_cand,
                    "nesterov/surr_delta":     surr_cand - surr_kp1,
                    "nesterov/accepted":       float(accepted),
                })
            else:
                theta_ref_next = theta_ref_candidate

            # KL between θ_ref_next and θ_{k+1} (Nesterov drift diagnostic)
            load_lora_params(policy, theta_ref_next)
            with torch.no_grad():
                lp_ref_next = compute_token_logprobs(
                    policy,
                    rollout["prompt_ids"], rollout["prompt_mask"],
                    rollout["response_ids"], rollout["response_mask"],
                    device,
                )
            load_lora_params(policy, theta_kp1)
            with torch.no_grad():
                lp_kp1 = compute_token_logprobs(
                    policy,
                    rollout["prompt_ids"], rollout["prompt_mask"],
                    rollout["response_ids"], rollout["response_mask"],
                    device,
                )
            kl_ref_vs_kp1 = ((lp_kp1 - lp_ref_next) *
                              rollout["response_mask"]).sum(dim=1).mean().item()

            # Nesterov drift norm (how far θ_ref is from θ_k)
            delta_sq = sum(
                (theta_ref_next[n] - theta_kp1[n]).pow(2).sum()
                for n in theta_ref_next
            )
            param_sq = sum(
                theta_kp1[n].pow(2).sum() for n in theta_kp1
            )
            delta_norm    = torch.sqrt(delta_sq).item()
            relative_norm = delta_norm / (torch.sqrt(param_sq).item() + 1e-8)

            nesterov_metrics.update({
                "nesterov/kl_ref_vs_kp1":    kl_ref_vs_kp1,
                "nesterov/delta_norm":       delta_norm,
                "nesterov/relative_norm":    relative_norm,
            })

            # Advance state
            theta_k   = theta_kp1
            theta_ref = theta_ref_next

            # Load θ_ref into policy for next rollout
            load_lora_params(policy, theta_ref)

        #  logging ─
        t_iter = time.time() - t_start
        rm_mean = rollout["rewards"].mean().item()
        rm_std  = rollout["rewards"].std().item()

        log_dict = {
            "train/rm_score_mean":    rm_mean,
            "train/rm_score_std":     rm_std,
            "train/kl_from_sft_mean": kl_mean,
            "train/kl_from_sft_max":  kl_max,
            "train/shaped_reward_mean": rollout["shaped_rewards"].mean().item(),
            "train/pg_loss":          inner_metrics["pg_loss"],
            "train/vf_loss":          inner_metrics["vf_loss"],
            "train/entropy":          inner_metrics["entropy"],
            "train/approx_kl":        inner_metrics["approx_kl"],
            "train/clipfrac":         inner_metrics["clipfrac"],
            "train/early_stop_epoch": inner_metrics["early_stop_epoch"],
            "train/lr":               optimizer.param_groups[0]["lr"],
            "perf/iter_time_s":       t_iter,
        }
        log_dict.update(nesterov_metrics)

        wandb_run.log(log_dict, step=iteration)

        print(
            f"[{iteration:4d}/{args.num_outer_iterations}] "
            f"RM={rm_mean:.3f}±{rm_std:.3f}  "
            f"KL_sft={kl_mean:.4f}  "
            f"pg={inner_metrics['pg_loss']:.4f}  "
            f"kl_inner={inner_metrics['approx_kl']:.4f}  "
            f"t={t_iter:.1f}s"
        )

        #  qualitative sample logging 
        if iteration % args.qual_check_every == 0:
            # Use val prompts for qualitative checks (not seen during training)
            log_qualitative_samples(
                policy, tokenizer,
                val_prompts[:args.qual_check_n],
                args, device, iteration, wandb_run,
            )

        #  checkpoint 
        if iteration % args.save_every == 0 or \
                iteration == args.num_outer_iterations:
            ckpt_dir = os.path.join(
                args.output_dir,
                run_name,
                f"iter_{iteration:04d}",
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            print(f"  Saved checkpoint: {ckpt_dir}")

    wandb_run.finish()
    print("Training complete")


if __name__ == "__main__":
    main()


"""
PPO-RLHF with outer-loop Nesterov momentum on the LoRA policy iterate.

Claim under test: outer-loop Nesterov momentum on the policy iterate
accelerates PPO-based RLHF, improving sample efficiency and/or final
reward without destabilizing training.

Conditions (controlled by --nesterov-alpha and --nesterov-adaptive):
  alpha=0.0                        -> standard PPO baseline
  alpha>0, --nesterov-adaptive=False -> Nesterov always-on
  alpha>0, --nesterov-adaptive=True  -> Nesterov adaptive (surrogate gate)

Tasks (controlled by --task):
  tldr   -> TL;DR summarization (CarperAI/openai_summarize_tldr)
  hh     -> HH-RLHF helpfulness (Anthropic/hh-rlhf)

Outer-loop Nesterov update rule (per outer iteration k):
  1. Collect rollout with π_{θ_ref}  (b_logprobs = log π_{θ_ref})
  2. Inner loop: K epochs of Adam optimizing PPO clip objective,
     ratio = π_θ / π_{θ_ref}, starting from θ_k  ->  θ_{k+1}
  3. Lookahead: θ_ref_candidate = θ_{k+1} + α·(θ_{k+1} − θ_k)
  4. Adaptive gate (if enabled): accept θ_ref_candidate only if
     clipped surrogate at candidate > surrogate at θ_{k+1};
     otherwise fall back to θ_ref_next = θ_{k+1}
  5. θ_k ← θ_{k+1};  θ_ref ← θ_ref_next
  6. Load θ_ref into model for next rollout

The Nesterov buffer (theta_k, theta_ref) operates on LoRA adapter
parameters only -- frozen base weights are excluded from cloning/loading.
This keeps memory cost negligible (~tens of MB per clone vs ~2GB for
full-param cloning).
Usage (login node smoke test, 5 iterations):
  python src/ppo_rlhf.py --task tldr --nesterov-alpha 0.0 --seed 1 \
      --num-outer-iterations 5 --rollout-batch-size 8 --track False
SLURM sweep:
  sbatch scripts/run_ppo_rlhf.sh --task tldr --nesterov-alpha 0.3 \
      --nesterov-adaptive False --seed 1
"""
