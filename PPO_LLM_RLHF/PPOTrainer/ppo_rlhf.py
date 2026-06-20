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
    parser.add_argument("--task", type=str, choices=["tldr", "hh"], required=True, help="tldr = summarization, hh = helpfulness")
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
    parser.add_argument("--inner-epochs", type=int, default=1, help="Adam epochs over each rollout batch")
    parser.add_argument("--mini-batch-size", type=int, default=16, help="minibatch size for inner loop gradient steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip coefficient epsilon")
    parser.add_argument("--kl-coef", type=float, default=0.2, help="per-token KL penalty coefficient in the reward")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="entropy bonus coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.1, help="value function loss coefficient")
    parser.add_argument("--target-kl", type=float, default=0.02, help="inner-loop early stopping KL threshold (per minibatch)")
    parser.add_argument("--gamma", type=float, default=1.0, help="discount factor (1.0 standard for RLHF)")
    parser.add_argument("--gae-lambda", type=float, default=1.0, help="GAE lambda")
    parser.add_argument("--logratio-clamp", type=float, default=20.0, help="clamp |logratio| before exp() as a numerical guardrail")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="clip value loss like PPO")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    # Generation
    parser.add_argument("--max-prompt-len", type=int, default=512, help="max tokens for prompt tokenization")
    parser.add_argument("--max-gen-len", type=int, default=128, help="max new tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    # Nesterov
    parser.add_argument("--nesterov-alpha", type=float, default=0.0, help="0.0 = standard PPO; >0 = Nesterov lookahead")
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
    args.lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    return args


# LoRA parameter helpers (Nesterov buffer operates on adapter params only)

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


def interpolate_lora_params(base: dict, delta_start: dict, delta_end: dict, alpha: float) -> dict:
    """θ_ref_candidate = θ_{k+1} + alpha * (θ_{k+1} - θ_k)."""
    return {
        name: base[name] + alpha * (base[name] - delta_start[name])
        for name in base
    }


# Token-level reward construction + GAE

def build_token_rewards(rm_scores, ref_logprobs, sft_logprobs, response_mask, kl_coef):
    """
    Dense per-token reward used by GAE:
      r_t = -kl_coef * (logπ_ref(t) - logπ_sft(t))     for every real token t
      r_T += RM_score                                   at the last real token
    rm_scores:    [B]
    ref_logprobs: [B, L_r]
    sft_logprobs: [B, L_r]
    response_mask:[B, L_r]
    Returns token_rewards [B, L_r].
    """
    kl_per_token = (ref_logprobs - sft_logprobs) * response_mask          # [B, L_r]
    token_rewards = -kl_coef * kl_per_token                                # [B, L_r]
    B, L_r = response_mask.shape
    last_idx = (response_mask.sum(dim=1).long() - 1).clamp(min=0)          # [B]
    token_rewards[torch.arange(B, device=token_rewards.device), last_idx] += rm_scores
    return token_rewards * response_mask


def compute_gae(rewards, values, mask, gamma, lam):
    """
    Generalized Advantage Estimation over the response token sequence.
    rewards/values/mask all [B, T]. Returns (advantages [B,T], returns [B,T]).
    Bootstrapping stops at padding (nextnonterminal = mask[t+1]).
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        if t == T - 1:
            nextvalue = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
            nextnonterminal = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
        else:
            nextvalue = values[:, t + 1]
            nextnonterminal = mask[:, t + 1]
        delta = rewards[:, t] + gamma * nextvalue * nextnonterminal - values[:, t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advantages[:, t] = lastgaelam
    returns = advantages + values
    return advantages * mask, returns * mask


# Per-token log-probs and value estimation

@torch.no_grad()
def compute_token_logprobs(model, prompt_ids, prompt_mask, response_ids, response_mask, device):
    """Per-token log-probs of response_ids given prompt. Returns [B, L_r]."""
    B, L_p = prompt_ids.shape
    L_r = response_ids.shape[1]

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    full_mask = torch.cat([prompt_mask, response_mask.long()], dim=1)

    logits = model(input_ids=full_ids, attention_mask=full_mask).logits
    response_logits = logits[:, L_p - 1: L_p + L_r - 1, :]
    log_probs = F.log_softmax(response_logits, dim=-1)
    token_logprobs = log_probs.gather(dim=2, index=response_ids.unsqueeze(-1)).squeeze(-1)
    return token_logprobs * response_mask


@torch.no_grad()
@torch.no_grad()
def compute_token_values(policy, value_head, prompt_ids, prompt_mask, response_ids, response_mask):
    L_p = prompt_ids.shape[1]
    L_r = response_ids.shape[1]
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    full_mask = torch.cat([prompt_mask, response_mask.long()], dim=1)
    hidden = policy(
        input_ids=full_ids,
        attention_mask=full_mask,
        output_hidden_states=True,
    ).hidden_states[-1]
    # Same alignment as token log-probs:
    # logits[:, L_p - 1 : L_p + L_r - 1, :]
    resp_hidden = hidden[:, L_p - 1: L_p + L_r - 1, :]

    return value_head(resp_hidden, response_mask)


# Per-token value head

class ValueHeadWrapper(torch.nn.Module):
    """Per-token value head: maps each response token's hidden state to V(s_t). Returns [B, L_r]."""
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, hidden_states, response_mask):
        v = self.linear(hidden_states.to(self.linear.weight.dtype)).squeeze(-1)  # [B, L_r]
        return v * response_mask


# Rollout collection

@torch.no_grad()
def collect_rollouts(policy, value_head, tokenizer, prompts, sft_model, reward_model, args, device):
    """
    Generate completions, score with RM, build per-token rewards, compute
    per-token ref log-probs and value estimates. Returns everything the
    inner loop needs for GAE-based PPO.
    """
    policy.eval()

    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
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

    output_ids = policy.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        generation_config=gen_cfg,
    )

    prompt_len = enc["input_ids"].shape[1]
    response_ids = output_ids[:, prompt_len:]            # [B, L_r]

    # Response mask: 1 up to and including the first EOS, 0 after
    B, L_r = response_ids.shape
    response_mask = torch.ones_like(response_ids, dtype=torch.float)
    for i in range(B):
        eos_positions = (response_ids[i] == tokenizer.eos_token_id).nonzero()
        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            response_mask[i, first_eos + 1:] = 0.0

    # Per-token ref log-probs (denominator of PPO ratio; policy currently holds θ_ref)
    ref_logprobs = compute_token_logprobs(
        policy, enc["input_ids"], enc["attention_mask"], response_ids, response_mask, device
    )

    # Per-token SFT log-probs (for KL-from-SFT penalty + monitoring)
    sft_logprobs = compute_token_logprobs(
        sft_model, enc["input_ids"], enc["attention_mask"], response_ids, response_mask, device
    )

    # Per-token value estimates V(s_t)
    ref_values = compute_token_values(
        policy, value_head, enc["input_ids"], enc["attention_mask"], response_ids, response_mask
    )

    # RM scores (scalar per sequence)
    completions = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    rm_scores = reward_model.score_batch(prompts, completions).to(dtype=torch.float32, device=device)

    # Whiten RM scores across the batch (zero mean, unit std) so the single
    # terminal RM reward is on a comparable scale to the accumulated per-token
    # KL penalty. Without this, raw DeBERTa logits (~[-6, 7]) are drowned by
    # the sum of ~max_gen_len per-token KL terms, and the policy degenerates
    # toward "stay near SFT, ignore RM". We keep the RAW rm_scores separately
    # for logging/monitoring (so the reported RM curve is in interpretable units).
    rm_scores_whitened = (rm_scores - rm_scores.mean()) / (rm_scores.std() + 1e-8)

    # Dense per-token reward: per-token KL penalty + whitened RM at last real token
    token_rewards = build_token_rewards(rm_scores_whitened, ref_logprobs, sft_logprobs, response_mask, args.kl_coef)

    # GAE
    advantages, returns = compute_gae(
        token_rewards, ref_values, response_mask, args.gamma, args.gae_lambda
    )

    # KL-from-SFT (summed per sequence) for monitoring / reward-hacking tracking
    kl_from_sft = ((ref_logprobs - sft_logprobs) * response_mask).sum(dim=1)  # [B]

    policy.train()

    return {
        "prompt_ids":     enc["input_ids"],
        "prompt_mask":    enc["attention_mask"],
        "response_ids":   response_ids,
        "response_mask":  response_mask,
        "ref_logprobs":   ref_logprobs,        # [B, L_r]
        "ref_values":     ref_values,          # [B, L_r]
        "advantages":     advantages,          # [B, L_r]
        "returns":        returns,             # [B, L_r]
        "rm_scores":      rm_scores,           # [B]
        "kl_from_sft":    kl_from_sft,         # [B]
        "completions":    completions,
    }


# PPO inner loop (GAE, ratio clamp, per-minibatch KL early stop)

def ppo_inner_loop(policy, value_head, optimizer, rollout, args, device):
    """
    args.inner_epochs passes of clipped PPO + clipped value loss over the
    rollout batch, with per-token advantages from GAE. Ratio is π_θ / π_{θ_ref}.

    For Nesterov runs the caller ensures the model holds θ_k on entry; on exit
    the model holds θ_{k+1}.
    """
    B = rollout["prompt_ids"].shape[0]

    prompt_ids    = rollout["prompt_ids"]
    prompt_mask   = rollout["prompt_mask"]
    response_ids  = rollout["response_ids"]
    response_mask = rollout["response_mask"]
    b_ref_logprobs = rollout["ref_logprobs"].detach()    # [B, L_r]
    b_ref_values   = rollout["ref_values"].detach()      # [B, L_r]
    b_advantages   = rollout["advantages"].detach()      # [B, L_r]
    b_returns      = rollout["returns"].detach()         # [B, L_r]

    # Normalize advantages once, over real tokens only
    if args.norm_adv:
        real = response_mask.bool()
        adv_real = b_advantages[real]
        if adv_real.numel() > 1 and adv_real.std() > 1e-8:
            b_advantages = (b_advantages - adv_real.mean()) / (adv_real.std() + 1e-8)
            b_advantages = b_advantages * response_mask

    metrics = {
        "pg_loss": [], "vf_loss": [], "entropy": [],
        "approx_kl": [], "clipfrac": [], "ratio_max": [], "grad_norm": [],
        "early_stop_epoch": args.inner_epochs,
    }

    b_inds = torch.arange(B, device=device)
    stop = False

    for epoch in range(args.inner_epochs):
        perm = b_inds[torch.randperm(B, device=device)]

        for start in range(0, B, args.mini_batch_size):
            mb_inds = perm[start: start + args.mini_batch_size]

            mb_prompt_ids    = prompt_ids[mb_inds]
            mb_prompt_mask   = prompt_mask[mb_inds]
            mb_response_ids  = response_ids[mb_inds]
            mb_response_mask = response_mask[mb_inds]
            mb_ref_lp        = b_ref_logprobs[mb_inds]      # [mb, L_r]
            mb_old_val       = b_ref_values[mb_inds]        # [mb, L_r]
            mb_adv           = b_advantages[mb_inds]        # [mb, L_r]
            mb_ret           = b_returns[mb_inds]           # [mb, L_r]

            mb_full_ids  = torch.cat([mb_prompt_ids, mb_response_ids], dim=1)
            mb_full_mask = torch.cat([mb_prompt_mask, mb_response_mask.long()], dim=1)

            output = policy(input_ids=mb_full_ids, attention_mask=mb_full_mask, output_hidden_states=True)
            logits = output.logits
            L_p = mb_prompt_ids.shape[1]
            L_r = mb_response_ids.shape[1]

            resp_logits = logits[:, L_p - 1: L_p + L_r - 1, :]      # [mb, L_r, V]
            log_probs = F.log_softmax(resp_logits, dim=-1)
            new_lp = log_probs.gather(2, mb_response_ids.unsqueeze(-1)).squeeze(-1) * mb_response_mask  # [mb, L_r]

            # Per-token ratio with logratio clamp (numerical guardrail, fix #4)
            logratio = (new_lp - mb_ref_lp) * mb_response_mask
            logratio = torch.clamp(logratio, -args.logratio_clamp, args.logratio_clamp)
            ratio = logratio.exp()

            with torch.no_grad():
                n_tok = mb_response_mask.sum() + 1e-8
                approx_kl = (((ratio - 1) - logratio) * mb_response_mask).sum().item() / n_tok.item()
                clipfrac = (((ratio - 1.0).abs() > args.clip_coef).float() * mb_response_mask).sum().item() / n_tok.item()
                ratio_max = ratio[mb_response_mask.bool()].max().item() if mb_response_mask.any() else 1.0

            # Per-token clipped policy loss (token-level advantages)
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_response_mask).sum() / n_tok

            # Per-token value loss, regressing to GAE returns (fix #2/#3), optionally clipped
            resp_hidden_mb = output.hidden_states[-1][:, L_p - 1: L_p + L_r - 1, :]   # [mb, L_r, H]
            new_val = value_head(resp_hidden_mb, mb_response_mask)  # [mb, L_r]
            if args.clip_vloss:
                v_clipped = mb_old_val + torch.clamp(new_val - mb_old_val, -args.clip_coef, args.clip_coef)
                vf1 = (new_val - mb_ret) ** 2
                vf2 = (v_clipped - mb_ret) ** 2
                vf_loss = 0.5 * (torch.max(vf1, vf2) * mb_response_mask).sum() / n_tok
            else:
                vf_loss = 0.5 * (((new_val - mb_ret) ** 2) * mb_response_mask).sum() / n_tok

            # Entropy (token-level)
            probs = F.softmax(resp_logits, dim=-1)
            entropy_tok = -(probs * log_probs).sum(dim=-1)         # [mb, L_r]
            entropy = (entropy_tok * mb_response_mask).sum() / n_tok

            loss = pg_loss - args.ent_coef * entropy + args.vf_coef * vf_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_head.parameters()), args.max_grad_norm
            )
            optimizer.step()

            metrics["pg_loss"].append(pg_loss.item())
            metrics["vf_loss"].append(vf_loss.item())
            metrics["entropy"].append(entropy.item())
            metrics["approx_kl"].append(approx_kl)
            metrics["clipfrac"].append(clipfrac)
            metrics["ratio_max"].append(ratio_max)
            metrics["grad_norm"].append(grad_norm.item())

            # Per-minibatch KL early stop (fix #5): break BEFORE the next bad update
            if args.target_kl is not None and approx_kl > args.target_kl:
                metrics["early_stop_epoch"] = epoch + 1
                stop = True
                break
        if stop:
            break

    return {k: (np.mean(v) if isinstance(v, list) and len(v) > 0 else v) for k, v in metrics.items()}


# Nesterov surrogate evaluation (adaptive gate)

@torch.no_grad()
def eval_clipped_surrogate(model, prompt_ids, prompt_mask, response_ids, response_mask,
                           ref_logprobs, advantages, clip_coef, logratio_clamp, device):
    """
    Clipped PPO surrogate at the model's current params, token-level, full batch.
    advantages: [B, L_r] per-token advantages.
    """
    L_p = prompt_ids.shape[1]
    L_r = response_ids.shape[1]
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    full_mask = torch.cat([prompt_mask, response_mask.long()], dim=1)

    logits = model(input_ids=full_ids, attention_mask=full_mask).logits
    resp_logits = logits[:, L_p - 1: L_p + L_r - 1, :]
    log_probs = F.log_softmax(resp_logits, dim=-1)
    new_lp = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1) * response_mask

    logratio = torch.clamp((new_lp - ref_logprobs) * response_mask, -logratio_clamp, logratio_clamp)
    ratio = logratio.exp()
    surr = torch.min(ratio * advantages,
                     torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * advantages)
    n_tok = response_mask.sum() + 1e-8
    return (surr * response_mask).sum().item() / n_tok.item()


# Qualitative check logger

@torch.no_grad()
def log_qualitative_samples(policy, tokenizer, prompts, args, device, iteration, wandb_run):
    policy.eval()
    sample_prompts = prompts[:args.qual_check_n]
    enc = tokenizer(sample_prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=args.max_prompt_len).to(device)
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_gen_len, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    output_ids = policy.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                 generation_config=gen_cfg)
    prompt_len = enc["input_ids"].shape[1]
    completions = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)

    table = wandb.Table(columns=["iteration", "prompt_tail", "completion"])
    for p, c in zip(sample_prompts, completions):
        table.add_data(iteration, p[-200:], c)
    wandb_run.log({"qualitative/samples": table}, step=iteration)
    policy.train()


# Main

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_name = (
        f"{args.task}__alpha{args.nesterov_alpha}__"
        f"{'adaptive' if args.nesterov_adaptive else 'always_on'}__"
        f"seed{args.seed}__{int(time.time())}"
    )
    if args.track:
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            group=args.wandb_group or f"{args.task}_nesterov_sweep",
            name=run_name, config=vars(args), save_code=False,
        )
    else:
        wandb_run = wandb.init(mode="disabled")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print(f"Loading dataset for task: {args.task}")
    if args.task == "tldr":
        ds = load_dataset("CarperAI/openai_summarize_tldr", cache_dir=args.hf_cache_dir)
        train_prompts = build_tldr_prompts(ds["train"])
        val_prompts = build_tldr_prompts(ds["valid"])
    else:
        ds = load_dataset("Anthropic/hh-rlhf", cache_dir=args.hf_cache_dir)
        train_prompts = build_hh_prompts(ds["train"])
        val_prompts = build_hh_prompts(ds["test"])
    print(f"Train prompts: {len(train_prompts)}, Val prompts: {len(val_prompts)}")

    print(f"Loading reward model: {args.reward_model}")
    reward_model = RewardModel(model_name=args.reward_model, cache_dir=args.hf_cache_dir, device=device)

    print(f"Loading policy: {args.policy_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, cache_dir=args.hf_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model, cache_dir=args.hf_cache_dir,
        torch_dtype=torch.bfloat16, device_map=device,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.lora_target_modules, bias="none",
    )
    policy = get_peft_model(base_model, lora_config)
    policy.print_trainable_parameters()

    sft_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model, cache_dir=args.hf_cache_dir,
        torch_dtype=torch.bfloat16, device_map=device,
    )
    sft_model.eval()
    for p in sft_model.parameters():
        p.requires_grad_(False)

    # Per-token value head
    hidden_size = base_model.config.hidden_size
    value_linear = torch.nn.Linear(hidden_size, 1).to(device)
    torch.nn.init.orthogonal_(value_linear.weight, gain=1.0)
    torch.nn.init.constant_(value_linear.bias, 0.0)
    value_head = ValueHeadWrapper(value_linear).to(device)

    trainable_params = [p for p in policy.parameters() if p.requires_grad] + list(value_head.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate, eps=1e-5)

    # Nesterov state
    use_nesterov = args.nesterov_alpha > 0.0
    if use_nesterov:
        theta_k = clone_lora_params(policy)
        theta_ref = clone_lora_params(policy)
    else:
        theta_k = None
        theta_ref = None

    rng = np.random.default_rng(args.seed)
    prompt_indices = np.arange(len(train_prompts))

    os.makedirs(args.output_dir, exist_ok=True)
    policy.train()

    for iteration in range(1, args.num_outer_iterations + 1):
        t_start = time.time()

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / args.num_outer_iterations
            for pg in optimizer.param_groups:
                pg["lr"] = frac * args.learning_rate

        idx = rng.choice(prompt_indices, size=args.rollout_batch_size, replace=False)
        batch_prompts = [train_prompts[i] for i in idx]

        # Rollout under θ_ref (policy currently holds θ_ref)
        rollout = collect_rollouts(policy, value_head, tokenizer, batch_prompts,
                                   sft_model, reward_model, args, device)

        kl_mean = rollout["kl_from_sft"].mean().item()
        kl_max = rollout["kl_from_sft"].max().item()

        # For Nesterov: inner loop must start from θ_k
        if use_nesterov:
            load_lora_params(policy, theta_k)

        inner_metrics = ppo_inner_loop(policy, value_head, optimizer, rollout, args, device)
        # model now holds θ_{k+1}

        nesterov_metrics = {}
        if use_nesterov:
            theta_kp1 = clone_lora_params(policy)
            theta_ref_candidate = interpolate_lora_params(
                base=theta_kp1, delta_start=theta_k, delta_end=theta_kp1, alpha=args.nesterov_alpha
            )

            if args.nesterov_adaptive:
                surr_kp1 = eval_clipped_surrogate(
                    policy, rollout["prompt_ids"], rollout["prompt_mask"],
                    rollout["response_ids"], rollout["response_mask"],
                    rollout["ref_logprobs"], rollout["advantages"],
                    args.clip_coef, args.logratio_clamp, device,
                )
                load_lora_params(policy, theta_ref_candidate)
                surr_cand = eval_clipped_surrogate(
                    policy, rollout["prompt_ids"], rollout["prompt_mask"],
                    rollout["response_ids"], rollout["response_mask"],
                    rollout["ref_logprobs"], rollout["advantages"],
                    args.clip_coef, args.logratio_clamp, device,
                )
                load_lora_params(policy, theta_kp1)
                accepted = surr_cand > surr_kp1
                theta_ref_next = theta_ref_candidate if accepted else theta_kp1
                nesterov_metrics.update({
                    "nesterov/surr_kp1": surr_kp1,
                    "nesterov/surr_candidate": surr_cand,
                    "nesterov/surr_delta": surr_cand - surr_kp1,
                    "nesterov/accepted": float(accepted),
                })
            else:
                theta_ref_next = theta_ref_candidate

            # KL between θ_ref_next and θ_{k+1} (Nesterov drift diagnostic)
            load_lora_params(policy, theta_ref_next)
            lp_ref_next = compute_token_logprobs(
                policy, rollout["prompt_ids"], rollout["prompt_mask"],
                rollout["response_ids"], rollout["response_mask"], device,
            )
            load_lora_params(policy, theta_kp1)
            lp_kp1 = compute_token_logprobs(
                policy, rollout["prompt_ids"], rollout["prompt_mask"],
                rollout["response_ids"], rollout["response_mask"], device,
            )
            kl_ref_vs_kp1 = ((lp_kp1 - lp_ref_next) * rollout["response_mask"]).sum(dim=1).mean().item()

            delta_sq = sum((theta_ref_next[n] - theta_kp1[n]).pow(2).sum() for n in theta_ref_next)
            param_sq = sum(theta_kp1[n].pow(2).sum() for n in theta_kp1)
            delta_norm = torch.sqrt(delta_sq).item()
            relative_norm = delta_norm / (torch.sqrt(param_sq).item() + 1e-8)
            nesterov_metrics.update({
                "nesterov/kl_ref_vs_kp1": kl_ref_vs_kp1,
                "nesterov/delta_norm": delta_norm,
                "nesterov/relative_norm": relative_norm,
            })

            theta_k = theta_kp1
            theta_ref = theta_ref_next
            load_lora_params(policy, theta_ref)

        t_iter = time.time() - t_start
        rm_mean = rollout["rm_scores"].mean().item()
        rm_std = rollout["rm_scores"].std().item()

        log_dict = {
            "train/rm_score_mean": rm_mean,
            "train/rm_score_std": rm_std,
            "train/kl_from_sft_mean": kl_mean,
            "train/kl_from_sft_max": kl_max,
            "train/pg_loss": inner_metrics["pg_loss"],
            "train/vf_loss": inner_metrics["vf_loss"],
            "train/entropy": inner_metrics["entropy"],
            "train/approx_kl": inner_metrics["approx_kl"],
            "train/clipfrac": inner_metrics["clipfrac"],
            "train/ratio_max": inner_metrics["ratio_max"],
            "train/early_stop_epoch": inner_metrics["early_stop_epoch"],
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/grad_norm": inner_metrics["grad_norm"],
            "perf/iter_time_s": t_iter,
        }
        log_dict.update(nesterov_metrics)
        wandb_run.log(log_dict, step=iteration)

        print(
            f"[{iteration:4d}/{args.num_outer_iterations}] "
            f"RM={rm_mean:.3f}±{rm_std:.3f}  KL_sft={kl_mean:.4f}  "
            f"pg={inner_metrics['pg_loss']:.4f}  kl_inner={inner_metrics['approx_kl']:.4f}  "
            f"ratio_max={inner_metrics['ratio_max']:.2f}  t={t_iter:.1f}s"
        )

        if iteration % args.qual_check_every == 0:
            log_qualitative_samples(policy, tokenizer, val_prompts[:args.qual_check_n],
                                    args, device, iteration, wandb_run)

        if iteration % args.save_every == 0 or iteration == args.num_outer_iterations:
            ckpt_dir = os.path.join(args.output_dir, run_name, f"iter_{iteration:04d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            print(f"  Saved checkpoint: {ckpt_dir}")

    wandb_run.finish()
    print("Training complete")


if __name__ == "__main__":
    main()