import re
import wandb
import pandas as pd
import numpy as np

ENTITY = "hajardoukhou-um6p"
PROJECT = "ppo_beta1_sweep"

METRIC = "charts/episodic_return"
STEP_KEY = "step"

OUTPUT_CSV = "beta1_final_mean_std_gap.csv"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

rows = []
skipped = []


def parse_beta1_from_text(text):
    """
    Extract beta1 from strings like:
    HalfCheetah-v5_b1_0p95_seed3
    Pusher-v5_b1_0p0
    """
    if not text:
        return None

    match = re.search(r"_b1_([0-9]+p?[0-9]*)", text)
    if match is None:
        return None

    beta_str = match.group(1).replace("p", ".")
    try:
        return float(beta_str)
    except ValueError:
        return None


def parse_seed_from_text(text):
    """
    Extract seed from strings like:
    HalfCheetah-v5_b1_0p95_seed3
    """
    if not text:
        return None

    match = re.search(r"_seed(\d+)", text)
    if match is None:
        return None

    try:
        return int(match.group(1))
    except ValueError:
        return None


def parse_env_from_text(text):
    """
    Extract env from strings like:
    HalfCheetah-v5_b1_0p95_seed3
    """
    if not text:
        return None

    if "_b1_" in text:
        return text.split("_b1_")[0]

    return None


def get_first_existing(config, keys):
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return None


for run in runs:
    config = run.config

        # 1. Robust metadata extraction
    
    env = get_first_existing(
        config,
        ["gym_id", "env_id", "env", "environment"]
    )

    beta1 = get_first_existing(
        config,
        [
            "beta1",
            "beta_1",
            "beta-one",
            "beta_one",
            "adam_beta1",
            "adam_beta_one",
        ]
    )

    seed = get_first_existing(
        config,
        ["seed", "random_seed"]
    )

    # Fallback parsing from W&B name/group
    if env is None:
        env = parse_env_from_text(run.name) or parse_env_from_text(run.group)

    if beta1 is None:
        beta1 = parse_beta1_from_text(run.name) or parse_beta1_from_text(run.group)

    if seed is None:
        seed = parse_seed_from_text(run.name)

    # Final validation
    if env is None or beta1 is None or seed is None:
        skipped.append(
            {
                "run_name": run.name,
                "group": run.group,
                "reason": "missing env/beta1/seed",
                "env": env,
                "beta1": beta1,
                "seed": seed,
            }
        )
        continue

    try:
        beta1 = float(beta1)
        seed = int(seed)
    except Exception:
        skipped.append(
            {
                "run_name": run.name,
                "group": run.group,
                "reason": "could not cast beta1/seed",
                "env": env,
                "beta1": beta1,
                "seed": seed,
            }
        )
        continue

        # 2. Robust metric extraction
    
    hist = []

    try:
        for row in run.scan_history(keys=[STEP_KEY, METRIC]):
            step = row.get(STEP_KEY)
            value = row.get(METRIC)

            if step is None or value is None:
                continue

            hist.append(
                {
                    "global_step": step,
                    "episodic_return": value,
                }
            )

    except Exception as e:
        skipped.append(
            {
                "run_name": run.name,
                "group": run.group,
                "reason": f"scan_history failed: {e}",
                "env": env,
                "beta1": beta1,
                "seed": seed,
            }
        )
        continue

    if len(hist) == 0:
        skipped.append(
            {
                "run_name": run.name,
                "group": run.group,
                "reason": "no history found",
                "env": env,
                "beta1": beta1,
                "seed": seed,
            }
        )
        continue

    df = pd.DataFrame(hist)

    if "global_step" not in df.columns or "episodic_return" not in df.columns:
        skipped.append(
            {
                "run_name": run.name,
                "group": run.group,
                "reason": "missing global_step or episodic_return in history",
                "env": env,
                "beta1": beta1,
                "seed": seed,
            }
        )
        continue

    df = df.dropna(subset=["global_step", "episodic_return"])

    if df.empty:
        skipped.append(
            {
                "run_name": run.name,
                "group": run.group,
                "reason": "history only contains NaNs",
                "env": env,
                "beta1": beta1,
                "seed": seed,
            }
        )
        continue

        # 3. Final-window score
    
    max_step = df["global_step"].max()
    final_start = 0.9 * max_step

    final_df = df[df["global_step"] >= final_start]

    if final_df.empty:
        skipped.append(
            {
                "run_name": run.name,
                "group": run.group,
                "reason": "empty final window",
                "env": env,
                "beta1": beta1,
                "seed": seed,
            }
        )
        continue

    final_score = final_df["episodic_return"].mean()

    rows.append(
        {
            "env": env,
            "beta1": beta1,
            "seed": seed,
            "run_name": run.name,
            "group": run.group,
            "max_step": max_step,
            "final_score": final_score,
        }
    )


# 4. Build seed-level dataframe safely

seed_scores = pd.DataFrame(rows)

print("Collected valid runs")
print(f"Number of valid seed-level runs: {len(seed_scores)}")

if len(skipped) > 0:
    skipped_df = pd.DataFrame(skipped)

    print("Skipped runs")

    print(f"Number of skipped runs: {len(skipped_df)}")
    print(skipped_df[["run_name", "group", "reason", "env", "beta1", "seed"]].head(30))

    skipped_df.to_csv("beta1_skipped_runs.csv", index=False)
    print("\nSaved skipped-run diagnostics to beta1_skipped_runs.csv")


if seed_scores.empty:
    print("\nERROR: No valid runs were collected.")
    print("This usually means one of these is wrong:")
    print(f"  - Metric name: {METRIC}")
    print(f"  - Step key: {STEP_KEY}")
    print("  - Metadata keys: gym_id, beta_one, seed")
    print("  - W&B project/entity")
    print("\nNo summary CSV was created.")
    raise SystemExit(1)


required_columns = ["env", "beta1", "seed", "final_score"]

missing_columns = [
    col for col in required_columns
    if col not in seed_scores.columns
]

if missing_columns:
    print("\nERROR: seed_scores is missing columns:")
    print(missing_columns)
    print("\nAvailable columns:")
    print(seed_scores.columns.tolist())
    raise SystemExit(1)


print("\nPreview of collected seed scores:")
print(seed_scores.head())


# 5. Aggregate over seeds

summary = (
    seed_scores
    .groupby(["env", "beta1"])
    .agg(
        n_seeds=("seed", "nunique"),
        final_mean=("final_score", "mean"),
        final_std=("final_score", "std"),
        best_seed_score=("final_score", "max"),
        worst_seed_score=("final_score", "min"),
        median_score=("final_score", "median"),
        q25_score=("final_score", lambda x: x.quantile(0.25)),
        q75_score=("final_score", lambda x: x.quantile(0.75)),
    )
    .reset_index()
)

summary["best_worst_gap"] = (
    summary["best_seed_score"] - summary["worst_seed_score"]
)

summary["iqr"] = (
    summary["q75_score"] - summary["q25_score"]
)

summary["relative_seed_spread"] = np.where(
    summary["final_mean"].abs() > 1e-8,
    summary["final_std"] / summary["final_mean"].abs(),
    np.nan,
)

summary = summary.sort_values(["env", "beta1"])

print("Summary")
print(summary)

summary.to_csv(OUTPUT_CSV, index=False)
seed_scores.to_csv("beta1_seed_scores.csv", index=False)

print(f"\nSaved summary to {OUTPUT_CSV}")
print("Saved seed-level scores to beta1_seed_scores.csv")