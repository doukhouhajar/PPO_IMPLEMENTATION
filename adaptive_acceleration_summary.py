#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wandb


DEFAULT_ENTITY = "hajardoukhou-um6p"
DEFAULT_PROJECT_PREFIX = "ppo_nesterov_adaptive_parameters"

ENVS = [
    "HalfCheetah-v5",
    "Walker2d-v5",
    "Ant-v5",
    "Humanoid-v5",
    "InvertedPendulum-v5",
    "Hopper-v5",
    "InvertedDoublePendulum-v5",
    "Swimmer-v5",
    "Pusher-v5",
    "Reacher-v5",
    "HumanoidStandup-v5",
]

RETURN_KEY = "charts/episodic_return"
ADAPTIVE_ACCEPTED_KEY = "debug/nesterov_accepted"

LAST_N_EPISODES = 200
ROLLING_WINDOW_EPISODES = 20

OUTPUT_COLUMNS = [
    "env",
    "distribution",
    "alpha",
    "n_mean_return",
    "mean_mean_return",
    "median_mean_return",
    "std_mean_return",
    "max_mean_return",
    "min_mean_return",
    "auc_return_mean",
    "auc_return_std",
    "steps_to_baseline_final_return_mean",
    "baseline_final_return_mean",
    "relative_final_improvement_pct_vs_alpha0",
    "sample_efficiency_step_gain_pct_vs_alpha0",
    "adaptive_acceptance_rate_mean",
    "adaptive_acceptance_rate_std",
]


def safe_float(value, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def parse_group_name(group_name: Optional[str]) -> Tuple[str, float]:
    text = (group_name or "").lower()

    if "beta" in text:
        distribution = "beta"
    elif "normal" in text or "gaussian" in text:
        distribution = "normal"
    else:
        distribution = "unknown"

    alpha = np.nan
    match = re.search(r"alpha(?P<alpha>\d+(?:[p.]\d+)?)", text)
    if match:
        alpha = safe_float(match.group("alpha").replace("p", "."))

    return distribution, alpha


def infer_run_identity(run) -> Tuple[str, float]:
    distribution, alpha = parse_group_name(run.group)

    if distribution == "unknown":
        text = " ".join(
            str(x or "")
            for x in [
                run.name,
                run.config.get("wandb_group_name"),
                run.config.get("distribution"),
                run.config.get("policy_distribution"),
                run.config.get("exp_name"),
            ]
        ).lower()

        if "beta" in text:
            distribution = "beta"
        elif "normal" in text or "gaussian" in text:
            distribution = "normal"

    if not np.isfinite(alpha):
        alpha = safe_float(run.config.get("nesterov_alpha"))

    return distribution, alpha


def fetch_return_curve(run) -> pd.DataFrame:
    try:
        rows = list(run.scan_history(keys=["_step", RETURN_KEY]))
    except Exception as exc:
        print(f"    [WARN] run={run.id}: could not scan return history: {exc}")
        rows = []

    if not rows:
        fallback = safe_float(run.summary.get(RETURN_KEY))
        if np.isfinite(fallback):
            print(f"    [INFO] run={run.id}: using summary fallback return={fallback:.2f}")
            return pd.DataFrame({"step": [np.nan], "episodic_return": [fallback]})

        print(f"    [WARN] run={run.id}: missing {RETURN_KEY}")
        return pd.DataFrame(columns=["step", "episodic_return"])

    df = pd.DataFrame(rows)
    if RETURN_KEY not in df.columns:
        return pd.DataFrame(columns=["step", "episodic_return"])

    if "_step" in df.columns:
        step = pd.to_numeric(df["_step"], errors="coerce")
    else:
        step = pd.Series(np.arange(len(df)), dtype=float)

    out = pd.DataFrame(
        {
            "step": step,
            "episodic_return": pd.to_numeric(df[RETURN_KEY], errors="coerce"),
        }
    ).dropna(subset=["episodic_return"])

    if out.empty:
        return pd.DataFrame(columns=["step", "episodic_return"])

    if out["step"].isna().all():
        out["step"] = np.arange(len(out), dtype=float)

    out = out.dropna(subset=["step"])
    out = out.sort_values("step").reset_index(drop=True)
    return out


def fetch_adaptive_acceptance_rate(run, alpha: float) -> float:
    if np.isfinite(alpha) and np.isclose(alpha, 0.0):
        return np.nan

    try:
        rows = list(run.scan_history(keys=[ADAPTIVE_ACCEPTED_KEY]))
    except Exception as exc:
        print(f"    [WARN] run={run.id}: could not scan adaptive history: {exc}")
        return np.nan

    if not rows:
        print(f"    [WARN] run={run.id}: missing {ADAPTIVE_ACCEPTED_KEY}")
        return np.nan

    df = pd.DataFrame(rows)
    if ADAPTIVE_ACCEPTED_KEY not in df.columns:
        return np.nan

    values = pd.to_numeric(df[ADAPTIVE_ACCEPTED_KEY], errors="coerce").dropna()
    if values.empty:
        return np.nan

    return float(values.mean())


def final_mean_return(curve: pd.DataFrame, last_n_episodes: int) -> float:
    if curve.empty:
        return np.nan
    return float(curve["episodic_return"].tail(last_n_episodes).mean())


def normalized_auc_return(curve: pd.DataFrame) -> float:
    if curve.empty:
        return np.nan

    tmp = curve[["step", "episodic_return"]].dropna().copy()
    if tmp.empty:
        return np.nan

    # Average duplicate steps if W&B/TensorBoard emitted several episode returns
    # at the same global_step.
    tmp = tmp.groupby("step", as_index=False)["episodic_return"].mean()

    x = tmp["step"].to_numpy(dtype=float)
    y = tmp["episodic_return"].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(y) == 0:
        return np.nan

    if len(y) == 1 or np.nanmax(x) == np.nanmin(x):
        return float(np.nanmean(y))

    return trapezoid_area(y, x) / float(np.nanmax(x) - np.nanmin(x))


def first_step_reaching_threshold(
    curve: pd.DataFrame,
    threshold: float,
    rolling_window_episodes: int,
) -> float:
    if curve.empty or not np.isfinite(threshold):
        return np.nan

    tmp = curve[["step", "episodic_return"]].dropna().copy()
    if tmp.empty:
        return np.nan

    tmp = tmp.sort_values("step").reset_index(drop=True)

    rolling_return = tmp["episodic_return"].rolling(
        window=rolling_window_episodes,
        min_periods=1,
    ).mean()

    reached = tmp.loc[rolling_return >= threshold]
    if reached.empty:
        return np.nan

    return safe_float(reached["step"].iloc[0])


def collect_raw_runs(args) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    api = wandb.Api(timeout=args.wandb_timeout)
    raw_records: List[dict] = []
    curves_by_run_id: Dict[str, pd.DataFrame] = {}

    for env in args.envs:
        project_path = f"{args.entity}/{args.project_prefix}_{env}"
        print(f"\nProject: {project_path}")

        try:
            runs = list(api.runs(project_path))
        except Exception as exc:
            print(f"[ERROR] Could not fetch project {project_path}: {exc}")
            continue

        print(f"Found {len(runs)} runs")

        for run in runs:
            distribution, alpha = infer_run_identity(run)
            seed = run.config.get("seed", np.nan)

            if args.only_finished and run.state != "finished":
                continue

            if distribution == "unknown" or not np.isfinite(alpha):
                print(
                    f"    [WARN] run={run.id}: could not infer distribution/alpha "
                    f"from group={run.group!r}, name={run.name!r}"
                )

            curve = fetch_return_curve(run)
            curves_by_run_id[run.id] = curve

            raw_records.append(
                {
                    "env": env,
                    "distribution": distribution,
                    "alpha": alpha,
                    "seed": seed,
                    "run_id": run.id,
                    "run_state": run.state,
                    "mean_return": final_mean_return(curve, args.last_n_episodes),
                    "auc_return": normalized_auc_return(curve),
                    "adaptive_acceptance_rate": fetch_adaptive_acceptance_rate(run, alpha),
                }
            )

    return pd.DataFrame(raw_records), curves_by_run_id


def build_summary(
    raw: pd.DataFrame,
    curves_by_run_id: Dict[str, pd.DataFrame],
    args,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    group_cols = ["env", "distribution", "alpha"]
    baseline_cols = ["env", "distribution"]

    summary = (
        raw.groupby(group_cols, dropna=False)
        .agg(
            n_mean_return=("mean_return", "count"),
            mean_mean_return=("mean_return", "mean"),
            median_mean_return=("mean_return", "median"),
            std_mean_return=("mean_return", "std"),
            max_mean_return=("mean_return", "max"),
            min_mean_return=("mean_return", "min"),
            auc_return_mean=("auc_return", "mean"),
            auc_return_std=("auc_return", "std"),
            adaptive_acceptance_rate_mean=("adaptive_acceptance_rate", "mean"),
            adaptive_acceptance_rate_std=("adaptive_acceptance_rate", "std"),
        )
        .reset_index()
    )

    baseline_final = (
        summary[np.isclose(summary["alpha"], 0.0, equal_nan=False)]
        [baseline_cols + ["mean_mean_return"]]
        .rename(columns={"mean_mean_return": "baseline_final_return_mean"})
    )

    summary = summary.merge(baseline_final, on=baseline_cols, how="left")

    raw_with_baseline = raw.merge(baseline_final, on=baseline_cols, how="left")

    steps_records = []
    for row in raw_with_baseline.itertuples(index=False):
        threshold = safe_float(getattr(row, "baseline_final_return_mean", np.nan))
        curve = curves_by_run_id.get(row.run_id, pd.DataFrame())

        steps = first_step_reaching_threshold(
            curve=curve,
            threshold=threshold,
            rolling_window_episodes=args.rolling_window_episodes,
        )

        steps_records.append(
            {
                "env": row.env,
                "distribution": row.distribution,
                "alpha": row.alpha,
                "run_id": row.run_id,
                "steps_to_baseline_final_return": steps,
            }
        )

    steps_df = pd.DataFrame(steps_records)
    steps_summary = (
        steps_df.groupby(group_cols, dropna=False)
        .agg(
            steps_to_baseline_final_return_mean=(
                "steps_to_baseline_final_return",
                "mean",
            )
        )
        .reset_index()
    )

    summary = summary.merge(steps_summary, on=group_cols, how="left")

    baseline_steps = (
        summary[np.isclose(summary["alpha"], 0.0, equal_nan=False)]
        [baseline_cols + ["steps_to_baseline_final_return_mean"]]
        .rename(
            columns={
                "steps_to_baseline_final_return_mean": (
                    "baseline_steps_to_baseline_final_return_mean"
                )
            }
        )
    )

    summary = summary.merge(baseline_steps, on=baseline_cols, how="left")

    denom_return = summary["baseline_final_return_mean"].abs().replace(0.0, np.nan)
    summary["relative_final_improvement_pct_vs_alpha0"] = (
        (
            summary["mean_mean_return"]
            - summary["baseline_final_return_mean"]
        )
        / denom_return
        * 100.0
    )

    # Positive means the configuration reaches the alpha=0 final-return level earlier.
    denom_steps = summary[
        "baseline_steps_to_baseline_final_return_mean"
    ].replace(0.0, np.nan)

    summary["sample_efficiency_step_gain_pct_vs_alpha0"] = (
        (
            summary["baseline_steps_to_baseline_final_return_mean"]
            - summary["steps_to_baseline_final_return_mean"]
        )
        / denom_steps
        * 100.0
    )

    is_baseline = np.isclose(summary["alpha"], 0.0, equal_nan=False)
    summary.loc[is_baseline, "relative_final_improvement_pct_vs_alpha0"] = 0.0
    summary.loc[is_baseline, "sample_efficiency_step_gain_pct_vs_alpha0"] = 0.0

    summary = summary[OUTPUT_COLUMNS].copy()
    summary = summary.sort_values(["env", "distribution", "alpha"]).reset_index(drop=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--entity", type=str, default=DEFAULT_ENTITY)
    parser.add_argument("--project-prefix", type=str, default=DEFAULT_PROJECT_PREFIX)
    parser.add_argument("--envs", nargs="+", default=ENVS)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/hajar.doukhou/PPO_IMPLEMENTATION/summary_tables",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="adaptive_accelerated_ppo_summary.csv",
    )

    parser.add_argument("--last-n-episodes", type=int, default=LAST_N_EPISODES)
    parser.add_argument(
        "--rolling-window-episodes",
        type=int,
        default=ROLLING_WINDOW_EPISODES,
        help="Rolling episode window used to decide when a run reaches the baseline final return.",
    )
    parser.add_argument("--wandb-timeout", type=int, default=90)
    parser.add_argument(
        "--only-finished",
        type=str_to_bool,
        default=False,
        help="If True, ignore W&B runs whose state is not 'finished'.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw, curves_by_run_id = collect_raw_runs(args)
    summary = build_summary(raw, curves_by_run_id, args)

    output_path = os.path.join(args.output_dir, args.output_file)
    summary.to_csv(output_path, index=False)

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.width", 220)

    if summary.empty:
        print("No valid runs found.")
    else:
        print(summary.to_string(index=False))

    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
