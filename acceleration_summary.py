import wandb
import pandas as pd
import re

entity = "hajardoukhou-um6p"
envs = [
    "HalfCheetah-v5", "Walker2d-v5", 
    "InvertedDoublePendulum-v5",
    "Swimmer-v5", "Pusher-v5", "Reacher-v5", "HumanoidStandup-v5"
]
# "Ant-v5", "Humanoid-v5", "InvertedPendulum-v5", "Hopper-v5", 

performance_metrics = ["charts/episodic_return"]
stability_metrics   = ["loss/approx_kl", "loss/explained_variance"]
momentum_metrics    = ["debug/nesterov_ref_delta_norm", "debug/nesterov_kl_ref_vs_corrected"]

performance_records = []
stability_records   = []
momentum_records    = []

def parse_group_name(group_name):
    match = re.match(r"(normal|beta)_parameters_alpha(\d+)p?(\d*)", group_name)
    if match:
        dist = match.group(1)
        alpha_str = match.group(2)
        alpha_dec = match.group(3) if match.group(3) else "0"
        alpha = float(f"{alpha_str}.{alpha_dec}")
        return dist, alpha
    else:
        return "unknown", -1

api = wandb.Api()

for env in envs:
    project_name = f"ppo_nesterov_corrected_parameters_{env}"
    runs = api.runs(f"{entity}/{project_name}")

    for run in runs:
        group_name = run.group or "unknown"
        dist, alpha = parse_group_name(group_name)

        # Performance
        perf_vals = [run.summary.get(m) for m in performance_metrics]
        performance_records.append({
            "env": env,
            "distribution": dist,
            "alpha": alpha,
            "mean_return": perf_vals[0],
        })

        # Stability
        stab_vals = [run.summary.get(m) for m in stability_metrics]
        stability_records.append({
            "env": env,
            "distribution": dist,
            "alpha": alpha,
            "approx_kl": stab_vals[0],
            "explained_variance": stab_vals[1],
        })

        # Momentum
        mom_vals = [run.summary.get(m) for m in momentum_metrics]
        momentum_records.append({
            "env": env,
            "distribution": dist,
            "alpha": alpha,
            "nesterov_ref_delta_norm": mom_vals[0],
            "nesterov_kl_ref_vs_corrected": mom_vals[1],
        })

df_perf = pd.DataFrame(performance_records)
df_stab = pd.DataFrame(stability_records)
df_mom  = pd.DataFrame(momentum_records)

# Aggregate across seeds
perf_summary = df_perf.groupby(["env","distribution","alpha"]).agg(
    mean_return=("mean_return","mean"),
    median_return=("mean_return","median"),
    std_return=("mean_return","std"),
    max_return=("mean_return","max"),
    min_return=("mean_return","min")
).reset_index()

stab_summary = df_stab.groupby(["env","distribution","alpha"]).agg(
    mean_approx_kl=("approx_kl","mean"),
    std_approx_kl=("approx_kl","std"),
    mean_explained_variance=("explained_variance","mean"),
    std_explained_variance=("explained_variance","std")
).reset_index()

mom_summary = df_mom.groupby(["env","distribution","alpha"]).agg(
    mean_nesterov_ref_delta_norm=("nesterov_ref_delta_norm","mean"),
    std_nesterov_ref_delta_norm=("nesterov_ref_delta_norm","std"),
    mean_nesterov_kl_ref_vs_corrected=("nesterov_kl_ref_vs_corrected","mean"),
    std_nesterov_kl_ref_vs_corrected=("nesterov_kl_ref_vs_corrected","std")
).reset_index()

perf_summary.to_csv("performance_summary.csv", index=False)
stab_summary.to_csv("stability_summary.csv", index=False)
mom_summary.to_csv("momentum_summary.csv", index=False)

print("Summary tables saved")
print("Performance Summary:")
print(perf_summary.head())
print("\nStability Summary:")
print(stab_summary.head())
print("\nMomentum Effect Summary:")
print(mom_summary.head())