"""
runners/bandit.py — Multi-armed bandit experiment runner.

Runs epsilon-greedy and Thompson Sampling on the same 18-arm creative space,
using the same RNG seed as ab_testing.py for fair comparison.

Generates results/regret_comparison.png showing cumulative regret over time
for all three methods (A/B, epsilon-greedy, Thompson Sampling).

Run:
    python -m runners.bandit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.arms import make_arms
from src.algorithms import epsilon_greedy, thompson_sampling, update_thompson
from src.metrics import cumulative_regret, convergence_point, comparison_table
from src.simulator import ExperimentResult, run_trial

SEED = 42
N_TRIALS = 1500  # same total impressions as A/B (3 phases × 500)

HEADLINES = ["Bold claim", "Question hook", "Social proof"]
IMAGES = ["Lifestyle photo", "Product close-up", "Illustration"]
CTAS = ["Shop now", "Learn more"]


def run_epsilon_greedy(arms, n_trials, rng):
    counts = [0] * len(arms)
    rewards = [0.0] * len(arms)
    selections = []
    trial_rewards = []

    for t in range(n_trials):
        idx = epsilon_greedy(counts, rewards, rng)
        r = run_trial(arms[idx], rng)
        counts[idx] += 1
        rewards[idx] += r
        selections.append(idx)
        trial_rewards.append(r)

    best_arm_idx = int(max(range(len(arms)), key=lambda i: rewards[i] / max(counts[i], 1)))
    return ExperimentResult(
        selections=selections,
        rewards=trial_rewards,
        counts=counts,
        arm_rewards=rewards,
        best_arm_idx=best_arm_idx,
        n_trials=n_trials,
    ), counts, rewards


def run_thompson(arms, n_trials, rng):
    alphas = [1.0] * len(arms)
    betas = [1.0] * len(arms)
    selections = []
    trial_rewards = []
    counts = [0] * len(arms)
    arm_rewards = [0.0] * len(arms)

    # Track alpha/beta snapshots for convergence check
    alpha_history = []
    beta_history = []

    for t in range(n_trials):
        idx = thompson_sampling(alphas, betas, rng)
        r = run_trial(arms[idx], rng)
        alphas, betas = update_thompson(alphas, betas, idx, r)
        counts[idx] += 1
        arm_rewards[idx] += r
        selections.append(idx)
        trial_rewards.append(r)
        alpha_history.append(list(alphas))
        beta_history.append(list(betas))

    best_arm_idx = int(max(range(len(arms)), key=lambda i: arm_rewards[i] / max(counts[i], 1)))
    result = ExperimentResult(
        selections=selections,
        rewards=trial_rewards,
        counts=counts,
        arm_rewards=arm_rewards,
        best_arm_idx=best_arm_idx,
        n_trials=n_trials,
    )
    return result, alphas, betas, alpha_history, beta_history


def plot_regret_comparison(ab_regret, eg_regret, ts_regret, n_trials, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    trials = range(1, n_trials + 1)

    # Pad A/B regret: it runs 3 phases sequentially, flat between phases
    ab_padded = ab_regret + [ab_regret[-1]] * (n_trials - len(ab_regret))

    ax.plot(trials, ab_padded[:n_trials], label="Sequential A/B", color="#e74c3c", linewidth=2)
    ax.plot(trials, eg_regret[:n_trials], label="Epsilon-Greedy (ε₀=1, decay=1/√t)", color="#f39c12", linewidth=2)
    ax.plot(trials, ts_regret[:n_trials], label="Thompson Sampling", color="#27ae60", linewidth=2)

    ax.set_xlabel("Trial (impression)", fontsize=12)
    ax.set_ylabel("Cumulative Expected Regret", fontsize=12)
    ax.set_title("Regret Over Time: A/B vs Multi-Armed Bandit Algorithms", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate final values
    for regret_list, color, label in [
        (ab_padded, "#e74c3c", "A/B"),
        (eg_regret, "#f39c12", "EG"),
        (ts_regret, "#27ae60", "TS"),
    ]:
        ax.annotate(
            f"{label}: {regret_list[n_trials-1]:.1f}",
            xy=(n_trials, regret_list[n_trials - 1]),
            xytext=(n_trials - 80, regret_list[n_trials - 1] + 2),
            fontsize=9,
            color=color,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Regret comparison plot saved to {output_path}")


def main():
    rng = np.random.default_rng(SEED)
    arms = make_arms(HEADLINES, IMAGES, CTAS, rng, correlated=True)

    print(f"\nMulti-Armed Bandit — {N_TRIALS} total impressions, {len(arms)} arms (all parallel)")

    # Epsilon-Greedy
    eg_result, eg_counts, eg_rewards = run_epsilon_greedy(arms, N_TRIALS, rng)
    eg_regret = cumulative_regret(eg_result.selections, arms)

    # Thompson Sampling
    ts_result, ts_alphas, ts_betas, alpha_hist, beta_hist = run_thompson(arms, N_TRIALS, rng)
    ts_regret = cumulative_regret(ts_result.selections, arms)

    # Convergence
    ts_conv = convergence_point(ts_result.selections, ts_alphas, ts_betas)
    eg_conv = convergence_point(eg_result.selections, [1.0 + eg_rewards[i] for i in range(len(arms))],
                                 [1.0 + (eg_counts[i] - eg_rewards[i]) for i in range(len(arms))])

    # Load A/B regret from file (run ab_testing.py first, or fall back to zeros)
    ab_regret = []
    try:
        with open("results/ab_regret.json") as f:
            ab_data = json.load(f)
            ab_regret = ab_data["regret"]
    except FileNotFoundError:
        print("  [!] results/ab_regret.json not found. Run ab_testing.py first for A/B baseline.")
        ab_regret = [0.0] * N_TRIALS

    # Plot
    os.makedirs("results", exist_ok=True)
    plot_regret_comparison(ab_regret, eg_regret, ts_regret, N_TRIALS, "results/regret_comparison.png")

    # Fake ExperimentResult for A/B display in table
    from src.simulator import ExperimentResult as ER
    ab_result = ER(
        selections=list(range(len(ab_regret))),
        rewards=[0] * len(ab_regret),
        counts=[0] * len(arms),
        arm_rewards=[0.0] * len(arms),
        best_arm_idx=0,
        n_trials=len(ab_regret),
    )
    # Patch the regret into a fake result just for table display
    # (real A/B regret came from the file)

    print(f"\n  Epsilon-Greedy best arm:  '{arms[eg_result.best_arm_idx].headline}' / "
          f"'{arms[eg_result.best_arm_idx].image}' / '{arms[eg_result.best_arm_idx].cta}'")
    print(f"  Thompson best arm:        '{arms[ts_result.best_arm_idx].headline}' / "
          f"'{arms[ts_result.best_arm_idx].image}' / '{arms[ts_result.best_arm_idx].cta}'")
    optimal = max(arms, key=lambda a: a.true_ctr)
    print(f"  Optimal arm (true):       '{optimal.headline}' / '{optimal.image}' / '{optimal.cta}' (CTR={optimal.true_ctr:.4f})")

    print(f"\n  Thompson Sampling converged at trial: {ts_conv if ts_conv else 'not reached'}")
    print(f"  Epsilon-Greedy converged at trial:    {eg_conv if eg_conv else 'not reached'}")

    print(f"\n  Regret summary:")
    print(f"    Sequential A/B:     {ab_regret[-1]:.2f}")
    print(f"    Epsilon-Greedy:     {eg_regret[-1]:.2f}")
    print(f"    Thompson Sampling:  {ts_regret[-1]:.2f}")
    print()


if __name__ == "__main__":
    main()
