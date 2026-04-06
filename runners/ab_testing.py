"""
runners/ab_testing.py — Sequential A/B experiment runner.

Demonstrates the status quo: test one creative variable at a time.
With 3 headlines × 3 images × 2 CTAs = 18 arms, this takes 3 sequential
phases to find the best combination — and can only ever find the locally
optimal pick within each phase, not the globally optimal arm.

Run:
    python -m runners.ab_testing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.arms import make_arms
from src.simulator import run_ab_experiment
from src.metrics import cumulative_regret, comparison_table

SEED = 42
N_TRIALS_PER_PHASE = 500  # impressions allocated per A/B phase (1500 total)

HEADLINES = ["Bold claim", "Question hook", "Social proof"]
IMAGES = ["Lifestyle photo", "Product close-up", "Illustration"]
CTAS = ["Shop now", "Learn more"]


def main():
    rng = np.random.default_rng(SEED)
    arms = make_arms(HEADLINES, IMAGES, CTAS, rng, correlated=True)

    print(f"\nA/B Testing — Sequential phases ({N_TRIALS_PER_PHASE} impressions/phase)")
    print(f"  {len(HEADLINES)} headlines × {len(IMAGES)} images × {len(CTAS)} CTAs = {len(arms)} arms")
    print(f"  Pairwise comparisons needed for exhaustive A/B: C({len(arms)},2) = {len(arms)*(len(arms)-1)//2}")
    print(f"  This runner tests {len(HEADLINES) + len(IMAGES) + len(CTAS)} arms across 3 phases instead.\n")

    result = run_ab_experiment(arms, N_TRIALS_PER_PHASE, rng)

    regret = cumulative_regret(result.selections, arms)
    best = arms[result.best_arm_idx]
    optimal = max(arms, key=lambda a: a.true_ctr)

    print(f"  A/B concluded best arm:   '{best.headline}' / '{best.image}' / '{best.cta}'")
    print(f"  True CTR of chosen arm:    {best.true_ctr:.4f}")
    print(f"  True CTR of optimal arm:   {optimal.true_ctr:.4f}")
    print(f"  Cumulative regret (final): {regret[-1]:.2f}")
    print(f"  Total impressions used:    {result.n_trials}")

    if best.headline != optimal.headline or best.image != optimal.image or best.cta != optimal.cta:
        print(f"\n  [!] A/B did NOT find the globally optimal arm.")
        print(f"      Optimal: '{optimal.headline}' / '{optimal.image}' / '{optimal.cta}'")
        print(f"      This is expected — sequential phases lock in suboptimal choices early.")

    # Save regret curve data for the bandit runner to include in combined plot
    import json
    os.makedirs("results", exist_ok=True)
    with open("results/ab_regret.json", "w") as f:
        json.dump({"regret": regret, "n_trials": result.n_trials}, f)

    print("\n  Regret data saved to results/ab_regret.json")
    print("  Run python -m runners.bandit to generate the full comparison plot.\n")


if __name__ == "__main__":
    main()
