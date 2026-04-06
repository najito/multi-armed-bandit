"""
algorithms.py — Pure selection functions for multi-armed bandit algorithms.

All functions are pure: they take state in and return an arm index out.
Callers own all mutable state (counts, rewards, alphas, betas).
No side effects. No shared mutable globals.

Functional style: each algorithm is a function, not a class.
"""

import math
import numpy as np


def epsilon_greedy(
    counts: list[int],
    rewards: list[float],
    rng: np.random.Generator,
    epsilon_0: float = 1.0,
) -> int:
    """
    Select an arm using epsilon-greedy with decaying exploration rate.

    Decay schedule: ε(t) = ε₀ / √(1 + t)
      where t = total trials so far (sum of counts).
      Sub-linear decay keeps exploration alive longer than 1/t, which
      collapses ε too aggressively early on.

    Arms with zero counts are always explored first (forced cold-start
    exploration), ensuring every arm gets at least one trial before
    exploitation begins.

    Returns the index of the selected arm.
    """
    n_arms = len(counts)
    t = sum(counts)

    # Force exploration of any unseen arm before any exploitation.
    unvisited = [i for i in range(n_arms) if counts[i] == 0]
    if unvisited:
        return int(rng.choice(unvisited))

    epsilon = epsilon_0 / math.sqrt(1 + t)

    if rng.random() < epsilon:
        return int(rng.integers(0, n_arms))

    ctrs = [rewards[i] / counts[i] for i in range(n_arms)]
    return int(np.argmax(ctrs))


def thompson_sampling(
    alphas: list[float],
    betas: list[float],
    rng: np.random.Generator,
) -> int:
    """
    Select an arm using Thompson Sampling over Beta posteriors.

    Each arm maintains Beta(α, β) where:
      α = clicks + 1   (successes, Laplace-smoothed)
      β = non-clicks + 1  (failures, Laplace-smoothed)

    At each trial: sample θₐ ~ Beta(αₐ, βₐ) for every arm, select argmax(θ).

    This is optimistic under uncertainty — with few impressions the Beta is
    wide and flat, so the algorithm explores naturally without any tunable ε.
    As data accumulates the posterior sharpens around the true CTR.

    No tunable parameters. The exploration/exploitation balance is entirely
    governed by posterior uncertainty.

    Returns the index of the selected arm.
    """
    samples = rng.beta(alphas, betas)
    return int(np.argmax(samples))


def update_thompson(
    alphas: list[float],
    betas: list[float],
    arm_idx: int,
    reward: int,
) -> tuple[list[float], list[float]]:
    """
    Return updated (alphas, betas) after observing reward ∈ {0, 1} for arm_idx.

    Pure function — returns new lists rather than mutating the inputs.
    """
    new_alphas = list(alphas)
    new_betas = list(betas)
    new_alphas[arm_idx] += reward
    new_betas[arm_idx] += 1 - reward
    return new_alphas, new_betas
