"""
metrics.py — Evaluation functions for experiment results.

All functions are pure: they take results and arms in, return numbers or
strings out. No plotting here — that lives in the runners.
"""

import math
from typing import Optional
from src.arms import Creative
from src.simulator import ExperimentResult


def cumulative_regret(
    selections: list[int],
    arms: list[Creative],
) -> list[float]:
    """
    Compute expected pseudo-regret at each trial.

    R(T) = T · μ* − Σ_{t=1}^{T} μ(aₜ)

    where μ* = max CTR across all arms (the optimal policy's reward rate),
    and μ(aₜ) = true CTR of the arm selected at trial t.

    This is deterministic given selections — it measures opportunity cost
    against the best possible policy, independent of realized noise.

    Returns a list of length len(selections) where entry t is R(t).
    """
    mu_star = max(a.true_ctr for a in arms)
    regret = []
    cumulative = 0.0
    for t, idx in enumerate(selections):
        cumulative += mu_star - arms[idx].true_ctr
        regret.append(cumulative)
    return regret


def convergence_point(
    selections: list[int],
    alphas: list[float],
    betas: list[float],
    k: int = 50,
) -> Optional[int]:
    """
    Return the first trial T* at which the experiment has converged.

    Convergence requires both conditions to hold simultaneously:
      1. The same arm has been selected for k consecutive trials.
      2. That arm's lower 95% credible interval exceeds the upper 95%
         credible interval of every other arm.

    Credible intervals use the Beta distribution's quantiles:
      lower = Beta(α, β).ppf(0.025)
      upper = Beta(α, β).ppf(0.975)

    Returns None if convergence was not reached.
    """
    from scipy.stats import beta as beta_dist

    n = len(selections)
    if n < k:
        return None

    for t in range(k - 1, n):
        # Condition 1: same arm selected for last k trials
        window = selections[t - k + 1 : t + 1]
        if len(set(window)) != 1:
            continue
        candidate = window[0]

        # Condition 2: posterior separation
        lower_candidate = beta_dist.ppf(0.025, alphas[candidate], betas[candidate])
        dominated = all(
            lower_candidate > beta_dist.ppf(0.975, alphas[i], betas[i])
            for i in range(len(alphas))
            if i != candidate
        )
        if dominated:
            return t

    return None


def comparison_table(
    arms: list[Creative],
    ab_result: ExperimentResult,
    ab_label: str,
    eg_result: ExperimentResult,
    eg_convergence: Optional[int],
    ts_result: ExperimentResult,
    ts_convergence: Optional[int],
) -> str:
    """
    Return a formatted comparison table as a string.

    Columns: Method | Trials | Clicks | Final Regret | Converged At | Best Arm CTR
    """
    optimal_ctr = max(a.true_ctr for a in arms)

    def row(label, result, convergence):
        regret = cumulative_regret(result.selections, arms)
        final_regret = regret[-1] if regret else 0.0
        clicks = sum(result.rewards)
        best_ctr = arms[result.best_arm_idx].true_ctr
        conv = str(convergence) if convergence is not None else "not reached"
        return f"  {label:<22} {result.n_trials:>8}  {clicks:>8}  {final_regret:>14.2f}  {conv:>14}  {best_ctr:.4f}"

    header = f"  {'Method':<22} {'Trials':>8}  {'Clicks':>8}  {'Final Regret':>14}  {'Converged At':>14}  Best Arm CTR"
    sep = "  " + "-" * 85
    optimal_row = f"\n  Optimal arm true CTR: {optimal_ctr:.4f}"

    lines = [
        "",
        "  Experiment Comparison",
        sep,
        header,
        sep,
        row(ab_label, ab_result, None),
        row("Epsilon-Greedy", eg_result, eg_convergence),
        row("Thompson Sampling", ts_result, ts_convergence),
        sep,
        optimal_row,
        "",
    ]
    return "\n".join(lines)
