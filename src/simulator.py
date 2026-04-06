"""
simulator.py — Experiment runners.

All functions are pure with respect to arm state — they never modify arms.
Randomness is passed in via rng so experiments are reproducible.

Two experiment modes:
  - run_ab_experiment: sequential A/B, one variable at a time
  - run_bandit_experiment: all arms in parallel, algorithm chooses each trial
"""

from typing import Callable, NamedTuple
import numpy as np

from src.arms import Creative


class ExperimentResult(NamedTuple):
    selections: list[int]       # arm index chosen at each trial
    rewards: list[int]          # 0 or 1 at each trial
    counts: list[int]           # impressions per arm at end
    arm_rewards: list[float]    # total clicks per arm at end
    best_arm_idx: int           # index of arm the experiment concluded was best
    n_trials: int


def run_trial(arm: Creative, rng: np.random.Generator) -> int:
    """Bernoulli draw from arm's true CTR. Returns 0 or 1."""
    return int(rng.random() < arm.true_ctr)


def run_ab_experiment(
    arms: list[Creative],
    n_trials_per_phase: int,
    rng: np.random.Generator,
) -> ExperimentResult:
    """
    Sequential A/B experiment: test one creative dimension at a time.

    Phase 1 — Headline: fix image[0] and cta[0], test all headlines.
               Pick headline with highest observed CTR.
    Phase 2 — Image: fix winning headline and cta[0], test all images.
               Pick image with highest observed CTR.
    Phase 3 — CTA: fix winning headline and image, test all CTAs.
               Pick CTA with highest observed CTR.

    This is the status quo approach: 3 sequential experiments to find the
    best creative across a 3×3×2 space. Each phase is isolated — later
    phases can't recover from a bad pick in an earlier one.

    n_trials_per_phase: impressions allocated to each phase. Total trials =
    n_trials_per_phase × 3.
    """
    # Infer unique values from arms
    headlines = list(dict.fromkeys(a.headline for a in arms))
    images = list(dict.fromkeys(a.image for a in arms))
    ctas = list(dict.fromkeys(a.cta for a in arms))

    all_selections: list[int] = []
    all_rewards: list[int] = []
    counts = [0] * len(arms)
    arm_rewards = [0.0] * len(arms)

    def arm_idx(h: str, img: str, cta: str) -> int:
        return next(i for i, a in enumerate(arms) if a.headline == h and a.image == img and a.cta == cta)

    def run_phase(candidates: list[int]) -> int:
        """Run n_trials_per_phase evenly across candidates. Return winner index."""
        trials_each = n_trials_per_phase // len(candidates)
        phase_clicks = {i: 0 for i in candidates}
        for idx in candidates:
            for _ in range(trials_each):
                r = run_trial(arms[idx], rng)
                all_selections.append(idx)
                all_rewards.append(r)
                counts[idx] += 1
                arm_rewards[idx] += r
                phase_clicks[idx] += r
        best = max(candidates, key=lambda i: phase_clicks[i] / max(counts[i], 1))
        return best

    # Phase 1: vary headline, fix image[0] and cta[0]
    phase1_candidates = [arm_idx(h, images[0], ctas[0]) for h in headlines]
    winner1 = run_phase(phase1_candidates)
    winning_headline = arms[winner1].headline

    # Phase 2: vary image, fix winning headline and cta[0]
    phase2_candidates = [arm_idx(winning_headline, img, ctas[0]) for img in images]
    winner2 = run_phase(phase2_candidates)
    winning_image = arms[winner2].image

    # Phase 3: vary CTA, fix winning headline and image
    phase3_candidates = [arm_idx(winning_headline, winning_image, cta) for cta in ctas]
    winner3 = run_phase(phase3_candidates)

    return ExperimentResult(
        selections=all_selections,
        rewards=all_rewards,
        counts=counts,
        arm_rewards=arm_rewards,
        best_arm_idx=winner3,
        n_trials=len(all_selections),
    )


def run_bandit_experiment(
    arms: list[Creative],
    algorithm_fn: Callable[[int], int],
    n_trials: int,
    reward_callback: Callable[[int, int], None],
    rng: np.random.Generator,
) -> ExperimentResult:
    """
    Generic MAB experiment runner.

    algorithm_fn(trial_number) -> arm_index: called each trial to select an arm.
    reward_callback(arm_index, reward): called after each trial so the caller
      can update algorithm state (counts, alphas, betas, etc.).

    The runner itself holds no algorithm state — all state management lives
    in the caller (see runners/bandit.py).
    """
    selections: list[int] = []
    rewards: list[int] = []
    counts = [0] * len(arms)
    arm_rewards = [0.0] * len(arms)

    for t in range(n_trials):
        idx = algorithm_fn(t)
        r = run_trial(arms[idx], rng)
        reward_callback(idx, r)
        selections.append(idx)
        rewards.append(r)
        counts[idx] += 1
        arm_rewards[idx] += r

    best_arm_idx = int(max(range(len(arms)), key=lambda i: arm_rewards[i] / max(counts[i], 1)))

    return ExperimentResult(
        selections=selections,
        rewards=rewards,
        counts=counts,
        arm_rewards=arm_rewards,
        best_arm_idx=best_arm_idx,
        n_trials=n_trials,
    )
