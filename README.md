# From A/B Testing to Multi-Armed Bandits: An Ad Creative Case Study

A case study in adaptive budget allocation for advertising creatives. This
repository demonstrates — with runnable code — why sequential A/B testing
breaks down when you have multiple creative variables to optimise, and how
multi-armed bandit (MAB) algorithms recover lost value.

The code was originally implemented in a functional language. This Python
reference implementation preserves that style: pure functions, immutable data,
no shared mutable state.

---

## The Problem

You have three creative dimensions to test:

| Variable  | Options                                             |
|-----------|-----------------------------------------------------|
| Headline  | "Bold claim", "Question hook", "Social proof"       |
| Image     | Lifestyle photo, Product close-up, Illustration     |
| CTA       | "Shop now", "Learn more"                            |

That gives **18 possible creatives** (`3 × 3 × 2`). The question is: which
combination drives the highest click-through rate?

---

## Why A/B Testing Fails in This Setup

### 1. Combinatorial explosion

Exhaustive pairwise A/B testing of 18 creatives requires:

```
C(18, 2) = 153 separate tests
```

At typical campaign traffic, each test needs hundreds to thousands of
impressions to reach significance. That's months of sequential testing for a
single campaign.

The standard workaround — test one variable at a time — reduces this to three
phases, but it introduces a worse problem.

### 2. Sequential phases lock in early mistakes

Phase 1 tests headlines (with image and CTA fixed at defaults).
Phase 2 tests images (with the *winning* headline from Phase 1).
Phase 3 tests CTAs (with the *winning* headline and image).

If a strong image could only shine with a specific headline that lost in Phase 1,
it never gets a fair trial. **The search space is conditionally pruned at each
phase.** You find the best headline in isolation, not the best creative overall.

In this simulation, A/B picked:

```
'Social proof' / 'Product close-up' / 'Learn more'   CTR: 0.1308
```

The globally optimal arm was:

```
'Question hook' / 'Product close-up' / 'Learn more'  CTR: 0.3037
```

"Question hook" lost the headline phase — likely because Product close-up
(its strongest image partner) wasn't paired with it during Phase 1. A/B
never recovered.

### 3. Wasted impressions on clear losers

A/B has no in-flight reallocation mechanism. Once a phase is running, budget
flows equally to all candidates — including ones that have already revealed
themselves as underperformers. This is pure wasted spend.

---

## The Solution: Multi-Armed Bandits

A MAB algorithm treats all 18 creatives as *arms* of a slot machine. At each
impression, it picks an arm to serve. Arms that perform well get more
impressions. Arms that underperform get fewer — but not zero, because the
algorithm keeps exploring in case it made an early mistake.

This turns the three-phase sequential problem into **one parallel experiment**.
All 18 arms compete from trial 1.

---

## How It Works

### Epsilon-Greedy

At each trial, flip a biased coin:

- With probability **ε**: serve a random arm (exploration)
- With probability **1 − ε**: serve the arm with the best observed CTR so far (exploitation)

The exploration rate decays over time:

```
ε(t) = ε₀ / √(1 + t)
```

Sub-linear decay (√t) keeps exploration alive longer than 1/t, which would
collapse exploration too aggressively in early trials. `ε₀ = 1.0` by default.

**For engineers:** see `src/algorithms.py:epsilon_greedy()`

### Thompson Sampling

Each arm maintains a probability distribution over its unknown true CTR — a
`Beta(α, β)` distribution where `α = clicks + 1` and `β = non-clicks + 1`.

At each trial:
1. Sample one value from each arm's Beta distribution
2. Serve the arm with the highest sample

Arms with little data have wide, uncertain distributions — so they get sampled
high often (exploration). Arms with lots of data have narrow distributions
centred on their true CTR (exploitation). No tunable ε. The balance emerges
from the data.

**For engineers:** see `src/algorithms.py:thompson_sampling()`

---

## Regret Curve Comparison

Cumulative regret measures the cost of not always serving the optimal arm:

```
R(T) = T · μ* − Σ μ(aₜ)
```

where `μ*` is the true CTR of the best arm and `μ(aₜ)` is the true CTR of
whatever arm was served at trial `t`. This is deterministic given the
algorithm's choices — it measures opportunity cost against a perfect oracle,
independent of click noise.

![Regret comparison](results/regret_comparison.png)

| Method             | Impressions | Final Regret | Notes                              |
|--------------------|-------------|--------------|------------------------------------|
| Sequential A/B     | 1500        | 184.46       | Locked in suboptimal headline early |
| Epsilon-Greedy     | 1500        | 40.86        | ~4.5× lower regret than A/B        |
| Thompson Sampling  | 1500        | 112.07       | See cold-start caveat below        |

---

## Running the Examples

```bash
# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy

# Run sequential A/B baseline
python -m runners.ab_testing

# Run MAB comparison and generate regret plot
python -m runners.bandit
```

Output: `results/regret_comparison.png`

All experiments use `seed=42` for reproducibility.

---

## Challenges and Caveats

### Cold-start and thin data per arm

This is the central challenge of applying MAB to advertising.

With 18 arms and 1500 total impressions, each arm receives on average 83
impressions — and that's assuming perfectly uniform exploration, which no
algorithm does. In practice, arms that look bad early get fewer impressions,
which means their CTR estimates stay noisy.

**Notice what happened in this simulation:** Thompson Sampling performed *worse*
than epsilon-greedy (regret 112 vs 41). This is not a bug — it is the
cold-start problem made visible.

Thompson starts with flat `Beta(1, 1)` priors on all 18 arms. With uniform
uncertainty across 18 arms, early sampling is nearly random. The algorithm
needs enough impressions to build informative posteriors before its
exploration/exploitation balance becomes effective. With only ~83 impressions
per arm on average, posteriors remain wide, and Thompson keeps sampling
arms that have already revealed themselves as poor performers.

Epsilon-greedy, by contrast, forces early exploration (visiting each arm at
least once), then decays toward exploitation. With only 1500 trials and 18
arms, that forced structure outperforms Thompson's purely posterior-driven
approach.

**Practical implication:** if your daily impressions are low relative to the
number of arms, Thompson Sampling may underperform simpler algorithms for
a long time. A minimum burn-in floor (e.g. 10 impressions per arm before
acting on results) is advisable before trusting any allocation decisions.

**Rule of thumb:** for Thompson Sampling to work well, budget for at least
`n_arms × 30` impressions before drawing conclusions. For 18 arms: ~540
impressions just to build basic posteriors.

### Convergence is not guaranteed to be fast

This simulation ran 1500 trials and neither MAB algorithm formally converged.

Convergence here is defined strictly: the same arm must be selected for 50
consecutive trials *and* its lower 95% credible interval must exceed the
upper 95% CI of every other arm. Both conditions are required — stability
alone is not enough.

Real campaigns often have insufficient budget to reach formal convergence. The
algorithm is still outperforming A/B in terms of regret even without converging
— it just means you should not interpret "the algorithm keeps picking arm X"
as confirmation that arm X is the global optimum.

**For engineers:** see `src/metrics.py:convergence_point()`

### Non-stationarity: creative fatigue and seasonality

Both algorithms assume reward distributions are stationary — that the CTR of
each creative is fixed. This is false in advertising.

Creative fatigue causes CTRs to decline over time. Seasonality causes CTRs
to shift based on context (day of week, holiday periods, news cycle). A MAB
algorithm that learned the best creative in week 1 may be serving a fatigued
ad in week 4 with no mechanism to notice.

Mitigations:
- **Sliding window Thompson:** only count clicks/non-clicks from the last N impressions per arm
- **Discounted Thompson:** multiply historical `α` and `β` by a decay factor `γ < 1` each day

Neither mitigation is implemented in this codebase — they are noted as natural
extensions.

### Structural correlation across arms

Creative dimensions are not independent. A strong headline tends to lift
*all* creatives that use it, not just one. The current implementation has an
optional `correlated=True` mode in `make_arms()` that seeds CTRs with
per-component offsets, modelling this real-world structure.

A production MAB system might exploit this structure explicitly — using
hierarchical priors that share information across arms that share a component.
This is sometimes called "contextual bandits" when the shared structure is
formalized. It is out of scope here but is the natural next step for
multi-dimensional creative optimisation.

### A/B testing has a genuine advantage: auditability

MAB is not strictly better than A/B in all dimensions.

A fixed-horizon A/B test with a pre-registered hypothesis is easier to audit,
explain to stakeholders, and reason about under regulatory scrutiny. The
statistical guarantee (type I error rate, power) is clearly defined before the
experiment starts.

MAB algorithms require ongoing monitoring and trust in the algorithm's
allocation decisions. Explaining to a stakeholder why 30% of impressions went
to "Shop now" and 70% to "Learn more" requires understanding the posterior
distributions at that moment — not a fixed significance threshold. For
organisations new to adaptive experimentation, this operational overhead is real.

---

## Code Structure

```
src/
  arms.py         Creative arm definitions — immutable NamedTuples
  algorithms.py   Pure selection functions: epsilon_greedy(), thompson_sampling()
  simulator.py    Experiment runners — no algorithm state
  metrics.py      cumulative_regret(), convergence_point(), comparison_table()

runners/
  ab_testing.py   Sequential A/B baseline
  bandit.py       MAB comparison + regret plot

results/
  regret_comparison.png   Generated by runners/bandit.py
```

---

## Dependencies

- Python 3.10+
- `numpy` — random draws, Beta sampling
- `matplotlib` — regret curve plot
- `scipy` — Beta distribution quantiles for convergence check
