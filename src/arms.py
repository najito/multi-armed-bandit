"""
arms.py — Creative arm definitions.

Each arm is an immutable combination of (headline, image, cta) with a hidden
true click-through rate (CTR). The true CTR is known only to the simulator;
algorithms never see it directly.

Functional style: no mutable state, no classes with methods. Arms are data.
"""

from typing import NamedTuple
import numpy as np


class Creative(NamedTuple):
    headline: str
    image: str
    cta: str
    true_ctr: float


def make_arms(
    headlines: list[str],
    images: list[str],
    ctas: list[str],
    rng: np.random.Generator,
    correlated: bool = False,
) -> list[Creative]:
    """
    Generate all headline × image × CTA combinations as Creative arms.

    true_ctr is drawn from Beta(2, 10) giving a realistic ~15% base rate.

    correlated=True: arms that share a creative component are seeded with
    correlated CTRs. A per-component baseline offset (±0.02) is added on top
    of the base draw. This models the real-world observation that a strong
    headline lifts all creatives that use it.
    """
    base_ctrs = rng.beta(2, 10, size=(len(headlines), len(images), len(ctas)))

    if correlated:
        headline_offsets = rng.uniform(-0.02, 0.02, size=len(headlines))
        image_offsets = rng.uniform(-0.02, 0.02, size=len(images))
        cta_offsets = rng.uniform(-0.02, 0.02, size=len(ctas))

    arms = []
    for hi, h in enumerate(headlines):
        for ii, img in enumerate(images):
            for ci, cta in enumerate(ctas):
                ctr = float(base_ctrs[hi, ii, ci])
                if correlated:
                    ctr = np.clip(
                        ctr + headline_offsets[hi] + image_offsets[ii] + cta_offsets[ci],
                        0.01,
                        0.99,
                    )
                arms.append(Creative(headline=h, image=img, cta=cta, true_ctr=float(ctr)))

    return arms
