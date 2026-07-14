"""Statistical confidence helpers for sampled evaluation."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def wilson_score_interval(successes: int, total: int, *, z: float = 1.96) -> tuple[float, float]:
    """Compute a Wilson score confidence interval for a binomial proportion."""

    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denominator = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 13,
) -> tuple[float, float]:
    """Bootstrap a confidence interval around the sample mean."""

    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _item in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lower_index = int(((1 - confidence) / 2) * len(means))
    upper_index = min(len(means) - 1, int((1 - (1 - confidence) / 2) * len(means)))
    return means[lower_index], means[upper_index]


__all__ = ["bootstrap_mean_interval", "wilson_score_interval"]
