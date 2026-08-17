"""Approximate prospective power calculations for two-group experiments."""
from __future__ import annotations

from math import ceil

from scipy.stats import norm


def two_group_sample_size(*, effect_size: float, alpha: float = 0.05, power: float = 0.8, two_sided: bool = True) -> dict[str, float | int]:
    """Estimate observations per group for a standardized mean difference.

    This is a normal-approximation planning calculation, not a substitute for a
    design-specific power analysis when variance, clustering, blocking, or
    non-Gaussian outcomes materially affect the experiment.
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if two_sided:
        z_alpha = norm.ppf(1 - alpha / 2)
    else:
        z_alpha = norm.ppf(1 - alpha)
    z_power = norm.ppf(power)
    n = ceil(2 * ((z_alpha + z_power) / effect_size) ** 2)
    return {"n_per_group": int(n), "effect_size": float(effect_size), "alpha": float(alpha), "power": float(power), "two_sided": bool(two_sided)}


__all__ = ["two_group_sample_size"]
