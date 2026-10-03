"""Approximate prospective power calculations for two-group experiments."""
from __future__ import annotations

from math import ceil, isfinite

from scipy.stats import norm


def two_group_sample_size(*, effect_size: float, alpha: float = 0.05, power: float = 0.8, two_sided: bool = True) -> dict[str, float | int]:
    """Estimate observations per group for a standardized mean difference.

    This is a normal-approximation planning calculation, not a substitute for a
    design-specific power analysis when variance, clustering, blocking, or
    non-Gaussian outcomes materially affect the experiment.
    """
    effect = float(effect_size)
    if not isfinite(effect) or effect <= 0:
        raise ValueError("effect_size must be a finite positive value")
    alpha_value = float(alpha)
    power_value = float(power)
    if not isfinite(alpha_value) or not 0 < alpha_value < 1:
        raise ValueError("alpha must be finite and in (0, 1)")
    if not isfinite(power_value) or not 0 < power_value < 1:
        raise ValueError("power must be finite and in (0, 1)")
    if two_sided:
        z_alpha = norm.ppf(1 - alpha_value / 2)
    else:
        z_alpha = norm.ppf(1 - alpha_value)
    z_power = norm.ppf(power_value)
    n = ceil(2 * ((z_alpha + z_power) / effect) ** 2)
    return {"n_per_group": int(n), "effect_size": effect, "alpha": alpha_value, "power": power_value, "two_sided": bool(two_sided)}


__all__ = ["two_group_sample_size"]
