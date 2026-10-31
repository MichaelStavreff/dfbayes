"""Bayesian linear contrasts of independent beta posteriors."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._core import (
    as_float_vector,
    check_prob_interval,
    is_na,
    rdiv,
)
from .results import BetaContrastOut


def _validate_beta_contrast(
    n1_vec,
    n2_vec,
    contrast_vec,
    a0_vec,
    b0_vec,
    prob_interval,
    samples,
):
    if is_na(samples):
        # R: if (NA < 10000) -> missing value where TRUE/FALSE needed
        raise ValueError("missing value where TRUE/FALSE needed")
    if float(samples) < 10000.0:
        raise ValueError(
            "For reliable results please use at least 10000 Monte Carlo samples"
        )

    n1 = as_float_vector(n1_vec, "n1_vec")
    n2 = as_float_vector(n2_vec, "n2_vec")
    contrast = as_float_vector(contrast_vec, "contrast_vec")
    a0 = as_float_vector(a0_vec, "a0_vec")
    b0 = as_float_vector(b0_vec, "b0_vec")

    if len(n2) != len(n1) or len(contrast) != len(n1):
        raise ValueError(
            "The vectors n1_vec, n2_vec, and contrast_vec must have the same length."
        )
    # R recycles a0_vec/b0_vec through ordinary vector arithmetic.  Support the
    # useful length-1 broadcast explicitly; other lengths are rejected rather
    # than silently broadcasting (documented in DECISION.md).
    if len(a0) == 1 and len(n1) != 1:
        a0 = np.repeat(a0, len(n1))
    elif len(a0) != len(n1):
        raise ValueError("a0_vec must have length 1 or the same length as n1_vec")
    if len(b0) == 1 and len(n1) != 1:
        b0 = np.repeat(b0, len(n1))
    elif len(b0) != len(n1):
        raise ValueError("b0_vec must have length 1 or the same length as n1_vec")

    for values, name in ((n1, "n1_vec"), (n2, "n2_vec")):
        if np.any(np.isnan(values)) or np.any(values < 0) or np.any(values != np.round(values)):
            raise ValueError(f"{name} values must be non-negative integers")

    if np.any(np.isnan(a0)) or np.any(a0 <= 0) or np.any(np.isinf(a0)):
        raise ValueError(
            "All values in both the a0_vec and b0_vec shape parameter vectors for the prior beta must be positive and finite."
        )
    if np.any(np.isnan(b0)) or np.any(b0 <= 0) or np.any(np.isinf(b0)):
        raise ValueError(
            "All values in both the a0_vec and b0_vec shape parameter vectors for the prior beta must be positive and finite."
        )

    if np.any(np.isnan(contrast)):
        raise ValueError("missing value where TRUE/FALSE needed")
    if float(np.sum(contrast[contrast > 0])) != 1.0:
        raise ValueError("The sum of the positive contrast coefficients must be 1")
    if float(np.sum(contrast[contrast < 0])) != -1.0:
        raise ValueError("The sum of the negative contrast coefficients must be -1")

    pi = check_prob_interval(
        prob_interval,
        allow_zero_one=True,
        message="The probability for the interval estimate must be between 0 and 1.",
    )

    a_vec = n1 + a0
    b_vec = n2 + b0
    phimeans = a_vec / (a_vec + b_vec)
    mean_delta = float(np.sum(contrast * phimeans))

    return n1, n2, contrast, a0, b0, pi, a_vec, b_vec, mean_delta


def _finish_beta_contrast(
    contrast: np.ndarray,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    a0: np.ndarray,
    b0: np.ndarray,
    pi: float,
    samples: int,
    mean_delta: float,
    delta: np.ndarray,
    delta0: np.ndarray,
) -> BetaContrastOut:
    """Deterministic downstream math, given frozen delta/delta0 draws."""
    delta = np.asarray(delta, dtype=float)
    delta0 = np.asarray(delta0, dtype=float)
    if len(delta) != samples or len(delta0) != samples:
        raise ValueError("frozen delta vectors must have length samples")

    delta_quantiles = np.percentile(delta, np.arange(101), method="linear")
    low_limit = float(np.percentile(delta, (1.0 - pi) / 2.0 * 100.0, method="linear"))
    up_limit = float(
        np.percentile(delta, (1.0 - (1.0 - pi) / 2.0) * 100.0, method="linear")
    )

    pr_h1 = float(np.sum(delta > 0)) / samples
    prior_pr_h1 = float(np.sum(delta0 > 0)) / samples

    if pr_h1 == 1.0 or prior_pr_h1 == 0.0:
        bf10 = float(samples)
    else:
        bf10 = rdiv(
            pr_h1 * (1.0 - prior_pr_h1),
            prior_pr_h1 * (1.0 - pr_h1),
        )

    return BetaContrastOut(
        mean=mean_delta,
        eti_lower=low_limit,
        eti_upper=up_limit,
        prob_positive_delta=pr_h1,
        prior_positive_delta=prior_pr_h1,
        bayes_factor=bf10,
        delta_quantiles=delta_quantiles,
        a_vec=a_vec,
        b_vec=b_vec,
        contrast_vec=contrast,
        prob_interval=pi,
        samples=samples,
        a0_vec=a0,
        b0_vec=b0,
    )


def dfba_beta_contrast(
    n1_vec,
    n2_vec,
    contrast_vec,
    a0_vec=None,
    b0_vec=None,
    prob_interval: float = 0.95,
    samples: int = 10000,
    rng: Any = None,
) -> BetaContrastOut:
    """Monte-Carlo Bayesian contrast among independent beta posteriors.

    ``rng`` is a numpy ``Generator``.  The sampling is vectorised and is *not*
    bit-compatible with R's Mersenne-Twister; the deterministic downstream
    maths is fixture-tested against frozen R draws (see ``DECISION.md``).
    """
    n1 = as_float_vector(n1_vec, "n1_vec")
    if a0_vec is None:
        a0_vec = np.ones(len(n1))
    if b0_vec is None:
        b0_vec = np.ones(len(n1))

    (
        n1, _n2, contrast, a0, b0, pi, a_vec, b_vec, mean_delta
    ) = _validate_beta_contrast(
        n1_vec, n2_vec, contrast_vec, a0_vec, b0_vec, prob_interval, samples
    )

    n_samples = int(samples)
    generator = np.random.default_rng() if rng is None else rng

    draws = generator.beta(a_vec, b_vec, size=(n_samples, len(a_vec)))
    delta = draws @ contrast
    draws0 = generator.beta(a0, b0, size=(n_samples, len(a0)))
    delta0 = draws0 @ contrast

    return _finish_beta_contrast(
        contrast=contrast,
        a_vec=a_vec,
        b_vec=b_vec,
        a0=a0,
        b0=b0,
        pi=pi,
        samples=n_samples,
        mean_delta=mean_delta,
        delta=delta,
        delta0=delta0,
    )
