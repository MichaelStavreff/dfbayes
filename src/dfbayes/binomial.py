"""Bayesian binomial rate inference."""

from __future__ import annotations

from ._core import check_nonnegative_count, check_positive_finite, check_prob_interval
from .beta import dfba_beta_descriptive
from .results import BinomialOut


def dfba_binomial(
    n1: float,
    n2: float,
    a0: float = 1.0,
    b0: float = 1.0,
    prob_interval: float = 0.95,
) -> BinomialOut:
    """Bayesian inference for a binomial rate parameter.

    ``n1`` and ``n2`` must be non-negative integers; ``a0``/``b0`` must be
    positive and finite.  NA/NaN anywhere raises ``ValueError``, matching R.
    """
    a0v, b0v = check_positive_finite(
        a0, b0, "Both a0 and b0 must be positive and finite"
    )
    pi = check_prob_interval(
        prob_interval,
        allow_zero_one=False,
        message="prob_interval must be greater than 0 and less than 1",
    )
    n1v = check_nonnegative_count(
        n1, "n1", "Neither n1 nor n2 can be negative", "n1 and n2 must be integers"
    )
    n2v = check_nonnegative_count(
        n2, "n2", "Neither n1 nor n2 can be negative", "n1 and n2 must be integers"
    )

    a_post = n1v + a0v
    b_post = n2v + b0v

    descriptive = dfba_beta_descriptive(a_post, b_post, prob_interval=pi)

    return BinomialOut(
        n1=n1v,
        n2=n2v,
        a0=a0v,
        b0=b0v,
        prob_interval=pi,
        a_post=a_post,
        b_post=b_post,
        phimean=descriptive.x_mean,
        phimedian=descriptive.x_median,
        phimode=descriptive.x_mode,
        eti_lower=descriptive.eti_lower,
        eti_upper=descriptive.eti_upper,
        hdi_lower=descriptive.hdi_lower,
        hdi_upper=descriptive.hdi_upper,
    )
