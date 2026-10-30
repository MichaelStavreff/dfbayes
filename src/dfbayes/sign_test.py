"""Bayesian sign test."""

from __future__ import annotations

import numpy as np

from ._core import (
    as_float_vector,
    check_positive_finite,
    check_prob_interval,
    pbeta,
)
from .beta import dfba_beta_bayes_factor, dfba_beta_descriptive
from .results import SignTestOut


def dfba_sign_test(
    Y1,
    Y2,
    a0: float = 1.0,
    b0: float = 1.0,
    prob_interval: float = 0.95,
) -> SignTestOut:
    """Bayesian sign test for paired continuous measurements.

    Pairs with NA/NaN in either vector are dropped listwise (R's
    ``complete.cases``).  Differences whose absolute value is at most
    ``sd(d) / 30000`` are treated as zero and removed from the sign counts.
    """
    y1_in = as_float_vector(Y1, "Y1")
    y2_in = as_float_vector(Y2, "Y2")
    if len(y1_in) != len(y2_in):
        raise ValueError(
            "Y1 and Y2 must have the same length. This function is for paired within-block data."
        )

    a0v, b0v = check_positive_finite(
        a0, b0, "Both a0 and b0 must be positive and finite."
    )
    pi = check_prob_interval(
        prob_interval,
        allow_zero_one=True,
        message="The probability for the interval estimation must be a proper proportion.",
    )

    # R: complete.cases(data.frame(Y1, Y2, Y1 - Y2)).  The difference column
    # matters for Inf - Inf, which is NaN and therefore also dropped.
    d_all = y1_in - y2_in
    mask = ~(np.isnan(y1_in) | np.isnan(y2_in) | np.isnan(d_all))
    y1 = y1_in[mask]
    y2 = y2_in[mask]
    d = d_all[mask]
    l1 = len(y1)

    if l1 < 3:
        raise ValueError(
            "There are not enough values in the Y1 and Y2 vectors for meaningful results."
        )

    sdd = float(np.std(d, ddof=1))
    threshold = sdd / 30000.0
    n = int(np.sum(np.abs(d) > threshold))

    if n == 0:
        raise ValueError("Y1 and Y2 differences are all trivial")

    n_pos = int(np.sum(d > threshold))
    n_neg = n - n_pos
    a_post = n_pos + a0v
    b_post = n_neg + b0v

    descriptive = dfba_beta_descriptive(a_post, b_post, prob_interval=pi)

    prior_h1 = 1.0 - pbeta(0.5, a0v, b0v)
    post_h1 = 1.0 - pbeta(0.5, a_post, b_post)

    # Upstream calls dfba_beta_bayes_factor() without forwarding a0/b0, so the
    # Bayes factor uses the default uniform prior even when the caller chose a
    # different prior.  Reproduce that exactly.
    bf = dfba_beta_bayes_factor(
        a_post, b_post, method="interval", H0=[0.0, 0.5]
    )

    return SignTestOut(
        Y1=y1,
        Y2=y2,
        a0=a0v,
        b0=b0v,
        prob_interval=pi,
        n_pos=n_pos,
        n_neg=n_neg,
        a_post=a_post,
        b_post=b_post,
        phimean=descriptive.x_mean,
        phimedian=descriptive.x_median,
        phimode=descriptive.x_mode,
        eti_lower=descriptive.eti_lower,
        eti_upper=descriptive.eti_upper,
        hdi_lower=descriptive.hdi_lower,
        hdi_upper=descriptive.hdi_upper,
        post_H1=post_h1,
        prior_H1=prior_h1,
        BF10=bf.BF10,
        BF01=bf.BF01,
    )
