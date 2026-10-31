"""Bayesian McNemar test for change."""

from __future__ import annotations

from typing import cast

from ._core import (
    check_nonnegative_count,
    check_positive_finite,
    check_prob_interval,
    qbeta,
)
from .beta import dfba_beta_bayes_factor
from .results import IntervalBFOut, McnemarOut


def dfba_mcnemar(
    n_01: float,
    n_10: float,
    a0: float = 1.0,
    b0: float = 1.0,
    prob_interval: float = 0.95,
) -> McnemarOut:
    """Bayesian repeated-measures McNemar test.

    Only the discordant counts ``n_01`` and ``n_10`` enter the analysis.
    """
    a0v, b0v = check_positive_finite(
        a0, b0, "Both a0 and b0 must be positive and finite."
    )
    pi = check_prob_interval(
        prob_interval,
        allow_zero_one=False,
        message="prob_interval must between 0 and 1",
    )
    n01 = check_nonnegative_count(
        n_01, "n_01",
        "Neither n_01 nor n_10 can be negative",
        "n_01 and n_10 must be integers",
    )
    n10 = check_nonnegative_count(
        n_10, "n_10",
        "Neither n_01 nor n_10 can be negative",
        "n_01 and n_10 must be integers",
    )

    a_post = a0v + n01
    b_post = b0v + n10

    post_mean = a_post / (a_post + b_post)
    post_median = qbeta(0.5, a_post, b_post)

    eti_lower = qbeta((1.0 - pi) / 2.0, a_post, b_post)
    eti_upper = qbeta(1.0 - ((1.0 - pi) / 2.0), a_post, b_post)

    bf_point = dfba_beta_bayes_factor(
        a_post=a_post, b_post=b_post, method="point", H0=0.5,
        a0=a0v, b0=b0v,
    )
    bf_interval = dfba_beta_bayes_factor(
        a_post=a_post, b_post=b_post, method="interval", H0=[0.0, 0.5],
        a0=a0v, b0=b0v,
    )

    return McnemarOut(
        n_10=n10,
        n_01=n01,
        prob_interval=pi,
        a0=a0v,
        b0=b0v,
        a_post=a_post,
        b_post=b_post,
        post_mean=post_mean,
        post_median=post_median,
        eti_lower=eti_lower,
        eti_upper=eti_upper,
        BF10point=bf_point.BF10,
        BF10interval=bf_interval.BF10,
        postH1=cast(IntervalBFOut, bf_interval).postH1,
    )
