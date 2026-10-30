"""Beta descriptive statistics and Bayes factors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._core import (
    as_scalar,
    check_positive_finite,
    check_prob_interval,
    dbeta,
    pbeta,
    qbeta,
    qbeta_vec,
    r_pow,
    rdiv,
)
from .results import (
    BetaDescriptiveOut,
    BetaOutputDF,
    IntervalBFOut,
    PointBFOut,
)


def dfba_beta_descriptive(
    a: float,
    b: float,
    prob_interval: float = 0.95,
) -> BetaDescriptiveOut:
    """Descriptive statistics for a beta distribution.

    Mirrors ``DFBA::dfba_beta_descriptive``.  An undefined mode or HDI is
    returned as ``None`` (R stores a logical ``NA`` there).
    """
    pi = check_prob_interval(
        prob_interval,
        allow_zero_one=True,
        message="The probability for the interval estimate must be between 0 and 1.",
    )
    av, bv = check_positive_finite(
        a, b,
        "Both the a and b shape parameters for a beta must be positive and finite",
    )

    phimean = av / (av + bv)
    phivariance = (av * bv) / (r_pow(av + bv, 2) * (av + bv + 1))
    phimedian = qbeta(0.5, av, bv)

    undefined_mode = (av == 1.0 and bv == 1.0) or av < 1.0 or bv < 1.0
    phimode = None if undefined_mode else (av - 1.0) / (av + bv - 2.0)

    qlequal = qbeta((1.0 - pi) / 2.0, av, bv)
    qhequal = qbeta(pi + ((1.0 - pi) / 2.0), av, bv)

    if undefined_mode:
        qlmin = None
        qhmax = None
    else:
        # R: seq(0, 1 - prob_interval, (1 - prob_interval)/1000)
        alpha_l = np.linspace(0.0, 1.0 - pi, 1001)
        ql = qbeta_vec(alpha_l, av, bv)
        qh = qbeta_vec(pi + alpha_l, av, bv)
        diff = qh - ql
        index = int(np.argmin(diff))  # first index attaining the minimum
        qlmin = float(ql[index])
        qhmax = float(qh[index])

    # R: seq(0, 1, .005) -> 201 points
    x = np.arange(201, dtype=float) * 0.005
    density = np.array([dbeta(float(xi), av, bv) for xi in x])
    cumulative = np.array(
        [pbeta(float(xi), av, bv) for xi in x]
    )
    outputdf = BetaOutputDF(x=x, density=density, cumulative_prob=cumulative)

    return BetaDescriptiveOut(
        a=av,
        b=bv,
        prob_interval=pi,
        x_mean=phimean,
        x_median=phimedian,
        x_mode=phimode,
        x_variance=phivariance,
        eti_lower=qlequal,
        eti_upper=qhequal,
        hdi_lower=qlmin,
        hdi_upper=qhmax,
        outputdf=outputdf,
    )


def dfba_beta_bayes_factor(
    a_post: float,
    b_post: float,
    method: str,
    H0: Any,
    a0: float = 1.0,
    b0: float = 1.0,
) -> PointBFOut | IntervalBFOut:
    """Bayes factor for a point or interval null hypothesis."""
    if method != "point" and method != "interval":
        raise ValueError("method must be either 'point' or 'interval'")

    a0v, b0v = check_positive_finite(
        a0, b0, "Both a0 and b0 must be positive and finite."
    )
    a_postv = as_scalar(a_post, "a_post")
    b_postv = as_scalar(b_post, "b_post")

    if (
        a_postv < a0v
        or b_postv < b0v
        or math.isnan(a_postv)
        or math.isnan(b_postv)
    ):
        raise ValueError(
            "Both a_post and b_post cannot be less than the respective a0 and b0 values"
        )

    if method == "point":
        # R checks length(H0), so a length-1 vector is accepted as well as a
        # scalar.
        if np.ndim(H0) == 1 and len(H0) == 1:
            H0 = H0[0]
        elif np.ndim(H0) != 0:
            raise ValueError("'H0' must be a single numeric value when method = 'point'")
        if H0 is None:
            raise ValueError("missing value where TRUE/FALSE needed")
        h0 = as_scalar(H0, "H0")
        if math.isnan(h0):
            raise ValueError("missing value where TRUE/FALSE needed")
        if h0 > 1.0 or h0 < 0.0:
            raise ValueError(
                "H0 must be greater than or equal to 0 and less than or equal to 1"
            )

        dprior = dbeta(h0, a0v, b0v)
        dpost = dbeta(h0, a_postv, b_postv)
        bf10 = math.inf if dpost == 0.0 else rdiv(dprior, dpost)
        bf01 = math.inf if dprior == 0.0 else rdiv(dpost, dprior)

        return PointBFOut(
            method=method,
            a_post=a_postv,
            b_post=b_postv,
            a0=a0v,
            b0=b0v,
            BF10=bf10,
            BF01=bf01,
            null_hypothesis=h0,
            dpriorH0=dprior,
            dpostH0=dpost,
        )

    # interval
    if np.ndim(H0) != 1 or len(H0) != 2:
        raise ValueError(
            "'H0' must be a vector of two values (lower and upper) when method = 'interval'"
        )
    if any(v is None for v in H0):
        raise ValueError("missing value where TRUE/FALSE needed")
    lower = as_scalar(H0[0], "H0[0]")
    upper = as_scalar(H0[1], "H0[1]")
    if math.isnan(lower) or math.isnan(upper):
        raise ValueError("missing value where TRUE/FALSE needed")
    if any(v > 1.0 or v < 0.0 for v in (lower, upper)):
        raise ValueError(
            "H0 interval limits must be greater than or equal to 0 and less than or equal to 1"
        )
    if lower >= upper:
        raise ValueError(
            "When method = 'interval', H0 upper limit must be greater than H0 lower limit"
        )

    ph0 = pbeta(upper, a0v, b0v) - pbeta(lower, a0v, b0v)
    ph1 = 1.0 - ph0
    post_h0 = pbeta(upper, a_postv, b_postv) - pbeta(lower, a_postv, b_postv)
    post_h1 = 1.0 - post_h0

    bf10 = (
        math.inf
        if post_h0 == 0.0
        else rdiv(rdiv(1.0, post_h0) - 1.0, rdiv(1.0, ph0) - 1.0)
    )
    bf01 = (
        math.inf
        if post_h1 == 0.0
        else rdiv(rdiv(1.0, ph0) - 1.0, rdiv(1.0, post_h0) - 1.0)
    )

    return IntervalBFOut(
        method=method,
        a_post=a_postv,
        b_post=b_postv,
        a0=a0v,
        b0=b0v,
        BF10=bf10,
        BF01=bf01,
        H0lower=lower,
        H0upper=upper,
        pH0=ph0,
        pH1=ph1,
        postH0=post_h0,
        postH1=post_h1,
    )
