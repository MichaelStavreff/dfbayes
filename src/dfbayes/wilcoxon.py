"""Bayesian Wilcoxon signed-rank analysis."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy.stats import rankdata

from ._core import (
    as_float_vector,
    check_positive_finite,
    check_prob_interval,
    is_na,
    pbeta,
    qbeta,
    qbeta_vec,
    rdiv,
)
from .mann_whitney import _OMEGA, _prior_vector
from .results import WilcoxonLargeOut, WilcoxonSmallOut


def _wilcoxon_small_from_counts(
    tpos: float,
    tneg: float,
    n: int,
    a0: float,
    b0: float,
    pi: float,
    samples: int,
    fphi: np.ndarray,
) -> WilcoxonSmallOut:
    """Deterministic small-sample maths, given the 200-bin likelihood counts."""
    prior = _prior_vector(a0, b0)
    fphi = np.asarray(fphi, dtype=float)
    if fphi.shape != (200,):
        raise ValueError("fphi must have length 200")

    weighted = prior * fphi
    total = float(np.sum(weighted))
    with np.errstate(divide="ignore", invalid="ignore"):
        phipost = weighted / total

    phiv = _OMEGA.copy()
    phibar = float(np.sum(phiv * phipost))
    cumulative = np.cumsum(phipost)

    target = (1.0 - pi) / 2.0
    idx = 0
    while idx < 200 and cumulative[idx] < target:
        idx += 1
    if idx >= 200:
        raise ValueError("missing value where TRUE/FALSE needed")
    ql_below = phiv[idx] - 0.0025
    if idx != 0:
        extrap = target - cumulative[idx - 1]
        prob_i = cumulative[idx] - cumulative[idx - 1]
    else:
        extrap = target
        prob_i = cumulative[0]
    hdi_lower = ql_below + 0.005 * rdiv(extrap, prob_i)

    target_u = 1.0 - ((1.0 - pi) / 2.0)
    idx = 0
    while idx < 200 and cumulative[idx] < target_u:
        idx += 1
    if idx >= 200:
        raise ValueError("missing value where TRUE/FALSE needed")
    qh_below = phiv[idx] - 0.0025
    if idx == 0:
        hdi_upper = float("nan")
    else:
        extrapup = target_u - cumulative[idx - 1]
        prob_iu = cumulative[idx] - cumulative[idx - 1]
        hdi_upper = qh_below + 0.005 * rdiv(extrapup, prob_iu)

    pr_h1 = 1.0 - cumulative[99]
    priorpr_h1 = 1.0 - float(np.cumsum(prior)[99])

    if pr_h1 == 1.0 or priorpr_h1 == 0.0:
        bf10 = float(samples)
    else:
        bf10 = rdiv(pr_h1 * (1.0 - priorpr_h1), priorpr_h1 * (1.0 - pr_h1))

    return WilcoxonSmallOut(
        T_pos=tpos,
        T_neg=tneg,
        n=n,
        prob_interval=pi,
        samples=samples,
        method="small",
        a0=a0,
        b0=b0,
        priorvector=prior,
        priorprH1=priorpr_h1,
        phiv=phiv,
        phipost=phipost,
        prH1=pr_h1,
        BF10=bf10,
        phibar=phibar,
        hdi_lower=hdi_lower,
        hdi_upper=hdi_upper,
        cumulative_phi=cumulative,
    )


def _fphi_counts(tpos: float, n: int, phi: float, samples: int, rng: Any) -> int:
    """Monte-Carlo count of ``T_pos == tpos`` at one phi value.

    This is the vectorised replacement for R's per-replication ``rbinom``
    loop.  Keeping it separate from the 200-bin loop lets tests probe an exact
    phi (notably 0.5, where the sign distribution is a Poisson binomial).
    """
    chunk = max(1, min(2000, int(2e7 // max(1, n))))
    ranks = np.arange(1, n + 1, dtype=float)
    matches = 0
    remaining = samples
    while remaining > 0:
        m = min(chunk, remaining)
        draws = rng.binomial(1, phi, size=(m, n))
        tz = (draws * ranks).sum(axis=1)
        matches += int(np.sum(tz == tpos))
        remaining -= m
    return matches


def _sample_fphi(tpos: float, n: int, samples: int, rng: Any) -> np.ndarray:
    """Vectorised replacement for R's 200 x samples rbinom() loops."""
    return np.array(
        [
            _fphi_counts(tpos, n, float(phi), samples, rng)
            for phi in _OMEGA
        ],
        dtype=float,
    )


def _wilcoxon_large(
    tpos: float,
    tneg: float,
    n: int,
    a0: float,
    b0: float,
    pi: float,
    samples: int,
) -> WilcoxonLargeOut:
    na0 = a0 - 1.0
    nb0 = b0 - 1.0
    term = (3.0 * tpos) / ((2.0 * n) + 2.0)
    na = term - 0.25
    nb = (((3.0 * n) - 1.0) / 4.0) - term
    a_post = na + na0 + 1.0
    b_post = nb + nb0 + 1.0

    post_mean = rdiv(a_post, a_post + b_post)
    post_median = qbeta(0.5, a_post, b_post)

    eti_lower = qbeta((1.0 - pi) * 0.5, a_post, b_post)
    eti_upper = qbeta(1.0 - (1.0 - pi) * 0.5, a_post, b_post)

    alpha_l = np.linspace(0.0, 1.0 - pi, 1001)
    ql = qbeta_vec(alpha_l, a_post, b_post)
    qh = qbeta_vec(pi + alpha_l, a_post, b_post)
    diff = qh - ql
    index = int(np.argmin(diff))
    hdi_lower = float(ql[index])
    hdi_upper = float(qh[index])

    pr_h1 = 1.0 - pbeta(0.5, a_post, b_post)
    priorpr_h1 = 1.0 - pbeta(0.5, na0 + 1.0, nb0 + 1.0)

    if pr_h1 == 1.0 or priorpr_h1 == 0.0:
        bf10 = math.inf
    else:
        bf10 = rdiv(pr_h1 * (1.0 - priorpr_h1), priorpr_h1 * (1.0 - pr_h1))

    return WilcoxonLargeOut(
        T_pos=tpos,
        T_neg=tneg,
        n=n,
        prob_interval=pi,
        samples=samples,
        method="large",
        a0=a0,
        b0=b0,
        a_post=a_post,
        b_post=b_post,
        post_mean=post_mean,
        post_median=post_median,
        priorprH1=priorpr_h1,
        prH1=pr_h1,
        BF10=bf10,
        eti_lower=eti_lower,
        eti_upper=eti_upper,
        hdi_lower=hdi_lower,
        hdi_upper=hdi_upper,
    )


def dfba_wilcoxon(
    Y1,
    Y2,
    a0: float = 1.0,
    b0: float = 1.0,
    prob_interval: float = 0.95,
    samples: int = 30000,
    method: str | None = None,
    hide_progress: bool = False,
    rng: Any = None,
) -> WilcoxonSmallOut | WilcoxonLargeOut:
    """Bayesian Wilcoxon signed-rank analysis for paired measurements.

    Pairs whose difference is NA/NaN are dropped.  Differences within
    ``sd(d)/30000`` of zero are treated as ties and removed.  Tied absolute
    differences receive average ranks.
    """
    y1_all = as_float_vector(Y1, "Y1")
    y2_all = as_float_vector(Y2, "Y2")
    if len(y1_all) != len(y2_all):
        raise ValueError(
            "Y1 and Y2 must have the same length. This function is for paired within-block data."
        )

    a0v, b0v = check_positive_finite(
        a0, b0, "Both a0 and b0 must be positive and finite"
    )
    pi = check_prob_interval(
        prob_interval,
        allow_zero_one=True,
        message="The probability for the interval estimate of phi_w must be a proper proportion.",
    )
    if is_na(samples):
        raise ValueError("missing value where TRUE/FALSE needed")
    if float(samples) < 10000.0:
        warnings.warn(
            "For reliable results, the recommended minimum number of Monte Carlo samples is 10000",
            UserWarning,
            stacklevel=2,
        )
    n_samples = int(samples)

    d_all = y1_all - y2_all
    mask = ~np.isnan(d_all)
    y1 = y1_all[mask]
    y2 = y2_all[mask]
    d = d_all[mask]
    l1 = len(d)

    if l1 < 3:
        raise ValueError(
            "There are not enough values in the Y1 and Y2 vectors for meaningful results."
        )

    sdd = float(np.std(d, ddof=1))
    threshold = sdd / 30000.0
    nonzero = np.abs(d) > threshold
    n = int(np.sum(nonzero))
    if n == 0:
        raise ValueError("Y1 and Y2 differences are all trivial")

    dt = d[nonzero]
    dta = np.abs(dt)
    dtar = rankdata(dta, method="average")
    dtars = dtar * dt / dta
    tpos = float(round(float(np.sum(dtar[dtars > 0]))))
    tneg = n * (n + 1) * 0.5 - tpos

    if method is None:
        method = "large" if n > 24 else "small"
    if method != "large" and method != "small":
        raise ValueError(
            "An explicit method stipulation must be either the word large or the word small."
        )

    if method == "small":
        generator = np.random.default_rng() if rng is None else rng
        fphi = _sample_fphi(tpos, n, n_samples, generator)
        return _wilcoxon_small_from_counts(
            tpos, tneg, n, a0v, b0v, pi, n_samples, fphi
        )

    return _wilcoxon_large(tpos, tneg, n, a0v, b0v, pi, n_samples)
