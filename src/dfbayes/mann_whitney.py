"""Bayesian Mann-Whitney U analysis."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from ._core import (
    as_float_vector,
    check_positive_finite,
    check_prob_interval,
    is_na,
    pbeta,
    qbeta,
    qbeta_vec,
    r_pow,
    rdiv,
)
from .results import MannWhitneyLargeOut, MannWhitneySmallOut

_OMEGA = np.arange(200, dtype=float) * 0.005 + 0.0025  # 0.0025 .. 0.9975


def _clean_vectors(E, C):
    e_all = as_float_vector(E, "E")
    c_all = as_float_vector(C, "C")
    e = e_all[~np.isnan(e_all)]
    c = c_all[~np.isnan(c_all)]
    e_missing = len(e_all) - len(e)
    c_missing = len(c_all) - len(c)
    if e_missing > 0:
        warnings.warn(
            f"Note: {e_missing} cells with missing values in E vector omitted",
            UserWarning,
            stacklevel=3,
        )
    if c_missing > 0:
        warnings.warn(
            f"Note: {c_missing} cells with missing values in C vector omitted",
            UserWarning,
            stacklevel=3,
        )
    return e, c


def _u_statistics(e: np.ndarray, c: np.ndarray) -> tuple[int, int]:
    ue = int(np.sum(e[:, None] > c[None, :]))
    uc = int(np.sum(c[:, None] > e[None, :]))
    return ue, uc


def _prior_vector(a0: float, b0: float) -> np.ndarray:
    upper = _OMEGA + 0.0025
    prior = np.empty(200, dtype=float)
    prior[0] = pbeta(float(upper[0]), a0, b0)
    for i in range(1, 200):
        prior[i] = pbeta(float(upper[i]), a0, b0) - pbeta(float(upper[i - 1]), a0, b0)
    return prior


def _mann_whitney_small_from_counts(
    e: np.ndarray,
    c: np.ndarray,
    ue: int,
    uc: int,
    a0: float,
    b0: float,
    pi: float,
    samples: int,
    fomega: np.ndarray,
) -> MannWhitneySmallOut:
    """Deterministic small-sample maths, given the 200-bin likelihood counts."""
    prior = _prior_vector(a0, b0)
    fomega = np.asarray(fomega, dtype=float)
    if fomega.shape != (200,):
        raise ValueError("fomega must have length 200")

    weighted = prior * fomega
    total = float(np.sum(weighted))
    with np.errstate(divide="ignore", invalid="ignore"):
        omegapost = weighted / total

    phiv = _OMEGA.copy()
    omegabar = float(np.sum(phiv * omegapost))
    cumulative = np.cumsum(omegapost)

    # Lower equal-tail limit (R's 1-based while loop, I == 1 branch included).
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
    eti_lower = ql_below + 0.005 * rdiv(extrap, prob_i)

    # Upper limit. For any valid prob_interval the loop advances past index 0;
    # R's I == 1 case would produce numeric(0), which Python cannot represent.
    target_u = 1.0 - ((1.0 - pi) / 2.0)
    idx = 0
    while idx < 200 and cumulative[idx] < target_u:
        idx += 1
    if idx >= 200:
        raise ValueError("missing value where TRUE/FALSE needed")
    qh_below = phiv[idx] - 0.0025
    if idx == 0:
        eti_upper = float("nan")
    else:
        extrapup = target_u - cumulative[idx - 1]
        prob_iu = cumulative[idx] - cumulative[idx - 1]
        eti_upper = qh_below + 0.005 * rdiv(extrapup, prob_iu)

    pr_h1 = 1.0 - cumulative[99]  # R's cumulative_omega[round(100)]
    priorpr_h1 = 1.0 - float(np.cumsum(prior)[99])

    if pr_h1 == 1.0 or priorpr_h1 == 0.0:
        bf10 = float(samples)
    else:
        bf10 = rdiv(pr_h1 * (1.0 - priorpr_h1), priorpr_h1 * (1.0 - pr_h1))

    return MannWhitneySmallOut(
        Emean=float(np.mean(e)),
        Cmean=float(np.mean(c)),
        n_E=len(e),
        n_C=len(c),
        U_E=ue,
        U_C=uc,
        prob_interval=pi,
        samples=samples,
        method="small",
        omega_E=phiv,
        omegapost=omegapost,
        cumulative_omega=cumulative,
        priorvector=prior,
        priorprH1=priorpr_h1,
        prH1=pr_h1,
        BF10=bf10,
        omegabar=omegabar,
        eti_lower=eti_lower,
        eti_upper=eti_upper,
    )


def _fomega_counts(
    e: np.ndarray,
    c: np.ndarray,
    ue: int,
    uc: int,
    omega: float,
    samples: int,
    rng: Any,
) -> int:
    """Monte-Carlo count of ``U == ue`` at one omega value.

    This is the vectorised replacement for R's per-replication ``rexp`` loops.
    Keeping it separate from the 200-bin loop lets tests probe an exact omega
    (notably 0.5, where the null Mann-Whitney distribution is known).
    """
    n_e = len(e)
    n_c = len(c)
    n_tot = ue + uc
    branch_a = n_e * n_c > n_tot

    # Cap the comparison tensor at roughly 20M booleans per chunk.
    cell = max(1, n_e * n_c)
    chunk = max(1, min(2000, int(2e7 // cell)))

    rate_ratio = (1.0 - omega) / omega
    matches = 0
    remaining = samples
    while remaining > 0:
        m = min(chunk, remaining)
        xe = rng.exponential(scale=1.0 / rate_ratio, size=(m, n_e))
        xc = rng.exponential(scale=1.0, size=(m, n_c))
        comparisons = xe[:, :, None] > xc[:, None, :]
        if branch_a:
            flat = comparisons.reshape(m, -1)[:, :n_tot]
            uz = flat.sum(axis=1)
        else:
            uz = comparisons.sum(axis=(1, 2))
        matches += int(np.sum(uz == ue))
        remaining -= m
    return matches


def _sample_fomega(
    e: np.ndarray,
    c: np.ndarray,
    ue: int,
    uc: int,
    samples: int,
    rng: Any,
) -> np.ndarray:
    """Vectorised replacement for R's 200 x samples rexp() loops."""
    return np.array(
        [
            _fomega_counts(e, c, ue, uc, float(omega), samples, rng)
            for omega in _OMEGA
        ],
        dtype=float,
    )


def _mann_whitney_large(
    e: np.ndarray,
    c: np.ndarray,
    ue: int,
    uc: int,
    a0: float,
    b0: float,
    pi: float,
    samples: int,
) -> MannWhitneyLargeOut:
    n_e = len(e)
    n_c = len(c)
    xs = ue / (ue + uc) if (ue + uc) != 0 else float("nan")
    if math.isnan(xs):
        raise ValueError("missing value where TRUE/FALSE needed")
    x = xs if xs >= 0.5 else 1.0 - xs

    nh = (2.0 * n_e * n_c) / (n_e + n_c)
    nh_pow = nh ** 1.1489
    y5 = nh_pow / (0.4972 + nh_pow)
    w4 = 0.8 - (1.0 / (1.0 + (1.833 * nh)))
    w3 = 0.6 - (1.0 / (1.0 + (2.111 * nh)))
    w2 = 0.4 - (1.0 / (1.0 + (2.520 * nh)))
    w1 = 0.2 - (1.0 / (1.0 + (4.813 * nh)))
    y4 = (y5 * w4) + (1.0 - w4) * 0.5
    y3 = (y5 * w3) + (1.0 - w3) * 0.5
    y2 = (y5 * w2) + (1.0 - w2) * 0.5
    y1 = (y5 * w1) + (1.0 - w1) * 0.5
    yy = np.array([0.5, y1, y2, y3, y4, y5], dtype=float)

    x2 = r_pow(x, 2)
    x3 = r_pow(x, 3)
    x4 = r_pow(x, 4)
    x5 = r_pow(x, 5)
    la0 = 252 - (1627 * x) + ((12500 * x2) - (15875 * x3) + (10000 * x4) - (2500 * x5)) / 3
    la1 = -1050 + ((42775 * x) / 6) - (38075 * 0.5 * x2) + ((75125 * x3) - (48750 * x4) + (12500 * x5)) / 3
    la2 = 1800 - (12650 * x) + ((104800 * x2) - (142250 * x3) + (95000 * x4) - (25000 * x5)) / 3
    la3 = -1575 + (11350 * x) + ((-96575 * x2) + (134750 * x3) - (92500 * x4) + (25000 * x5)) / 3
    la4 = 700 + (14900 * x2) + (15000 * x4) - ((15425 * x) + (63875 * x3) + (12500 * x5)) / 3
    la5 = -126 + (1879 * 0.5 * x) + ((-16625 * 0.5 * x2) + (12125 * x3) - (8750 * x4) + (2500 * x5)) / 3
    la = np.array([la0, la1, la2, la3, la4, la5], dtype=float)

    ombar = float(np.sum(yy * la))
    absum = nh * (1.028 + (0.75 * x)) + 2
    a = ombar * absum
    b = (1.0 - ombar) * absum
    omegabar = ombar
    if xs < 0.5:
        a = (1.0 - ombar) * absum
        b = ombar * absum
        omegabar = 1.0 - ombar

    na = a - 1.0
    nb = b - 1.0
    a_post = a0 + na
    b_post = b0 + nb
    a = a_post
    b = b_post

    post_mean = rdiv(a, a + b)
    post_median = qbeta(0.5, a, b)
    eti_lower = qbeta((1.0 - pi) * 0.5, a, b)
    eti_upper = qbeta(1.0 - (1.0 - pi) * 0.5, a, b)

    alpha_l = np.linspace(0.0, 1.0 - pi, 1001)
    ql = qbeta_vec(alpha_l, a, b)
    qh = qbeta_vec(pi + alpha_l, a, b)
    diff = qh - ql
    index = int(np.argmin(diff))
    hdi_lower = float(ql[index])
    hdi_upper = float(qh[index])

    pr_h1 = 1.0 - pbeta(0.5, a, b)
    priorpr_h1 = 1.0 - pbeta(0.5, a0, b0)
    if pr_h1 == 1.0 or priorpr_h1 == 0.0:
        bf10 = math.inf
    else:
        bf10 = rdiv(pr_h1 * (1.0 - priorpr_h1), priorpr_h1 * (1.0 - pr_h1))

    return MannWhitneyLargeOut(
        Emean=float(np.mean(e)),
        Cmean=float(np.mean(c)),
        n_E=n_e,
        n_C=n_c,
        U_E=ue,
        U_C=uc,
        prob_interval=pi,
        a0=a0,
        b0=b0,
        a_post=a_post,
        b_post=b_post,
        post_mean=post_mean,
        post_median=post_median,
        method="large",
        omegabar=omegabar,
        eti_lower=eti_lower,
        eti_upper=eti_upper,
        hdi_lower=hdi_lower,
        hdi_upper=hdi_upper,
        priorprH1=priorpr_h1,
        prH1=pr_h1,
        BF10=bf10,
    )


def dfba_mann_whitney(
    E,
    C,
    a0: float = 1.0,
    b0: float = 1.0,
    prob_interval: float = 0.95,
    samples: int = 30000,
    method: str | None = None,
    hide_progress: bool = False,
    rng: Any = None,
) -> MannWhitneySmallOut | MannWhitneyLargeOut:
    """Bayesian Mann-Whitney U analysis.

    ``method="small"`` uses vectorised numpy Monte Carlo; ``method="large"`` is
    the closed-form beta approximation.  The small-sample deterministic maths
    is fixture-tested against frozen R ``fomega`` counts.
    """
    a0v, b0v = check_positive_finite(
        a0, b0, "Both a0 and b0 must be positive and finite."
    )
    pi = check_prob_interval(
        prob_interval,
        allow_zero_one=True,
        message="The probability for the interval estimate of phi_w must be a proper proportion.",
    )
    if is_na(samples):
        # R: if (NA < 10000) -> missing value where TRUE/FALSE needed
        raise ValueError("missing value where TRUE/FALSE needed")
    if float(samples) < 10000.0:
        warnings.warn(
            "For reliable results, the recommended minimum number of Monte Carlo samples is 10000",
            UserWarning,
            stacklevel=2,
        )
    n_samples = int(samples)

    e, c = _clean_vectors(E, C)
    if len(e) == 0 or len(c) == 0:
        raise ValueError("The E and C vectors must have a length greater than 0.")

    ue, uc = _u_statistics(e, c)
    nh = (2.0 * len(e) * len(c)) / (len(e) + len(c))

    if method is None:
        method = "large" if nh >= 20.0 else "small"
    if method != "large" and method != "small":
        raise ValueError(
            "An explicit method stipulation must be either the word large or small."
        )

    if method == "small":
        generator = np.random.default_rng() if rng is None else rng
        fomega = _sample_fomega(e, c, ue, uc, n_samples, generator)
        return _mann_whitney_small_from_counts(
            e, c, ue, uc, a0v, b0v, pi, n_samples, fomega
        )

    return _mann_whitney_large(e, c, ue, uc, a0v, b0v, pi, n_samples)
