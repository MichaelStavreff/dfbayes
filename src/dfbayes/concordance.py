"""Bivariate concordance (tau-A) and Goodman-Kruskal gamma."""

from __future__ import annotations

import math
import warnings
from collections import Counter

import numpy as np
from scipy.stats import rankdata

from ._core import (
    as_float_vector,
    check_positive_finite,
    is_na,
    qbeta,
    rdiv,
)
from .results import BivariateConcordanceOut, BivariateConcordanceStarOut, GammaOut


def _pairwise_complete(x, y) -> tuple[np.ndarray, np.ndarray, int]:
    x_arr = as_float_vector(x, "x")
    y_arr = as_float_vector(y, "y")

    # R's data.frame() recycles a length-1 argument and errors otherwise.
    if len(x_arr) != len(y_arr):
        if len(x_arr) == 1:
            x_arr = np.repeat(x_arr, len(y_arr))
        elif len(y_arr) == 1:
            y_arr = np.repeat(y_arr, len(x_arr))
        else:
            raise ValueError(
                f"arguments imply differing number of rows: {len(x_arr)}, {len(y_arr)}"
            )

    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    removed = int(len(x_arr) - np.sum(mask))
    if removed > 0:
        warnings.warn(
            f"{removed} row(s) removed due to missing data",
            UserWarning,
            stacklevel=3,
        )
    return x_arr[mask], y_arr[mask], removed


def _tie_loss(values: list) -> int:
    counts = Counter(values)
    return sum(t * (t - 1) // 2 for t in counts.values() if t > 1)


def _concordance_counts(x: np.ndarray, y: np.ndarray) -> tuple[int, float, float, int, int]:
    """Return (n_max, nc, nd, Tx, Ty) using R's tau-A tie correction."""
    n = len(x)
    tx = _tie_loss(x.tolist())
    ty = _tie_loss(y.tolist())
    txy = _tie_loss(list(zip(x.tolist(), y.tolist())))
    n_max = n * (n - 1) / 2.0 - tx - ty + txy

    if n == 0:
        return n_max, 0, 0.0, tx, ty

    xrank = rankdata(x, method="average")
    yrank = rankdata(y, method="average")
    # R: order(x1, -y1) -- ascending x, descending y
    order = np.lexsort((-y, x))
    yrank_sorted = yrank[order]

    nc = 0
    for i in range(n - 1):
        nc += int(np.sum(yrank_sorted[i + 1:] > yrank_sorted[i]))
    nd = n_max - nc
    return n_max, nc, float(nd), tx, ty


def dfba_bivariate_concordance(
    x,
    y,
    a0: float = 1.0,
    b0: float = 1.0,
    prob_interval: float = 0.95,
    fitting_parameters=None,
) -> BivariateConcordanceOut | BivariateConcordanceStarOut:
    """Distribution-free Bayesian correlation/concordance (tau-A).

    Rows with NA/NaN in either vector are removed pairwise with a warning.
    ``prob_interval`` is deliberately not validated here, matching upstream:
    an NA interval propagates into the interval fields (returned as ``None``).
    """
    x1, y1, _ = _pairwise_complete(x, y)

    a0v, b0v = check_positive_finite(
        a0, b0, "Both the a0 and b0 shape parameters must be positive and finite"
    )

    n_max, nc, nd, _, _ = _concordance_counts(x1, y1)
    tau = rdiv(nc - nd, n_max)
    pc = (tau + 1.0) / 2.0

    a_post = a0v + nc
    b_post = b0v + nd
    post_median = qbeta(0.5, a_post, b_post)

    if is_na(prob_interval):
        pi = float("nan")
        pi_out = None
        eti_lower = None
        eti_upper = None
    else:
        pi = float(prob_interval)
        pi_out = pi
        eti_lower = qbeta((1.0 - pi) / 2.0, a_post, b_post)
        eti_upper = qbeta(1.0 - (1.0 - pi) / 2.0, a_post, b_post)

    if fitting_parameters is None:
        return BivariateConcordanceOut(
            tau=tau,
            nc=nc,
            nd=nd,
            sample_p=pc,
            a0=a0v,
            b0=b0v,
            a_post=a_post,
            b_post=b_post,
            post_median=post_median,
            prob_interval=pi_out,
            eti_lower=eti_lower,
            eti_upper=eti_upper,
        )

    m = float(fitting_parameters)
    if math.isnan(m):
        # R's arithmetic with NA yields NA in every adjusted field.
        return BivariateConcordanceStarOut(
            tau=tau, nc=nc, nd=nd, sample_p=pc,
            a0=a0v, b0=b0v, a_post=a_post, b_post=b_post,
            post_median=post_median, prob_interval=pi_out,
            eti_lower=eti_lower, eti_upper=eti_upper,
            tau_star=None, nc_star=None, nd_star=nd,
            sample_p_star=None, a_post_star=None, b_post_star=b_post,
            post_median_star=None, eti_lower_star=None, eti_upper_star=None,
        )

    lc = len(x1) * m - m * (m + 1.0) / 2.0
    nc_star = nc - lc
    tau_star = rdiv(nc_star - nd, nc_star + nd)
    pc_star = (tau_star + 1.0) / 2.0
    a_post_star = a0v + nc_star
    post_median_star = qbeta(0.5, a_post_star, b_post)
    if is_na(prob_interval):
        eti_lower_star = None
        eti_upper_star = None
    else:
        eti_lower_star = qbeta((1.0 - pi) / 2.0, a_post_star, b_post)
        eti_upper_star = qbeta(1.0 - (1.0 - pi) / 2.0, a_post_star, b_post)

    return BivariateConcordanceStarOut(
        tau=tau, nc=nc, nd=nd, sample_p=pc,
        a0=a0v, b0=b0v, a_post=a_post, b_post=b_post,
        post_median=post_median, prob_interval=pi_out,
        eti_lower=eti_lower, eti_upper=eti_upper,
        tau_star=tau_star, nc_star=nc_star, nd_star=nd,
        sample_p_star=pc_star, a_post_star=a_post_star,
        b_post_star=b_post, post_median_star=post_median_star,
        eti_lower_star=eti_lower_star, eti_upper_star=eti_upper_star,
    )


def dfba_gamma(
    x,
    a0: float = 1.0,
    b0: float = 1.0,
    prob_interval: float = 0.95,
) -> GammaOut:
    """Goodman-Kruskal gamma from a rank-order contingency matrix.

    The input must be a 2-D array (R matrix or table).  Cell counts are
    expanded to paired observations in row-major order, exactly as upstream's
    ``rep``/``table`` round-trip does.
    """
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError("input must be in matrix or table format")

    a0v, b0v = check_positive_finite(
        a0, b0, "Both the a0 and b0 shape parameters must be positive and finite."
    )

    if not np.all(np.isfinite(arr)):
        # R: rep(..., times = NA) -> "invalid 'times' argument"
        raise ValueError("invalid 'times' argument")
    if np.any(arr < 0):
        raise ValueError("invalid 'times' argument")

    xs: list[int] = []
    ys: list[int] = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            count = int(arr[i, j])  # R's rep() truncates non-integer times
            if count:
                xs.extend([i + 1] * count)
                ys.extend([j + 1] * count)

    base = dfba_bivariate_concordance(xs, ys, a0v, b0v, prob_interval)

    return GammaOut(
        gamma=base.tau,
        a0=a0v,
        b0=b0v,
        sample_p=base.sample_p,
        nc=base.nc,
        nd=base.nd,
        a_post=base.a_post,
        b_post=base.b_post,
        prob_interval=base.prob_interval,
        post_median=base.post_median,
        eti_lower=base.eti_lower,
        eti_upper=base.eti_upper,
    )
