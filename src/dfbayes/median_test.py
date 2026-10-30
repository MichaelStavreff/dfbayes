"""Bayesian median test."""

from __future__ import annotations

import math

import numpy as np

from ._core import as_float_vector, check_positive_finite, pbeta, rdiv
from .results import MedianTestOut


def dfba_median_test(
    E,
    C,
    a0: float = 1.0,
    b0: float = 1.0,
) -> MedianTestOut:
    """Bayesian median test for two independent groups.

    NA/NaN values are removed independently from ``E`` and ``C`` (R does not
    pair the vectors here).  Values equal to the combined median count as
    below-median, matching ``<=`` in the R source.
    """
    a0v, b0v = check_positive_finite(
        a0, b0, "Both a0 and b0 must be positive and finite."
    )

    e_all = as_float_vector(E, "E")
    c_all = as_float_vector(C, "C")
    e = e_all[~np.isnan(e_all)]
    c = c_all[~np.isnan(c_all)]

    combined = np.concatenate([e, c])
    med = None if len(combined) == 0 else float(np.median(combined))

    n_e_above = int(np.sum(e > med)) if med is not None else 0
    n_e_below = int(np.sum(e <= med)) if med is not None else 0
    n_c_above = int(np.sum(c > med)) if med is not None else 0
    n_c_below = int(np.sum(c <= med)) if med is not None else 0

    a_post = n_e_above + a0v
    b_post = n_c_above + b0v

    e_base = rdiv(len(e), len(e) + len(c))
    c_base = 1.0 - e_base

    post_ehi = 1.0 - pbeta(e_base, a_post, b_post)
    post_chi = 1.0 - post_ehi
    prior_ehi = 1.0 - pbeta(e_base, a0v, b0v)
    prior_chi = 1.0 - prior_ehi

    # R's ifelse() returns NA (not NaN) when the tested condition is NA.
    # This happens when both groups are empty and Ebaserate is 0/0.
    if math.isnan(post_chi):
        bf10e = None
    elif post_chi == 0.0:
        bf10e = math.inf
    else:
        bf10e = rdiv(rdiv(1.0, post_chi) - 1.0, rdiv(1.0, prior_chi) - 1.0)

    if math.isnan(post_ehi):
        bf01e = None
    elif post_ehi == 0.0:
        bf01e = math.inf
    else:
        bf01e = rdiv(rdiv(1.0, prior_chi) - 1.0, rdiv(1.0, post_chi) - 1.0)

    return MedianTestOut(
        median=med,
        nE=len(e),
        nC=len(c),
        Ebaserate=e_base,
        Cbaserate=c_base,
        nEabove=n_e_above,
        nCabove=n_c_above,
        nEbelow=n_e_below,
        nCbelow=n_c_below,
        a0=a0v,
        b0=b0v,
        a_post=a_post,
        b_post=b_post,
        postEhi=post_ehi,
        postChi=post_chi,
        priorEhi=prior_ehi,
        priorChi=prior_chi,
        BF10E=bf10e,
        BF10C=bf01e,
    )
