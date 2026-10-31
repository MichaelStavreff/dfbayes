"""Tests for the Monte-Carlo functions.

Two kinds of check are used, following the RNG policy in ``DECISION.md``:

* deterministic replay: feed the frozen R intermediate (``fomega``, ``fphi``,
  ``delta``/``delta0``) into the Python downstream maths and compare every
  field with the R output;
* sampling checks: compare the sampler against a known analytic distribution
  and check that it is deterministic under a pinned seed.  numpy sampling is
  never claimed to be bit-compatible with R.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import rankdata

import dfbayes as d
from dfbayes.contrast import _finish_beta_contrast
from dfbayes.mann_whitney import (
    _fomega_counts,
    _mann_whitney_small_from_counts,
    _sample_fomega,
)
from dfbayes.wilcoxon import (
    _average_ranks,
    _fphi_counts,
    _sample_fphi,
    _wilcoxon_small_from_counts,
)

from ._support import assert_result_matches, decode, load_stochastic

STOCHASTIC = load_stochastic()

pytestmark = pytest.mark.skipif(
    not STOCHASTIC, reason="stochastic fixtures not generated (DFBA_STOCHASTIC=1)"
)


def _frozen(case: dict) -> dict:
    return {name: decode(value) for name, value in case["frozen"].items()}


def _fields(case: dict) -> dict:
    return {name: decode(value) for name, value in case["fields"]}


# ---------------------------------------------------------------------------
# Deterministic replay of the frozen R intermediates
# ---------------------------------------------------------------------------


def test_replay_beta_contrast_two_conditions() -> None:
    case = STOCHASTIC["beta_contrast_2"]
    frozen = _frozen(case)
    fields = _fields(case)

    result = _finish_beta_contrast(
        contrast=fields["contrast_vec"],
        a_vec=fields["a_vec"],
        b_vec=fields["b_vec"],
        a0=fields["a0_vec"],
        b0=fields["b0_vec"],
        pi=fields["prob_interval"],
        samples=fields["samples"],
        mean_delta=fields["mean"],
        delta=frozen["delta"],
        delta0=frozen["delta0"],
    )
    assert_result_matches("beta_contrast_2", result, case)


def test_replay_beta_contrast_four_conditions() -> None:
    case = STOCHASTIC["beta_contrast_4"]
    frozen = _frozen(case)
    fields = _fields(case)

    result = _finish_beta_contrast(
        contrast=fields["contrast_vec"],
        a_vec=fields["a_vec"],
        b_vec=fields["b_vec"],
        a0=fields["a0_vec"],
        b0=fields["b0_vec"],
        pi=fields["prob_interval"],
        samples=fields["samples"],
        mean_delta=fields["mean"],
        delta=frozen["delta"],
        delta0=frozen["delta0"],
    )
    assert_result_matches("beta_contrast_4", result, case)


def _replay_mw(name: str):
    case = STOCHASTIC[name]
    frozen = _frozen(case)
    fields = _fields(case)
    # The small-result class does not expose a0/b0; both fixture calls use the
    # default uniform prior.  The frozen priorvector would catch a mismatch.
    e = np.full(int(fields["n_E"]), float(fields["Emean"]))
    c = np.full(int(fields["n_C"]), float(fields["Cmean"]))
    return _mann_whitney_small_from_counts(
        e,
        c,
        int(fields["U_E"]),
        int(fields["U_C"]),
        1.0,
        1.0,
        float(fields["prob_interval"]),
        int(fields["samples"]),
        frozen["fomega"],
    )


def test_stochastic_mw_cases_cover_both_branches() -> None:
    a = _fields(STOCHASTIC["mw_small_branch_a"])
    b = _fields(STOCHASTIC["mw_small_branch_b"])
    assert a["n_E"] * a["n_C"] > a["U_E"] + a["U_C"]
    assert b["n_E"] * b["n_C"] <= b["U_E"] + b["U_C"]


def test_replay_mw_small_branch_a() -> None:
    case = STOCHASTIC["mw_small_branch_a"]
    assert_result_matches("mw_small_branch_a", _replay_mw("mw_small_branch_a"), case)


def test_replay_mw_small_branch_b() -> None:
    case = STOCHASTIC["mw_small_branch_b"]
    assert_result_matches("mw_small_branch_b", _replay_mw("mw_small_branch_b"), case)


def test_replay_wilcoxon_small() -> None:
    case = STOCHASTIC["wil_small"]
    frozen = _frozen(case)
    fields = _fields(case)
    result = _wilcoxon_small_from_counts(
        float(fields["T_pos"]),
        float(fields["T_neg"]),
        int(fields["n"]),
        float(fields["a0"]),
        float(fields["b0"]),
        float(fields["prob_interval"]),
        int(fields["samples"]),
        frozen["fphi"],
    )
    assert_result_matches("wil_small", result, case)


# ---------------------------------------------------------------------------
# Analytic checks at the null value (omega = phi = 0.5)
# ---------------------------------------------------------------------------


def _mann_whitney_null_pmf(n_e: int, n_c: int) -> np.ndarray:
    """Exact null distribution of the Mann-Whitney U statistic.

    At ``omega = 0.5`` the two exponential samples are exchangeable, so all
    interleavings of ``n_e`` E's and ``n_c`` C's are equally likely.  Building
    the sequence left to right, appending an E after ``j`` C's contributes
    ``j`` inversions and appending a C contributes none.
    """
    # dp[i][j][u] = number of arrangements of i E's and j C's with u inversions.
    dp = [[[0] * (n_e * n_c + 1) for _ in range(n_c + 1)] for _ in range(n_e + 1)]
    dp[0][0][0] = 1
    for i in range(n_e + 1):
        for j in range(n_c + 1):
            for u, count in enumerate(dp[i][j]):
                if count == 0:
                    continue
                if i < n_e:
                    dp[i + 1][j][u + j] += count
                if j < n_c:
                    dp[i][j + 1][u] += count
    counts = np.asarray(dp[n_e][n_c], dtype=float)
    return counts / math.comb(n_e + n_c, n_e)


def test_mw_fomega_matches_null_branch_b() -> None:
    e = np.array([2.0, 4.0, 6.0])
    c = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    ue, uc = 6, 9
    assert len(e) * len(c) <= ue + uc

    samples = 100_000
    count = _fomega_counts(
        e, c, ue, uc, 0.5, samples, np.random.default_rng(101)
    )
    expected = _mann_whitney_null_pmf(len(e), len(c))[ue]
    tolerance = 5.0 * math.sqrt(expected * (1.0 - expected) / samples)
    assert abs(count / samples - expected) <= tolerance


def test_mw_fomega_matches_null_branch_a() -> None:
    e = np.array([1.0, 1.0, 2.0, 2.0])
    c = np.array([1.0, 1.0, 2.0, 2.0])
    ue, uc = 4, 4
    assert len(e) * len(c) > ue + uc

    samples = 100_000
    count = _fomega_counts(
        e, c, ue, uc, 0.5, samples, np.random.default_rng(102)
    )
    # Branch A counts the first Ntot=8 row-major comparisons.  Those are the
    # complete 2 x 4 block (E1,E2) x (C1..C4), so the null is U(2, 4).
    expected = _mann_whitney_null_pmf(2, 4)[ue]
    tolerance = 5.0 * math.sqrt(expected * (1.0 - expected) / samples)
    assert abs(count / samples - expected) <= tolerance


def _poisson_binomial_pmf(weights: np.ndarray, phi: float) -> np.ndarray:
    total = int(np.sum(weights))
    pmf = np.zeros(total + 1, dtype=float)
    pmf[0] = 1.0
    for weight in weights:
        updated = pmf * (1.0 - phi)
        updated[int(weight) :] += pmf[: -int(weight)] * phi
        pmf = updated
    return pmf


def test_wilcoxon_fphi_matches_poisson_binomial_null() -> None:
    n = 5
    tpos = 8
    phi = 0.5
    samples = 100_000

    count = _fphi_counts(
        tpos, n, phi, samples, np.random.default_rng(103)
    )
    expected = _poisson_binomial_pmf(np.arange(1, n + 1), phi)[tpos]
    tolerance = 5.0 * math.sqrt(expected * (1.0 - expected) / samples)
    assert abs(count / samples - expected) <= tolerance


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_mw_sampler_is_deterministic() -> None:
    e = np.array([2.0, 4.0, 6.0])
    c = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    first = _sample_fomega(
        e, c, 6, 9, 10_000, np.random.default_rng(7)
    )
    second = _sample_fomega(
        e, c, 6, 9, 10_000, np.random.default_rng(7)
    )
    other = _sample_fomega(
        e, c, 6, 9, 10_000, np.random.default_rng(8)
    )
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert first.shape == (200,)


def test_wilcoxon_sampler_is_deterministic() -> None:
    first = _sample_fphi(8, 5, 10_000, np.random.default_rng(7))
    second = _sample_fphi(8, 5, 10_000, np.random.default_rng(7))
    other = _sample_fphi(8, 5, 10_000, np.random.default_rng(8))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert first.shape == (200,)


def test_average_ranks_matches_scipy_including_ties() -> None:
    """The fast rank helper must reproduce R's average tie handling exactly."""
    rng = np.random.default_rng(20261106)
    cases = [
        np.array([], dtype=float),
        np.array([1.0]),
        np.array([2.0, 2.0]),
        np.full(7, 3.5),
        np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0]),
        np.array([-2.0, -0.0, 0.0, 2.0]),
        np.round(rng.normal(size=50), 1),
        rng.integers(0, 4, size=40).astype(float),
    ]
    for values in cases:
        np.testing.assert_array_equal(
            _average_ranks(values), rankdata(values, method="average")
        )


def test_beta_contrast_sampler_is_deterministic() -> None:
    kwargs = {
        "n1_vec": [22, 15],
        "n2_vec": [18, 25],
        "contrast_vec": [1, -1],
    }
    first = d.dfba_beta_contrast(**kwargs, rng=np.random.default_rng(7))
    second = d.dfba_beta_contrast(**kwargs, rng=np.random.default_rng(7))
    other = d.dfba_beta_contrast(**kwargs, rng=np.random.default_rng(8))
    assert np.array_equal(first.delta_quantiles, second.delta_quantiles)
    assert not np.array_equal(first.delta_quantiles, other.delta_quantiles)


# ---------------------------------------------------------------------------
# Public wrappers: shape, range, and internal consistency
# ---------------------------------------------------------------------------


def test_mw_small_public_wrapper() -> None:
    result = d.dfba_mann_whitney(
        [2, 4, 6],
        [1, 3, 5, 7, 9],
        method="small",
        samples=10_000,
        rng=np.random.default_rng(11),
    )
    assert result.method == "small"
    assert result.U_E == 6 and result.U_C == 9
    assert result.omegapost.shape == (200,)
    assert result.cumulative_omega.shape == (200,)
    assert result.omegapost.min() >= 0.0
    assert result.cumulative_omega[-1] == pytest.approx(1.0)
    assert 0.0 < result.prH1 < 1.0
    assert result.BF10 > 0.0
    assert 0.0 <= result.eti_lower <= result.eti_upper <= 1.0


@pytest.mark.parametrize(
    "n_e, n_c, expected",
    [
        (19, 21, "small"),  # nH = 2*19*21/40 = 19.95
        (20, 20, "large"),  # nH = 2*20*20/40 = 20.00
    ],
)
def test_method_none_selects_at_the_nH_20_boundary(n_e, n_c, expected) -> None:
    """Pin both sides of ``nH >= 20`` in R's automatic method selection."""
    e = np.linspace(0.0, 1.0, n_e)
    c = np.linspace(0.0, 1.0, n_c)
    result = d.dfba_mann_whitney(
        e,
        c,
        samples=10_000,
        rng=np.random.default_rng(42),
    )
    assert result.method == expected
    assert result.R_CLASS == f"dfba_mann_whitney_{expected}_out"



def test_wilcoxon_small_public_wrapper() -> None:
    result = d.dfba_wilcoxon(
        [1.49, 0.64, 0.96, 2.34, 0.78],
        [0.53, 0.55, 0.58, 0.97, 0.60],
        method="small",
        samples=10_000,
        rng=np.random.default_rng(11),
    )
    assert result.method == "small"
    assert result.n == 5
    assert result.T_pos + result.T_neg == 15
    assert result.phipost.shape == (200,)
    assert result.cumulative_phi[-1] == pytest.approx(1.0)
    assert 0.0 < result.prH1 < 1.0
    assert result.BF10 > 0.0
    assert 0.0 <= result.hdi_lower <= result.hdi_upper <= 1.0


def test_beta_contrast_public_wrapper() -> None:
    result = d.dfba_beta_contrast(
        [22, 15],
        [18, 25],
        [1, -1],
        rng=np.random.default_rng(11),
    )
    assert result.delta_quantiles.shape == (101,)
    assert np.all(np.diff(result.delta_quantiles) >= 0.0)
    assert 0.0 <= result.prob_positive_delta <= 1.0
    assert 0.0 <= result.prior_positive_delta <= 1.0
    assert result.bayes_factor > 0.0
    assert result.eti_lower <= result.eti_upper
