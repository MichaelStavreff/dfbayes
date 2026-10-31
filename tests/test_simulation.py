"""Tests for the Phase 2 simulation and power functions.

numpy sampling is not bit-compatible with R's Mersenne-Twister, so these tests
check the documented contract: parameterisation of each model, reproducibility
under a pinned numpy seed, structural fields, and the frozen-R summary replay
(which lives in the deterministic fixture suite).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import dfbayes as d
from dfbayes.simulation import _summarise

EULER_GAMMA = 0.5772156649015329


@pytest.mark.parametrize(
    "model",
    [
        "normal",
        "weibull",
        "cauchy",
        "lognormal",
        "chisquare",
        "logistic",
        "exponential",
        "gumbel",
        "pareto",
    ],
)
@pytest.mark.parametrize("design", ["paired", "independent"])
def test_sim_data_returns_both_conditions(model, design):
    rng = np.random.default_rng(20261101)
    result = d.dfba_sim_data(
        n=20,
        model=model,
        design=design,
        delta=0.4,
        shape1=1.0,
        shape2=1.0,
        rng=rng,
    )
    assert result.design == design
    assert len(result.E) == 20
    assert len(result.C) == 20
    assert np.all(np.isfinite(result.E))
    assert np.all(np.isfinite(result.C))
    assert 0.0 <= result.pvalue <= 1.0
    assert 0.0 <= result.prH1 <= 1.0


def test_sim_data_is_reproducible_with_a_seed():
    kwargs = dict(n=40, model="normal", design="paired", delta=0.5)
    first = d.dfba_sim_data(rng=np.random.default_rng(7), **kwargs)
    second = d.dfba_sim_data(rng=np.random.default_rng(7), **kwargs)
    np.testing.assert_array_equal(first.E, second.E)
    np.testing.assert_array_equal(first.C, second.C)
    assert first.pvalue == second.pvalue
    assert first.prH1 == second.prH1


def test_sim_data_paired_block_effect_cancels_in_differences():
    kwargs = dict(n=50, model="normal", design="paired", delta=0.5, shape1=2.0, shape2=3.0)
    plain = d.dfba_sim_data(block_max=0.0, rng=np.random.default_rng(11), **kwargs)
    blocked = d.dfba_sim_data(block_max=5.0, rng=np.random.default_rng(11), **kwargs)
    np.testing.assert_allclose(plain.E - plain.C, blocked.E - blocked.C)
    assert plain.pvalue == pytest.approx(blocked.pvalue)
    assert plain.prH1 == pytest.approx(blocked.prH1)


def test_sim_data_independent_block_effect_changes_values():
    kwargs = dict(n=50, model="normal", design="independent", delta=0.5)
    plain = d.dfba_sim_data(block_max=0.0, rng=np.random.default_rng(12), **kwargs)
    blocked = d.dfba_sim_data(block_max=5.0, rng=np.random.default_rng(12), **kwargs)
    assert not np.allclose(plain.E, blocked.E)
    assert not np.allclose(plain.C, blocked.C)


@pytest.mark.parametrize(
    "model, kwargs, expected_c_mean, expected_e_mean",
    [
        ("normal", dict(shape1=2.0, shape2=3.0), 0.0, 0.5),
        ("weibull", dict(shape1=2.0, shape2=3.0), math.gamma(1.5), 0.5 + math.gamma(4 / 3)),
        ("lognormal", dict(shape1=0.5, shape2=0.7), math.exp(0.125), math.exp(0.5 + 0.245)),
        ("chisquare", dict(shape1=4.0, shape2=6.0), 4.0, 6.5),
        ("logistic", dict(shape1=2.0, shape2=3.0), 0.0, 0.5),
        ("exponential", dict(shape1=2.0, shape2=4.0), 0.5, 0.75),
        ("gumbel", dict(shape1=2.0, shape2=3.0), 2.0 * EULER_GAMMA, 0.5 + 3.0 * EULER_GAMMA),
    ],
)
def test_model_means(model, kwargs, expected_c_mean, expected_e_mean):
    result = d.dfba_sim_data(
        n=6000,
        model=model,
        design="paired",
        delta=0.5,
        block_max=0.0,
        rng=np.random.default_rng(20261102),
        **kwargs,
    )
    assert float(np.mean(result.C)) == pytest.approx(expected_c_mean, abs=0.12)
    assert float(np.mean(result.E)) == pytest.approx(expected_e_mean, abs=0.12)


def test_cauchy_scale_and_location():
    result = d.dfba_sim_data(
        n=8000,
        model="cauchy",
        design="paired",
        delta=0.5,
        shape1=1.0,
        shape2=2.0,
        rng=np.random.default_rng(20261103),
    )
    assert float(np.median(result.C)) == pytest.approx(0.0, abs=0.12)
    assert float(np.median(result.E)) == pytest.approx(0.5, abs=0.15)


def test_pareto_support_and_median():
    result = d.dfba_sim_data(
        n=6000,
        model="pareto",
        design="paired",
        delta=0.5,
        shape1=1.0,
        shape2=2.0,
        rng=np.random.default_rng(20261104),
    )
    alpha1 = 1.16
    alpha2 = 2.32
    assert float(np.min(result.C)) >= 1.0
    assert float(np.min(result.E)) >= 1.5
    assert float(np.median(result.C)) == pytest.approx(2 ** (1 / alpha1), abs=0.12)
    assert float(np.median(result.E)) == pytest.approx(
        1.5 * 2 ** (1 / alpha2), abs=0.12
    )


def test_sim_data_validation_errors_are_r_messages():
    with pytest.raises(ValueError, match="positive difference"):
        d.dfba_sim_data(n=20, model="normal", design="paired", delta=-1)
    with pytest.raises(ValueError, match="20 or larger"):
        d.dfba_sim_data(n=19, model="normal", design="paired", delta=1)
    with pytest.raises(ValueError, match="stipulated model"):
        d.dfba_sim_data(n=20, model="bad", design="paired", delta=1)
    with pytest.raises(ValueError, match="stipulated design"):
        d.dfba_sim_data(n=20, model="normal", design="bad", delta=1)


def test_summary_fast_path_matches_full_analysis():
    # The power loops only need prH1; _summarise skips the interval/HDI work.
    # Pin it to the public analysis so the optimisation cannot drift.
    rng = np.random.default_rng(20261116)
    for design in ("paired", "independent"):
        for _ in range(25):
            e = rng.normal(0.4, 1.0, 25)
            c = rng.normal(0.0, 1.0, 25)
            _, pr_h1 = _summarise(e, c, design, 1.0, 1.0)
            if design == "paired":
                full = d.dfba_wilcoxon(e, c, method="large")
            else:
                full = d.dfba_mann_whitney(e, c, method="large")
            assert pr_h1 == full.prH1


def test_power_curve_structure_and_reproducibility():
    kwargs = dict(
        n=20,
        model="normal",
        design="paired",
        delta_step=0.1,
        samples=8,
        rng=np.random.default_rng(20261105),
    )
    result = d.dfba_power_curve(**kwargs)
    assert result.R_CLASS == "dfba_power_curve_out"
    assert result.n == 20.0
    assert result.nsims == 8
    assert result.outputdf.columns == ("delta_value", "Bayes_power", "t_power")
    assert result.outputdf.shape == (21, 3)
    np.testing.assert_allclose(result.delta_vec, np.arange(21) * 0.1)
    assert np.all((result.outputdf.Bayes_power >= 0) & (result.outputdf.Bayes_power <= 1))
    assert np.all((result.outputdf.t_power >= 0) & (result.outputdf.t_power <= 1))

    repeated = d.dfba_power_curve(
        n=20,
        model="normal",
        design="paired",
        delta_step=0.1,
        samples=8,
        rng=np.random.default_rng(20261105),
    )
    np.testing.assert_array_equal(
        result.outputdf.Bayes_power, repeated.outputdf.Bayes_power
    )
    np.testing.assert_array_equal(result.outputdf.t_power, repeated.outputdf.t_power)


def test_power_curve_rounds_n_like_r():
    result = d.dfba_power_curve(
        n=20.4,
        model="normal",
        design="paired",
        samples=2,
        rng=np.random.default_rng(20261106),
    )
    assert result.n == 20.0


def test_power_curve_validates_delta_step():
    with pytest.raises(ValueError, match="nonnegative value for delta"):
        d.dfba_power_curve(n=20, model="normal", design="paired", delta_step=-0.1)


@pytest.mark.parametrize(
    "samples",
    [None, float("nan"), float("inf"), float("-inf"), -1, -0.6],
)
def test_samples_rejected_where_r_also_errors(samples):
    with pytest.raises(ValueError, match="invalid 'times' argument"):
        d.dfba_power_curve(
            n=20, model="normal", design="paired", samples=samples
        )


@pytest.mark.parametrize("samples", [0, 0.4, 0.5, -0.4, -0.5])
def test_samples_rounding_to_zero_are_rejected_divergence(samples):
    # R accepts these and returns nsims=0 with NaN/Inf power. The port rejects
    # them deliberately; see DECISION.md item 12.
    with pytest.raises(ValueError, match="samples must be a positive integer"):
        d.dfba_power_curve(
            n=20, model="normal", design="paired", samples=samples
        )


@pytest.mark.parametrize("samples, expected", [(0.6, 1), (1.4, 1), (1.5, 2), (2.5, 2)])
def test_samples_are_rounded_like_r(samples, expected):
    result = d.dfba_power_curve(
        n=20,
        model="normal",
        design="paired",
        samples=samples,
        rng=np.random.default_rng(20261115),
    )
    assert result.nsims == expected



def test_bayes_vs_t_power_structure_and_reproducibility():
    result = d.dfba_bayes_vs_t_power(
        n_min=20,
        delta=0.5,
        model="normal",
        design="paired",
        samples=6,
        rng=np.random.default_rng(20261107),
    )
    assert result.R_CLASS == "dfba_t_power_out"
    assert result.nsims == 6
    assert result.deltav == 0.5
    assert result.outputdf.columns == ("sample_size", "Bayes_power", "t_power")
    assert result.outputdf.shape == (11, 3)
    np.testing.assert_array_equal(result.outputdf.sample_size, np.arange(20, 71, 5))
    assert np.all((result.outputdf.Bayes_power >= 0) & (result.outputdf.Bayes_power <= 1))
    assert np.all((result.outputdf.t_power >= 0) & (result.outputdf.t_power <= 1))

    repeated = d.dfba_bayes_vs_t_power(
        n_min=20,
        delta=0.5,
        model="normal",
        design="paired",
        samples=6,
        rng=np.random.default_rng(20261107),
    )
    np.testing.assert_array_equal(
        result.outputdf.Bayes_power, repeated.outputdf.Bayes_power
    )
    np.testing.assert_array_equal(result.outputdf.t_power, repeated.outputdf.t_power)


def test_bayes_vs_t_power_sample_size_grid_respects_n_min():
    result = d.dfba_bayes_vs_t_power(
        n_min=27,
        delta=0.5,
        model="normal",
        design="independent",
        samples=2,
        rng=np.random.default_rng(20261108),
    )
    np.testing.assert_array_equal(result.outputdf.sample_size, np.arange(27, 78, 5))


def test_output_table_is_pandas_friendly():
    pd = pytest.importorskip("pandas")
    result = d.dfba_power_curve(
        n=20,
        model="normal",
        design="paired",
        samples=2,
        rng=np.random.default_rng(20261109),
    )
    table = result.outputdf
    assert table.columns == ("delta_value", "Bayes_power", "t_power")
    assert len(table) == 21
    np.testing.assert_array_equal(table["delta_value"], table.delta_value)
    frame = table.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == list(table.columns)
    assert len(frame) == len(table)
    assert table.to_dict()["Bayes_power"] == table.Bayes_power.tolist()


def test_sim_data_result_helpers():
    result = d.dfba_sim_data(
        n=20,
        model="normal",
        design="paired",
        delta=0.4,
        rng=np.random.default_rng(20261113),
    )
    payload = result.to_dict()
    assert payload["E"] == result.E.tolist()
    assert payload["C"] == result.C.tolist()
    assert payload["design"] == "paired"
    assert '"pvalue"' in result.to_json()


def test_power_result_helpers_and_batch_dataframe():
    pd = pytest.importorskip("pandas")
    result = d.dfba_power_curve(
        n=20,
        model="normal",
        design="paired",
        samples=2,
        rng=np.random.default_rng(20261114),
    )
    payload = result.to_dict()
    assert payload["outputdf"]["delta_value"] == result.delta_vec.tolist()
    frame = result.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert "outputdf" not in frame.columns
    frame_with_arrays = result.to_dataframe(include_arrays=True)
    assert frame_with_arrays.loc[0, "outputdf"]["delta_value"] == result.delta_vec.tolist()

    batch = d.results_to_dataframe([result])
    assert list(batch["result_class"]) == ["dfba_power_curve_out"]
