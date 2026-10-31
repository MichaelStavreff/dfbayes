"""Tests for the optional matplotlib plotting layer."""

from __future__ import annotations

import pytest

# Plotting is an optional extra. Skip the whole module cleanly when the
# minimal-dependency CI leg runs without `dfbayes[plot]`.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from dfbayes.plotting import plot_result
from dfbayes.results import DFBAResult

from .test_results import RESULTS


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_every_result_plots_to_an_axes(name):
    result = RESULTS[name]
    ax = result.plot()
    assert isinstance(ax, Axes)
    assert ax.get_title()
    assert ax.get_xlabel()
    assert ax.get_ylabel()


def test_plot_accepts_existing_axes():
    result = RESULTS["dfba_binomial_out"]
    _, ax = plt.subplots()
    returned = result.plot(ax=ax)
    assert returned is ax


def test_plot_accepts_title_and_figsize():
    result = RESULTS["dfba_mcnemar_out"]
    ax = plot_result(result, title="Custom title", figsize=(4, 3))
    assert ax.get_title() == "Custom title"
    assert ax.figure.get_size_inches().tolist() == pytest.approx([4.0, 3.0])


def test_plot_marks_null_hypothesis():
    result = RESULTS["dfba_point_BF_out"]
    ax = result.plot()
    labels = [line.get_label() for line in ax.get_lines()]
    assert "H0" in labels


def test_plot_marks_interval_null_region():
    result = RESULTS["dfba_interval_BF_out"]
    ax = result.plot()
    assert any(
        "H0 interval" in (patch.get_label() or "") for patch in ax.patches
    )


def test_plot_marks_delta_zero_for_contrast():
    result = RESULTS["dfba_beta_contrast_out"]
    ax = result.plot()
    assert "delta = 0" in [line.get_label() for line in ax.get_lines()]


def test_plot_unknown_result_raises():
    class Unknown(DFBAResult):
        R_CLASS = "dfba_unknown_out"

    with pytest.raises(NotImplementedError, match="dfba_unknown_out"):
        Unknown().plot()


def _sim_data(design: str):
    import numpy as np

    import dfbayes as d

    return d.dfba_sim_data(
        n=20,
        model="normal",
        design=design,
        delta=0.4,
        rng=np.random.default_rng(20261110),
    )


def test_sim_data_plot_independent():
    ax = _sim_data("independent").plot()
    assert isinstance(ax, Axes)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["E", "C"]


def test_sim_data_plot_paired():
    ax = _sim_data("paired").plot()
    assert isinstance(ax, Axes)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["E - C"]


def test_power_curve_plot():
    import numpy as np

    import dfbayes as d

    result = d.dfba_power_curve(
        n=20,
        model="normal",
        design="paired",
        samples=2,
        rng=np.random.default_rng(20261111),
    )
    ax = result.plot()
    assert isinstance(ax, Axes)
    assert [line.get_label() for line in ax.get_lines()] == ["Bayesian", "frequentist t"]
    assert ax.get_ylim() == pytest.approx((0.0, 1.0))


def test_bayes_vs_t_power_plot():
    import numpy as np

    import dfbayes as d

    result = d.dfba_bayes_vs_t_power(
        n_min=20,
        delta=0.4,
        model="normal",
        design="paired",
        samples=2,
        rng=np.random.default_rng(20261112),
    )
    ax = result.plot()
    assert isinstance(ax, Axes)
    assert [line.get_label() for line in ax.get_lines()] == ["Bayesian", "frequentist t"]
