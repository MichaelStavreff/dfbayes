"""Tests for the optional matplotlib plotting layer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
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
