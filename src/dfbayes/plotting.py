"""Matplotlib visualisations for DFBA result objects.

Plotting is an optional dependency: install ``dfbayes[plot]``.  Every plot
returns a matplotlib ``Axes`` so callers can compose or save figures themselves.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_TITLES = {
    "dfba_binomial_out": "Binomial posterior for phi",
    "dfba_sign_test_out": "Sign-test posterior for phi",
    "dfba_median_test_out": "Median-test posterior for phi",
    "dfba_beta_descriptive_out": "Beta posterior",
    "dfba_point_BF_out": "Point Bayes factor",
    "dfba_interval_BF_out": "Interval Bayes factor",
    "dfba_beta_contrast_out": "Posterior quantiles for the contrast",
    "dfba_gamma_out": "Gamma concordance posterior",
    "dfba_mann_whitney_small_out": "Mann-Whitney posterior for omega",
    "dfba_mann_whitney_large_out": "Mann-Whitney posterior for omega",
    "dfba_wilcoxon_small_out": "Wilcoxon posterior for phi",
    "dfba_wilcoxon_large_out": "Wilcoxon posterior for phi",
    "dfba_mcnemar_out": "McNemar posterior for phi",
    "dfba_bivariate_concordance_out": "Concordance posterior for tau",
    "dfba_bivariate_concordance_star_out": "Concordance posterior for tau",
    "dfba_sim_data_out": "Simulated data",
    "dfba_power_curve_out": "Bayesian and frequentist power by delta",
    "dfba_t_power_out": "Bayesian and frequentist power by sample size",
}


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Plotting requires matplotlib. Install it with "
            "`pip install 'dfbayes[plot]'`."
        ) from exc
    return plt


def _axes(ax: Any, figsize: tuple[float, float] | None):
    if ax is not None:
        return ax
    _, created = _pyplot().subplots(figsize=figsize or (6.4, 4.0))
    return created


def _pair(result: Any, lower: str, upper: str) -> tuple[float, float] | None:
    low = getattr(result, lower, None)
    high = getattr(result, upper, None)
    if low is None or high is None:
        return None
    try:
        low_f = float(low)
        high_f = float(high)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(low_f) and np.isfinite(high_f)):
        return None
    return (low_f, high_f)


def _shade(
    ax: Any,
    interval: tuple[float, float] | None,
    color: str,
    label: str,
    *,
    alpha: float = 0.15,
) -> None:
    if interval is None:
        return
    ax.axvspan(
        interval[0],
        interval[1],
        color=color,
        alpha=alpha,
        linewidth=0,
        label=label,
    )


def _finish(ax: Any, title: str, xlabel: str, ylabel: str):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
    if unique_handles:
        ax.legend(unique_handles, unique_labels, frameon=False)
    return ax


def _plot_beta_pdf(result: Any, ax: Any) -> None:
    from scipy.stats import beta as beta_distribution

    a = float(result.a_post)
    b = float(result.b_post)
    x = np.linspace(0.0, 1.0, 500)
    ax.plot(x, beta_distribution.pdf(x, a, b), color="C0", label="posterior")
    _shade(ax, _pair(result, "eti_lower", "eti_upper"), "C0", "ETI")
    _shade(ax, _pair(result, "hdi_lower", "hdi_upper"), "C1", "HDI")

    mean = result.posterior_mean
    median = result.posterior_median
    if mean is not None and np.isfinite(mean):
        ax.axvline(float(mean), color="C0", linestyle="--", linewidth=1.0, label="mean")
    if median is not None and np.isfinite(median):
        ax.axvline(float(median), color="C3", linestyle=":", linewidth=1.0, label="median")


def _plot_beta_outputdf(result: Any, ax: Any) -> None:
    output = result.outputdf
    ax.plot(output.x, output.density, color="C0", label="posterior")
    _shade(ax, _pair(result, "eti_lower", "eti_upper"), "C0", "ETI")
    _shade(ax, _pair(result, "hdi_lower", "hdi_upper"), "C1", "HDI")

    mean = result.posterior_mean
    median = result.posterior_median
    if mean is not None and np.isfinite(mean):
        ax.axvline(float(mean), color="C0", linestyle="--", linewidth=1.0, label="mean")
    if median is not None and np.isfinite(median):
        ax.axvline(float(median), color="C3", linestyle=":", linewidth=1.0, label="median")


def _plot_discrete(result: Any, ax: Any, x: Any, probability: Any, mean: Any) -> None:
    x_values = np.asarray(x, dtype=float)
    probabilities = np.asarray(probability, dtype=float)
    ax.plot(x_values, probabilities, color="C0", label="posterior")
    ax.fill_between(
        x_values, 0.0, probabilities, color="C0", alpha=0.15, linewidth=0
    )
    _shade(ax, _pair(result, "eti_lower", "eti_upper"), "C0", "ETI")
    _shade(ax, _pair(result, "hdi_lower", "hdi_upper"), "C1", "HDI")
    if mean is not None and np.isfinite(mean):
        ax.axvline(float(mean), color="C0", linestyle="--", linewidth=1.0, label="mean")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)


def _plot_delta(result: Any, ax: Any) -> None:
    quantiles = np.asarray(result.delta_quantiles, dtype=float)
    probabilities = np.linspace(0.0, 1.0, len(quantiles))
    ax.plot(quantiles, probabilities, color="C0", label="posterior quantiles")
    ax.axvline(0.0, color="0.3", linestyle="--", linewidth=1.0, label="delta = 0")
    _shade(ax, _pair(result, "eti_lower", "eti_upper"), "C0", "ETI")
    mean = getattr(result, "mean", None)
    if mean is not None and np.isfinite(mean):
        ax.axvline(float(mean), color="C0", linestyle="--", linewidth=1.0, label="mean")


def _plot_power(ax: Any, x: Any, bayes: Any, frequentist: Any) -> None:
    x_values = np.asarray(x, dtype=float)
    ax.plot(
        x_values,
        np.asarray(bayes, dtype=float),
        color="C0",
        marker="o",
        label="Bayesian",
    )
    ax.plot(
        x_values,
        np.asarray(frequentist, dtype=float),
        color="C1",
        marker="o",
        linestyle="--",
        label="frequentist t",
    )
    ax.set_ylim(0.0, 1.0)


def _plot_sim_data(result: Any, ax: Any) -> None:
    e = np.asarray(result.E, dtype=float)
    c = np.asarray(result.C, dtype=float)
    if result.design == "independent":
        ax.boxplot([e, c])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["E", "C"])
    else:
        ax.boxplot([e - c])
        ax.set_xticks([1])
        ax.set_xticklabels(["E - C"])


def plot_result(
    result: Any,
    *,
    ax: Any = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Plot one DFBA result and return the matplotlib ``Axes``."""
    ax = _axes(ax, figsize)
    title = title or _TITLES.get(result.R_CLASS, result.R_CLASS)

    if result.R_CLASS == "dfba_sim_data_out":
        _plot_sim_data(result, ax)
        return _finish(ax, title, "group", "simulated value")

    if result.R_CLASS == "dfba_power_curve_out":
        _plot_power(
            ax,
            result.outputdf.delta_value,
            result.outputdf.Bayes_power,
            result.outputdf.t_power,
        )
        return _finish(ax, title, "delta", "power estimate")

    if result.R_CLASS == "dfba_t_power_out":
        _plot_power(
            ax,
            result.outputdf.sample_size,
            result.outputdf.Bayes_power,
            result.outputdf.t_power,
        )
        return _finish(ax, title, "sample size", "power estimate")

    if hasattr(result, "outputdf"):
        _plot_beta_outputdf(result, ax)
        return _finish(ax, title, "phi", "density")

    if hasattr(result, "omegapost") and hasattr(result, "omega_E"):
        _plot_discrete(
            result,
            ax,
            result.omega_E,
            result.omegapost,
            getattr(result, "omegabar", None),
        )
        return _finish(ax, title, "omega", "posterior probability")

    if hasattr(result, "phipost") and hasattr(result, "phiv"):
        _plot_discrete(
            result,
            ax,
            result.phiv,
            result.phipost,
            getattr(result, "phibar", None),
        )
        return _finish(ax, title, "phi", "posterior probability")

    if hasattr(result, "delta_quantiles"):
        _plot_delta(result, ax)
        return _finish(ax, title, "delta", "posterior probability")

    if hasattr(result, "null_hypothesis"):
        _plot_beta_pdf(result, ax)
        null = float(result.null_hypothesis)
        ax.axvline(null, color="C2", linestyle="--", linewidth=1.0, label="H0")
        return _finish(ax, title, "phi", "density")

    if hasattr(result, "H0lower") and hasattr(result, "H0upper"):
        _plot_beta_pdf(result, ax)
        ax.axvspan(
            float(result.H0lower),
            float(result.H0upper),
            color="C2",
            alpha=0.12,
            linewidth=0,
            label="H0 interval",
        )
        return _finish(ax, title, "phi", "density")

    if hasattr(result, "a_post") and hasattr(result, "b_post"):
        _plot_beta_pdf(result, ax)
        return _finish(ax, title, "phi", "density")

    if hasattr(result, "a") and hasattr(result, "b"):
        _plot_beta_pdf(result, ax)
        return _finish(ax, title, "phi", "density")

    raise NotImplementedError(
        f"No plot is defined for {result.R_CLASS or type(result).__name__}."
    )
