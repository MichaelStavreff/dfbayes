"""Result objects mirroring DFBA's S4 list subclasses.

Each class's ``R_FIELDS`` tuple reproduces the slot names *and order* of the
corresponding R class.  The one exception is
``dfba_bivariate_concordance_star_out``: R appends a second ``prob_interval``
slot, which a Python dataclass cannot represent twice.  ``R_FIELDS`` records the
duplicate for ``summary()`` and fixture checks, while the dataclass has a single
``prob_interval`` attribute.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np


def _format_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size <= 6:
            return np.array2string(value, precision=6)
        return f"<array shape={value.shape} dtype={value.dtype}>"
    if isinstance(value, BetaOutputDF):
        return f"<data.frame {len(value)} rows x 3 cols>"
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if np.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return repr(value)
    return repr(value)


def _plain_value(value: Any) -> Any:
    """Convert numpy containers to JSON-friendly Python objects."""
    if isinstance(value, (BetaOutputDF, OutputTable)):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` for strict JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _first_attr(result: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in result.R_FIELDS:
            return getattr(result, name)
    return None


def _evidence_label(bf: float) -> str:
    """Jeffreys-style evidence category for a Bayes factor >= 1."""
    if bf < 3.0:
        return "anecdotal"
    if bf < 10.0:
        return "moderate"
    if bf < 30.0:
        return "strong"
    if bf < 100.0:
        return "very strong"
    return "extreme"


class DFBAResult:
    """Base class for all R S4 list-subclass results."""

    R_CLASS: ClassVar[str] = ""
    R_FIELDS: ClassVar[tuple[str, ...]] = ()

    def summary(self) -> str:
        """Print every slot in R order, including duplicated star slots."""
        lines = [f"{self.R_CLASS}"]
        for name in self.R_FIELDS:
            lines.append(f"  {name}: {_format_value(getattr(self, name))}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        parts: list[str] = []
        for name in self.R_FIELDS[:4]:
            parts.append(f"{name}={_format_value(getattr(self, name))}")
        return f"{type(self).__name__}({', '.join(parts)}, ...)"

    def __str__(self) -> str:
        return self.summary()

    # -- Pythonic accessors -------------------------------------------------

    @property
    def bf10(self) -> float | None:
        """Bayes factor for H1 over H0, when the result has one.

        ``dfba_mcnemar`` reports both a point and an interval Bayes factor;
        this accessor returns the point value.  The median test reports one
        factor per group, so it has no single value and returns ``None``.
        """
        return _first_attr(
            self, ("BF10", "bayes_factor", "BF10point", "BF10interval")
        )

    @property
    def credible_interval(self) -> tuple[float, float] | None:
        """Equal-tailed interval if available, otherwise the HDI."""
        for lower, upper in (
            ("eti_lower", "eti_upper"),
            ("hdi_lower", "hdi_upper"),
        ):
            if lower in self.R_FIELDS and upper in self.R_FIELDS:
                low = getattr(self, lower)
                high = getattr(self, upper)
                if low is not None and high is not None:
                    return (low, high)
        return None

    @property
    def posterior_mean(self) -> float | None:
        return _first_attr(
            self,
            ("post_mean", "phimean", "omegabar", "phibar", "mean", "x_mean"),
        )

    @property
    def posterior_median(self) -> float | None:
        return _first_attr(
            self, ("post_median", "phimedian", "x_median", "median")
        )

    @property
    def probability_h1(self) -> float | None:
        return _first_attr(
            self, ("prH1", "postH1", "post_H1", "prob_positive_delta")
        )

    # -- Interchange --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the R slots as a plain Python dictionary.

        Array slots become lists.  The duplicated ``prob_interval`` slot in the
        star concordance result collapses to one key, matching the dataclass.
        """
        result: dict[str, Any] = {}
        for name in self.R_FIELDS:
            if name not in result:
                result[name] = _plain_value(getattr(self, name))
        return result

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialise the result to strict JSON.

        R's ``NA``/``NaN``/``Inf`` have no JSON representation, so non-finite
        values become ``null``.  Use ``to_dict()`` if you need to preserve
        them.
        """
        return json.dumps(
            _json_safe(self.to_dict()), indent=indent, allow_nan=False
        )

    def to_dataframe(self, *, include_arrays: bool = False):
        """Return a one-row pandas DataFrame of the result.

        Scalar slots are included by default.  Set ``include_arrays=True`` to
        add array-valued slots as object columns; pandas is an optional
        dependency (``pip install 'dfbayes[pandas]'``).
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "to_dataframe() requires pandas. Install it with "
                "`pip install 'dfbayes[pandas]'`."
            ) from exc

        row: dict[str, Any] = {}
        for name in dict.fromkeys(self.R_FIELDS):
            value = getattr(self, name)
            if isinstance(value, BetaOutputDF):
                if include_arrays:
                    row[name] = value.to_dict()
                continue
            if isinstance(value, OutputTable):
                if include_arrays:
                    row[name] = value.to_dict()
                continue
            if isinstance(value, np.ndarray):
                if include_arrays:
                    row[name] = value.tolist()
                continue
            row[name] = value
        return pd.DataFrame([row])

    def _repr_html_(self) -> str:
        """Compact HTML table for Jupyter notebooks."""
        rows = []
        for name in dict.fromkeys(self.R_FIELDS):
            value = _format_value(getattr(self, name))
            rows.append(
                "<tr><th style='text-align:left;padding:2px 8px'>"
                f"{html.escape(name)}</th>"
                "<td style='text-align:left;padding:2px 8px'>"
                f"{html.escape(value)}</td></tr>"
            )
        return (
            "<div class='dfbayes-result'>"
            f"<strong>{html.escape(self.R_CLASS)}</strong>"
            "<table style='border-collapse:collapse'>"
            + "".join(rows)
            + "</table></div>"
        )

    def interpret(self) -> str:
        """Return a short, human-readable Bayes-factor interpretation."""
        bf = self.bf10
        if bf is None:
            return (
                f"{self.R_CLASS} has no single Bayes factor; "
                "inspect the result fields directly."
            )
        try:
            value = float(bf)
        except (TypeError, ValueError):
            return f"BF10 = {bf!r}"

        if math.isnan(value):
            return "BF10 is NaN: the Bayes factor is undefined."
        if math.isinf(value):
            side = "H1" if value > 0 else "H0"
            return f"BF10 is infinite: extreme evidence for {side}."

        if value < 1.0:
            bf01 = math.inf if value == 0.0 else 1.0 / value
            return (
                f"BF10 = {value:.3g} (BF01 = {bf01:.3g}): "
                f"{_evidence_label(bf01)} evidence for H0."
            )
        return (
            f"BF10 = {value:.3g}: "
            f"{_evidence_label(value)} evidence for H1."
        )

    def plot(self, ax=None, **kwargs):
        """Plot the result with matplotlib and return the ``Axes``.

        Plotting is optional; install ``dfbayes[plot]``.  Pass ``ax=`` to draw
        on an existing axes.
        """
        from .plotting import plot_result

        return plot_result(self, ax=ax, **kwargs)


@dataclass
class BetaOutputDF:
    """Stand-in for R's ``outputdf`` data frame (no pandas dependency)."""

    x: np.ndarray
    density: np.ndarray
    cumulative_prob: np.ndarray

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.density = np.asarray(self.density, dtype=float)
        self.cumulative_prob = np.asarray(self.cumulative_prob, dtype=float)
        if not (len(self.x) == len(self.density) == len(self.cumulative_prob)):
            raise ValueError("BetaOutputDF columns must have equal length")

    @property
    def columns(self) -> tuple[str, str, str]:
        return ("x", "density", "cumulative_prob")

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.x), 3)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self.columns:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[np.ndarray]:
        for name in self.columns:
            yield getattr(self, name)

    def to_dict(self) -> dict[str, list[float]]:
        return {name: getattr(self, name).tolist() for name in self.columns}

    def to_dataframe(self):
        """Return the three columns as a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "to_dataframe() requires pandas. Install it with "
                "`pip install 'dfbayes[pandas]'`."
            ) from exc
        return pd.DataFrame(self.to_dict())


@dataclass
class OutputTable:
    """A small stand-in for an R data frame, with optional pandas export.

    Columns are stored in insertion order.  Attribute access is provided for
    convenience, so ``table.Bayes_power`` and ``table["Bayes_power"]`` are
    equivalent.
    """

    data: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        self.data = {
            name: np.asarray(values, dtype=float)
            for name, values in self.data.items()
        }
        lengths = {len(values) for values in self.data.values()}
        if len(lengths) > 1:
            raise ValueError("OutputTable columns must have equal length")

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.data)

    @property
    def shape(self) -> tuple[int, int]:
        length = len(next(iter(self.data.values()))) if self.data else 0
        return (length, len(self.data))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self.data:
            raise KeyError(key)
        return self.data[key]

    def __getattr__(self, name: str) -> np.ndarray:
        data = self.__dict__.get("data", {})
        if name in data:
            return data[name]
        raise AttributeError(name)

    def __iter__(self) -> Iterator[np.ndarray]:
        for name in self.columns:
            yield self.data[name]

    def to_dict(self) -> dict[str, list[float]]:
        return {name: values.tolist() for name, values in self.data.items()}

    def to_dataframe(self):
        """Return the table as a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "to_dataframe() requires pandas. Install it with "
                "`pip install 'dfbayes[pandas]'`."
            ) from exc
        return pd.DataFrame(self.to_dict())


# ---------------------------------------------------------------------------
# closed-form result classes
# ---------------------------------------------------------------------------

@dataclass(repr=False)
class BinomialOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_binomial_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "n1", "n2", "a0", "b0", "prob_interval", "a_post", "b_post",
        "phimean", "phimedian", "phimode", "eti_lower", "eti_upper",
        "hdi_lower", "hdi_upper",
    )

    n1: float
    n2: float
    a0: float
    b0: float
    prob_interval: float
    a_post: float
    b_post: float
    phimean: float
    phimedian: float
    phimode: float | None
    eti_lower: float
    eti_upper: float
    hdi_lower: float | None
    hdi_upper: float | None


@dataclass(repr=False)
class SignTestOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_sign_test_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "Y1", "Y2", "a0", "b0", "prob_interval", "n_pos", "n_neg",
        "a_post", "b_post", "phimean", "phimedian", "phimode",
        "eti_lower", "eti_upper", "hdi_lower", "hdi_upper",
        "post_H1", "prior_H1", "BF10", "BF01",
    )

    Y1: np.ndarray
    Y2: np.ndarray
    a0: float
    b0: float
    prob_interval: float
    n_pos: int
    n_neg: int
    a_post: float
    b_post: float
    phimean: float
    phimedian: float
    phimode: float | None
    eti_lower: float
    eti_upper: float
    hdi_lower: float | None
    hdi_upper: float | None
    post_H1: float
    prior_H1: float
    BF10: float
    BF01: float


@dataclass(repr=False)
class MedianTestOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_median_test_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "median", "nE", "nC", "Ebaserate", "Cbaserate", "nEabove",
        "nCabove", "nEbelow", "nCbelow", "a0", "b0", "a_post", "b_post",
        "postEhi", "postChi", "priorEhi", "priorChi", "BF10E", "BF10C",
    )

    median: float | None
    nE: int
    nC: int
    Ebaserate: float
    Cbaserate: float
    nEabove: int
    nCabove: int
    nEbelow: int
    nCbelow: int
    a0: float
    b0: float
    a_post: float
    b_post: float
    postEhi: float
    postChi: float
    priorEhi: float
    priorChi: float
    BF10E: float | None
    BF10C: float | None


@dataclass(repr=False)
class BetaDescriptiveOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_beta_descriptive_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "a", "b", "prob_interval", "x_mean", "x_median", "x_mode",
        "x_variance", "eti_lower", "eti_upper", "hdi_lower", "hdi_upper",
        "outputdf",
    )

    a: float
    b: float
    prob_interval: float
    x_mean: float
    x_median: float
    x_mode: float | None
    x_variance: float
    eti_lower: float
    eti_upper: float
    hdi_lower: float | None
    hdi_upper: float | None
    outputdf: BetaOutputDF


@dataclass(repr=False)
class PointBFOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_point_BF_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "method", "a_post", "b_post", "a0", "b0", "BF10", "BF01",
        "null_hypothesis", "dpriorH0", "dpostH0",
    )

    method: str
    a_post: float
    b_post: float
    a0: float
    b0: float
    BF10: float
    BF01: float
    null_hypothesis: float
    dpriorH0: float
    dpostH0: float


@dataclass(repr=False)
class IntervalBFOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_interval_BF_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "method", "a_post", "b_post", "a0", "b0", "BF10", "BF01",
        "H0lower", "H0upper", "pH0", "pH1", "postH0", "postH1",
    )

    method: str
    a_post: float
    b_post: float
    a0: float
    b0: float
    BF10: float
    BF01: float
    H0lower: float
    H0upper: float
    pH0: float
    pH1: float
    postH0: float
    postH1: float


@dataclass(repr=False)
class BetaContrastOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_beta_contrast_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "mean", "eti_lower", "eti_upper", "prob_positive_delta",
        "prior_positive_delta", "bayes_factor", "delta_quantiles",
        "a_vec", "b_vec", "contrast_vec", "prob_interval", "samples",
        "a0_vec", "b0_vec",
    )

    mean: float
    eti_lower: float
    eti_upper: float
    prob_positive_delta: float
    prior_positive_delta: float
    bayes_factor: float
    delta_quantiles: np.ndarray
    a_vec: np.ndarray
    b_vec: np.ndarray
    contrast_vec: np.ndarray
    prob_interval: float
    samples: int
    a0_vec: np.ndarray
    b0_vec: np.ndarray


@dataclass(repr=False)
class GammaOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_gamma_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "gamma", "a0", "b0", "sample_p", "nc", "nd", "a_post", "b_post",
        "prob_interval", "post_median", "eti_lower", "eti_upper",
    )

    gamma: float
    a0: float
    b0: float
    sample_p: float
    nc: int
    nd: float
    a_post: float
    b_post: float
    prob_interval: float | None
    post_median: float
    eti_lower: float | None
    eti_upper: float | None


@dataclass(repr=False)
class MannWhitneySmallOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_mann_whitney_small_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "Emean", "Cmean", "n_E", "n_C", "U_E", "U_C", "prob_interval",
        "samples", "method", "omega_E", "omegapost", "cumulative_omega",
        "priorvector", "priorprH1", "prH1", "BF10", "omegabar",
        "eti_lower", "eti_upper",
    )

    Emean: float
    Cmean: float
    n_E: int
    n_C: int
    U_E: int
    U_C: int
    prob_interval: float
    samples: int
    method: str
    omega_E: np.ndarray
    omegapost: np.ndarray
    cumulative_omega: np.ndarray
    priorvector: np.ndarray
    priorprH1: float
    prH1: float
    BF10: float
    omegabar: float
    eti_lower: float
    eti_upper: float


@dataclass(repr=False)
class MannWhitneyLargeOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_mann_whitney_large_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "Emean", "Cmean", "n_E", "n_C", "U_E", "U_C", "prob_interval",
        "a0", "b0", "a_post", "b_post", "post_mean", "post_median",
        "method", "omegabar", "eti_lower", "eti_upper", "hdi_lower",
        "hdi_upper", "priorprH1", "prH1", "BF10",
    )

    Emean: float
    Cmean: float
    n_E: int
    n_C: int
    U_E: int
    U_C: int
    prob_interval: float
    a0: float
    b0: float
    a_post: float
    b_post: float
    post_mean: float
    post_median: float
    method: str
    omegabar: float
    eti_lower: float
    eti_upper: float
    hdi_lower: float
    hdi_upper: float
    priorprH1: float
    prH1: float
    BF10: float


@dataclass(repr=False)
class WilcoxonSmallOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_wilcoxon_small_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "T_pos", "T_neg", "n", "prob_interval", "samples", "method",
        "a0", "b0", "priorvector", "priorprH1", "phiv", "phipost",
        "prH1", "BF10", "phibar", "hdi_lower", "hdi_upper",
        "cumulative_phi",
    )

    T_pos: float
    T_neg: float
    n: int
    prob_interval: float
    samples: int
    method: str
    a0: float
    b0: float
    priorvector: np.ndarray
    priorprH1: float
    phiv: np.ndarray
    phipost: np.ndarray
    prH1: float
    BF10: float
    phibar: float
    hdi_lower: float
    hdi_upper: float
    cumulative_phi: np.ndarray


@dataclass(repr=False)
class WilcoxonLargeOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_wilcoxon_large_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "T_pos", "T_neg", "n", "prob_interval", "samples", "method",
        "a0", "b0", "a_post", "b_post", "post_mean", "post_median",
        "priorprH1", "prH1", "BF10", "eti_lower", "eti_upper",
        "hdi_lower", "hdi_upper",
    )

    T_pos: float
    T_neg: float
    n: int
    prob_interval: float
    samples: int
    method: str
    a0: float
    b0: float
    a_post: float
    b_post: float
    post_mean: float
    post_median: float
    priorprH1: float
    prH1: float
    BF10: float
    eti_lower: float
    eti_upper: float
    hdi_lower: float
    hdi_upper: float


@dataclass(repr=False)
class McnemarOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_mcnemar_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "n_10", "n_01", "prob_interval", "a0", "b0", "a_post", "b_post",
        "post_mean", "post_median", "eti_lower", "eti_upper",
        "BF10point", "BF10interval", "postH1",
    )

    n_10: float
    n_01: float
    prob_interval: float
    a0: float
    b0: float
    a_post: float
    b_post: float
    post_mean: float
    post_median: float
    eti_lower: float
    eti_upper: float
    BF10point: float
    BF10interval: float
    postH1: float


@dataclass(repr=False)
class BivariateConcordanceOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_bivariate_concordance_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "tau", "nc", "nd", "sample_p", "a0", "b0", "a_post", "b_post",
        "post_median", "prob_interval", "eti_lower", "eti_upper",
    )

    tau: float
    nc: int
    nd: float
    sample_p: float
    a0: float
    b0: float
    a_post: float
    b_post: float
    post_median: float
    prob_interval: float | None
    eti_lower: float | None
    eti_upper: float | None


@dataclass(repr=False)
class BivariateConcordanceStarOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_bivariate_concordance_star_out"
    # R appends a second "prob_interval" slot. Keep it here for summary().
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "tau", "nc", "nd", "sample_p", "a0", "b0", "a_post", "b_post",
        "post_median", "prob_interval", "eti_lower", "eti_upper",
        "tau_star", "nc_star", "nd_star", "sample_p_star", "a_post_star",
        "b_post_star", "post_median_star", "prob_interval",
        "eti_lower_star", "eti_upper_star",
    )

    tau: float
    nc: int
    nd: float
    sample_p: float
    a0: float
    b0: float
    a_post: float
    b_post: float
    post_median: float
    prob_interval: float | None
    eti_lower: float | None
    eti_upper: float | None
    tau_star: float | None
    nc_star: float | None
    nd_star: float | None
    sample_p_star: float | None
    a_post_star: float | None
    b_post_star: float | None
    post_median_star: float | None
    eti_lower_star: float | None
    eti_upper_star: float | None


# ---------------------------------------------------------------------------
# simulation result classes
# ---------------------------------------------------------------------------

@dataclass(repr=False)
class SimDataOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_sim_data_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = ("pvalue", "prH1", "E", "C", "design")

    pvalue: float
    prH1: float
    E: np.ndarray
    C: np.ndarray
    design: str


@dataclass(repr=False)
class PowerCurveOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_power_curve_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "n", "nsims", "model", "design", "a0", "b0", "effect_crit",
        "block_max", "outputdf",
    )

    n: float
    nsims: int
    model: str
    design: str
    a0: float
    b0: float
    effect_crit: float
    block_max: float
    outputdf: OutputTable

    @property
    def delta_vec(self) -> np.ndarray:
        """The delta grid stored in ``outputdf.delta_value``."""
        return self.outputdf.delta_value


@dataclass(repr=False)
class TPowerOut(DFBAResult):
    R_CLASS: ClassVar[str] = "dfba_t_power_out"
    R_FIELDS: ClassVar[tuple[str, ...]] = (
        "nsims", "model", "design", "effect_crit", "deltav", "a0", "b0",
        "block_max", "outputdf",
    )

    nsims: int
    model: str
    design: str
    effect_crit: float
    deltav: float
    a0: float
    b0: float
    block_max: float
    outputdf: OutputTable


def results_to_dataframe(
    results: Iterable[DFBAResult],
    *,
    include_arrays: bool = False,
    include_class: bool = True,
):
    """Combine many result objects into one pandas DataFrame.

    Rows follow the input order.  ``result_class`` records the R S4 class name
    so heterogeneous batches can be filtered.  Columns are the union of the
    scalar slots; array slots are omitted unless ``include_arrays=True``.
    pandas is an optional dependency (``pip install 'dfbayes[pandas]'``).
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "results_to_dataframe() requires pandas. Install it with "
            "`pip install 'dfbayes[pandas]'`."
        ) from exc

    frames = []
    for result in results:
        if not isinstance(result, DFBAResult):
            raise TypeError(
                "results_to_dataframe() expects DFBAResult instances, "
                f"got {type(result).__name__}"
            )
        frame = result.to_dataframe(include_arrays=include_arrays)
        if include_class:
            frame.insert(0, "result_class", result.R_CLASS)
        frames.append(frame)

    columns = ["result_class"] if include_class else None
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True, sort=False)
