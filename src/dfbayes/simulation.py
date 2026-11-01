"""Simulation and power tools.

These functions mirror DFBA's ``dfba_sim_data``, ``dfba_power_curve`` and
``dfba_bayes_vs_t_power``.  Data generation uses numpy's ``Generator`` rather
than R's Mersenne-Twister, so draws are not bit-compatible with R; the
downstream statistics are computed with the same closed-form formulas.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import stdtr

from ._core import check_positive_finite, is_na, rdiv
from .mann_whitney import _mann_whitney_prh1
from .results import OutputTable, PowerCurveOut, SimDataOut, TPowerOut
from .wilcoxon import _wilcoxon_prh1

_MODELS = (
    "normal",
    "weibull",
    "cauchy",
    "lognormal",
    "chisquare",
    "logistic",
    "exponential",
    "gumbel",
    "pareto",
)
_DESIGNS = ("paired", "independent")

_MODEL_ERROR = (
    "The set of distributions for model are: \n\t\t"
    + "\n\t\t".join(_MODELS)
    + "\nThe stipulated model is not on the list"
)
_DESIGN_ERROR = (
    "The set of distributions for design are: \n\t\t"
    + "\n\t\t".join(_DESIGNS)
    + "\nThe stipulated design is not on the list"
)

_SHAPE_MESSAGES = {
    "normal": "shape1 and shape2 are standard deviations and must be positive values.",
    "weibull": (
        "shape1 and shape2 are the Weibull shape parameters for the respective "
        "control and experiment conditions and must be positive values."
    ),
    "cauchy": (
        "shape1 and shape2 are the Cauchy distribution scale factors for the "
        "respective control and experiment conditions and must be positive values."
    ),
    "lognormal": (
        "shape1 and shape2 are the sdlog values for the respective control and "
        "experiment conditions and must be positive values."
    ),
    "chisquare": (
        "shape1 and shape2 are df for respectively the control and experiment "
        "conditions and must be positive values."
    ),
    "logistic": "shape1 and shape2 are scale factors and must be positive values.",
    "exponential": "shape1 and shape2 are scale factors and must be positive values.",
    "gumbel": "shape1 and shape2 are scale factors and must be positive values.",
}
_PARETO_ERROR = "shape1 and shape2 must be greater than or equal to 1."


def _as_number(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a single numeric value") from exc


def _check_model(model: Any) -> str:
    if is_na(model):
        raise ValueError("missing value where TRUE/FALSE needed")
    if model not in _MODELS:
        raise ValueError(_MODEL_ERROR)
    return model


def _check_design(design: Any) -> str:
    if is_na(design):
        raise ValueError("missing value where TRUE/FALSE needed")
    if design not in _DESIGNS:
        raise ValueError(_DESIGN_ERROR)
    return design


def _check_n(n: Any, message: str) -> int:
    if is_na(n):
        raise ValueError("missing value where TRUE/FALSE needed")
    value = _as_number(n, "n")
    if math.isinf(value):
        raise ValueError("missing value where TRUE/FALSE needed")
    if value < 20.0 or value % 1.0 != 0.0:
        raise ValueError(message)
    return int(value)


def _check_effect_crit(effect_crit: Any, message: str) -> float:
    if is_na(effect_crit):
        raise ValueError("missing value where TRUE/FALSE needed")
    value = _as_number(effect_crit, "effect_crit")
    if value < 0.0 or value > 1.0:
        raise ValueError(message)
    return value


def _check_block_max(block_max: Any) -> float:
    if is_na(block_max):
        raise ValueError("block_max must be nonnegative")
    value = _as_number(block_max, "block_max")
    if value < 0.0:
        raise ValueError("block_max must be nonnegative")
    return value


def _check_shapes(model: str, shape1: Any, shape2: Any) -> tuple[float, float]:
    if is_na(shape1) or is_na(shape2):
        raise ValueError("missing value where TRUE/FALSE needed")
    s1 = _as_number(shape1, "shape1")
    s2 = _as_number(shape2, "shape2")
    if model == "pareto":
        if s1 < 1.0 or s2 < 1.0:
            raise ValueError(_PARETO_ERROR)
    elif s1 <= 0.0 or s2 <= 0.0:
        raise ValueError(_SHAPE_MESSAGES[model])
    return s1, s2


def _generator(rng: Any) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def _draw(
    model: str,
    n: int,
    delta: float,
    shape1: float,
    shape2: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the control and experimental vectors for one model."""
    if model == "normal":
        c = rng.normal(0.0, shape1, n)
        e = rng.normal(delta, shape2, n)
    elif model == "weibull":
        c = rng.weibull(shape1, n)
        e = delta + rng.weibull(shape2, n)
    elif model == "cauchy":
        c = 0.0 + shape1 * rng.standard_cauchy(n)
        e = delta + shape2 * rng.standard_cauchy(n)
    elif model == "lognormal":
        c = rng.lognormal(0.0, shape1, n)
        e = rng.lognormal(delta, shape2, n)
    elif model == "chisquare":
        c = rng.chisquare(shape1, n)
        e = delta + rng.chisquare(shape2, n)
    elif model == "logistic":
        c = rng.logistic(0.0, shape1, n)
        e = rng.logistic(delta, shape2, n)
    elif model == "exponential":
        c = rng.exponential(1.0 / shape1, n)
        e = delta + rng.exponential(1.0 / shape2, n)
    elif model == "gumbel":
        g = rng.uniform(0.00001, 0.99999, n)
        c = -shape1 * np.log(np.log(1.0 / g))
        g2 = rng.uniform(0.00001, 0.99999, n)
        e = delta - shape2 * np.log(np.log(1.0 / g2))
    elif model == "pareto":
        g = rng.uniform(0.00001, 0.99999, n)
        c = 1.0 / (1.0 - g) ** (1.0 / (1.16 * shape1))
        g2 = rng.uniform(0.00001, 0.99999, n)
        e = (1.0 + delta) / (1.0 - g2) ** (1.0 / (1.16 * shape2))
    else:  # pragma: no cover - validated by _check_model
        raise ValueError(_MODEL_ERROR)
    return np.asarray(c, dtype=float), np.asarray(e, dtype=float)


def _summarise(
    e: np.ndarray,
    c: np.ndarray,
    design: str,
    a0: float,
    b0: float,
) -> tuple[float, float]:
    """Return the one-tailed t p-value and the Bayesian posterior probability."""
    n = len(e)
    if design == "independent":
        numerator = float(np.mean(e) - np.mean(c))
        denominator = math.sqrt(
            (float(np.var(e, ddof=1)) + float(np.var(c, ddof=1))) / n
        )
        t_sample = rdiv(numerator, denominator)
        # Student-t survival function, without the rv_continuous dispatch
        # overhead.  scipy's t.sf(x, df) is exactly stdtr(df, -x).
        pvalue = float(stdtr(2 * n - 2, -t_sample))
        pr_h1 = _mann_whitney_prh1(e, c, a0, b0)
    else:
        d = e - c
        t_sample = rdiv(math.sqrt(n) * float(np.mean(d)), float(np.std(d, ddof=1)))
        pvalue = float(stdtr(n - 1, -t_sample))
        pr_h1 = _wilcoxon_prh1(e, c, a0, b0)
    return pvalue, float(pr_h1)


def _simulate(
    n: int,
    a0: float,
    b0: float,
    model: str,
    design: str,
    delta: float,
    shape1: float,
    shape2: float,
    block_max: float,
    rng: Any,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Draw and summarise one data set; all arguments must be validated.

    Split out from :func:`dfba_sim_data` so the power loops do not repeat the
    argument validation or build a result object for every replication.
    """
    generator = _generator(rng)
    c, e = _draw(model, n, delta, shape1, shape2, generator)

    # R's runif(n, 0, 0) returns zeros without consuming RNG state, so the
    # port skips the draw when block_max is exactly zero.  The added values
    # would be zeros, so the returned data are unchanged; only the position
    # of a caller-supplied Generator moves in lockstep with R.
    if design == "independent":
        if block_max > 0.0:
            b_ex = generator.uniform(0.0, block_max, n)
            b_cn = generator.uniform(0.0, block_max, n)
            e = e + b_ex
            c = c + b_cn
    else:
        if block_max > 0.0:
            block = generator.uniform(0.0, block_max, n)
            e = e + block
            c = c + block

    pvalue, pr_h1 = _summarise(e, c, design, a0, b0)
    return e, c, pvalue, pr_h1


def dfba_sim_data(
    n: int = 20,
    a0: float = 1.0,
    b0: float = 1.0,
    model: str | None = None,
    design: str | None = None,
    delta: float | None = None,
    shape1: float = 1.0,
    shape2: float = 1.0,
    block_max: float = 0.0,
    rng: Any = None,
) -> SimDataOut:
    """Generate two simulated conditions and compare them.

    ``model`` is one of ``normal``, ``weibull``, ``cauchy``, ``lognormal``,
    ``chisquare``, ``logistic``, ``exponential``, ``gumbel`` or ``pareto``;
    ``design`` is ``paired`` or ``independent``.  ``rng`` is an optional numpy
    ``Generator`` for reproducible draws.  Sampling is not bit-compatible with
    R's RNG; see ``DECISION.md``.
    """
    if is_na(delta):
        raise ValueError("missing value where TRUE/FALSE needed")
    delta_value = _as_number(delta, "delta")
    if delta_value < 0.0:
        raise ValueError(
            "The function requires a positive difference in the location of the two conditions."
        )

    n_value = _check_n(
        n, "The function requires an integer that is 20 or larger for sample size"
    )
    model_value = _check_model(model)
    design_value = _check_design(design)
    block_value = _check_block_max(block_max)
    a0v, b0v = check_positive_finite(
        a0, b0, "Both a0 and b0 must be positive and finite."
    )
    s1, s2 = _check_shapes(model_value, shape1, shape2)

    e, c, pvalue, pr_h1 = _simulate(
        n_value,
        a0v,
        b0v,
        model_value,
        design_value,
        delta_value,
        s1,
        s2,
        block_value,
        rng,
    )
    return SimDataOut(
        pvalue=pvalue,
        prH1=pr_h1,
        E=e,
        C=c,
        design=design_value,
    )


def _check_nsims(samples: Any) -> int:
    if is_na(samples):
        # R reaches rep(NA, round(samples)) and fails here.
        raise ValueError("invalid 'times' argument")
    value = _as_number(samples, "samples")
    if not math.isfinite(value):
        # R fails inside rep(NA, round(samples)) for Inf and NaN.
        raise ValueError("invalid 'times' argument")
    rounded = round(value)
    if rounded < 0:
        # Same R failure: rep(NA, -1).
        raise ValueError("invalid 'times' argument")
    if rounded == 0:
        # R accepts this and returns nsims = 0 with NaN/Inf power values from
        # a pathological 1:0 loop. Reject it deliberately; see DECISION.md.
        raise ValueError("samples must be a positive integer")
    return rounded


def dfba_power_curve(
    n: int = 20,
    a0: float = 1.0,
    b0: float = 1.0,
    delta_step: float = 0.05,
    model: str | None = None,
    design: str | None = None,
    effect_crit: float = 0.95,
    shape1: float = 1.0,
    shape2: float = 1.0,
    block_max: float = 0.0,
    samples: int = 1000,
    hide_progress: bool = False,
    rng: Any = None,
) -> PowerCurveOut:
    """Estimate Bayesian and frequentist power over 21 delta values.

    ``hide_progress`` is accepted for R API compatibility but no progress
    output is produced.  ``rng`` is an optional numpy ``Generator``.
    """
    if is_na(delta_step):
        raise ValueError("missing value where TRUE/FALSE needed")
    step = _as_number(delta_step, "delta_step")
    if step < 0.0:
        raise ValueError("The function requires a nonnegative value for delta.")

    if is_na(n):
        raise ValueError("missing value where TRUE/FALSE needed")
    n_input = _as_number(n, "n")
    if not math.isfinite(n_input):
        raise ValueError("missing value where TRUE/FALSE needed")
    n_value = round(n_input)
    if n_value < 20:
        raise ValueError("The function requires n to be an integer that is 20 or larger")

    model_value = _check_model(model)
    design_value = _check_design(design)
    effect_value = _check_effect_crit(
        effect_crit,
        "The effect_crit value must be a nonzero number less than 1.",
    )
    a0v, b0v = check_positive_finite(
        a0, b0, "Both a0 and b0 must be positive and finite."
    )
    block_value = _check_block_max(block_max)
    s1, s2 = _check_shapes(model_value, shape1, shape2)
    nsims = _check_nsims(samples)

    generator = _generator(rng)
    delta_vec = np.arange(21, dtype=float) * step
    detect_bayes = np.empty(21, dtype=float)
    detect_t = np.empty(21, dtype=float)
    for i, delta in enumerate(delta_vec):
        bayes = np.empty(nsims, dtype=float)
        tpvalue = np.empty(nsims, dtype=float)
        for j in range(nsims):
            _, _, pvalue, pr_h1 = _simulate(
                n_value,
                a0v,
                b0v,
                model_value,
                design_value,
                float(delta),
                s1,
                s2,
                block_value,
                generator,
            )
            bayes[j] = pr_h1
            tpvalue[j] = pvalue
        detect_bayes[i] = float(np.sum(bayes > effect_value)) / nsims
        detect_t[i] = float(np.sum(tpvalue < 1.0 - effect_value)) / nsims

    outputdf = OutputTable(
        {
            "delta_value": delta_vec,
            "Bayes_power": detect_bayes,
            "t_power": detect_t,
        }
    )
    return PowerCurveOut(
        n=float(n_value),
        nsims=nsims,
        model=model_value,
        design=design_value,
        a0=a0v,
        b0=b0v,
        effect_crit=effect_value,
        block_max=block_value,
        outputdf=outputdf,
    )


def dfba_bayes_vs_t_power(
    n_min: int = 20,
    delta: float | None = None,
    model: str | None = None,
    design: str | None = None,
    effect_crit: float = 0.95,
    shape1: float = 1.0,
    shape2: float = 1.0,
    samples: int = 1000,
    a0: float = 1.0,
    b0: float = 1.0,
    block_max: float = 0.0,
    hide_progress: bool = False,
    rng: Any = None,
) -> TPowerOut:
    """Estimate Bayesian and frequentist power over 11 sample sizes.

    Sample sizes run from ``n_min`` to ``n_min + 50`` in steps of five.
    ``hide_progress`` is accepted for R API compatibility but no progress
    output is produced.  ``rng`` is an optional numpy ``Generator``.
    """
    if is_na(delta):
        raise ValueError("missing value where TRUE/FALSE needed")
    delta_value = _as_number(delta, "delta")
    if delta_value < 0.0:
        raise ValueError("The function requires a nonnegative value for delta.")

    n_min_value = _check_n(n_min, "n_min must be an integer that is 20 or larger")
    a0v, b0v = check_positive_finite(
        a0, b0, "Both a0 and b0 must be positive and finite."
    )
    model_value = _check_model(model)
    design_value = _check_design(design)
    effect_value = _check_effect_crit(
        effect_crit,
        "The effect_crit value must be a number nonzero number less than 1.",
    )
    s1, s2 = _check_shapes(model_value, shape1, shape2)
    block_value = _check_block_max(block_max)
    nsims = _check_nsims(samples)

    generator = _generator(rng)
    sample_sizes = np.arange(n_min_value, n_min_value + 51, 5, dtype=float)
    detect_bayes = np.empty(len(sample_sizes), dtype=float)
    detect_t = np.empty(len(sample_sizes), dtype=float)
    for i, n_value in enumerate(sample_sizes):
        bayes = np.empty(nsims, dtype=float)
        tpvalue = np.empty(nsims, dtype=float)
        for j in range(nsims):
            _, _, pvalue, pr_h1 = _simulate(
                int(n_value),
                a0v,
                b0v,
                model_value,
                design_value,
                delta_value,
                s1,
                s2,
                block_value,
                generator,
            )
            bayes[j] = pr_h1
            tpvalue[j] = pvalue
        detect_bayes[i] = float(np.sum(bayes > effect_value)) / nsims
        detect_t[i] = float(np.sum(tpvalue < 1.0 - effect_value)) / nsims

    outputdf = OutputTable(
        {
            "sample_size": sample_sizes,
            "Bayes_power": detect_bayes,
            "t_power": detect_t,
        }
    )
    return TPowerOut(
        nsims=nsims,
        model=model_value,
        design=design_value,
        effect_crit=effect_value,
        deltav=delta_value,
        a0=a0v,
        b0=b0v,
        block_max=block_value,
        outputdf=outputdf,
    )
