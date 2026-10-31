"""Shared numerical and validation helpers.

The port deliberately mirrors R's scalar behaviour, including division by zero
(``Inf``/``-Inf``/``NaN``) and the fact that ``is.na(NaN)`` is TRUE.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import betainc, betaincinv, betaln

# ---------------------------------------------------------------------------
# NA / NaN
# ---------------------------------------------------------------------------

def is_na(value: Any) -> bool:
    """R's ``is.na``: TRUE for both NA (None) and NaN."""
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def as_float_vector(values: Any, name: str = "values") -> np.ndarray:
    """Convert a sequence to a 1-D float array, mapping None to NaN.

    This is the input-side equivalent of R's numeric vector coercion; ``None``
    and ``NaN`` both become ``NaN`` and are treated as missing.
    """
    if values is None:
        arr = np.array([np.nan], dtype=float)
    elif isinstance(values, np.ndarray):
        arr = values.astype(float, copy=False)
        if arr.ndim == 0:
            arr = arr.reshape(1)
    else:
        try:
            arr = np.asarray(list(values), dtype=float)
        except TypeError:
            # R treats a scalar as a length-1 vector.
            arr = np.array([float(values)], dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    return arr


def as_scalar(value: Any, name: str) -> float:
    """Coerce a scalar argument to float, mapping None to NaN."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a single numeric value") from exc


def rdiv(numerator: float, denominator: float) -> float:
    """Division with R's semantics: 0/0 -> NaN, x/0 -> +/-Inf."""
    num = float(numerator)
    den = float(denominator)
    if math.isnan(num) or math.isnan(den):
        return float("nan")
    if den == 0.0:
        if num == 0.0:
            return float("nan")
        return math.copysign(math.inf, num)
    return num / den


def r_pow(x: float, y: float) -> float:
    """R's ``^`` for the cases used upstream.

    R evaluates a non-negative integer power by repeated multiplication
    (``x^4`` is ``((x*x)*x)*x``), not libm ``pow``.  The two differ by one ulp
    often enough to matter: the Mann-Whitney polynomial subtracts terms of
    order 1e3 to obtain a result of order 1e-2, so a one-ulp change in ``x^4``
    is amplified to ~1e-12 in the output.
    """
    xf = float(x)
    yf = float(y)
    if yf >= 0.0 and yf.is_integer():
        result = 1.0
        for _ in range(int(yf)):
            result *= xf
        return result
    return xf ** yf


def rseq(start: float, stop: float, by: float, *, n: int | None = None) -> np.ndarray:
    """Approximate R's ``seq`` for the fixed grids used upstream.

    Upstream only uses ``seq`` with a known number of points (201, 1001, 200,
    101), so callers pass ``n`` and we use ``linspace``.  The endpoint is
    included, matching R.
    """
    if n is None:
        raise TypeError("rseq requires n")
    return np.linspace(start, stop, n)


# ---------------------------------------------------------------------------
# Beta distribution functions (R's pbeta/qbeta/dbeta equivalents)
# ---------------------------------------------------------------------------

def pbeta(q: float, a: float, b: float) -> float:
    if is_na(q) or is_na(a) or is_na(b):
        return float("nan")
    return float(betainc(a, b, q))


def pbeta_vec(q: np.ndarray, a: float, b: float) -> np.ndarray:
    return betainc(a, b, np.asarray(q, dtype=float))


def qbeta(p: float, a: float, b: float) -> float:
    if is_na(p) or is_na(a) or is_na(b):
        return float("nan")
    return float(betaincinv(a, b, p))


def qbeta_vec(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return betaincinv(a, b, np.asarray(p, dtype=float))


def dbeta(x: float, a: float, b: float) -> float:
    """Beta density, including R's boundary conventions."""
    if is_na(x) or is_na(a) or is_na(b):
        return float("nan")
    x = float(x)
    if x < 0.0 or x > 1.0:
        return 0.0
    if x == 0.0:
        if a < 1.0:
            return math.inf
        if a == 1.0:
            return float(b)
        return 0.0
    if x == 1.0:
        if b < 1.0:
            return math.inf
        if b == 1.0:
            return float(a)
        return 0.0
    log_density = (a - 1.0) * math.log(x) + (b - 1.0) * math.log1p(-x) - float(betaln(a, b))
    return math.exp(log_density)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def check_positive_finite(a: Any, b: Any, message: str) -> tuple[float, float]:
    av = as_scalar(a, "a")
    bv = as_scalar(b, "b")
    for value in (av, bv):
        if math.isnan(value) or math.isinf(value) or value <= 0.0:
            raise ValueError(message)
    return av, bv


def check_prob_interval(value: Any, *, allow_zero_one: bool, message: str) -> float:
    p = as_scalar(value, "prob_interval")
    if math.isnan(p):
        # R reaches `if (NA)` and raises a base error, not the custom stop().
        raise ValueError("missing value where TRUE/FALSE needed")
    if allow_zero_one:
        if p < 0.0 or p > 1.0:
            raise ValueError(message)
    else:
        if p <= 0.0 or p >= 1.0:
            raise ValueError(message)
    return p


def check_nonnegative_count(
    value: Any,
    name: str,
    negative_message: str,
    integer_message: str,
) -> int | float:
    """Validate a count in R's order: negativity/NA first, integrality second.

    Preserves whether the caller supplied an int or a float, matching R's
    observable ``integer`` vs ``double`` distinction for inputs.
    """
    if value is None:
        raise ValueError(negative_message)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(negative_message) from exc
    if math.isnan(numeric) or numeric < 0.0:
        raise ValueError(negative_message)
    if numeric != round(numeric):
        raise ValueError(integer_message)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return numeric


def result_fields(obj: Any) -> list[tuple[str, Any]]:
    """Return (name, value) pairs in R slot order."""
    return [(name, getattr(obj, name)) for name in obj.R_FIELDS]
