"""Shared fixture loading and comparison helpers for the test suite."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def decode(value: Any) -> Any:
    """Decode the tagged JSON produced by ``tests/fixtures/generate.R``."""
    if isinstance(value, dict):
        if "__na__" in value:
            return None
        if "__nan__" in value:
            return float("nan")
        if "__inf__" in value:
            return math.inf if value["__inf__"] > 0 else -math.inf
        if "__df__" in value:
            frame = value["__df__"]
            return {
                name: np.array([decode(v) for v in column], dtype=float)
                for name, column in zip(frame["names"], frame["columns"])
            }
        if "__matrix__" in value:
            matrix = value["__matrix__"]
            return np.array(
                [decode(v) for v in matrix["data"]], dtype=float
            ).reshape(matrix["dim"], order="F")
        if "__list__" in value:
            return {name: decode(item) for name, item in value["__list__"]}
        raise ValueError(f"unknown fixture tag: {sorted(value)}")
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def load_json(name: str) -> dict:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_fixtures() -> dict:
    return load_json("fixtures.json")["cases"]


def load_inputs() -> dict:
    return {name: decode(value) for name, value in load_json("inputs.json").items()}


def load_stochastic() -> dict:
    path = FIXTURE_DIR / "stochastic.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["cases"]


def tolerance(case_name: str, field_name: str) -> tuple[float, float]:
    """Return ``(rtol, atol)`` for one compared value.

    The default is derived from the deterministic fixtures: every value outside
    the Mann-Whitney large branch agrees to better than 1e-11 relative.  The
    large branch evaluates a Lagrange polynomial by subtracting O(1e3) terms to
    get an O(1e-2) result, then forms ``BF10`` as a ratio of beta tail
    probabilities.  The observed maximum there is 2.4e-11 relative, so that
    field gets 1e-9.  The absolute floor keeps near-zero quantities (for example
    a Bayes factor around 3e-9 formed by cancellation) meaningful.
    """
    if case_name.startswith("mw_large") and field_name == "BF10":
        return 1e-9, 1e-12
    return 1e-11, 1e-12


def assert_scalar_close(actual: Any, expected: Any, *, rtol: float, atol: float) -> None:
    if expected is None:
        assert actual is None, f"expected NA (None), got {actual!r}"
        return
    if isinstance(expected, str):
        assert actual == expected, f"expected {expected!r}, got {actual!r}"
        return
    if isinstance(expected, bool):
        assert actual is expected
        return

    expected_float = float(expected)
    assert actual is not None, f"expected {expected!r}, got None"
    actual_float = float(actual)

    if math.isnan(expected_float):
        assert math.isnan(actual_float), f"expected NaN, got {actual!r}"
    elif math.isinf(expected_float):
        assert actual_float == expected_float, f"expected {expected!r}, got {actual!r}"
    else:
        difference = abs(actual_float - expected_float)
        assert difference <= atol + rtol * abs(expected_float), (
            f"expected {expected_float!r}, got {actual_float!r} "
            f"(abs diff {difference!r}, rtol {rtol!r}, atol {atol!r})"
        )


def assert_array_close(actual: Any, expected: np.ndarray, *, rtol: float, atol: float) -> None:
    actual_array = np.asarray(actual, dtype=float)
    assert actual_array.shape == expected.shape, (
        f"shape mismatch: expected {expected.shape}, got {actual_array.shape}"
    )
    if expected.size == 0:
        return
    np.testing.assert_allclose(
        actual_array,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )


def assert_outputdf_close(actual: Any, expected: dict, *, rtol: float, atol: float) -> None:
    assert tuple(actual.columns) == tuple(expected), (
        f"data-frame columns: expected {tuple(expected)}, got {actual.columns}"
    )
    for column, values in expected.items():
        assert_array_close(actual[column], values, rtol=rtol, atol=atol)


def assert_result_matches(case_name: str, result: Any, fixture: dict) -> None:
    """Compare a result dataclass against one golden fixture case."""
    assert result.R_CLASS == fixture["class"], (
        f"R class: expected {fixture['class']!r}, got {result.R_CLASS!r}"
    )

    expected_fields = fixture["fields"]
    expected_names = tuple(name for name, _ in expected_fields)
    assert result.R_FIELDS == expected_names, (
        f"field order: expected {expected_names}, got {result.R_FIELDS}"
    )

    for field_name, encoded in expected_fields:
        expected = decode(encoded)
        if isinstance(expected, list):
            expected = np.asarray(expected, dtype=float)
        actual = getattr(result, field_name)
        rtol, atol = tolerance(case_name, field_name)

        if isinstance(expected, dict):  # outputdf
            assert_outputdf_close(actual, expected, rtol=rtol, atol=atol)
        elif isinstance(expected, np.ndarray):
            assert_array_close(actual, expected, rtol=rtol, atol=atol)
        else:
            assert_scalar_close(actual, expected, rtol=rtol, atol=atol)


def expected_warnings(fixture: dict) -> list[str]:
    """R warnings and messages that the port maps to deliberate Python warnings.

    Progress messages contain a carriage return and are opt-in in the port, so
    they are excluded here.
    """
    messages: list[str] = []
    for key in ("warnings", "messages"):
        for message in fixture.get(key) or []:
            stripped = message.strip()
            if "% complete" in stripped:
                continue
            messages.append(stripped)
    return messages
