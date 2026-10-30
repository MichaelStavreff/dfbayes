"""Golden-fixture tests for the deterministic Phase 1 functions."""

from __future__ import annotations

import warnings

import pytest

from ._support import assert_result_matches, expected_warnings, load_fixtures
from .cases import CASES

FIXTURES = load_fixtures()


def test_every_fixture_has_a_python_call() -> None:
    assert set(FIXTURES) == set(CASES)


@pytest.mark.parametrize("case_name", sorted(FIXTURES))
def test_deterministic_case(case_name: str) -> None:
    fixture = FIXTURES[case_name]
    call = CASES[case_name]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "error" in fixture:
            with pytest.raises(ValueError) as excinfo:
                call()
            assert str(excinfo.value) == fixture["error"], (
                f"{case_name}: expected R error {fixture['error']!r}, "
                f"got {str(excinfo.value)!r}"
            )
        else:
            assert_result_matches(case_name, call(), fixture)

    got_warnings = [
        str(item.message).strip()
        for item in caught
        if issubclass(item.category, UserWarning)
    ]
    assert got_warnings == expected_warnings(fixture), (
        f"{case_name}: expected warnings {expected_warnings(fixture)!r}, "
        f"got {got_warnings!r}"
    )
