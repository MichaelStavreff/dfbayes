"""Tests for the Pythonic result helpers in :mod:`dfbayes.results`."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import dfbayes as d
from dfbayes.results import PointBFOut

from .cases import (
    BC_X,
    BC_Y,
    GAMMA_MAT,
    GROUP1,
    GROUP2,
    GROUP_A,
    GROUP_B,
    GROUP_C,
    GROUP_D,
    MEASURE_1,
    MEASURE_2,
    WIL_A,
    WIL_B,
    WIL_C,
    WIL_E,
)


def _one_of_each() -> dict[str, object]:
    """Build one result object for every closed-form result class."""
    rng = np.random.default_rng(20240601)
    return {
        "dfba_binomial_out": d.dfba_binomial(16, 2),
        "dfba_sign_test_out": d.dfba_sign_test(MEASURE_1, MEASURE_2),
        "dfba_median_test_out": d.dfba_median_test(GROUP1, GROUP2),
        "dfba_beta_descriptive_out": d.dfba_beta_descriptive(38, 55),
        "dfba_point_BF_out": d.dfba_beta_bayes_factor(377, 123, "point", 0.75),
        "dfba_interval_BF_out": d.dfba_beta_bayes_factor(
            17, 5, "interval", [0, 0.5]
        ),
        "dfba_beta_contrast_out": d.dfba_beta_contrast(
            [22, 15], [18, 25], [1, -1], rng=rng
        ),
        "dfba_gamma_out": d.dfba_gamma(GAMMA_MAT),
        "dfba_mann_whitney_small_out": d.dfba_mann_whitney(
            E=GROUP_C, C=GROUP_D, method="small", rng=rng
        ),
        "dfba_mann_whitney_large_out": d.dfba_mann_whitney(E=GROUP_A, C=GROUP_B),
        "dfba_wilcoxon_small_out": d.dfba_wilcoxon(
            Y1=WIL_A, Y2=WIL_B, method="small", rng=rng
        ),
        "dfba_wilcoxon_large_out": d.dfba_wilcoxon(Y1=WIL_E, Y2=WIL_C),
        "dfba_mcnemar_out": d.dfba_mcnemar(17, 2),
        "dfba_bivariate_concordance_out": d.dfba_bivariate_concordance(BC_X, BC_Y),
        "dfba_bivariate_concordance_star_out": d.dfba_bivariate_concordance(
            BC_X, BC_Y, fitting_parameters=3
        ),
    }


RESULTS = _one_of_each()


def test_every_result_class_is_covered():
    # BetaOutputDF is a nested helper, not a top-level S4 result class.
    assert len(RESULTS) == 15
    assert {type(result).__name__ for result in RESULTS.values()} == {
        "BinomialOut",
        "SignTestOut",
        "MedianTestOut",
        "BetaDescriptiveOut",
        "PointBFOut",
        "IntervalBFOut",
        "BetaContrastOut",
        "GammaOut",
        "MannWhitneySmallOut",
        "MannWhitneyLargeOut",
        "WilcoxonSmallOut",
        "WilcoxonLargeOut",
        "McnemarOut",
        "BivariateConcordanceOut",
        "BivariateConcordanceStarOut",
    }


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_to_dict_matches_r_fields(name):
    result = RESULTS[name]
    payload = result.to_dict()
    assert list(payload) == list(dict.fromkeys(result.R_FIELDS))
    for field in payload:
        value = getattr(result, field)
        if isinstance(value, np.ndarray):
            assert payload[field] == value.tolist()
        elif hasattr(value, "to_dict"):
            assert payload[field] == value.to_dict()
        else:
            assert payload[field] == value


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_to_json_is_strict_and_round_trips(name):
    payload = RESULTS[name].to_json()
    assert json.loads(payload) == json.loads(
        json.dumps(_json_ready(RESULTS[name].to_dict()))
    )


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def test_to_json_maps_non_finite_to_null():
    result = PointBFOut(
        method="point",
        a_post=1.0,
        b_post=1.0,
        a0=1.0,
        b0=1.0,
        BF10=float("nan"),
        BF01=float("inf"),
        null_hypothesis=0.5,
        dpriorH0=1.0,
        dpostH0=1.0,
    )
    payload = json.loads(result.to_json())
    assert payload["BF10"] is None
    assert payload["BF01"] is None


def test_to_dict_preserves_non_finite_values():
    result = PointBFOut(
        method="point",
        a_post=1.0,
        b_post=1.0,
        a0=1.0,
        b0=1.0,
        BF10=float("nan"),
        BF01=float("inf"),
        null_hypothesis=0.5,
        dpriorH0=1.0,
        dpostH0=1.0,
    )
    assert math.isnan(result.to_dict()["BF10"])
    assert math.isinf(result.to_dict()["BF01"])


def test_to_dataframe_is_one_row_without_arrays():
    pd = pytest.importorskip("pandas")
    result = RESULTS["dfba_sign_test_out"]
    frame = result.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert "Y1" not in frame.columns
    assert "BF10" in frame.columns
    assert frame.loc[0, "BF10"] == pytest.approx(result.BF10)


def test_to_dataframe_can_include_arrays():
    pd = pytest.importorskip("pandas")
    result = RESULTS["dfba_sign_test_out"]
    frame = result.to_dataframe(include_arrays=True)
    assert isinstance(frame, pd.DataFrame)
    assert frame.loc[0, "Y1"] == result.Y1.tolist()


def test_beta_outputdf_to_dict_and_dataframe():
    pd = pytest.importorskip("pandas")
    output = RESULTS["dfba_beta_descriptive_out"].outputdf
    payload = output.to_dict()
    assert set(payload) == set(output.columns)
    assert {"x", "density", "cumulative_prob"} <= set(payload)
    assert isinstance(payload["x"], list)
    frame = output.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == list(output.columns)
    assert len(frame) == len(output)


def test_summary_and_str_list_fields_in_r_order():
    result = RESULTS["dfba_bivariate_concordance_star_out"]
    text = str(result)
    lines = text.splitlines()
    assert lines[0] == result.R_CLASS
    reported = [line.split(":", 1)[0].strip() for line in lines[1:]]
    assert reported == list(result.R_FIELDS)
    assert text.count("prob_interval") == 2


def test_repr_is_compact():
    text = repr(RESULTS["dfba_binomial_out"])
    assert text.startswith("BinomialOut(")
    assert text.endswith(", ...)")


def test_html_repr_escapes_and_collapses_duplicate_fields():
    result = RESULTS["dfba_bivariate_concordance_star_out"]
    markup = result._repr_html_()
    assert result.R_CLASS in markup
    assert markup.count(">prob_interval<") == 1
    assert "<table" in markup


def test_bf10_uses_point_estimate_for_mcnemar():
    result = RESULTS["dfba_mcnemar_out"]
    assert result.bf10 == pytest.approx(result.BF10point)
    assert result.bf10 != pytest.approx(result.BF10interval)


def test_bf10_is_none_when_there_is_no_single_factor():
    assert RESULTS["dfba_median_test_out"].bf10 is None
    assert RESULTS["dfba_beta_descriptive_out"].bf10 is None


def test_bf10_alias_resolution_order():
    result = PointBFOut(
        method="point",
        a_post=1.0,
        b_post=1.0,
        a0=1.0,
        b0=1.0,
        BF10=4.0,
        BF01=0.25,
        null_hypothesis=0.5,
        dpriorH0=1.0,
        dpostH0=1.0,
    )
    assert result.bf10 == 4.0


def test_credible_interval_prefers_eti_then_hdi():
    binomial = RESULTS["dfba_binomial_out"]
    assert binomial.credible_interval == pytest.approx(
        (binomial.eti_lower, binomial.eti_upper)
    )
    wilcoxon = RESULTS["dfba_wilcoxon_small_out"]
    assert wilcoxon.credible_interval == pytest.approx(
        (wilcoxon.hdi_lower, wilcoxon.hdi_upper)
    )


def test_posterior_mean_covers_x_mean():
    descriptive = RESULTS["dfba_beta_descriptive_out"]
    assert descriptive.posterior_mean == pytest.approx(descriptive.x_mean)
    binomial = RESULTS["dfba_binomial_out"]
    assert binomial.posterior_mean == pytest.approx(binomial.phimean)


def test_posterior_median_covers_x_median():
    descriptive = RESULTS["dfba_beta_descriptive_out"]
    assert descriptive.posterior_median == pytest.approx(descriptive.x_median)


def test_probability_h1_aliases():
    sign = RESULTS["dfba_sign_test_out"]
    assert sign.probability_h1 == pytest.approx(sign.post_H1)
    interval = RESULTS["dfba_interval_BF_out"]
    assert interval.probability_h1 == pytest.approx(interval.postH1)


def test_interpret_labels_evidence_for_h1():
    result = PointBFOut(
        method="point",
        a_post=1.0,
        b_post=1.0,
        a0=1.0,
        b0=1.0,
        BF10=5.0,
        BF01=0.2,
        null_hypothesis=0.5,
        dpriorH0=1.0,
        dpostH0=1.0,
    )
    assert "moderate evidence for H1" in result.interpret()


def test_interpret_reports_bf01_for_h0():
    result = PointBFOut(
        method="point",
        a_post=1.0,
        b_post=1.0,
        a0=1.0,
        b0=1.0,
        BF10=0.5,
        BF01=2.0,
        null_hypothesis=0.5,
        dpriorH0=1.0,
        dpostH0=1.0,
    )
    text = result.interpret()
    assert "BF01 = 2" in text
    assert "anecdotal evidence for H0" in text


def test_interpret_handles_nan_and_infinity():
    nan_result = PointBFOut(
        method="point",
        a_post=1.0,
        b_post=1.0,
        a0=1.0,
        b0=1.0,
        BF10=float("nan"),
        BF01=float("nan"),
        null_hypothesis=0.5,
        dpriorH0=1.0,
        dpostH0=1.0,
    )
    assert "undefined" in nan_result.interpret()

    inf_result = PointBFOut(
        method="point",
        a_post=1.0,
        b_post=1.0,
        a0=1.0,
        b0=1.0,
        BF10=float("inf"),
        BF01=0.0,
        null_hypothesis=0.5,
        dpriorH0=1.0,
        dpostH0=1.0,
    )
    assert "extreme evidence for H1" in inf_result.interpret()


def test_interpret_without_bf10():
    text = RESULTS["dfba_median_test_out"].interpret()
    assert "no single Bayes factor" in text


def test_results_to_dataframe_batches_rows():
    pd = pytest.importorskip("pandas")
    results = [
        d.dfba_binomial(16, 2),
        d.dfba_binomial(5, 5),
    ]
    frame = d.results_to_dataframe(results)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame["result_class"]) == ["dfba_binomial_out"] * 2
    assert list(frame["n1"]) == [16.0, 5.0]
    assert frame.index.tolist() == [0, 1]


def test_results_to_dataframe_unions_heterogeneous_columns():
    pd = pytest.importorskip("pandas")
    frame = d.results_to_dataframe(
        [d.dfba_binomial(16, 2), d.dfba_mcnemar(17, 2)]
    )
    assert isinstance(frame, pd.DataFrame)
    assert "n1" in frame.columns
    assert "n_10" in frame.columns
    assert pd.isna(frame.loc[0, "n_10"])
    assert pd.isna(frame.loc[1, "n1"])


def test_results_to_dataframe_can_include_arrays_and_skip_class():
    pytest.importorskip("pandas")
    frame = d.results_to_dataframe(
        [d.dfba_sign_test(MEASURE_1, MEASURE_2)],
        include_arrays=True,
        include_class=False,
    )
    assert "result_class" not in frame.columns
    assert frame.loc[0, "Y1"] == RESULTS["dfba_sign_test_out"].Y1.tolist()


def test_results_to_dataframe_empty_input():
    pd = pytest.importorskip("pandas")
    frame = d.results_to_dataframe([])
    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
    assert list(frame.columns) == ["result_class"]


def test_results_to_dataframe_rejects_non_results():
    pytest.importorskip("pandas")
    with pytest.raises(TypeError, match="DFBAResult"):
        d.results_to_dataframe([object()])


def test_pandas_series_are_accepted_as_vectors():
    pd = pytest.importorskip("pandas")
    result = d.dfba_sign_test(
        pd.Series(MEASURE_1), pd.Series(MEASURE_2)
    )
    expected = d.dfba_sign_test(MEASURE_1, MEASURE_2)
    assert result.BF10 == pytest.approx(expected.BF10)
    assert result.n_pos == expected.n_pos
    assert result.n_neg == expected.n_neg


def test_pandas_series_nan_is_treated_as_missing():
    pd = pytest.importorskip("pandas")
    y1 = pd.Series([1.0, 2.0, None, 4.0])
    y2 = pd.Series([0.0, 3.0, 1.0, 5.0])
    result = d.dfba_sign_test(y1, y2)
    expected = d.dfba_sign_test([1.0, 2.0, None, 4.0], [0.0, 3.0, 1.0, 5.0])
    assert result.BF10 == pytest.approx(expected.BF10)
    assert result.n_pos == expected.n_pos
