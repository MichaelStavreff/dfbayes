"""Distribution-free Bayesian procedures, ported from R's DFBA 0.1.0.

The closed-form core covers eleven analysis functions; the simulation helpers
``dfba_sim_data``, ``dfba_power_curve`` and ``dfba_bayes_vs_t_power`` cover
study planning.
"""

from __future__ import annotations

from .beta import dfba_beta_bayes_factor, dfba_beta_descriptive
from .binomial import dfba_binomial
from .concordance import dfba_bivariate_concordance, dfba_gamma
from .contrast import dfba_beta_contrast
from .mann_whitney import dfba_mann_whitney
from .mcnemar import dfba_mcnemar
from .median_test import dfba_median_test
from .results import (
    BetaContrastOut,
    BetaDescriptiveOut,
    BetaOutputDF,
    BinomialOut,
    BivariateConcordanceOut,
    BivariateConcordanceStarOut,
    DFBAResult,
    GammaOut,
    IntervalBFOut,
    MannWhitneyLargeOut,
    MannWhitneySmallOut,
    McnemarOut,
    MedianTestOut,
    OutputTable,
    PointBFOut,
    PowerCurveOut,
    SignTestOut,
    SimDataOut,
    TPowerOut,
    WilcoxonLargeOut,
    WilcoxonSmallOut,
    results_to_dataframe,
)
from .sign_test import dfba_sign_test
from .simulation import (
    dfba_bayes_vs_t_power,
    dfba_power_curve,
    dfba_sim_data,
)
from .wilcoxon import dfba_wilcoxon

__version__ = "0.1.1"

__all__ = [  # noqa: RUF022 - grouped by kind for readability
    "__version__",
    # functions
    "dfba_beta_bayes_factor",
    "dfba_beta_contrast",
    "dfba_beta_descriptive",
    "dfba_binomial",
    "dfba_bivariate_concordance",
    "dfba_gamma",
    "dfba_mann_whitney",
    "dfba_mcnemar",
    "dfba_median_test",
    "dfba_sign_test",
    "dfba_wilcoxon",
    # simulation and power
    "dfba_sim_data",
    "dfba_power_curve",
    "dfba_bayes_vs_t_power",
    # result objects
    "DFBAResult",
    "BetaContrastOut",
    "BetaDescriptiveOut",
    "BetaOutputDF",
    "BinomialOut",
    "BivariateConcordanceOut",
    "BivariateConcordanceStarOut",
    "GammaOut",
    "IntervalBFOut",
    "MannWhitneyLargeOut",
    "MannWhitneySmallOut",
    "McnemarOut",
    "MedianTestOut",
    "OutputTable",
    "PointBFOut",
    "PowerCurveOut",
    "SignTestOut",
    "SimDataOut",
    "TPowerOut",
    "WilcoxonLargeOut",
    "WilcoxonSmallOut",
    # batch helpers
    "results_to_dataframe",
]
