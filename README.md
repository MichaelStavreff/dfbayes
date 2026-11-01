# dfbayes

[![CI](https://github.com/MichaelStavreff/dfbayes/actions/workflows/ci.yml/badge.svg)](https://github.com/MichaelStavreff/dfbayes/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dfbayes.svg)](https://pypi.org/project/dfbayes/)

Distribution-free Bayesian analysis for Python: a faithful port of the R
package **DFBA** 0.1.0 by Daniel H. Barch and Richard A. Chechile.

`dfbayes` implements Chechile's distribution-free Bayesian procedures:
closed-form posterior intervals and Bayes factors for binomial, sign, median,
Mann-Whitney, Wilcoxon, McNemar, gamma/tau-A concordance, and beta contrasts,
plus the simulation and power tools for planning studies. There is no MCMC, no
quadrature, and no numerical optimisation.

The port keeps the R functions' field names and field order, and reproduces
their numeric results. The PyPI name `dfba` belongs to an unrelated
dynamic-flux-balance package, so this port is distributed as `dfbayes`.

## Install

```sh
pip install dfbayes
# or, with pandas and plotting support:
pip install 'dfbayes[all]'
```

Python 3.10–3.14 are supported. Core dependencies are numpy (>=1.24) and scipy
(>=1.10). pandas and matplotlib are optional:

| Extra | Enables |
| --- | --- |
| `dfbayes[pandas]` | `to_dataframe()` |
| `dfbayes[plot]` | `plot()` |
| `dfbayes[all]` | both |

## Quick start

```python
from dfbayes import dfba_binomial, dfba_mann_whitney, dfba_sign_test

res = dfba_binomial(n1=16, n2=2)
print(res.phimean, res.eti_lower, res.eti_upper)
print(res.posterior_mean)        # phimean
print(res.credible_interval)     # (eti_lower, eti_upper)

sign = dfba_sign_test(
    [1.49, 0.64, 0.96, 2.34, 0.78, 1.29],
    [0.53, 0.71, 0.62, 1.01, 0.55, 0.91],
)
print(sign.bf10)                 # Bayes factor for H1 over H0
print(sign.interpret())          # "BF10 = ...: ... evidence for H1."

mw = dfba_mann_whitney(
    E=[43, 45, 47, 50, 54, 58, 60, 63, 69, 84, 85, 91,
       99, 127, 130, 147, 165, 175, 193, 228, 252, 276],
    C=[0, 1, 2, 3, 5, 14, 15, 23, 23, 25, 27, 32, 57,
       105, 115, 158, 161, 181, 203, 290],
)
print(mw.method, mw.BF10)
```

Result objects are dataclasses whose field names match the R S4 list slots
exactly, in the same order. `summary()` prints the fields in R order; `repr()`
gives a compact view.

## What's included

| Area | Functions |
| --- | --- |
| Proportions | `dfba_binomial`, `dfba_mcnemar` |
| Beta posteriors | `dfba_beta_descriptive`, `dfba_beta_bayes_factor`, `dfba_beta_contrast` |
| Paired and two-sample tests | `dfba_sign_test`, `dfba_median_test`, `dfba_mann_whitney`, `dfba_wilcoxon` |
| Association | `dfba_bivariate_concordance`, `dfba_gamma` |
| Simulation and power | `dfba_sim_data`, `dfba_power_curve`, `dfba_bayes_vs_t_power` |

`dfba_wilcoxon` and `dfba_mann_whitney` choose between the simulated
small-sample posterior and the closed-form large-sample approximation using R's
rule (Wilcoxon switches at `n > 24`, Mann-Whitney at `nh >= 20`, where `nh` is
the harmonic mean of the two group sizes). Pass `method="small"` or
`method="large"` to override. The small-sample path is Monte Carlo, so pass
`rng=` for reproducible output.

### Simulation and power

```python
import numpy as np
from dfbayes import dfba_sim_data, dfba_power_curve, dfba_bayes_vs_t_power

sim = dfba_sim_data(
    n=50,
    model="normal",
    design="paired",
    delta=0.4,
    shape1=1,
    shape2=4,
    rng=np.random.default_rng(2026),
)
print(sim.pvalue, sim.prH1)

curve = dfba_power_curve(
    n=85,
    model="normal",
    design="independent",
    samples=250,
    rng=np.random.default_rng(2026),
)
print(curve.outputdf.to_dataframe())

by_n = dfba_bayes_vs_t_power(
    n_min=40,
    delta=0.45,
    model="normal",
    design="paired",
    samples=250,
    rng=np.random.default_rng(2026),
)
by_n.plot()
```

The simulation models are `normal`, `weibull`, `cauchy`, `lognormal`,
`chisquare`, `logistic`, `exponential`, `gumbel`, and `pareto`. Designs are
`paired` and `independent`. Sampling uses numpy's `Generator`, so it is not
bit-compatible with R's `set.seed()`; pass `rng=` for reproducible runs.

### Working with results

```python
res.to_dict()        # plain Python dict; arrays become lists
res.to_json()        # strict JSON; NaN/Inf become null
res.to_dataframe()   # one-row pandas DataFrame (scalar slots)
res.to_dataframe(include_arrays=True)
res._repr_html_()    # compact HTML table in Jupyter
```

`to_dataframe()` omits array-valued slots by default and includes them as
object columns when `include_arrays=True`. `BetaOutputDF` (the density table
returned by `dfba_beta_descriptive`) also has `to_dict()` and `to_dataframe()`.

Batch workflows stack results into a single frame:

```python
from dfbayes import dfba_mcnemar, results_to_dataframe

batch = [dfba_mcnemar(n_01=n, n_10=20 - n) for n in range(1, 20)]
frame = results_to_dataframe(batch)
frame[frame["BF10point"] > 3]
```

`results_to_dataframe()` adds a `result_class` column and unions heterogeneous
fields. Use `include_arrays=True` to keep array slots as object columns, or
`include_class=False` to omit the class column. pandas Series are accepted
anywhere a numeric vector is expected:

```python
import pandas as pd
from dfbayes import dfba_sign_test

dfba_sign_test(pd.Series(y1), pd.Series(y2))
```

### Plotting

```python
import matplotlib.pyplot as plt

ax = res.plot()                 # returns the matplotlib Axes
ax = mw.plot(ax=ax)             # draw on an existing axes
plt.savefig("posterior.png")
```

`plot()` chooses a sensible view per result: beta densities with ETI/HDI bands,
discrete omega/phi posteriors, contrast quantiles, Bayes-factor null regions,
simulated-data boxplots, or Bayesian-vs-frequentist power curves.

## Differences from the R package

These are deliberate and documented rather than accidental:

- R's S4 `plot()` methods are not ported. `dfbayes` ships an optional,
  Python-native matplotlib layer instead.
- Random draws use numpy, not R's Mersenne-Twister, so `set.seed()` has no
  equivalent. Pass `rng=` for reproducibility.
- R's `message()` output becomes a Python `UserWarning`, and the interactive
  progress counters are not printed (`hide_progress` is accepted for
  compatibility).
- `samples` must round to at least 1 and `effect_crit` must be non-zero.
  Upstream accepts both and returns degenerate output.
- The star concordance class has a duplicated `prob_interval` slot upstream.
  The Python dataclass exposes one field, and `summary()` still lists it twice,
  in R order.

[`DECISION.md`](https://github.com/MichaelStavreff/dfbayes/blob/main/DECISION.md)
records every divergence with its rationale, and
[`COVERAGE.md`](https://github.com/MichaelStavreff/dfbayes/blob/main/COVERAGE.md)
maps each R source branch to the test that pins it.

## Contributing

Development setup, the fixture policy, and the release process are in
[`CONTRIBUTING.md`](https://github.com/MichaelStavreff/dfbayes/blob/main/CONTRIBUTING.md).
Bug reports and pull requests are welcome; a change is correct when it matches
R's DFBA 0.1.0, not when it looks idiomatic on its own.

## Attribution and licence

`dfbayes` is a derivative work of the CRAN package **DFBA 0.1.0**
(Distribution-Free Bayesian Analysis):

- **Authors:** Daniel H. Barch and Richard A. Chechile, Tufts University
- **Upstream source:** <https://CRAN.R-project.org/package=DFBA>
- **CRAN publication:** 2023-12-13
- **Upstream licence:** GPL-2

The upstream copyright notice and authorship are preserved in `NOTICE` and
`AUTHORS`, as required by GPL-2 section 2(a). This port is licensed
**GPL-2.0-only**, matching upstream. A copy of the licence is in `LICENSE`.

The upstream R sources are vendored read-only under `upstream/` so the fixtures
can be regenerated; they are not part of the installed Python package.
