# dfbayes

A faithful Python port of the R package **DFBA** (Distribution-Free Bayesian
Analysis) by Daniel H. Barch and Richard A. Chechile, version 0.1.0.

`dfbayes` implements Chechile's distribution-free Bayesian procedures:
closed-form posterior intervals and Bayes factors for binomial, sign, median,
Mann-Whitney, Wilcoxon, McNemar, gamma/tau-A concordance, and beta contrasts.
There is no MCMC, no quadrature, and no numerical optimisation.

The PyPI name `dfba` belongs to an unrelated dynamic-flux-balance package, so
this port is distributed as `dfbayes`.

## Status

Phase 1 is the deterministic closed-form core plus the `method="large"`
approximations. Phase 2 adds the simulation and power tools
(`dfba_sim_data`, `dfba_power_curve`, `dfba_bayes_vs_t_power`). The stochastic
small-sample paths are ported with vectorised numpy sampling and tested against
frozen R intermediates; see `DECISION.md`.

R's S4 `plot()` methods are not ported. `dfbayes` instead ships an optional,
Python-native matplotlib layer described below. See `COVERAGE.md` for the
R-branch-to-test map.

## Install

```sh
uv sync
# or, for pandas and plotting support:
pip install 'dfbayes[all]'
```

Core dependencies are numpy and scipy. pandas and matplotlib are optional:

| Extra | Enables |
| --- | --- |
| `dfbayes[pandas]` | `to_dataframe()` |
| `dfbayes[plot]` | `plot()` |
| `dfbayes[all]` | both |

## Usage

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

### Interchange and notebooks

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

### Batch workflows

```python
from dfbayes import dfba_mcnemar, results_to_dataframe

batch = [dfba_mcnemar(n_01=n, n_10=20 - n) for n in range(1, 20)]
frame = results_to_dataframe(batch)
frame[frame["BF10point"] > 3]
```

`results_to_dataframe()` stacks one row per result, adds a `result_class`
column, and unions heterogeneous fields. Use `include_arrays=True` to keep
array slots as object columns, or `include_class=False` to omit the class
column. pandas Series are accepted anywhere a numeric vector is expected:

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


## Development

```sh
just build      # uv sync
just test       # uv run pytest
just fixtures   # regenerate golden JSON by running R's DFBA 0.1.0
```

Fixtures are generated by running the R original. Never hand-write expected
values. If the port and R disagree, the port is wrong.

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

Upstream R sources are vendored read-only under `upstream/` for fixture
generation and are not part of the installed Python package.
