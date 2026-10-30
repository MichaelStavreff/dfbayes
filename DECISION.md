# Decision record — `dfbayes`

## Upstream pin

- Package: **DFBA**
- Version: **0.1.0**
- CRAN publication: 2023-12-13
- Tarball: `upstream/DFBA_0.1.0.tar.gz` (vendored read-only)
- Installed from source with `R CMD INSTALL`; R 4.6.1 at `/usr/bin/R`.
- Upstream licence: **GPL-2** (no `LICENSE` file). This port is
  **GPL-2.0-only**.

`testthat` is not installed in the fixture environment; the fixture generator
uses base R plus `jsonlite` and does not require `testthat`. The upstream
`tests/testthat/` suite was read as a behavioural oracle, but our fixtures come
from calling the installed R functions directly.

## Distribution and import name

`dfba` on PyPI is an unrelated dynamic-flux-balance package. This port is
distributed and imported as **`dfbayes`**. `dfbayes` returned 404 on PyPI when
checked on 2026-10-29.

## Target language: pure Python + numpy

Confirmed. No Rust, no PyO3, no compiled extension. The reasons are:

1. Upstream has no compiled code at all (`NeedsCompilation: no`), so there is
   no kernel to translate and no performance gap to close.
2. The compute that does exist is Monte Carlo, which vectorises. Rust would buy
   a constant factor at best, not an order of magnitude.
3. The API returns 18 S4 list-subclass result objects with 15–20 fields each.
   An FFI boundary would marshal a 20-field struct per call for functions whose
   actual compute is microseconds. Net negative.
4. Pure Python ships as a single `py3-none-any` wheel. `mixtime`'s
   `DECISION.md` documents the maturin/abi3/universal2 cost we would otherwise
   repeat.

Note: "no quadrature/optimisation" is true but is *not* the reason Python wins.
The heavy paths are Monte Carlo:

- `dfba_mann_whitney(method="small")`: 200-point omega grid × `samples`
  replications, each drawing `rexp(n_E)` and `rexp(n_C)` and doing an
  `n_E × n_C` comparison loop.
- `dfba_wilcoxon(method="small")`: same structure.
- `dfba_beta_contrast`: `samples` × K individual `rbeta(1, ...)` draws.
- `dfba_power_curve` / `dfba_bayes_vs_t_power`: 21 × `nsims` full simulations.

The R original makes roughly 12 million R-level `rexp()` calls in the
Mann-Whitney small path. The port draws the whole replication batch at once and
uses broadcasting for the comparison matrix; it does not transliterate the
loops line-for-line.

## NA policy

R distinguishes logical `NA` (structural) from `NA_real_` (propagated), and
`is.na()` is TRUE for both `NA` and `NaN`.

- **Inputs:** `None` and `NaN` are both treated as missing, matching `is.na()`.
  Scalar validation mirrors R's `if (...)` branches and raises `ValueError`.
- **Outputs:** all output NA is **`None`**. Genuine `0/0` remains
  `float("nan")`. This preserves the NA-vs-NaN distinction that affects
  arithmetic, but collapses R's logical-vs-double NA distinction; Python has no
  typed NA. `dfbayes.is_na()` treats `None` and `NaN` as missing.
- **Deletion semantics are per function** (see `AGENTS.md`):
  pairwise/listwise drop for `dfba_sign_test`, `dfba_wilcoxon`, and
  `dfba_bivariate_concordance`; independent per-vector drop for
  `dfba_median_test` and `dfba_mann_whitney`.
- `dfba_bivariate_concordance(prob_interval=NA)` is not validated by R; the
  interval fields become NA. The port returns `None` there.
- `dfba_gamma` with an NA cell raises, because R's `rep(..., times=NA)`
  raises `invalid 'times' argument`. The port mirrors that error.

## RNG policy: (a) freeze the narrowest intermediate

R's `rexp()` uses inversion from `unif_rand()`; `rbeta()` uses a different
algorithm again. Bit-exact reproduction is a multi-day subproject and buys
nothing: R's own small-sample output already varies with `samples`, so
cross-language `set.seed()` reproducibility is not a property users rely on.

Policy:

- `dfba_mann_whitney(method="small")` → freeze `fomega` (length-200 counts).
- `dfba_wilcoxon(method="small")` → freeze `fphi` (length-200 counts).
- `dfba_beta_contrast` → freeze `delta` and `delta0`.
- `dfba_power_curve` / `dfba_bayes_vs_t_power` → separate the simulation loop
  from the aggregation. Test aggregation exactly against synthetic
  `bayesprH1`/`tpvalue` arrays; test the simulation path for shape, range, and
  monotonicity. Numeric power values are not fixture-pinned.
- The sampling wrappers use numpy. They are **not** claimed to match R.
  Two tests per stochastic function: (1) analytic-vs-Monte-Carlo empirical
  probability within ~5 standard errors at the null value (`omega=0.5` uses the
  exact Mann-Whitney U distribution; `phi=0.5` uses the Poisson binomial);
  (2) same seed → identical output, different seed → different output.
- `dfba_beta_contrast` uses R's `quantile()` default type 7;
  `numpy.percentile` matches it.

Pinned seeds are recorded with each fixture case in
`tests/fixtures/stochastic.json`:

| case | seed | frozen intermediate |
| --- | ---: | --- |
| `beta_contrast_2` | 20261030 | `delta`, `delta0` (10000 each) |
| `beta_contrast_4` | 20261031 | `delta`, `delta0` (10000 each) |
| `mw_small_branch_a` | 20261032 | `fomega` (200 bins) |
| `mw_small_branch_b` | 20261033 | `fomega` (200 bins) |
| `wil_small` | 20261034 | `fphi` (200 bins) |

The two Mann-Whitney cases sit on opposite sides of R's branch test
`length(E)*length(C) > UE + UC`:

- branch A: `E = C = c(1,1,2,2)` → 16 comparisons, `Ntot = 8`;
- branch B: `E = c(2,4,6)`, `C = c(1,3,5,7,9)` → 15 comparisons,
  `Ntot = 15`.

Branch A counts only the first `Ntot` row-major comparisons; branch B counts
all of them. Both are preserved and replay-tested against the frozen `fomega`.

## Tolerances

`pbeta`/`qbeta` are TOMS 708 in R and Boost in scipy. Agreement is expected
around 1e-12, not bit-identical. Tolerances are derived from the fixtures by
measuring the observed relative/absolute difference, then rounded up to a
documented bound. They are never loosened to make a test pass; a larger
difference is a port bug or a documented divergence.

The suite default is `rtol=1e-11, atol=1e-12`. Observed maxima outside the
Mann-Whitney large branch are below `3e-12` relative, driven by the final
`pbeta`/`qbeta` calls. Two exceptions are documented and handled explicitly in
`tests/_support.py`:

- `mw_large*` `BF10`: `rtol=1e-9`. R's large-branch polynomial is evaluated by
  subtracting O(1e3) terms to leave an O(1e-2) result, and `BF10` is then a
  ratio of two beta tails. The observed maximum relative difference is
  `2.40e-11`, so `1e-9` is a conservative bound, not a license to drift.
- `mc_large` `BF10interval`/`postH1`: the relative difference is `7.4e-8`
  because the value itself is `2.99e-9`, formed by cancellation. The absolute
  difference is about `2.2e-16`, well inside the shared `atol=1e-12`; no
  field-specific `rtol` is needed.

R's `^` on doubles is repeated multiplication, not libm `pow`; the port uses
`r_pow()` where R's one-ulp behaviour is observable (`mann_whitney` polynomial
and `beta` variance).

## Documented divergences and deliberate choices

These are intentional and covered by tests; none changes a deterministic
numeric result.

1. **Progress output.** R's small-sample functions emit `\r...% complete` via
   `message()` unless `hide_progress=TRUE`. The port emits no progress output by
   default and has no `\r` rewriting. This is display-only.
2. **R `message()` → Python `UserWarning`.** Sample-size advice and the
   NA-cell notes become `UserWarning`s, preserving the text. `dfba_beta_contrast`
   raises `ValueError` for `samples < 10000` exactly as R stops; NA `samples`
   raises `missing value where TRUE/FALSE needed`.
3. **`a0_vec`/`b0_vec` recycling in `dfba_beta_contrast`.** R silently recycles
   mismatched lengths through vector arithmetic. The port broadcasts length 1
   and rejects other mismatched lengths with `ValueError` rather than silently
   recycling.
4. **Star concordance duplicate field.** R's `dfba_bivariate_concordance_out`
   with `fitting_parameters` has two `prob_interval` slots. The dataclass has
   one field; `R_FIELDS` records the duplicate so `summary()` can reproduce R's
   field order.
5. **Logical NA.** `dfba_beta_descriptive` returns logical `NA` for an
   undefined mode/HDI. Python has no typed NA, so these are `None`; see the NA
   policy above.
6. **`dfba_sign_test` Bayes factor prior.** Upstream computes the BF with the
   default uniform prior even when the caller sets `a0`/`b0`. The port
   reproduces that bug; the posterior interval still uses the caller's prior.
7. **`dfba_gamma` matrix expansion.** R expands a table/matrix row-major
   (`x` = row, `y` = column) and uses `int(count)` per cell. Python accepts a
   2-D array and indexes it explicitly; column-major storage is not assumed.
8. **`plot()` and text `show()`.** The 15 R `plot()` S4 methods are not
   ported. In their place the package offers an optional, Python-native
   matplotlib layer (`result.plot()`) that returns an `Axes`; it is a
   usability extension, not a reproduction of R's base-graphics output.
   `summary()` reproduces field names/order, not R's exact `sprintf` layout.
9. **Result helpers.** `to_dict()`, `to_json()`, `to_dataframe()`, `bf10`,
   `credible_interval`, `posterior_mean`, `posterior_median`,
   `probability_h1`, `interpret()`, `_repr_html_()`, and
   `results_to_dataframe()` are additions, not R API. `to_json()` maps
   non-finite values to `null` because JSON has no NaN or infinity.
   `to_dataframe()` omits array slots unless `include_arrays=True`.

## Out of scope (Phase 1)

- The 15 R `plot()` S4 methods. The Python-native plotting layer above replaces
  them for Python users but does not reproduce R's base-graphics output.
- Phase 2: `dfba_sim_data`, `dfba_power_curve`, `dfba_bayes_vs_t_power`.
  Their deterministic aggregation is designed in the RNG section above and is
  implemented after Phase 1 is green.
- `summary()`/`show()` byte-for-byte text formatting. Field names and order are
  matched; the R `sprintf`/`cat` layout is not reproduced exactly.
