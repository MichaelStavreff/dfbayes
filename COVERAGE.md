# Coverage map — R source branch to test

Fixture case names refer to `tests/fixtures/fixtures.json`; stochastic case
names refer to `tests/fixtures/stochastic.json`. The deterministic suite runs
every fixture in `tests/test_deterministic.py`; the stochastic suite is
`tests/test_stochastic.py`; Phase 2 model/power behaviour is in
`tests/test_simulation.py`.

`upstream/` is read-only. Expected values are produced by running R DFBA 0.1.0,
never by hand.

## `dfba_binomial` (`dfba_binomial.R`)

| R branch | Tests |
| --- | --- |
| default uniform prior | `binomial_default`, `binomial_zero_zero` |
| non-default prior (`a0`, `b0`) | `binomial_prior_2_3`, `binomial_jeffreys_99` |
| `prob_interval` 0.5 / 0.99 / 0.999 | `binomial_half_interval`, `binomial_jeffreys_99`, `binomial_interval_999` |
| all successes / all failures | `binomial_all_success`, `binomial_all_fail` |
| large counts | `binomial_large` |
| negative / non-integer counts | `binomial_err_n1_neg`, `binomial_err_n2_neg`, `binomial_err_n1_nonint`, `binomial_err_n2_nonint` |
| non-positive / infinite / NA prior | `binomial_err_a0_zero`, `binomial_err_b0_zero`, `binomial_err_a0_inf`, `binomial_err_b0_inf`, `binomial_err_a0_na`, `binomial_err_b0_na` |
| NA counts | `binomial_err_n1_na`, `binomial_err_n2_na` |
| `prob_interval` at 0 / 1 / NA | `binomial_err_pi_zero`, `binomial_err_pi_one`, `binomial_err_pi_na` |

## `dfba_beta_descriptive` (`dfba_beta_descriptive.R`)

| R branch | Tests |
| --- | --- |
| regular mode/HDI | `bd_38_55`, `bd_38_55_90`, `bd_38_55_99`, `bd_large` |
| mode undefined (`a==1 & b==1`, or both ≤ 1) | `bd_1_1`, `bd_2_2`, `bd_1_2` |
| mode defined with one shape ≤ 1 | `bd_half_2`, `bd_2_half` |
| `prob_interval` 0.5 | `bd_interval_half` |
| non-positive / infinite / NA shape | `bd_err_a_zero`, `bd_err_b_zero`, `bd_err_a_inf`, `bd_err_b_inf`, `bd_err_a_na`, `bd_err_b_na` |
| `prob_interval` < 0 / > 1 / NA | `bd_err_pi_neg`, `bd_err_pi_gt1`, `bd_err_pi_na` |

## `dfba_beta_bayes_factor` (`dfba_beta_bayes_factor.R`)

| R branch | Tests |
| --- | --- |
| interval H0 | `bf_interval`, `bf_interval_narrow`, `bf_interval_jeffreys`, `bf_interval_full` |
| point H0 | `bf_point`, `bf_point_jeffreys`, `bf_point_zero`, `bf_point_one` |
| point density 0 → `BF10=NaN` / `BF01=Inf` | `bf_point_zero`, `bf_point_one` |
| interval H0 = (0,1) → `BF10=NaN` / `BF01=Inf` | `bf_interval_full` |
| bad method | `bf_err_method` |
| point H0 length / out of range | `bf_err_point_len2`, `bf_err_point_gt1`, `bf_err_point_neg` |
| interval H0 length / range / order | `bf_err_interval_len1`, `bf_err_interval_gt1`, `bf_err_interval_order` |
| prior validation and posterior below prior | `bf_err_a0_zero`, `bf_err_a_post_lt_a0`, `bf_err_b_post_lt_b0`, `bf_err_a_post_na` |

## `dfba_sign_test` (`dfba_sign_test.R`)

| R branch | Tests |
| --- | --- |
| positive/negative sign counts | `sign_default`, `sign_jeffreys_99` |
| zero differences removed | `sign_with_zeros`, `sign_three_pairs` |
| pairwise NA removal | `sign_with_na`, `sign_nan` |
| `prob_interval=0` | `sign_pi_zero` |
| BF always uses the uniform prior (upstream bug) | `sign_jeffreys_99` |
| length mismatch / too few / all trivial | `sign_err_length`, `sign_err_too_few`, `sign_err_all_zero` |
| NA prior / invalid interval | `sign_err_a0_na`, `sign_err_b0_na`, `sign_err_pi_neg`, `sign_err_pi_gt1`, `sign_err_pi_na` |

## `dfba_median_test` (`dfba_median_test.R`)

| R branch | Tests |
| --- | --- |
| E > C and C > E counts | `med_example`, `med_jeffreys` |
| ties | `med_ties`, `med_all_equal` |
| independent NA removal | `med_na` |
| one/both vectors empty | `med_one_empty`, `med_both_empty` |
| invalid prior | `med_err_a0_na`, `med_err_b0_na`, `med_err_a0_zero`, `med_err_b0_inf` |

## `dfba_gamma` (`dfba_gamma.R`)

| R branch | Tests |
| --- | --- |
| matrix/table input | `gamma_example`, `gamma_table`, `gamma_2x2` |
| zero cells | `gamma_zeros` |
| 1×1, 1×N, N×1 | `gamma_1x1`, `gamma_1xN`, `gamma_Nx1` |
| NA cell → `invalid 'times' argument` | `gamma_na` |
| non-matrix input | `gamma_not_matrix` |
| invalid prior / interval | `gamma_err_a0_na`, `gamma_err_b0_zero`, `gamma_err_pi_99` |

## `dfba_mcnemar` (`dfba_mcnemar.R`)

| R branch | Tests |
| --- | --- |
| `n_01 > n_10`, `n_10 > n_01` | `mc_default`, `mc_all_01`, `mc_all_10` |
| both zero | `mc_zero_zero` |
| large counts and cancellation | `mc_large` |
| Jeffreys prior / 99% interval | `mc_jeffreys_99` |
| negative / non-integer / NA counts | `mc_err_neg`, `mc_err_nonint`, `mc_err_na` |
| invalid prior / interval | `mc_err_a0_na`, `mc_err_pi_zero`, `mc_err_pi_one`, `mc_err_pi_na` |

## `dfba_bivariate_concordance` (`dfba_bivariate_concordance.R`)

| R branch | Tests |
| --- | --- |
| `nc`, `nd`, `tau` from concordant/discordant pairs | `bc_example`, `bc_jeffreys`, `bc_pi_99` |
| tied pairs removed from `nc`/`nd` | `bc_ties_doc` |
| all equal / empty / single pair / two pairs | `bc_all_equal`, `bc_empty`, `bc_single_pair`, `bc_two_pairs` |
| pairwise NA row removal warning | `bc_na_rows` |
| NA `prob_interval` propagates | `bc_pi_na` |
| `fitting_parameters` 0 / 2 / 3 / NA | `bc_star_0`, `bc_star_2`, `bc_star_3`, `bc_star_na` |
| invalid prior / unequal lengths | `bc_err_a0_na`, `bc_err_b0_zero`, `bc_err_length` |

The star class has R's duplicated `prob_interval` slot; see `DECISION.md`.

## `dfba_mann_whitney` (`dfba_mann_whitney.R`)

| R branch | Tests |
| --- | --- |
| `method=NULL`, `nH >= 20` → large | `mw_large_example`, `test_method_none_selects_at_the_nH_20_boundary[20-20-large]` |
| `method=NULL`, `nH < 20` → small | `test_method_none_selects_at_the_nH_20_boundary[19-21-small]`; forced small `mw_small_branch_*` |
| explicit `method="large"` | `mw_large_forced_small_data`, `mw_large_forced_19` |
| explicit `method="small"` | `mw_small_branch_a`, `mw_small_branch_b` (stochastic) |
| large branch `xs >= .5` | `mw_large_example`, `mw_large_jeffreys`, `mw_large_99` |
| large branch `xs < .5` (swap a/b) | `mw_large_reversed` |
| `nH` exactly at the 20 boundary | `mw_large_nH_boundary` |
| independent NA cell removal + messages | `mw_large_na` |
| all values equal → `missing value where TRUE/FALSE needed` | `mw_large_all_equal` |
| small branch A (`length(E)*length(C) > Ntot`) | `mw_small_branch_a` (stochastic, frozen `fomega`) |
| small branch B (`length(E)*length(C) <= Ntot`) | `mw_small_branch_b` (stochastic, frozen `fomega`) |
| `samples < 10000` message → warning | `mw_large_low_samples` |
| invalid prior / interval / method | `mw_err_a0_na`, `mw_err_b0_zero`, `mw_err_pi_neg`, `mw_err_pi_gt1`, `mw_err_pi_na`, `mw_err_method` |
| empty E / C | `mw_err_empty_e`, `mw_err_empty_c` |
| analytic null at `omega=0.5` (branch A and B) | `test_mw_fomega_matches_null_branch_a/b` |
| sampler determinism | `test_mw_sampler_is_deterministic` |
| public wrapper shape/range | `test_mw_small_public_wrapper` |

## `dfba_wilcoxon` (`dfba_wilcoxon.R`)

| R branch | Tests |
| --- | --- |
| `method=NULL`, `n > 24` → large | `wil_large_example` |
| explicit `method="large"` | `wil_large_forced_small` |
| explicit `method="small"` | `wil_small` (stochastic, frozen `fphi`) |
| ties get average ranks | `wil_large_ties` |
| zero/trivial differences removed | `wil_large_zero_diffs` |
| pairwise NA removal | `wil_large_na` |
| Jeffreys prior / 99% interval | `wil_large_jeffreys`, `wil_large_99` |
| `samples < 10000` message → warning | `wil_large_low_samples` |
| length mismatch / too few / all trivial | `wil_err_length`, `wil_err_too_few`, `wil_err_all_equal` |
| invalid prior / interval / method | `wil_err_a0_na`, `wil_err_b0_zero`, `wil_err_pi_neg`, `wil_err_pi_gt1`, `wil_err_pi_na`, `wil_err_method` |
| analytic null at `phi=0.5` (Poisson binomial) | `test_wilcoxon_fphi_matches_poisson_binomial_null` |
| sampler determinism | `test_wilcoxon_sampler_is_deterministic` |
| average tie ranks used by `tpos`/`tneg` | `test_average_ranks_matches_scipy_including_ties` |
| public wrapper shape/range | `test_wilcoxon_small_public_wrapper` |

## `dfba_beta_contrast` (`dfba_beta_contrast.R`)

| R branch | Tests |
| --- | --- |
| two conditions, `delta`/`delta0` frozen | `beta_contrast_2` (stochastic) |
| four conditions / orthogonal contrast | `beta_contrast_4` (stochastic) |
| `prH1==1 | priorprH1==0` shortcut | `_finish_beta_contrast` unit path; public smoke |
| `samples < 10000` / NA samples | `bcon_err_samples_low`, `bcon_err_samples_na` |
| vector length mismatch | `bcon_err_length` |
| non-integer / NA / negative counts | `bcon_err_n1_nonint`, `bcon_err_n1_na`, `bcon_err_n2_neg` |
| invalid prior | `bcon_err_a0_zero`, `bcon_err_a0_inf`, `bcon_err_a0_na`, `bcon_err_b0_na` |
| positive/negative contrast sums | `bcon_err_contrast_pos`, `bcon_err_contrast_neg` |
| NA contrast / invalid interval | `bcon_err_contrast_na`, `bcon_err_pi_neg`, `bcon_err_pi_na` |
| `a0_vec`/`b0_vec` length-1 broadcast (documented divergence) | public wrapper + `DECISION.md` |
| sampler determinism | `test_beta_contrast_sampler_is_deterministic` |
| public wrapper shape/range | `test_beta_contrast_public_wrapper` |

## `dfba_sim_data` (`dfba_sim_data.R`)

| R branch | Tests |
| --- | --- |
| negative delta | `sim_err_delta_neg` |
| `n < 20` / non-integer | `sim_err_n_lt20`, `sim_err_n_nonint` |
| invalid model / design | `sim_err_model`, `sim_err_design` |
| negative block effect | `sim_err_block_neg` |
| invalid prior | `sim_err_a0_zero` |
| model shape validation | `sim_err_shape_normal`, `sim_err_shape_pareto` |
| independent summary (t + Mann-Whitney) against frozen R E/C | `sim_frozen_independent` |
| paired summary (t + Wilcoxon) against frozen R E/C | `sim_frozen_paired` |
| nine model parameterisations | `test_model_means`, `test_cauchy_scale_and_location`, `test_pareto_support_and_median` |
| paired block effect cancels in differences | `test_sim_data_paired_block_effect_cancels_in_differences` |
| independent block effects change both conditions | `test_sim_data_independent_block_effect_changes_values` |
| `block_max=0` does not consume RNG state (R's `runif(n,0,0)` behaviour) | `test_zero_block_max_does_not_advance_the_generator` |
| pinned-seed reproducibility | `test_sim_data_is_reproducible_with_a_seed` |
| output shape, design, ranges | `test_sim_data_returns_both_conditions` |

## `dfba_power_curve` (`dfba_power_curve.R`)

| R branch | Tests |
| --- | --- |
| negative `delta_step` | `power_err_delta_step_neg` |
| `n < 20` | `power_err_n_lt20` |
| invalid model / design / effect criterion / prior / block | `power_err_model`, `power_err_design`, `power_err_effect_crit`, `power_err_a0_zero`, `power_err_block_neg` |
| 21-point delta grid and 21 power rows | `test_power_curve_structure_and_reproducibility` |
| `n` rounded like R | `test_power_curve_rounds_n_like_r` |
| `samples` errors for NA/NaN/Inf/negative | `test_samples_rejected_where_r_also_errors` |
| `samples` rounding to zero (deliberate divergence) | `test_samples_rounding_to_zero_are_rejected_divergence` |
| `samples` rounded like R | `test_samples_are_rounded_like_r` |
| pinned-seed reproducibility | `test_power_curve_structure_and_reproducibility` |
| summary fast path equals full analysis | `test_summary_fast_path_matches_full_analysis` |
| power loop shares the draw path with `dfba_sim_data` | `test_power_loop_uses_the_public_draw_path` |
| output table and pandas export | `test_output_table_is_pandas_friendly`, `test_power_result_helpers_and_batch_dataframe` |
| plotting | `test_power_curve_plot` |

## `dfba_bayes_vs_t_power` (`dfba_bayes_vs_t_power.R`)

| R branch | Tests |
| --- | --- |
| negative delta | `bvt_err_delta_neg` |
| `n_min < 20` / non-integer | `bvt_err_n_min_lt20`, `bvt_err_n_min_nonint` |
| invalid prior / model / design / effect criterion | `bvt_err_a0_zero`, `bvt_err_model`, `bvt_err_design`, `bvt_err_effect_crit` |
| 11 sample sizes from `n_min` to `n_min+50` | `test_bayes_vs_t_power_structure_and_reproducibility`, `test_bayes_vs_t_power_sample_size_grid_respects_n_min` |
| `samples` validation (shared `_check_nsims`) | `test_samples_rejected_where_r_also_errors`, `test_samples_rounding_to_zero_are_rejected_divergence`, `test_samples_are_rounded_like_r` |
| pinned-seed reproducibility | `test_bayes_vs_t_power_structure_and_reproducibility` |
| plotting | `test_bayes_vs_t_power_plot` |

## `methods-*.R`

| R source | Coverage |
| --- | --- |
| `show` methods | `summary()`/`str()` field order in `tests/test_results.py` |
| `plot` methods | Python-native `result.plot()` dispatch in `tests/test_plotting.py`; not byte-compatible with R base graphics |
