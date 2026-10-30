#!/usr/bin/env Rscript
# Generate golden fixtures by running the R original DFBA 0.1.0.
#
# Usage:
#   Rscript tests/fixtures/generate.R
#   DFBA_STOCHASTIC=1 Rscript tests/fixtures/generate.R   # also freeze RNG intermediates
#
# Never hand-write expected values. Every number in fixtures/*.json comes from
# calling the installed DFBA package below.

suppressPackageStartupMessages(library(DFBA))

stopifnot(as.character(packageVersion("DFBA")) == "0.1.0")

out_dir <- file.path("tests", "fixtures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

## ---------------------------------------------------------------------------
## Minimal full-precision JSON writer
##
## jsonlite::toJSON() is available, but its default `digits` rounds and it
## cannot represent R's distinction between NA and NaN. This writer emits
## %.17g doubles (round-trip exact) and tagged objects for NA/NaN/Inf.
## ---------------------------------------------------------------------------

json_escape <- function(s) {
  s <- gsub("\\\\", "\\\\\\\\", s)
  s <- gsub("\"", "\\\\\"", s)
  s <- gsub("\n", "\\\\n", s)
  s <- gsub("\r", "\\\\r", s)
  s <- gsub("\t", "\\\\t", s)
  paste0("\"", s, "\"")
}

na_type <- function(x) {
  if (is.integer(x)) "integer"
  else if (is.logical(x)) "logical"
  else if (is.character(x)) "character"
  else "double"
}

scalar_json <- function(x) {
  if (is.na(x)) {
    if (is.nan(x)) return('{"__nan__":true}')
    return(paste0('{"__na__":"', na_type(x), '"}'))
  }
  if (is.infinite(x)) {
    return(if (x > 0) '{"__inf__":1}' else '{"__inf__":-1}')
  }
  if (is.logical(x)) return(if (x) "true" else "false")
  if (is.integer(x)) return(format(x, scientific = FALSE))
  if (is.character(x)) return(json_escape(x))
  sprintf("%.17g", x)
}

atomic_json <- function(x, force_array = FALSE) {
  if (length(x) == 0) return("[]")
  if (!force_array && length(x) == 1) return(scalar_json(x))
  paste0("[", paste(vapply(x, scalar_json, character(1)), collapse = ","), "]")
}

to_json <- function(x, force_array = FALSE) {
  if (is.null(x)) return("null")

  if (is.data.frame(x)) {
    cols <- lapply(x, atomic_json, force_array = TRUE)
    return(paste0(
      '{"__df__":{"names":', atomic_json(names(x), TRUE),
      ',"columns":[', paste(cols, collapse = ","), ']',
      ',"nrow":', nrow(x), "}}"
    ))
  }

  if (is.matrix(x)) {
    dn <- dimnames(x)
    dn_json <- if (is.null(dn)) "null" else paste0(
      "[", paste(vapply(dn, function(z) {
        if (is.null(z)) "null" else atomic_json(z, TRUE)
      }, character(1)), collapse = ","), "]"
    )
    return(paste0(
      '{"__matrix__":{"dim":[', nrow(x), ",", ncol(x), "]",
      ',"data":', atomic_json(as.vector(x), TRUE),
      ',"dimnames":', dn_json, "}}"
    ))
  }

  if (isS4(x)) {
    return(to_json(as.list(x)))
  }

  if (is.list(x)) {
    nms <- names(x)
    if (!is.null(nms) && any(nzchar(nms))) {
      pairs <- vapply(seq_along(x), function(i) {
        paste0("[", json_escape(nms[[i]]), ",", to_json(x[[i]]), "]")
      }, character(1))
      return(paste0('{"__list__":[', paste(pairs, collapse = ","), "]}"))
    }
    return(paste0("[", paste(vapply(x, to_json, character(1)), collapse = ","), "]"))
  }

  atomic_json(x, force_array = force_array)
}

## ---------------------------------------------------------------------------
## Case runner
## ---------------------------------------------------------------------------

run_case <- function(name, call_text, expr) {
  warnings <- character()
  messages <- character()

  res <- tryCatch(
    withCallingHandlers(
      eval(expr, envir = parent.frame()),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      },
      message = function(m) {
        messages <<- c(messages, conditionMessage(m))
        invokeRestart("muffleMessage")
      }
    ),
    error = function(e) {
      structure(list(message = conditionMessage(e), class = class(e)[1]),
                class = "__fixture_error__")
    }
  )

  header <- paste0(
    '"call":', json_escape(call_text)
  )
  if (length(warnings)) {
    header <- paste0(header, ',"warnings":', atomic_json(warnings, TRUE))
  }
  if (length(messages)) {
    header <- paste0(header, ',"messages":', atomic_json(messages, TRUE))
  }

  if (inherits(res, "__fixture_error__")) {
    return(paste0("{", header,
                  ',"error":', json_escape(res$message),
                  ',"error_class":', json_escape(res$class), "}"))
  }

  cls <- class(res)[1]
  fields <- as.list(res)
  field_json <- vapply(seq_along(fields), function(i) {
    paste0("[", json_escape(names(fields)[[i]]), ",", to_json(fields[[i]]), "]")
  }, character(1))

  paste0("{", header,
         ',"class":', json_escape(cls),
         ',"fields":[', paste(field_json, collapse = ","), "]}")
}

case_list <- list()

add <- function(name, call_text, expr) {
  case_list[[name]] <<- run_case(name, call_text, expr)
}

## ---------------------------------------------------------------------------
## Deterministic cases
## ---------------------------------------------------------------------------

# dfba_binomial -------------------------------------------------------------
add("binomial_default", "dfba_binomial(n1=16, n2=2)", quote(dfba_binomial(n1 = 16, n2 = 2)))
add("binomial_jeffreys_99", "dfba_binomial(16, 2, a0=.5, b0=.5, prob_interval=.99)",
    quote(dfba_binomial(16, 2, a0 = .5, b0 = .5, prob_interval = .99)))
add("binomial_zero_zero", "dfba_binomial(0, 0)", quote(dfba_binomial(0, 0)))
add("binomial_all_success", "dfba_binomial(5, 0)", quote(dfba_binomial(5, 0)))
add("binomial_all_fail", "dfba_binomial(0, 5)", quote(dfba_binomial(0, 5)))
add("binomial_large", "dfba_binomial(1000, 900)", quote(dfba_binomial(1000, 900)))
add("binomial_half_interval", "dfba_binomial(10, 10, prob_interval=.5)",
    quote(dfba_binomial(10, 10, prob_interval = .5)))
add("binomial_prior_2_3", "dfba_binomial(7, 3, a0=2, b0=3)",
    quote(dfba_binomial(7, 3, a0 = 2, b0 = 3)))
add("binomial_interval_999", "dfba_binomial(3, 4, prob_interval=.999)",
    quote(dfba_binomial(3, 4, prob_interval = .999)))
add("binomial_err_n1_neg", "dfba_binomial(-1, 2)", quote(dfba_binomial(-1, 2)))
add("binomial_err_n2_neg", "dfba_binomial(1, -2)", quote(dfba_binomial(1, -2)))
add("binomial_err_n1_nonint", "dfba_binomial(1.5, 2)", quote(dfba_binomial(1.5, 2)))
add("binomial_err_n2_nonint", "dfba_binomial(1, 2.5)", quote(dfba_binomial(1, 2.5)))
add("binomial_err_a0_zero", "dfba_binomial(1, 2, a0=0)", quote(dfba_binomial(1, 2, a0 = 0)))
add("binomial_err_b0_zero", "dfba_binomial(1, 2, b0=0)", quote(dfba_binomial(1, 2, b0 = 0)))
add("binomial_err_a0_inf", "dfba_binomial(1, 2, a0=Inf)", quote(dfba_binomial(1, 2, a0 = Inf)))
add("binomial_err_b0_inf", "dfba_binomial(1, 2, b0=Inf)", quote(dfba_binomial(1, 2, b0 = Inf)))
add("binomial_err_a0_na", "dfba_binomial(1, 2, a0=NA)", quote(dfba_binomial(1, 2, a0 = NA)))
add("binomial_err_b0_na", "dfba_binomial(1, 2, b0=NA)", quote(dfba_binomial(1, 2, b0 = NA)))
add("binomial_err_n1_na", "dfba_binomial(NA, 2)", quote(dfba_binomial(NA, 2)))
add("binomial_err_n2_na", "dfba_binomial(1, NA)", quote(dfba_binomial(1, NA)))
add("binomial_err_pi_zero", "dfba_binomial(1, 2, prob_interval=0)",
    quote(dfba_binomial(1, 2, prob_interval = 0)))
add("binomial_err_pi_one", "dfba_binomial(1, 2, prob_interval=1)",
    quote(dfba_binomial(1, 2, prob_interval = 1)))
add("binomial_err_pi_na", "dfba_binomial(1, 2, prob_interval=NA)",
    quote(dfba_binomial(1, 2, prob_interval = NA)))

# dfba_beta_descriptive -----------------------------------------------------
add("bd_38_55", "dfba_beta_descriptive(38, 55)", quote(dfba_beta_descriptive(38, 55)))
add("bd_38_55_90", "dfba_beta_descriptive(38, 55, .90)",
    quote(dfba_beta_descriptive(38, 55, prob_interval = .90)))
add("bd_38_55_99", "dfba_beta_descriptive(38, 55, .99)",
    quote(dfba_beta_descriptive(38, 55, prob_interval = .99)))
add("bd_1_1", "dfba_beta_descriptive(1, 1)", quote(dfba_beta_descriptive(1, 1)))
add("bd_half_2", "dfba_beta_descriptive(.5, 2)", quote(dfba_beta_descriptive(.5, 2)))
add("bd_2_half", "dfba_beta_descriptive(2, .5)", quote(dfba_beta_descriptive(2, .5)))
add("bd_2_2", "dfba_beta_descriptive(2, 2)", quote(dfba_beta_descriptive(2, 2)))
add("bd_1_2", "dfba_beta_descriptive(1, 2)", quote(dfba_beta_descriptive(1, 2)))
add("bd_large", "dfba_beta_descriptive(1000, 2000)", quote(dfba_beta_descriptive(1000, 2000)))
add("bd_interval_half", "dfba_beta_descriptive(5, 7, prob_interval=.5)",
    quote(dfba_beta_descriptive(5, 7, prob_interval = .5)))
add("bd_err_a_zero", "dfba_beta_descriptive(0, 1)", quote(dfba_beta_descriptive(0, 1)))
add("bd_err_b_zero", "dfba_beta_descriptive(1, 0)", quote(dfba_beta_descriptive(1, 0)))
add("bd_err_a_inf", "dfba_beta_descriptive(Inf, 1)", quote(dfba_beta_descriptive(Inf, 1)))
add("bd_err_b_inf", "dfba_beta_descriptive(1, Inf)", quote(dfba_beta_descriptive(1, Inf)))
add("bd_err_a_na", "dfba_beta_descriptive(NA, 1)", quote(dfba_beta_descriptive(NA, 1)))
add("bd_err_b_na", "dfba_beta_descriptive(1, NA)", quote(dfba_beta_descriptive(1, NA)))
add("bd_err_pi_neg", "dfba_beta_descriptive(1, 2, prob_interval=-.1)",
    quote(dfba_beta_descriptive(1, 2, prob_interval = -.1)))
add("bd_err_pi_gt1", "dfba_beta_descriptive(1, 2, prob_interval=1.1)",
    quote(dfba_beta_descriptive(1, 2, prob_interval = 1.1)))
add("bd_err_pi_na", "dfba_beta_descriptive(1, 2, prob_interval=NA)",
    quote(dfba_beta_descriptive(1, 2, prob_interval = NA)))

# dfba_beta_bayes_factor ----------------------------------------------------
add("bf_interval", "dfba_beta_bayes_factor(17, 5, 'interval', c(0,.5))",
    quote(dfba_beta_bayes_factor(17, 5, method = "interval", H0 = c(0, .5))))
add("bf_point", "dfba_beta_bayes_factor(377, 123, 'point', .75)",
    quote(dfba_beta_bayes_factor(377, 123, method = "point", H0 = .75)))
add("bf_point_jeffreys", "dfba_beta_bayes_factor(377.5, 123.5, 'point', .75, a0=.5, b0=.5)",
    quote(dfba_beta_bayes_factor(377.5, 123.5, method = "point", H0 = .75, a0 = .5, b0 = .5)))
add("bf_interval_narrow", "dfba_beta_bayes_factor(273, 278, 'interval', c(.4975,.5025))",
    quote(dfba_beta_bayes_factor(273, 278, method = "interval", H0 = c(.4975, .5025))))
add("bf_interval_full", "dfba_beta_bayes_factor(10, 10, 'interval', c(0,1))",
    quote(dfba_beta_bayes_factor(10, 10, method = "interval", H0 = c(0, 1))))
add("bf_point_zero", "dfba_beta_bayes_factor(10, 10, 'point', 0)",
    quote(dfba_beta_bayes_factor(10, 10, method = "point", H0 = 0)))
add("bf_point_one", "dfba_beta_bayes_factor(10, 10, 'point', 1)",
    quote(dfba_beta_bayes_factor(10, 10, method = "point", H0 = 1)))
add("bf_interval_jeffreys", "dfba_beta_bayes_factor(17.5, 5.5, 'interval', c(0,.5), a0=.5, b0=.5)",
    quote(dfba_beta_bayes_factor(17.5, 5.5, method = "interval", H0 = c(0, .5), a0 = .5, b0 = .5)))
add("bf_err_method", "dfba_beta_bayes_factor(17, 5, 'bad', c(0,.5))",
    quote(dfba_beta_bayes_factor(17, 5, method = "bad", H0 = c(0, .5))))
add("bf_err_point_len2", "dfba_beta_bayes_factor(17, 5, 'point', c(.5,.6))",
    quote(dfba_beta_bayes_factor(17, 5, method = "point", H0 = c(.5, .6))))
add("bf_err_interval_len1", "dfba_beta_bayes_factor(17, 5, 'interval', .5)",
    quote(dfba_beta_bayes_factor(17, 5, method = "interval", H0 = .5)))
add("bf_err_point_gt1", "dfba_beta_bayes_factor(17, 5, 'point', 1.1)",
    quote(dfba_beta_bayes_factor(17, 5, method = "point", H0 = 1.1)))
add("bf_err_point_neg", "dfba_beta_bayes_factor(17, 5, 'point', -.1)",
    quote(dfba_beta_bayes_factor(17, 5, method = "point", H0 = -.1)))
add("bf_err_interval_gt1", "dfba_beta_bayes_factor(17, 5, 'interval', c(0,1.1))",
    quote(dfba_beta_bayes_factor(17, 5, method = "interval", H0 = c(0, 1.1))))
add("bf_err_interval_order", "dfba_beta_bayes_factor(17, 5, 'interval', c(.6,.5))",
    quote(dfba_beta_bayes_factor(17, 5, method = "interval", H0 = c(.6, .5))))
add("bf_err_a0_zero", "dfba_beta_bayes_factor(17, 5, 'interval', c(0,.5), a0=0)",
    quote(dfba_beta_bayes_factor(17, 5, method = "interval", H0 = c(0, .5), a0 = 0)))
add("bf_err_a_post_lt_a0", "dfba_beta_bayes_factor(1, 5, 'interval', c(0,.5), a0=2)",
    quote(dfba_beta_bayes_factor(1, 5, method = "interval", H0 = c(0, .5), a0 = 2)))
add("bf_err_b_post_lt_b0", "dfba_beta_bayes_factor(5, 1, 'interval', c(0,.5), b0=2)",
    quote(dfba_beta_bayes_factor(5, 1, method = "interval", H0 = c(0, .5), b0 = 2)))
add("bf_err_a_post_na", "dfba_beta_bayes_factor(NA, 5, 'interval', c(0,.5))",
    quote(dfba_beta_bayes_factor(NA, 5, method = "interval", H0 = c(0, .5))))

# dfba_sign_test ------------------------------------------------------------
measure_1 <- c(1.49, 0.64, 0.96, 2.34, 0.78, 1.29, 0.72, 1.52,
               0.62, 1.67, 1.19, 0.860)
measure_2 <- c(0.53, 0.55, 0.58, 0.97, 0.60, 0.22, 0.05, 13.14,
               0.63, 0.33, 0.91, 0.37)
add("sign_default", "dfba_sign_test(measure_1, measure_2)",
    quote(dfba_sign_test(measure_1, measure_2)))
add("sign_jeffreys_99", "dfba_sign_test(measure_1, measure_2, a0=.5, b0=.5, prob_interval=.99)",
    quote(dfba_sign_test(measure_1, measure_2, a0 = .5, b0 = .5, prob_interval = .99)))
add("sign_with_na", "dfba_sign_test(Y1 with NA[3], Y2 with NA[7])",
    quote({
      y1 <- measure_1; y1[3] <- NA
      y2 <- measure_2; y2[7] <- NA
      dfba_sign_test(y1, y2)
    }))
add("sign_with_zeros", "dfba_sign_test(c(1,2,2,3,4), c(0,2,1,3,5))",
    quote(dfba_sign_test(c(1, 2, 2, 3, 4), c(0, 2, 1, 3, 5))))
add("sign_nan", "dfba_sign_test(Y1 with NaN[2], Y2)",
    quote({
      y1 <- measure_1; y1[2] <- NaN
      dfba_sign_test(y1, measure_2)
    }))
add("sign_three_pairs", "dfba_sign_test(c(1,2,3), c(0,0,0))",
    quote(dfba_sign_test(c(1, 2, 3), c(0, 0, 0))))
add("sign_pi_zero", "dfba_sign_test(c(1,2,3,4), c(0,0,0,0), prob_interval=0)",
    quote(dfba_sign_test(c(1, 2, 3, 4), c(0, 0, 0, 0), prob_interval = 0)))
add("sign_err_length", "dfba_sign_test(c(1,2,3), c(0,0))",
    quote(dfba_sign_test(c(1, 2, 3), c(0, 0))))
add("sign_err_too_few", "dfba_sign_test(c(1,2,NA), c(0,0,0))",
    quote(dfba_sign_test(c(1, 2, NA), c(0, 0, 0))))
add("sign_err_all_zero", "dfba_sign_test(c(1,1,1,1), c(1,1,1,1))",
    quote(dfba_sign_test(c(1, 1, 1, 1), c(1, 1, 1, 1))))
add("sign_err_a0_na", "dfba_sign_test(measure_1, measure_2, a0=NA)",
    quote(dfba_sign_test(measure_1, measure_2, a0 = NA)))
add("sign_err_b0_na", "dfba_sign_test(measure_1, measure_2, b0=NA)",
    quote(dfba_sign_test(measure_1, measure_2, b0 = NA)))
add("sign_err_pi_neg", "dfba_sign_test(measure_1, measure_2, prob_interval=-.1)",
    quote(dfba_sign_test(measure_1, measure_2, prob_interval = -.1)))
add("sign_err_pi_gt1", "dfba_sign_test(measure_1, measure_2, prob_interval=1.1)",
    quote(dfba_sign_test(measure_1, measure_2, prob_interval = 1.1)))
add("sign_err_pi_na", "dfba_sign_test(measure_1, measure_2, prob_interval=NA)",
    quote(dfba_sign_test(measure_1, measure_2, prob_interval = NA)))

# dfba_median_test ----------------------------------------------------------
group1 <- c(12.90, 10.84, 22.67, 10.64, 10.67, 10.79, 13.55, 10.95, 12.19,
            12.76, 10.89, 11.02, 14.27, 13.98, 11.52, 13.49, 11.22, 15.07,
            15.74, 19.00)
group2 <- c(4.63, 58.64, 5.07, 4.66, 4.13, 3.92, 3.39, 3.57, 3.56, 3.39)
add("med_example", "dfba_median_test(group1, group2)", quote(dfba_median_test(group1, group2)))
add("med_jeffreys", "dfba_median_test(group1, group2, a0=.5, b0=.5)",
    quote(dfba_median_test(group1, group2, a0 = .5, b0 = .5)))
add("med_ties", "dfba_median_test(c(1,2,2,3), c(2,2,4,5))",
    quote(dfba_median_test(c(1, 2, 2, 3), c(2, 2, 4, 5))))
add("med_all_equal", "dfba_median_test(rep(2,5), rep(2,4))",
    quote(dfba_median_test(rep(2, 5), rep(2, 4))))
add("med_na", "dfba_median_test(c(1,NA,3,4), c(2,5,NA,6))",
    quote(dfba_median_test(c(1, NA, 3, 4), c(2, 5, NA, 6))))
add("med_one_empty", "dfba_median_test(c(1,2,3), numeric(0))",
    quote(dfba_median_test(c(1, 2, 3), numeric(0))))
add("med_both_empty", "dfba_median_test(numeric(0), numeric(0))",
    quote(dfba_median_test(numeric(0), numeric(0))))
add("med_err_a0_na", "dfba_median_test(group1, group2, a0=NA)",
    quote(dfba_median_test(group1, group2, a0 = NA)))
add("med_err_b0_na", "dfba_median_test(group1, group2, b0=NA)",
    quote(dfba_median_test(group1, group2, b0 = NA)))
add("med_err_a0_zero", "dfba_median_test(group1, group2, a0=0)",
    quote(dfba_median_test(group1, group2, a0 = 0)))
add("med_err_b0_inf", "dfba_median_test(group1, group2, b0=Inf)",
    quote(dfba_median_test(group1, group2, b0 = Inf)))

# dfba_gamma ----------------------------------------------------------------
gamma_mat <- matrix(c(38, 4, 5, 0, 6, 40, 1, 2, 4, 8, 20, 30),
                    ncol = 4, byrow = TRUE)
colnames(gamma_mat) <- c("C1", "C2", "C3", "C4")
rownames(gamma_mat) <- c("R1", "R2", "R3")
add("gamma_example", "dfba_gamma(matrix(...))", quote(dfba_gamma(gamma_mat)))
add("gamma_table", "dfba_gamma(as.table(matrix(...)))", quote(dfba_gamma(as.table(gamma_mat))))
add("gamma_2x2", "dfba_gamma(matrix(c(10,2,3,8),2,2))",
    quote(dfba_gamma(matrix(c(10, 2, 3, 8), 2, 2))))
add("gamma_zeros", "dfba_gamma(matrix(c(0,1,2,0),2,2))",
    quote(dfba_gamma(matrix(c(0, 1, 2, 0), 2, 2))))
add("gamma_1x1", "dfba_gamma(matrix(5,1,1))", quote(dfba_gamma(matrix(5, 1, 1))))
add("gamma_1xN", "dfba_gamma(matrix(c(1,2,3), nrow=1))",
    quote(dfba_gamma(matrix(c(1, 2, 3), nrow = 1))))
add("gamma_Nx1", "dfba_gamma(matrix(c(1,2,3), ncol=1))",
    quote(dfba_gamma(matrix(c(1, 2, 3), ncol = 1))))
add("gamma_na", "dfba_gamma(matrix(c(1,NA,2,3),2,2))",
    quote(dfba_gamma(matrix(c(1, NA, 2, 3), 2, 2))))
add("gamma_not_matrix", "dfba_gamma(c(1,2,3))", quote(dfba_gamma(c(1, 2, 3))))
add("gamma_err_a0_na", "dfba_gamma(gamma_mat, a0=NA)", quote(dfba_gamma(gamma_mat, a0 = NA)))
add("gamma_err_b0_zero", "dfba_gamma(gamma_mat, b0=0)", quote(dfba_gamma(gamma_mat, b0 = 0)))
add("gamma_err_pi_99", "dfba_gamma(gamma_mat, prob_interval=.99)",
    quote(dfba_gamma(gamma_mat, prob_interval = .99)))

# dfba_mcnemar --------------------------------------------------------------
add("mc_default", "dfba_mcnemar(17, 2)", quote(dfba_mcnemar(n_01 = 17, n_10 = 2)))
add("mc_jeffreys_99", "dfba_mcnemar(17, 2, a0=.5, b0=.5, prob_interval=.99)",
    quote(dfba_mcnemar(17, 2, a0 = .5, b0 = .5, prob_interval = .99)))
add("mc_zero_zero", "dfba_mcnemar(0, 0)", quote(dfba_mcnemar(0, 0)))
add("mc_all_01", "dfba_mcnemar(10, 0)", quote(dfba_mcnemar(10, 0)))
add("mc_all_10", "dfba_mcnemar(0, 10)", quote(dfba_mcnemar(0, 10)))
add("mc_large", "dfba_mcnemar(100, 200)", quote(dfba_mcnemar(100, 200)))
add("mc_err_neg", "dfba_mcnemar(-1, 2)", quote(dfba_mcnemar(-1, 2)))
add("mc_err_nonint", "dfba_mcnemar(1.5, 2)", quote(dfba_mcnemar(1.5, 2)))
add("mc_err_na", "dfba_mcnemar(NA, 2)", quote(dfba_mcnemar(NA, 2)))
add("mc_err_a0_na", "dfba_mcnemar(1, 2, a0=NA)", quote(dfba_mcnemar(1, 2, a0 = NA)))
add("mc_err_pi_zero", "dfba_mcnemar(1, 2, prob_interval=0)",
    quote(dfba_mcnemar(1, 2, prob_interval = 0)))
add("mc_err_pi_one", "dfba_mcnemar(1, 2, prob_interval=1)",
    quote(dfba_mcnemar(1, 2, prob_interval = 1)))
add("mc_err_pi_na", "dfba_mcnemar(1, 2, prob_interval=NA)",
    quote(dfba_mcnemar(1, 2, prob_interval = NA)))

# dfba_bivariate_concordance ------------------------------------------------
bc_x <- c(47, 39, 47, 42, 44, 46, 39, 37, 29, 42, 54, 33, 44, 31, 28, 49, 32, 37, 46, 55, 31)
bc_y <- c(36, 40, 49, 45, 30, 38, 39, 44, 27, 48, 49, 51, 27, 36, 30, 44, 42, 41, 35, 49, 33)
add("bc_example", "dfba_bivariate_concordance(bc_x, bc_y)",
    quote(dfba_bivariate_concordance(bc_x, bc_y)))
add("bc_ties_doc", "dfba_bivariate_concordance(c(3.8,4.7,4.7,4.7,11.8), c(5.9,-4.1,7.3,7.3,38.9))",
    quote(dfba_bivariate_concordance(c(3.8, 4.7, 4.7, 4.7, 11.8),
                                     c(5.9, -4.1, 7.3, 7.3, 38.9))))
add("bc_all_equal", "dfba_bivariate_concordance(c(1,1,1), c(2,2,2))",
    quote(dfba_bivariate_concordance(c(1, 1, 1), c(2, 2, 2))))
add("bc_empty", "dfba_bivariate_concordance(numeric(0), numeric(0))",
    quote(dfba_bivariate_concordance(numeric(0), numeric(0))))
add("bc_single_pair", "dfba_bivariate_concordance(1, 2)",
    quote(dfba_bivariate_concordance(1, 2)))
add("bc_two_pairs", "dfba_bivariate_concordance(c(1,2), c(2,1))",
    quote(dfba_bivariate_concordance(c(1, 2), c(2, 1))))
add("bc_na_rows", "dfba_bivariate_concordance(x with NA[1], y with NA[3])",
    quote({
      x <- bc_x; x[1] <- NA
      y <- bc_y; y[3] <- NA
      dfba_bivariate_concordance(x, y)
    }))
add("bc_pi_na", "dfba_bivariate_concordance(bc_x, bc_y, prob_interval=NA)",
    quote(dfba_bivariate_concordance(bc_x, bc_y, prob_interval = NA)))
add("bc_star_3", "dfba_bivariate_concordance(bc_x, bc_y, fitting.parameters=3)",
    quote(dfba_bivariate_concordance(bc_x, bc_y, fitting.parameters = 3)))
add("bc_star_2", "dfba_bivariate_concordance(bc_x, bc_y, fitting.parameters=2)",
    quote(dfba_bivariate_concordance(bc_x, bc_y, fitting.parameters = 2)))
add("bc_star_0", "dfba_bivariate_concordance(bc_x, bc_y, fitting.parameters=0)",
    quote(dfba_bivariate_concordance(bc_x, bc_y, fitting.parameters = 0)))
add("bc_star_na", "dfba_bivariate_concordance(bc_x, bc_y, fitting.parameters=NA)",
    quote(dfba_bivariate_concordance(bc_x, bc_y, fitting.parameters = NA)))
add("bc_jeffreys", "dfba_bivariate_concordance(bc_x, bc_y, a0=.5, b0=.5)",
    quote(dfba_bivariate_concordance(bc_x, bc_y, a0 = .5, b0 = .5)))
add("bc_pi_99", "dfba_bivariate_concordance(bc_x, bc_y, prob_interval=.99)",
    quote(dfba_bivariate_concordance(bc_x, bc_y, prob_interval = .99)))
add("bc_err_a0_na", "dfba_bivariate_concordance(bc_x, bc_y, a0=NA)",
    quote(dfba_bivariate_concordance(bc_x, bc_y, a0 = NA)))
add("bc_err_b0_zero", "dfba_bivariate_concordance(bc_x, bc_y, b0=0)",
    quote(dfba_bivariate_concordance(bc_x, bc_y, b0 = 0)))
add("bc_err_length", "dfba_bivariate_concordance(c(1,2,3), c(1,2))",
    quote(dfba_bivariate_concordance(c(1, 2, 3), c(1, 2))))

## ---------------------------------------------------------------------------
## method="large" cases
## ---------------------------------------------------------------------------

groupA <- c(43, 45, 47, 50, 54, 58, 60, 63, 69, 84, 85, 91, 99, 127, 130,
            147, 165, 175, 193, 228, 252, 276)
groupB <- c(0, 1, 2, 3, 5, 14, 15, 23, 23, 25, 27, 32, 57, 105, 115, 158,
            161, 181, 203, 290)
groupC <- c(96.49, 96.78, 97.26, 98.85, 99.75, 100.14, 101.15, 101.39,
            102.58, 107.22, 107.70, 113.26)
groupD <- c(101.16, 102.09, 103.14, 104.70, 105.27, 108.22, 108.32, 108.51,
            109.88, 110.32, 110.55, 113.42)

add("mw_large_example", "dfba_mann_whitney(groupA, groupB)",
    quote(dfba_mann_whitney(E = groupA, C = groupB)))
add("mw_large_jeffreys", "dfba_mann_whitney(groupA, groupB, a0=.5, b0=.5)",
    quote(dfba_mann_whitney(E = groupA, C = groupB, a0 = .5, b0 = .5)))
add("mw_large_reversed", "dfba_mann_whitney(groupB, groupA, a0=.5, b0=.5)",
    quote(dfba_mann_whitney(E = groupB, C = groupA, a0 = .5, b0 = .5)))
add("mw_large_99", "dfba_mann_whitney(groupA, groupB, a0=.5, b0=.5, prob_interval=.99)",
    quote(dfba_mann_whitney(E = groupA, C = groupB, a0 = .5, b0 = .5, prob_interval = .99)))
add("mw_large_forced_small_data", "dfba_mann_whitney(groupC, groupD, method='large')",
    quote(dfba_mann_whitney(E = groupC, C = groupD, method = "large")))
add("mw_large_na", "dfba_mann_whitney(E with NA[1], C with NA[2], method='large')",
    quote({
      e <- groupA; e[1] <- NA
      cc <- groupB; cc[2] <- NA
      dfba_mann_whitney(E = e, C = cc, method = "large")
    }))
add("mw_large_all_equal", "dfba_mann_whitney(rep(1,10), rep(1,10), method='large')",
    quote(dfba_mann_whitney(rep(1, 10), rep(1, 10), method = "large")))
set.seed(20261029)
mw_bound_e <- rnorm(20, 1)
mw_bound_c <- rnorm(20, 0)
add("mw_large_nH_boundary", "dfba_mann_whitney(nE=nC=20, method=NULL)",
    quote(dfba_mann_whitney(E = mw_bound_e, C = mw_bound_c)))
add("mw_large_forced_19", "dfba_mann_whitney(nE=nC=19, method='large')",
    quote(dfba_mann_whitney(E = mw_bound_e[1:19], C = mw_bound_c[1:19], method = "large")))
add("mw_large_low_samples", "dfba_mann_whitney(groupA, groupB, samples=100, method='large')",
    quote(dfba_mann_whitney(E = groupA, C = groupB, samples = 100, method = "large")))
add("mw_err_a0_na", "dfba_mann_whitney(groupA, groupB, a0=NA)",
    quote(dfba_mann_whitney(E = groupA, C = groupB, a0 = NA)))
add("mw_err_b0_zero", "dfba_mann_whitney(groupA, groupB, b0=0)",
    quote(dfba_mann_whitney(E = groupA, C = groupB, b0 = 0)))
add("mw_err_pi_neg", "dfba_mann_whitney(groupA, groupB, prob_interval=-.1)",
    quote(dfba_mann_whitney(E = groupA, C = groupB, prob_interval = -.1)))
add("mw_err_pi_gt1", "dfba_mann_whitney(groupA, groupB, prob_interval=1.1)",
    quote(dfba_mann_whitney(E = groupA, C = groupB, prob_interval = 1.1)))
add("mw_err_pi_na", "dfba_mann_whitney(groupA, groupB, prob_interval=NA)",
    quote(dfba_mann_whitney(E = groupA, C = groupB, prob_interval = NA)))
add("mw_err_method", "dfba_mann_whitney(groupA, groupB, method='bad')",
    quote(dfba_mann_whitney(E = groupA, C = groupB, method = "bad")))
add("mw_err_empty_e", "dfba_mann_whitney(numeric(0), groupB)",
    quote(dfba_mann_whitney(E = numeric(0), C = groupB)))
add("mw_err_empty_c", "dfba_mann_whitney(groupA, numeric(0))",
    quote(dfba_mann_whitney(E = groupA, C = numeric(0))))

wil_e <- c(6.45, 5.65, 4.34, 5.92, 2.84, 13.06, 6.61, 5.47, 4.49, 6.39, 6.63,
           3.55, 3.76, 5.61, 7.45, 6.41, 10.16, 6.26, 8.46, 2.29, 3.16, 5.68,
           4.13, 2.94, 4.87, 4.44, 3.13, 8.87)
wil_c <- c(2.89, 4.19, 3.22, 6.50, 3.10, 4.19, 5.13, 3.77, 2.71, 2.58, 7.59,
           2.68, 4.98, 2.35, 5.15, 8.46, 3.77, 8.83, 4.06, 2.50, 5.48, 2.80,
           8.89, 3.19, 9.36, 4.58, 2.94, 4.75)
wil_a <- c(1.49, 0.64, 0.96, 2.34, 0.78, 1.29, 0.72, 1.52, 0.62, 1.67, 1.19, 0.86)
wil_b <- c(0.53, 0.55, 0.58, 0.97, 0.60, 0.22, 0.05, 13.14, 0.63, 0.33, 0.91, 0.37)

add("wil_large_example", "dfba_wilcoxon(wil_e, wil_c)",
    quote(dfba_wilcoxon(Y1 = wil_e, Y2 = wil_c)))
add("wil_large_forced_small", "dfba_wilcoxon(wil_a, wil_b, method='large')",
    quote(dfba_wilcoxon(Y1 = wil_a, Y2 = wil_b, method = "large")))
add("wil_large_ties", "dfba_wilcoxon(c(1,2,3,4,5), c(0,1,1,2,2), method='large')",
    quote(dfba_wilcoxon(Y1 = c(1, 2, 3, 4, 5), Y2 = c(0, 1, 1, 2, 2), method = "large")))
add("wil_large_zero_diffs", "dfba_wilcoxon(c(1,2,3,4,5,6), c(1,2,0,4,5,3), method='large')",
    quote(dfba_wilcoxon(Y1 = c(1, 2, 3, 4, 5, 6), Y2 = c(1, 2, 0, 4, 5, 3), method = "large")))
add("wil_large_na", "dfba_wilcoxon(Y1 with NA[1], Y2 with NA[3], method='large')",
    quote({
      y1 <- wil_e; y1[1] <- NA
      y2 <- wil_c; y2[3] <- NA
      dfba_wilcoxon(Y1 = y1, Y2 = y2, method = "large")
    }))
add("wil_large_jeffreys", "dfba_wilcoxon(wil_e, wil_c, a0=.5, b0=.5)",
    quote(dfba_wilcoxon(Y1 = wil_e, Y2 = wil_c, a0 = .5, b0 = .5)))
add("wil_large_99", "dfba_wilcoxon(wil_e, wil_c, prob_interval=.99)",
    quote(dfba_wilcoxon(Y1 = wil_e, Y2 = wil_c, prob_interval = .99)))
add("wil_large_low_samples", "dfba_wilcoxon(wil_e, wil_c, samples=100, method='large')",
    quote(dfba_wilcoxon(Y1 = wil_e, Y2 = wil_c, samples = 100, method = "large")))
add("wil_err_length", "dfba_wilcoxon(c(1,2,3), c(1,2))",
    quote(dfba_wilcoxon(Y1 = c(1, 2, 3), Y2 = c(1, 2))))
add("wil_err_all_equal", "dfba_wilcoxon(rep(1,30), rep(1,30), method='large')",
    quote(dfba_wilcoxon(Y1 = rep(1, 30), Y2 = rep(1, 30), method = "large")))
add("wil_err_too_few", "dfba_wilcoxon(c(1,2,NA), c(0,0,0))",
    quote(dfba_wilcoxon(Y1 = c(1, 2, NA), Y2 = c(0, 0, 0))))
add("wil_err_a0_na", "dfba_wilcoxon(wil_e, wil_c, a0=NA)",
    quote(dfba_wilcoxon(Y1 = wil_e, Y2 = wil_c, a0 = NA)))
add("wil_err_b0_zero", "dfba_wilcoxon(wil_e, wil_c, b0=0)",
    quote(dfba_wilcoxon(Y1 = wil_e, Y2 = wil_c, b0 = 0)))
add("wil_err_pi_neg", "dfba_wilcoxon(wil_e, wil_c, prob_interval=-.1)",
    quote(dfba_wilcoxon(Y1 = wil_e, Y2 = wil_c, prob_interval = -.1)))
add("wil_err_pi_gt1", "dfba_wilcoxon(wil_e, wil_c, prob_interval=1.1)",
    quote(dfba_wilcoxon(Y1 = wil_e, Y2 = wil_c, prob_interval = 1.1)))
add("wil_err_pi_na", "dfba_wilcoxon(wil_e, wil_c, prob_interval=NA)",
    quote(dfba_wilcoxon(Y1 = wil_e, Y2 = wil_c, prob_interval = NA)))
add("wil_err_method", "dfba_wilcoxon(wil_e, wil_c, method='bad')",
    quote(dfba_wilcoxon(Y1 = wil_e, Y2 = wil_c, method = "bad")))

# dfba_beta_contrast: validation only (the sampling path is stochastic) ------
add("bcon_err_samples_low", "dfba_beta_contrast(c(22,15), c(18,25), c(1,-1), samples=100)",
    quote(dfba_beta_contrast(c(22, 15), c(18, 25), c(1, -1), samples = 100)))
add("bcon_err_samples_na", "dfba_beta_contrast(c(22,15), c(18,25), c(1,-1), samples=NA)",
    quote(dfba_beta_contrast(c(22, 15), c(18, 25), c(1, -1), samples = NA)))
add("bcon_err_length", "dfba_beta_contrast(c(22,15), c(18,25), c(1,-1,1))",
    quote(dfba_beta_contrast(c(22, 15), c(18, 25), c(1, -1, 1))))
add("bcon_err_n1_nonint", "dfba_beta_contrast(c(22.5,15), c(18,25), c(1,-1))",
    quote(dfba_beta_contrast(c(22.5, 15), c(18, 25), c(1, -1))))
add("bcon_err_n1_na", "dfba_beta_contrast(c(22,NA), c(18,25), c(1,-1))",
    quote(dfba_beta_contrast(c(22, NA), c(18, 25), c(1, -1))))
add("bcon_err_n2_neg", "dfba_beta_contrast(c(22,15), c(-18,25), c(1,-1))",
    quote(dfba_beta_contrast(c(22, 15), c(-18, 25), c(1, -1))))
add("bcon_err_a0_zero", "dfba_beta_contrast(c(22,15), c(18,25), c(1,-1), a0_vec=c(0,1))",
    quote(dfba_beta_contrast(c(22, 15), c(18, 25), c(1, -1), a0_vec = c(0, 1))))
add("bcon_err_a0_inf", "dfba_beta_contrast(c(22,15), c(18,25), c(1,-1), a0_vec=c(Inf,1))",
    quote(dfba_beta_contrast(c(22, 15), c(18, 25), c(1, -1), a0_vec = c(Inf, 1))))
add("bcon_err_a0_na", "dfba_beta_contrast(c(22,15), c(18,25), c(1,-1), a0_vec=c(NA,1))",
    quote(dfba_beta_contrast(c(22, 15), c(18, 25), c(1, -1), a0_vec = c(NA, 1))))
add("bcon_err_b0_na", "dfba_beta_contrast(c(22,15), c(18,25), c(1,-1), b0_vec=c(1,NA))",
    quote(dfba_beta_contrast(c(22, 15), c(18, 25), c(1, -1), b0_vec = c(1, NA))))
add("bcon_err_contrast_pos", "dfba_beta_contrast(c(22,15), c(18,25), c(.5,-1))",
    quote(dfba_beta_contrast(c(22, 15), c(18, 25), c(.5, -1))))
add("bcon_err_contrast_neg", "dfba_beta_contrast(c(22,15), c(18,25), c(1,-.5))",
    quote(dfba_beta_contrast(c(22, 15), c(18, 25), c(1, -.5))))
add("bcon_err_contrast_na", "dfba_beta_contrast(c(22,15), c(18,25), c(1,NA))",
    quote(dfba_beta_contrast(c(22, 15), c(18, 25), c(1, NA))))
add("bcon_err_pi_neg", "dfba_beta_contrast(c(22,15), c(18,25), c(1,-1), prob_interval=-.1)",
    quote(dfba_beta_contrast(c(22, 15), c(18, 25), c(1, -1), prob_interval = -.1)))
add("bcon_err_pi_na", "dfba_beta_contrast(c(22,15), c(18,25), c(1,-1), prob_interval=NA)",
    quote(dfba_beta_contrast(c(22, 15), c(18, 25), c(1, -1), prob_interval = NA)))

# dfba_sim_data -------------------------------------------------------------
sim_frozen_E <- c(1.2, 0.8, 2.1, 1.9, 0.4, 1.5, 2.8, 1.1, 0.9, 2.3,
                  1.7, 0.6, 2.5, 1.4, 0.3, 1.8, 2.0, 1.0, 2.6, 0.7)
sim_frozen_C <- c(0.5, 0.2, 1.1, 0.9, 0.1, 0.7, 1.8, 0.4, 0.3, 1.3,
                  0.8, 0.2, 1.5, 0.6, 0.1, 1.0, 1.2, 0.4, 1.6, 0.3)
add("sim_frozen_independent", "frozen E/C, independent summary", quote({
  E <- sim_frozen_E
  C <- sim_frozen_C
  tnum <- mean(E) - mean(C)
  tdem <- sqrt((var(E) + var(C)) / length(E))
  t_sample <- tnum / tdem
  list(
    pvalue = pt(t_sample, 2 * length(E) - 2, lower.tail = FALSE),
    prH1 = dfba_mann_whitney(E, C, 1, 1, method = "large")$prH1
  )
}))
add("sim_frozen_paired", "frozen E/C, paired summary", quote({
  E <- sim_frozen_E
  C <- sim_frozen_C
  d <- E - C
  t_sample <- (sqrt(length(d)) * mean(d)) / sd(d)
  list(
    pvalue = pt(t_sample, length(d) - 1, lower.tail = FALSE),
    prH1 = dfba_wilcoxon(E, C, 1, 1, method = "large")$prH1
  )
}))
add("sim_err_delta_neg", "dfba_sim_data(n=20, model='normal', design='paired', delta=-.1)",
    quote(dfba_sim_data(n = 20, model = "normal", design = "paired", delta = -.1)))
add("sim_err_n_lt20", "dfba_sim_data(n=19, model='normal', design='paired', delta=.1)",
    quote(dfba_sim_data(n = 19, model = "normal", design = "paired", delta = .1)))
add("sim_err_n_nonint", "dfba_sim_data(n=20.5, model='normal', design='paired', delta=.1)",
    quote(dfba_sim_data(n = 20.5, model = "normal", design = "paired", delta = .1)))
add("sim_err_model", "dfba_sim_data(n=20, model='bad', design='paired', delta=.1)",
    quote(dfba_sim_data(n = 20, model = "bad", design = "paired", delta = .1)))
add("sim_err_design", "dfba_sim_data(n=20, model='normal', design='bad', delta=.1)",
    quote(dfba_sim_data(n = 20, model = "normal", design = "bad", delta = .1)))
add("sim_err_block_neg", "dfba_sim_data(n=20, model='normal', design='paired', delta=.1, block_max=-1)",
    quote(dfba_sim_data(n = 20, model = "normal", design = "paired", delta = .1, block_max = -1)))
add("sim_err_a0_zero", "dfba_sim_data(n=20, model='normal', design='paired', delta=.1, a0=0)",
    quote(dfba_sim_data(n = 20, model = "normal", design = "paired", delta = .1, a0 = 0)))
add("sim_err_shape_normal", "dfba_sim_data(n=20, model='normal', design='paired', delta=.1, shape1=0)",
    quote(dfba_sim_data(n = 20, model = "normal", design = "paired", delta = .1, shape1 = 0)))
add("sim_err_shape_pareto", "dfba_sim_data(n=20, model='pareto', design='paired', delta=.1, shape1=.5)",
    quote(dfba_sim_data(n = 20, model = "pareto", design = "paired", delta = .1, shape1 = .5)))

# dfba_power_curve ----------------------------------------------------------
add("power_err_delta_step_neg", "dfba_power_curve(n=20, model='normal', design='paired', delta_step=-.1)",
    quote(dfba_power_curve(n = 20, model = "normal", design = "paired", delta_step = -.1)))
add("power_err_n_lt20", "dfba_power_curve(n=19, model='normal', design='paired')",
    quote(dfba_power_curve(n = 19, model = "normal", design = "paired")))
add("power_err_model", "dfba_power_curve(n=20, model='bad', design='paired')",
    quote(dfba_power_curve(n = 20, model = "bad", design = "paired")))
add("power_err_design", "dfba_power_curve(n=20, model='normal', design='bad')",
    quote(dfba_power_curve(n = 20, model = "normal", design = "bad")))
add("power_err_effect_crit", "dfba_power_curve(n=20, model='normal', design='paired', effect_crit=1.1)",
    quote(dfba_power_curve(n = 20, model = "normal", design = "paired", effect_crit = 1.1)))
add("power_err_a0_zero", "dfba_power_curve(n=20, model='normal', design='paired', a0=0)",
    quote(dfba_power_curve(n = 20, model = "normal", design = "paired", a0 = 0)))
add("power_err_block_neg", "dfba_power_curve(n=20, model='normal', design='paired', block_max=-1)",
    quote(dfba_power_curve(n = 20, model = "normal", design = "paired", block_max = -1)))

# dfba_bayes_vs_t_power -----------------------------------------------------
add("bvt_err_delta_neg", "dfba_bayes_vs_t_power(n_min=20, delta=-.1, model='normal', design='paired')",
    quote(dfba_bayes_vs_t_power(n_min = 20, delta = -.1, model = "normal", design = "paired")))
add("bvt_err_n_min_lt20", "dfba_bayes_vs_t_power(n_min=19, delta=.1, model='normal', design='paired')",
    quote(dfba_bayes_vs_t_power(n_min = 19, delta = .1, model = "normal", design = "paired")))
add("bvt_err_n_min_nonint", "dfba_bayes_vs_t_power(n_min=20.5, delta=.1, model='normal', design='paired')",
    quote(dfba_bayes_vs_t_power(n_min = 20.5, delta = .1, model = "normal", design = "paired")))
add("bvt_err_a0_zero", "dfba_bayes_vs_t_power(n_min=20, delta=.1, model='normal', design='paired', a0=0)",
    quote(dfba_bayes_vs_t_power(n_min = 20, delta = .1, model = "normal", design = "paired", a0 = 0)))
add("bvt_err_model", "dfba_bayes_vs_t_power(n_min=20, delta=.1, model='bad', design='paired')",
    quote(dfba_bayes_vs_t_power(n_min = 20, delta = .1, model = "bad", design = "paired")))
add("bvt_err_design", "dfba_bayes_vs_t_power(n_min=20, delta=.1, model='normal', design='bad')",
    quote(dfba_bayes_vs_t_power(n_min = 20, delta = .1, model = "normal", design = "bad")))
add("bvt_err_effect_crit", "dfba_bayes_vs_t_power(n_min=20, delta=.1, model='normal', design='paired', effect_crit=1.1)",
    quote(dfba_bayes_vs_t_power(n_min = 20, delta = .1, model = "normal", design = "paired", effect_crit = 1.1)))

## ---------------------------------------------------------------------------
## Write deterministic + large fixtures
## ---------------------------------------------------------------------------

# Record the input vectors verbatim so the Python tests never have to recreate
# R's RNG (the nH-boundary cases use rnorm() under a pinned seed). Expected
# values still come only from the calls above.
fixture_inputs <- list(
  measure_1 = measure_1,
  measure_2 = measure_2,
  group1 = group1,
  group2 = group2,
  gamma_mat = gamma_mat,
  bc_x = bc_x,
  bc_y = bc_y,
  sim_frozen_e = sim_frozen_E,
  sim_frozen_c = sim_frozen_C,
  groupA = groupA,
  groupB = groupB,
  groupC = groupC,
  groupD = groupD,
  wil_e = wil_e,
  wil_c = wil_c,
  wil_a = wil_a,
  wil_b = wil_b,
  mw_bound_e = mw_bound_e,
  mw_bound_c = mw_bound_c
)
inputs_json <- paste0(
  "{",
  paste(vapply(names(fixture_inputs), function(n) {
    paste0(json_escape(n), ":", to_json(fixture_inputs[[n]], TRUE))
  }, character(1)), collapse = ","),
  "}"
)
writeLines(inputs_json, file.path(out_dir, "inputs.json"), useBytes = TRUE)

meta <- paste0(
  '"meta":{"r_version":', json_escape(R.version.string),
  ',"dfba_version":"0.1.0"',
  ',"generated_utc":', json_escape(format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")),
  "}"
)
body <- paste0(
  "{", meta, ',"cases":{',
  paste(vapply(names(case_list), function(n) {
    paste0(json_escape(n), ":", case_list[[n]])
  }, character(1)), collapse = ","),
  "}}"
)
writeLines(body, file.path(out_dir, "fixtures.json"), useBytes = TRUE)
cat("wrote", length(case_list), "cases to", file.path(out_dir, "fixtures.json"), "\n")

## ---------------------------------------------------------------------------
## Stochastic cases: freeze the narrowest R intermediate
##
## Enabled with DFBA_STOCHASTIC=1. These are slow because they run the R
## original's Monte Carlo. The seeds are pinned here and recorded in the
## fixture manifest.
## ---------------------------------------------------------------------------

if (identical(Sys.getenv("DFBA_STOCHASTIC"), "1")) {
  stochastic <- list()

  add_stoch <- function(name, call_text, seed, expr) {
    set.seed(seed)
    stochastic[[name]] <<- run_case(name, call_text, expr)
    stochastic[[name]] <<- paste0(
      '{"seed":', seed, ",",
      substring(stochastic[[name]], 2)
    )
  }

  # dfba_beta_contrast: freeze delta and delta0 ----------------------------
  capture_beta_contrast <- function(seed, n1_vec, n2_vec, contrast_vec,
                                    a0_vec = rep(1, length(n1_vec)),
                                    b0_vec = rep(1, length(n1_vec)),
                                    prob_interval = .95, samples = 10000) {
    set.seed(seed)
    delta <- delta0 <- NULL
    trace("dfba_beta_contrast", where = asNamespace("DFBA"),
          exit = quote({
            assign(".__delta", delta, envir = .GlobalEnv)
            assign(".__delta0", delta0, envir = .GlobalEnv)
          }), print = FALSE)
    on.exit(untrace("dfba_beta_contrast", where = asNamespace("DFBA")), add = TRUE)
    out <- DFBA::dfba_beta_contrast(n1_vec, n2_vec, contrast_vec,
                              a0_vec = a0_vec, b0_vec = b0_vec,
                              prob_interval = prob_interval, samples = samples)
    out
  }

  # Two conditions (l1 = 2)
  set.seed(20261030)
  bc2 <- capture_beta_contrast(20261030, c(22, 15), c(18, 25), c(1, -1))
  stochastic[["beta_contrast_2"]] <- run_case(
    "beta_contrast_2",
    "set.seed(20261030); dfba_beta_contrast(c(22,15), c(18,25), c(1,-1), samples=10000)",
    quote(bc2)
  )
  stochastic[["beta_contrast_2"]] <- paste0(
    '{"seed":20261030,"frozen":{"delta":', to_json(.__delta),
    ',"delta0":', to_json(.__delta0), "},",
    substring(stochastic[["beta_contrast_2"]], 2)
  )

  # Four conditions (l1 = 4), orthogonal contrast
  set.seed(20261031)
  bc4 <- capture_beta_contrast(20261031, c(22, 15, 13, 21), c(18, 25, 27, 19),
                               c(.5, .5, -.5, -.5))
  stochastic[["beta_contrast_4"]] <- run_case(
    "beta_contrast_4",
    "set.seed(20261031); dfba_beta_contrast(c(22,15,13,21), c(18,25,27,19), c(.5,.5,-.5,-.5))",
    quote(bc4)
  )
  stochastic[["beta_contrast_4"]] <- paste0(
    '{"seed":20261031,"frozen":{"delta":', to_json(.__delta),
    ',"delta0":', to_json(.__delta0), "},",
    substring(stochastic[["beta_contrast_4"]], 2)
  )

  # dfba_mann_whitney small: freeze fomega --------------------------------
  capture_mw <- function(seed, E, C, ...) {
    set.seed(seed)
    fomega <- NULL
    trace("dfba_mann_whitney", where = asNamespace("DFBA"),
          exit = quote({
            if (exists("fomega", inherits = FALSE)) {
              assign(".__fomega", fomega, envir = .GlobalEnv)
            } else {
              assign(".__fomega", NULL, envir = .GlobalEnv)
            }
          }), print = FALSE)
    on.exit(untrace("dfba_mann_whitney", where = asNamespace("DFBA")), add = TRUE)
    out <- DFBA::dfba_mann_whitney(E = E, C = C, method = "small", hide_progress = TRUE, ...)
    out
  }

  # Shape A: branch length(E)*length(C) > Ntot (many ties in E/C)
  mw_small_a_E <- c(1, 1, 2, 2)
  mw_small_a_C <- c(1, 1, 2, 2)
  set.seed(20261032)
  mw_a <- capture_mw(20261032, mw_small_a_E, mw_small_a_C, samples = 10000)
  stochastic[["mw_small_branch_a"]] <- run_case(
    "mw_small_branch_a",
    "set.seed(20261032); dfba_mann_whitney(c(1,1,2,2), c(1,1,2,2), method='small', samples=10000, hide_progress=TRUE)",
    quote(mw_a)
  )
  stochastic[["mw_small_branch_a"]] <- paste0(
    '{"seed":20261032,"frozen":{"fomega":', to_json(.__fomega), "},",
    substring(stochastic[["mw_small_branch_a"]], 2)
  )

  # Shape B: branch length(E)*length(C) <= Ntot (here UE=6, UC=9, Ntot=15)
  mw_small_b_E <- c(2, 4, 6)
  mw_small_b_C <- c(1, 3, 5, 7, 9)
  set.seed(20261033)
  mw_b <- capture_mw(20261033, mw_small_b_E, mw_small_b_C, samples = 10000)
  stochastic[["mw_small_branch_b"]] <- run_case(
    "mw_small_branch_b",
    "set.seed(20261033); dfba_mann_whitney(c(2,4,6), c(1,3,5,7,9), method='small', samples=10000, hide_progress=TRUE)",
    quote(mw_b)
  )
  stochastic[["mw_small_branch_b"]] <- paste0(
    '{"seed":20261033,"frozen":{"fomega":', to_json(.__fomega), "},",
    substring(stochastic[["mw_small_branch_b"]], 2)
  )

  # dfba_wilcoxon small: freeze fomega ------------------------------------
  capture_wil <- function(seed, Y1, Y2, ...) {
    set.seed(seed)
    fphi <- NULL
    trace("dfba_wilcoxon", where = asNamespace("DFBA"),
          exit = quote({
            if (exists("fphi", inherits = FALSE)) {
              assign(".__fphi", fphi, envir = .GlobalEnv)
            } else {
              assign(".__fphi", NULL, envir = .GlobalEnv)
            }
          }), print = FALSE)
    on.exit(untrace("dfba_wilcoxon", where = asNamespace("DFBA")), add = TRUE)
    out <- DFBA::dfba_wilcoxon(Y1 = Y1, Y2 = Y2, method = "small", hide_progress = TRUE, ...)
    out
  }

  set.seed(20261034)
  wil_small <- capture_wil(20261034, c(1.49, 0.64, 0.96, 2.34, 0.78),
                           c(0.53, 0.55, 0.58, 0.97, 0.60), samples = 10000)
  stochastic[["wil_small"]] <- run_case(
    "wil_small",
    "set.seed(20261034); dfba_wilcoxon(c(1.49,0.64,0.96,2.34,0.78), c(0.53,0.55,0.58,0.97,0.60), method='small', samples=10000, hide_progress=TRUE)",
    quote(wil_small)
  )
  stochastic[["wil_small"]] <- paste0(
    '{"seed":20261034,"frozen":{"fphi":', to_json(.__fphi), "},",
    substring(stochastic[["wil_small"]], 2)
  )

  stoch_body <- paste0(
    "{", meta, ',"cases":{',
    paste(vapply(names(stochastic), function(n) {
      paste0(json_escape(n), ":", stochastic[[n]])
    }, character(1)), collapse = ","),
    "}}"
  )
  writeLines(stoch_body, file.path(out_dir, "stochastic.json"), useBytes = TRUE)
  cat("wrote", length(stochastic), "stochastic cases to",
      file.path(out_dir, "stochastic.json"), "\n")
}
