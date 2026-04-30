# test-bvs-mh-zinb.R
# Tests for outcome_type = "ZIC" (Zero-Inflated Negative Binomial)

local_zinb_fixture <- function(n = 60L, p = 12L, seed = 808L,
                               pi_true = 0.3, r_true = 2.0) {
  skip_if_not_installed("Matrix")
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  X_sp <- as(Matrix::Matrix(X, sparse = TRUE), "dgCMatrix")
  S_ggm <- Matrix::crossprod(X_sp)

  R1 <- matrix(0L, p, p)
  R1[1:4, 1:4] <- 1L
  diag(R1) <- 0L

  R2 <- matrix(0L, p, p)
  R2[3:7, 3:7] <- 1L
  diag(R2) <- 0L

  beta_true <- c(rep(0.8, 3), rep(0, p - 3))
  mu <- exp(0.5 + as.numeric(X %*% beta_true))
  z_true <- rbinom(n, 1, pi_true)
  y <- ifelse(z_true == 1, 0L, rnbinom(n, size = r_true, mu = mu))

  list(
    X = X, X_sp = X_sp, S_ggm = S_ggm, R1 = R1, R2 = R2,
    y = y, n = n, p = p, pi_true = pi_true, r_true = r_true,
    zero_frac = mean(y == 0)
  )
}

# ==========================================================================
# 1. Smoke tests: ZIC across dense backends with fixed r
# ==========================================================================
test_that("ZIC smoke-runs across all dense MH backends (fixed r)", {
  skip_on_cran()
  fx <- local_zinb_fixture()

  backend_specs <- list(
    list(name = "fixed", adj_type = "fixed", x = fx$X,
         extra = list(adj_fixed = fx$R1)),
    list(name = "dual_fixed", adj_type = "dual_fixed", x = fx$X,
         extra = list(adj_fixed = fx$R1, adj_fixed2 = fx$R2)),
    list(name = "ggm_dense", adj_type = "ggm", x = fx$X,
         extra = list()),
    list(name = "ggm_fixed_dense", adj_type = "ggm_fixed", x = fx$X,
         extra = list(adj_fixed = fx$R1))
  )

  for (bs in backend_specs) {
    label <- paste0("ZIC/", bs$name)
    args <- c(list(
      X = bs$x, y = fx$y,
      outcome_type = "ZIC",
      adj_type = bs$adj_type,
      zinb_r = 2.0,
      niter = 200L, burnin = 50L, thin = 1L
    ), bs$extra)
    fit <- do.call(bvs_mh, args)

    expect_s3_class(fit, "bvs")
    expect_true(is.numeric(fit$gamma_pip), label = label)
    expect_equal(length(fit$gamma_pip), fx$p, label = label)
    expect_true(all(fit$gamma_pip >= 0 & fit$gamma_pip <= 1), label = label)
    expect_true(all(is.finite(fit$beta)), label = label)
  }
})

# ==========================================================================
# 2. ZIC with estimated r (MH)
# ==========================================================================
test_that("ZIC works with zinb_estimate_r = TRUE", {
  skip_on_cran()
  fx <- local_zinb_fixture()

  fit <- bvs_mh(
    fx$X, fx$y,
    outcome_type = "ZIC",
    adj_type = "fixed", adj_fixed = fx$R1,
    zinb_r = 1.0, zinb_estimate_r = TRUE,
    niter = 300L, burnin = 100L
  )
  expect_s3_class(fit, "bvs")
  expect_true(all(is.finite(fit$gamma_pip)))
  expect_true(!is.null(fit$zinb_r_final))
  expect_true(fit$zinb_r_final > 0)
})

# ==========================================================================
# 3. ZIC returns zinb_pi and zinb_r_final
# ==========================================================================
test_that("ZIC returns zinb_pi and zinb_r_final", {
  skip_on_cran()
  fx <- local_zinb_fixture()

  fit <- bvs_mh(
    fx$X, fx$y,
    outcome_type = "ZIC",
    adj_type = "fixed", adj_fixed = fx$R1,
    zinb_r = 2.0,
    niter = 200L, burnin = 50L
  )
  expect_true(!is.null(fit$zinb_pi))
  expect_true(fit$zinb_pi > 0 && fit$zinb_pi < 1)
  expect_true(!is.null(fit$zinb_r_final))
  expect_true(fit$zinb_r_final > 0)
})

# ==========================================================================
# 4. Non-ZIC outcomes don't return ZINB fields
# ==========================================================================
test_that("non-ZIC outcomes have NULL zinb fields", {
  skip_on_cran()
  fx <- local_zinb_fixture()

  fit <- bvs_mh(
    fx$X, fx$y,
    outcome_type = "count",
    adj_type = "fixed", adj_fixed = fx$R1,
    niter = 200L, burnin = 50L
  )
  expect_null(fit$zinb_pi)
  expect_null(fit$zinb_r_final)
})

# ==========================================================================
# 5. ZIC input validation
# ==========================================================================
test_that("ZIC rejects non-integer y", {
  skip_on_cran()
  fx <- local_zinb_fixture()
  y_bad <- rnorm(fx$n)
  expect_error(
    bvs_mh(fx$X, y_bad, outcome_type = "ZIC",
            adj_type = "fixed", adj_fixed = fx$R1,
            niter = 50L, burnin = 10L),
    "must be finite non-negative integers"
  )
})

test_that("ZIC rejects negative y", {
  skip_on_cran()
  fx <- local_zinb_fixture()
  y_neg <- fx$y
  y_neg[1] <- -1L
  expect_error(
    bvs_mh(fx$X, y_neg, outcome_type = "ZIC",
            adj_type = "fixed", adj_fixed = fx$R1,
            niter = 50L, burnin = 10L),
    "must be finite non-negative integers"
  )
})

test_that("ZIC rejects invalid zinb_r", {
  skip_on_cran()
  fx <- local_zinb_fixture()
  expect_error(
    bvs_mh(fx$X, fx$y, outcome_type = "ZIC",
            adj_type = "fixed", adj_fixed = fx$R1,
            zinb_r = -1, niter = 50L, burnin = 10L),
    "zinb_r"
  )
})

# ==========================================================================
# 6. ZINB prior parameters
# ==========================================================================
test_that("ZIC works with custom Beta prior on pi", {
  skip_on_cran()
  fx <- local_zinb_fixture()

  fit <- bvs_mh(
    fx$X, fx$y,
    outcome_type = "ZIC",
    adj_type = "fixed", adj_fixed = fx$R1,
    zinb_a_pi = 2.0, zinb_b_pi = 5.0,
    niter = 200L, burnin = 50L
  )
  expect_s3_class(fit, "bvs")
  expect_true(fit$zinb_pi > 0 && fit$zinb_pi < 1)
})

# ==========================================================================
# 7. Sparse backend smoke test
# ==========================================================================
test_that("ZIC works with sparse GGM backends", {
  skip_on_cran()
  fx <- local_zinb_fixture()

  fit <- bvs_mh(
    fx$X_sp, fx$y,
    outcome_type = "ZIC",
    adj_type = "ggm", sparse = TRUE,
    S_ggm = fx$S_ggm,
    store_beta = FALSE, store_gamma = FALSE,
    store_Z_list = FALSE, store_Z_pip = TRUE,
    zinb_r = 2.0,
    niter = 200L, burnin = 50L
  )
  expect_s3_class(fit, "bvs")
  expect_true(all(is.finite(fit$gamma_pip)))
})

# ==========================================================================
# 8. All-zeros y should produce high pi estimate
# ==========================================================================
test_that("ZIC handles high zero fraction gracefully", {
  skip_on_cran()
  fx <- local_zinb_fixture()
  y_mostly_zero <- fx$y
  y_mostly_zero[y_mostly_zero > 0] <- 0L
  y_mostly_zero[1] <- 1L  # at least one non-zero

  fit <- bvs_mh(
    fx$X, y_mostly_zero,
    outcome_type = "ZIC",
    adj_type = "fixed", adj_fixed = fx$R1,
    niter = 200L, burnin = 50L
  )
  expect_s3_class(fit, "bvs")
  expect_true(all(is.finite(fit$beta)))
})

# ==========================================================================
# 9. Backward compatibility: existing count outcome unaffected
# ==========================================================================
test_that("standard count outcome is unaffected by ZINB additions", {
  skip_on_cran()
  fx <- local_zinb_fixture()
  fit <- bvs_mh(
    fx$X, fx$y,
    outcome_type = "count",
    adj_type = "fixed", adj_fixed = fx$R1,
    niter = 200L, burnin = 50L
  )
  expect_s3_class(fit, "bvs")
  expect_null(fit$zinb_pi)
})
