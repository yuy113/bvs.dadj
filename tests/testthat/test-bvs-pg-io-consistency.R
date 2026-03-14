test_that("bvs_pg returns documented core input/output metadata", {
  set.seed(808)
  n <- 48
  p <- 18
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(rep(1, 4), rep(0, p - 4))
  y <- rbinom(n, 1, plogis(as.numeric(X %*% beta_true) / 3))

  R <- matrix(0L, p, p)
  R[1:6, 1:6] <- 1L
  diag(R) <- 0L

  fit <- bvs_pg(
    X, y,
    adj_type = "fixed",
    adj_fixed = R,
    niter = 6L,
    burnin = 3L
  )

  expect_s3_class(fit, "bvs")
  expect_identical(fit$sampler, "pg")
  expect_identical(fit$adj_type, "fixed")
  expect_equal(fit$niter, 6L)
  expect_equal(fit$burnin, 3L)
  expect_equal(fit$n, n)
  expect_equal(fit$p, p)
  expect_equal(fit$ntau, 1L)
  expect_length(fit$alpha, 6L)
  expect_length(fit$sigmasq, 6L)
  expect_length(fit$gamma_pip, p)
  expect_identical(fit$outcome_type, "binary")
  expect_identical(fit$thin, 1L)
})
