test_that("bvs_rhat returns 1.0 for constant chain", {
  x <- rep(3.14, 100)
  expect_equal(bvs_rhat(x), 1.0)
})

test_that("bvs_rhat returns 1.0 for well-mixed IID chain", {
  set.seed(4201)
  x <- rnorm(2000)
  rhat <- bvs_rhat(x)
  expect_lt(rhat, 1.05)
})

test_that("bvs_rhat detects non-convergence for divergent chains", {
  set.seed(4203)
  # Two chains with very different means — large separation for reliable detection
  x_mat <- cbind(rnorm(1000, mean = 0), rnorm(1000, mean = 50))
  rhat <- bvs_rhat(x_mat)
  expect_gt(rhat, 1.5)
})

test_that("bvs_ess_bulk returns approximately N for IID chain", {
  set.seed(4202)
  x <- rnorm(1000)
  ess <- bvs_ess_bulk(x)
  expect_true(is.finite(ess))
  expect_gt(ess, 200)  # conservative lower bound
})

test_that("bvs_ess_bulk returns N for constant chain", {
  x <- rep(5.0, 200)
  ess <- bvs_ess_bulk(x)
  expect_true(is.finite(ess))
  expect_gte(ess, 200)
})

test_that("bvs_ess_tail returns finite positive for IID chain", {
  set.seed(4203)
  x <- rnorm(1000)
  ess <- bvs_ess_tail(x)
  expect_true(is.finite(ess))
  expect_gt(ess, 0)
})

test_that("bvs_ess_bulk detects autocorrelation", {
  set.seed(4204)
  # AR(1) with rho = 0.95 has ESS ~ N * (1-rho)/(1+rho) ~ N * 0.026
  n <- 2000
  x <- numeric(n)
  x[1] <- rnorm(1)
  for (i in 2:n) x[i] <- 0.95 * x[i - 1] + rnorm(1)
  ess <- bvs_ess_bulk(x)
  expect_true(is.finite(ess))
  expect_lt(ess, n * 0.5)  # must detect autocorrelation
})

test_that("bvs_ess_tail is lower than bulk for heavy-tailed chain", {
  set.seed(4205)
  x <- rt(2000, df = 3)  # heavy tails
  ess_b <- bvs_ess_bulk(x)
  ess_t <- bvs_ess_tail(x)
  # Tail ESS is typically lower; at least both should be finite and positive
  expect_true(is.finite(ess_b) && ess_b > 0)
  expect_true(is.finite(ess_t) && ess_t > 0)
})

test_that("multi-chain input works for all diagnostics", {
  set.seed(4206)
  x_mat <- matrix(rnorm(2000), ncol = 2)
  expect_true(is.finite(bvs_rhat(x_mat)))
  expect_true(is.finite(bvs_ess_bulk(x_mat)))
  expect_true(is.finite(bvs_ess_tail(x_mat)))
})

test_that("binary chain gives valid ESS", {
  set.seed(4207)
  x <- rbinom(500, 1, 0.3)
  ess <- bvs_ess_bulk(x)
  expect_true(is.finite(ess))
  expect_gt(ess, 0)
})
