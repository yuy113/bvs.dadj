test_that("count mode works for dense fixed adjacency", {
  set.seed(101)
  n <- 60
  p <- 16
  X <- matrix(rnorm(n * p), n, p)
  R <- matrix(0L, p, p)
  R[1:5, 1:5] <- 1L
  diag(R) <- 0L

  mu_count <- exp(0.2 + X[, 1] / 4 - X[, 2] / 5)
  y <- rnbinom(n, size = 3, mu = mu_count)

  fit <- bvs_mh(
    X, y,
    outcome_type = "count",
    adj_type = "fixed",
    adj_fixed = R,
    niter = 8,
    burnin = 4
  )

  expect_s3_class(fit, "bvs")
  expect_identical(fit$outcome_type, "count")
  expect_length(fit$alpha, 8)
  expect_equal(fit$p, p)
  expect_equal(fit$n, n)
  expect_true(all(is.finite(fit$alpha)))
})

test_that("count mode runs in both dense and sparse ggm backends", {
  skip_if_not_installed("Matrix")

  set.seed(202)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- rnbinom(n, size = 2.5, mu = exp(0.1 + X[, 1] / 3 - X[, 2] / 6))

  fit_dense <- bvs_mh(
    X, y,
    outcome_type = "count",
    adj_type = "ggm",
    sparse = FALSE,
    niter = 6,
    burnin = 3
  )

  X_sp <- as(Matrix::Matrix(X, sparse = TRUE), "dgCMatrix")
  S_ggm <- Matrix::crossprod(X_sp)
  fit_sparse <- bvs_mh(
    X_sp, y,
    outcome_type = "count",
    adj_type = "ggm",
    sparse = TRUE,
    S_ggm = S_ggm,
    niter = 6,
    burnin = 3,
    store_beta = FALSE,
    store_gamma = FALSE,
    store_Z_list = FALSE,
    store_Z_pip = TRUE
  )

  expect_identical(fit_dense$outcome_type, "count")
  expect_identical(fit_sparse$outcome_type, "count")
  expect_length(fit_dense$alpha, 6)
  expect_length(fit_sparse$alpha, 6)
  expect_equal(fit_dense$p, p)
  expect_equal(fit_sparse$p, p)
})

test_that("sparse count mode enforces p >= 10000 S_ggm guard", {
  skip_if_not_installed("Matrix")
  skip_on_cran()

  set.seed(303)
  n <- 40
  p <- 10001
  X <- Matrix::rsparsematrix(n, p, density = 0.001)
  y <- rnbinom(n, size = 2.5, mu = 2.0)

  expect_error(
    bvs_mh(
      X, y,
      outcome_type = "count",
      adj_type = "ggm",
      sparse = TRUE,
      niter = 4,
      burnin = 2
    ),
    "p >= 10000"
  )

  S_ggm <- Matrix::crossprod(X)
  fit <- bvs_mh(
    X, y,
    outcome_type = "count",
    adj_type = "ggm",
    sparse = TRUE,
    S_ggm = S_ggm,
    niter = 4,
    burnin = 2,
    store_beta = FALSE,
    store_gamma = FALSE,
    store_Z_list = FALSE,
    store_Z_pip = TRUE
  )

  expect_identical(fit$outcome_type, "count")
  expect_equal(fit$p, p)
  expect_length(fit$alpha, 4)
})
