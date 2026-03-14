test_that("locally-balanced sparse gamma proposal is wired in sparse backends", {
  candidates <- c(
    file.path("src", "BayesLogit_Sparse_Helpers.h"),
    file.path("..", "src", "BayesLogit_Sparse_Helpers.h"),
    file.path("..", "..", "src", "BayesLogit_Sparse_Helpers.h")
  )
  src_path <- candidates[file.exists(candidates)][1]
  if (is.na(src_path) || !nzchar(src_path)) {
    skip("Source tree unavailable in installed test environment.")
  }

  hdr <- readLines(src_path, warn = FALSE)
  expect_true(any(grepl("init_lb_single_scores\\(", hdr)))
  expect_true(any(grepl("init_lb_dual_scores\\(", hdr)))
  expect_true(any(grepl("build_lb_single_delta\\(", hdr)))
  expect_true(any(grepl("build_lb_dual_delta\\(", hdr)))
  expect_true(any(grepl("apply_lb_delta\\(", hdr)))

  cpps <- c(
    "BayesLogit_SingleNet_SparseGGM.cpp",
    "BayesLogit_DualNet_SparseGGM.cpp",
    "BayesLogit_PG_SingleAdj_SparseGGM.cpp",
    "BayesLogit_PG_DualNet_SparseGGM.cpp"
  )
  src_dir <- dirname(src_path)
  for (f in cpps) {
    lines <- readLines(file.path(src_dir, f), warn = FALSE)
    expect_true(any(grepl("bool use_lb_gamma = true", lines, fixed = TRUE)))
  }
})

test_that("use_lb_gamma argument is validated in R wrappers", {
  set.seed(9011)
  n <- 20
  p <- 8
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)
  R <- matrix(0L, p, p)

  # NA should be rejected
  expect_error(
    bvs_pg(X, y, adj_type = "fixed", adj_fixed = R, niter = 4L, burnin = 2L,
           use_lb_gamma = NA),
    "use_lb_gamma"
  )
  expect_error(
    bvs_mh(X, y, adj_type = "fixed", adj_fixed = R, niter = 4L, burnin = 2L,
           use_lb_gamma = NA),
    "use_lb_gamma"
  )
  # Non-logical types should be rejected
  expect_error(
    bvs_pg(X, y, adj_type = "fixed", adj_fixed = R, niter = 4L, burnin = 2L,
           use_lb_gamma = 1),
    "use_lb_gamma"
  )
  expect_error(
    bvs_mh(X, y, adj_type = "fixed", adj_fixed = R, niter = 4L, burnin = 2L,
           use_lb_gamma = "yes"),
    "use_lb_gamma"
  )
  # Logical vectors of length > 1 should be rejected
  expect_error(
    bvs_pg(X, y, adj_type = "fixed", adj_fixed = R, niter = 4L, burnin = 2L,
           use_lb_gamma = c(TRUE, FALSE)),
    "use_lb_gamma"
  )
  expect_error(
    bvs_mh(X, y, adj_type = "fixed", adj_fixed = R, niter = 4L, burnin = 2L,
           use_lb_gamma = c(TRUE, FALSE)),
    "use_lb_gamma"
  )
})

test_that("targeted ESS benchmark runs for sparse locally-balanced gamma", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  skip_if_not_installed("coda")

  # Use p=30 with moderate density so the uniform proposal reliably visits
  # signal variables within the allowed iteration budget (niter=120, 60 saves).
  set.seed(1401)
  n <- 60
  p <- 30
  X <- Matrix::rsparsematrix(n, p, density = 0.15)
  beta_true <- c(1.2, -1.0, 0.8, 0.0, 0.0)
  eta <- as.numeric(X[, 1:5] %*% beta_true)
  y <- rbinom(n, 1, plogis(eta))
  S_ggm <- Matrix::crossprod(X)

  fit_with_toggle <- function(use_lb_gamma) {
    set.seed(1402)
    bvs_pg(
      X = X,
      y = y,
      adj_type = "ggm",
      sparse = TRUE,
      S_ggm = S_ggm,
      niter = 120L,
      burnin = 60L,
      thin = 1L,
      n_mh_gamma = 4L,
      block_size = 1L,
      use_lb_gamma = use_lb_gamma,
      store_beta = FALSE,
      store_gamma = TRUE,
      store_Z_list = FALSE,
      store_Z_pip = FALSE
    )
  }

  fit_uniform <- fit_with_toggle(FALSE)
  fit_lb <- fit_with_toggle(TRUE)

  gamma_chain <- function(fit, j) {
    vapply(
      fit$gamma,
      function(draw) {
        idx <- BVS.DAdj:::.normalize_sparse_idx(draw, fit$p)
        as.numeric(j %in% idx)
      },
      numeric(1)
    )
  }

  ess_for_fit <- function(fit, vars = 1:5) {
    vapply(
      vars,
      function(j) {
        chain <- gamma_chain(fit, j)
        as.numeric(coda::effectiveSize(chain))
      },
      numeric(1)
    )
  }

  ess0 <- ess_for_fit(fit_uniform)
  ess1 <- ess_for_fit(fit_lb)
  fin0 <- ess0[is.finite(ess0) & ess0 > 0]
  fin1 <- ess1[is.finite(ess1) & ess1 > 0]

  # At least one signal variable must mix in each mode.
  expect_gt(length(fin0), 0L)
  expect_gt(length(fin1), 0L)

  # Benchmark guard: locally balanced mode should not collapse ESS.
  med0 <- stats::median(fin0)
  med1 <- stats::median(fin1)
  expect_gte(med1, 0.5 * med0)
})

test_that("targeted ESS benchmark runs for sparse MH locally-balanced gamma", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  skip_if_not_installed("coda")

  set.seed(2401)
  n <- 60
  p <- 120
  X <- Matrix::rsparsematrix(n, p, density = 0.06)
  beta_true <- c(0.9, -0.7, 0.6, 0.0, 0.0)
  eta <- as.numeric(X[, 1:5] %*% beta_true)
  y <- rbinom(n, 1, plogis(eta))
  S_ggm <- Matrix::crossprod(X)

  fit_with_toggle <- function(use_lb_gamma) {
    set.seed(2402)
    bvs_mh(
      X = X,
      y = y,
      outcome_type = "binary",
      adj_type = "ggm",
      sparse = TRUE,
      S_ggm = S_ggm,
      niter = 80L,
      burnin = 40L,
      thin = 1L,
      n_mh_gamma = 4L,
      use_lb_gamma = use_lb_gamma,
      store_beta = FALSE,
      store_gamma = TRUE,
      store_Z_list = FALSE,
      store_Z_pip = TRUE
    )
  }

  fit_uniform <- fit_with_toggle(FALSE)
  fit_lb <- fit_with_toggle(TRUE)

  gamma_chain <- function(fit, j) {
    vapply(
      fit$gamma,
      function(draw) {
        idx <- BVS.DAdj:::.normalize_sparse_idx(draw, fit$p)
        as.numeric(j %in% idx)
      },
      numeric(1)
    )
  }

  ess_for_fit <- function(fit, vars = 1:5) {
    vapply(
      vars,
      function(j) {
        chain <- gamma_chain(fit, j)
        as.numeric(coda::effectiveSize(chain))
      },
      numeric(1)
    )
  }

  ess0 <- ess_for_fit(fit_uniform)
  ess1 <- ess_for_fit(fit_lb)
  fin0 <- ess0[is.finite(ess0) & ess0 > 0]
  fin1 <- ess1[is.finite(ess1) & ess1 > 0]

  # At least one signal variable must mix in each mode.
  expect_gt(length(fin0), 0L)
  expect_gt(length(fin1), 0L)
  expect_gte(stats::median(fin1), 0.5 * stats::median(fin0))
})
