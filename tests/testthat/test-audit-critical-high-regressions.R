reconstruct_sparse_beta_draw <- function(draw, p) {
  beta <- numeric(p)
  if (is.null(draw) || length(draw) == 0) {
    return(beta)
  }

  m <- as.matrix(draw)
  if (!nrow(m) || ncol(m) < 2) {
    return(beta)
  }

  idx <- as.integer(round(m[, 1]))
  val <- as.numeric(m[, 2])
  if (any(idx == 0L)) {
    idx <- idx + 1L
  }

  ok <- is.finite(idx) & is.finite(val) & idx >= 1L & idx <= p
  beta[idx[ok]] <- val[ok]
  beta
}

test_that("HLP-1 regression: estimate_glasso_adj returns symmetric adjacency", {
  skip_if_not_installed("huge")

  set.seed(4201)
  n <- 140
  p <- 28
  z <- matrix(rnorm(n * 5), n, 5)
  w <- matrix(rnorm(5 * p), 5, p)
  X <- z %*% w + matrix(rnorm(n * p, sd = 0.25), n, p)

  adj <- estimate_glasso_adj(
    X,
    criterion = "ebic",
    nlambda = 40,
    lambda.min.ratio = 0.01,
    symmetrize = TRUE
  )

  expect_true(is.matrix(adj))
  expect_equal(adj, t(adj))
  expect_true(all(diag(adj) == 0L))
  expect_true(all(adj %in% c(0L, 1L)))
})

test_that("SPG-2 regression: dual sparse Moller uses 2 CFTP calls", {
  candidates <- c(
    file.path("src", "BayesLogit_Sparse_Helpers.h"),
    file.path("..", "src", "BayesLogit_Sparse_Helpers.h"),
    file.path("..", "..", "src", "BayesLogit_Sparse_Helpers.h")
  )
  src_path <- candidates[file.exists(candidates)][1]
  if (is.na(src_path) || !nzchar(src_path)) {
    skip("Source tree unavailable in installed test environment.")
  }

  src <- readLines(src_path, warn = FALSE)
  start <- grep("^\\[\\[maybe_unused\\]\\] static void moller_update_dual_sparse\\(", src)
  stop <- grep("^static bool validate_and_convert_y\\(", src)

  expect_length(start, 1L)
  expect_gte(length(stop), 1L)
  section <- src[start:(stop[1] - 1L)]

  n_calls <- sum(grepl("proppwilson_dual_sparse\\(", section))
  expect_equal(n_calls, 2L)
  expect_true(any(grepl("eta1_new, eta2", section, fixed = TRUE)))
  expect_true(any(grepl("eta1, eta2_new", section, fixed = TRUE)))
  expect_false(any(grepl("eta1, eta2, T_max", section, fixed = TRUE)))
})

test_that("SPG-3 regression: sparse PG PCG path reflects beta0 prior mean", {
  skip_if_not_installed("Matrix")

  set.seed(4202)
  n <- 36
  p <- 64
  X <- as(Matrix::Matrix(0, n, p, sparse = TRUE), "dgCMatrix")
  y <- rbinom(n, 1, 0.5)
  S_ggm <- Matrix::crossprod(X)

  run_fit <- function(beta0_val) {
    bvs_pg(
      X = X,
      y = y,
      adj_type = "ggm",
      sparse = TRUE,
      S_ggm = S_ggm,
      niter = 24L,
      burnin = 12L,
      thin = 1L,
      mu = 4.0,
      beta0 = beta0_val,
      n_mh_gamma = 2L,
      block_size = 4L,
      pcg_threshold = 2L,
      store_beta = TRUE,
      store_gamma = TRUE,
      store_Z_list = FALSE,
      store_Z_pip = FALSE
    )
  }

  set.seed(4203)
  fit_beta0 <- run_fit(0.0)
  set.seed(4203)
  fit_beta2 <- run_fit(2.0)

  active_sizes <- vapply(fit_beta2$gamma, length, integer(1))
  expect_true(any(active_sizes > 2L))

  mean_beta0 <- mean(vapply(
    fit_beta0$beta,
    function(draw) mean(reconstruct_sparse_beta_draw(draw, p)),
    numeric(1)
  ))
  mean_beta2 <- mean(vapply(
    fit_beta2$beta,
    function(draw) mean(reconstruct_sparse_beta_draw(draw, p)),
    numeric(1)
  ))

  expect_gt(mean_beta2, mean_beta0 + 0.3)
})
