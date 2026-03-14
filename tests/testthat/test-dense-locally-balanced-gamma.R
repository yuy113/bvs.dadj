test_that("use_lb_gamma parameter exists in dense FixedAdj function signatures", {
  candidates <- c(
    file.path("src", "BayesLogit_Sparse_Helpers.h"),
    file.path("..", "src", "BayesLogit_Sparse_Helpers.h"),
    file.path("..", "..", "src", "BayesLogit_Sparse_Helpers.h")
  )
  src_path <- candidates[file.exists(candidates)][1]
  if (is.na(src_path) || !nzchar(src_path)) {
    skip("Source tree unavailable in installed test environment.")
  }

  src_dir <- dirname(src_path)
  dense_cpps <- c(
    "BayesLogit_SingleNet_FixedAdj.cpp",
    "BayesLogit_DualNet_FixedAdj.cpp",
    "BayesLogit_PG_SingleAdj.cpp",
    "BayesLogit_PG_DualAdj.cpp",
    "BayesLogit_SingleNet_GGM.cpp",
    "BayesLogit_DualNet_GGM.cpp",
    "BayesLogit_PG_SingleAdj_GGM_Moller.cpp",
    "BayesLogit_PG_GGM_Moller.cpp"
  )
  for (f in dense_cpps) {
    fpath <- file.path(src_dir, f)
    if (!file.exists(fpath)) next
    lines <- readLines(fpath, warn = FALSE)
    expect_true(
      any(grepl("use_lb_gamma", lines)),
      info = paste("use_lb_gamma missing in", f)
    )
  }
})

test_that("dense LB helper functions exist in Sparse_Helpers.h", {
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
  expect_true(any(grepl("init_lb_single_scores_dense\\(", hdr)))
  expect_true(any(grepl("build_lb_single_delta_dense\\(", hdr)))
  expect_true(any(grepl("init_lb_dual_scores_dense\\(", hdr)))
  expect_true(any(grepl("build_lb_dual_delta_dense\\(", hdr)))
  expect_true(any(grepl("init_lb_single_scores_ggm\\(", hdr)))
  expect_true(any(grepl("build_lb_single_delta_ggm\\(", hdr)))
  expect_true(any(grepl("init_lb_dual_scores_ggm\\(", hdr)))
  expect_true(any(grepl("build_lb_dual_delta_ggm\\(", hdr)))
})

test_that("use_lb_gamma argument is validated for dense backends", {
  set.seed(8011)
  n <- 20
  p <- 8
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)
  R <- matrix(0L, p, p)

  # NA should be rejected (fixed adj uses dense backend)
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
})

test_that("dense fixed-adj PG backend runs with use_lb_gamma=TRUE", {
  skip_on_cran()
  set.seed(8012)
  n <- 30
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1, -0.8, rep(0, p - 2))
  y <- rbinom(n, 1, plogis(X %*% beta_true))
  R <- diag(0L, p)
  R[1, 2] <- R[2, 1] <- 1L

  fit <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R,
                niter = 20L, burnin = 10L,
                use_lb_gamma = TRUE)
  expect_s3_class(fit, "bvs")
  expect_equal(fit$p, p)
})

test_that("dense fixed-adj MH backend runs with use_lb_gamma=TRUE", {
  skip_on_cran()
  set.seed(8013)
  n <- 30
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1, -0.8, rep(0, p - 2))
  y <- rbinom(n, 1, plogis(X %*% beta_true))
  R <- diag(0L, p)
  R[1, 2] <- R[2, 1] <- 1L

  fit <- bvs_mh(X, y, outcome_type = "binary",
                adj_type = "fixed", adj_fixed = R,
                niter = 20L, burnin = 10L,
                use_lb_gamma = TRUE)
  expect_s3_class(fit, "bvs")
  expect_equal(fit$p, p)
})

test_that("dense dual-fixed PG backend runs with use_lb_gamma=TRUE", {
  skip_on_cran()
  set.seed(8014)
  n <- 30
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)
  R1 <- diag(0L, p); R1[1, 2] <- R1[2, 1] <- 1L
  R2 <- diag(0L, p); R2[3, 4] <- R2[4, 3] <- 1L

  fit <- bvs_pg(X, y, adj_type = "dual_fixed",
                adj_fixed = R1, adj_fixed2 = R2,
                niter = 20L, burnin = 10L,
                use_lb_gamma = TRUE)
  expect_s3_class(fit, "bvs")
})
