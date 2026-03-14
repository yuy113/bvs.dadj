test_that("HLP-2 regression: .prepare_adj preserves negative edges", {
  p <- 5L
  adj_in <- matrix(0, p, p)
  adj_in[1, 3] <- -1
  adj_in[4, 2] <- -2
  adj_in[5, 5] <- 9

  adj_out <- BVS.DAdj:::.prepare_adj(adj_in, p, "adj_in")

  expect_type(adj_out, "integer")
  expect_equal(dim(adj_out), c(p, p))
  expect_equal(adj_out, t(adj_out))
  expect_true(all(diag(adj_out) == 0L))
  expect_identical(adj_out[1, 3], 1L)
  expect_identical(adj_out[3, 1], 1L)
  expect_identical(adj_out[4, 2], 1L)
  expect_identical(adj_out[2, 4], 1L)
})

test_that("RMH-3 regression: bvs_pg validates niter, burnin, thin", {
  set.seed(901)
  n <- 20
  p <- 8
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)
  R <- matrix(0L, p, p)

  expect_error(
    bvs_pg(X, y, adj_type = "fixed", adj_fixed = R, niter = 0L, burnin = 1L),
    regexp = "niter|is\\.numeric|assertion|TRUE"
  )
  expect_error(
    bvs_pg(X, y, adj_type = "fixed", adj_fixed = R, niter = 4L, burnin = -1L),
    regexp = "burnin|is\\.numeric|assertion|TRUE"
  )
  expect_error(
    bvs_pg(X, y, adj_type = "fixed", adj_fixed = R, niter = 4L, burnin = 1L, thin = 0L),
    regexp = "thin|is\\.numeric|assertion|TRUE"
  )
})

test_that("RMH-3 regression: bvs_mh validates niter, burnin, thin", {
  set.seed(902)
  n <- 20
  p <- 8
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)
  R <- matrix(0L, p, p)

  expect_error(
    bvs_mh(X, y, adj_type = "fixed", adj_fixed = R, niter = 0L, burnin = 1L),
    regexp = "niter|is\\.numeric|assertion|TRUE"
  )
  expect_error(
    bvs_mh(X, y, adj_type = "fixed", adj_fixed = R, niter = 4L, burnin = -1L),
    regexp = "burnin|is\\.numeric|assertion|TRUE"
  )
  expect_error(
    bvs_mh(X, y, adj_type = "fixed", adj_fixed = R, niter = 4L, burnin = 1L, thin = 0L),
    regexp = "thin|is\\.numeric|assertion|TRUE"
  )
})

build_stub_bvs <- function(thin = 1L) {
  set.seed(903)
  fit <- list(
    beta = matrix(rnorm(40), nrow = 10, ncol = 4),
    gamma = matrix(rbinom(40, 1, 0.4), nrow = 10, ncol = 4),
    alpha = rnorm(10),
    tau = matrix(rnorm(20), nrow = 10, ncol = 2),
    gamma_pip = c(0.6, 0.2, 0.7, 0.1),
    sampler = "mh",
    adj_type = "fixed",
    p = 4L,
    n = 25L,
    niter = 10L,
    thin = as.integer(thin)
  )
  class(fit) <- "bvs"
  fit
}

test_that("SUM-2 regression: as.mcmc.bvs honors fit thinning interval", {
  fit <- build_stub_bvs(thin = 5L)
  mc <- coda::as.mcmc(fit)
  expect_s3_class(mc, "mcmc")
  expect_identical(as.integer(attr(mc, "mcpar")[3]), 5L)
})

test_that("SUM-1 regression: tau summary quantiles follow cred_level", {
  fit <- build_stub_bvs(thin = 1L)
  s <- summary(fit, cred_level = 0.80, hpd_level = 0.90)

  expect_true(is.data.frame(s$tau_summary))
  expect_true("Q10" %in% colnames(s$tau_summary))
  expect_true("Q90" %in% colnames(s$tau_summary))
})

test_that("NUM-2 regression: Cox linpred clamp is tightened in numerics", {
  candidates <- c(
    file.path("src", "BayesLogit_Numerics.h"),
    file.path("..", "src", "BayesLogit_Numerics.h"),
    file.path("..", "..", "src", "BayesLogit_Numerics.h")
  )
  src_path <- candidates[file.exists(candidates)][1]
  if (is.na(src_path) || !nzchar(src_path)) {
    skip("Source tree unavailable in installed test environment.")
  }

  src <- readLines(src_path, warn = FALSE)
  expect_true(any(grepl("clamp_finite\\(linpred\\(i\\), -500\\.0, 500\\.0, 0\\.0\\)", src)))
  expect_true(any(grepl("clamp_finite\\(linpred_\\(i\\), -500\\.0, 500\\.0, 0\\.0\\)", src)))
  expect_false(any(grepl("clamp_finite\\(linpred\\(i\\), -1e6, 1e6, 0\\.0\\)", src)))
  expect_false(any(grepl("clamp_finite\\(linpred_\\(i\\), -1e6, 1e6, 0\\.0\\)", src)))
})

test_that("PCG warm-start regression: active beta initializes CG state", {
  hdr_candidates <- c(
    file.path("src", "BayesLogit_BlockPG.h"),
    file.path("..", "src", "BayesLogit_BlockPG.h"),
    file.path("..", "..", "src", "BayesLogit_BlockPG.h")
  )
  hdr_path <- hdr_candidates[file.exists(hdr_candidates)][1]
  if (is.na(hdr_path) || !nzchar(hdr_path)) {
    skip("Source tree unavailable in installed test environment.")
  }
  hdr <- readLines(hdr_path, warn = FALSE)
  expect_gte(sum(grepl("beta_out\\.n_elem\\) == p_act && beta_out\\.is_finite\\(\\)", hdr)), 2L)
  expect_false(any(grepl("arma::vec x = arma::zeros<arma::vec>\\(p_act\\)", hdr)))

  dense_files <- c(
    "BayesLogit_PG_DualAdj.cpp",
    "BayesLogit_PG_SingleAdj.cpp",
    "BayesLogit_PG_GGM_Moller.cpp",
    "BayesLogit_PG_SingleAdj_GGM_Moller.cpp"
  )
  for (f in dense_files) {
    path <- file.path(dirname(hdr_path), f)
    lines <- readLines(path, warn = FALSE)
    has_beta_warmstart <- any(grepl("arma::vec beta_act = beta\\.elem\\(active\\);", lines)) ||
      any(grepl("arma::vec beta_act = beta_vec\\.elem\\(active_uv\\);", lines))
    expect_true(has_beta_warmstart)
  }

  sparse_files <- c(
    "BayesLogit_PG_DualNet_SparseGGM.cpp",
    "BayesLogit_PG_SingleAdj_SparseGGM.cpp"
  )
  for (f in sparse_files) {
    path <- file.path(dirname(hdr_path), f)
    lines <- readLines(path, warn = FALSE)
    expect_true(any(grepl("beta_act\\(k\\) = beta_vec\\(active_idx\\[k\\]\\);", lines)))
  }
})
