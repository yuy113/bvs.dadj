local_hmc_fixture <- function(n = 32L, p = 12L, seed = 818L) {
  skip_if_not_installed("Matrix")
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  X_sp <- as(Matrix::Matrix(X, sparse = TRUE), "dgCMatrix")
  S_ggm <- Matrix::crossprod(X_sp)

  R1 <- matrix(0L, p, p)
  R1[1:4, 1:4] <- 1L
  diag(R1) <- 0L

  R2 <- matrix(0L, p, p)
  R2[5:8, 5:8] <- 1L
  diag(R2) <- 0L

  Z <- matrix(rnorm(n * 2L), n, 2L)
  beta_true <- c(1.1, -0.8, 0.6, rep(0, p - 3L))
  tau_true <- c(0.4, -0.3)
  alpha_true <- -0.15
  eta <- as.numeric(alpha_true + X %*% beta_true + Z %*% tau_true)

  y_binary <- rbinom(n, 1, plogis(eta / 3))
  y_tte <- rexp(n, rate = exp((eta - alpha_true) / 6))
  event <- rbinom(n, 1, 0.75)
  if (sum(event == 1L) < 2L) {
    event[1:2] <- 1L
  }

  list(
    X = X, X_sp = X_sp, S_ggm = S_ggm, R1 = R1, R2 = R2, Z = Z,
    y_binary = y_binary, y_tte = y_tte, event = event, p = p, n = n
  )
}

test_that("bvs_mh runs HMC and NUTS for binary and TTE across backends", {
  skip_on_cran()
  fx <- local_hmc_fixture()

  specs <- list(
    list(
      name = "fixed_hmc_binary",
      args = list(
        X = fx$X, y = fx$y_binary, z_dat = fx$Z,
        outcome_type = "binary", adj_type = "fixed", adj_fixed = fx$R1,
        alg_type = "HMC"
      )
    ),
    list(
      name = "dual_fixed_nuts_tte",
      args = list(
        X = fx$X, y = fx$y_tte, event = fx$event, z_dat = fx$Z,
        outcome_type = "TTE", adj_type = "dual_fixed",
        adj_fixed = fx$R1, adj_fixed2 = fx$R2,
        alg_type = "NUTS"
      )
    ),
    list(
      name = "ggm_hmc_binary",
      args = list(
        X = fx$X, y = fx$y_binary, z_dat = fx$Z,
        outcome_type = "binary", adj_type = "ggm",
        alg_type = "HMC"
      )
    ),
    list(
      name = "ggm_fixed_nuts_tte",
      args = list(
        X = fx$X, y = fx$y_tte, event = fx$event, z_dat = fx$Z,
        outcome_type = "TTE", adj_type = "ggm_fixed", adj_fixed = fx$R1,
        alg_type = "NUTS"
      )
    ),
    list(
      name = "ggm_sparse_hmc_binary",
      args = list(
        X = fx$X_sp, y = fx$y_binary, z_dat = fx$Z,
        outcome_type = "binary", adj_type = "ggm", sparse = TRUE,
        S_ggm = fx$S_ggm,
        store_beta = FALSE, store_gamma = FALSE,
        store_Z_list = FALSE, store_Z_pip = TRUE,
        alg_type = "HMC"
      )
    ),
    list(
      name = "ggm_fixed_sparse_nuts_tte",
      args = list(
        X = fx$X_sp, y = fx$y_tte, event = fx$event, z_dat = fx$Z,
        outcome_type = "TTE", adj_type = "ggm_fixed", sparse = TRUE,
        adj_fixed = fx$R1, S_ggm = fx$S_ggm,
        store_beta = FALSE, store_gamma = FALSE,
        store_Z_list = FALSE, store_Z_pip = TRUE,
        alg_type = "NUTS"
      )
    )
  )

  for (spec in specs) {
    fit <- do.call(
      bvs_mh,
      c(
        list(
          niter = 4L,
          burnin = 3L,
          thin = 1L,
          n_thin_gb = 1L,
          n_mh_gamma = 2L
        ),
        spec$args
      )
    )

    expect_s3_class(fit, "bvs")
    expect_true(fit$sampler %in% c("hmc", "nuts"), info = spec$name)
    expect_equal(fit$p, fx$p, info = spec$name)
    expect_equal(fit$n, fx$n, info = spec$name)
    expect_equal(length(fit$alpha), 4L, info = spec$name)
    expect_true(all(is.finite(fit$sigmasq)), info = spec$name)
    expect_true(all(is.finite(fit$gamma_pip)), info = spec$name)
    expect_false(is.null(fit$tau), info = spec$name)
    expect_true(all(is.finite(fit$tau)), info = spec$name)
    expect_true(is.list(fit$hmc_nuts_diagnostics), info = spec$name)
    expect_true(all(c(
      "n_steps", "accept_rate", "mean_accept_prob", "n_divergent",
      "n_max_treedepth", "mean_tree_depth", "mean_abs_energy_error",
      "max_abs_energy_error", "final_epsilon", "epsilon_history",
      "energy_error_history"
    ) %in% names(fit$hmc_nuts_diagnostics)), info = spec$name)
    expect_true(fit$hmc_nuts_diagnostics$n_steps >= 1L, info = spec$name)
    expect_equal(
      length(fit$hmc_nuts_diagnostics$epsilon_history),
      fit$hmc_nuts_diagnostics$n_steps,
      info = spec$name
    )
    if (identical(fit$outcome_type, "TTE")) {
      expect_true(all(abs(fit$alpha) < 1e-10), info = spec$name)
    }
  }
})

test_that("unsupported HMC/NUTS outcomes fall back to MH", {
  fx <- local_hmc_fixture(seed = 919L)
  y_cont <- as.numeric(fx$X[, 1] + rnorm(fx$n, sd = 0.2))

  expect_warning(
    fit <- bvs_mh(
      fx$X, y_cont,
      outcome_type = "continuous",
      adj_type = "fixed", adj_fixed = fx$R1,
      alg_type = "NUTS",
      niter = 4L, burnin = 2L
    ),
    "falling back to 'MH'"
  )
  expect_identical(fit$sampler, "mh")
})

test_that("n_thin_gb warning is emitted for HMC/NUTS when > 1", {
  fx <- local_hmc_fixture(seed = 173L)
  expect_warning(
    bvs_mh(
      X = fx$X, y = fx$y_binary, z_dat = fx$Z,
      outcome_type = "binary", adj_type = "fixed", adj_fixed = fx$R1,
      alg_type = "HMC",
      niter = 4L, burnin = 2L, n_thin_gb = 2L
    ),
    "n_thin_gb=1 is typically sufficient"
  )
})
