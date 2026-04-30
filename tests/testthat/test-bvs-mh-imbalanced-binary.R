# test-bvs-mh-imbalanced-binary.R
# Tests for outcome_type = "imbalanced_binary" (Approach A: logit_t, Approach B: cloglog)

local_imbalanced_fixture <- function(n = 60L, p = 12L, seed = 707L,
                                     prevalence = 0.1) {
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

  beta_true <- c(rep(1.2, 3), rep(0, p - 3))
  intercept <- qlogis(prevalence)
  eta <- intercept + as.numeric(X %*% beta_true)
  y <- rbinom(n, 1, plogis(eta))
  # Ensure at least 2 events and 2 non-events

  if (sum(y == 1) < 2L) y[1:2] <- 1L
  if (sum(y == 0) < 2L) y[(n - 1):n] <- 0L

  list(
    X = X, X_sp = X_sp, S_ggm = S_ggm, R1 = R1, R2 = R2,
    y = y, n = n, p = p, prevalence = mean(y)
  )
}

# ==========================================================================
# 1. Smoke tests: both links x all backends x MH
# ==========================================================================
test_that("imbalanced_binary smoke-runs both links across all MH backends", {
  skip_on_cran()
  fx <- local_imbalanced_fixture()

  backend_specs <- list(
    list(name = "fixed", adj_type = "fixed", sparse = FALSE, x = fx$X,
         extra = list(adj_fixed = fx$R1)),
    list(name = "dual_fixed", adj_type = "dual_fixed", sparse = FALSE, x = fx$X,
         extra = list(adj_fixed = fx$R1, adj_fixed2 = fx$R2)),
    list(name = "ggm_dense", adj_type = "ggm", sparse = FALSE, x = fx$X,
         extra = list()),
    list(name = "ggm_fixed_dense", adj_type = "ggm_fixed", sparse = FALSE,
         x = fx$X, extra = list(adj_fixed = fx$R1))
  )

  link_specs <- list(
    list(name = "logit_t", link = "logit_t"),
    list(name = "cloglog", link = "cloglog")
  )

  for (lk in link_specs) {
    for (bs in backend_specs) {
      label <- paste0(lk$name, "/", bs$name)
      args <- c(list(
        X = bs$x, y = fx$y,
        outcome_type = "imbalanced_binary",
        imbalanced_link = lk$link,
        adj_type = bs$adj_type,
        niter = 200L, burnin = 50L, thin = 1L
      ), bs$extra)
      fit <- do.call(bvs_mh, args)

      expect_s3_class(fit, "bvs")
      expect_true(is.numeric(fit$gamma_pip), label = label)
      expect_equal(length(fit$gamma_pip), fx$p, label = label)
      expect_true(all(fit$gamma_pip >= 0 & fit$gamma_pip <= 1), label = label)
      # No NaN or Inf in beta
      expect_true(all(is.finite(fit$beta)), label = label)
    }
  }
})

# ==========================================================================
# 2. HMC/NUTS smoke test for imbalanced_binary
# ==========================================================================
test_that("imbalanced_binary works with HMC and NUTS", {
  skip_on_cran()
  fx <- local_imbalanced_fixture()

  for (alg in c("HMC", "NUTS")) {
    for (lk in c("logit_t", "cloglog")) {
      label <- paste0(lk, "/", alg)
      fit <- bvs_mh(
        fx$X, fx$y,
        outcome_type = "imbalanced_binary",
        imbalanced_link = lk,
        adj_type = "fixed", adj_fixed = fx$R1,
        niter = 200L, burnin = 50L,
        alg_type = alg
      )
      expect_s3_class(fit, "bvs")
      expect_true(all(is.finite(fit$gamma_pip)), label = label)
    }
  }
})

# ==========================================================================
# 3. logit_t returns lambda_t
# ==========================================================================
test_that("logit_t returns lambda_t latent scales", {
  skip_on_cran()
  fx <- local_imbalanced_fixture()
  fit <- bvs_mh(
    fx$X, fx$y,
    outcome_type = "imbalanced_binary",
    imbalanced_link = "logit_t",
    adj_type = "fixed", adj_fixed = fx$R1,
    niter = 200L, burnin = 50L
  )
  expect_true(!is.null(fit$lambda_t))
  expect_equal(length(fit$lambda_t), fx$p)
  expect_true(all(fit$lambda_t > 0))
})

# ==========================================================================
# 4. cloglog does NOT return lambda_t (uses Gaussian prior)
# ==========================================================================
test_that("cloglog returns lambda_t (all ones or non-null)", {
  skip_on_cran()
  fx <- local_imbalanced_fixture()
  fit <- bvs_mh(
    fx$X, fx$y,
    outcome_type = "imbalanced_binary",
    imbalanced_link = "cloglog",
    adj_type = "fixed", adj_fixed = fx$R1,
    niter = 200L, burnin = 50L
  )
  # lambda_t is returned for all imbalanced_binary; for cloglog it stays at 1
  expect_true(!is.null(fit$lambda_t))
})

# ==========================================================================
# 5. t_df and t_scale validation
# ==========================================================================
test_that("invalid t_df and t_scale are caught", {
  skip_on_cran()
  fx <- local_imbalanced_fixture()
  expect_error(
    bvs_mh(fx$X, fx$y, outcome_type = "imbalanced_binary",
            adj_type = "fixed", adj_fixed = fx$R1,
            t_df = -1, niter = 50L, burnin = 10L),
    "t_df"
  )
  expect_error(
    bvs_mh(fx$X, fx$y, outcome_type = "imbalanced_binary",
            adj_type = "fixed", adj_fixed = fx$R1,
            t_scale = 0, niter = 50L, burnin = 10L),
    "t_scale"
  )
})

# ==========================================================================
# 6. y validation for imbalanced_binary
# ==========================================================================
test_that("imbalanced_binary rejects non-binary y", {
  skip_on_cran()
  fx <- local_imbalanced_fixture()
  y_bad <- rnorm(fx$n)
  expect_error(
    bvs_mh(fx$X, y_bad, outcome_type = "imbalanced_binary",
            adj_type = "fixed", adj_fixed = fx$R1,
            niter = 50L, burnin = 10L),
    "must be in"
  )
})

# ==========================================================================
# 7. Backward compatibility: existing binary still works
# ==========================================================================
test_that("standard binary outcome is unaffected by new parameters", {
  skip_on_cran()
  fx <- local_imbalanced_fixture()
  fit <- bvs_mh(
    fx$X, fx$y,
    outcome_type = "binary",
    adj_type = "fixed", adj_fixed = fx$R1,
    niter = 200L, burnin = 50L
  )
  expect_s3_class(fit, "bvs")
  expect_null(fit$lambda_t)
})

# ==========================================================================
# 8. Student-t df variants: Cauchy (df=1) vs Student-t(7)
# ==========================================================================
test_that("different t_df values produce valid results", {
  skip_on_cran()
  fx <- local_imbalanced_fixture()
  for (df_val in c(1, 3, 7)) {
    fit <- bvs_mh(
      fx$X, fx$y,
      outcome_type = "imbalanced_binary",
      imbalanced_link = "logit_t",
      t_df = df_val, t_scale = 2.5,
      adj_type = "fixed", adj_fixed = fx$R1,
      niter = 200L, burnin = 50L
    )
    expect_s3_class(fit, "bvs")
    expect_true(all(is.finite(fit$gamma_pip)), label = paste("t_df =", df_val))
  }
})

# ==========================================================================
# 9. Sparse backend smoke test
# ==========================================================================
test_that("imbalanced_binary works with sparse GGM backends", {
  skip_on_cran()
  fx <- local_imbalanced_fixture()

  for (lk in c("logit_t", "cloglog")) {
    fit <- bvs_mh(
      fx$X_sp, fx$y,
      outcome_type = "imbalanced_binary",
      imbalanced_link = lk,
      adj_type = "ggm", sparse = TRUE,
      S_ggm = fx$S_ggm,
      store_beta = FALSE, store_gamma = FALSE,
      store_Z_list = FALSE, store_Z_pip = TRUE,
      niter = 200L, burnin = 50L
    )
    expect_s3_class(fit, "bvs")
    expect_true(all(is.finite(fit$gamma_pip)), label = paste0("sparse/", lk))
  }
})
