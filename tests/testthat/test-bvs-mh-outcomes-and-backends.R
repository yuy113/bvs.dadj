local_mh_fixture <- function(n = 36L, p = 14L, seed = 606L) {
  skip_if_not_installed("Matrix")
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  X_sp <- as(Matrix::Matrix(X, sparse = TRUE), "dgCMatrix")
  S_ggm <- Matrix::crossprod(X_sp)

  R1 <- matrix(0L, p, p)
  R1[1:5, 1:5] <- 1L
  diag(R1) <- 0L

  R2 <- matrix(0L, p, p)
  R2[3:8, 3:8] <- 1L
  diag(R2) <- 0L

  beta_true <- c(rep(0.9, 4), rep(0, p - 4))
  eta <- as.numeric(X %*% beta_true)

  y_binary <- rbinom(n, 1, plogis(eta / 3))
  y_cont <- eta / 2 + rnorm(n, sd = 0.6)
  y_count <- rnbinom(n, size = 3, mu = exp(0.2 + eta / 6))
  y_tte <- rexp(n, rate = exp(eta / 7))
  event <- rbinom(n, 1, 0.75)
  if (sum(event == 1L) < 2L) {
    event[1:2] <- 1L
  }

  list(
    X = X, X_sp = X_sp, S_ggm = S_ggm, R1 = R1, R2 = R2,
    y_binary = y_binary, y_cont = y_cont, y_count = y_count,
    y_tte = y_tte, event = event, n = n, p = p
  )
}

test_that("bvs_mh smoke-runs all outcomes across all MH backends", {
  skip_on_cran()
  fx <- local_mh_fixture()

  backend_specs <- list(
    list(name = "fixed", adj_type = "fixed", sparse = FALSE, x = fx$X, extra = list(adj_fixed = fx$R1)),
    list(name = "dual_fixed", adj_type = "dual_fixed", sparse = FALSE, x = fx$X, extra = list(adj_fixed = fx$R1, adj_fixed2 = fx$R2)),
    list(name = "ggm_dense", adj_type = "ggm", sparse = FALSE, x = fx$X, extra = list()),
    list(name = "ggm_fixed_dense", adj_type = "ggm_fixed", sparse = FALSE, x = fx$X, extra = list(adj_fixed = fx$R1)),
    list(
      name = "ggm_sparse", adj_type = "ggm", sparse = TRUE, x = fx$X_sp,
      extra = list(S_ggm = fx$S_ggm, store_beta = FALSE, store_gamma = FALSE, store_Z_list = FALSE, store_Z_pip = TRUE)
    ),
    list(
      name = "ggm_fixed_sparse", adj_type = "ggm_fixed", sparse = TRUE, x = fx$X_sp,
      extra = list(adj_fixed = fx$R1, S_ggm = fx$S_ggm, store_beta = FALSE, store_gamma = FALSE, store_Z_list = FALSE, store_Z_pip = TRUE)
    )
  )

  outcome_specs <- list(
    list(name = "binary", y = fx$y_binary, event = NULL),
    list(name = "continuous", y = fx$y_cont, event = NULL),
    list(name = "count", y = fx$y_count, event = NULL),
    list(name = "TTE", y = fx$y_tte, event = fx$event)
  )

  for (b in backend_specs) {
    for (o in outcome_specs) {
      fit <- do.call(
        bvs_mh,
        c(
          list(
            X = b$x,
            y = o$y,
            event = o$event,
            outcome_type = o$name,
            adj_type = b$adj_type,
            sparse = b$sparse,
            niter = 4L,
            burnin = 2L,
            thin = 1L,
            n_thin_gb = 1L,
            n_mh_gamma = 2L
          ),
          b$extra
        )
      )

      expect_s3_class(fit, "bvs")
      expect_identical(fit$outcome_type, o$name)
      expect_identical(fit$adj_type, b$adj_type)
      expect_equal(fit$p, fx$p)
      expect_equal(fit$n, fx$n)
      expect_length(fit$alpha, 4L)
      expect_true(all(is.finite(fit$sigmasq)))
      if (identical(o$name, "TTE")) {
        expect_true(all(abs(fit$alpha) < 1e-10))
      }
    }
  }
})

test_that("bvs_mh validates outcome-specific inputs", {
  fx <- local_mh_fixture(seed = 707L)

  expect_error(
    bvs_mh(
      fx$X, c(0, 2, rep(0, fx$n - 2)),
      outcome_type = "binary",
      adj_type = "fixed",
      adj_fixed = fx$R1,
      niter = 4L,
      burnin = 2L
    ),
    "outcome_type='binary'"
  )

  y_bad_cont <- fx$y_cont
  y_bad_cont[1] <- Inf
  expect_error(
    bvs_mh(
      fx$X, y_bad_cont,
      outcome_type = "continuous",
      adj_type = "fixed",
      adj_fixed = fx$R1,
      niter = 4L,
      burnin = 2L
    ),
    "outcome_type='continuous'"
  )

  expect_error(
    bvs_mh(
      fx$X, fx$y_tte,
      outcome_type = "TTE",
      adj_type = "fixed",
      adj_fixed = fx$R1,
      niter = 4L,
      burnin = 2L
    ),
    "event"
  )

  expect_error(
    bvs_mh(
      fx$X, fx$y_tte,
      event = rep(0, fx$n),
      outcome_type = "TTE",
      adj_type = "fixed",
      adj_fixed = fx$R1,
      niter = 4L,
      burnin = 2L
    ),
    "at least one event==1"
  )

  y_bad_count <- fx$y_count
  y_bad_count[2] <- y_bad_count[2] + 0.25
  expect_error(
    bvs_mh(
      fx$X, y_bad_count,
      outcome_type = "count",
      adj_type = "fixed",
      adj_fixed = fx$R1,
      niter = 4L,
      burnin = 2L
    ),
    "non-negative integers"
  )
})
