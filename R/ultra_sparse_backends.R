# Ultra-sparse wrappers for high-dimensional sparse-GGM backends.

.as_dgC <- function(M) {
  if (inherits(M, "dgCMatrix")) {
    return(M)
  }
  if (inherits(M, "Matrix") && !inherits(M, "generalMatrix")) {
    M <- as(M, "generalMatrix")
  }
  as(as(M, "CsparseMatrix"), "dgCMatrix")
}

.init_ultra_sparse_state <- function(y, p, beta_init, gamma_init, alpha_init) {
  if (is.null(beta_init)) {
    beta_init <- numeric(p)
  }
  if (is.null(gamma_init)) {
    gamma_init <- integer(p)
  }
  if (is.null(alpha_init)) {
    y01 <- as.numeric(y)
    if (all(y01 %in% c(-1, 1))) {
      y01 <- 0.5 * (y01 + 1)
    }
    ybar <- mean(y01, na.rm = TRUE)
    ybar <- min(max(ybar, 1e-4), 1 - 1e-4)
    alpha_init <- qlogis(ybar)
  }
  list(beta_init = beta_init, gamma_init = gamma_init, alpha_init = alpha_init)
}

.sparse_diag_drop <- function(M) {
  d <- Matrix::diag(M)
  if (length(d) == 0 || all(d == 0)) {
    return(M)
  }
  M - Matrix::Diagonal(n = nrow(M), x = d)
}

.prepare_sparse_S_triplet <- function(X, S_ggm = NULL, p_max_crossprod = 1e5) {
  if (is.null(S_ggm)) {
    p <- ncol(X)
    if (p >= p_max_crossprod) {
      stop("For large p, provide sparse S_ggm explicitly; crossprod(X) is disabled.")
    }
    S_ggm <- Matrix::crossprod(X)
  }

  S_ggm <- .as_dgC(S_ggm)
  p <- ncol(S_ggm)
  if (nrow(S_ggm) != p) {
    stop("S_ggm must be square.")
  }

  S_diag <- as.numeric(Matrix::diag(S_ggm))

  S_i <- S_ggm@i
  S_p <- S_ggm@p
  S_x <- S_ggm@x

  off_i <- integer(length(S_i))
  off_x <- numeric(length(S_x))
  off_p <- integer(p + 1L)
  off_p[1] <- 0L
  ptr <- 0L

  for (col in seq_len(p)) {
    start <- S_p[col] + 1L
    end <- S_p[col + 1L]
    n_in_col <- 0L
    if (start <= end) {
      idx <- start:end
      rows <- S_i[idx]
      keep <- rows != (col - 1L)
      n_in_col <- sum(keep)
      if (n_in_col > 0L) {
        pos <- (ptr + 1L):(ptr + n_in_col)
        off_i[pos] <- rows[keep]
        off_x[pos] <- S_x[idx][keep]
        ptr <- ptr + n_in_col
      }
    }
    off_p[col + 1L] <- ptr
  }

  if (ptr < length(off_i)) {
    off_i <- off_i[seq_len(ptr)]
    off_x <- off_x[seq_len(ptr)]
  }

  list(
    S_i = as.integer(off_i),
    S_p = as.integer(off_p),
    S_x = as.numeric(off_x),
    S_diag = as.numeric(S_diag),
    p = as.integer(p)
  )
}

.prepare_sparse_adj_triplet <- function(adj_fixed) {
  A <- .as_dgC(adj_fixed)
  if (nrow(A) != ncol(A)) {
    stop("adj_fixed must be square.")
  }

  A <- A + Matrix::t(A)
  A@x[] <- 1
  A <- Matrix::drop0(A)
  A <- .sparse_diag_drop(A)
  A <- Matrix::drop0(A)

  list(
    R_i = as.integer(A@i),
    R_p = as.integer(A@p),
    p = as.integer(ncol(A))
  )
}

.reconstruct_sparse_pip <- function(result, p, store_Z_pip) {
  if (!isTRUE(store_Z_pip)) {
    result$Z_pip <- NULL
    result$R_ggm_median <- NULL
    result$Z_pip_row <- NULL
    result$Z_pip_col <- NULL
    result$Z_pip_val <- NULL
    return(result)
  }

  pr <- result$Z_pip_row
  pc <- result$Z_pip_col
  pv <- result$Z_pip_val

  if (length(pr) > 0L) {
    Z_pip <- Matrix::sparseMatrix(
      i = c(pr + 1L, pc + 1L),
      j = c(pc + 1L, pr + 1L),
      x = c(pv, pv),
      dims = c(p, p),
      repr = "C"
    )
  } else {
    Z_pip <- Matrix::sparseMatrix(
      i = integer(0), j = integer(0), x = numeric(0),
      dims = c(p, p), repr = "C"
    )
  }

  result$Z_pip <- Z_pip
  result$R_ggm_median <- (Z_pip > 0.5) * 1
  result$Z_pip_row <- NULL
  result$Z_pip_col <- NULL
  result$Z_pip_val <- NULL
  result
}

.reconstruct_gamma_pip <- function(result, p, n_save, store_gamma) {
  if (!isTRUE(store_gamma) || !is.list(result$gamma)) {
    result$gamma_pip <- NULL
    return(result)
  }

  pip_counts <- numeric(p)
  for (s in seq_along(result$gamma)) {
    idx <- result$gamma[[s]]
    if (length(idx) > 0L) {
      pip_counts[idx + 1L] <- pip_counts[idx + 1L] + 1
    }
  }
  result$gamma_pip <- pip_counts / n_save
  result
}

#' Ultra-sparse MH BVS with single sparse GGM adjacency
#' @export
bvs_mh_singlenet_ultra_sparse <- function(
    X, y, S_ggm = NULL,
    niter = 60000L, burnin = 10000L, thin = 1L,
    nu0 = 2, sigmasq0 = 1.5, h = 1.5,
    mu = -log(1 / 0.3 - 1), alpha0 = 0, beta0 = 0,
    n_mh_gamma = 3L,
    eta_sd = 0.5, mu_tilde = -4, eta1_tilde = 0.075,
    e_eta = 2, f_eta = 1,
    Tmax = 64L, proposal_type = 1L,
    v0_ggm = 0.015^2, v1_ggm = (50^2) * (0.015^2), pii_ggm = NULL,
    beta_init = NULL, gamma_init = NULL, alpha_init = NULL,
    store_beta = FALSE, store_gamma = FALSE,
    store_Z_list = FALSE, store_Z_pip = TRUE,
    seed = NULL) {

  if (!is.null(seed)) {
    set.seed(seed)
  }

  X <- .as_dgC(X)
  y <- as.numeric(y)

  n <- nrow(X)
  p <- ncol(X)
  if (length(y) != n) {
    stop("length(y) must equal nrow(X).")
  }
  if (is.null(pii_ggm)) {
    pii_ggm <- 30 / max(1, (p - 1))
  }

  if (is.null(beta_init) || is.null(gamma_init) || is.null(alpha_init)) {
    init <- .init_ultra_sparse_state(y, p, beta_init, gamma_init, alpha_init)
    beta_init <- init$beta_init
    gamma_init <- init$gamma_init
    alpha_init <- init$alpha_init
  }

  sp <- .prepare_sparse_S_triplet(X, S_ggm = S_ggm)

  result <- BayesLogit_SingleNet_SparseGGM_UltraSparse(
    X = X, y = y,
    S_i = sp$S_i, S_p_csc = sp$S_p, S_x = sp$S_x, S_diag = sp$S_diag,
    p_ggm = sp$p,
    niter = as.integer(niter), burnin = as.integer(burnin),
    mu = as.double(mu), nu0 = as.double(nu0), sigmasq0 = as.double(sigmasq0),
    alpha0 = as.double(alpha0), beta0 = as.double(beta0),
    h = as.double(h), e = as.double(e_eta), f = as.double(f_eta),
    v0_ggm = as.double(v0_ggm), v1_ggm = as.double(v1_ggm),
    pii_ggm = as.double(pii_ggm),
    eta_sd = as.double(eta_sd), mu_tilde = as.double(mu_tilde),
    eta1_tilde = as.double(eta1_tilde),
    T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
    n_mh_gamma = as.integer(n_mh_gamma), thin = as.integer(thin),
    beta_in = as.numeric(beta_init),
    gamma_in = as.integer(gamma_init),
    alpha_in = as.double(alpha_init),
    store_beta = isTRUE(store_beta), store_gamma = isTRUE(store_gamma),
    store_Z_list = isTRUE(store_Z_list), store_Z_pip = isTRUE(store_Z_pip)
  )

  n_save <- as.integer(niter %/% max(1L, as.integer(thin)))
  result <- .reconstruct_sparse_pip(result, p, store_Z_pip)
  result <- .reconstruct_gamma_pip(result, p, n_save, store_gamma)
  result$n_save <- n_save
  result$sampler <- "mh"
  result$adj_type <- "ggm"
  class(result) <- c("bvs_ultra_sparse", "bvs")
  result
}

#' Ultra-sparse MH BVS with dual network (sparse GGM + sparse fixed adjacency)
#' @export
bvs_mh_dualnet_ultra_sparse <- function(
    X, y, adj_fixed, S_ggm = NULL,
    niter = 60000L, burnin = 10000L, thin = 1L,
    nu0 = 2, sigmasq0 = 1.5, h = 1.5,
    mu = -log(1 / 0.3 - 1), alpha0 = 0, beta0 = 0,
    n_mh_gamma = 3L,
    eta1_sd = 0.5, eta2_sd = 0.5,
    mu_tilde = -4, eta1_tilde = 0.075, eta2_tilde = 0.065,
    e_eta = 2, f_eta = 1,
    Tmax = 64L, proposal_type = 1L,
    v0_ggm = 0.015^2, v1_ggm = (50^2) * (0.015^2), pii_ggm = NULL,
    beta_init = NULL, gamma_init = NULL, alpha_init = NULL,
    store_beta = FALSE, store_gamma = FALSE,
    store_Z_list = FALSE, store_Z_pip = TRUE,
    seed = NULL) {

  if (!is.null(seed)) {
    set.seed(seed)
  }

  X <- .as_dgC(X)
  y <- as.numeric(y)

  n <- nrow(X)
  p <- ncol(X)
  if (length(y) != n) {
    stop("length(y) must equal nrow(X).")
  }
  if (is.null(pii_ggm)) {
    pii_ggm <- 30 / max(1, (p - 1))
  }

  fixed_sp <- .prepare_sparse_adj_triplet(adj_fixed)
  if (fixed_sp$p != p) {
    stop("adj_fixed dimension must match ncol(X).")
  }

  if (is.null(beta_init) || is.null(gamma_init) || is.null(alpha_init)) {
    init <- .init_ultra_sparse_state(y, p, beta_init, gamma_init, alpha_init)
    beta_init <- init$beta_init
    gamma_init <- init$gamma_init
    alpha_init <- init$alpha_init
  }

  sp <- .prepare_sparse_S_triplet(X, S_ggm = S_ggm)

  result <- BayesLogit_DualNet_SparseGGM_UltraSparse(
    X = X, y = y,
    S_i = sp$S_i, S_p_csc = sp$S_p, S_x = sp$S_x, S_diag = sp$S_diag,
    R_fix_i = fixed_sp$R_i, R_fix_p_csc = fixed_sp$R_p,
    p_ggm = sp$p,
    niter = as.integer(niter), burnin = as.integer(burnin),
    mu = as.double(mu), nu0 = as.double(nu0), sigmasq0 = as.double(sigmasq0),
    alpha0 = as.double(alpha0), beta0 = as.double(beta0),
    h = as.double(h), e = as.double(e_eta), f = as.double(f_eta),
    v0_ggm = as.double(v0_ggm), v1_ggm = as.double(v1_ggm),
    pii_ggm = as.double(pii_ggm),
    eta1_sd = as.double(eta1_sd), eta2_sd = as.double(eta2_sd),
    mu_tilde = as.double(mu_tilde), eta1_tilde = as.double(eta1_tilde),
    eta2_tilde = as.double(eta2_tilde),
    T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
    n_mh_gamma = as.integer(n_mh_gamma), thin = as.integer(thin),
    beta_in = as.numeric(beta_init),
    gamma_in = as.integer(gamma_init),
    alpha_in = as.double(alpha_init),
    store_beta = isTRUE(store_beta), store_gamma = isTRUE(store_gamma),
    store_Z_list = isTRUE(store_Z_list), store_Z_pip = isTRUE(store_Z_pip)
  )

  n_save <- as.integer(niter %/% max(1L, as.integer(thin)))
  result <- .reconstruct_sparse_pip(result, p, store_Z_pip)
  result <- .reconstruct_gamma_pip(result, p, n_save, store_gamma)
  result$n_save <- n_save
  result$sampler <- "mh"
  result$adj_type <- "ggm_fixed"
  class(result) <- c("bvs_ultra_sparse", "bvs")
  result
}

#' Ultra-sparse PG BVS with single sparse GGM adjacency
#' @export
bvs_pg_singlenet_ultra_sparse <- function(
    X, y, S_ggm = NULL,
    niter = 60000L, burnin = 10000L, thin = 1L,
    nu0 = 2, sigmasq0 = 1.5, h = 1.5,
    mu = -log(1 / 0.3 - 1), alpha0 = 0, beta0 = 0,
    n_mh_gamma = 3L,
    eta_sd = 0.5,
    mu_tilde = -4, eta1_tilde = 0.075,
    e_eta = 2, f_eta = 1,
    Tmax = 64L, proposal_type = 1L,
    v0_ggm = 0.015^2, v1_ggm = (50^2) * (0.015^2), pii_ggm = NULL,
    beta_init = NULL, gamma_init = NULL, alpha_init = NULL,
    store_beta = FALSE, store_gamma = FALSE,
    store_Z_list = FALSE, store_Z_pip = TRUE,
    seed = NULL) {

  if (!is.null(seed)) {
    set.seed(seed)
  }

  X <- .as_dgC(X)
  y <- as.numeric(y)

  n <- nrow(X)
  p <- ncol(X)
  if (length(y) != n) {
    stop("length(y) must equal nrow(X).")
  }
  if (is.null(pii_ggm)) {
    pii_ggm <- 30 / max(1, (p - 1))
  }

  if (is.null(beta_init) || is.null(gamma_init) || is.null(alpha_init)) {
    init <- .init_ultra_sparse_state(y, p, beta_init, gamma_init, alpha_init)
    beta_init <- init$beta_init
    gamma_init <- init$gamma_init
    alpha_init <- init$alpha_init
  }

  sp <- .prepare_sparse_S_triplet(X, S_ggm = S_ggm)

  result <- BayesLogit_PG_SingleNet_SparseGGM_UltraSparse(
    X = X, y = y,
    S_i = sp$S_i, S_p_csc = sp$S_p, S_x = sp$S_x, S_diag = sp$S_diag,
    p_ggm = sp$p,
    niter = as.integer(niter), burnin = as.integer(burnin),
    mu = as.double(mu), nu0 = as.double(nu0), sigmasq0 = as.double(sigmasq0),
    alpha0 = as.double(alpha0), beta0 = as.double(beta0),
    h = as.double(h), n_mh_gamma = as.integer(n_mh_gamma),
    v0_ggm = as.double(v0_ggm), v1_ggm = as.double(v1_ggm),
    pii_ggm = as.double(pii_ggm),
    eta_sd = as.double(eta_sd), mu_tilde = as.double(mu_tilde),
    eta1_tilde = as.double(eta1_tilde),
    e_eta = as.double(e_eta), f_eta = as.double(f_eta),
    T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
    thin = as.integer(thin),
    beta_in = as.numeric(beta_init),
    gamma_in = as.integer(gamma_init),
    alpha_in = as.double(alpha_init),
    store_beta = isTRUE(store_beta), store_gamma = isTRUE(store_gamma),
    store_Z_list = isTRUE(store_Z_list), store_Z_pip = isTRUE(store_Z_pip)
  )

  n_save <- as.integer(niter %/% max(1L, as.integer(thin)))
  result <- .reconstruct_sparse_pip(result, p, store_Z_pip)
  result <- .reconstruct_gamma_pip(result, p, n_save, store_gamma)
  result$n_save <- n_save
  result$sampler <- "pg"
  result$adj_type <- "ggm"
  class(result) <- c("bvs_ultra_sparse", "bvs")
  result
}

#' Ultra-sparse PG BVS with dual network (sparse GGM + sparse fixed adjacency)
#' @export
bvs_pg_dualnet_ultra_sparse <- function(
    X, y, adj_fixed, S_ggm = NULL,
    niter = 60000L, burnin = 10000L, thin = 1L,
    nu0 = 2, sigmasq0 = 1.5, h = 1.5,
    mu = -log(1 / 0.3 - 1), alpha0 = 0, beta0 = 0,
    n_mh_gamma = 3L,
    eta1_sd = 0.5, eta2_sd = 0.5,
    mu_tilde = -4, eta1_tilde = 0.075, eta2_tilde = 0.065,
    e_eta = 2, f_eta = 1,
    Tmax = 64L, proposal_type = 1L,
    v0_ggm = 0.015^2, v1_ggm = (50^2) * (0.015^2), pii_ggm = NULL,
    beta_init = NULL, gamma_init = NULL, alpha_init = NULL,
    store_beta = FALSE, store_gamma = FALSE,
    store_Z_list = FALSE, store_Z_pip = TRUE,
    seed = NULL) {

  if (!is.null(seed)) {
    set.seed(seed)
  }

  X <- .as_dgC(X)
  y <- as.numeric(y)

  n <- nrow(X)
  p <- ncol(X)
  if (length(y) != n) {
    stop("length(y) must equal nrow(X).")
  }
  if (is.null(pii_ggm)) {
    pii_ggm <- 30 / max(1, (p - 1))
  }

  fixed_sp <- .prepare_sparse_adj_triplet(adj_fixed)
  if (fixed_sp$p != p) {
    stop("adj_fixed dimension must match ncol(X).")
  }

  if (is.null(beta_init) || is.null(gamma_init) || is.null(alpha_init)) {
    init <- .init_ultra_sparse_state(y, p, beta_init, gamma_init, alpha_init)
    beta_init <- init$beta_init
    gamma_init <- init$gamma_init
    alpha_init <- init$alpha_init
  }

  sp <- .prepare_sparse_S_triplet(X, S_ggm = S_ggm)

  result <- BayesLogit_PG_DualNet_SparseGGM_UltraSparse(
    X = X, y = y,
    S_i = sp$S_i, S_p_csc = sp$S_p, S_x = sp$S_x, S_diag = sp$S_diag,
    R_fix_i = fixed_sp$R_i, R_fix_p_csc = fixed_sp$R_p,
    p_ggm = sp$p,
    niter = as.integer(niter), burnin = as.integer(burnin),
    mu = as.double(mu), nu0 = as.double(nu0), sigmasq0 = as.double(sigmasq0),
    alpha0 = as.double(alpha0), beta0 = as.double(beta0),
    h = as.double(h), n_mh_gamma = as.integer(n_mh_gamma),
    v0_ggm = as.double(v0_ggm), v1_ggm = as.double(v1_ggm),
    pii_ggm = as.double(pii_ggm),
    eta1_sd = as.double(eta1_sd), eta2_sd = as.double(eta2_sd),
    mu_tilde = as.double(mu_tilde), eta1_tilde = as.double(eta1_tilde),
    eta2_tilde = as.double(eta2_tilde),
    e_eta = as.double(e_eta), f_eta = as.double(f_eta),
    T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
    thin = as.integer(thin),
    beta_in = as.numeric(beta_init),
    gamma_in = as.integer(gamma_init),
    alpha_in = as.double(alpha_init),
    store_beta = isTRUE(store_beta), store_gamma = isTRUE(store_gamma),
    store_Z_list = isTRUE(store_Z_list), store_Z_pip = isTRUE(store_Z_pip)
  )

  n_save <- as.integer(niter %/% max(1L, as.integer(thin)))
  result <- .reconstruct_sparse_pip(result, p, store_Z_pip)
  result <- .reconstruct_gamma_pip(result, p, n_save, store_gamma)
  result$n_save <- n_save
  result$sampler <- "pg"
  result$adj_type <- "ggm_fixed"
  class(result) <- c("bvs_ultra_sparse", "bvs")
  result
}
