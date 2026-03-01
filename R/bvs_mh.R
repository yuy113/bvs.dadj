#' Bayesian Variable Selection via Metropolis-Hastings
#'
#' Bayesian variable selection for logistic regression with an Ising/MRF prior
#' on the inclusion indicators gamma, using Metropolis-Hastings updates for
#' beta, alpha, and gamma.  The Ising coupling parameter eta is updated via
#' Moller et al. (2006) auxiliary variable MH with Propp-Wilson perfect
#' simulation.
#'
#' @param X         An \code{n x p} design matrix (numeric).
#' @param y         A binary response vector of length \code{n} (0/1).
#' @param adj_type  Character string specifying the adjacency structure.
#'   One of:
#'   \describe{
#'     \item{\code{"fixed"}}{Single fixed adjacency matrix
#'       (supplied via \code{adj_fixed}).}
#'     \item{\code{"dual_fixed"}}{Two fixed adjacency matrices
#'       (supplied via \code{adj_fixed} and \code{adj_fixed2}).}
#'     \item{\code{"glasso"}}{Single adjacency estimated from X via graphical
#'       lasso (\pkg{huge} package), then used as a fixed matrix.}
#'     \item{\code{"glasso_fixed"}}{Two fixed adjacency sources: one estimated
#'       from X via graphical lasso (\pkg{huge}) and one external fixed matrix
#'       supplied via \code{adj_fixed}.}
#'     \item{\code{"ggm"}}{Single adjacency learned jointly via Bayesian GGM
#'       SSVS (Wang et al. 2012) during MCMC.}
#'     \item{\code{"ggm_fixed"}}{Two sources: one learned from Bayesian GGM
#'       and one fixed (supplied via \code{adj_fixed}).}
#'   }
#' @param adj_fixed   A \code{p x p} binary adjacency matrix (for
#'   \code{"fixed"}, \code{"dual_fixed"}, \code{"glasso_fixed"},
#'   \code{"ggm_fixed"}).
#' @param adj_fixed2  A second \code{p x p} binary adjacency matrix
#'   (for \code{"dual_fixed"} only).
#' @param sparse    Logical; use the sparse high-dimensional backend
#'   (default \code{FALSE}).  Only applicable for \code{adj_type = "ggm"} or
#'   \code{"ggm_fixed"}.
#' @param S_ggm Optional sparse GGM scatter matrix. For ultra-sparse mode,
#'   supply this explicitly for very large \code{p}.
#' @param store_beta Logical; in ultra-sparse mode, store beta draws.
#' @param store_gamma Logical; in ultra-sparse mode, store gamma draws.
#' @param store_Z_list Logical; in ultra-sparse mode, store per-iteration
#'   graph snapshots.
#' @param store_Z_pip Logical; in ultra-sparse mode, accumulate sparse edge PIPs.
#' @param glasso_criterion  Character; criterion for graphical lasso model
#'   selection: \code{"ebic"} (default) or \code{"ric"}.
#'
#' @param niter     Number of post-burn-in MCMC iterations (default 60000).
#' @param burnin    Number of burn-in iterations (default 10000).
#' @param thin      Thinning interval (default 1).
#' @param n_thin_gb Number of inner MH sub-iterations for gamma and beta
#'   per MCMC iteration (default 3). Only used by dense backends; sparse
#'   backends ignore this parameter for computational efficiency.
#'
#' @param nu0       Inverse-Gamma shape for sigma^2 (default 2).
#' @param sigmasq0  Inverse-Gamma scale for sigma^2 (default 1.5).
#' @param h         Intercept variance inflation factor (default 1.5).
#' @param mu        Ising external field controlling sparsity
#'   (default \code{-log(1/0.3 - 1)}).
#' @param alpha0    Prior mean for intercept (default 0).
#' @param beta0     Prior mean for coefficients (default 0).
#'
#' @param n_mh_gamma  Number of gamma MH updates per MCMC iteration
#'   (default 3; used by GGM/sparse backends).
#' @param eta1_sd   Upper bound for eta1 (single-eta models or dual models, default 0.5).
#' @param eta2_sd   Upper bound for eta2 (dual-eta models, default 0.5).
#' @param mu_tilde  Auxiliary MRF external field for Moller update
#'   (default -4).
#' @param eta1_tilde Auxiliary eta1 coupling (default 0.075).
#' @param eta2_tilde Auxiliary eta2 coupling (default 0.065).
#' @param e_eta     Beta prior shape \code{a} for eta (default 2).
#' @param f_eta     Beta prior shape \code{b} for eta (default 1).
#' @param Tmax      Maximum Propp-Wilson doubling time (default 64).
#' @param proposal_type  Eta proposal kernel: 0 = uniform, 1 = truncated
#'   normal (default 1).
#'
#' @param v0_ggm    GGM SSVS spike variance (default \code{0.015^2}).
#' @param v1_ggm    GGM SSVS slab variance (default \code{50^2 * 0.015^2}).
#' @param pii_ggm   GGM SSVS inclusion probability
#'   (default \code{30/(p-1)}).
#' @param lambda_ggm GGM prior scale (default 1).
#'
#' @param beta_init  Optional initial beta (numeric vector of length p).
#' @param gamma_init Optional initial gamma (integer vector of length p).
#' @param alpha_init Optional initial intercept (numeric scalar).
#'
#' @return An object of class \code{"bvs"}, a named list with:
#'   \describe{
#'     \item{beta}{Matrix of posterior beta samples (niter x p).}
#'     \item{gamma}{Posterior gamma samples (dense matrix or sparse list,
#'       depending on backend/storage options).}
#'     \item{alpha}{Vector of posterior alpha (intercept) samples.}
#'     \item{sigmasq}{Vector of posterior sigma^2 samples.}
#'     \item{eta1}{Vector of eta1 (dynamic or single-graph coupling) samples
#'       (if applicable).}
#'     \item{eta2}{Vector of eta2 (fixed coupling) samples
#'       (if applicable).}
#'     \item{Z_list}{List of GGM adjacency snapshots (if applicable).}
#'     \item{call}{The matched call.}
#'     \item{adj_type}{The adjacency type used.}
#'     \item{sampler}{Character: \code{"mh"}.}
#'   }
#'
#' @details
#' The model is:
#' \deqn{y_i | X, beta, alpha ~ Bernoulli(logistic(alpha + X_i beta))}
#' \deqn{beta_j | gamma_j, sigma^2 ~ gamma_j N(0, sigma^2) + (1-gamma_j) delta_0}
#' \deqn{P(gamma | eta, R) propto exp(mu sum(gamma_j) + eta sum_{R} gamma_j gamma_k)}
#'
#' where R is the adjacency matrix from the chosen \code{adj_type}.
#'
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 200
#' p <- 50
#' X <- matrix(rnorm(n * p), n, p)
#' beta_true <- c(rep(1, 5), rep(0, p - 5))
#' y <- rbinom(n, 1, plogis(X %*% beta_true))
#'
#' # Single fixed adjacency
#' R <- diag(0, p)
#' R[1:5, 1:5] <- 1
#' diag(R) <- 0
#' fit <- bvs_mh(X, y, adj_type = "fixed", adj_fixed = R, niter = 5000)
#' summary(fit)
#'
#' # Glasso-estimated adjacency with EBIC
#' fit2 <- bvs_mh(X, y,
#'   adj_type = "glasso", glasso_criterion = "ebic",
#'   niter = 5000
#' )
#'
#' # Dual network: glasso-estimated + external fixed adjacency
#' fit2b <- bvs_mh(X, y,
#'   adj_type = "glasso_fixed", adj_fixed = R,
#'   glasso_criterion = "ebic", niter = 5000
#' )
#'
#' # Sparse Bayesian GGM adjacency
#' fit3 <- bvs_mh(X, y, adj_type = "ggm", sparse = TRUE, niter = 5000)
#' }
#'
#' @seealso \code{\link{bvs_pg}}, \code{\link{summary.bvs}},
#'   \code{\link{estimate_glasso_adj}}
#'
#' @importFrom grDevices devAskNewPage
#' @importFrom stats qlogis var
#' @importFrom utils head
#' @export
bvs_mh <- function(X, y,
                   adj_type = c(
                     "fixed", "dual_fixed", "glasso",
                     "glasso_fixed", "ggm", "ggm_fixed"
                   ),
                   adj_fixed = NULL,
                   adj_fixed2 = NULL,
                   sparse = FALSE,
                   S_ggm = NULL,
                   store_beta = FALSE,
                   store_gamma = FALSE,
                   store_Z_list = FALSE,
                   store_Z_pip = TRUE,
                   glasso_criterion = c("ebic", "ric"),
                   # MCMC control
                   niter = 60000L,
                   burnin = 10000L,
                   thin = 1L,
                   n_thin_gb = 3L,
                   # Variable selection priors
                   nu0 = 2, sigmasq0 = 1.5, h = 1.5,
                   mu = -log(1 / 0.3 - 1),
                   alpha0 = 0, beta0 = 0,
                   # Gamma / Ising
                   n_mh_gamma = 3L,
                   eta1_sd = 0.5, eta2_sd = 0.5,
                   mu_tilde = -4,
                   eta1_tilde = 5, eta2_tilde = 5,
                   e_eta = 1, f_eta = 1,
                   Tmax = 64L,
                   proposal_type = 1L,
                   # GGM SSVS
                   v0_ggm = 0.015^2,
                   v1_ggm = 50^2*0.015^2,
                   pii_ggm = 0.02,
                   lambda_ggm = 1,
                   # Init (optional)
                   beta_init = NULL,
                   gamma_init = NULL,
                   alpha_init = NULL) {
  mc <- match.call()
  adj_type <- match.arg(adj_type)
  glasso_criterion <- match.arg(glasso_criterion)

  if (isTRUE(sparse) && !adj_type %in% c("ggm", "ggm_fixed")) {
    warning("sparse=TRUE is only implemented for adj_type='ggm' or 'ggm_fixed'; using dense backend.",
      call. = FALSE
    )
  }

  # Dimensions
  use_sparse_backend <- isTRUE(sparse) && adj_type %in% c("ggm", "ggm_fixed")
  if (use_sparse_backend) {
    X <- .as_dgC(X)
  } else {
    X <- as.matrix(X)
  }
  y <- as.numeric(y)
  n <- nrow(X)
  p <- ncol(X)

  # Derived GGM defaults
  if (is.null(v1_ggm)) v1_ggm <- (50^2) * v0_ggm
  if (is.null(pii_ggm)) pii_ggm <- 30 / (p - 1)

  # Initialisation
  if (is.null(beta_init) || is.null(gamma_init) || is.null(alpha_init)) {
    if (use_sparse_backend) {
      init <- .init_ultra_sparse_state(y, p, beta_init, gamma_init, alpha_init)
      beta_init <- init$beta_init
      gamma_init <- init$gamma_init
      alpha_init <- init$alpha_init
    } else {
      init <- .init_mcmc(p, mu, nu0, sigmasq0, alpha0, beta0, h)
      if (is.null(beta_init)) beta_init <- init$beta_init
      if (is.null(gamma_init)) gamma_init <- init$gamma_init
      if (is.null(alpha_init)) alpha_init <- init$alpha_init
    }
  }

  # ---- Dispatch ----
  # Note: parameter names must match the C++ signatures exactly.
  # Most dense MH backends use (e, f); BayesLogit_DualNet_GGM and
  # sparse backends use (e_eta, f_eta). Check RcppExports.R per backend.
  result <- switch(adj_type,
    "fixed" = {
      if (is.null(adj_fixed)) {
        stop("adj_type='fixed' requires 'adj_fixed'.")
      }
      R_mat <- .prepare_adj(adj_fixed, p, "adj_fixed")
      BayesLogit_SingleNet_FixedAdj(
        X = X, y = y, R_fix_int = R_mat,
        niter = as.integer(niter), burnin = as.integer(burnin),
        mu = mu, nu0 = nu0, sigmasq0 = sigmasq0,
        alpha0 = alpha0, beta0 = beta0, h = h,
        e_eta = e_eta, f_eta = f_eta,
        eta1_sd = eta1_sd, mu_tilde = mu_tilde,
        eta1_tilde = eta1_tilde,
        T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
        thin = as.integer(thin), n_thin_gb = as.integer(n_thin_gb),
        beta_in = beta_init, gamma_in = as.integer(gamma_init),
        alpha_in = alpha_init
      )
    },
    "dual_fixed" = {
      if (is.null(adj_fixed) || is.null(adj_fixed2)) {
        stop("adj_type='dual_fixed' requires both 'adj_fixed' and 'adj_fixed2'.")
      }
      R1 <- .prepare_adj(adj_fixed, p, "adj_fixed")
      R2 <- .prepare_adj(adj_fixed2, p, "adj_fixed2")
      BayesLogit_DualNet_FixedAdj(
        X = X, y = y, R_dyn_int = R1, R_fix_int = R2,
        niter = as.integer(niter), burnin = as.integer(burnin),
        mu = mu, nu0 = nu0, sigmasq0 = sigmasq0,
        alpha0 = alpha0, beta0 = beta0, h = h,
        e_eta = e_eta, f_eta = f_eta,
        eta1_sd = eta1_sd, eta2_sd = eta2_sd,
        mu_tilde = mu_tilde,
        eta1_tilde = eta1_tilde, eta2_tilde = eta2_tilde,
        T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
        thin = as.integer(thin), n_thin_gb = as.integer(n_thin_gb),
        beta_in = beta_init, gamma_in = as.integer(gamma_init),
        alpha_in = alpha_init
      )
    },
    "glasso" = {
      adj_est <- estimate_glasso_adj(X, criterion = glasso_criterion)
      R_mat <- .prepare_adj(adj_est, p, "glasso_adj")
      BayesLogit_SingleNet_FixedAdj(
        X = X, y = y, R_fix_int = R_mat,
        niter = as.integer(niter), burnin = as.integer(burnin),
        mu = mu, nu0 = nu0, sigmasq0 = sigmasq0,
        alpha0 = alpha0, beta0 = beta0, h = h,
        e_eta = e_eta, f_eta = f_eta,
        eta1_sd = eta1_sd, mu_tilde = mu_tilde,
        eta1_tilde = eta1_tilde,
        T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
        thin = as.integer(thin), n_thin_gb = as.integer(n_thin_gb),
        beta_in = beta_init, gamma_in = as.integer(gamma_init),
        alpha_in = alpha_init
      )
    },
    "glasso_fixed" = {
      if (is.null(adj_fixed)) {
        stop("adj_type='glasso_fixed' requires 'adj_fixed'.")
      }
      R_fix <- .prepare_adj(adj_fixed, p, "adj_fixed")
      adj_est <- estimate_glasso_adj(X, criterion = glasso_criterion)
      R_glasso <- .prepare_adj(adj_est, p, "glasso_adj")
      BayesLogit_DualNet_FixedAdj(
        X = X, y = y, R_dyn_int = R_glasso, R_fix_int = R_fix,
        niter = as.integer(niter), burnin = as.integer(burnin),
        mu = mu, nu0 = nu0, sigmasq0 = sigmasq0,
        alpha0 = alpha0, beta0 = beta0, h = h,
        e_eta = e_eta, f_eta = f_eta,
        eta1_sd = eta1_sd, eta2_sd = eta2_sd,
        mu_tilde = mu_tilde,
        eta1_tilde = eta1_tilde, eta2_tilde = eta2_tilde,
        T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
        thin = as.integer(thin), n_thin_gb = as.integer(n_thin_gb),
        beta_in = beta_init, gamma_in = as.integer(gamma_init),
        alpha_in = alpha_init
      )
    },
    "ggm" = {
      if (sparse) {
        sp <- .prepare_sparse_S_triplet(X, S_ggm = S_ggm)
        sparse_result <- BayesLogit_SingleNet_SparseGGM(
          X = X, y = y,
          S_i = sp$S_i, S_p_csc = sp$S_p, S_x = sp$S_x,
          S_diag = sp$S_diag, p_ggm = sp$p,
          niter = as.integer(niter), burnin = as.integer(burnin),
          mu = mu, nu0 = nu0, sigmasq0 = sigmasq0,
          alpha0 = alpha0, beta0 = beta0, h = h,
          e_eta = e_eta, f_eta = f_eta,
          v0_ggm = v0_ggm, v1_ggm = v1_ggm,
          pii_ggm = pii_ggm,
          eta1_sd = eta1_sd, mu_tilde = mu_tilde,
          eta1_tilde = eta1_tilde,
          T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
          n_mh_gamma = as.integer(n_mh_gamma),
          thin = as.integer(thin),
          beta_in = beta_init, gamma_in = as.integer(gamma_init),
          alpha_in = alpha_init,
          store_beta = isTRUE(store_beta),
          store_gamma = isTRUE(store_gamma),
          store_Z_list = isTRUE(store_Z_list),
          store_Z_pip = isTRUE(store_Z_pip)
        )

        n_save <- as.integer(niter %/% max(1L, as.integer(thin)))
        sparse_result <- .reconstruct_sparse_pip(sparse_result, p, store_Z_pip)
        sparse_result <- .reconstruct_gamma_pip(sparse_result, p, n_save, store_gamma)
        sparse_result
      } else {
        S_ggm <- crossprod(X)
        BayesLogit_SingleNet_GGM(
          X = X, y = y, S_ggm = S_ggm, n_ggm = as.double(n),
          niter = as.integer(niter), burnin = as.integer(burnin),
          mu = mu, nu0 = nu0, sigmasq0 = sigmasq0,
          alpha0 = alpha0, beta0 = beta0, h = h,
          e_eta = e_eta, f_eta = f_eta,
          v0_ggm = v0_ggm, v1_ggm = v1_ggm,
          pii_ggm = pii_ggm, lambda_ggm = lambda_ggm,
          eta1_sd = eta1_sd, mu_tilde = mu_tilde,
          eta1_tilde = eta1_tilde,
          T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
          thin = as.integer(thin), n_thin_gb = as.integer(n_thin_gb),
          beta_in = beta_init, gamma_in = as.integer(gamma_init),
          alpha_in = alpha_init
        )
      }
    },
    "ggm_fixed" = {
      if (is.null(adj_fixed)) {
        stop("adj_type='ggm_fixed' requires 'adj_fixed'.")
      }
      if (sparse) {
        fixed_sp <- .prepare_sparse_adj_triplet(adj_fixed)
        if (fixed_sp$p != p) {
          stop("adj_fixed dimension must match ncol(X).")
        }
        sp <- .prepare_sparse_S_triplet(X, S_ggm = S_ggm)
        sparse_result <- BayesLogit_DualNet_SparseGGM(
          X = X, y = y,
          S_i = sp$S_i, S_p_csc = sp$S_p, S_x = sp$S_x,
          S_diag = sp$S_diag,
          R_fix_i = fixed_sp$R_i, R_fix_p_csc = fixed_sp$R_p,
          p_ggm = sp$p,
          niter = as.integer(niter), burnin = as.integer(burnin),
          mu = mu, nu0 = nu0, sigmasq0 = sigmasq0,
          alpha0 = alpha0, beta0 = beta0, h = h,
          e_eta = e_eta, f_eta = f_eta,
          v0_ggm = v0_ggm, v1_ggm = v1_ggm,
          pii_ggm = pii_ggm,
          eta1_sd = eta1_sd, eta2_sd = eta2_sd,
          mu_tilde = mu_tilde,
          eta1_tilde = eta1_tilde, eta2_tilde = eta2_tilde,
          T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
          n_mh_gamma = as.integer(n_mh_gamma),
          thin = as.integer(thin),
          beta_in = beta_init, gamma_in = as.integer(gamma_init),
          alpha_in = alpha_init,
          store_beta = isTRUE(store_beta),
          store_gamma = isTRUE(store_gamma),
          store_Z_list = isTRUE(store_Z_list),
          store_Z_pip = isTRUE(store_Z_pip)
        )

        n_save <- as.integer(niter %/% max(1L, as.integer(thin)))
        sparse_result <- .reconstruct_sparse_pip(sparse_result, p, store_Z_pip)
        sparse_result <- .reconstruct_gamma_pip(sparse_result, p, n_save, store_gamma)
        sparse_result
      } else {
        R_fix <- .prepare_adj(adj_fixed, p, "adj_fixed")
        S_ggm <- crossprod(X)
        BayesLogit_DualNet_GGM(
          X = X, y = y, S_ggm = S_ggm, n_ggm = as.double(n),
          R_fix_int = R_fix,
          niter = as.integer(niter), burnin = as.integer(burnin),
          mu = mu, nu0 = nu0, sigmasq0 = sigmasq0,
          alpha0 = alpha0, beta0 = beta0, h = h,
          n_mh_gamma = as.integer(n_mh_gamma),
          v0_ggm = v0_ggm, v1_ggm = v1_ggm,
          pii_ggm = pii_ggm, lambda_ggm = lambda_ggm,
          eta1_sd = eta1_sd, eta2_sd = eta2_sd,
          mu_tilde = mu_tilde,
          eta1_tilde = eta1_tilde, eta2_tilde = eta2_tilde,
          e_eta = e_eta, f_eta = f_eta,
          T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
          thin = as.integer(thin), n_thin_gb = as.integer(n_thin_gb),
          beta_in = beta_init, gamma_in = as.integer(gamma_init),
          alpha_in = alpha_init
        )
      }
    }
  )

  # Package result
  out <- list(
    beta = result$beta,
    gamma = result$gamma,
    alpha = if (!is.null(result$alpha)) as.vector(result$alpha) else NULL,
    sigmasq = if (!is.null(result$sigmasq)) as.vector(result$sigmasq) else NULL,
    eta1 = result$eta1,
    eta2 = result$eta2,
    Z_list = result$Z_list,
    Z_pip = result$Z_pip,
    gamma_pip = result$gamma_pip,
    call = mc,
    adj_type = adj_type,
    sampler = "mh",
    niter = niter,
    burnin = burnin,
    p = p,
    n = n
  )
  class(out) <- "bvs"
  out
}
