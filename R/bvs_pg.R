#' Bayesian Variable Selection via Polya-Gamma Augmentation
#'
#' Bayesian variable selection for logistic regression with an Ising/MRF prior
#' on the inclusion indicators gamma, using Polya-Gamma data augmentation for
#' exact Gibbs sampling of beta and alpha.  The Ising coupling parameter eta is
#' updated via Moller et al. (2006) auxiliary variable MH with Propp-Wilson
#' perfect simulation.
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
#'     \item{\code{"ggm"}}{Single adjacency learned jointly via Bayesian GGM
#'       SSVS (Wang et al. 2012) during MCMC.}
#'     \item{\code{"ggm_fixed"}}{Two sources: one learned from Bayesian GGM
#'       and one fixed (supplied via \code{adj_fixed}).}
#'   }
#' @param adj_fixed   A \code{p x p} binary adjacency matrix (for
#'   \code{"fixed"}, \code{"dual_fixed"}, \code{"ggm_fixed"}).
#' @param adj_fixed2  A second \code{p x p} binary adjacency matrix
#'   (for \code{"dual_fixed"} only).
#' @param sparse    Logical; use the sparse high-dimensional backend
#'   (default \code{FALSE}).  Applicable for \code{adj_type = "ggm"} or
#'   \code{"ggm_fixed"}.
#' @param glasso_criterion  Character; criterion for graphical lasso model
#'   selection: \code{"ebic"} (default) or \code{"ric"}.
#'
#' @param niter     Number of post-burn-in MCMC iterations (default 60000).
#' @param burnin    Number of burn-in iterations (default 10000).
#' @param thin      Thinning interval (default 1, only for sparse backends).
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
#'   (default 3).
#' @param eta_sd    Upper bound for eta (single-eta models, default 0.5).
#' @param eta1_sd   Upper bound for eta1 (dual-eta models, default 0.5).
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
#'     \item{gamma}{Matrix of posterior gamma samples (niter x p).}
#'     \item{alpha}{Vector of posterior alpha (intercept) samples.}
#'     \item{sigmasq}{Vector of posterior sigma^2 samples.}
#'     \item{eta_dyn}{Vector of eta1 (dynamic GGM coupling) samples
#'       (if applicable).}
#'     \item{eta_fix}{Vector of eta2 (fixed coupling) samples
#'       (if applicable).}
#'     \item{Z_list}{List of GGM adjacency snapshots (if applicable).}
#'     \item{call}{The matched call.}
#'     \item{adj_type}{The adjacency type used.}
#'     \item{sampler}{Character: \code{"pg"}.}
#'   }
#'
#' @details
#' This function uses Polya-Gamma data augmentation (Polson, Scott, and
#' Windle 2013) to replace the Metropolis-Hastings updates for beta and
#' alpha with exact Gibbs sampling steps, improving mixing and convergence.
#'
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 200; p <- 50
#' X <- matrix(rnorm(n * p), n, p)
#' beta_true <- c(rep(1, 5), rep(0, p - 5))
#' y <- rbinom(n, 1, plogis(X %*% beta_true))
#'
#' # PG with glasso adjacency (EBIC selection)
#' fit <- bvs_pg(X, y, adj_type = "glasso", glasso_criterion = "ebic",
#'               niter = 5000)
#' summary(fit)
#'
#' # PG with sparse GGM adjacency
#' fit2 <- bvs_pg(X, y, adj_type = "ggm", sparse = TRUE, niter = 5000)
#' }
#'
#' @seealso \code{\link{bvs_mh}}, \code{\link{summary.bvs}},
#'   \code{\link{estimate_glasso_adj}}
#'
#' @export
bvs_pg <- function(X, y,
                   adj_type = c("fixed", "dual_fixed", "glasso",
                                "ggm", "ggm_fixed"),
                   adj_fixed  = NULL,
                   adj_fixed2 = NULL,
                   sparse = FALSE,
                   glasso_criterion = c("ebic", "ric"),

                   # MCMC control
                   niter  = 60000L,
                   burnin = 10000L,
                   thin   = 1L,

                   # Variable selection priors
                   nu0 = 2, sigmasq0 = 1.5, h = 1.5,
                   mu = -log(1/0.3 - 1),
                   alpha0 = 0, beta0 = 0,

                   # Gamma / Ising
                   n_mh_gamma = 3L,
                   eta_sd  = 0.5,
                   eta1_sd = 0.5, eta2_sd = 0.5,
                   mu_tilde = -4,
                   eta1_tilde = 0.075, eta2_tilde = 0.065,
                   e_eta = 2, f_eta = 1,
                   Tmax = 64L,
                   proposal_type = 1L,

                   # GGM SSVS
                   v0_ggm = 0.015^2,
                   v1_ggm = NULL,
                   pii_ggm = NULL,
                   lambda_ggm = 1,

                   # Init (optional)
                   beta_init  = NULL,
                   gamma_init = NULL,
                   alpha_init = NULL) {

  mc <- match.call()
  adj_type <- match.arg(adj_type)
  glasso_criterion <- match.arg(glasso_criterion)

  # Dimensions
  X <- as.matrix(X)
  y <- as.numeric(y)
  n <- nrow(X); p <- ncol(X)

  # Derived GGM defaults
  if (is.null(v1_ggm)) v1_ggm <- (50^2) * v0_ggm
  if (is.null(pii_ggm)) pii_ggm <- 30 / (p - 1)

  # Initialisation
  if (is.null(beta_init) || is.null(gamma_init) || is.null(alpha_init)) {
    init <- .init_mcmc(p, mu, nu0, sigmasq0, alpha0, beta0, h)
    if (is.null(beta_init))  beta_init  <- init$beta_init
    if (is.null(gamma_init)) gamma_init <- init$gamma_init
    if (is.null(alpha_init)) alpha_init <- init$alpha_init
  }

  # ---- Dispatch ----
  result <- switch(adj_type,

    "fixed" = {
      if (is.null(adj_fixed))
        stop("adj_type='fixed' requires 'adj_fixed'.")
      R_mat <- .prepare_adj(adj_fixed, p, "adj_fixed")
      BayesLogit_PG_SingleAdj(
        X = X, y = y, R_adj_int = R_mat,
        niter = as.integer(niter), burnin = as.integer(burnin),
        mu = mu, nu0 = nu0, sigmasq0 = sigmasq0,
        alpha0 = alpha0, beta0 = beta0, h = h,
        n_mh_gamma = as.integer(n_mh_gamma),
        eta_sd = eta_sd, mu_tilde = mu_tilde,
        eta_tilde = eta1_tilde,
        e = e_eta, f = f_eta,
        T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
        beta_init = beta_init, gamma_init = as.integer(gamma_init),
        alpha_in = alpha_init)
    },

    "dual_fixed" = {
      if (is.null(adj_fixed) || is.null(adj_fixed2))
        stop("adj_type='dual_fixed' requires both 'adj_fixed' and 'adj_fixed2'.")
      R1 <- .prepare_adj(adj_fixed,  p, "adj_fixed")
      R2 <- .prepare_adj(adj_fixed2, p, "adj_fixed2")
      BayesLogit_PG_DualAdj(
        X = X, y = y,
        R_glasso_int = R1, R_fix_int = R2,
        niter = as.integer(niter), burnin = as.integer(burnin),
        mu = mu, nu0 = nu0, sigmasq0 = sigmasq0,
        alpha0 = alpha0, beta0 = beta0, h = h,
        n_mh_gamma = as.integer(n_mh_gamma),
        eta1_sd = eta1_sd, eta2_sd = eta2_sd,
        mu_tilde = mu_tilde,
        eta1_tilde = eta1_tilde, eta2_tilde = eta2_tilde,
        e = e_eta, f = f_eta,
        T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
        beta_init = beta_init, gamma_init = as.integer(gamma_init),
        alpha_in = alpha_init)
    },

    "glasso" = {
      adj_est <- estimate_glasso_adj(X, criterion = glasso_criterion)
      R_mat <- .prepare_adj(adj_est, p, "glasso_adj")
      BayesLogit_PG_SingleAdj(
        X = X, y = y, R_adj_int = R_mat,
        niter = as.integer(niter), burnin = as.integer(burnin),
        mu = mu, nu0 = nu0, sigmasq0 = sigmasq0,
        alpha0 = alpha0, beta0 = beta0, h = h,
        n_mh_gamma = as.integer(n_mh_gamma),
        eta_sd = eta_sd, mu_tilde = mu_tilde,
        eta_tilde = eta1_tilde,
        e = e_eta, f = f_eta,
        T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
        beta_init = beta_init, gamma_init = as.integer(gamma_init),
        alpha_in = alpha_init)
    },

    "ggm" = {
      if (sparse) {
        sp <- prepare_sparse_S(X)
        BayesLogit_PG_SingleAdj_SparseGGM(
          X = X, y = y,
          S_i = sp$S_i, S_p_csc = sp$S_p, S_x = sp$S_x,
          S_diag = sp$S_diag, p_ggm = sp$p,
          niter = as.integer(niter), burnin = as.integer(burnin),
          mu = mu, nu0 = nu0, sigmasq0 = sigmasq0,
          alpha0 = alpha0, beta0 = beta0, h = h,
          n_mh_gamma = as.integer(n_mh_gamma),
          v0_ggm = v0_ggm, v1_ggm = v1_ggm,
          pii_ggm = pii_ggm,
          eta_sd = eta_sd, mu_tilde = mu_tilde,
          eta1_tilde = eta1_tilde,
          e_eta = e_eta, f_eta = f_eta,
          T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
          thin = as.integer(thin),
          beta_in = beta_init, gamma_in = as.integer(gamma_init),
          alpha_in = alpha_init)
      } else {
        S_ggm <- crossprod(X)
        BayesLogit_PG_SingleAdj_GGM_Moller(
          X = X, y = y, S_ggm = S_ggm,
          n_ggm = as.double(n),
          niter = as.integer(niter), burnin = as.integer(burnin),
          mu = mu, nu0 = nu0, sigmasq0 = sigmasq0,
          alpha0 = alpha0, beta0 = beta0, h = h,
          n_mh_gamma = as.integer(n_mh_gamma),
          v0_ggm = v0_ggm, v1_ggm = v1_ggm,
          pii_ggm = pii_ggm, lambda_ggm = lambda_ggm,
          eta_sd = eta_sd, mu_tilde = mu_tilde,
          eta1_tilde = eta1_tilde,
          e_eta = e_eta, f_eta = f_eta,
          T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
          thin = as.integer(thin),
          beta_init = beta_init, gamma_init = as.integer(gamma_init),
          alpha_in = alpha_init)
      }
    },

    "ggm_fixed" = {
      if (is.null(adj_fixed))
        stop("adj_type='ggm_fixed' requires 'adj_fixed'.")
      R_fix <- .prepare_adj(adj_fixed, p, "adj_fixed")
      if (sparse) {
        sp <- prepare_sparse_S(X)
        BayesLogit_PG_DualNet_SparseGGM(
          X = X, y = y,
          S_i = sp$S_i, S_p_csc = sp$S_p, S_x = sp$S_x,
          S_diag = sp$S_diag,
          R_fix_int = R_fix, p_ggm = sp$p,
          niter = as.integer(niter), burnin = as.integer(burnin),
          mu = mu, nu0 = nu0, sigmasq0 = sigmasq0,
          alpha0 = alpha0, beta0 = beta0, h = h,
          n_mh_gamma = as.integer(n_mh_gamma),
          v0_ggm = v0_ggm, v1_ggm = v1_ggm,
          pii_ggm = pii_ggm,
          eta1_sd = eta1_sd, eta2_sd = eta2_sd,
          mu_tilde = mu_tilde,
          eta1_tilde = eta1_tilde, eta2_tilde = eta2_tilde,
          e_eta = e_eta, f_eta = f_eta,
          T_max = as.integer(Tmax), proposal_type = as.integer(proposal_type),
          thin = as.integer(thin),
          beta_in = beta_init, gamma_in = as.integer(gamma_init),
          alpha_in = alpha_init)
      } else {
        S_ggm <- crossprod(X)
        BayesLogit_PG_GGM_Moller(
          X = X, y = y, S_ggm = S_ggm,
          n_ggm = as.double(n), R_fix = R_fix,
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
          beta_init = beta_init, gamma_init = as.integer(gamma_init),
          alpha_in = alpha_init)
      }
    }
  )

  # Package result
  out <- list(
    beta    = result$beta,
    gamma   = result$gamma,
    alpha   = if (!is.null(result$alpha)) as.vector(result$alpha) else NULL,
    sigmasq = if (!is.null(result$sigmasq)) as.vector(result$sigmasq) else NULL,
    eta_dyn = if (!is.null(result$eta_dyn)) result$eta_dyn else result$eta1,
    eta_fix = if (!is.null(result$eta_fix)) result$eta_fix else result$eta2,
    Z_list  = result$Z_list,
    Z_pip   = result$Z_pip,
    call    = mc,
    adj_type = adj_type,
    sampler  = "pg",
    niter    = niter,
    burnin   = burnin,
    p        = p,
    n        = n
  )
  class(out) <- "bvs"
  out
}
