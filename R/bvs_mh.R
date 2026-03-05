#' Bayesian Variable Selection via Metropolis-Hastings
#'
#' Bayesian variable selection with an Ising/MRF prior on the inclusion
#' indicators gamma. For \code{outcome_type = "binary"} (default), the model
#' uses logistic regression and Metropolis-Hastings updates. For
#' \code{outcome_type = "continuous"}, the model uses a Gaussian likelihood
#' with conjugate Normal/Inverse-Gamma updates for \code{beta}, \code{alpha},
#' \code{tau}, and \code{sigmasq}. For \code{outcome_type = "TTE"}, the model
#' uses a Cox proportional hazards partial likelihood (Breslow ties) for
#' right-censored time-to-event outcomes. For
#' \code{outcome_type = "count"}, the model uses an overdispersed count
#' likelihood via a Poisson-Gamma mixture (negative-binomial representation).
#' The Ising coupling parameter eta is updated
#' via Moller et al. (2006) auxiliary variable MH with Propp-Wilson perfect
#' simulation.
#'
#' @param X         An \code{n x p} design matrix (numeric).
#' @param y         Response vector of length \code{n}. For
#'   \code{outcome_type = "binary"}, values must be in \code{\{0,1\}} or
#'   \code{\{-1,1\}}. For \code{outcome_type = "continuous"}, values must be
#'   finite numeric. For \code{outcome_type = "TTE"}, values are survival/event
#'   times and must be finite and strictly positive. For
#'   \code{outcome_type = "count"}, values must be finite non-negative integers.
#' @param event      Event indicator vector of length \code{n} for
#'   \code{outcome_type = "TTE"} (\code{1}=event, \code{0}=censored; or
#'   \code{\{-1,1\}} allowed). Ignored for non-TTE outcomes.
#' @param outcome_type Character outcome model:
#'   \code{"binary"} (logistic; default) or \code{"continuous"}
#'   (Gaussian with conjugate normal-inverse-gamma updates) or \code{"TTE"}
#'   (Cox partial likelihood for right-censored time-to-event outcomes) or
#'   \code{"count"} (negative-binomial via Poisson-Gamma augmentation).
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
#'   supply this explicitly for very large \code{p}. For sparse backends with
#'   \code{p >= 10000}, \code{S_ggm} must be provided.
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
#'   (default \code{-log(1/0.1 - 1)}).
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
#' @param e_eta     Beta prior shape \code{a} for eta (default 1).
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
#' @param z_dat      An optional \code{n x q} matrix of additional covariates
#'   whose regression coefficients (tau) are always included in the model
#'   (not subject to variable selection). Default \code{NULL} (no extra
#'   covariates).
#' @param tau0       Prior mean for tau coefficients (default 0).
#' @param htau       Tau variance inflation factor: \code{sigmasq_tau = htau * sigmasq}
#'   (default 1.5).
#' @param tau_init   Optional initial tau (numeric vector of length \code{ncol(z_dat)}).
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
#'     \item{gamma_pip}{Posterior inclusion probabilities for gamma
#'       (length p).}
#'     \item{alpha}{Vector of posterior alpha (intercept) samples.}
#'     \item{sigmasq}{Vector of posterior sigma^2 samples.}
#'     \item{eta1}{Vector of eta1 (dynamic or single-graph coupling) samples
#'       (if applicable).}
#'     \item{eta2}{Vector of eta2 (fixed coupling) samples
#'       (if applicable).}
#'     \item{Z_list}{List of GGM adjacency snapshots (if applicable).}
#'     \item{Z_pip}{Posterior edge inclusion probabilities for GGM adjacency
#'       (if applicable).}
#'     \item{tau}{Matrix of posterior tau (z_dat coefficient) samples
#'       (niter x q), or NULL if z_dat was not supplied.}
#'     \item{call}{The matched call.}
#'     \item{adj_type}{The adjacency type used.}
#'     \item{sampler}{Character: \code{"mh"}.}
#'     \item{outcome_type}{Character outcome model used (\code{"binary"},
#'       \code{"continuous"}, \code{"TTE"}, or \code{"count"}).}
#'     \item{niter}{Number of stored post-burn-in iterations.}
#'     \item{burnin}{Number of burn-in iterations discarded.}
#'     \item{p}{Number of predictors in \code{X}.}
#'     \item{ntau}{Number of always-included covariates in \code{z_dat}.}
#'     \item{n}{Number of observations.}
#'   }
#'
#' @details
#' The outcome model depends on \code{outcome_type}:
#' \deqn{y_i | X, beta, alpha, z_i, tau \sim Bernoulli(logistic(alpha + X_i beta + z_i tau)) \quad \text{if binary}}
#' \deqn{y_i | X, beta, alpha, z_i, tau, \sigma^2 \sim N(alpha + X_i beta + z_i tau,\sigma^2) \quad \text{if continuous}}
#' \deqn{\log L(\beta,\tau) = \sum_{i:\delta_i=1}\left(\eta_i - \log\sum_{j \in \mathcal{R}(t_i)} e^{\eta_j}\right), \ \eta_i = X_i\beta + z_i\tau \quad \text{if TTE}}
#' \deqn{y_i | w_i, X, beta, alpha, z_i, tau \sim \mathrm{Poisson}(w_i \exp(\alpha + X_i\beta + z_i\tau)), \ \ w_i \sim \mathrm{Gamma}(r,r) \quad \text{if count}}
#' \deqn{beta_j | gamma_j, sigma^2 ~ gamma_j N(0, sigma^2) + (1-gamma_j) delta_0}
#' \deqn{tau_k | sigma^2 \sim N(tau0, htau * sigma^2)}
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
#'
#' # Continuous outcome (Gaussian)
#' y_cont <- as.numeric(X %*% beta_true + rnorm(n, sd = 0.5))
#' fit4 <- bvs_mh(
#'   X, y_cont,
#'   outcome_type = "continuous",
#'   adj_type = "fixed", adj_fixed = R,
#'   niter = 5000
#' )
#'
#' # Overdispersed count outcome (negative-binomial representation)
#' y_count <- rnbinom(n, size = 3, mu = exp(0.4 + as.numeric(X %*% beta_true) / 4))
#' fit_count <- bvs_mh(
#'   X, y_count,
#'   outcome_type = "count",
#'   adj_type = "fixed", adj_fixed = R,
#'   niter = 5000
#' )
#'
#' # Time-to-event outcome (Cox partial likelihood)
#' tte_time <- rexp(n, rate = exp(as.numeric(X %*% beta_true) / 4))
#' tte_event <- rbinom(n, 1, 0.7)
#' fit5 <- bvs_mh(
#'   X, tte_time,
#'   event = tte_event, outcome_type = "TTE",
#'   adj_type = "fixed", adj_fixed = R,
#'   niter = 5000
#' )
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
                   event = NULL,
                   outcome_type = c("binary", "continuous", "TTE", "count"),
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
                   mu = -log(1 / 0.1 - 1),
                   alpha0 = 0, beta0 = 0,
                   # Gamma / Ising
                   n_mh_gamma = 3L,
                   eta1_sd = 0.5, eta2_sd = 0.5,
                   mu_tilde = -4,
                   eta1_tilde = 0.075, eta2_tilde = 0.065,
                   e_eta = 1, f_eta = 1,
                   Tmax = 64L,
                   proposal_type = 1L,
                   # GGM SSVS
                   v0_ggm = 0.015^2,
                   v1_ggm = NULL,
                   pii_ggm = NULL,
                   lambda_ggm = 1,
                   # Init (optional)
                   beta_init = NULL,
                   gamma_init = NULL,
                   alpha_init = NULL,
                   z_dat = NULL,
                   tau0 = 0,
                   htau = 1.5,
                   tau_init = NULL) {
  mc <- match.call()
  alpha_init_missing <- is.null(alpha_init)
  if (is.character(outcome_type) && length(outcome_type) == 1L && identical(outcome_type, "contiousou")) {
    outcome_type <- "continuous"
  }
  outcome_type <- match.arg(outcome_type)
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
  if (outcome_type == "binary") {
    y_ok_01 <- all(y %in% c(0, 1))
    y_ok_11 <- all(y %in% c(-1, 1))
    if (!y_ok_01 && y_ok_11) {
      y <- 0.5 * (y + 1)
    } else if (!y_ok_01) {
      stop("For outcome_type='binary', y must be in {0,1} or {-1,1}.")
    }
    if (!is.null(event)) {
      warning("Argument 'event' is ignored unless outcome_type='TTE'.", call. = FALSE)
      event <- NULL
    }
  } else if (outcome_type == "continuous") {
    if (!all(is.finite(y))) {
      stop("For outcome_type='continuous', y must be finite numeric.")
    }
    if (!is.null(event)) {
      warning("Argument 'event' is ignored unless outcome_type='TTE'.", call. = FALSE)
      event <- NULL
    }
  } else if (outcome_type == "TTE") {
    if (!all(is.finite(y))) {
      stop("For outcome_type='TTE', survival times must be finite.")
    }
    if (any(y <= 0)) {
      stop("For outcome_type='TTE', survival times must be > 0.")
    }
    if (is.null(event)) {
      stop("For outcome_type='TTE', 'event' must be provided.")
    }
    event <- as.numeric(event)
    # exact nrow(X) check is done below after X coercion
    ev_ok_01 <- all(event %in% c(0, 1))
    ev_ok_11 <- all(event %in% c(-1, 1))
    if (!ev_ok_01 && ev_ok_11) {
      event <- 0.5 * (event + 1)
    } else if (!ev_ok_01) {
      stop("For outcome_type='TTE', event must be in {0,1} or {-1,1}.")
    }
    if (sum(event == 1) < 1L) {
      stop("For outcome_type='TTE', at least one event==1 is required.")
    }
  } else if (outcome_type == "count") {
    if (!all(is.finite(y))) {
      stop("For outcome_type='count', y must be finite non-negative integers.")
    }
    y_round <- round(y)
    if (any(abs(y - y_round) > 1e-8) || any(y_round < 0)) {
      stop("For outcome_type='count', y must be finite non-negative integers.")
    }
    y <- y_round
    if (!is.null(event)) {
      warning("Argument 'event' is ignored unless outcome_type='TTE'.", call. = FALSE)
      event <- NULL
    }
  } else {
    stop("outcome_type must be one of 'binary', 'continuous', 'TTE', or 'count'.")
  }
  n <- nrow(X)
  p <- ncol(X)
  if (length(y) != n) {
    stop("length(y) must equal nrow(X).")
  }
  if (outcome_type == "TTE" && length(event) != n) {
    stop("For outcome_type='TTE', length(event) must equal nrow(X).")
  }

  # Derived GGM defaults
  if (is.null(v1_ggm)) v1_ggm <- (50^2) * v0_ggm
  if (is.null(pii_ggm)) pii_ggm <- 30 / (p - 1)

  # Initialisation
  if (is.null(beta_init) || is.null(gamma_init) || is.null(alpha_init)) {
    if (use_sparse_backend) {
      init <- .init_ultra_sparse_state(y, p, beta_init, gamma_init, alpha_init, outcome_type)
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
  if (outcome_type == "TTE") {
    alpha_init <- 0
  } else if (outcome_type == "count" && alpha_init_missing) {
    alpha_init <- log(mean(y) + 1e-4)
  }

  # Z_dat defaults
  z_dat_provided <- !is.null(z_dat)
  if (!z_dat_provided) {
    z_dat <- matrix(0, nrow = n, ncol = 1)
  } else {
    z_dat <- as.matrix(z_dat)
    if (nrow(z_dat) != n) stop("nrow(z_dat) must equal nrow(X).")
  }
  ntau <- ncol(z_dat)
  if (is.null(tau_init)) {
    tau_init <- rep(tau0, ntau)
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
        alpha_in = alpha_init,
        Z_dat = z_dat, tau0 = tau0, htau = htau, tau_in = tau_init,
        event = event,
        outcome_type = outcome_type
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
        alpha_in = alpha_init,
        Z_dat = z_dat, tau0 = tau0, htau = htau, tau_in = tau_init,
        event = event,
        outcome_type = outcome_type
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
        alpha_in = alpha_init,
        Z_dat = z_dat, tau0 = tau0, htau = htau, tau_in = tau_init,
        event = event,
        outcome_type = outcome_type
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
        alpha_in = alpha_init,
        Z_dat = z_dat, tau0 = tau0, htau = htau, tau_in = tau_init,
        event = event,
        outcome_type = outcome_type
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
          Z_dat = z_dat, tau0 = tau0, htau = htau, tau_in = tau_init,
          event = event,
          store_beta = isTRUE(store_beta),
          store_gamma = isTRUE(store_gamma),
          store_Z_list = isTRUE(store_Z_list),
          store_Z_pip = isTRUE(store_Z_pip),
          outcome_type = outcome_type
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
          alpha_in = alpha_init,
          Z_dat = z_dat, tau0 = tau0, htau = htau, tau_in = tau_init,
          event = event,
          outcome_type = outcome_type
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
          Z_dat = z_dat, tau0 = tau0, htau = htau, tau_in = tau_init,
          event = event,
          store_beta = isTRUE(store_beta),
          store_gamma = isTRUE(store_gamma),
          store_Z_list = isTRUE(store_Z_list),
          store_Z_pip = isTRUE(store_Z_pip),
          outcome_type = outcome_type
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
          alpha_in = alpha_init,
          Z_dat = z_dat, tau0 = tau0, htau = htau, tau_in = tau_init,
          event = event,
          outcome_type = outcome_type
        )
      }
    }
  )

  # Package result
  gamma_pip <- result$gamma_pip
  if (is.null(gamma_pip) && !is.null(result$gamma)) {
    if (is.matrix(result$gamma)) {
      gamma_pip <- colMeans(result$gamma)
    } else if (is.list(result$gamma)) {
      n_save <- as.integer(niter %/% max(1L, as.integer(thin)))
      tmp <- list(gamma = result$gamma)
      tmp <- .reconstruct_gamma_pip(tmp, p, n_save, store_gamma = TRUE)
      gamma_pip <- tmp$gamma_pip
    }
  }

  out <- list(
    beta = result$beta,
    gamma = result$gamma,
    alpha = if (!is.null(result$alpha)) as.vector(result$alpha) else NULL,
    sigmasq = if (!is.null(result$sigmasq)) as.vector(result$sigmasq) else NULL,
    tau = if (z_dat_provided && !is.null(result$tau)) result$tau else NULL,
    eta1 = result$eta1,
    eta2 = result$eta2,
    Z_list = result$Z_list,
    Z_pip = result$Z_pip,
    gamma_pip = gamma_pip,
    call = mc,
    adj_type = adj_type,
    sampler = "mh",
    outcome_type = outcome_type,
    niter = niter,
    burnin = burnin,
    p = p,
    ntau = ntau,
    n = n
  )
  class(out) <- "bvs"
  out
}
