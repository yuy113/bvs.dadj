#' Prepare Sparse Sample Covariance for C++ GGM Backend
#'
#' Thresholds the sample covariance matrix S and converts it to compressed
#' sparse column (CSC) format for efficient input to the sparse GGM C++ backends.
#'
#' @param X  An \code{n x p} design matrix (observations in rows).
#' @param threshold  Entries in S with |S_ij| < threshold are zeroed.
#'   Default \code{1e-4}.
#'
#' @return A named list with components:
#'   \describe{
#'     \item{S_i}{Integer vector of row indices (0-based, CSC).}
#'     \item{S_p}{Integer vector of column pointers (0-based, CSC, length p+1).}
#'     \item{S_x}{Numeric vector of nonzero values.}
#'     \item{S_diag}{Numeric vector of diagonal entries of S (length p).}
#'     \item{p}{Integer, number of variables (columns of X).}
#'   }
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(500 * 20), 500, 20)
#' sp <- prepare_sparse_S(X)
#' str(sp)
#'
#' @export
prepare_sparse_S <- function(X, threshold = 1e-4) {

  if (!is.matrix(X)) X <- as.matrix(X)

  p <- ncol(X)
  n <- nrow(X)
  S <- crossprod(X)          # p x p scatter matrix (X'X)
  S_diag <- diag(S)

  # Threshold small off-diagonal entries
  S_thr <- S
  diag(S_thr) <- 0
  S_thr[abs(S_thr) < threshold] <- 0

  # Upper triangle (column-major) -> CSC
  S_upper <- S_thr
  S_upper[lower.tri(S_upper)] <- 0

  sm <- Matrix::sparseMatrix(i = which(S_upper != 0, arr.ind = TRUE)[, 1],
                             j = which(S_upper != 0, arr.ind = TRUE)[, 2],
                             x = S_upper[S_upper != 0],
                             dims = c(p, p))

  sm_summary <- Matrix::summary(sm)
  S_csc <- as(sm, "CsparseMatrix")

  list(
    S_i    = as.integer(S_csc@i),          # 0-based row indices
    S_p    = as.integer(S_csc@p),          # 0-based col pointers
    S_x    = as.double(S_csc@x),           # nonzero values
    S_diag = as.double(S_diag),            # diagonal
    p      = as.integer(p)
  )
}


#' Estimate Adjacency Matrix via Graphical Lasso
#'
#' Uses the \pkg{huge} package to estimate a sparse precision matrix via the
#' graphical lasso algorithm and extract the binary adjacency matrix.
#'
#' @param X  An \code{n x p} design matrix.
#' @param criterion  Model selection criterion: \code{"ebic"} (extended BIC,
#'   default) or \code{"ric"} (rotation information criterion).
#' @param nlambda  Number of lambda values for the regularization path
#'   (default 30).
#' @param lambda.min.ratio  Ratio of smallest to largest lambda (default 0.01).
#' @param symmetrize  Logical; symmetrize the adjacency and keep upper
#'   triangle only (default TRUE).
#'
#' @return A \code{p x p} integer adjacency matrix (0/1), symmetric with
#'   zero diagonal.
#'
#' @details Requires the \pkg{huge} package to be installed.  The function
#'   runs \code{huge()} with \code{method = "glasso"} and selects the optimal
#'   model via \code{huge.select()} with the specified criterion.
#'
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 200; p <- 50
#' X <- matrix(rnorm(n * p), n, p)
#' adj <- estimate_glasso_adj(X, criterion = "ebic")
#' sum(adj)  # number of detected edges
#' }
#'
#' @export
estimate_glasso_adj <- function(X,
                                criterion = c("ebic", "ric"),
                                nlambda = 30,
                                lambda.min.ratio = 0.01,
                                symmetrize = TRUE) {

  criterion <- match.arg(criterion)

  if (!requireNamespace("huge", quietly = TRUE)) {
    stop("Package 'huge' is required. Install it via install.packages('huge').")
  }

  fit <- huge::huge(X, method = "glasso",
                    nlambda = nlambda,
                    lambda.min.ratio = lambda.min.ratio,
                    verbose = FALSE)
  sel <- huge::huge.select(fit, criterion = criterion, verbose = FALSE)

  # Extract precision matrix -> binary adjacency
  prec <- as.matrix(sel$opt.icov)
  p    <- ncol(prec)
  adj  <- matrix(as.integer(prec != 0), p, p)
  diag(adj) <- 0L

  if (symmetrize) {
    adj <- pmax(adj, t(adj))           # symmetrize
    adj[lower.tri(adj)] <- 0L          # upper-tri only
  }

  adj
}


#' Initialize MCMC Parameters for BVS
#'
#' Generates default initial values for beta, gamma, alpha, and sigma^2.
#'
#' @param p     Number of predictors.
#' @param mu    Ising external field.
#' @param nu0   IG shape parameter.
#' @param sigmasq0  IG scale parameter.
#' @param alpha0  Prior mean for intercept.
#' @param beta0   Prior mean for coefficients.
#' @param h     Intercept variance inflation.
#'
#' @return A named list with components: beta_init, gamma_init, alpha_init.
#'
#' @keywords internal
.init_mcmc <- function(p, mu, nu0, sigmasq0, alpha0, beta0, h) {
  omega <- exp(mu) / (1 + exp(mu))
  gamma_init <- rbinom(p, 1, omega)
  beta_init  <- rep(0, p)
  sigmasq    <- 1 / rgamma(1, nu0 / 2, nu0 * sigmasq0 / 2)
  sigma_beta <- sqrt(sigmasq)
  sigma_alpha <- sqrt(h * sigmasq)
  alpha_init  <- rnorm(1, alpha0, sigma_alpha)
  beta_init[gamma_init == 1] <- rnorm(sum(gamma_init), beta0, sigma_beta)

  list(beta_init  = as.numeric(beta_init),
       gamma_init = as.integer(gamma_init),
       alpha_init = as.double(alpha_init))
}


#' Prepare and Validate a Fixed Adjacency Matrix
#'
#' Ensures the matrix is symmetric, binary, integer, with zero diagonal.
#'
#' @param adj A matrix (p x p).
#' @param p   Expected dimension.
#' @param name Character name for error messages.
#'
#' @return An integer matrix (p x p).
#'
#' @keywords internal
.prepare_adj <- function(adj, p, name = "adj") {
  adj <- as.matrix(adj)
  if (nrow(adj) != p || ncol(adj) != p) {
    stop(sprintf("'%s' must be a %d x %d matrix", name, p, p))
  }
  adj <- (adj + t(adj))
  adj <- ifelse(adj > 0, 1L, 0L)
  diag(adj) <- 0L
  storage.mode(adj) <- "integer"
  adj
}
