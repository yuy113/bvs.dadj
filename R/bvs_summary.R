#' Summarize BVS Results
#'
#' Computes posterior inclusion probabilities (PIP), posterior means,
#' credible intervals (quantile-based), Highest Posterior Density (HPD)
#' intervals (via \code{coda}), and MCMC convergence diagnostics (bulk and
#' tail Effective Sample Size following Vehtari et al. 2021).
#'
#' @param object  An object of class \code{"bvs"}.
#' @param pip_threshold  Threshold on PIP for declaring a variable selected
#'   (default 0.5).
#' @param cred_level  Probability level for quantile-based credible intervals
#'   (default 0.95).
#' @param hpd_level  Probability level for HPD intervals (default 0.95).
#' @param ...  Additional arguments (ignored).
#'
#' @return A list with:
#'   \describe{
#'     \item{pip}{Posterior inclusion probabilities (numeric vector, length \code{p}).}
#'     \item{selected}{Integer indices of selected variables (PIP > \code{pip_threshold}).}
#'     \item{nselected}{Number of selected variables.}
#'     \item{summary_beta}{Data frame with columns \code{Mean}, \code{SD},
#'       \code{PIP}, \code{Q_low}, \code{Q_up}, \code{HPD_low}, \code{HPD_up},
#'       \code{ESS} (bulk ESS, rank-normalised), \code{ESS_tail} (tail ESS =
#'       min ESS at 5th and 95th percentiles). Rows correspond to predictors.
#'       ESS and ESS_tail are computed via \code{\link{bvs_ess_bulk}} and
#'       \code{\link{bvs_ess_tail}} (Vehtari et al. 2021).}
#'     \item{summary_alpha}{Named numeric vector for intercept: Mean, SD,
#'       quantile bounds, HPD bounds, and ESS.}
#'     \item{summary_tau}{Data frame for always-included covariates (if any).}
#'     \item{cred_level}{Credible interval level used.}
#'     \item{hpd_level}{HPD interval level used.}
#'     \item{sampler}{Character sampler identifier from the fit object.}
#'     \item{adj_type}{Character adjacency type from the fit object.}
#'   }
#'
#' @references
#'   Vehtari A, Gelman A, Simpson D, Carpenter B, Burkner P-C (2021).
#'   Rank-normalization, folding, and localization: An improved R-hat for
#'   assessing convergence of MCMC. \emph{Bayesian Analysis} 16(2), 667--718.
#'   DOI: 10.1214/20-BA1221.
#'
#' @examples
#' \dontrun{
#' fit <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R, niter = 5000)
#' s <- summary(fit, cred_level = 0.90, hpd_level = 0.90)
#' head(s$summary_beta)
#' # ESS and ESS_tail (Vehtari et al. 2021) for each predictor:
#' s$summary_beta[, c("ESS", "ESS_tail")]
#' plot(fit, type = "trace")
#' }
#'
#' @importFrom coda HPDinterval mcmc effectiveSize
#' @method summary bvs
#' @export
summary.bvs <- function(object, pip_threshold = 0.5,
                        cred_level = 0.95, hpd_level = 0.95, ...) {
  pip <- .extract_pip_from_bvs(object)
  selected <- which(pip > pip_threshold)

  # Extract beta samples (handling sparse list vs dense matrix)
  beta_samples <- .get_beta_matrix(object)
  p <- object$p

  # Initialize results
  beta_mean <- numeric(p)
  beta_sd <- numeric(p)
  beta_q_low <- numeric(p)
  beta_q_up <- numeric(p)
  beta_hpd_low <- numeric(p)
  beta_hpd_up <- numeric(p)
  beta_ess <- rep(NA_real_, p)
  beta_ess_tail <- rep(NA_real_, p)

  q_probs <- c((1 - cred_level) / 2, 1 - (1 - cred_level) / 2)

  if (!is.null(beta_samples) && nrow(beta_samples) > 0) {
    beta_mean <- colMeans(beta_samples)
    beta_sd <- apply(beta_samples, 2, stats::sd)

    # Quantiles
    quants <- apply(beta_samples, 2, stats::quantile, probs = q_probs)
    beta_q_low <- quants[1, ]
    beta_q_up <- quants[2, ]

    # HPD and ESS using coda
    mcmc_obj <- coda::mcmc(beta_samples)

    # HPD
    hpd_res <- tryCatch(
      {
        coda::HPDinterval(mcmc_obj, prob = hpd_level)
      },
      error = function(e) {
        matrix(0, ncol = 2, nrow = p)
      }
    )

    if (nrow(hpd_res) == p) {
      beta_hpd_low <- hpd_res[, 1]
      beta_hpd_up <- hpd_res[, 2]
    }

    # ESS — use rank-normalized bulk + tail ESS (Vehtari et al. 2021)
    beta_ess <- tryCatch(
      apply(beta_samples, 2, bvs_ess_bulk),
      error = function(e) rep(NA_real_, p)
    )
    beta_ess_tail <- tryCatch(
      apply(beta_samples, 2, bvs_ess_tail),
      error = function(e) rep(NA_real_, p)
    )
  }

  summary_beta <- data.frame(
    Mean = beta_mean,
    SD = beta_sd,
    PIP = pip,
    Q_low = beta_q_low,
    Q_up = beta_q_up,
    HPD_low = beta_hpd_low,
    HPD_up = beta_hpd_up,
    ESS = beta_ess,
    ESS_tail = beta_ess_tail
  )
  colnames(summary_beta)[4:7] <- c(
    paste0("Q", round(q_probs[1] * 100, 1)),
    paste0("Q", round(q_probs[2] * 100, 1)),
    paste0("HPD", round(hpd_level * 100, 1), "_low"),
    paste0("HPD", round(hpd_level * 100, 1), "_up")
  )

  # Alpha summary
  summary_alpha <- NULL
  if (!is.null(object$alpha)) {
    alpha_vec <- as.numeric(object$alpha)
    alpha_mcmc <- coda::mcmc(alpha_vec)

    a_mean <- mean(alpha_vec)
    a_sd <- stats::sd(alpha_vec)
    a_quant <- stats::quantile(alpha_vec, probs = q_probs)
    a_hpd <- tryCatch(coda::HPDinterval(alpha_mcmc, prob = hpd_level), error = function(e) c(NA, NA))
    a_ess <- tryCatch(bvs_ess_bulk(alpha_vec), error = function(e) NA)

    summary_alpha <- c(
      Mean = a_mean,
      SD = a_sd,
      Q_low = a_quant[1],
      Q_up = a_quant[2],
      HPD_low = a_hpd[1],
      HPD_up = a_hpd[2],
      ESS = a_ess
    )
    names(summary_alpha)[3:6] <- c(
      paste0("Q", round(q_probs[1] * 100, 1)),
      paste0("Q", round(q_probs[2] * 100, 1)),
      paste0("HPD", round(hpd_level * 100, 1), "_low"),
      paste0("HPD", round(hpd_level * 100, 1), "_up")
    )
  }

  # --- tau summary ---
  tau_summary <- NULL
  if (!is.null(object$tau) && ncol(object$tau) > 0) {
    ntau <- ncol(object$tau)
    tau_mean <- colMeans(object$tau)
    tau_sd <- apply(object$tau, 2, stats::sd)
    # SUM-1: Use cred_level for tau quantiles (was hardcoded 0.025/0.975)
    tau_q_probs <- c((1 - cred_level) / 2, 0.5, 1 - (1 - cred_level) / 2)
    tau_q <- t(apply(object$tau, 2, quantile, probs = tau_q_probs))
    tau_summary <- data.frame(
      Variable = paste0("tau", seq_len(ntau)),
      Mean = tau_mean,
      SD = tau_sd,
      Q_low = tau_q[, 1],
      Median = tau_q[, 2],
      Q_up = tau_q[, 3]
    )
    colnames(tau_summary)[4] <- paste0("Q", round(tau_q_probs[1] * 100, 1))
    colnames(tau_summary)[6] <- paste0("Q", round(tau_q_probs[3] * 100, 1))
    if (requireNamespace("coda", quietly = TRUE)) {
      tau_ess <- apply(object$tau, 2, bvs_ess_bulk)
      tau_summary$ESS <- tau_ess
    }
  }

  out <- list(
    pip = pip,
    selected = selected,
    summary_beta = summary_beta,
    summary_alpha = summary_alpha,
    tau_summary = tau_summary,
    nselected = length(selected),
    sampler = object$sampler,
    adj_type = object$adj_type,
    outcome_type = object$outcome_type,
    p = object$p,
    n = object$n,
    niter = object$niter,
    cred_level = cred_level,
    hpd_level = hpd_level,
    zinb_pi = object$zinb_pi,
    zinb_r_final = object$zinb_r_final
  )
  class(out) <- "summary.bvs"
  out
}

#' @method print summary.bvs
#' @importFrom utils head
#' @export
print.summary.bvs <- function(x, ...) {
  cat("Bayesian Variable Selection (", toupper(x$sampler),
    ", adj_type='", x$adj_type, "')\n",
    sep = ""
  )
  cat("  n =", x$n, ", p =", x$p, ", niter =", x$niter, "\n")
  cat("  Variables selected (PIP > 0.5):", x$nselected, "\n")

  if (!is.null(x$summary_alpha)) {
    cat("\nIntercept (Alpha):\n")
    print(round(x$summary_alpha, 4))
  }

  if (!is.null(x$tau_summary)) {
    cat("\nAdditional covariate coefficients (tau):\n")
    print(x$tau_summary, row.names = FALSE)
  }

  if (!is.null(x$zinb_pi) || !is.null(x$zinb_r_final)) {
    cat("\nZero-Inflated Count (ZIC) Parameters:\n")
    if (!is.null(x$zinb_pi))
      cat("  Estimated pi (zero-inflation prob):", round(x$zinb_pi, 4), "\n")
    if (!is.null(x$zinb_r_final))
      cat("  Final r (NB dispersion):           ", round(x$zinb_r_final, 4), "\n")
  }

  if (x$nselected > 0) {
    cat("\nSelected Variables (Top 10 by PIP):\n")
    sel_df <- x$summary_beta[x$selected, , drop = FALSE]
    sel_df <- sel_df[order(sel_df$PIP, decreasing = TRUE), , drop = FALSE]
    print(head(round(sel_df, 4), 10))
    if (nrow(sel_df) > 10) cat("... (", nrow(sel_df) - 10, " more selected)\n")
  } else {
    cat("\nNo variables selected.\n")
  }
  invisible(x)
}


#' Print BVS Object
#'
#' @param x An object of class \code{"bvs"}.
#' @param ... Additional arguments (ignored).
#'
#' @method print bvs
#' @export
print.bvs <- function(x, ...) {
  cat("BayesVarSel fit (", toupper(x$sampler),
    ", adj_type='", x$adj_type, "')\n",
    sep = ""
  )
  cat("  n =", x$n, ", p =", x$p, "\n")
  cat("  Iterations:", x$niter, "\n")
  cat("  Use summary() for posterior summaries and diagnostics.\n")
  invisible(x)
}


#' Plot BVS Results (PIP, Trace, ACF)
#'
#' Provides plots for Posterior Inclusion Probabilities (PIP),
#' trace plots of coefficients, and autocorrelation functions (ACF)
#' for MCMC convergence assessment.
#'
#' @param x  An object of class \code{"bvs"}.
#' @param type  Type of plot: \code{"pip"} (bar plot of PIPs),
#'   \code{"trace"} (trace plots of coefficients),
#'   \code{"acf"} (autocorrelation of coefficients),
#'   or \code{"all"} (all of the above).
#' @param pip_threshold  Threshold line for PIP plot (default 0.5).
#' @param top_n  Number of top variables (by PIP) to show in trace/ACF plots (default 5).
#' @param vars  Optional vector of indices or names of specific variables to plot.
#'   Overrides \code{top_n} if provided.
#' @param ...  Additional graphical arguments passed to plotting functions.
#'
#' @examples
#' \dontrun{
#' fit <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R, niter = 5000)
#' plot(fit, type = "pip")
#' plot(fit, type = "trace", top_n = 4)
#' }
#'
#' @importFrom graphics barplot abline legend par plot lines text box grid
#' @importFrom stats acf
#' @method plot bvs
#' @export
plot.bvs <- function(x, type = c("pip", "trace", "acf", "all"),
                     pip_threshold = 0.5, top_n = 5, vars = NULL, ...) {
  type <- match.arg(type)

  pip <- .extract_pip_from_bvs(x)
  p <- x$p

  plot_indices <- NULL
  if (!is.null(vars)) {
    if (is.character(vars)) stop("Please provide numeric indices for 'vars'.")
    plot_indices <- vars
  } else {
    plot_indices <- order(pip, decreasing = TRUE)[1:min(top_n, p)]
  }
  plot_indices <- plot_indices[plot_indices >= 1 & plot_indices <= p]

  if (type == "all") {
    oask <- devAskNewPage(TRUE)
    on.exit(devAskNewPage(oask))
    plot.bvs(x, type = "pip", pip_threshold = pip_threshold, ...)
    plot.bvs(x, type = "trace", vars = plot_indices, ...)
    plot.bvs(x, type = "acf", vars = plot_indices, ...)
    return(invisible(NULL))
  }

  if (type == "pip") {
    barplot(pip,
      las = 2, ylim = c(0, 1),
      main = paste0(
        "Posterior Inclusion Probabilities (",
        toupper(x$sampler), ")"
      ),
      ylab = "PIP", col = ifelse(pip > pip_threshold, "steelblue", "grey70"),
      border = NA, ...
    )
    abline(h = pip_threshold, lty = 2, col = "red", lwd = 1.5)
    legend("topright",
      legend = paste("threshold =", pip_threshold),
      lty = 2, col = "red", bty = "n"
    )
  } else if (type == "trace") {
    beta_mat <- .get_beta_matrix(x, indices = plot_indices)
    n_plots <- length(plot_indices)
    rows <- floor(sqrt(n_plots))
    cols <- ceiling(n_plots / rows)
    op <- par(mfrow = c(rows, cols), mar = c(2, 4, 2, 1))
    on.exit(par(op))

    for (i in seq_along(plot_indices)) {
      idx <- plot_indices[i]
      vals <- beta_mat[, match(idx, plot_indices)]
      plot(vals,
        type = "l", main = paste0("Trace: Beta[", idx, "]"),
        ylab = "Value", xlab = "Iter", col = "steelblue", ...
      )
      grid()
    }
  } else if (type == "acf") {
    beta_mat <- .get_beta_matrix(x, indices = plot_indices)
    n_plots <- length(plot_indices)
    rows <- floor(sqrt(n_plots))
    cols <- ceiling(n_plots / rows)
    op <- par(mfrow = c(rows, cols), mar = c(2, 4, 2, 1))
    on.exit(par(op))

    for (i in seq_along(plot_indices)) {
      idx <- plot_indices[i]
      vals <- beta_mat[, match(idx, plot_indices)]
      if (stats::var(vals) < 1e-12) {
        plot(0, 0, type = "n", main = paste0("ACF: Beta[", idx, "]"), xlab = "Lag", ylab = "ACF", axes = FALSE)
        text(0, 0, "Constant Chain")
        box()
      } else {
        stats::acf(vals, main = paste0("ACF: Beta[", idx, "]"), ...)
      }
    }
  }
}

#' Convert BVS Object to MCMC Object
#'
#' Extracts posterior samples of coefficients (beta) and intercept (alpha)
#' as a \code{coda::mcmc} object.
#'
#' @param x An object of class \code{"bvs"}.
#' @param vars Optional vector of variable indices or names to include.
#'   If NULL, includes all variables (warning: high memory usage for large p).
#'   Can be useful to focus diagnostics on a subset of interesting variables.
#' @param ... Additional arguments (ignored).
#'
#' @return An object of class \code{"mcmc"}.
#' @importFrom coda mcmc as.mcmc
#' @method as.mcmc bvs
#' @export
as.mcmc.bvs <- function(x, vars = NULL, ...) {
  beta_mat <- .get_beta_matrix(x, indices = vars)

  if (!is.null(x$alpha)) {
    alpha_vec <- as.numeric(x$alpha)
    if (nrow(beta_mat) != length(alpha_vec)) {
      warning("Alpha and Beta chain lengths differ? Using Beta length.")
    }
    # Bind alpha as first column
    out <- cbind(Alpha = alpha_vec, beta_mat)
  } else {
    out <- beta_mat
  }

  # Add tau columns if present
  if (!is.null(x$tau)) {
    colnames_tau <- paste0("tau", seq_len(ncol(x$tau)))
    out <- cbind(out, x$tau)
    if (!is.null(colnames(out))) {
      # Update colnames to include tau names
      old_names <- colnames(out)[1:(ncol(out) - ncol(x$tau))]
      colnames(out) <- c(old_names, colnames_tau)
    }
  }

  # Set column names for beta if not present
  if (is.null(colnames(out))) {
    p_cols <- ncol(out)
    start_idx <- if (!is.null(x$alpha)) 2 else 1
    if (p_cols >= start_idx) {
      beta_idx <- seq_len(p_cols - start_idx + 1)
      if (!is.null(vars) && is.numeric(vars)) {
        beta_names <- paste0("Beta", vars)
      } else {
        beta_names <- paste0("Beta", beta_idx)
      }
      colnames(out)[start_idx:p_cols] <- beta_names
    }
  }

  # SUM-2: Pass actual thinning interval from the fit object
  thin_val <- if (!is.null(x$thin)) as.integer(x$thin) else 1L
  coda::mcmc(out, start = 1, thin = thin_val)
}

#' Gelman-Rubin Diagnostics for BVS
#'
#' Computes the Gelman-Rubin convergence diagnostic for a list of BVS fit objects
#' (representing multiple chains).
#'
#' @param fits A list of \code{"bvs"} objects derived from running the same model
#'   with different starting values or seeds.
#' @param vars Optional vector of variable indices to include in the diagnostic.
#'   If NULL, computes for all variables (can be slow for high-dimensional data).
#' @param ... Additional arguments passed to \code{coda::gelman.diag}.
#'
#' @return An object of class \code{"gelman.diag"} from the \code{coda} package.
#'   Contains point estimates and upper confidence limits for the potential
#'   scale reduction factor (PSRF).
#'
#' @examples
#' \dontrun{
#' fit1 <- bvs_pg(X, y, ...)
#' fit2 <- bvs_pg(X, y, ...)
#' diag <- bvs_gelman_diag(list(fit1, fit2))
#' print(diag)
#' }
#'
#' @importFrom coda gelman.diag mcmc.list
#' @importFrom stats var
#' @export
bvs_gelman_diag <- function(fits, vars = NULL, ...) {
  if (!is.list(fits) || length(fits) < 2) {
    stop("Please provide a list of at least 2 'bvs' fit objects.")
  }

  # Convert each fit to mcmc
  chain_list <- lapply(fits, function(f) {
    if (!inherits(f, "bvs")) stop("All elements in 'fits' must be 'bvs' objects.")
    as.mcmc.bvs(f, vars = vars)
  })

  # Check chain lengths equal
  ns <- vapply(chain_list, nrow, integer(1))
  if (var(ns) > 0) {
    min_n <- min(ns)
    warning("Chains have different lengths. Truncating to minimum length: ", min_n)
    # SUM-3: Wrap truncated matrices in coda::mcmc() to preserve class
    chain_list <- lapply(chain_list, function(ch) {
      coda::mcmc(ch[1:min_n, , drop = FALSE])
    })
  }

  mcmc_list <- coda::mcmc.list(chain_list)

  # Compute diagnostics
  # multivariate=FALSE is usually safer for high-dim/sparse BVS
  coda::gelman.diag(mcmc_list, multivariate = FALSE, ...)
}

# ---------------------------------------------------------------------------
# Modern convergence diagnostics (Vehtari et al. 2021, Bayesian Analysis)
# ---------------------------------------------------------------------------

#' Rank-Normalized Split-R-hat
#'
#' Computes the rank-normalized split-\eqn{\hat{R}} convergence diagnostic
#' following Vehtari, Gelman, Simpson, Carpenter, and Burkner (2021).
#' For a single chain, the chain is split in half to create two pseudo-chains
#' (detecting within-chain non-stationarity). For multiple chains (matrix
#' input, one column per chain), each chain is split in half first.
#' All draws are then rank-normalized (replacing draws by their standard
#' normal quantiles) before the classical between- and within-chain variance
#' estimator is applied. The returned value is
#' \eqn{\max(\hat{R}_{\mathrm{bulk}}, \hat{R}_{\mathrm{tail}})} where
#' \eqn{\hat{R}_{\mathrm{tail}}} is computed on the folded (absolute deviation
#' from the median) draws.
#'
#' When the \pkg{posterior} package is installed, computation is delegated to
#' \code{posterior::rhat()} for speed and consistency.
#'
#' @param x Numeric vector (single chain, split in half automatically) or
#'   matrix (rows = draws, columns = chains).
#' @return Scalar \eqn{\hat{R}} value. Values below 1.01 are considered
#'   acceptable; values above 1.05 indicate convergence failure.
#' @seealso \code{\link{bvs_ess_bulk}}, \code{\link{bvs_ess_tail}}
#' @references
#'   Vehtari A, Gelman A, Simpson D, Carpenter B, Burkner P-C (2021).
#'   Rank-normalization, folding, and localization: An improved R-hat for
#'   assessing convergence of MCMC (with discussion). \emph{Bayesian Analysis},
#'   16(2), 667--718. DOI: 10.1214/20-BA1221.
#' @export
bvs_rhat <- function(x) {
  x_mat <- .ensure_matrix(x)
  fallback <- function() {
    bulk <- .rhat_basic(.z_scale(.split_chains(x_mat)))
    tail <- .rhat_basic(.z_scale(.split_chains(.fold_draws(x_mat))))
    max(bulk, tail)
  }
  if (requireNamespace("posterior", quietly = TRUE)) {
    out <- tryCatch(
      posterior::rhat(posterior::as_draws_matrix(x_mat)),
      error = function(e) NA_real_
    )
    if (length(out) == 1L && is.finite(out)) {
      return(as.numeric(out))
    }
  }
  fallback()
}

#' Bulk Effective Sample Size
#'
#' Computes bulk ESS on rank-normalized, split chains following
#' Vehtari et al. (2021). Bulk ESS measures sampling efficiency for
#' the body of the posterior distribution using Geyer's initial positive
#' sequence estimator applied to the rank-normalized draws. It is more
#' reliable than the classical autocorrelation-based ESS when chains are
#' heavy-tailed or exhibit non-Gaussian behaviour.
#'
#' When the \pkg{posterior} package is installed, computation is delegated to
#' \code{posterior::ess_bulk()} for speed and consistency.
#'
#' @param x Numeric vector (single chain, split in half automatically) or
#'   matrix (rows = draws, columns = chains).
#' @return Scalar bulk ESS value. Rule of thumb: bulk ESS > 100 per chain is
#'   acceptable.
#' @seealso \code{\link{bvs_rhat}}, \code{\link{bvs_ess_tail}}
#' @references
#'   Vehtari A, Gelman A, Simpson D, Carpenter B, Burkner P-C (2021).
#'   Rank-normalization, folding, and localization: An improved R-hat for
#'   assessing convergence of MCMC (with discussion). \emph{Bayesian Analysis},
#'   16(2), 667--718. DOI: 10.1214/20-BA1221.
#' @export
bvs_ess_bulk <- function(x) {
  x_mat <- .ensure_matrix(x)
  fallback <- function() {
    z <- .z_scale(.split_chains(x_mat))
    .ess_multi_chain(z)
  }
  if (requireNamespace("posterior", quietly = TRUE)) {
    out <- tryCatch(
      posterior::ess_bulk(posterior::as_draws_matrix(x_mat)),
      error = function(e) NA_real_
    )
    if (length(out) == 1L && is.finite(out)) {
      return(as.numeric(out))
    }
  }
  fallback()
}

#' Tail Effective Sample Size
#'
#' Computes tail ESS as \eqn{\min(\mathrm{ESS}_{0.05},\, \mathrm{ESS}_{0.95})},
#' following Vehtari et al. (2021). Tail ESS measures sampling efficiency in
#' the tails of the posterior and is particularly sensitive to poor mixing of
#' extreme quantiles (heavy-tailed posteriors, multimodal chains). It is
#' computed by applying Geyer's initial positive sequence estimator to indicator
#' sequences \eqn{I(x \le q)} for \eqn{q} at the 5th and 95th empirical
#' percentiles.
#'
#' When the \pkg{posterior} package is installed, computation is delegated to
#' \code{posterior::ess_tail()} for speed and consistency.
#'
#' @param x Numeric vector (single chain, split in half automatically) or
#'   matrix (rows = draws, columns = chains).
#' @return Scalar tail ESS value. Rule of thumb: tail ESS > 100 per chain is
#'   acceptable.
#' @seealso \code{\link{bvs_rhat}}, \code{\link{bvs_ess_bulk}}
#' @references
#'   Vehtari A, Gelman A, Simpson D, Carpenter B, Burkner P-C (2021).
#'   Rank-normalization, folding, and localization: An improved R-hat for
#'   assessing convergence of MCMC (with discussion). \emph{Bayesian Analysis},
#'   16(2), 667--718. DOI: 10.1214/20-BA1221.
#' @export
bvs_ess_tail <- function(x) {
  x_mat <- .ensure_matrix(x)
  fallback <- function() {
    q05 <- .ess_quantile(x_mat, 0.05)
    q95 <- .ess_quantile(x_mat, 0.95)
    min(q05, q95)
  }
  if (requireNamespace("posterior", quietly = TRUE)) {
    out <- tryCatch(
      posterior::ess_tail(posterior::as_draws_matrix(x_mat)),
      error = function(e) NA_real_
    )
    if (length(out) == 1L && is.finite(out)) {
      return(as.numeric(out))
    }
  }
  fallback()
}

# --- Internal helpers for Vehtari et al. (2021) diagnostics ---

.ensure_matrix <- function(x) {
  if (is.matrix(x)) return(x)
  matrix(x, ncol = 1)
}

.split_chains <- function(x_mat) {
  # Split each chain in half, doubling the number of chains
  n <- nrow(x_mat)
  half <- n %/% 2
  if (half < 2) return(x_mat)
  m <- ncol(x_mat)
  out <- matrix(NA_real_, nrow = half, ncol = 2 * m)
  for (j in seq_len(m)) {
    out[, 2 * j - 1] <- x_mat[1:half, j]
    out[, 2 * j]     <- x_mat[(n - half + 1):n, j]
  }
  out
}

.z_scale <- function(x_mat) {
  # Rank-normalize across ALL chains pooled, then reshape back
  # (per Vehtari et al. 2021: ranks computed over all draws jointly)
  n <- nrow(x_mat)
  m <- ncol(x_mat)
  all_vals <- as.vector(x_mat)
  r <- rank(all_vals, ties.method = "average")
  N <- length(r)
  z <- stats::qnorm((r - 3 / 8) / (N + 1 / 4))
  matrix(z, nrow = n, ncol = m)
}

.fold_draws <- function(x_mat) {
  # Fold around the median: |x - median(x)|
  med <- stats::median(x_mat)
  abs(x_mat - med)
}

.rhat_basic <- function(x_mat) {
  # Classical split-R-hat from within- and between-chain variances
  m <- ncol(x_mat)
  n <- nrow(x_mat)
  if (m < 2 || n < 2) return(1.0)

  chain_means <- colMeans(x_mat)
  chain_vars <- apply(x_mat, 2, stats::var)
  grand_mean <- mean(chain_means)

  W <- mean(chain_vars)            # within-chain variance
  B <- n * stats::var(chain_means) # between-chain variance

  if (W < .Machine$double.eps) return(1.0)

  var_hat <- ((n - 1) / n) * W + (1 / n) * B
  sqrt(var_hat / W)
}

.ess_multi_chain <- function(x_mat) {
  # Aggregate ESS across chains using Geyer's initial positive sequence
  m <- ncol(x_mat)
  n <- nrow(x_mat)
  if (n < 4) return(NA_real_)

  chain_means <- colMeans(x_mat)
  chain_vars <- apply(x_mat, 2, stats::var)
  W <- mean(chain_vars)
  if (W < .Machine$double.eps) return(as.numeric(n * m))

  # R2-FIX: cap max_lag at min(n-1, max(20, floor(n/2)), 1000). Previous
  # max_lag = n - 1 made stats::acf O(n^2), which is wasted work for long
  # chains -- Geyer's IPS truncates the positive-pair sum well before the
  # tail, so high lags contribute nothing but cost. Cap matches the
  # posterior package's practical default.
  max_lag <- min(n - 1L, max(20L, n %/% 2L), 1000L)
  max_lag <- as.integer(max_lag)
  acf_sum <- numeric(max_lag)
  for (j in seq_len(m)) {
    chain_centered <- x_mat[, j] - chain_means[j]
    ac <- stats::acf(chain_centered, lag.max = max_lag, plot = FALSE)$acf[-1]
    acf_sum[seq_along(ac)] <- acf_sum[seq_along(ac)] + ac * chain_vars[j]
  }
  rho_hat <- acf_sum / (m * W)

  # Geyer's initial positive sequence estimator
  tau <- -1.0
  k <- 1
  while (k <= length(rho_hat) - 1) {
    pair_sum <- rho_hat[k] + rho_hat[k + 1]
    if (pair_sum < 0) break
    tau <- tau + 2 * pair_sum
    k <- k + 2
  }
  tau <- max(tau, 1 / n)
  as.numeric(n * m / tau)
}

.ess_quantile <- function(x_mat, prob) {
  # ESS for a specific quantile indicator
  q_val <- stats::quantile(x_mat, prob)
  indicator <- ifelse(x_mat <= q_val, 1.0, 0.0)
  ind_mat <- matrix(indicator, nrow = nrow(x_mat), ncol = ncol(x_mat))
  .ess_multi_chain(.split_chains(ind_mat))
}

.extract_pip_from_bvs <- function(object) {
  if (!is.null(object$gamma_pip)) {
    return(as.numeric(object$gamma_pip))
  }
  gamma <- object$gamma
  if (is.matrix(gamma)) {
    return(colMeans(gamma))
  }
  if (is.list(gamma)) {
    p <- object$p
    if (is.null(p) || p < 1L) stop("Cannot infer p for sparse gamma samples.")
    n_save <- length(gamma)
    if (n_save < 1L) {
      return(rep(0, p))
    }
    pip_counts <- numeric(p)
    for (s in seq_len(n_save)) {
      idx1 <- .normalize_sparse_idx(gamma[[s]], p)
      if (length(idx1) > 0L) {
        pip_counts[idx1] <- pip_counts[idx1] + 1
      }
    }
    return(pip_counts / n_save)
  }
  stop("Posterior inclusion probabilities are unavailable.")
}

.get_beta_matrix <- function(object, indices = NULL) {
  beta <- object$beta
  p <- object$p
  n_save <- if (is.matrix(beta)) nrow(beta) else length(beta)
  if (is.null(p)) p <- if (is.matrix(beta)) ncol(beta) else 0L
  target_indices <- if (is.null(indices)) seq_len(p) else indices
  n_target <- length(target_indices)
  if (n_target == 0) {
    return(NULL)
  }

  if (is.matrix(beta)) {
    return(beta[, target_indices, drop = FALSE])
  }
  if (is.list(beta)) {
    out_mat <- matrix(0.0, nrow = n_save, ncol = n_target)
    for (s in seq_len(n_save)) {
      bm <- beta[[s]]
      if (is.null(bm) || length(bm) == 0L) next
      bm <- as.matrix(bm)
      if (nrow(bm) > 0 && ncol(bm) >= 2) {
        idx_raw <- as.integer(bm[, 1])
        val <- as.numeric(bm[, 2])

        # Support both historical 0-based and current 1-based sparse indices.
        idx_a <- idx_raw
        ok_a <- idx_a >= 1L & idx_a <= p
        idx_b <- idx_raw + 1L
        ok_b <- idx_b >= 1L & idx_b <= p
        use_b <- sum(ok_b) > sum(ok_a)
        idx_1 <- if (use_b) idx_b else idx_a
        ok <- if (use_b) ok_b else ok_a

        match_pos <- match(idx_1[ok], target_indices)
        valid <- !is.na(match_pos)
        if (any(valid)) {
          out_mat[s, match_pos[valid]] <- val[ok][valid]
        }
      }
    }
    return(out_mat)
  }
  return(NULL)
}
