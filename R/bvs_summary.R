#' Summarize BVS Results
#'
#' Computes posterior inclusion probabilities (PIP), posterior means,
#' credible intervals (quantile-based), and Highest Posterior Density (HPD)
#' intervals (via \code{coda}) for probability of inclusion and coefficients.
#' Also provides MCMC convergence diagnostics (Effective Sample Size).
#'
#' @param object  An object of class \code{"bvs"}.
#' @param pip_threshold  Threshold on PIP for declaring a variable selected
#'   (default 0.5).
#' @param cred_level  Probability level for quantile-based credible intervals (default 0.95).
#' @param hpd_level  Probability level for HPD intervals (default 0.95).
#' @param ...  Additional arguments (ignored).
#'
#' @return A list with:
#'   \describe{
#'     \item{pip}{Posterior inclusion probabilities (length p).}
#'     \item{selected}{Indices of selected variables (PIP > threshold).}
#'     \item{summary_beta}{Data frame with columns: mean, sd,
#'           quantile_lower, quantile_upper, hpd_lower, hpd_upper, ess. (Rows = predictors)}
#'     \item{summary_alpha}{Summary vector for intercept (mean, quantiles, HPD, ESS).}
#'     \item{nselected}{Number of selected variables.}
#'     \item{cred_level}{Sampled credible level.}
#'     \item{hpd_level}{Sampled HPD level.}
#'   }
#'
#' @examples
#' \dontrun{
#' fit <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R, niter = 5000)
#' s <- summary(fit, cred_level = 0.90, hpd_level = 0.90)
#' head(s$summary_beta)
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

    # ESS
    ess_res <- tryCatch(
      {
        coda::effectiveSize(mcmc_obj)
      },
      error = function(e) rep(NA_real_, p)
    )
    beta_ess <- as.numeric(ess_res)
  }

  summary_beta <- data.frame(
    Mean = beta_mean,
    SD = beta_sd,
    PIP = pip,
    Q_low = beta_q_low,
    Q_up = beta_q_up,
    HPD_low = beta_hpd_low,
    HPD_up = beta_hpd_up,
    ESS = beta_ess
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
    a_ess <- tryCatch(coda::effectiveSize(alpha_mcmc), error = function(e) NA)

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
    tau_q <- t(apply(object$tau, 2, quantile, probs = c(0.025, 0.5, 0.975)))
    tau_summary <- data.frame(
      Variable = paste0("tau", seq_len(ntau)),
      Mean = tau_mean,
      SD = tau_sd,
      Q2.5 = tau_q[, 1],
      Median = tau_q[, 2],
      Q97.5 = tau_q[, 3]
    )
    if (requireNamespace("coda", quietly = TRUE)) {
      tau_ess <- apply(object$tau, 2, function(x) coda::effectiveSize(x))
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
    p = object$p,
    n = object$n,
    niter = object$niter,
    cred_level = cred_level,
    hpd_level = hpd_level
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

  coda::mcmc(out, start = 1, thin = 1)
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
    chain_list <- lapply(chain_list, function(ch) ch[1:min_n, , drop = FALSE])
  }

  mcmc_list <- coda::mcmc.list(chain_list)

  # Compute diagnostics
  # multivariate=FALSE is usually safer for high-dim/sparse BVS
  coda::gelman.diag(mcmc_list, multivariate = FALSE, ...)
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
