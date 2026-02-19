#' Summarize BVS Results
#'
#' Computes posterior inclusion probabilities (PIP), posterior means,
#' and 95\% credible intervals for each predictor.
#'
#' @param object  An object of class \code{"bvs"}.
#' @param pip_threshold  Threshold on PIP for declaring a variable selected
#'   (default 0.5).
#' @param ...  Additional arguments (ignored).
#'
#' @return A list with:
#'   \describe{
#'     \item{pip}{Posterior inclusion probabilities (length p).}
#'     \item{selected}{Indices of selected variables (PIP > threshold).}
#'     \item{beta_mean}{Posterior mean of beta (length p).}
#'     \item{beta_lower}{2.5\% quantile of beta (length p).}
#'     \item{beta_upper}{97.5\% quantile of beta (length p).}
#'     \item{alpha_mean}{Posterior mean of alpha.}
#'     \item{nselected}{Number of selected variables.}
#'   }
#'
#' @examples
#' \dontrun{
#' fit <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R, niter = 5000)
#' s <- summary(fit)
#' s$selected
#' }
#'
#' @method summary bvs
#' @export
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
    if (is.null(p) || p < 1L) {
      stop("Cannot infer p for sparse gamma samples.")
    }
    n_save <- length(gamma)
    if (n_save < 1L) {
      return(rep(0, p))
    }
    pip_counts <- numeric(p)
    for (s in seq_len(n_save)) {
      idx <- gamma[[s]]
      if (length(idx) > 0L) {
        idx1 <- as.integer(idx) + 1L
        idx1 <- idx1[idx1 >= 1L & idx1 <= p]
        if (length(idx1) > 0L) {
          pip_counts[idx1] <- pip_counts[idx1] + 1
        }
      }
    }
    return(pip_counts / n_save)
  }

  stop("Posterior inclusion probabilities are unavailable.")
}

.extract_beta_summary_from_bvs <- function(object) {
  beta <- object$beta
  p <- object$p

  if (is.matrix(beta)) {
    return(list(
      beta_mean = colMeans(beta),
      beta_lower = apply(beta, 2, quantile, probs = 0.025),
      beta_upper = apply(beta, 2, quantile, probs = 0.975)
    ))
  }

  if (is.list(beta)) {
    n_save <- length(beta)
    if (is.null(p) || p < 1L || n_save < 1L) {
      return(list(beta_mean = NULL, beta_lower = NULL, beta_upper = NULL))
    }

    # Guard against accidental large dense materialization.
    if ((as.double(p) * as.double(n_save)) > 2e7) {
      return(list(beta_mean = NULL, beta_lower = NULL, beta_upper = NULL))
    }

    beta_mat <- matrix(0, nrow = n_save, ncol = p)
    for (s in seq_len(n_save)) {
      bm <- beta[[s]]
      if (is.null(bm) || length(bm) == 0L) {
        next
      }
      bm <- as.matrix(bm)
      if (nrow(bm) > 0L && ncol(bm) >= 2L) {
        idx <- as.integer(bm[, 1]) + 1L
        val <- as.numeric(bm[, 2])
        keep <- idx >= 1L & idx <= p & is.finite(val)
        if (any(keep)) {
          beta_mat[s, idx[keep]] <- val[keep]
        }
      }
    }
    return(list(
      beta_mean = colMeans(beta_mat),
      beta_lower = apply(beta_mat, 2, quantile, probs = 0.025),
      beta_upper = apply(beta_mat, 2, quantile, probs = 0.975)
    ))
  }

  list(beta_mean = NULL, beta_lower = NULL, beta_upper = NULL)
}

summary.bvs <- function(object, pip_threshold = 0.5, ...) {

  pip <- .extract_pip_from_bvs(object)
  selected <- which(pip > pip_threshold)

  beta_sum <- .extract_beta_summary_from_bvs(object)
  alpha_mean <- if (!is.null(object$alpha)) mean(object$alpha) else NA_real_

  out <- list(
    pip        = pip,
    selected   = selected,
    beta_mean  = beta_sum$beta_mean,
    beta_lower = beta_sum$beta_lower,
    beta_upper = beta_sum$beta_upper,
    alpha_mean = alpha_mean,
    nselected  = length(selected),
    sampler    = object$sampler,
    adj_type   = object$adj_type,
    p          = object$p,
    n          = object$n,
    niter      = object$niter
  )
  class(out) <- "summary.bvs"
  out
}

#' @method print summary.bvs
#' @export
print.summary.bvs <- function(x, ...) {
  cat("Bayesian Variable Selection (", toupper(x$sampler),
      ", adj_type='", x$adj_type, "')\n", sep = "")
  cat("  n =", x$n, ", p =", x$p, ", niter =", x$niter, "\n")
  cat("  Variables selected (PIP > 0.5):", x$nselected, "\n")
  if (x$nselected > 0 && x$nselected <= 30) {
    cat("  Selected indices:", paste(x$selected, collapse = ", "), "\n")
  }
  cat("  Intercept (posterior mean):", round(x$alpha_mean, 4), "\n")
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
      ", adj_type='", x$adj_type, "')\n", sep = "")
  cat("  n =", x$n, ", p =", x$p, "\n")
  cat("  Iterations:", x$niter, "\n")
  cat("  Use summary() for posterior summaries.\n")
  invisible(x)
}


#' Plot BVS Posterior Inclusion Probabilities
#'
#' Creates a barplot of posterior inclusion probabilities with a threshold line.
#'
#' @param x  An object of class \code{"bvs"}.
#' @param pip_threshold  Threshold line (default 0.5).
#' @param top_n  Show only top N variables by PIP (default: all).
#' @param ...  Additional graphical arguments.
#'
#' @examples
#' \dontrun{
#' fit <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R, niter = 5000)
#' plot(fit)
#' }
#'
#' @method plot bvs
#' @export
plot.bvs <- function(x, pip_threshold = 0.5, top_n = NULL, ...) {

  pip <- .extract_pip_from_bvs(x)

  if (!is.null(top_n)) {
    ord <- order(pip, decreasing = TRUE)[1:min(top_n, length(pip))]
    pip <- pip[ord]
    names(pip) <- paste0("X", ord)
  }

  barplot(pip, las = 2, ylim = c(0, 1),
          main = paste0("Posterior Inclusion Probabilities (",
                        toupper(x$sampler), ")"),
          ylab = "PIP", col = ifelse(pip > pip_threshold, "steelblue", "grey70"),
          border = NA, ...)
  abline(h = pip_threshold, lty = 2, col = "red", lwd = 1.5)
  legend("topright", legend = paste("threshold =", pip_threshold),
         lty = 2, col = "red", bty = "n")
}
