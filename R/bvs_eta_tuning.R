#' Phase-Transition Profile for Eta Tuning
#'
#' Runs \code{\link{phase_transition}} on a binary adjacency matrix and extracts
#' a practical phase-transition anchor (\code{eta_star}) from the mean model-size
#' curve. The anchor can be used to build eta upper-bound grids for
#' \code{eta1_sd} and \code{eta2_sd}.
#'
#' @param adj A square binary adjacency matrix.
#' @param mu Ising external field used in the phase-transition sweep.
#' @param min_eta Minimum eta value for the sweep.
#' @param max_eta Maximum eta value for the sweep.
#' @param step_size Eta step size for the sweep.
#' @param num_rep Number of Propp-Wilson replications per eta value.
#' @param Tmax Maximum Propp-Wilson doubling horizon.
#'
#' @return A list with phase-transition diagnostics:
#' \describe{
#'   \item{eta_seq}{Eta grid used in the sweep.}
#'   \item{phase_matrix}{Integer matrix of model sizes (eta x replication).}
#'   \item{mean_model_size}{Mean model size at each eta value.}
#'   \item{sd_model_size}{Model-size SD at each eta value.}
#'   \item{eta_star}{Estimated transition anchor (largest mean-size jump).}
#'   \item{eta_star_index}{1-based index of \code{eta_star}.}
#' }
#'
#' @export
bvs_phase_eta_profile <- function(adj,
                                  # nolint start: object_length_linter.
                                  mu = -log(1 / 0.1 - 1),
                                  min_eta = 0,
                                  max_eta = 1.5,
                                  step_size = 0.05,
                                  num_rep = 8L,
                                  Tmax = 64L) {
  # nolint end: object_length_linter.
  adj <- as.matrix(adj)
  if (!is.numeric(adj) || nrow(adj) != ncol(adj)) {
    stop("'adj' must be a square numeric matrix.")
  }
  p <- nrow(adj)
  adj <- .prepare_adj(adj, p, "adj")

  phase_mat <- phase_transition(
    adj = adj,
    mu = mu,
    min_eta = min_eta,
    max_eta = max_eta,
    step_size = step_size,
    num_rep = as.integer(num_rep),
    Tmax = as.integer(Tmax)
  )

  eta_seq <- seq(min_eta, max_eta, by = step_size)
  n_eta <- min(length(eta_seq), nrow(phase_mat))
  eta_seq <- eta_seq[seq_len(n_eta)]
  phase_mat <- phase_mat[seq_len(n_eta), , drop = FALSE]

  mean_size <- rowMeans(phase_mat)
  sd_size <- apply(phase_mat, 1L, stats::sd)
  if (length(mean_size) <= 1L) {
    eta_star_idx <- 1L
  } else {
    jump <- abs(diff(mean_size))
    eta_star_idx <- which.max(jump) + 1L
  }
  eta_star <- eta_seq[eta_star_idx]

  list(
    eta_seq = eta_seq,
    phase_matrix = phase_mat,
    mean_model_size = mean_size,
    sd_model_size = sd_size,
    eta_star = eta_star,
    eta_star_index = eta_star_idx
  )
}

#' Build Single-Network Eta Tuning Grid
#'
#' Creates a candidate grid for single-adjacency models
#' (\code{fixed}, \code{glasso}, \code{ggm}) over \code{eta1_sd} and the
#' corresponding Moller hyperparameters.
#'
#' If \code{eta1_sd} is omitted and \code{adj} is provided, \code{eta1_sd} values
#' are generated as fractions of the phase-transition anchor from
#' \code{\link{bvs_phase_eta_profile}}.
#'
#' @param eta1_sd Optional numeric vector of candidate \code{eta1_sd}.
#' @param adj Optional adjacency matrix for phase-based grid construction.
#' @param eta1_frac Fractions of phase-transition anchor for constructing
#'   \code{eta1_sd} when \code{eta1_sd} is NULL.
#' @param mu Ising external field used for phase profiling.
#' @param phase_min_eta Minimum eta used for phase profiling.
#' @param phase_max_eta Maximum eta used for phase profiling.
#' @param phase_step_size Eta step size used for phase profiling.
#' @param phase_num_rep Replications used for phase profiling.
#' @param phase_Tmax Propp-Wilson horizon used for phase profiling.
#' @param eta_floor Minimum allowed eta upper bound.
#' @param eta_cap Maximum allowed eta upper bound.
#' @param mu_tilde Candidate vector for Moller auxiliary \code{mu_tilde}.
#' @param eta1_tilde Candidate vector for Moller auxiliary \code{eta1_tilde}.
#' @param e_eta Candidate vector for Moller Beta prior shape \code{a}.
#' @param f_eta Candidate vector for Moller Beta prior shape \code{b}.
#'
#' @return A data.frame grid with columns
#'   \code{eta1_sd, mu_tilde, eta1_tilde, e_eta, f_eta}.
#'   If phase profiling is used, a \code{"phase_profile"} attribute is attached.
#'
#' @export
bvs_eta_grid_single <- function(
  eta1_sd = NULL,
  adj = NULL,
  eta1_frac = seq(0.2, 1.0, by = 0.1),
  mu = -log(1 / 0.1 - 1),
  phase_min_eta = 0,
  phase_max_eta = 1.5,
  phase_step_size = 0.05,
  phase_num_rep = 8L,
  phase_Tmax = 64L,
  eta_floor = 1e-4,
  eta_cap = 2.0,
  mu_tilde = -4,
  eta1_tilde = 0.075,
  e_eta = 1,
  f_eta = 1
) {
  phase_profile <- NULL
  if (is.null(eta1_sd)) {
    if (is.null(adj)) {
      stop("Provide either 'eta1_sd' directly or an 'adj' matrix for phase-based eta grid.")
    }
    phase_profile <- bvs_phase_eta_profile(
      adj = adj, mu = mu,
      min_eta = phase_min_eta, max_eta = phase_max_eta,
      step_size = phase_step_size, num_rep = phase_num_rep,
      Tmax = phase_Tmax
    )
    eta1_sd <- phase_profile$eta_star * as.numeric(eta1_frac)
  }

  eta1_sd <- sort(unique(as.numeric(eta1_sd)))
  eta1_sd <- pmin(eta_cap, pmax(eta_floor, eta1_sd))

  grid <- expand.grid(
    eta1_sd = eta1_sd,
    mu_tilde = as.numeric(mu_tilde),
    eta1_tilde = as.numeric(eta1_tilde),
    e_eta = as.numeric(e_eta),
    f_eta = as.numeric(f_eta),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )

  attr(grid, "phase_profile") <- phase_profile
  grid
}

#' Build Dual-Network Eta Tuning Grid
#'
#' Creates a candidate grid for dual-adjacency models
#' (\code{dual_fixed}, \code{glasso_fixed}, \code{ggm_fixed}) over
#' \code{eta1_sd}, \code{eta2_sd}, and corresponding Moller hyperparameters.
#'
#' If \code{eta1_sd}/\code{eta2_sd} are omitted and \code{adj1}/\code{adj2}
#' are provided, candidate values are generated as fractions of each network's
#' phase-transition anchor.
#'
#' @param eta1_sd Optional numeric vector of candidate \code{eta1_sd}.
#' @param eta2_sd Optional numeric vector of candidate \code{eta2_sd}.
#' @param adj1 Optional first adjacency matrix for phase-based \code{eta1_sd}.
#' @param adj2 Optional second adjacency matrix for phase-based \code{eta2_sd}.
#' @param eta1_frac Fractions of network-1 phase anchor.
#' @param eta2_frac Fractions of network-2 phase anchor.
#' @param mu Ising external field used for phase profiling.
#' @param phase_min_eta Minimum eta used for phase profiling.
#' @param phase_max_eta Maximum eta used for phase profiling.
#' @param phase_step_size Eta step size used for phase profiling.
#' @param phase_num_rep Replications used for phase profiling.
#' @param phase_Tmax Propp-Wilson horizon used for phase profiling.
#' @param eta_floor Minimum allowed eta upper bound.
#' @param eta_cap Maximum allowed eta upper bound.
#' @param mu_tilde Candidate vector for Moller auxiliary \code{mu_tilde}.
#' @param eta1_tilde Candidate vector for Moller auxiliary \code{eta1_tilde}.
#' @param eta2_tilde Candidate vector for Moller auxiliary \code{eta2_tilde}.
#' @param e_eta Candidate vector for Moller Beta prior shape \code{a}.
#' @param f_eta Candidate vector for Moller Beta prior shape \code{b}.
#'
#' @return A data.frame grid with columns
#'   \code{eta1_sd, eta2_sd, mu_tilde, eta1_tilde, eta2_tilde, e_eta, f_eta}.
#'   If phase profiling is used, \code{"phase_profile1"} and
#'   \code{"phase_profile2"} attributes are attached.
#'
#' @export
bvs_eta_grid_dual <- function(
  eta1_sd = NULL,
  eta2_sd = NULL,
  adj1 = NULL,
  adj2 = NULL,
  eta1_frac = seq(0.2, 1.0, by = 0.1),
  eta2_frac = seq(0.2, 1.0, by = 0.1),
  mu = -log(1 / 0.1 - 1),
  phase_min_eta = 0,
  phase_max_eta = 1.5,
  phase_step_size = 0.05,
  phase_num_rep = 8L,
  phase_Tmax = 64L,
  eta_floor = 1e-4,
  eta_cap = 2.0,
  mu_tilde = -4,
  eta1_tilde = 0.075,
  eta2_tilde = 0.065,
  e_eta = 1,
  f_eta = 1
) {
  phase_profile1 <- NULL
  phase_profile2 <- NULL

  if (is.null(eta1_sd)) {
    if (is.null(adj1)) {
      stop("Provide either 'eta1_sd' directly or 'adj1' for phase-based eta1 grid.")
    }
    phase_profile1 <- bvs_phase_eta_profile(
      adj = adj1, mu = mu,
      min_eta = phase_min_eta, max_eta = phase_max_eta,
      step_size = phase_step_size, num_rep = phase_num_rep,
      Tmax = phase_Tmax
    )
    eta1_sd <- phase_profile1$eta_star * as.numeric(eta1_frac)
  }

  if (is.null(eta2_sd)) {
    if (is.null(adj2)) {
      stop("Provide either 'eta2_sd' directly or 'adj2' for phase-based eta2 grid.")
    }
    phase_profile2 <- bvs_phase_eta_profile(
      adj = adj2, mu = mu,
      min_eta = phase_min_eta, max_eta = phase_max_eta,
      step_size = phase_step_size, num_rep = phase_num_rep,
      Tmax = phase_Tmax
    )
    eta2_sd <- phase_profile2$eta_star * as.numeric(eta2_frac)
  }

  eta1_sd <- sort(unique(as.numeric(eta1_sd)))
  eta2_sd <- sort(unique(as.numeric(eta2_sd)))
  eta1_sd <- pmin(eta_cap, pmax(eta_floor, eta1_sd))
  eta2_sd <- pmin(eta_cap, pmax(eta_floor, eta2_sd))

  grid <- expand.grid(
    eta1_sd = eta1_sd,
    eta2_sd = eta2_sd,
    mu_tilde = as.numeric(mu_tilde),
    eta1_tilde = as.numeric(eta1_tilde),
    eta2_tilde = as.numeric(eta2_tilde),
    e_eta = as.numeric(e_eta),
    f_eta = as.numeric(f_eta),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )

  attr(grid, "phase_profile1") <- phase_profile1
  attr(grid, "phase_profile2") <- phase_profile2
  grid
}

#' Build Reasonable Single-Network Eta/Moller Grid
#'
#' Convenience constructor for single-adjacency tuning grids that couples
#' \code{eta1_tilde} to \code{eta1_sd} through scale factors
#' (\code{eta1_tilde = eta1_sd * eta1_tilde_frac}).
#'
#' This is useful for phase-transition-guided tuning where Moller auxiliary
#' interaction should remain proportional to the candidate Ising upper bound.
#'
#' @inheritParams bvs_eta_grid_single
#' @param eta1_tilde_frac Fractions used to derive \code{eta1_tilde} from
#'   \code{eta1_sd}.
#'
#' @return A data.frame with columns
#'   \code{eta1_sd, mu_tilde, eta1_tilde, e_eta, f_eta}.
#'   If phase profiling is used, a \code{"phase_profile"} attribute is attached.
#'
#' @export
bvs_eta_grid_single_reasonable <- function(
  eta1_sd = NULL,
  adj = NULL,
  eta1_frac = seq(0.2, 1.2, by = 0.1),
  mu = stats::qlogis(0.02),
  phase_min_eta = 0,
  phase_max_eta = 1.5,
  phase_step_size = 0.05,
  phase_num_rep = 8L,
  phase_Tmax = 64L,
  eta_floor = 1e-4,
  eta_cap = 2.0,
  mu_tilde = c(stats::qlogis(0.01), stats::qlogis(0.02), stats::qlogis(0.05)),
  eta1_tilde_frac = c(0.2, 0.35, 0.5),
  e_eta = c(1, 2),
  f_eta = c(1, 2)
) {
  base_grid <- bvs_eta_grid_single(
    eta1_sd = eta1_sd,
    adj = adj,
    eta1_frac = eta1_frac,
    mu = mu,
    phase_min_eta = phase_min_eta,
    phase_max_eta = phase_max_eta,
    phase_step_size = phase_step_size,
    phase_num_rep = phase_num_rep,
    phase_Tmax = phase_Tmax,
    eta_floor = eta_floor,
    eta_cap = eta_cap,
    mu_tilde = mu_tilde,
    eta1_tilde = 0,
    e_eta = e_eta,
    f_eta = f_eta
  )

  eta_sd_vec <- sort(unique(as.numeric(base_grid$eta1_sd)))
  eta_sd_vec <- pmin(eta_cap, pmax(eta_floor, eta_sd_vec))
  eta1_tilde_frac <- as.numeric(eta1_tilde_frac)
  eta1_tilde_frac <- eta1_tilde_frac[is.finite(eta1_tilde_frac) & eta1_tilde_frac > 0]
  if (!length(eta1_tilde_frac)) {
    stop("'eta1_tilde_frac' must contain at least one positive finite value.")
  }

  grid <- expand.grid(
    eta1_sd = eta_sd_vec,
    mu_tilde = as.numeric(mu_tilde),
    eta1_tilde_frac = eta1_tilde_frac,
    e_eta = as.numeric(e_eta),
    f_eta = as.numeric(f_eta),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  grid$eta1_tilde <- pmax(1e-6, pmin(grid$eta1_sd * grid$eta1_tilde_frac, grid$eta1_sd - 1e-6))
  grid$eta1_tilde_frac <- NULL

  attr(grid, "phase_profile") <- attr(base_grid, "phase_profile")
  grid
}

#' Build Reasonable Dual-Network Eta/Moller Grid
#'
#' Convenience constructor for dual-adjacency tuning grids that couples
#' \code{eta1_tilde} to \code{eta1_sd} and \code{eta2_tilde} to \code{eta2_sd}
#' through scale factors.
#'
#' @inheritParams bvs_eta_grid_dual
#' @param eta1_tilde_frac Fractions used to derive \code{eta1_tilde} from
#'   \code{eta1_sd}.
#' @param eta2_tilde_frac Fractions used to derive \code{eta2_tilde} from
#'   \code{eta2_sd}.
#'
#' @return A data.frame with columns
#'   \code{eta1_sd, eta2_sd, mu_tilde, eta1_tilde, eta2_tilde, e_eta, f_eta}.
#'   If phase profiling is used, \code{"phase_profile1"} and
#'   \code{"phase_profile2"} attributes are attached.
#'
#' @export
bvs_eta_grid_dual_reasonable <- function(
  eta1_sd = NULL,
  eta2_sd = NULL,
  adj1 = NULL,
  adj2 = NULL,
  eta1_frac = seq(0.2, 1.2, by = 0.1),
  eta2_frac = seq(0.2, 1.2, by = 0.1),
  mu = stats::qlogis(0.02),
  phase_min_eta = 0,
  phase_max_eta = 1.5,
  phase_step_size = 0.05,
  phase_num_rep = 8L,
  phase_Tmax = 64L,
  eta_floor = 1e-4,
  eta_cap = 2.0,
  mu_tilde = c(stats::qlogis(0.01), stats::qlogis(0.02), stats::qlogis(0.05)),
  eta1_tilde_frac = c(0.15, 0.3, 0.45),
  eta2_tilde_frac = c(0.15, 0.3, 0.45),
  e_eta = c(1, 2),
  f_eta = c(1, 2)
) {
  base_grid <- bvs_eta_grid_dual(
    eta1_sd = eta1_sd,
    eta2_sd = eta2_sd,
    adj1 = adj1,
    adj2 = adj2,
    eta1_frac = eta1_frac,
    eta2_frac = eta2_frac,
    mu = mu,
    phase_min_eta = phase_min_eta,
    phase_max_eta = phase_max_eta,
    phase_step_size = phase_step_size,
    phase_num_rep = phase_num_rep,
    phase_Tmax = phase_Tmax,
    eta_floor = eta_floor,
    eta_cap = eta_cap,
    mu_tilde = mu_tilde,
    eta1_tilde = 0,
    eta2_tilde = 0,
    e_eta = e_eta,
    f_eta = f_eta
  )

  eta1_sd_vec <- sort(unique(as.numeric(base_grid$eta1_sd)))
  eta2_sd_vec <- sort(unique(as.numeric(base_grid$eta2_sd)))
  eta1_sd_vec <- pmin(eta_cap, pmax(eta_floor, eta1_sd_vec))
  eta2_sd_vec <- pmin(eta_cap, pmax(eta_floor, eta2_sd_vec))

  eta1_tilde_frac <- as.numeric(eta1_tilde_frac)
  eta1_tilde_frac <- eta1_tilde_frac[is.finite(eta1_tilde_frac) & eta1_tilde_frac > 0]
  if (!length(eta1_tilde_frac)) {
    stop("'eta1_tilde_frac' must contain at least one positive finite value.")
  }
  eta2_tilde_frac <- as.numeric(eta2_tilde_frac)
  eta2_tilde_frac <- eta2_tilde_frac[is.finite(eta2_tilde_frac) & eta2_tilde_frac > 0]
  if (!length(eta2_tilde_frac)) {
    stop("'eta2_tilde_frac' must contain at least one positive finite value.")
  }

  grid <- expand.grid(
    eta1_sd = eta1_sd_vec,
    eta2_sd = eta2_sd_vec,
    mu_tilde = as.numeric(mu_tilde),
    eta1_tilde_frac = eta1_tilde_frac,
    eta2_tilde_frac = eta2_tilde_frac,
    e_eta = as.numeric(e_eta),
    f_eta = as.numeric(f_eta),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  grid$eta1_tilde <- pmax(1e-6, pmin(grid$eta1_sd * grid$eta1_tilde_frac, grid$eta1_sd - 1e-6))
  grid$eta2_tilde <- pmax(1e-6, pmin(grid$eta2_sd * grid$eta2_tilde_frac, grid$eta2_sd - 1e-6))
  grid$eta1_tilde_frac <- NULL
  grid$eta2_tilde_frac <- NULL

  attr(grid, "phase_profile1") <- attr(base_grid, "phase_profile1")
  attr(grid, "phase_profile2") <- attr(base_grid, "phase_profile2")
  grid
}

#' Compute Selection Metrics from PIPs
#'
#' Computes selected set size and, when ground-truth indices are supplied,
#' confusion-matrix metrics including sensitivity and FDR.
#'
#' @param pip Numeric vector of posterior inclusion probabilities.
#' @param truth_idx Optional integer vector of true associated variable indices
#'   (1-based). If NULL, supervised metrics are returned as NA.
#' @param pip_threshold PIP threshold for selection.
#' @param p Optional total number of predictors. Defaults to \code{length(pip)}.
#'
#' @return A one-row data.frame with metrics:
#' \code{selected_size, tp, fp, fn, tn, sensitivity, specificity, precision, fdr, f1}.
#'
#' @export
bvs_selection_metrics <- function(pip,
                                  truth_idx = NULL,
                                  pip_threshold = 0.5,
                                  p = length(pip)) {
  pip <- as.numeric(pip)
  if (length(pip) != p) {
    stop("'pip' length must match 'p'.")
  }

  selected <- which(pip >= pip_threshold)
  n_sel <- length(selected)

  if (is.null(truth_idx)) {
    return(data.frame(
      selected_size = n_sel,
      tp = NA_real_, fp = NA_real_, fn = NA_real_, tn = NA_real_,
      sensitivity = NA_real_, specificity = NA_real_,
      precision = NA_real_, fdr = NA_real_, f1 = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  truth_idx <- as.integer(truth_idx)
  truth_idx <- truth_idx[is.finite(truth_idx)]
  truth_idx <- unique(truth_idx[truth_idx >= 1L & truth_idx <= p])
  truth_flag <- logical(p)
  truth_flag[truth_idx] <- TRUE

  sel_flag <- logical(p)
  sel_flag[selected] <- TRUE

  tp <- sum(sel_flag & truth_flag)
  fp <- sum(sel_flag & !truth_flag)
  fn <- sum(!sel_flag & truth_flag)
  tn <- sum(!sel_flag & !truth_flag)

  sensitivity <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  specificity <- if ((tn + fp) > 0) tn / (tn + fp) else NA_real_
  precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  fdr <- if ((tp + fp) > 0) fp / (tp + fp) else 0
  f1 <- if ((2 * tp + fp + fn) > 0) 2 * tp / (2 * tp + fp + fn) else NA_real_

  data.frame(
    selected_size = n_sel,
    tp = tp, fp = fp, fn = fn, tn = tn,
    sensitivity = sensitivity, specificity = specificity,
    precision = precision, fdr = fdr, f1 = f1,
    stringsAsFactors = FALSE
  )
}

.tune_extract_pip <- function(fit) {
  if (!is.null(fit$gamma_pip)) {
    return(as.numeric(fit$gamma_pip))
  }
  gamma <- fit$gamma
  p <- as.integer(fit$p)
  if (is.matrix(gamma)) {
    return(colMeans(gamma != 0))
  }
  if (is.list(gamma)) {
    if (length(gamma) == 0L) {
      return(rep(0, p))
    }
    counts <- numeric(p)
    for (k in seq_along(gamma)) {
      idx <- .normalize_sparse_idx(gamma[[k]], p)
      if (length(idx)) counts[idx] <- counts[idx] + 1
    }
    return(counts / length(gamma))
  }
  stop("Cannot extract posterior inclusion probabilities from fit.")
}

.rbind_fill_df <- function(df_list) {
  if (!length(df_list)) {
    return(data.frame())
  }
  all_cols <- unique(unlist(lapply(df_list, names), use.names = FALSE))
  filled <- lapply(df_list, function(df) {
    miss <- setdiff(all_cols, names(df))
    if (length(miss)) {
      for (nm in miss) df[[nm]] <- NA
    }
    df[, all_cols, drop = FALSE]
  })
  do.call(rbind, filled)
}

.as_binary_adj_from_S <- function(S_ggm, q = 0.98) {
  if (inherits(S_ggm, "Matrix")) {
    A <- Matrix::as.matrix((abs(S_ggm) > 0) * 1)
  } else {
    A <- as.matrix(S_ggm)
    if (!is.numeric(A) || nrow(A) != ncol(A)) {
      stop("'S_ggm' must be a square numeric matrix or sparse Matrix object.")
    }
    diag(A) <- 0
    off <- abs(A[upper.tri(A)])
    off <- off[is.finite(off)]
    if (length(off) == 0L) {
      A[, ] <- 0
    } else {
      thr <- as.numeric(stats::quantile(off, probs = q, na.rm = TRUE))
      A <- (abs(A) >= thr) * 1
    }
  }
  A <- (A + t(A) > 0) * 1
  diag(A) <- 0
  A
}

.derive_phase_adjs <- function(adj_type, p,
                               X, adj_fixed, adj_fixed2, S_ggm,
                               glasso_criterion,
                               phase_adj1, phase_adj2,
                               phase_sggm_q,
                               get_glasso_adj) {
  # User overrides take priority.
  if (!is.null(phase_adj1)) {
    phase_adj1 <- .prepare_adj(phase_adj1, p, "phase_adj1")
  }
  if (!is.null(phase_adj2)) {
    phase_adj2 <- .prepare_adj(phase_adj2, p, "phase_adj2")
  }

  if (adj_type == "fixed" && is.null(phase_adj1)) {
    phase_adj1 <- .prepare_adj(adj_fixed, p, "adj_fixed")
  }
  if (adj_type == "dual_fixed") {
    if (is.null(phase_adj1)) phase_adj1 <- .prepare_adj(adj_fixed, p, "adj_fixed")
    if (is.null(phase_adj2)) phase_adj2 <- .prepare_adj(adj_fixed2, p, "adj_fixed2")
  }
  if (adj_type == "glasso" && is.null(phase_adj1)) {
    phase_adj1 <- .prepare_adj(get_glasso_adj(), p, "glasso_adj")
  }
  if (adj_type == "glasso_fixed") {
    if (is.null(phase_adj1)) phase_adj1 <- .prepare_adj(get_glasso_adj(), p, "glasso_adj")
    if (is.null(phase_adj2)) phase_adj2 <- .prepare_adj(adj_fixed, p, "adj_fixed")
  }
  if (adj_type == "ggm" && is.null(phase_adj1)) {
    if (!is.null(S_ggm)) {
      phase_adj1 <- .prepare_adj(.as_binary_adj_from_S(S_ggm, q = phase_sggm_q), p, "phase_adj1")
    } else {
      phase_adj1 <- .prepare_adj(get_glasso_adj(), p, "glasso_adj")
    }
  }
  if (adj_type == "ggm_fixed") {
    if (is.null(phase_adj1)) {
      if (!is.null(S_ggm)) {
        phase_adj1 <- .prepare_adj(.as_binary_adj_from_S(S_ggm, q = phase_sggm_q), p, "phase_adj1")
      } else {
        phase_adj1 <- .prepare_adj(get_glasso_adj(), p, "glasso_adj")
      }
    }
    if (is.null(phase_adj2)) phase_adj2 <- .prepare_adj(adj_fixed, p, "adj_fixed")
  }

  list(adj1 = phase_adj1, adj2 = phase_adj2)
}

#' Tune Eta Hyperparameters Under FDR Constraint
#'
#' Runs a grid search over Ising upper-bound hyperparameters
#' (\code{eta1_sd} or \code{eta1_sd}/\code{eta2_sd}) and corresponding
#' Moller hyperparameters for Bayesian variable selection via \code{bvs_mh}
#' or \code{bvs_pg}.
#'
#' The routine targets \code{FDR <= target_fdr} and selects the best candidate
#' by maximum sensitivity among feasible candidates.
#'
#' @param X Design matrix.
#' @param y Binary response.
#' @param adj_type Adjacency type passed to \code{bvs_mh}/\code{bvs_pg}.
#' @param sampler Sampler type: \code{"mh"} or \code{"pg"}.
#' @param truth_idx Optional ground-truth active indices (required for
#'   supervised FDR/sensitivity optimization).
#' @param adj_fixed Fixed adjacency matrix when required by \code{adj_type}.
#' @param adj_fixed2 Second fixed adjacency matrix for \code{dual_fixed}.
#' @param sparse Logical; use sparse backend for \code{ggm}/\code{ggm_fixed}.
#' @param S_ggm Optional precomputed sparse/dense GGM scatter matrix.
#' @param glasso_criterion Glasso criterion (\code{"ebic"} or \code{"ric"}).
#' @param grid Optional preconstructed tuning grid.
#' @param eta1_sd Optional single-eta (or dual-eta network 1) candidate vector.
#' @param eta2_sd Optional dual-eta candidate vector for fixed network.
#' @param eta1_frac Phase-fraction grid for single-eta (or dual-eta network 1) auto-grid.
#' @param eta2_frac Phase-fraction grid for dual \code{eta2_sd} auto-grid.
#' @param phase_adj1 Optional adjacency override for phase profiling of network 1.
#' @param phase_adj2 Optional adjacency override for phase profiling of network 2.
#' @param phase_min_eta Minimum eta for phase profiling.
#' @param phase_max_eta Maximum eta for phase profiling.
#' @param phase_step_size Eta step size for phase profiling.
#' @param phase_num_rep Replications for phase profiling.
#' @param phase_Tmax Propp-Wilson horizon for phase profiling.
#' @param phase_sggm_q Quantile for converting dense \code{S_ggm} to binary
#'   adjacency for phase profiling.
#' @param mu_tilde Candidate values of Moller auxiliary \code{mu_tilde}.
#' @param eta1_tilde Candidate values of Moller \code{eta1_tilde}.
#' @param eta2_tilde Candidate values of Moller \code{eta2_tilde}.
#' @param e_eta Candidate values for Moller Beta prior shape \code{a}.
#' @param f_eta Candidate values for Moller Beta prior shape \code{b}.
#' @param niter,burnin,thin MCMC controls passed to the sampler.
#' @param n_mh_gamma Number of gamma MH updates per MCMC iteration. If NULL,
#'   defaults adapt to dimensionality (75 for \eqn{p \ge 1000}, 25 for
#'   \eqn{300 \le p < 1000}, otherwise 10).
#' @param mu,nu0,sigmasq0,h,alpha0,beta0 Other model priors.
#' @param Tmax,proposal_type Moller/Propp-Wilson controls.
#' @param v0_ggm,v1_ggm,pii_ggm,lambda_ggm GGM prior controls.
#' @param pip_threshold PIP threshold to define selected variables.
#' @param target_fdr Target upper bound for FDR.
#' @param seed Optional base random seed. If supplied, candidate \code{i}
#'   runs with seed \code{seed + i - 1}.
#' @param keep_fits Logical; store all fitted objects.
#' @param verbose Logical; print progress.
#'
#' @return An object of class \code{"bvs_eta_tune"} with components:
#' \describe{
#'   \item{results}{Candidate-by-candidate tuning table.}
#'   \item{best_index}{Row index of selected best candidate.}
#'   \item{best}{One-row data.frame of best candidate.}
#'   \item{best_fit}{Best fit object (if \code{keep_fits = TRUE}).}
#'   \item{fits}{All fit objects (if \code{keep_fits = TRUE}).}
#'   \item{grid}{Evaluated tuning grid.}
#'   \item{phase_profile}{Phase profile(s) used for auto-grid construction.}
#' }
#'
#' @export
bvs_tune_eta <- function(
  X, y,
  adj_type = c("fixed", "dual_fixed", "glasso", "glasso_fixed", "ggm", "ggm_fixed"),
  sampler = c("mh", "pg"),
  truth_idx = NULL,
  adj_fixed = NULL,
  adj_fixed2 = NULL,
  sparse = FALSE,
  S_ggm = NULL,
  glasso_criterion = c("ebic", "ric"),
  grid = NULL,
  eta1_sd = NULL,
  eta2_sd = NULL,
  eta1_frac = seq(0.2, 1.0, by = 0.1),
  eta2_frac = seq(0.2, 1.0, by = 0.1),
  phase_adj1 = NULL,
  phase_adj2 = NULL,
  phase_min_eta = 0,
  phase_max_eta = 1.5,
  phase_step_size = 0.05,
  phase_num_rep = 8L,
  phase_Tmax = 64L,
  phase_sggm_q = 0.98,
  mu_tilde = -4,
  eta1_tilde = 0.075,
  eta2_tilde = 0.065,
  e_eta = 1,
  f_eta = 1,
  niter = 50000L,
  burnin = 20000L,
  thin = 1L,
  n_mh_gamma = NULL,
  mu = stats::qlogis(0.02),
  nu0 = 2,
  sigmasq0 = 1.5,
  h = 1.5,
  alpha0 = 0,
  beta0 = 0,
  Tmax = 64L,
  proposal_type = 1L,
  v0_ggm = 0.015^2,
  v1_ggm = NULL,
  pii_ggm = NULL,
  lambda_ggm = 1,
  z_dat = NULL,           # TUN-2: Forward z_dat (always-included covariates)
  tau0 = 0,               # TUN-2: Forward tau0 (prior mean for tau)
  htau = 1.5,             # TUN-2: Forward htau (tau variance inflation factor)
  tau_init = NULL,         # TUN-2: Forward tau_init
  pip_threshold = 0.50,
  target_fdr = 0.05,
  seed = NULL,
  keep_fits = FALSE,
  verbose = TRUE
) {
  adj_type <- match.arg(adj_type)
  sampler <- match.arg(sampler)
  glasso_criterion <- match.arg(glasso_criterion)

  if (!isTRUE(sparse)) X <- as.matrix(X)
  y <- as.numeric(y)
  p <- ncol(X)

  if (is.null(n_mh_gamma)) {
    n_mh_gamma <- if (p >= 1000L) 75L else if (p >= 300L) 25L else 10L
  }
  n_mh_gamma <- as.integer(n_mh_gamma)
  if (!is.finite(n_mh_gamma) || n_mh_gamma < 1L) {
    stop("'n_mh_gamma' must be a positive integer.")
  }

  if (!is.null(truth_idx)) {
    truth_idx <- as.integer(truth_idx)
    truth_idx <- truth_idx[is.finite(truth_idx)]
    truth_idx <- unique(truth_idx[truth_idx >= 1L & truth_idx <= p])
  }

  if (adj_type %in% c("fixed", "glasso_fixed", "ggm_fixed", "dual_fixed") &&
    is.null(adj_fixed)) {
    stop("adj_type='", adj_type, "' requires 'adj_fixed'.")
  }
  if (adj_type == "dual_fixed" && is.null(adj_fixed2)) {
    stop("adj_type='dual_fixed' requires 'adj_fixed2'.")
  }

  fit_fun <- if (sampler == "mh") bvs_mh else bvs_pg
  is_dual <- adj_type %in% c("dual_fixed", "glasso_fixed", "ggm_fixed")

  # Cache glasso adjacency so tuning loops do not re-estimate it repeatedly.
  adj_glasso_cache <- NULL
  get_glasso_adj <- function() {
    if (is.null(adj_glasso_cache)) {
      adj_glasso_cache <<- estimate_glasso_adj(X, criterion = glasso_criterion)
    }
    adj_glasso_cache
  }

  run_adj_type <- adj_type
  run_adj_fixed <- adj_fixed
  run_adj_fixed2 <- adj_fixed2

  if (adj_type == "glasso") {
    run_adj_type <- "fixed"
    run_adj_fixed <- get_glasso_adj()
  } else if (adj_type == "glasso_fixed") {
    run_adj_type <- "dual_fixed"
    run_adj_fixed <- get_glasso_adj()
    run_adj_fixed2 <- adj_fixed
  }

  run_sparse <- isTRUE(sparse) && run_adj_type %in% c("ggm", "ggm_fixed")
  run_S_ggm <- if (run_sparse) S_ggm else NULL

  phase_profile <- NULL
  if (is.null(grid)) {
    phase_adjs <- .derive_phase_adjs(
      adj_type = adj_type, p = p, X = X,
      adj_fixed = adj_fixed, adj_fixed2 = adj_fixed2, S_ggm = S_ggm,
      glasso_criterion = glasso_criterion,
      phase_adj1 = phase_adj1, phase_adj2 = phase_adj2,
      phase_sggm_q = phase_sggm_q,
      get_glasso_adj = get_glasso_adj
    )

    if (is_dual) {
      grid <- bvs_eta_grid_dual(
        eta1_sd = eta1_sd,
        eta2_sd = eta2_sd,
        adj1 = phase_adjs$adj1,
        adj2 = phase_adjs$adj2,
        eta1_frac = eta1_frac,
        eta2_frac = eta2_frac,
        mu = mu,
        phase_min_eta = phase_min_eta,
        phase_max_eta = phase_max_eta,
        phase_step_size = phase_step_size,
        phase_num_rep = phase_num_rep,
        phase_Tmax = phase_Tmax,
        mu_tilde = mu_tilde,
        eta1_tilde = eta1_tilde,
        eta2_tilde = eta2_tilde,
        e_eta = e_eta,
        f_eta = f_eta
      )
      phase_profile <- list(
        network1 = attr(grid, "phase_profile1"),
        network2 = attr(grid, "phase_profile2")
      )
    } else {
      grid <- bvs_eta_grid_single(
        eta1_sd = eta1_sd,
        adj = phase_adjs$adj1,
        eta1_frac = eta1_frac,
        mu = mu,
        phase_min_eta = phase_min_eta,
        phase_max_eta = phase_max_eta,
        phase_step_size = phase_step_size,
        phase_num_rep = phase_num_rep,
        phase_Tmax = phase_Tmax,
        mu_tilde = mu_tilde,
        eta1_tilde = eta1_tilde,
        e_eta = e_eta,
        f_eta = f_eta
      )
      phase_profile <- list(network1 = attr(grid, "phase_profile"))
    }
  }

  grid <- as.data.frame(grid, stringsAsFactors = FALSE)
  if (nrow(grid) < 1L) stop("Tuning grid is empty.")

  if (verbose) {
    message(
      "Running ", nrow(grid), " tuning candidates [sampler=", sampler,
      ", adj_type=", adj_type, "]."
    )
  }

  pb <- NULL
  if (verbose && nrow(grid) > 1L) {
    pb <- utils::txtProgressBar(min = 0, max = nrow(grid), style = 3)
  }
  on.exit(
    {
      if (!is.null(pb)) close(pb)
    },
    add = TRUE
  )

  fits <- if (isTRUE(keep_fits)) vector("list", nrow(grid)) else NULL
  results <- vector("list", nrow(grid))

  for (i in seq_len(nrow(grid))) {
    if (!is.null(seed)) set.seed(as.integer(seed) + i - 1L)

    gi <- grid[i, , drop = FALSE]
    args_i <- list(
      X = X, y = y,
      adj_type = run_adj_type,
      adj_fixed = run_adj_fixed,
      adj_fixed2 = run_adj_fixed2,
      sparse = run_sparse,
      S_ggm = run_S_ggm,
      store_beta = FALSE,
      store_gamma = TRUE,
      store_Z_list = FALSE,
      store_Z_pip = FALSE,
      glasso_criterion = glasso_criterion,
      niter = as.integer(niter),
      burnin = as.integer(burnin),
      thin = as.integer(thin),
      nu0 = nu0, sigmasq0 = sigmasq0, h = h,
      mu = mu, alpha0 = alpha0, beta0 = beta0,
      n_mh_gamma = as.integer(n_mh_gamma),
      mu_tilde = as.numeric(gi$mu_tilde),
      eta1_tilde = as.numeric(gi$eta1_tilde),
      eta2_tilde = if ("eta2_tilde" %in% names(gi)) as.numeric(gi$eta2_tilde) else eta2_tilde,
      e_eta = as.numeric(gi$e_eta),
      f_eta = as.numeric(gi$f_eta),
      Tmax = as.integer(Tmax),
      proposal_type = as.integer(proposal_type),
      v0_ggm = v0_ggm,
      v1_ggm = v1_ggm,
      pii_ggm = pii_ggm,
      lambda_ggm = lambda_ggm,
      z_dat = z_dat,          # TUN-2: Forward z_dat
      tau0 = tau0,            # TUN-2: Forward tau0
      htau = htau,            # TUN-2: Forward htau
      tau_init = tau_init     # TUN-2: Forward tau_init
    )

    if (is_dual) {
      args_i$eta1_sd <- as.numeric(gi$eta1_sd)
      args_i$eta2_sd <- as.numeric(gi$eta2_sd)
    } else {
      args_i$eta1_sd <- as.numeric(gi$eta1_sd)
    }

    fit_time <- system.time({
      fit_i <- do.call(fit_fun, args_i)
    })

    pip <- .tune_extract_pip(fit_i)
    met <- bvs_selection_metrics(
      pip = pip, truth_idx = truth_idx,
      pip_threshold = pip_threshold, p = p
    )

    out_i <- cbind(
      gi,
      met,
      elapsed_sec = as.numeric(fit_time["elapsed"]),
      iter_per_sec = niter / pmax(1e-12, as.numeric(fit_time["elapsed"])),
      stringsAsFactors = FALSE
    )
    results[[i]] <- out_i

    if (isTRUE(keep_fits)) fits[[i]] <- fit_i

    if (!is.null(pb)) utils::setTxtProgressBar(pb, i)
  }

  res_df <- do.call(rbind, results)

  has_supervised <- !is.null(truth_idx) && length(truth_idx) > 0L
  best_idx <- NA_integer_
  if (has_supervised) {
    feasible <- is.finite(res_df$fdr) & (res_df$fdr <= target_fdr)
    sens <- ifelse(is.finite(res_df$sensitivity), res_df$sensitivity, -Inf)
    fdr_val <- ifelse(is.finite(res_df$fdr), res_df$fdr, Inf)
    n_sel <- ifelse(is.finite(res_df$selected_size), res_df$selected_size, Inf)

    if (any(feasible)) {
      cand <- which(feasible)
      ord <- order(-sens[cand], fdr_val[cand], n_sel[cand])
      best_idx <- cand[ord[1L]]
    } else {
      ord <- order(fdr_val, -sens, n_sel)
      best_idx <- ord[1L]
    }
  }

  out <- list(
    call = match.call(),
    sampler = sampler,
    adj_type = adj_type,
    run_adj_type = run_adj_type,
    target_fdr = target_fdr,
    pip_threshold = pip_threshold,
    truth_idx = truth_idx,
    grid = grid,
    results = res_df,
    best_index = best_idx,
    best = if (is.na(best_idx)) NULL else res_df[best_idx, , drop = FALSE],
    best_fit = if (isTRUE(keep_fits) && !is.na(best_idx)) fits[[best_idx]] else NULL,
    fits = if (isTRUE(keep_fits)) fits else NULL,
    phase_profile = phase_profile
  )
  class(out) <- "bvs_eta_tune"
  out
}

#' Tune Eta Hyperparameters at a Fixed PIP Threshold
#'
#' Wrapper around \code{\link{bvs_tune_eta}} that enforces a constant
#' \code{pip_threshold} (default 0.5) while optimizing eta hyperparameters under
#' an FDR constraint.
#'
#' @inheritParams bvs_tune_eta
#' @param ... Arguments passed to \code{\link{bvs_tune_eta}}.
#' @param pip_threshold_fixed Fixed PIP threshold used for all candidates.
#'
#' @return A \code{"bvs_eta_tune"} object (same structure as
#'   \code{\link{bvs_tune_eta}}).
#'
#' @export
bvs_tune_eta_fixed_pip <- function(...,
                                   pip_threshold_fixed = 0.5,
                                   target_fdr = 0.05) {
  pip_threshold_fixed <- as.numeric(pip_threshold_fixed)[1]
  if (!is.finite(pip_threshold_fixed) ||
    pip_threshold_fixed <= 0 ||
    pip_threshold_fixed >= 1) {
    stop("'pip_threshold_fixed' must be a finite scalar in (0, 1).")
  }

  args <- list(...)
  args$pip_threshold <- NULL
  args$target_fdr <- NULL
  args$pip_threshold <- pip_threshold_fixed
  args$target_fdr <- target_fdr

  out <- do.call(bvs_tune_eta, args)
  out$pip_threshold <- pip_threshold_fixed
  out$target_fdr <- target_fdr
  out
}

#' Tune Eta for Four Standard Scenarios (Sim1/Sim2/Real1/Real2)
#'
#' Convenience batch runner over four scenarios using a constant PIP threshold
#' (default 0.5) and shared FDR target (default 0.05).
#'
#' This helper is designed for the common workflow of two simulation studies
#' and two real-data applications.
#'
#' @param sim1,sim2,real1,real2 Scenario argument lists. Each list should contain
#'   the scenario-specific inputs required by \code{\link{bvs_tune_eta}}
#'   (for example \code{X}, \code{y}, \code{adj_type}, adjacency inputs, and
#'   optional \code{truth_idx}).
#' @param pip_threshold_fixed Fixed PIP threshold used for all scenarios.
#' @param target_fdr Target upper bound for FDR.
#' @param require_truth Logical; if TRUE, each scenario must include non-empty
#'   \code{truth_idx} for supervised FDR/sensitivity optimization.
#' @param verbose Logical; print progress across scenarios.
#' @param ... Common arguments passed to every scenario (for example
#'   \code{sampler}, \code{niter}, \code{burnin}, \code{thin}, and other MCMC
#'   hyperparameters).
#'
#' @return A list with:
#' \describe{
#'   \item{scenario_results}{Named list of \code{"bvs_eta_tune"} objects.}
#'   \item{best_table}{One-row-per-scenario summary of selected candidates.}
#' }
#'
#' @export
bvs_tune_eta_four_examples <- function(sim1, sim2, real1, real2,
                                       pip_threshold_fixed = 0.5,
                                       target_fdr = 0.05,
                                       require_truth = TRUE,
                                       verbose = TRUE,
                                       ...) {
  scenarios <- list(sim1 = sim1, sim2 = sim2, real1 = real1, real2 = real2)

  for (nm in names(scenarios)) {
    if (!is.list(scenarios[[nm]])) {
      stop("Scenario '", nm, "' must be a list of arguments for bvs_tune_eta.")
    }
    if (isTRUE(require_truth)) {
      ti <- scenarios[[nm]]$truth_idx
      if (is.null(ti) || !length(ti)) {
        stop(
          "Scenario '", nm, "' is missing 'truth_idx'. ",
          "Set 'require_truth = FALSE' to run unsupervised tuning."
        )
      }
    }
  }

  args <- list(...)
  args$scenarios <- scenarios
  args$pip_threshold <- pip_threshold_fixed
  args$target_fdr <- target_fdr
  args$verbose <- verbose

  do.call(bvs_tune_eta_batch, args)
}

#' Tune Eta Hyperparameters Across Multiple Scenarios
#'
#' Batch wrapper over \code{\link{bvs_tune_eta}} for running tuning across
#' multiple simulation/real-data scenarios.
#'
#' @param scenarios Named list of scenario lists. Each scenario should contain
#'   arguments accepted by \code{\link{bvs_tune_eta}} (for example
#'   \code{X}, \code{y}, \code{adj_type}, \code{truth_idx}, and adjacency inputs).
#' @param ... Common arguments passed to every scenario. Scenario-specific values
#'   override common values.
#' @param verbose Logical; print progress across scenarios.
#'
#' @return A list with:
#' \describe{
#'   \item{scenario_results}{Named list of \code{"bvs_eta_tune"} objects.}
#'   \item{best_table}{One-row-per-scenario summary of selected best candidate.}
#' }
#'
#' @export
bvs_tune_eta_batch <- function(scenarios, ..., verbose = TRUE) {
  if (!is.list(scenarios) || length(scenarios) < 1L) {
    stop("'scenarios' must be a non-empty list.")
  }
  common <- list(...)

  nm <- names(scenarios)
  if (is.null(nm)) nm <- paste0("scenario_", seq_along(scenarios))
  names(scenarios) <- nm

  out <- vector("list", length(scenarios))
  names(out) <- nm

  for (k in seq_along(scenarios)) {
    sc_name <- nm[k]
    if (verbose) message("[", sc_name, "] tuning started.")

    sc <- scenarios[[k]]
    if (!is.list(sc)) stop("Each scenario must be a list of arguments.")
    args_k <- utils::modifyList(common, sc, keep.null = TRUE)
    out[[k]] <- do.call(bvs_tune_eta, args_k)

    if (verbose) message("[", sc_name, "] tuning finished.")
  }

  best_rows <- lapply(seq_along(out), function(k) {
    fit_k <- out[[k]]
    if (is.null(fit_k$best)) {
      data.frame(
        scenario = nm[k],
        best_index = NA_integer_,
        stringsAsFactors = FALSE
      )
    } else {
      cbind(
        data.frame(scenario = nm[k], best_index = fit_k$best_index, stringsAsFactors = FALSE),
        fit_k$best
      )
    }
  })

  list(
    scenario_results = out,
    best_table = .rbind_fill_df(best_rows)
  )
}

#' @method print bvs_eta_tune
#' @export
print.bvs_eta_tune <- function(x, ...) {
  cat("BVS eta tuning result\n")
  cat("  sampler   :", x$sampler, "\n")
  cat("  adj_type  :", x$adj_type, "\n")
  cat("  candidates:", nrow(x$results), "\n")
  if (!is.null(x$best)) {
    cat("  best index:", x$best_index, "\n")
    if ("fdr" %in% names(x$best) && "sensitivity" %in% names(x$best)) {
      cat("  best FDR  :", format(x$best$fdr, digits = 4), "\n")
      cat("  best Sens.:", format(x$best$sensitivity, digits = 4), "\n")
    }
  } else {
    cat("  best index: <not selected; truth_idx not provided>\n")
  }
  invisible(x)
}
