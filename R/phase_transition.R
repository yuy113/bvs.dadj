#' Phase Transition Detection
#'
#' Sweeps the Ising coupling parameter eta across a range and generates gamma
#' vectors via Propp-Wilson perfect simulation at each value to visualize the
#' phase transition in model size.
#'
#' @param adj  A \code{p x p} binary adjacency matrix.
#' @param adj2  Optional second adjacency matrix (for dual-eta phase transition).
#' @param mu  Ising external field (default \code{-log(1/0.3-1)}).
#' @param min_eta  Minimum eta value (default 0).
#' @param max_eta  Maximum eta value (default 2).
#' @param step_size  Step size for eta grid (default 0.01).
#' @param num_rep  Number of Propp-Wilson replications per eta (default 10).
#' @param Tmax  Maximum Propp-Wilson doubling time (default 64).
#'
#' @return An integer matrix of model sizes (one row per eta value,
#'   one column per replication).
#'
#' @details
#' For dual-eta mode (when \code{adj2} is supplied), both eta1 and eta2 are
#' swept jointly.  The phase transition point indicates where the Ising prior
#' transitions from sparse to dense model selection.
#'
#' @examples
#' \dontrun{
#' p <- 50
#' R <- diag(0, p)
#' R[1:10, 1:10] <- 1; diag(R) <- 0
#' pt <- phase_transition(R, max_eta = 1.5)
#' matplot(pt, type = "l", xlab = "eta index", ylab = "model size")
#' }
#'
#' @export
phase_transition <- function(adj,
                             adj2 = NULL,
                             mu = -log(1/0.3 - 1),
                             min_eta = 0,
                             max_eta = 2,
                             step_size = 0.01,
                             num_rep = 10L,
                             Tmax = 64L) {

  p <- nrow(adj)
  R1 <- .prepare_adj(adj, p, "adj")

  if (is.null(adj2)) {
    # Single-eta phase transition
    phase_transit_1eta(
      R1 = R1,
      T_max = as.integer(Tmax), mu = mu,
      min_eta = min_eta, max_eta = max_eta,
      num_rep = as.integer(num_rep),
      step_size = step_size)
  } else {
    R2 <- .prepare_adj(adj2, p, "adj2")
    phase_transit_2eta(
      R1 = R1, R2 = R2,
      T_max = as.integer(Tmax), mu = mu,
      min_eta = min_eta, max_eta = max_eta,
      num_rep = as.integer(num_rep),
      step_size = step_size)
  }
}
