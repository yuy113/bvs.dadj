#' @useDynLib BVS.DAdj, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @importFrom Matrix sparseMatrix
#' @importFrom graphics abline barplot legend
#' @importFrom methods as slotNames
#' @importFrom stats plogis rbinom rgamma rnorm quantile
#' @keywords internal
"_PACKAGE"

## Package documentation
#' BVS.DAdj: Bayesian Variable Selection with Dual-Adjacency Priors
#'
#' @description MCMC tools for Bayesian variable selection with Ising/MRF
#'   priors. The \code{bvs_mh} sampler uses Metropolis-Hastings (MH) by
#'   default, with optional Hamiltonian Monte Carlo (HMC) and No-U-Turn
#'   Sampler (NUTS) active parameter-block updates for binary and
#'   time-to-event models; it supports binary logistic outcomes,
#'   continuous Gaussian outcomes, right-censored time-to-event outcomes
#'   via Cox partial likelihood, and overdispersed count outcomes via a
#'   negative-binomial (Poisson-Gamma) representation. The PG sampler
#'   targets binary logistic outcomes.
#' @name BVS.DAdj-package
NULL
