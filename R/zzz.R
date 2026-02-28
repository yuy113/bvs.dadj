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
#' @description MCMC tools for Bayesian variable selection in logistic
#'   regression with Ising/MRF priors.
#' @name BVS.DAdj-package
NULL
