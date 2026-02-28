// Shared numerics helpers for robust linear algebra in MCMC updates.
#ifndef BVS_DADJ_BAYESLOGIT_NUMERICS_H
#define BVS_DADJ_BAYESLOGIT_NUMERICS_H

#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>

namespace bvs_dadj {

inline double clamp_finite(double x, double lo, double hi,
                           double fallback = 0.0) {
  if (!std::isfinite(x))
    return fallback;
  if (x < lo)
    return lo;
  if (x > hi)
    return hi;
  return x;
}

inline void sanitize_vec_inplace(arma::vec &v, double lo, double hi,
                                 double fallback = 0.0) {
  for (arma::uword i = 0; i < v.n_elem; ++i)
    v(i) = clamp_finite(v(i), lo, hi, fallback);
}

inline void sanitize_sym_mat_inplace(arma::mat &A, double abs_max = 1e12,
                                     double diag_floor = 1e-10) {
  if (A.n_rows != A.n_cols)
    return;

  const arma::uword p = A.n_rows;
  for (arma::uword i = 0; i < p; ++i) {
    for (arma::uword j = i; j < p; ++j) {
      double aij = clamp_finite(A(i, j), -abs_max, abs_max, 0.0);
      double aji = clamp_finite(A(j, i), -abs_max, abs_max, 0.0);
      double v = 0.5 * (aij + aji);
      A(i, j) = v;
      A(j, i) = v;
    }
    A(i, i) = clamp_finite(A(i, i), diag_floor, abs_max, diag_floor);
  }
}

inline bool robust_chol_inplace(arma::mat &chol_upper, arma::mat &A,
                                int max_attempts = 8,
                                double jitter_init = 1e-10,
                                double jitter_mult = 10.0) {
  if (A.n_rows != A.n_cols)
    return false;

  if (A.n_elem == 0) {
    chol_upper.set_size(0, 0);
    return true;
  }

  // Numerical safety: finite cleanup + symmetry before Cholesky.
  sanitize_sym_mat_inplace(A);

  if (arma::chol(chol_upper, arma::symmatu(A)))
    return true;

  double dscale = arma::mean(arma::abs(A.diag()));
  if (!std::isfinite(dscale) || dscale <= 0.0)
    dscale = 1.0;

  double jitter = std::max(jitter_init, 1e-12 * dscale);
  for (int k = 0; k < max_attempts; ++k) {
    A.diag() += jitter;
    if (arma::chol(chol_upper, arma::symmatu(A)))
      return true;
    jitter *= jitter_mult;
  }

  return false;
}

inline bool robust_chol(arma::mat &chol_upper, const arma::mat &A_in,
                        int max_attempts = 8,
                        double jitter_init = 1e-10,
                        double jitter_mult = 10.0) {
  arma::mat A = A_in;
  return robust_chol_inplace(chol_upper, A, max_attempts, jitter_init,
                             jitter_mult);
}

// -----------------------------------------------------------------------
// Robust Metropolis-Hastings acceptance
//
// Avoids deadlock from log(0) when R::runif returns exactly 0 by using
// the probability form  U < exp(log_ratio)  instead of  log(U) < log_ratio.
// Also guards against NaN and +/-Inf in the log-ratio.
// -----------------------------------------------------------------------
inline bool safe_mh_accept(double log_ratio) {
  if (std::isnan(log_ratio))
    return false; // numerical error → reject
  if (log_ratio >= 0.0)
    return true; // always accept (includes +Inf)
  // Use probability form: avoids log(0) deadlock entirely
  return R::runif(0.0, 1.0) < std::exp(log_ratio);
}

} // namespace bvs_dadj

#endif
