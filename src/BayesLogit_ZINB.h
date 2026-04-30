// BayesLogit_ZINB.h
// ---------------------------------------------------------------------------
// Numerics for outcome_type = "ZIC":
//   Bayesian Zero-Inflated Negative Binomial (ZINB) model.
//
//   Model:
//     Z_i ~ Bernoulli(pi)              [structural zero indicator]
//     Y_i | Z_i=1  => Y_i = 0         [structural zero]
//     Y_i | Z_i=0  => Y_i ~ NB(r, mu_i)  [at-risk count]
//     log(mu_i) = alpha + X_i*beta + Z_dat_i*tau
//
//   Data augmentation:
//     Z_i Gibbs-sampled from full conditional
//     pi ~ Beta(a_pi, b_pi) conjugate Gibbs
//     r optionally MH-updated via log-normal proposal
//     w_i ~ Gamma(r + y_i, r + mu_i) Poisson-Gamma augmentation for at-risk obs
//
// Design: all ZINB-specific code centralized here, header-only.
// ---------------------------------------------------------------------------
#ifndef BVS_DADJ_BAYESLOGIT_ZINB_H
#define BVS_DADJ_BAYESLOGIT_ZINB_H

#include "BayesLogit_Numerics.h"
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>

namespace bvs_zinb {

// ============================================================================
// 1.  Numerical helpers
// ============================================================================

static const double ZINB_PI_LO  = 1e-12;
static const double ZINB_PI_HI  = 1.0 - 1e-12;
static const double ZINB_R_LO   = 1e-6;
static const double ZINB_R_HI   = 1e6;
static const double ZINB_ETA_LO = -50.0;
static const double ZINB_ETA_HI =  50.0;

// Numerically stable log(exp(a) + exp(b))
inline double log_sum_exp(double a, double b) {
  if (!std::isfinite(a) && !std::isfinite(b)) return -std::numeric_limits<double>::infinity();
  if (!std::isfinite(a)) return b;
  if (!std::isfinite(b)) return a;
  double mx = std::max(a, b);
  return mx + std::log1p(std::exp(-std::fabs(a - b)));
}

// NB(y; r, mu) log-likelihood for a single observation
// NB(y; r, mu) = Gamma(r+y)/(Gamma(r)*y!) * (r/(r+mu))^r * (mu/(r+mu))^y
inline double nb_loglik(double y, double mu, double r) {
  mu = std::max(mu, 1e-300);
  r  = std::max(r,  ZINB_R_LO);
  double log_p = std::log(r / (r + mu));       // log(r/(r+mu))
  double log_q = std::log(mu / (r + mu));       // log(mu/(r+mu))
  return std::lgamma(r + y) - std::lgamma(r) - std::lgamma(y + 1.0)
         + r * log_p + y * log_q;
}

// ============================================================================
// 2.  ZINB marginal log-likelihood (single observation)
// ============================================================================
//   P(Y=0) = pi + (1-pi) * (r/(r+mu))^r
//   P(Y=k>0) = (1-pi) * NB(k; r, mu)

inline double zinb_loglik_obs(double y, double mu, double r, double pi) {
  mu = std::max(mu, 1e-300);
  r  = std::max(r,  ZINB_R_LO);
  pi = std::max(ZINB_PI_LO, std::min(ZINB_PI_HI, pi));

  if (y < 0.5) {
    // Y = 0:  log(pi + (1-pi) * NB(0; r, mu))
    double log_pi   = std::log(pi);
    double log_nb0  = r * std::log(r / (r + mu));  // log NB(0; r, mu)
    double log_1mpi = std::log(1.0 - pi);
    return log_sum_exp(log_pi, log_1mpi + log_nb0);
  } else {
    // Y = k > 0:  log((1-pi) * NB(k; r, mu))
    return std::log(1.0 - pi) + nb_loglik(y, mu, r);
  }
}

// ============================================================================
// 3.  Gibbs update for latent Z_i (zero-inflation indicators)
// ============================================================================
//   Y_i > 0:  Z_i = 0 (deterministic)
//   Y_i = 0:  Z_i ~ Bernoulli(p_i*)
//     p_i* = pi / (pi + (1-pi)*(r/(r+mu_i))^r)

inline void gibbs_update_Z(arma::uvec &Z_latent,
                           const arma::uvec &y_count,
                           const arma::vec &Xb_total,
                           double alpha,
                           double r, double pi,
                           arma::uword n) {
  pi = std::max(ZINB_PI_LO, std::min(ZINB_PI_HI, pi));
  r  = std::max(r, ZINB_R_LO);
  const double log_pi   = std::log(pi);
  const double log_1mpi = std::log(1.0 - pi);

  for (arma::uword i = 0; i < n; ++i) {
    if (y_count(i) > 0u) {
      Z_latent(i) = 0u;  // cannot be structural zero
      continue;
    }
    // Y_i = 0: compute P(Z_i=1 | Y_i=0)
    double eta = bvs_dadj::clamp_finite(alpha + Xb_total(i),
                                        ZINB_ETA_LO, ZINB_ETA_HI, 0.0);
    double mu_i = std::exp(eta);
    double log_nb0 = r * std::log(r / (r + mu_i));  // log NB(0; r, mu_i)
    // log P(Z=1, Y=0) = log(pi)
    // log P(Z=0, Y=0) = log(1-pi) + log_nb0
    double log_p_struct = log_pi;
    double log_p_count  = log_1mpi + log_nb0;
    // p_star = exp(log_p_struct) / (exp(log_p_struct) + exp(log_p_count))
    //        = 1 / (1 + exp(log_p_count - log_p_struct))
    double diff = log_p_count - log_p_struct;
    double p_star;
    if (diff > 30.0) {
      p_star = 0.0;  // count overwhelmingly more likely
    } else if (diff < -30.0) {
      p_star = 1.0;  // structural zero overwhelmingly more likely
    } else {
      p_star = 1.0 / (1.0 + std::exp(diff));
    }
    p_star = std::max(ZINB_PI_LO, std::min(ZINB_PI_HI, p_star));
    Z_latent(i) = (R::runif(0.0, 1.0) < p_star) ? 1u : 0u;
  }
}

// ============================================================================
// 4.  Gibbs update for scalar pi (conjugate Beta)
// ============================================================================
//   pi | Z ~ Beta(a_pi + n_struct, b_pi + n_atrisk)

inline double gibbs_update_pi(const arma::uvec &Z_latent, arma::uword n,
                              double a_pi, double b_pi) {
  double n_struct = 0.0;
  for (arma::uword i = 0; i < n; ++i)
    n_struct += (Z_latent(i) == 1u) ? 1.0 : 0.0;
  double n_atrisk = static_cast<double>(n) - n_struct;
  double pi_new = R::rbeta(a_pi + n_struct, b_pi + n_atrisk);
  return std::max(ZINB_PI_LO, std::min(ZINB_PI_HI, pi_new));
}

// ============================================================================
// 5.  MH update for NB dispersion r (log-normal proposal)
// ============================================================================
// IMPORTANT: The proposal is symmetric on the LOG scale:
//   log(r*) = log(r) + N(0, prop_sd^2)
// But the target density is on the ORIGINAL r scale. The Jacobian of the
// transformation r = exp(log_r) is |dr/d(log_r)| = r, so:
//   q(r* | r) = (1/r*) * Normal(log r*; log r, prop_sd^2)
//   q(r | r*) = (1/r)  * Normal(log r; log r*, prop_sd^2)
//   log(q(r | r*) / q(r* | r)) = log(r*) - log(r)
// This Jacobian must be ADDED to the log acceptance ratio.

inline double mh_update_r(double r_curr,
                          const arma::uvec &y_count,
                          const arma::vec &Xb_total,
                          double alpha,
                          const arma::uvec &Z_latent,
                          arma::uword n,
                          double a_r, double b_r,
                          double prop_sd = 0.5,
                          int n_mh_r = 3) {
  double r_out = r_curr;
  for (int attempt = 0; attempt < n_mh_r; ++attempt) {
    // Propose on log scale
    double log_r_curr = std::log(r_out);
    double log_r_prop = R::rnorm(log_r_curr, prop_sd);
    double r_prop = std::exp(log_r_prop);
    r_prop = std::max(ZINB_R_LO, std::min(ZINB_R_HI, r_prop));

    // Log-likelihood ratio (at-risk observations only)
    double ll_prop = 0.0, ll_curr = 0.0;
    for (arma::uword i = 0; i < n; ++i) {
      if (Z_latent(i) == 1u) continue;  // skip structural zeros
      double eta = bvs_dadj::clamp_finite(alpha + Xb_total(i),
                                          ZINB_ETA_LO, ZINB_ETA_HI, 0.0);
      double mu_i = std::exp(eta);
      double yi = static_cast<double>(y_count(i));
      ll_prop += nb_loglik(yi, mu_i, r_prop);
      ll_curr += nb_loglik(yi, mu_i, r_out);
    }

    // Gamma prior on r: log p(r) = (a_r-1)*log(r) - b_r*r + const
    double log_prior_diff = (a_r - 1.0) * (std::log(r_prop) - std::log(r_out))
                            - b_r * (r_prop - r_out);

    // Jacobian correction for log-normal proposal: log(r_prop) - log(r_curr)
    double log_jacobian = std::log(r_prop) - std::log(r_out);

    double log_ratio = (ll_prop - ll_curr) + log_prior_diff + log_jacobian;

    if (bvs_dadj::safe_mh_accept(log_ratio))
      r_out = r_prop;
  }
  return r_out;
}

// ============================================================================
// 6.  Conditional Poisson log-likelihood (at-risk obs only)
// ============================================================================
//   For Z_i = 0:  y_i * eta_i - exp(eta_i)  where eta = alpha + Xb + log_w
//   For Z_i = 1:  skip (structural zero)

inline double calc_loglik_zic_conditional(
    const arma::uvec &y_count,
    const arma::vec &z,        // Xb_total - Z_tau  (predictor without offset)
    double alpha,
    const arma::vec &offset,   // Z_tau
    const arma::vec &log_w,
    const arma::uvec &Z_latent,
    arma::uword n) {
  double ll = 0.0;
  for (arma::uword i = 0; i < n; ++i) {
    if (Z_latent(i) == 1u) continue;  // structural zero
    const double eta = bvs_dadj::clamp_finite(
        alpha + z(i) + offset(i) + log_w(i), ZINB_ETA_LO, ZINB_ETA_HI, 0.0);
    ll += static_cast<double>(y_count(i)) * eta - std::exp(eta);
  }
  return ll;
}

// ============================================================================
// 7.  MALA residual for ZIC (single observation)
// ============================================================================
//   Z_i = 1 (structural zero): 0.0 (no gradient)
//   Z_i = 0 (at-risk):  y_i - exp(eta_i)

inline double zic_mala_residual(unsigned int y_count_i,
                                double alpha_plus_Xb_total_i,
                                double log_w_i,
                                unsigned int Z_latent_i) {
  if (Z_latent_i == 1u) return 0.0;
  double eta = bvs_dadj::clamp_finite(
      alpha_plus_Xb_total_i + log_w_i, ZINB_ETA_LO, ZINB_ETA_HI, 0.0);
  return static_cast<double>(y_count_i) - std::exp(eta);
}

// ============================================================================
// 8.  Refresh Poisson-Gamma latent w_i for at-risk obs
// ============================================================================
//   Z_i = 1: w_i = 1, log_w_i = 0 (no augmentation needed)
//   Z_i = 0: w_i ~ Gamma(r + y_i, r + mu_i)

inline void refresh_zic_poisson_gamma(
    arma::vec &w_count, arma::vec &log_w_count,
    const arma::uvec &y_count,
    const arma::vec &Xb_total,
    double alpha, double r,
    const arma::uvec &Z_latent,
    arma::uword n) {
  r = std::max(r, ZINB_R_LO);
  for (arma::uword i = 0; i < n; ++i) {
    if (Z_latent(i) == 1u) {
      w_count(i) = 1.0;
      log_w_count(i) = 0.0;
      continue;
    }
    double eta = bvs_dadj::clamp_finite(alpha + Xb_total(i),
                                        ZINB_ETA_LO, ZINB_ETA_HI, 0.0);
    double mu_i = std::exp(eta);
    double shape_post = r + static_cast<double>(y_count(i));
    double rate_post  = r + mu_i;
    double wi = R::rgamma(shape_post, 1.0 / rate_post);
    if (!std::isfinite(wi) || wi <= 0.0)
      wi = shape_post / std::max(rate_post, 1e-12);
    w_count(i) = wi;
    log_w_count(i) = std::log(wi);
  }
}

// ============================================================================
// 9.  Column-wise LL diff for sparse gamma update (ZIC)
// ============================================================================
//   Same as count but skips structural zeros (Z_i=1)

inline double column_ll_diff_zic(
    const arma::uvec &y_count,
    const arma::vec &Xb, double alpha,
    const arma::vec &log_w,
    const arma::uvec &Z_latent,
    const arma::uword *col_ptr,
    const arma::uword *row_idx,
    const double *xvals,
    int j, double db) {
  if (std::abs(db) < 1e-16) return 0.0;

  double ll_diff = 0.0;
  arma::uword start = col_ptr[j];
  arma::uword end   = col_ptr[j + 1];
  for (arma::uword k = start; k < end; ++k) {
    int i = static_cast<int>(row_idx[k]);
    if (Z_latent(i) == 1u) continue;  // structural zero
    double xij = xvals[k];
    const double eta_base = alpha + Xb(i) + log_w(i);
    const double eta_old = bvs_dadj::clamp_finite(eta_base, ZINB_ETA_LO, ZINB_ETA_HI, 0.0);
    const double eta_new = bvs_dadj::clamp_finite(eta_base + db * xij, ZINB_ETA_LO, ZINB_ETA_HI, 0.0);
    ll_diff += static_cast<double>(y_count(i)) * (eta_new - eta_old)
               - (std::exp(eta_new) - std::exp(eta_old));
  }
  return ll_diff;
}

} // namespace bvs_zinb

#endif // BVS_DADJ_BAYESLOGIT_ZINB_H
