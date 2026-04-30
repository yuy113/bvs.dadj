// BayesLogit_Imbalanced.h
// ---------------------------------------------------------------------------
// Numerics for outcome_type = "imbalanced_binary":
//   Approach A  (logit_t):  logistic likelihood + Student-t / Cauchy prior
//   Approach B  (cloglog):  complementary log-log likelihood + Gaussian prior
//
// Design: all imbalanced-specific code is centralised here so that existing
// headers grow minimally.  Functions are header-only (inline / static inline)
// for zero link-time overhead.
// ---------------------------------------------------------------------------
#ifndef BVS_DADJ_BAYESLOGIT_IMBALANCED_H
#define BVS_DADJ_BAYESLOGIT_IMBALANCED_H

#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <string>

namespace bvs_imbalanced {

// ============================================================================
// 1.  Link-type parsing
// ============================================================================
//   0 = logit_t  (Approach A: logistic link + Student-t prior)
//   1 = cloglog  (Approach B: complementary log-log link)

inline int parse_imbalanced_link(const std::string &link) {
  if (link == "logit_t") return 0;
  if (link == "cloglog") return 1;
  Rcpp::stop("imbalanced_link must be 'logit_t' or 'cloglog'; got '%s'.",
             link.c_str());
  return -1; // unreachable
}

// ============================================================================
// 2.  Complementary log-log likelihood  (single observation)
// ============================================================================
//
//   pi  = 1 - exp(-mu),   mu = exp(eta)
//
//   y=1:  log(pi)   = log(1 - exp(-mu))  = log1p(-exp(-mu))
//   y=0:  log(1-pi) = -mu
//
//   Stability:
//     mu > 30     =>  log(pi) ~ -exp(-mu) ~ 0       [underflow-safe]
//     mu < 1e-10  =>  log(pi) ~ log(mu) = eta       [first-order Taylor]
//     eta clamped to [-30, 20] (exp(exp(20)) ~ 5e8, safe; exp(exp(30)) OFs)

static const double CLOGLOG_ETA_LO = -30.0;
static const double CLOGLOG_ETA_HI =  20.0;

inline double loglik_cloglog_obs(double yi, double eta) {
  // Clamp eta for numerical safety
  if (!std::isfinite(eta))
    eta = (eta > 0.0) ? CLOGLOG_ETA_HI : CLOGLOG_ETA_LO;
  eta = std::max(CLOGLOG_ETA_LO, std::min(CLOGLOG_ETA_HI, eta));

  const double mu = std::exp(eta);  // always >= 0

  if (yi > 0.5) {
    // y = 1:  log(1 - exp(-mu))
    if (mu > 30.0) {
      // pi ~ 1, log(pi) ~ -exp(-mu) ~ 0
      return -std::exp(-mu);
    }
    if (mu < 1e-10) {
      // Taylor: log(1 - exp(-mu)) ~ log(mu) = eta
      return eta;
    }
    return std::log1p(-std::exp(-mu));
  } else {
    // y = 0:  log(1 - pi) = -mu
    return -mu;
  }
}

// ============================================================================
// 3.  Cloglog full log-likelihood (dense)
// ============================================================================

inline double calc_loglik_cloglog_full(const arma::vec &y,
                                       const arma::vec &Xb_total,
                                       double alpha) {
  const arma::uword n = y.n_elem;
  double ll = 0.0;
  for (arma::uword i = 0; i < n; ++i)
    ll += loglik_cloglog_obs(y(i), alpha + Xb_total(i));
  return ll;
}

// ============================================================================
// 4.  Cloglog gradient contribution per observation  (MALA residual)
// ============================================================================
//
//   y=1:  d logL / d eta =  mu * S / (1 - S)   where S = exp(-mu)
//   y=0:  d logL / d eta = -mu
//
//   Stability:
//     mu > 30   =>  mu*S/(1-S) ~ mu*exp(-mu) -> 0   (underflow-safe)
//     mu < 1e-10 =>  mu*S/(1-S) ~ 1                  (Taylor)
//     When 1 - S < 1e-15 use max(1-S, 1e-15) to avoid 0/0

inline double cloglog_residual(double yi, double eta) {
  eta = std::max(CLOGLOG_ETA_LO, std::min(CLOGLOG_ETA_HI, eta));
  const double mu = std::exp(eta);

  if (yi > 0.5) {
    // y = 1
    if (mu > 30.0)
      return mu * std::exp(-mu);  // ~ 0
    if (mu < 1e-10)
      return 1.0;                 // Taylor limit
    const double S = std::exp(-mu);
    return mu * S / std::max(1.0 - S, 1e-15);
  } else {
    // y = 0
    return -mu;
  }
}

// ============================================================================
// 5.  Student-t latent scale-mixture Gibbs sampler
// ============================================================================
//
//   beta_j | gamma_j=1, lambda_j, sigma^2  ~  N(beta0, sigma^2 * s^2 * lambda_j)
//   lambda_j  ~  IG(nu/2, nu/2)
//
//   Full conditional:
//     lambda_j | beta_j, gamma_j=1  ~  IG( (nu+1)/2,  (nu + (beta_j - beta0)^2 / (sigma^2 s^2)) / 2 )
//     lambda_j | gamma_j=0          ~  IG( nu/2,  nu/2 )   [draw from prior]
//
//   Clamp lambda to [1e-8, 1e8] for numerical safety.

static const double LAMBDA_LO = 1e-8;
static const double LAMBDA_HI = 1e8;

inline void gibbs_update_lambda(arma::vec &lambda,
                                const arma::vec &beta,
                                const arma::uvec &gamma,
                                double beta0,
                                double sigmasq,
                                double t_df,
                                double t_scale) {
  const arma::uword p = beta.n_elem;
  const double s2 = t_scale * t_scale;
  const double half_df = 0.5 * t_df;

  for (arma::uword j = 0; j < p; ++j) {
    double shape, rate;
    if (gamma(j) == 1u) {
      const double bj = beta(j) - beta0;
      shape = half_df + 0.5;                        // (nu+1)/2
      rate  = half_df + 0.5 * bj * bj / (sigmasq * s2);  // (nu + bj^2/(sig^2 s^2)) / 2
    } else {
      shape = half_df;
      rate  = half_df;
    }

    // Sample from IG(shape, rate) = 1 / Gamma(shape, 1/rate)
    const double gam_sample = R::rgamma(shape, 1.0 / rate);
    double lam = (gam_sample > 0.0) ? (1.0 / gam_sample) : LAMBDA_HI;

    // Clamp
    lambda(j) = std::max(LAMBDA_LO, std::min(LAMBDA_HI, lam));
  }
}

// ============================================================================
// 6.  Effective prior variance for beta_j
// ============================================================================
//
//   logit_t (Approach A):  sigma^2 * s^2 * lambda_j
//   cloglog (Approach B):  sigma^2                     (unchanged Gaussian)
//   non-imbalanced:        sigma^2                     (standard)

inline double beta_prior_var_j(double sigmasq,
                               int imb_link_code,
                               double t_scale,
                               double lambda_j) {
  if (imb_link_code == 0) {
    // logit_t: Student-t via scale mixture
    return sigmasq * t_scale * t_scale * lambda_j;
  }
  // cloglog or non-imbalanced: standard Gaussian prior
  return sigmasq;
}

// ============================================================================
// 7.  Cloglog column-wise LL diff (for sparse gamma update)
// ============================================================================
//   Used by sparse backends: given CSC representation of X,
//   compute sum_i [ loglik_cloglog(y_i, eta_new) - loglik_cloglog(y_i, eta_old) ]
//   where eta_new = eta_old + db * X_{i,j}  for nonzero entries only.

inline double column_ll_diff_cloglog(const arma::vec &y,
                                     const arma::vec &Xb,
                                     double alpha,
                                     const arma::uword *col_ptr,
                                     const arma::uword *row_idx,
                                     const double *xvals,
                                     int j, double db) {
  if (std::abs(db) < 1e-16)
    return 0.0;

  double ll_diff = 0.0;
  const arma::uword start = col_ptr[j];
  const arma::uword end   = col_ptr[j + 1];
  for (arma::uword k = start; k < end; ++k) {
    const int i = static_cast<int>(row_idx[k]);
    const double xij = xvals[k];
    const double psi_old = alpha + Xb(i);
    const double psi_new = psi_old + db * xij;
    ll_diff += loglik_cloglog_obs(y(i), psi_new) -
               loglik_cloglog_obs(y(i), psi_old);
  }
  return ll_diff;
}

// ============================================================================
// 8.  Cloglog full loglik computation (dense, from Xb_total)
// ============================================================================
//   Used in MALA full loglik recalculation.

inline double cloglog_full_loglik(const arma::vec &y,
                                  double alpha,
                                  const arma::vec &Xb_total) {
  return calc_loglik_cloglog_full(y, Xb_total, alpha);
}

} // namespace bvs_imbalanced

#endif // BVS_DADJ_BAYESLOGIT_IMBALANCED_H
