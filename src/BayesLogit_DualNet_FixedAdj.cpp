// [[Rcpp::depends(RcppArmadillo)]]
#include "BayesLogit_Numerics.h"
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <random>

namespace {

// =============================================================================
// SECTION 1: HELPER FUNCTIONS
// =============================================================================

// Logistic Log-Likelihood (numerically stable)
static inline double calc_loglik(const arma::vec &y, const arma::vec &z,
                                 double alpha) {
  const arma::uword n = y.n_elem;
  double term1 = 0.0, term2 = 0.0;
  for (arma::uword i = 0; i < n; ++i) {
    double x = alpha + z(i);
    term1 += y(i) * x;
    if (x > 0.0)
      term2 += x + std::log1p(std::exp(-x));
    else
      term2 += std::log1p(std::exp(x));
  }
  return term1 - term2;
}

// Normal PDF (not log)
static inline double normal_pdf(double x, double mu, double sigma) {
  double z = (x - mu) / sigma;
  return std::exp(-0.5 * z * z) / (sigma * std::sqrt(2.0 * M_PI));
}

// Normal CDF via approximation of erf
static inline double approx_erf(double x) {
  double y = 1.0 / (1.0 + 0.3275911 * std::abs(x));
  double val =
      1.0 - (((((+1.061405429 * y - 1.453152027) * y + 1.421413741) * y -
               0.284496736) *
                  y +
              0.254829592) *
             y) *
                std::exp(-x * x);
  return (x >= 0.0) ? val : -val;
}

static inline double normal_cdf(double x, double mu, double sigma) {
  return 0.5 * (1.0 + approx_erf((x - mu) / (sigma * std::sqrt(2.0))));
}

// Log Beta density: log Beta(x | a, b)
static inline double log_beta_pdf(double x, double a, double b) {
  x = std::max(1e-12, std::min(1.0 - 1e-12, x));
  return std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
         (a - 1.0) * std::log(x) + (b - 1.0) * std::log(1.0 - x);
}

// =============================================================================
// PROPP-WILSON PERFECT SIMULATION
//
// Generates a sample omega from the MRF distribution:
//   P(omega | mu, eta1, eta2, R1, R2) propto
//       exp{ mu * sum(omega_j) + eta1 * sum_{R1} omega_j omega_k
//                               + eta2 * sum_{R2} omega_j omega_k }
//
// Uses monotone coupling with upper/lower chains running backwards in time.
// If coalescence is not achieved by T_max, falls back to Gibbs sampling
// from the upper chain.
//
// Reference: Propp & Wilson (1996), Stingo et al. (2011)
// =============================================================================
static Rcpp::IntegerVector proppwilson_omega(const Rcpp::IntegerMatrix &R1,
                                             const Rcpp::IntegerMatrix &R2,
                                             double mu, double eta1,
                                             double eta2, unsigned int T_max) {
  const unsigned int p = R1.ncol();
  unsigned int T = 2;

  Rcpp::IntegerVector x_up(p, 0);
  Rcpp::IntegerVector x_down(p, 1);
  Rcpp::NumericVector pi_up(p), pi_down(p);

  int seed_base = static_cast<int>(std::floor(R::runif(0.0, 1.0) * 1000.0)) + 1;

  while (Rcpp::sum(x_up != x_down) > 0) {

    // Reset chains
    for (unsigned int k = 0; k < p; ++k) {
      x_up[k] = 0;
      x_down[k] = 1;
    }

    // Run chains from -T to -1
    for (int t = -(int)T; t <= -1; ++t) {

      // Deterministic seed for this time step for coupling
      int seed2 = -t * seed_base;
      std::mt19937 gen(seed2);
      std::uniform_real_distribution<double> unif01(0.0, 1.0);

      for (unsigned int i = 0; i < p; ++i) {
        double ker_up = 0.0, ker_down = 0.0;
        for (unsigned int j = 0; j < p; ++j) {
          double r1ij = R1(i, j);
          double r2ij = R2(i, j);
          ker_up += eta1 * r1ij * x_up[j] + eta2 * r2ij * x_up[j];
          ker_down += eta1 * r1ij * x_down[j] + eta2 * r2ij * x_down[j];
        }
        pi_up[i] = 1.0 / (1.0 + std::exp(-(mu + ker_up)));
        pi_down[i] = 1.0 / (1.0 + std::exp(-(mu + ker_down)));

        double u = unif01(gen);
        x_up[i] = (pi_up[i] > u) ? 1 : 0;
        x_down[i] = (pi_down[i] > u) ? 1 : 0;
      }
    }

    T = 2 * T;

    // Safety: if T exceeds T_max, force coalescence via Gibbs from upper
    if (T >= T_max) {
      // Identify disagreeing sites
      std::vector<unsigned int> diff_idx;
      for (unsigned int k = 0; k < p; ++k) {
        if (x_up[k] != x_down[k])
          diff_idx.push_back(k);
      }

      // Run extra Gibbs sweeps on disagreeing sites from x_up
      std::mt19937 gen_fb(seed_base + 99999);
      std::uniform_real_distribution<double> unif01_fb(0.0, 1.0);
      for (int sweep = 0; sweep < 100; ++sweep) {
        for (unsigned int idx = 0; idx < diff_idx.size(); ++idx) {
          unsigned int m = diff_idx[idx];
          double ker = 0.0;
          for (unsigned int k = 0; k < p; ++k) {
            ker += eta1 * R1(m, k) * x_up[k] + eta2 * R2(m, k) * x_up[k];
          }
          double prob = 1.0 / (1.0 + std::exp(-(mu + ker)));
          x_up[m] = (unif01_fb(gen_fb) < prob) ? 1 : 0;
        }
      }
      // Force coalescence
      for (unsigned int k = 0; k < p; ++k)
        x_down[k] = x_up[k];
    }
  }

  return x_up;
}

// =============================================================================
// MOLLER ET AL. (2006) MH UPDATE FOR eta1, eta2
//
// Uses auxiliary variables omega sampled via Propp-Wilson perfect simulation
// to handle the intractable normalizing constant in the MRF prior.
//
// Two proposal kernels available:
//   proposal_type = 0: Uniform on [max(0, eta-0.01), min(eta_sd, eta+0.01)]
//   proposal_type = 1: Truncated Normal(eta, eta_sd) on (0, eta_sd)
//
// Prior: eta_i / eta_i_sd ~ Beta(e, f), so eta_i ~ scaled Beta on (0, eta_i_sd)
// =============================================================================
static void
moller_update_eta(const Rcpp::IntegerMatrix &R1, const Rcpp::IntegerMatrix &R2,
                  double mu, double &eta1, double &eta2, double eta1_sd,
                  double eta2_sd, double mu_tilde, double eta1_tilde,
                  double eta2_tilde, const arma::uvec &gamma, double e_eta,
                  double f_eta, unsigned int T_max, int proposal_type) {
  const unsigned int p = R1.ncol();

  // --- Propose new eta1, eta2 ---
  double eta1_new, eta2_new;

  double log_prop_ratio_eta1 = 0.0;
  double log_prop_ratio_eta2 = 0.0;

  if (proposal_type == 0) {
    // Uniform proposal: U[max(0, eta-0.01), min(eta_sd, eta+0.01)]
    double a1 = std::max(0.0, eta1 - 0.01);
    double b1 = std::min(eta1_sd, eta1 + 0.01);
    eta1_new = R::runif(a1, b1);

    double a2 = std::max(0.0, eta2 - 0.01);
    double b2 = std::min(eta2_sd, eta2 + 0.01);
    eta2_new = R::runif(a2, b2);

    // Reverse proposal ranges
    double c1 = std::max(0.0, eta1_new - 0.01);
    double d1 = std::min(eta1_sd, eta1_new + 0.01);
    double c2 = std::max(0.0, eta2_new - 0.01);
    double d2 = std::min(eta2_sd, eta2_new + 0.01);

    // log q(eta|eta_new) - log q(eta_new|eta) = log(b-a) - log(d-c)
    log_prop_ratio_eta1 = std::log(b1 - a1) - std::log(d1 - c1);
    log_prop_ratio_eta2 = std::log(b2 - a2) - std::log(d2 - c2);
  } else {
    // Truncated normal proposal: N(eta, eta_sd) truncated to (0, eta_sd)
    int attempts = 0;
    do {
      eta1_new = R::rnorm(eta1, eta1_sd);
      if (++attempts > 10000) {
        eta1_new = eta1;
        break;
      }
    } while (eta1_new <= 0.0 || eta1_new >= eta1_sd);

    attempts = 0;
    do {
      eta2_new = R::rnorm(eta2, eta2_sd);
      if (++attempts > 10000) {
        eta2_new = eta2;
        break;
      }
    } while (eta2_new <= 0.0 || eta2_new >= eta2_sd);

    // Transition density ratio for truncated normal
    double log_q_fwd_1 = std::log(normal_pdf(eta1_new, eta1, eta1_sd)) -
                         std::log(normal_cdf(eta1_sd, eta1, eta1_sd) -
                                  normal_cdf(0.0, eta1, eta1_sd));
    double log_q_rev_1 = std::log(normal_pdf(eta1, eta1_new, eta1_sd)) -
                         std::log(normal_cdf(eta1_sd, eta1_new, eta1_sd) -
                                  normal_cdf(0.0, eta1_new, eta1_sd));
    log_prop_ratio_eta1 = log_q_rev_1 - log_q_fwd_1;

    double log_q_fwd_2 = std::log(normal_pdf(eta2_new, eta2, eta2_sd)) -
                         std::log(normal_cdf(eta2_sd, eta2, eta2_sd) -
                                  normal_cdf(0.0, eta2, eta2_sd));
    double log_q_rev_2 = std::log(normal_pdf(eta2, eta2_new, eta2_sd)) -
                         std::log(normal_cdf(eta2_sd, eta2_new, eta2_sd) -
                                  normal_cdf(0.0, eta2_new, eta2_sd));
    log_prop_ratio_eta2 = log_q_rev_2 - log_q_fwd_2;
  }

  // Clamp to valid range
  eta1_new = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1_new));
  eta2_new = std::max(1e-8, std::min(eta2_sd - 1e-8, eta2_new));

  // --- Generate auxiliary variables via Propp-Wilson ---
  Rcpp::IntegerVector omega_eta1 =
      proppwilson_omega(R1, R2, mu, eta1, eta2, T_max);
  Rcpp::IntegerVector omega_eta2 =
      proppwilson_omega(R1, R2, mu, eta1, eta2, T_max);
  Rcpp::IntegerVector omega_eta1_new =
      proppwilson_omega(R1, R2, mu, eta1_new, eta2, T_max);
  Rcpp::IntegerVector omega_eta2_new =
      proppwilson_omega(R1, R2, mu, eta1, eta2_new, T_max);

  // --- Compute sufficient statistics ---
  int B_R1 = 0, B_R2 = 0;
  int A_om1_R1 = 0, A_om1_R2 = 0;
  int A_om2_R1 = 0, A_om2_R2 = 0;
  int A_om1n_R1 = 0, A_om1n_R2 = 0;
  int A_om2n_R1 = 0, A_om2n_R2 = 0;
  int sum_om1 = 0, sum_om2 = 0, sum_om1n = 0, sum_om2n = 0;

  for (unsigned int j = 0; j < p; ++j) {
    sum_om1 += omega_eta1[j];
    sum_om2 += omega_eta2[j];
    sum_om1n += omega_eta1_new[j];
    sum_om2n += omega_eta2_new[j];

    for (unsigned int k = j + 1; k < p; ++k) {
      int r1jk = R1(j, k);
      int r2jk = R2(j, k);
      if (r1jk == 0 && r2jk == 0)
        continue;

      B_R1 += (int)gamma(j) * r1jk * (int)gamma(k);
      B_R2 += (int)gamma(j) * r2jk * (int)gamma(k);

      A_om1_R1 += omega_eta1[j] * r1jk * omega_eta1[k];
      A_om1_R2 += omega_eta1[j] * r2jk * omega_eta1[k];
      A_om2_R1 += omega_eta2[j] * r1jk * omega_eta2[k];
      A_om2_R2 += omega_eta2[j] * r2jk * omega_eta2[k];
      A_om1n_R1 += omega_eta1_new[j] * r1jk * omega_eta1_new[k];
      A_om1n_R2 += omega_eta1_new[j] * r2jk * omega_eta1_new[k];
      A_om2n_R1 += omega_eta2_new[j] * r1jk * omega_eta2_new[k];
      A_om2n_R2 += omega_eta2_new[j] * r2jk * omega_eta2_new[k];
    }
  }

  // --- Prior ratio: log Beta(eta_new/eta_sd | e,f) - log Beta(eta/eta_sd |
  // e,f) ---
  double log_prior_eta1 = log_beta_pdf(eta1_new / eta1_sd, e_eta, f_eta) -
                          log_beta_pdf(eta1 / eta1_sd, e_eta, f_eta);
  double log_prior_eta2 = log_beta_pdf(eta2_new / eta2_sd, e_eta, f_eta) -
                          log_beta_pdf(eta2 / eta2_sd, e_eta, f_eta);

  // --- MH ratio (Moller et al. 2006) ---
  // Target ratio for eta1: (eta1_new - eta1) * gamma'R1*gamma
  double log_target_eta1 = (eta1_new - eta1) * B_R1 + log_prior_eta1;
  double log_aux_eta1 = mu_tilde * (sum_om1n - sum_om1) +
                        eta1_tilde * (A_om1n_R1 - A_om1_R1) +
                        eta2_tilde * (A_om1n_R2 - A_om1_R2);
  // Normalizing constant cancellation term
  double log_norm_eta1 = mu * (sum_om1 - sum_om1n) + eta1 * A_om1_R1 -
                         eta1_new * A_om1n_R1 + eta2 * (A_om1_R2 - A_om1n_R2);

  double log_MH_eta1 =
      log_target_eta1 + log_aux_eta1 + log_norm_eta1 + log_prop_ratio_eta1;

  // Same for eta2
  double log_target_eta2 = (eta2_new - eta2) * B_R2 + log_prior_eta2;
  double log_aux_eta2 = mu_tilde * (sum_om2n - sum_om2) +
                        eta1_tilde * (A_om2n_R1 - A_om2_R1) +
                        eta2_tilde * (A_om2n_R2 - A_om2_R2);
  double log_norm_eta2 = mu * (sum_om2 - sum_om2n) + eta2 * A_om2_R2 -
                         eta2_new * A_om2n_R2 + eta1 * (A_om2_R1 - A_om2n_R1);

  double log_MH_eta2 =
      log_target_eta2 + log_aux_eta2 + log_norm_eta2 + log_prop_ratio_eta2;

  // --- Accept/reject ---
  if (bvs_dadj::safe_mh_accept(log_MH_eta1)) {
    eta1 = eta1_new;
  }
  if (bvs_dadj::safe_mh_accept(log_MH_eta2)) {
    eta2 = eta2_new;
  }
}

} // anonymous namespace
// =============================================================================
// MAIN FUNCTION
// =============================================================================

// [[Rcpp::export]]
Rcpp::List BayesLogit_DualNet_FixedAdj(
    const arma::mat &X, const arma::vec &y, Rcpp::IntegerMatrix R_dyn_int,
    Rcpp::IntegerMatrix R_fix_int, int niter, int burnin, double mu, double nu0,
    double sigmasq0, double alpha0, double beta0, double h, double e_eta,
    double f_eta, double eta1_sd, double eta2_sd, double mu_tilde,
    double eta1_tilde, double eta2_tilde, unsigned int T_max, int proposal_type,
    int thin = 1, int n_thin_gb = 3,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0) {
  Rcpp::RNGScope scope;

  const arma::uword p = X.n_cols;
  if ((unsigned int)R_dyn_int.nrow() != p ||
      (unsigned int)R_dyn_int.ncol() != p)
    Rcpp::stop("R_dyn dimensions must match p = %d", p);
  if ((unsigned int)R_fix_int.nrow() != p ||
      (unsigned int)R_fix_int.ncol() != p)
    Rcpp::stop("R_fix dimensions must match p = %d", p);
  if (thin < 1)
    thin = 1;
  if (n_thin_gb < 1)
    n_thin_gb = 1;

  // Create arma::mat versions for fast submatrix operations in gamma updates
  arma::mat R_dyn(p, p);
  arma::mat R_fix(p, p);
  for (arma::uword i = 0; i < p; ++i) {
    for (arma::uword j = 0; j < p; ++j) {
      R_dyn(i, j) = static_cast<double>(R_dyn_int(i, j));
      R_fix(i, j) = static_cast<double>(R_fix_int(i, j));
    }
  }

  // 1. INITIALIZATION
  arma::vec beta(p, arma::fill::zeros);
  arma::uvec gamma(p, arma::fill::zeros);
  if (beta_in.isNotNull())
    beta = Rcpp::as<arma::vec>(beta_in);
  if (gamma_in.isNotNull())
    gamma = Rcpp::as<arma::uvec>(gamma_in);

  double alpha = alpha_in;
  double sigmasq = 1.0;
  double eta1 = 0.01; // Dynamic Network Weight
  double eta2 = 0.01; // Fixed Network Weight

  arma::vec z = X * beta;
  double loglik = calc_loglik(y, z, alpha);

  // Output Storage (thinned)
  int n_save = niter / thin;
  arma::mat beta_out(n_save, p);
  arma::umat gamma_out(n_save, p);
  arma::vec eta1_out(n_save), eta2_out(n_save);
  arma::vec alpha_out(n_save);
  arma::vec sigmasq_out(n_save);

  // 2. MCMC LOOP
  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {

    if (iter > 0 && iter % 10000 == 0) {
      Rcpp::checkUserInterrupt();
      int model_size = (int)arma::accu(gamma);
      Rcpp::Rcout << "Iter: " << iter << " | Model size: " << model_size
                  << " | eta1: " << eta1 << " | eta2: " << eta2 << "\n";
    }

    // ---------------------------------------------------------
    // STEP A+B: Gamma + Beta with inner thinning (n_thin_gb rounds)
    // ---------------------------------------------------------
    arma::uvec active;
    for (int thin_gb = 0; thin_gb < n_thin_gb; ++thin_gb) {

      // STEP A: Gamma (Variable Selection) via MH with dual Ising prior
      for (arma::uword k = 0; k < 5; ++k) {
        arma::uword j =
            static_cast<arma::uword>(std::floor(R::runif(0.0, (double)p)));
        if (j >= p)
          j = p - 1;

        int g_curr = gamma(j);
        int g_prop = 1 - g_curr;
        double b_curr = beta(j);
        double b_prop =
            (g_prop == 1) ? R::rnorm(beta0, std::sqrt(sigmasq)) : 0.0;

        arma::vec z_prop = z + (b_prop - b_curr) * X.col(j);
        double ll_prop = calc_loglik(y, z_prop, alpha);

        double diff = static_cast<double>(g_prop - g_curr);
        double neigh_dyn = 0.0, neigh_fix = 0.0;
        for (arma::uword i = 0; i < p; ++i) {
          if (i == j)
            continue;
          if (R_dyn(i, j) != 0.0)
            neigh_dyn += gamma(i);
          if (R_fix(i, j) != 0.0)
            neigh_fix += gamma(i);
        }

        double ising_diff = diff * (mu + eta1 * neigh_dyn + eta2 * neigh_fix);
        double log_ratio = (ll_prop - loglik) + ising_diff;

        if (bvs_dadj::safe_mh_accept(log_ratio)) {
          gamma(j) = g_prop;
          beta(j) = b_prop;
          z = z_prop;
          loglik = ll_prop;
        }
      }

      // STEP B: Beta (Metropolis updates for active variables)
      active = arma::find(gamma == 1);
      for (arma::uword k = 0; k < active.n_elem; ++k) {
        arma::uword j = active(k);
        double b_prop = R::rnorm(beta(j), std::sqrt(sigmasq) * 0.1);
        arma::vec z_prop = z + (b_prop - beta(j)) * X.col(j);
        double ll_prop = calc_loglik(y, z_prop, alpha);
        double pr_curr = -0.5 * std::pow(beta(j) - beta0, 2) / sigmasq;
        double pr_prop = -0.5 * std::pow(b_prop - beta0, 2) / sigmasq;
        if (bvs_dadj::safe_mh_accept((ll_prop - loglik) +
                                      (pr_prop - pr_curr))) {
          beta(j) = b_prop;
          z = z_prop;
          loglik = ll_prop;
        }
      }

    } // end inner thinning for gamma + beta

    // ---------------------------------------------------------
    // STEP C: Alpha (Metropolis update)
    // ---------------------------------------------------------
    double a_prop = R::rnorm(alpha, std::sqrt(h * sigmasq));
    double ll_a_prop = calc_loglik(y, z, a_prop);
    double pr_a_curr = -0.5 * std::pow(alpha - alpha0, 2) / (h * sigmasq);
    double pr_a_prop = -0.5 * std::pow(a_prop - alpha0, 2) / (h * sigmasq);
    if (bvs_dadj::safe_mh_accept((ll_a_prop - loglik) +
                                   (pr_a_prop - pr_a_curr))) {
      alpha = a_prop;
      loglik = ll_a_prop;
    }

    // ---------------------------------------------------------
    // STEP D: Update SigmaSq (log-normal proposal)
    // ---------------------------------------------------------
    {
      double sig_prop = std::exp(std::log(sigmasq) + R::rnorm(0, 0.2));
      sig_prop = std::max(sig_prop, 1e-10);
      double shape = nu0 / 2.0;
      double scale = sigmasq0 * nu0 / 2.0;

      double lp_sig_curr = -(shape + 1.0) * std::log(sigmasq) - scale / sigmasq;
      double lp_sig_prop =
          -(shape + 1.0) * std::log(sig_prop) - scale / sig_prop;

      // In-place ss_beta computation
      double n_active_beta = (double)active.n_elem;
      double ss_beta = 0.0;
      for (arma::uword k = 0; k < active.n_elem; ++k) {
        double d = beta(active(k)) - beta0;
        ss_beta += d * d;
      }

      double lp_beta_curr =
          -0.5 * n_active_beta * std::log(sigmasq) - 0.5 * ss_beta / sigmasq;
      double lp_beta_prop =
          -0.5 * n_active_beta * std::log(sig_prop) - 0.5 * ss_beta / sig_prop;

      double lp_alpha_curr = -0.5 * std::log(h * sigmasq) -
                             0.5 * std::pow(alpha - alpha0, 2) / (h * sigmasq);
      double lp_alpha_prop = -0.5 * std::log(h * sig_prop) -
                             0.5 * std::pow(alpha - alpha0, 2) / (h * sig_prop);

      double log_mh_sig = (lp_sig_prop - lp_sig_curr) +
                          (lp_beta_prop - lp_beta_curr) +
                          (lp_alpha_prop - lp_alpha_curr) +
                          (std::log(sig_prop) - std::log(sigmasq));
      if (bvs_dadj::safe_mh_accept(log_mh_sig))
        sigmasq = sig_prop;
    }

    // ---------------------------------------------------------
    // STEP E: Update eta1, eta2 via Moller et al. (2006)
    //         with Propp-Wilson perfect simulation
    // ---------------------------------------------------------
    moller_update_eta(R_dyn_int, R_fix_int, mu, eta1, eta2, eta1_sd, eta2_sd,
                      mu_tilde, eta1_tilde, eta2_tilde, gamma, e_eta, f_eta,
                      T_max, proposal_type);

    // ---------------------------------------------------------
    // STORE SAMPLES (with thinning)
    // ---------------------------------------------------------
    if (iter >= burnin && (iter - burnin) % thin == 0) {
      int s = (iter - burnin) / thin;
      if (s < n_save) {
        beta_out.row(s) = beta.t();
        for (arma::uword j = 0; j < p; ++j)
          gamma_out(s, j) = gamma(j);
        eta1_out(s) = eta1;
        eta2_out(s) = eta2;
        alpha_out(s) = alpha;
        sigmasq_out(s) = sigmasq;
      }
    }
  }

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_out, Rcpp::Named("gamma") = gamma_out,
      Rcpp::Named("eta1") = eta1_out, Rcpp::Named("eta2") = eta2_out,
      Rcpp::Named("alpha") = alpha_out, Rcpp::Named("sigmasq") = sigmasq_out);
}
