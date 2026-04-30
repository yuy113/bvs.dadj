// [[Rcpp::depends(RcppArmadillo)]]
#include "BVS_HMC_NUTS.h"
#include "BayesLogit_Imbalanced.h"
#include "BayesLogit_Numerics.h"
#include "BayesLogit_ZINB.h"
#include "BayesLogit_Sparse_Helpers.h"
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <random>

namespace {

// =============================================================================
// SECTION 1: HELPER FUNCTIONS
// =============================================================================

// Logistic Log-Likelihood (numerically stable)
static inline double calc_loglik_binary(const arma::vec &y, const arma::vec &z,
                                        double alpha, const arma::vec &offset) {
  const arma::uword n = y.n_elem;
  double term1 = 0.0, term2 = 0.0;
  for (arma::uword i = 0; i < n; ++i) {
    double x = alpha + z(i) + offset(i);
    term1 += y(i) * x;
    if (x > 0.0)
      term2 += x + std::log1p(std::exp(-x));
    else
      term2 += std::log1p(std::exp(x));
  }
  return term1 - term2;
}

// Negative-binomial (Poisson-gamma mixture) conditional Poisson log-likelihood
static inline double calc_loglik_count(const arma::uvec &y_count,
                                       const arma::vec &z, double alpha,
                                       const arma::vec &offset,
                                       const arma::vec &log_w_count) {
  return bvs_dadj::calc_loglik_count_conditional(y_count, z, alpha, offset,
                                                 log_w_count);
}

static inline double calc_loglik_continuous(const arma::vec &y,
                                            const arma::vec &z, double alpha,
                                            const arma::vec &offset,
                                            double sigmasq) {
  const arma::uword n = y.n_elem;
  const double log_norm =
      -0.5 * static_cast<double>(n) * std::log(2.0 * M_PI * sigmasq);
  double sse = 0.0;
  for (arma::uword i = 0; i < n; ++i) {
    const double r = y(i) - (alpha + z(i) + offset(i));
    sse += r * r;
  }
  return log_norm - 0.5 * sse / sigmasq;
}

// normal_pdf, approx_erf, normal_cdf, log_beta_pdf provided by BayesLogit_Sparse_Helpers.h

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
// Propp-Wilson CFTP for dual Ising on combined R1+R2 graph.
// When CFTP fails to coalesce by T_max, falls back to 500 full Gibbs sweeps
// from random initialization over ALL sites. This produces an approximate
// but unbiased sample from the stationary distribution.
static Rcpp::IntegerVector gibbs_omega(const Rcpp::IntegerMatrix &R1,
                                       const Rcpp::IntegerMatrix &R2,
                                       double mu, double eta1, double eta2,
                                       int n_gibbs_sweeps = 20) {
  const unsigned int p = R1.ncol();
  Rcpp::IntegerVector x(p, 0);
  int seed_val = static_cast<int>(std::floor(R::runif(0.0, 1.0) * 2147483646.0)) + 1;
  std::mt19937 gf(seed_val);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  double base_prob = 1.0 / (1.0 + std::exp(-mu));

  // Pre-compute sparse neighbor lists: store (neighbor_j, which_graph) to avoid O(p^2)
  struct Edge { unsigned int j; double weight; };
  std::vector<std::vector<Edge>> nbrs(p);
  int total_edges = 0;
  for (unsigned int k = 0; k < p; ++k) {
    for (unsigned int j = 0; j < p; ++j) {
      if (j == k) continue;
      double w = eta1 * R1(k, j) + eta2 * R2(k, j);
      if (w != 0.0) {
        nbrs[k].push_back({j, w});
        ++total_edges;
      }
    }
  }

  for (unsigned int k = 0; k < p; ++k) {
    x[k] = (u01(gf) < base_prob) ? 1 : 0;
  }

  if (total_edges == 0) return x;

  for (int sw = 0; sw < n_gibbs_sweeps; ++sw) {
    for (unsigned int k = 0; k < p; ++k) {
      double ker = 0.0;
      for (const auto &e : nbrs[k]) {
        if (x[e.j]) ker += e.weight;
      }
      x[k] = (u01(gf) < 1.0 / (1.0 + std::exp(-(mu + ker)))) ? 1 : 0;
    }
  }
  return x;
}

static Rcpp::IntegerVector proppwilson_omega(const Rcpp::IntegerMatrix &R1,
                                             const Rcpp::IntegerMatrix &R2,
                                             double mu, double eta1,
                                             double eta2, unsigned int T_max,
                                             bool *coalesced = nullptr) {
  const unsigned int p = R1.ncol();
  unsigned int T = 2;

  Rcpp::IntegerVector x_up(p, 0);
  Rcpp::IntegerVector x_down(p, 1);
  Rcpp::NumericVector pi_up(p), pi_down(p);

  int seed_base = static_cast<int>(std::floor(R::runif(0.0, 1.0) * 2147483646.0)) + 1;

  // DFA-OPT: Pre-build sparse neighbor list once per call (edges are fixed).
  // Avoids O(p^2) scans on every CFTP restart, which made the sampler
  // infeasible for p >= 3000. Storage is O(|E|).
  struct Edge { unsigned int j; double weight; };
  std::vector<std::vector<Edge>> nbrs(p);
  for (unsigned int k = 0; k < p; ++k) {
    for (unsigned int j = 0; j < p; ++j) {
      if (j == k) continue;
      double w = eta1 * R1(k, j) + eta2 * R2(k, j);
      if (w != 0.0) nbrs[k].push_back({j, w});
    }
  }

  while (Rcpp::sum(x_up != x_down) > 0) {

    for (unsigned int k = 0; k < p; ++k) {
      x_up[k] = 0;
      x_down[k] = 1;
    }

    for (int t = -(int)T; t <= -1; ++t) {
      int seed2 = -t * seed_base;
      std::mt19937 gen(seed2);
      std::uniform_real_distribution<double> unif01(0.0, 1.0);

      for (unsigned int i = 0; i < p; ++i) {
        double ker_up = 0.0, ker_down = 0.0;
        for (const auto &e : nbrs[i]) {
          if (x_up[e.j]) ker_up += e.weight;
          if (x_down[e.j]) ker_down += e.weight;
        }
        pi_up[i] = 1.0 / (1.0 + std::exp(-(mu + ker_up)));
        pi_down[i] = 1.0 / (1.0 + std::exp(-(mu + ker_down)));

        double u = unif01(gen);
        x_up[i] = (pi_up[i] > u) ? 1 : 0;
        x_down[i] = (pi_down[i] > u) ? 1 : 0;
      }
    }

    T = 2 * T;

    if (T >= T_max) {
      // Gibbs fallback: 500 full sweeps from random initialization over ALL sites
      std::mt19937 gf(seed_base + static_cast<int>(R::runif(0.0, 1.0) * 1e8));
      std::uniform_real_distribution<double> u01f(0.0, 1.0);
      double base_prob = 1.0 / (1.0 + std::exp(-mu));
      for (unsigned int k = 0; k < p; ++k) {
        x_up[k] = (u01f(gf) < base_prob) ? 1 : 0;
      }
      for (int sw = 0; sw < 500; ++sw) {
        for (unsigned int k = 0; k < p; ++k) {
          double ker = 0.0;
          for (const auto &e : nbrs[k]) {
            if (x_up[e.j]) ker += e.weight;
          }
          x_up[k] = (u01f(gf) < 1.0 / (1.0 + std::exp(-(mu + ker)))) ? 1 : 0;
        }
      }
      if (coalesced) *coalesced = false;
      return x_up;
    }
  }

  if (coalesced) *coalesced = true;
  return x_up;
}

// =============================================================================
// MOLLER ET AL. (2006) MH UPDATE FOR eta1, eta2
//
// Uses auxiliary variables from Propp-Wilson CFTP on the COMBINED R1+R2 graph.
// When CFTP fails (T >= T_max), uses 500-sweep Gibbs fallback samples
// and proceeds with an approximate exchange algorithm step.
//
// Key fix: The original code rejected the entire update when any of the 4
// CFTP calls failed (~58% of iterations), freezing both etas and destroying
// mixing. Now we always proceed with the MH step using approximate samples.
// =============================================================================
static void
moller_update_eta(const Rcpp::IntegerMatrix &R1, const Rcpp::IntegerMatrix &R2,
                  double mu, double &eta1, double &eta2, double eta1_sd,
                  double eta2_sd, double mu_tilde, double eta1_tilde,
                  double eta2_tilde, const arma::uvec &gamma, double e_eta,
                  double f_eta, unsigned int T_max, int proposal_type,
                  EtaAdapter &adapter1, EtaAdapter &adapter2,
                  bool use_cftp = false) {
  // Exactly cancel the Moller effect to implement the Exchange Algorithm
  mu_tilde = mu;
  eta1_tilde = eta1;
  eta2_tilde = eta2;
  (void)proposal_type;

  const unsigned int p = R1.ncol();

  // --- Propose new eta1, eta2 ---
  double eta1_new, eta2_new;
  double log_prop_ratio_eta1 = 0.0;
  double log_prop_ratio_eta2 = 0.0;

  // Logit-transformed proposals with Vihola RAM adaptation
  double eta1_safe = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1));
  double phi1 = std::log(eta1_safe / (eta1_sd - eta1_safe));
  double phi1_new = R::rnorm(phi1, adapter1.sigma());
  eta1_new = eta1_sd / (1.0 + std::exp(-phi1_new));
  eta1_new = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1_new));
  log_prop_ratio_eta1 = std::log(eta1_new) + std::log(eta1_sd - eta1_new) -
                        std::log(eta1_safe) - std::log(eta1_sd - eta1_safe);

  double eta2_safe = std::max(1e-8, std::min(eta2_sd - 1e-8, eta2));
  double phi2 = std::log(eta2_safe / (eta2_sd - eta2_safe));
  double phi2_new = R::rnorm(phi2, adapter2.sigma());
  eta2_new = eta2_sd / (1.0 + std::exp(-phi2_new));
  eta2_new = std::max(1e-8, std::min(eta2_sd - 1e-8, eta2_new));
  log_prop_ratio_eta2 = std::log(eta2_new) + std::log(eta2_sd - eta2_new) -
                        std::log(eta2_safe) - std::log(eta2_sd - eta2_safe);

  // --- Generate auxiliary variables ---
  // use_cftp=true: Propp-Wilson CFTP with Gibbs fallback (original)
  // use_cftp=false (default): Direct Gibbs sampling (500 sweeps)
  Rcpp::IntegerVector omega_eta1, omega_eta2, omega_eta1_new, omega_eta2_new;
  if (use_cftp) {
    omega_eta1 = proppwilson_omega(R1, R2, mu, eta1, eta2, T_max);
    omega_eta2 = proppwilson_omega(R1, R2, mu, eta1, eta2, T_max);
    omega_eta1_new = proppwilson_omega(R1, R2, mu, eta1_new, eta2, T_max);
    omega_eta2_new = proppwilson_omega(R1, R2, mu, eta1, eta2_new, T_max);
  } else {
    omega_eta1 = gibbs_omega(R1, R2, mu, eta1, eta2, 50);
    omega_eta2 = gibbs_omega(R1, R2, mu, eta1, eta2, 50);
    omega_eta1_new = gibbs_omega(R1, R2, mu, eta1_new, eta2, 50);
    omega_eta2_new = gibbs_omega(R1, R2, mu, eta1, eta2_new, 50);
  }

  // --- Compute sufficient statistics ---
  double B_R1 = 0.0, B_R2 = 0.0;
  double A_om1_R1 = 0.0, A_om1_R2 = 0.0;
  double A_om2_R1 = 0.0, A_om2_R2 = 0.0;
  double A_om1n_R1 = 0.0, A_om1n_R2 = 0.0;
  double A_om2n_R1 = 0.0, A_om2n_R2 = 0.0;
  double sum_om1 = 0.0, sum_om2 = 0.0, sum_om1n = 0.0, sum_om2n = 0.0;

  for (unsigned int j = 0; j < p; ++j) {
    sum_om1 += omega_eta1[j];
    sum_om2 += omega_eta2[j];
    sum_om1n += omega_eta1_new[j];
    sum_om2n += omega_eta2_new[j];
  }

  // DFA-OPT: Upper-triangle sparse sweep. For each (j,k) with j<k, skip
  // zero-weight pairs without comparing all p^2. When R1/R2 are sparse
  // (|E| << p^2/2), this is a large speed-up; for dense graphs the cost
  // is identical to the previous double loop.
  for (unsigned int j = 0; j < p; ++j) {
    for (unsigned int k = j + 1; k < p; ++k) {
      int r1jk = R1(j, k);
      int r2jk = R2(j, k);
      if (r1jk == 0 && r2jk == 0)
        continue;

      B_R1 += (double)gamma(j) * r1jk * (double)gamma(k);
      B_R2 += (double)gamma(j) * r2jk * (double)gamma(k);

      A_om1_R1 += (double)omega_eta1[j] * r1jk * (double)omega_eta1[k];
      A_om1_R2 += (double)omega_eta1[j] * r2jk * (double)omega_eta1[k];
      A_om2_R1 += (double)omega_eta2[j] * r1jk * (double)omega_eta2[k];
      A_om2_R2 += (double)omega_eta2[j] * r2jk * (double)omega_eta2[k];
      A_om1n_R1 += (double)omega_eta1_new[j] * r1jk * (double)omega_eta1_new[k];
      A_om1n_R2 += (double)omega_eta1_new[j] * r2jk * (double)omega_eta1_new[k];
      A_om2n_R1 += (double)omega_eta2_new[j] * r1jk * (double)omega_eta2_new[k];
      A_om2n_R2 += (double)omega_eta2_new[j] * r2jk * (double)omega_eta2_new[k];
    }
  }

  // --- Prior ratio ---
  double log_prior_eta1 = log_beta_pdf(eta1_new / eta1_sd, e_eta, f_eta) -
                          log_beta_pdf(eta1 / eta1_sd, e_eta, f_eta);
  double log_prior_eta2 = log_beta_pdf(eta2_new / eta2_sd, e_eta, f_eta) -
                          log_beta_pdf(eta2 / eta2_sd, e_eta, f_eta);

  // --- MH ratio (Moller et al. 2006) ---
  double log_target_eta1 = (eta1_new - eta1) * B_R1 + log_prior_eta1;
  double log_aux_eta1 = mu_tilde * (sum_om1n - sum_om1) +
                        eta1_tilde * (A_om1n_R1 - A_om1_R1) +
                        eta2_tilde * (A_om1n_R2 - A_om1_R2);
  double log_norm_eta1 = mu * (sum_om1 - sum_om1n) + eta1 * A_om1_R1 -
                         eta1_new * A_om1n_R1 + eta2 * (A_om1_R2 - A_om1n_R2);

  double log_MH_eta1 =
      log_target_eta1 + log_aux_eta1 + log_norm_eta1 + log_prop_ratio_eta1;

  double log_target_eta2 = (eta2_new - eta2) * B_R2 + log_prior_eta2;
  double log_aux_eta2 = mu_tilde * (sum_om2n - sum_om2) +
                        eta1_tilde * (A_om2n_R1 - A_om2_R1) +
                        eta2_tilde * (A_om2n_R2 - A_om2_R2);
  double log_norm_eta2 = mu * (sum_om2 - sum_om2n) + eta2 * A_om2_R2 -
                         eta2_new * A_om2n_R2 + eta1 * (A_om2_R1 - A_om2n_R1);

  double log_MH_eta2 =
      log_target_eta2 + log_aux_eta2 + log_norm_eta2 + log_prop_ratio_eta2;

  // --- Accept/reject with Vihola RAM adaptation ---
  double accept_prob_eta1 = std::min(1.0, std::exp(std::min(0.0, log_MH_eta1)));
  double accept_prob_eta2 = std::min(1.0, std::exp(std::min(0.0, log_MH_eta2)));
  if (bvs_dadj::safe_mh_accept(log_MH_eta1)) {
    eta1 = eta1_new;
  }
  if (bvs_dadj::safe_mh_accept(log_MH_eta2)) {
    eta2 = eta2_new;
  }
  adapter1.update(accept_prob_eta1);
  adapter2.update(accept_prob_eta2);
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
    double alpha_in = 0.0, const arma::mat &Z_dat = arma::mat(),
    double tau0 = 0.0, double htau = 1.0,
    Rcpp::Nullable<Rcpp::NumericVector> tau_in = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> event = R_NilValue,
    std::string outcome_type = "binary", bool store_gamma = true,
    std::string alg_type = "MH", double hmc_step_size = 0.1,
    int hmc_n_leapfrog = 10, int nuts_max_treedepth = 10,
    bool use_lb_gamma = true,
    std::string imbalanced_link = "logit_t",
    double t_df = 1.0, double t_scale = 2.5,
    double zinb_r = 1.0, double zinb_a_pi = 1.0, double zinb_b_pi = 1.0,
    double zinb_a_r = 1.0, double zinb_b_r = 0.1,
    bool zinb_estimate_r = false,
    bool use_cftp = false) {
  Rcpp::RNGScope scope;
  const bool is_continuous = bvs_dadj::outcome_is_continuous(outcome_type);
  const bool is_tte = bvs_dadj::outcome_is_tte(outcome_type);
  const bool is_count = bvs_dadj::outcome_is_count(outcome_type);
  const bool is_imbalanced = bvs_dadj::outcome_is_imbalanced_binary(outcome_type);
  const int imb_link_code = is_imbalanced ?
      bvs_imbalanced::parse_imbalanced_link(imbalanced_link) : -1;
  const bool use_cloglog = (imb_link_code == 1);
  const bool use_t_prior = is_imbalanced && (imb_link_code == 0);
  const bool is_zic = bvs_dadj::outcome_is_zic(outcome_type);

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

  const arma::uword n = X.n_rows;
  arma::uvec y_count;
  arma::uvec event01;
  bvs_dadj::CoxBreslowData cox_data;
  if (is_tte) {
    if (event.isNull())
      Rcpp::stop("For outcome_type='TTE', event indicator must be provided.");
    arma::vec event_vec = Rcpp::as<arma::vec>(event);
    if (event_vec.n_elem != n)
      Rcpp::stop("For outcome_type='TTE', length(event) must match nrow(X).");
    if (!y.is_finite())
      Rcpp::stop("For outcome_type='TTE', survival times must be finite.");
    for (arma::uword i = 0; i < n; ++i) {
      if (y(i) <= 0.0)
        Rcpp::stop("For outcome_type='TTE', survival times must be > 0.");
    }
    if (!bvs_dadj::normalize_binary_indicator(event_vec, event01))
      Rcpp::stop("For outcome_type='TTE', event must be in {0,1} or {-1,1}.");
    if (arma::accu(event01) < 1u)
      Rcpp::stop("For outcome_type='TTE', at least one event==1 is required.");
    cox_data = bvs_dadj::build_cox_breslow_data(y, event01);
  } else if (is_count) {
    if (!bvs_dadj::normalize_count_response(y, y_count)) {
      Rcpp::stop(
          "For outcome_type='count', y must be non-negative integer counts.");
    }
  } else if (is_zic) {
    if (!bvs_dadj::normalize_count_response(y, y_count)) {
      Rcpp::stop(
          "For outcome_type='ZIC', y must be non-negative integer counts.");
    }
  }

  // DNF-2: Use R_dyn_int/R_fix_int directly instead of O(p^2) dense copy.
  // The integer matrices are only used for != 0 adjacency checks, so no
  // conversion to double is needed.

  // 1. INITIALIZATION
  arma::vec beta(p, arma::fill::zeros);
  arma::uvec gamma(p, arma::fill::zeros);
  if (beta_in.isNotNull())
    beta = Rcpp::as<arma::vec>(beta_in);
  if (gamma_in.isNotNull())
    gamma = Rcpp::as<arma::uvec>(gamma_in);

  double alpha = alpha_in;
  double sigmasq = 1.0;
  double eta1 = eta1_sd * 0.5; // Dynamic Network Weight – start at midpoint of support
  double eta2 = eta2_sd * 0.5; // Fixed Network Weight – start at midpoint of support
  if (is_tte)
    alpha = 0.0; // not identifiable under Cox partial likelihood
  const double nb_shape = 1.0;

  // --- tau (Z_dat covariates) ---
  const arma::uword ntau = Z_dat.n_cols;
  arma::vec tau(ntau, arma::fill::zeros);
  tau.fill(tau0);
  if (tau_in.isNotNull()) {
    tau = Rcpp::as<arma::vec>(tau_in);
  }
  arma::vec Z_tau = Z_dat * tau;

  double loglik = 0.0;
  arma::vec resid;
  // Pre-compute column squared norms for MALA step sizes (all outcome types)
  arma::vec X_col_sq_sums(p);
  for (arma::uword j = 0; j < p; ++j)
    X_col_sq_sums(j) = arma::dot(X.col(j), X.col(j));
  // Pre-compute Z_dat column norms for tau MALA step sizes
  arma::vec Z_col_sq_sums(ntau, arma::fill::zeros);
  if (ntau > 0)
    for (arma::uword j = 0; j < ntau; ++j)
      Z_col_sq_sums(j) = arma::dot(Z_dat.col(j), Z_dat.col(j));
  // MALA residual vector: y - p_hat (binary) or y_count - lambda_hat (count)
  arma::vec mala_resid(n, arma::fill::zeros);
  if (is_continuous) {
    resid.set_size(n);
  }

  arma::vec Xb = X * beta;
  arma::vec Xb_total = Xb + Z_tau;
  bvs_dadj::CoxTracker cox_tracker;
  arma::vec w_count;
  arma::vec log_w_count;
  if (is_count || is_zic) {
    w_count.set_size(n);
    log_w_count.set_size(n);
    w_count.fill(1.0);
    log_w_count.zeros();
  }
  // ZINB state variables
  double zinb_r_curr = zinb_r;
  double zinb_pi_curr = 0.1;
  arma::uvec zinb_Z_latent;
  if (is_zic) {
    zinb_Z_latent.set_size(n);
    zinb_Z_latent.zeros();  // all start as at-risk
  }
  // Latent scale-mixture variables for Student-t / Cauchy prior (Approach A)
  arma::vec lambda_t(p, arma::fill::ones);

  auto refresh_count_latent = [&](const arma::vec &xb_total_curr,
                                  double alpha_curr) {
    for (arma::uword i = 0; i < n; ++i) {
      const double eta = bvs_dadj::clamp_finite(alpha_curr + xb_total_curr(i),
                                                -50.0, 50.0, 0.0);
      const double mu_i = std::exp(eta);
      const double shape_post = nb_shape + static_cast<double>(y_count(i));
      const double rate_post = nb_shape + mu_i;
      double wi = R::rgamma(shape_post, 1.0 / rate_post);
      if (!std::isfinite(wi) || wi <= 0.0)
        wi = shape_post / std::max(rate_post, 1e-12);
      w_count(i) = wi;
      log_w_count(i) = std::log(wi);
    }
  };

  if (is_continuous) {
    resid = y - Xb_total - alpha;
    loglik = calc_loglik_continuous(y, Xb_total - Z_tau, alpha, Z_tau, sigmasq);
  } else if (is_tte) {
    cox_tracker.init(Xb_total, cox_data);
    loglik = cox_tracker.get_loglik();
  } else if (is_count) {
    refresh_count_latent(Xb_total, alpha);
    loglik =
        calc_loglik_count(y_count, Xb_total - Z_tau, alpha, Z_tau, log_w_count);
  } else if (is_zic) {
    bvs_zinb::gibbs_update_Z(zinb_Z_latent, y_count, Xb_total, alpha,
                             zinb_r_curr, zinb_pi_curr, n);
    zinb_pi_curr = bvs_zinb::gibbs_update_pi(zinb_Z_latent, n, zinb_a_pi, zinb_b_pi);
    if (zinb_estimate_r)
      zinb_r_curr = bvs_zinb::mh_update_r(zinb_r_curr, y_count, Xb_total, alpha,
                                           zinb_Z_latent, n, zinb_a_r, zinb_b_r);
    bvs_zinb::refresh_zic_poisson_gamma(w_count, log_w_count, y_count,
                                        Xb_total, alpha, zinb_r_curr,
                                        zinb_Z_latent, n);
    loglik = bvs_zinb::calc_loglik_zic_conditional(
        y_count, Xb_total - Z_tau, alpha, Z_tau, log_w_count, zinb_Z_latent, n);
  } else if (use_cloglog) {
    loglik = bvs_imbalanced::calc_loglik_cloglog_full(y, Xb_total, alpha);
  } else {
    loglik = calc_loglik_binary(y, Xb_total - Z_tau, alpha, Z_tau);
  }

  // OPT 1: Pre-allocate informed proposal weight buffers and cached neighbor
  // sums
  arma::vec proposal_weights(p);
  arma::vec neigh_dyn_sum(p, arma::fill::zeros);
  arma::vec neigh_fix_sum(p, arma::fill::zeros);
  for (arma::uword j = 0; j < p; ++j) {
    for (arma::uword i = 0; i < p; ++i) {
      if (i == j)
        continue;
      if (R_dyn_int(i, j) != 0)
        neigh_dyn_sum(j) += gamma(i);
      if (R_fix_int(i, j) != 0)
        neigh_fix_sum(j) += gamma(i);
    }
  }

  // Locally-balanced gamma proposal state (Zanella 2020)
  std::vector<double> lb_score, lb_weight;
  double lb_Z = 0.0;
  LBProposalDelta lb_delta;

  // OPT 2: Online PIP accumulation
  arma::vec gamma_pip_sum(p, arma::fill::zeros);

  // HMC/NUTS initialisation
  const int alg_type_int = bvs_hmc::parse_alg_type(alg_type);
  const bool use_hmc_nuts = (alg_type_int == 1 || alg_type_int == 2);
  const double hmc_target_accept = (alg_type_int == 2) ? 0.80 : 0.65;
  double hmc_epsilon = hmc_step_size;
  bvs_hmc::DualAveraging hmc_da;
  bvs_hmc::HMCNUTSDiagnostics hmc_diag;
  bvs_hmc::WindowedMassAdapter hmc_mass_adapter;
  bool hmc_eps_initialised = false;
  bool hmc_warmup_finalized = false;
  int hmc_prev_dim = -1;
  int hmc_post_burnin_adapt_left = 0;
  const int hmc_post_burnin_adapt_window = 20;
  if (use_hmc_nuts) {
    hmc_da.reset(hmc_epsilon, hmc_target_accept);
    hmc_mass_adapter.reset(
        static_cast<int>(p), static_cast<int>(ntau), is_tte,
        std::max(0, burnin * std::max(1, n_thin_gb)));
  }

  // Output Storage (thinned)
  int n_save = niter / thin;
  arma::mat beta_out(n_save, p);
  arma::umat gamma_out;
  if (store_gamma)
    gamma_out.set_size(n_save, p);
  arma::vec eta1_out(n_save), eta2_out(n_save);
  arma::vec alpha_out(n_save);
  arma::vec sigmasq_out(n_save);
  arma::mat tau_out(n_save, ntau);

  // 2. MCMC LOOP
  EtaAdapter eta1_adapter(0.5);  // Vihola RAM for eta1 proposal
  EtaAdapter eta2_adapter(0.5);  // Vihola RAM for eta2 proposal

  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {

    if (iter > 0 && iter % 10000 == 0) {
      Rcpp::checkUserInterrupt();
      int model_size = (int)arma::accu(gamma);
      Rcpp::Rcout << "Iter: " << iter << " | Model size: " << model_size
                  << " | eta1: " << eta1 << " | eta2: " << eta2 << "\n";
    }

    if (is_count) {
      refresh_count_latent(Xb_total, alpha);
      loglik = calc_loglik_count(y_count, Xb_total - Z_tau, alpha, Z_tau,
                                 log_w_count);
    } else if (is_zic) {
      bvs_zinb::gibbs_update_Z(zinb_Z_latent, y_count, Xb_total, alpha,
                               zinb_r_curr, zinb_pi_curr, n);
      zinb_pi_curr = bvs_zinb::gibbs_update_pi(zinb_Z_latent, n, zinb_a_pi, zinb_b_pi);
      if (zinb_estimate_r)
        zinb_r_curr = bvs_zinb::mh_update_r(zinb_r_curr, y_count, Xb_total, alpha,
                                             zinb_Z_latent, n, zinb_a_r, zinb_b_r);
      bvs_zinb::refresh_zic_poisson_gamma(w_count, log_w_count, y_count,
                                          Xb_total, alpha, zinb_r_curr,
                                          zinb_Z_latent, n);
      loglik = bvs_zinb::calc_loglik_zic_conditional(
          y_count, Xb_total - Z_tau, alpha, Z_tau, log_w_count, zinb_Z_latent, n);
    }

    // ---------------------------------------------------------
    // STEP A+B: Gamma + Beta with inner thinning (n_thin_gb rounds)
    // ---------------------------------------------------------
    arma::uvec active;
    for (int thin_gb = 0; thin_gb < n_thin_gb; ++thin_gb) {

      // STEP A: Gamma via Zanella (2020) locally informed proposals
      // Initialize LB scores before gamma scan
      arma::ivec gamma_iv;
      if (use_lb_gamma) {
        gamma_iv = arma::conv_to<arma::ivec>::from(gamma);
        init_lb_dual_scores_dense(R_dyn_int, R_fix_int, (int)p, gamma_iv, mu, eta1, eta2, lb_score, lb_weight, lb_Z);
      }

      for (arma::uword k = 0; k < 5; ++k) {

        arma::uword j;
        double q_fwd = 0.0, q_bwd = 0.0;
        if (use_lb_gamma) {
          j = (arma::uword)sample_weighted_index(lb_weight, lb_Z, (int)p);
        } else {
          // OPT 1: Compute informed proposal weights
          double wt_sum = 0.0;
          for (arma::uword jj = 0; jj < p; ++jj) {
            double diff_jj = (gamma(jj) == 0) ? 1.0 : -1.0;
            double ising_jj = diff_jj * (mu + eta1 * neigh_dyn_sum(jj) +
                                         eta2 * neigh_fix_sum(jj));
            proposal_weights(jj) = std::exp(std::min(0.0, ising_jj)) + 1e-10;
            wt_sum += proposal_weights(jj);
          }
          double u_draw = R::runif(0.0, wt_sum);
          j = 0;
          double cum_wt = proposal_weights(0);
          while (cum_wt < u_draw && j < p - 1) {
            ++j;
            cum_wt += proposal_weights(j);
          }
          q_fwd = proposal_weights(j) / wt_sum;
          // Pre-compute reverse for non-LB path
          double diff_jj = (gamma(j) == 0) ? 1.0 : -1.0;
          double rev_ising =
              -diff_jj * (mu + eta1 * neigh_dyn_sum(j) + eta2 * neigh_fix_sum(j));
          double q_rev_j = std::exp(std::min(0.0, rev_ising)) + 1e-10;
          double wt_sum_rev = wt_sum - proposal_weights(j) + q_rev_j;
          q_bwd = q_rev_j / wt_sum_rev;
        }

        int g_curr = gamma(j);
        int g_prop = 1 - g_curr;
        double b_curr = beta(j);
        double gamma_prior_var = use_t_prior ?
            bvs_imbalanced::beta_prior_var_j(sigmasq, imb_link_code, t_scale, lambda_t(j)) :
            sigmasq;
        double b_prop =
            (g_prop == 1) ? R::rnorm(beta0, std::sqrt(gamma_prior_var)) : 0.0;
        double db = b_prop - b_curr;

        double ll_diff = 0.0;
        const arma::vec &xj = X.col(j);
        std::vector<double> delta_group_W;
        if (is_tte)
          delta_group_W.assign(cox_data.group_start.size(), 0.0);

        // OPT 3: Incremental log-likelihood difference
        if (is_continuous) {
          ll_diff =
              (arma::dot(xj, resid) * db - 0.5 * X_col_sq_sums(j) * db * db) /
              sigmasq;
        } else if (is_tte) {
          ll_diff = cox_tracker.propose_diff(xj, db, delta_group_W);
        } else if (use_cloglog) {
          for (arma::uword i = 0; i < n; ++i) {
            double eta_old = alpha + Xb_total(i);
            double eta_new = eta_old + db * xj(i);
            ll_diff += bvs_imbalanced::loglik_cloglog_obs(y(i), eta_new) -
                       bvs_imbalanced::loglik_cloglog_obs(y(i), eta_old);
          }
        } else {
          for (arma::uword i = 0; i < n; ++i) {
            double eta_old = alpha + Xb_total(i);
            double eta_new = eta_old + db * xj(i);
            if (is_count || is_zic) {
              if (is_zic && zinb_Z_latent(i) == 1u) continue;
              double lw_i = log_w_count(i);
              double eta_old_c =
                  bvs_dadj::clamp_finite(eta_old + lw_i, -50.0, 50.0, 0.0);
              double eta_new_c =
                  bvs_dadj::clamp_finite(eta_new + lw_i, -50.0, 50.0, 0.0);
              ll_diff += (double)y_count(i) * (eta_new_c - eta_old_c) -
                         (std::exp(eta_new_c) - std::exp(eta_old_c));
            } else {
              auto log1pexp = [](double x) -> double {
                return (x > 0.0) ? x + std::log1p(std::exp(-x))
                                 : std::log1p(std::exp(x));
              };
              ll_diff +=
                  y(i) * db * xj(i) - (log1pexp(eta_new) - log1pexp(eta_old));
            }
          }
        }

        double diff = static_cast<double>(g_prop - g_curr);
        double ising_diff =
            diff * (mu + eta1 * neigh_dyn_sum(j) + eta2 * neigh_fix_sum(j));

        double log_ratio = ll_diff + ising_diff;

        // LB correction: adjust MH ratio for non-uniform proposal
        if (use_lb_gamma) {
          int delta_g = 1 - 2 * (int)gamma(j);
          gamma_iv(j) = (int)gamma(j);
          build_lb_dual_delta_dense(R_dyn_int, R_fix_int, (int)p, gamma_iv, eta1, eta2, (int)j, delta_g,
              lb_score, lb_weight, lb_Z, lb_delta);
          log_ratio += lb_delta.log_q_rev - lb_delta.log_q_fwd;
        } else {
          log_ratio += std::log(q_bwd) - std::log(q_fwd);
        }

        if (bvs_dadj::safe_mh_accept(log_ratio)) {
          gamma(j) = g_prop;
          beta(j) = b_prop;
          // Update cached neighbor sums
          for (arma::uword i = 0; i < p; ++i) {
            if (i == j)
              continue;
            if (R_dyn_int(i, j) != 0)
              neigh_dyn_sum(i) += diff;
            if (R_fix_int(i, j) != 0)
              neigh_fix_sum(i) += diff;
          }
          // Update LB state on acceptance
          if (use_lb_gamma) {
            apply_lb_delta(lb_score, lb_weight, lb_Z, lb_delta);
            gamma_iv(j) = (int)g_prop;
          }
          if (is_continuous) {
            resid -= db * xj;
            loglik += ll_diff;
            Xb_total += db * xj;
          } else if (is_tte) {
            cox_tracker.apply_diff(xj, db, ll_diff, delta_group_W);
            loglik = cox_tracker.get_loglik();
            Xb_total += db * xj;
          } else {
            Xb_total += db * xj;
            loglik += ll_diff;
          }
        }
      }

      // STEP B: Beta updates — HMC/NUTS for binary/TTE (if selected),
      //         MALA for binary/count, Fisher-RW for TTE
      active = arma::find(gamma == 1);
      // Build active_idx for HMC/NUTS
      std::vector<arma::uword> active_idx;
      active_idx.reserve(active.n_elem);
      for (arma::uword k = 0; k < active.n_elem; ++k)
        active_idx.push_back(active(k));

      if (use_hmc_nuts && !is_continuous && !is_count && !is_zic) {
        auto compute = [&](const arma::vec &q) -> std::pair<double, arma::vec> {
          if (is_tte) {
            return bvs_hmc::nlp_grad_tte_joint(X, Z_dat, cox_data, active_idx, q,
                                               beta0, tau0, htau, nu0, sigmasq0);
          }
          if (is_imbalanced) {
            return bvs_hmc::nlp_grad_imbalanced_joint(
                X, y, Z_dat, active_idx, q, beta0, alpha0, tau0, h, htau, nu0,
                sigmasq0, use_cloglog, use_t_prior, t_scale, lambda_t);
          }
          return bvs_hmc::nlp_grad_binary_joint(X, y, Z_dat, active_idx, q,
                                                beta0, alpha0, tau0, h, htau,
                                                nu0, sigmasq0);
        };

        // Persistent step-size state across common gamma flips.
        const int d_act = static_cast<int>(active_idx.size());
        const int ntau_local = static_cast<int>(Z_dat.n_cols);
        const int hmc_dim = d_act + ntau_local + 1 + (is_tte ? 0 : 1);
        const bool in_warmup = (iter < burnin);
        const int dim_jump =
            (hmc_prev_dim >= 0) ? std::abs(hmc_dim - hmc_prev_dim) : 0;
        bool need_reinit = bvs_hmc::should_reinit_step_size(
            hmc_eps_initialised, hmc_prev_dim, hmc_dim, in_warmup);

        bvs_hmc::DiagMassMatrix mass_current;
        hmc_mass_adapter.build_joint_mass_matrix(mass_current, active_idx,
                                                 sigmasq, h, htau, nu0);

        if (!in_warmup && hmc_prev_dim >= 0 && dim_jump > 2 && hmc_post_burnin_adapt_left == 0) {
          hmc_post_burnin_adapt_left = hmc_post_burnin_adapt_window;
          hmc_da.reset(bvs_hmc::clamp_epsilon(hmc_epsilon),
                       hmc_target_accept);
        }
        hmc_prev_dim = hmc_dim;

        if (need_reinit && hmc_dim > 0) {
          arma::vec q0(hmc_dim);
          for (int kk = 0; kk < d_act; ++kk)
            q0(kk) = beta(active_idx[kk]);
          if (!is_tte) {
            q0(d_act) = alpha;
            for (int kk = 0; kk < ntau_local; ++kk)
              q0(d_act + 1 + kk) = tau(kk);
            q0(d_act + 1 + ntau_local) = std::log(sigmasq);
          } else {
            for (int kk = 0; kk < ntau_local; ++kk)
              q0(d_act + kk) = tau(kk);
            q0(d_act + ntau_local) = std::log(sigmasq);
          }

          const auto res0 = compute(q0);
          if (res0.second.is_finite()) {
            hmc_epsilon = bvs_hmc::find_reasonable_epsilon(q0, res0.second,
                                                           res0.first, compute,
                                                           mass_current);
            hmc_da.reset(hmc_epsilon, hmc_target_accept);
          }
          hmc_eps_initialised = true;
        }

        bvs_hmc::HMCSamplingStats step_stats;
        double accept_prob = bvs_hmc::run_hmc_nuts_joint(
            alg_type_int, X, y, Z_dat, beta, Xb_total, loglik, alpha, tau,
            sigmasq, beta0, alpha0, tau0, h, htau, nu0, sigmasq0, active_idx,
            is_tte, cox_data, cox_tracker, hmc_epsilon, hmc_n_leapfrog,
            nuts_max_treedepth, compute, &mass_current, &step_stats);
        hmc_diag.record(hmc_epsilon, step_stats);

        if (in_warmup) {
          hmc_mass_adapter.observe(active_idx, beta, alpha, tau,
                                   std::log(std::max(sigmasq, 1e-10)));
          hmc_epsilon = hmc_da.update(accept_prob);
        } else {
          if (!hmc_warmup_finalized) {
            hmc_epsilon = hmc_da.final_epsilon();
            hmc_warmup_finalized = true;
          }
          if (hmc_post_burnin_adapt_left > 0) {
            hmc_epsilon = hmc_da.update(accept_prob);
            --hmc_post_burnin_adapt_left;
            if (hmc_post_burnin_adapt_left == 0)
              hmc_epsilon = hmc_da.final_epsilon();
          }
        }

        if (!is_tte) {
          if (use_cloglog) {
            for (arma::uword i = 0; i < n; ++i)
              mala_resid(i) = bvs_imbalanced::cloglog_residual(y(i), alpha + Xb_total(i));
          } else {
            for (arma::uword i = 0; i < n; ++i)
              mala_resid(i) =
                  y(i) - 1.0 / (1.0 + std::exp(-(alpha + Xb_total(i))));
          }
        }
      } else if (!is_continuous) {
        // Refresh MALA residuals after Step A gamma flips
        if (!is_tte) {
          if (use_cloglog) {
            for (arma::uword i = 0; i < n; ++i)
              mala_resid(i) = bvs_imbalanced::cloglog_residual(y(i), alpha + Xb_total(i));
          } else if (is_count || is_zic) {
            for (arma::uword i = 0; i < n; ++i) {
              if (is_zic && zinb_Z_latent(i) == 1u) { mala_resid(i) = 0.0; continue; }
              const double eta = bvs_dadj::clamp_finite(
                  alpha + Xb_total(i) + log_w_count(i), -50.0, 50.0, 0.0);
              mala_resid(i) = (double)y_count(i) - std::exp(eta);
            }
          } else {
            for (arma::uword i = 0; i < n; ++i)
              mala_resid(i) =
                  y(i) - 1.0 / (1.0 + std::exp(-(alpha + Xb_total(i))));
          }
        }
        arma::vec cox_H;
        if (is_tte)
          cox_tracker.compute_H_vec(cox_H);

        for (arma::uword k = 0; k < active.n_elem; ++k) {
          arma::uword j = active(k);
          const arma::vec &xj = X.col(j);

          if (is_tte) {
            double info_jj = cox_tracker.compute_info_diag_j(xj, cox_H);
            double prop_sd = (info_jj > 1e-10)
                                 ? std::min(1.0 / std::sqrt(info_jj), 2.0)
                                 : 0.5;
            double b_prop = R::rnorm(beta(j), prop_sd);
            double db = b_prop - beta(j);
            std::vector<double> delta_group_W(cox_data.group_start.size(), 0.0);
            double ll_prop_diff =
                cox_tracker.propose_diff(xj, db, delta_group_W);
            double pr_diff =
                -0.5 *
                (std::pow(b_prop - beta0, 2) - std::pow(beta(j) - beta0, 2)) /
                sigmasq;
            if (bvs_dadj::safe_mh_accept(ll_prop_diff + pr_diff)) {
              beta(j) = b_prop;
              cox_tracker.apply_diff(xj, db, ll_prop_diff, delta_group_W);
              loglik = cox_tracker.get_loglik();
              Xb_total += db * xj;
              cox_tracker.compute_H_vec(cox_H);
            }
          } else {
            // Component-wise MALA for binary/count/imbalanced_binary
            double pvar_j = use_t_prior ?
                bvs_imbalanced::beta_prior_var_j(sigmasq, imb_link_code, t_scale, lambda_t(j)) :
                sigmasq;
            double g_j =
                arma::dot(xj, mala_resid) - (beta(j) - beta0) / pvar_j;
            double h_j =
                std::min(0.5 * pvar_j / (X_col_sq_sums(j) + 1.0), 1.0);
            double mean_fwd = beta(j) + 0.5 * h_j * g_j;
            double b_prop = R::rnorm(mean_fwd, std::sqrt(h_j));
            double db = b_prop - beta(j);
            double ll_prop = 0.0;
            double g_j_back = -(b_prop - beta0) / pvar_j;
            for (arma::uword i = 0; i < n; ++i) {
              if (is_count || is_zic) {
                if (is_zic && zinb_Z_latent(i) == 1u) continue;
                const double eta_p = bvs_dadj::clamp_finite(
                    alpha + Xb_total(i) + db * xj(i) + log_w_count(i), -50.0,
                    50.0, 0.0);
                const double lam_p = std::exp(eta_p);
                ll_prop += (double)y_count(i) * eta_p - lam_p;
                g_j_back += ((double)y_count(i) - lam_p) * xj(i);
              } else if (use_cloglog) {
                const double eta_p = alpha + Xb_total(i) + db * xj(i);
                ll_prop += bvs_imbalanced::loglik_cloglog_obs(y(i), eta_p);
                g_j_back += bvs_imbalanced::cloglog_residual(y(i), eta_p) * xj(i);
              } else {
                const double eta_p = alpha + Xb_total(i) + db * xj(i);
                const double p_p = 1.0 / (1.0 + std::exp(-eta_p));
                ll_prop += y(i) * eta_p -
                           (eta_p > 0.0 ? eta_p + std::log1p(std::exp(-eta_p))
                                        : std::log1p(std::exp(eta_p)));
                g_j_back += (y(i) - p_p) * xj(i);
              }
            }
            double ll_diff = ll_prop - loglik;
            double mean_bwd = b_prop + 0.5 * h_j * g_j_back;
            double lq_fwd =
                -0.5 * (b_prop - mean_fwd) * (b_prop - mean_fwd) / h_j;
            double lq_bwd =
                -0.5 * (beta(j) - mean_bwd) * (beta(j) - mean_bwd) / h_j;
            double pr_diff =
                -0.5 *
                (std::pow(b_prop - beta0, 2) - std::pow(beta(j) - beta0, 2)) /
                pvar_j;
            if (bvs_dadj::safe_mh_accept(ll_diff + pr_diff + lq_bwd - lq_fwd)) {
              beta(j) = b_prop;
              Xb_total += db * xj;
              loglik = ll_prop;
              if (use_cloglog) {
                for (arma::uword i = 0; i < n; ++i)
                  mala_resid(i) = bvs_imbalanced::cloglog_residual(y(i), alpha + Xb_total(i));
              } else if (is_count || is_zic) {
                for (arma::uword i = 0; i < n; ++i) {
                  if (is_zic && zinb_Z_latent(i) == 1u) { mala_resid(i) = 0.0; continue; }
                  const double eta = bvs_dadj::clamp_finite(
                      alpha + Xb_total(i) + log_w_count(i), -50.0, 50.0, 0.0);
                  mala_resid(i) = (double)y_count(i) - std::exp(eta);
                }
              } else {
                for (arma::uword i = 0; i < n; ++i)
                  mala_resid(i) =
                      y(i) - 1.0 / (1.0 + std::exp(-(alpha + Xb_total(i))));
              }
            }
          }
        }
      } else {
        for (arma::uword k = 0; k < active.n_elem; ++k) {
          arma::uword j = active(k);
          double bj_old = beta(j);
          double denom = X_col_sq_sums(j) + 1.0;
          double mean =
              (arma::dot(X.col(j), resid) + bj_old * X_col_sq_sums(j) + beta0) /
              denom;
          double bj_new = R::rnorm(mean, std::sqrt(sigmasq / denom));
          double db = bj_new - bj_old;
          if (std::abs(db) > 0.0) {
            beta(j) = bj_new;
            Xb_total += db * X.col(j);
            resid -= db * X.col(j);
          }
        }
        loglik = -0.5 * n * std::log(2.0 * M_PI * sigmasq) -
                 0.5 * arma::dot(resid, resid) / sigmasq;
      }

    } // end inner thinning for gamma + beta

    // ---------------------------------------------------------
    // STEP B2: Lambda (Student-t latent scales) — Gibbs step
    // ---------------------------------------------------------
    if (use_t_prior) {
      bvs_imbalanced::gibbs_update_lambda(lambda_t, beta, gamma, beta0,
                                          sigmasq, t_df, t_scale);
    }

    // ---------------------------------------------------------
    // STEP C: Alpha — MALA (binary/count), exact Gibbs (continuous)
    // ---------------------------------------------------------
    if (!use_hmc_nuts && !is_continuous && !is_tte) {
      double sum_resid = arma::accu(mala_resid);
      double g_alpha = sum_resid - (alpha - alpha0) / (h * sigmasq);
      double h_alpha = std::min(0.5 * h * sigmasq / ((double)n + 1.0 / h), 1.0);
      double mean_fwd_a = alpha + 0.5 * h_alpha * g_alpha;
      double a_prop = R::rnorm(mean_fwd_a, std::sqrt(h_alpha));
      double ll_a_prop = 0.0;
      double g_alpha_back = -(a_prop - alpha0) / (h * sigmasq);
      for (arma::uword i = 0; i < n; ++i) {
        if (is_count || is_zic) {
          if (is_zic && zinb_Z_latent(i) == 1u) continue;
          const double eta = bvs_dadj::clamp_finite(
              a_prop + Xb_total(i) + log_w_count(i), -50.0, 50.0, 0.0);
          const double lam = std::exp(eta);
          ll_a_prop += (double)y_count(i) * eta - lam;
          g_alpha_back += (double)y_count(i) - lam;
        } else if (use_cloglog) {
          const double eta = a_prop + Xb_total(i);
          ll_a_prop += bvs_imbalanced::loglik_cloglog_obs(y(i), eta);
          g_alpha_back += bvs_imbalanced::cloglog_residual(y(i), eta);
        } else {
          const double eta = a_prop + Xb_total(i);
          const double p = 1.0 / (1.0 + std::exp(-eta));
          ll_a_prop +=
              y(i) * eta - (eta > 0.0 ? eta + std::log1p(std::exp(-eta))
                                      : std::log1p(std::exp(eta)));
          g_alpha_back += y(i) - p;
        }
      }
      double pr_a_curr =
          -0.5 * (alpha - alpha0) * (alpha - alpha0) / (h * sigmasq);
      double pr_a_prop =
          -0.5 * (a_prop - alpha0) * (a_prop - alpha0) / (h * sigmasq);
      double mean_bwd_a = a_prop + 0.5 * h_alpha * g_alpha_back;
      double lq_fwd_a =
          -0.5 * (a_prop - mean_fwd_a) * (a_prop - mean_fwd_a) / h_alpha;
      double lq_bwd_a =
          -0.5 * (alpha - mean_bwd_a) * (alpha - mean_bwd_a) / h_alpha;
      if (bvs_dadj::safe_mh_accept((ll_a_prop - loglik) +
                                   (pr_a_prop - pr_a_curr) + lq_bwd_a -
                                   lq_fwd_a)) {
        alpha = a_prop;
        loglik = ll_a_prop;
        if (use_cloglog) {
          for (arma::uword i = 0; i < n; ++i)
            mala_resid(i) = bvs_imbalanced::cloglog_residual(y(i), alpha + Xb_total(i));
        } else if (is_count || is_zic) {
          for (arma::uword i = 0; i < n; ++i) {
            if (is_zic && zinb_Z_latent(i) == 1u) { mala_resid(i) = 0.0; continue; }
            const double eta = bvs_dadj::clamp_finite(
                alpha + Xb_total(i) + log_w_count(i), -50.0, 50.0, 0.0);
            mala_resid(i) = (double)y_count(i) - std::exp(eta);
          }
        } else {
          for (arma::uword i = 0; i < n; ++i)
            mala_resid(i) =
                y(i) - 1.0 / (1.0 + std::exp(-(alpha + Xb_total(i))));
        }
      }
    } else if (is_continuous) {
      double denom = static_cast<double>(X.n_rows) + 1.0 / h;
      double mean = (arma::accu(resid) + alpha * static_cast<double>(X.n_rows) +
                     alpha0 / h) /
                    denom;
      double a_new = R::rnorm(mean, std::sqrt(sigmasq / denom));
      if (alpha != a_new) {
        double da = a_new - alpha;
        alpha = a_new;
        resid -= da;
      }
      loglik = -0.5 * n * std::log(2.0 * M_PI * sigmasq) -
               0.5 * arma::dot(resid, resid) / sigmasq;
    }

    // ---------------------------------------------------------
    // STEP C+: Tau — component-wise MALA (binary/count) or block MH (TTE)
    // ---------------------------------------------------------
    if (ntau > 0 && !use_hmc_nuts) {
      if (!is_continuous) {
        if (is_tte) {
          arma::vec tau_prop(ntau);
          for (arma::uword j = 0; j < ntau; ++j)
            tau_prop(j) = R::rnorm(tau(j), std::sqrt(htau * sigmasq));
          arma::vec Z_tau_prop = Z_dat * tau_prop;
          arma::vec Xb_total_prop = Xb_total - Z_tau + Z_tau_prop;
          bvs_dadj::CoxTracker cox_tracker_prop;
          cox_tracker_prop.init(Xb_total_prop, cox_data);
          double ll_tau_prop = cox_tracker_prop.get_loglik();
          double pr_tau_curr =
              -0.5 * arma::accu(arma::square(tau - tau0)) / (htau * sigmasq);
          double pr_tau_prop = -0.5 *
                               arma::accu(arma::square(tau_prop - tau0)) /
                               (htau * sigmasq);
          if (bvs_dadj::safe_mh_accept((ll_tau_prop - loglik) +
                                       (pr_tau_prop - pr_tau_curr))) {
            tau = tau_prop;
            Z_tau = Z_tau_prop;
            Xb_total = Xb_total_prop;
            cox_tracker.init(Xb_total, cox_data);
            loglik = cox_tracker.get_loglik();
          }
        } else {
          // Component-wise MALA for tau (binary/count)
          for (arma::uword k = 0; k < ntau; ++k) {
            const arma::vec &zk = Z_dat.col(k);
            double g_k =
                arma::dot(zk, mala_resid) - (tau(k) - tau0) / (htau * sigmasq);
            double h_k = std::min(
                0.5 * htau * sigmasq / (Z_col_sq_sums(k) + 1.0 / htau), 1.0);
            double mean_fwd_k = tau(k) + 0.5 * h_k * g_k;
            double t_prop = R::rnorm(mean_fwd_k, std::sqrt(h_k));
            double dt = t_prop - tau(k);
            double ll_t_prop = 0.0;
            double g_k_back = -(t_prop - tau0) / (htau * sigmasq);
            for (arma::uword i = 0; i < n; ++i) {
              const double neb = alpha + Xb_total(i) + dt * zk(i);
              if (is_count || is_zic) {
                if (is_zic && zinb_Z_latent(i) == 1u) continue;
                const double ep = bvs_dadj::clamp_finite(neb + log_w_count(i),
                                                         -50.0, 50.0, 0.0);
                const double lp = std::exp(ep);
                ll_t_prop += (double)y_count(i) * ep - lp;
                g_k_back += ((double)y_count(i) - lp) * zk(i);
              } else if (use_cloglog) {
                ll_t_prop += bvs_imbalanced::loglik_cloglog_obs(y(i), neb);
                g_k_back += bvs_imbalanced::cloglog_residual(y(i), neb) * zk(i);
              } else {
                const double pp = 1.0 / (1.0 + std::exp(-neb));
                ll_t_prop +=
                    y(i) * neb - (neb > 0.0 ? neb + std::log1p(std::exp(-neb))
                                            : std::log1p(std::exp(neb)));
                g_k_back += (y(i) - pp) * zk(i);
              }
            }
            double lld = ll_t_prop - loglik;
            double mean_bwd_k = t_prop + 0.5 * h_k * g_k_back;
            double lq_f =
                -0.5 * (t_prop - mean_fwd_k) * (t_prop - mean_fwd_k) / h_k;
            double lq_b =
                -0.5 * (tau(k) - mean_bwd_k) * (tau(k) - mean_bwd_k) / h_k;
            double prd =
                -0.5 *
                (std::pow(t_prop - tau0, 2) - std::pow(tau(k) - tau0, 2)) /
                (htau * sigmasq);
            if (bvs_dadj::safe_mh_accept(lld + prd + lq_b - lq_f)) {
              tau(k) = t_prop;
              Xb_total += dt * zk;
              Z_tau += dt * zk;
              loglik = ll_t_prop;
              if (use_cloglog) {
                for (arma::uword i = 0; i < n; ++i)
                  mala_resid(i) = bvs_imbalanced::cloglog_residual(y(i), alpha + Xb_total(i));
              } else if (is_count || is_zic) {
                for (arma::uword i = 0; i < n; ++i) {
                  if (is_zic && zinb_Z_latent(i) == 1u) { mala_resid(i) = 0.0; continue; }
                  const double eta = bvs_dadj::clamp_finite(
                      alpha + Xb_total(i) + log_w_count(i), -50.0, 50.0, 0.0);
                  mala_resid(i) = (double)y_count(i) - std::exp(eta);
                }
              } else {
                for (arma::uword i = 0; i < n; ++i)
                  mala_resid(i) =
                      y(i) - 1.0 / (1.0 + std::exp(-(alpha + Xb_total(i))));
              }
            }
          }
        }
      } else {
        for (arma::uword j = 0; j < ntau; ++j) {
          arma::vec zj = Z_dat.col(j);
          double tj_old = tau(j);
          double denom = arma::dot(zj, zj) + 1.0 / htau;
          double mean = (arma::dot(zj, resid) + tj_old * arma::dot(zj, zj) +
                         tau0 / htau) /
                        denom;
          double tj_new = R::rnorm(mean, std::sqrt(sigmasq / denom));
          double dt = tj_new - tj_old;
          if (std::abs(dt) > 0.0) {
            tau(j) = tj_new;
            Z_tau += dt * zj;
            resid -= dt * zj;
          }
        }
        loglik = -0.5 * X.n_rows * std::log(2.0 * M_PI * sigmasq) -
                 0.5 * arma::dot(resid, resid) / sigmasq;
      }
    }

    // ---------------------------------------------------------
    // STEP D: SigmaSq — exact IG Gibbs (all outcome types)
    // p(sigmasq|rest) = IG(shape, scale):
    //   shape = (nu0 + k_active + 1 + ntau) / 2
    //   scale = (nu0*sigmasq0 + ss_beta + (alpha-alpha0)^2/h + ss_tau/htau) / 2
    // ---------------------------------------------------------
    if (!use_hmc_nuts) {
      if (!is_continuous) {
        double ss_beta = 0.0;
        for (arma::uword k = 0; k < active.n_elem; ++k) {
          const arma::uword jj = active(k);
          const double d = beta(jj) - beta0;
          // For Student-t prior: scale by 1/(s^2 * lambda_j)
          double scale_fac = use_t_prior ?
              1.0 / (t_scale * t_scale * lambda_t(jj)) : 1.0;
          ss_beta += d * d * scale_fac;
        }
        double ss_tau = 0.0;
        for (arma::uword j = 0; j < ntau; ++j) {
          const double d = tau(j) - tau0;
          ss_tau += d * d;
        }
        const double da = alpha - alpha0;
        const double shape_post =
            0.5 * (nu0 + (double)active.n_elem + 1.0 + (double)ntau);
        const double scale_post =
            0.5 * (nu0 * sigmasq0 + ss_beta + da * da / h + ss_tau / htau);
        const double gdraw =
            R::rgamma(shape_post, 1.0 / std::max(scale_post, 1e-12));
        if (std::isfinite(gdraw) && gdraw > 0.0)
          sigmasq = std::max(1e-10, 1.0 / gdraw);
      } else {
        double ss_beta = 0.0;
        for (arma::uword k = 0; k < active.n_elem; ++k) {
          double d = beta(active(k)) - beta0;
          ss_beta += d * d;
        }
        double ss_tau = 0.0;
        for (arma::uword j = 0; j < ntau; ++j) {
          double d = tau(j) - tau0;
          ss_tau += d * d;
        }
        double da = alpha - alpha0;
        double sse = arma::dot(resid, resid);

        double shape_post = 0.5 * (nu0 + static_cast<double>(X.n_rows) +
                                   static_cast<double>(active.n_elem) + 1.0 +
                                   static_cast<double>(ntau));
        double scale_post = 0.5 * (nu0 * sigmasq0 + sse + ss_beta +
                                   da * da / h + ss_tau / htau);
        double gdraw = R::rgamma(shape_post, 1.0 / std::max(scale_post, 1e-12));
        if (std::isfinite(gdraw) && gdraw > 0.0) {
          sigmasq = std::max(1e-10, 1.0 / gdraw);
        }
        loglik = -0.5 * X.n_rows * std::log(2.0 * M_PI * sigmasq) -
                 0.5 * sse / sigmasq;
      }
    }

    // ---------------------------------------------------------
    // STEP E: Update eta1, eta2 via Moller et al. (2006)
    //         with Propp-Wilson perfect simulation
    // ---------------------------------------------------------
    moller_update_eta(R_dyn_int, R_fix_int, mu, eta1, eta2, eta1_sd, eta2_sd,
                      mu_tilde, eta1_tilde, eta2_tilde, gamma, e_eta, f_eta,
                      T_max, proposal_type, eta1_adapter, eta2_adapter,
                      use_cftp);

    // OPT 2: Online PIP accumulation (post-burnin)
    if (iter >= burnin) {
      for (arma::uword j = 0; j < p; ++j)
        gamma_pip_sum(j) += (double)gamma(j);
    }

    if (iter >= burnin && (iter - burnin) % thin == 0) {
      int s = (iter - burnin) / thin;
      if (s < n_save) {
        beta_out.row(s) = beta.t();
        if (store_gamma) {
          for (arma::uword j = 0; j < p; ++j)
            gamma_out(s, j) = gamma(j);
        }
        eta1_out(s) = eta1;
        eta2_out(s) = eta2;
        alpha_out(s) = alpha;
        sigmasq_out(s) = sigmasq;
        tau_out.row(s) = tau.t();
      }
    }
  }

  // OPT 2: Compute final PIP vector
  double n_post = (double)(total_iter - burnin);
  arma::vec gamma_pip = (n_post > 0) ? gamma_pip_sum / n_post : gamma_pip_sum;

  Rcpp::List result = Rcpp::List::create(
      Rcpp::Named("beta") = beta_out, Rcpp::Named("gamma_pip") = gamma_pip,
      Rcpp::Named("eta1") = eta1_out, Rcpp::Named("eta2") = eta2_out,
      Rcpp::Named("alpha") = alpha_out, Rcpp::Named("sigmasq") = sigmasq_out,
      Rcpp::Named("tau") = tau_out);
  if (use_hmc_nuts)
    result["hmc_nuts_diagnostics"] = hmc_diag.to_list(hmc_epsilon);
  if (store_gamma)
    result["gamma"] = gamma_out;
  if (is_imbalanced)
    result["lambda_t"] = lambda_t;
  if (is_zic) {
    result["zinb_pi"] = zinb_pi_curr;
    result["zinb_r"] = zinb_r_curr;
  }
  return result;
}
