// [[Rcpp::depends(RcppArmadillo)]]
#include "BVS_HMC_NUTS.h"
#include "BayesLogit_Numerics.h"
#include "BayesLogit_ZINB.h"
#include "BayesLogit_Sparse_Helpers.h"
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace {

// =============================================================================
// SECTION 1: HELPER FUNCTIONS
// =============================================================================

// Efficient vector/matrix subsetting for GGM column-wise sampler
static inline void vec_remove_idx(const arma::vec &src, arma::vec &dest,
                                  arma::uword k, arma::uword p) {
  if (k > 0)
    dest.head(k) = src.head(k);
  if (k < p - 1)
    dest.tail(p - 1 - k) = src.tail(p - 1 - k);
}

static inline void mat_remove_rowcol(const arma::mat &src, arma::mat &dest,
                                     arma::uword k, arma::uword p) {
  if (k > 0)
    dest.submat(0, 0, k - 1, k - 1) = src.submat(0, 0, k - 1, k - 1);
  if (k > 0 && k < p - 1)
    dest.submat(0, k, k - 1, p - 2) = src.submat(0, k + 1, k - 1, p - 1);
  if (k < p - 1 && k > 0)
    dest.submat(k, 0, p - 2, k - 1) = src.submat(k + 1, 0, p - 1, k - 1);
  if (k < p - 1)
    dest.submat(k, k, p - 2, p - 2) = src.submat(k + 1, k + 1, p - 1, p - 1);
}

static inline void mat_insert_rowcol(arma::mat &src, const arma::mat &dest,
                                     arma::uword k, arma::uword p) {
  if (k > 0)
    src.submat(0, 0, k - 1, k - 1) = dest.submat(0, 0, k - 1, k - 1);
  if (k > 0 && k < p - 1)
    src.submat(0, k + 1, k - 1, p - 1) = dest.submat(0, k, k - 1, p - 2);
  if (k < p - 1 && k > 0)
    src.submat(k + 1, 0, p - 1, k - 1) = dest.submat(k, 0, p - 2, k - 1);
  if (k < p - 1)
    src.submat(k + 1, k + 1, p - 1, p - 1) = dest.submat(k, k, p - 2, p - 2);
}

static inline arma::vec solve_spd_chol_upper(const arma::mat &U,
                                             const arma::vec &b) {
  arma::vec y = arma::solve(arma::trimatl(U.t()), b, arma::solve_opts::fast);
  arma::vec x = arma::solve(arma::trimatu(U), y, arma::solve_opts::fast);
  return x;
}

static inline void rnorm_into(arma::vec &out) {
  for (arma::uword i = 0; i < out.n_elem; ++i)
    out(i) = R::rnorm(0.0, 1.0);
}

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
// GIBBS SAMPLING FOR AUXILIARY OMEGA (Single Adjacency)
// Default method (use_cftp=false): 20 full Gibbs sweeps from random init.
// OPT: accepts uint8 Z_ggm directly — no R_dyn_int copy needed
// OPT: precomputes sparse neighbor lists from Z to avoid O(p^2) inner loops
// =============================================================================
static Rcpp::IntegerVector gibbs_omega(const arma::Mat<uint8_t> &Z,
                                       double mu, double eta1,
                                       int n_gibbs_sweeps = 20) {
  const unsigned int p = Z.n_cols;

  // Precompute sparse neighbor lists from dense Z (built once per call)
  std::vector<std::vector<unsigned int>> nbr(p);
  for (unsigned int k = 0; k < p; ++k) {
    for (unsigned int j = 0; j < p; ++j) {
      if (j != k && Z(k, j) != 0) nbr[k].push_back(j);
    }
  }

  Rcpp::IntegerVector x(p, 0);
  int seed_val = static_cast<int>(std::floor(R::runif(0.0, 1.0) * 2147483646.0)) + 1;
  std::mt19937 gf(seed_val);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  double base_prob = 1.0 / (1.0 + std::exp(-mu));
  for (unsigned int k = 0; k < p; ++k) {
    x[k] = (u01(gf) < base_prob) ? 1 : 0;
  }
  for (int sw = 0; sw < n_gibbs_sweeps; ++sw) {
    for (unsigned int k = 0; k < p; ++k) {
      double ker = 0.0;
      for (unsigned int j : nbr[k]) {
        ker += eta1 * x[j];
      }
      x[k] = (u01(gf) < 1.0 / (1.0 + std::exp(-(mu + ker)))) ? 1 : 0;
    }
  }
  return x;
}

// =============================================================================
// PROPP-WILSON PERFECT SIMULATION (Single Adjacency)
// OPT: accepts uint8 Z_ggm directly — no R_dyn_int copy needed
// OPT: skips j==i in kernel (diagonal is always 0)
// =============================================================================
static Rcpp::IntegerVector proppwilson_omega(const arma::Mat<uint8_t> &Z,
                                             double mu, double eta1,
                                             unsigned int T_max,
                                             bool *coalesced = nullptr) {
  const unsigned int p = Z.n_cols;
  unsigned int T = 2;

  Rcpp::IntegerVector x_up(p, 0);
  Rcpp::IntegerVector x_down(p, 1);
  Rcpp::NumericVector pi_up(p), pi_down(p);

  int seed_base = static_cast<int>(std::floor(R::runif(0.0, 1.0) * 2147483646.0)) + 1;

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
        for (unsigned int j = 0; j < p; ++j) {
          if (j == i)
            continue; // OPT: skip self
          if (Z(i, j) == 0)
            continue; // OPT: skip zero entries
          ker_up += eta1 * x_up[j];
          ker_down += eta1 * x_down[j];
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
      std::vector<unsigned int> diff_idx;
      for (unsigned int k = 0; k < p; ++k) {
        if (x_up[k] != x_down[k])
          diff_idx.push_back(k);
      }
      std::mt19937 gen_fb(seed_base + 99999);
      std::uniform_real_distribution<double> unif01_fb(0.0, 1.0);
      for (int sweep = 0; sweep < 100; ++sweep) {
        for (unsigned int idx = 0; idx < diff_idx.size(); ++idx) {
          unsigned int m = diff_idx[idx];
          double ker = 0.0;
          for (unsigned int k = 0; k < p; ++k) {
            if (k == m)
              continue; // OPT: skip self
            if (Z(m, k) == 0)
              continue; // OPT: skip zero entries
            ker += eta1 * x_up[k];
          }
          double prob = 1.0 / (1.0 + std::exp(-(mu + ker)));
          x_up[m] = (unif01_fb(gen_fb) < prob) ? 1 : 0;
        }
      }
      if (coalesced) *coalesced = false; // IMP-2: Signal CFTP failure
      for (unsigned int k = 0; k < p; ++k)
        x_down[k] = x_up[k];
    }
  }
  if (coalesced) *coalesced = true; // IMP-2: Exact coalescence
  return x_up;
}

// =============================================================================
// MOLLER ET AL. (2006) MH UPDATE FOR SINGLE eta1
// OPT: accepts uint8 Z_ggm directly — no R_dyn_int copy needed
// =============================================================================
static void moller_update_eta(const arma::Mat<uint8_t> &Z, double mu,
                              double &eta1, double eta1_sd, double mu_tilde,
                              double eta1_tilde, const arma::uvec &gamma,
                              double e_eta, double f_eta, unsigned int T_max,
                              int proposal_type, EtaAdapter &adapter1,
                              bool use_cftp = false) {
  // Exactly cancel the Moller effect to implement the Exchange Algorithm
  mu_tilde = mu;
  eta1_tilde = eta1;

  const unsigned int p = Z.n_cols;
  double eta1_new;
  double log_prop_ratio_eta1 = 0.0;

  // Logit-transformed proposal with Vihola RAM adaptation
  double eta1_safe = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1));
  double phi = std::log(eta1_safe / (eta1_sd - eta1_safe));
  double phi_new = R::rnorm(phi, adapter1.sigma());
  eta1_new = eta1_sd / (1.0 + std::exp(-phi_new));
  eta1_new = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1_new));
  log_prop_ratio_eta1 = std::log(eta1_new) + std::log(eta1_sd - eta1_new)
                      - std::log(eta1_safe) - std::log(eta1_sd - eta1_safe);

  // Generate auxiliary variables for Exchange Algorithm
  Rcpp::IntegerVector omega_eta1, omega_eta1_new;
  if (use_cftp) {
    // CFTP (Propp-Wilson) with Gibbs fallback — optional
    bool c1 = true, c2 = true;
    omega_eta1 = proppwilson_omega(Z, mu, eta1, T_max, &c1);
    omega_eta1_new = proppwilson_omega(Z, mu, eta1_new, T_max, &c2);
    if (!c1 || !c2) { adapter1.update(0.0); return; } // Reject: preserve chain exactness
  } else {
    // Standard Gibbs sampling (default) — no CFTP overhead
    omega_eta1 = gibbs_omega(Z, mu, eta1, 20);
    omega_eta1_new = gibbs_omega(Z, mu, eta1_new, 20);
  }

  // Compute sufficient statistics
  double B_R1 = 0.0;
  double A_om1_R1 = 0.0, A_om1n_R1 = 0.0;
  double sum_om1 = 0.0, sum_om1n = 0.0;

  for (unsigned int j = 0; j < p; ++j) {
    sum_om1 += omega_eta1[j];
    sum_om1n += omega_eta1_new[j];
    for (unsigned int k = j + 1; k < p; ++k) {
      if (Z(j, k) == 0)
        continue;
      B_R1 += (double)gamma(j) * (double)gamma(k);
      A_om1_R1 += (double)omega_eta1[j] * (double)omega_eta1[k];
      A_om1n_R1 += (double)omega_eta1_new[j] * (double)omega_eta1_new[k];
    }
  }

  // Prior ratio (scaled Beta)
  double log_prior_eta1 = log_beta_pdf(eta1_new / eta1_sd, e_eta, f_eta) -
                          log_beta_pdf(eta1 / eta1_sd, e_eta, f_eta);

  // Log MH ratio = log_target + log_aux + log_norm + log_prop
  double log_target_eta1 = (eta1_new - eta1) * B_R1 + log_prior_eta1;
  double log_aux_eta1 =
      mu_tilde * (sum_om1n - sum_om1) + eta1_tilde * (A_om1n_R1 - A_om1_R1);
  double log_norm_eta1 =
      mu * (sum_om1 - sum_om1n) + eta1 * A_om1_R1 - eta1_new * A_om1n_R1;
  double log_MH_eta1 =
      log_target_eta1 + log_aux_eta1 + log_norm_eta1 + log_prop_ratio_eta1;

  double accept_prob = std::min(1.0, std::exp(std::min(0.0, log_MH_eta1)));
  if (bvs_dadj::safe_mh_accept(log_MH_eta1))
    eta1 = eta1_new;
  adapter1.update(accept_prob);
}

} // anonymous namespace
// =============================================================================
// MAIN FUNCTION: Single-Network GGM Bayesian Variable Selection
// Uses only ONE adjacency source (GGM-estimated from Wang et al. 2012)
// =============================================================================

// [[Rcpp::export]]
Rcpp::List BayesLogit_SingleNet_GGM(
    const arma::mat &X, const arma::vec &y, const arma::mat &S_ggm,
    double n_ggm, int niter, int burnin, double mu, double nu0, double sigmasq0,
    double alpha0, double beta0, double h, double e_eta, double f_eta,
    double v0_ggm, double v1_ggm, double pii_ggm, double lambda_ggm,
    double eta1_sd, double mu_tilde, double eta1_tilde, unsigned int T_max,
    int proposal_type, int thin = 1, int n_thin_gb = 3,
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
  const bool is_zic = bvs_dadj::outcome_is_zic(outcome_type);
  const int imb_link_code = is_imbalanced ?
      bvs_imbalanced::parse_imbalanced_link(imbalanced_link) : -1;
  const bool use_cloglog = (imb_link_code == 1);
  const bool use_t_prior = is_imbalanced && (imb_link_code == 0);

  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  if (S_ggm.n_cols != p)
    Rcpp::stop("GGM S matrix dim mismatch");
  if (thin < 1)
    thin = 1;
  if (n_thin_gb < 1)
    n_thin_gb = 1;

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

  // 1. INITIALIZATION
  arma::vec beta(p, arma::fill::zeros);
  arma::uvec gamma(p, arma::fill::zeros);
  if (beta_in.isNotNull())
    beta = Rcpp::as<arma::vec>(beta_in);
  if (gamma_in.isNotNull())
    gamma = Rcpp::as<arma::uvec>(gamma_in);

  double alpha = alpha_in;
  double sigmasq = 1.0;
  double eta1 = eta1_sd * 0.5; // start at midpoint of support
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

  arma::vec Xb = X * beta;
  arma::vec Xb_total = Xb + Z_tau;
  double loglik = 0.0;
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
    loglik = calc_loglik_continuous(y, Xb_total - Z_tau, alpha, Z_tau, sigmasq);
  } else if (is_tte) {
    cox_tracker.init(Xb_total, cox_data);
    loglik = cox_tracker.get_loglik();
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
  } else if (is_count) {
    refresh_count_latent(Xb_total, alpha);
    loglik =
        calc_loglik_count(y_count, Xb_total - Z_tau, alpha, Z_tau, log_w_count);
  } else if (use_cloglog) {
    loglik = bvs_imbalanced::calc_loglik_cloglog_full(y, Xb_total, alpha);
  } else {
    loglik = calc_loglik_binary(y, Xb_total - Z_tau, alpha, Z_tau);
  }
  // The original `eval_non_gaussian_loglik` lambda and its usage are removed.

  // GGM Init — Z_ggm is the ONLY adjacency storage (uint8, 1 byte/elem)
  // OPT: removed redundant R_dyn (arma::mat, 8 bytes/elem) and
  //       R_dyn_int (Rcpp::IntegerMatrix, 4 bytes/elem)
  arma::mat Sig_ggm = arma::diagmat(S_ggm.diag());
  arma::Mat<uint8_t> Z_ggm(p, p, arma::fill::zeros);

  // GGM Buffers (pre-allocated)
  arma::mat C_ggm(p, p), invC11(p - 1, p - 1), Ci(p - 1, p - 1),
      U_ggm(p - 1, p - 1);
  arma::vec Sig12(p - 1), S_i(p - 1), tau_temp(p - 1);
  arma::vec mu_i(p - 1), b_ggm(p - 1), eps_ggm(p - 1), invC11beta(p - 1),
      Sig12_new(p - 1);
  arma::vec v0_vec(p - 1), v1_vec(p - 1), w1(p - 1), w2(p - 1), w_ggm(p - 1);

  v0_vec.fill(v0_ggm);
  v1_vec.fill(v1_ggm);
  double log_pii_ggm = std::log(pii_ggm);
  double log_1_pii_ggm = std::log(1.0 - pii_ggm);
  double a_gam = 0.5 * n_ggm + 1.0;

  // Pre-compute column squared norms for MALA step sizes (all outcome types)
  arma::vec X_col_sq_sums(p);
  for (arma::uword j = 0; j < p; ++j)
    X_col_sq_sums(j) = arma::dot(X.col(j), X.col(j));
  arma::vec Z_col_sq_sums(ntau, arma::fill::zeros);
  if (ntau > 0)
    for (arma::uword j = 0; j < ntau; ++j)
      Z_col_sq_sums(j) = arma::dot(Z_dat.col(j), Z_dat.col(j));
  // MALA residual vector
  arma::vec mala_resid(n, arma::fill::zeros);

  // OPT: Pre-allocate active indices buffer (avoids arma::find allocation)
  std::vector<arma::uword> active_idx;
  active_idx.reserve(p);

  // OPT 1: Pre-allocate informed proposal weight buffers
  arma::vec proposal_weights(p);

  // --- Locally-balanced gamma proposal state (Zanella 2020) ---
  // R2-FIX: removed unused Rcpp::IntegerMatrix Z_ggm_int(p, p) snapshot;
  // the LB helpers now accept Z_ggm (uint8) directly so no copy is needed.
  std::vector<double> lb_score, lb_weight;
  double lb_Z = 0.0;
  LBProposalDelta lb_delta;

  // OPT 2: Online PIP accumulation
  arma::vec gamma_pip_sum(p, arma::fill::zeros);

  // OPT: Pre-allocate sparse edge storage (reused every saved iteration)
  std::vector<int> edge_rows, edge_cols;
  const arma::uword max_edges = p * (p - 1) / 2;
  edge_rows.reserve(std::min(max_edges, (arma::uword)1000));
  edge_cols.reserve(std::min(max_edges, (arma::uword)1000));

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
  arma::vec eta1_out(n_save);
  arma::vec alpha_out(n_save);
  arma::vec sigmasq_out(n_save);
  arma::mat tau_out(n_save, ntau);

  // Sparse Z storage
  Rcpp::List Z_list(n_save);
  arma::mat Z_pip(p, p, arma::fill::zeros);

  // 2. MCMC LOOP
  EtaAdapter eta1_adapter(0.5);  // Vihola RAM for eta1 proposal

  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {

    if (iter > 0 && iter % 10000 == 0) {
      Rcpp::checkUserInterrupt();
      int edge_count = 0;
      for (arma::uword r = 0; r < p; ++r)
        for (arma::uword c = r + 1; c < p; ++c)
          if (Z_ggm(r, c) == 1)
            edge_count++;
      int model_size = (int)arma::accu(gamma);
      Rcpp::Rcout << "Iter: " << iter << " | Model size: " << model_size
                  << " | GGM Edges: " << edge_count << " | eta1: " << eta1
                  << "\n";
    }

    if (is_zic) {
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
    } else if (is_count) {
      refresh_count_latent(Xb_total, alpha);
      loglik = calc_loglik_count(y_count, Xb_total - Z_tau, alpha, Z_tau,
                                 log_w_count);
    }

    // ---------------------------------------------------------
    // STEP A: Update Dynamic Network (GGM via SSVS, Wang et al. 2012)
    // FIX SNG-1: Removed redundant O(p^3) inv_sympd (saves massive compute).
    //   C_ggm is no longer maintained; only Sig_ggm is tracked.
    // FIX SNG-3: Backup invC11 before Schur complement; restore on Cholesky failure.
    // FIX PSGM-1: Periodic numerical reset of Sig_ggm.
    // IMP-4: Dimension-adaptive reset frequency: max(20, 5000/p).
    //   Drift grows as O(iter * p * eps * kappa), so reset more often for large p.
    // ---------------------------------------------------------
    const int ggm_reset_freq = std::max(20, (int)(5000 / p));
    if (iter > 0 && iter % ggm_reset_freq == 0) {
      // Periodic reset: recompute Sig_ggm from its inverse to remove drift
      bvs_dadj::sanitize_sym_mat_inplace(Sig_ggm, 1e10, 1e-8);
      arma::mat C_tmp;
      bool inv_ok = arma::inv_sympd(C_tmp, arma::symmatu(Sig_ggm));
      if (inv_ok) {
        arma::mat Sig_tmp;
        inv_ok = arma::inv_sympd(Sig_tmp, arma::symmatu(C_tmp));
        if (inv_ok) Sig_ggm = Sig_tmp;
      }
    }

    for (arma::uword i = 0; i < p; ++i) {
      vec_remove_idx(Sig_ggm.col(i), Sig12, i, p);
      mat_remove_rowcol(Sig_ggm, invC11, i, p);
      vec_remove_idx(S_ggm.col(i), S_i, i, p);

      for (arma::uword k = 0, idx = 0; k < p; ++k) {
        if (k == i)
          continue;
        tau_temp(idx++) = (Z_ggm(k, i) == 1) ? v1_ggm : v0_ggm;
      }

      double Sigii = Sig_ggm(i, i);

      // FIX SNG-3: backup invC11 before Schur complement subtraction
      arma::mat invC11_backup = invC11;
      invC11 -= (1.0 / Sigii) * (Sig12 * Sig12.t());
      Ci = (S_ggm(i, i) + lambda_ggm) * invC11;
      Ci.diag() += 1.0 / tau_temp;

      bool ok = arma::chol(U_ggm, arma::symmatu(Ci));
      if (!ok) {
        double jitter = 1e-8;
        for (int attempt = 0; attempt < 5 && !ok; ++attempt) {
          Ci.diag() += jitter;
          ok = arma::chol(U_ggm, arma::symmatu(Ci));
          jitter *= 10.0;
        }
        if (!ok) {
          // Restore invC11 and skip this column
          invC11 = invC11_backup;
          mat_insert_rowcol(Sig_ggm, invC11, i, p);
          continue;
        }
      }

      mu_i = -solve_spd_chol_upper(U_ggm, S_i);
      rnorm_into(eps_ggm);
      arma::vec delta =
          arma::solve(arma::trimatu(U_ggm), eps_ggm, arma::solve_opts::fast);
      b_ggm = mu_i + delta;

      double s_ii = S_ggm(i, i);
      double gam_val = R::rgamma(a_gam, 2.0 / (s_ii + lambda_ggm));

      invC11beta = invC11 * b_ggm;
      Sig12_new = -invC11beta / gam_val;
      invC11 += (1.0 / gam_val) * (invC11beta * invC11beta.t());

      arma::uword k = 0;
      for (arma::uword j = 0; j < p; ++j) {
        if (j == i)
          continue;
        Sig_ggm(j, i) = Sig12_new(k);
        Sig_ggm(i, j) = Sig12_new(k);
        k++;
      }
      Sig_ggm(i, i) = 1.0 / gam_val;
      mat_insert_rowcol(Sig_ggm, invC11, i, p);

      // SSVS edge selection
      arma::vec b2 = arma::square(b_ggm);
      w1 = -0.5 * arma::log(v0_vec) - 0.5 * b2 / v0_vec + log_1_pii_ggm;
      w2 = -0.5 * arma::log(v1_vec) - 0.5 * b2 / v1_vec + log_pii_ggm;
      arma::vec wmax = arma::max(w1, w2);
      w_ggm =
          arma::exp(w2 - wmax) / (arma::exp(w1 - wmax) + arma::exp(w2 - wmax));

      k = 0;
      for (arma::uword j = 0; j < p; ++j) {
        if (j == i)
          continue;
        uint8_t val = (R::runif(0, 1) < w_ggm(k)) ? 1 : 0;
        Z_ggm(j, i) = val;
        Z_ggm(i, j) = val;
        k++;
      }
    }

    // ---------------------------------------------------------
    // STEP B: Update Logistic Parameters
    // ---------------------------------------------------------

    // Re-initialize LB scores after GGM sweep (Z_ggm changed)
    // R2-FIX: avoid the O(p^2) Z_ggm -> Z_ggm_int copy each iteration by
    // calling the uint8 overload of init_lb_single_scores_ggm directly on
    // Z_ggm. For p=1000 this saves ~1M assignments per iteration.
    if (use_lb_gamma) {
      arma::ivec gamma_iv(p);
      for (arma::uword j = 0; j < p; ++j) gamma_iv(j) = (int)gamma(j);
      init_lb_single_scores_ggm(Z_ggm, (int)p, gamma_iv, mu, eta1,
                                 lb_score, lb_weight, lb_Z);
    }

    // 1+2. Gamma + Beta with inner thinning (n_thin_gb rounds)
    for (int thin_gb = 0; thin_gb < n_thin_gb; ++thin_gb) {

      // 1. Gamma via locally informed proposals
      for (arma::uword kk = 0; kk < 5; ++kk) {

        int j;
        double wt_sum = 0.0;  // used by fallback path
        if (use_lb_gamma) {
          j = sample_weighted_index(lb_weight, lb_Z, (int)p);
        } else {
          // Fallback: ad-hoc informed proposal weights
          for (arma::uword jj = 0; jj < p; ++jj) {
            double diff_jj = (gamma(jj) == 0) ? 1.0 : -1.0;
            double nd = 0.0;
            for (arma::uword ii = 0; ii < p; ++ii) {
              if (ii != jj && Z_ggm(ii, jj) == 1)
                nd += gamma(ii);
            }
            double ising_jj = diff_jj * (mu + eta1 * nd);
            proposal_weights(jj) = std::exp(std::min(0.0, ising_jj)) + 1e-10;
            wt_sum += proposal_weights(jj);
          }
          double u_draw = R::runif(0.0, wt_sum);
          arma::uword jj_sel = 0;
          double cum_wt = proposal_weights(0);
          while (cum_wt < u_draw && jj_sel < p - 1) {
            ++jj_sel;
            cum_wt += proposal_weights(jj_sel);
          }
          j = (int)jj_sel;
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
          arma::vec z_prop = Xb_total - Z_tau + db * xj;
          ll_diff =
              calc_loglik_continuous(y, z_prop, alpha, Z_tau, sigmasq) - loglik;
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
        double neigh_dyn = 0.0;
        for (arma::uword i = 0; i < p; ++i) {
          if (i != (arma::uword)j && Z_ggm(i, (arma::uword)j) == 1)
            neigh_dyn += gamma(i);
        }
        double ising_diff = diff * (mu + eta1 * neigh_dyn);

        double log_ratio = ll_diff + ising_diff;

        if (use_lb_gamma) {
          int delta_g = 1 - 2 * g_curr;
          arma::ivec gamma_iv(p);
          for (arma::uword jj = 0; jj < p; ++jj) gamma_iv(jj) = (int)gamma(jj);
          // R2-FIX: use uint8 overload directly on Z_ggm (no copy).
          build_lb_single_delta_ggm(Z_ggm, (int)p, gamma_iv, eta1,
                                     j, delta_g, lb_score, lb_weight, lb_Z, lb_delta);
          log_ratio += lb_delta.log_q_rev - lb_delta.log_q_fwd;
        } else {
          // Ad-hoc reverse proposal weight (fallback)
          double q_fwd = proposal_weights((arma::uword)j) / wt_sum;
          double rev_ising = -diff * (mu + eta1 * neigh_dyn);
          double q_rev_j = std::exp(std::min(0.0, rev_ising)) + 1e-10;
          double wt_sum_rev = wt_sum - proposal_weights((arma::uword)j) + q_rev_j;
          double q_bwd = q_rev_j / wt_sum_rev;
          log_ratio += std::log(q_bwd) - std::log(q_fwd);
        }

        if (bvs_dadj::safe_mh_accept(log_ratio)) {
          gamma(j) = static_cast<uint8_t>(g_prop);
          beta(j) = b_prop;
          if (is_continuous) {
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
          if (use_lb_gamma) {
            apply_lb_delta(lb_score, lb_weight, lb_Z, lb_delta);
          }
        }
      }

      // OPT: Build active indices manually (avoids arma::find heap allocation)
      active_idx.clear();
      for (arma::uword j = 0; j < p; ++j) {
        if (gamma(j) == 1)
          active_idx.push_back(j);
      }

      // 2. Beta — HMC/NUTS for binary/TTE (if selected),
      //         MALA (binary/count), Fisher-RW (TTE), Gibbs (continuous)
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
        const int d_act = (int)active_idx.size();
        const int ntau_local = (int)Z_dat.n_cols;
        const int dim_q = d_act + ntau_local + 1 + (is_tte ? 0 : 1);
        const bool in_warmup = (iter < burnin);
        const int dim_jump =
            (hmc_prev_dim >= 0) ? std::abs(dim_q - hmc_prev_dim) : 0;
        bool need_reinit = bvs_hmc::should_reinit_step_size(
            hmc_eps_initialised, hmc_prev_dim, dim_q, in_warmup);

        bvs_hmc::DiagMassMatrix mass_current;
        hmc_mass_adapter.build_joint_mass_matrix(mass_current, active_idx,
                                                 sigmasq, h, htau, nu0);

        if (!in_warmup && hmc_prev_dim >= 0 && dim_jump > 2 && hmc_post_burnin_adapt_left == 0) {
          hmc_post_burnin_adapt_left = hmc_post_burnin_adapt_window;
          hmc_da.reset(bvs_hmc::clamp_epsilon(hmc_epsilon),
                       hmc_target_accept);
        }
        hmc_prev_dim = dim_q;

        if (need_reinit && dim_q > 0) {
          arma::vec q0(dim_q);

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

          auto res0 = compute(q0);
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
        // Refresh MALA residuals after gamma flips
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

        for (arma::uword k = 0; k < active_idx.size(); ++k) {
          arma::uword j = active_idx[k];
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
                const double ep = bvs_dadj::clamp_finite(
                    alpha + Xb_total(i) + db * xj(i) + log_w_count(i), -50.0,
                    50.0, 0.0);
                const double lp = std::exp(ep);
                ll_prop += (double)y_count(i) * ep - lp;
                g_j_back += ((double)y_count(i) - lp) * xj(i);
              } else if (use_cloglog) {
                const double ep = alpha + Xb_total(i) + db * xj(i);
                ll_prop += bvs_imbalanced::loglik_cloglog_obs(y(i), ep);
                g_j_back += bvs_imbalanced::cloglog_residual(y(i), ep) * xj(i);
              } else {
                const double ep = alpha + Xb_total(i) + db * xj(i);
                const double pp = 1.0 / (1.0 + std::exp(-ep));
                ll_prop +=
                    y(i) * ep - (ep > 0.0 ? ep + std::log1p(std::exp(-ep))
                                          : std::log1p(std::exp(ep)));
                g_j_back += (y(i) - pp) * xj(i);
              }
            }
            double lld = ll_prop - loglik;
            double mean_bwd = b_prop + 0.5 * h_j * g_j_back;
            double lqf = -0.5 * (b_prop - mean_fwd) * (b_prop - mean_fwd) / h_j;
            double lqb =
                -0.5 * (beta(j) - mean_bwd) * (beta(j) - mean_bwd) / h_j;
            double prd =
                -0.5 *
                (std::pow(b_prop - beta0, 2) - std::pow(beta(j) - beta0, 2)) /
                pvar_j;
            if (bvs_dadj::safe_mh_accept(lld + prd + lqb - lqf)) {
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
        for (arma::uword k = 0; k < active_idx.size(); ++k) {
          arma::uword j = active_idx[k];
          arma::vec xj = X.col(j);
          double bj_old = beta(j);
          // FIX: Xb_total already includes Z_tau, so don't add Z_tau again
          arma::vec resid = y - (alpha + Xb_total - bj_old * xj);
          double denom = arma::dot(xj, xj) + 1.0;
          double mean = (arma::dot(xj, resid) + beta0) / denom;
          double bj_new = R::rnorm(mean, std::sqrt(sigmasq / denom));
          beta(j) = bj_new;
          Xb_total += (bj_new - bj_old) * xj; // Changed z to Xb_total
        }
        // FIX: Xb_total already includes Z_tau; pass Xb_total - Z_tau as z
        // so calc_loglik_continuous computes y - (alpha + z + offset) correctly
        loglik =
            calc_loglik_continuous(y, Xb_total - Z_tau, alpha, Z_tau, sigmasq);
      }

    } // end inner thinning for gamma + beta

    // ---------------------------------------------------------
    // STEP B2: Lambda (Student-t latent scales) — Gibbs step
    // ---------------------------------------------------------
    if (use_t_prior) {
      bvs_imbalanced::gibbs_update_lambda(lambda_t, beta, gamma, beta0,
                                          sigmasq, t_df, t_scale);
    }

    // 3. Alpha — MALA (binary/count), exact Gibbs (continuous)
    if (!is_continuous && !is_tte && !use_hmc_nuts) {
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
      double denom = static_cast<double>(n) + 1.0 / h;
      arma::vec resid = y - Xb_total;
      double mean = (arma::accu(resid) + alpha0 / h) / denom;
      alpha = R::rnorm(mean, std::sqrt(sigmasq / denom));
      loglik =
          calc_loglik_continuous(y, Xb_total - Z_tau, alpha, Z_tau, sigmasq);
    }

    // 3+. Tau — component-wise MALA (binary/count), block MH (TTE)
    if (ntau > 0 && !use_hmc_nuts) {
      if (!is_continuous) {
        if (is_tte) {
          arma::vec tau_prop(ntau);
          for (arma::uword j = 0; j < ntau; ++j)
            tau_prop(j) = R::rnorm(tau(j), std::sqrt(htau * sigmasq));
          arma::vec Z_tau_prop = Z_dat * tau_prop;
          arma::vec Xb_total_prop(n);
          for (arma::uword i = 0; i < n; ++i)
            Xb_total_prop(i) = Xb_total(i) - Z_tau(i) + Z_tau_prop(i);
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
            double mbwd = t_prop + 0.5 * h_k * g_k_back;
            double lqf =
                -0.5 * (t_prop - mean_fwd_k) * (t_prop - mean_fwd_k) / h_k;
            double lqb = -0.5 * (tau(k) - mbwd) * (tau(k) - mbwd) / h_k;
            double prd =
                -0.5 *
                (std::pow(t_prop - tau0, 2) - std::pow(tau(k) - tau0, 2)) /
                (htau * sigmasq);
            if (bvs_dadj::safe_mh_accept(lld + prd + lqb - lqf)) {
              tau(k) = t_prop;
              Xb_total += dt * zk;
              Z_tau += dt * zk;
              loglik = ll_t_prop;
              if (is_count || is_zic) {
                for (arma::uword i = 0; i < n; ++i) {
                  if (is_zic && zinb_Z_latent(i) == 1u) { mala_resid(i) = 0.0; continue; }
                  const double eta = bvs_dadj::clamp_finite(
                      alpha + Xb_total(i) + log_w_count(i), -50.0, 50.0, 0.0);
                  mala_resid(i) = (double)y_count(i) - std::exp(eta);
                }
              } else if (use_cloglog) {
                for (arma::uword i = 0; i < n; ++i)
                  mala_resid(i) = bvs_imbalanced::cloglog_residual(y(i), alpha + Xb_total(i));
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
          // FIX: Xb_total includes Z_tau which contains tj_old*zj;
          // to remove only j-th tau component, ADD back tj_old*zj
          arma::vec resid = y - alpha - Xb_total + tj_old * zj;
          double denom = arma::dot(zj, zj) + 1.0 / htau;
          double mean = (arma::dot(zj, resid) + tau0 / htau) / denom;
          double tj_new = R::rnorm(mean, std::sqrt(sigmasq / denom));
          tau(j) = tj_new;
          Z_tau += (tj_new - tj_old) * zj;
        }
        Xb_total = X * beta + Z_tau;
        loglik =
            calc_loglik_continuous(y, Xb_total - Z_tau, alpha, Z_tau, sigmasq);
      }
    }

    // 4. Update SigmaSq — exact Inverse-Gamma Gibbs for all outcome types.
    // For non-continuous outcomes the likelihood does not depend on sigmasq,
    // so the conjugate IG posterior is exact (no MH step needed).
    if (!use_hmc_nuts) {
      if (!is_continuous) {
        // Sufficient statistics
        double ss_beta = 0.0;
        for (arma::uword k = 0; k < active_idx.size(); ++k) {
          const arma::uword jj = active_idx[k];
          const double d = beta(jj) - beta0;
          // For Student-t prior: scale by 1/(s^2 * lambda_j)
          double scale_fac = use_t_prior ?
              1.0 / (t_scale * t_scale * lambda_t(jj)) : 1.0;
          ss_beta += d * d * scale_fac;
        }
        double ss_tau = 0.0;
        for (arma::uword j = 0; j < ntau; ++j) {
          double d = tau(j) - tau0;
          ss_tau += d * d;
        }
        const double da = alpha - alpha0;
        const double shape_post =
            0.5 * (nu0 + (double)active_idx.size() + 1.0 + (double)ntau);
        const double scale_post =
            0.5 * (nu0 * sigmasq0 + ss_beta + da * da / h + ss_tau / htau);
        const double gdraw =
            R::rgamma(shape_post, 1.0 / std::max(scale_post, 1e-12));
        if (std::isfinite(gdraw) && gdraw > 0.0)
          sigmasq = std::max(1e-10, 1.0 / gdraw);
      } else {
        double ss_beta = 0.0;
        for (arma::uword k = 0; k < active_idx.size(); ++k) {
          double d = beta(active_idx[k]) - beta0;
          ss_beta += d * d;
        }
        double ss_tau = 0.0;
        for (arma::uword j = 0; j < ntau; ++j) {
          double d = tau(j) - tau0;
          ss_tau += d * d;
        }
        double da = alpha - alpha0;
        arma::vec resid = y - (alpha + Xb_total);
        double sse = arma::dot(resid, resid);

        double shape_post = 0.5 * (nu0 + static_cast<double>(n) +
                                   static_cast<double>(active_idx.size()) +
                                   1.0 + static_cast<double>(ntau));
        double scale_post = 0.5 * (nu0 * sigmasq0 + sse + ss_beta +
                                   da * da / h + ss_tau / htau);
        double gdraw = R::rgamma(shape_post, 1.0 / std::max(scale_post, 1e-12));
        if (std::isfinite(gdraw) && gdraw > 0.0) {
          sigmasq = std::max(1e-10, 1.0 / gdraw);
        }
        Xb_total = X * beta + Z_tau;
        loglik =
            calc_loglik_continuous(y, Xb_total - Z_tau, alpha, Z_tau, sigmasq);
      }
    }

    // 5. Update eta1 via Moller et al. (2006) — passes Z_ggm directly
    moller_update_eta(Z_ggm, mu, eta1, eta1_sd, mu_tilde, eta1_tilde, gamma,
                      e_eta, f_eta, T_max, proposal_type, eta1_adapter,
                      use_cftp);

    if (iter >= burnin && (iter - burnin) % thin == 0) {
      int s = (iter - burnin) / thin;
      if (s < n_save) {
        // OPT 2: Accumulate PIP only at thinned save points (n_save times total)
        for (arma::uword j = 0; j < p; ++j)
          gamma_pip_sum(j) += (double)gamma(j);
        beta_out.row(s) = beta.t();
        if (store_gamma) {
          for (arma::uword j = 0; j < p; ++j)
            gamma_out(s, j) = gamma(j);
        }
        eta1_out(s) = eta1;
        alpha_out(s) = alpha;
        sigmasq_out(s) = sigmasq;
        tau_out.row(s) = tau.t();

        // OPT: reuse pre-allocated edge vectors
        // FIX: Accumulate Z_pip (was never accumulated → always returned zeros)
        edge_rows.clear();
        edge_cols.clear();
        for (arma::uword r = 0; r < p; ++r) {
          for (arma::uword c = r + 1; c < p; ++c) {
            if (Z_ggm(r, c) == 1) {
              Z_pip(r, c) += 1.0; // FIX: accumulate for posterior edge PIPs
              edge_rows.push_back((int)r);
              edge_cols.push_back((int)c);
            }
          }
        }
        int ne = (int)edge_rows.size();
        if (ne > 0) {
          Rcpp::IntegerMatrix edges(ne, 2);
          for (int ei = 0; ei < ne; ++ei) {
            edges(ei, 0) = edge_rows[ei];
            edges(ei, 1) = edge_cols[ei];
          }
          Z_list[s] = edges;
        } else {
          Z_list[s] = Rcpp::IntegerMatrix(0, 2);
        }
      }
    }
  }

  // Normalize Z_pip to posterior inclusion probabilities
  // FIX SNG-2: Z_pip is accumulated only at thinned save iterations (n_save times),
  // so divide by n_save, not n_post = niter (which is thin * n_save).
  double n_pip_denom = (double)n_save;
  if (n_pip_denom > 0)
    Z_pip /= n_pip_denom;
  // Symmetrize
  Z_pip = Z_pip + Z_pip.t();

  // OPT 2: Compute final PIP vector
  arma::vec gamma_pip = (n_pip_denom > 0) ? gamma_pip_sum / n_pip_denom : gamma_pip_sum;

  Rcpp::List result = Rcpp::List::create(
      Rcpp::Named("beta") = beta_out, Rcpp::Named("gamma_pip") = gamma_pip,
      Rcpp::Named("eta1") = eta1_out, Rcpp::Named("alpha") = alpha_out,
      Rcpp::Named("sigmasq") = sigmasq_out, Rcpp::Named("Z_list") = Z_list,
      Rcpp::Named("Z_pip") = Z_pip, Rcpp::Named("tau") = tau_out);
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
