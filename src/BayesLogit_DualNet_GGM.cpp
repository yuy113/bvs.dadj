// =============================================================================
// BayesLogit_DualNet_GGM_refactored.cpp
//
// Refactored dual-network BVS MH sampler:
//   - Binary logistic (MH) or continuous Gaussian (conjugate updates)
//   - Wang (2012) Bayesian GGM for dynamic adjacency (dense S, sparse Z)
//   - Fixed external adjacency matrix
//   - Moller et al. (2006) + Propp-Wilson for eta1/eta2 MH updates
//
// Key changes from original BayesLogit_DualNet_GGM.cpp:
//   FIX  1: Removed redundant O(p^3) inv_sympd (C_ggm never read)
//   FIX  2: Sparse adjacency lists for Z_active (dynamic) and R_fix_adj (fixed)
//   FIX  3: Propp-Wilson coupling-from-the-past replaces 5-sweep Gibbs
//   FIX  4: Complete Beta(e,f) prior for eta2 (was missing (f-1)log(1-eta2))
//   FIX  5: Eta proposals bounded to (0, eta_sd)
//   FIX  6: Running Z_pip replaces O(niter*p^2) Z_list storage
//   FIX  7: Thinning parameter added
//   FIX  8: n_mh_gamma is a parameter (was hardcoded to 5)
//   FIX  9: Incremental edge counter (no O(p^2) scan for diagnostics)
//   FIX 10: Safe index generation for gamma proposals
//   FIX 11: Sigmasq proposal floor to prevent underflow
//   FIX 12: ARMA_NO_DEBUG for release performance
//   FIX 13: Pre-allocated all temporary buffers
//   FIX 14: Ising energy uses upper-triangle only (no diagonal leak)
//
// Memory: O(n*p + p^2 + |E_dyn| + |E_fix|) — down from O(niter*p^2)
// Per-iter cost: O(p^3) for GGM sweep + O(p * n_mh_gamma) for gamma MH
//   (removed redundant O(p^3) inv_sympd per iteration)
// =============================================================================

// [[Rcpp::depends(RcppArmadillo)]]
#include "BVS_HMC_NUTS.h"
#include "BayesLogit_Numerics.h"
#include "BayesLogit_ZINB.h"
#include "BayesLogit_Sparse_Helpers.h"
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>
#include <vector>

// =============================================================================
// SECTION 1: HELPER FUNCTIONS
// =============================================================================

// --- Sub-matrix extraction helpers (unchanged, correct) ---
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

// FIX: Renamed parameters for clarity (target ← submat)
static inline void mat_insert_rowcol(arma::mat &target, const arma::mat &submat,
                                     arma::uword k, arma::uword p) {
  if (k > 0)
    target.submat(0, 0, k - 1, k - 1) = submat.submat(0, 0, k - 1, k - 1);
  if (k > 0 && k < p - 1)
    target.submat(0, k + 1, k - 1, p - 1) = submat.submat(0, k, k - 1, p - 2);
  if (k < p - 1 && k > 0)
    target.submat(k + 1, 0, p - 1, k - 1) = submat.submat(k, 0, p - 2, k - 1);
  if (k < p - 1)
    target.submat(k + 1, k + 1, p - 1, p - 1) =
        submat.submat(k, k, p - 2, p - 2);
}

// --- Triangular solve via pre-computed Cholesky ---
static inline arma::vec solve_spd_chol_upper(const arma::mat &U,
                                             const arma::vec &b) {
  arma::vec y = arma::solve(arma::trimatl(U.t()), b, arma::solve_opts::fast);
  return arma::solve(arma::trimatu(U), y, arma::solve_opts::fast);
}

// --- Fill vector with iid N(0,1) (avoids temporary allocation) ---
static inline void rnorm_into(arma::vec &out) {
  for (arma::uword i = 0; i < out.n_elem; ++i)
    out(i) = R::rnorm(0.0, 1.0);
}

// --- Logistic log-likelihood: sum_i [y_i * eta_i - log(1 + exp(eta_i))] ---
static inline double calc_loglik_binary(const arma::vec &y, const arma::vec &z,
                                        double alpha, const arma::vec &offset) {
  const arma::uword n = y.n_elem;
  double term1 = 0.0, term2 = 0.0;
  for (arma::uword i = 0; i < n; ++i) {
    double x = alpha + z(i) + offset(i);
    term1 += y(i) * x;
    // Numerically stable log(1 + exp(x))
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

// log_beta_pdf provided by BayesLogit_Sparse_Helpers.h

// =============================================================================
// GIBBS SAMPLING FOR AUXILIARY OMEGA (Dual Adjacency, sparse)
// Default method (use_cftp=false): 20 full Gibbs sweeps from random init.
// Uses sparse adjacency lists for both dynamic and fixed networks.
// =============================================================================
static void
gibbs_omega_dual_sparse(const std::vector<std::unordered_set<int>> &Z_active,
                        const std::vector<std::vector<int>> &R_fix_adj, int p,
                        double mu, double eta1, double eta2,
                        std::vector<int> &result,
                        int n_gibbs_sweeps = 20) {
  int seed_val = static_cast<int>(std::floor(R::runif(0.0, 1.0) * 2147483646.0)) + 1;
  std::mt19937 gf(seed_val);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  double base_prob = 1.0 / (1.0 + std::exp(-mu));
  for (int k = 0; k < p; ++k) {
    result[k] = (u01(gf) < base_prob) ? 1 : 0;
  }
  for (int sw = 0; sw < n_gibbs_sweeps; ++sw) {
    for (int k = 0; k < p; ++k) {
      double ker = 0.0;
      for (int j : Z_active[k]) {
        if (result[j]) ker += eta1;
      }
      for (int j : R_fix_adj[k]) {
        if (result[j]) ker += eta2;
      }
      result[k] = (u01(gf) < 1.0 / (1.0 + std::exp(-(mu + ker)))) ? 1 : 0;
    }
  }
}

// =============================================================================
// SECTION 2: PROPP-WILSON COUPLING-FROM-THE-PAST
//
// FIX 3: Replaces the 5-sweep Gibbs sampler. Uses sparse adjacency lists
// for both dynamic and fixed networks. Cost: O(p * avg_degree) per sweep.
// Pre-allocated workspace eliminates heap allocations per call.
// =============================================================================
static void
proppwilson_dual_sparse(const std::vector<std::unordered_set<int>> &Z_active,
                        const std::vector<std::vector<int>> &R_fix_adj, int p,
                        double mu, double eta1, double eta2, unsigned int T_max,
                        // Pre-allocated workspace
                        std::vector<int> &x_up, std::vector<int> &x_down,
                        std::vector<int> &result) {
  unsigned int T = 2;

  int seed_base = static_cast<int>(std::floor(R::runif(0.0, 1.0) * 2147483646.0)) + 1;

  auto not_coalesced = [&]() {
    for (int k = 0; k < p; ++k)
      if (x_up[k] != x_down[k])
        return true;
    return false;
  };

  // Initialize: x_up = all-zeros (minimal), x_down = all-ones (maximal)
  for (int k = 0; k < p; ++k) {
    x_up[k] = 0;
    x_down[k] = 1;
  }

  while (not_coalesced()) {
    // Reset to extremal configurations
    for (int k = 0; k < p; ++k) {
      x_up[k] = 0;
      x_down[k] = 1;
    }

    // Run chain from time -T to -1 using same random numbers
    for (int t = -(int)T; t <= -1; ++t) {
      int seed2 = -t * seed_base;
      std::mt19937 gen(seed2);
      std::uniform_real_distribution<double> unif01(0.0, 1.0);

      for (int i = 0; i < p; ++i) {
        double ker_up = 0.0, ker_down = 0.0;

        // Dynamic GGM neighbors (eta1) — binary conditional add
        for (int j : Z_active[i]) {
          if (x_up[j])
            ker_up += eta1;
          if (x_down[j])
            ker_down += eta1;
        }
        // Fixed external neighbors (eta2) — binary conditional add
        for (int j : R_fix_adj[i]) {
          if (x_up[j])
            ker_up += eta2;
          if (x_down[j])
            ker_down += eta2;
        }

        double pi_up = 1.0 / (1.0 + std::exp(-(mu + ker_up)));
        double pi_down = 1.0 / (1.0 + std::exp(-(mu + ker_down)));

        double u = unif01(gen);
        x_up[i] = (pi_up > u) ? 1 : 0;
        x_down[i] = (pi_down > u) ? 1 : 0;
      }
    }

    T = 2 * T;

    // Fallback if T exceeds T_max: force coalescence via extra Gibbs sweeps
    if (T >= T_max) {
      std::vector<int> diff_idx;
      for (int k = 0; k < p; ++k)
        if (x_up[k] != x_down[k])
          diff_idx.push_back(k);

      std::mt19937 gen_fb(seed_base + 99999);
      std::uniform_real_distribution<double> unif01_fb(0.0, 1.0);
      for (int sweep = 0; sweep < 100; ++sweep) {
        for (int idx = 0; idx < (int)diff_idx.size(); ++idx) {
          int m = diff_idx[idx];
          double ker = 0.0;
          for (int j : Z_active[m])
            if (x_up[j])
              ker += eta1;
          for (int j : R_fix_adj[m])
            if (x_up[j])
              ker += eta2;
          double prob = 1.0 / (1.0 + std::exp(-(mu + ker)));
          x_up[m] = (unif01_fb(gen_fb) < prob) ? 1 : 0;
        }
      }
      for (int k = 0; k < p; ++k)
        x_down[k] = x_up[k];
    }
  }

  // Copy coalesced result
  for (int k = 0; k < p; ++k)
    result[k] = x_up[k];
}

// =============================================================================
// SECTION 3: MOLLER MH UPDATE FOR DUAL ETA
//
// FIX 4: Complete Beta(e,f) prior for both eta1 and eta2.
// FIX 5: Proposals bounded to (0, eta_sd).
// Uses sparse sufficient statistics: O(|E_dyn| + |E_fix|) not O(p^2).
// =============================================================================
static void
moller_update_dual(const std::vector<std::unordered_set<int>> &Z_active,
                   const std::vector<std::vector<int>> &R_fix_adj, int p,
                   double mu, double &eta1, double &eta2, double eta1_sd,
                   double eta2_sd, double mu_tilde, double eta1_tilde,
                   double eta2_tilde, const arma::uvec &gamma, double e,
                   double f, unsigned int T_max, int proposal_type,
                   // Pre-allocated PW workspace
                   std::vector<int> &pw_x_up, std::vector<int> &pw_x_down,
                   std::vector<int> &om1, std::vector<int> &om2,
                   std::vector<int> &om1n, std::vector<int> &om2n,
                   EtaAdapter &adapter1, EtaAdapter &adapter2,
                   bool use_cftp = false) {

  // Exactly cancel the Moller effect to implement the Exchange Algorithm
  mu_tilde = mu;
  eta1_tilde = eta1;
  eta2_tilde = eta2;

  double eta1_new, eta2_new;
  double log_prop_ratio_eta1 = 0.0, log_prop_ratio_eta2 = 0.0;

  // Logit-transformed proposals with Vihola RAM adaptation
  double eta1_safe = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1));
  double phi1 = std::log(eta1_safe / (eta1_sd - eta1_safe));
  double phi1_new = R::rnorm(phi1, adapter1.sigma());
  eta1_new = eta1_sd / (1.0 + std::exp(-phi1_new));
  eta1_new = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1_new));
  log_prop_ratio_eta1 = std::log(eta1_new) + std::log(eta1_sd - eta1_new)
                      - std::log(eta1_safe) - std::log(eta1_sd - eta1_safe);

  double eta2_safe = std::max(1e-8, std::min(eta2_sd - 1e-8, eta2));
  double phi2 = std::log(eta2_safe / (eta2_sd - eta2_safe));
  double phi2_new = R::rnorm(phi2, adapter2.sigma());
  eta2_new = eta2_sd / (1.0 + std::exp(-phi2_new));
  eta2_new = std::max(1e-8, std::min(eta2_sd - 1e-8, eta2_new));
  log_prop_ratio_eta2 = std::log(eta2_new) + std::log(eta2_sd - eta2_new)
                      - std::log(eta2_safe) - std::log(eta2_sd - eta2_safe);

  // --- 4 auxiliary variable draws ---
  // om1:  drawn under (eta1,     eta2)       for eta1 update
  // om1n: drawn under (eta1_new, eta2)       for eta1 update
  // om2:  drawn under (eta1,     eta2)       for eta2 update
  // om2n: drawn under (eta1,     eta2_new)   for eta2 update
  if (use_cftp) {
    // CFTP (Propp-Wilson) — optional
    proppwilson_dual_sparse(Z_active, R_fix_adj, p, mu, eta1, eta2, T_max,
                            pw_x_up, pw_x_down, om1);
    proppwilson_dual_sparse(Z_active, R_fix_adj, p, mu, eta1, eta2, T_max,
                            pw_x_up, pw_x_down, om2);
    proppwilson_dual_sparse(Z_active, R_fix_adj, p, mu, eta1_new, eta2, T_max,
                            pw_x_up, pw_x_down, om1n);
    proppwilson_dual_sparse(Z_active, R_fix_adj, p, mu, eta1, eta2_new, T_max,
                            pw_x_up, pw_x_down, om2n);
  } else {
    // Standard Gibbs sampling (default) — no CFTP overhead
    gibbs_omega_dual_sparse(Z_active, R_fix_adj, p, mu, eta1, eta2, om1, 50);
    gibbs_omega_dual_sparse(Z_active, R_fix_adj, p, mu, eta1, eta2, om2, 50);
    gibbs_omega_dual_sparse(Z_active, R_fix_adj, p, mu, eta1_new, eta2, om1n, 50);
    gibbs_omega_dual_sparse(Z_active, R_fix_adj, p, mu, eta1, eta2_new, om2n, 50);
  }

  // --- Compute sparse sufficient statistics ---
  // Sums of omega vectors
  double sum_om1 = 0.0, sum_om2 = 0.0, sum_om1n = 0.0, sum_om2n = 0.0;
  for (int j = 0; j < p; ++j) {
    sum_om1 += om1[j];
    sum_om2 += om2[j];
    sum_om1n += om1n[j];
    sum_om2n += om2n[j];
  }

  // Edge products for dynamic network (upper triangle only → no double-count)
  double B_R1 = 0.0; // gamma^T R_dyn gamma (edges only)
  double A_om1_R1 = 0.0, A_om1n_R1 = 0.0, A_om2_R1 = 0.0, A_om2n_R1 = 0.0;
  for (int j = 0; j < p; ++j) {
    for (int k : Z_active[j]) {
      if (k <= j)
        continue; // upper triangle only
      double gj = (double)gamma(j), gk = (double)gamma(k);
      B_R1 += gj * gk;
      A_om1_R1 += (double)om1[j] * (double)om1[k];
      A_om1n_R1 += (double)om1n[j] * (double)om1n[k];
      A_om2_R1 += (double)om2[j] * (double)om2[k];
      A_om2n_R1 += (double)om2n[j] * (double)om2n[k];
    }
  }

  // Edge products for fixed network (upper triangle only)
  double B_R2 = 0.0;
  double A_om1_R2 = 0.0, A_om1n_R2 = 0.0, A_om2_R2 = 0.0, A_om2n_R2 = 0.0;
  for (int j = 0; j < p; ++j) {
    for (int k : R_fix_adj[j]) {
      if (k <= j)
        continue;
      double gj = (double)gamma(j), gk = (double)gamma(k);
      B_R2 += gj * gk;
      A_om1_R2 += (double)om1[j] * (double)om1[k];
      A_om1n_R2 += (double)om1n[j] * (double)om1n[k];
      A_om2_R2 += (double)om2[j] * (double)om2[k];
      A_om2n_R2 += (double)om2n[j] * (double)om2n[k];
    }
  }

  // --- eta1 Metropolis-Hastings ---
  // FIX 4: Full Beta(e,f) prior on eta1/eta1_sd
  double log_prior_eta1 = log_beta_pdf(eta1_new / eta1_sd, e, f) -
                          log_beta_pdf(eta1 / eta1_sd, e, f);
  double log_target_eta1 = (eta1_new - eta1) * B_R1 + log_prior_eta1;
  double log_aux_eta1 = mu_tilde * (sum_om1n - sum_om1) +
                        eta1_tilde * (A_om1n_R1 - A_om1_R1) +
                        eta2_tilde * (A_om1n_R2 - A_om1_R2);
  double log_norm_eta1 = mu * (sum_om1 - sum_om1n) + eta1 * A_om1_R1 -
                         eta1_new * A_om1n_R1 + eta2 * (A_om1_R2 - A_om1n_R2);
  double log_MH_eta1 =
      log_target_eta1 + log_aux_eta1 + log_norm_eta1 + log_prop_ratio_eta1;

  // --- eta2 Metropolis-Hastings ---
  // FIX 4: Full Beta(e,f) prior on eta2/eta2_sd (was missing (f-1)log(1-x))
  double log_prior_eta2 = log_beta_pdf(eta2_new / eta2_sd, e, f) -
                          log_beta_pdf(eta2 / eta2_sd, e, f);
  double log_target_eta2 = (eta2_new - eta2) * B_R2 + log_prior_eta2;
  double log_aux_eta2 = mu_tilde * (sum_om2n - sum_om2) +
                        eta1_tilde * (A_om2n_R1 - A_om2_R1) +
                        eta2_tilde * (A_om2n_R2 - A_om2_R2);
  double log_norm_eta2 = mu * (sum_om2 - sum_om2n) + eta2 * A_om2_R2 -
                         eta2_new * A_om2n_R2 + eta1 * (A_om2_R1 - A_om2n_R1);
  double log_MH_eta2 =
      log_target_eta2 + log_aux_eta2 + log_norm_eta2 + log_prop_ratio_eta2;

  // Accept/reject independently with Vihola RAM adaptation
  double accept_prob_eta1 = std::min(1.0, std::exp(std::min(0.0, log_MH_eta1)));
  double accept_prob_eta2 = std::min(1.0, std::exp(std::min(0.0, log_MH_eta2)));
  if (bvs_dadj::safe_mh_accept(log_MH_eta1))
    eta1 = eta1_new;
  if (bvs_dadj::safe_mh_accept(log_MH_eta2))
    eta2 = eta2_new;
  adapter1.update(accept_prob_eta1);
  adapter2.update(accept_prob_eta2);
}

// =============================================================================
// SECTION 4: MAIN FUNCTION
//
// BayesLogit_DualNet_GGM (Refactored):
//   Dense S_ggm input (Wang 2012 column sweep)
//   Sparse Z_active for learned edges
//   Sparse R_fix_adj for fixed edges
//   Moller + Propp-Wilson for eta sampling
//
// Memory: O(n*p + p^2 + |E_dyn| + |E_fix|) — no Z_list
// =============================================================================

// [[Rcpp::export]]
Rcpp::List BayesLogit_DualNet_GGM(
    const arma::mat &X, const arma::vec &y, const arma::mat &S_ggm,
    double n_ggm, const Rcpp::IntegerMatrix &R_fix_int, int niter, int burnin,
    double mu, double nu0, double sigmasq0, double alpha0, double beta0,
    double h, int n_mh_gamma, double v0_ggm, double v1_ggm, double pii_ggm,
    double lambda_ggm, double eta1_sd, double eta2_sd, double mu_tilde,
    double eta1_tilde, double eta2_tilde, double e_eta, double f_eta,
    unsigned int T_max, int proposal_type, int thin = 1, int n_thin_gb = 3,
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

  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  if (S_ggm.n_rows != p || S_ggm.n_cols != p)
    Rcpp::stop("S_ggm dimensions (%d x %d) must match p = %d",
               (int)S_ggm.n_rows, (int)S_ggm.n_cols, (int)p);
  if ((arma::uword)R_fix_int.nrow() != p || (arma::uword)R_fix_int.ncol() != p)
    Rcpp::stop("R_fix dimensions must match p = %d", (int)p);
  if (thin < 1)
    thin = 1;
  if (n_thin_gb < 1)
    n_thin_gb = 1;
  if (n_mh_gamma < 1)
    n_mh_gamma = 1;

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

  // =========================================================================
  // 1. BUILD SPARSE ADJACENCY LIST FOR FIXED NETWORK
  // =========================================================================
  // FIX 2: Sparse adjacency list — O(|E_fix|) storage, O(degree) lookups
  std::vector<std::vector<int>> R_fix_adj(p);
  for (arma::uword i = 0; i < p; ++i) {
    for (arma::uword j = 0; j < p; ++j) {
      if (i != j && R_fix_int(i, j) != 0)
        R_fix_adj[i].push_back((int)j);
    }
  }

  // =========================================================================
  // 2. INITIALIZATION
  // =========================================================================
  arma::vec beta_vec(p, arma::fill::zeros);
  arma::uvec gamma(p, arma::fill::zeros);
  if (beta_in.isNotNull())
    beta_vec = Rcpp::as<arma::vec>(beta_in);
  if (gamma_in.isNotNull())
    gamma = Rcpp::as<arma::uvec>(gamma_in);

  double alpha = alpha_in;
  double sigmasq = 1.0;
  double eta1 = eta1_sd * 0.5;
  double eta2 = eta2_sd * 0.5;
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

  arma::vec z = X * beta_vec;
  arma::vec Xb_total = z + Z_tau;
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

  // --- GGM state ---
  arma::mat Sig_ggm = arma::diagmat(S_ggm.diag());
  // FIX 2: Sparse edge state — O(|E_dyn|) memory instead of O(p^2)
  std::vector<std::unordered_set<int>> Z_active(p);
  int n_edges = 0; // FIX 9: incremental edge counter

  // --- GGM pre-allocated work buffers ---
  const arma::uword pm1 = p - 1;
  arma::mat invC11(pm1, pm1), Ci(pm1, pm1), U_ggm(pm1, pm1);
  arma::vec Sig12(pm1), S_i(pm1), tau_temp(pm1);
  arma::vec mu_i(pm1), b_ggm(pm1), eps_ggm(pm1);
  arma::vec invC11beta(pm1), Sig12_new(pm1);
  arma::vec b2_buf(pm1), w1_buf(pm1), w2_buf(pm1), wmax_buf(pm1),
      w_ggm_buf(pm1);

  // Pre-computed SSVS constants
  double log_pii_ggm = std::log(pii_ggm);
  double log_1_pii_ggm = std::log(1.0 - pii_ggm);
  double log_v0_half = -0.5 * std::log(v0_ggm);
  double log_v1_half = -0.5 * std::log(v1_ggm);
  double inv_v0 = 1.0 / v0_ggm;
  double inv_v1 = 1.0 / v1_ggm;
  double a_gam = 0.5 * n_ggm + 1.0;

  // --- Logistic pre-allocated buffers ---
  arma::vec z_prop(n); // FIX 13: pre-allocate proposal vector
  std::vector<arma::uword> active_idx;
  active_idx.reserve(p);

  // OPT 1: Pre-allocate informed proposal weight buffers
  arma::vec proposal_weights(p);

  // --- Locally-balanced gamma proposal state (Zanella 2020) ---
  Rcpp::IntegerMatrix Z_ggm_int(p, p);  // dense snapshot for LB helpers
  std::vector<double> lb_score, lb_weight;
  double lb_Z = 0.0;
  LBProposalDelta lb_delta;

  // Pre-compute fixed neighbor sums for Ising prior
  arma::vec neigh_fix_sum(p, arma::fill::zeros);
  for (arma::uword j = 0; j < p; ++j)
    for (int nbr : R_fix_adj[j])
      neigh_fix_sum(j) += gamma(nbr);

  // OPT 2: Online PIP accumulation
  arma::vec gamma_pip_sum(p, arma::fill::zeros);

  // --- MALA / Fisher-RW pre-computed column norms ---
  arma::vec X_col_sq_sums(p);
  for (arma::uword j = 0; j < p; ++j)
    X_col_sq_sums(j) = arma::dot(X.col(j), X.col(j));
  arma::vec Z_col_sq_sums(ntau, arma::fill::zeros);
  if (ntau > 0)
    for (arma::uword j = 0; j < ntau; ++j)
      Z_col_sq_sums(j) = arma::dot(Z_dat.col(j), Z_dat.col(j));
  // Gradient residuals for MALA (binary: y - p_hat; count: y - mu_hat)
  arma::vec mala_resid(n, arma::fill::zeros);

  // --- Propp-Wilson pre-allocated workspace ---
  std::vector<int> pw_x_up(p), pw_x_down(p);
  std::vector<int> pw_om1(p), pw_om2(p), pw_om1n(p), pw_om2n(p);

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

  // --- Output storage ---
  int n_save = niter / thin;
  arma::mat beta_out(n_save, p);
  arma::umat gamma_out;
  if (store_gamma)
    gamma_out.set_size(n_save, p);
  arma::vec eta1_out(n_save), eta2_out(n_save);
  arma::vec alpha_out(n_save), sigmasq_out(n_save);
  arma::mat tau_out(n_save, ntau);
  // FIX 6: Running posterior edge inclusion probability instead of Z_list
  // Saves ~800 MB for p=200, niter=5000
  arma::mat Z_pip(p, p, arma::fill::zeros); // accumulate, divide at end
  Rcpp::List Z_list(n_save);
  std::vector<int> edge_rows, edge_cols;
  const arma::uword max_edges = p * (p - 1) / 2;
  edge_rows.reserve(std::min(max_edges, (arma::uword)1000));
  edge_cols.reserve(std::min(max_edges, (arma::uword)1000));

  // =========================================================================
  // 3. MCMC LOOP
  // =========================================================================
  EtaAdapter eta1_adapter(0.5);  // Vihola RAM for eta1 proposal
  EtaAdapter eta2_adapter(0.5);  // Vihola RAM for eta2 proposal

  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {

    // Periodic diagnostics + interrupt check
    if (iter > 0 && iter % 5000 == 0) {
      Rcpp::checkUserInterrupt();
      int model_size = (int)arma::accu(gamma);
      Rcpp::Rcout << "Iter: " << iter << " | Model: " << model_size
                  << " | Edges: " << n_edges << " | eta1: " << eta1
                  << " | eta2: " << eta2 << "\n";
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

    // Pre-compute sqrt(sigmasq) once per iteration
    double sd_sig = std::sqrt(sigmasq);
    // Pre-compute Fisher information H_i vector for TTE MALA
    arma::vec cox_H;

    // -----------------------------------------------------------------
    // STEP A: Wang (2012) GGM Block-Gibbs Column Sweep
    //
    // FIX 1: Removed the redundant C_ggm = inv_sympd(Sig_ggm) call.
    // The concentration matrix was maintained but never read.
    // This saves O(p^3) FLOPS per iteration — the biggest speedup.
    //
    // The column sweep updates Sig_ggm in-place via rank-1 updates:
    //   invC11 = Sig_{-i,-i} - Sig_{12} Sig_{12}^T / Sig_{ii}
    //          = C_{-i,-i}^{-1}  (Schur complement)
    //   ... sample beta, update Sig_ggm ...
    //   invC11 += (1/gamma) * (invC11*b)(invC11*b)^T
    //          = new Sig_{-i,-i}
    // -----------------------------------------------------------------
    for (arma::uword i = 0; i < p; ++i) {
      // Extract sub-vectors
      vec_remove_idx(Sig_ggm.col(i), Sig12, i, p);
      mat_remove_rowcol(Sig_ggm, invC11, i, p); // Sig_{-i,-i}
      vec_remove_idx(S_ggm.col(i), S_i, i, p);

      // Build tau vector from current edge state
      for (arma::uword k = 0, idx = 0; k < p; ++k) {
        if (k == i)
          continue;
        tau_temp(idx++) = (Z_active[i].count((int)k) > 0) ? v1_ggm : v0_ggm;
      }

      double Sigii = Sig_ggm(i, i);
      double s_ii = S_ggm(i, i);

      // Schur complement: C_{-i,-i}^{-1} = Sig_{-i,-i} - Sig_12*Sig_12^T/Sig_ii
      invC11 -= (1.0 / Sigii) * (Sig12 * Sig12.t());

      // Posterior precision for off-diagonal concentration column
      Ci = (s_ii + lambda_ggm) * invC11;
      Ci.diag() += 1.0 / tau_temp;

      // Cholesky of posterior precision
      bool ok = arma::chol(U_ggm, arma::symmatu(Ci));
      if (!ok) {
        Ci.diag() += 1e-6;
        ok = arma::chol(U_ggm, arma::symmatu(Ci));
        if (!ok)
          continue; // skip this column on failure
      }

      // Posterior mean and sample
      mu_i = -solve_spd_chol_upper(U_ggm, S_i);
      rnorm_into(eps_ggm);
      arma::vec delta =
          arma::solve(arma::trimatu(U_ggm), eps_ggm, arma::solve_opts::fast);
      b_ggm = mu_i + delta;

      // Sample diagonal concentration: gamma ~ Gamma(a, rate)
      double gam_val =
          R::rgamma(a_gam, 2.0 / (s_ii + lambda_ggm)); // scale = 1/rate

      // Update Sig_ggm via rank-1 formulas
      invC11beta = invC11 * b_ggm;
      Sig12_new = -invC11beta / gam_val;
      invC11 += (1.0 / gam_val) * (invC11beta * invC11beta.t());
      // Now invC11 = new Sig_{-i,-i}

      // Write back to Sig_ggm
      {
        arma::uword k = 0;
        for (arma::uword j = 0; j < p; ++j) {
          if (j == i)
            continue;
          Sig_ggm(j, i) = Sig12_new(k);
          Sig_ggm(i, j) = Sig12_new(k);
          k++;
        }
      }
      Sig_ggm(i, i) = 1.0 / gam_val;
      mat_insert_rowcol(Sig_ggm, invC11, i, p);

      // --- SSVS Edge Selection with Incremental Tracking ---
      // FIX 9: Update n_edges incrementally instead of scanning full matrix
      {
        arma::uword k = 0;
        for (arma::uword j = 0; j < p; ++j) {
          if (j == i)
            continue;

          double bk2 = b_ggm(k) * b_ggm(k);
          double w1 = log_v0_half - 0.5 * bk2 * inv_v0 + log_1_pii_ggm;
          double w2 = log_v1_half - 0.5 * bk2 * inv_v1 + log_pii_ggm;
          double wm = std::max(w1, w2);
          double prob =
              std::exp(w2 - wm) / (std::exp(w1 - wm) + std::exp(w2 - wm));

          bool was_active = (Z_active[i].count((int)j) > 0);
          bool now_active = (R::runif(0, 1) < prob);

          if (now_active && !was_active) {
            Z_active[i].insert((int)j);
            Z_active[j].insert((int)i);
            n_edges++;
          } else if (!now_active && was_active) {
            Z_active[i].erase((int)j);
            Z_active[j].erase((int)i);
            n_edges--;
          }

          k++;
        }
      }
    } // end GGM column sweep

    // Re-initialize LB scores after GGM sweep (Z_ggm changed)
    if (use_lb_gamma) {
      for (arma::uword r = 0; r < p; ++r)
        for (arma::uword c = 0; c < p; ++c)
          Z_ggm_int(r, c) = (Z_active[r].count((int)c) > 0) ? 1 : 0;
      arma::ivec gamma_iv(p);
      for (arma::uword j = 0; j < p; ++j) gamma_iv(j) = (int)gamma(j);
      init_lb_dual_scores_ggm(Z_ggm_int, R_fix_int, (int)p, gamma_iv, mu,
                               eta1, eta2, lb_score, lb_weight, lb_Z);
    }

    // -----------------------------------------------------------------
    // STEP B+C: Gamma + Beta with inner thinning (n_thin_gb rounds)
    // -----------------------------------------------------------------
    for (int thin_gb = 0; thin_gb < n_thin_gb; ++thin_gb) {

      // STEP B: Gamma (Variable Selection) via locally informed proposals
      for (int mh = 0; mh < n_mh_gamma; ++mh) {

        int j;
        double wt_sum = 0.0;  // used by fallback path
        if (use_lb_gamma) {
          j = sample_weighted_index(lb_weight, lb_Z, (int)p);
        } else {
          // Fallback: ad-hoc informed proposal weights
          for (arma::uword jj = 0; jj < p; ++jj) {
            double diff_jj = (gamma(jj) == 0) ? 1.0 : -1.0;
            double nd = 0.0;
            for (int nbr : Z_active[jj])
              nd += gamma(nbr);
            double ising_jj =
                diff_jj * (mu + eta1 * nd + eta2 * neigh_fix_sum(jj));
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

        int g_curr = (int)gamma(j);
        int g_prop = 1 - g_curr;
        double b_curr = beta_vec(j);
        double gamma_prior_var = use_t_prior ?
            bvs_imbalanced::beta_prior_var_j(sigmasq, imb_link_code, t_scale, lambda_t(j)) :
            sigmasq;
        double b_prop =
            (g_prop == 1) ? R::rnorm(beta0, std::sqrt(gamma_prior_var)) : 0.0;

        double db = b_prop - b_curr;
        double ll_diff = 0.0;
        const arma::vec &xj = X.col(j);
        std::vector<double> delta_group_W;

        if (is_tte) {
          delta_group_W.assign(cox_data.group_start.size(), 0.0);
        }

        // --- OPT 3: Incremental log-likelihood difference ---
        if (is_continuous) {
          arma::vec Xb_total_prop = Xb_total + db * xj;
          ll_diff = calc_loglik_continuous(y, Xb_total_prop - Z_tau, alpha,
                                           Z_tau, sigmasq) -
                    loglik;
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
          // Inline delta computation
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

        double diff = (double)(g_prop - g_curr);
        double neigh_dyn = 0.0;
        for (int nbr : Z_active[j])
          neigh_dyn += gamma(nbr);

        double ising_diff =
            diff * (mu + eta1 * neigh_dyn + eta2 * neigh_fix_sum(j));

        double log_ratio = ll_diff + ising_diff;

        if (use_lb_gamma) {
          int delta_g = 1 - 2 * g_curr;
          arma::ivec gamma_iv(p);
          for (arma::uword jj = 0; jj < p; ++jj) gamma_iv(jj) = (int)gamma(jj);
          build_lb_dual_delta_ggm(Z_ggm_int, R_fix_int, (int)p, gamma_iv,
                                   eta1, eta2, j, delta_g,
                                   lb_score, lb_weight, lb_Z, lb_delta);
          log_ratio += lb_delta.log_q_rev - lb_delta.log_q_fwd;
        } else {
          // Ad-hoc reverse proposal weight (fallback)
          double q_fwd = proposal_weights((arma::uword)j) / wt_sum;
          double rev_ising_jj =
              -diff * (mu + eta1 * neigh_dyn + eta2 * neigh_fix_sum(j));
          double q_rev_j = std::exp(std::min(0.0, rev_ising_jj)) + 1e-10;
          double wt_sum_rev = wt_sum - proposal_weights((arma::uword)j) + q_rev_j;
          double q_bwd = q_rev_j / wt_sum_rev;
          log_ratio += std::log(q_bwd) - std::log(q_fwd);
        }

        if (bvs_dadj::safe_mh_accept(log_ratio)) {
          gamma(j) = g_prop;
          beta_vec(j) = b_prop;
          // Update cached fixed neighbor sums
          for (int nbr : R_fix_adj[j])
            neigh_fix_sum(nbr) += diff;
          z += db * xj;  // FIX: keep z = X*beta in sync with beta_vec
          Xb_total += db * xj;
          if (is_tte) {
            cox_tracker.apply_diff(xj, db, ll_diff, delta_group_W);
          }
          if (use_lb_gamma) {
            apply_lb_delta(lb_score, lb_weight, lb_Z, lb_delta);
          }
          if (is_continuous) {
            loglik = calc_loglik_continuous(y, Xb_total - Z_tau, alpha, Z_tau,
                                            sigmasq);
          } else if (!is_tte) {
            loglik += ll_diff;
          } else { // is_tte
            loglik = cox_tracker.get_loglik();
          }
        }
      }

      // Build active index set
      active_idx.clear();
      for (arma::uword j = 0; j < p; ++j)
        if (gamma(j) == 1)
          active_idx.push_back(j);

      // STEP B: Beta updates — HMC/NUTS for binary/TTE (if selected),
      //         MALA for binary/count, Fisher-RW for TTE
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
            q0(kk) = beta_vec(active_idx[kk]);
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
            alg_type_int, X, y, Z_dat, beta_vec, Xb_total, loglik, alpha, tau,
            sigmasq, beta0, alpha0, tau0, h, htau, nu0, sigmasq0, active_idx,
            is_tte, cox_data, cox_tracker, hmc_epsilon, hmc_n_leapfrog,
            nuts_max_treedepth, compute, &mass_current, &step_stats);
        hmc_diag.record(hmc_epsilon, step_stats);

        if (in_warmup) {
          hmc_mass_adapter.observe(active_idx, beta_vec, alpha, tau,
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
        if (is_tte) {
          // --- Fisher information-scaled random walk for TTE ---
          cox_tracker.compute_H_vec(cox_H);
          for (arma::uword k = 0; k < active_idx.size(); ++k) {
            arma::uword j = active_idx[k];
            const arma::vec &xj = X.col(j);
            double I_jj = cox_tracker.compute_info_diag_j(xj, cox_H);
            double step = (I_jj > 1e-12) ? 1.0 / std::sqrt(I_jj) : sd_sig;
            step = std::min(step, 2.0 * sd_sig);
            double db = R::rnorm(0.0, step);
            double b_prop = beta_vec(j) + db;
            std::vector<double> delta_group_W(cox_data.group_start.size(), 0.0);
            double ll_diff = cox_tracker.propose_diff(xj, db, delta_group_W);
            double pr_curr = -0.5 * std::pow(beta_vec(j) - beta0, 2) / sigmasq;
            double pr_prop = -0.5 * std::pow(b_prop - beta0, 2) / sigmasq;
            if (bvs_dadj::safe_mh_accept(ll_diff + (pr_prop - pr_curr))) {
              beta_vec(j) = b_prop;
              z += db * xj;
              Xb_total += db * xj;
              cox_tracker.apply_diff(xj, db, ll_diff, delta_group_W);
              loglik = cox_tracker.get_loglik();
              // Refresh H_vec after accept to keep Fisher info current
              cox_tracker.compute_H_vec(cox_H);
            }
          }
        } else {
          // --- Component-wise MALA for binary/count/imbalanced_binary ---
          // Refresh gradient residuals from current state
          for (arma::uword i = 0; i < n; ++i) {
            double eta_i = alpha + Xb_total(i);
            if (use_cloglog) {
              mala_resid(i) = bvs_imbalanced::cloglog_residual(y(i), eta_i);
            } else if (is_count || is_zic) {
              if (is_zic && zinb_Z_latent(i) == 1u) { mala_resid(i) = 0.0; continue; }
              double mu_i = std::exp(bvs_dadj::clamp_finite(
                  eta_i + log_w_count(i), -50.0, 50.0, 0.0));
              mala_resid(i) = (double)y_count(i) - mu_i;
            } else {
              double p_i = 1.0 / (1.0 + std::exp(-eta_i));
              mala_resid(i) = y(i) - p_i;
            }
          }
          for (arma::uword k = 0; k < active_idx.size(); ++k) {
            arma::uword j = active_idx[k];
            const arma::vec &xj = X.col(j);
            // Component-wise MALA for binary/count/imbalanced_binary
            double pvar_j = use_t_prior ?
                bvs_imbalanced::beta_prior_var_j(sigmasq, imb_link_code, t_scale, lambda_t(j)) :
                sigmasq;
            double g_j =
                arma::dot(xj, mala_resid) - (beta_vec(j) - beta0) / pvar_j;
            double h_j =
                std::min(0.5 * pvar_j / (X_col_sq_sums(j) + 1.0), 1.0);
            double b_prop = beta_vec(j) + 0.5 * h_j * g_j +
                            std::sqrt(h_j) * R::rnorm(0.0, 1.0);
            double db = b_prop - beta_vec(j);
            // Compute ll_prop and backward gradient in one O(n) pass
            double ll_prop = 0.0;
            double g_j_back = -(b_prop - beta0) / pvar_j;
            for (arma::uword i = 0; i < n; ++i) {
              double new_eta = alpha + Xb_total(i) + db * xj(i);
              if (is_count || is_zic) {
                if (is_zic && zinb_Z_latent(i) == 1u) continue;
                double eta_c = bvs_dadj::clamp_finite(
                    new_eta + log_w_count(i), -50.0, 50.0, 0.0);
                double mu_new = std::exp(eta_c);
                ll_prop +=
                    (double)y_count(i) * eta_c - mu_new;
                g_j_back += ((double)y_count(i) - mu_new) * xj(i);
              } else if (use_cloglog) {
                ll_prop += bvs_imbalanced::loglik_cloglog_obs(y(i), new_eta);
                g_j_back += bvs_imbalanced::cloglog_residual(y(i), new_eta) * xj(i);
              } else {
                double p_new = 1.0 / (1.0 + std::exp(-new_eta));
                ll_prop +=
                    y(i) * new_eta -
                    (new_eta > 0.0 ? new_eta + std::log1p(std::exp(-new_eta))
                                   : std::log1p(std::exp(new_eta)));
                g_j_back += (y(i) - p_new) * xj(i);
              }
            }
            // Backward proposal centre
            double b_back = b_prop + 0.5 * h_j * g_j_back;
            double pr_curr = -0.5 * std::pow(beta_vec(j) - beta0, 2) / pvar_j;
            double pr_prop = -0.5 * std::pow(b_prop - beta0, 2) / pvar_j;
            // MALA log proposal correction
            double lq_fwd =
                -0.5 * std::pow(b_prop - (beta_vec(j) + 0.5 * h_j * g_j), 2) /
                h_j;
            double lq_bwd = -0.5 * std::pow(beta_vec(j) - b_back, 2) / h_j;
            double log_mh =
                (ll_prop - loglik) + (pr_prop - pr_curr) + (lq_bwd - lq_fwd);
            if (bvs_dadj::safe_mh_accept(log_mh)) {
              beta_vec(j) = b_prop;
              z += db * xj;
              Xb_total += db * xj;
              loglik = ll_prop;
              // Incremental residual update
              for (arma::uword i = 0; i < n; ++i) {
                double new_eta = alpha + Xb_total(i);
                if (use_cloglog) {
                  mala_resid(i) = bvs_imbalanced::cloglog_residual(y(i), new_eta);
                } else if (is_count || is_zic) {
                  if (is_zic && zinb_Z_latent(i) == 1u) { mala_resid(i) = 0.0; continue; }
                  double mu_i = std::exp(bvs_dadj::clamp_finite(
                      new_eta + log_w_count(i), -50.0, 50.0, 0.0));
                  mala_resid(i) = (double)y_count(i) - mu_i;
                } else {
                  mala_resid(i) = y(i) - 1.0 / (1.0 + std::exp(-new_eta));
                }
              }
            }
          }
        }
      } else {
        for (arma::uword k = 0; k < active_idx.size(); ++k) {
          arma::uword j = active_idx[k];
          arma::vec xj = X.col(j);
          double bj_old = beta_vec(j);
          arma::vec resid = y - (alpha + Z_tau + z - bj_old * xj);
          double denom = arma::dot(xj, xj) + 1.0;
          double mean = (arma::dot(xj, resid) + beta0) / denom;
          double bj_new = R::rnorm(mean, std::sqrt(sigmasq / denom));
          beta_vec(j) = bj_new;
          z += (bj_new - bj_old) * xj;
        }
        Xb_total = z + Z_tau; // Update Xb_total after z changes
        loglik = calc_loglik_continuous(y, z, alpha, Z_tau, sigmasq);
      }
    } // end inner thinning for gamma + beta

    // -----------------------------------------------------------------
    // STEP C2: Lambda (Student-t latent scales) — Gibbs step
    // -----------------------------------------------------------------
    if (use_t_prior) {
      bvs_imbalanced::gibbs_update_lambda(lambda_t, beta_vec, gamma, beta0,
                                          sigmasq, t_df, t_scale);
    }

    // -----------------------------------------------------------------
    // STEP D: Alpha — MALA for binary/count; conjugate Gibbs for continuous.
    // -----------------------------------------------------------------
    {
      if (!is_continuous && !is_tte && !use_hmc_nuts) {
        // Refresh mala_resid (beta loop may have changed Xb_total)
        double sum_resid = 0.0;
        for (arma::uword i = 0; i < n; ++i) {
          double eta_i = alpha + Xb_total(i);
          if (use_cloglog) {
            mala_resid(i) = bvs_imbalanced::cloglog_residual(y(i), eta_i);
          } else if (is_count || is_zic) {
            if (is_zic && zinb_Z_latent(i) == 1u) { mala_resid(i) = 0.0; continue; }
            double mu_i = std::exp(bvs_dadj::clamp_finite(
                eta_i + log_w_count(i), -50.0, 50.0, 0.0));
            mala_resid(i) = (double)y_count(i) - mu_i;
          } else {
            double p_i = 1.0 / (1.0 + std::exp(-eta_i));
            mala_resid(i) = y(i) - p_i;
          }
          sum_resid += mala_resid(i);
        }
        double g_alpha = sum_resid - (alpha - alpha0) / (h * sigmasq);
        double h_alpha =
            std::min(0.5 * h * sigmasq / ((double)n + 1.0 / h), 1.0);
        double a_prop = alpha + 0.5 * h_alpha * g_alpha +
                        std::sqrt(h_alpha) * R::rnorm(0.0, 1.0);
        double ll_prop_alpha = 0.0;
        double g_alpha_back = -(a_prop - alpha0) / (h * sigmasq);
        for (arma::uword i = 0; i < n; ++i) {
          double new_eta = a_prop + Xb_total(i);
          if (is_count || is_zic) {
            if (is_zic && zinb_Z_latent(i) == 1u) continue;
            double mu_i = std::exp(bvs_dadj::clamp_finite(
                new_eta + log_w_count(i), -50.0, 50.0, 0.0));
            ll_prop_alpha +=
                (double)y_count(i) * (new_eta + log_w_count(i)) - mu_i;
            g_alpha_back += (double)y_count(i) - mu_i;
          } else if (use_cloglog) {
            ll_prop_alpha += bvs_imbalanced::loglik_cloglog_obs(y(i), new_eta);
            g_alpha_back += bvs_imbalanced::cloglog_residual(y(i), new_eta);
          } else {
            double p_i = 1.0 / (1.0 + std::exp(-new_eta));
            ll_prop_alpha +=
                y(i) * new_eta - (new_eta > 0.0
                                      ? new_eta + std::log1p(std::exp(-new_eta))
                                      : std::log1p(std::exp(new_eta)));
            g_alpha_back += y(i) - p_i;
          }
        }
        double a_back = a_prop + 0.5 * h_alpha * g_alpha_back;
        double pr_a_curr = -0.5 * std::pow(alpha - alpha0, 2) / (h * sigmasq);
        double pr_a_prop = -0.5 * std::pow(a_prop - alpha0, 2) / (h * sigmasq);
        double lq_fwd =
            -0.5 * std::pow(a_prop - (alpha + 0.5 * h_alpha * g_alpha), 2) /
            h_alpha;
        double lq_bwd = -0.5 * std::pow(alpha - a_back, 2) / h_alpha;
        if (bvs_dadj::safe_mh_accept((ll_prop_alpha - loglik) +
                                     (pr_a_prop - pr_a_curr) +
                                     (lq_bwd - lq_fwd))) {
          alpha = a_prop;
          loglik = ll_prop_alpha;
          // Update mala_resid to reflect new alpha
          for (arma::uword i = 0; i < n; ++i) {
            double new_eta = alpha + Xb_total(i);
            if (use_cloglog) {
              mala_resid(i) = bvs_imbalanced::cloglog_residual(y(i), new_eta);
            } else if (is_count || is_zic) {
              if (is_zic && zinb_Z_latent(i) == 1u) { mala_resid(i) = 0.0; continue; }
              double mu_i = std::exp(bvs_dadj::clamp_finite(
                  new_eta + log_w_count(i), -50.0, 50.0, 0.0));
              mala_resid(i) = (double)y_count(i) - mu_i;
            } else {
              mala_resid(i) = y(i) - 1.0 / (1.0 + std::exp(-new_eta));
            }
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
      // No alpha update for TTE as it's not identifiable
    }

    // -----------------------------------------------------------------
    // STEP D+: Tau — component-wise MALA (binary/count); block MH (TTE);
    //           exact conjugate Gibbs (continuous).
    // -----------------------------------------------------------------
    if (ntau > 0 && !use_hmc_nuts) {
      if (!is_continuous) {
        if (is_tte) {
          // Block MH for TTE (TTE loglik requires full re-init of CoxTracker)
          arma::vec tau_prop(ntau);
          for (arma::uword j = 0; j < ntau; ++j)
            tau_prop(j) = R::rnorm(tau(j), std::sqrt(htau * sigmasq));
          arma::vec Z_tau_prop = Z_dat * tau_prop;
          arma::vec Xb_total_prop = z + Z_tau_prop;
          bvs_dadj::CoxTracker cox_tracker_prop;
          cox_tracker_prop.init(Xb_total_prop, cox_data);
          double ll_diff = cox_tracker_prop.get_loglik() - loglik;
          double pr_tau_curr =
              -0.5 * arma::accu(arma::square(tau - tau0)) / (htau * sigmasq);
          double pr_tau_prop = -0.5 *
                               arma::accu(arma::square(tau_prop - tau0)) /
                               (htau * sigmasq);
          if (bvs_dadj::safe_mh_accept(ll_diff + (pr_tau_prop - pr_tau_curr))) {
            tau = tau_prop;
            Z_tau = Z_tau_prop;
            Xb_total = z + Z_tau;
            cox_tracker.init(Xb_total, cox_data);
            loglik = cox_tracker.get_loglik();
          }
        } else {
          // Component-wise MALA for binary/count/imbalanced_binary
          // Refresh mala_resid from current state
          for (arma::uword i = 0; i < n; ++i) {
            double eta_i = alpha + Xb_total(i);
            if (use_cloglog) {
              mala_resid(i) = bvs_imbalanced::cloglog_residual(y(i), eta_i);
            } else if (is_count || is_zic) {
              if (is_zic && zinb_Z_latent(i) == 1u) { mala_resid(i) = 0.0; continue; }
              double mu_i = std::exp(bvs_dadj::clamp_finite(
                  eta_i + log_w_count(i), -50.0, 50.0, 0.0));
              mala_resid(i) = (double)y_count(i) - mu_i;
            } else {
              mala_resid(i) = y(i) - 1.0 / (1.0 + std::exp(-eta_i));
            }
          }
          for (arma::uword k = 0; k < ntau; ++k) {
            const arma::vec &zk = Z_dat.col(k);
            double g_k =
                arma::dot(zk, mala_resid) - (tau(k) - tau0) / (htau * sigmasq);
            double h_k = std::min(
                0.5 * htau * sigmasq / (Z_col_sq_sums(k) + 1.0 / htau), 1.0);
            double t_prop =
                tau(k) + 0.5 * h_k * g_k + std::sqrt(h_k) * R::rnorm(0.0, 1.0);
            double dt = t_prop - tau(k);
            // Compute ll_prop and g_k_back in one pass
            double ll_prop = 0.0;
            double g_k_back = -(t_prop - tau0) / (htau * sigmasq);
            for (arma::uword i = 0; i < n; ++i) {
              double new_eta = alpha + Xb_total(i) + dt * zk(i);
              if (is_count || is_zic) {
                if (is_zic && zinb_Z_latent(i) == 1u) continue;
                double mu_i = std::exp(bvs_dadj::clamp_finite(
                    new_eta + log_w_count(i), -50.0, 50.0, 0.0));
                ll_prop +=
                    (double)y_count(i) * (new_eta + log_w_count(i)) - mu_i;
                g_k_back += ((double)y_count(i) - mu_i) * zk(i);
              } else if (use_cloglog) {
                ll_prop += bvs_imbalanced::loglik_cloglog_obs(y(i), new_eta);
                g_k_back += bvs_imbalanced::cloglog_residual(y(i), new_eta) * zk(i);
              } else {
                double p_i = 1.0 / (1.0 + std::exp(-new_eta));
                ll_prop +=
                    y(i) * new_eta -
                    (new_eta > 0.0 ? new_eta + std::log1p(std::exp(-new_eta))
                                   : std::log1p(std::exp(new_eta)));
                g_k_back += (y(i) - p_i) * zk(i);
              }
            }
            double t_back = t_prop + 0.5 * h_k * g_k_back;
            double pr_curr =
                -0.5 * std::pow(tau(k) - tau0, 2) / (htau * sigmasq);
            double pr_prop =
                -0.5 * std::pow(t_prop - tau0, 2) / (htau * sigmasq);
            double lq_fwd =
                -0.5 * std::pow(t_prop - (tau(k) + 0.5 * h_k * g_k), 2) / h_k;
            double lq_bwd = -0.5 * std::pow(tau(k) - t_back, 2) / h_k;
            if (bvs_dadj::safe_mh_accept((ll_prop - loglik) +
                                         (pr_prop - pr_curr) +
                                         (lq_bwd - lq_fwd))) {
              tau(k) = t_prop;
              Z_tau += dt * zk;
              Xb_total = z + Z_tau;
              loglik = ll_prop;
              // Update mala_resid
              for (arma::uword i = 0; i < n; ++i) {
                double new_eta = alpha + Xb_total(i);
                if (use_cloglog) {
                  mala_resid(i) = bvs_imbalanced::cloglog_residual(y(i), new_eta);
                } else if (is_count || is_zic) {
                  if (is_zic && zinb_Z_latent(i) == 1u) { mala_resid(i) = 0.0; continue; }
                  double mu_i = std::exp(bvs_dadj::clamp_finite(
                      new_eta + log_w_count(i), -50.0, 50.0, 0.0));
                  mala_resid(i) = (double)y_count(i) - mu_i;
                } else {
                  mala_resid(i) = y(i) - 1.0 / (1.0 + std::exp(-new_eta));
                }
              }
            }
          }
        }
      } else {
        for (arma::uword j = 0; j < ntau; ++j) {
          arma::vec zj = Z_dat.col(j);
          double tj_old = tau(j);
          arma::vec resid = y - alpha - z - Z_tau + tj_old * zj;
          double denom = arma::dot(zj, zj) + 1.0 / htau;
          double mean = (arma::dot(zj, resid) + tau0 / htau) / denom;
          double tj_new = R::rnorm(mean, std::sqrt(sigmasq / denom));
          tau(j) = tj_new;
          Z_tau += (tj_new - tj_old) * zj;
        }
        Xb_total = z + Z_tau; // Update Xb_total after Z_tau changes
        loglik =
            calc_loglik_continuous(y, Xb_total - Z_tau, alpha, Z_tau, sigmasq);
      }
    }

    // -----------------------------------------------------------------
    // STEP E: SigmaSq — exact Inverse-Gamma Gibbs for all outcome types.
    // For non-continuous outcomes the likelihood does not depend on sigmasq;
    // the conjugate IG posterior is exact — no MH step needed.
    // -----------------------------------------------------------------
    if (!use_hmc_nuts) {
      if (!is_continuous) {
        // Sufficient statistics
        double ss_beta = 0.0;
        for (arma::uword k = 0; k < active_idx.size(); ++k) {
          const arma::uword jj = active_idx[k];
          const double d = beta_vec(jj) - beta0;
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
          double d = beta_vec(active_idx[k]) - beta0;
          ss_beta += d * d;
        }
        double ss_tau = 0.0;
        for (arma::uword j = 0; j < ntau; ++j) {
          double d = tau(j) - tau0;
          ss_tau += d * d;
        }
        double da = alpha - alpha0;
        arma::vec resid = y - (alpha + z + Z_tau);
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
        loglik = calc_loglik_continuous(y, z, alpha, Z_tau, sigmasq);
      }
    }

    // -----------------------------------------------------------------
    // STEP F: Dual-eta Moller + Propp-Wilson
    // FIX 3, 4, 5: Proper exchange algorithm with complete priors
    // -----------------------------------------------------------------
    moller_update_dual(Z_active, R_fix_adj, (int)p, mu, eta1, eta2, eta1_sd,
                       eta2_sd, mu_tilde, eta1_tilde, eta2_tilde, gamma, e_eta,
                       f_eta, T_max, proposal_type, pw_x_up, pw_x_down, pw_om1,
                       pw_om2, pw_om1n, pw_om2n, eta1_adapter, eta2_adapter,
                       use_cftp);

    // OPT 2: Online PIP accumulation (post-burnin)
    if (iter >= burnin) {
      for (arma::uword j = 0; j < p; ++j)
        gamma_pip_sum(j) += (double)gamma(j);
    }

    // -----------------------------------------------------------------
    // STORE SAMPLES
    // -----------------------------------------------------------------
    if (iter >= burnin && (iter - burnin) % thin == 0) {
      int s = (iter - burnin) / thin;
      if (s < n_save) {
        beta_out.row(s) = beta_vec.t();
        if (store_gamma) {
          for (arma::uword j = 0; j < p; ++j)
            gamma_out(s, j) = gamma(j);
        }
        eta1_out(s) = eta1;
        eta2_out(s) = eta2;
        alpha_out(s) = alpha;
        sigmasq_out(s) = sigmasq;
        tau_out.row(s) = tau.t();

        edge_rows.clear();
        edge_cols.clear();
        for (arma::uword r = 0; r < p; ++r) {
          for (int c : Z_active[r]) {
            if (c > (int)r) {
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

    // FIX 6: Accumulate edge inclusion counts (post-burnin only)
    if (iter >= burnin) {
      for (arma::uword j = 0; j < p; ++j) {
        for (int nbr : Z_active[j]) {
          if (nbr > (int)j)
            Z_pip(j, (arma::uword)nbr) += 1.0;
        }
      }
    }
  } // end MCMC loop

  // Normalize Z_pip to posterior inclusion probabilities
  double n_post = (double)(total_iter - burnin);
  if (n_post > 0)
    Z_pip /= n_post;
  // Symmetrize
  Z_pip = Z_pip + Z_pip.t();

  // OPT 2: Compute final PIP vector
  arma::vec gamma_pip = (n_post > 0) ? gamma_pip_sum / n_post : gamma_pip_sum;

  Rcpp::List result = Rcpp::List::create(
      Rcpp::Named("beta") = beta_out, Rcpp::Named("gamma_pip") = gamma_pip,
      Rcpp::Named("eta1") = eta1_out, Rcpp::Named("eta2") = eta2_out,
      Rcpp::Named("alpha") = alpha_out, Rcpp::Named("sigmasq") = sigmasq_out,
      Rcpp::Named("Z_list") = Z_list, Rcpp::Named("Z_pip") = Z_pip,
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
