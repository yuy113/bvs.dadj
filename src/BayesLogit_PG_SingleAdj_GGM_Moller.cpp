// =============================================================================
// BayesLogit_PG_SingleAdj_GGM_Moller_refactored.cpp
//
// Refactored single-network BVS sampler:
//   - Polya-Gamma augmented logistic regression (joint Gibbs for beta)
//   - Wang (2012) Bayesian GGM (dense S_ggm) for dynamic adjacency
//   - Moller et al. (2006) + Propp-Wilson for eta1 MH update
//   - Single Ising prior: P(gamma|mu,eta1,R_ggm) propto
//       exp{ mu * sum(gamma_j) + eta1 * sum_{j<k in R} gamma_j*gamma_k }
//
// Key changes from original:
//   FIX  1: Removed redundant O(p^3) inv_sympd + C_ggm (never read)
//   FIX  2: Sparse Z_active replaces dense Z_ggm, R_ggm, R_ggm_int
//   FIX  3: Propp-Wilson uses pre-allocated std::vector (no Rcpp allocs)
//   FIX  4: PW + Moller use sparse adjacency — O(|E|) not O(p^2)
//   FIX  5: Moller sufficient stats use upper-triangle only (no double-count)
//   FIX  6: Consistent Ising parameterization (no extra factor of 2)
//   FIX  7: PG sampler K=20 + C++ RNG (was K=200 + R::rexp)
//   FIX  8: Z_pip (running avg) + sparse Z_list (edge indices only)
//   FIX  9: Thinning parameter added
//   FIX 10: sigmasq proposal floor
//   FIX 11: Incremental edge counter
//   FIX 12: Cached Xb = X*beta (eliminates redundant O(np) products)
//   FIX 13: Pre-allocated z_prop, active_idx, element-wise SSVS
//   FIX 14: Progressive Cholesky jittering
//   FIX 15: y {-1,1} -> {0,1} auto-conversion
//
// Memory: O(n*p + p^2 + |E|) — down from O(n*p + p^2 + niter*p^2/2)
// Per-iter: O(p^2) for GGM sweep, O(n*p_active) for PG beta, O(|E|) for Moller
//   (removed O(p^3) inv_sympd)
// =============================================================================

// [[Rcpp::depends(RcppArmadillo)]]
#include "BayesLogit_BlockPG.h"
#include "BayesLogit_Numerics.h"
#include "BayesLogit_Sparse_Helpers.h"
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>
#include <vector>

// =============================================================================
// SECTION 1: HELPERS
// =============================================================================

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

static inline arma::vec solve_spd_chol_upper(const arma::mat &U,
                                             const arma::vec &b) {
  arma::vec y = arma::solve(arma::trimatl(U.t()), b, arma::solve_opts::fast);
  return arma::solve(arma::trimatu(U), y, arma::solve_opts::fast);
}

static inline void rnorm_into(arma::vec &out) {
  for (arma::uword i = 0; i < out.n_elem; ++i)
    out(i) = R::rnorm(0.0, 1.0);
}

// --- Logistic log-likelihood (numerically stable) ---
static inline double calc_loglik(const arma::vec &y, const arma::vec &z,
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

// log_beta_pdf, normal_pdf, approx_erf, normal_cdf provided by BayesLogit_Sparse_Helpers.h

// =============================================================================
// SECTION 2: POLYA-GAMMA(1, z) SAMPLER
//
// FIX 7: K=20 (tail < 1e-8 for |z|<10), C++ RNG (avoids R hook overhead).
// Net: 90% fewer RNG calls (200n -> 20n per iteration).
// =============================================================================
// PG_K now provided by BayesLogit_Sparse_Helpers.h

// PSGM-5: Use R::rexp() instead of C++ mt19937 for PG sampler.
// This ensures results are reproducible via R's set.seed(), matching
// the convention used by all other backends.
static double sample_pg_approx(double z) {
  z = std::abs(z) * 0.5;
  const double c2 = z * z;
  const double PI2 = M_PI * M_PI;
  const double c2_over_pi2 = c2 / PI2;
  const double INV_2PI2 = 1.0 / (2.0 * PI2);
  double sum = 0.0;
  for (int k = 0; k < PG_K; ++k) {
    double g = R::rexp(1.0);
    double kh = k + 0.5;
    double den = kh * kh + c2_over_pi2;
    sum += g / den;
  }
  return std::max(1e-6, sum * INV_2PI2);
}

// =============================================================================
// SECTION 3: PROPP-WILSON (1-eta, sparse)
//
// FIX 3: Pre-allocated std::vector workspace (no Rcpp heap allocation).
// FIX 4: Sparse adjacency — O(|E|) per sweep instead of O(p^2).
// =============================================================================
static void
proppwilson_sparse_1eta(const std::vector<std::unordered_set<int>> &Z_active,
                        int p, double mu, double eta1, unsigned int T_max,
                        std::vector<int> &x_up, std::vector<int> &x_down,
                        std::vector<int> &result,
                        bool *coalesced = nullptr) {
  unsigned int T = 2;
  int seed_base = static_cast<int>(std::floor(R::runif(0.0, 1.0) * 2147483646.0)) + 1;

  auto not_coalesced = [&]() {
    for (int k = 0; k < p; ++k)
      if (x_up[k] != x_down[k])
        return true;
    return false;
  };

  for (int k = 0; k < p; ++k) {
    x_up[k] = 0;
    x_down[k] = 1;
  }

  while (not_coalesced()) {
    for (int k = 0; k < p; ++k) {
      x_up[k] = 0;
      x_down[k] = 1;
    }

    for (int t = -(int)T; t <= -1; ++t) {
      int seed2 = -t * seed_base;
      std::mt19937 gen(seed2);
      std::uniform_real_distribution<double> unif01(0.0, 1.0);

      for (int i = 0; i < p; ++i) {
        double ker_up = 0.0, ker_down = 0.0;
        for (int j : Z_active[i]) {
          if (x_up[j])
            ker_up += eta1;
          if (x_down[j])
            ker_down += eta1;
        }
        double pi_up = 1.0 / (1.0 + std::exp(-(mu + ker_up)));
        double pi_down = 1.0 / (1.0 + std::exp(-(mu + ker_down)));
        double u = unif01(gen);
        x_up[i] = (pi_up > u) ? 1 : 0;
        x_down[i] = (pi_down > u) ? 1 : 0;
      }
    }

    T = 2 * T;
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
          double prob = 1.0 / (1.0 + std::exp(-(mu + ker)));
          x_up[m] = (unif01_fb(gen_fb) < prob) ? 1 : 0;
        }
      }
      for (int k = 0; k < p; ++k)
        result[k] = x_up[k];
      if (coalesced) *coalesced = false; // IMP-2: Signal CFTP failure
      return; // early exit: skip normal coalescence path
    }
  }
  if (coalesced) *coalesced = true; // IMP-2: Exact coalescence
  for (int k = 0; k < p; ++k)
    result[k] = x_up[k];
}

// =============================================================================
// SECTION 4: MOLLER MH UPDATE (1-eta, sparse)
//
// FIX 5: Sufficient stats over upper triangle only (no double-count).
// FIX 6: Consistent parameterization with gamma MH step.
// =============================================================================
static void moller_update_1eta_sparse(
    const std::vector<std::unordered_set<int>> &Z_active, int p, double mu,
    double &eta1, double eta1_sd, double mu_tilde, double eta1_tilde,
    const arma::uvec &gamma, double e_eta, double f_eta, unsigned int T_max,
    int proposal_type, std::vector<int> &pw_x_up, std::vector<int> &pw_x_down,
    std::vector<int> &om1, std::vector<int> &om1n,
    EtaAdapter &adapter1) {
  // Exactly cancel the Moller effect to implement the Exchange Algorithm
  mu_tilde = mu;
  eta1_tilde = eta1;

  double eta1_new;
  double log_prop_ratio = 0.0;

  // IMP-1: Logit-transformed proposal for eta1 with Vihola RAM.
  double eta1_safe = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1));
  double phi1 = std::log(eta1_safe / (eta1_sd - eta1_safe));
  double phi1_new = R::rnorm(phi1, adapter1.sigma());
  eta1_new = eta1_sd / (1.0 + std::exp(-phi1_new));
  eta1_new = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1_new));
  log_prop_ratio = std::log(eta1_new) + std::log(eta1_sd - eta1_new) -
                   std::log(eta1_safe) - std::log(eta1_sd - eta1_safe);

  // 2 PW calls for single eta
  // IMP-2: Track CFTP coalescence; reject proposal if any call fails.
  bool c1 = true, c2 = true;
  proppwilson_sparse_1eta(Z_active, p, mu, eta1, T_max, pw_x_up, pw_x_down,
                          om1, &c1);
  proppwilson_sparse_1eta(Z_active, p, mu, eta1_new, T_max, pw_x_up, pw_x_down,
                          om1n, &c2);
  if (!c1 || !c2) { adapter1.update(0.0); return; } // Reject: preserve chain exactness

  // FIX 5: Sufficient stats — upper triangle only, O(|E|)
  // L-3: use double to prevent int overflow for large dense graphs
  double sum_om1 = 0.0, sum_om1n = 0.0;
  for (int j = 0; j < p; ++j) {
    sum_om1 += om1[j];
    sum_om1n += om1n[j];
  }

  double B_R1 = 0.0, A_om1_R1 = 0.0, A_om1n_R1 = 0.0;
  for (int j = 0; j < p; ++j) {
    for (int k : Z_active[j]) {
      if (k <= j)
        continue; // upper triangle only
      B_R1 += (double)gamma(j) * (double)gamma(k);
      A_om1_R1 += (double)om1[j] * (double)om1[k];
      A_om1n_R1 += (double)om1n[j] * (double)om1n[k];
    }
  }

  double log_prior = log_beta_pdf(eta1_new / eta1_sd, e_eta, f_eta) -
                     log_beta_pdf(eta1 / eta1_sd, e_eta, f_eta);
  double log_target = (eta1_new - eta1) * B_R1 + log_prior;
  double log_aux =
      mu_tilde * (sum_om1n - sum_om1) + eta1_tilde * (A_om1n_R1 - A_om1_R1);
  double log_norm =
      mu * (sum_om1 - sum_om1n) + eta1 * A_om1_R1 - eta1_new * A_om1n_R1;

  double log_MH = log_target + log_aux + log_norm + log_prop_ratio;

  double accept_prob = std::min(1.0, std::exp(std::min(0.0, log_MH)));
  if (bvs_dadj::safe_mh_accept(log_MH))
    eta1 = eta1_new;
  adapter1.update(accept_prob);
}

// =============================================================================
// SECTION 5: PHASE TRANSITION DETECTION (exported utility)
// =============================================================================
// [[Rcpp::export]]
Rcpp::IntegerMatrix phase_transit_1eta(Rcpp::IntegerMatrix R1, int T_max,
                                       double mu, double min_eta,
                                       double max_eta, unsigned int num_rep,
                                       double step_size = 0.01) {
  int p = R1.ncol();

  // Build sparse adjacency from dense input
  std::vector<std::unordered_set<int>> Z_adj(p);
  for (int i = 0; i < p; ++i)
    for (int j = 0; j < p; ++j)
      if (i != j && R1(i, j) != 0)
        Z_adj[i].insert(j);

  unsigned int len_eta =
      static_cast<unsigned int>((max_eta - min_eta) / step_size) + 1;
  Rcpp::IntegerMatrix gamma_output(len_eta, num_rep);

  std::vector<int> pw_x_up(p), pw_x_down(p), result(p);
  double eta_tmp = min_eta;
  for (unsigned int i = 0; i < len_eta; ++i) {
    for (unsigned int j = 0; j < num_rep; ++j) {
      proppwilson_sparse_1eta(Z_adj, p, mu, eta_tmp, T_max, pw_x_up, pw_x_down,
                              result);
      int s = 0;
      for (int k = 0; k < p; ++k)
        s += result[k];
      gamma_output(i, j) = s;
    }
    eta_tmp += step_size;
  }
  return gamma_output;
}

// =============================================================================
// SECTION 6: MAIN MCMC SAMPLER
//
// Memory: O(n*p + p^2 + |E| + n_save*avg_edges)  (sparse Z_list)
// Per-iter: O(p^2) GGM sweep + O(n*p_active) PG beta + O(|E|) Moller
// =============================================================================

// [[Rcpp::export]]
Rcpp::List BayesLogit_PG_SingleAdj_GGM_Moller(
    const arma::mat &X, const arma::vec &y, const arma::mat &S_ggm,
    double n_ggm, const arma::mat &Z_dat, int niter, int burnin, double mu,
    double nu0, double sigmasq0, double alpha0, double beta0, double h,
    int n_mh_gamma, double v0_ggm, double v1_ggm, double pii_ggm,
    double lambda_ggm, double eta1_sd, double mu_tilde, double eta1_tilde,
    double e_eta, double f_eta, unsigned int T_max, int proposal_type,
    int thin = 1, Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0, double tau0 = 0.0, double htau = 1.5,
    Rcpp::Nullable<Rcpp::NumericVector> tau_in = R_NilValue, int block_size = 1,
    int pcg_threshold = 500, bool use_lb_gamma = true) {

  Rcpp::RNGScope scope;

  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  if (S_ggm.n_cols != p || S_ggm.n_rows != p)
    Rcpp::stop("S_ggm dimensions must match p = %d", (int)p);
  if (thin < 1)
    thin = 1;
  if (n_mh_gamma < 1)
    n_mh_gamma = 1;

  // --- Enforce y in {0,1}; auto-convert {-1,1} ---  (FIX 15)
  arma::vec y01 = y;
  bool is01 = true, is11 = true;
  for (arma::uword i = 0; i < y01.n_elem; ++i) {
    double yi = y01(i);
    if (std::fabs(yi) > 1e-12 && std::fabs(yi - 1.0) > 1e-12)
      is01 = false;
    if (std::fabs(yi + 1.0) > 1e-12 && std::fabs(yi - 1.0) > 1e-12)
      is11 = false;
  }
  if (!is01 && is11) {
    y01 = 0.5 * (y01 + 1.0);
    Rcpp::Rcout << "Note: y in {-1,1}; converted to {0,1}.\n";
  } else if (!is01) {
    Rcpp::stop("y must be binary in {0,1} (or {-1,1}).");
  }

  // =========================================================================
  // 1. INITIALIZATION
  // =========================================================================
  arma::vec beta_vec(p, arma::fill::zeros);
  if (beta_in.isNotNull()) {
    beta_vec = Rcpp::as<arma::vec>(beta_in);
  }

  arma::uvec gamma(p, arma::fill::zeros);
  if (gamma_in.isNotNull()) {
    gamma = Rcpp::as<arma::uvec>(gamma_in);
  } else {
    // If gamma not provided, infer from beta or start full/empty?
    // Default to empty if beta is zero, or infer.
    for (arma::uword j = 0; j < p; ++j) {
      if (std::abs(beta_vec(j)) > 1e-6)
        gamma(j) = 1;
    }
  }
  double alpha = alpha_in;
  double sigmasq = 1.0;
  double eta1 = std::min(0.01, eta1_sd * 0.5);

  // --- tau (Z_dat covariates) ---
  const arma::uword ntau = Z_dat.n_cols;
  arma::vec tau(ntau, arma::fill::zeros);
  tau.fill(tau0);
  if (tau_in.isNotNull()) {
    tau = Rcpp::as<arma::vec>(tau_in);
  }
  arma::vec Z_tau = Z_dat * tau;

  arma::vec Xb = X * beta_vec; // FIX 12: cached X*beta
  arma::vec omega_pg(n, arma::fill::ones);
  const arma::vec kappa = y01 - 0.5;

  // --- GGM state: sparse edge representation ---  (FIX 2)
  arma::mat Sig_ggm = arma::diagmat(S_ggm.diag());
  std::vector<std::unordered_set<int>> Z_active(p);
  int n_edges = 0; // FIX 11: incremental counter

  // --- GGM pre-allocated work buffers ---
  const arma::uword pm1 = p - 1;
  arma::mat invC11(pm1, pm1), Ci(pm1, pm1), U_ggm(pm1, pm1);
  arma::vec Sig12(pm1), S_i(pm1), tau_temp(pm1);
  arma::vec mu_i(pm1), b_ggm(pm1), eps_ggm(pm1);
  arma::vec invC11beta(pm1), Sig12_new(pm1);

  // Pre-computed SSVS constants
  const double log_pii = std::log(pii_ggm);
  const double log_1m_pii = std::log(1.0 - pii_ggm);
  const double log_v0_half = -0.5 * std::log(v0_ggm);
  const double log_v1_half = -0.5 * std::log(v1_ggm);
  const double inv_v0 = 1.0 / v0_ggm;
  const double inv_v1 = 1.0 / v1_ggm;
  const double a_gam = 0.5 * n_ggm + 1.0;

  // --- Logistic pre-allocated buffers ---
  arma::vec z_prop(n); // FIX 13
  arma::vec lin(n);
  std::vector<arma::uword> active_idx;
  active_idx.reserve(p);

  // --- PW pre-allocated workspace (2 calls for single eta) ---
  std::vector<int> pw_x_up(p), pw_x_down(p);
  std::vector<int> pw_om1(p), pw_om1n(p);

  // PSGM-5: Removed C++ mt19937 RNG; now uses R::rexp() in sample_pg_approx.
  // This ensures reproducibility via R's set.seed().

  // --- Locally-balanced gamma proposal state (Zanella 2020) ---
  Rcpp::IntegerMatrix Z_ggm_int(p, p);  // dense snapshot for LB helpers
  std::vector<double> lb_score, lb_weight;
  double lb_Z = 0.0;
  LBProposalDelta lb_delta;

  // --- Output storage ---
  int n_save = niter / thin;
  arma::mat beta_out(n_save, p);
  arma::umat gamma_out(n_save, p);
  arma::vec eta1_out(n_save);
  arma::vec alpha_out(n_save), sigmasq_out(n_save);
  arma::mat tau_out(n_save, ntau);

  // FIX 8: Running posterior edge inclusion probability
  arma::mat Z_pip(p, p, arma::fill::zeros);

  // Sparse Z_list: each entry stores only (row, col) of non-zero edges
  // in upper triangle. Memory: O(n_save * avg_edges) instead of O(n_save *
  // p^2).
  Rcpp::List Z_list(n_save);
  std::vector<int> edge_rows, edge_cols;
  edge_rows.reserve(std::max(1000, n_edges));
  edge_cols.reserve(std::max(1000, n_edges));

  // =========================================================================
  // 2. MCMC LOOP
  // =========================================================================
  EtaAdapter eta1_adapter(0.5);  // Vihola RAM for eta1 proposal

  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {

    if (iter > 0 && iter % 5000 == 0) {
      Rcpp::checkUserInterrupt();
      int model_size = (int)active_idx.size();
      Rcpp::Rcout << "Iter: " << iter << " | Model: " << model_size
                  << " | Edges: " << n_edges << " | eta1: " << eta1 << "\n";
    }

    double sd_sig = std::sqrt(sigmasq);

    // -----------------------------------------------------------------
    // STEP A: Wang (2012) GGM Block-Gibbs Column Sweep
    //
    // FIX 1: Removed inv_sympd + C_ggm (saves O(p^3) per iteration).
    // FIX 13: Element-wise SSVS with incremental edge tracking.
    // FIX PSGM-1: Periodic numerical reset of Sig_ggm every 100 iterations
    //   to prevent floating-point drift from Schur complement updates.
    // -----------------------------------------------------------------
    // IMP-4: Dimension-adaptive reset frequency: max(20, 5000/p).
    const int ggm_reset_freq = std::max(20, (int)(5000 / p));
    if (iter > 0 && iter % ggm_reset_freq == 0) {
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

      // Build tau vector from current edge state
      for (arma::uword k = 0, idx = 0; k < p; ++k) {
        if (k == i)
          continue;
        tau_temp(idx++) = (Z_active[i].count((int)k) > 0) ? v1_ggm : v0_ggm;
      }

      double Sigii = Sig_ggm(i, i);
      double s_ii = S_ggm(i, i);

      // Schur complement: C_{-i,-i}^{-1}
      invC11 -= (1.0 / Sigii) * (Sig12 * Sig12.t());

      // Posterior precision
      Ci = (s_ii + lambda_ggm) * invC11;
      Ci.diag() += 1.0 / tau_temp;

      // FIX 14: Progressive Cholesky jittering
      bool ok = arma::chol(U_ggm, arma::symmatu(Ci));
      if (!ok) {
        double jitter = 1e-8;
        for (int attempt = 0; attempt < 5 && !ok; ++attempt) {
          Ci.diag() += jitter;
          ok = arma::chol(U_ggm, arma::symmatu(Ci));
          jitter *= 10.0;
        }
        if (!ok)
          continue;
      }

      // Sample off-diagonal concentration
      mu_i = -solve_spd_chol_upper(U_ggm, S_i);
      rnorm_into(eps_ggm);
      arma::vec delta =
          arma::solve(arma::trimatu(U_ggm), eps_ggm, arma::solve_opts::fast);
      b_ggm = mu_i + delta;

      // Sample diagonal concentration
      double gam_val = R::rgamma(a_gam, 2.0 / (s_ii + lambda_ggm));

      // Update Sig_ggm via rank-1 formulas
      invC11beta = invC11 * b_ggm;
      Sig12_new = -invC11beta / gam_val;
      invC11 += (1.0 / gam_val) * (invC11beta * invC11beta.t());

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

      // --- SSVS Edge Selection: element-wise with incremental tracking ---
      {
        arma::uword k = 0;
        for (arma::uword j = 0; j < p; ++j) {
          if (j == i)
            continue;

          double bk2 = b_ggm(k) * b_ggm(k);
          double w1 = log_v0_half - 0.5 * bk2 * inv_v0 + log_1m_pii;
          double w2 = log_v1_half - 0.5 * bk2 * inv_v1 + log_pii;
          double wm = std::max(w1, w2);
          double prob =
              std::exp(w2 - wm) / (std::exp(w1 - wm) + std::exp(w2 - wm));

          bool was_active = (Z_active[i].count((int)j) > 0);
          bool now_active = (R::runif(0, 1) < prob);

          // FIX 11: Incremental edge counter
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
      init_lb_single_scores_ggm(Z_ggm_int, (int)p, gamma_iv, mu, eta1,
                                 lb_score, lb_weight, lb_Z);
    }

    // -----------------------------------------------------------------
    // STEP B: Polya-Gamma Augmentation + Gibbs Beta / Alpha
    //
    // FIX 7: PG K=20 + C++ RNG.
    // FIX 12: Uses cached Xb.
    // FIX 14: Progressive Cholesky jittering.
    // -----------------------------------------------------------------
    lin = alpha + Xb + Z_tau;
    for (arma::uword i = 0; i < n; ++i) {
      omega_pg(i) = sample_pg_approx(lin(i));
      // PSGM-4: Clamp omega_pg to [1e-8, 1e6] to prevent degenerate prec
      omega_pg(i) = std::max(1e-8, std::min(1e6, omega_pg(i)));
    }

    // Build active indices (manual, no arma::find)
    active_idx.clear();
    for (arma::uword j = 0; j < p; ++j)
      if (gamma(j) == 1)
        active_idx.push_back(j);
    const arma::uword p_active = (arma::uword)active_idx.size();

    // Block Gibbs update for beta (active variables)
    if (p_active > 0) {
      arma::uvec active_uv(active_idx.data(), p_active, false, true);
      arma::mat X_act = X.cols(active_uv);

      if (block_size > 1 && (int)p_active > pcg_threshold) {
        // PCG path for large active sets
        bvs_dadj_block::PCGConfig pcg_cfg(1e-4, 200, pcg_threshold);
        arma::vec beta_act = beta_vec.elem(active_uv);
        bool pcg_ok = bvs_dadj_block::pcg_sample_beta(
            beta_act, X_act, omega_pg, kappa, alpha, 1.0 / sigmasq, pcg_cfg,
            &Z_tau, beta0);
        if (pcg_ok && beta_act.n_elem == p_active) {
          beta_vec.zeros();
          beta_vec.elem(active_uv) = beta_act;
        }
      } else {
        // Original Cholesky path
        arma::mat Xt_Om = X_act.t();
        Xt_Om.each_row() %= omega_pg.t();
        arma::mat prec_beta = Xt_Om * X_act;
        prec_beta.diag() += 1.0 / sigmasq;

        arma::vec z_star = kappa - omega_pg * alpha - omega_pg % Z_tau;
        // FIX: Include beta0 prior mean in RHS (was missing)
        arma::vec mean_rhs = X_act.t() * z_star +
            (beta0 / sigmasq) * arma::ones<arma::vec>(p_active);

        arma::mat L_prec;
        bool chol_ok = arma::chol(L_prec, arma::symmatu(prec_beta));
        if (!chol_ok) {
          double jitter = 1e-8;
          for (int attempt = 0; attempt < 5 && !chol_ok; ++attempt) {
            prec_beta.diag() += jitter;
            chol_ok = arma::chol(L_prec, arma::symmatu(prec_beta));
            jitter *= 10.0;
          }
        }
        if (chol_ok) {
          arma::vec m_beta = arma::solve(arma::trimatl(L_prec.t()), mean_rhs,
                                         arma::solve_opts::fast);
          m_beta = arma::solve(arma::trimatu(L_prec), m_beta,
                               arma::solve_opts::fast);
          arma::vec zz = arma::randn<arma::vec>(p_active);
          arma::vec b_draw = m_beta + arma::solve(arma::trimatu(L_prec), zz,
                                                  arma::solve_opts::fast);
          beta_vec.zeros();
          beta_vec.elem(active_uv) = b_draw;
        }
      }
    } else {
      beta_vec.zeros();
    }

    // FIX 12: Recompute Xb once after beta update
    Xb = X * beta_vec;

    // Intercept alpha (conjugate Gibbs)
    {
      double sum_omega = arma::accu(omega_pg);
      double prec_alpha = sum_omega + 1.0 / (h * sigmasq);
      double var_alpha = 1.0 / prec_alpha;
      arma::vec resid =
          kappa - omega_pg % (Xb + Z_tau); // FIX 12: uses cached Xb
      double mean_alpha =
          var_alpha * (arma::accu(resid) + alpha0 / (h * sigmasq));
      alpha = R::rnorm(mean_alpha, std::sqrt(var_alpha));
      // PSGM-3: Clamp alpha to prevent extreme intercept values
      alpha = std::max(-60.0, std::min(60.0, alpha));
    }

    // -----------------------------------------------------------------
    // STEP C: Gamma via MH with single Ising prior (eta1 x GGM)
    //
    // FIX 6: Consistent parameterization (no extra factor of 2).
    //   E(gamma) = eta1 * sum_{j<k} R(j,k)*gamma_j*gamma_k
    //   => P(gamma_j=1|rest) = sigma(mu + eta1*sum_{k~j} gamma_k)
    // FIX 13: Pre-allocated z_prop.
    // -----------------------------------------------------------------
    if (block_size > 1) {
      // Block update: SW + Uncollapsed Gibbs (single GGM adjacency)
      auto gamma_u8 = bvs_dadj_block::gamma_to_uint8(gamma);
      auto neigh_fn = [&](int jj, std::function<void(int)> cb) {
        for (int nbr : Z_active[jj])
          cb(nbr);
      };
      auto proposal = bvs_dadj_block::swendsen_wang_single(
          gamma_u8, eta1, (int)p, block_size, neigh_fn);
      auto block = bvs_dadj_block::flatten_clusters(proposal);
      if (!block.empty()) {
        bvs_dadj_block::uncollapsed_gamma_sweep_single(
            gamma_u8, beta_vec, Xb, X, y01, alpha, sigmasq, beta0, mu, eta1,
            block, neigh_fn);
        bvs_dadj_block::uint8_to_gamma(gamma, gamma_u8);
      }
    } else {
      // Original single-variable MH
      arma::vec z = Xb;
      double loglik = calc_loglik(y01, z, alpha, Z_tau);

      // L-4: Build gamma_iv once before the MH loop; update incrementally on accept.
      arma::ivec gamma_iv_mh;
      if (use_lb_gamma) {
        gamma_iv_mh.set_size(p);
        for (arma::uword jj = 0; jj < p; ++jj) gamma_iv_mh(jj) = (int)gamma(jj);
      }

      for (int mh = 0; mh < n_mh_gamma; ++mh) {
        int j;
        if (use_lb_gamma) {
          j = sample_weighted_index(lb_weight, lb_Z, (int)p);
        } else {
          j = static_cast<int>(std::floor(R::runif(0.0, (double)p)));
          if (j >= (int)p) j = (int)p - 1;
        }

        int g_curr = (int)gamma(j);
        int g_prop = 1 - g_curr;
        double b_curr = beta_vec(j);
        double b_prop = (g_prop == 1) ? R::rnorm(beta0, sd_sig) : 0.0;

        z_prop = z + (b_prop - b_curr) * X.col(j);
        double ll_prop = calc_loglik(y01, z_prop, alpha, Z_tau);

        double diff = (double)(g_prop - g_curr);

        double neigh_ggm = 0.0;
        for (int nbr : Z_active[j])
          neigh_ggm += gamma(nbr);

        double ising_diff = diff * (mu + eta1 * neigh_ggm);
        double log_ratio = (ll_prop - loglik) + ising_diff;

        if (use_lb_gamma) {
          int delta_g = 1 - 2 * g_curr;
          build_lb_single_delta_ggm(Z_ggm_int, (int)p, gamma_iv_mh, eta1,
                                     j, delta_g, lb_score, lb_weight, lb_Z, lb_delta);
          log_ratio += lb_delta.log_q_rev - lb_delta.log_q_fwd;
        }

        if (bvs_dadj::safe_mh_accept(log_ratio)) {
          gamma(j) = g_prop;
          beta_vec(j) = b_prop;
          z = z_prop;
          loglik = ll_prop;
          if (use_lb_gamma) {
            gamma_iv_mh(j) = g_prop; // L-4: incremental update
            apply_lb_delta(lb_score, lb_weight, lb_Z, lb_delta);
          }
        }
      }

      Xb = z;
    }

    // -----------------------------------------------------------------
    // STEP C.5: tau Gibbs step =======================
    if (ntau > 0) {
      arma::mat Zt_Om = Z_dat.t();
      Zt_Om.each_row() %= omega_pg.t();
      arma::mat prec_tau = Zt_Om * Z_dat;
      prec_tau.diag() += 1.0 / (htau * sigmasq);

      arma::vec z_star_tau = kappa - omega_pg * alpha - omega_pg % Xb;
      arma::vec mean_rhs_tau =
          Z_dat.t() * z_star_tau +
          (tau0 / (htau * sigmasq)) * arma::ones<arma::vec>(ntau);

      arma::mat L_tau;
      bool tau_ok = arma::chol(L_tau, arma::symmatu(prec_tau));
      if (!tau_ok) {
        double jitter = 1e-8;
        for (int attempt = 0; attempt < 5 && !tau_ok; ++attempt) {
          prec_tau.diag() += jitter;
          tau_ok = arma::chol(L_tau, arma::symmatu(prec_tau));
          jitter *= 10.0;
        }
      }
      if (tau_ok) {
        arma::vec m_tau = arma::solve(arma::trimatl(L_tau.t()), mean_rhs_tau,
                                      arma::solve_opts::fast);
        m_tau =
            arma::solve(arma::trimatu(L_tau), m_tau, arma::solve_opts::fast);
        arma::vec zz_tau = arma::randn<arma::vec>(ntau);
        arma::vec noise_tau =
            arma::solve(arma::trimatu(L_tau), zz_tau, arma::solve_opts::fast);
        tau = m_tau + noise_tau;
        Z_tau = Z_dat * tau;
      }
    }

    // -----------------------------------------------------------------
    // STEP D: SigmaSq — exact Inverse-Gamma Gibbs.
    // PG logistic likelihood does not depend on sigmasq; the conjugate
    // IG posterior for the (beta, alpha, tau) priors is exact.
    // -----------------------------------------------------------------
    {
      // Rebuild active_idx after gamma MH (gamma may have changed)
      active_idx.clear();
      for (arma::uword j = 0; j < p; ++j)
        if (gamma(j) == 1)
          active_idx.push_back(j);

      double ss_b = 0.0;
      for (arma::uword k = 0; k < active_idx.size(); ++k) {
        double d = beta_vec(active_idx[k]) - beta0;
        ss_b += d * d;
      }
      double ss_tau = 0.0;
      for (arma::uword j = 0; j < ntau; ++j) {
        double d = tau(j) - tau0;
        ss_tau += d * d;
      }
      const double da = alpha - alpha0;
      const double shape_post =
          0.5 * nu0 + 0.5 * ((double)active_idx.size() + 1.0 + (double)ntau);
      const double scale_post =
          0.5 * nu0 * sigmasq0 + 0.5 * (ss_b + da * da / h + ss_tau / htau);
      const double gdraw =
          R::rgamma(shape_post, 1.0 / std::max(scale_post, 1e-12));
      if (std::isfinite(gdraw) && gdraw > 0.0)
        // PSGM-4: Clamp sigmasq to [1e-10, 1e10] to prevent degenerate prec
        sigmasq = std::max(1e-10, std::min(1e10, 1.0 / gdraw));
    }

    // -----------------------------------------------------------------
    // STEP E: eta1 via Moller + sparse Propp-Wilson (single eta)
    // FIX 3,4,5: Sparse PW, upper-triangle sufficient stats.
    // -----------------------------------------------------------------
    moller_update_1eta_sparse(
        Z_active, (int)p, mu, eta1, eta1_sd, mu_tilde, eta1_tilde, gamma, e_eta,
        f_eta, T_max, proposal_type, pw_x_up, pw_x_down, pw_om1, pw_om1n,
        eta1_adapter);

    // -----------------------------------------------------------------
    // STORE SAMPLES
    // -----------------------------------------------------------------
    if (iter >= burnin && (iter - burnin) % thin == 0) {
      int s = (iter - burnin) / thin;
      if (s < n_save) {
        beta_out.row(s) = beta_vec.t();
        for (arma::uword j = 0; j < p; ++j)
          gamma_out(s, j) = gamma(j);
        eta1_out(s) = eta1;
        alpha_out(s) = alpha;
        sigmasq_out(s) = sigmasq;
        tau_out.row(s) = tau.t();

        // Sparse Z_list: store only (row, col) of active edges (upper tri)
        // FIX 8: Also accumulate Z_pip at thinned save points only
        edge_rows.clear();
        edge_cols.clear();
        for (arma::uword c = 0; c < p; ++c) {
          for (int r : Z_active[c]) {
            if (r < (int)c) {
              Z_pip((arma::uword)r, c) += 1.0;
              edge_rows.push_back(r);
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
  } // end MCMC loop

  // Normalize Z_pip to posterior inclusion probabilities
  // Divide by n_save (number of thinned saves) to match accumulation convention
  double n_post = (double)n_save;
  if (n_post > 0)
    Z_pip /= n_post;
  Z_pip = Z_pip + Z_pip.t(); // symmetrize

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_out, Rcpp::Named("gamma") = gamma_out,
      Rcpp::Named("eta1") = eta1_out, Rcpp::Named("alpha") = alpha_out,
      Rcpp::Named("sigmasq") = sigmasq_out, Rcpp::Named("tau") = tau_out,
      Rcpp::Named("Z_pip") = Z_pip, Rcpp::Named("Z_list") = Z_list);
}
