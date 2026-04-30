// [[Rcpp::depends(RcppArmadillo)]]
#include "BayesLogit_BlockPG.h"
#include "BayesLogit_Numerics.h"
#include "BayesLogit_Sparse_Helpers.h"
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {

// =============================================================================
// Polya-Gamma(1, z) approximate sampler (truncated series, K=200 terms)
// PG(1,z) =_d (1/(2 pi^2)) sum_{k=0}^inf G_k / ((k+0.5)^2 + (z/(2pi))^2)
// =============================================================================
static double sample_pg_approx(double z) {
  if (!std::isfinite(z))
    return 0.25;
  z = bvs_dadj::clamp_finite(std::abs(z) * 0.5, 0.0, 60.0, 0.0);
  const double c2 = z * z;
  const double PI2 = M_PI * M_PI;
  const double INV_2PI2 = 1.0 / (2.0 * PI2);
  double sum = 0.0;
  for (int k = 0; k < 200; ++k) {
    double g = R::rexp(1.0);
    double kh = k + 0.5;
    double den = kh * kh + c2 / PI2;
    den = bvs_dadj::clamp_finite(den, 1e-12, 1e12, 1.0);
    sum += g / den;
  }
  double out = sum * INV_2PI2;
  return bvs_dadj::clamp_finite(out, 1e-8, 1e6, 0.25);
}

// =============================================================================
// Log-likelihood for logistic regression (numerically stable)
// =============================================================================
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
// IMP-2: Added bool* coalesced output to signal CFTP success/failure.
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
          if (j == i) continue;  // FIX DNF-1: Skip self-loops in Ising kernel
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
            if (k == m) continue;  // FIX DNF-1: Skip self-loops in fallback Gibbs
            ker += eta1 * R1(m, k) * x_up[k] + eta2 * R2(m, k) * x_up[k];
          }
          double prob = 1.0 / (1.0 + std::exp(-(mu + ker)));
          x_up[m] = (unif01_fb(gen_fb) < prob) ? 1 : 0;
        }
      }
      // Force coalescence
      for (unsigned int k = 0; k < p; ++k)
        x_down[k] = x_up[k];
      if (coalesced) *coalesced = false; // IMP-2: Signal CFTP failure
      return x_up;
    }
  }

  if (coalesced) *coalesced = true; // IMP-2: Exact coalescence
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
//
// Parameters:
//   R1, R2      : two p x p adjacency matrices
//   mu          : MRF external field
//   eta1, eta2  : current coupling parameters
//   eta1_sd, eta2_sd : phase transition values (upper bounds for eta1, eta2)
//   mu_tilde, eta1_tilde, eta2_tilde : auxiliary MRF parameters for omega
//   gamma       : current inclusion indicators
//   e, f        : Beta prior hyperparameters
//   T_max       : max doubling time for Propp-Wilson
//   proposal_type : 0 = uniform, 1 = truncated normal
//
// Returns updated eta1, eta2
// =============================================================================
static void
moller_update_eta(const Rcpp::IntegerMatrix &R1, const Rcpp::IntegerMatrix &R2,
                  double mu, double &eta1, double &eta2, double eta1_sd,
                  double eta2_sd, double mu_tilde, double eta1_tilde,
                  double eta2_tilde, const arma::uvec &gamma, double e_eta,
                  double f_eta, unsigned int T_max, int proposal_type,
                  EtaAdapter &adapter1, EtaAdapter &adapter2) {
  // Exactly cancel the Moller effect to implement the Exchange Algorithm
  mu_tilde = mu;
  eta1_tilde = eta1;
  eta2_tilde = eta2;

  const unsigned int p = R1.ncol();

  // --- Propose new eta1, eta2 via logit-transform with Vihola RAM ---
  double eta1_new, eta2_new;
  double log_prop_ratio_eta1 = 0.0;
  double log_prop_ratio_eta2 = 0.0;

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

  // --- Generate auxiliary variables via Propp-Wilson ---
  // IMP-2: Track CFTP coalescence; reject proposal if any call fails.
  bool coal1 = true, coal2 = true, coal3 = true, coal4 = true;
  Rcpp::IntegerVector omega_eta1 =
      proppwilson_omega(R1, R2, mu, eta1, eta2, T_max, &coal1);
  Rcpp::IntegerVector omega_eta2 =
      proppwilson_omega(R1, R2, mu, eta1, eta2, T_max, &coal2);
  Rcpp::IntegerVector omega_eta1_new =
      proppwilson_omega(R1, R2, mu, eta1_new, eta2, T_max, &coal3);
  Rcpp::IntegerVector omega_eta2_new =
      proppwilson_omega(R1, R2, mu, eta1, eta2_new, T_max, &coal4);
  if (!coal1 || !coal2 || !coal3 || !coal4) return; // Reject: preserve chain exactness

  // --- Compute sufficient statistics (L-3: use double to prevent int overflow) ---
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

  // --- Prior ratio: log Beta(eta_new/eta_sd | e,f) - log Beta(eta/eta_sd |
  // e,f) ---
  double log_prior_eta1 = log_beta_pdf(eta1_new / eta1_sd, e_eta, f_eta) -
                          log_beta_pdf(eta1 / eta1_sd, e_eta, f_eta);
  double log_prior_eta2 = log_beta_pdf(eta2_new / eta2_sd, e_eta, f_eta) -
                          log_beta_pdf(eta2 / eta2_sd, e_eta, f_eta);

  // --- MH ratio (MÃ¶ller et al. 2006) ---
  // Target ratio for eta1: (eta1_new - eta1) * gamma'R1*gamma
  // Auxiliary normalizing constant ratio approximated by omega terms
  // Auxiliary MRF contribution with tilde parameters
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
// PHASE TRANSITION DETECTION
//
// Sweep eta from min_eta to max_eta, generate gamma via Propp-Wilson at each,
// return matrix of model sizes for plotting.
// =============================================================================

// [[Rcpp::export]]
Rcpp::IntegerMatrix
phase_transit_2eta_fixadj(Rcpp::IntegerMatrix R1, Rcpp::IntegerMatrix R2,
                          int T_max, double mu, double min_eta, double max_eta,
                          unsigned int num_rep, double step_size = 0.01) {
  if (min_eta > max_eta)
    Rcpp::stop("min_eta must be <= max_eta");
  unsigned int p = R1.ncol();
  unsigned int len_eta =
      static_cast<unsigned int>((max_eta - min_eta) / step_size) + 1;

  Rcpp::IntegerMatrix gamma_output(len_eta, num_rep);
  double eta_tmp = min_eta;

  for (unsigned int i = 0; i < len_eta; ++i) {
    for (unsigned int j = 0; j < num_rep; ++j) {
      Rcpp::IntegerVector g =
          proppwilson_omega(R1, R2, mu, eta_tmp, eta_tmp, T_max);
      int s = 0;
      for (unsigned int k = 0; k < p; ++k)
        s += g[k];
      gamma_output(i, j) = s;
    }
    eta_tmp += step_size;
  }

  return gamma_output;
}

// =============================================================================
// MAIN MCMC SAMPLER
//
// Bayesian variable selection for logistic regression with:
//   - Polya-Gamma data augmentation
//   - Dual-network Ising/MRF prior on gamma with TWO adjacency matrices:
//       R_glasso : estimated from graphical lasso (data-driven)
//       R_fix    : fixed from external pathway/biological knowledge
//   - Two coupling parameters eta1 (R_glasso), eta2 (R_fix)
//     updated via Moller et al. (2006) auxiliary variable MH
//     with Propp-Wilson perfect simulation
//   - Prior: eta_i / eta_i_sd ~ Beta(e, f)
// =============================================================================

// [[Rcpp::export]]
Rcpp::List BayesLogit_PG_DualAdj(
    const arma::mat &X, const arma::vec &y, Rcpp::IntegerMatrix R_glasso_int,
    Rcpp::IntegerMatrix R_fix_int, const arma::mat &Z_dat, int niter,
    int burnin, double mu, double nu0, double sigmasq0, double alpha0,
    double beta0, double h, int n_mh_gamma, double eta1_sd, double eta2_sd,
    double mu_tilde, double eta1_tilde, double eta2_tilde, double e_eta,
    double f_eta, unsigned int T_max, int proposal_type, int thin = 1,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0, double tau0 = 0.0, double htau = 1.5,
    Rcpp::Nullable<Rcpp::NumericVector> tau_in = R_NilValue,
    double eta1_init = 0.01, double eta2_init = 0.01, double sigmasq_init = 1.0,
    int block_size = 1, int pcg_threshold = 500, bool use_lb_gamma = true) {
  Rcpp::RNGScope scope;

  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;

  if (thin < 1)
    thin = 1;

  // Output Storage
  int n_save = niter / thin;
  arma::mat beta_out(n_save, p);
  arma::umat gamma_out(n_save, p);
  arma::vec eta1_out(n_save);
  arma::vec eta2_out(n_save);
  arma::vec alpha_out(n_save);
  arma::vec sigmasq_out(n_save);
  arma::mat tau_out(n_save, Z_dat.n_cols);

  // Initialisation
  arma::vec beta(p, arma::fill::zeros);
  arma::uvec gamma(p, arma::fill::zeros);

  if (beta_in.isNotNull()) {
    beta = Rcpp::as<arma::vec>(beta_in);
  }
  if (gamma_in.isNotNull()) {
    gamma = Rcpp::as<arma::uvec>(gamma_in);
  }

  double alpha = alpha_in;
  double sigmasq = sigmasq_init;
  double eta1 = eta1_init; // initial eta for R_glasso
  double eta2 = eta2_init; // initial eta for R_fix

  // --- tau (Z_dat covariates) ---
  const arma::uword ntau = Z_dat.n_cols;
  arma::vec tau(ntau, arma::fill::zeros);
  tau.fill(tau0);
  if (tau_in.isNotNull()) {
    tau = Rcpp::as<arma::vec>(tau_in);
  }
  arma::vec Z_tau = Z_dat * tau;

  if ((unsigned int)R_glasso_int.nrow() != p ||
      (unsigned int)R_glasso_int.ncol() != p)
    Rcpp::stop("R_glasso dimensions must match p = %d", p);
  if ((unsigned int)R_fix_int.nrow() != p ||
      (unsigned int)R_fix_int.ncol() != p)
    Rcpp::stop("R_fix dimensions must match p = %d", p);

  // Also create arma::mat versions for fast submatrix operations
  arma::mat R_glasso(p, p);
  arma::mat R_fix(p, p);
  for (arma::uword i = 0; i < p; ++i) {
    for (arma::uword j = 0; j < p; ++j) {
      R_glasso(i, j) = static_cast<double>(R_glasso_int(i, j));
      R_fix(i, j) = static_cast<double>(R_fix_int(i, j));
    }
  }

  // ---- Initialisation -------------------------------------------------------
  // Initialisation moved above to handle Nullable inputs
  arma::vec z = X * beta;
  arma::vec omega_pg(n, arma::fill::ones);
  // FIX: Convert y from {-1,+1} to {0,1} for PG kappa computation
  arma::vec y01 = y;
  for (arma::uword i = 0; i < n; ++i) {
    if (y01(i) < 0.0) y01(i) = 0.0;
  }
  const arma::vec kappa = y01 - 0.5;

  // ---- Output storage -------------------------------------------------------
  // Output storage declarations moved above to handle thinning
  // ---- Temp buffer ----------------------------------------------------------
  arma::vec lin(n);

  EtaAdapter eta1_adapter(0.5);  // Vihola RAM for eta1 proposal
  EtaAdapter eta2_adapter(0.5);  // Vihola RAM for eta2 proposal

  // ==========================================================================
  // MCMC LOOP
  // ==========================================================================
  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {

    if (iter % 10000 == 0 && iter > 0) {
      Rcpp::checkUserInterrupt();
      int model_size = (int)arma::accu(gamma);
      Rcpp::Rcout << "Iter: " << iter << " | Model size: " << model_size
                  << " | eta1: " << eta1 << " | eta2: " << eta2 << "\n";
    }

    // ==================================================================
    // STEP A: Polya-Gamma augmentation + beta / alpha update
    // ==================================================================
    sigmasq = bvs_dadj::clamp_finite(sigmasq, 1e-10, 1e10, 1.0);
    lin = alpha + X * beta + Z_tau;
    bvs_dadj::sanitize_vec_inplace(lin, -60.0, 60.0, 0.0);
    for (arma::uword i = 0; i < n; ++i)
      omega_pg(i) = sample_pg_approx(lin(i));
    bvs_dadj::sanitize_vec_inplace(omega_pg, 1e-8, 1e6, 1.0);

    arma::uvec active = arma::find(gamma == 1);
    const arma::uword p_active = active.n_elem;

    if (p_active > 0) {
      arma::mat X_act = X.cols(active);

      if (block_size > 1 && (int)p_active > pcg_threshold) {
        // PCG path for large active sets
        bvs_dadj_block::PCGConfig pcg_cfg(1e-4, 200, pcg_threshold);
        arma::vec beta_act = beta.elem(active);
        bool pcg_ok = bvs_dadj_block::pcg_sample_beta(
            beta_act, X_act, omega_pg, kappa, alpha, 1.0 / sigmasq, pcg_cfg,
            &Z_tau, beta0);
        if (pcg_ok && beta_act.n_elem == p_active) {
          beta.zeros();
          beta.elem(active) = beta_act;
        }
      } else {
        // Original Cholesky path
        arma::mat Xt_Om = X_act.t();
        Xt_Om.each_row() %= omega_pg.t();
        arma::mat prec_beta = Xt_Om * X_act;
        prec_beta.diag() += 1.0 / sigmasq;
        bvs_dadj::sanitize_sym_mat_inplace(prec_beta, 1e10, 1e-8);

        arma::vec z_star = kappa - omega_pg * alpha - omega_pg % Z_tau;
        // FIX: Include beta0 prior mean in RHS (was missing)
        arma::vec mean_rhs = X_act.t() * z_star +
            (beta0 / sigmasq) * arma::ones<arma::vec>(p_active);
        bvs_dadj::sanitize_vec_inplace(mean_rhs, -1e12, 1e12, 0.0);

        arma::mat L_prec;
        bool chol_ok = bvs_dadj::robust_chol_inplace(L_prec, prec_beta);

        if (chol_ok) {
          arma::vec m_beta = arma::solve(arma::trimatl(L_prec.t()), mean_rhs,
                                         arma::solve_opts::fast);
          m_beta = arma::solve(arma::trimatu(L_prec), m_beta,
                               arma::solve_opts::fast);
          bvs_dadj::sanitize_vec_inplace(m_beta, -1e8, 1e8, 0.0);
          arma::vec zz = arma::randn<arma::vec>(p_active);
          arma::vec b_draw = m_beta + arma::solve(arma::trimatu(L_prec), zz,
                                                  arma::solve_opts::fast);
          bvs_dadj::sanitize_vec_inplace(b_draw, -1e8, 1e8, 0.0);
          beta.zeros();
          beta.elem(active) = b_draw;
        }
      }
    } else {
      beta.zeros();
    }

    // --- Intercept alpha ---
    {
      double sum_omega =
          bvs_dadj::clamp_finite(arma::accu(omega_pg), 1e-8, 1e12, (double)n);
      double prec_alpha = sum_omega + 1.0 / (h * sigmasq);
      prec_alpha = bvs_dadj::clamp_finite(prec_alpha, 1e-8, 1e12, 1.0);
      double var_alpha = 1.0 / prec_alpha;
      arma::vec resid = kappa - omega_pg % (X * beta + Z_tau);
      bvs_dadj::sanitize_vec_inplace(resid, -1e8, 1e8, 0.0);
      double mean_alpha =
          var_alpha * (arma::accu(resid) + alpha0 / (h * sigmasq));
      mean_alpha = bvs_dadj::clamp_finite(mean_alpha, -60.0, 60.0, 0.0);
      alpha = R::rnorm(mean_alpha, std::sqrt(var_alpha));
      alpha = bvs_dadj::clamp_finite(alpha, -60.0, 60.0, 0.0);
    }

    // ==================================================================
    // STEP B: Gamma (variable selection) update
    // ==================================================================
    // Locally-balanced gamma proposal state (Zanella 2020)
    std::vector<double> lb_score, lb_weight;
    double lb_Z = 0.0;
    LBProposalDelta lb_delta;
    if (block_size > 1) {
      // Block update: SW + Uncollapsed Gibbs (dual adjacency)
      auto gamma_u8 = bvs_dadj_block::gamma_to_uint8(gamma);
      auto neigh_glasso_fn = [&](int jj, std::function<void(int)> cb) {
        for (arma::uword i = 0; i < p; ++i) {
          if ((arma::uword)jj != i && R_glasso(i, jj) != 0.0)
            cb(static_cast<int>(i));
        }
      };
      auto neigh_fix_fn = [&](int jj, std::function<void(int)> cb) {
        for (arma::uword i = 0; i < p; ++i) {
          if ((arma::uword)jj != i && R_fix(i, jj) != 0.0)
            cb(static_cast<int>(i));
        }
      };
      auto proposal = bvs_dadj_block::swendsen_wang_dual(
          gamma_u8, eta1, eta2, (int)p, block_size, neigh_glasso_fn,
          neigh_fix_fn);
      auto block = bvs_dadj_block::flatten_clusters(proposal);
      if (!block.empty()) {
        z = X * beta;
        // R2-FIX: pass Z_tau so the block likelihood uses the full predictor.
        bvs_dadj_block::uncollapsed_gamma_sweep_dual(
            gamma_u8, beta, z, X, y, alpha, sigmasq, beta0, mu, eta1, eta2,
            block, neigh_glasso_fn, neigh_fix_fn, &Z_tau);
        bvs_dadj_block::uint8_to_gamma(gamma, gamma_u8);
      }
    } else {
      // Original single-variable MH
      z = X * beta;
      double loglik = calc_loglik(y01, z, alpha, Z_tau);
      if (!std::isfinite(loglik))
        loglik = -std::numeric_limits<double>::infinity();
      double sd_beta = std::sqrt(sigmasq);
      if (!std::isfinite(sd_beta) || sd_beta <= 0.0)
        sd_beta = 1.0;

      // Initialize LB scores before gamma scan
      arma::ivec gamma_iv;
      if (use_lb_gamma) {
        gamma_iv = arma::conv_to<arma::ivec>::from(gamma);
        init_lb_dual_scores_dense(R_glasso_int, R_fix_int, (int)p, gamma_iv, mu, eta1, eta2, lb_score, lb_weight, lb_Z);
      }

      for (int mh = 0; mh < n_mh_gamma; ++mh) {
        int j;
        if (use_lb_gamma) {
          j = sample_weighted_index(lb_weight, lb_Z, (int)p);
        } else {
          j = static_cast<int>(std::floor(R::runif(0.0, (double)p)));
          if (j >= (int)p)
            j = (int)p - 1;
        }

        int g_curr = gamma(j);
        int g_prop = 1 - g_curr;

        double b_curr = beta(j);
        double b_prop = (g_prop == 1) ? R::rnorm(beta0, sd_beta) : 0.0;
        b_prop = bvs_dadj::clamp_finite(b_prop, -1e8, 1e8, 0.0);

        arma::vec z_prop = z + (b_prop - b_curr) * X.col(j);
        if (!z_prop.is_finite())
          continue;
        double ll_prop = calc_loglik(y01, z_prop, alpha, Z_tau);
        if (!std::isfinite(ll_prop))
          continue;

        double diff = static_cast<double>(g_prop - g_curr);

        // Dual Ising neighbourhood sufficient statistics
        double neigh_glasso = 0.0, neigh_fix = 0.0;
        for (arma::uword i = 0; i < p; ++i) {
          if ((arma::uword)j == i)
            continue;
          if (R_glasso(i, j) != 0.0)
            neigh_glasso += gamma(i);
          if (R_fix(i, j) != 0.0)
            neigh_fix += gamma(i);
        }

        double ising_diff =
            diff * (mu + eta1 * neigh_glasso + eta2 * neigh_fix);

        double log_ratio = (ll_prop - loglik) + ising_diff;
        if (!std::isfinite(log_ratio))
          continue;

        // LB correction: adjust MH ratio for non-uniform proposal
        if (use_lb_gamma) {
          int delta_g = 1 - 2 * (int)gamma(j);  // +1 if currently 0, -1 if currently 1
          gamma_iv(j) = (int)gamma(j);  // ensure sync
          build_lb_dual_delta_dense(R_glasso_int, R_fix_int, (int)p, gamma_iv, eta1, eta2, j, delta_g,
              lb_score, lb_weight, lb_Z, lb_delta);
          log_ratio += lb_delta.log_q_rev - lb_delta.log_q_fwd;
        }

        if (bvs_dadj::safe_mh_accept(log_ratio)) {
          gamma(j) = g_prop;
          beta(j) = b_prop;
          z = z_prop;
          loglik = ll_prop;
          // Update LB state on acceptance
          if (use_lb_gamma) {
            apply_lb_delta(lb_score, lb_weight, lb_Z, lb_delta);
            gamma_iv(j) = (int)g_prop;
          }
        }
      }
    }

    // ===================== STEP B.5: tau Gibbs step =======================
    if (ntau > 0) {
      arma::mat Zt_Om = Z_dat.t();
      Zt_Om.each_row() %= omega_pg.t();
      arma::mat prec_tau = Zt_Om * Z_dat;
      prec_tau.diag() += 1.0 / (htau * sigmasq);

      arma::vec z_star_tau = kappa - omega_pg * alpha - omega_pg % (X * beta);
      arma::vec mean_rhs_tau =
          Z_dat.t() * z_star_tau +
          (tau0 / (htau * sigmasq)) * arma::ones<arma::vec>(ntau);

      arma::mat L_tau;
      bool tau_ok = bvs_dadj::robust_chol_inplace(L_tau, prec_tau);
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

    // ==================================================================
    // STEP C: sigmasq — exact Inverse-Gamma Gibbs.
    // The PG logistic likelihood does not depend on sigmasq; the
    // conjugate IG posterior for the (beta, alpha, tau) priors is exact.
    // ==================================================================
    {
      double n_act = static_cast<double>(arma::accu(gamma));
      double ss_b = 0.0;
      if (n_act > 0) {
        arma::uvec act = arma::find(gamma == 1);
        ss_b = arma::accu(arma::square(beta.elem(act) - beta0));
      }
      double ss_tau = 0.0;
      for (arma::uword j = 0; j < ntau; ++j) {
        double d = tau(j) - tau0;
        ss_tau += d * d;
      }
      const double da = alpha - alpha0;
      const double shape_post = 0.5 * nu0 + 0.5 * (n_act + 1.0 + (double)ntau);
      const double scale_post =
          0.5 * nu0 * sigmasq0 + 0.5 * (ss_b + da * da / h + ss_tau / htau);
      const double gdraw =
          R::rgamma(shape_post, 1.0 / std::max(scale_post, 1e-12));
      if (std::isfinite(gdraw) && gdraw > 0.0)
        sigmasq = std::max(1e-10, 1.0 / gdraw);
      sigmasq = bvs_dadj::clamp_finite(sigmasq, 1e-10, 1e10, 1.0);
    }

    // ==================================================================
    // STEP D: eta1, eta2 via Moller et al. (2006)
    //         with Propp-Wilson perfect simulation
    // ==================================================================
    moller_update_eta(R_glasso_int, R_fix_int, mu, eta1, eta2, eta1_sd, eta2_sd,
                      mu_tilde, eta1_tilde, eta2_tilde, gamma, e_eta, f_eta,
                      T_max, proposal_type, eta1_adapter, eta2_adapter);

    // ==================================================================
    // STORE SAMPLES (post burn-in)
    // ==================================================================
    if (iter >= burnin && (iter - burnin) % thin == 0) {
      int idx = (iter - burnin) / thin;
      beta_out.row(idx) = beta.t();
      gamma_out.row(idx) = gamma.t();
      eta1_out(idx) = eta1;
      eta2_out(idx) = eta2;
      alpha_out(idx) = alpha;
      sigmasq_out(idx) = sigmasq;
      tau_out.row(idx) = tau.t();
    }
  }

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_out, Rcpp::Named("gamma") = gamma_out,
      Rcpp::Named("eta1") = eta1_out, Rcpp::Named("eta2") = eta2_out,
      Rcpp::Named("alpha") = alpha_out, Rcpp::Named("sigmasq") = sigmasq_out,
      Rcpp::Named("tau") = tau_out);
}
