// [[Rcpp::depends(RcppArmadillo)]]
#include "BayesLogit_BlockPG.h"
#include "BayesLogit_Numerics.h"
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace {

// =============================================================================
// HELPER FUNCTIONS  (sub-matrix removal / insertion for Wang 2012 GGM)
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
  return arma::solve(arma::trimatu(U), y, arma::solve_opts::fast);
}

static inline void rnorm_into(arma::vec &out) {
  const arma::uword n = out.n_elem;
  for (arma::uword i = 0; i < n; ++i)
    out(i) = R::rnorm(0.0, 1.0);
}

// =============================================================================
// Polya-Gamma(1, z) approximate sampler (truncated series, K=200 terms)
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

// =============================================================================
// Normal PDF / CDF / erf approximation (for truncated normal proposal)
// =============================================================================
static inline double normal_pdf(double x, double mu, double sigma) {
  double z = (x - mu) / sigma;
  return std::exp(-0.5 * z * z) / (sigma * std::sqrt(2.0 * M_PI));
}

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

// =============================================================================
// Log Beta density: log Beta(x | a, b)
// =============================================================================
static inline double log_beta_pdf(double x, double a, double b) {
  // Clamp x away from 0 and 1 for numerical stability
  x = std::max(1e-12, std::min(1.0 - 1e-12, x));
  return std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
         (a - 1.0) * std::log(x) + (b - 1.0) * std::log(1.0 - x);
}

// =============================================================================
// PROPP-WILSON PERFECT SIMULATION
//
// Generates omega from MRF:
//   P(omega | mu, eta1, eta2, R1, R2) propto
//     exp{ mu sum(omega_j) + eta1 sum_{R1} omega_j omega_k
//                           + eta2 sum_{R2} omega_j omega_k }
//
// R1 = dynamic GGM adjacency (changes every MCMC iteration)
// R2 = fixed external pathway adjacency
//
// Monotone coupling with upper/lower chains. Falls back to Gibbs at T_max.
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
          int r1ij = R1(i, j);
          int r2ij = R2(i, j);
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
    if (T >= T_max) {
      std::vector<unsigned int> diff_idx;
      for (unsigned int k = 0; k < p; ++k)
        if (x_up[k] != x_down[k])
          diff_idx.push_back(k);

      std::mt19937 gen_fb(seed_base + 99999);
      std::uniform_real_distribution<double> unif01_fb(0.0, 1.0);
      for (int sweep = 0; sweep < 100; ++sweep) {
        for (unsigned int idx = 0; idx < diff_idx.size(); ++idx) {
          unsigned int m = diff_idx[idx];
          double ker = 0.0;
          for (unsigned int k = 0; k < p; ++k)
            ker += eta1 * R1(m, k) * x_up[k] + eta2 * R2(m, k) * x_up[k];
          double prob = 1.0 / (1.0 + std::exp(-(mu + ker)));
          x_up[m] = (unif01_fb(gen_fb) < prob) ? 1 : 0;
        }
      }
      for (unsigned int k = 0; k < p; ++k)
        x_down[k] = x_up[k];
    }
  }
  return x_up;
}

// =============================================================================
// MOLLER ET AL. (2006) MH UPDATE FOR eta1, eta2
//
// R1 = R_dyn (current GGM adjacency snapshot), R2 = R_fix (external pathway)
// eta1 couples R_dyn, eta2 couples R_fix
// Prior: eta_i / eta_i_sd ~ Beta(e, f)
//
// proposal_type = 0: Uniform on [max(0, eta-0.01), min(eta_sd, eta+0.01)]
// proposal_type = 1: Truncated Normal(eta, eta_sd) on (0, eta_sd)
// =============================================================================
static void
moller_update_eta(const Rcpp::IntegerMatrix &R1, const Rcpp::IntegerMatrix &R2,
                  double mu, double &eta1, double &eta2, double eta1_sd,
                  double eta2_sd, double mu_tilde, double eta1_tilde,
                  double eta2_tilde, const arma::uvec &gamma, double e_eta,
                  double f_eta, unsigned int T_max, int proposal_type) {
  const unsigned int p = R1.ncol();
  double eta1_new, eta2_new;
  double log_prop_ratio_eta1 = 0.0;
  double log_prop_ratio_eta2 = 0.0;

  if (proposal_type == 0) {
    double a1 = std::max(0.0, eta1 - 0.01);
    double b1 = std::min(eta1_sd, eta1 + 0.01);
    eta1_new = R::runif(a1, b1);
    double a2 = std::max(0.0, eta2 - 0.01);
    double b2 = std::min(eta2_sd, eta2 + 0.01);
    eta2_new = R::runif(a2, b2);

    double c1 = std::max(0.0, eta1_new - 0.01);
    double d1 = std::min(eta1_sd, eta1_new + 0.01);
    double c2 = std::max(0.0, eta2_new - 0.01);
    double d2 = std::min(eta2_sd, eta2_new + 0.01);

    log_prop_ratio_eta1 = std::log(b1 - a1) - std::log(d1 - c1);
    log_prop_ratio_eta2 = std::log(b2 - a2) - std::log(d2 - c2);
  } else {
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

  eta1_new = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1_new));
  eta2_new = std::max(1e-8, std::min(eta2_sd - 1e-8, eta2_new));

  // Generate auxiliary variables via Propp-Wilson
  Rcpp::IntegerVector omega_eta1 =
      proppwilson_omega(R1, R2, mu, eta1, eta2, T_max);
  Rcpp::IntegerVector omega_eta2 =
      proppwilson_omega(R1, R2, mu, eta1, eta2, T_max);
  Rcpp::IntegerVector omega_eta1_new =
      proppwilson_omega(R1, R2, mu, eta1_new, eta2, T_max);
  Rcpp::IntegerVector omega_eta2_new =
      proppwilson_omega(R1, R2, mu, eta1, eta2_new, T_max);

  // Sufficient statistics
  int B_R1 = 0, B_R2 = 0;
  int A_om1_R1 = 0, A_om1_R2 = 0, A_om2_R1 = 0, A_om2_R2 = 0;
  int A_om1n_R1 = 0, A_om1n_R2 = 0, A_om2n_R1 = 0, A_om2n_R2 = 0;
  int sum_om1 = 0, sum_om2 = 0, sum_om1n = 0, sum_om2n = 0;

  for (unsigned int j = 0; j < p; ++j) {
    sum_om1 += omega_eta1[j];
    sum_om2 += omega_eta2[j];
    sum_om1n += omega_eta1_new[j];
    sum_om2n += omega_eta2_new[j];
    for (unsigned int k = j + 1; k < p; ++k) {
      int r1jk = R1(j, k), r2jk = R2(j, k);
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

  double log_prior_eta1 = log_beta_pdf(eta1_new / eta1_sd, e_eta, f_eta) -
                          log_beta_pdf(eta1 / eta1_sd, e_eta, f_eta);
  double log_prior_eta2 = log_beta_pdf(eta2_new / eta2_sd, e_eta, f_eta) -
                          log_beta_pdf(eta2 / eta2_sd, e_eta, f_eta);

  // MH ratio (MÃ¶ller et al. 2006)
  double log_MH_eta1 =
      (eta1_new - eta1) * B_R1 + log_prior_eta1 +
      mu_tilde * (sum_om1n - sum_om1) + eta1_tilde * (A_om1n_R1 - A_om1_R1) +
      eta2_tilde * (A_om1n_R2 - A_om1_R2) + mu * (sum_om1 - sum_om1n) +
      eta1 * A_om1_R1 - eta1_new * A_om1n_R1 + eta2 * (A_om1_R2 - A_om1n_R2) +
      log_prop_ratio_eta1;

  double log_MH_eta2 =
      (eta2_new - eta2) * B_R2 + log_prior_eta2 +
      mu_tilde * (sum_om2n - sum_om2) + eta1_tilde * (A_om2n_R1 - A_om2_R1) +
      eta2_tilde * (A_om2n_R2 - A_om2_R2) + mu * (sum_om2 - sum_om2n) +
      eta2 * A_om2_R2 - eta2_new * A_om2n_R2 + eta1 * (A_om2_R1 - A_om2n_R1) +
      log_prop_ratio_eta2;

  if (bvs_dadj::safe_mh_accept(log_MH_eta1))
    eta1 = eta1_new;
  if (bvs_dadj::safe_mh_accept(log_MH_eta2))
    eta2 = eta2_new;
}

} // anonymous namespace
// =============================================================================
// PHASE TRANSITION DETECTION (exported utility)
// =============================================================================
// [[Rcpp::export]]
Rcpp::IntegerMatrix phase_transit_2eta(Rcpp::IntegerMatrix R1,
                                       Rcpp::IntegerMatrix R2, int T_max,
                                       double mu, double min_eta,
                                       double max_eta, unsigned int num_rep,
                                       double step_size = 0.01) {
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
//   - Bayesian GGM (Wang 2012 block-Gibbs) for learning dynamic network R_dyn
//   - Fixed external pathway network R_fix
//   - Dual-network Ising/MRF prior on gamma:
//       eta1 couples R_dyn (GGM-learned), eta2 couples R_fix (external)
//   - eta1, eta2 updated via MÃ¶ller et al. (2006) auxiliary variable MH
//     with Propp-Wilson perfect simulation
//   - Prior: eta_i / eta_i_sd ~ Beta(e_eta, f_eta)
// =============================================================================

// [[Rcpp::export]]
Rcpp::List BayesLogit_PG_GGM_Moller(
    const arma::mat &X, const arma::vec &y, const arma::mat &S_ggm,
    double n_ggm, const arma::mat &R_fix, const arma::mat &Z_dat, int niter,
    int burnin, double mu, double nu0, double sigmasq0, double alpha0,
    double beta0, double h, int n_mh_gamma, double v0_ggm, double v1_ggm,
    double pii_ggm, double lambda_ggm, double eta1_sd, double eta2_sd,
    double mu_tilde, double eta1_tilde, double eta2_tilde, double e_eta,
    double f_eta, unsigned int T_max, int proposal_type, int thin = 1,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0, double tau0 = 0.0, double htau = 1.5,
    Rcpp::Nullable<Rcpp::NumericVector> tau_in = R_NilValue, int block_size = 1,
    int pcg_threshold = 500) {
  Rcpp::RNGScope scope;

  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  if (S_ggm.n_cols != p || S_ggm.n_rows != p)
    Rcpp::stop("S_ggm dimensions must match p = %d", p);
  if (R_fix.n_cols != p || R_fix.n_rows != p)
    Rcpp::stop("R_fix dimensions must match p = %d", p);

  if (thin < 1)
    thin = 1;

  // ---- Initialisation -------------------------------------------------------
  arma::vec beta(p, arma::fill::zeros);
  arma::uvec gamma(p, arma::fill::zeros);
  if (beta_in.isNotNull())
    beta = Rcpp::as<arma::vec>(beta_in);
  if (gamma_in.isNotNull())
    gamma = Rcpp::as<arma::uvec>(gamma_in);

  double alpha = alpha_in;
  double sigmasq = 1.0;
  double eta1 = std::min(0.01, eta1_sd * 0.5); // initial eta for R_dyn
  double eta2 = std::min(0.01, eta2_sd * 0.5); // initial eta for R_fix

  // --- tau (Z_dat covariates) ---
  const arma::uword ntau = Z_dat.n_cols;
  arma::vec tau(ntau, arma::fill::zeros);
  tau.fill(tau0);
  if (tau_in.isNotNull()) {
    tau = Rcpp::as<arma::vec>(tau_in);
  }
  arma::vec Z_tau = Z_dat * tau;

  arma::vec z = X * beta;
  arma::vec omega_pg(n, arma::fill::ones);
  const arma::vec kappa = y - 0.5;

  // ---- GGM state (Wang 2012) -----------------------------------------------
  arma::mat Sig_ggm = arma::diagmat(S_ggm.diag());
  arma::Mat<uint8_t> Z_ggm(p, p, arma::fill::zeros);
  arma::mat R_dyn(p, p, arma::fill::zeros);

  // ---- GGM pre-allocated buffers -------------------------------------------
  const arma::uword pm1 = p - 1;
  arma::mat C_ggm(p, p);
  arma::mat invC11(pm1, pm1), Ci(pm1, pm1), U_ggm(pm1, pm1);
  arma::vec Sig12(pm1), S_i(pm1), tau_temp(pm1);
  arma::vec mu_i(pm1), b_ggm(pm1), eps_ggm(pm1);
  arma::vec invC11beta(pm1), Sig12_new(pm1);
  arma::vec v0_vec(pm1), v1_vec(pm1), w1(pm1), w2(pm1), w_ggm(pm1);

  v0_vec.fill(v0_ggm);
  v1_vec.fill(v1_ggm);
  const double log_pii = std::log(pii_ggm);
  const double log_1m_pii = std::log(1.0 - pii_ggm);
  const double a_gam = 0.5 * n_ggm + 1.0;

  // ---- R_fix as IntegerMatrix for Propp-Wilson (fixed, never changes) ------
  Rcpp::IntegerMatrix R_fix_int(p, p);
  for (arma::uword i = 0; i < p; ++i)
    for (arma::uword j = 0; j < p; ++j)
      R_fix_int(i, j) = (R_fix(i, j) != 0.0) ? 1 : 0;

  // ---- R_dyn IntegerMatrix snapshot (rebuilt each iteration for MÃ¶ller) ----
  Rcpp::IntegerMatrix R_dyn_int(p, p);

  // ---- Output Storage ------------------------------------------------------
  int n_save = niter / thin;
  arma::mat beta_out(n_save, p);
  arma::umat gamma_out(n_save, p);
  arma::vec eta1_out(n_save);
  arma::vec eta2_out(n_save);
  arma::vec alpha_out(n_save);
  arma::vec sigmasq_out(n_save);
  arma::mat tau_out(n_save, ntau);

  // Sparse Z_list: each entry stores only (row, col) edge indices from upper
  // triangle
  Rcpp::List Z_list(n_save);

  arma::vec lin(n);

  // ==========================================================================
  // MCMC LOOP
  // ==========================================================================
  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {

    if (iter % 5000 == 0 && iter > 0) {
      Rcpp::checkUserInterrupt();
      arma::uword edge_count = 0;
      for (arma::uword r = 0; r < p; ++r)
        for (arma::uword c = r + 1; c < p; ++c)
          edge_count += Z_ggm(r, c);
      int model_size = (int)arma::accu(gamma);
      Rcpp::Rcout << "Iter: " << iter << " | GGM Edges: " << edge_count
                  << " | Model: " << model_size << " | eta1: " << eta1
                  << " | eta2: " << eta2 << "\n";
    }

    // ==================================================================
    // STEP A: GGM Block-Gibbs (Wang 2012)
    // ==================================================================
    bvs_dadj::sanitize_sym_mat_inplace(Sig_ggm, 1e10, 1e-8);
    bool inv_ok = arma::inv_sympd(C_ggm, arma::symmatu(Sig_ggm));
    if (!inv_ok) {
      arma::mat Sig_try = Sig_ggm;
      double jitter = 1e-8;
      for (int attempt = 0; attempt < 6 && !inv_ok; ++attempt) {
        Sig_try.diag() += jitter;
        inv_ok = arma::inv_sympd(C_ggm, arma::symmatu(Sig_try));
        jitter *= 10.0;
      }
      if (inv_ok) {
        Sig_ggm = Sig_try;
      } else {
        C_ggm.eye();
      }
    }

    for (arma::uword i = 0; i < p; ++i) {
      vec_remove_idx(Sig_ggm.col(i), Sig12, i, p);
      vec_remove_idx(S_ggm.col(i), S_i, i, p);

      mat_remove_rowcol(Sig_ggm, invC11, i, p);
      invC11 -= (1.0 / Sig_ggm(i, i)) * (Sig12 * Sig12.t());

      for (arma::uword k = 0, idx = 0; k < p; ++k) {
        if (k == i)
          continue;
        tau_temp(idx++) = (Z_ggm(k, i) == 1) ? v1_ggm : v0_ggm;
      }

      double s_ii = S_ggm(i, i);
      Ci = (s_ii + lambda_ggm) * invC11;
      Ci.diag() += 1.0 / tau_temp;
      bvs_dadj::sanitize_sym_mat_inplace(Ci, 1e10, 1e-8);

      bool ok = bvs_dadj::robust_chol_inplace(U_ggm, Ci);
      if (!ok)
        continue;

      mu_i = -solve_spd_chol_upper(U_ggm, S_i);
      bvs_dadj::sanitize_vec_inplace(mu_i, -1e8, 1e8, 0.0);
      rnorm_into(eps_ggm);
      arma::vec delta =
          arma::solve(arma::trimatu(U_ggm), eps_ggm, arma::solve_opts::fast);
      b_ggm = mu_i + delta;
      bvs_dadj::sanitize_vec_inplace(b_ggm, -1e8, 1e8, 0.0);

      {
        arma::uword k = 0;
        for (arma::uword j = 0; j < p; ++j) {
          if (j == i)
            continue;
          C_ggm(j, i) = b_ggm(k);
          C_ggm(i, j) = b_ggm(k);
          k++;
        }
      }

      double gam_rate = 0.5 * (s_ii + lambda_ggm);
      gam_rate = bvs_dadj::clamp_finite(gam_rate, 1e-10, 1e10, 1.0);
      double gam_val = R::rgamma(a_gam, 1.0 / gam_rate);
      gam_val = bvs_dadj::clamp_finite(gam_val, 1e-10, 1e10, 1.0);
      double cval = arma::dot(b_ggm, invC11 * b_ggm);
      C_ggm(i, i) = gam_val + cval;

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

      // Edge indicator update (spike-and-slab)
      arma::vec b2 = arma::square(b_ggm);
      w1 = -0.5 * arma::log(v0_vec) - 0.5 * b2 / v0_vec + log_1m_pii;
      w2 = -0.5 * arma::log(v1_vec) - 0.5 * b2 / v1_vec + log_pii;
      arma::vec wmax = arma::max(w1, w2);
      arma::vec exp_w1 = arma::exp(w1 - wmax);
      arma::vec exp_w2 = arma::exp(w2 - wmax);
      w_ggm = exp_w2 / (exp_w1 + exp_w2);

      {
        arma::uword k = 0;
        for (arma::uword j = 0; j < p; ++j) {
          if (j == i)
            continue;
          bool active = (R::runif(0.0, 1.0) < w_ggm(k));
          Z_ggm(j, i) = active ? 1 : 0;
          Z_ggm(i, j) = active ? 1 : 0;
          R_dyn(j, i) = active ? 1.0 : 0.0;
          R_dyn(i, j) = active ? 1.0 : 0.0;
          k++;
        }
      }
    } // end GGM column sweep

    // ==================================================================
    // STEP B: Polya-Gamma augmentation + beta / alpha update
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
        bvs_dadj_block::PCGConfig pcg_cfg(1e-6, 200, pcg_threshold);
        arma::vec beta_act;
        bool pcg_ok = bvs_dadj_block::pcg_sample_beta(
            beta_act, X_act, omega_pg, kappa, alpha, 1.0 / sigmasq, pcg_cfg);
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
        arma::vec mean_rhs = X_act.t() * z_star;
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

    // Intercept alpha
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
    // STEP C: Gamma (variable selection) update
    // ==================================================================
    if (block_size > 1) {
      // Block update: SW + Uncollapsed Gibbs (dual: GGM + fixed)
      auto gamma_u8 = bvs_dadj_block::gamma_to_uint8(gamma);
      auto neigh_dyn_fn = [&](int jj, std::function<void(int)> cb) {
        for (arma::uword i = 0; i < p; ++i) {
          if ((arma::uword)jj != i && R_dyn(i, jj) != 0.0)
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
          gamma_u8, eta1, eta2, (int)p, block_size, neigh_dyn_fn, neigh_fix_fn);
      auto block = bvs_dadj_block::flatten_clusters(proposal);
      if (!block.empty()) {
        z = X * beta;
        bvs_dadj_block::uncollapsed_gamma_sweep_dual(
            gamma_u8, beta, z, X, y, alpha, sigmasq, beta0, mu, eta1, eta2,
            block, neigh_dyn_fn, neigh_fix_fn);
        bvs_dadj_block::uint8_to_gamma(gamma, gamma_u8);
      }
    } else {
      // Original single-variable MH
      z = X * beta;
      double loglik = calc_loglik(y, z, alpha, Z_tau);
      if (!std::isfinite(loglik))
        loglik = -std::numeric_limits<double>::infinity();
      double sd_beta = std::sqrt(sigmasq);
      if (!std::isfinite(sd_beta) || sd_beta <= 0.0)
        sd_beta = 1.0;

      for (int mh = 0; mh < n_mh_gamma; ++mh) {
        int j = static_cast<int>(std::floor(R::runif(0.0, (double)p)));
        if (j >= (int)p)
          j = (int)p - 1;

        int g_curr = gamma(j);
        int g_prop = 1 - g_curr;

        double b_curr = beta(j);
        double b_prop = (g_prop == 1) ? R::rnorm(beta0, sd_beta) : 0.0;
        b_prop = bvs_dadj::clamp_finite(b_prop, -1e8, 1e8, 0.0);

        arma::vec z_prop = z + (b_prop - b_curr) * X.col(j);
        if (!z_prop.is_finite())
          continue;
        double ll_prop = calc_loglik(y, z_prop, alpha, Z_tau);
        if (!std::isfinite(ll_prop))
          continue;

        double diff = static_cast<double>(g_prop - g_curr);

        double neigh_dyn = 0.0, neigh_fix = 0.0;
        for (arma::uword i = 0; i < p; ++i) {
          if ((arma::uword)j == i)
            continue;
          if (R_dyn(i, j) != 0.0)
            neigh_dyn += gamma(i);
          if (R_fix(i, j) != 0.0)
            neigh_fix += gamma(i);
        }

        double ising_diff = diff * (mu + eta1 * neigh_dyn + eta2 * neigh_fix);

        double log_ratio = (ll_prop - loglik) + ising_diff;
        if (!std::isfinite(log_ratio))
          continue;

        if (bvs_dadj::safe_mh_accept(log_ratio)) {
          gamma(j) = g_prop;
          beta(j) = b_prop;
          z = z_prop;
          loglik = ll_prop;
        }
      }
    }

    // ==================================================================
    // STEP C.5: tau Gibbs step =======================
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
    // STEP D: sigmasq — exact Inverse-Gamma Gibbs.
    // PG logistic likelihood does not depend on sigmasq; the conjugate
    // IG posterior for the (beta, alpha, tau) priors is exact.
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
      const double shape_post =
          0.5 * nu0 + 0.5 * (n_act + 1.0 + (double)ntau);
      const double scale_post =
          0.5 * nu0 * sigmasq0 + 0.5 * (ss_b + da * da / h + ss_tau / htau);
      const double gdraw = R::rgamma(shape_post, 1.0 / std::max(scale_post, 1e-12));
      if (std::isfinite(gdraw) && gdraw > 0.0)
        sigmasq = std::max(1e-10, 1.0 / gdraw);
      sigmasq = bvs_dadj::clamp_finite(sigmasq, 1e-10, 1e10, 1.0);
    }

    // ==================================================================
    // STEP E: eta1, eta2 via MÃ¶ller et al. (2006) with Propp-Wilson
    //
    // Rebuild R_dyn_int snapshot from current Z_ggm for Propp-Wilson
    // R1 = R_dyn (dynamic, from GGM), R2 = R_fix (external pathway)
    // ==================================================================
    {
      for (arma::uword r = 0; r < p; ++r)
        for (arma::uword c = 0; c < p; ++c)
          R_dyn_int(r, c) = (Z_ggm(r, c) == 1) ? 1 : 0;

      moller_update_eta(R_dyn_int, R_fix_int, mu, eta1, eta2, eta1_sd, eta2_sd,
                        mu_tilde, eta1_tilde, eta2_tilde, gamma, e_eta, f_eta,
                        T_max, proposal_type);
    }

    // ==================================================================
    // STORE SAMPLES (post burn-in)
    // ==================================================================
    // ---- Save Samples -------------------------------------------------------
    if (iter >= burnin && (iter - burnin) % thin == 0) {
      int idx = (iter - burnin) / thin;
      beta_out.row(idx) = beta.t();
      gamma_out.row(idx) = gamma.t();
      eta1_out(idx) = eta1;
      eta2_out(idx) = eta2;
      alpha_out(idx) = alpha;
      sigmasq_out(idx) = sigmasq;
      tau_out.row(idx) = tau.t();

      // Sparse Z storage: only store (row, col) of edges in upper triangle
      std::vector<int> edge_rows, edge_cols;
      for (arma::uword r = 0; r < p; ++r)
        for (arma::uword c = r + 1; c < p; ++c)
          if (Z_ggm(r, c) == 1) {
            edge_rows.push_back(r);
            edge_cols.push_back(c);
          }
      int ne = (int)edge_rows.size();
      Rcpp::IntegerMatrix edges(std::max(ne, 0), 2);
      for (int ei = 0; ei < ne; ++ei) {
        edges(ei, 0) = edge_rows[ei];
        edges(ei, 1) = edge_cols[ei];
      }
      Z_list[idx] = edges;
    }
  } // end MCMC

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_out, Rcpp::Named("gamma") = gamma_out,
      Rcpp::Named("eta1") = eta1_out, Rcpp::Named("eta2") = eta2_out,
      Rcpp::Named("alpha") = alpha_out, Rcpp::Named("sigmasq") = sigmasq_out,
      Rcpp::Named("tau") = tau_out, Rcpp::Named("Z_list") = Z_list);
}
