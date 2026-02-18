// [[Rcpp::depends(RcppArmadillo)]]
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
  return std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
         (a - 1.0) * std::log(x) + (b - 1.0) * std::log(1.0 - x);
}

// =============================================================================
// PROPP-WILSON PERFECT SIMULATION (Single Adjacency)
// OPT: accepts uint8 Z_ggm directly — no R_dyn_int copy needed
// OPT: skips j==i in kernel (diagonal is always 0)
// =============================================================================
static Rcpp::IntegerVector proppwilson_omega(const arma::Mat<uint8_t> &Z,
                                             double mu, double eta1,
                                             unsigned int T_max) {
  const unsigned int p = Z.n_cols;
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
      for (unsigned int k = 0; k < p; ++k)
        x_down[k] = x_up[k];
    }
  }
  return x_up;
}

// =============================================================================
// MOLLER ET AL. (2006) MH UPDATE FOR SINGLE eta1
// OPT: accepts uint8 Z_ggm directly — no R_dyn_int copy needed
// =============================================================================
static void moller_update_eta(const arma::Mat<uint8_t> &Z, double mu,
                              double &eta1, double eta1_sd, double mu_tilde,
                              double eta1_tilde, const arma::uvec &gamma,
                              double e, double f, unsigned int T_max,
                              int proposal_type) {
  const unsigned int p = Z.n_cols;
  double eta1_new;
  double log_prop_ratio_eta1 = 0.0;

  if (proposal_type == 0) {
    // Uniform random walk proposal
    double a1 = std::max(0.0, eta1 - 0.01);
    double b1 = std::min(eta1_sd, eta1 + 0.01);
    eta1_new = R::runif(a1, b1);
    double c1 = std::max(0.0, eta1_new - 0.01);
    double d1 = std::min(eta1_sd, eta1_new + 0.01);
    log_prop_ratio_eta1 = std::log(b1 - a1) - std::log(d1 - c1);
  } else {
    // Truncated normal proposal
    int attempts = 0;
    do {
      eta1_new = R::rnorm(eta1, eta1_sd);
      if (++attempts > 10000) {
        eta1_new = eta1;
        break;
      }
    } while (eta1_new <= 0.0 || eta1_new >= eta1_sd);

    double log_q_fwd_1 = std::log(normal_pdf(eta1_new, eta1, eta1_sd)) -
                         std::log(normal_cdf(eta1_sd, eta1, eta1_sd) -
                                  normal_cdf(0.0, eta1, eta1_sd));
    double log_q_rev_1 = std::log(normal_pdf(eta1, eta1_new, eta1_sd)) -
                         std::log(normal_cdf(eta1_sd, eta1_new, eta1_sd) -
                                  normal_cdf(0.0, eta1_new, eta1_sd));
    log_prop_ratio_eta1 = log_q_rev_1 - log_q_fwd_1;
  }

  eta1_new = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1_new));

  // Generate auxiliary variables via Propp-Wilson (uses Z_ggm directly)
  Rcpp::IntegerVector omega_eta1 = proppwilson_omega(Z, mu, eta1, T_max);
  Rcpp::IntegerVector omega_eta1_new =
      proppwilson_omega(Z, mu, eta1_new, T_max);

  // Compute sufficient statistics
  int B_R1 = 0;
  int A_om1_R1 = 0, A_om1n_R1 = 0;
  int sum_om1 = 0, sum_om1n = 0;

  for (unsigned int j = 0; j < p; ++j) {
    sum_om1 += omega_eta1[j];
    sum_om1n += omega_eta1_new[j];
    for (unsigned int k = j + 1; k < p; ++k) {
      if (Z(j, k) == 0)
        continue;
      B_R1 += (int)gamma(j) * (int)gamma(k);
      A_om1_R1 += omega_eta1[j] * omega_eta1[k];
      A_om1n_R1 += omega_eta1_new[j] * omega_eta1_new[k];
    }
  }

  // Prior ratio (scaled Beta)
  double log_prior_eta1 = log_beta_pdf(eta1_new / eta1_sd, e, f) -
                          log_beta_pdf(eta1 / eta1_sd, e, f);

  // Log MH ratio = log_target + log_aux + log_norm + log_prop
  double log_target_eta1 = (eta1_new - eta1) * B_R1 + log_prior_eta1;
  double log_aux_eta1 =
      mu_tilde * (sum_om1n - sum_om1) + eta1_tilde * (A_om1n_R1 - A_om1_R1);
  double log_norm_eta1 =
      mu * (sum_om1 - sum_om1n) + eta1 * A_om1_R1 - eta1_new * A_om1n_R1;
  double log_MH_eta1 =
      log_target_eta1 + log_aux_eta1 + log_norm_eta1 + log_prop_ratio_eta1;

  if (std::log(R::runif(0.0, 1.0)) < log_MH_eta1)
    eta1 = eta1_new;
}


} // anonymous namespace
// =============================================================================
// MAIN FUNCTION: Single-Network GGM Bayesian Variable Selection
// Uses only ONE adjacency source (GGM-estimated from Wang et al. 2012)
// =============================================================================

// [[Rcpp::export]]
Rcpp::List BayesLogit_SingleNet_GGM(
    const arma::mat &X, const arma::vec &y, const arma::mat &S_ggm,
    double n_ggm,
    int niter, int burnin,
    double mu, double nu0, double sigmasq0, double alpha0, double beta0,
    double h, double e, double f,
    double v0_ggm, double v1_ggm, double pii_ggm, double lambda_ggm,
    double eta1_sd, double mu_tilde, double eta1_tilde, unsigned int T_max,
    int proposal_type,
    int thin = 1,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0) {
  Rcpp::RNGScope scope;

  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  if (S_ggm.n_cols != p)
    Rcpp::stop("GGM S matrix dim mismatch");
  if (thin < 1)
    thin = 1;

  // 1. INITIALIZATION
  arma::vec beta(p, arma::fill::zeros);
  arma::uvec gamma(p, arma::fill::zeros);
  if (beta_in.isNotNull())
    beta = Rcpp::as<arma::vec>(beta_in);
  if (gamma_in.isNotNull())
    gamma = Rcpp::as<arma::uvec>(gamma_in);

  double alpha = alpha_in;
  double sigmasq = 1.0;
  double eta1 = 0.01;

  arma::vec z = X * beta;
  double loglik = calc_loglik(y, z, alpha);

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

  // OPT: Pre-allocate z_prop buffer (reused every proposal)
  arma::vec z_prop(n);

  // OPT: Pre-allocate active indices buffer (avoids arma::find allocation)
  std::vector<arma::uword> active_idx;
  active_idx.reserve(p);

  // OPT: Pre-allocate sparse edge storage (reused every saved iteration)
  std::vector<int> edge_rows, edge_cols;
  const arma::uword max_edges = p * (p - 1) / 2;
  edge_rows.reserve(std::min(max_edges, (arma::uword)1000));
  edge_cols.reserve(std::min(max_edges, (arma::uword)1000));

  // Output Storage (thinned)
  int n_save = niter / thin;
  arma::mat beta_out(n_save, p);
  arma::umat gamma_out(n_save, p);
  arma::vec eta1_out(n_save);
  arma::vec alpha_out(n_save);
  arma::vec sigmasq_out(n_save);

  // Sparse Z storage
  Rcpp::List Z_list(n_save);

  // 2. MCMC LOOP
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

    // ---------------------------------------------------------
    // STEP A: Update Dynamic Network (GGM via SSVS, Wang et al. 2012)
    // ---------------------------------------------------------
    C_ggm = arma::inv_sympd(arma::symmatu(Sig_ggm));

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

      invC11 -= (1.0 / Sigii) * (Sig12 * Sig12.t());
      Ci = (S_ggm(i, i) + lambda_ggm) * invC11;
      Ci.diag() += 1.0 / tau_temp;

      bool ok = arma::chol(U_ggm, arma::symmatu(Ci));
      if (!ok) {
        Ci.diag() += 1e-8;
        arma::chol(U_ggm, arma::symmatu(Ci));
      }

      mu_i = -solve_spd_chol_upper(U_ggm, S_i);
      rnorm_into(eps_ggm);
      arma::vec delta =
          arma::solve(arma::trimatu(U_ggm), eps_ggm, arma::solve_opts::fast);
      b_ggm = mu_i + delta;

      arma::uword k = 0;
      for (arma::uword j = 0; j < p; ++j) {
        if (j == i)
          continue;
        C_ggm(j, i) = b_ggm(k);
        C_ggm(i, j) = b_ggm(k);
        k++;
      }
      double gam_val =
          R::rgamma(a_gam, 1.0 / (0.5 * (S_ggm(i, i) + lambda_ggm)));
      double cval = arma::as_scalar(b_ggm.t() * invC11 * b_ggm);
      C_ggm(i, i) = gam_val + cval;

      invC11beta = invC11 * b_ggm;
      Sig12_new = -invC11beta / gam_val;
      invC11 += (1.0 / gam_val) * (invC11beta * invC11beta.t());

      k = 0;
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

    // 1. Gamma (Variable Selection) — single-network Ising prior
    for (arma::uword kk = 0; kk < 5; ++kk) {
      arma::uword j =
          static_cast<arma::uword>(std::floor(R::runif(0.0, (double)p)));
      if (j >= p)
        j = p - 1;

      int g_curr = gamma(j);
      int g_prop = 1 - g_curr;
      double b_curr = beta(j);
      double b_prop = (g_prop == 1) ? R::rnorm(beta0, std::sqrt(sigmasq)) : 0.0;

      // OPT: reuse pre-allocated z_prop
      z_prop = z + (b_prop - b_curr) * X.col(j);
      double ll_prop = calc_loglik(y, z_prop, alpha);

      double diff = static_cast<double>(g_prop - g_curr);
      // Single-network: only GGM-estimated neighbors via Z_ggm directly
      double neigh_dyn = 0.0;
      for (arma::uword i = 0; i < p; ++i) {
        if (i == j)
          continue;
        if (Z_ggm(i, j) == 1)
          neigh_dyn += gamma(i);
      }

      double ising_diff = diff * (mu + eta1 * neigh_dyn);
      double log_ratio = (ll_prop - loglik) + ising_diff;

      if (std::log(R::runif(0, 1)) < log_ratio) {
        gamma(j) = g_prop;
        beta(j) = b_prop;
        z = z_prop;
        loglik = ll_prop;
      }
    }

    // OPT: Build active indices manually (avoids arma::find heap allocation)
    active_idx.clear();
    for (arma::uword j = 0; j < p; ++j) {
      if (gamma(j) == 1)
        active_idx.push_back(j);
    }

    // 2. Beta (Metropolis updates for active variables)
    for (arma::uword k = 0; k < active_idx.size(); ++k) {
      arma::uword j = active_idx[k];
      double b_prop = R::rnorm(beta(j), std::sqrt(sigmasq) * 0.1);
      // OPT: reuse pre-allocated z_prop
      z_prop = z + (b_prop - beta(j)) * X.col(j);
      double ll_prop = calc_loglik(y, z_prop, alpha);
      double pr_curr = -0.5 * std::pow(beta(j) - beta0, 2) / sigmasq;
      double pr_prop = -0.5 * std::pow(b_prop - beta0, 2) / sigmasq;
      if (std::log(R::runif(0, 1)) < (ll_prop - loglik) + (pr_prop - pr_curr)) {
        beta(j) = b_prop;
        z = z_prop;
        loglik = ll_prop;
      }
    }

    // 3. Alpha (Metropolis update)
    double a_prop = R::rnorm(alpha, std::sqrt(h * sigmasq));
    double ll_a_prop = calc_loglik(y, z, a_prop);
    double pr_a_curr = -0.5 * std::pow(alpha - alpha0, 2) / (h * sigmasq);
    double pr_a_prop = -0.5 * std::pow(a_prop - alpha0, 2) / (h * sigmasq);
    if (std::log(R::runif(0, 1)) <
        (ll_a_prop - loglik) + (pr_a_prop - pr_a_curr)) {
      alpha = a_prop;
      loglik = ll_a_prop;
    }

    // 4. Update SigmaSq (log-normal proposal)
    {
      double sig_prop = std::exp(std::log(sigmasq) + R::rnorm(0, 0.2));
      sig_prop = std::max(sig_prop, 1e-10);
      double shape = nu0 / 2.0;
      double scale = sigmasq0 * nu0 / 2.0;

      double lp_sig_curr = -(shape + 1.0) * std::log(sigmasq) - scale / sigmasq;
      double lp_sig_prop =
          -(shape + 1.0) * std::log(sig_prop) - scale / sig_prop;

      // OPT: in-place ss_beta using active_idx (no arma::find)
      double n_active_beta = (double)active_idx.size();
      double ss_beta = 0.0;
      for (arma::uword k = 0; k < active_idx.size(); ++k) {
        double d = beta(active_idx[k]) - beta0;
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
      if (std::log(R::runif(0, 1)) < log_mh_sig)
        sigmasq = sig_prop;
    }

    // 5. Update eta1 via Moller et al. (2006) — passes Z_ggm directly
    moller_update_eta(Z_ggm, mu, eta1, eta1_sd, mu_tilde, eta1_tilde, gamma, e,
                      f, T_max, proposal_type);

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
        alpha_out(s) = alpha;
        sigmasq_out(s) = sigmasq;

        // OPT: reuse pre-allocated edge vectors
        edge_rows.clear();
        edge_cols.clear();
        for (arma::uword r = 0; r < p; ++r) {
          for (arma::uword c = r + 1; c < p; ++c) {
            if (Z_ggm(r, c) == 1) {
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

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_out, Rcpp::Named("gamma") = gamma_out,
      Rcpp::Named("eta1") = eta1_out, Rcpp::Named("alpha") = alpha_out,
      Rcpp::Named("sigmasq") = sigmasq_out, Rcpp::Named("Z_list") = Z_list);
}
