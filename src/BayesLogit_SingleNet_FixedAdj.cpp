// [[Rcpp::depends(RcppArmadillo)]]
#include "BayesLogit_Numerics.h"
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

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
  x = std::max(1e-12, std::min(1.0 - 1e-12, x)); // guard against 0 or 1
  return std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
         (a - 1.0) * std::log(x) + (b - 1.0) * std::log(1.0 - x);
}

// =============================================================================
// PROPP-WILSON PERFECT SIMULATION (Single Adjacency)
//
// Generates omega ~ MRF:
//   P(omega | mu, eta1, R) propto
//       exp{ mu * sum(omega_j) + eta1 * sum_R omega_j omega_k }
//
// OPT: skips j==i (self) and R(i,j)==0 (no edge) in kernel
// =============================================================================
static Rcpp::IntegerVector proppwilson_omega(const Rcpp::IntegerMatrix &R1,
                                             double mu, double eta1,
                                             unsigned int T_max) {
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
          if (j == i)
            continue; // OPT: skip self
          int r1ij = R1(i, j);
          if (r1ij == 0)
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
            if (R1(m, k) == 0)
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
//
// Two proposal kernels:
//   proposal_type=0: Uniform on [max(0,eta-0.01), min(eta_sd,eta+0.01)]
//   proposal_type=1: Truncated Normal(eta, eta_sd) on (0, eta_sd)
//
// Prior: eta1/eta_sd ~ Beta(e,f)
// =============================================================================
static void moller_update_eta(const Rcpp::IntegerMatrix &R1, double mu,
                              double &eta1, double eta1_sd, double mu_tilde,
                              double eta1_tilde, const arma::uvec &gamma,
                              double e_eta, double f_eta, unsigned int T_max,
                              int proposal_type) {
  const unsigned int p = R1.ncol();
  double eta1_new;
  double log_prop_ratio = 0.0;

  if (proposal_type == 0) {
    double a1 = std::max(0.0, eta1 - 0.01);
    double b1 = std::min(eta1_sd, eta1 + 0.01);
    eta1_new = R::runif(a1, b1);
    double c1 = std::max(0.0, eta1_new - 0.01);
    double d1 = std::min(eta1_sd, eta1_new + 0.01);
    log_prop_ratio = std::log(b1 - a1) - std::log(d1 - c1);
  } else {
    int attempts = 0;
    do {
      eta1_new = R::rnorm(eta1, eta1_sd);
      if (++attempts > 10000) {
        eta1_new = eta1;
        break;
      }
    } while (eta1_new <= 0.0 || eta1_new >= eta1_sd);

    double log_q_fwd = std::log(normal_pdf(eta1_new, eta1, eta1_sd)) -
                       std::log(normal_cdf(eta1_sd, eta1, eta1_sd) -
                                normal_cdf(0.0, eta1, eta1_sd));
    double log_q_rev = std::log(normal_pdf(eta1, eta1_new, eta1_sd)) -
                       std::log(normal_cdf(eta1_sd, eta1_new, eta1_sd) -
                                normal_cdf(0.0, eta1_new, eta1_sd));
    log_prop_ratio = log_q_rev - log_q_fwd;
  }

  eta1_new = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1_new));

  // Generate auxiliary variables via Propp-Wilson
  Rcpp::IntegerVector omega_curr = proppwilson_omega(R1, mu, eta1, T_max);
  Rcpp::IntegerVector omega_new = proppwilson_omega(R1, mu, eta1_new, T_max);

  // Compute sufficient statistics
  int B_R1 = 0;
  int A_curr_R1 = 0, A_new_R1 = 0;
  int sum_curr = 0, sum_new = 0;

  for (unsigned int j = 0; j < p; ++j) {
    sum_curr += omega_curr[j];
    sum_new += omega_new[j];
    for (unsigned int k = j + 1; k < p; ++k) {
      int r1jk = R1(j, k);
      if (r1jk == 0)
        continue;
      B_R1 += (int)gamma(j) * (int)gamma(k);
      A_curr_R1 += omega_curr[j] * omega_curr[k];
      A_new_R1 += omega_new[j] * omega_new[k];
    }
  }

  // Prior ratio (scaled Beta)
  double log_prior = log_beta_pdf(eta1_new / eta1_sd, e_eta, f_eta) -
                     log_beta_pdf(eta1 / eta1_sd, e_eta, f_eta);

  // Log MH ratio (Möller et al. 2006)
  double log_target = (eta1_new - eta1) * B_R1 + log_prior;
  double log_aux =
      mu_tilde * (sum_new - sum_curr) + eta1_tilde * (A_new_R1 - A_curr_R1);
  double log_norm =
      mu * (sum_curr - sum_new) + eta1 * A_curr_R1 - eta1_new * A_new_R1;

  double log_MH = log_target + log_aux + log_norm + log_prop_ratio;

  if (bvs_dadj::safe_mh_accept(log_MH))
    eta1 = eta1_new;
}

} // anonymous namespace
// =============================================================================
// MAIN FUNCTION: Single Fixed Adjacency Bayesian Variable Selection
//
// Takes ONE fixed adjacency matrix (from external information).
// No GGM estimation — adjacency is fixed throughout the MCMC.
// =============================================================================

// [[Rcpp::export]]
Rcpp::List BayesLogit_SingleNet_FixedAdj(
    const arma::mat &X, const arma::vec &y, Rcpp::IntegerMatrix R_fix_int,
    int niter, int burnin, double mu, double nu0, double sigmasq0,
    double alpha0, double beta0, double h, double e_eta, double f_eta,
    double eta1_sd, double mu_tilde, double eta1_tilde, unsigned int T_max,
    int proposal_type, int thin = 1, int n_thin_gb = 3,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0) {
  Rcpp::RNGScope scope;

  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  if ((unsigned int)R_fix_int.nrow() != p ||
      (unsigned int)R_fix_int.ncol() != p)
    Rcpp::stop("R_fix dimensions must match p = %d", p);
  if (thin < 1)
    thin = 1;
  if (n_thin_gb < 1)
    n_thin_gb = 1;

  // 1. INITIALIZATION
  arma::vec beta(p, arma::fill::zeros);
  arma::uvec gamma(p, arma::fill::zeros);
  if (beta_in.isNotNull())
    beta = Rcpp::as<arma::vec>(beta_in);
  if (gamma_in.isNotNull())
    gamma = Rcpp::as<arma::uvec>(gamma_in);

  double alpha = alpha_in;
  double sigmasq = 1.0;
  double eta1 = std::min(0.01, eta1_sd * 0.5);

  arma::vec z = X * beta;
  double loglik = calc_loglik(y, z, alpha);

  // OPT: Pre-allocate z_prop buffer (reused every proposal)
  arma::vec z_prop(n);

  // OPT: Pre-allocate active indices buffer (avoids arma::find allocation)
  std::vector<arma::uword> active_idx;
  active_idx.reserve(p);

  // Output Storage (thinned)
  int n_save = niter / thin;
  arma::mat beta_out(n_save, p);
  arma::umat gamma_out(n_save, p);
  arma::vec eta_out(n_save);
  arma::vec alpha_out(n_save);
  arma::vec sigmasq_out(n_save);

  // 2. MCMC LOOP
  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {

    if (iter > 0 && iter % 10000 == 0) {
      Rcpp::checkUserInterrupt();
      int model_size = (int)arma::accu(gamma);
      Rcpp::Rcout << "Iter: " << iter << " | Model size: " << model_size
                  << " | eta: " << eta1 << "\n";
    }

    // ---------------------------------------------------------
    // STEP A+B: Gamma + Beta with inner thinning (n_thin_gb rounds)
    // ---------------------------------------------------------
    for (int thin_gb = 0; thin_gb < n_thin_gb; ++thin_gb) {

      // STEP A: Gamma (Variable Selection) via MH with single Ising prior
      for (arma::uword kk = 0; kk < 5; ++kk) {
        arma::uword j =
            static_cast<arma::uword>(std::floor(R::runif(0.0, (double)p)));
        if (j >= p)
          j = p - 1;

        int g_curr = gamma(j);
        int g_prop = 1 - g_curr;
        double b_curr = beta(j);
        double b_prop =
            (g_prop == 1) ? R::rnorm(beta0, std::sqrt(sigmasq)) : 0.0;

        // OPT: reuse pre-allocated z_prop
        z_prop = z + (b_prop - b_curr) * X.col(j);
        double ll_prop = calc_loglik(y, z_prop, alpha);

        double diff = static_cast<double>(g_prop - g_curr);
        // Single adjacency: only fixed network neighbors
        double neigh = 0.0;
        for (arma::uword i = 0; i < p; ++i) {
          if (i == j)
            continue;
          if (R_fix_int(i, j) != 0)
            neigh += gamma(i);
        }

        double ising_diff = diff * (mu + eta1 * neigh);
        double log_ratio = (ll_prop - loglik) + ising_diff;

        if (bvs_dadj::safe_mh_accept(log_ratio)) {
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

      // STEP B: Beta (Metropolis updates for active variables)
      for (arma::uword k = 0; k < active_idx.size(); ++k) {
        arma::uword j = active_idx[k];
        double b_prop = R::rnorm(beta(j), std::sqrt(sigmasq) * 0.1);
        // OPT: reuse pre-allocated z_prop
        z_prop = z + (b_prop - beta(j)) * X.col(j);
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
      sig_prop = std::max(sig_prop, 1e-10); // OPT: guard against underflow

      double shape = nu0 / 2.0;
      double scale = sigmasq0 * nu0 / 2.0;

      double lp_sig_curr = -(shape + 1.0) * std::log(sigmasq) - scale / sigmasq;
      double lp_sig_prop =
          -(shape + 1.0) * std::log(sig_prop) - scale / sig_prop;

      // OPT: in-place ss_beta using active_idx (no arma::find, no temp vec)
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
      if (bvs_dadj::safe_mh_accept(log_mh_sig))
        sigmasq = sig_prop;
    }

    // ---------------------------------------------------------
    // STEP E: Update eta via Moller et al. (2006) + Propp-Wilson
    // ---------------------------------------------------------
    moller_update_eta(R_fix_int, mu, eta1, eta1_sd, mu_tilde, eta1_tilde, gamma,
                      e_eta, f_eta, T_max, proposal_type);

    // ---------------------------------------------------------
    // STORE SAMPLES (with thinning)
    // ---------------------------------------------------------
    if (iter >= burnin && (iter - burnin) % thin == 0) {
      int s = (iter - burnin) / thin;
      if (s < n_save) {
        beta_out.row(s) = beta.t();
        for (arma::uword j = 0; j < p; ++j)
          gamma_out(s, j) = gamma(j);
        eta_out(s) = eta1;
        alpha_out(s) = alpha;
        sigmasq_out(s) = sigmasq;
      }
    }
  }

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_out, Rcpp::Named("gamma") = gamma_out,
      Rcpp::Named("eta1") = eta_out, Rcpp::Named("alpha") = alpha_out,
      Rcpp::Named("sigmasq") = sigmasq_out);
}
