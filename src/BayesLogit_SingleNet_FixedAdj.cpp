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

// Gaussian Log-Likelihood for continuous outcome
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
    double alpha_in = 0.0, const arma::mat &Z_dat = arma::mat(),
    double tau0 = 0.0, double htau = 1.0,
    Rcpp::Nullable<Rcpp::NumericVector> tau_in = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> event = R_NilValue,
    std::string outcome_type = "binary") {
  Rcpp::RNGScope scope;
  const bool is_continuous = bvs_dadj::outcome_is_continuous(outcome_type);
  const bool is_tte = bvs_dadj::outcome_is_tte(outcome_type);
  const bool is_count = bvs_dadj::outcome_is_count(outcome_type);

  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  if ((unsigned int)R_fix_int.nrow() != p ||
      (unsigned int)R_fix_int.ncol() != p)
    Rcpp::stop("R_fix dimensions must match p = %d", p);
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
  double eta1 = std::min(0.01, eta1_sd * 0.5);
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
  if (is_count) {
    w_count.set_size(n);
    log_w_count.set_size(n);
    w_count.fill(1.0);
    log_w_count.zeros();
  }
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
    loglik = -0.5 * n * std::log(2.0 * M_PI * sigmasq) -
             0.5 * arma::dot(resid, resid) / sigmasq;
  } else if (is_tte) {
    cox_tracker.init(Xb_total, cox_data);
    loglik = cox_tracker.get_loglik();
  } else if (is_count) {
    refresh_count_latent(Xb_total, alpha);
    loglik =
        calc_loglik_count(y_count, Xb_total - Z_tau, alpha, Z_tau, log_w_count);
  } else {
    loglik = calc_loglik_binary(y, Xb_total - Z_tau, alpha, Z_tau);
  }

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
  arma::mat tau_out(n_save, ntau);

  // 2. MCMC LOOP
  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {

    if (iter > 0 && iter % 10000 == 0) {
      Rcpp::checkUserInterrupt();
      int model_size = (int)arma::accu(gamma);
      Rcpp::Rcout << "Iter: " << iter << " | Model size: " << model_size
                  << " | eta: " << eta1 << "\n";
    }

    if (is_count) {
      refresh_count_latent(Xb_total, alpha);
      loglik = calc_loglik_count(y_count, Xb_total - Z_tau, alpha, Z_tau,
                                 log_w_count);
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

        double diff = static_cast<double>(g_prop - g_curr);
        double db = b_prop - b_curr;
        double ll_diff = 0.0;
        const arma::vec &xj = X.col(j);
        std::vector<double> delta_group_W;

        if (is_tte) {
          delta_group_W.assign(cox_data.group_start.size(), 0.0);
        }

        if (is_continuous) {
          ll_diff =
              (arma::dot(xj, resid) * db - 0.5 * X_col_sq_sums(j) * db * db) /
              sigmasq;
        } else if (is_tte) {
          ll_diff = cox_tracker.propose_diff(xj, db, delta_group_W);
        } else {
          arma::vec z_prop = Xb_total - Z_tau + db * xj;
          if (is_count) {
            ll_diff =
                calc_loglik_count(y_count, z_prop, alpha, Z_tau, log_w_count) -
                loglik;
          } else {
            ll_diff = calc_loglik_binary(y, z_prop, alpha, Z_tau) - loglik;
          }
        }

        // Single adjacency: only fixed network neighbors
        double neigh = 0.0;
        for (arma::uword i = 0; i < p; ++i) {
          if (i == j)
            continue;
          if (R_fix_int(i, j) != 0)
            neigh += gamma(i);
        }

        double ising_diff = diff * (mu + eta1 * neigh);
        double log_ratio = ll_diff + ising_diff;

        if (bvs_dadj::safe_mh_accept(log_ratio)) {
          gamma(j) = g_prop;
          beta(j) = b_prop;
          if (is_continuous) {
            resid -= db * xj;
            loglik += ll_diff;
            Xb_total += db * xj; // Keep Xb_total updated for other steps
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

      // OPT: Build active indices manually (avoids arma::find heap allocation)
      active_idx.clear();
      for (arma::uword j = 0; j < p; ++j) {
        if (gamma(j) == 1)
          active_idx.push_back(j);
      }

      // STEP B: Beta updates — MALA for binary/count, Fisher-RW for TTE
      if (!is_continuous) {
        // Refresh MALA residuals from current Xb_total (after Step A gamma
        // flips)
        if (!is_tte) {
          if (is_count) {
            for (arma::uword i = 0; i < n; ++i) {
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
        // Pre-compute H_vec for TTE Fisher information scaling (once per loop)
        arma::vec cox_H;
        if (is_tte)
          cox_tracker.compute_H_vec(cox_H);

        for (arma::uword k = 0; k < active_idx.size(); ++k) {
          arma::uword j = active_idx[k];
          const arma::vec &xj = X.col(j);
          double b_prop, db;

          if (is_tte) {
            // Fisher information-scaled random walk for TTE
            double info_jj = cox_tracker.compute_info_diag_j(xj, cox_H);
            double prop_sd = (info_jj > 1e-10)
                                 ? std::min(1.0 / std::sqrt(info_jj), 2.0)
                                 : 0.5;
            b_prop = R::rnorm(beta(j), prop_sd);
            db = b_prop - beta(j);
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
              // Refresh H_vec after each TTE accept for accurate scaling
              cox_tracker.compute_H_vec(cox_H);
            }
          } else {
            // Component-wise MALA for binary/count
            double g_j =
                arma::dot(xj, mala_resid) - (beta(j) - beta0) / sigmasq;
            double h_j =
                std::min(0.5 * sigmasq / (X_col_sq_sums(j) + 1.0), 1.0);
            double mean_fwd = beta(j) + 0.5 * h_j * g_j;
            b_prop = R::rnorm(mean_fwd, std::sqrt(h_j));
            db = b_prop - beta(j);

            // Single O(n) pass: compute ll_prop and backward gradient
            double ll_prop = 0.0;
            double g_j_back = -(b_prop - beta0) / sigmasq;
            for (arma::uword i = 0; i < n; ++i) {
              if (is_count) {
                const double eta_p = bvs_dadj::clamp_finite(
                    alpha + Xb_total(i) + db * xj(i) + log_w_count(i), -50.0,
                    50.0, 0.0);
                const double lam_p = std::exp(eta_p);
                ll_prop += (double)y_count(i) * eta_p - lam_p;
                g_j_back += ((double)y_count(i) - lam_p) * xj(i);
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
                sigmasq;
            if (bvs_dadj::safe_mh_accept(ll_diff + pr_diff + lq_bwd - lq_fwd)) {
              beta(j) = b_prop;
              Xb_total += db * xj;
              loglik = ll_prop;
              // Incrementally refresh mala_resid from updated Xb_total
              if (is_count) {
                for (arma::uword i = 0; i < n; ++i) {
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
    // STEP C: Alpha — MALA for binary/count, exact Gibbs for continuous
    // ---------------------------------------------------------
    if (!is_continuous && !is_tte) {
      // Ensure mala_resid is current (refreshed at end of thin_gb loop)
      double sum_resid = arma::accu(mala_resid);
      double g_alpha = sum_resid - (alpha - alpha0) / (h * sigmasq);
      double h_alpha = std::min(0.5 * h * sigmasq / ((double)n + 1.0 / h), 1.0);
      double mean_fwd_a = alpha + 0.5 * h_alpha * g_alpha;
      double a_prop = R::rnorm(mean_fwd_a, std::sqrt(h_alpha));
      // O(n) pass: ll_prop and backward gradient
      double ll_a_prop = 0.0;
      double g_alpha_back = -(a_prop - alpha0) / (h * sigmasq);
      for (arma::uword i = 0; i < n; ++i) {
        if (is_count) {
          const double eta = bvs_dadj::clamp_finite(
              a_prop + Xb_total(i) + log_w_count(i), -50.0, 50.0, 0.0);
          const double lam = std::exp(eta);
          ll_a_prop += (double)y_count(i) * eta - lam;
          g_alpha_back += (double)y_count(i) - lam;
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
        // Refresh mala_resid with updated alpha
        if (is_count) {
          for (arma::uword i = 0; i < n; ++i) {
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
      double mean =
          (arma::accu(resid) + alpha * static_cast<double>(n) + alpha0 / h) /
          denom;
      double a_new = R::rnorm(mean, std::sqrt(sigmasq / denom));
      double da = a_new - alpha;
      if (std::abs(da) > 0.0) {
        alpha = a_new;
        resid -= da;
      }
      loglik = -0.5 * n * std::log(2.0 * M_PI * sigmasq) -
               0.5 * arma::dot(resid, resid) / sigmasq;
    }

    // ---------------------------------------------------------
    // STEP C+: Tau — component-wise MALA (binary/count) or block MH (TTE)
    // ---------------------------------------------------------
    if (ntau > 0) {
      if (!is_continuous) {
        if (is_tte) {
          // Block random-walk MH for tau under TTE (CoxTracker re-init)
          arma::vec tau_prop(ntau);
          for (arma::uword j = 0; j < ntau; ++j)
            tau_prop(j) = R::rnorm(tau(j), std::sqrt(htau * sigmasq));
          arma::vec Z_tau_prop = Z_dat * tau_prop;
          arma::vec Xb_total_prop(n);
          for (arma::uword i = 0; i < n; ++i)
            Xb_total_prop(i) = Xb_total(i) - Z_tau(i) + Z_tau_prop(i);
          double ll_tau_prop =
              bvs_dadj::cox_loglik_breslow(Xb_total_prop, cox_data);
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
            double inv_htau = 1.0 / htau;
            double h_k = std::min(
                0.5 * htau * sigmasq / (Z_col_sq_sums(k) + inv_htau), 1.0);
            double mean_fwd_k = tau(k) + 0.5 * h_k * g_k;
            double t_prop = R::rnorm(mean_fwd_k, std::sqrt(h_k));
            double dt = t_prop - tau(k);
            // O(n) pass: ll_prop and backward gradient
            double ll_t_prop = 0.0;
            double g_k_back = -(t_prop - tau0) / (htau * sigmasq);
            for (arma::uword i = 0; i < n; ++i) {
              const double new_eta_base = alpha + Xb_total(i) + dt * zk(i);
              if (is_count) {
                const double eta_p = bvs_dadj::clamp_finite(
                    new_eta_base + log_w_count(i), -50.0, 50.0, 0.0);
                const double lam_p = std::exp(eta_p);
                ll_t_prop += (double)y_count(i) * eta_p - lam_p;
                g_k_back += ((double)y_count(i) - lam_p) * zk(i);
              } else {
                const double p_p = 1.0 / (1.0 + std::exp(-new_eta_base));
                ll_t_prop +=
                    y(i) * new_eta_base -
                    (new_eta_base > 0.0
                         ? new_eta_base + std::log1p(std::exp(-new_eta_base))
                         : std::log1p(std::exp(new_eta_base)));
                g_k_back += (y(i) - p_p) * zk(i);
              }
            }
            double ll_diff_k = ll_t_prop - loglik;
            double mean_bwd_k = t_prop + 0.5 * h_k * g_k_back;
            double lq_fwd_k =
                -0.5 * (t_prop - mean_fwd_k) * (t_prop - mean_fwd_k) / h_k;
            double lq_bwd_k =
                -0.5 * (tau(k) - mean_bwd_k) * (tau(k) - mean_bwd_k) / h_k;
            double pr_diff_k =
                -0.5 *
                (std::pow(t_prop - tau0, 2) - std::pow(tau(k) - tau0, 2)) /
                (htau * sigmasq);
            if (bvs_dadj::safe_mh_accept(ll_diff_k + pr_diff_k + lq_bwd_k -
                                         lq_fwd_k)) {
              tau(k) = t_prop;
              Xb_total += dt * zk;
              Z_tau += dt * zk;
              loglik = ll_t_prop;
              // Refresh mala_resid
              if (is_count) {
                for (arma::uword i = 0; i < n; ++i) {
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
        loglik = -0.5 * n * std::log(2.0 * M_PI * sigmasq) -
                 0.5 * arma::dot(resid, resid) / sigmasq;
      }
    }

    // ---------------------------------------------------------
    // STEP D: SigmaSq — exact IG Gibbs (non-continuous) or IG Gibbs
    // (continuous) p(sigmasq | rest) = IG(shape_post, scale_post) for all
    // outcome types:
    //   shape = (nu0 + k_active + 1 + ntau) / 2
    //   scale = (nu0*sigmasq0 + ss_beta + (alpha-alpha0)^2/h + ss_tau/htau) / 2
    // (likelihood for binary/count/TTE doesn't depend on sigmasq — only priors
    // do)
    // ---------------------------------------------------------
    {
      if (!is_continuous) {
        double ss_beta = 0.0;
        for (arma::uword k = 0; k < active_idx.size(); ++k) {
          const double d = beta(active_idx[k]) - beta0;
          ss_beta += d * d;
        }
        double ss_tau = 0.0;
        for (arma::uword j = 0; j < ntau; ++j) {
          const double d = tau(j) - tau0;
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
        loglik =
            -0.5 * n * std::log(2.0 * M_PI * sigmasq) - 0.5 * sse / sigmasq;
      }
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
        tau_out.row(s) = tau.t();
      }
    }
  }

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_out, Rcpp::Named("gamma") = gamma_out,
      Rcpp::Named("eta1") = eta_out, Rcpp::Named("alpha") = alpha_out,
      Rcpp::Named("sigmasq") = sigmasq_out, Rcpp::Named("tau") = tau_out);
}
