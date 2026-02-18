// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

namespace {

// =============================================================================
// Polya-Gamma(1, z) approximate sampler (truncated series, K=200 terms)
// =============================================================================
static double sample_pg_approx(double z) {
  z = std::abs(z) * 0.5;
  const double c2 = z * z;
  const double PI2 = M_PI * M_PI;
  const double INV_2PI2 = 1.0 / (2.0 * PI2);

  double sum = 0.0;
  for (int k = 0; k < 200; ++k) {
    double g = R::rexp(1.0);
    double kh = k + 0.5;
    double den = kh * kh + c2 / PI2;
    sum += g / den;
  }
  return sum * INV_2PI2;
}

// =============================================================================
// Log-likelihood for logistic regression (numerically stable)
// =============================================================================
static inline double calc_loglik(const arma::vec &y01, const arma::vec &z,
                                 double alpha) {
  const arma::uword n = y01.n_elem;
  double term1 = 0.0, term2 = 0.0;
  for (arma::uword i = 0; i < n; ++i) {
    double x = alpha + z(i);
    term1 += y01(i) * x;
    if (x > 0.0)
      term2 += x + std::log1p(std::exp(-x));
    else
      term2 += std::log1p(std::exp(x));
  }
  return term1 - term2;
}

// =============================================================================
// Logistic helper (stable for large |x|)
// =============================================================================
static inline double logistic(double x) {
  if (x >= 0.0) {
    double z = std::exp(-x);
    return 1.0 / (1.0 + z);
  }
  double z = std::exp(x);
  return z / (1.0 + z);
}

// =============================================================================
// Normal PDF/CDF support (for truncated normal eta proposals)
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
// Log Beta density: log Beta(x | a, b) with safe clamping
// =============================================================================
static inline double log_beta_pdf(double x, double a, double b) {
  x = std::max(1e-12, std::min(1.0 - 1e-12, x));
  return std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
         (a - 1.0) * std::log(x) + (b - 1.0) * std::log(1.0 - x);
}

// =============================================================================
// Adjacency preprocessing (neighbors + edge list)
// =============================================================================
using Index = arma::uword;
using NeighborList = std::vector<std::vector<Index>>;
using EdgeList = std::vector<std::pair<Index, Index>>;

static void build_neighbors_edges(const Rcpp::IntegerMatrix &R1,
                                  NeighborList &neighbors, EdgeList &edges) {
  const Index p = R1.ncol();
  neighbors.assign(p, {});
  edges.clear();
  for (Index i = 0; i < p; ++i) {
    for (Index j = i + 1; j < p; ++j) {
      const int r = (R1(i, j) != 0 || R1(j, i) != 0) ? 1 : 0;
      if (r != 0) {
        neighbors[i].push_back(j);
        neighbors[j].push_back(i);
        edges.emplace_back(i, j);
      }
    }
  }
}

// =============================================================================
// PROPP-WILSON PERFECT SIMULATION (SINGLE ADJACENCY)
// =============================================================================
static Rcpp::IntegerVector proppwilson_omega_1eta(const NeighborList &neighbors,
                                                  double mu, double eta,
                                                  unsigned int T_max) {
  const unsigned int p = static_cast<unsigned int>(neighbors.size());
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
        const auto &nb = neighbors[i];
        for (Index idx = 0; idx < nb.size(); ++idx) {
          Index j = nb[idx];
          ker_up += eta * x_up[j];
          ker_down += eta * x_down[j];
        }
        pi_up[i] = logistic(mu + ker_up);
        pi_down[i] = logistic(mu + ker_down);

        double u = unif01(gen);
        x_up[i] = (pi_up[i] > u) ? 1 : 0;
        x_down[i] = (pi_down[i] > u) ? 1 : 0;
      }
    }

    T = 2 * T;

    if (T >= T_max) {
      std::vector<unsigned int> diff_idx;
      diff_idx.reserve(p);
      for (unsigned int k = 0; k < p; ++k)
        if (x_up[k] != x_down[k])
          diff_idx.push_back(k);

      std::mt19937 gen_fb(seed_base + 99999);
      std::uniform_real_distribution<double> unif01_fb(0.0, 1.0);

      for (int sweep = 0; sweep < 100; ++sweep) {
        for (unsigned int idx = 0; idx < diff_idx.size(); ++idx) {
          unsigned int m = diff_idx[idx];
          double ker = 0.0;
          const auto &nb = neighbors[m];
          for (Index k = 0; k < nb.size(); ++k)
            ker += eta * x_up[nb[k]];
          double prob = logistic(mu + ker);
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
// Moller MH update for eta (single adjacency), using unique-pair (j<k)
// =============================================================================
static void moller_update_eta_1(const NeighborList &neighbors,
                                const EdgeList &edges, double mu, double &eta,
                                double eta_sd, double mu_tilde,
                                double eta_tilde, const arma::uvec &gamma,
                                double e, double f, unsigned int T_max,
                                int proposal_type) {
  const unsigned int p = static_cast<unsigned int>(neighbors.size());

  double eta_new;
  double log_prop_ratio = 0.0;

  if (proposal_type == 0) {
    double a = std::max(0.0, eta - 0.01);
    double b = std::min(eta_sd, eta + 0.01);
    eta_new = R::runif(a, b);

    double c = std::max(0.0, eta_new - 0.01);
    double d = std::min(eta_sd, eta_new + 0.01);
    log_prop_ratio = std::log(b - a) - std::log(d - c);
  } else {
    int attempts = 0;
    do {
      eta_new = R::rnorm(eta, eta_sd);
      if (++attempts > 10000) {
        eta_new = eta;
        break;
      }
    } while (eta_new <= 0.0 || eta_new >= eta_sd);

    double denom_fwd = std::max(1e-16, normal_cdf(eta_sd, eta, eta_sd) -
                                           normal_cdf(0.0, eta, eta_sd));
    double denom_rev = std::max(1e-16, normal_cdf(eta_sd, eta_new, eta_sd) -
                                           normal_cdf(0.0, eta_new, eta_sd));

    double log_q_fwd =
        std::log(normal_pdf(eta_new, eta, eta_sd)) - std::log(denom_fwd);
    double log_q_rev =
        std::log(normal_pdf(eta, eta_new, eta_sd)) - std::log(denom_rev);

    log_prop_ratio = log_q_rev - log_q_fwd;
  }

  eta_new = std::max(1e-8, std::min(eta_sd - 1e-8, eta_new));

  Rcpp::IntegerVector omega_cur =
      proppwilson_omega_1eta(neighbors, mu, eta, T_max);
  Rcpp::IntegerVector omega_new =
      proppwilson_omega_1eta(neighbors, mu, eta_new, T_max);

  int B_R = 0;
  int A_om_R = 0, A_om_new_R = 0;
  int sum_om = 0, sum_om_new = 0;

  for (unsigned int j = 0; j < p; ++j) {
    sum_om += omega_cur[j];
    sum_om_new += omega_new[j];
  }
  for (const auto &e : edges) {
    Index j = e.first;
    Index k = e.second;
    B_R += (int)gamma(j) * (int)gamma(k);
    A_om_R += omega_cur[j] * omega_cur[k];
    A_om_new_R += omega_new[j] * omega_new[k];
  }

  double log_prior =
      log_beta_pdf(eta_new / eta_sd, e, f) - log_beta_pdf(eta / eta_sd, e, f);

  double log_target = (eta_new - eta) * (double)B_R + log_prior;
  double log_aux = mu_tilde * (double)(sum_om_new - sum_om) +
                   eta_tilde * (double)(A_om_new_R - A_om_R);
  double log_norm = mu * (double)(sum_om - sum_om_new) + eta * (double)A_om_R -
                    eta_new * (double)A_om_new_R;

  double log_MH = log_target + log_aux + log_norm + log_prop_ratio;

  if (std::log(R::runif(0.0, 1.0)) < log_MH)
    eta = eta_new;
}

} // anonymous namespace
// =============================================================================
// [[Rcpp::export]]
Rcpp::List BayesLogit_PG_SingleAdj(
    const arma::mat &X, const arma::vec &y, Rcpp::IntegerMatrix R_adj_int,
    int niter, int burnin, double mu, double nu0, double sigmasq0,
    double alpha0, double beta0, double h, int n_mh_gamma, double eta_sd,
    double mu_tilde, double eta_tilde, double e, double f, unsigned int T_max,
    int proposal_type, int thin = 1,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0, double eta_init = 0.01, double sigmasq_init = 1.0) {
  Rcpp::RNGScope scope;

  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;

  if ((unsigned int)R_adj_int.nrow() != p ||
      (unsigned int)R_adj_int.ncol() != p)
    Rcpp::stop("R_adj dimensions must match p.");

  if (thin < 1)
    thin = 1;

  // Output Storage
  int n_save = niter / thin;
  arma::mat beta_out(n_save, p);
  arma::umat gamma_out(n_save, p);
  arma::vec eta_out(n_save);
  arma::vec alpha_out(n_save);
  arma::vec sigmasq_out(n_save);

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
  double eta = eta_init;

  // --- enforce y in {0,1}; auto-convert {-1,1} -> {0,1} ---
  arma::vec y01 = y;
  bool is01 = true, is11 = true;
  for (arma::uword i = 0; i < y01.n_elem; ++i) {
    double yi = y01(i);
    if (std::fabs(yi - 0.0) > 1e-12 && std::fabs(yi - 1.0) > 1e-12)
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

  // --- adjacency precompute (neighbors + edge list) ---
  NeighborList neighbors;
  EdgeList edges;
  build_neighbors_edges(R_adj_int, neighbors, edges);

  // HARD RULE: gamma=0 => beta=0
  beta.elem(arma::find(gamma == 0)).zeros();

  arma::vec z = X * beta;
  arma::vec omega_pg(n, arma::fill::ones);
  const arma::vec kappa = y01 - 0.5;

  arma::vec lin(n);

  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {

    if (iter % 5000 == 0 && iter > 0) {
      Rcpp::checkUserInterrupt();
      Rcpp::Rcout << "Iter: " << iter
                  << " | Model size: " << (int)arma::accu(gamma)
                  << " | eta: " << eta << " | sigmasq: " << sigmasq << "\n";
    }

    // ===================== STEP A: PG + beta/alpha =====================
    lin = alpha + X * beta;
    for (arma::uword i = 0; i < n; ++i)
      omega_pg(i) = sample_pg_approx(lin(i));

    arma::uvec active = arma::find(gamma == 1);
    const arma::uword p_active = active.n_elem;

    // HARD RULE: beta inactive = 0 always
    beta.zeros();

    if (p_active > 0) {
      arma::mat X_act = X.cols(active);

      arma::mat Xt_Om = X_act.t();
      Xt_Om.each_row() %= omega_pg.t();

      arma::mat prec_beta = Xt_Om * X_act;
      prec_beta.diag() += 1.0 / sigmasq;
      // Enforce symmetry to avoid numerical warnings in chol()
      arma::mat prec_sym = 0.5 * (prec_beta + prec_beta.t());

      arma::vec z_star = kappa - omega_pg * alpha;
      arma::vec mean_rhs = X_act.t() * z_star +
                           (beta0 / sigmasq) * arma::ones<arma::vec>(p_active);

      arma::mat L_prec;
      bool ok = arma::chol(L_prec, arma::symmatu(prec_sym));
      if (!ok) {
        double jitter = 1e-8;
        for (int attempt = 0; attempt < 5 && !ok; ++attempt) {
          prec_sym.diag() += jitter;
          ok = arma::chol(L_prec, arma::symmatu(prec_sym));
          jitter *= 10.0;
        }
      }

      if (ok) {
        arma::vec tmp = arma::solve(arma::trimatl(L_prec.t()), mean_rhs,
                                    arma::solve_opts::fast);
        arma::vec m_beta =
            arma::solve(arma::trimatu(L_prec), tmp, arma::solve_opts::fast);

        arma::vec zz = arma::randn<arma::vec>(p_active);
        arma::vec noise =
            arma::solve(arma::trimatu(L_prec), zz, arma::solve_opts::fast);

        beta.elem(active) = m_beta + noise;
      }
    }

    // alpha
    {
      double sum_omega = arma::accu(omega_pg);
      double prec_alpha = sum_omega + 1.0 / (h * sigmasq);
      double var_alpha = 1.0 / prec_alpha;

      arma::vec resid = kappa - omega_pg % (X * beta);
      double mean_alpha =
          var_alpha * (arma::accu(resid) + alpha0 / (h * sigmasq));
      alpha = R::rnorm(mean_alpha, std::sqrt(var_alpha));
    }

    // ===================== STEP B: gamma MH (hard beta=0 if gamma=0) =====
    z = X * beta;
    double loglik = calc_loglik(y01, z, alpha);

    for (int mh = 0; mh < n_mh_gamma; ++mh) {
      int j = (int)std::floor(R::runif(0.0, (double)p));
      if (j >= (int)p)
        j = (int)p - 1;
      arma::uword ju = (arma::uword)j;

      if (gamma(ju) == 0)
        beta(ju) = 0.0;

      int g_curr = (int)gamma(ju);
      int g_prop = 1 - g_curr;

      double b_curr = beta(ju);
      double b_prop = (g_prop == 1) ? R::rnorm(beta0, std::sqrt(sigmasq)) : 0.0;

      arma::vec z_prop = z + (b_prop - b_curr) * X.col(ju);
      double ll_prop = calc_loglik(y01, z_prop, alpha);

      double diff = (double)(g_prop - g_curr);

      double neigh = 0.0;
      const auto &nb = neighbors[ju];
      for (Index k = 0; k < nb.size(); ++k) {
        neigh += (double)gamma(nb[k]);
      }

      double log_ratio = (ll_prop - loglik) + diff * (mu + eta * neigh);

      if (std::log(R::runif(0.0, 1.0)) < log_ratio) {
        gamma(ju) = (arma::uword)g_prop;
        beta(ju) = b_prop;
        z = z_prop;
        loglik = ll_prop;
      }

      if (gamma(ju) == 0)
        beta(ju) = 0.0;
    }

    beta.elem(arma::find(gamma == 0)).zeros();

    // ===================== STEP C: sigmasq strict IG Gibbs ===============
    {
      const double shape0 = 0.5 * nu0;
      const double scale0 = 0.5 * nu0 * sigmasq0;

      const double p_act = (double)arma::accu(gamma);

      double ss_b = 0.0;
      if (p_act > 0.0) {
        arma::uvec act = arma::find(gamma == 1);
        ss_b = arma::accu(arma::square(beta.elem(act) - beta0));
      }

      const double ss_a = (alpha - alpha0) * (alpha - alpha0) / h;

      const double shape_post = shape0 + 0.5 * (p_act + 1.0);
      const double scale_post = scale0 + 0.5 * (ss_b + ss_a);

      const double g = R::rgamma(shape_post, 1.0 / scale_post);
      sigmasq = 1.0 / g;
    }

    // ===================== STEP D: eta update (Moller + PW) ===============
    moller_update_eta_1(neighbors, edges, mu, eta, eta_sd, mu_tilde, eta_tilde,
                        gamma, e, f, T_max, proposal_type);

    // ===================== STORE =======================================
    // Store
    if (iter >= burnin && (iter - burnin) % thin == 0) {
      int idx = (iter - burnin) / thin;
      beta_out.row(idx) = beta.t();
      gamma_out.row(idx) = gamma.t();
      eta_out(idx) = eta;
      alpha_out(idx) = alpha;
      sigmasq_out(idx) = sigmasq;
    }
  }

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_out, Rcpp::Named("gamma") = gamma_out,
      Rcpp::Named("eta") = eta_out, Rcpp::Named("alpha") = alpha_out,
      Rcpp::Named("sigmasq") = sigmasq_out);
}
