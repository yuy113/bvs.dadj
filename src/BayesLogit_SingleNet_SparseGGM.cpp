// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {


// =============================================================================
// SECTION 1: HELPER FUNCTIONS
// =============================================================================

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

static inline double log_beta_pdf(double x, double a, double b) {
  x = std::max(1e-12, std::min(1.0 - 1e-12, x));
  return std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
         (a - 1.0) * std::log(x) + (b - 1.0) * std::log(1.0 - x);
}

// =============================================================================
// SECTION 2: SPARSE GRAPH STRUCTURE
// =============================================================================
struct SparseGraph {
  std::vector<std::vector<int>> adj;
  std::vector<std::vector<double>> val;
  int p;

  SparseGraph(int nodes) : p(nodes) {
    adj.resize(p);
    val.resize(p);
  }

  void sort_indices() {
    for (int i = 0; i < p; ++i) {
      int d = (int)adj[i].size();
      if (d > 1) {
        std::vector<std::pair<int, double>> temp(d);
        for (int k = 0; k < d; ++k)
          temp[k] = std::make_pair(adj[i][k], val[i][k]);
        std::sort(temp.begin(), temp.end());
        for (int k = 0; k < d; ++k) {
          adj[i][k] = temp[k].first;
          val[i][k] = temp[k].second;
        }
      }
    }
  }
};

// =============================================================================
// SECTION 3: SPARSE PROPP-WILSON PERFECT SIMULATION
//
// FIX: Pre-allocated workspace (x_up, x_down, result passed in).
// FIX: Binary conditional adds instead of multiplication.
// =============================================================================
static void
proppwilson_sparse(const std::vector<std::unordered_set<int>> &Z_active, int p,
                   double mu, double eta1, unsigned int T_max,
                   std::vector<int> &x_up, std::vector<int> &x_down,
                   std::vector<int> &result) {
  unsigned int T = 2;
  int seed_base = static_cast<int>(std::floor(R::runif(0.0, 1.0) * 1000.0)) + 1;

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
        // FIX: conditional adds (binary x_up/x_down)
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
        x_down[k] = x_up[k];
    }
  }
  for (int k = 0; k < p; ++k)
    result[k] = x_up[k];
}

// =============================================================================
// SECTION 4: SPARSE MOLLER MH UPDATE FOR eta
//
// FIX: Uses pre-allocated PW workspace (no heap alloc per call).
// OPT: Sufficient stats via Z_active edges: O(|E|) not O(p²).
// =============================================================================
static void moller_update_eta_sparse(
    const std::vector<std::unordered_set<int>> &Z_active, int p, double mu,
    double &eta1, double eta_sd, double mu_tilde, double eta1_tilde,
    const arma::uvec &gamma, double e, double f, unsigned int T_max,
    int proposal_type, std::vector<int> &pw_x_up, std::vector<int> &pw_x_down,
    std::vector<int> &om_curr, std::vector<int> &om_new) {
  double eta1_new;
  double log_prop_ratio = 0.0;

  if (proposal_type == 0) {
    double a1 = std::max(0.0, eta1 - 0.01);
    double b1 = std::min(eta_sd, eta1 + 0.01);
    eta1_new = R::runif(a1, b1);
    double c1 = std::max(0.0, eta1_new - 0.01);
    double d1 = std::min(eta_sd, eta1_new + 0.01);
    log_prop_ratio = std::log(b1 - a1) - std::log(d1 - c1);
  } else {
    int attempts = 0;
    do {
      eta1_new = R::rnorm(eta1, eta_sd);
      if (++attempts > 10000) {
        eta1_new = eta1;
        break;
      }
    } while (eta1_new <= 0.0 || eta1_new >= eta_sd);
    double log_q_fwd = std::log(normal_pdf(eta1_new, eta1, eta_sd)) -
                       std::log(normal_cdf(eta_sd, eta1, eta_sd) -
                                normal_cdf(0.0, eta1, eta_sd));
    double log_q_rev = std::log(normal_pdf(eta1, eta1_new, eta_sd)) -
                       std::log(normal_cdf(eta_sd, eta1_new, eta_sd) -
                                normal_cdf(0.0, eta1_new, eta_sd));
    log_prop_ratio = log_q_rev - log_q_fwd;
  }

  eta1_new = std::max(1e-8, std::min(eta_sd - 1e-8, eta1_new));

  // FIX: Uses pre-allocated workspace
  proppwilson_sparse(Z_active, p, mu, eta1, T_max, pw_x_up, pw_x_down, om_curr);
  proppwilson_sparse(Z_active, p, mu, eta1_new, T_max, pw_x_up, pw_x_down,
                     om_new);

  int B_R1 = 0;
  int A_curr_R1 = 0, A_new_R1 = 0;
  int sum_curr = 0, sum_new = 0;

  for (int j = 0; j < p; ++j) {
    sum_curr += om_curr[j];
    sum_new += om_new[j];
  }

  for (int j = 0; j < p; ++j) {
    for (int k : Z_active[j]) {
      if (k <= j)
        continue;
      B_R1 += (int)gamma(j) * (int)gamma(k);
      A_curr_R1 += om_curr[j] * om_curr[k];
      A_new_R1 += om_new[j] * om_new[k];
    }
  }

  double log_prior =
      log_beta_pdf(eta1_new / eta_sd, e, f) - log_beta_pdf(eta1 / eta_sd, e, f);
  double log_target = (eta1_new - eta1) * B_R1 + log_prior;
  double log_aux =
      mu_tilde * (sum_new - sum_curr) + eta1_tilde * (A_new_R1 - A_curr_R1);
  double log_norm =
      mu * (sum_curr - sum_new) + eta1 * A_curr_R1 - eta1_new * A_new_R1;

  double log_MH = log_target + log_aux + log_norm + log_prop_ratio;

  if (std::log(R::runif(0.0, 1.0)) < log_MH)
    eta1 = eta1_new;
}


} // anonymous namespace
// =============================================================================
// SECTION 5: MAIN FUNCTION
//
// Memory: O(n*p + p*d_max^2 + |E|)
// =============================================================================

// [[Rcpp::export]]
Rcpp::List BayesLogit_SingleNet_SparseGGM(
    const arma::mat &X, const arma::vec &y, const Rcpp::IntegerVector &S_i,
    const Rcpp::IntegerVector &S_p_csc, const Rcpp::NumericVector &S_x,
    const Rcpp::NumericVector &S_diag, int p_ggm, int niter, int burnin,
    double mu, double nu0, double sigmasq0, double alpha0, double beta0,
    double h, double e, double f,
    double v0_ggm, double v1_ggm, double pii_ggm,
    double eta_sd, double mu_tilde, double eta1_tilde, unsigned int T_max,
    int proposal_type,
    int n_mh_gamma = 5,
    int thin = 1,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0) {
  Rcpp::RNGScope scope;

  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  if ((int)p != p_ggm)
    Rcpp::stop("p_ggm (%d) must match ncol(X) (%d)", p_ggm, (int)p);
  if (thin < 1)
    thin = 1;

  // =========================================================================
  // 1. BUILD SPARSE GRAPH FROM CSC INPUT
  // =========================================================================
  SparseGraph G(p);
  for (int col = 0; col < (int)p; ++col) {
    for (int k = S_p_csc[col]; k < S_p_csc[col + 1]; ++k) {
      int row = S_i[k];
      G.adj[col].push_back(row);
      G.val[col].push_back(S_x[k]);
    }
  }
  G.sort_indices();

  int d_max = 0;
  for (int i = 0; i < (int)p; ++i) {
    int d = (int)G.adj[i].size();
    if (d > d_max)
      d_max = d;
  }

  // Neighbor→local_index hash map
  std::vector<std::unordered_map<int, int>> nbr_idx_map(p);
  for (int i = 0; i < (int)p; ++i) {
    const std::vector<int> &nbrs = G.adj[i];
    for (int k = 0; k < (int)nbrs.size(); ++k)
      nbr_idx_map[i][nbrs[k]] = k;
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
  double eta1 = std::min(0.01, eta_sd * 0.5);

  arma::vec z = X * beta_vec;
  double loglik = calc_loglik(y, z, alpha);

  // Sparse edge state
  std::vector<std::unordered_set<int>> Z_active(p);
  int n_edges = 0;
  for (int i = 0; i < (int)p; ++i)
    for (int nbr : G.adj[i])
      Z_active[i].insert(nbr);
  for (int i = 0; i < (int)p; ++i)
    for (int nbr : Z_active[i])
      if (nbr > i)
        n_edges++;

  // GGM pre-allocated buffers
  arma::mat A_sub, Ud_work;
  arma::vec s_ggm, noise_ggm, v_ggm;
  if (d_max > 0) {
    A_sub.set_size(d_max, d_max);
    Ud_work.set_size(d_max, d_max); // FIX: pre-allocated Ud copy buffer
    s_ggm.set_size(d_max);
    noise_ggm.set_size(d_max);
    v_ggm.set_size(d_max);
  }

  // Pre-computed SSVS constants
  double log_pii_ggm = std::log(pii_ggm);
  double log_1_pii_ggm = std::log(1.0 - pii_ggm);
  double log_v0_half = -0.5 * std::log(v0_ggm);
  double log_v1_half = -0.5 * std::log(v1_ggm);
  double inv_v0 = 1.0 / v0_ggm;
  double inv_v1 = 1.0 / v1_ggm;

  // Logistic pre-allocated buffers
  arma::vec z_prop(n);
  std::vector<arma::uword> active_idx;
  active_idx.reserve(p);

  // FIX: PW pre-allocated workspace (eliminates 2×p heap allocs per Möller)
  std::vector<int> pw_x_up(p), pw_x_down(p);
  std::vector<int> pw_om_curr(p), pw_om_new(p);

  // Output storage
  int n_save = niter / thin;
  arma::mat beta_out(n_save, p);
  arma::umat gamma_out(n_save, p);
  arma::vec eta_out(n_save);
  arma::vec alpha_out(n_save);
  arma::vec sigmasq_out(n_save);
  Rcpp::List Z_list(n_save);

  std::vector<int> edge_rows, edge_cols;
  edge_rows.reserve(std::max(1000, n_edges));
  edge_cols.reserve(std::max(1000, n_edges));

  // =========================================================================
  // 3. MCMC LOOP
  // =========================================================================
  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {

    if (iter > 0 && iter % 5000 == 0) {
      Rcpp::checkUserInterrupt();
      int model_size = (int)active_idx.size(); // FIX: cached, no arma::accu
      Rcpp::Rcout << "Iter: " << iter << " | Model: " << model_size
                  << " | Edges: " << n_edges << " | eta: " << eta1 << "\n";
    }

    // OPT: pre-compute sqrt(sigmasq) once per iteration
    double sd_sig = std::sqrt(sigmasq);

    // -----------------------------------------------------------------------
    // STEP A: Sparse GGM Column Sweep (SSVS)
    // -----------------------------------------------------------------------
    if (d_max > 0) {
      for (int i = 0; i < (int)p; ++i) {
        const std::vector<int> &neighbors = G.adj[i];
        int d = (int)neighbors.size();
        if (d < 1)
          continue;

        for (int r = 0; r < d; ++r) {
          int u = neighbors[r];
          A_sub(r, r) = S_diag[u];
          for (int c = r + 1; c < d; ++c) {
            int v = neighbors[c];
            double w = 0.0;
            auto it = nbr_idx_map[u].find(v);
            if (it != nbr_idx_map[u].end())
              w = G.val[u][it->second];
            A_sub(r, c) = w;
            A_sub(c, r) = w;
          }
        }

        for (int k = 0; k < d; ++k) {
          int nbr = neighbors[k];
          bool active = (Z_active[i].count(nbr) > 0);
          double v_val = active ? v1_ggm : v0_ggm;
          v_ggm(k) = v_val;
          A_sub(k, k) += (1.0 / v_val);
        }

        arma::mat A_chol_tmp = A_sub.submat(0, 0, d - 1, d - 1);
        arma::mat U_chol_tmp;
        bool ok = arma::chol(U_chol_tmp, arma::symmatu(A_chol_tmp));
        if (!ok) {
          for (int k = 0; k < d; ++k)
            A_sub(k, k) += 1e-5;
          A_chol_tmp = A_sub.submat(0, 0, d - 1, d - 1);
          ok = arma::chol(U_chol_tmp, arma::symmatu(A_chol_tmp));
          if (!ok)
            continue;
        }

        // FIX: Copy into pre-allocated Ud_work (no d×d heap alloc per node)
        for (int r2 = 0; r2 < d; ++r2)
          for (int c2 = r2; c2 < d; ++c2)
            Ud_work(r2, c2) = U_chol_tmp(r2, c2);

        for (int k = 0; k < d; ++k)
          s_ggm(k) = G.val[i][k];

        arma::vec neg_s = -s_ggm.head(d);
        arma::mat Ud_view(Ud_work.memptr(), d, d, false, true);
        const double min_diag = Ud_view.diag().min();
        if (!std::isfinite(min_diag) || min_diag <= 1e-12)
          continue;

        arma::vec y_tmp;
        bool solve_ok = arma::solve(
            y_tmp, arma::trimatl(Ud_view.t()), neg_s,
            arma::solve_opts::fast + arma::solve_opts::no_approx);
        if (!solve_ok)
          continue;
        arma::vec mu_sub;
        solve_ok = arma::solve(mu_sub, arma::trimatu(Ud_view), y_tmp,
                               arma::solve_opts::fast +
                                   arma::solve_opts::no_approx);
        if (!solve_ok)
          continue;

        for (int k = 0; k < d; ++k)
          noise_ggm(k) = R::rnorm(0.0, 1.0);
        arma::vec delta;
        solve_ok = arma::solve(delta, arma::trimatu(Ud_view), noise_ggm.head(d),
                               arma::solve_opts::fast +
                                   arma::solve_opts::no_approx);
        if (!solve_ok)
          continue;
        arma::vec b_ggm = mu_sub + delta;

        for (int k = 0; k < d; ++k) {
          double b2 = b_ggm(k) * b_ggm(k);
          double w1 = log_v0_half - 0.5 * b2 * inv_v0 + log_1_pii_ggm;
          double w2 = log_v1_half - 0.5 * b2 * inv_v1 + log_pii_ggm;
          double w_max = std::max(w1, w2);
          double prob = std::exp(w2 - w_max) /
                        (std::exp(w1 - w_max) + std::exp(w2 - w_max));

          int nbr = neighbors[k];
          bool was_active = (Z_active[i].count(nbr) > 0);
          bool now_active = (R::runif(0, 1) < prob);

          if (now_active && !was_active) {
            Z_active[i].insert(nbr);
            Z_active[nbr].insert(i);
            n_edges++;
          } else if (!now_active && was_active) {
            Z_active[i].erase(nbr);
            Z_active[nbr].erase(i);
            n_edges--;
          }
        }
      }
    }

    // -----------------------------------------------------------------------
    // STEP B: Gamma (Variable Selection) — Ising prior via sparse Z_active
    // OPT: n_mh_gamma now a parameter (was hardcoded 5)
    // OPT: sd_sig pre-computed once
    // -----------------------------------------------------------------------
    for (int mh = 0; mh < n_mh_gamma; ++mh) {
      arma::uword j =
          static_cast<arma::uword>(std::floor(R::runif(0.0, (double)p)));
      if (j >= p)
        j = p - 1;

      int g_curr = gamma(j);
      int g_prop = 1 - g_curr;
      double b_curr = beta_vec(j);
      double b_prop = (g_prop == 1) ? R::rnorm(beta0, sd_sig) : 0.0;

      z_prop = z + (b_prop - b_curr) * X.col(j);
      double ll_prop = calc_loglik(y, z_prop, alpha);

      double diff = static_cast<double>(g_prop - g_curr);
      double neigh = 0.0;
      for (int nbr : Z_active[j])
        neigh += gamma(nbr);

      double ising_diff = diff * (mu + eta1 * neigh);
      double log_ratio = (ll_prop - loglik) + ising_diff;

      if (std::log(R::runif(0, 1)) < log_ratio) {
        gamma(j) = g_prop;
        beta_vec(j) = b_prop;
        z = z_prop;
        loglik = ll_prop;
      }
    }

    // Build active indices
    active_idx.clear();
    for (arma::uword j = 0; j < p; ++j)
      if (gamma(j) == 1)
        active_idx.push_back(j);

    // -----------------------------------------------------------------------
    // STEP C: Beta (Metropolis updates for active variables)
    // OPT: sd_sig pre-computed, used directly
    // -----------------------------------------------------------------------
    {
      double sd_beta = sd_sig * 0.1;
      for (arma::uword k = 0; k < active_idx.size(); ++k) {
        arma::uword j = active_idx[k];
        double b_prop = R::rnorm(beta_vec(j), sd_beta);
        z_prop = z + (b_prop - beta_vec(j)) * X.col(j);
        double ll_prop = calc_loglik(y, z_prop, alpha);
        // OPT: x*x instead of std::pow(x, 2)
        double d_curr = beta_vec(j) - beta0;
        double d_prop = b_prop - beta0;
        double pr_curr = -0.5 * d_curr * d_curr / sigmasq;
        double pr_prop = -0.5 * d_prop * d_prop / sigmasq;
        if (std::log(R::runif(0, 1)) <
            (ll_prop - loglik) + (pr_prop - pr_curr)) {
          beta_vec(j) = b_prop;
          z = z_prop;
          loglik = ll_prop;
        }
      }
    }

    // -----------------------------------------------------------------------
    // STEP D: Alpha (Metropolis update)
    // -----------------------------------------------------------------------
    {
      double a_prop = R::rnorm(alpha, std::sqrt(h * sigmasq));
      double ll_a_prop = calc_loglik(y, z, a_prop);
      // OPT: x*x instead of std::pow(x, 2)
      double da_curr = alpha - alpha0;
      double da_prop = a_prop - alpha0;
      double pr_a_curr = -0.5 * da_curr * da_curr / (h * sigmasq);
      double pr_a_prop = -0.5 * da_prop * da_prop / (h * sigmasq);
      if (std::log(R::runif(0, 1)) <
          (ll_a_prop - loglik) + (pr_a_prop - pr_a_curr)) {
        alpha = a_prop;
        loglik = ll_a_prop;
      }
    }

    // -----------------------------------------------------------------------
    // STEP E: SigmaSq (log-normal proposal)
    // OPT: x*x, uses cached active_idx
    // -----------------------------------------------------------------------
    {
      double sig_prop = std::exp(std::log(sigmasq) + R::rnorm(0, 0.2));
      sig_prop = std::max(sig_prop, 1e-10);

      double shape = nu0 / 2.0;
      double scale = sigmasq0 * nu0 / 2.0;
      double lp_sig_curr = -(shape + 1.0) * std::log(sigmasq) - scale / sigmasq;
      double lp_sig_prop =
          -(shape + 1.0) * std::log(sig_prop) - scale / sig_prop;

      double n_active_beta = (double)active_idx.size();
      double ss_beta = 0.0;
      for (arma::uword k = 0; k < active_idx.size(); ++k) {
        double d2 = beta_vec(active_idx[k]) - beta0;
        ss_beta += d2 * d2;
      }

      double lp_beta_curr =
          -0.5 * n_active_beta * std::log(sigmasq) - 0.5 * ss_beta / sigmasq;
      double lp_beta_prop =
          -0.5 * n_active_beta * std::log(sig_prop) - 0.5 * ss_beta / sig_prop;

      // OPT: x*x instead of std::pow(x, 2)
      double da = alpha - alpha0;
      double lp_alpha_curr =
          -0.5 * std::log(h * sigmasq) - 0.5 * da * da / (h * sigmasq);
      double lp_alpha_prop =
          -0.5 * std::log(h * sig_prop) - 0.5 * da * da / (h * sig_prop);

      double log_mh_sig =
          (lp_sig_prop - lp_sig_curr) + (lp_beta_prop - lp_beta_curr) +
          (lp_alpha_prop - lp_alpha_curr) + std::log(sig_prop / sigmasq);
      if (std::log(R::runif(0, 1)) < log_mh_sig)
        sigmasq = sig_prop;
    }

    // -----------------------------------------------------------------------
    // STEP F: Update eta via sparse Moller + Propp-Wilson
    // FIX: passes pre-allocated PW workspace
    // -----------------------------------------------------------------------
    moller_update_eta_sparse(Z_active, (int)p, mu, eta1, eta_sd, mu_tilde,
                             eta1_tilde, gamma, e, f, T_max, proposal_type,
                             pw_x_up, pw_x_down, pw_om_curr, pw_om_new);

    // -----------------------------------------------------------------------
    // STORE SAMPLES
    // -----------------------------------------------------------------------
    if (iter >= burnin && (iter - burnin) % thin == 0) {
      int s = (iter - burnin) / thin;
      if (s < n_save) {
        beta_out.row(s) = beta_vec.t();
        for (arma::uword j = 0; j < p; ++j)
          gamma_out(s, j) = gamma(j);
        eta_out(s) = eta1;
        alpha_out(s) = alpha;
        sigmasq_out(s) = sigmasq;

        edge_rows.clear();
        edge_cols.clear();
        for (int c = 0; c < (int)p; ++c) {
          for (int r : Z_active[c]) {
            if (r < c) {
              edge_rows.push_back(r);
              edge_cols.push_back(c);
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
      Rcpp::Named("eta") = eta_out, Rcpp::Named("alpha") = alpha_out,
      Rcpp::Named("sigmasq") = sigmasq_out, Rcpp::Named("Z_list") = Z_list);
}
