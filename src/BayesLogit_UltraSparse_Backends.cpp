// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace {

static inline bool sorted_contains(const std::vector<int> &v, int val) {
  return std::binary_search(v.begin(), v.end(), val);
}

static inline bool sorted_insert(std::vector<int> &v, int val) {
  auto it = std::lower_bound(v.begin(), v.end(), val);
  if (it != v.end() && *it == val)
    return false;
  v.insert(it, val);
  return true;
}

static inline bool sorted_erase(std::vector<int> &v, int val) {
  auto it = std::lower_bound(v.begin(), v.end(), val);
  if (it == v.end() || *it != val)
    return false;
  v.erase(it);
  return true;
}

static inline int sorted_index(const std::vector<int> &v, int val) {
  auto it = std::lower_bound(v.begin(), v.end(), val);
  if (it != v.end() && *it == val)
    return static_cast<int>(it - v.begin());
  return -1;
}

static inline void activate_gamma(int j, std::vector<int> &active_idx,
                                  std::vector<int> &active_pos) {
  if (active_pos[j] >= 0)
    return;
  active_pos[j] = static_cast<int>(active_idx.size());
  active_idx.push_back(j);
}

static inline void deactivate_gamma(int j, std::vector<int> &active_idx,
                                    std::vector<int> &active_pos) {
  int pos = active_pos[j];
  if (pos < 0)
    return;
  int last = active_idx.back();
  active_idx[pos] = last;
  active_pos[last] = pos;
  active_idx.pop_back();
  active_pos[j] = -1;
}

struct SparseS {
  std::vector<std::vector<int>> adj;
  std::vector<std::vector<double>> val;
  std::vector<double> diag;
  int p;

  explicit SparseS(int p_) : p(p_) {
    adj.resize(p);
    val.resize(p);
    diag.resize(p, 0.0);
  }

  double lookup(int u, int v) const {
    int idx = sorted_index(adj[u], v);
    return (idx >= 0) ? val[u][idx] : 0.0;
  }
};

struct SparseAdj {
  std::vector<std::vector<int>> adj;
  int p;

  explicit SparseAdj(int p_) : p(p_) { adj.resize(p); }
};

static SparseS build_sparse_s(const Rcpp::IntegerVector &S_i,
                              const Rcpp::IntegerVector &S_p_csc,
                              const Rcpp::NumericVector &S_x,
                              const Rcpp::NumericVector &S_diag, int p) {
  if (S_p_csc.size() != p + 1)
    Rcpp::stop("S_p_csc must have length p + 1.");
  if (S_diag.size() != p)
    Rcpp::stop("S_diag must have length p.");

  SparseS S(p);
  for (int col = 0; col < p; ++col) {
    S.diag[col] = S_diag[col];
    int start = S_p_csc[col];
    int end = S_p_csc[col + 1];
    if (start < 0 || end < start || end > S_i.size() || end > S_x.size())
      Rcpp::stop("Invalid CSC pointers for S.");

    const int d = end - start;
    S.adj[col].resize(d);
    S.val[col].resize(d);
    for (int k = 0; k < d; ++k) {
      int row = S_i[start + k];
      if (row < 0 || row >= p)
        Rcpp::stop("S_i contains out-of-range row index.");
      S.adj[col][k] = row;
      S.val[col][k] = S_x[start + k];
    }

    if (d > 1) {
      std::vector<std::pair<int, double>> tmp(d);
      for (int k = 0; k < d; ++k)
        tmp[k] = std::make_pair(S.adj[col][k], S.val[col][k]);
      std::sort(tmp.begin(), tmp.end());
      for (int k = 0; k < d; ++k) {
        S.adj[col][k] = tmp[k].first;
        S.val[col][k] = tmp[k].second;
      }
    }
  }
  return S;
}

static SparseAdj build_sparse_adj(const Rcpp::IntegerVector &R_i,
                                  const Rcpp::IntegerVector &R_p_csc, int p,
                                  bool remove_diag) {
  if (R_p_csc.size() != p + 1)
    Rcpp::stop("R_p_csc must have length p + 1.");

  SparseAdj R(p);
  for (int col = 0; col < p; ++col) {
    int start = R_p_csc[col];
    int end = R_p_csc[col + 1];
    if (start < 0 || end < start || end > R_i.size())
      Rcpp::stop("Invalid CSC pointers for fixed adjacency.");

    std::vector<int> col_adj;
    col_adj.reserve(end - start);
    for (int k = start; k < end; ++k) {
      int row = R_i[k];
      if (row < 0 || row >= p)
        Rcpp::stop("R_i contains out-of-range row index.");
      if (remove_diag && row == col)
        continue;
      col_adj.push_back(row);
    }

    std::sort(col_adj.begin(), col_adj.end());
    col_adj.erase(std::unique(col_adj.begin(), col_adj.end()), col_adj.end());
    R.adj[col].swap(col_adj);
  }
  return R;
}

static const int PG_K = 20;

static double sample_pg(double z, std::mt19937 &rng) {
  z = std::abs(z) * 0.5;
  const double c2 = z * z;
  const double PI2 = M_PI * M_PI;
  const double c2_over_pi2 = c2 / PI2;
  const double INV_2PI2 = 1.0 / (2.0 * PI2);

  std::exponential_distribution<double> exp_dist(1.0);
  double s = 0.0;
  for (int k = 0; k < PG_K; ++k) {
    double g = exp_dist(rng);
    double kh = k + 0.5;
    s += g / (kh * kh + c2_over_pi2);
  }
  return s * INV_2PI2;
}

static inline double log_beta_pdf(double x, double a, double b) {
  x = std::max(1e-12, std::min(1.0 - 1e-12, x));
  return std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
         (a - 1.0) * std::log(x) + (b - 1.0) * std::log(1.0 - x);
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

static inline double loglik_obs(double yi, double psi) {
  if (psi > 0.0)
    return yi * psi - psi - std::log1p(std::exp(-psi));
  return yi * psi - std::log1p(std::exp(psi));
}

static inline double calc_loglik_full(const arma::vec &y, const arma::vec &Xb,
                                      double alpha) {
  const int n = static_cast<int>(y.n_elem);
  double ll = 0.0;
  for (int i = 0; i < n; ++i)
    ll += loglik_obs(y(i), alpha + Xb(i));
  return ll;
}

static inline double column_ll_diff(const arma::vec &y, const arma::vec &Xb,
                                    double alpha,
                                    const arma::uword *col_ptr,
                                    const arma::uword *row_idx,
                                    const double *xvals, int j, double db) {
  if (std::abs(db) < 1e-16)
    return 0.0;

  double ll_diff = 0.0;
  arma::uword start = col_ptr[j];
  arma::uword end = col_ptr[j + 1];
  for (arma::uword k = start; k < end; ++k) {
    int i = static_cast<int>(row_idx[k]);
    double xij = xvals[k];
    double psi_old = alpha + Xb(i);
    double psi_new = psi_old + db * xij;
    ll_diff += loglik_obs(y(i), psi_new) - loglik_obs(y(i), psi_old);
  }
  return ll_diff;
}

static inline void apply_column_update(arma::vec &Xb, const arma::uword *col_ptr,
                                       const arma::uword *row_idx,
                                       const double *xvals, int j, double db) {
  if (std::abs(db) < 1e-16)
    return;
  arma::uword start = col_ptr[j];
  arma::uword end = col_ptr[j + 1];
  for (arma::uword k = start; k < end; ++k)
    Xb(static_cast<int>(row_idx[k])) += db * xvals[k];
}

static void proppwilson_single_sparse(
    const std::vector<std::vector<int>> &Z_adj, int p, double mu, double eta1,
    unsigned int T_max, std::vector<int> &x_up, std::vector<int> &x_down,
    std::vector<int> &result) {
  unsigned int T = 2;
  int seed_base =
      static_cast<int>(std::floor(R::runif(0.0, 1.0) * 1000.0)) + 1;

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
      std::mt19937 gen(-t * seed_base);
      std::uniform_real_distribution<double> u01(0.0, 1.0);

      for (int i = 0; i < p; ++i) {
        double ku = 0.0, kd = 0.0;
        for (int j : Z_adj[i]) {
          if (x_up[j])
            ku += eta1;
          if (x_down[j])
            kd += eta1;
        }
        double pu = 1.0 / (1.0 + std::exp(-(mu + ku)));
        double pd = 1.0 / (1.0 + std::exp(-(mu + kd)));
        double u = u01(gen);
        x_up[i] = (pu > u) ? 1 : 0;
        x_down[i] = (pd > u) ? 1 : 0;
      }
    }

    T *= 2;
    if (T >= T_max) {
      std::mt19937 gf(seed_base + 99999);
      std::uniform_real_distribution<double> u01f(0.0, 1.0);
      for (int sw = 0; sw < 100; ++sw) {
        for (int k = 0; k < p; ++k) {
          if (x_up[k] == x_down[k])
            continue;
          double ker = 0.0;
          for (int j : Z_adj[k])
            if (x_up[j])
              ker += eta1;
          x_up[k] = (u01f(gf) < 1.0 / (1.0 + std::exp(-(mu + ker)))) ? 1 : 0;
        }
      }
      for (int k = 0; k < p; ++k)
        x_down[k] = x_up[k];
    }
  }

  for (int k = 0; k < p; ++k)
    result[k] = x_up[k];
}

static void proppwilson_dual_sparse(
    const std::vector<std::vector<int>> &Z_adj,
    const std::vector<std::vector<int>> &R_fix_adj, int p, double mu,
    double eta1, double eta2, unsigned int T_max, std::vector<int> &x_up,
    std::vector<int> &x_down, std::vector<int> &result) {
  unsigned int T = 2;
  int seed_base =
      static_cast<int>(std::floor(R::runif(0.0, 1.0) * 1000.0)) + 1;

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
      std::mt19937 gen(-t * seed_base);
      std::uniform_real_distribution<double> u01(0.0, 1.0);

      for (int i = 0; i < p; ++i) {
        double ku = 0.0, kd = 0.0;
        for (int j : Z_adj[i]) {
          if (x_up[j])
            ku += eta1;
          if (x_down[j])
            kd += eta1;
        }
        for (int j : R_fix_adj[i]) {
          if (x_up[j])
            ku += eta2;
          if (x_down[j])
            kd += eta2;
        }

        double pu = 1.0 / (1.0 + std::exp(-(mu + ku)));
        double pd = 1.0 / (1.0 + std::exp(-(mu + kd)));
        double u = u01(gen);
        x_up[i] = (pu > u) ? 1 : 0;
        x_down[i] = (pd > u) ? 1 : 0;
      }
    }

    T *= 2;
    if (T >= T_max) {
      std::mt19937 gf(seed_base + 99999);
      std::uniform_real_distribution<double> u01f(0.0, 1.0);
      for (int sw = 0; sw < 100; ++sw) {
        for (int k = 0; k < p; ++k) {
          if (x_up[k] == x_down[k])
            continue;
          double ker = 0.0;
          for (int j : Z_adj[k])
            if (x_up[j])
              ker += eta1;
          for (int j : R_fix_adj[k])
            if (x_up[j])
              ker += eta2;
          x_up[k] = (u01f(gf) < 1.0 / (1.0 + std::exp(-(mu + ker)))) ? 1 : 0;
        }
      }
      for (int k = 0; k < p; ++k)
        x_down[k] = x_up[k];
    }
  }

  for (int k = 0; k < p; ++k)
    result[k] = x_up[k];
}

static void moller_update_single_sparse(
    const std::vector<std::vector<int>> &Z_adj, int p, double mu, double &eta1,
    double eta_sd, double mu_tilde, double eta1_tilde,
    const std::vector<uint8_t> &gamma, double e, double f, unsigned int T_max,
    int proposal_type, std::vector<int> &pw_up, std::vector<int> &pw_dn,
    std::vector<int> &om1, std::vector<int> &om1n) {
  double eta1_new;
  double lpr = 0.0;

  if (proposal_type == 0) {
    double a = std::max(0.0, eta1 - 0.01);
    double b = std::min(eta_sd, eta1 + 0.01);
    eta1_new = R::runif(a, b);
    double c = std::max(0.0, eta1_new - 0.01);
    double d = std::min(eta_sd, eta1_new + 0.01);
    lpr = std::log(b - a) - std::log(d - c);
  } else {
    int att = 0;
    do {
      eta1_new = R::rnorm(eta1, eta_sd);
      if (++att > 10000) {
        eta1_new = eta1;
        break;
      }
    } while (eta1_new <= 0.0 || eta1_new >= eta_sd);

    double lqf = std::log(normal_pdf(eta1_new, eta1, eta_sd)) -
                 std::log(normal_cdf(eta_sd, eta1, eta_sd) -
                          normal_cdf(0.0, eta1, eta_sd));
    double lqr = std::log(normal_pdf(eta1, eta1_new, eta_sd)) -
                 std::log(normal_cdf(eta_sd, eta1_new, eta_sd) -
                          normal_cdf(0.0, eta1_new, eta_sd));
    lpr = lqr - lqf;
  }

  eta1_new = std::max(1e-8, std::min(eta_sd - 1e-8, eta1_new));

  proppwilson_single_sparse(Z_adj, p, mu, eta1, T_max, pw_up, pw_dn, om1);
  proppwilson_single_sparse(Z_adj, p, mu, eta1_new, T_max, pw_up, pw_dn,
                            om1n);

  int sum1 = 0, sum1n = 0, B = 0, A1 = 0, A1n = 0;
  for (int j = 0; j < p; ++j) {
    sum1 += om1[j];
    sum1n += om1n[j];
    for (int k : Z_adj[j]) {
      if (k <= j)
        continue;
      B += static_cast<int>(gamma[j]) * static_cast<int>(gamma[k]);
      A1 += om1[j] * om1[k];
      A1n += om1n[j] * om1n[k];
    }
  }

  double lp = log_beta_pdf(eta1_new / eta_sd, e, f) -
              log_beta_pdf(eta1 / eta_sd, e, f);
  double lmh = (eta1_new - eta1) * B + lp + mu_tilde * (sum1n - sum1) +
               eta1_tilde * (A1n - A1) + mu * (sum1 - sum1n) + eta1 * A1 -
               eta1_new * A1n + lpr;

  if (std::log(R::runif(0.0, 1.0)) < lmh)
    eta1 = eta1_new;
}

static void moller_update_dual_sparse(
    const std::vector<std::vector<int>> &Z_adj,
    const std::vector<std::vector<int>> &R_fix_adj, int p, double mu,
    double &eta1, double &eta2, double eta1_sd, double eta2_sd,
    double mu_tilde, double eta1_tilde, double eta2_tilde,
    const std::vector<uint8_t> &gamma, double e, double f, unsigned int T_max,
    int proposal_type, std::vector<int> &pw_up, std::vector<int> &pw_dn,
    std::vector<int> &om1, std::vector<int> &om2, std::vector<int> &om1n,
    std::vector<int> &om2n) {
  double eta1_new, eta2_new;
  double lpr1 = 0.0, lpr2 = 0.0;

  if (proposal_type == 0) {
    double a1 = std::max(0.0, eta1 - 0.01);
    double b1 = std::min(eta1_sd, eta1 + 0.01);
    eta1_new = R::runif(a1, b1);
    double c1 = std::max(0.0, eta1_new - 0.01);
    double d1 = std::min(eta1_sd, eta1_new + 0.01);
    lpr1 = std::log(b1 - a1) - std::log(d1 - c1);

    double a2 = std::max(0.0, eta2 - 0.01);
    double b2 = std::min(eta2_sd, eta2 + 0.01);
    eta2_new = R::runif(a2, b2);
    double c2 = std::max(0.0, eta2_new - 0.01);
    double d2 = std::min(eta2_sd, eta2_new + 0.01);
    lpr2 = std::log(b2 - a2) - std::log(d2 - c2);
  } else {
    int att = 0;
    do {
      eta1_new = R::rnorm(eta1, eta1_sd);
      if (++att > 10000) {
        eta1_new = eta1;
        break;
      }
    } while (eta1_new <= 0.0 || eta1_new >= eta1_sd);

    att = 0;
    do {
      eta2_new = R::rnorm(eta2, eta2_sd);
      if (++att > 10000) {
        eta2_new = eta2;
        break;
      }
    } while (eta2_new <= 0.0 || eta2_new >= eta2_sd);

    double lqf1 = std::log(normal_pdf(eta1_new, eta1, eta1_sd)) -
                  std::log(normal_cdf(eta1_sd, eta1, eta1_sd) -
                           normal_cdf(0.0, eta1, eta1_sd));
    double lqr1 = std::log(normal_pdf(eta1, eta1_new, eta1_sd)) -
                  std::log(normal_cdf(eta1_sd, eta1_new, eta1_sd) -
                           normal_cdf(0.0, eta1_new, eta1_sd));
    lpr1 = lqr1 - lqf1;

    double lqf2 = std::log(normal_pdf(eta2_new, eta2, eta2_sd)) -
                  std::log(normal_cdf(eta2_sd, eta2, eta2_sd) -
                           normal_cdf(0.0, eta2, eta2_sd));
    double lqr2 = std::log(normal_pdf(eta2, eta2_new, eta2_sd)) -
                  std::log(normal_cdf(eta2_sd, eta2_new, eta2_sd) -
                           normal_cdf(0.0, eta2_new, eta2_sd));
    lpr2 = lqr2 - lqf2;
  }

  eta1_new = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1_new));
  eta2_new = std::max(1e-8, std::min(eta2_sd - 1e-8, eta2_new));

  proppwilson_dual_sparse(Z_adj, R_fix_adj, p, mu, eta1, eta2, T_max, pw_up,
                          pw_dn, om1);
  proppwilson_dual_sparse(Z_adj, R_fix_adj, p, mu, eta1, eta2, T_max, pw_up,
                          pw_dn, om2);
  proppwilson_dual_sparse(Z_adj, R_fix_adj, p, mu, eta1_new, eta2, T_max,
                          pw_up, pw_dn, om1n);
  proppwilson_dual_sparse(Z_adj, R_fix_adj, p, mu, eta1, eta2_new, T_max,
                          pw_up, pw_dn, om2n);

  int sum1 = 0, sum2 = 0, sum1n = 0, sum2n = 0;
  for (int j = 0; j < p; ++j) {
    sum1 += om1[j];
    sum2 += om2[j];
    sum1n += om1n[j];
    sum2n += om2n[j];
  }

  int B1 = 0, A11 = 0, A11n = 0, A21 = 0, A21n = 0;
  for (int j = 0; j < p; ++j) {
    for (int k : Z_adj[j]) {
      if (k <= j)
        continue;
      B1 += static_cast<int>(gamma[j]) * static_cast<int>(gamma[k]);
      A11 += om1[j] * om1[k];
      A11n += om1n[j] * om1n[k];
      A21 += om2[j] * om2[k];
      A21n += om2n[j] * om2n[k];
    }
  }

  int B2 = 0, A12 = 0, A12n = 0, A22 = 0, A22n = 0;
  for (int j = 0; j < p; ++j) {
    for (int k : R_fix_adj[j]) {
      if (k <= j)
        continue;
      B2 += static_cast<int>(gamma[j]) * static_cast<int>(gamma[k]);
      A12 += om1[j] * om1[k];
      A12n += om1n[j] * om1n[k];
      A22 += om2[j] * om2[k];
      A22n += om2n[j] * om2n[k];
    }
  }

  double lp1 = log_beta_pdf(eta1_new / eta1_sd, e, f) -
               log_beta_pdf(eta1 / eta1_sd, e, f);
  double log_target1 = (eta1_new - eta1) * B1 + lp1;
  double log_aux1 = mu_tilde * (sum1n - sum1) + eta1_tilde * (A11n - A11) +
                    eta2_tilde * (A12n - A12);
  double log_norm1 = mu * (sum1 - sum1n) + eta1 * A11 - eta1_new * A11n +
                     eta2 * (A12 - A12n);
  double log_mh1 = log_target1 + log_aux1 + log_norm1 + lpr1;

  double lp2 = log_beta_pdf(eta2_new / eta2_sd, e, f) -
               log_beta_pdf(eta2 / eta2_sd, e, f);
  double log_target2 = (eta2_new - eta2) * B2 + lp2;
  double log_aux2 = mu_tilde * (sum2n - sum2) + eta1_tilde * (A21n - A21) +
                    eta2_tilde * (A22n - A22);
  double log_norm2 = mu * (sum2 - sum2n) + eta2 * A22 - eta2_new * A22n +
                     eta1 * (A21 - A21n);
  double log_mh2 = log_target2 + log_aux2 + log_norm2 + lpr2;

  if (std::log(R::runif(0.0, 1.0)) < log_mh1)
    eta1 = eta1_new;
  if (std::log(R::runif(0.0, 1.0)) < log_mh2)
    eta2 = eta2_new;
}

static bool validate_and_convert_y(const arma::vec &y, arma::vec &y01) {
  const int n = static_cast<int>(y.n_elem);
  y01 = y;
  bool ok01 = true, ok11 = true;
  for (int i = 0; i < n; ++i) {
    double yi = y(i);
    if (std::fabs(yi) > 1e-12 && std::fabs(yi - 1.0) > 1e-12)
      ok01 = false;
    if (std::fabs(yi + 1.0) > 1e-12 && std::fabs(yi - 1.0) > 1e-12)
      ok11 = false;
  }
  if (!ok01 && ok11)
    y01 = 0.5 * (y01 + 1.0);
  return ok01 || ok11;
}

static void ggm_column_sweep_sparse(
    const SparseS &S, std::vector<std::vector<int>> &Z_active, int p,
    double log_pii, double log_1pii, double lv0h, double lv1h, double iv0,
    double iv1, arma::mat &A_sub, arma::vec &s_ggm, arma::vec &noise_ggm,
    int &n_edges) {
  const int d_max = static_cast<int>(A_sub.n_rows);
  if (d_max <= 0)
    return;

  for (int i = 0; i < p; ++i) {
    const std::vector<int> &nbrs = S.adj[i];
    const int d = static_cast<int>(nbrs.size());
    if (d < 1)
      continue;

    for (int r = 0; r < d; ++r) {
      int u = nbrs[r];
      A_sub(r, r) = S.diag[u];
      for (int c = r + 1; c < d; ++c) {
        int v = nbrs[c];
        double w = S.lookup(u, v);
        A_sub(r, c) = w;
        A_sub(c, r) = w;
      }
    }

    for (int k = 0; k < d; ++k) {
      int nbr = nbrs[k];
      bool active = sorted_contains(Z_active[i], nbr);
      A_sub(k, k) += active ? iv1 : iv0;
    }

    arma::mat A_chol = A_sub.submat(0, 0, d - 1, d - 1);
    arma::mat U;
    bool ok = arma::chol(U, arma::symmatu(A_chol));
    if (!ok) {
      double jit = 1e-8;
      for (int att = 0; att < 5 && !ok; ++att) {
        A_chol.diag() += jit;
        ok = arma::chol(U, arma::symmatu(A_chol));
        jit *= 10.0;
      }
      if (!ok)
        continue;
    }

    for (int k = 0; k < d; ++k)
      s_ggm(k) = S.val[i][k];

    arma::vec neg_s = -s_ggm.head(d);
    arma::vec yt, mu_sub, delta;
    bool solve_ok = arma::solve(yt, arma::trimatl(U.t()), neg_s,
                                arma::solve_opts::fast +
                                    arma::solve_opts::no_approx);
    if (!solve_ok)
      continue;
    solve_ok = arma::solve(mu_sub, arma::trimatu(U), yt,
                           arma::solve_opts::fast +
                               arma::solve_opts::no_approx);
    if (!solve_ok)
      continue;

    for (int k = 0; k < d; ++k)
      noise_ggm(k) = R::rnorm(0.0, 1.0);
    solve_ok = arma::solve(delta, arma::trimatu(U), noise_ggm.head(d),
                           arma::solve_opts::fast +
                               arma::solve_opts::no_approx);
    if (!solve_ok)
      continue;

    arma::vec b = mu_sub + delta;
    for (int k = 0; k < d; ++k) {
      int nbr = nbrs[k];
      double b2 = b(k) * b(k);
      double w1 = lv0h - 0.5 * b2 * iv0 + log_1pii;
      double w2 = lv1h - 0.5 * b2 * iv1 + log_pii;
      double wm = std::max(w1, w2);
      double prob = std::exp(w2 - wm) / (std::exp(w1 - wm) + std::exp(w2 - wm));

      bool was = sorted_contains(Z_active[i], nbr);
      bool now = (R::runif(0.0, 1.0) < prob);
      if (now && !was) {
        sorted_insert(Z_active[i], nbr);
        sorted_insert(Z_active[nbr], i);
        ++n_edges;
      } else if (!now && was) {
        sorted_erase(Z_active[i], nbr);
        sorted_erase(Z_active[nbr], i);
        --n_edges;
      }
    }
  }
}

static void init_sparse_pip_counters(
    const SparseS &S, bool store_Z_pip,
    std::vector<std::vector<int>> &Z_pip_cnt) {
  if (!store_Z_pip)
    return;
  const int p = S.p;
  Z_pip_cnt.resize(p);
  for (int j = 0; j < p; ++j)
    Z_pip_cnt[j].assign(S.adj[j].size(), 0);
}

static void accumulate_sparse_pip(const SparseS &S,
                                  const std::vector<std::vector<int>> &Z_active,
                                  bool store_Z_pip,
                                  std::vector<std::vector<int>> &Z_pip_cnt) {
  if (!store_Z_pip)
    return;

  const int p = S.p;
  for (int j = 0; j < p; ++j) {
    for (int nbr : Z_active[j]) {
      if (nbr <= j)
        continue;
      int idx = sorted_index(S.adj[j], nbr);
      if (idx >= 0)
        Z_pip_cnt[j][idx]++;
    }
  }
}

static Rcpp::List build_sparse_pip_triplet(const SparseS &S,
                                           bool store_Z_pip,
                                           const std::vector<std::vector<int>> &Z_pip_cnt,
                                           double n_post) {
  int total = 0;
  if (store_Z_pip) {
    for (int j = 0; j < S.p; ++j) {
      const int d = static_cast<int>(S.adj[j].size());
      for (int k = 0; k < d; ++k) {
        if (S.adj[j][k] > j && Z_pip_cnt[j][k] > 0)
          ++total;
      }
    }
  }

  Rcpp::IntegerVector row(total);
  Rcpp::IntegerVector col(total);
  Rcpp::NumericVector val(total);

  if (store_Z_pip) {
    int idx = 0;
    for (int j = 0; j < S.p; ++j) {
      const int d = static_cast<int>(S.adj[j].size());
      for (int k = 0; k < d; ++k) {
        int nbr = S.adj[j][k];
        int cnt = Z_pip_cnt[j][k];
        if (nbr <= j || cnt <= 0)
          continue;
        row[idx] = j;
        col[idx] = nbr;
        val[idx] = (n_post > 0.0) ? static_cast<double>(cnt) / n_post : 0.0;
        ++idx;
      }
    }
  }

  return Rcpp::List::create(Rcpp::Named("row") = row,
                            Rcpp::Named("col") = col,
                            Rcpp::Named("val") = val);
}

static void maybe_store_sparse_state(
    int iter, int burnin, int thin, int n_save,
    bool store_beta, bool store_gamma, bool store_Z_list,
    const std::vector<int> &active_idx, const arma::vec &beta_vec,
    const std::vector<std::vector<int>> &Z_active,
    arma::vec &eta1_out, arma::vec &eta2_out, bool dual_eta,
    arma::vec &alpha_out, arma::vec &sigmasq_out,
    double eta1, double eta2, double alpha, double sigmasq,
    Rcpp::List &beta_out_list, Rcpp::List &gamma_out_list, Rcpp::List &Z_list,
    std::vector<int> &edge_r, std::vector<int> &edge_c) {
  if (iter < burnin || ((iter - burnin) % thin) != 0)
    return;

  int s = (iter - burnin) / thin;
  if (s < 0 || s >= n_save)
    return;

  eta1_out(s) = eta1;
  if (dual_eta)
    eta2_out(s) = eta2;
  alpha_out(s) = alpha;
  sigmasq_out(s) = sigmasq;

  const int na = static_cast<int>(active_idx.size());

  if (store_beta) {
    if (na > 0) {
      Rcpp::NumericMatrix bm(na, 2);
      for (int k = 0; k < na; ++k) {
        bm(k, 0) = active_idx[k];
        bm(k, 1) = beta_vec(active_idx[k]);
      }
      beta_out_list[s] = bm;
    } else {
      beta_out_list[s] = Rcpp::NumericMatrix(0, 2);
    }
  }

  if (store_gamma) {
    if (na > 0) {
      Rcpp::IntegerVector gv(na);
      for (int k = 0; k < na; ++k)
        gv[k] = active_idx[k];
      gamma_out_list[s] = gv;
    } else {
      gamma_out_list[s] = Rcpp::IntegerVector(0);
    }
  }

  if (store_Z_list) {
    const int p = static_cast<int>(Z_active.size());
    edge_r.clear();
    edge_c.clear();
    for (int c = 0; c < p; ++c) {
      for (int r : Z_active[c]) {
        if (r < c) {
          edge_r.push_back(r);
          edge_c.push_back(c);
        }
      }
    }
    int ne = static_cast<int>(edge_r.size());
    if (ne > 0) {
      Rcpp::IntegerMatrix em(ne, 2);
      for (int k = 0; k < ne; ++k) {
        em(k, 0) = edge_r[k];
        em(k, 1) = edge_c[k];
      }
      Z_list[s] = em;
    } else {
      Z_list[s] = Rcpp::IntegerMatrix(0, 2);
    }
  }
}

} // namespace

// [[Rcpp::export]]
Rcpp::List BayesLogit_SingleNet_SparseGGM_UltraSparse(
    const arma::sp_mat &X, const arma::vec &y,
    const Rcpp::IntegerVector &S_i, const Rcpp::IntegerVector &S_p_csc,
    const Rcpp::NumericVector &S_x, const Rcpp::NumericVector &S_diag,
    int p_ggm, int niter, int burnin,
    double mu, double nu0, double sigmasq0, double alpha0, double beta0,
    double h, double e, double f,
    double v0_ggm, double v1_ggm, double pii_ggm,
    double eta_sd, double mu_tilde, double eta1_tilde, unsigned int T_max,
    int proposal_type, int n_mh_gamma = 5, int thin = 1,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0,
    bool store_beta = true, bool store_gamma = true,
    bool store_Z_list = false, bool store_Z_pip = true) {
  Rcpp::RNGScope scope;

  const int n = static_cast<int>(X.n_rows);
  const int p = static_cast<int>(X.n_cols);
  if (p != p_ggm)
    Rcpp::stop("p_ggm (%d) != ncol(X) (%d)", p_ggm, p);
  if (thin < 1)
    thin = 1;
  if (n_mh_gamma < 1)
    n_mh_gamma = 1;

  arma::vec y01;
  if (!validate_and_convert_y(y, y01))
    Rcpp::stop("y must be binary {0,1} or {-1,1}.");

  SparseS S = build_sparse_s(S_i, S_p_csc, S_x, S_diag, p);

  int d_max = 0;
  for (int j = 0; j < p; ++j)
    d_max = std::max(d_max, static_cast<int>(S.adj[j].size()));

  arma::vec beta_vec(p, arma::fill::zeros);
  std::vector<uint8_t> gamma(p, 0);
  if (beta_in.isNotNull()) {
    arma::vec b = Rcpp::as<arma::vec>(beta_in);
    if (static_cast<int>(b.n_elem) == p)
      beta_vec = b;
  }
  if (gamma_in.isNotNull()) {
    Rcpp::IntegerVector g = Rcpp::as<Rcpp::IntegerVector>(gamma_in);
    if (g.size() == p) {
      for (int j = 0; j < p; ++j)
        gamma[j] = static_cast<uint8_t>(g[j] != 0);
    }
  }

  std::vector<int> active_idx;
  active_idx.reserve(std::min(p, 100000));
  std::vector<int> active_pos(p, -1);
  for (int j = 0; j < p; ++j) {
    if (gamma[j]) {
      active_pos[j] = static_cast<int>(active_idx.size());
      active_idx.push_back(j);
    }
  }

  double alpha = alpha_in;
  double sigmasq = 1.0;
  double eta1 = std::min(0.01, eta_sd * 0.5);

  std::vector<std::vector<int>> Z_active(p);
  int n_edges = 0;
  for (int j = 0; j < p; ++j) {
    Z_active[j] = S.adj[j];
    for (int nbr : Z_active[j])
      if (nbr > j)
        ++n_edges;
  }

  arma::vec Xb = X * beta_vec;
  double loglik = calc_loglik_full(y01, Xb, alpha);

  arma::mat A_sub;
  arma::vec s_ggm, noise_ggm;
  if (d_max > 0) {
    A_sub.set_size(d_max, d_max);
    s_ggm.set_size(d_max);
    noise_ggm.set_size(d_max);
  }

  const double log_pii = std::log(pii_ggm);
  const double log_1pii = std::log(1.0 - pii_ggm);
  const double lv0h = -0.5 * std::log(v0_ggm);
  const double lv1h = -0.5 * std::log(v1_ggm);
  const double iv0 = 1.0 / v0_ggm;
  const double iv1 = 1.0 / v1_ggm;

  std::vector<int> pw_up(p), pw_dn(p), pw_om1(p), pw_om1n(p);

  std::vector<std::vector<int>> Z_pip_cnt;
  init_sparse_pip_counters(S, store_Z_pip, Z_pip_cnt);

  const int n_save = niter / thin;
  arma::vec eta1_out(n_save, arma::fill::zeros);
  arma::vec eta2_out;
  arma::vec alpha_out(n_save, arma::fill::zeros);
  arma::vec sigmasq_out(n_save, arma::fill::zeros);

  Rcpp::List beta_out_list = store_beta ? Rcpp::List(n_save) : Rcpp::List();
  Rcpp::List gamma_out_list = store_gamma ? Rcpp::List(n_save) : Rcpp::List();
  Rcpp::List Z_list = store_Z_list ? Rcpp::List(n_save) : Rcpp::List();

  std::vector<int> edge_r;
  std::vector<int> edge_c;
  edge_r.reserve(std::max(1000, n_edges));
  edge_c.reserve(std::max(1000, n_edges));

  const arma::uword *col_ptr = X.col_ptrs;
  const arma::uword *row_idx = X.row_indices;
  const double *xvals = X.values;

  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {
    if (iter > 0 && (iter % 2000) == 0)
      Rcpp::checkUserInterrupt();

    const double sd_sig = std::sqrt(sigmasq);

    ggm_column_sweep_sparse(S, Z_active, p, log_pii, log_1pii, lv0h, lv1h, iv0,
                            iv1, A_sub, s_ggm, noise_ggm, n_edges);

    for (int mh = 0; mh < n_mh_gamma; ++mh) {
      int j = static_cast<int>(std::floor(R::runif(0.0, static_cast<double>(p))));
      if (j >= p)
        j = p - 1;

      int g_curr = static_cast<int>(gamma[j]);
      int g_prop = 1 - g_curr;
      double b_curr = beta_vec(j);
      double b_prop = (g_prop == 1) ? R::rnorm(beta0, sd_sig) : 0.0;
      double db = b_prop - b_curr;

      double ll_diff = column_ll_diff(y01, Xb, alpha, col_ptr, row_idx, xvals,
                                      j, db);
      double neigh = 0.0;
      for (int nbr : Z_active[j])
        neigh += static_cast<int>(gamma[nbr]);

      double ising = static_cast<double>(g_prop - g_curr) * (mu + eta1 * neigh);
      double lmh = ll_diff + ising;

      if (std::log(R::runif(0.0, 1.0)) < lmh) {
        gamma[j] = static_cast<uint8_t>(g_prop);
        beta_vec(j) = b_prop;
        loglik += ll_diff;
        apply_column_update(Xb, col_ptr, row_idx, xvals, j, db);
        if (g_prop == 1)
          activate_gamma(j, active_idx, active_pos);
        else
          deactivate_gamma(j, active_idx, active_pos);
      }
    }

    const double sd_beta = 0.1 * sd_sig;
    const int na = static_cast<int>(active_idx.size());
    for (int k = 0; k < na; ++k) {
      int j = active_idx[k];
      double b_curr = beta_vec(j);
      double b_prop = R::rnorm(b_curr, sd_beta);
      double db = b_prop - b_curr;

      double ll_diff = column_ll_diff(y01, Xb, alpha, col_ptr, row_idx, xvals,
                                      j, db);
      double d_curr = b_curr - beta0;
      double d_prop = b_prop - beta0;
      double prior_diff = -0.5 * (d_prop * d_prop - d_curr * d_curr) / sigmasq;
      double lmh = ll_diff + prior_diff;

      if (std::log(R::runif(0.0, 1.0)) < lmh) {
        beta_vec(j) = b_prop;
        loglik += ll_diff;
        apply_column_update(Xb, col_ptr, row_idx, xvals, j, db);
      }
    }

    {
      double a_prop = R::rnorm(alpha, std::sqrt(h * sigmasq));
      double ll_prop = calc_loglik_full(y01, Xb, a_prop);
      double d_curr = alpha - alpha0;
      double d_prop = a_prop - alpha0;
      double prior_diff =
          -0.5 * (d_prop * d_prop - d_curr * d_curr) / (h * sigmasq);
      double lmh = (ll_prop - loglik) + prior_diff;
      if (std::log(R::runif(0.0, 1.0)) < lmh) {
        alpha = a_prop;
        loglik = ll_prop;
      }
    }

    {
      double sig_prop = std::exp(std::log(sigmasq) + R::rnorm(0.0, 0.2));
      sig_prop = std::max(sig_prop, 1e-10);

      double sh = nu0 / 2.0;
      double sc = sigmasq0 * nu0 / 2.0;
      double lp_c = -(sh + 1.0) * std::log(sigmasq) - sc / sigmasq;
      double lp_p = -(sh + 1.0) * std::log(sig_prop) - sc / sig_prop;

      double ss = 0.0;
      for (int j : active_idx) {
        double d = beta_vec(j) - beta0;
        ss += d * d;
      }

      double n_act = static_cast<double>(active_idx.size());
      double lb_c = -0.5 * n_act * std::log(sigmasq) - 0.5 * ss / sigmasq;
      double lb_p = -0.5 * n_act * std::log(sig_prop) - 0.5 * ss / sig_prop;

      double da = alpha - alpha0;
      double la_c = -0.5 * std::log(h * sigmasq) - 0.5 * da * da / (h * sigmasq);
      double la_p = -0.5 * std::log(h * sig_prop) - 0.5 * da * da / (h * sig_prop);

      double lmh = (lp_p + lb_p + la_p) - (lp_c + lb_c + la_c) +
                   std::log(sig_prop / sigmasq);
      if (std::log(R::runif(0.0, 1.0)) < lmh)
        sigmasq = sig_prop;
    }

    moller_update_single_sparse(Z_active, p, mu, eta1, eta_sd, mu_tilde,
                                eta1_tilde, gamma, e, f, T_max, proposal_type,
                                pw_up, pw_dn, pw_om1, pw_om1n);

    maybe_store_sparse_state(iter, burnin, thin, n_save, store_beta,
                             store_gamma, store_Z_list, active_idx, beta_vec,
                             Z_active, eta1_out, eta2_out, false, alpha_out,
                             sigmasq_out, eta1, 0.0, alpha, sigmasq,
                             beta_out_list, gamma_out_list, Z_list, edge_r,
                             edge_c);

    if (iter >= burnin)
      accumulate_sparse_pip(S, Z_active, store_Z_pip, Z_pip_cnt);
  }

  double n_post = static_cast<double>(total_iter - burnin);
  Rcpp::List pip_trip = build_sparse_pip_triplet(S, store_Z_pip, Z_pip_cnt, n_post);

  SEXP beta_out_sexp = store_beta ? static_cast<SEXP>(beta_out_list) : R_NilValue;
  SEXP gamma_out_sexp = store_gamma ? static_cast<SEXP>(gamma_out_list) : R_NilValue;
  SEXP z_list_sexp = store_Z_list ? static_cast<SEXP>(Z_list) : R_NilValue;

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_out_sexp,
      Rcpp::Named("gamma") = gamma_out_sexp,
      Rcpp::Named("eta1") = eta1_out,
      Rcpp::Named("alpha") = alpha_out,
      Rcpp::Named("sigmasq") = sigmasq_out,
      Rcpp::Named("Z_list") = z_list_sexp,
      Rcpp::Named("Z_pip_row") = pip_trip["row"],
      Rcpp::Named("Z_pip_col") = pip_trip["col"],
      Rcpp::Named("Z_pip_val") = pip_trip["val"],
      Rcpp::Named("p") = p,
      Rcpp::Named("n") = n);
}

// [[Rcpp::export]]
Rcpp::List BayesLogit_DualNet_SparseGGM_UltraSparse(
    const arma::sp_mat &X, const arma::vec &y,
    const Rcpp::IntegerVector &S_i, const Rcpp::IntegerVector &S_p_csc,
    const Rcpp::NumericVector &S_x, const Rcpp::NumericVector &S_diag,
    const Rcpp::IntegerVector &R_fix_i,
    const Rcpp::IntegerVector &R_fix_p_csc,
    int p_ggm, int niter, int burnin,
    double mu, double nu0, double sigmasq0, double alpha0, double beta0,
    double h, double e, double f,
    double v0_ggm, double v1_ggm, double pii_ggm,
    double eta1_sd, double eta2_sd, double mu_tilde, double eta1_tilde,
    double eta2_tilde, unsigned int T_max, int proposal_type,
    int n_mh_gamma = 5, int thin = 1,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0,
    bool store_beta = true, bool store_gamma = true,
    bool store_Z_list = false, bool store_Z_pip = true) {
  Rcpp::RNGScope scope;

  const int n = static_cast<int>(X.n_rows);
  const int p = static_cast<int>(X.n_cols);
  if (p != p_ggm)
    Rcpp::stop("p_ggm (%d) != ncol(X) (%d)", p_ggm, p);
  if (thin < 1)
    thin = 1;
  if (n_mh_gamma < 1)
    n_mh_gamma = 1;

  arma::vec y01;
  if (!validate_and_convert_y(y, y01))
    Rcpp::stop("y must be binary {0,1} or {-1,1}.");

  SparseS S = build_sparse_s(S_i, S_p_csc, S_x, S_diag, p);
  SparseAdj R_fix = build_sparse_adj(R_fix_i, R_fix_p_csc, p, true);

  int d_max = 0;
  for (int j = 0; j < p; ++j)
    d_max = std::max(d_max, static_cast<int>(S.adj[j].size()));

  arma::vec beta_vec(p, arma::fill::zeros);
  std::vector<uint8_t> gamma(p, 0);
  if (beta_in.isNotNull()) {
    arma::vec b = Rcpp::as<arma::vec>(beta_in);
    if (static_cast<int>(b.n_elem) == p)
      beta_vec = b;
  }
  if (gamma_in.isNotNull()) {
    Rcpp::IntegerVector g = Rcpp::as<Rcpp::IntegerVector>(gamma_in);
    if (g.size() == p) {
      for (int j = 0; j < p; ++j)
        gamma[j] = static_cast<uint8_t>(g[j] != 0);
    }
  }

  std::vector<int> active_idx;
  active_idx.reserve(std::min(p, 100000));
  std::vector<int> active_pos(p, -1);
  for (int j = 0; j < p; ++j) {
    if (gamma[j]) {
      active_pos[j] = static_cast<int>(active_idx.size());
      active_idx.push_back(j);
    }
  }

  double alpha = alpha_in;
  double sigmasq = 1.0;
  double eta1 = std::min(0.01, eta1_sd * 0.5);
  double eta2 = std::min(0.01, eta2_sd * 0.5);

  std::vector<std::vector<int>> Z_active(p);
  int n_edges = 0;
  for (int j = 0; j < p; ++j) {
    Z_active[j] = S.adj[j];
    for (int nbr : Z_active[j])
      if (nbr > j)
        ++n_edges;
  }

  arma::vec Xb = X * beta_vec;
  double loglik = calc_loglik_full(y01, Xb, alpha);

  arma::mat A_sub;
  arma::vec s_ggm, noise_ggm;
  if (d_max > 0) {
    A_sub.set_size(d_max, d_max);
    s_ggm.set_size(d_max);
    noise_ggm.set_size(d_max);
  }

  const double log_pii = std::log(pii_ggm);
  const double log_1pii = std::log(1.0 - pii_ggm);
  const double lv0h = -0.5 * std::log(v0_ggm);
  const double lv1h = -0.5 * std::log(v1_ggm);
  const double iv0 = 1.0 / v0_ggm;
  const double iv1 = 1.0 / v1_ggm;

  std::vector<int> pw_up(p), pw_dn(p), pw_om1(p), pw_om2(p), pw_om1n(p),
      pw_om2n(p);

  std::vector<std::vector<int>> Z_pip_cnt;
  init_sparse_pip_counters(S, store_Z_pip, Z_pip_cnt);

  const int n_save = niter / thin;
  arma::vec eta1_out(n_save, arma::fill::zeros);
  arma::vec eta2_out(n_save, arma::fill::zeros);
  arma::vec alpha_out(n_save, arma::fill::zeros);
  arma::vec sigmasq_out(n_save, arma::fill::zeros);

  Rcpp::List beta_out_list = store_beta ? Rcpp::List(n_save) : Rcpp::List();
  Rcpp::List gamma_out_list = store_gamma ? Rcpp::List(n_save) : Rcpp::List();
  Rcpp::List Z_list = store_Z_list ? Rcpp::List(n_save) : Rcpp::List();

  std::vector<int> edge_r;
  std::vector<int> edge_c;
  edge_r.reserve(std::max(1000, n_edges));
  edge_c.reserve(std::max(1000, n_edges));

  const arma::uword *col_ptr = X.col_ptrs;
  const arma::uword *row_idx = X.row_indices;
  const double *xvals = X.values;

  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {
    if (iter > 0 && (iter % 2000) == 0)
      Rcpp::checkUserInterrupt();

    const double sd_sig = std::sqrt(sigmasq);

    ggm_column_sweep_sparse(S, Z_active, p, log_pii, log_1pii, lv0h, lv1h, iv0,
                            iv1, A_sub, s_ggm, noise_ggm, n_edges);

    for (int mh = 0; mh < n_mh_gamma; ++mh) {
      int j = static_cast<int>(std::floor(R::runif(0.0, static_cast<double>(p))));
      if (j >= p)
        j = p - 1;

      int g_curr = static_cast<int>(gamma[j]);
      int g_prop = 1 - g_curr;
      double b_curr = beta_vec(j);
      double b_prop = (g_prop == 1) ? R::rnorm(beta0, sd_sig) : 0.0;
      double db = b_prop - b_curr;

      double ll_diff = column_ll_diff(y01, Xb, alpha, col_ptr, row_idx, xvals,
                                      j, db);
      double neigh_dyn = 0.0;
      for (int nbr : Z_active[j])
        neigh_dyn += static_cast<int>(gamma[nbr]);
      double neigh_fix = 0.0;
      for (int nbr : R_fix.adj[j])
        neigh_fix += static_cast<int>(gamma[nbr]);

      double ising = static_cast<double>(g_prop - g_curr) *
                     (mu + eta1 * neigh_dyn + eta2 * neigh_fix);
      double lmh = ll_diff + ising;

      if (std::log(R::runif(0.0, 1.0)) < lmh) {
        gamma[j] = static_cast<uint8_t>(g_prop);
        beta_vec(j) = b_prop;
        loglik += ll_diff;
        apply_column_update(Xb, col_ptr, row_idx, xvals, j, db);
        if (g_prop == 1)
          activate_gamma(j, active_idx, active_pos);
        else
          deactivate_gamma(j, active_idx, active_pos);
      }
    }

    const double sd_beta = 0.1 * sd_sig;
    const int na = static_cast<int>(active_idx.size());
    for (int k = 0; k < na; ++k) {
      int j = active_idx[k];
      double b_curr = beta_vec(j);
      double b_prop = R::rnorm(b_curr, sd_beta);
      double db = b_prop - b_curr;

      double ll_diff = column_ll_diff(y01, Xb, alpha, col_ptr, row_idx, xvals,
                                      j, db);
      double d_curr = b_curr - beta0;
      double d_prop = b_prop - beta0;
      double prior_diff = -0.5 * (d_prop * d_prop - d_curr * d_curr) / sigmasq;
      double lmh = ll_diff + prior_diff;

      if (std::log(R::runif(0.0, 1.0)) < lmh) {
        beta_vec(j) = b_prop;
        loglik += ll_diff;
        apply_column_update(Xb, col_ptr, row_idx, xvals, j, db);
      }
    }

    {
      double a_prop = R::rnorm(alpha, std::sqrt(h * sigmasq));
      double ll_prop = calc_loglik_full(y01, Xb, a_prop);
      double d_curr = alpha - alpha0;
      double d_prop = a_prop - alpha0;
      double prior_diff =
          -0.5 * (d_prop * d_prop - d_curr * d_curr) / (h * sigmasq);
      double lmh = (ll_prop - loglik) + prior_diff;
      if (std::log(R::runif(0.0, 1.0)) < lmh) {
        alpha = a_prop;
        loglik = ll_prop;
      }
    }

    {
      double sig_prop = std::exp(std::log(sigmasq) + R::rnorm(0.0, 0.2));
      sig_prop = std::max(sig_prop, 1e-10);

      double sh = nu0 / 2.0;
      double sc = sigmasq0 * nu0 / 2.0;
      double lp_c = -(sh + 1.0) * std::log(sigmasq) - sc / sigmasq;
      double lp_p = -(sh + 1.0) * std::log(sig_prop) - sc / sig_prop;

      double ss = 0.0;
      for (int j : active_idx) {
        double d = beta_vec(j) - beta0;
        ss += d * d;
      }

      double n_act = static_cast<double>(active_idx.size());
      double lb_c = -0.5 * n_act * std::log(sigmasq) - 0.5 * ss / sigmasq;
      double lb_p = -0.5 * n_act * std::log(sig_prop) - 0.5 * ss / sig_prop;

      double da = alpha - alpha0;
      double la_c = -0.5 * std::log(h * sigmasq) - 0.5 * da * da / (h * sigmasq);
      double la_p = -0.5 * std::log(h * sig_prop) - 0.5 * da * da / (h * sig_prop);

      double lmh = (lp_p + lb_p + la_p) - (lp_c + lb_c + la_c) +
                   std::log(sig_prop / sigmasq);
      if (std::log(R::runif(0.0, 1.0)) < lmh)
        sigmasq = sig_prop;
    }

    moller_update_dual_sparse(
        Z_active, R_fix.adj, p, mu, eta1, eta2, eta1_sd, eta2_sd, mu_tilde,
        eta1_tilde, eta2_tilde, gamma, e, f, T_max, proposal_type, pw_up,
        pw_dn, pw_om1, pw_om2, pw_om1n, pw_om2n);

    maybe_store_sparse_state(iter, burnin, thin, n_save, store_beta,
                             store_gamma, store_Z_list, active_idx, beta_vec,
                             Z_active, eta1_out, eta2_out, true, alpha_out,
                             sigmasq_out, eta1, eta2, alpha, sigmasq,
                             beta_out_list, gamma_out_list, Z_list, edge_r,
                             edge_c);

    if (iter >= burnin)
      accumulate_sparse_pip(S, Z_active, store_Z_pip, Z_pip_cnt);
  }

  double n_post = static_cast<double>(total_iter - burnin);
  Rcpp::List pip_trip = build_sparse_pip_triplet(S, store_Z_pip, Z_pip_cnt, n_post);

  SEXP beta_out_sexp = store_beta ? static_cast<SEXP>(beta_out_list) : R_NilValue;
  SEXP gamma_out_sexp = store_gamma ? static_cast<SEXP>(gamma_out_list) : R_NilValue;
  SEXP z_list_sexp = store_Z_list ? static_cast<SEXP>(Z_list) : R_NilValue;

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_out_sexp,
      Rcpp::Named("gamma") = gamma_out_sexp,
      Rcpp::Named("eta1") = eta1_out,
      Rcpp::Named("eta2") = eta2_out,
      Rcpp::Named("alpha") = alpha_out,
      Rcpp::Named("sigmasq") = sigmasq_out,
      Rcpp::Named("Z_list") = z_list_sexp,
      Rcpp::Named("Z_pip_row") = pip_trip["row"],
      Rcpp::Named("Z_pip_col") = pip_trip["col"],
      Rcpp::Named("Z_pip_val") = pip_trip["val"],
      Rcpp::Named("p") = p,
      Rcpp::Named("n") = n);
}

// [[Rcpp::export]]
Rcpp::List BayesLogit_PG_SingleNet_SparseGGM_UltraSparse(
    const arma::sp_mat &X, const arma::vec &y,
    const Rcpp::IntegerVector &S_i, const Rcpp::IntegerVector &S_p_csc,
    const Rcpp::NumericVector &S_x, const Rcpp::NumericVector &S_diag,
    int p_ggm, int niter, int burnin,
    double mu, double nu0, double sigmasq0, double alpha0, double beta0,
    double h, int n_mh_gamma,
    double v0_ggm, double v1_ggm, double pii_ggm,
    double eta_sd, double mu_tilde, double eta1_tilde,
    double e_eta, double f_eta, unsigned int T_max, int proposal_type,
    int thin = 1,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0,
    bool store_beta = true, bool store_gamma = true,
    bool store_Z_list = false, bool store_Z_pip = true) {
  Rcpp::RNGScope scope;

  const int n = static_cast<int>(X.n_rows);
  const int p = static_cast<int>(X.n_cols);
  if (p != p_ggm)
    Rcpp::stop("p_ggm (%d) != ncol(X) (%d)", p_ggm, p);
  if (thin < 1)
    thin = 1;
  if (n_mh_gamma < 1)
    n_mh_gamma = 1;

  arma::vec y01;
  if (!validate_and_convert_y(y, y01))
    Rcpp::stop("y must be binary {0,1} or {-1,1}.");

  SparseS S = build_sparse_s(S_i, S_p_csc, S_x, S_diag, p);

  int d_max = 0;
  for (int j = 0; j < p; ++j)
    d_max = std::max(d_max, static_cast<int>(S.adj[j].size()));

  arma::vec beta_vec(p, arma::fill::zeros);
  std::vector<uint8_t> gamma(p, 0);
  if (beta_in.isNotNull()) {
    arma::vec b = Rcpp::as<arma::vec>(beta_in);
    if (static_cast<int>(b.n_elem) == p)
      beta_vec = b;
  }
  if (gamma_in.isNotNull()) {
    Rcpp::IntegerVector g = Rcpp::as<Rcpp::IntegerVector>(gamma_in);
    if (g.size() == p) {
      for (int j = 0; j < p; ++j)
        gamma[j] = static_cast<uint8_t>(g[j] != 0);
    }
  }

  std::vector<int> active_idx;
  active_idx.reserve(std::min(p, 100000));
  std::vector<int> active_pos(p, -1);
  for (int j = 0; j < p; ++j) {
    if (gamma[j]) {
      active_pos[j] = static_cast<int>(active_idx.size());
      active_idx.push_back(j);
    }
  }

  double alpha = alpha_in;
  double sigmasq = 1.0;
  double eta1 = std::min(0.01, eta_sd * 0.5);

  std::vector<std::vector<int>> Z_active(p);
  int n_edges = 0;
  for (int j = 0; j < p; ++j) {
    Z_active[j] = S.adj[j];
    for (int nbr : Z_active[j])
      if (nbr > j)
        ++n_edges;
  }

  arma::vec Xb = X * beta_vec;
  arma::vec omega_pg(n, arma::fill::ones);
  const arma::vec kappa = y01 - 0.5;

  arma::mat A_sub;
  arma::vec s_ggm, noise_ggm;
  if (d_max > 0) {
    A_sub.set_size(d_max, d_max);
    s_ggm.set_size(d_max);
    noise_ggm.set_size(d_max);
  }

  const double log_pii = std::log(pii_ggm);
  const double log_1pii = std::log(1.0 - pii_ggm);
  const double lv0h = -0.5 * std::log(v0_ggm);
  const double lv1h = -0.5 * std::log(v1_ggm);
  const double iv0 = 1.0 / v0_ggm;
  const double iv1 = 1.0 / v1_ggm;

  std::vector<int> pw_up(p), pw_dn(p), pw_om1(p), pw_om1n(p);
  std::mt19937 pg_rng(static_cast<unsigned int>(R::runif(0.0, 1.0) * 1e9));

  std::vector<std::vector<int>> Z_pip_cnt;
  init_sparse_pip_counters(S, store_Z_pip, Z_pip_cnt);

  const int n_save = niter / thin;
  arma::vec eta1_out(n_save, arma::fill::zeros);
  arma::vec eta2_out;
  arma::vec alpha_out(n_save, arma::fill::zeros);
  arma::vec sigmasq_out(n_save, arma::fill::zeros);

  Rcpp::List beta_out_list = store_beta ? Rcpp::List(n_save) : Rcpp::List();
  Rcpp::List gamma_out_list = store_gamma ? Rcpp::List(n_save) : Rcpp::List();
  Rcpp::List Z_list = store_Z_list ? Rcpp::List(n_save) : Rcpp::List();

  std::vector<int> edge_r;
  std::vector<int> edge_c;
  edge_r.reserve(std::max(1000, n_edges));
  edge_c.reserve(std::max(1000, n_edges));

  const arma::uword *col_ptr = X.col_ptrs;
  const arma::uword *row_idx = X.row_indices;
  const double *xvals = X.values;

  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {
    if (iter > 0 && (iter % 2000) == 0)
      Rcpp::checkUserInterrupt();

    const double sd_sig = std::sqrt(sigmasq);

    ggm_column_sweep_sparse(S, Z_active, p, log_pii, log_1pii, lv0h, lv1h, iv0,
                            iv1, A_sub, s_ggm, noise_ggm, n_edges);

    {
      arma::vec lin = alpha + Xb;
      for (int i = 0; i < n; ++i)
        omega_pg(i) = sample_pg(lin(i), pg_rng);
    }

    const int p_act = static_cast<int>(active_idx.size());
    bool beta_refreshed = false;
    if (p_act > 0) {
      arma::mat X_act(n, p_act, arma::fill::zeros);
      for (int k = 0; k < p_act; ++k) {
        int j = active_idx[k];
        for (arma::sp_mat::const_col_iterator it = X.begin_col(j);
             it != X.end_col(j); ++it) {
          X_act(it.row(), k) = (*it);
        }
      }

      arma::mat Xt_Om = X_act.t();
      Xt_Om.each_row() %= omega_pg.t();
      arma::mat prec = Xt_Om * X_act;
      prec.diag() += 1.0 / sigmasq;

      arma::vec z_star = kappa - omega_pg * alpha;
      arma::vec rhs = X_act.t() * z_star;

      arma::mat L;
      bool chol_ok = arma::chol(L, arma::symmatu(prec));
      if (!chol_ok) {
        double jit = 1e-8;
        for (int att = 0; att < 5 && !chol_ok; ++att) {
          prec.diag() += jit;
          chol_ok = arma::chol(L, arma::symmatu(prec));
          jit *= 10.0;
        }
      }

      if (chol_ok) {
        arma::vec mb;
        bool solve_ok = arma::solve(mb, arma::trimatl(L.t()), rhs,
                                    arma::solve_opts::fast +
                                        arma::solve_opts::no_approx);
        if (solve_ok) {
          solve_ok = arma::solve(mb, arma::trimatu(L), mb,
                                 arma::solve_opts::fast +
                                     arma::solve_opts::no_approx);
        }

        if (solve_ok) {
          arma::vec zz = arma::randn<arma::vec>(p_act);
          arma::vec pert;
          solve_ok = arma::solve(pert, arma::trimatu(L), zz,
                                 arma::solve_opts::fast +
                                     arma::solve_opts::no_approx);
          if (solve_ok) {
            arma::vec bd = mb + pert;
            beta_vec.zeros();
            for (int k = 0; k < p_act; ++k)
              beta_vec(active_idx[k]) = bd(k);
            Xb = X_act * bd;
            beta_refreshed = true;
          }
        }
      }
    } else {
      beta_vec.zeros();
      Xb.zeros();
      beta_refreshed = true;
    }

    if (!beta_refreshed) {
      // Keep current Xb as-is when PG beta solve fails and beta is unchanged.
      // This avoids an unnecessary O(nnz(X)) sparse mat-vec multiply.
    }

    {
      double sum_om = arma::accu(omega_pg);
      double prec_a = sum_om + 1.0 / (h * sigmasq);
      double var_a = 1.0 / prec_a;
      arma::vec resid = kappa - omega_pg % Xb;
      alpha = R::rnorm(var_a * (arma::accu(resid) + alpha0 / (h * sigmasq)),
                       std::sqrt(var_a));
    }

    for (int mh = 0; mh < n_mh_gamma; ++mh) {
      int j = static_cast<int>(std::floor(R::runif(0.0, static_cast<double>(p))));
      if (j >= p)
        j = p - 1;

      int g_curr = static_cast<int>(gamma[j]);
      int g_prop = 1 - g_curr;
      double b_curr = beta_vec(j);
      double b_prop = (g_prop == 1) ? R::rnorm(beta0, sd_sig) : 0.0;
      double db = b_prop - b_curr;

      double ll_diff = column_ll_diff(y01, Xb, alpha, col_ptr, row_idx, xvals,
                                      j, db);
      double neigh_dyn = 0.0;
      for (int nbr : Z_active[j])
        neigh_dyn += static_cast<int>(gamma[nbr]);

      double ising = static_cast<double>(g_prop - g_curr) * (mu + eta1 * neigh_dyn);
      double lmh = ll_diff + ising;

      if (std::log(R::runif(0.0, 1.0)) < lmh) {
        gamma[j] = static_cast<uint8_t>(g_prop);
        beta_vec(j) = b_prop;
        apply_column_update(Xb, col_ptr, row_idx, xvals, j, db);
        if (g_prop == 1)
          activate_gamma(j, active_idx, active_pos);
        else
          deactivate_gamma(j, active_idx, active_pos);
      }
    }

    {
      double sig_prop = std::exp(std::log(sigmasq) + R::rnorm(0.0, 0.2));
      sig_prop = std::max(sig_prop, 1e-10);

      double sh = nu0 / 2.0;
      double sc = sigmasq0 * nu0 / 2.0;
      double lp_c = -(sh + 1.0) * std::log(sigmasq) - sc / sigmasq;
      double lp_p = -(sh + 1.0) * std::log(sig_prop) - sc / sig_prop;

      double ss = 0.0;
      for (int j : active_idx) {
        double d = beta_vec(j) - beta0;
        ss += d * d;
      }

      double n_act = static_cast<double>(active_idx.size());
      double lb_c = -0.5 * n_act * std::log(sigmasq) - 0.5 * ss / sigmasq;
      double lb_p = -0.5 * n_act * std::log(sig_prop) - 0.5 * ss / sig_prop;

      double da = alpha - alpha0;
      double la_c = -0.5 * std::log(h * sigmasq) - 0.5 * da * da / (h * sigmasq);
      double la_p = -0.5 * std::log(h * sig_prop) - 0.5 * da * da / (h * sig_prop);

      double lmh = (lp_p + lb_p + la_p) - (lp_c + lb_c + la_c) +
                   std::log(sig_prop / sigmasq);
      if (std::log(R::runif(0.0, 1.0)) < lmh)
        sigmasq = sig_prop;
    }

    moller_update_single_sparse(
        Z_active, p, mu, eta1, eta_sd, mu_tilde, eta1_tilde, gamma, e_eta,
        f_eta, T_max, proposal_type, pw_up, pw_dn, pw_om1, pw_om1n);

    maybe_store_sparse_state(iter, burnin, thin, n_save, store_beta,
                             store_gamma, store_Z_list, active_idx, beta_vec,
                             Z_active, eta1_out, eta2_out, false, alpha_out,
                             sigmasq_out, eta1, 0.0, alpha, sigmasq,
                             beta_out_list, gamma_out_list, Z_list, edge_r,
                             edge_c);

    if (iter >= burnin)
      accumulate_sparse_pip(S, Z_active, store_Z_pip, Z_pip_cnt);
  }

  double n_post = static_cast<double>(total_iter - burnin);
  Rcpp::List pip_trip = build_sparse_pip_triplet(S, store_Z_pip, Z_pip_cnt, n_post);

  SEXP beta_out_sexp = store_beta ? static_cast<SEXP>(beta_out_list) : R_NilValue;
  SEXP gamma_out_sexp = store_gamma ? static_cast<SEXP>(gamma_out_list) : R_NilValue;
  SEXP z_list_sexp = store_Z_list ? static_cast<SEXP>(Z_list) : R_NilValue;

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_out_sexp,
      Rcpp::Named("gamma") = gamma_out_sexp,
      Rcpp::Named("eta1") = eta1_out,
      Rcpp::Named("alpha") = alpha_out,
      Rcpp::Named("sigmasq") = sigmasq_out,
      Rcpp::Named("Z_list") = z_list_sexp,
      Rcpp::Named("Z_pip_row") = pip_trip["row"],
      Rcpp::Named("Z_pip_col") = pip_trip["col"],
      Rcpp::Named("Z_pip_val") = pip_trip["val"],
      Rcpp::Named("p") = p,
      Rcpp::Named("n") = n);
}

// [[Rcpp::export]]
Rcpp::List BayesLogit_PG_DualNet_SparseGGM_UltraSparse(
    const arma::sp_mat &X, const arma::vec &y,
    const Rcpp::IntegerVector &S_i, const Rcpp::IntegerVector &S_p_csc,
    const Rcpp::NumericVector &S_x, const Rcpp::NumericVector &S_diag,
    const Rcpp::IntegerVector &R_fix_i,
    const Rcpp::IntegerVector &R_fix_p_csc,
    int p_ggm, int niter, int burnin,
    double mu, double nu0, double sigmasq0, double alpha0, double beta0,
    double h, int n_mh_gamma,
    double v0_ggm, double v1_ggm, double pii_ggm,
    double eta1_sd, double eta2_sd, double mu_tilde, double eta1_tilde,
    double eta2_tilde, double e_eta, double f_eta, unsigned int T_max,
    int proposal_type, int thin = 1,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0,
    bool store_beta = true, bool store_gamma = true,
    bool store_Z_list = false, bool store_Z_pip = true) {
  Rcpp::RNGScope scope;

  const int n = static_cast<int>(X.n_rows);
  const int p = static_cast<int>(X.n_cols);
  if (p != p_ggm)
    Rcpp::stop("p_ggm (%d) != ncol(X) (%d)", p_ggm, p);
  if (thin < 1)
    thin = 1;
  if (n_mh_gamma < 1)
    n_mh_gamma = 1;

  arma::vec y01;
  if (!validate_and_convert_y(y, y01))
    Rcpp::stop("y must be binary {0,1} or {-1,1}.");

  SparseS S = build_sparse_s(S_i, S_p_csc, S_x, S_diag, p);
  SparseAdj R_fix = build_sparse_adj(R_fix_i, R_fix_p_csc, p, true);

  int d_max = 0;
  for (int j = 0; j < p; ++j)
    d_max = std::max(d_max, static_cast<int>(S.adj[j].size()));

  arma::vec beta_vec(p, arma::fill::zeros);
  std::vector<uint8_t> gamma(p, 0);
  if (beta_in.isNotNull()) {
    arma::vec b = Rcpp::as<arma::vec>(beta_in);
    if (static_cast<int>(b.n_elem) == p)
      beta_vec = b;
  }
  if (gamma_in.isNotNull()) {
    Rcpp::IntegerVector g = Rcpp::as<Rcpp::IntegerVector>(gamma_in);
    if (g.size() == p) {
      for (int j = 0; j < p; ++j)
        gamma[j] = static_cast<uint8_t>(g[j] != 0);
    }
  }

  std::vector<int> active_idx;
  active_idx.reserve(std::min(p, 100000));
  std::vector<int> active_pos(p, -1);
  for (int j = 0; j < p; ++j) {
    if (gamma[j]) {
      active_pos[j] = static_cast<int>(active_idx.size());
      active_idx.push_back(j);
    }
  }

  double alpha = alpha_in;
  double sigmasq = 1.0;
  double eta1 = std::min(0.01, eta1_sd * 0.5);
  double eta2 = std::min(0.01, eta2_sd * 0.5);

  std::vector<std::vector<int>> Z_active(p);
  int n_edges = 0;
  for (int j = 0; j < p; ++j) {
    Z_active[j] = S.adj[j];
    for (int nbr : Z_active[j])
      if (nbr > j)
        ++n_edges;
  }

  arma::vec Xb = X * beta_vec;
  arma::vec omega_pg(n, arma::fill::ones);
  const arma::vec kappa = y01 - 0.5;

  arma::mat A_sub;
  arma::vec s_ggm, noise_ggm;
  if (d_max > 0) {
    A_sub.set_size(d_max, d_max);
    s_ggm.set_size(d_max);
    noise_ggm.set_size(d_max);
  }

  const double log_pii = std::log(pii_ggm);
  const double log_1pii = std::log(1.0 - pii_ggm);
  const double lv0h = -0.5 * std::log(v0_ggm);
  const double lv1h = -0.5 * std::log(v1_ggm);
  const double iv0 = 1.0 / v0_ggm;
  const double iv1 = 1.0 / v1_ggm;

  std::vector<int> pw_up(p), pw_dn(p), pw_om1(p), pw_om2(p), pw_om1n(p),
      pw_om2n(p);

  std::mt19937 pg_rng(static_cast<unsigned int>(R::runif(0.0, 1.0) * 1e9));

  std::vector<std::vector<int>> Z_pip_cnt;
  init_sparse_pip_counters(S, store_Z_pip, Z_pip_cnt);

  const int n_save = niter / thin;
  arma::vec eta1_out(n_save, arma::fill::zeros);
  arma::vec eta2_out(n_save, arma::fill::zeros);
  arma::vec alpha_out(n_save, arma::fill::zeros);
  arma::vec sigmasq_out(n_save, arma::fill::zeros);

  Rcpp::List beta_out_list = store_beta ? Rcpp::List(n_save) : Rcpp::List();
  Rcpp::List gamma_out_list = store_gamma ? Rcpp::List(n_save) : Rcpp::List();
  Rcpp::List Z_list = store_Z_list ? Rcpp::List(n_save) : Rcpp::List();

  std::vector<int> edge_r;
  std::vector<int> edge_c;
  edge_r.reserve(std::max(1000, n_edges));
  edge_c.reserve(std::max(1000, n_edges));

  const arma::uword *col_ptr = X.col_ptrs;
  const arma::uword *row_idx = X.row_indices;
  const double *xvals = X.values;

  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {
    if (iter > 0 && (iter % 2000) == 0)
      Rcpp::checkUserInterrupt();

    const double sd_sig = std::sqrt(sigmasq);

    ggm_column_sweep_sparse(S, Z_active, p, log_pii, log_1pii, lv0h, lv1h, iv0,
                            iv1, A_sub, s_ggm, noise_ggm, n_edges);

    {
      arma::vec lin = alpha + Xb;
      for (int i = 0; i < n; ++i)
        omega_pg(i) = sample_pg(lin(i), pg_rng);
    }

    const int p_act = static_cast<int>(active_idx.size());
    bool beta_refreshed = false;
    if (p_act > 0) {
      arma::mat X_act(n, p_act, arma::fill::zeros);
      for (int k = 0; k < p_act; ++k) {
        int j = active_idx[k];
        for (arma::sp_mat::const_col_iterator it = X.begin_col(j);
             it != X.end_col(j); ++it) {
          X_act(it.row(), k) = (*it);
        }
      }

      arma::mat Xt_Om = X_act.t();
      Xt_Om.each_row() %= omega_pg.t();
      arma::mat prec = Xt_Om * X_act;
      prec.diag() += 1.0 / sigmasq;

      arma::vec z_star = kappa - omega_pg * alpha;
      arma::vec rhs = X_act.t() * z_star;

      arma::mat L;
      bool chol_ok = arma::chol(L, arma::symmatu(prec));
      if (!chol_ok) {
        double jit = 1e-8;
        for (int att = 0; att < 5 && !chol_ok; ++att) {
          prec.diag() += jit;
          chol_ok = arma::chol(L, arma::symmatu(prec));
          jit *= 10.0;
        }
      }

      if (chol_ok) {
        arma::vec mb;
        bool solve_ok = arma::solve(mb, arma::trimatl(L.t()), rhs,
                                    arma::solve_opts::fast +
                                        arma::solve_opts::no_approx);
        if (solve_ok) {
          solve_ok = arma::solve(mb, arma::trimatu(L), mb,
                                 arma::solve_opts::fast +
                                     arma::solve_opts::no_approx);
        }

        if (solve_ok) {
          arma::vec zz = arma::randn<arma::vec>(p_act);
          arma::vec pert;
          solve_ok = arma::solve(pert, arma::trimatu(L), zz,
                                 arma::solve_opts::fast +
                                     arma::solve_opts::no_approx);
          if (solve_ok) {
            arma::vec bd = mb + pert;
            beta_vec.zeros();
            for (int k = 0; k < p_act; ++k)
              beta_vec(active_idx[k]) = bd(k);
            Xb = X_act * bd;
            beta_refreshed = true;
          }
        }
      }
    } else {
      beta_vec.zeros();
      Xb.zeros();
      beta_refreshed = true;
    }

    if (!beta_refreshed) {
      // Keep current Xb as-is when PG beta solve fails and beta is unchanged.
      // This avoids an unnecessary O(nnz(X)) sparse mat-vec multiply.
    }

    {
      double sum_om = arma::accu(omega_pg);
      double prec_a = sum_om + 1.0 / (h * sigmasq);
      double var_a = 1.0 / prec_a;
      arma::vec resid = kappa - omega_pg % Xb;
      alpha = R::rnorm(var_a * (arma::accu(resid) + alpha0 / (h * sigmasq)),
                       std::sqrt(var_a));
    }

    for (int mh = 0; mh < n_mh_gamma; ++mh) {
      int j = static_cast<int>(std::floor(R::runif(0.0, static_cast<double>(p))));
      if (j >= p)
        j = p - 1;

      int g_curr = static_cast<int>(gamma[j]);
      int g_prop = 1 - g_curr;
      double b_curr = beta_vec(j);
      double b_prop = (g_prop == 1) ? R::rnorm(beta0, sd_sig) : 0.0;
      double db = b_prop - b_curr;

      double ll_diff = column_ll_diff(y01, Xb, alpha, col_ptr, row_idx, xvals,
                                      j, db);
      double neigh_dyn = 0.0;
      for (int nbr : Z_active[j])
        neigh_dyn += static_cast<int>(gamma[nbr]);
      double neigh_fix = 0.0;
      for (int nbr : R_fix.adj[j])
        neigh_fix += static_cast<int>(gamma[nbr]);

      double ising = static_cast<double>(g_prop - g_curr) *
                     (mu + eta1 * neigh_dyn + eta2 * neigh_fix);
      double lmh = ll_diff + ising;

      if (std::log(R::runif(0.0, 1.0)) < lmh) {
        gamma[j] = static_cast<uint8_t>(g_prop);
        beta_vec(j) = b_prop;
        apply_column_update(Xb, col_ptr, row_idx, xvals, j, db);
        if (g_prop == 1)
          activate_gamma(j, active_idx, active_pos);
        else
          deactivate_gamma(j, active_idx, active_pos);
      }
    }

    {
      double sig_prop = std::exp(std::log(sigmasq) + R::rnorm(0.0, 0.2));
      sig_prop = std::max(sig_prop, 1e-10);

      double sh = nu0 / 2.0;
      double sc = sigmasq0 * nu0 / 2.0;
      double lp_c = -(sh + 1.0) * std::log(sigmasq) - sc / sigmasq;
      double lp_p = -(sh + 1.0) * std::log(sig_prop) - sc / sig_prop;

      double ss = 0.0;
      for (int j : active_idx) {
        double d = beta_vec(j) - beta0;
        ss += d * d;
      }

      double n_act = static_cast<double>(active_idx.size());
      double lb_c = -0.5 * n_act * std::log(sigmasq) - 0.5 * ss / sigmasq;
      double lb_p = -0.5 * n_act * std::log(sig_prop) - 0.5 * ss / sig_prop;

      double da = alpha - alpha0;
      double la_c = -0.5 * std::log(h * sigmasq) - 0.5 * da * da / (h * sigmasq);
      double la_p = -0.5 * std::log(h * sig_prop) - 0.5 * da * da / (h * sig_prop);

      double lmh = (lp_p + lb_p + la_p) - (lp_c + lb_c + la_c) +
                   std::log(sig_prop / sigmasq);
      if (std::log(R::runif(0.0, 1.0)) < lmh)
        sigmasq = sig_prop;
    }

    moller_update_dual_sparse(
        Z_active, R_fix.adj, p, mu, eta1, eta2, eta1_sd, eta2_sd, mu_tilde,
        eta1_tilde, eta2_tilde, gamma, e_eta, f_eta, T_max, proposal_type,
        pw_up, pw_dn, pw_om1, pw_om2, pw_om1n, pw_om2n);

    maybe_store_sparse_state(iter, burnin, thin, n_save, store_beta,
                             store_gamma, store_Z_list, active_idx, beta_vec,
                             Z_active, eta1_out, eta2_out, true, alpha_out,
                             sigmasq_out, eta1, eta2, alpha, sigmasq,
                             beta_out_list, gamma_out_list, Z_list, edge_r,
                             edge_c);

    if (iter >= burnin)
      accumulate_sparse_pip(S, Z_active, store_Z_pip, Z_pip_cnt);
  }

  double n_post = static_cast<double>(total_iter - burnin);
  Rcpp::List pip_trip = build_sparse_pip_triplet(S, store_Z_pip, Z_pip_cnt, n_post);

  SEXP beta_out_sexp = store_beta ? static_cast<SEXP>(beta_out_list) : R_NilValue;
  SEXP gamma_out_sexp = store_gamma ? static_cast<SEXP>(gamma_out_list) : R_NilValue;
  SEXP z_list_sexp = store_Z_list ? static_cast<SEXP>(Z_list) : R_NilValue;

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_out_sexp,
      Rcpp::Named("gamma") = gamma_out_sexp,
      Rcpp::Named("eta1") = eta1_out,
      Rcpp::Named("eta2") = eta2_out,
      Rcpp::Named("alpha") = alpha_out,
      Rcpp::Named("sigmasq") = sigmasq_out,
      Rcpp::Named("Z_list") = z_list_sexp,
      Rcpp::Named("Z_pip_row") = pip_trip["row"],
      Rcpp::Named("Z_pip_col") = pip_trip["col"],
      Rcpp::Named("Z_pip_val") = pip_trip["val"],
      Rcpp::Named("p") = p,
      Rcpp::Named("n") = n);
}
