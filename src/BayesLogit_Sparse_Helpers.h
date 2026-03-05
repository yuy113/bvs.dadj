// [[Rcpp::depends(RcppArmadillo)]]
#include "BayesLogit_Numerics.h"
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

static inline double clamp_scalar(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

static inline bool finite_vec(const arma::vec &x) {
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    if (!std::isfinite(x(i)))
      return false;
  }
  return true;
}

static inline void clamp_vec_inplace(arma::vec &x, double lo, double hi) {
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    x(i) = clamp_scalar(x(i), lo, hi);
  }
}

struct ConstSparseS {
  const int *col_ptrs;
  const int *row_idx;
  const double *values;
  const double *diag;
  int p;
  int nnz;

  ConstSparseS(const Rcpp::IntegerVector &S_i,
               const Rcpp::IntegerVector &S_p_csc,
               const Rcpp::NumericVector &S_x,
               const Rcpp::NumericVector &S_diag, int p_)
      : col_ptrs(S_p_csc.begin()), row_idx(S_i.begin()), values(S_x.begin()),
        diag(S_diag.begin()), p(p_), nnz(S_i.size()) {}

  double lookup(int u, int v) const {
    int start = col_ptrs[v];
    int end = col_ptrs[v + 1];
    auto it = std::lower_bound(row_idx + start, row_idx + end, u);
    if (it != row_idx + end && *it == u) {
      return values[it - row_idx];
    }
    return 0.0;
  }

  int find_edge_index(int u, int v) const {
    int start = col_ptrs[v];
    int end = col_ptrs[v + 1];
    auto it = std::lower_bound(row_idx + start, row_idx + end, u);
    if (it != row_idx + end && *it == u) {
      return static_cast<int>(it - row_idx);
    }
    return -1;
  }
};

struct ConstSparseAdj {
  const int *col_ptrs;
  const int *row_idx;
  int p;
  int nnz;

  ConstSparseAdj(const Rcpp::IntegerVector &R_i,
                 const Rcpp::IntegerVector &R_p_csc, int p_)
      : col_ptrs(R_p_csc.begin()), row_idx(R_i.begin()), p(p_),
        nnz(R_i.size()) {}
};

static const int PG_K = 20;
static const double LINPRED_CLIP = 30.0;
static const double PG_OMEGA_MIN = 1e-6;
static const double SIGMASQ_MIN = 1e-10;
static const double SIGMASQ_MAX = 1e4;
static const double BETA_ABS_MAX = 30.0;

static inline double stable_logistic(double x) {
  x = clamp_scalar(x, -LINPRED_CLIP, LINPRED_CLIP);
  if (x >= 0.0) {
    const double ex = std::exp(-x);
    return 1.0 / (1.0 + ex);
  }
  const double ex = std::exp(x);
  return ex / (1.0 + ex);
}

[[maybe_unused]] static double sample_pg(double z, std::mt19937 &rng) {
  if (!std::isfinite(z))
    z = 0.0;
  z = std::abs(clamp_scalar(z, -LINPRED_CLIP, LINPRED_CLIP)) * 0.5;
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
  return std::max(PG_OMEGA_MIN, s * INV_2PI2);
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
  if (!std::isfinite(psi))
    psi = (psi > 0.0) ? LINPRED_CLIP : -LINPRED_CLIP;
  psi = clamp_scalar(psi, -LINPRED_CLIP, LINPRED_CLIP);
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

static inline double calc_loglik_full_gaussian(const arma::vec &y,
                                               const arma::vec &Xb,
                                               double alpha, double sigmasq) {
  const int n = static_cast<int>(y.n_elem);
  const double log_norm =
      -0.5 * static_cast<double>(n) * std::log(2.0 * M_PI * sigmasq);
  double sse = 0.0;
  for (int i = 0; i < n; ++i) {
    double r = y(i) - (alpha + Xb(i));
    sse += r * r;
  }
  return log_norm - 0.5 * sse / sigmasq;
}

static inline double calc_loglik_full_count(const arma::uvec &y_count,
                                            const arma::vec &Xb, double alpha,
                                            const arma::vec &log_w) {
  const int n = static_cast<int>(y_count.n_elem);
  double ll = 0.0;
  for (int i = 0; i < n; ++i) {
    const double eta =
        clamp_scalar(alpha + Xb(i) + log_w(i), -LINPRED_CLIP, LINPRED_CLIP);
    ll += static_cast<double>(y_count(i)) * eta - std::exp(eta);
  }
  return ll;
}

static inline double column_ll_diff(const arma::vec &y, const arma::vec &Xb,
                                    double alpha, const arma::uword *col_ptr,
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

static inline double column_ll_diff_count(const arma::uvec &y_count,
                                          const arma::vec &Xb, double alpha,
                                          const arma::vec &log_w,
                                          const arma::uword *col_ptr,
                                          const arma::uword *row_idx,
                                          const double *xvals, int j,
                                          double db) {
  if (std::abs(db) < 1e-16)
    return 0.0;

  double ll_diff = 0.0;
  arma::uword start = col_ptr[j];
  arma::uword end = col_ptr[j + 1];
  for (arma::uword k = start; k < end; ++k) {
    int i = static_cast<int>(row_idx[k]);
    double xij = xvals[k];
    const double eta_base = alpha + Xb(i) + log_w(i);
    const double eta_old =
        clamp_scalar(eta_base, -LINPRED_CLIP, LINPRED_CLIP);
    const double eta_new =
        clamp_scalar(eta_base + db * xij, -LINPRED_CLIP, LINPRED_CLIP);
    ll_diff += static_cast<double>(y_count(i)) * (eta_new - eta_old) -
               (std::exp(eta_new) - std::exp(eta_old));
  }
  return ll_diff;
}

static inline void refresh_count_latent_gamma(const arma::uvec &y_count,
                                              const arma::vec &Xb, double alpha,
                                              double nb_shape,
                                              arma::vec &w_count,
                                              arma::vec &log_w) {
  const arma::uword n = y_count.n_elem;
  if (Xb.n_elem != n || w_count.n_elem != n || log_w.n_elem != n)
    Rcpp::stop("Length mismatch in refresh_count_latent_gamma.");
  for (arma::uword i = 0; i < n; ++i) {
    const double eta =
        clamp_scalar(alpha + Xb(i), -LINPRED_CLIP, LINPRED_CLIP);
    const double mu = std::exp(eta);
    const double shape_post = nb_shape + static_cast<double>(y_count(i));
    const double rate_post = nb_shape + mu;
    double wi = R::rgamma(shape_post, 1.0 / rate_post);
    if (!std::isfinite(wi) || wi <= 0.0)
      wi = shape_post / std::max(rate_post, 1e-12);
    w_count(i) = wi;
    log_w(i) = std::log(wi);
  }
}

static inline double
column_ll_diff_gaussian(const arma::vec &y, const arma::vec &Xb, double alpha,
                        const arma::uword *col_ptr, const arma::uword *row_idx,
                        const double *xvals, int j, double db, double sigmasq) {
  if (std::abs(db) < 1e-16)
    return 0.0;

  double ll_diff = 0.0;
  arma::uword start = col_ptr[j];
  arma::uword end = col_ptr[j + 1];
  for (arma::uword k = start; k < end; ++k) {
    int i = static_cast<int>(row_idx[k]);
    double xij = xvals[k];
    double mu_old = alpha + Xb(i);
    double r_old = y(i) - mu_old;
    double r_new = r_old - db * xij;
    ll_diff += -0.5 * (r_new * r_new - r_old * r_old) / sigmasq;
  }
  return ll_diff;
}

static inline double column_ll_diff_gaussian_resid(const arma::vec &resid,
                                                   const double Xj_sq_sum,
                                                   const arma::uword *col_ptr,
                                                   const arma::uword *row_idx,
                                                   const double *xvals, int j,
                                                   double db, double sigmasq) {
  if (std::abs(db) < 1e-16)
    return 0.0;

  double sum_x_r = 0.0;
  arma::uword start = col_ptr[j];
  arma::uword end = col_ptr[j + 1];
  for (arma::uword k = start; k < end; ++k) {
    sum_x_r += resid(row_idx[k]) * xvals[k];
  }
  return -0.5 * (db * db * Xj_sq_sum - 2.0 * db * sum_x_r) / sigmasq;
}

static inline void apply_column_update(arma::vec &Xb,
                                       const arma::uword *col_ptr,
                                       const arma::uword *row_idx,
                                       const double *xvals, int j, double db) {
  if (std::abs(db) < 1e-16)
    return;
  arma::uword start = col_ptr[j];
  arma::uword end = col_ptr[j + 1];
  for (arma::uword k = start; k < end; ++k)
    Xb(static_cast<int>(row_idx[k])) += db * xvals[k];
}

static inline void apply_column_update_resid(arma::vec &resid,
                                             const arma::uword *col_ptr,
                                             const arma::uword *row_idx,
                                             const double *xvals, int j,
                                             double db) {
  if (std::abs(db) < 1e-16)
    return;
  arma::uword start = col_ptr[j];
  arma::uword end = col_ptr[j + 1];
  for (arma::uword k = start; k < end; ++k)
    resid(static_cast<int>(row_idx[k])) -= db * xvals[k];
}

static void proppwilson_single_sparse(
    const ConstSparseS &S, const std::vector<uint8_t> &Z_active_flag, int p,
    double mu, double eta1, unsigned int T_max, std::vector<int> &x_up,
    std::vector<int> &x_down, std::vector<int> &result) {
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
      std::mt19937 gen(-t * seed_base);
      std::uniform_real_distribution<double> u01(0.0, 1.0);

      for (int i = 0; i < p; ++i) {
        double ku = 0.0, kd = 0.0;
        int start = S.col_ptrs[i], end = S.col_ptrs[i + 1];
        for (int idx = start; idx < end; ++idx) {
          if (Z_active_flag[idx]) {
            int j = S.row_idx[idx];
            if (x_up[j])
              ku += eta1;
            if (x_down[j])
              kd += eta1;
          }
        }
        double pu = stable_logistic(mu + ku);
        double pd = stable_logistic(mu + kd);
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
          int start = S.col_ptrs[k], end = S.col_ptrs[k + 1];
          for (int idx = start; idx < end; ++idx) {
            if (Z_active_flag[idx]) {
              int j = S.row_idx[idx];
              if (x_up[j])
                ker += eta1;
            }
          }
          x_up[k] = (u01f(gf) < stable_logistic(mu + ker)) ? 1 : 0;
        }
      }
      for (int k = 0; k < p; ++k)
        x_down[k] = x_up[k];
    }
  }

  for (int k = 0; k < p; ++k)
    result[k] = x_up[k];
}

static void proppwilson_dual_sparse(const ConstSparseS &S,
                                    const ConstSparseAdj &R_fix,
                                    const std::vector<uint8_t> &Z_active_flag,
                                    int p, double mu, double eta1, double eta2,
                                    unsigned int T_max, std::vector<int> &x_up,
                                    std::vector<int> &x_down,
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
      std::mt19937 gen(-t * seed_base);
      std::uniform_real_distribution<double> u01(0.0, 1.0);

      for (int i = 0; i < p; ++i) {
        double ku = 0.0, kd = 0.0;
        int s_start = S.col_ptrs[i], s_end = S.col_ptrs[i + 1];
        for (int idx = s_start; idx < s_end; ++idx) {
          if (Z_active_flag[idx]) {
            int j = S.row_idx[idx];
            if (x_up[j])
              ku += eta1;
            if (x_down[j])
              kd += eta1;
          }
        }
        int r_start = R_fix.col_ptrs[i], r_end = R_fix.col_ptrs[i + 1];
        for (int idx = r_start; idx < r_end; ++idx) {
          int j = R_fix.row_idx[idx];
          if (x_up[j])
            ku += eta2;
          if (x_down[j])
            kd += eta2;
        }

        double pu = stable_logistic(mu + ku);
        double pd = stable_logistic(mu + kd);
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
          int s_start = S.col_ptrs[k], s_end = S.col_ptrs[k + 1];
          for (int idx = s_start; idx < s_end; ++idx) {
            if (Z_active_flag[idx]) {
              int j = S.row_idx[idx];
              if (x_up[j])
                ker += eta1;
            }
          }
          int r_start = R_fix.col_ptrs[k], r_end = R_fix.col_ptrs[k + 1];
          for (int idx = r_start; idx < r_end; ++idx) {
            int j = R_fix.row_idx[idx];
            if (x_up[j])
              ker += eta2;
          }
          x_up[k] = (u01f(gf) < stable_logistic(mu + ker)) ? 1 : 0;
        }
      }
      for (int k = 0; k < p; ++k)
        x_down[k] = x_up[k];
    }
  }

  for (int k = 0; k < p; ++k)
    result[k] = x_up[k];
}

[[maybe_unused]] static void
proppwilson_dual_sparse(const std::vector<std::vector<int>> &Z_adj,
                        const std::vector<std::vector<int>> &R_fix_adj, int p,
                        double mu, double eta1, double eta2, unsigned int T_max,
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

        double pu = stable_logistic(mu + ku);
        double pd = stable_logistic(mu + kd);
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
          x_up[k] = (u01f(gf) < stable_logistic(mu + ker)) ? 1 : 0;
        }
      }
      for (int k = 0; k < p; ++k)
        x_down[k] = x_up[k];
    }
  }

  for (int k = 0; k < p; ++k)
    result[k] = x_up[k];
}

[[maybe_unused]] static void moller_update_single_sparse(
    const ConstSparseS &S, const std::vector<uint8_t> &Z_active_flag, int p,
    double mu, double &eta1, double eta1_sd, double mu_tilde, double eta1_tilde,
    const std::vector<uint8_t> &gamma, double e_eta, double f_eta,
    unsigned int T_max, int proposal_type, std::vector<int> &pw_up,
    std::vector<int> &pw_dn, std::vector<int> &om1, std::vector<int> &om1n) {
  double eta1_new;
  double lpr = 0.0;

  if (proposal_type == 0) {
    double a = std::max(0.0, eta1 - 0.01);
    double b = std::min(eta1_sd, eta1 + 0.01);
    eta1_new = R::runif(a, b);
    double c = std::max(0.0, eta1_new - 0.01);
    double d = std::min(eta1_sd, eta1_new + 0.01);
    lpr = std::log(b - a) - std::log(d - c);
  } else {
    int att = 0;
    do {
      eta1_new = R::rnorm(eta1, eta1_sd);
      if (++att > 10000) {
        eta1_new = eta1;
        break;
      }
    } while (eta1_new <= 0.0 || eta1_new >= eta1_sd);

    double lqf = std::log(normal_pdf(eta1_new, eta1, eta1_sd)) -
                 std::log(normal_cdf(eta1_sd, eta1, eta1_sd) -
                          normal_cdf(0.0, eta1, eta1_sd));
    double lqr = std::log(normal_pdf(eta1, eta1_new, eta1_sd)) -
                 std::log(normal_cdf(eta1_sd, eta1_new, eta1_sd) -
                          normal_cdf(0.0, eta1_new, eta1_sd));
    lpr = lqr - lqf;
  }

  eta1_new = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1_new));

  proppwilson_single_sparse(S, Z_active_flag, p, mu, eta1, T_max, pw_up, pw_dn,
                            om1);
  proppwilson_single_sparse(S, Z_active_flag, p, mu, eta1_new, T_max, pw_up,
                            pw_dn, om1n);

  int sum1 = 0, sum1n = 0, B = 0, A1 = 0, A1n = 0;
  for (int j = 0; j < p; ++j) {
    sum1 += om1[j];
    sum1n += om1n[j];
    int start = S.col_ptrs[j], end = S.col_ptrs[j + 1];
    for (int idx = start; idx < end; ++idx) {
      if (Z_active_flag[idx]) {
        int k = S.row_idx[idx];
        if (k <= j)
          continue;
        B += static_cast<int>(gamma[j]) * static_cast<int>(gamma[k]);
        A1 += om1[j] * om1[k];
        A1n += om1n[j] * om1n[k];
      }
    }
  }

  double lp = log_beta_pdf(eta1_new / eta1_sd, e_eta, f_eta) -
              log_beta_pdf(eta1 / eta1_sd, e_eta, f_eta);
  double lmh = (eta1_new - eta1) * B + lp + mu_tilde * (sum1n - sum1) +
               eta1_tilde * (A1n - A1) + mu * (sum1 - sum1n) + eta1 * A1 -
               eta1_new * A1n + lpr;

  if (bvs_dadj::safe_mh_accept(lmh))
    eta1 = eta1_new;
}

[[maybe_unused]] static void moller_update_dual_sparse(
    const ConstSparseS &S, const ConstSparseAdj &R_fix,
    const std::vector<uint8_t> &Z_active_flag, int p, double mu, double &eta1,
    double &eta2, double eta1_sd, double eta2_sd, double mu_tilde,
    double eta1_tilde, double eta2_tilde, const std::vector<uint8_t> &gamma,
    double e_eta, double f_eta, unsigned int T_max, int proposal_type,
    std::vector<int> &pw_up, std::vector<int> &pw_dn, std::vector<int> &om1,
    std::vector<int> &om2, std::vector<int> &om1n, std::vector<int> &om2n) {
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

  proppwilson_dual_sparse(S, R_fix, Z_active_flag, p, mu, eta1, eta2, T_max,
                          pw_up, pw_dn, om1);
  proppwilson_dual_sparse(S, R_fix, Z_active_flag, p, mu, eta1, eta2, T_max,
                          pw_up, pw_dn, om2);
  proppwilson_dual_sparse(S, R_fix, Z_active_flag, p, mu, eta1_new, eta2, T_max,
                          pw_up, pw_dn, om1n);
  proppwilson_dual_sparse(S, R_fix, Z_active_flag, p, mu, eta1, eta2_new, T_max,
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
    int s_start = S.col_ptrs[j], s_end = S.col_ptrs[j + 1];
    for (int idx = s_start; idx < s_end; ++idx) {
      if (Z_active_flag[idx]) {
        int k = S.row_idx[idx];
        if (k <= j)
          continue;
        B1 += static_cast<int>(gamma[j]) * static_cast<int>(gamma[k]);
        A11 += om1[j] * om1[k];
        A11n += om1n[j] * om1n[k];
        A21 += om2[j] * om2[k];
        A21n += om2n[j] * om2n[k];
      }
    }
  }

  int B2 = 0, A12 = 0, A12n = 0, A22 = 0, A22n = 0;
  for (int j = 0; j < p; ++j) {
    int r_start = R_fix.col_ptrs[j], r_end = R_fix.col_ptrs[j + 1];
    for (int idx = r_start; idx < r_end; ++idx) {
      int k = R_fix.row_idx[idx];
      if (k <= j)
        continue;
      B2 += static_cast<int>(gamma[j]) * static_cast<int>(gamma[k]);
      A12 += om1[j] * om1[k];
      A12n += om1n[j] * om1n[k];
      A22 += om2[j] * om2[k];
      A22n += om2n[j] * om2n[k];
    }
  }

  double lp1 = log_beta_pdf(eta1_new / eta1_sd, e_eta, f_eta) -
               log_beta_pdf(eta1 / eta1_sd, e_eta, f_eta);
  double log_target1 = (eta1_new - eta1) * B1 + lp1;
  double log_aux1 = mu_tilde * (sum1n - sum1) + eta1_tilde * (A11n - A11) +
                    eta2_tilde * (A12n - A12);
  double log_norm1 =
      mu * (sum1 - sum1n) + eta1 * A11 - eta1_new * A11n + eta2 * (A12 - A12n);
  double log_mh1 = log_target1 + log_aux1 + log_norm1 + lpr1;

  double lp2 = log_beta_pdf(eta2_new / eta2_sd, e_eta, f_eta) -
               log_beta_pdf(eta2 / eta2_sd, e_eta, f_eta);
  double log_target2 = (eta2_new - eta2) * B2 + lp2;
  double log_aux2 = mu_tilde * (sum2n - sum2) + eta1_tilde * (A21n - A21) +
                    eta2_tilde * (A22n - A22);
  double log_norm2 =
      mu * (sum2 - sum2n) + eta2 * A22 - eta2_new * A22n + eta1 * (A21 - A21n);
  double log_mh2 = log_target2 + log_aux2 + log_norm2 + lpr2;

  if (bvs_dadj::safe_mh_accept(log_mh1))
    eta1 = eta1_new;
  if (bvs_dadj::safe_mh_accept(log_mh2))
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

static void ggm_column_sweep_sparse(const ConstSparseS &S,
                                    std::vector<uint8_t> &Z_active_flag, int p,
                                    double log_pii, double log_1pii,
                                    double lv0h, double lv1h, double iv0,
                                    double iv1, arma::mat &A_sub,
                                    arma::vec &s_ggm, arma::vec &noise_ggm,
                                    int &n_edges) {
  const int d_max = static_cast<int>(A_sub.n_rows);
  if (d_max <= 0)
    return;

  for (int i = 0; i < p; ++i) {
    const int start = S.col_ptrs[i];
    const int end = S.col_ptrs[i + 1];
    const int d = end - start;
    if (d < 1)
      continue;

    for (int r = 0; r < d; ++r) {
      int u = S.row_idx[start + r];
      A_sub(r, r) = S.diag[u];
      for (int c = r + 1; c < d; ++c) {
        int v = S.row_idx[start + c];
        double w = S.lookup(u, v);
        A_sub(r, c) = w;
        A_sub(c, r) = w;
      }
    }

    for (int k = 0; k < d; ++k) {
      bool active = Z_active_flag[start + k];
      A_sub(k, k) += active ? iv1 : iv0;
    }

    arma::mat A_chol = A_sub.submat(0, 0, d - 1, d - 1);
    arma::mat U;
    bool ok = bvs_dadj::robust_chol_inplace(U, A_chol);
    if (!ok)
      continue;

    for (int k = 0; k < d; ++k)
      s_ggm(k) = S.values[start + k];

    arma::vec neg_s = -s_ggm.head(d);
    arma::vec yt, mu_sub, delta;
    bool solve_ok =
        arma::solve(yt, arma::trimatl(U.t()), neg_s,
                    arma::solve_opts::fast + arma::solve_opts::no_approx);
    if (!solve_ok)
      continue;
    solve_ok =
        arma::solve(mu_sub, arma::trimatu(U), yt,
                    arma::solve_opts::fast + arma::solve_opts::no_approx);
    if (!solve_ok)
      continue;

    for (int k = 0; k < d; ++k)
      noise_ggm(k) = R::rnorm(0.0, 1.0);
    solve_ok =
        arma::solve(delta, arma::trimatu(U), noise_ggm.head(d),
                    arma::solve_opts::fast + arma::solve_opts::no_approx);
    if (!solve_ok)
      continue;

    arma::vec b = mu_sub + delta;
    for (int k = 0; k < d; ++k) {
      int nbr = S.row_idx[start + k];
      double b2 = b(k) * b(k);
      double w1 = lv0h - 0.5 * b2 * iv0 + log_1pii;
      double w2 = lv1h - 0.5 * b2 * iv1 + log_pii;
      double wm = std::max(w1, w2);
      double prob = std::exp(w2 - wm) / (std::exp(w1 - wm) + std::exp(w2 - wm));

      int idx_ij = start + k;
      bool was = Z_active_flag[idx_ij];
      bool now = (R::runif(0.0, 1.0) < prob);
      if (now && !was) {
        Z_active_flag[idx_ij] = 1;
        int idx_ji = S.find_edge_index(i, nbr);
        if (idx_ji >= 0)
          Z_active_flag[idx_ji] = 1;
        ++n_edges;
      } else if (!now && was) {
        Z_active_flag[idx_ij] = 0;
        int idx_ji = S.find_edge_index(i, nbr);
        if (idx_ji >= 0)
          Z_active_flag[idx_ji] = 0;
        --n_edges;
      }
    }
  }
}

static void init_sparse_pip_counters(const ConstSparseS &S, bool store_Z_pip,
                                     std::vector<int> &Z_pip_cnt) {
  if (!store_Z_pip)
    return;
  Z_pip_cnt.assign(S.nnz, 0);
}

static void accumulate_sparse_pip(const ConstSparseS &S,
                                  const std::vector<uint8_t> &Z_active_flag,
                                  bool store_Z_pip,
                                  std::vector<int> &Z_pip_cnt) {
  if (!store_Z_pip)
    return;
  for (int j = 0; j < S.p; ++j) {
    int start = S.col_ptrs[j], end = S.col_ptrs[j + 1];
    for (int idx = start; idx < end; ++idx) {
      int nbr = S.row_idx[idx];
      if (nbr > j && Z_active_flag[idx]) {
        Z_pip_cnt[idx]++;
      }
    }
  }
}

static Rcpp::List build_sparse_pip_triplet(const ConstSparseS &S,
                                           bool store_Z_pip,
                                           const std::vector<int> &Z_pip_cnt,
                                           double n_post) {
  int total = 0;
  if (store_Z_pip) {
    for (int j = 0; j < S.p; ++j) {
      int start = S.col_ptrs[j], end = S.col_ptrs[j + 1];
      for (int idx = start; idx < end; ++idx) {
        if (S.row_idx[idx] > j && Z_pip_cnt[idx] > 0)
          ++total;
      }
    }
  }

  Rcpp::IntegerVector row(total);
  Rcpp::IntegerVector col(total);
  Rcpp::NumericVector val(total);

  if (store_Z_pip) {
    int c = 0;
    for (int j = 0; j < S.p; ++j) {
      int start = S.col_ptrs[j], end = S.col_ptrs[j + 1];
      for (int idx = start; idx < end; ++idx) {
        int nbr = S.row_idx[idx];
        int cnt = Z_pip_cnt[idx];
        if (nbr > j && cnt > 0) {
          row[c] = j;
          col[c] = nbr;
          val[c] = (n_post > 0.0) ? static_cast<double>(cnt) / n_post : 0.0;
          ++c;
        }
      }
    }
  }

  return Rcpp::List::create(Rcpp::Named("row") = row, Rcpp::Named("col") = col,
                            Rcpp::Named("val") = val);
}

static void maybe_store_sparse_state(
    int iter, int burnin, int thin, int n_save, bool store_beta,
    bool store_gamma, bool store_Z_list, const std::vector<int> &active_idx,
    const arma::vec &beta_vec, const ConstSparseS &S,
    const std::vector<uint8_t> &Z_active_flag, arma::vec &eta1_out,
    arma::vec &eta2_out, bool dual_eta, arma::vec &alpha_out,
    arma::vec &sigmasq_out, double eta1, double eta2, double alpha,
    double sigmasq, Rcpp::List &beta_out_list, Rcpp::List &gamma_out_list,
    Rcpp::List &Z_list, std::vector<int> &edge_r, std::vector<int> &edge_c) {
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
        bm(k, 0) = active_idx[k] + 1; // 1-based index in R
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
        gv[k] = active_idx[k] + 1;
      gamma_out_list[s] = gv;
    } else {
      gamma_out_list[s] = Rcpp::IntegerVector(0);
    }
  }

  if (store_Z_list) {
    edge_r.clear();
    edge_c.clear();
    for (int j = 0; j < S.p; ++j) {
      int start = S.col_ptrs[j], end = S.col_ptrs[j + 1];
      for (int idx = start; idx < end; ++idx) {
        int nbr = S.row_idx[idx];
        if (nbr > j && Z_active_flag[idx]) {
          edge_r.push_back(j + 1);
          edge_c.push_back(nbr + 1);
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
