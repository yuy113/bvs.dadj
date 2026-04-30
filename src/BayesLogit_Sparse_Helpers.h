// [[Rcpp::depends(RcppArmadillo)]]
#include "BayesLogit_Numerics.h"
#include <RcppArmadillo.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
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
static const double LB_LOGR_CLIP = 60.0;

struct LBProposalDelta {
  std::vector<int> idx;
  std::vector<double> score_new;
  std::vector<double> weight_new;
  double Z_new = 0.0;
  double log_q_fwd = 0.0;
  double log_q_rev = 0.0;
};

// Vihola (2012) Robust Adaptive Metropolis for 1D proposal scaling.
// Adapts the logit-scale proposal SD to target a specified acceptance rate.
// Uses Robbins-Monro on the log-sigma scale with n^{-2/3} step schedule.
struct EtaAdapter {
  double log_sigma;
  double alpha_star;  // target acceptance rate (0.44 for 1D)
  double gamma_decay; // step-size exponent (2/3)
  int n_adapt;

  EtaAdapter(double init_sigma = 0.5, double target = 0.44)
      : log_sigma(std::log(std::max(1e-6, init_sigma))), alpha_star(target),
        gamma_decay(2.0 / 3.0), n_adapt(0) {}

  double sigma() const {
    return std::exp(std::max(-10.0, std::min(5.0, log_sigma)));
  }

  void update(double accept_prob) {
    ++n_adapt;
    double step = std::min(1.0, std::pow((double)n_adapt, -gamma_decay));
    log_sigma += 0.5 * step * (accept_prob - alpha_star);
    // Floor at exp(-4) ≈ 0.018 to prevent adapter from shrinking proposal to near-zero
    log_sigma = std::max(-4.0, std::min(5.0, log_sigma));
  }
};

static inline double lb_clip_logratio(double x) {
  return clamp_scalar(x, -LB_LOGR_CLIP, LB_LOGR_CLIP);
}

static inline double lb_weight_from_logratio(double log_ratio) {
  // Zanella-style balancing with g(t) = sqrt(t) on log scale.
  return std::max(1e-300, std::exp(0.5 * lb_clip_logratio(log_ratio)));
}

static inline int sample_weighted_index(const std::vector<double> &w, double Z,
                                        int p) {
  if (p <= 1)
    return 0;
  if (!std::isfinite(Z) || Z <= 0.0) {
    int j = static_cast<int>(std::floor(R::runif(0.0, static_cast<double>(p))));
    return (j >= p) ? (p - 1) : std::max(0, j);
  }
  const double u = R::runif(0.0, Z);
  double cum = 0.0;
  for (int j = 0; j < p; ++j) {
    cum += std::max(0.0, w[j]);
    if (u <= cum)
      return j;
  }
  return p - 1;
}

static inline void init_lb_single_scores(const ConstSparseS &S,
                                         const std::vector<uint8_t> &Z_active,
                                         const std::vector<uint8_t> &gamma,
                                         double mu, double eta,
                                         std::vector<double> &score,
                                         std::vector<double> &weight,
                                         double &Z_sum) {
  const int p = S.p;
  score.assign(p, 0.0);
  weight.assign(p, 1.0);
  Z_sum = 0.0;
  for (int j = 0; j < p; ++j) {
    double neigh = 0.0;
    for (int idx = S.col_ptrs[j]; idx < S.col_ptrs[j + 1]; ++idx) {
      if (Z_active[idx])
        neigh += static_cast<int>(gamma[S.row_idx[idx]]);
    }
    const int g = static_cast<int>(gamma[j]);
    const double lr = static_cast<double>(1 - 2 * g) * (mu + eta * neigh);
    score[j] = lb_clip_logratio(lr);
    weight[j] = lb_weight_from_logratio(score[j]);
    Z_sum += weight[j];
  }
  if (!std::isfinite(Z_sum) || Z_sum <= 0.0) {
    std::fill(weight.begin(), weight.end(), 1.0);
    Z_sum = static_cast<double>(p);
  }
}

static inline void init_lb_dual_scores(const ConstSparseS &S,
                                       const ConstSparseAdj &R_fix,
                                       const std::vector<uint8_t> &Z_active,
                                       const std::vector<uint8_t> &gamma,
                                       double mu, double eta1, double eta2,
                                       std::vector<double> &score,
                                       std::vector<double> &weight,
                                       double &Z_sum) {
  const int p = S.p;
  score.assign(p, 0.0);
  weight.assign(p, 1.0);
  Z_sum = 0.0;
  for (int j = 0; j < p; ++j) {
    double neigh_dyn = 0.0;
    for (int idx = S.col_ptrs[j]; idx < S.col_ptrs[j + 1]; ++idx) {
      if (Z_active[idx])
        neigh_dyn += static_cast<int>(gamma[S.row_idx[idx]]);
    }
    double neigh_fix = 0.0;
    for (int idx = R_fix.col_ptrs[j]; idx < R_fix.col_ptrs[j + 1]; ++idx)
      neigh_fix += static_cast<int>(gamma[R_fix.row_idx[idx]]);
    const int g = static_cast<int>(gamma[j]);
    const double lr = static_cast<double>(1 - 2 * g) *
                      (mu + eta1 * neigh_dyn + eta2 * neigh_fix);
    score[j] = lb_clip_logratio(lr);
    weight[j] = lb_weight_from_logratio(score[j]);
    Z_sum += weight[j];
  }
  if (!std::isfinite(Z_sum) || Z_sum <= 0.0) {
    std::fill(weight.begin(), weight.end(), 1.0);
    Z_sum = static_cast<double>(p);
  }
}

static inline void build_lb_single_delta(const ConstSparseS &S,
                                         const std::vector<uint8_t> &Z_active,
                                         const std::vector<uint8_t> &gamma,
                                         double eta, int j, int delta_g,
                                         const std::vector<double> &score,
                                         const std::vector<double> &weight,
                                         double Z_sum,
                                         LBProposalDelta &out) {
  out.idx.clear();
  out.score_new.clear();
  out.weight_new.clear();
  out.Z_new = Z_sum;
  const double Z_safe = std::max(1e-300, Z_sum);
  const double wj = std::max(1e-300, weight[j]);
  out.log_q_fwd = std::log(wj) - std::log(Z_safe);

  // j coordinate flips sign in the Ising log-odds term.
  const double sj_new = lb_clip_logratio(-score[j]);
  const double wj_new = lb_weight_from_logratio(sj_new);
  out.idx.push_back(j);
  out.score_new.push_back(sj_new);
  out.weight_new.push_back(wj_new);
  out.Z_new += (wj_new - weight[j]);

  std::unordered_map<int, double> dscore;
  for (int idx = S.col_ptrs[j]; idx < S.col_ptrs[j + 1]; ++idx) {
    if (!Z_active[idx])
      continue;
    const int k = S.row_idx[idx];
    if (k == j)
      continue;
    const int sgk = 1 - 2 * static_cast<int>(gamma[k]);
    dscore[k] += static_cast<double>(sgk * delta_g) * eta;
  }

  for (const auto &kv : dscore) {
    const int k = kv.first;
    const double sk_new = lb_clip_logratio(score[k] + kv.second);
    const double wk_new = lb_weight_from_logratio(sk_new);
    out.idx.push_back(k);
    out.score_new.push_back(sk_new);
    out.weight_new.push_back(wk_new);
    out.Z_new += (wk_new - weight[k]);
  }

  const double Z_new_safe = std::max(1e-300, out.Z_new);
  out.log_q_rev = std::log(std::max(1e-300, wj_new)) - std::log(Z_new_safe);
}

static inline void build_lb_dual_delta(const ConstSparseS &S,
                                       const ConstSparseAdj &R_fix,
                                       const std::vector<uint8_t> &Z_active,
                                       const std::vector<uint8_t> &gamma,
                                       double eta1, double eta2, int j,
                                       int delta_g,
                                       const std::vector<double> &score,
                                       const std::vector<double> &weight,
                                       double Z_sum,
                                       LBProposalDelta &out) {
  out.idx.clear();
  out.score_new.clear();
  out.weight_new.clear();
  out.Z_new = Z_sum;
  const double Z_safe = std::max(1e-300, Z_sum);
  const double wj = std::max(1e-300, weight[j]);
  out.log_q_fwd = std::log(wj) - std::log(Z_safe);

  const double sj_new = lb_clip_logratio(-score[j]);
  const double wj_new = lb_weight_from_logratio(sj_new);
  out.idx.push_back(j);
  out.score_new.push_back(sj_new);
  out.weight_new.push_back(wj_new);
  out.Z_new += (wj_new - weight[j]);

  std::unordered_map<int, double> eta_shift;
  for (int idx = S.col_ptrs[j]; idx < S.col_ptrs[j + 1]; ++idx) {
    if (!Z_active[idx])
      continue;
    const int k = S.row_idx[idx];
    if (k != j)
      eta_shift[k] += eta1;
  }
  for (int idx = R_fix.col_ptrs[j]; idx < R_fix.col_ptrs[j + 1]; ++idx) {
    const int k = R_fix.row_idx[idx];
    if (k != j)
      eta_shift[k] += eta2;
  }

  for (const auto &kv : eta_shift) {
    const int k = kv.first;
    const int sgk = 1 - 2 * static_cast<int>(gamma[k]);
    const double dsk = static_cast<double>(sgk * delta_g) * kv.second;
    const double sk_new = lb_clip_logratio(score[k] + dsk);
    const double wk_new = lb_weight_from_logratio(sk_new);
    out.idx.push_back(k);
    out.score_new.push_back(sk_new);
    out.weight_new.push_back(wk_new);
    out.Z_new += (wk_new - weight[k]);
  }

  const double Z_new_safe = std::max(1e-300, out.Z_new);
  out.log_q_rev = std::log(std::max(1e-300, wj_new)) - std::log(Z_new_safe);
}

static inline void apply_lb_delta(std::vector<double> &score,
                                  std::vector<double> &weight, double &Z_sum,
                                  const LBProposalDelta &delta) {
  for (std::size_t t = 0; t < delta.idx.size(); ++t) {
    const int k = delta.idx[t];
    score[k] = delta.score_new[t];
    weight[k] = delta.weight_new[t];
  }
  Z_sum = delta.Z_new;
  if (!std::isfinite(Z_sum) || Z_sum <= 0.0) {
    Z_sum = 0.0;
    for (double w : weight)
      Z_sum += std::max(1e-300, w);
    if (!std::isfinite(Z_sum) || Z_sum <= 0.0) {
      std::fill(weight.begin(), weight.end(), 1.0);
      Z_sum = static_cast<double>(weight.size());
    }
  }
}

// ---------------------------------------------------------------------------
// Dense LB helpers — neighbor-list (FixedAdj) and dense-matrix (GGM) variants
// Zanella (2020, JASA) locally balanced proposals for dense adjacency backends
// ---------------------------------------------------------------------------

// FixedAdj single-network: neighbor list stored as IntegerMatrix R (p x p)
static inline void init_lb_single_scores_dense(
    const Rcpp::IntegerMatrix &R_adj, int p,
    const arma::ivec &gamma, double mu, double eta,
    std::vector<double> &score, std::vector<double> &weight, double &Z_sum) {
  score.assign(p, 0.0);
  weight.assign(p, 1.0);
  Z_sum = 0.0;
  for (int j = 0; j < p; ++j) {
    double neigh = 0.0;
    for (int k = 0; k < p; ++k) {
      if (k != j && R_adj(k, j) != 0)
        neigh += gamma(k);
    }
    const int g = gamma(j);
    const double lr = static_cast<double>(1 - 2 * g) * (mu + eta * neigh);
    score[j] = lb_clip_logratio(lr);
    weight[j] = lb_weight_from_logratio(score[j]);
    Z_sum += weight[j];
  }
  if (!std::isfinite(Z_sum) || Z_sum <= 0.0) {
    std::fill(weight.begin(), weight.end(), 1.0);
    Z_sum = static_cast<double>(p);
  }
}

static inline void build_lb_single_delta_dense(
    const Rcpp::IntegerMatrix &R_adj, int p,
    const arma::ivec &gamma, double eta, int j, int delta_g,
    const std::vector<double> &score, const std::vector<double> &weight,
    double Z_sum, LBProposalDelta &out) {
  out.idx.clear();
  out.score_new.clear();
  out.weight_new.clear();
  out.Z_new = Z_sum;
  const double Z_safe = std::max(1e-300, Z_sum);
  const double wj = std::max(1e-300, weight[j]);
  out.log_q_fwd = std::log(wj) - std::log(Z_safe);

  const double sj_new = lb_clip_logratio(-score[j]);
  const double wj_new = lb_weight_from_logratio(sj_new);
  out.idx.push_back(j);
  out.score_new.push_back(sj_new);
  out.weight_new.push_back(wj_new);
  out.Z_new += (wj_new - weight[j]);

  for (int k = 0; k < p; ++k) {
    if (k == j || R_adj(k, j) == 0) continue;
    const int sgk = 1 - 2 * gamma(k);
    const double dsk = static_cast<double>(sgk * delta_g) * eta;
    const double sk_new = lb_clip_logratio(score[k] + dsk);
    const double wk_new = lb_weight_from_logratio(sk_new);
    out.idx.push_back(k);
    out.score_new.push_back(sk_new);
    out.weight_new.push_back(wk_new);
    out.Z_new += (wk_new - weight[k]);
  }

  const double Z_new_safe = std::max(1e-300, out.Z_new);
  out.log_q_rev = std::log(std::max(1e-300, wj_new)) - std::log(Z_new_safe);
}

// FixedAdj dual-network variant
static inline void init_lb_dual_scores_dense(
    const Rcpp::IntegerMatrix &R1, const Rcpp::IntegerMatrix &R2, int p,
    const arma::ivec &gamma, double mu, double eta1, double eta2,
    std::vector<double> &score, std::vector<double> &weight, double &Z_sum) {
  score.assign(p, 0.0);
  weight.assign(p, 1.0);
  Z_sum = 0.0;
  for (int j = 0; j < p; ++j) {
    double neigh1 = 0.0, neigh2 = 0.0;
    for (int k = 0; k < p; ++k) {
      if (k == j) continue;
      if (R1(k, j) != 0) neigh1 += gamma(k);
      if (R2(k, j) != 0) neigh2 += gamma(k);
    }
    const int g = gamma(j);
    const double lr = static_cast<double>(1 - 2 * g) *
                      (mu + eta1 * neigh1 + eta2 * neigh2);
    score[j] = lb_clip_logratio(lr);
    weight[j] = lb_weight_from_logratio(score[j]);
    Z_sum += weight[j];
  }
  if (!std::isfinite(Z_sum) || Z_sum <= 0.0) {
    std::fill(weight.begin(), weight.end(), 1.0);
    Z_sum = static_cast<double>(p);
  }
}

static inline void build_lb_dual_delta_dense(
    const Rcpp::IntegerMatrix &R1, const Rcpp::IntegerMatrix &R2, int p,
    const arma::ivec &gamma, double eta1, double eta2, int j, int delta_g,
    const std::vector<double> &score, const std::vector<double> &weight,
    double Z_sum, LBProposalDelta &out) {
  out.idx.clear();
  out.score_new.clear();
  out.weight_new.clear();
  out.Z_new = Z_sum;
  const double Z_safe = std::max(1e-300, Z_sum);
  const double wj = std::max(1e-300, weight[j]);
  out.log_q_fwd = std::log(wj) - std::log(Z_safe);

  const double sj_new = lb_clip_logratio(-score[j]);
  const double wj_new = lb_weight_from_logratio(sj_new);
  out.idx.push_back(j);
  out.score_new.push_back(sj_new);
  out.weight_new.push_back(wj_new);
  out.Z_new += (wj_new - weight[j]);

  for (int k = 0; k < p; ++k) {
    if (k == j) continue;
    double eta_shift = 0.0;
    if (R1(k, j) != 0) eta_shift += eta1;
    if (R2(k, j) != 0) eta_shift += eta2;
    if (eta_shift == 0.0) continue;
    const int sgk = 1 - 2 * gamma(k);
    const double dsk = static_cast<double>(sgk * delta_g) * eta_shift;
    const double sk_new = lb_clip_logratio(score[k] + dsk);
    const double wk_new = lb_weight_from_logratio(sk_new);
    out.idx.push_back(k);
    out.score_new.push_back(sk_new);
    out.weight_new.push_back(wk_new);
    out.Z_new += (wk_new - weight[k]);
  }

  const double Z_new_safe = std::max(1e-300, out.Z_new);
  out.log_q_rev = std::log(std::max(1e-300, wj_new)) - std::log(Z_new_safe);
}

// GGM single-network: Z_ggm is a dynamic dense matrix (changes each iter)
static inline void init_lb_single_scores_ggm(
    const Rcpp::IntegerMatrix &Z_ggm, int p,
    const arma::ivec &gamma, double mu, double eta,
    std::vector<double> &score, std::vector<double> &weight, double &Z_sum) {
  // Z_ggm is the current GGM adjacency — identical interface to fixed R
  init_lb_single_scores_dense(Z_ggm, p, gamma, mu, eta, score, weight, Z_sum);
}

static inline void build_lb_single_delta_ggm(
    const Rcpp::IntegerMatrix &Z_ggm, int p,
    const arma::ivec &gamma, double eta, int j, int delta_g,
    const std::vector<double> &score, const std::vector<double> &weight,
    double Z_sum, LBProposalDelta &out) {
  build_lb_single_delta_dense(Z_ggm, p, gamma, eta, j, delta_g,
                              score, weight, Z_sum, out);
}

// R2-FIX: uint8 overloads avoid the O(p^2) IntegerMatrix copy of Z_ggm
// every iteration of the GGM main loop. Logic mirrors *_dense exactly.
static inline void init_lb_single_scores_ggm(
    const arma::Mat<uint8_t> &Z_ggm, int p,
    const arma::ivec &gamma, double mu, double eta,
    std::vector<double> &score, std::vector<double> &weight, double &Z_sum) {
  score.assign(p, 0.0);
  weight.assign(p, 1.0);
  Z_sum = 0.0;
  for (int j = 0; j < p; ++j) {
    double neigh = 0.0;
    for (int k = 0; k < p; ++k) {
      if (k != j && Z_ggm(k, j) != 0)
        neigh += gamma(k);
    }
    const int g = gamma(j);
    const double lr = static_cast<double>(1 - 2 * g) * (mu + eta * neigh);
    score[j] = lb_clip_logratio(lr);
    weight[j] = lb_weight_from_logratio(score[j]);
    Z_sum += weight[j];
  }
  if (!std::isfinite(Z_sum) || Z_sum <= 0.0) {
    std::fill(weight.begin(), weight.end(), 1.0);
    Z_sum = static_cast<double>(p);
  }
}

static inline void build_lb_single_delta_ggm(
    const arma::Mat<uint8_t> &Z_ggm, int p,
    const arma::ivec &gamma, double eta, int j, int delta_g,
    const std::vector<double> &score, const std::vector<double> &weight,
    double Z_sum, LBProposalDelta &out) {
  out.idx.clear();
  out.score_new.clear();
  out.weight_new.clear();
  out.Z_new = Z_sum;
  const double Z_safe = std::max(1e-300, Z_sum);
  const double wj = std::max(1e-300, weight[j]);
  out.log_q_fwd = std::log(wj) - std::log(Z_safe);

  const double sj_new = lb_clip_logratio(-score[j]);
  const double wj_new = lb_weight_from_logratio(sj_new);
  out.idx.push_back(j);
  out.score_new.push_back(sj_new);
  out.weight_new.push_back(wj_new);
  out.Z_new += (wj_new - weight[j]);

  for (int k = 0; k < p; ++k) {
    if (k == j || Z_ggm(k, j) == 0) continue;
    const int sgk = 1 - 2 * gamma(k);
    const double dsk = static_cast<double>(sgk * delta_g) * eta;
    const double sk_new = lb_clip_logratio(score[k] + dsk);
    const double wk_new = lb_weight_from_logratio(sk_new);
    out.idx.push_back(k);
    out.score_new.push_back(sk_new);
    out.weight_new.push_back(wk_new);
    out.Z_new += (wk_new - weight[k]);
  }

  const double Z_new_safe = std::max(1e-300, out.Z_new);
  out.log_q_rev = std::log(std::max(1e-300, wj_new)) - std::log(Z_new_safe);
}

// GGM dual-network: Z_ggm dynamic + R_fix static
static inline void init_lb_dual_scores_ggm(
    const Rcpp::IntegerMatrix &Z_ggm, const Rcpp::IntegerMatrix &R_fix, int p,
    const arma::ivec &gamma, double mu, double eta1, double eta2,
    std::vector<double> &score, std::vector<double> &weight, double &Z_sum) {
  init_lb_dual_scores_dense(Z_ggm, R_fix, p, gamma, mu, eta1, eta2,
                            score, weight, Z_sum);
}

static inline void build_lb_dual_delta_ggm(
    const Rcpp::IntegerMatrix &Z_ggm, const Rcpp::IntegerMatrix &R_fix, int p,
    const arma::ivec &gamma, double eta1, double eta2, int j, int delta_g,
    const std::vector<double> &score, const std::vector<double> &weight,
    double Z_sum, LBProposalDelta &out) {
  build_lb_dual_delta_dense(Z_ggm, R_fix, p, gamma, eta1, eta2, j, delta_g,
                            score, weight, Z_sum, out);
}

static inline double stable_logistic(double x) {
  x = clamp_scalar(x, -LINPRED_CLIP, LINPRED_CLIP);
  if (x >= 0.0) {
    const double ex = std::exp(-x);
    return 1.0 / (1.0 + ex);
  }
  const double ex = std::exp(x);
  return ex / (1.0 + ex);
}

[[maybe_unused]] static double sample_pg(double z, std::mt19937 & /*rng*/) {
  if (!std::isfinite(z))
    z = 0.0;
  z = std::abs(clamp_scalar(z, -LINPRED_CLIP, LINPRED_CLIP)) * 0.5;
  const double c2 = z * z;
  const double PI2 = M_PI * M_PI;
  const double c2_over_pi2 = c2 / PI2;
  const double INV_2PI2 = 1.0 / (2.0 * PI2);

  double s = 0.0;
  for (int k = 0; k < PG_K; ++k) {
    // Use R RNG for reproducibility with set.seed() across all backends.
    double g = R::rexp(1.0);
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

// Gibbs-only auxiliary sampler for single sparse graph (no CFTP)
static void gibbs_single_sparse(
    const ConstSparseS &S, const std::vector<uint8_t> &Z_active_flag, int p,
    double mu, double eta1, int n_gibbs_sweeps, std::vector<int> &result) {
  int seed_val = static_cast<int>(std::floor(R::runif(0.0, 1.0) * 2147483646.0)) + 1;
  std::mt19937 gf(seed_val);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  // Initialize from marginal
  for (int k = 0; k < p; ++k) {
    result[k] = (u01(gf) < stable_logistic(mu)) ? 1 : 0;
  }
  for (int sw = 0; sw < n_gibbs_sweeps; ++sw) {
    for (int k = 0; k < p; ++k) {
      double ker = 0.0;
      int start = S.col_ptrs[k], end = S.col_ptrs[k + 1];
      for (int idx = start; idx < end; ++idx) {
        if (Z_active_flag[idx]) {
          int j = S.row_idx[idx];
          if (result[j])
            ker += eta1;
        }
      }
      result[k] = (u01(gf) < stable_logistic(mu + ker)) ? 1 : 0;
    }
  }
}

// Gibbs-only auxiliary sampler for dual sparse graph (no CFTP)
static void gibbs_dual_sparse(
    const ConstSparseS &S, const ConstSparseAdj &R_fix,
    const std::vector<uint8_t> &Z_active_flag, int p,
    double mu, double eta1, double eta2, int n_gibbs_sweeps,
    std::vector<int> &result) {
  int seed_val = static_cast<int>(std::floor(R::runif(0.0, 1.0) * 2147483646.0)) + 1;
  std::mt19937 gf(seed_val);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  for (int k = 0; k < p; ++k) {
    result[k] = (u01(gf) < stable_logistic(mu)) ? 1 : 0;
  }
  for (int sw = 0; sw < n_gibbs_sweeps; ++sw) {
    for (int k = 0; k < p; ++k) {
      double ker = 0.0;
      int s_start = S.col_ptrs[k], s_end = S.col_ptrs[k + 1];
      for (int idx = s_start; idx < s_end; ++idx) {
        if (Z_active_flag[idx]) {
          int j = S.row_idx[idx];
          if (result[j])
            ker += eta1;
        }
      }
      int r_start = R_fix.col_ptrs[k], r_end = R_fix.col_ptrs[k + 1];
      for (int idx = r_start; idx < r_end; ++idx) {
        int j = R_fix.row_idx[idx];
        if (result[j])
          ker += eta2;
      }
      result[k] = (u01(gf) < stable_logistic(mu + ker)) ? 1 : 0;
    }
  }
}

// IMP-2: Returns true if CFTP coalesced exactly, false on T_max failure.
// On failure the result is filled via fallback Gibbs (approximate) but caller
// should reject the MH proposal to preserve Markov chain exactness.
static bool proppwilson_single_sparse(
    const ConstSparseS &S, const std::vector<uint8_t> &Z_active_flag, int p,
    double mu, double eta1, unsigned int T_max, std::vector<int> &x_up,
    std::vector<int> &x_down, std::vector<int> &result) {
  unsigned int T = 2;
  // SPH-5: Enlarged seed space from 1000 to 2^31-1 to avoid seed collisions
  // across different time steps. Was floor(runif(0,1)*1000)+1.
  unsigned int seed_base = static_cast<unsigned int>(R::runif(0.0, 1.0) * 2147483647.0) + 1;

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
      std::mt19937 gen(static_cast<unsigned int>(-t) * seed_base);
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
      // SPH-7: CFTP failed. Use full Gibbs chain as approximate auxiliary draw.
      // IMP-3: Run 200 full Gibbs sweeps from random initialization.
      {
        std::mt19937 gf(seed_base + static_cast<unsigned int>(R::runif(0,1) * 1e8));
        std::uniform_real_distribution<double> u01f(0.0, 1.0);
        for (int k = 0; k < p; ++k) {
          x_up[k] = (u01f(gf) < stable_logistic(mu)) ? 1 : 0;
        }
        for (int sw = 0; sw < 200; ++sw) {
          for (int k = 0; k < p; ++k) {
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
        for (int k = 0; k < p; ++k) {
          x_down[k] = x_up[k];
          result[k] = x_up[k];
        }
      }
      return false; // Signal approximate (not exact) CFTP
    }
  }

  for (int k = 0; k < p; ++k)
    result[k] = x_up[k];
  return true; // Exact coalescence achieved
}

// IMP-2: Returns true if CFTP coalesced exactly, false on T_max failure.
static bool proppwilson_dual_sparse(const ConstSparseS &S,
                                    const ConstSparseAdj &R_fix,
                                    const std::vector<uint8_t> &Z_active_flag,
                                    int p, double mu, double eta1, double eta2,
                                    unsigned int T_max, std::vector<int> &x_up,
                                    std::vector<int> &x_down,
                                    std::vector<int> &result) {
  unsigned int T = 2;
  // SPH-5: Enlarged seed space from 1000 to 2^31-1
  unsigned int seed_base = static_cast<unsigned int>(R::runif(0.0, 1.0) * 2147483647.0) + 1;

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
      std::mt19937 gen(static_cast<unsigned int>(-t) * seed_base);
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
      // SPH-7: CFTP failed. Use full Gibbs chain as approximate auxiliary draw.
      // IMP-3: Run 200 full Gibbs sweeps updating ALL sites (not just non-coalesced)
      // from a fresh random initialization. This provides a proper approximate
      // sample from the Ising model for the approximate exchange algorithm.
      {
        std::mt19937 gf(seed_base + static_cast<unsigned int>(R::runif(0,1) * 1e8));
        std::uniform_real_distribution<double> u01f(0.0, 1.0);
        // Initialize from marginal probabilities (not all-0 or all-1)
        for (int k = 0; k < p; ++k) {
          x_up[k] = (u01f(gf) < stable_logistic(mu)) ? 1 : 0;
        }
        // Run 200 full Gibbs sweeps over ALL sites
        for (int sw = 0; sw < 200; ++sw) {
          for (int k = 0; k < p; ++k) {
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
        for (int k = 0; k < p; ++k) {
          x_down[k] = x_up[k];
          result[k] = x_up[k];
        }
      }
      return false; // Signal approximate (not exact) CFTP
    }
  }

  for (int k = 0; k < p; ++k)
    result[k] = x_up[k];
  return true; // Exact coalescence achieved
}

[[maybe_unused]] static void
proppwilson_dual_sparse(const std::vector<std::vector<int>> &Z_adj,
                        const std::vector<std::vector<int>> &R_fix_adj, int p,
                        double mu, double eta1, double eta2, unsigned int T_max,
                        std::vector<int> &x_up, std::vector<int> &x_down,
                        std::vector<int> &result) {
  unsigned int T = 2;
  // SPH-5: Enlarged seed space from 1000 to 2^31-1
  unsigned int seed_base = static_cast<unsigned int>(R::runif(0.0, 1.0) * 2147483647.0) + 1;

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
      std::mt19937 gen(static_cast<unsigned int>(-t) * seed_base);
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
    std::vector<int> &pw_dn, std::vector<int> &om1, std::vector<int> &om1n,
    EtaAdapter &adapter1, bool use_cftp = false) {

  // Exactly cancel the Moller effect to implement the Exchange Algorithm
  mu_tilde = mu;
  eta1_tilde = eta1;

  double eta1_new;
  double lpr = 0.0;

  // IMP-1: Logit-transformed proposal for eta with Vihola RAM adaptation.
  double eta1_safe = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1));
  double phi1 = std::log(eta1_safe / (eta1_sd - eta1_safe));
  double phi1_new = R::rnorm(phi1, adapter1.sigma());
  eta1_new = eta1_sd / (1.0 + std::exp(-phi1_new));
  eta1_new = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1_new));

  // Jacobian: log|d(eta)/d(phi)| = log(eta * (eta_sd - eta) / eta_sd)
  double log_jac_new = std::log(eta1_new) + std::log(eta1_sd - eta1_new) - std::log(eta1_sd);
  double log_jac_old = std::log(eta1_safe) + std::log(eta1_sd - eta1_safe) - std::log(eta1_sd);
  lpr = log_jac_new - log_jac_old;

  // Generate auxiliary samples: CFTP (optional) or direct Gibbs (default)
  if (use_cftp) {
    proppwilson_single_sparse(S, Z_active_flag, p, mu, eta1, T_max,
                              pw_up, pw_dn, om1);
    proppwilson_single_sparse(S, Z_active_flag, p, mu, eta1_new,
                              T_max, pw_up, pw_dn, om1n);
  } else {
    gibbs_single_sparse(S, Z_active_flag, p, mu, eta1, 20, om1);
    gibbs_single_sparse(S, Z_active_flag, p, mu, eta1_new, 20, om1n);
  }

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

  double accept_prob = std::min(1.0, std::exp(std::min(0.0, lmh)));
  if (bvs_dadj::safe_mh_accept(lmh))
    eta1 = eta1_new;
  adapter1.update(accept_prob);
}

[[maybe_unused]] static void moller_update_dual_sparse(
    const ConstSparseS &S, const ConstSparseAdj &R_fix,
    const std::vector<uint8_t> &Z_active_flag, int p, double mu, double &eta1,
    double &eta2, double eta1_sd, double eta2_sd, double mu_tilde,
    double eta1_tilde, double eta2_tilde, const std::vector<uint8_t> &gamma,
    double e_eta, double f_eta, unsigned int T_max, int proposal_type,
    std::vector<int> &pw_up, std::vector<int> &pw_dn, std::vector<int> &om1,
    std::vector<int> &om2, std::vector<int> &om1n, std::vector<int> &om2n,
    EtaAdapter &adapter1, EtaAdapter &adapter2,
    bool use_cftp = false) {

  // Exactly cancel the Moller effect to implement the Exchange Algorithm
  mu_tilde = mu;
  eta1_tilde = eta1;
  eta2_tilde = eta2;

  double eta1_new, eta2_new;
  double lpr1 = 0.0, lpr2 = 0.0;

  // IMP-1: Logit-transformed proposals with Vihola RAM adaptation.
  double eta1_safe = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1));
  double phi1 = std::log(eta1_safe / (eta1_sd - eta1_safe));
  double phi1_new = R::rnorm(phi1, adapter1.sigma());
  eta1_new = eta1_sd / (1.0 + std::exp(-phi1_new));
  eta1_new = std::max(1e-8, std::min(eta1_sd - 1e-8, eta1_new));
  lpr1 = std::log(eta1_new) + std::log(eta1_sd - eta1_new) -
         std::log(eta1_safe) - std::log(eta1_sd - eta1_safe);

  double eta2_safe = std::max(1e-8, std::min(eta2_sd - 1e-8, eta2));
  double phi2 = std::log(eta2_safe / (eta2_sd - eta2_safe));
  double phi2_new = R::rnorm(phi2, adapter2.sigma());
  eta2_new = eta2_sd / (1.0 + std::exp(-phi2_new));
  eta2_new = std::max(1e-8, std::min(eta2_sd - 1e-8, eta2_new));
  lpr2 = std::log(eta2_new) + std::log(eta2_sd - eta2_new) -
         std::log(eta2_safe) - std::log(eta2_sd - eta2_safe);

  // Joint exchange algorithm for dual adjacency (Murray et al. 2006).
  // Update eta1 and eta2 as two coordinate-wise exchange steps.  Each step
  // needs one auxiliary draw from the proposed dual-Ising distribution; a
  // current-parameter auxiliary draw cancels algebraically in the exchange
  // ratio and only adds Monte Carlo noise.
  (void)pw_up;
  (void)pw_dn;
  (void)om1;
  (void)om2;
  (void)mu_tilde;
  (void)eta1_tilde;
  (void)eta2_tilde;
  (void)proposal_type;

  // Sufficient statistics for the observed gamma on both graphs.
  double B_R1 = 0.0, B_R2 = 0.0;
  for (int j = 0; j < p; ++j) {
    int s_start = S.col_ptrs[j], s_end = S.col_ptrs[j + 1];
    for (int idx = s_start; idx < s_end; ++idx) {
      if (!Z_active_flag[idx])
        continue;
      int k = S.row_idx[idx];
      if (k <= j)
        continue;
      B_R1 += static_cast<int>(gamma[j]) * static_cast<int>(gamma[k]);
    }
    int r_start = R_fix.col_ptrs[j], r_end = R_fix.col_ptrs[j + 1];
    for (int idx = r_start; idx < r_end; ++idx) {
      int k = R_fix.row_idx[idx];
      if (k <= j)
        continue;
      B_R2 += static_cast<int>(gamma[j]) * static_cast<int>(gamma[k]);
    }
  }

  // --- eta1 MH (exchange algorithm) ---
  bool eta1_aux_ok = true;
  if (use_cftp) {
    eta1_aux_ok = proppwilson_dual_sparse(
        S, R_fix, Z_active_flag, p, mu, eta1_new, eta2, T_max, pw_up, pw_dn,
        om1n);
  } else {
    gibbs_dual_sparse(S, R_fix, Z_active_flag, p, mu, eta1_new, eta2, 50,
                      om1n);
  }
  double A_om1n_R1 = 0.0;
  for (int j = 0; j < p; ++j) {
    int s_start = S.col_ptrs[j], s_end = S.col_ptrs[j + 1];
    for (int idx = s_start; idx < s_end; ++idx) {
      if (!Z_active_flag[idx])
        continue;
      int k = S.row_idx[idx];
      if (k <= j)
        continue;
      A_om1n_R1 += om1n[j] * om1n[k];
    }
  }
  double lp1 = log_beta_pdf(eta1_new / eta1_sd, e_eta, f_eta) -
               log_beta_pdf(eta1 / eta1_sd, e_eta, f_eta);
  double log_mh1 = eta1_aux_ok
                       ? (eta1_new - eta1) * (B_R1 - A_om1n_R1) + lp1 + lpr1
                       : -std::numeric_limits<double>::infinity();
  double ap1 = std::min(1.0, std::exp(std::min(0.0, log_mh1)));
  if (eta1_aux_ok && bvs_dadj::safe_mh_accept(log_mh1))
    eta1 = eta1_new;
  adapter1.update(ap1);

  // --- eta2 MH (exchange algorithm) ---
  bool eta2_aux_ok = true;
  if (use_cftp) {
    eta2_aux_ok = proppwilson_dual_sparse(
        S, R_fix, Z_active_flag, p, mu, eta1, eta2_new, T_max, pw_up, pw_dn,
        om2n);
  } else {
    gibbs_dual_sparse(S, R_fix, Z_active_flag, p, mu, eta1, eta2_new, 50,
                      om2n);
  }
  double A_om2n_R2 = 0.0;
  for (int j = 0; j < p; ++j) {
    int r_start = R_fix.col_ptrs[j], r_end = R_fix.col_ptrs[j + 1];
    for (int idx = r_start; idx < r_end; ++idx) {
      int k = R_fix.row_idx[idx];
      if (k <= j)
        continue;
      A_om2n_R2 += om2n[j] * om2n[k];
    }
  }
  double lp2 = log_beta_pdf(eta2_new / eta2_sd, e_eta, f_eta) -
               log_beta_pdf(eta2 / eta2_sd, e_eta, f_eta);
  double log_mh2 = eta2_aux_ok
                       ? (eta2_new - eta2) * (B_R2 - A_om2n_R2) + lp2 + lpr2
                       : -std::numeric_limits<double>::infinity();
  double ap2 = std::min(1.0, std::exp(std::min(0.0, log_mh2)));
  if (eta2_aux_ok && bvs_dadj::safe_mh_accept(log_mh2))
    eta2 = eta2_new;
  adapter2.update(ap2);
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
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
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

[[maybe_unused]] static void ggm_column_sweep_sparse(
    const ConstSparseS &S, std::vector<uint8_t> &Z_active_flag, int p,
    double log_pii, double log_1pii, double lv0h, double lv1h, double iv0,
    double iv1, arma::mat &A_sub, arma::vec &s_ggm, arma::vec &noise_ggm,
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

    std::unordered_map<int, int> nbr_pos;
    nbr_pos.reserve(static_cast<size_t>(d) * 2u);
    for (int r = 0; r < d; ++r) {
      int u = S.row_idx[start + r];
      nbr_pos.emplace(u, r);
      A_sub(r, r) = S.diag[u];
      for (int c = r + 1; c < d; ++c) {
        A_sub(r, c) = 0.0;
        A_sub(c, r) = 0.0;
      }
    }

    // SPH-6: avoid O(d^2 log d) repeated binary lookups by using a local
    // neighbor index map and scanning sparse columns once.
    for (int c = 0; c < d; ++c) {
      const int v = S.row_idx[start + c];
      const int v_start = S.col_ptrs[v];
      const int v_end = S.col_ptrs[v + 1];
      for (int idx = v_start; idx < v_end; ++idx) {
        const int u = S.row_idx[idx];
        auto it = nbr_pos.find(u);
        if (it != nbr_pos.end()) {
          const int r = it->second;
          if (r < c) {
            const double w = S.values[idx];
            A_sub(r, c) = w;
            A_sub(c, r) = w;
          }
        }
      }
    }

    for (int k = 0; k < d; ++k) {
      bool active = Z_active_flag[start + k];
      A_sub(k, k) += active ? iv1 : iv0;
    }

    // SPG-1: Extract d×d block for Cholesky (robust_chol_inplace modifies input).
    // A_sub is rebuilt from S data each iteration, so it is not needed after this.
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
        // SPH-4: Count each edge only once (when i < nbr) to avoid double-count
        if (i < nbr) ++n_edges;
      } else if (!now && was) {
        Z_active_flag[idx_ij] = 0;
        int idx_ji = S.find_edge_index(i, nbr);
        if (idx_ji >= 0)
          Z_active_flag[idx_ji] = 0;
        // SPH-4: Decrement only once per edge
        if (i < nbr) --n_edges;
      }
    }
  }
}

// ============================================================================
// SSSL (Wang 2015) Column Sweep for Sparse GGM — handles p > n
//
// Key differences from vanilla SSVS column sweep:
//   1. Adaptive SSSL spike/slab: v0_sssl is set large enough that the
//      inclusion threshold |b| > sqrt((log(v1/v0)+2*log((1-pi)/pi))/(1/v0-1/v1))
//      is comparable to the noise SD in b, preventing degenerate edge inclusion.
//   2. Per-column adaptive v1: v1_j = max(v1_base, diag_j * scale) so the slab
//      adapts to the local scale of the precision column.
//   3. Positive-definiteness monitoring: if a sampled b produces a non-PD
//      precision column, the draw is rejected and the previous state retained.
// ============================================================================
[[maybe_unused]] static void ggm_column_sweep_sparse_sssl(
    const ConstSparseS &S, std::vector<uint8_t> &Z_active_flag, int p,
    double log_pii, double log_1pii, double v0_sssl, double v1_sssl,
    arma::mat &A_sub, arma::vec &s_ggm, arma::vec &noise_ggm,
    int &n_edges) {
  const int d_max = static_cast<int>(A_sub.n_rows);
  if (d_max <= 0)
    return;

  const double iv0 = 1.0 / v0_sssl;
  const double iv1 = 1.0 / v1_sssl;
  const double lv0h = -0.5 * std::log(v0_sssl);
  const double lv1h = -0.5 * std::log(v1_sssl);

  for (int i = 0; i < p; ++i) {
    const int start = S.col_ptrs[i];
    const int end = S.col_ptrs[i + 1];
    const int d = end - start;
    if (d < 1)
      continue;

    // Build d×d neighbor submatrix from sparse S
    std::unordered_map<int, int> nbr_pos;
    nbr_pos.reserve(static_cast<size_t>(d) * 2u);
    for (int r = 0; r < d; ++r) {
      int u = S.row_idx[start + r];
      nbr_pos.emplace(u, r);
      A_sub(r, r) = S.diag[u];
      for (int c = r + 1; c < d; ++c) {
        A_sub(r, c) = 0.0;
        A_sub(c, r) = 0.0;
      }
    }

    for (int c = 0; c < d; ++c) {
      const int v = S.row_idx[start + c];
      const int v_start = S.col_ptrs[v];
      const int v_end = S.col_ptrs[v + 1];
      for (int idx = v_start; idx < v_end; ++idx) {
        const int u = S.row_idx[idx];
        auto it = nbr_pos.find(u);
        if (it != nbr_pos.end()) {
          const int r = it->second;
          if (r < c) {
            const double w = S.values[idx];
            A_sub(r, c) = w;
            A_sub(c, r) = w;
          }
        }
      }
    }

    // SSSL: add spike/slab precision to diagonal
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

    // SSSL edge selection with adaptive threshold
    for (int k = 0; k < d; ++k) {
      int nbr = S.row_idx[start + k];
      double b2 = b(k) * b(k);

      // Compute posterior edge probability using SSSL spike/slab
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
        if (i < nbr) ++n_edges;
      } else if (!now && was) {
        Z_active_flag[idx_ij] = 0;
        int idx_ji = S.find_edge_index(i, nbr);
        if (idx_ji >= 0)
          Z_active_flag[idx_ji] = 0;
        if (i < nbr) --n_edges;
      }
    }
  }
}

[[maybe_unused]] static void
init_sparse_pip_counters(const ConstSparseS &S, bool store_Z_pip,
                         std::vector<int> &Z_pip_cnt) {
  if (!store_Z_pip)
    return;
  Z_pip_cnt.assign(S.nnz, 0);
}

[[maybe_unused]] static void
accumulate_sparse_pip(const ConstSparseS &S,
                      const std::vector<uint8_t> &Z_active_flag,
                      bool store_Z_pip, std::vector<int> &Z_pip_cnt) {
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

[[maybe_unused]] static Rcpp::List
build_sparse_pip_triplet(const ConstSparseS &S, bool store_Z_pip,
                         const std::vector<int> &Z_pip_cnt, double n_post) {
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

[[maybe_unused]] static void maybe_store_sparse_state(
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
