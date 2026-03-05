// Shared numerics helpers for robust linear algebra in MCMC updates.
#ifndef BVS_DADJ_BAYESLOGIT_NUMERICS_H
#define BVS_DADJ_BAYESLOGIT_NUMERICS_H

#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace bvs_dadj {

inline double clamp_finite(double x, double lo, double hi,
                           double fallback = 0.0) {
  if (!std::isfinite(x))
    return fallback;
  if (x < lo)
    return lo;
  if (x > hi)
    return hi;
  return x;
}

inline void sanitize_vec_inplace(arma::vec &v, double lo, double hi,
                                 double fallback = 0.0) {
  for (arma::uword i = 0; i < v.n_elem; ++i)
    v(i) = clamp_finite(v(i), lo, hi, fallback);
}

inline void sanitize_sym_mat_inplace(arma::mat &A, double abs_max = 1e12,
                                     double diag_floor = 1e-10) {
  if (A.n_rows != A.n_cols)
    return;

  const arma::uword p = A.n_rows;
  for (arma::uword i = 0; i < p; ++i) {
    for (arma::uword j = i; j < p; ++j) {
      double aij = clamp_finite(A(i, j), -abs_max, abs_max, 0.0);
      double aji = clamp_finite(A(j, i), -abs_max, abs_max, 0.0);
      double v = 0.5 * (aij + aji);
      A(i, j) = v;
      A(j, i) = v;
    }
    A(i, i) = clamp_finite(A(i, i), diag_floor, abs_max, diag_floor);
  }
}

inline bool robust_chol_inplace(arma::mat &chol_upper, arma::mat &A,
                                int max_attempts = 8,
                                double jitter_init = 1e-10,
                                double jitter_mult = 10.0) {
  if (A.n_rows != A.n_cols)
    return false;

  if (A.n_elem == 0) {
    chol_upper.set_size(0, 0);
    return true;
  }

  // Numerical safety: finite cleanup + symmetry before Cholesky.
  sanitize_sym_mat_inplace(A);

  if (arma::chol(chol_upper, arma::symmatu(A)))
    return true;

  double dscale = arma::mean(arma::abs(A.diag()));
  if (!std::isfinite(dscale) || dscale <= 0.0)
    dscale = 1.0;

  double jitter = std::max(jitter_init, 1e-12 * dscale);
  for (int k = 0; k < max_attempts; ++k) {
    A.diag() += jitter;
    if (arma::chol(chol_upper, arma::symmatu(A)))
      return true;
    jitter *= jitter_mult;
  }

  return false;
}

inline bool robust_chol(arma::mat &chol_upper, const arma::mat &A_in,
                        int max_attempts = 8, double jitter_init = 1e-10,
                        double jitter_mult = 10.0) {
  arma::mat A = A_in;
  return robust_chol_inplace(chol_upper, A, max_attempts, jitter_init,
                             jitter_mult);
}

// -----------------------------------------------------------------------
// Robust Metropolis-Hastings acceptance
//
// Avoids deadlock from log(0) when R::runif returns exactly 0 by using
// the probability form  U < exp(log_ratio)  instead of  log(U) < log_ratio.
// Also guards against NaN and +/-Inf in the log-ratio.
// -----------------------------------------------------------------------
inline bool safe_mh_accept(double log_ratio) {
  if (std::isnan(log_ratio))
    return false; // numerical error → reject
  if (log_ratio >= 0.0)
    return true; // always accept (includes +Inf)
  // Use probability form: avoids log(0) deadlock entirely
  return R::runif(0.0, 1.0) < std::exp(log_ratio);
}

inline bool outcome_is_continuous(const std::string &outcome_type) {
  if (outcome_type == "binary")
    return false;
  if (outcome_type == "continuous")
    return true;
  if (outcome_type == "TTE")
    return false;
  if (outcome_type == "count")
    return false;
  Rcpp::stop(
      "outcome_type must be 'binary', 'continuous', 'TTE', or 'count'.");
  return false;
}

inline bool outcome_is_tte(const std::string &outcome_type) {
  if (outcome_type == "TTE")
    return true;
  if (outcome_type == "binary" || outcome_type == "continuous" ||
      outcome_type == "count")
    return false;
  Rcpp::stop(
      "outcome_type must be 'binary', 'continuous', 'TTE', or 'count'.");
  return false;
}

inline bool outcome_is_count(const std::string &outcome_type) {
  if (outcome_type == "count")
    return true;
  if (outcome_type == "binary" || outcome_type == "continuous" ||
      outcome_type == "TTE")
    return false;
  Rcpp::stop(
      "outcome_type must be 'binary', 'continuous', 'TTE', or 'count'.");
  return false;
}

inline bool normalize_binary_indicator(const arma::vec &x, arma::uvec &x01) {
  const arma::uword n = x.n_elem;
  x01.set_size(n);

  bool ok01 = true, ok11 = true;
  for (arma::uword i = 0; i < n; ++i) {
    const double xi = x(i);
    if (!std::isfinite(xi))
      return false;
    if (std::fabs(xi) > 1e-12 && std::fabs(xi - 1.0) > 1e-12)
      ok01 = false;
    if (std::fabs(xi + 1.0) > 1e-12 && std::fabs(xi - 1.0) > 1e-12)
      ok11 = false;
  }

  if (!ok01 && !ok11)
    return false;

  if (ok01) {
    for (arma::uword i = 0; i < n; ++i)
      x01(i) = (x(i) > 0.5) ? 1u : 0u;
  } else {
    for (arma::uword i = 0; i < n; ++i)
      x01(i) = (x(i) > 0.0) ? 1u : 0u;
  }
  return true;
}

inline bool normalize_count_response(const arma::vec &y, arma::uvec &y_count) {
  const arma::uword n = y.n_elem;
  y_count.set_size(n);
  for (arma::uword i = 0; i < n; ++i) {
    const double yi = y(i);
    if (!std::isfinite(yi))
      return false;
    if (yi < -1e-12)
      return false;
    const double y_round = std::round(yi);
    if (std::fabs(yi - y_round) > 1e-8)
      return false;
    y_count(i) = static_cast<arma::uword>(std::max(0.0, y_round));
  }
  return true;
}

inline double calc_loglik_count_conditional(const arma::uvec &y_count,
                                            const arma::vec &z, double alpha,
                                            const arma::vec &offset,
                                            const arma::vec &log_w) {
  const arma::uword n = y_count.n_elem;
  if (z.n_elem != n || offset.n_elem != n || log_w.n_elem != n)
    Rcpp::stop("Length mismatch in calc_loglik_count_conditional.");
  double ll = 0.0;
  for (arma::uword i = 0; i < n; ++i) {
    const double eta = clamp_finite(alpha + z(i) + offset(i) + log_w(i), -50.0,
                                    50.0, 0.0);
    ll += static_cast<double>(y_count(i)) * eta - std::exp(eta);
  }
  return ll;
}

struct CoxBreslowData {
  arma::uvec order;          // indices sorted by time descending
  arma::uvec event_sorted01; // event indicator in sorted order (0/1)
  std::vector<arma::uword> group_start;
  std::vector<arma::uword> group_end;
};

inline CoxBreslowData build_cox_breslow_data(const arma::vec &time,
                                             const arma::uvec &event01) {
  const arma::uword n = time.n_elem;
  if (event01.n_elem != n)
    Rcpp::stop("time and event must have the same length.");

  CoxBreslowData out;
  out.order = arma::sort_index(time, "descend");
  out.event_sorted01.set_size(n);

  if (n == 0)
    return out;

  arma::vec t_sorted(n);
  for (arma::uword k = 0; k < n; ++k) {
    const arma::uword idx = out.order(k);
    t_sorted(k) = time(idx);
    out.event_sorted01(k) = event01(idx);
  }

  arma::uword start = 0;
  while (start < n) {
    arma::uword end = start;
    const double t0 = t_sorted(start);
    const double tol = 1e-12 * std::max(1.0, std::fabs(t0));
    while (end + 1 < n && std::fabs(t_sorted(end + 1) - t0) <= tol)
      ++end;
    out.group_start.push_back(start);
    out.group_end.push_back(end);
    start = end + 1;
  }
  return out;
}

inline double cox_loglik_breslow(const arma::vec &linpred,
                                 const CoxBreslowData &cox) {
  const arma::uword n = cox.order.n_elem;
  if (linpred.n_elem != n)
    Rcpp::stop("linpred length mismatch in cox_loglik_breslow.");
  if (n == 0)
    return 0.0;

  double ll = 0.0;
  double running_max = -std::numeric_limits<double>::infinity();
  double running_sum_scaled = 0.0; // sum exp(eta - running_max)

  const arma::uword ng = static_cast<arma::uword>(cox.group_start.size());
  for (arma::uword g = 0; g < ng; ++g) {
    const arma::uword gs = cox.group_start[g];
    const arma::uword ge = cox.group_end[g];

    double event_eta_sum = 0.0;
    double d = 0.0;

    for (arma::uword pos = gs; pos <= ge; ++pos) {
      const arma::uword i = cox.order(pos);
      const double eta = clamp_finite(linpred(i), -1e6, 1e6, 0.0);

      if (!std::isfinite(running_max)) {
        running_max = eta;
        running_sum_scaled = 1.0;
      } else if (eta <= running_max) {
        running_sum_scaled += std::exp(eta - running_max);
      } else {
        running_sum_scaled =
            running_sum_scaled * std::exp(running_max - eta) + 1.0;
        running_max = eta;
      }

      if (cox.event_sorted01(pos) != 0u) {
        event_eta_sum += eta;
        d += 1.0;
      }
    }

    if (d > 0.0) {
      const double log_denom =
          running_max + std::log(std::max(running_sum_scaled, 1e-300));
      ll += event_eta_sum - d * log_denom;
    }
  }
  return ll;
}

// -----------------------------------------------------------------------------
// O(nnz + G) Cox Tracker for MCMC Proposals
// -----------------------------------------------------------------------------
class CoxTracker {
private:
  const CoxBreslowData *cox_;
  arma::uword n_;
  arma::uword ng_;

  arma::vec linpred_;
  double max_eta_;
  arma::vec W_;                 // exp(linpred_ - max_eta_)
  std::vector<double> group_W_; // sum of W in each group
  std::vector<double> S_; // cumulative risk sum S_[g] = sum_{h<=g} group_W_[h]
  double ll_;             // current log-likelihood

  std::vector<arma::uword> item_to_group_; // item i -> group g
  std::vector<double> d_g_;                // number of events in each group
  std::vector<double> is_event_;           // 1.0 if event, 0.0 otherwise

public:
  CoxTracker() : cox_(nullptr), n_(0), ng_(0), max_eta_(0), ll_(0) {}

  void init(const arma::vec &linpred, const CoxBreslowData &cox) {
    cox_ = &cox;
    n_ = linpred.n_elem;
    ng_ = cox.group_start.size();
    linpred_ = linpred;

    item_to_group_.assign(n_, 0);
    d_g_.assign(ng_, 0.0);
    is_event_.assign(n_, 0.0);

    for (arma::uword g = 0; g < ng_; ++g) {
      for (arma::uword pos = cox.group_start[g]; pos <= cox.group_end[g];
           ++pos) {
        arma::uword i = cox.order(pos);
        item_to_group_[i] = g;
        if (cox.event_sorted01(pos) != 0u) {
          d_g_[g] += 1.0;
          is_event_[i] = 1.0;
        }
      }
    }

    recompute();
  }

  void recompute() {
    max_eta_ = -std::numeric_limits<double>::infinity();
    for (arma::uword i = 0; i < n_; ++i) {
      double eta = clamp_finite(linpred_(i), -1e6, 1e6, 0.0);
      linpred_(i) = eta;
      if (eta > max_eta_)
        max_eta_ = eta;
    }

    W_.set_size(n_);
    for (arma::uword i = 0; i < n_; ++i) {
      W_(i) = std::exp(linpred_(i) - max_eta_);
    }

    group_W_.assign(ng_, 0.0);
    for (arma::uword g = 0; g < ng_; ++g) {
      for (arma::uword pos = cox_->group_start[g]; pos <= cox_->group_end[g];
           ++pos) {
        group_W_[g] += W_(cox_->order(pos));
      }
    }

    S_.assign(ng_, 0.0);
    double cur_S = 0.0;
    ll_ = 0.0;
    for (arma::uword g = 0; g < ng_; ++g) {
      cur_S += group_W_[g];
      S_[g] = cur_S;

      if (d_g_[g] > 0.0) {
        double event_eta_sum = 0.0;
        for (arma::uword pos = cox_->group_start[g]; pos <= cox_->group_end[g];
             ++pos) {
          if (cox_->event_sorted01(pos) != 0u) {
            event_eta_sum += linpred_(cox_->order(pos));
          }
        }
        ll_ += event_eta_sum -
               d_g_[g] * (max_eta_ + std::log(std::max(cur_S, 1e-300)));
      }
    }
  }

  double get_loglik() const { return ll_; }
  const arma::vec &get_linpred() const { return linpred_; }

  // O(nnz + G) proposal evaluation
  double propose_diff(const arma::uword *col_ptr, const arma::uword *row_idx,
                      const double *xvals, arma::uword j, double db,
                      std::vector<double> &delta_group_W) const {
    arma::uword start = col_ptr[j];
    arma::uword end = col_ptr[j + 1];

    delta_group_W.assign(ng_, 0.0);
    double delta_event_eta_sum = 0.0;

    for (arma::uword idx = start; idx < end; ++idx) {
      arma::uword i = row_idx[idx];
      double val = xvals[idx];
      double shift = db * val;

      arma::uword g = item_to_group_[i];
      delta_group_W[g] += W_(i) * (std::exp(shift) - 1.0);

      if (is_event_[i] > 0.0) {
        delta_event_eta_sum += shift;
      }
    }

    double ll_diff = delta_event_eta_sum;
    double current_delta_S = 0.0;
    for (arma::uword g = 0; g < ng_; ++g) {
      current_delta_S += delta_group_W[g];
      if (d_g_[g] > 0.0) {
        double new_S = S_[g] + current_delta_S;
        ll_diff -= d_g_[g] * (std::log(std::max(new_S, 1e-300)) -
                              std::log(std::max(S_[g], 1e-300)));
      }
    }

    return ll_diff;
  }

  // O(N + G) dense proposal evaluation
  double propose_diff(const arma::vec &xj, double db,
                      std::vector<double> &delta_group_W) const {
    delta_group_W.assign(ng_, 0.0);
    double delta_event_eta_sum = 0.0;

    for (arma::uword i = 0; i < n_; ++i) {
      double shift = db * xj[i];
      if (shift == 0.0)
        continue;

      arma::uword g = item_to_group_[i];
      delta_group_W[g] += W_(i) * (std::exp(shift) - 1.0);

      if (is_event_[i] > 0.0) {
        delta_event_eta_sum += shift;
      }
    }

    double ll_diff = delta_event_eta_sum;
    double current_delta_S = 0.0;
    for (arma::uword g = 0; g < ng_; ++g) {
      current_delta_S += delta_group_W[g];
      if (d_g_[g] > 0.0) {
        double new_S = S_[g] + current_delta_S;
        ll_diff -= d_g_[g] * (std::log(std::max(new_S, 1e-300)) -
                              std::log(std::max(S_[g], 1e-300)));
      }
    }

    return ll_diff;
  }

  // O(nnz + G) state update using previously calculated delta_group_W
  void apply_diff(const arma::uword *col_ptr, const arma::uword *row_idx,
                  const double *xvals, arma::uword j, double db, double ll_diff,
                  const std::vector<double> &delta_group_W) {
    arma::uword start = col_ptr[j];
    arma::uword end = col_ptr[j + 1];

    for (arma::uword idx = start; idx < end; ++idx) {
      arma::uword i = row_idx[idx];
      double shift = db * xvals[idx];

      linpred_(i) += shift;
      W_(i) *=
          std::exp(shift); // equivalently W_(i) = std::exp(linpred_ - max_eta_)
    }

    double current_delta_S = 0.0;
    for (arma::uword g = 0; g < ng_; ++g) {
      group_W_[g] += delta_group_W[g];
      current_delta_S += delta_group_W[g];
      S_[g] += current_delta_S;
    }
    ll_ += ll_diff;
  }

  // O(N + G) dense state update using previously calculated delta_group_W
  void apply_diff(const arma::vec &xj, double db, double ll_diff,
                  const std::vector<double> &delta_group_W) {
    for (arma::uword i = 0; i < n_; ++i) {
      double shift = db * xj[i];
      if (shift == 0.0)
        continue;

      linpred_(i) += shift;
      W_(i) *= std::exp(shift);
    }

    double current_delta_S = 0.0;
    for (arma::uword g = 0; g < ng_; ++g) {
      group_W_[g] += delta_group_W[g];
      current_delta_S += delta_group_W[g];
      S_[g] += current_delta_S;
    }
    ll_ += ll_diff;
  }

  // -----------------------------------------------------------------------
  // Accessors for MALA / Fisher-information step-size computation
  // -----------------------------------------------------------------------

  // Internal W_i = exp(linpred_i - max_eta_) — used for gradient/info calcs
  const arma::vec &get_W() const { return W_; }

  // Compute H_i = sum_{g >= item_to_group_[i]} d_g / S_[g]
  // The Cox score for variable j is:
  //   U_j = sum_i x_ij * (delta_i - W_i * H_i)
  // O(n + G) time.
  void compute_H_vec(arma::vec &H) const {
    H.set_size(n_);
    if (n_ == 0)
      return;
    // Suffix sum over groups: H_suffix[g] = sum_{h=g}^{ng_-1} d_g_[h]/S_[h]
    std::vector<double> H_suffix(ng_ + 1, 0.0);
    for (int g = (int)ng_ - 1; g >= 0; --g)
      H_suffix[g] = H_suffix[g + 1] +
                    (S_[g] > 1e-300 ? d_g_[g] / S_[g] : 0.0);
    for (arma::uword i = 0; i < n_; ++i)
      H[i] = H_suffix[item_to_group_[i]];
  }

  // Approximate Fisher information diagonal for variable j:
  //   I_jj ≈ sum_i x_ij^2 * W_i * H_i
  // H must have been pre-computed via compute_H_vec().  O(n).
  double compute_info_diag_j(const arma::vec &xj,
                             const arma::vec &H) const {
    double info = 0.0;
    for (arma::uword i = 0; i < n_; ++i)
      info += xj[i] * xj[i] * W_(i) * H[i];
    return info;
  }
};

} // namespace bvs_dadj

#endif
