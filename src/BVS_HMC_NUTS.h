// HMC and NUTS samplers for joint beta updates in BVS MCMC.
// Replaces component-wise MALA/Fisher-RW for binary and TTE outcomes.
//
// References:
//   Neal (2011). MCMC using Hamiltonian dynamics. Handbook of MCMC.
//   Hoffman & Gelman (2014). The No-U-Turn Sampler. JMLR 15:1593-1623.
//
// Optimizations over baseline:
//   1. Diagonal mass matrix preconditioning (prior-informed) for better mixing
//   2. O(n*d) Xb recomputation instead of O(n*p) in run_hmc_nuts_joint
//   3. Persistent step-size state across common gamma flips
//   4. Windowed warmup mass adaptation (25, 50, 100, ...)
//   5. Consistent parameter bounds after HMC/NUTS steps
//   6. Move semantics for NUTS tree nodes

#ifndef BVS_HMC_NUTS_H
#define BVS_HMC_NUTS_H

#include "BayesLogit_Numerics.h"
#include "BayesLogit_Imbalanced.h"
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace bvs_hmc {

static constexpr double BVS_HMC_EPS_MIN = 1e-6;
static constexpr double BVS_HMC_EPS_MAX = 10.0;
static constexpr double BVS_HMC_DELTA_MAX = 1000.0;
static constexpr double BVS_HMC_INV_MASS_MIN = 1e-10;
static constexpr double BVS_HMC_INV_MASS_MAX = 1e4;

static inline double clamp_epsilon(double eps) {
  return std::max(BVS_HMC_EPS_MIN, std::min(eps, BVS_HMC_EPS_MAX));
}

static inline double clamp_inv_mass(double v) {
  if (!std::isfinite(v))
    return 1.0;
  return std::max(BVS_HMC_INV_MASS_MIN, std::min(v, BVS_HMC_INV_MASS_MAX));
}

struct HMCSamplingStats {
  double accept_prob;
  bool accepted;
  bool divergent;
  bool hit_max_treedepth;
  int tree_depth;
  double energy_error;

  HMCSamplingStats()
      : accept_prob(0.0), accepted(false), divergent(false),
        hit_max_treedepth(false), tree_depth(0), energy_error(0.0) {}
};

struct HMCNUTSDiagnostics {
  int n_steps;
  int n_accepted;
  int n_divergent;
  int n_max_treedepth;
  double sum_accept_prob;
  double sum_abs_energy_error;
  double max_abs_energy_error;
  double sum_tree_depth;
  std::vector<double> epsilon_history;
  std::vector<double> energy_error_history;

  HMCNUTSDiagnostics()
      : n_steps(0), n_accepted(0), n_divergent(0), n_max_treedepth(0),
        sum_accept_prob(0.0), sum_abs_energy_error(0.0), max_abs_energy_error(0.0),
        sum_tree_depth(0.0) {}

  inline void record(double epsilon_used, const HMCSamplingStats &s) {
    ++n_steps;
    if (s.accepted)
      ++n_accepted;
    if (s.divergent)
      ++n_divergent;
    if (s.hit_max_treedepth)
      ++n_max_treedepth;
    sum_accept_prob += s.accept_prob;
    const double abs_e = std::abs(s.energy_error);
    sum_abs_energy_error += abs_e;
    max_abs_energy_error = std::max(max_abs_energy_error, abs_e);
    sum_tree_depth += static_cast<double>(std::max(0, s.tree_depth));
    epsilon_history.push_back(epsilon_used);
    energy_error_history.push_back(s.energy_error);
  }

  inline Rcpp::List to_list(double final_epsilon) const {
    const double denom = (n_steps > 0) ? static_cast<double>(n_steps) : 1.0;
    return Rcpp::List::create(
        Rcpp::Named("n_steps") = n_steps,
        Rcpp::Named("accept_rate") =
            (n_steps > 0) ? (static_cast<double>(n_accepted) / denom) : NA_REAL,
        Rcpp::Named("mean_accept_prob") =
            (n_steps > 0) ? (sum_accept_prob / denom) : NA_REAL,
        Rcpp::Named("n_divergent") = n_divergent,
        Rcpp::Named("n_max_treedepth") = n_max_treedepth,
        Rcpp::Named("mean_tree_depth") =
            (n_steps > 0) ? (sum_tree_depth / denom) : NA_REAL,
        Rcpp::Named("mean_abs_energy_error") =
            (n_steps > 0) ? (sum_abs_energy_error / denom) : NA_REAL,
        Rcpp::Named("max_abs_energy_error") = max_abs_energy_error,
        Rcpp::Named("final_epsilon") = final_epsilon,
        Rcpp::Named("epsilon_history") = Rcpp::wrap(epsilon_history),
        Rcpp::Named("energy_error_history") = Rcpp::wrap(energy_error_history));
  }
};

static inline double log_sum_exp2(double a, double b) {
  if (!std::isfinite(a))
    return b;
  if (!std::isfinite(b))
    return a;
  const double m = std::max(a, b);
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

// ============================================================================
// Diagonal mass matrix for HMC/NUTS preconditioning
//
// Uses the convention M^{-1} = diag(inv_mass) so that:
//   - Kinetic energy: T(p) = 0.5 * sum(p_i^2 * inv_mass_i)
//   - Position update: q_i += eps * inv_mass_i * p_i
//   - Momentum sampling: p_i ~ N(0, 1/inv_mass_i)
//
// Setting inv_mass_i = Var_post(q_i) equalises oscillation frequencies,
// dramatically improving mixing when parameter scales differ (e.g.,
// beta vs alpha vs tau vs log_sigmasq).
// ============================================================================
struct DiagMassMatrix {
  arma::vec inv_mass;       // M^{-1} diagonal: position-update scale
  arma::vec sqrt_inv_mass;  // sqrt(inv_mass): precomputed for efficiency

  DiagMassMatrix() = default;

  void set_identity(int d) {
    inv_mass.ones(d);
    sqrt_inv_mass.ones(d);
  }

  bool is_valid() const {
    return inv_mass.n_elem > 0 && inv_mass.is_finite();
  }

  // Kinetic energy: T(p) = 0.5 * p^T * M^{-1} * p
  double kinetic_energy(const arma::vec &p) const {
    double ke = 0.0;
    for (arma::uword i = 0; i < p.n_elem; ++i)
      ke += p(i) * p(i) * inv_mass(i);
    return 0.5 * ke;
  }

  // Sample momentum: p ~ N(0, M) where M_ii = 1/inv_mass_i
  void sample_momentum(arma::vec &p) const {
    for (arma::uword i = 0; i < p.n_elem; ++i)
      p(i) = R::rnorm(0.0, 1.0) / sqrt_inv_mass(i);
  }
};

// Build a prior-informed diagonal mass matrix for the joint HMC state.
// q layout for binary: [beta_active(d), alpha(1), tau(ntau), log_sigmasq(1)]
// q layout for TTE:    [beta_active(d), tau(ntau), log_sigmasq(1)]
//
// inv_mass_i ≈ Var_prior(q_i):
//   beta_k:       sigmasq
//   alpha:        h * sigmasq
//   tau_k:        htau * sigmasq
//   log_sigmasq:  trigamma(nu0/2)
static inline void build_prior_mass_matrix(
    DiagMassMatrix &mass, int d, int ntau, bool is_tte,
    double sigmasq, double h, double htau, double nu0) {

  int dim_q = d + ntau + 1 + (is_tte ? 0 : 1);
  if ((int)mass.inv_mass.n_elem != dim_q) {
    mass.inv_mass.set_size(dim_q);
    mass.sqrt_inv_mass.set_size(dim_q);
  }

  // Clamp sigmasq to avoid degenerate mass
  double sig2 = std::max(sigmasq, 1e-8);

  // Beta prior variance
  for (int k = 0; k < d; ++k) {
    mass.inv_mass(k) = sig2;
  }

  if (!is_tte) {
    // Alpha prior variance
    mass.inv_mass(d) = std::max(h * sig2, 1e-8);
    // Tau prior variance
    for (int k = 0; k < ntau; ++k)
      mass.inv_mass(d + 1 + k) = std::max(htau * sig2, 1e-8);
    double nu0_safe = std::max(nu0, 1e-6);
    double log_sig_var = R::trigamma(0.5 * nu0_safe);
    if (!std::isfinite(log_sig_var) || log_sig_var <= 0.0)
      log_sig_var = 2.0 / std::max(nu0_safe, 0.1);
    mass.inv_mass(d + 1 + ntau) = std::max(log_sig_var, 0.01);
  } else {
    // Tau prior variance (no alpha for TTE)
    for (int k = 0; k < ntau; ++k)
      mass.inv_mass(d + k) = std::max(htau * sig2, 1e-8);
    double nu0_safe = std::max(nu0, 1e-6);
    double log_sig_var = R::trigamma(0.5 * nu0_safe);
    if (!std::isfinite(log_sig_var) || log_sig_var <= 0.0)
      log_sig_var = 2.0 / std::max(nu0_safe, 0.1);
    mass.inv_mass(d + ntau) = std::max(log_sig_var, 0.01);
  }

  // Clamp all entries for safety
  for (arma::uword i = 0; i < mass.inv_mass.n_elem; ++i) {
    mass.inv_mass(i) = clamp_inv_mass(mass.inv_mass(i));
    mass.sqrt_inv_mass(i) = std::sqrt(mass.inv_mass(i));
  }
}

// ============================================================================
// Dual averaging for step size adaptation (Hoffman & Gelman, 2014, Alg. 5)
// ============================================================================
struct DualAveraging {
  double mu;              // log(10 * epsilon0)
  double log_epsilon_bar; // smoothed log step size
  double H_bar;           // smoothed acceptance statistic
  double gamma;           // shrinkage toward mu
  double t0;              // early iteration stabilisation
  double kappa;           // step size schedule exponent
  double delta;           // target acceptance rate
  int m;                  // iteration counter

  DualAveraging()
      : mu(0), log_epsilon_bar(0), H_bar(0), gamma(0.05), t0(10.0), kappa(0.75),
        delta(0.65), m(0) {}

  void reset(double epsilon0, double target_accept) {
    mu = std::log(10.0 * epsilon0);
    log_epsilon_bar = std::log(epsilon0);
    H_bar = 0.0;
    delta = target_accept;
    m = 0;
  }

  // Returns adapted epsilon for the *next* iteration.
  double update(double accept_stat) {
    ++m;
    double w = 1.0 / ((double)m + t0);
    H_bar = (1.0 - w) * H_bar + w * (delta - accept_stat);
    double log_eps = mu - std::sqrt((double)m) / gamma * H_bar;
    double mk = std::pow((double)m, -kappa);
    log_epsilon_bar = mk * log_eps + (1.0 - mk) * log_epsilon_bar;
    return clamp_epsilon(std::exp(log_eps));
  }

  double final_epsilon() const { return clamp_epsilon(std::exp(log_epsilon_bar)); }
};

struct OnlineVariance1D {
  int n;
  double mean;
  double m2;

  OnlineVariance1D() : n(0), mean(0.0), m2(0.0) {}

  inline void reset() {
    n = 0;
    mean = 0.0;
    m2 = 0.0;
  }

  inline void add(double x) {
    if (!std::isfinite(x))
      return;
    ++n;
    const double delta = x - mean;
    mean += delta / static_cast<double>(n);
    const double delta2 = x - mean;
    m2 += delta * delta2;
  }

  inline bool has_variance() const { return n > 1; }

  inline double variance() const {
    if (n <= 1)
      return std::numeric_limits<double>::quiet_NaN();
    return m2 / static_cast<double>(n - 1);
  }
};

// Windowed warmup mass adaptation schedule (25, 50, 100, 200, ...).
// Tracks beta moments by variable index so gamma flips do not destroy history.
struct WindowedMassAdapter {
  int p_total;
  int ntau;
  bool has_alpha;
  int warmup_total_steps;
  int warmup_seen_steps;
  int window_size;
  int window_seen;
  bool enabled;

  std::vector<OnlineVariance1D> beta_window;
  std::vector<double> beta_var;
  std::vector<int> beta_n;

  OnlineVariance1D alpha_window;
  double alpha_var;
  int alpha_n;

  std::vector<OnlineVariance1D> tau_window;
  std::vector<double> tau_var;
  std::vector<int> tau_n;

  OnlineVariance1D log_sig_window;
  double log_sig_var;
  int log_sig_n;

  WindowedMassAdapter()
      : p_total(0), ntau(0), has_alpha(false), warmup_total_steps(0),
        warmup_seen_steps(0), window_size(0), window_seen(0), enabled(false),
        alpha_var(0.0), alpha_n(0), log_sig_var(0.0), log_sig_n(0) {}

  inline void reset(int p_total_, int ntau_, bool is_tte, int warmup_steps) {
    p_total = std::max(0, p_total_);
    ntau = std::max(0, ntau_);
    has_alpha = !is_tte;
    warmup_total_steps = std::max(0, warmup_steps);
    warmup_seen_steps = 0;
    window_seen = 0;
    enabled = warmup_total_steps > 0;
    window_size = enabled ? std::min(25, warmup_total_steps) : 0;

    beta_window.assign(p_total, OnlineVariance1D());
    beta_var.assign(p_total, 0.0);
    beta_n.assign(p_total, 0);

    alpha_window.reset();
    alpha_var = 0.0;
    alpha_n = 0;

    tau_window.assign(ntau, OnlineVariance1D());
    tau_var.assign(ntau, 0.0);
    tau_n.assign(ntau, 0);

    log_sig_window.reset();
    log_sig_var = 0.0;
    log_sig_n = 0;
  }

  inline bool is_adapting() const {
    return enabled && (warmup_seen_steps < warmup_total_steps);
  }

  inline void reset_window_accumulators() {
    for (int j = 0; j < p_total; ++j)
      beta_window[j].reset();
    if (has_alpha)
      alpha_window.reset();
    for (int k = 0; k < ntau; ++k)
      tau_window[k].reset();
    log_sig_window.reset();
    window_seen = 0;
  }

  inline void commit_window() {
    for (int j = 0; j < p_total; ++j) {
      if (beta_window[j].has_variance()) {
        beta_var[j] = clamp_inv_mass(beta_window[j].variance());
        beta_n[j] = beta_window[j].n;
      }
    }
    if (has_alpha && alpha_window.has_variance()) {
      alpha_var = clamp_inv_mass(alpha_window.variance());
      alpha_n = alpha_window.n;
    }
    for (int k = 0; k < ntau; ++k) {
      if (tau_window[k].has_variance()) {
        tau_var[k] = clamp_inv_mass(tau_window[k].variance());
        tau_n[k] = tau_window[k].n;
      }
    }
    if (log_sig_window.has_variance()) {
      log_sig_var = clamp_inv_mass(log_sig_window.variance());
      log_sig_n = log_sig_window.n;
    }
  }

  inline void close_window_if_needed() {
    if (!enabled || window_size <= 0 || window_seen < window_size)
      return;

    commit_window();
    reset_window_accumulators();

    const int remaining = warmup_total_steps - warmup_seen_steps;
    if (remaining > 0)
      window_size = std::min(window_size * 2, remaining);
    else
      window_size = 0;
  }

  template <typename IndexVec>
  inline void observe(const IndexVec &active_idx, const arma::vec &beta,
                      double alpha, const arma::vec &tau,
                      double log_sigmasq) {
    if (!is_adapting())
      return;

    for (const auto &j_raw : active_idx) {
      const arma::uword j = static_cast<arma::uword>(j_raw);
      if (j < static_cast<arma::uword>(p_total))
        beta_window[static_cast<int>(j)].add(beta(j));
    }
    if (has_alpha)
      alpha_window.add(alpha);
    for (int k = 0; k < ntau && k < static_cast<int>(tau.n_elem); ++k)
      tau_window[k].add(tau(k));
    log_sig_window.add(log_sigmasq);

    ++warmup_seen_steps;
    ++window_seen;
    close_window_if_needed();

    if (!is_adapting() && window_seen > 0) {
      // Flush the final partial window at warmup end.
      commit_window();
      reset_window_accumulators();
    }
  }

  template <typename IndexVec>
  inline void build_joint_mass_matrix(DiagMassMatrix &mass,
                                      const IndexVec &active_idx,
                                      double sigmasq, double h, double htau,
                                      double nu0) const {
    const int d = static_cast<int>(active_idx.size());
    build_prior_mass_matrix(mass, d, ntau, !has_alpha, sigmasq, h, htau, nu0);

    for (int k = 0; k < d; ++k) {
      const arma::uword j =
          static_cast<arma::uword>(active_idx[static_cast<arma::uword>(k)]);
      if (j < static_cast<arma::uword>(beta_n.size()) && beta_n[j] > 1)
        mass.inv_mass(k) = beta_var[j];
    }

    int offset = d;
    if (has_alpha) {
      if (alpha_n > 1)
        mass.inv_mass(offset) = alpha_var;
      ++offset;
    }
    for (int k = 0; k < ntau; ++k) {
      if (k < static_cast<int>(tau_n.size()) && tau_n[k] > 1)
        mass.inv_mass(offset + k) = tau_var[k];
    }
    if (offset + ntau < static_cast<int>(mass.inv_mass.n_elem) && log_sig_n > 1)
      mass.inv_mass(offset + ntau) = log_sig_var;

    for (arma::uword i = 0; i < mass.inv_mass.n_elem; ++i) {
      mass.inv_mass(i) = clamp_inv_mass(mass.inv_mass(i));
      mass.sqrt_inv_mass(i) = std::sqrt(mass.inv_mass(i));
    }
  }
};

static inline bool should_reinit_step_size(bool hmc_eps_initialised, int prev_dim,
                                           int curr_dim, bool in_warmup) {
  if (!hmc_eps_initialised)
    return true;
  if (!in_warmup || prev_dim < 0)
    return false;
  const int dim_jump = std::abs(curr_dim - prev_dim);
  const int large_jump = std::max(10, prev_dim / 2);
  return dim_jump > large_jump;
}

// ============================================================================
// Find a reasonable initial step size (Hoffman & Gelman, 2014, Alg. 4)
// Mass-matrix aware version.
// ============================================================================
template <typename ComputeFn>
static inline void debug_check_gradients_once(const arma::vec &q,
                                              const arma::vec &grad, double nlp,
                                              ComputeFn &compute) {
#ifdef BVS_DEBUG_GRADIENTS
  static thread_local bool checked = false;
  if (checked)
    return;
  checked = true;

  if (!grad.is_finite() || !std::isfinite(nlp) || q.n_elem == 0) {
    Rcpp::Rcout << "[BVS_DEBUG_GRADIENTS] skipped (non-finite baseline)\n";
    return;
  }

  std::vector<arma::uword> idx;
  if (q.n_elem <= 8) {
    idx.reserve(q.n_elem);
    for (arma::uword j = 0; j < q.n_elem; ++j)
      idx.push_back(j);
  } else {
    idx = {0u, q.n_elem / 2u, q.n_elem - 1u};
  }

  const double tol_abs = 1e-4;
  const double tol_rel = 1e-3;
  bool any_warn = false;
  for (arma::uword j : idx) {
    const double h =
        1e-6 * std::max(1.0, std::abs(static_cast<double>(q(j))));
    arma::vec qp = q, qm = q;
    qp(j) += h;
    qm(j) -= h;
    const auto rp = compute(qp);
    const auto rm = compute(qm);
    if (!std::isfinite(rp.first) || !std::isfinite(rm.first))
      continue;
    const double fd = (rp.first - rm.first) / (2.0 * h);
    const double ad = grad(j);
    const double abs_err = std::abs(fd - ad);
    const double rel_err =
        abs_err / std::max(1.0, std::max(std::abs(fd), std::abs(ad)));
    if (abs_err > tol_abs && rel_err > tol_rel) {
      any_warn = true;
      Rcpp::Rcout << "[BVS_DEBUG_GRADIENTS] idx=" << j << " fd=" << fd
                  << " ad=" << ad << " abs_err=" << abs_err
                  << " rel_err=" << rel_err << "\n";
    }
  }
  if (!any_warn)
    Rcpp::Rcout << "[BVS_DEBUG_GRADIENTS] finite-difference check passed\n";
#else
  (void)q;
  (void)grad;
  (void)nlp;
  (void)compute;
#endif
}

template <typename ComputeFn>
static inline double find_reasonable_epsilon(const arma::vec &q,
                                             const arma::vec &grad, double nlp,
                                             ComputeFn &compute,
                                             const DiagMassMatrix &mass) {
  const int d = (int)q.n_elem;
  if (d == 0)
    return 0.1;

  if (!grad.is_finite() || !std::isfinite(nlp))
    return clamp_epsilon(1.0);

  debug_check_gradients_once(q, grad, nlp, compute);

  double epsilon = 1.0;

  // Sample test momentum ~ N(0, M)
  arma::vec r(d);
  mass.sample_momentum(r);

  // One leapfrog step with mass matrix
  arma::vec q1 = q;
  arma::vec r1 = r;
  r1 -= 0.5 * epsilon * grad;
  for (int i = 0; i < d; ++i)
    q1(i) += epsilon * mass.inv_mass(i) * r1(i);
  auto res1 = compute(q1);
  r1 -= 0.5 * epsilon * res1.second;

  double H0 = nlp + mass.kinetic_energy(r);
  double H1 = res1.first + mass.kinetic_energy(r1);
  double log_ratio = -(H1 - H0);

  int a = (log_ratio > std::log(0.5)) ? 1 : -1;

  for (int count = 0; count < 50; ++count) {
    if ((a == 1 && log_ratio <= std::log(0.5)) ||
        (a == -1 && log_ratio >= std::log(0.5)))
      break;

    epsilon *= std::pow(2.0, (double)a);

    r1 = r;
    q1 = q;
    r1 -= 0.5 * epsilon * grad;
    for (int i = 0; i < d; ++i)
      q1(i) += epsilon * mass.inv_mass(i) * r1(i);
    auto res2 = compute(q1);
    r1 -= 0.5 * epsilon * res2.second;

    H1 = res2.first + mass.kinetic_energy(r1);
    log_ratio = -(H1 - H0);
    if (!std::isfinite(log_ratio)) {
      epsilon *= 0.5;
      break;
    }
  }

  return clamp_epsilon(epsilon);
}

// Backward-compatible overload without mass matrix
template <typename ComputeFn>
static inline double find_reasonable_epsilon(const arma::vec &q,
                                             const arma::vec &grad, double nlp,
                                             ComputeFn &compute) {
  DiagMassMatrix mass;
  mass.set_identity((int)q.n_elem);
  return find_reasonable_epsilon(q, grad, nlp, compute, mass);
}

// ============================================================================
// HMC transition (Neal, 2011)
//
// Performs L leapfrog steps with step size epsilon and diagonal mass matrix,
// then accept/reject.
// Returns the acceptance probability (for step-size adaptation).
// q, nlp, grad are updated in-place on acceptance.
// ============================================================================
template <typename ComputeFn>
static inline double hmc_step(arma::vec &q, double &nlp, arma::vec &grad,
                              double epsilon, int L, ComputeFn &compute,
                              const DiagMassMatrix &mass,
                              HMCSamplingStats *stats_out = nullptr) {
  HMCSamplingStats stats;
  const int d = (int)q.n_elem;
  if (d == 0) {
    stats.accept_prob = 1.0;
    stats.accepted = true;
    if (stats_out)
      *stats_out = stats;
    return 1.0;
  }

  const double epsilon_j =
      clamp_epsilon(epsilon * (0.8 + 0.4 * R::runif(0.0, 1.0)));
  const int L_scaled = std::max(1, (int)std::ceil(std::sqrt((double)d)));
  const int L_base = std::max(1, std::max(L, L_scaled));
  const int L_actual =
      1 + (int)std::floor(R::runif(0.0, 2.0 * (double)L_base));
  stats.tree_depth = L_actual;

  // Sample momentum ~ N(0, M)
  arma::vec p(d);
  mass.sample_momentum(p);

  double H_current = nlp + mass.kinetic_energy(p);

  // Leapfrog integration with mass matrix preconditioning
  arma::vec q_prop = q;
  arma::vec p_prop = p;
  arma::vec grad_prop = grad;

  // Half step for momentum
  p_prop -= 0.5 * epsilon_j * grad_prop;

  for (int l = 0; l < L_actual - 1; ++l) {
    // Full position step: q += eps * M^{-1} * p
    for (int i = 0; i < d; ++i)
      q_prop(i) += epsilon_j * mass.inv_mass(i) * p_prop(i);
    auto res = compute(q_prop);
    grad_prop = res.second;
    // Check for numerical issues
    if (!std::isfinite(res.first) || !grad_prop.is_finite()) {
      stats.accept_prob = 0.0;
      stats.accepted = false;
      stats.divergent = true;
      stats.energy_error = std::numeric_limits<double>::infinity();
      if (stats_out)
        *stats_out = stats;
      return 0.0;
    }
    // Full momentum step
    p_prop -= epsilon_j * grad_prop;
  }

  // Final full position step
  for (int i = 0; i < d; ++i)
    q_prop(i) += epsilon_j * mass.inv_mass(i) * p_prop(i);

  // Final half momentum step
  auto res_final = compute(q_prop);
  double nlp_prop = res_final.first;
  grad_prop = res_final.second;
  p_prop -= 0.5 * epsilon_j * grad_prop;

  if (!std::isfinite(nlp_prop) || !grad_prop.is_finite()) {
    stats.accept_prob = 0.0;
    stats.accepted = false;
    stats.divergent = true;
    stats.energy_error = std::numeric_limits<double>::infinity();
    if (stats_out)
      *stats_out = stats;
    return 0.0;
  }

  // Negate momentum for reversibility (not needed for accept ratio)
  double H_prop = nlp_prop + mass.kinetic_energy(p_prop);
  const double energy_error = H_prop - H_current;

  double log_alpha = -(H_prop - H_current);
  double accept_prob = std::min(1.0, std::exp(std::min(log_alpha, 0.0)));
  if (!std::isfinite(accept_prob))
    accept_prob = 0.0;
  const bool divergent = !std::isfinite(energy_error) ||
                         std::abs(energy_error) > BVS_HMC_DELTA_MAX;
  if (divergent)
    accept_prob = 0.0;

  bool accepted = false;
  const double u = std::max(R::runif(0.0, 1.0), std::numeric_limits<double>::min());
  if (!divergent && std::isfinite(log_alpha) &&
      std::log(u) < log_alpha) {
    q = q_prop;
    nlp = nlp_prop;
    grad = grad_prop;
    accepted = true;
  }

  stats.accept_prob = accept_prob;
  stats.accepted = accepted;
  stats.divergent = divergent;
  stats.energy_error = std::abs(energy_error);
  if (stats_out)
    *stats_out = stats;

  return accept_prob;
}

// Backward-compatible overload without mass matrix
template <typename ComputeFn>
static inline double hmc_step(arma::vec &q, double &nlp, arma::vec &grad,
                              double epsilon, int L, ComputeFn &compute,
                              HMCSamplingStats *stats_out = nullptr) {
  DiagMassMatrix mass;
  mass.set_identity((int)q.n_elem);
  return hmc_step(q, nlp, grad, epsilon, L, compute, mass, stats_out);
}

// ============================================================================
// NUTS transition (Hoffman & Gelman, 2014) with multinomial trajectory
// sampling weights (Betancourt, 2017 / modern Stan implementation).
//
// Recursive tree building with multinomial sampling and U-turn criterion.
// Returns average acceptance probability (for step-size adaptation).
// q, nlp, grad are updated in-place.
// ============================================================================

struct NUTSNode {
  arma::vec theta_minus, theta_plus;
  arma::vec r_minus, r_plus;
  arma::vec grad_minus, grad_plus;
  arma::vec theta_sample;
  double nlp_sample;
  arma::vec grad_sample;
  double log_sum_weight;
  bool has_sample;
  bool stop;
  bool divergent;
  double max_abs_energy_error;
  double sum_alpha;
  int n_alpha;

  NUTSNode()
      : nlp_sample(0.0), log_sum_weight(-std::numeric_limits<double>::infinity()),
        has_sample(false), stop(false), divergent(false), max_abs_energy_error(0.0),
        sum_alpha(0.0), n_alpha(0) {}
  NUTSNode(NUTSNode &&) = default;
  NUTSNode &operator=(NUTSNode &&) = default;
  NUTSNode(const NUTSNode &) = default;
  NUTSNode &operator=(const NUTSNode &) = default;
};

template <typename ComputeFn>
static inline NUTSNode nuts_build_tree(const arma::vec &theta, const arma::vec &r,
                                       const arma::vec &grad_at_theta, int v,
                                       int depth, double epsilon, double H0,
                                       double delta_max, ComputeFn &compute,
                                       const DiagMassMatrix &mass) {
  NUTSNode node;

  if (depth == 0) {
    // Base case: one leapfrog step in direction v with mass matrix
    arma::vec r_new = r - 0.5 * (v * epsilon) * grad_at_theta;
    arma::vec theta_new(theta.n_elem);
    for (arma::uword i = 0; i < theta.n_elem; ++i)
      theta_new(i) = theta(i) + (v * epsilon) * mass.inv_mass(i) * r_new(i);
    auto res = compute(theta_new);
    double nlp_new = res.first;
    arma::vec grad_new = std::move(res.second);
    r_new -= 0.5 * (v * epsilon) * grad_new;

    double H_new = nlp_new + mass.kinetic_energy(r_new);
    bool valid = std::isfinite(H_new);
    const double energy_error = H_new - H0;
    // NUTS-FIX: match hmc_step divergence criterion (two-sided |energy_error|).
    // A Hamiltonian trajectory should preserve H exactly; any large deviation
    // (positive or negative) indicates an integrator failure, not a good proposal.
    const bool divergent = !valid || (std::abs(energy_error) > delta_max);

    node.theta_minus = theta_new;
    node.theta_plus = theta_new;
    node.r_minus = r_new;
    node.r_plus = r_new;
    node.grad_minus = grad_new;
    node.grad_plus = grad_new;
    if (valid && !divergent) {
      node.theta_sample = std::move(theta_new);
      node.nlp_sample = nlp_new;
      node.grad_sample = std::move(grad_new);
      node.log_sum_weight = -H_new;
      node.has_sample = true;
    }
    node.stop = divergent;
    node.divergent = divergent;
    node.max_abs_energy_error =
        std::isfinite(energy_error) ? std::abs(energy_error)
                                    : std::numeric_limits<double>::infinity();
    double alpha = valid ? std::min(1.0, std::exp(-(H_new - H0))) : 0.0;
    if (!std::isfinite(alpha))
      alpha = 0.0;
    node.sum_alpha = alpha;
    node.n_alpha = 1;
    return node;
  }

  // Recursion: build first half-tree
  node = nuts_build_tree(theta, r, grad_at_theta, v, depth - 1, epsilon, H0,
                         delta_max, compute, mass);

  if (!node.stop) {
    // Build second half-tree from the boundary
    NUTSNode node2;
    if (v == -1) {
      node2 =
          nuts_build_tree(node.theta_minus, node.r_minus, node.grad_minus,
                          v, depth - 1, epsilon, H0, delta_max, compute, mass);
      node.theta_minus = std::move(node2.theta_minus);
      node.r_minus = std::move(node2.r_minus);
      node.grad_minus = std::move(node2.grad_minus);
    } else {
      node2 =
          nuts_build_tree(node.theta_plus, node.r_plus, node.grad_plus, v,
                          depth - 1, epsilon, H0, delta_max, compute, mass);
      node.theta_plus = std::move(node2.theta_plus);
      node.r_plus = std::move(node2.r_plus);
      node.grad_plus = std::move(node2.grad_plus);
    }

    // Multinomial sample from combined tree using exp(-H) weights.
    const double log_sum = log_sum_exp2(node.log_sum_weight, node2.log_sum_weight);
    if (node2.has_sample && std::isfinite(node2.log_sum_weight) &&
        std::isfinite(log_sum)) {
      const double prob_take_node2 = std::exp(node2.log_sum_weight - log_sum);
      if (!node.has_sample || R::runif(0.0, 1.0) < prob_take_node2) {
        node.theta_sample = std::move(node2.theta_sample);
        node.nlp_sample = node2.nlp_sample;
        node.grad_sample = std::move(node2.grad_sample);
        node.has_sample = true;
      }
    }
    node.log_sum_weight = log_sum;
    node.sum_alpha += node2.sum_alpha;
    node.n_alpha += node2.n_alpha;
    node.divergent = node.divergent || node2.divergent;
    node.max_abs_energy_error =
        std::max(node.max_abs_energy_error, node2.max_abs_energy_error);

    // U-turn check on combined tree using generalised criterion:
    // dot(dtheta, M^{-1} * r) for mass-preconditioned U-turn
    if (!node2.stop) {
      arma::vec dtheta = node.theta_plus - node.theta_minus;
      // Generalised U-turn: check dot(dtheta, M^{-1} * r_{minus/plus})
      double check_minus = 0.0, check_plus = 0.0;
      for (arma::uword i = 0; i < dtheta.n_elem; ++i) {
        check_minus += dtheta(i) * mass.inv_mass(i) * node.r_minus(i);
        check_plus += dtheta(i) * mass.inv_mass(i) * node.r_plus(i);
      }
      if (check_minus < 0 || check_plus < 0) {
        node.stop = true;
      }
    } else {
      node.stop = true;
    }
  }

  return node;
}

template <typename ComputeFn>
static inline double nuts_step(arma::vec &q, double &nlp, arma::vec &grad,
                               double epsilon, int max_treedepth,
                               ComputeFn &compute,
                               const DiagMassMatrix &mass,
                               HMCSamplingStats *stats_out = nullptr) {
  HMCSamplingStats stats;
  const int d = (int)q.n_elem;
  if (d == 0) {
    stats.accept_prob = 1.0;
    stats.accepted = true;
    if (stats_out)
      *stats_out = stats;
    return 1.0;
  }

  const double epsilon_j =
      clamp_epsilon(epsilon * (0.8 + 0.4 * R::runif(0.0, 1.0)));

  // Sample momentum ~ N(0, M)
  arma::vec r(d);
  mass.sample_momentum(r);

  double H0 = nlp + mass.kinetic_energy(r);
  double log_sum_weight = -H0;

  arma::vec theta_minus = q;
  arma::vec theta_plus = q;
  arma::vec r_minus = r;
  arma::vec r_plus = r;
  arma::vec grad_minus = grad;
  arma::vec grad_plus = grad;

  double sum_alpha = 0.0;
  int n_alpha = 0;
  bool accepted = false;
  bool divergent = false;
  bool terminated_early = false;
  int depth_reached = 0;
  double max_abs_energy_error = 0.0;

  for (int j = 0; j < max_treedepth; ++j) {
    depth_reached = j + 1;
    // Choose direction uniformly
    int v = (R::runif(0.0, 1.0) < 0.5) ? -1 : 1;

    NUTSNode tree;
    if (v == -1) {
      tree = nuts_build_tree(theta_minus, r_minus, grad_minus, v, j, epsilon_j,
                             H0, BVS_HMC_DELTA_MAX, compute, mass);
      theta_minus = std::move(tree.theta_minus);
      r_minus = std::move(tree.r_minus);
      grad_minus = std::move(tree.grad_minus);
    } else {
      tree = nuts_build_tree(theta_plus, r_plus, grad_plus, v, j, epsilon_j, H0,
                             BVS_HMC_DELTA_MAX, compute, mass);
      theta_plus = std::move(tree.theta_plus);
      r_plus = std::move(tree.r_plus);
      grad_plus = std::move(tree.grad_plus);
    }

    // Multinomial candidate selection across trajectory states.
    if (tree.has_sample && std::isfinite(tree.log_sum_weight)) {
      const double log_total_new = log_sum_exp2(log_sum_weight, tree.log_sum_weight);
      if (std::isfinite(log_total_new)) {
        const double prob_take_tree = std::exp(tree.log_sum_weight - log_total_new);
        if (R::runif(0.0, 1.0) < prob_take_tree) {
          q = std::move(tree.theta_sample);
          nlp = tree.nlp_sample;
          grad = std::move(tree.grad_sample);
          accepted = true;
        }
        log_sum_weight = log_total_new;
      }
    }

    sum_alpha += tree.sum_alpha;
    n_alpha += tree.n_alpha;
    divergent = divergent || tree.divergent;
    max_abs_energy_error =
        std::max(max_abs_energy_error, tree.max_abs_energy_error);

    // U-turn check on full tree (mass-preconditioned)
    if (tree.stop) {
      terminated_early = true;
      break;
    }
    arma::vec dtheta = theta_plus - theta_minus;
    double check_minus = 0.0, check_plus = 0.0;
    for (int i = 0; i < d; ++i) {
      check_minus += dtheta(i) * mass.inv_mass(i) * r_minus(i);
      check_plus += dtheta(i) * mass.inv_mass(i) * r_plus(i);
    }
    if (check_minus < 0 || check_plus < 0) {
      terminated_early = true;
      break;
    }
  }

  double accept_prob = (n_alpha > 0) ? sum_alpha / (double)n_alpha : 0.0;
  if (!std::isfinite(accept_prob))
    accept_prob = 0.0;

  stats.accept_prob = accept_prob;
  stats.accepted = accepted;
  stats.divergent = divergent;
  stats.hit_max_treedepth =
      (depth_reached >= max_treedepth) && !terminated_early;
  stats.tree_depth = depth_reached;
  stats.energy_error = max_abs_energy_error;
  if (stats_out)
    *stats_out = stats;

  return accept_prob;
}

// Backward-compatible overloads without mass matrix
template <typename ComputeFn>
static inline double nuts_step(arma::vec &q, double &nlp, arma::vec &grad,
                               double epsilon, int max_treedepth,
                               ComputeFn &compute,
                               HMCSamplingStats *stats_out = nullptr) {
  DiagMassMatrix mass;
  mass.set_identity((int)q.n_elem);
  return nuts_step(q, nlp, grad, epsilon, max_treedepth, compute, mass,
                   stats_out);
}

// ============================================================================
// Neg-log-posterior + gradient for BINARY logistic regression
//
// neg_logpost = -sum_i [y_i*eta_i - log(1+exp(eta_i))] + prior
// gradient[k] = -sum_i [(y_i - sigmoid(eta_i)) * X_{i,active[k]}] + prior'
//
// eta_i = alpha + Xb_inactive(i) + sum_k X_{i,active[k]} * q(k)
// ============================================================================
static inline std::pair<double, arma::vec>
nlp_grad_binary(const arma::mat &X, const arma::vec &y,
                const arma::vec &Xb_inactive, double alpha,
                const std::vector<arma::uword> &active_idx, const arma::vec &q,
                double beta0, double sigmasq) {
  const arma::uword n = X.n_rows;
  const int d = (int)active_idx.size();

  // Compute active contribution to linear predictor
  arma::vec eta(n);
  for (arma::uword i = 0; i < n; ++i)
    eta(i) = alpha + Xb_inactive(i);
  for (int k = 0; k < d; ++k) {
    const arma::vec &xk = X.col(active_idx[k]);
    for (arma::uword i = 0; i < n; ++i)
      eta(i) += xk(i) * q(k);
  }

  double nlp = 0.0;
  arma::vec grad(d, arma::fill::zeros);
  arma::vec diff_vec(n, arma::fill::zeros);

  for (arma::uword i = 0; i < n; ++i) {
    double ei = bvs_dadj::clamp_finite(eta(i), -30.0, 30.0, 0.0);
    double pi = 1.0 / (1.0 + std::exp(-ei));
    // neg log-likelihood contribution
    if (ei > 0.0)
      nlp += -y(i) * ei + ei + std::log1p(std::exp(-ei));
    else
      nlp += -y(i) * ei + std::log1p(std::exp(ei));
    double diff = -(y(i) - pi);
    diff_vec(i) = diff;
  }
  for (int k = 0; k < d; ++k) {
    const arma::vec &xk = X.col(active_idx[k]);
    double acc = 0.0;
    for (arma::uword i = 0; i < n; ++i)
      acc += diff_vec(i) * xk(i);
    grad(k) += acc;
  }

  // Prior: 0.5 * (q_k - beta0)^2 / sigmasq
  for (int k = 0; k < d; ++k) {
    double bk = q(k) - beta0;
    nlp += 0.5 * bk * bk / sigmasq;
    grad(k) += bk / sigmasq;
  }

  return {nlp, grad};
}

// ============================================================================
// Neg-log-posterior + gradient for TTE (Cox partial likelihood, Breslow ties)
//
// Uses bvs_dadj::CoxBreslowData for group structure.
// eta_i = Xb_inactive(i) + sum_k X_{i,active[k]} * q(k)
// (alpha is 0 for TTE)
// ============================================================================
static inline std::pair<double, arma::vec>
nlp_grad_tte(const arma::mat &X, const arma::vec &Xb_inactive,
             const bvs_dadj::CoxBreslowData &cox_data,
             const std::vector<arma::uword> &active_idx, const arma::vec &q,
             double beta0, double sigmasq) {
  const arma::uword n = X.n_rows;
  const int d = (int)active_idx.size();
  const arma::uword ng = (arma::uword)cox_data.group_start.size();

  // Compute full linear predictor
  arma::vec linpred(n);
  for (arma::uword i = 0; i < n; ++i)
    linpred(i) = Xb_inactive(i);
  for (int k = 0; k < d; ++k) {
    const arma::vec &xk = X.col(active_idx[k]);
    for (arma::uword i = 0; i < n; ++i)
      linpred(i) += xk(i) * q(k);
  }

  // Compute log-likelihood and gradient via a single pass over groups.
  // Groups are sorted by descending event time; risk set grows as we process.
  double ll = 0.0;
  arma::vec grad(d, arma::fill::zeros);

  // Running risk-set sums (log-sum-exp stable)
  double max_eta = -std::numeric_limits<double>::infinity();
  double W_sum = 0.0;                     // sum exp(eta - max_eta) in risk set
  arma::vec XW_sum(d, arma::fill::zeros); // sum X_k * exp(eta - max_eta)

  for (arma::uword g = 0; g < ng; ++g) {
    double d_g = 0.0;
    arma::vec event_X_sum(d, arma::fill::zeros);
    double event_eta_sum = 0.0;

    for (arma::uword pos = cox_data.group_start[g];
         pos <= cox_data.group_end[g]; ++pos) {
      arma::uword i = cox_data.order(pos);
      double eta_i = bvs_dadj::clamp_finite(linpred(i), -1e6, 1e6, 0.0);

      // Add to risk set with log-sum-exp stability
      if (!std::isfinite(max_eta)) {
        max_eta = eta_i;
        W_sum = 1.0;
        for (int k = 0; k < d; ++k)
          XW_sum(k) = X(i, active_idx[k]);
      } else if (eta_i <= max_eta) {
        double w = std::exp(eta_i - max_eta);
        W_sum += w;
        for (int k = 0; k < d; ++k)
          XW_sum(k) += X(i, active_idx[k]) * w;
      } else {
        double ratio = std::exp(max_eta - eta_i);
        W_sum = W_sum * ratio + 1.0;
        for (int k = 0; k < d; ++k)
          XW_sum(k) = XW_sum(k) * ratio + X(i, active_idx[k]);
        max_eta = eta_i;
      }

      if (cox_data.event_sorted01(pos) != 0u) {
        d_g += 1.0;
        event_eta_sum += eta_i;
        for (int k = 0; k < d; ++k)
          event_X_sum(k) += X(i, active_idx[k]);
      }
    }

    if (d_g > 0.0 && W_sum > 0.0) {
      double log_W = max_eta + std::log(std::max(W_sum, 1e-300));
      ll += event_eta_sum - d_g * log_W;
      for (int k = 0; k < d; ++k)
        grad(k) += -(event_X_sum(k) - d_g * XW_sum(k) / W_sum);
    }
  }

  double nlp = -ll;

  // Prior: 0.5 * (q_k - beta0)^2 / sigmasq
  for (int k = 0; k < d; ++k) {
    double bk = q(k) - beta0;
    nlp += 0.5 * bk * bk / sigmasq;
    grad(k) += bk / sigmasq;
  }

  return {nlp, grad};
}

// ============================================================================
// Neg-log-posterior + gradient for Binary Logistic (Joint)
//
// Updates active beta, alpha, tau, and log_sigmasq jointly
// q layout: [beta_active (d), alpha (1), tau (ntau), log_sigmasq (1)]
// ============================================================================
static inline std::pair<double, arma::vec> nlp_grad_binary_joint(
    const arma::mat &X, const arma::vec &y, const arma::mat &Z_dat,
    const std::vector<arma::uword> &active_idx, const arma::vec &q,
    double beta0, double alpha0, double tau0, double h, double htau, double nu0,
    double sigmasq0) {
  const arma::uword n = y.n_elem;
  const int d = (int)active_idx.size();
  const int ntau = (int)Z_dat.n_cols;

  double alpha = q(d);
  double log_sigmasq = q(d + 1 + ntau);
  // HMC-1: Clamp log_sigmasq during leapfrog to prevent exp() overflow.
  // Without this, log_sigmasq can temporarily exceed 700 during leapfrog
  // integration, causing sigmasq = exp(log_sigmasq) = Inf.
  log_sigmasq = std::max(-23.0, std::min(9.2, log_sigmasq));
  double sigmasq = std::exp(log_sigmasq);

  static thread_local arma::vec eta;
  if (eta.n_elem != n) eta.set_size(n);
  eta.fill(alpha);
  for (int k = 0; k < d; ++k) {
    const arma::vec &xk = X.col(active_idx[k]);
    double b = q(k);
    for (arma::uword i = 0; i < n; ++i)
      eta(i) += xk(i) * b;
  }
  for (int k = 0; k < ntau; ++k) {
    const arma::vec &zk = Z_dat.col(k);
    double t = q(d + 1 + k);
    for (arma::uword i = 0; i < n; ++i)
      eta(i) += zk(i) * t;
  }

  double nlp = 0.0;
  arma::vec diff_vec(n, arma::fill::zeros);
  static thread_local arma::vec grad;
  if (grad.n_elem != q.n_elem) grad.set_size(q.n_elem);
  grad.zeros();

  for (arma::uword i = 0; i < n; ++i) {
    double ei = bvs_dadj::clamp_finite(eta(i), -30.0, 30.0, 0.0);
    double pi = 1.0 / (1.0 + std::exp(-ei));
    if (ei > 0.0)
      nlp += -y(i) * ei + ei + std::log1p(std::exp(-ei));
    else
      nlp += -y(i) * ei + std::log1p(std::exp(ei));

    const double diff = -(y(i) - pi);
    diff_vec(i) = diff;
    grad(d) += diff;
    for (int k = 0; k < ntau; ++k)
      grad(d + 1 + k) += diff * Z_dat(i, k);
  }
  for (int k = 0; k < d; ++k) {
    const arma::vec &xk = X.col(active_idx[k]);
    double acc = 0.0;
    for (arma::uword i = 0; i < n; ++i)
      acc += diff_vec(i) * xk(i);
    grad(k) += acc;
  }

  double sum_beta_sq = 0.0;
  for (int k = 0; k < d; ++k) {
    double bk = q(k) - beta0;
    sum_beta_sq += bk * bk;
    grad(k) += bk / sigmasq;
  }
  nlp += 0.5 * d * log_sigmasq + 0.5 * sum_beta_sq / sigmasq;

  double a_diff = q(d) - alpha0;
  nlp += 0.5 * log_sigmasq + 0.5 * a_diff * a_diff / (h * sigmasq);
  grad(d) += a_diff / (h * sigmasq);

  double sum_tau_sq = 0.0;
  for (int k = 0; k < ntau; ++k) {
    double tk = q(d + 1 + k) - tau0;
    sum_tau_sq += tk * tk;
    grad(d + 1 + k) += tk / (htau * sigmasq);
  }
  nlp += 0.5 * ntau * log_sigmasq + 0.5 * sum_tau_sq / (htau * sigmasq);

  nlp += (nu0 / 2.0) * log_sigmasq + (nu0 * sigmasq0) / (2.0 * sigmasq);

  double grad_q_sig = (nu0 / 2.0) - (nu0 * sigmasq0) / (2.0 * sigmasq);
  grad_q_sig += 0.5 * d - 0.5 * sum_beta_sq / sigmasq;
  grad_q_sig += 0.5 - 0.5 * a_diff * a_diff / (h * sigmasq);
  grad_q_sig += 0.5 * ntau - 0.5 * sum_tau_sq / (htau * sigmasq);
  grad(d + 1 + ntau) = grad_q_sig;

  return {nlp, grad};
}

// ============================================================================
// Neg-log-posterior + gradient for TTE (Joint)
// q layout: [beta_active (d), tau (ntau), log_sigmasq (1)]
// ============================================================================
static inline std::pair<double, arma::vec>
nlp_grad_tte_joint(const arma::mat &X, const arma::mat &Z_dat,
                   const bvs_dadj::CoxBreslowData &cox_data,
                   const std::vector<arma::uword> &active_idx,
                   const arma::vec &q, double beta0, double tau0, double htau,
                   double nu0, double sigmasq0) {
  const arma::uword n = X.n_rows;
  const int d = (int)active_idx.size();
  const int ntau = (int)Z_dat.n_cols;
  const arma::uword ng = (arma::uword)cox_data.group_start.size();

  double log_sigmasq = q(d + ntau);
  // HMC-1: Clamp log_sigmasq during leapfrog
  log_sigmasq = std::max(-23.0, std::min(9.2, log_sigmasq));
  double sigmasq = std::exp(log_sigmasq);

  static thread_local arma::vec linpred;
  if (linpred.n_elem != n)
    linpred.set_size(n);
  linpred.zeros();
  for (int k = 0; k < d; ++k) {
    const arma::vec &xk = X.col(active_idx[k]);
    double b = q(k);
    for (arma::uword i = 0; i < n; ++i)
      linpred(i) += xk(i) * b;
  }
  for (int k = 0; k < ntau; ++k) {
    const arma::vec &zk = Z_dat.col(k);
    double t = q(d + k);
    for (arma::uword i = 0; i < n; ++i)
      linpred(i) += zk(i) * t;
  }

  double max_lp = -std::numeric_limits<double>::infinity();
  for (arma::uword i = 0; i < n; ++i) {
    linpred(i) = bvs_dadj::clamp_finite(linpred(i), -1e6, 1e6, 0.0);
    if (linpred(i) > max_lp)
      max_lp = linpred(i);
  }
  if (!std::isfinite(max_lp))
    max_lp = 0.0;

  static thread_local arma::vec exp_shifted;
  if (exp_shifted.n_elem != n)
    exp_shifted.set_size(n);
  for (arma::uword i = 0; i < n; ++i)
    exp_shifted(i) = std::exp(linpred(i) - max_lp);

  static thread_local arma::vec event_weight;
  if (event_weight.n_elem != n)
    event_weight.set_size(n);
  event_weight.zeros();

  static thread_local arma::vec risk_weight;
  if (risk_weight.n_elem != n)
    risk_weight.set_size(n);
  risk_weight.zeros();

  static thread_local arma::uvec group_of_obs;
  if (group_of_obs.n_elem != n)
    group_of_obs.set_size(n);
  group_of_obs.zeros();

  static thread_local arma::vec group_coef;
  if (group_coef.n_elem != ng)
    group_coef.set_size(ng);
  group_coef.zeros();

  double ll = 0.0;
  double W_sum = 0.0;
  for (arma::uword g = 0; g < ng; ++g) {
    double d_g = 0.0;
    double event_eta_sum = 0.0;
    for (arma::uword pos = cox_data.group_start[g];
         pos <= cox_data.group_end[g]; ++pos) {
      const arma::uword i = cox_data.order(pos);
      W_sum += exp_shifted(i);
      group_of_obs(i) = g;
      if (cox_data.event_sorted01(pos) != 0u) {
        event_weight(i) = 1.0;
        d_g += 1.0;
        event_eta_sum += linpred(i);
      }
    }
    if (d_g > 0.0) {
      const double denom = std::max(W_sum, 1e-300);
      ll += event_eta_sum - d_g * (max_lp + std::log(denom));
      group_coef(g) = d_g / denom;
    }
  }

  double suffix = 0.0;
  for (int g = static_cast<int>(ng) - 1; g >= 0; --g) {
    suffix += group_coef(static_cast<arma::uword>(g));
    group_coef(static_cast<arma::uword>(g)) = suffix;
  }
  for (arma::uword i = 0; i < n; ++i)
    risk_weight(i) = exp_shifted(i) * group_coef(group_of_obs(i));

  // thread_local buffers are copied out by value in the return pair; do not
  // change these functions to return references.
  static thread_local arma::vec grad;
  if (grad.n_elem != q.n_elem)
    grad.set_size(q.n_elem);
  grad.zeros();

  // Cache-friendly access in column-major storage: fixed column, scan rows.
  for (int k = 0; k < d; ++k) {
    const arma::vec &xk = X.col(active_idx[k]);
    double risk_k = 0.0;
    double event_k = 0.0;
    for (arma::uword i = 0; i < n; ++i) {
      risk_k += xk(i) * risk_weight(i);
      event_k += xk(i) * event_weight(i);
    }
    grad(k) = risk_k - event_k;
  }
  for (int k = 0; k < ntau; ++k) {
    const arma::vec &zk = Z_dat.col(k);
    double risk_k = 0.0;
    double event_k = 0.0;
    for (arma::uword i = 0; i < n; ++i) {
      risk_k += zk(i) * risk_weight(i);
      event_k += zk(i) * event_weight(i);
    }
    grad(d + k) = risk_k - event_k;
  }

  double nlp = -ll;

  double sum_beta_sq = 0.0;
  for (int k = 0; k < d; ++k) {
    double bk = q(k) - beta0;
    sum_beta_sq += bk * bk;
    grad(k) += bk / sigmasq;
  }
  nlp += 0.5 * d * log_sigmasq + 0.5 * sum_beta_sq / sigmasq;

  double sum_tau_sq = 0.0;
  for (int k = 0; k < ntau; ++k) {
    double tk = q(d + k) - tau0;
    sum_tau_sq += tk * tk;
    grad(d + k) += tk / (htau * sigmasq);
  }
  nlp += 0.5 * ntau * log_sigmasq + 0.5 * sum_tau_sq / (htau * sigmasq);

  nlp += (nu0 / 2.0) * log_sigmasq + (nu0 * sigmasq0) / (2.0 * sigmasq);

  double grad_q_sig = (nu0 / 2.0) - (nu0 * sigmasq0) / (2.0 * sigmasq);
  grad_q_sig += 0.5 * d - 0.5 * sum_beta_sq / sigmasq;
  grad_q_sig += 0.5 * ntau - 0.5 * sum_tau_sq / (htau * sigmasq);
  grad(d + ntau) = grad_q_sig;

  return {nlp, grad};
}

template <typename IndexVec>
static inline void sparse_active_add_to_eta(
    arma::vec &eta, const arma::uword *col_ptr, const arma::uword *row_idx,
    const double *xvals, const IndexVec &active_idx, const arma::vec &q) {
  const int d = static_cast<int>(active_idx.size());
  for (int k = 0; k < d; ++k) {
    const arma::uword j = static_cast<arma::uword>(active_idx[k]);
    const double beta_k = q(k);
    if (std::abs(beta_k) < 1e-16)
      continue;
    for (arma::uword idx = col_ptr[j]; idx < col_ptr[j + 1]; ++idx)
      eta(row_idx[idx]) += xvals[idx] * beta_k;
  }
}

template <typename IndexVec>
static inline void sparse_active_xt_weighted(
    arma::vec &out, const arma::uword *col_ptr, const arma::uword *row_idx,
    const double *xvals, const IndexVec &active_idx,
    const arma::vec &weights) {
  const int d = static_cast<int>(active_idx.size());
  out.set_size(d);
  out.zeros();
  for (int k = 0; k < d; ++k) {
    const arma::uword j = static_cast<arma::uword>(active_idx[k]);
    double acc = 0.0;
    for (arma::uword idx = col_ptr[j]; idx < col_ptr[j + 1]; ++idx)
      acc += xvals[idx] * weights(row_idx[idx]);
    out(k) = acc;
  }
}

template <typename IndexVec>
static inline std::pair<double, arma::vec> nlp_grad_binary_joint_sparse(
    const arma::uword n, const arma::uword *col_ptr, const arma::uword *row_idx,
    const double *xvals, const arma::vec &y, const arma::mat &Z_dat,
    const IndexVec &active_idx, const arma::vec &q, double beta0,
    double alpha0, double tau0, double h, double htau, double nu0,
    double sigmasq0) {
  const int d = static_cast<int>(active_idx.size());
  const int ntau = static_cast<int>(Z_dat.n_cols);

  const double alpha = q(d);
  // HMC-1: Clamp log_sigmasq during leapfrog
  const double log_sigmasq = std::max(-23.0, std::min(9.2, q(d + 1 + ntau)));
  const double sigmasq = std::exp(log_sigmasq);

  static thread_local arma::vec eta;
  if (eta.n_elem != n) eta.set_size(n);
  eta.fill(alpha);
  sparse_active_add_to_eta(eta, col_ptr, row_idx, xvals, active_idx, q);
  for (int k = 0; k < ntau; ++k) {
    const double tau_k = q(d + 1 + k);
    if (std::abs(tau_k) < 1e-16)
      continue;
    const arma::vec &zk = Z_dat.col(k);
    for (arma::uword i = 0; i < n; ++i)
      eta(i) += zk(i) * tau_k;
  }

  double nlp = 0.0;
  static thread_local arma::vec diff;
  if (diff.n_elem != n) diff.set_size(n);
  diff.zeros();
  // thread_local buffers are copied out by value in the return pair; do not
  // change these functions to return references.
  static thread_local arma::vec grad;
  if (grad.n_elem != q.n_elem) grad.set_size(q.n_elem);
  grad.zeros();

  for (arma::uword i = 0; i < n; ++i) {
    const double ei = bvs_dadj::clamp_finite(eta(i), -30.0, 30.0, 0.0);
    const double pi = 1.0 / (1.0 + std::exp(-ei));
    if (ei > 0.0)
      nlp += -y(i) * ei + ei + std::log1p(std::exp(-ei));
    else
      nlp += -y(i) * ei + std::log1p(std::exp(ei));
    diff(i) = -(y(i) - pi);
    grad(d) += diff(i);
  }

  static thread_local arma::vec grad_beta;
  sparse_active_xt_weighted(grad_beta, col_ptr, row_idx, xvals, active_idx,
                            diff);
  for (int k = 0; k < d; ++k)
    grad(k) = grad_beta(k);

  for (int k = 0; k < ntau; ++k) {
    const arma::vec &zk = Z_dat.col(k);
    for (arma::uword i = 0; i < n; ++i)
      grad(d + 1 + k) += diff(i) * zk(i);
  }

  double sum_beta_sq = 0.0;
  for (int k = 0; k < d; ++k) {
    const double bk = q(k) - beta0;
    sum_beta_sq += bk * bk;
    grad(k) += bk / sigmasq;
  }
  nlp += 0.5 * d * log_sigmasq + 0.5 * sum_beta_sq / sigmasq;

  const double a_diff = q(d) - alpha0;
  nlp += 0.5 * log_sigmasq + 0.5 * a_diff * a_diff / (h * sigmasq);
  grad(d) += a_diff / (h * sigmasq);

  double sum_tau_sq = 0.0;
  for (int k = 0; k < ntau; ++k) {
    const double tk = q(d + 1 + k) - tau0;
    sum_tau_sq += tk * tk;
    grad(d + 1 + k) += tk / (htau * sigmasq);
  }
  nlp += 0.5 * ntau * log_sigmasq + 0.5 * sum_tau_sq / (htau * sigmasq);

  nlp += (nu0 / 2.0) * log_sigmasq + (nu0 * sigmasq0) / (2.0 * sigmasq);

  double grad_q_sig = (nu0 / 2.0) - (nu0 * sigmasq0) / (2.0 * sigmasq);
  grad_q_sig += 0.5 * d - 0.5 * sum_beta_sq / sigmasq;
  grad_q_sig += 0.5 - 0.5 * a_diff * a_diff / (h * sigmasq);
  grad_q_sig += 0.5 * ntau - 0.5 * sum_tau_sq / (htau * sigmasq);
  grad(d + 1 + ntau) = grad_q_sig;

  return {nlp, grad};
}

// ============================================================================
// Neg-log-posterior + gradient for imbalanced binary (dense, Joint)
// Supports both logit_t (Approach A) and cloglog (Approach B)
// q layout: [beta_active(d), alpha(1), tau(ntau), log_sigmasq(1)]
// ============================================================================
static inline std::pair<double, arma::vec> nlp_grad_imbalanced_joint(
    const arma::mat &X, const arma::vec &y, const arma::mat &Z_dat,
    const std::vector<arma::uword> &active_idx, const arma::vec &q,
    double beta0, double alpha0, double tau0, double h, double htau, double nu0,
    double sigmasq0,
    bool use_cloglog, bool use_t_prior,
    double t_scale, const arma::vec &lambda_t) {
  const arma::uword n = y.n_elem;
  const int d = (int)active_idx.size();
  const int ntau = (int)Z_dat.n_cols;

  double alpha = q(d);
  double log_sigmasq = q(d + 1 + ntau);
  log_sigmasq = std::max(-23.0, std::min(9.2, log_sigmasq));
  double sigmasq = std::exp(log_sigmasq);

  static thread_local arma::vec eta;
  if (eta.n_elem != n) eta.set_size(n);
  eta.fill(alpha);
  for (int k = 0; k < d; ++k) {
    const arma::vec &xk = X.col(active_idx[k]);
    double b = q(k);
    for (arma::uword i = 0; i < n; ++i)
      eta(i) += xk(i) * b;
  }
  for (int k = 0; k < ntau; ++k) {
    const arma::vec &zk = Z_dat.col(k);
    double t = q(d + 1 + k);
    for (arma::uword i = 0; i < n; ++i)
      eta(i) += zk(i) * t;
  }

  double nlp = 0.0;
  arma::vec diff_vec(n, arma::fill::zeros);
  static thread_local arma::vec grad;
  if (grad.n_elem != q.n_elem) grad.set_size(q.n_elem);
  grad.zeros();

  if (use_cloglog) {
    // Complementary log-log likelihood + gradient
    for (arma::uword i = 0; i < n; ++i) {
      double ei = std::max(bvs_imbalanced::CLOGLOG_ETA_LO,
                           std::min(bvs_imbalanced::CLOGLOG_ETA_HI, eta(i)));
      nlp -= bvs_imbalanced::loglik_cloglog_obs(y(i), ei);
      // Gradient contribution: -(d logL / d eta)
      diff_vec(i) = -bvs_imbalanced::cloglog_residual(y(i), ei);
      grad(d) += diff_vec(i);
      for (int k = 0; k < ntau; ++k)
        grad(d + 1 + k) += diff_vec(i) * Z_dat(i, k);
    }
  } else {
    // Standard logistic likelihood (same as nlp_grad_binary_joint)
    for (arma::uword i = 0; i < n; ++i) {
      double ei = bvs_dadj::clamp_finite(eta(i), -30.0, 30.0, 0.0);
      double pi = 1.0 / (1.0 + std::exp(-ei));
      if (ei > 0.0)
        nlp += -y(i) * ei + ei + std::log1p(std::exp(-ei));
      else
        nlp += -y(i) * ei + std::log1p(std::exp(ei));
      const double diff = -(y(i) - pi);
      diff_vec(i) = diff;
      grad(d) += diff;
      for (int k = 0; k < ntau; ++k)
        grad(d + 1 + k) += diff * Z_dat(i, k);
    }
  }
  for (int k = 0; k < d; ++k) {
    const arma::vec &xk = X.col(active_idx[k]);
    double acc = 0.0;
    for (arma::uword i = 0; i < n; ++i)
      acc += diff_vec(i) * xk(i);
    grad(k) += acc;
  }

  // Prior on beta: N(beta0, pvar_j) where pvar_j depends on approach
  double sum_beta_sq_eff = 0.0;  // sum of beta_k^2 / pvar_k (excluding sigmasq)
  for (int k = 0; k < d; ++k) {
    double bk = q(k) - beta0;
    double pvar_k = use_t_prior ?
        (t_scale * t_scale * lambda_t(active_idx[k])) : 1.0;
    sum_beta_sq_eff += bk * bk / pvar_k;
    grad(k) += bk / (sigmasq * pvar_k);
  }
  nlp += 0.5 * d * log_sigmasq + 0.5 * sum_beta_sq_eff / sigmasq;

  double a_diff = q(d) - alpha0;
  nlp += 0.5 * log_sigmasq + 0.5 * a_diff * a_diff / (h * sigmasq);
  grad(d) += a_diff / (h * sigmasq);

  double sum_tau_sq = 0.0;
  for (int k = 0; k < ntau; ++k) {
    double tk = q(d + 1 + k) - tau0;
    sum_tau_sq += tk * tk;
    grad(d + 1 + k) += tk / (htau * sigmasq);
  }
  nlp += 0.5 * ntau * log_sigmasq + 0.5 * sum_tau_sq / (htau * sigmasq);

  nlp += (nu0 / 2.0) * log_sigmasq + (nu0 * sigmasq0) / (2.0 * sigmasq);

  double grad_q_sig = (nu0 / 2.0) - (nu0 * sigmasq0) / (2.0 * sigmasq);
  grad_q_sig += 0.5 * d - 0.5 * sum_beta_sq_eff / sigmasq;
  grad_q_sig += 0.5 - 0.5 * a_diff * a_diff / (h * sigmasq);
  grad_q_sig += 0.5 * ntau - 0.5 * sum_tau_sq / (htau * sigmasq);
  grad(d + 1 + ntau) = grad_q_sig;

  return {nlp, grad};
}

// ============================================================================
// Neg-log-posterior + gradient for imbalanced binary (sparse, Joint)
// ============================================================================
template <typename IndexVec>
static inline std::pair<double, arma::vec> nlp_grad_imbalanced_joint_sparse(
    const arma::uword n, const arma::uword *col_ptr, const arma::uword *row_idx,
    const double *xvals, const arma::vec &y, const arma::mat &Z_dat,
    const IndexVec &active_idx, const arma::vec &q, double beta0,
    double alpha0, double tau0, double h, double htau, double nu0,
    double sigmasq0,
    bool use_cloglog, bool use_t_prior,
    double t_scale, const arma::vec &lambda_t) {
  const int d = static_cast<int>(active_idx.size());
  const int ntau = static_cast<int>(Z_dat.n_cols);

  const double alpha = q(d);
  const double log_sigmasq = std::max(-23.0, std::min(9.2, q(d + 1 + ntau)));
  const double sigmasq = std::exp(log_sigmasq);

  static thread_local arma::vec eta;
  if (eta.n_elem != n) eta.set_size(n);
  eta.fill(alpha);
  sparse_active_add_to_eta(eta, col_ptr, row_idx, xvals, active_idx, q);
  for (int k = 0; k < ntau; ++k) {
    const double tau_k = q(d + 1 + k);
    if (std::abs(tau_k) < 1e-16) continue;
    const arma::vec &zk = Z_dat.col(k);
    for (arma::uword i = 0; i < n; ++i)
      eta(i) += zk(i) * tau_k;
  }

  double nlp = 0.0;
  static thread_local arma::vec diff;
  if (diff.n_elem != n) diff.set_size(n);
  diff.zeros();
  static thread_local arma::vec grad;
  if (grad.n_elem != q.n_elem) grad.set_size(q.n_elem);
  grad.zeros();

  if (use_cloglog) {
    for (arma::uword i = 0; i < n; ++i) {
      double ei = std::max(bvs_imbalanced::CLOGLOG_ETA_LO,
                           std::min(bvs_imbalanced::CLOGLOG_ETA_HI, eta(i)));
      nlp -= bvs_imbalanced::loglik_cloglog_obs(y(i), ei);
      diff(i) = -bvs_imbalanced::cloglog_residual(y(i), ei);
      grad(d) += diff(i);
    }
  } else {
    for (arma::uword i = 0; i < n; ++i) {
      const double ei = bvs_dadj::clamp_finite(eta(i), -30.0, 30.0, 0.0);
      const double pi = 1.0 / (1.0 + std::exp(-ei));
      if (ei > 0.0)
        nlp += -y(i) * ei + ei + std::log1p(std::exp(-ei));
      else
        nlp += -y(i) * ei + std::log1p(std::exp(ei));
      diff(i) = -(y(i) - pi);
      grad(d) += diff(i);
    }
  }

  static thread_local arma::vec grad_beta;
  sparse_active_xt_weighted(grad_beta, col_ptr, row_idx, xvals, active_idx, diff);
  for (int k = 0; k < d; ++k)
    grad(k) = grad_beta(k);
  for (int k = 0; k < ntau; ++k) {
    const arma::vec &zk = Z_dat.col(k);
    for (arma::uword i = 0; i < n; ++i)
      grad(d + 1 + k) += diff(i) * zk(i);
  }

  double sum_beta_sq_eff = 0.0;
  for (int k = 0; k < d; ++k) {
    const double bk = q(k) - beta0;
    double pvar_k = use_t_prior ?
        (t_scale * t_scale * lambda_t(static_cast<arma::uword>(active_idx[k]))) : 1.0;
    sum_beta_sq_eff += bk * bk / pvar_k;
    grad(k) += bk / (sigmasq * pvar_k);
  }
  nlp += 0.5 * d * log_sigmasq + 0.5 * sum_beta_sq_eff / sigmasq;

  const double a_diff = q(d) - alpha0;
  nlp += 0.5 * log_sigmasq + 0.5 * a_diff * a_diff / (h * sigmasq);
  grad(d) += a_diff / (h * sigmasq);

  double sum_tau_sq = 0.0;
  for (int k = 0; k < ntau; ++k) {
    const double tk = q(d + 1 + k) - tau0;
    sum_tau_sq += tk * tk;
    grad(d + 1 + k) += tk / (htau * sigmasq);
  }
  nlp += 0.5 * ntau * log_sigmasq + 0.5 * sum_tau_sq / (htau * sigmasq);

  nlp += (nu0 / 2.0) * log_sigmasq + (nu0 * sigmasq0) / (2.0 * sigmasq);

  double grad_q_sig = (nu0 / 2.0) - (nu0 * sigmasq0) / (2.0 * sigmasq);
  grad_q_sig += 0.5 * d - 0.5 * sum_beta_sq_eff / sigmasq;
  grad_q_sig += 0.5 - 0.5 * a_diff * a_diff / (h * sigmasq);
  grad_q_sig += 0.5 * ntau - 0.5 * sum_tau_sq / (htau * sigmasq);
  grad(d + 1 + ntau) = grad_q_sig;

  return {nlp, grad};
}

template <typename IndexVec>
static inline std::pair<double, arma::vec> nlp_grad_tte_joint_sparse(
    const arma::uword n, const arma::uword *col_ptr, const arma::uword *row_idx,
    const double *xvals, const arma::mat &Z_dat,
    const bvs_dadj::CoxBreslowData &cox_data, const IndexVec &active_idx,
    const arma::vec &q, double beta0, double tau0, double htau, double nu0,
    double sigmasq0) {
  const int d = static_cast<int>(active_idx.size());
  const int ntau = static_cast<int>(Z_dat.n_cols);
  const arma::uword ng = static_cast<arma::uword>(cox_data.group_start.size());

  // HMC-1: Clamp log_sigmasq during leapfrog
  const double log_sigmasq = std::max(-23.0, std::min(9.2, q(d + ntau)));
  const double sigmasq = std::exp(log_sigmasq);

  static thread_local arma::vec linpred;
  if (linpred.n_elem != n) linpred.set_size(n);
  linpred.zeros();
  sparse_active_add_to_eta(linpred, col_ptr, row_idx, xvals, active_idx, q);
  for (int k = 0; k < ntau; ++k) {
    const double tau_k = q(d + k);
    if (std::abs(tau_k) < 1e-16)
      continue;
    const arma::vec &zk = Z_dat.col(k);
    for (arma::uword i = 0; i < n; ++i)
      linpred(i) += zk(i) * tau_k;
  }

  double max_lp = -std::numeric_limits<double>::infinity();
  for (arma::uword i = 0; i < n; ++i) {
    linpred(i) = bvs_dadj::clamp_finite(linpred(i), -1e6, 1e6, 0.0);
    if (linpred(i) > max_lp)
      max_lp = linpred(i);
  }
  if (!std::isfinite(max_lp))
    max_lp = 0.0;

  static thread_local arma::vec risk_weight;
  if (risk_weight.n_elem != n) risk_weight.set_size(n);
  risk_weight.zeros();

  static thread_local arma::vec event_weight;
  if (event_weight.n_elem != n) event_weight.set_size(n);
  event_weight.zeros();

  static thread_local arma::vec exp_shifted;
  if (exp_shifted.n_elem != n) exp_shifted.set_size(n);
  exp_shifted.zeros();

  static thread_local arma::uvec group_of_obs;
  if (group_of_obs.n_elem != n) group_of_obs.set_size(n);
  group_of_obs.zeros();

  static thread_local arma::vec group_coef;
  if (group_coef.n_elem != ng) group_coef.set_size(ng);
  group_coef.zeros();

  for (arma::uword i = 0; i < n; ++i)
    exp_shifted(i) = std::exp(linpred(i) - max_lp);

  double ll = 0.0;
  double W_sum = 0.0;
  for (arma::uword g = 0; g < ng; ++g) {
    double d_g = 0.0;
    double event_eta_sum = 0.0;
    for (arma::uword pos = cox_data.group_start[g];
         pos <= cox_data.group_end[g]; ++pos) {
      const arma::uword i = cox_data.order(pos);
      W_sum += exp_shifted(i);
      group_of_obs(i) = g;
      if (cox_data.event_sorted01(pos) != 0u) {
        event_weight(i) = 1.0;
        d_g += 1.0;
        event_eta_sum += linpred(i);
      }
    }
    if (d_g > 0.0) {
      ll += event_eta_sum - d_g * (max_lp + std::log(std::max(W_sum, 1e-300)));
      group_coef(g) = d_g / std::max(W_sum, 1e-300);
    }
  }

  double suffix = 0.0;
  for (int g = static_cast<int>(ng) - 1; g >= 0; --g) {
    suffix += group_coef(static_cast<arma::uword>(g));
    group_coef(static_cast<arma::uword>(g)) = suffix;
  }
  for (arma::uword i = 0; i < n; ++i)
    risk_weight(i) = exp_shifted(i) * group_coef(group_of_obs(i));

  // thread_local buffers are copied out by value in the return pair; do not
  // change these functions to return references.
  static thread_local arma::vec grad;
  if (grad.n_elem != q.n_elem) grad.set_size(q.n_elem);
  grad.zeros();
  static thread_local arma::vec risk_beta;
  static thread_local arma::vec event_beta;
  sparse_active_xt_weighted(risk_beta, col_ptr, row_idx, xvals, active_idx,
                            risk_weight);
  sparse_active_xt_weighted(event_beta, col_ptr, row_idx, xvals, active_idx,
                            event_weight);
  for (int k = 0; k < d; ++k)
    grad(k) = risk_beta(k) - event_beta(k);

  for (int k = 0; k < ntau; ++k) {
    const arma::vec &zk = Z_dat.col(k);
    double risk_tau_k = 0.0;
    double event_tau_k = 0.0;
    for (arma::uword i = 0; i < n; ++i) {
      risk_tau_k += zk(i) * risk_weight(i);
      event_tau_k += zk(i) * event_weight(i);
    }
    grad(d + k) = risk_tau_k - event_tau_k;
  }

  double nlp = -ll;

  double sum_beta_sq = 0.0;
  for (int k = 0; k < d; ++k) {
    const double bk = q(k) - beta0;
    sum_beta_sq += bk * bk;
    grad(k) += bk / sigmasq;
  }
  nlp += 0.5 * d * log_sigmasq + 0.5 * sum_beta_sq / sigmasq;

  double sum_tau_sq = 0.0;
  for (int k = 0; k < ntau; ++k) {
    const double tk = q(d + k) - tau0;
    sum_tau_sq += tk * tk;
    grad(d + k) += tk / (htau * sigmasq);
  }
  nlp += 0.5 * ntau * log_sigmasq + 0.5 * sum_tau_sq / (htau * sigmasq);

  nlp += (nu0 / 2.0) * log_sigmasq + (nu0 * sigmasq0) / (2.0 * sigmasq);

  double grad_q_sig = (nu0 / 2.0) - (nu0 * sigmasq0) / (2.0 * sigmasq);
  grad_q_sig += 0.5 * d - 0.5 * sum_beta_sq / sigmasq;
  grad_q_sig += 0.5 * ntau - 0.5 * sum_tau_sq / (htau * sigmasq);
  grad(d + ntau) = grad_q_sig;

  return {nlp, grad};
}

// ============================================================================
// Convenience: run one HMC or NUTS iteration for active beta, alpha, tau, sig
//
// Optimizations:
//   1. Uses diagonal mass matrix (prior-informed) for better mixing
//   2. Recomputes Xb_total in O(n*d) instead of O(n*p)
//   3. Clamps parameters to consistent bounds after the step
// ============================================================================
template <typename ComputeFn>
static inline double run_hmc_nuts_joint(
    int alg_type_int, const arma::mat &X, const arma::vec &y,
    const arma::mat &Z_dat, arma::vec &beta, arma::vec &Xb_total,
    double &loglik, double &alpha, arma::vec &tau, double &sigmasq,
    double beta0, double alpha0, double tau0, double h, double htau, double nu0,
    double sigmasq0, const std::vector<arma::uword> &active_idx, bool is_tte,
    const bvs_dadj::CoxBreslowData &cox_data, bvs_dadj::CoxTracker &cox_tracker,
    double &hmc_epsilon, int hmc_n_leapfrog, int nuts_max_treedepth,
    ComputeFn &compute, const DiagMassMatrix *mass_override = nullptr,
    HMCSamplingStats *step_stats = nullptr) {

  const int d = (int)active_idx.size();
  const int ntau = (int)Z_dat.n_cols;
  const arma::uword n = X.n_rows;

  int dim_q = d + ntau + 1 + (is_tte ? 0 : 1);
  arma::vec q(dim_q);

  for (int k = 0; k < d; ++k)
    q(k) = beta(active_idx[k]);
  if (!is_tte) {
    q(d) = alpha;
    for (int k = 0; k < ntau; ++k)
      q(d + 1 + k) = tau(k);
    q(d + 1 + ntau) = std::log(sigmasq);
  } else {
    for (int k = 0; k < ntau; ++k)
      q(d + k) = tau(k);
    q(d + ntau) = std::log(sigmasq);
  }

  // Build prior-informed mass matrix
  DiagMassMatrix mass;
  if (mass_override != nullptr &&
      mass_override->inv_mass.n_elem == static_cast<arma::uword>(dim_q) &&
      mass_override->is_valid()) {
    mass = *mass_override;
  } else {
    build_prior_mass_matrix(mass, d, ntau, is_tte, sigmasq, h, htau, nu0);
  }

  struct JointGradCache {
    bool valid;
    const void *x_ptr;
    const void *active_ptr;
    arma::uword active_size;
    std::size_t active_hash;
    bool is_tte;
    arma::vec q;
    arma::vec grad;
    double nlp;
    JointGradCache()
        : valid(false), x_ptr(nullptr), active_ptr(nullptr), active_size(0),
          active_hash(0), is_tte(false), nlp(0.0) {}
  };
  static thread_local JointGradCache cache;

  const void *x_ptr = static_cast<const void *>(X.memptr());
  const void *active_ptr =
      active_idx.empty()
          ? nullptr
          : static_cast<const void *>(active_idx.data());
  std::size_t active_hash = 1469598103934665603ull;
  for (arma::uword j : active_idx) {
    active_hash ^= static_cast<std::size_t>(j) + 0x9e3779b97f4a7c15ull +
                   (active_hash << 6) + (active_hash >> 2);
  }
  const bool cache_hit =
      cache.valid && cache.x_ptr == x_ptr && cache.active_ptr == active_ptr &&
      cache.active_size == active_idx.size() && cache.active_hash == active_hash &&
      cache.is_tte == is_tte &&
      cache.q.n_elem == q.n_elem && arma::approx_equal(cache.q, q, "absdiff", 0.0);

  double nlp0;
  arma::vec grad0;
  if (cache_hit) {
    nlp0 = cache.nlp;
    grad0 = cache.grad;
  } else {
    auto res0 = compute(q);
    nlp0 = res0.first;
    grad0 = std::move(res0.second);
  }

  HMCSamplingStats local_stats;
  double accept_prob;
  if (alg_type_int == 1) {
    accept_prob = hmc_step(q, nlp0, grad0, hmc_epsilon, hmc_n_leapfrog, compute,
                           mass, &local_stats);
  } else {
    accept_prob = nuts_step(q, nlp0, grad0, hmc_epsilon, nuts_max_treedepth,
                            compute, mass, &local_stats);
  }
  if (step_stats)
    *step_stats = local_stats;

  if (std::isfinite(nlp0) && grad0.is_finite()) {
    cache.valid = true;
    cache.x_ptr = x_ptr;
    cache.active_ptr = active_ptr;
    cache.active_size = active_idx.size();
    cache.active_hash = active_hash;
    cache.is_tte = is_tte;
    cache.q = q;
    cache.grad = grad0;
    cache.nlp = nlp0;
  } else {
    cache.valid = false;
  }

  // Fast path: rejected proposals keep the current state exactly.
  if (!local_stats.accepted)
    return accept_prob;

  // Unpack and clamp parameters for robustness
  for (int k = 0; k < d; ++k) {
    double bk = bvs_dadj::clamp_finite(q(k), -30.0, 30.0, 0.0);
    beta(active_idx[k]) = bk;
  }
  if (!is_tte) {
    alpha = bvs_dadj::clamp_finite(q(d), -30.0, 30.0, 0.0);
    for (int k = 0; k < ntau; ++k)
      tau(k) = bvs_dadj::clamp_finite(q(d + 1 + k), -30.0, 30.0, 0.0);
    double log_sig = bvs_dadj::clamp_finite(q(d + 1 + ntau), -23.0, 9.2, 0.0);
    sigmasq = std::exp(log_sig); // ≈ [1e-10, 1e4]
  } else {
    for (int k = 0; k < ntau; ++k)
      tau(k) = bvs_dadj::clamp_finite(q(d + k), -30.0, 30.0, 0.0);
    double log_sig = bvs_dadj::clamp_finite(q(d + ntau), -23.0, 9.2, 0.0);
    sigmasq = std::exp(log_sig);
  }

  // Recompute Xb_total in O(n*d) instead of O(n*p):
  // only active betas are non-zero
  Xb_total.set_size(n);
  Xb_total.zeros();
  for (int k = 0; k < d; ++k) {
    const arma::vec &xk = X.col(active_idx[k]);
    double bk = beta(active_idx[k]);
    for (arma::uword i = 0; i < n; ++i)
      Xb_total(i) += xk(i) * bk;
  }
  // Add Z_dat * tau
  for (int k = 0; k < ntau; ++k) {
    const arma::vec &zk = Z_dat.col(k);
    double tk = tau(k);
    for (arma::uword i = 0; i < n; ++i)
      Xb_total(i) += zk(i) * tk;
  }

  if (is_tte) {
    cox_tracker.init(Xb_total, cox_data);
    loglik = cox_tracker.get_loglik();
  } else {
    loglik = 0.0;
    for (arma::uword i = 0; i < n; ++i) {
      double ei = alpha + Xb_total(i);
      if (ei > 0.0)
        loglik += y(i) * ei - ei - std::log1p(std::exp(-ei));
      else
        loglik += y(i) * ei - std::log1p(std::exp(ei));
    }
  }

  return accept_prob;
}

// ============================================================================
// Convenience: run one HMC or NUTS iteration for active beta only
//
// alg_type_int: 1 = HMC, 2 = NUTS
// Returns acceptance probability for adaptation.
// Updates beta, Xb_total, loglik in place.
// For TTE, reinitialises cox_tracker after the step.
// ============================================================================
template <typename ComputeFn>
static inline double
run_hmc_nuts_beta(int alg_type_int, const arma::mat &X, const arma::vec &y,
                  const arma::vec &Z_tau, arma::vec &beta, arma::vec &Xb_total,
                  double &loglik, double alpha, double beta0, double sigmasq,
                  const std::vector<arma::uword> &active_idx, bool is_tte,
                  const bvs_dadj::CoxBreslowData &cox_data,
                  bvs_dadj::CoxTracker &cox_tracker, double &hmc_epsilon,
                  int hmc_n_leapfrog, int nuts_max_treedepth,
                  ComputeFn &compute, HMCSamplingStats *step_stats = nullptr) {
  const int d = (int)active_idx.size();
  if (d == 0)
    return 1.0;

  // Extract active beta
  arma::vec q(d);
  for (int k = 0; k < d; ++k)
    q(k) = beta(active_idx[k]);

  // Simple mass matrix for beta-only: use prior variance
  DiagMassMatrix mass;
  mass.inv_mass.set_size(d);
  mass.sqrt_inv_mass.set_size(d);
  double sig2 = std::max(sigmasq, 1e-8);
  for (int k = 0; k < d; ++k) {
    mass.inv_mass(k) = sig2;
    mass.sqrt_inv_mass(k) = std::sqrt(sig2);
  }

  // Compute initial neg-log-post and gradient
  auto res0 = compute(q);
  double nlp0 = res0.first;
  arma::vec grad0 = res0.second;

  HMCSamplingStats local_stats;
  double accept_prob;
  if (alg_type_int == 1) {
    accept_prob = hmc_step(q, nlp0, grad0, hmc_epsilon, hmc_n_leapfrog, compute,
                           mass, &local_stats);
  } else {
    accept_prob = nuts_step(q, nlp0, grad0, hmc_epsilon, nuts_max_treedepth,
                            compute, mass, &local_stats);
  }
  if (step_stats)
    *step_stats = local_stats;

  if (!local_stats.accepted)
    return accept_prob;

  // Write back active beta with bounds
  for (int k = 0; k < d; ++k)
    beta(active_idx[k]) = bvs_dadj::clamp_finite(q(k), -30.0, 30.0, 0.0);

  // Recompute Xb_total in O(n*d) instead of O(n*p)
  const arma::uword n = X.n_rows;
  Xb_total.set_size(n);
  for (arma::uword i = 0; i < n; ++i)
    Xb_total(i) = Z_tau(i);
  for (int k = 0; k < d; ++k) {
    const arma::vec &xk = X.col(active_idx[k]);
    double bk = beta(active_idx[k]);
    for (arma::uword i = 0; i < n; ++i)
      Xb_total(i) += xk(i) * bk;
  }

  if (is_tte) {
    cox_tracker.init(Xb_total, cox_data);
    loglik = cox_tracker.get_loglik();
  } else {
    // Binary: recompute loglik
    loglik = 0.0;
    for (arma::uword i = 0; i < n; ++i) {
      double ei = alpha + Xb_total(i);
      if (ei > 0.0)
        loglik += y(i) * ei - ei - std::log1p(std::exp(-ei));
      else
        loglik += y(i) * ei - std::log1p(std::exp(ei));
    }
  }

  return accept_prob;
}

// ============================================================================
// Parse alg_type string to int: 0=MH, 1=HMC, 2=NUTS
// ============================================================================
static inline int parse_alg_type(const std::string &alg_type) {
  if (alg_type == "MH" || alg_type == "mh")
    return 0;
  if (alg_type == "HMC" || alg_type == "hmc")
    return 1;
  if (alg_type == "NUTS" || alg_type == "nuts")
    return 2;
  Rcpp::stop("alg_type must be 'MH', 'HMC', or 'NUTS'.");
  return 0;
}

} // namespace bvs_hmc

#endif // BVS_HMC_NUTS_H
