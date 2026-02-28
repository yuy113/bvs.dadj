// =============================================================================
// BayesLogit_BlockPG.h
//
// Shared infrastructure for block-100 gamma updates in PG samplers:
//   1. Union-Find for Swendsen-Wang cluster detection
//   2. Swendsen-Wang cluster proposal (single & dual adjacency)
//   3. Woodbury rank-k covariance updates (bordered matrix inverse)
//   4. Preconditioned Conjugate Gradient (PCG) beta sampler
//   5. Uncollapsed Gibbs sweep (gamma | beta, no matrix inversion)
//   6. Block MH acceptance for collapsed mode
//
// All functions are static inline in an anonymous namespace to avoid ODR
// violations across translation units (each .cpp gets its own copy).
// =============================================================================

#ifndef BVS_DADJ_BAYESLOGIT_BLOCKPG_H
#define BVS_DADJ_BAYESLOGIT_BLOCKPG_H

#include "BayesLogit_Numerics.h"
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

namespace bvs_dadj_block {

// =============================================================================
// SECTION 1: UNION-FIND
// =============================================================================

struct UnionFind {
  std::vector<int> parent;
  std::vector<int> rank_;

  explicit UnionFind(int n) : parent(n), rank_(n, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]]; // path halving
      x = parent[x];
    }
    return x;
  }

  void unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b)
      return;
    if (rank_[a] < rank_[b])
      std::swap(a, b);
    parent[b] = a;
    if (rank_[a] == rank_[b])
      ++rank_[a];
  }

  // Returns clusters as vector of vectors. Only includes clusters of size >= 2
  // or clusters containing at least one member with the given gamma state.
  std::vector<std::vector<int>> get_components(int n) {
    std::vector<std::vector<int>> comp_map;
    std::vector<int> root_to_idx(n, -1);
    int next_idx = 0;
    for (int i = 0; i < n; ++i) {
      int r = find(i);
      if (root_to_idx[r] < 0) {
        root_to_idx[r] = next_idx++;
        comp_map.emplace_back();
      }
      comp_map[root_to_idx[r]].push_back(i);
    }
    return comp_map;
  }
};

// =============================================================================
// SECTION 2: SWENDSEN-WANG CLUSTER BUILDER
// =============================================================================

struct ClusterProposal {
  std::vector<std::vector<int>> clusters;
  int total_flipped;
};

// Single-adjacency Swendsen-Wang
// for_each_neighbor(j, callback): iterates over neighbors of j, calling
// callback(k) for each neighbor k
template <typename NeighborFn>
inline ClusterProposal
swendsen_wang_single(const std::vector<uint8_t> &gamma, double eta, int p,
                     int block_size, NeighborFn for_each_neighbor) {
  ClusterProposal result;
  result.total_flipped = 0;

  if (p <= 0 || block_size <= 0)
    return result;

  // Bond activation probability
  double p_bond = 1.0 - std::exp(-2.0 * std::max(0.0, eta));
  p_bond = std::min(1.0, std::max(0.0, p_bond));

  UnionFind uf(p);

  // Activate bonds between same-state neighbors
  for (int i = 0; i < p; ++i) {
    for_each_neighbor(i, [&](int j) {
      if (j > i && gamma[i] == gamma[j]) {
        if (R::runif(0.0, 1.0) < p_bond) {
          uf.unite(i, j);
        }
      }
    });
  }

  // Extract clusters
  auto all_clusters = uf.get_components(p);

  // Shuffle and select clusters up to block_size budget
  // Use Fisher-Yates partial shuffle
  int n_clusters = static_cast<int>(all_clusters.size());
  int budget = block_size;

  for (int i = 0; i < n_clusters && budget > 0; ++i) {
    // Pick random remaining cluster
    int swap_idx = i + static_cast<int>(std::floor(
                           R::runif(0.0, (double)(n_clusters - i))));
    if (swap_idx >= n_clusters)
      swap_idx = n_clusters - 1;
    std::swap(all_clusters[i], all_clusters[swap_idx]);

    int csize = static_cast<int>(all_clusters[i].size());
    if (csize <= budget) {
      result.clusters.push_back(std::move(all_clusters[i]));
      result.total_flipped += csize;
      budget -= csize;
    }
  }

  return result;
}

// Dual-adjacency Swendsen-Wang
// Two neighbor functions for two adjacency matrices
template <typename NeighborFn1, typename NeighborFn2>
inline ClusterProposal
swendsen_wang_dual(const std::vector<uint8_t> &gamma, double eta1, double eta2,
                   int p, int block_size, NeighborFn1 for_each_neighbor1,
                   NeighborFn2 for_each_neighbor2) {
  ClusterProposal result;
  result.total_flipped = 0;

  if (p <= 0 || block_size <= 0)
    return result;

  double p_bond1 = 1.0 - std::exp(-2.0 * std::max(0.0, eta1));
  double p_bond2 = 1.0 - std::exp(-2.0 * std::max(0.0, eta2));
  p_bond1 = std::min(1.0, std::max(0.0, p_bond1));
  p_bond2 = std::min(1.0, std::max(0.0, p_bond2));

  UnionFind uf(p);

  // Activate bonds from adjacency 1
  for (int i = 0; i < p; ++i) {
    for_each_neighbor1(i, [&](int j) {
      if (j > i && gamma[i] == gamma[j]) {
        if (R::runif(0.0, 1.0) < p_bond1) {
          uf.unite(i, j);
        }
      }
    });
  }

  // Activate bonds from adjacency 2
  for (int i = 0; i < p; ++i) {
    for_each_neighbor2(i, [&](int j) {
      if (j > i && gamma[i] == gamma[j]) {
        if (R::runif(0.0, 1.0) < p_bond2) {
          uf.unite(i, j);
        }
      }
    });
  }

  auto all_clusters = uf.get_components(p);
  int n_clusters = static_cast<int>(all_clusters.size());
  int budget = block_size;

  for (int i = 0; i < n_clusters && budget > 0; ++i) {
    int swap_idx = i + static_cast<int>(std::floor(
                           R::runif(0.0, (double)(n_clusters - i))));
    if (swap_idx >= n_clusters)
      swap_idx = n_clusters - 1;
    std::swap(all_clusters[i], all_clusters[swap_idx]);

    int csize = static_cast<int>(all_clusters[i].size());
    if (csize <= budget) {
      result.clusters.push_back(std::move(all_clusters[i]));
      result.total_flipped += csize;
      budget -= csize;
    }
  }

  return result;
}

// =============================================================================
// SECTION 3: WOODBURY RANK-K COVARIANCE UPDATE
// =============================================================================

struct CovarianceState {
  arma::mat Sigma;         // p_active x p_active posterior covariance
  std::vector<int> active; // current active set indices
  int n_since_reset;
  bool valid;

  CovarianceState() : n_since_reset(0), valid(false) {}
};

// Full recomputation: Sigma = (X_act' * diag(omega) * X_act + inv_sigmasq *
// I)^{-1}
inline bool reset_covariance(CovarianceState &state, const arma::mat &X,
                              const arma::vec &omega_pg, double inv_sigmasq,
                              const std::vector<int> &active_idx) {
  int p_act = static_cast<int>(active_idx.size());
  state.active = active_idx;
  state.n_since_reset = 0;

  if (p_act == 0) {
    state.Sigma.set_size(0, 0);
    state.valid = true;
    return true;
  }

  arma::uvec act(active_idx.size());
  for (size_t i = 0; i < active_idx.size(); ++i)
    act(i) = static_cast<arma::uword>(active_idx[i]);

  arma::mat X_act = X.cols(act);
  arma::mat Xt_Om = X_act.t();
  Xt_Om.each_row() %= omega_pg.t();
  arma::mat prec = Xt_Om * X_act;
  prec.diag() += inv_sigmasq;

  bvs_dadj::sanitize_sym_mat_inplace(prec);

  arma::mat L;
  bool ok = bvs_dadj::robust_chol_inplace(L, prec);
  if (!ok) {
    state.valid = false;
    return false;
  }

  // Sigma = L^{-T} * L^{-1}  (inverse of precision via Cholesky)
  arma::mat Linv;
  ok = arma::inv(Linv, arma::trimatu(L));
  if (!ok) {
    state.valid = false;
    return false;
  }
  state.Sigma = Linv * Linv.t();
  state.valid = true;
  return true;
}

// Woodbury: add k columns to active set (bordered matrix inverse)
// add_indices: the column indices (in X) being activated
// Returns false on numerical failure (caller should fallback to full reset)
inline bool woodbury_add_columns(CovarianceState &state, const arma::mat &X,
                                  const arma::vec &omega_pg,
                                  double inv_sigmasq,
                                  const std::vector<int> &add_indices) {
  if (add_indices.empty())
    return true;
  if (!state.valid)
    return false;

  int p_old = static_cast<int>(state.active.size());
  int k = static_cast<int>(add_indices.size());
  int n = static_cast<int>(X.n_rows);

  // B = X_old' * diag(omega) * X_new  (p_old x k)
  // D_k = X_new' * diag(omega) * X_new + inv_sigmasq * I_k  (k x k)
  arma::mat X_new(n, k);
  for (int i = 0; i < k; ++i)
    X_new.col(i) = X.col(add_indices[i]);

  arma::mat B;
  if (p_old > 0) {
    arma::mat X_old(n, p_old);
    for (int i = 0; i < p_old; ++i)
      X_old.col(i) = X.col(state.active[i]);

    arma::mat Xt_old_Om = X_old.t();
    Xt_old_Om.each_row() %= omega_pg.t();
    B = Xt_old_Om * X_new; // p_old x k
  } else {
    B.set_size(0, k);
  }

  arma::mat Xt_new_Om = X_new.t();
  Xt_new_Om.each_row() %= omega_pg.t();
  arma::mat D_k = Xt_new_Om * X_new; // k x k
  D_k.diag() += inv_sigmasq;

  // Schur complement: S = D_k - B' * Sigma_old * B
  arma::mat S = D_k;
  if (p_old > 0) {
    arma::mat SigB = state.Sigma * B; // p_old x k
    S -= B.t() * SigB;                // k x k
  }

  // Invert S (k x k)
  bvs_dadj::sanitize_sym_mat_inplace(S);
  arma::mat Sinv;
  bool ok = arma::inv_sympd(Sinv, arma::symmatu(S));
  if (!ok) {
    // Try with jitter
    S.diag() += 1e-8;
    ok = arma::inv_sympd(Sinv, arma::symmatu(S));
    if (!ok)
      return false;
  }

  // Build new Sigma of size (p_old + k) x (p_old + k)
  int p_new = p_old + k;
  arma::mat Sigma_new(p_new, p_new);

  if (p_old > 0) {
    arma::mat SigB = state.Sigma * B; // p_old x k
    arma::mat SigB_Sinv = SigB * Sinv; // p_old x k

    // Sigma_11 = Sigma_old + SigB * Sinv * SigB'
    Sigma_new.submat(0, 0, p_old - 1, p_old - 1) =
        state.Sigma + SigB_Sinv * SigB.t();

    // Sigma_12 = -SigB * Sinv
    Sigma_new.submat(0, p_old, p_old - 1, p_new - 1) = -SigB_Sinv;

    // Sigma_21 = -Sinv * SigB'
    Sigma_new.submat(p_old, 0, p_new - 1, p_old - 1) = -SigB_Sinv.t();
  }

  // Sigma_22 = Sinv
  Sigma_new.submat(p_old, p_old, p_new - 1, p_new - 1) = Sinv;

  state.Sigma = Sigma_new;
  for (int idx : add_indices)
    state.active.push_back(idx);
  state.n_since_reset++;
  return true;
}

// Woodbury: remove k columns from active set (deflation via Schur complement)
inline bool woodbury_remove_columns(CovarianceState &state,
                                     const std::vector<int> &rem_indices) {
  if (rem_indices.empty())
    return true;
  if (!state.valid)
    return false;

  int p_old = static_cast<int>(state.active.size());
  int k = static_cast<int>(rem_indices.size());

  // Find positions of rem_indices in state.active
  std::vector<int> rem_pos;
  rem_pos.reserve(k);
  for (int idx : rem_indices) {
    for (int i = 0; i < p_old; ++i) {
      if (state.active[i] == idx) {
        rem_pos.push_back(i);
        break;
      }
    }
  }
  if (static_cast<int>(rem_pos.size()) != k)
    return false;

  // Sort rem_pos for easier permutation
  std::sort(rem_pos.begin(), rem_pos.end());

  // Build keep positions
  std::vector<int> keep_pos;
  keep_pos.reserve(p_old - k);
  {
    int ri = 0;
    for (int i = 0; i < p_old; ++i) {
      if (ri < k && rem_pos[ri] == i) {
        ++ri;
      } else {
        keep_pos.push_back(i);
      }
    }
  }

  int p_new = static_cast<int>(keep_pos.size());

  if (p_new == 0) {
    state.Sigma.set_size(0, 0);
    state.active.clear();
    state.n_since_reset++;
    return true;
  }

  // Extract sub-blocks
  arma::uvec keep_uv(keep_pos.size());
  for (size_t i = 0; i < keep_pos.size(); ++i)
    keep_uv(i) = static_cast<arma::uword>(keep_pos[i]);
  arma::uvec rem_uv(rem_pos.size());
  for (size_t i = 0; i < rem_pos.size(); ++i)
    rem_uv(i) = static_cast<arma::uword>(rem_pos[i]);

  arma::mat Sig11 = state.Sigma.submat(keep_uv, keep_uv); // (p_new x p_new)
  arma::mat Sig12 = state.Sigma.submat(keep_uv, rem_uv);  // (p_new x k)
  arma::mat Sig22 = state.Sigma.submat(rem_uv, rem_uv);   // (k x k)

  // New covariance: Sig11 - Sig12 * Sig22^{-1} * Sig12'
  arma::mat Sig22_inv;
  bool ok = arma::inv_sympd(Sig22_inv, arma::symmatu(Sig22));
  if (!ok) {
    Sig22.diag() += 1e-8;
    ok = arma::inv_sympd(Sig22_inv, arma::symmatu(Sig22));
    if (!ok)
      return false;
  }

  state.Sigma = Sig11 - Sig12 * Sig22_inv * Sig12.t();

  // Update active list
  std::vector<int> new_active;
  new_active.reserve(p_new);
  for (int pos : keep_pos)
    new_active.push_back(state.active[pos]);
  state.active = std::move(new_active);
  state.n_since_reset++;
  return true;
}

// =============================================================================
// SECTION 4: PCG BETA SAMPLER
// =============================================================================

struct PCGConfig {
  double tol;
  int max_iter;
  int pcg_threshold; // use PCG when p_active > this

  PCGConfig() : tol(1e-6), max_iter(200), pcg_threshold(500) {}
  PCGConfig(double t, int m, int th) : tol(t), max_iter(m), pcg_threshold(th) {}
};

// Dense PCG beta sampler
// Samples beta ~ N(m, Sigma) where Sigma = P^{-1}, P = X_act'*Omega*X_act +
// inv_sigmasq*I Uses Papandreou-Yuille CG sampling trick
inline bool pcg_sample_beta(arma::vec &beta_out, const arma::mat &X_act,
                             const arma::vec &omega_pg, const arma::vec &kappa,
                             double alpha, double inv_sigmasq,
                             const PCGConfig &cfg) {
  int n = static_cast<int>(X_act.n_rows);
  int p_act = static_cast<int>(X_act.n_cols);

  if (p_act == 0) {
    beta_out.set_size(0);
    return true;
  }

  // Diagonal preconditioner: M[j] = sum_i(omega_i * X_ij^2) + inv_sigmasq
  arma::vec M(p_act);
  for (int j = 0; j < p_act; ++j) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
      s += omega_pg(i) * X_act(i, j) * X_act(i, j);
    M(j) = s + inv_sigmasq;
  }
  // Avoid division by zero
  for (int j = 0; j < p_act; ++j)
    M(j) = std::max(M(j), 1e-12);

  // Precision-times-vector lambda (no explicit P formed)
  auto Pv = [&](const arma::vec &v) -> arma::vec {
    arma::vec Xv = X_act * v;     // n x 1
    Xv %= omega_pg;                // element-wise multiply
    arma::vec result = X_act.t() * Xv; // p_act x 1
    result += inv_sigmasq * v;
    return result;
  };

  // RHS: X_act' * (kappa - omega * alpha)
  arma::vec z_star = kappa - omega_pg * alpha;
  arma::vec rhs = X_act.t() * z_star;

  // Noise for sampling: b = rhs + sqrt(M) .* z where z ~ N(0,I)
  arma::vec z_noise = arma::randn<arma::vec>(p_act);
  arma::vec noise = arma::sqrt(M) % z_noise;
  arma::vec b = rhs + noise;

  // PCG solve: P * beta = b
  arma::vec x = arma::zeros<arma::vec>(p_act); // initial guess
  arma::vec r = b - Pv(x);
  arma::vec z = r / M;                          // preconditioner solve
  arma::vec p_cg = z;
  double rz = arma::dot(r, z);
  double b_norm = arma::norm(b);
  if (b_norm < 1e-16)
    b_norm = 1.0;

  for (int iter = 0; iter < cfg.max_iter; ++iter) {
    if (arma::norm(r) / b_norm < cfg.tol)
      break;

    arma::vec Ap = Pv(p_cg);
    double pAp = arma::dot(p_cg, Ap);
    if (pAp < 1e-30)
      break;

    double alpha_cg = rz / pAp;
    x += alpha_cg * p_cg;
    r -= alpha_cg * Ap;

    arma::vec z_new = r / M;
    double rz_new = arma::dot(r, z_new);
    double beta_cg = rz_new / std::max(rz, 1e-30);

    p_cg = z_new + beta_cg * p_cg;
    z = z_new;
    rz = rz_new;
  }

  beta_out = x;
  bvs_dadj::sanitize_vec_inplace(beta_out, -30.0, 30.0, 0.0);
  return true;
}

// Sparse PCG beta sampler (for sp_mat X)
inline bool pcg_sample_beta_sparse(arma::vec &beta_out,
                                    const arma::sp_mat &X,
                                    const std::vector<int> &active_idx,
                                    const arma::vec &omega_pg,
                                    const arma::vec &kappa, double alpha,
                                    double inv_sigmasq, const PCGConfig &cfg) {
  int n = static_cast<int>(X.n_rows);
  int p_act = static_cast<int>(active_idx.size());

  if (p_act == 0) {
    beta_out.set_size(0);
    return true;
  }

  // Build dense X_act from sparse columns
  arma::mat X_act(n, p_act, arma::fill::zeros);
  for (int k = 0; k < p_act; ++k) {
    int j = active_idx[k];
    for (arma::sp_mat::const_col_iterator it = X.begin_col(j);
         it != X.end_col(j); ++it) {
      X_act(it.row(), k) = (*it);
    }
  }

  // Delegate to dense PCG
  return pcg_sample_beta(beta_out, X_act, omega_pg, kappa, alpha, inv_sigmasq,
                          cfg);
}

// =============================================================================
// SECTION 5: UNCOLLAPSED GIBBS SWEEP
// =============================================================================

// Logistic log-likelihood helper (stable)
inline double loglik_obs_stable(double yi, double psi) {
  psi = bvs_dadj::clamp_finite(psi, -30.0, 30.0, 0.0);
  if (psi > 0.0)
    return yi * psi - psi - std::log1p(std::exp(-psi));
  return yi * psi - std::log1p(std::exp(psi));
}

// Compute log-likelihood difference when changing X.col(j) contribution by db
// Dense version
inline double dense_column_ll_diff(const arma::vec &y, const arma::vec &Xb,
                                    double alpha_val, const arma::mat &X, int j,
                                    double db) {
  if (std::abs(db) < 1e-16)
    return 0.0;
  int n = static_cast<int>(y.n_elem);
  double diff = 0.0;
  for (int i = 0; i < n; ++i) {
    double psi_old = alpha_val + Xb(i);
    double psi_new = psi_old + db * X(i, j);
    diff += loglik_obs_stable(y(i), psi_new) - loglik_obs_stable(y(i), psi_old);
  }
  return diff;
}

// Uncollapsed Gibbs sweep for a block of variables (single adjacency)
// Updates gamma and beta in-place for the given block, conditions on current
// beta
template <typename NeighborFn>
inline void uncollapsed_gamma_sweep_single(
    std::vector<uint8_t> &gamma, arma::vec &beta, arma::vec &Xb,
    const arma::mat &X, const arma::vec &y, double alpha_val, double sigmasq,
    double beta0, double mu, double eta,
    const std::vector<int> &block_indices, NeighborFn for_each_neighbor) {

  double sd_beta = std::sqrt(sigmasq);
  if (!std::isfinite(sd_beta) || sd_beta <= 0.0)
    sd_beta = 1.0;
  for (int j : block_indices) {
    int g_curr = static_cast<int>(gamma[j]);
    double b_curr = beta(j);

    // Compute neighborhood sum
    double neigh_sum = 0.0;
    for_each_neighbor(j, [&](int k) { neigh_sum += gamma[k]; });

    // Ising prior log-odds for gamma_j = 1
    double log_prior_odds = mu + eta * neigh_sum;

    if (g_curr == 0) {
      // Propose activation: sample beta_j from prior
      double b_prop = R::rnorm(beta0, sd_beta);
      b_prop = bvs_dadj::clamp_finite(b_prop, -30.0, 30.0, 0.0);
      double db = b_prop - 0.0; // b_curr is 0
      double ll_diff = dense_column_ll_diff(y, Xb, alpha_val, X, j, db);
      double log_ratio = ll_diff + log_prior_odds;

      if (bvs_dadj::safe_mh_accept(log_ratio)) {
        gamma[j] = 1;
        beta(j) = b_prop;
        // Update Xb incrementally
        for (arma::uword i = 0; i < X.n_rows; ++i)
          Xb(i) += db * X(i, j);
      }
    } else {
      // Propose deactivation: set beta_j = 0
      double db = 0.0 - b_curr;
      double ll_diff = dense_column_ll_diff(y, Xb, alpha_val, X, j, db);
      double log_ratio = ll_diff - log_prior_odds;

      if (bvs_dadj::safe_mh_accept(log_ratio)) {
        gamma[j] = 0;
        beta(j) = 0.0;
        for (arma::uword i = 0; i < X.n_rows; ++i)
          Xb(i) += db * X(i, j);
      }
    }
  }
}

// Uncollapsed Gibbs sweep for dual adjacency
template <typename NeighborFn1, typename NeighborFn2>
inline void uncollapsed_gamma_sweep_dual(
    std::vector<uint8_t> &gamma, arma::vec &beta, arma::vec &Xb,
    const arma::mat &X, const arma::vec &y, double alpha_val, double sigmasq,
    double beta0, double mu, double eta1, double eta2,
    const std::vector<int> &block_indices, NeighborFn1 for_each_neighbor1,
    NeighborFn2 for_each_neighbor2) {

  double sd_beta = std::sqrt(sigmasq);
  if (!std::isfinite(sd_beta) || sd_beta <= 0.0)
    sd_beta = 1.0;

  for (int j : block_indices) {
    int g_curr = static_cast<int>(gamma[j]);
    double b_curr = beta(j);

    double neigh1 = 0.0, neigh2 = 0.0;
    for_each_neighbor1(j, [&](int k) { neigh1 += gamma[k]; });
    for_each_neighbor2(j, [&](int k) { neigh2 += gamma[k]; });

    double log_prior_odds = mu + eta1 * neigh1 + eta2 * neigh2;

    if (g_curr == 0) {
      double b_prop = R::rnorm(beta0, sd_beta);
      b_prop = bvs_dadj::clamp_finite(b_prop, -30.0, 30.0, 0.0);
      double db = b_prop;
      double ll_diff = dense_column_ll_diff(y, Xb, alpha_val, X, j, db);
      double log_ratio = ll_diff + log_prior_odds;

      if (bvs_dadj::safe_mh_accept(log_ratio)) {
        gamma[j] = 1;
        beta(j) = b_prop;
        for (arma::uword i = 0; i < X.n_rows; ++i)
          Xb(i) += db * X(i, j);
      }
    } else {
      double db = -b_curr;
      double ll_diff = dense_column_ll_diff(y, Xb, alpha_val, X, j, db);
      double log_ratio = ll_diff - log_prior_odds;

      if (bvs_dadj::safe_mh_accept(log_ratio)) {
        gamma[j] = 0;
        beta(j) = 0.0;
        for (arma::uword i = 0; i < X.n_rows; ++i)
          Xb(i) += db * X(i, j);
      }
    }
  }
}

// Sparse uncollapsed Gibbs sweep (for sp_mat X)
// Uses CSC structure for incremental Xb updates
template <typename NeighborFn>
inline void uncollapsed_gamma_sweep_single_sparse(
    std::vector<uint8_t> &gamma, arma::vec &beta, arma::vec &Xb,
    const arma::sp_mat &X, const arma::vec &y, double alpha_val,
    double sigmasq, double beta0, double mu, double eta,
    const std::vector<int> &block_indices, NeighborFn for_each_neighbor) {

  double sd_beta = std::sqrt(sigmasq);
  if (!std::isfinite(sd_beta) || sd_beta <= 0.0)
    sd_beta = 1.0;

  const arma::uword *col_ptr = X.col_ptrs;
  const arma::uword *row_idx = X.row_indices;
  const double *xvals = X.values;

  for (int j : block_indices) {
    int g_curr = static_cast<int>(gamma[j]);
    double b_curr = beta(j);

    double neigh_sum = 0.0;
    for_each_neighbor(j, [&](int k) { neigh_sum += gamma[k]; });
    double log_prior_odds = mu + eta * neigh_sum;

    if (g_curr == 0) {
      double b_prop = R::rnorm(beta0, sd_beta);
      b_prop = bvs_dadj::clamp_finite(b_prop, -30.0, 30.0, 0.0);
      double db = b_prop;

      // Sparse column log-likelihood difference
      double ll_diff = 0.0;
      arma::uword start = col_ptr[j], end = col_ptr[j + 1];
      for (arma::uword idx = start; idx < end; ++idx) {
        int i = static_cast<int>(row_idx[idx]);
        double xij = xvals[idx];
        double psi_old = alpha_val + Xb(i);
        double psi_new = psi_old + db * xij;
        ll_diff +=
            loglik_obs_stable(y(i), psi_new) - loglik_obs_stable(y(i), psi_old);
      }

      double log_ratio = ll_diff + log_prior_odds;
      if (bvs_dadj::safe_mh_accept(log_ratio)) {
        gamma[j] = 1;
        beta(j) = b_prop;
        for (arma::uword idx = start; idx < end; ++idx)
          Xb(static_cast<int>(row_idx[idx])) += db * xvals[idx];
      }
    } else {
      double db = -b_curr;

      double ll_diff = 0.0;
      arma::uword start = col_ptr[j], end = col_ptr[j + 1];
      for (arma::uword idx = start; idx < end; ++idx) {
        int i = static_cast<int>(row_idx[idx]);
        double xij = xvals[idx];
        double psi_old = alpha_val + Xb(i);
        double psi_new = psi_old + db * xij;
        ll_diff +=
            loglik_obs_stable(y(i), psi_new) - loglik_obs_stable(y(i), psi_old);
      }

      double log_ratio = ll_diff - log_prior_odds;
      if (bvs_dadj::safe_mh_accept(log_ratio)) {
        gamma[j] = 0;
        beta(j) = 0.0;
        for (arma::uword idx = start; idx < end; ++idx)
          Xb(static_cast<int>(row_idx[idx])) += db * xvals[idx];
      }
    }
  }
}

// Dual sparse uncollapsed
template <typename NeighborFn1, typename NeighborFn2>
inline void uncollapsed_gamma_sweep_dual_sparse(
    std::vector<uint8_t> &gamma, arma::vec &beta, arma::vec &Xb,
    const arma::sp_mat &X, const arma::vec &y, double alpha_val,
    double sigmasq, double beta0, double mu, double eta1, double eta2,
    const std::vector<int> &block_indices, NeighborFn1 for_each_neighbor1,
    NeighborFn2 for_each_neighbor2) {

  double sd_beta = std::sqrt(sigmasq);
  if (!std::isfinite(sd_beta) || sd_beta <= 0.0)
    sd_beta = 1.0;

  const arma::uword *col_ptr = X.col_ptrs;
  const arma::uword *row_idx = X.row_indices;
  const double *xvals = X.values;

  for (int j : block_indices) {
    int g_curr = static_cast<int>(gamma[j]);
    double b_curr = beta(j);

    double neigh1 = 0.0, neigh2 = 0.0;
    for_each_neighbor1(j, [&](int k) { neigh1 += gamma[k]; });
    for_each_neighbor2(j, [&](int k) { neigh2 += gamma[k]; });
    double log_prior_odds = mu + eta1 * neigh1 + eta2 * neigh2;

    if (g_curr == 0) {
      double b_prop = R::rnorm(beta0, sd_beta);
      b_prop = bvs_dadj::clamp_finite(b_prop, -30.0, 30.0, 0.0);
      double db = b_prop;

      double ll_diff = 0.0;
      arma::uword start = col_ptr[j], end = col_ptr[j + 1];
      for (arma::uword idx = start; idx < end; ++idx) {
        int i = static_cast<int>(row_idx[idx]);
        double xij = xvals[idx];
        double psi_old = alpha_val + Xb(i);
        double psi_new = psi_old + db * xij;
        ll_diff +=
            loglik_obs_stable(y(i), psi_new) - loglik_obs_stable(y(i), psi_old);
      }

      if (bvs_dadj::safe_mh_accept(ll_diff + log_prior_odds)) {
        gamma[j] = 1;
        beta(j) = b_prop;
        for (arma::uword idx = start; idx < end; ++idx)
          Xb(static_cast<int>(row_idx[idx])) += db * xvals[idx];
      }
    } else {
      double db = -b_curr;

      double ll_diff = 0.0;
      arma::uword start = col_ptr[j], end = col_ptr[j + 1];
      for (arma::uword idx = start; idx < end; ++idx) {
        int i = static_cast<int>(row_idx[idx]);
        double xij = xvals[idx];
        double psi_old = alpha_val + Xb(i);
        double psi_new = psi_old + db * xij;
        ll_diff +=
            loglik_obs_stable(y(i), psi_new) - loglik_obs_stable(y(i), psi_old);
      }

      if (bvs_dadj::safe_mh_accept(ll_diff - log_prior_odds)) {
        gamma[j] = 0;
        beta(j) = 0.0;
        for (arma::uword idx = start; idx < end; ++idx)
          Xb(static_cast<int>(row_idx[idx])) += db * xvals[idx];
      }
    }
  }
}

// =============================================================================
// SECTION 6: BLOCK GAMMA UPDATE ORCHESTRATOR
// =============================================================================

// Collect all cluster members into a flat block for the uncollapsed sweep
inline std::vector<int>
flatten_clusters(const ClusterProposal &proposal) {
  std::vector<int> block;
  block.reserve(proposal.total_flipped);
  for (const auto &cluster : proposal.clusters) {
    for (int j : cluster)
      block.push_back(j);
  }
  return block;
}

// Adapter: convert arma::uvec gamma to std::vector<uint8_t> for SW functions
inline std::vector<uint8_t> gamma_to_uint8(const arma::uvec &gamma) {
  std::vector<uint8_t> g(gamma.n_elem);
  for (arma::uword i = 0; i < gamma.n_elem; ++i)
    g[i] = static_cast<uint8_t>(gamma(i));
  return g;
}

// Copy back from uint8_t to arma::uvec
inline void uint8_to_gamma(arma::uvec &gamma,
                            const std::vector<uint8_t> &g) {
  for (arma::uword i = 0; i < gamma.n_elem; ++i)
    gamma(i) = static_cast<arma::uword>(g[i]);
}

} // namespace bvs_dadj_block

#endif // BVS_DADJ_BAYESLOGIT_BLOCKPG_H
