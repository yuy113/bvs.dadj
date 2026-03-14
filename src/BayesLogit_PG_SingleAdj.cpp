// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]
#include "BayesLogit_BlockPG.h"
#include "BayesLogit_Numerics.h"
#include "BayesLogit_Sparse_Helpers.h"
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace {

// =============================================================================
// Polya-Gamma(1, z) approximate sampler (truncated series, K=200 terms)
// =============================================================================
static double sample_pg_approx(double z) {
  if (!std::isfinite(z))
    return 0.25;
  z = bvs_dadj::clamp_finite(std::abs(z) * 0.5, 0.0, 60.0, 0.0);
  const double c2 = z * z;
  const double PI2 = M_PI * M_PI;
  const double INV_2PI2 = 1.0 / (2.0 * PI2);

  double sum = 0.0;
  for (int k = 0; k < 200; ++k) {
    double g = R::rexp(1.0);
    double kh = k + 0.5;
    double den = kh * kh + c2 / PI2;
    den = bvs_dadj::clamp_finite(den, 1e-12, 1e12, 1.0);
    sum += g / den;
  }
  double out = sum * INV_2PI2;
  return bvs_dadj::clamp_finite(out, 1e-8, 1e6, 0.25);
}

// =============================================================================
// Log-likelihood for logistic regression (numerically stable)
// =============================================================================
static inline double calc_loglik(const arma::vec &y01, const arma::vec &z,
                                 double alpha, const arma::vec &offset) {
  const arma::uword n = y01.n_elem;
  double term1 = 0.0, term2 = 0.0;
  for (arma::uword i = 0; i < n; ++i) {
    double x = alpha + z(i) + offset(i);
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

// normal_pdf, approx_erf, normal_cdf, log_beta_pdf provided by BayesLogit_Sparse_Helpers.h

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
                                                  unsigned int T_max,
                                                  bool *coalesced = nullptr) {
  const unsigned int p = static_cast<unsigned int>(neighbors.size());
  unsigned int T = 2;

  Rcpp::IntegerVector x_up(p, 0);
  Rcpp::IntegerVector x_down(p, 1);
  Rcpp::NumericVector pi_up(p), pi_down(p);

  int seed_base = static_cast<int>(std::floor(R::runif(0.0, 1.0) * 2147483646.0)) + 1;

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
      if (coalesced) *coalesced = false; // IMP-2: Signal CFTP failure
      return x_up;
      for (unsigned int k = 0; k < p; ++k)
        x_down[k] = x_up[k];
    }
  }
  if (coalesced) *coalesced = true;
  return x_up;
}

// =============================================================================
// Moller MH update for eta (single adjacency), using unique-pair (j<k)
// =============================================================================
static void moller_update_eta_1(const NeighborList &neighbors,
                                const EdgeList &edges, double mu, double &eta,
                                double eta1_sd, double mu_tilde,
                                double eta1_tilde, const arma::uvec &gamma,
                                double e_eta, double f_eta, unsigned int T_max,
                                int proposal_type,
                                EtaAdapter &adapter1) {
  // Exactly cancel the Moller effect to implement the Exchange Algorithm
  mu_tilde = mu;
  eta1_tilde = eta;

  const unsigned int p = static_cast<unsigned int>(neighbors.size());

  double eta_new;
  double log_prop_ratio = 0.0;

  // Logit-transformed proposal with Vihola RAM adaptive SD.
  double eta_safe = std::max(1e-8, std::min(eta1_sd - 1e-8, eta));
  double phi = std::log(eta_safe / (eta1_sd - eta_safe));
  double phi_new = R::rnorm(phi, adapter1.sigma());
  eta_new = eta1_sd / (1.0 + std::exp(-phi_new));
  eta_new = std::max(1e-8, std::min(eta1_sd - 1e-8, eta_new));
  log_prop_ratio = std::log(eta_new) + std::log(eta1_sd - eta_new) -
                   std::log(eta_safe) - std::log(eta1_sd - eta_safe);

  bool c1 = true, c2 = true;
  Rcpp::IntegerVector omega_cur =
      proppwilson_omega_1eta(neighbors, mu, eta, T_max, &c1);
  Rcpp::IntegerVector omega_new =
      proppwilson_omega_1eta(neighbors, mu, eta_new, T_max, &c2);
  if (!c1 || !c2) { adapter1.update(0.0); return; }

  // L-3: use double to prevent int overflow for large dense graphs
  double B_R = 0.0;
  double A_om_R = 0.0, A_om_new_R = 0.0;
  double sum_om = 0.0, sum_om_new = 0.0;

  for (unsigned int j = 0; j < p; ++j) {
    sum_om += omega_cur[j];
    sum_om_new += omega_new[j];
  }
  for (const auto &e : edges) {
    Index j = e.first;
    Index k = e.second;
    B_R += (double)gamma(j) * (double)gamma(k);
    A_om_R += (double)omega_cur[j] * (double)omega_cur[k];
    A_om_new_R += (double)omega_new[j] * (double)omega_new[k];
  }

  double log_prior = log_beta_pdf(eta_new / eta1_sd, e_eta, f_eta) -
                     log_beta_pdf(eta / eta1_sd, e_eta, f_eta);

  double log_target = (eta_new - eta) * B_R + log_prior;
  double log_aux = mu_tilde * (sum_om_new - sum_om) +
                   eta1_tilde * (A_om_new_R - A_om_R);
  double log_norm = mu * (sum_om - sum_om_new) + eta * A_om_R -
                    eta_new * A_om_new_R;

  double log_MH = log_target + log_aux + log_norm + log_prop_ratio;

  double accept_prob = std::min(1.0, std::exp(std::min(0.0, log_MH)));
  if (bvs_dadj::safe_mh_accept(log_MH))
    eta = eta_new;
  adapter1.update(accept_prob);
}

} // anonymous namespace
// =============================================================================
// [[Rcpp::export]]
Rcpp::List BayesLogit_PG_SingleAdj(
    const arma::mat &X, const arma::vec &y, Rcpp::IntegerMatrix R_adj_int,
    const arma::mat &Z_dat, int niter, int burnin, double mu, double nu0,
    double sigmasq0, double alpha0, double beta0, double h, int n_mh_gamma,
    double eta1_sd, double mu_tilde, double eta1_tilde, double e_eta,
    double f_eta, unsigned int T_max, int proposal_type, int thin = 1,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0, double tau0 = 0.0, double htau = 1.5,
    Rcpp::Nullable<Rcpp::NumericVector> tau_in = R_NilValue,
    double eta1_init = 0.01, double sigmasq_init = 1.0, int block_size = 1,
    int pcg_threshold = 500, bool use_lb_gamma = true) {
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
  arma::mat tau_out(n_save, Z_dat.n_cols);

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
  double eta = eta1_init;

  // --- tau (Z_dat covariates) ---
  const arma::uword ntau = Z_dat.n_cols;
  arma::vec tau(ntau, arma::fill::zeros);
  tau.fill(tau0);
  if (tau_in.isNotNull()) {
    tau = Rcpp::as<arma::vec>(tau_in);
  }
  arma::vec Z_tau = Z_dat * tau;

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

  EtaAdapter eta1_adapter(0.5);  // Vihola RAM for eta1 proposal

  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {

    if (iter % 5000 == 0 && iter > 0) {
      Rcpp::checkUserInterrupt();
      Rcpp::Rcout << "Iter: " << iter
                  << " | Model size: " << (int)arma::accu(gamma)
                  << " | eta: " << eta << " | sigmasq: " << sigmasq << "\n";
    }

    // ===================== STEP A: PG + beta/alpha =====================
    sigmasq = bvs_dadj::clamp_finite(sigmasq, 1e-10, 1e10, 1.0);
    lin = alpha + X * beta + Z_tau;
    bvs_dadj::sanitize_vec_inplace(lin, -60.0, 60.0, 0.0);
    for (arma::uword i = 0; i < n; ++i)
      omega_pg(i) = sample_pg_approx(lin(i));
    bvs_dadj::sanitize_vec_inplace(omega_pg, 1e-8, 1e6, 1.0);

    arma::uvec active = arma::find(gamma == 1);
    const arma::uword p_active = active.n_elem;

    // HARD RULE: beta inactive = 0 always
    beta.zeros();

    if (p_active > 0) {
      arma::mat X_act = X.cols(active);

      if (block_size > 1 && (int)p_active > pcg_threshold) {
        // PCG path for large active sets
        bvs_dadj_block::PCGConfig pcg_cfg(1e-4, 200, pcg_threshold);
        arma::vec beta_act = beta.elem(active);
        bool pcg_ok = bvs_dadj_block::pcg_sample_beta(
            beta_act, X_act, omega_pg, kappa, alpha, 1.0 / sigmasq, pcg_cfg,
            &Z_tau, beta0);
        if (pcg_ok && beta_act.n_elem == p_active)
          beta.elem(active) = beta_act;
      } else {
        // Original Cholesky path
        arma::mat Xt_Om = X_act.t();
        Xt_Om.each_row() %= omega_pg.t();

        arma::mat prec_beta = Xt_Om * X_act;
        prec_beta.diag() += 1.0 / sigmasq;
        bvs_dadj::sanitize_sym_mat_inplace(prec_beta, 1e10, 1e-8);

        arma::vec z_star = kappa - omega_pg * alpha - omega_pg % Z_tau;
        bvs_dadj::sanitize_vec_inplace(z_star, -1e8, 1e8, 0.0);
        arma::vec mean_rhs =
            X_act.t() * z_star +
            (beta0 / sigmasq) * arma::ones<arma::vec>(p_active);
        bvs_dadj::sanitize_vec_inplace(mean_rhs, -1e12, 1e12, 0.0);

        arma::mat L_prec;
        bool ok = bvs_dadj::robust_chol_inplace(L_prec, prec_beta);

        if (ok) {
          arma::vec tmp = arma::solve(arma::trimatl(L_prec.t()), mean_rhs,
                                      arma::solve_opts::fast);
          arma::vec m_beta =
              arma::solve(arma::trimatu(L_prec), tmp, arma::solve_opts::fast);
          bvs_dadj::sanitize_vec_inplace(m_beta, -1e8, 1e8, 0.0);

          arma::vec zz = arma::randn<arma::vec>(p_active);
          arma::vec noise =
              arma::solve(arma::trimatu(L_prec), zz, arma::solve_opts::fast);
          bvs_dadj::sanitize_vec_inplace(noise, -1e8, 1e8, 0.0);

          beta.elem(active) = m_beta + noise;
        }
      }
    }

    // alpha
    {
      double sum_omega =
          bvs_dadj::clamp_finite(arma::accu(omega_pg), 1e-8, 1e12, (double)n);
      double prec_alpha = sum_omega + 1.0 / (h * sigmasq);
      prec_alpha = bvs_dadj::clamp_finite(prec_alpha, 1e-8, 1e12, 1.0);
      double var_alpha = 1.0 / prec_alpha;

      arma::vec resid = kappa - omega_pg % (X * beta + Z_tau);
      bvs_dadj::sanitize_vec_inplace(resid, -1e8, 1e8, 0.0);
      double mean_alpha =
          var_alpha * (arma::accu(resid) + alpha0 / (h * sigmasq));
      mean_alpha = bvs_dadj::clamp_finite(mean_alpha, -60.0, 60.0, 0.0);
      alpha = R::rnorm(mean_alpha, std::sqrt(var_alpha));
      alpha = bvs_dadj::clamp_finite(alpha, -60.0, 60.0, 0.0);
    }

    // ===================== STEP B: gamma update =========================
    // Locally-balanced gamma proposal state (Zanella 2020)
    std::vector<double> lb_score, lb_weight;
    double lb_Z = 0.0;
    LBProposalDelta lb_delta;

    if (block_size > 1) {
      // Block update: Swendsen-Wang + Uncollapsed Gibbs
      auto gamma_u8 = bvs_dadj_block::gamma_to_uint8(gamma);
      auto neigh_fn = [&](int jj, std::function<void(int)> cb) {
        for (Index k = 0; k < neighbors[jj].size(); ++k)
          cb(static_cast<int>(neighbors[jj][k]));
      };
      auto proposal = bvs_dadj_block::swendsen_wang_single(
          gamma_u8, eta, (int)p, block_size, neigh_fn);
      auto block = bvs_dadj_block::flatten_clusters(proposal);
      if (!block.empty()) {
        z = X * beta;
        bvs_dadj_block::uncollapsed_gamma_sweep_single(
            gamma_u8, beta, z, X, y01, alpha, sigmasq, beta0, mu, eta, block,
            neigh_fn);
        bvs_dadj_block::uint8_to_gamma(gamma, gamma_u8);
      }
      beta.elem(arma::find(gamma == 0)).zeros();
    } else {
      // Original single-variable MH
      z = X * beta;
      double loglik = calc_loglik(y01, z, alpha, Z_tau);
      if (!std::isfinite(loglik))
        loglik = -std::numeric_limits<double>::infinity();
      double sd_beta = std::sqrt(sigmasq);
      if (!std::isfinite(sd_beta) || sd_beta <= 0.0)
        sd_beta = 1.0;

      // Initialize LB scores before gamma scan
      arma::ivec gamma_iv;
      if (use_lb_gamma) {
        gamma_iv = arma::conv_to<arma::ivec>::from(gamma);
        init_lb_single_scores_dense(R_adj_int, (int)p, gamma_iv, mu, eta, lb_score, lb_weight, lb_Z);
      }

      for (int mh = 0; mh < n_mh_gamma; ++mh) {
        int j;
        if (use_lb_gamma) {
          j = sample_weighted_index(lb_weight, lb_Z, (int)p);
        } else {
          j = (int)std::floor(R::runif(0.0, (double)p));
          if (j >= (int)p)
            j = (int)p - 1;
        }
        arma::uword ju = (arma::uword)j;

        if (gamma(ju) == 0)
          beta(ju) = 0.0;

        int g_curr = (int)gamma(ju);
        int g_prop = 1 - g_curr;

        double b_curr = beta(ju);
        double b_prop = (g_prop == 1) ? R::rnorm(beta0, sd_beta) : 0.0;
        b_prop = bvs_dadj::clamp_finite(b_prop, -1e8, 1e8, 0.0);

        arma::vec z_prop = z + (b_prop - b_curr) * X.col(ju);
        if (!z_prop.is_finite())
          continue;
        double ll_prop = calc_loglik(y01, z_prop, alpha, Z_tau);
        if (!std::isfinite(ll_prop))
          continue;

        double diff = (double)(g_prop - g_curr);

        double neigh = 0.0;
        const auto &nb = neighbors[ju];
        for (Index k = 0; k < nb.size(); ++k) {
          neigh += (double)gamma(nb[k]);
        }

        double log_ratio = (ll_prop - loglik) + diff * (mu + eta * neigh);
        if (!std::isfinite(log_ratio))
          continue;

        // LB correction: adjust MH ratio for non-uniform proposal
        if (use_lb_gamma) {
          int delta_g = 1 - 2 * (int)gamma(ju);  // +1 if currently 0, -1 if currently 1
          gamma_iv(ju) = (int)gamma(ju);  // ensure sync
          build_lb_single_delta_dense(R_adj_int, (int)p, gamma_iv, eta, j, delta_g,
              lb_score, lb_weight, lb_Z, lb_delta);
          log_ratio += lb_delta.log_q_rev - lb_delta.log_q_fwd;
        }

        if (bvs_dadj::safe_mh_accept(log_ratio)) {
          gamma(ju) = (arma::uword)g_prop;
          beta(ju) = b_prop;
          z = z_prop;
          loglik = ll_prop;
          // Update LB state on acceptance
          if (use_lb_gamma) {
            apply_lb_delta(lb_score, lb_weight, lb_Z, lb_delta);
            gamma_iv(ju) = (int)g_prop;
          }
        }

        if (gamma(ju) == 0)
          beta(ju) = 0.0;
      }

      beta.elem(arma::find(gamma == 0)).zeros();
    }

    // ===================== STEP B.5: tau Gibbs step =======================
    if (ntau > 0) {
      arma::mat Zt_Om = Z_dat.t();
      Zt_Om.each_row() %= omega_pg.t();
      arma::mat prec_tau = Zt_Om * Z_dat;
      prec_tau.diag() += 1.0 / (htau * sigmasq);

      arma::vec z_star_tau = kappa - omega_pg * alpha - omega_pg % (X * beta);
      arma::vec mean_rhs_tau =
          Z_dat.t() * z_star_tau +
          (tau0 / (htau * sigmasq)) * arma::ones<arma::vec>(ntau);

      arma::mat L_tau;
      bool tau_ok = bvs_dadj::robust_chol_inplace(L_tau, prec_tau);
      if (tau_ok) {
        arma::vec m_tau = arma::solve(arma::trimatl(L_tau.t()), mean_rhs_tau,
                                      arma::solve_opts::fast);
        m_tau =
            arma::solve(arma::trimatu(L_tau), m_tau, arma::solve_opts::fast);
        arma::vec zz_tau = arma::randn<arma::vec>(ntau);
        arma::vec noise_tau =
            arma::solve(arma::trimatu(L_tau), zz_tau, arma::solve_opts::fast);
        tau = m_tau + noise_tau;
        Z_tau = Z_dat * tau;
      }
    }

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

      double ss_tau = 0.0;
      if (ntau > 0) {
        ss_tau = arma::accu(arma::square(tau - tau0)) / htau;
      }

      const double ss_a = (alpha - alpha0) * (alpha - alpha0) / h;

      const double shape_post = shape0 + 0.5 * (p_act + 1.0 + (double)ntau);
      const double scale_post = scale0 + 0.5 * (ss_b + ss_a + ss_tau);

      const double scale_safe = std::max(scale_post, 1e-12);
      const double g = R::rgamma(shape_post, 1.0 / scale_safe);
      if (std::isfinite(g) && g > 0.0)
        sigmasq = 1.0 / g;
      sigmasq = bvs_dadj::clamp_finite(sigmasq, 1e-10, 1e10, 1.0);
    }

    // ===================== STEP D: eta update (Moller + PW) ===============
    moller_update_eta_1(neighbors, edges, mu, eta, eta1_sd, mu_tilde,
                        eta1_tilde, gamma, e_eta, f_eta, T_max, proposal_type,
                        eta1_adapter);

    // ===================== STORE =======================================
    // Store
    if (iter >= burnin && (iter - burnin) % thin == 0) {
      int idx = (iter - burnin) / thin;
      beta_out.row(idx) = beta.t();
      gamma_out.row(idx) = gamma.t();
      eta_out(idx) = eta;
      alpha_out(idx) = alpha;
      sigmasq_out(idx) = sigmasq;
      tau_out.row(idx) = tau.t();
    }
  }

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_out, Rcpp::Named("gamma") = gamma_out,
      Rcpp::Named("eta1") = eta_out, Rcpp::Named("alpha") = alpha_out,
      Rcpp::Named("sigmasq") = sigmasq_out, Rcpp::Named("tau") = tau_out);
}
