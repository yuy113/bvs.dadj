// [[Rcpp::depends(RcppArmadillo)]]
#include "BVS_HMC_NUTS.h"
#include "BayesLogit_Sparse_Helpers.h"
#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List BayesLogit_DualNet_SparseGGM(
    const arma::sp_mat &X, const arma::vec &y, const Rcpp::IntegerVector &S_i,
    const Rcpp::IntegerVector &S_p_csc, const Rcpp::NumericVector &S_x,
    const Rcpp::NumericVector &S_diag, const Rcpp::IntegerVector &R_fix_i,
    const Rcpp::IntegerVector &R_fix_p_csc, int p_ggm, int niter, int burnin,
    double mu, double nu0, double sigmasq0, double alpha0, double beta0,
    double h, double e_eta, double f_eta, double v0_ggm, double v1_ggm,
    double pii_ggm, double eta1_sd, double eta2_sd, double mu_tilde,
    double eta1_tilde, double eta2_tilde, unsigned int T_max, int proposal_type,
    int n_mh_gamma = 5, int thin = 1,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0, const arma::mat &Z_dat = arma::mat(),
    double tau0 = 0.0, double htau = 1.0,
    Rcpp::Nullable<Rcpp::NumericVector> tau_in = R_NilValue,
    bool store_beta = true, bool store_gamma = true, bool store_Z_list = false,
    bool store_Z_pip = true,
    Rcpp::Nullable<Rcpp::NumericVector> event = R_NilValue,
    std::string outcome_type = "binary", std::string alg_type = "MH",
    double hmc_step_size = 0.1, int hmc_n_leapfrog = 10,
    int nuts_max_treedepth = 10, bool use_lb_gamma = true) {
  Rcpp::RNGScope scope;
  const bool is_continuous = bvs_dadj::outcome_is_continuous(outcome_type);
  const bool is_tte = bvs_dadj::outcome_is_tte(outcome_type);
  const bool is_count = bvs_dadj::outcome_is_count(outcome_type);

  const int n = static_cast<int>(X.n_rows);
  const int p = static_cast<int>(X.n_cols);
  if (p != p_ggm)
    Rcpp::stop("p_ggm (%d) != ncol(X) (%d)", p_ggm, p);
  if (thin < 1)
    thin = 1;
  if (n_mh_gamma < 1)
    n_mh_gamma = 1;

  arma::vec y_work;
  arma::uvec y_count;
  arma::uvec event01;
  bvs_dadj::CoxBreslowData cox_data;
  if (is_continuous) {
    y_work = y;
    if (!y_work.is_finite())
      Rcpp::stop("For outcome_type='continuous', y must be finite numeric.");
  } else if (is_tte) {
    y_work = y;
    if (!y_work.is_finite())
      Rcpp::stop("For outcome_type='TTE', survival times must be finite.");
    for (int i = 0; i < n; ++i) {
      if (y_work(i) <= 0.0)
        Rcpp::stop("For outcome_type='TTE', survival times must be > 0.");
    }
    if (event.isNull())
      Rcpp::stop("For outcome_type='TTE', event indicator must be provided.");
    arma::vec event_vec = Rcpp::as<arma::vec>(event);
    if (static_cast<int>(event_vec.n_elem) != n)
      Rcpp::stop("For outcome_type='TTE', length(event) must match nrow(X).");
    if (!bvs_dadj::normalize_binary_indicator(event_vec, event01))
      Rcpp::stop("For outcome_type='TTE', event must be in {0,1} or {-1,1}.");
    if (arma::accu(event01) < 1u)
      Rcpp::stop("For outcome_type='TTE', at least one event==1 is required.");
    cox_data = bvs_dadj::build_cox_breslow_data(y_work, event01);
  } else if (is_count) {
    if (!bvs_dadj::normalize_count_response(y, y_count))
      Rcpp::stop(
          "For outcome_type='count', y must be non-negative integer counts.");
  } else {
    if (!validate_and_convert_y(y, y_work))
      Rcpp::stop("For outcome_type='binary', y must be in {0,1} or {-1,1}.");
  }

  ConstSparseS S(S_i, S_p_csc, S_x, S_diag, p);
  ConstSparseAdj R_fix(R_fix_i, R_fix_p_csc, p);

  int d_max = 0;
  for (int j = 0; j < p; ++j) {
    int d = S.col_ptrs[j + 1] - S.col_ptrs[j];
    d_max = std::max(d_max, d);
  }

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
  if (is_tte)
    alpha = 0.0;

  // --- tau (Z_dat covariates) ---
  const arma::uword ntau = Z_dat.n_cols;
  arma::vec tau(ntau, arma::fill::zeros);
  tau.fill(tau0);
  if (tau_in.isNotNull()) {
    tau = Rcpp::as<arma::vec>(tau_in);
  }
  arma::vec Z_tau = Z_dat * tau;

  std::vector<uint8_t> Z_active_flag(S.nnz, 1);
  int n_edges = 0;
  for (int j = 0; j < p; ++j) {
    int start = S.col_ptrs[j], end = S.col_ptrs[j + 1];
    for (int idx = start; idx < end; ++idx) {
      if (S.row_idx[idx] > j)
        ++n_edges;
    }
  }

  arma::vec Xb = X * beta_vec;
  arma::vec Xb_total = Xb + Z_tau;
  double loglik = 0.0;
  const double nb_shape = 1.0;
  arma::vec w_count;
  arma::vec log_w_count;
  if (is_count) {
    w_count.set_size(n);
    log_w_count.set_size(n);
    w_count.fill(1.0);
    log_w_count.zeros();
  }
  bvs_dadj::CoxTracker cox_tracker;

  if (is_continuous) {
    loglik = calc_loglik_full_gaussian(y_work, Xb_total, alpha, sigmasq);
  } else if (is_tte) {
    cox_tracker.init(Xb_total, cox_data);
    loglik = cox_tracker.get_loglik();
  } else if (is_count) {
    refresh_count_latent_gamma(y_count, Xb_total, alpha, nb_shape, w_count,
                               log_w_count);
    loglik = calc_loglik_full_count(y_count, Xb_total, alpha, log_w_count);
  } else {
    loglik = calc_loglik_full(y_work, Xb_total, alpha);
  }

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

  std::vector<int> Z_pip_cnt;
  init_sparse_pip_counters(S, store_Z_pip, Z_pip_cnt);

  const int n_save = niter / thin;
  arma::vec eta1_out(n_save, arma::fill::zeros);
  arma::vec eta2_out(n_save, arma::fill::zeros);
  arma::vec alpha_out(n_save, arma::fill::zeros);
  arma::vec sigmasq_out(n_save, arma::fill::zeros);
  arma::mat tau_out(n_save, ntau);
  arma::vec gamma_pip_cnt(p, arma::fill::zeros);
  int n_post_gamma = 0;

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

  std::vector<double> delta_group_W;
  if (is_tte) {
    delta_group_W.assign(cox_data.group_start.size(), 0.0);
  }

  arma::vec resid;
  // Pre-compute column squared norms for MALA step sizes (all outcomes)
  arma::vec X_col_sq_sums(p, arma::fill::zeros);
  for (int j = 0; j < p; ++j) {
    double sum = 0.0;
    for (arma::uword idx = col_ptr[j]; idx < col_ptr[j + 1]; ++idx)
      sum += xvals[idx] * xvals[idx];
    X_col_sq_sums(j) = sum;
  }
  // Gradient residuals for MALA (binary: y-p_hat; count: y-mu_hat)
  arma::vec mala_resid(n, arma::fill::zeros);
  if (is_continuous) {
    resid = y_work - alpha - Xb_total;
  }

  // HMC/NUTS initialisation
  const int alg_type_int = bvs_hmc::parse_alg_type(alg_type);
  const bool use_hmc_nuts = (alg_type_int == 1 || alg_type_int == 2);
  const double hmc_target_accept = (alg_type_int == 2) ? 0.80 : 0.65;
  double hmc_epsilon = hmc_step_size;
  bvs_hmc::DualAveraging hmc_da;
  bvs_hmc::HMCNUTSDiagnostics hmc_diag;
  bvs_hmc::WindowedMassAdapter hmc_mass_adapter;
  bool hmc_eps_initialised = false;
  bool hmc_warmup_finalized = false;
  int hmc_prev_dim = -1;
  int hmc_post_burnin_adapt_left = 0;
  const int hmc_post_burnin_adapt_window = 20;
  if (use_hmc_nuts) {
    hmc_da.reset(hmc_epsilon, hmc_target_accept);
    hmc_mass_adapter.reset(p, ntau, is_tte, std::max(0, burnin));
  }

  EtaAdapter eta1_adapter(0.5);  // Vihola RAM for eta1 proposal
  EtaAdapter eta2_adapter(0.5);  // Vihola RAM for eta2 proposal

  const int total_iter = niter + burnin;
  for (int iter = 0; iter < total_iter; ++iter) {
    if (iter > 0 && (iter % 5000) == 0) {
      Rcpp::checkUserInterrupt();
      int model_size = 0;
      for (int j = 0; j < p; ++j)
        model_size += static_cast<int>(gamma[j]);
      Rcpp::Rcout << "Iter: " << iter << " | Model size: " << model_size
                  << " | Edges: " << n_edges << " | eta1: " << eta1
                  << " | eta2: " << eta2 << "\n";
    }

    sigmasq = clamp_scalar(sigmasq, SIGMASQ_MIN, SIGMASQ_MAX);
    const double sd_sig = std::sqrt(sigmasq);

    if (is_count) {
      refresh_count_latent_gamma(y_count, Xb_total, alpha, nb_shape, w_count,
                                 log_w_count);
      loglik = calc_loglik_full_count(y_count, Xb_total, alpha, log_w_count);
    }

    ggm_column_sweep_sparse(S, Z_active_flag, p, log_pii, log_1pii, lv0h, lv1h,
                            iv0, iv1, A_sub, s_ggm, noise_ggm, n_edges);

    const bool lb_gamma = use_lb_gamma;
    std::vector<double> lb_score, lb_weight;
    double lb_Z = 0.0;
    if (lb_gamma) {
      init_lb_dual_scores(S, R_fix, Z_active_flag, gamma, mu, eta1, eta2,
                          lb_score, lb_weight, lb_Z);
    }

    for (int mh = 0; mh < n_mh_gamma; ++mh) {
      int j = 0;
      if (lb_gamma) {
        j = sample_weighted_index(lb_weight, lb_Z, p);
      } else {
        j = static_cast<int>(std::floor(R::runif(0.0, static_cast<double>(p))));
        if (j >= p)
          j = p - 1;
      }

      int g_curr = static_cast<int>(gamma[j]);
      int g_prop = 1 - g_curr;
      const int delta_g = g_prop - g_curr;
      double b_curr = beta_vec(j);
      double b_prop = (g_prop == 1) ? R::rnorm(beta0, sd_sig) : 0.0;
      double db = b_prop - b_curr;

      double ll_diff = 0.0;
      if (is_continuous) {
        ll_diff = column_ll_diff_gaussian_resid(
            resid, X_col_sq_sums(j), col_ptr, row_idx, xvals, j, db, sigmasq);
      } else if (is_tte) {
        ll_diff = cox_tracker.propose_diff(col_ptr, row_idx, xvals, j, db,
                                           delta_group_W);
      } else if (is_count) {
        ll_diff = column_ll_diff_count(y_count, Xb_total, alpha, log_w_count,
                                       col_ptr, row_idx, xvals, j, db);
      } else {
        ll_diff = column_ll_diff(y_work, Xb_total, alpha, col_ptr, row_idx,
                                 xvals, j, db);
      }
      double neigh_dyn = 0.0;
      int start = S.col_ptrs[j], end = S.col_ptrs[j + 1];
      for (int idx = start; idx < end; ++idx) {
        if (Z_active_flag[idx])
          neigh_dyn += static_cast<int>(gamma[S.row_idx[idx]]);
      }
      double neigh_fix = 0.0;
      int r_start = R_fix.col_ptrs[j], r_end = R_fix.col_ptrs[j + 1];
      for (int idx = r_start; idx < r_end; ++idx) {
        neigh_fix += static_cast<int>(gamma[R_fix.row_idx[idx]]);
      }

      double ising = static_cast<double>(delta_g) *
                     (mu + eta1 * neigh_dyn + eta2 * neigh_fix);
      double lmh = ll_diff + ising;

      LBProposalDelta lb_delta;
      if (lb_gamma) {
        build_lb_dual_delta(S, R_fix, Z_active_flag, gamma, eta1, eta2, j,
                            delta_g, lb_score, lb_weight, lb_Z, lb_delta);
        lmh += (lb_delta.log_q_rev - lb_delta.log_q_fwd);
      }

      if (bvs_dadj::safe_mh_accept(lmh)) {
        gamma[j] = static_cast<uint8_t>(g_prop);
        beta_vec(j) = b_prop;
        if (is_continuous) {
          apply_column_update_resid(resid, col_ptr, row_idx, xvals, j, -db);
          loglik += ll_diff;
        } else if (is_tte) {
          cox_tracker.apply_diff(col_ptr, row_idx, xvals, j, db, ll_diff,
                                 delta_group_W);
          loglik = cox_tracker.get_loglik();
        } else {
          apply_column_update(Xb, col_ptr, row_idx, xvals, j, db);
          apply_column_update(Xb_total, col_ptr, row_idx, xvals, j, db);
          loglik += ll_diff;
        }
        if (g_prop == 1)
          activate_gamma(j, active_idx, active_pos);
        else
          deactivate_gamma(j, active_idx, active_pos);
        if (lb_gamma) {
          apply_lb_delta(lb_score, lb_weight, lb_Z, lb_delta);
        }
      }
    }

    const int na = static_cast<int>(active_idx.size());
    if (use_hmc_nuts && !is_continuous && !is_count) {
      auto compute = [&](const arma::vec &q) -> std::pair<double, arma::vec> {
        if (is_tte) {
          return bvs_hmc::nlp_grad_tte_joint_sparse(
              static_cast<arma::uword>(n), col_ptr, row_idx, xvals, Z_dat,
              cox_data, active_idx, q, beta0, tau0, htau, nu0, sigmasq0);
        }
        return bvs_hmc::nlp_grad_binary_joint_sparse(
            static_cast<arma::uword>(n), col_ptr, row_idx, xvals, y_work, Z_dat,
            active_idx, q, beta0, alpha0, tau0, h, htau, nu0, sigmasq0);
      };

      int d_act = na;
      int ntau_local = (int)Z_dat.n_cols;
      int dim_q = d_act + ntau_local + 1 + (is_tte ? 0 : 1);
      arma::vec q_curr(dim_q);

      for (int k = 0; k < d_act; ++k)
        q_curr(k) = beta_vec(active_idx[k]);
      if (!is_tte) {
        q_curr(d_act) = alpha;
        for (int k = 0; k < ntau_local; ++k)
          q_curr(d_act + 1 + k) = tau(k);
        q_curr(d_act + 1 + ntau_local) = std::log(sigmasq);
      } else {
        for (int k = 0; k < ntau_local; ++k)
          q_curr(d_act + k) = tau(k);
        q_curr(d_act + ntau_local) = std::log(sigmasq);
      }

      // Build mass matrix for preconditioning
      bvs_hmc::DiagMassMatrix mass;
      hmc_mass_adapter.build_joint_mass_matrix(mass, active_idx, sigmasq, h,
                                               htau, nu0);

      // Persistent step-size state across common gamma flips.
      const bool in_warmup = (iter < burnin);
      const int dim_jump =
          (hmc_prev_dim >= 0) ? std::abs(dim_q - hmc_prev_dim) : 0;
      bool need_reinit = bvs_hmc::should_reinit_step_size(
          hmc_eps_initialised, hmc_prev_dim, dim_q, in_warmup);

      if (!in_warmup && hmc_prev_dim >= 0 && dim_jump > 2 && hmc_post_burnin_adapt_left == 0) {
        hmc_post_burnin_adapt_left = hmc_post_burnin_adapt_window;
        hmc_da.reset(bvs_hmc::clamp_epsilon(hmc_epsilon), hmc_target_accept);
      }
      hmc_prev_dim = dim_q;

      if (need_reinit && dim_q > 0) {
        auto res0 = compute(q_curr);
        if (res0.second.is_finite()) {
          hmc_epsilon = bvs_hmc::find_reasonable_epsilon(q_curr, res0.second,
                                                         res0.first, compute,
                                                         mass);
          hmc_da.reset(hmc_epsilon, hmc_target_accept);
        }
        hmc_eps_initialised = true;
      }

      arma::vec q_new = q_curr;
      auto res0 = compute(q_new);
      double nlp0 = res0.first;
      arma::vec grad0 = res0.second;

      bvs_hmc::HMCSamplingStats step_stats;
      double accept_prob;
      if (alg_type_int == 1) {
        accept_prob = bvs_hmc::hmc_step(q_new, nlp0, grad0, hmc_epsilon,
                                        hmc_n_leapfrog, compute, mass,
                                        &step_stats);
      } else {
        accept_prob = bvs_hmc::nuts_step(q_new, nlp0, grad0, hmc_epsilon,
                                         nuts_max_treedepth, compute, mass,
                                         &step_stats);
      }
      hmc_diag.record(hmc_epsilon, step_stats);

      if (step_stats.accepted) {
        // Write back the clamped joint state before rebuilding cached predictors.
        for (int k = 0; k < d_act; ++k) {
          int j = active_idx[k];
          double beta_new = bvs_dadj::clamp_finite(q_new(k), -30.0, 30.0, 0.0);
          double db = beta_new - beta_vec(j);
          if (std::abs(db) > 0.0) {
            beta_vec(j) = beta_new;
            apply_column_update(Xb, col_ptr, row_idx, xvals, j, db);
          }
        }

        if (!is_tte) {
          alpha = bvs_dadj::clamp_finite(q_new(d_act), -30.0, 30.0, 0.0);
          for (int k = 0; k < ntau_local; ++k)
            tau(k) =
                bvs_dadj::clamp_finite(q_new(d_act + 1 + k), -30.0, 30.0, 0.0);
          sigmasq = std::exp(bvs_dadj::clamp_finite(
              q_new(d_act + 1 + ntau_local), -23.0, 9.2, 0.0));
        } else {
          for (int k = 0; k < ntau_local; ++k)
            tau(k) = bvs_dadj::clamp_finite(q_new(d_act + k), -30.0, 30.0, 0.0);
          sigmasq = std::exp(bvs_dadj::clamp_finite(
              q_new(d_act + ntau_local), -23.0, 9.2, 0.0));
        }
        Z_tau = Z_dat * tau;
        Xb_total = Xb + Z_tau;

        if (is_tte) {
          cox_tracker.init(Xb_total, cox_data);
          loglik = cox_tracker.get_loglik();
        } else {
          loglik = calc_loglik_full(y_work, Xb_total, alpha);
        }
      }

      if (in_warmup) {
        hmc_mass_adapter.observe(active_idx, beta_vec, alpha, tau,
                                 std::log(std::max(sigmasq, 1e-10)));
        hmc_epsilon = hmc_da.update(accept_prob);
      } else {
        if (!hmc_warmup_finalized) {
          hmc_epsilon = hmc_da.final_epsilon();
          hmc_warmup_finalized = true;
        }
        if (hmc_post_burnin_adapt_left > 0) {
          hmc_epsilon = hmc_da.update(accept_prob);
          --hmc_post_burnin_adapt_left;
          if (hmc_post_burnin_adapt_left == 0)
            hmc_epsilon = hmc_da.final_epsilon();
        }
      }

      if (step_stats.accepted && !is_tte) {
        for (int i = 0; i < n; ++i)
          mala_resid(i) =
              y_work(i) - 1.0 / (1.0 + std::exp(-(alpha + Xb_total(i))));
      }
    } else if (!is_continuous) {
      if (is_tte) {
        // --- Fisher information-scaled RW for TTE (sparse) ---
        arma::vec cox_H;
        cox_tracker.compute_H_vec(cox_H);
        const arma::vec &W_cox = cox_tracker.get_W();
        for (int k = 0; k < na; ++k) {
          int j = active_idx[k];
          double b_curr = beta_vec(j);
          double I_jj = 0.0;
          for (arma::uword idx = col_ptr[j]; idx < col_ptr[j + 1]; ++idx) {
            int i = static_cast<int>(row_idx[idx]);
            I_jj += xvals[idx] * xvals[idx] * W_cox(i) * cox_H(i);
          }
          double step = (I_jj > 1e-12) ? 1.0 / std::sqrt(I_jj) : sd_sig;
          step = std::min(step, 2.0 * sd_sig);
          double db = R::rnorm(0.0, step);
          double b_prop = b_curr + db;
          double ll_diff = cox_tracker.propose_diff(col_ptr, row_idx, xvals, j,
                                                    db, delta_group_W);
          double d_curr = b_curr - beta0;
          double d_prop = b_prop - beta0;
          if (bvs_dadj::safe_mh_accept(
                  ll_diff -
                  0.5 * (d_prop * d_prop - d_curr * d_curr) / sigmasq)) {
            beta_vec(j) = b_prop;
            cox_tracker.apply_diff(col_ptr, row_idx, xvals, j, db, ll_diff,
                                   delta_group_W);
            loglik = cox_tracker.get_loglik();
            cox_tracker.compute_H_vec(cox_H);
          }
        }
      } else {
        // --- Component-wise sparse MALA for binary and count ---
        // Refresh mala_resid from current state
        for (int i = 0; i < n; ++i) {
          double eta_i = alpha + Xb_total(i);
          if (is_count) {
            double mu_i = std::exp(bvs_dadj::clamp_finite(
                eta_i + log_w_count(i), -50.0, 50.0, 0.0));
            mala_resid(i) = static_cast<double>(y_count(i)) - mu_i;
          } else {
            mala_resid(i) = y_work(i) - 1.0 / (1.0 + std::exp(-eta_i));
          }
        }
        for (int k = 0; k < na; ++k) {
          int j = active_idx[k];
          double b_curr = beta_vec(j);
          double g_j = -(b_curr - beta0) / sigmasq;
          for (arma::uword idx = col_ptr[j]; idx < col_ptr[j + 1]; ++idx)
            g_j += xvals[idx] * mala_resid(static_cast<int>(row_idx[idx]));
          double h_j = std::min(0.5 * sigmasq / (X_col_sq_sums(j) + 1.0), 1.0);
          double b_prop =
              b_curr + 0.5 * h_j * g_j + std::sqrt(h_j) * R::rnorm(0.0, 1.0);
          double db = b_prop - b_curr;
          double ll_diff = 0.0;
          double g_j_back = -(b_prop - beta0) / sigmasq;
          for (arma::uword idx = col_ptr[j]; idx < col_ptr[j + 1]; ++idx) {
            int i = static_cast<int>(row_idx[idx]);
            double old_eta = alpha + Xb_total(i);
            double new_eta = old_eta + db * xvals[idx];
            if (is_count) {
              double old_mu = std::exp(bvs_dadj::clamp_finite(
                  old_eta + log_w_count(i), -50.0, 50.0, 0.0));
              double new_mu = std::exp(bvs_dadj::clamp_finite(
                  new_eta + log_w_count(i), -50.0, 50.0, 0.0));
              ll_diff += (static_cast<double>(y_count(i)) *
                              (new_eta + log_w_count(i)) -
                          new_mu) -
                         (static_cast<double>(y_count(i)) *
                              (old_eta + log_w_count(i)) -
                          old_mu);
              g_j_back +=
                  (static_cast<double>(y_count(i)) - new_mu) * xvals[idx];
            } else {
              double p_new = 1.0 / (1.0 + std::exp(-new_eta));
              double old_ll =
                  y_work(i) * old_eta -
                  (old_eta > 0.0 ? old_eta + std::log1p(std::exp(-old_eta))
                                 : std::log1p(std::exp(old_eta)));
              double new_ll =
                  y_work(i) * new_eta -
                  (new_eta > 0.0 ? new_eta + std::log1p(std::exp(-new_eta))
                                 : std::log1p(std::exp(new_eta)));
              ll_diff += new_ll - old_ll;
              g_j_back += (y_work(i) - p_new) * xvals[idx];
            }
          }
          double b_back = b_prop + 0.5 * h_j * g_j_back;
          double d_curr = b_curr - beta0;
          double d_prop = b_prop - beta0;
          double lq_fwd =
              -0.5 * std::pow(b_prop - (b_curr + 0.5 * h_j * g_j), 2) / h_j;
          double lq_bwd = -0.5 * std::pow(b_curr - b_back, 2) / h_j;
          double lmh = ll_diff -
                       0.5 * (d_prop * d_prop - d_curr * d_curr) / sigmasq +
                       (lq_bwd - lq_fwd);
          if (bvs_dadj::safe_mh_accept(lmh)) {
            beta_vec(j) = b_prop;
            for (arma::uword idx = col_ptr[j]; idx < col_ptr[j + 1]; ++idx) {
              int i = static_cast<int>(row_idx[idx]);
              Xb(i) += db * xvals[idx];
              Xb_total(i) += db * xvals[idx];
              double new_eta = alpha + Xb_total(i);
              if (is_count) {
                double mu_i = std::exp(bvs_dadj::clamp_finite(
                    new_eta + log_w_count(i), -50.0, 50.0, 0.0));
                mala_resid(i) = static_cast<double>(y_count(i)) - mu_i;
              } else {
                mala_resid(i) = y_work(i) - 1.0 / (1.0 + std::exp(-new_eta));
              }
            }
            loglik += ll_diff;
          }
        }
      }
    } else {
      for (int k = 0; k < na; ++k) {
        int j = active_idx[k];
        double b_curr = beta_vec(j);
        arma::uword start = col_ptr[j], end = col_ptr[j + 1];
        double sum_x_resid = 0.0;
        for (arma::uword idx = start; idx < end; ++idx) {
          sum_x_resid += xvals[idx] * resid(row_idx[idx]);
        }
        double denom = X_col_sq_sums(j) + 1.0;
        double mean = (sum_x_resid + b_curr * X_col_sq_sums(j) + beta0) / denom;
        double b_new = R::rnorm(mean, std::sqrt(sigmasq / denom));
        double db = b_new - b_curr;
        if (std::abs(db) > 0.0) {
          beta_vec(j) = b_new;
          apply_column_update_resid(resid, col_ptr, row_idx, xvals, j, db);
        }
      }
      loglik = -0.5 * n * std::log(2.0 * M_PI * sigmasq) -
               0.5 * arma::dot(resid, resid) / sigmasq;
    }

    // Alpha — MALA for binary/count; conjugate Gibbs for continuous.
    {
      if (!is_continuous && !is_tte && !use_hmc_nuts) {
        double sum_resid = 0.0;
        for (int i = 0; i < n; ++i) {
          double eta_i = alpha + Xb_total(i);
          if (is_count) {
            double mu_i = std::exp(bvs_dadj::clamp_finite(
                eta_i + log_w_count(i), -50.0, 50.0, 0.0));
            mala_resid(i) = static_cast<double>(y_count(i)) - mu_i;
          } else {
            mala_resid(i) = y_work(i) - 1.0 / (1.0 + std::exp(-eta_i));
          }
          sum_resid += mala_resid(i);
        }
        double g_alpha = sum_resid - (alpha - alpha0) / (h * sigmasq);
        double h_alpha =
            std::min(0.5 * h * sigmasq / ((double)n + 1.0 / h), 1.0);
        double a_prop = alpha + 0.5 * h_alpha * g_alpha +
                        std::sqrt(h_alpha) * R::rnorm(0.0, 1.0);
        double ll_prop_alpha = 0.0;
        double g_alpha_back = -(a_prop - alpha0) / (h * sigmasq);
        for (int i = 0; i < n; ++i) {
          double new_eta = a_prop + Xb_total(i);
          if (is_count) {
            double mu_i = std::exp(bvs_dadj::clamp_finite(
                new_eta + log_w_count(i), -50.0, 50.0, 0.0));
            ll_prop_alpha +=
                static_cast<double>(y_count(i)) * (new_eta + log_w_count(i)) -
                mu_i;
            g_alpha_back += static_cast<double>(y_count(i)) - mu_i;
          } else {
            double p_i = 1.0 / (1.0 + std::exp(-new_eta));
            ll_prop_alpha +=
                y_work(i) * new_eta -
                (new_eta > 0.0 ? new_eta + std::log1p(std::exp(-new_eta))
                               : std::log1p(std::exp(new_eta)));
            g_alpha_back += y_work(i) - p_i;
          }
        }
        double a_back = a_prop + 0.5 * h_alpha * g_alpha_back;
        double d_curr = alpha - alpha0;
        double d_prop = a_prop - alpha0;
        double lq_fwd =
            -0.5 * std::pow(a_prop - (alpha + 0.5 * h_alpha * g_alpha), 2) /
            h_alpha;
        double lq_bwd = -0.5 * std::pow(alpha - a_back, 2) / h_alpha;
        double lmh = (ll_prop_alpha - loglik) -
                     0.5 * (d_prop * d_prop - d_curr * d_curr) / (h * sigmasq) +
                     (lq_bwd - lq_fwd);
        if (bvs_dadj::safe_mh_accept(lmh)) {
          alpha = a_prop;
          loglik = ll_prop_alpha;
          for (int i = 0; i < n; ++i) {
            double new_eta = alpha + Xb_total(i);
            if (is_count) {
              double mu_i = std::exp(bvs_dadj::clamp_finite(
                  new_eta + log_w_count(i), -50.0, 50.0, 0.0));
              mala_resid(i) = static_cast<double>(y_count(i)) - mu_i;
            } else {
              mala_resid(i) = y_work(i) - 1.0 / (1.0 + std::exp(-new_eta));
            }
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
    }

    // SigmaSq — exact Inverse-Gamma Gibbs for all outcome types.
    // For non-continuous outcomes the likelihood does not depend on sigmasq.
    if (!use_hmc_nuts) {
      if (!is_continuous) {
        double ss = 0.0;
        for (int j : active_idx) {
          double d = beta_vec(j) - beta0;
          ss += d * d;
        }
        double ss_tau = 0.0;
        for (arma::uword j = 0; j < ntau; ++j) {
          double d = tau(j) - tau0;
          ss_tau += d * d;
        }
        const double da = alpha - alpha0;
        const double shape_post =
            0.5 * nu0 + 0.5 * ((double)active_idx.size() + 1.0 + (double)ntau);
        const double scale_post =
            0.5 * nu0 * sigmasq0 + 0.5 * (ss + da * da / h + ss_tau / htau);
        const double gdraw =
            R::rgamma(shape_post, 1.0 / std::max(scale_post, 1e-12));
        if (std::isfinite(gdraw) && gdraw > 0.0)
          sigmasq = std::max(1e-10, 1.0 / gdraw);
      } else {
        double ss_beta = 0.0;
        for (int j : active_idx) {
          double d = beta_vec(j) - beta0;
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
        double gdraw = R::rgamma(shape_post, 1.0 / scale_post);
        if (std::isfinite(gdraw) && gdraw > 0.0) {
          sigmasq = std::max(1e-10, 1.0 / gdraw);
        }
        loglik =
            -0.5 * n * std::log(2.0 * M_PI * sigmasq) - 0.5 * sse / sigmasq;
      }
    }

    // --- Tau MH step ---
    if (ntau > 0 && !use_hmc_nuts) {
      if (!is_continuous) {
        arma::vec tau_prop(ntau);
        arma::vec Z_tau_prop(n);
        arma::vec Xb_total_prop(n);
        for (arma::uword j = 0; j < ntau; ++j) {
          tau_prop(j) = R::rnorm(tau(j), std::sqrt(htau * sigmasq));
        }
        Z_tau_prop = Z_dat * tau_prop;
        for (arma::uword i = 0; i < n; ++i) {
          Xb_total_prop(i) = Xb_total(i) - Z_tau(i) + Z_tau_prop(i);
        }
        // Full recalculation for tau update since tau modifies the whole linear
        // predictor
        double ll_prop = 0.0;
        if (is_tte) {
          ll_prop = bvs_dadj::cox_loglik_breslow(Xb_total_prop, cox_data);
        } else if (is_count) {
          ll_prop = calc_loglik_full_count(y_count, Xb_total_prop, alpha,
                                           log_w_count);
        } else {
          ll_prop = calc_loglik_full(y_work, Xb_total_prop, alpha);
        }
        double pr_curr =
            -0.5 * arma::accu(arma::square(tau - tau0)) / (htau * sigmasq);
        double pr_prop =
            -0.5 * arma::accu(arma::square(tau_prop - tau0)) / (htau * sigmasq);
        double lmh = (ll_prop - loglik) + (pr_prop - pr_curr);
        if (bvs_dadj::safe_mh_accept(lmh)) {
          tau = tau_prop;
          Z_tau = Z_tau_prop;
          Xb_total = Xb_total_prop;
          if (is_tte) {
            cox_tracker.init(Xb_total, cox_data);
            loglik = cox_tracker.get_loglik();
          } else {
            loglik = ll_prop;
          }
        }
      } else {
        for (arma::uword j = 0; j < ntau; ++j) {
          arma::vec zj = Z_dat.col(j);
          double tj_old = tau(j);
          double denom = arma::dot(zj, zj) + 1.0 / htau;
          double sum_z_resid = arma::dot(zj, resid);
          double mean =
              (sum_z_resid + tj_old * arma::dot(zj, zj) + tau0 / htau) / denom;
          double tj_new = R::rnorm(mean, std::sqrt(sigmasq / denom));
          double dt = tj_new - tj_old;
          if (std::abs(dt) > 0.0) {
            tau(j) = tj_new;
            resid -= dt * zj;
          }
        }
        loglik = -0.5 * n * std::log(2.0 * M_PI * sigmasq) -
                 0.5 * arma::dot(resid, resid) / sigmasq;
      }
    }

    moller_update_dual_sparse(
        S, R_fix, Z_active_flag, p, mu, eta1, eta2, eta1_sd, eta2_sd, mu_tilde,
        eta1_tilde, eta2_tilde, gamma, e_eta, f_eta, T_max, proposal_type,
        pw_up, pw_dn, pw_om1, pw_om2, pw_om1n, pw_om2n,
        eta1_adapter, eta2_adapter);

    maybe_store_sparse_state(
        iter, burnin, thin, n_save, store_beta, store_gamma, store_Z_list,
        active_idx, beta_vec, S, Z_active_flag, eta1_out, eta2_out, true,
        alpha_out, sigmasq_out, eta1, eta2, alpha, sigmasq, beta_out_list,
        gamma_out_list, Z_list, edge_r, edge_c);

    if (iter >= burnin && (iter - burnin) % thin == 0) {
      int idx = (iter - burnin) / thin;
      tau_out.row(idx) = tau.t();
    }

    if (iter >= burnin) {
      accumulate_sparse_pip(S, Z_active_flag, store_Z_pip, Z_pip_cnt);
      for (int j = 0; j < p; ++j)
        gamma_pip_cnt(j) += static_cast<double>(gamma[j]);
      ++n_post_gamma;
    }
  }

  double n_post = static_cast<double>(total_iter - burnin);
  Rcpp::List pip_trip =
      build_sparse_pip_triplet(S, store_Z_pip, Z_pip_cnt, n_post);

  SEXP beta_out_sexp =
      store_beta ? static_cast<SEXP>(beta_out_list) : R_NilValue;
  SEXP gamma_out_sexp =
      store_gamma ? static_cast<SEXP>(gamma_out_list) : R_NilValue;
  SEXP z_list_sexp = store_Z_list ? static_cast<SEXP>(Z_list) : R_NilValue;

  arma::vec gamma_pip_out =
      (n_post_gamma > 0) ? (gamma_pip_cnt / static_cast<double>(n_post_gamma))
                         : arma::vec(p, arma::fill::zeros);

  Rcpp::List result = Rcpp::List::create(
      Rcpp::Named("beta") = beta_out_sexp,
      Rcpp::Named("gamma") = gamma_out_sexp, Rcpp::Named("eta1") = eta1_out,
      Rcpp::Named("eta2") = eta2_out, Rcpp::Named("alpha") = alpha_out,
      Rcpp::Named("sigmasq") = sigmasq_out, Rcpp::Named("tau") = tau_out,
      Rcpp::Named("Z_list") = z_list_sexp,
      Rcpp::Named("gamma_pip") = gamma_pip_out,
      Rcpp::Named("Z_pip_row") = pip_trip["row"],
      Rcpp::Named("Z_pip_col") = pip_trip["col"],
      Rcpp::Named("Z_pip_val") = pip_trip["val"], Rcpp::Named("p") = p,
      Rcpp::Named("n") = n);
  if (use_hmc_nuts)
    result["hmc_nuts_diagnostics"] = hmc_diag.to_list(hmc_epsilon);
  return result;
}
