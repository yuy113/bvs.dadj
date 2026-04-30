// [[Rcpp::depends(RcppArmadillo)]]
#include "BayesLogit_BlockPG.h"
#include "BayesLogit_Sparse_Helpers.h"
#include <RcppArmadillo.h>

// [[Rcpp::export]]
Rcpp::List BayesLogit_PG_DualNet_SparseGGM(
    const arma::sp_mat &X, const arma::vec &y, const Rcpp::IntegerVector &S_i,
    const Rcpp::IntegerVector &S_p_csc, const Rcpp::NumericVector &S_x,
    const Rcpp::NumericVector &S_diag, const Rcpp::IntegerVector &R_fix_i,
    const Rcpp::IntegerVector &R_fix_p_csc, int p_ggm, int niter, int burnin,
    double mu, double nu0, double sigmasq0, double alpha0, double beta0,
    double h, int n_mh_gamma, double v0_ggm, double v1_ggm, double pii_ggm,
    double eta1_sd, double eta2_sd, double mu_tilde, double eta1_tilde,
    double eta2_tilde, double e_eta, double f_eta, unsigned int T_max,
    int proposal_type, int thin = 1,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0, const arma::mat &Z_dat = arma::mat(),
    double tau0 = 0.0, double htau = 1.0,
    Rcpp::Nullable<Rcpp::NumericVector> tau_in = R_NilValue,
    bool store_beta = true, bool store_gamma = true, bool store_Z_list = false,
    bool store_Z_pip = true, int block_size = 1, int pcg_threshold = 500,
    bool use_lb_gamma = true,
    bool use_sssl = false,
    double v0_sssl = 0.01,
    double v1_sssl = 1.0,
    bool use_cftp = false) {
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
  double eta1 = eta1_sd * 0.5;
  double eta2 = eta2_sd * 0.5;

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

  // M-4: pg_rng removed; sample_pg() uses R::rexp() internally for reproducibility.
  static std::mt19937 pg_rng_dummy(0); // placeholder — ignored by sample_pg()

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

    if (use_sssl) {
      ggm_column_sweep_sparse_sssl(S, Z_active_flag, p, log_pii, log_1pii,
                                   v0_sssl, v1_sssl, A_sub, s_ggm, noise_ggm,
                                   n_edges);
    } else {
      ggm_column_sweep_sparse(S, Z_active_flag, p, log_pii, log_1pii, lv0h,
                              lv1h, iv0, iv1, A_sub, s_ggm, noise_ggm, n_edges);
    }

    {
      for (int i = 0; i < n; ++i) {
        const double lp =
            clamp_scalar(alpha + Xb(i) + Z_tau(i), -LINPRED_CLIP, LINPRED_CLIP);
        omega_pg(i) = sample_pg(lp, pg_rng_dummy);
      }
    }

    const int p_act = static_cast<int>(active_idx.size());
    bool beta_refreshed = false;
    if (p_act > 0) {
      if (block_size > 1 && p_act > pcg_threshold) {
        // PCG path for large active sets (sparse X)
        bvs_dadj_block::PCGConfig pcg_cfg(1e-4, 200, pcg_threshold);
        arma::vec beta_act(p_act);
        for (int k = 0; k < p_act; ++k)
          beta_act(k) = beta_vec(active_idx[k]);
        bool pcg_ok = bvs_dadj_block::pcg_sample_beta_sparse(
            beta_act, X, active_idx, omega_pg, kappa, alpha, 1.0 / sigmasq,
            pcg_cfg, &Z_tau, beta0);
        if (pcg_ok && static_cast<int>(beta_act.n_elem) == p_act) {
          clamp_vec_inplace(beta_act, -BETA_ABS_MAX, BETA_ABS_MAX);
          beta_vec.zeros();
          for (int k = 0; k < p_act; ++k)
            beta_vec(active_idx[k]) = beta_act(k);
          Xb = X * beta_vec;
          beta_refreshed = true;
        }
      } else {
        // Original Cholesky path
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

        arma::vec z_star = kappa - omega_pg * alpha - omega_pg % Z_tau;
        // FIX SPG-3: Include beta0 prior mean in RHS
        arma::vec rhs = X_act.t() * z_star +
            (beta0 / sigmasq) * arma::ones<arma::vec>(p_act);

        arma::mat L;
        bool chol_ok = bvs_dadj::robust_chol_inplace(L, prec);

        if (chol_ok) {
          arma::vec mb;
          bool solve_ok =
              arma::solve(mb, arma::trimatl(L.t()), rhs,
                          arma::solve_opts::fast + arma::solve_opts::no_approx);
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
              if (finite_vec(bd)) {
                clamp_vec_inplace(bd, -BETA_ABS_MAX, BETA_ABS_MAX);
                beta_vec.zeros();
                for (int k = 0; k < p_act; ++k)
                  beta_vec(active_idx[k]) = bd(k);
                Xb = X_act * bd;
                beta_refreshed = true;
              }
            }
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
      arma::vec resid = kappa - omega_pg % Xb - omega_pg % Z_tau;
      alpha = R::rnorm(var_a * (arma::accu(resid) + alpha0 / (h * sigmasq)),
                       std::sqrt(var_a));
    }

    if (block_size > 1) {
      // --- Block Swendsen-Wang + Uncollapsed Gibbs (dual sparse) ---
      auto neigh_dyn_fn = [&](int jj, std::function<void(int)> cb) {
        int s = S.col_ptrs[jj], e = S.col_ptrs[jj + 1];
        for (int idx = s; idx < e; ++idx) {
          if (Z_active_flag[idx])
            cb(S.row_idx[idx]);
        }
      };
      auto neigh_fix_fn = [&](int jj, std::function<void(int)> cb) {
        int rs = R_fix.col_ptrs[jj], re = R_fix.col_ptrs[jj + 1];
        for (int idx = rs; idx < re; ++idx) {
          cb(R_fix.row_idx[idx]);
        }
      };
      // R2-FIX: argument order had been (gamma, p, eta1, eta2, ...) which the
      // compiler silently accepted via int↔double conversions. The intended
      // signature is (gamma, eta1, eta2, p, block_size, ...) — passing p as
      // eta1 made the bond probability ~1 and truncated p to (int)eta2 == 0,
      // which made the SW step a complete no-op for sparse dual GGM.
      auto clusters = bvs_dadj_block::swendsen_wang_dual(
          gamma, eta1, eta2, (int)p, block_size, neigh_dyn_fn, neigh_fix_fn);
      auto flat = bvs_dadj_block::flatten_clusters(clusters);

      // R2-FIX: pass Z_tau through so the block likelihood ratio uses the
      // full linear predictor alpha + X*beta + Z*tau (not just alpha + X*beta).
      bvs_dadj_block::uncollapsed_gamma_sweep_dual_sparse(
          gamma, beta_vec, Xb, X, y01, alpha, sigmasq, beta0, mu, eta1, eta2,
          flat, neigh_dyn_fn, neigh_fix_fn, &Z_tau);

      // Rebuild active_idx / active_pos from scratch
      active_idx.clear();
      active_pos.assign(p, -1);
      for (int j = 0; j < p; ++j) {
        if (gamma[j] == 1) {
          active_pos[j] = static_cast<int>(active_idx.size());
          active_idx.push_back(j);
        }
      }
    } else {
      // --- Original single-variable MH ---
      // FIX: Include Z_tau in linear predictor for likelihood computation
      arma::vec Xb_full = Xb + Z_tau;
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
        b_prop = clamp_scalar(b_prop, -BETA_ABS_MAX, BETA_ABS_MAX);
        double db = b_prop - b_curr;

        // FIX: Use Xb_full (= Xb + Z_tau) instead of Xb for correct loglik
        double ll_diff =
            column_ll_diff(y01, Xb_full, alpha, col_ptr, row_idx, xvals, j, db);
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
          apply_column_update(Xb, col_ptr, row_idx, xvals, j, db);
          apply_column_update(Xb_full, col_ptr, row_idx, xvals, j, db);
          if (g_prop == 1)
            activate_gamma(j, active_idx, active_pos);
          else
            deactivate_gamma(j, active_idx, active_pos);
          if (lb_gamma) {
            apply_lb_delta(lb_score, lb_weight, lb_Z, lb_delta);
          }
        }
      }
    }

    // SigmaSq — exact Inverse-Gamma Gibbs.
    // PG logistic likelihood does not depend on sigmasq; the conjugate
    // IG posterior for the (beta, alpha, tau) priors is exact.
    {
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
        sigmasq = std::max(SIGMASQ_MIN, 1.0 / gdraw);
      sigmasq = clamp_scalar(sigmasq, SIGMASQ_MIN, SIGMASQ_MAX);
    }

    // --- Tau Gibbs step ---
    if (ntau > 0) {
      arma::mat Zt_Om = Z_dat.t();
      Zt_Om.each_row() %= omega_pg.t();
      arma::mat prec_tau = Zt_Om * Z_dat;
      prec_tau.diag() += 1.0 / (htau * sigmasq);

      arma::vec z_star_tau = kappa - omega_pg * alpha - omega_pg % Xb;
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

    moller_update_dual_sparse(
        S, R_fix, Z_active_flag, p, mu, eta1, eta2, eta1_sd, eta2_sd, mu_tilde,
        eta1_tilde, eta2_tilde, gamma, e_eta, f_eta, T_max, proposal_type,
        pw_up, pw_dn, pw_om1, pw_om2, pw_om1n, pw_om2n,
        eta1_adapter, eta2_adapter, use_cftp);

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

  return Rcpp::List::create(
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
}
