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
    double alpha_in = 0.0, bool store_beta = true, bool store_gamma = true,
    bool store_Z_list = false, bool store_Z_pip = true, int block_size = 1,
    int pcg_threshold = 500) {
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
  double eta1 = std::min(0.01, eta1_sd * 0.5);
  double eta2 = std::min(0.01, eta2_sd * 0.5);

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

  std::mt19937 pg_rng(static_cast<unsigned int>(R::runif(0.0, 1.0) * 1e9));

  std::vector<int> Z_pip_cnt;
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

    ggm_column_sweep_sparse(S, Z_active_flag, p, log_pii, log_1pii, lv0h, lv1h,
                            iv0, iv1, A_sub, s_ggm, noise_ggm, n_edges);

    {
      for (int i = 0; i < n; ++i) {
        const double lp =
            clamp_scalar(alpha + Xb(i), -LINPRED_CLIP, LINPRED_CLIP);
        omega_pg(i) = sample_pg(lp, pg_rng);
      }
    }

    const int p_act = static_cast<int>(active_idx.size());
    bool beta_refreshed = false;
    if (p_act > 0) {
      if (block_size > 1 && p_act > pcg_threshold) {
        // PCG path for large active sets (sparse X)
        bvs_dadj_block::PCGConfig pcg_cfg(1e-6, 200, pcg_threshold);
        arma::vec beta_act;
        bool pcg_ok = bvs_dadj_block::pcg_sample_beta_sparse(
            beta_act, X, active_idx, omega_pg, kappa, alpha, 1.0 / sigmasq,
            pcg_cfg);
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

        arma::vec z_star = kappa - omega_pg * alpha;
        arma::vec rhs = X_act.t() * z_star;

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
      arma::vec resid = kappa - omega_pg % Xb;
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
      auto clusters = bvs_dadj_block::swendsen_wang_dual(
          gamma, p, eta1, eta2, block_size, neigh_dyn_fn, neigh_fix_fn);
      auto flat = bvs_dadj_block::flatten_clusters(clusters);

      bvs_dadj_block::uncollapsed_gamma_sweep_dual_sparse(
          gamma, beta_vec, Xb, X, y01, alpha, sigmasq, beta0, mu, eta1, eta2,
          flat, neigh_dyn_fn, neigh_fix_fn);

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
      for (int mh = 0; mh < n_mh_gamma; ++mh) {
        int j =
            static_cast<int>(std::floor(R::runif(0.0, static_cast<double>(p))));
        if (j >= p)
          j = p - 1;

        int g_curr = static_cast<int>(gamma[j]);
        int g_prop = 1 - g_curr;
        double b_curr = beta_vec(j);
        double b_prop = (g_prop == 1) ? R::rnorm(beta0, sd_sig) : 0.0;
        b_prop = clamp_scalar(b_prop, -BETA_ABS_MAX, BETA_ABS_MAX);
        double db = b_prop - b_curr;

        double ll_diff =
            column_ll_diff(y01, Xb, alpha, col_ptr, row_idx, xvals, j, db);
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

        double ising = static_cast<double>(g_prop - g_curr) *
                       (mu + eta1 * neigh_dyn + eta2 * neigh_fix);
        double lmh = ll_diff + ising;

        if (bvs_dadj::safe_mh_accept(lmh)) {
          gamma[j] = static_cast<uint8_t>(g_prop);
          beta_vec(j) = b_prop;
          apply_column_update(Xb, col_ptr, row_idx, xvals, j, db);
          if (g_prop == 1)
            activate_gamma(j, active_idx, active_pos);
          else
            deactivate_gamma(j, active_idx, active_pos);
        }
      }
    }

    {
      double sig_prop = std::exp(std::log(sigmasq) + R::rnorm(0.0, 0.2));
      sig_prop = clamp_scalar(sig_prop, SIGMASQ_MIN, SIGMASQ_MAX);

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
      double la_c =
          -0.5 * std::log(h * sigmasq) - 0.5 * da * da / (h * sigmasq);
      double la_p =
          -0.5 * std::log(h * sig_prop) - 0.5 * da * da / (h * sig_prop);

      double lmh = (lp_p + lb_p + la_p) - (lp_c + lb_c + la_c) +
                   std::log(sig_prop / sigmasq);
      if (bvs_dadj::safe_mh_accept(lmh))
        sigmasq = sig_prop;

      sigmasq = clamp_scalar(sigmasq, SIGMASQ_MIN, SIGMASQ_MAX);
    }

    moller_update_dual_sparse(
        S, R_fix, Z_active_flag, p, mu, eta1, eta2, eta1_sd, eta2_sd, mu_tilde,
        eta1_tilde, eta2_tilde, gamma, e_eta, f_eta, T_max, proposal_type,
        pw_up, pw_dn, pw_om1, pw_om2, pw_om1n, pw_om2n);

    maybe_store_sparse_state(
        iter, burnin, thin, n_save, store_beta, store_gamma, store_Z_list,
        active_idx, beta_vec, S, Z_active_flag, eta1_out, eta2_out, true,
        alpha_out, sigmasq_out, eta1, eta2, alpha, sigmasq, beta_out_list,
        gamma_out_list, Z_list, edge_r, edge_c);

    if (iter >= burnin)
      accumulate_sparse_pip(S, Z_active_flag, store_Z_pip, Z_pip_cnt);
  }

  double n_post = static_cast<double>(total_iter - burnin);
  Rcpp::List pip_trip =
      build_sparse_pip_triplet(S, store_Z_pip, Z_pip_cnt, n_post);

  SEXP beta_out_sexp =
      store_beta ? static_cast<SEXP>(beta_out_list) : R_NilValue;
  SEXP gamma_out_sexp =
      store_gamma ? static_cast<SEXP>(gamma_out_list) : R_NilValue;
  SEXP z_list_sexp = store_Z_list ? static_cast<SEXP>(Z_list) : R_NilValue;

  return Rcpp::List::create(
      Rcpp::Named("beta") = beta_out_sexp,
      Rcpp::Named("gamma") = gamma_out_sexp, Rcpp::Named("eta1") = eta1_out,
      Rcpp::Named("eta2") = eta2_out, Rcpp::Named("alpha") = alpha_out,
      Rcpp::Named("sigmasq") = sigmasq_out, Rcpp::Named("Z_list") = z_list_sexp,
      Rcpp::Named("Z_pip_row") = pip_trip["row"],
      Rcpp::Named("Z_pip_col") = pip_trip["col"],
      Rcpp::Named("Z_pip_val") = pip_trip["val"], Rcpp::Named("p") = p,
      Rcpp::Named("n") = n);
}
