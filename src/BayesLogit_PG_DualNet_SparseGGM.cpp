// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// Forward declaration from BayesLogit_UltraSparse_Backends.cpp
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
    int proposal_type, int thin,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in,
    double alpha_in,
    bool store_beta, bool store_gamma,
    bool store_Z_list, bool store_Z_pip);

// [[Rcpp::export]]
Rcpp::List BayesLogit_PG_DualNet_SparseGGM(
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
    int proposal_type,
    int thin = 1,
    Rcpp::Nullable<Rcpp::NumericVector> beta_in = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> gamma_in = R_NilValue,
    double alpha_in = 0.0,
    bool store_beta = true,
    bool store_gamma = true,
    bool store_Z_list = false,
    bool store_Z_pip = true) {

  return BayesLogit_PG_DualNet_SparseGGM_UltraSparse(
      X, y,
      S_i, S_p_csc, S_x, S_diag,
      R_fix_i, R_fix_p_csc,
      p_ggm, niter, burnin,
      mu, nu0, sigmasq0, alpha0, beta0,
      h, n_mh_gamma,
      v0_ggm, v1_ggm, pii_ggm,
      eta1_sd, eta2_sd, mu_tilde, eta1_tilde,
      eta2_tilde, e_eta, f_eta, T_max,
      proposal_type, thin,
      beta_in, gamma_in, alpha_in,
      store_beta, store_gamma, store_Z_list, store_Z_pip);
}
