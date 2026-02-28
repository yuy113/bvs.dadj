# BVS.DAdj News

## BVS.DAdj 0.1.0

Initial release.

### New features

* Two MCMC samplers: `bvs_mh()` (Metropolis-Hastings) and `bvs_pg()` (Pólya-Gamma
  augmented Gibbs), each supporting six adjacency types:
  `"fixed"`, `"dual_fixed"`, `"glasso"`, `"glasso_fixed"`, `"ggm"`, `"ggm_fixed"`.

* **12 optimised C++ backends** covering all combinations of sampler (MH / PG),
  network topology (single / dual), adjacency estimation (fixed / GGM / sparse GGM),
  and compute mode (dense / sparse CSC).

* **`n_thin_gb` parameter** (default `3`) in `bvs_mh()`:
  Runs `n_thin_gb` inner MH sub-iterations for the gamma (variable selection) and
  beta (regression coefficient) updates per stored MCMC sample, providing denser
  exploration of the binary-inclusion space without increasing stored output size.
  Applies to dense MH backends only; sparse GGM backends omit this for computational
  efficiency.

* **Robust MH acceptance** (`bvs_dadj::safe_mh_accept()` in `BayesLogit_Numerics.h`):
  All 51 MH acceptance checks across all 12 C++ backends now use the probability-form
  test `U < exp(log_ratio)` instead of `log(U) < log_ratio`. This eliminates the
  potential deadlock when `R::runif(0, 1)` returns exactly 0 (producing `log(0) = -Inf`)
  and correctly handles `NaN` (always reject) and `log_ratio ≥ 0` (always accept).

* **Eta hyperparameter tuning workflow** via:
  - `phase_transition()` / `bvs_phase_eta_profile()` — locate the critical Ising
    coupling strength by sweeping η with Propp-Wilson perfect simulation.
  - `bvs_eta_grid_single()` / `bvs_eta_grid_dual()` — build candidate grids of
    (`eta_sd`, `mu_tilde`, `eta_tilde`) anchored at the phase-transition point.
  - `bvs_tune_eta()` / `bvs_tune_eta_fixed_pip()` — select optimal hyperparameters
    under an FDR constraint across short pilot chains.

* **Rich diagnostics**:
  - `summary.bvs()` — PIP, posterior mean, credible and HPD intervals, ESS for all
    coefficients.
  - `plot.bvs()` — PIP bar chart, MCMC trace, and ACF plots.
  - `as.mcmc.bvs()` — convert to `coda::mcmc` for external diagnostics.
  - `bvs_gelman_diag()` — Gelman-Rubin PSRF across independent chains.

* **Sparse pre-computation helpers**:
  - `prepare_sparse_S()` — pre-compute the scatter matrix in CSC format for repeated
    use with sparse GGM backends.
  - `estimate_glasso_adj()` — frequency-domain graphical lasso adjacency estimation
    via `huge` (EBIC or RIC criterion).

### Bug fixes

* Fixed partial argument match warnings: calls to `BayesLogit_DualNet_FixedAdj()`,
  `BayesLogit_DualNet_GGM()`, `BayesLogit_SingleNet_GGM()`, and
  `BayesLogit_PG_SingleAdj.cpp` were passing `e = e_eta, f = f_eta` (partial match)
  instead of the full argument names `e_eta = e_eta, f_eta = f_eta`. (#R-codoc-1)

* Fixed missing `methods` import: `methods::hasSlot()` usage in
  `ultra_sparse_backends.R` was replaced by `"x" %in% slotNames(M)` and the
  `slotNames` function is now properly imported via `@importFrom methods slotNames`.

* Fixed stale Rd documentation: `bvs_eta_grid_single.Rd` listed parameters `eta_sd`
  and `eta_frac` (old names) instead of the current `eta1_sd` and `eta1_frac`.
  Documentation regenerated with `roxygen2::roxygenise()`.

* Added `@param ...` documentation to `bvs_tune_eta_fixed_pip()` to resolve the
  "undocumented arguments" `R CMD check` warning.

* Added `.Rbuildignore` patterns to exclude vignette output files (`.html`, `.md`,
  `_cache/`, `_files/`, `_saved_runs/`, build logs) from the package tarball,
  resolving the "non-portable file names" `R CMD check` error caused by paths
  exceeding 100 bytes.

### Internal changes

* All MH acceptance checks migrated from the classic `std::log(R::runif(0,1)) < log_ratio`
  idiom to `bvs_dadj::safe_mh_accept(log_ratio)`, covering:
  - 4 dense MH C++ files (gamma, beta, alpha, σ² checks + Möller eta checks)
  - 4 dense PG C++ files (gamma, sigma², eta checks)
  - 4 sparse GGM C++ files (gamma, sigma² checks, plus Möller helpers in
    `BayesLogit_Sparse_Helpers.h`)

* `BayesLogit_PG_SingleAdj_GGM_Moller.cpp` now includes `BayesLogit_Numerics.h`
  directly (previously it relied on transitive inclusion which is fragile).

* `bvs_mh.R` wrapper updated to pass `n_thin_gb = as.integer(n_thin_gb)` to all
  dense MH backends and to document the new parameter in roxygen.
