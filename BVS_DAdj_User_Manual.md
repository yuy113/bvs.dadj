# BVS.DAdj User Manual

**Bayesian Variable Selection with Dual-Adjacency Network-Informed Ising Priors**

Version 0.1.0 · Yubing Yao, Mahlet Tadesse, Raji Balasubramanian

---

## Table of Contents

1. [Overview](#1-overview)
2. [Statistical Model](#2-statistical-model)
3. [MCMC Algorithms](#3-mcmc-algorithms)
4. [Core Functions: `bvs_mh` and `bvs_pg`](#4-core-functions-bvs_mh-and-bvs_pg)
5. [Adjacency Types](#5-adjacency-types)
6. [Numerical Robustness: Robust MH Acceptance](#6-numerical-robustness-robust-mh-acceptance)
7. [Inner Gamma/Beta Thinning (`n_thin_gb`)](#7-inner-gammabeta-thinning-n_thin_gb)
8. [Post-Processing: Summary, Plots, and Diagnostics](#8-post-processing-summary-plots-and-diagnostics)
9. [Eta Hyperparameter Tuning](#9-eta-hyperparameter-tuning)
10. [Pre-computation Helpers](#10-pre-computation-helpers)
11. [Choosing Dense vs. Sparse Backends](#11-choosing-dense-vs-sparse-backends)
12. [C++ Backend Architecture](#12-c-backend-architecture)
13. [Complete Worked Examples](#13-complete-worked-examples)
14. [Parameter Reference Tables](#14-parameter-reference-tables)

---

## 1. Overview

`BVS.DAdj` is an R package for Bayesian variable selection in logistic regression. It places an **Ising/Markov Random Field (MRF) prior** on the binary inclusion indicators γ, allowing the selection prior to be informed by one or two adjacency (network) matrices that encode prior knowledge about which predictors are likely co-selected.

The package is designed for:

- **Single-network** settings where one source of structural information is available (e.g., a biological pathway graph or a graphical model learned from the predictor matrix).
- **Dual-network** settings where two sources of structural information are combined: one fixed external network and one estimated from the data.
- **High-dimensional settings** (`p ≫ n`) via sparse C++ backends that operate entirely in compressed-sparse-column (CSC) format and use O(|E|) neighbourhood sums instead of O(p²) dense operations.

Two primary MCMC sampling algorithms are provided:

| Function | Algorithm | Beta/alpha update | Typical use |
|---|---|---|---|
| `bvs_mh()` | Metropolis-Hastings | Random-walk MH proposal | Flexible; any adjacency type |
| `bvs_pg()` | Pólya-Gamma Gibbs | Exact Gibbs via data augmentation | Improved mixing for beta/alpha |

---

## 2. Statistical Model

The likelihood is a logistic regression:

```
y_i | X, β, α  ~  Bernoulli( logistic(α + x_i'β) ),   i = 1, …, n
```

The spike-and-slab prior on coefficients conditional on the inclusion indicator γ:

```
β_j | γ_j, σ²  ~  γ_j · N(β₀, σ²)  +  (1 − γ_j) · δ₀
```

The Ising MRF prior on the inclusion indicator vector γ:

```
P(γ | η, R) ∝ exp{ μ · Σ_j γ_j  +  η₁ · Σ_{(j,k)∈R₁} γ_j γ_k  +  η₂ · Σ_{(j,k)∈R₂} γ_j γ_k }
```

where R₁ (and R₂ in dual-network models) are binary symmetric adjacency matrices. The coupling parameter η captures how strongly adjacent predictors tend to be co-selected or co-excluded.

Hyperpriors:

- `σ² ~ Inverse-Gamma(ν₀/2, ν₀σ₀²/2)`
- `η_k ~ Uniform(0, η_k_sd)` with a Beta(e_eta, f_eta) shape prior enforced by the Möller auxiliary variable construction.
- `α ~ N(α₀, h · σ²)` (intercept prior scaled by variance).

---

## 3. MCMC Algorithms

### Metropolis-Hastings (`bvs_mh`)

Per-iteration steps:

1. **Gamma and beta** (inner loop, repeated `n_thin_gb` times):
   - Propose γ_j flip with β_j drawn from the spike-and-slab prior if activating, or set to 0 if deactivating.
   - MH accept/reject using the logistic likelihood ratio and Ising prior ratio.
   - Propose a Random-walk Normal perturbation to each active β_j (continuous update).
2. **Alpha** — Random-walk Normal MH.
3. **σ²** — Log-normal random-walk MH.
4. **η (coupling)** — Möller et al. (2006) auxiliary variable MH with Propp-Wilson perfect simulation to handle the intractable Ising normalising constant.
5. **GGM adjacency** *(when `adj_type ∈ {"ggm", "ggm_fixed"}`)* — Wang (2012) Bayesian GGM column sweep with spike-and-slab precision priors.

All MH acceptance steps use the robust `safe_mh_accept()` function (see [Section 6](#6-numerical-robustness-robust-mh-acceptance)).

### Pólya-Gamma Gibbs (`bvs_pg`)

Per-iteration steps:

1. **Omega (PG latent variables)** — Sample ω_i ~ PG(1, α + x_i'β) for each observation.
2. **Beta and alpha** — Exact closed-form Gibbs draw from a multivariate normal, conditioned on ω and γ.
3. **Gamma (variable selection)** — Reversible-jump MH update for each predictor in random order, using the augmented likelihood.
4. **σ²** — Log-normal random-walk MH.
5. **η** — Möller auxiliary variable MH (same as `bvs_mh`).
6. **GGM adjacency** *(if applicable)* — Wang (2012) column sweep.

The Pólya-Gamma augmentation makes the beta/alpha update *exact* (no MH), which substantially improves mixing for regression coefficients.

---

## 4. Core Functions: `bvs_mh` and `bvs_pg`

### 4.1 `bvs_mh` — Metropolis-Hastings Sampler

```r
bvs_mh(
  X, y,
  adj_type  = c("fixed", "dual_fixed", "glasso", "glasso_fixed", "ggm", "ggm_fixed"),
  adj_fixed  = NULL,        # p×p binary adjacency matrix (required for fixed types)
  adj_fixed2 = NULL,        # second p×p matrix (required for dual_fixed only)
  sparse     = FALSE,       # TRUE → sparse backend (ggm / ggm_fixed only)
  S_ggm      = NULL,        # optional pre-computed scatter matrix for ultra-large p
  store_beta    = FALSE,    # (sparse mode) store full beta draws
  store_gamma   = FALSE,    # (sparse mode) store full gamma draws
  store_Z_list  = FALSE,    # (sparse mode) store GGM adjacency snapshots
  store_Z_pip   = TRUE,     # (sparse mode) accumulate GGM edge PIPs
  glasso_criterion = c("ebic", "ric"),
  # --- MCMC control ---
  niter  = 60000L,          # post-burn-in iterations
  burnin = 10000L,          # burn-in iterations (discarded)
  thin   = 1L,              # storage thinning interval
  n_thin_gb = 3L,           # inner MH sub-iterations for gamma+beta (dense only)
  # --- Variable selection priors ---
  nu0      = 2,             # IG shape for σ²
  sigmasq0 = 1.5,           # IG scale for σ²
  h        = 1.5,           # intercept variance inflation factor
  mu       = -log(1/0.3 - 1), # Ising external field (sparsity)
  alpha0   = 0,             # prior mean for intercept
  beta0    = 0,             # prior mean for coefficients
  # --- Gamma / Ising MH ---
  n_mh_gamma  = 3L,         # gamma flip proposals per iteration (GGM/sparse only)
  eta1_sd     = 0.5,        # upper bound Uniform prior on η₁
  eta2_sd     = 0.5,        # upper bound Uniform prior on η₂ (dual models)
  mu_tilde    = -4,         # Möller auxiliary external field
  eta1_tilde  = 0.075,      # Möller auxiliary η₁ coupling
  eta2_tilde  = 0.065,      # Möller auxiliary η₂ coupling (dual models)
  e_eta = 2, f_eta = 1,     # Beta(e, f) prior shape on η
  Tmax  = 64L,              # Propp-Wilson maximum doubling horizon
  proposal_type = 1L,       # η proposal: 0=Uniform, 1=truncated Normal
  # --- GGM SSVS priors ---
  v0_ggm    = 0.015^2,      # spike variance
  v1_ggm    = NULL,         # slab variance (default: 50² × v0_ggm)
  pii_ggm   = NULL,         # edge inclusion probability (default: 30/(p-1))
  lambda_ggm = 1,           # GGM diagonal prior scale
  # --- Initialisation ---
  beta_init  = NULL,        # initial beta (length-p numeric vector)
  gamma_init = NULL,        # initial gamma (length-p 0/1 integer vector)
  alpha_init = NULL         # initial intercept (scalar)
)
```

**Key difference from `bvs_pg`:** The `n_thin_gb` parameter (default `3`) runs `n_thin_gb` MH sub-iterations for gamma and beta at each MCMC step. This effectively provides 3× more mixing for the binary inclusion indicators and continuous coefficients per recorded sample, at the cost of 3× computation for those steps (other parameters — alpha, σ², η — remain one update per iteration). Sparse GGM backends ignore `n_thin_gb` for computational efficiency.

### 4.2 `bvs_pg` — Pólya-Gamma Gibbs Sampler

```r
bvs_pg(
  X, y,
  adj_type  = c("fixed", "dual_fixed", "glasso", "glasso_fixed", "ggm", "ggm_fixed"),
  adj_fixed  = NULL,
  adj_fixed2 = NULL,
  sparse     = FALSE,
  S_ggm      = NULL,
  store_beta    = FALSE,
  store_gamma   = FALSE,
  store_Z_list  = FALSE,
  store_Z_pip   = TRUE,
  glasso_criterion = c("ebic", "ric"),
  # --- MCMC control ---
  niter  = 60000L,
  burnin = 10000L,
  thin   = 1L,
  # --- Variable selection priors ---
  nu0 = 2, sigmasq0 = 1.5, h = 1.5,
  mu = -log(1/0.3 - 1), alpha0 = 0, beta0 = 0,
  # --- Gamma / Ising MH ---
  n_mh_gamma = 3L,
  eta1_sd = 0.5, eta2_sd = 0.5,
  mu_tilde = -4, eta1_tilde = 0.075, eta2_tilde = 0.065,
  e_eta = 2, f_eta = 1,
  Tmax = 64L, proposal_type = 1L,
  # --- GGM SSVS priors ---
  v0_ggm = 0.015^2, v1_ggm = NULL, pii_ggm = NULL, lambda_ggm = 1,
  # --- Initialisation ---
  beta_init = NULL, gamma_init = NULL, alpha_init = NULL
)
```

`bvs_pg` does not have `n_thin_gb` because the Pólya-Gamma augmentation gives exact Gibbs draws for beta and alpha — repeated proposals are not needed for mixing.

### 4.3 Return Value

Both functions return an S3 object of class `"bvs"`:

| Field | Type | Present when |
|---|---|---|
| `beta` | `niter × p` matrix or list | always (dense) / `store_beta = TRUE` (sparse) |
| `gamma` | `niter × p` matrix or list | always (dense) / `store_gamma = TRUE` (sparse) |
| `gamma_pip` | numeric vector, length `p` | always |
| `alpha` | numeric vector, length `niter` | always |
| `sigmasq` | numeric vector, length `niter` | always |
| `eta1` | numeric vector | single- and dual-network |
| `eta2` | numeric vector | dual-network models only |
| `Z_list` | list of sparse adjacency snapshots | `store_Z_list = TRUE` |
| `Z_pip` | sparse matrix (`p × p`) | `store_Z_pip = TRUE` (GGM types) |
| `call` | matched call | always |
| `adj_type`, `sampler` | character | always |
| `niter`, `burnin`, `p`, `n` | integer | always |

> **Note on short chains:** The toy examples in this manual use very short MCMC chains (`niter = 5`, `burnin = 2`) for quick illustration only. In practice use at least `niter = 20000`, `burnin = 5000` (and more for large `p` or sparse models).

**Toy Example — `bvs_pg` with glasso adjacency:**
```r
set.seed(123)
n <- 200; p <- 50
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(rep(1.5, 5), rep(0, p - 5))
y <- rbinom(n, 1, plogis(X %*% beta_true))

fit_pg <- bvs_pg(X, y, adj_type = "glasso", glasso_criterion = "ebic",
                 niter = 5, burnin = 2)
summary(fit_pg)
```

**Toy Example — `bvs_mh` with fixed adjacency and inner thinning:**
```r
R <- matrix(0L, p, p)
R[1:5, 1:5] <- 1L; diag(R) <- 0L

fit_mh <- bvs_mh(X, y, adj_type = "fixed", adj_fixed = R,
                 niter = 5, burnin = 2, n_thin_gb = 3L)
plot(fit_mh, type = "pip")
```

**Toy Example — dual-fixed adjacency:**
```r
R1 <- matrix(rbinom(p * p, 1, 0.05), p, p)
R2 <- matrix(rbinom(p * p, 1, 0.02), p, p)
R1 <- (R1 + t(R1) > 0) * 1L; diag(R1) <- 0L
R2 <- (R2 + t(R2) > 0) * 1L; diag(R2) <- 0L

fit_dual <- bvs_pg(X, y, adj_type = "dual_fixed",
                   adj_fixed = R1, adj_fixed2 = R2,
                   niter = 5, burnin = 2)
summary(fit_dual)
```

**Toy Example — sparse Bayesian GGM backend (p ≫ n):**
```r
fit_sparse <- bvs_mh(X, y, adj_type = "ggm", sparse = TRUE,
                     niter = 5, burnin = 2)
sum(summary(fit_sparse)$pip > 0.5)  # estimated model size
```

---

## 5. Adjacency Types

| `adj_type` | Etas | Adjacency source | `sparse` option |
|---|---|---|---|
| `"fixed"` | 1 | External `adj_fixed` matrix | No |
| `"dual_fixed"` | 2 | `adj_fixed` + `adj_fixed2` | No |
| `"glasso"` | 1 | Frequentist graphical lasso (EBIC or RIC selection, requires `huge`) | No |
| `"glasso_fixed"` | 2 | Glasso-estimated + external `adj_fixed` | No |
| `"ggm"` | 1 | Bayesian GGM SSVS (Wang 2012), learned jointly during MCMC | Yes |
| `"ggm_fixed"` | 2 | Bayesian GGM + external `adj_fixed` | Yes |

For `"glasso"` and `"glasso_fixed"` the adjacency is estimated once before MCMC begins and held fixed throughout. For `"ggm"` and `"ggm_fixed"` the adjacency is updated at every MCMC iteration using the Wang (2012) GGM column sweep.

---

## 6. Numerical Robustness: Robust MH Acceptance

All MH acceptance steps across every C++ backend use the `bvs_dadj::safe_mh_accept(log_ratio)` helper defined in `BayesLogit_Numerics.h`:

```cpp
inline bool safe_mh_accept(double log_ratio) {
  if (std::isnan(log_ratio)) return false;  // numerical error → reject
  if (log_ratio >= 0.0)      return true;   // always accept (includes +Inf)
  // Probability form: U < exp(log_ratio) — avoids log(0) deadlock entirely
  return R::runif(0.0, 1.0) < std::exp(log_ratio);
}
```

The three cases handled:

| Situation | Cause | Behaviour |
|---|---|---|
| `log_ratio = NaN` | Invalid likelihood (e.g., `log(negative)`) | Always reject |
| `log_ratio ≥ 0` | Proposal better than or equal to current | Always accept |
| `log_ratio < 0` | Partial acceptance | Compare `U ~ Uniform(0,1)` to `exp(log_ratio)` — never calls `log(0)` |

The classic idiom `std::log(R::runif(0, 1)) < log_ratio` can deadlock if `runif` returns exactly 0 (causing `log(0) = -Inf`, which equals `log_ratio` only if `log_ratio` is also `-Inf` — an edge case that can freeze in degenerate configurations). `safe_mh_accept` eliminates this entirely by working in probability space.

This robustness mechanism is applied to **all** MH steps across all 12 C++ backends:

- Gamma (variable selection) MH acceptance
- Beta (regression coefficient) MH acceptance
- Alpha (intercept) MH acceptance
- σ² (variance) MH acceptance
- η₁ and η₂ (coupling) Möller auxiliary MH acceptance

---

## 7. Inner Gamma/Beta Thinning (`n_thin_gb`)

**What it does:** In `bvs_mh()`, the `n_thin_gb` parameter (default `3`) wraps the gamma and beta MH update blocks in an inner `for` loop that runs `n_thin_gb` times per recorded MCMC iteration. Only every `n_thin_gb`-th state of γ and β is kept for inference.

**Why it helps:** Discrete Bernoulli indicators γ and random-walk beta proposals can mix slowly relative to continuous parameters. Running 3× more proposals per stored sample — while keeping the other parameters (α, σ², η, GGM) at 1 update per stored sample — provides denser exploration of the variable-selection space without inflating the stored output size.

**Where it applies:** Dense MH backends only:

- `BayesLogit_SingleNet_FixedAdj.cpp`
- `BayesLogit_DualNet_FixedAdj.cpp`
- `BayesLogit_SingleNet_GGM.cpp`
- `BayesLogit_DualNet_GGM.cpp`

Sparse GGM backends (`BayesLogit_SingleNet_SparseGGM.cpp`, `BayesLogit_DualNet_SparseGGM.cpp`) do **not** implement inner thinning — each gamma flip in the sparse backend involves a different column of the sparse scatter matrix and already benefits from the `n_mh_gamma` outer loop, making the additional inner loop redundant for computational efficiency.

`bvs_pg()` does not have `n_thin_gb` because the Pólya-Gamma beta/alpha update is an exact Gibbs step that does not benefit from repeated proposals.

**Setting `n_thin_gb`:** Values of 2–5 are typical. Higher values improve mixing but increase computation linearly. Setting `n_thin_gb = 1` reproduces the original single-proposal behaviour.

---

## 8. Post-Processing: Summary, Plots, and Diagnostics

### 8.1 `summary.bvs`

```r
summary(object, pip_threshold = 0.5, cred_level = 0.95, hpd_level = 0.95, ...)
```

Computes for each predictor:

- **PIP** — Posterior inclusion probability = proportion of MCMC samples with γ_j = 1.
- **Posterior mean** of β_j (over all samples, including zeros).
- **Posterior mean conditional on selection** (over samples where γ_j = 1).
- **Credible interval** — quantile-based `cred_level` interval for β_j.
- **HPD interval** — highest-posterior-density interval at `hpd_level` (via `coda`).
- **ESS** — effective sample size for β_j.

Returns a named list including `pip`, `selected` (integer indices with PIP > `pip_threshold`), `nselected`, `summary_beta` (data frame), `summary_alpha`, `sampler`, `adj_type`.

```r
summ <- summary(fit_pg, pip_threshold = 0.5)
print(summ$summary_beta[summ$selected, ])  # rows for selected variables
cat("Model size:", summ$nselected, "\n")
```

### 8.2 `plot.bvs`

```r
plot(x, type = c("pip", "trace", "acf", "all"),
     pip_threshold = 0.5, top_n = 5, vars = NULL, ...)
```

| `type` | Output |
|---|---|
| `"pip"` | Bar chart of posterior inclusion probabilities with a reference line at `pip_threshold` |
| `"trace"` | MCMC trace plots for the `top_n` predictors with highest PIP (or explicit `vars`) |
| `"acf"` | Autocorrelation function plots for the same variables |
| `"all"` | All three panels in sequence |

```r
plot(fit_mh, type = "pip", pip_threshold = 0.5)
plot(fit_mh, type = "acf", vars = c(1, 2, 5))
```

### 8.3 `as.mcmc.bvs` — Convert to `coda` Object

```r
mcmc_obj <- as.mcmc.bvs(x, vars = NULL, ...)
```

Converts posterior beta draws to a `coda::mcmc` object for use with `coda` diagnostics (e.g., `effectiveSize`, `heidel.diag`, `geweke.diag`).

```r
library(coda)
mcmc_obj <- as.mcmc.bvs(fit_pg, vars = 1:10)
effectiveSize(mcmc_obj)
```

### 8.4 `bvs_gelman_diag` — Gelman-Rubin PSRF

```r
bvs_gelman_diag(fits, vars = NULL, ...)
```

Computes the Gelman-Rubin potential scale reduction factor (PSRF) across independent chains. Requires at least two `"bvs"` objects run with the same model specification.

```r
# Run two independent chains
fit1 <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R,
               niter = 10000, burnin = 2000)
fit2 <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R,
               niter = 10000, burnin = 2000)

diag_res <- bvs_gelman_diag(list(fit1, fit2))
print(diag_res)             # multivariate PSRF
print(diag_res$psrf[1:10,]) # per-variable PSRF for first 10 predictors
```

A PSRF close to 1 (< 1.1 is the standard threshold) indicates convergence.

---

## 9. Eta Hyperparameter Tuning

The Ising coupling parameter η governs how strongly neighbouring predictors co-select. Its prior is Uniform(0, `eta_sd`) combined with the Möller auxiliary variable construction. Choosing `eta_sd` and the Möller hyperparameters (`mu_tilde`, `eta1_tilde`, `eta2_tilde`) is crucial for valid inference.

The package provides a three-stage tuning workflow:

### Stage 1 — Phase transition profiling

Locate the **critical coupling strength** η* at which the Ising prior transitions from sparse to dense models for your specific adjacency matrix.

```r
pt <- phase_transition(R, mu = -log(1/0.3 - 1),
                       min_eta = 0, max_eta = 1.5,
                       step_size = 0.05, num_rep = 10, Tmax = 64L)
matplot(pt, type = "l", xlab = "eta step index", ylab = "model size",
        main = "Ising phase transition profile")
```

`phase_transition()` returns an integer matrix (`length(eta_seq) × num_rep`) of model sizes from Propp-Wilson perfect simulation. The transition point η* is where model size jumps sharply.

### Stage 2 — Build a candidate hyperparameter grid

```r
# Single-network model (one eta):
grid1 <- bvs_eta_grid_single(
  adj         = R,
  eta1_frac   = seq(0.2, 1.0, by = 0.2),  # fractions of eta* to try
  mu_tilde    = c(-4, -5),
  eta1_tilde  = c(0.05, 0.10, 0.15),
  e_eta       = 2, f_eta = 1
)
head(grid1)

# Dual-network model (two etas):
grid2 <- bvs_eta_grid_dual(
  adj1        = R1, adj2 = R2,
  eta1_frac   = seq(0.2, 1.0, by = 0.2),
  eta2_frac   = seq(0.2, 1.0, by = 0.2)
)
```

### Stage 3 — Evaluate and select hyperparameters

```r
# Run short pilot chains for each grid row, then select by FDR-controlled PIP:
best <- bvs_tune_eta(
  X, y,
  adj_type  = "fixed", adj_fixed = R,
  sampler   = "pg",
  eta_grid  = grid1,
  niter     = 5000L, burnin = 1000L,
  pip_threshold = 0.5,
  target_fdr    = 0.05
)
print(best$best_row)   # optimal hyperparameter row
print(best$result_df)  # full grid results

# Convenience wrapper: fixes PIP threshold and optimises FDR:
best_fixed <- bvs_tune_eta_fixed_pip(
  X, y,
  adj_type = "fixed", adj_fixed = R,
  eta_grid = grid1,
  pip_threshold_fixed = 0.5,
  target_fdr = 0.05
)
```

---

## 10. Pre-computation Helpers

### 10.1 `prepare_sparse_S`

```r
prepare_sparse_S(X, threshold = 1e-4)
```

Pre-computes the scatter matrix `S = X'X / n` in CSC format for the sparse GGM backends. Entries with `|S_ij| < threshold` are treated as structural zeros. Use this when the same `X` is used across many MCMC runs to avoid recomputing the scatter matrix repeatedly.

```r
S_csc <- prepare_sparse_S(X, threshold = 1e-4)
fit <- bvs_mh(X, y, adj_type = "ggm", sparse = TRUE,
              S_ggm = S_csc, niter = 5, burnin = 2)
```

Increasing `threshold` produces a sparser scatter matrix and faster GGM column sweeps, at the cost of approximating small off-diagonal entries as zero. For most genomic applications `threshold ∈ [1e-4, 1e-3]` is adequate.

### 10.2 `estimate_glasso_adj`

```r
estimate_glasso_adj(X, criterion = c("ebic", "ric"),
                    nlambda = 30, lambda.min.ratio = 0.01,
                    symmetrize = TRUE)
```

Calls `huge::huge()` and `huge::huge.select()` to estimate a sparse precision matrix from `X` via graphical lasso, then returns a symmetric binary adjacency matrix suitable for `adj_fixed`. Requires the `huge` package.

```r
R_learned <- estimate_glasso_adj(X, criterion = "ebic")
fit <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R_learned,
              niter = 10000, burnin = 2000)
```

Note: `"glasso"` and `"glasso_fixed"` adjacency types call this function internally. Supply `adj_fixed = estimate_glasso_adj(X)` only when you want to reuse the same estimated graph across multiple `bvs_mh`/`bvs_pg` calls.

---

## 11. Choosing Dense vs. Sparse Backends

| Criterion | Dense backend (`sparse = FALSE`) | Sparse backend (`sparse = TRUE`) |
|---|---|---|
| **Dimension** | `p` up to ~1000 | `p` up to ~10 000+ |
| **Beta storage** | Full `niter × p` matrix | Sparse COO list (set `store_beta = TRUE`) |
| **Gamma storage** | Full `niter × p` matrix | Sparse index list (set `store_gamma = TRUE`) |
| **GGM sweep** | Dense `p × p` scatter matrix | CSC-format scatter matrix (off-diagonal thresholded) |
| **Ising neighbour sum** | Dense O(p) loop | Sparse O(degree) loop |
| **`n_thin_gb`** | Supported (default 3) | Not applicable |
| **Thinning** | `thin` parameter | `thin` parameter |

**Rule of thumb:** If `p × n` fits comfortably in memory as a dense double matrix and `p ≤ 1000`, use the dense backend. If `p > 500` and `p ≫ n`, consider `sparse = TRUE` with `S_ggm = prepare_sparse_S(X)`.

For sparse backends, set `store_beta = FALSE` and `store_gamma = FALSE` unless memory permits, and use `gamma_pip` (always computed) as the primary inference target.

---

## 12. C++ Backend Architecture

The package contains 12 C++ translation units in `src/` plus two shared headers. The R dispatcher in `bvs_mh()` and `bvs_pg()` automatically routes to the appropriate backend based on `sampler × adj_type × sparse`.

### Dense MH backends (thinned inner gamma/beta loop)

| File | Network | Adjacency | Eta |
|---|---|---|---|
| `BayesLogit_SingleNet_FixedAdj.cpp` | Single | Fixed | 1 |
| `BayesLogit_DualNet_FixedAdj.cpp` | Dual | 2 Fixed | 2 |
| `BayesLogit_SingleNet_GGM.cpp` | Single | Bayesian GGM | 1 |
| `BayesLogit_DualNet_GGM.cpp` | Dual | GGM + Fixed | 2 |

### Dense PG backends (exact Gibbs for beta/alpha)

| File | Network | Adjacency | Eta |
|---|---|---|---|
| `BayesLogit_PG_SingleAdj.cpp` | Single | Fixed | 1 |
| `BayesLogit_PG_DualAdj.cpp` | Dual | 2 Fixed | 2 |
| `BayesLogit_PG_SingleAdj_GGM_Moller.cpp` | Single | Bayesian GGM | 1 |
| `BayesLogit_PG_GGM_Moller.cpp` | Dual | GGM + Fixed | 2 |

### Sparse GGM backends (CSC format, O(|E|) operations)

| File | Sampler | Network | Adjacency |
|---|---|---|---|
| `BayesLogit_SingleNet_SparseGGM.cpp` | MH | Single | Sparse GGM |
| `BayesLogit_DualNet_SparseGGM.cpp` | MH | Dual | Sparse GGM + Fixed |
| `BayesLogit_PG_SingleAdj_SparseGGM.cpp` | PG | Single | Sparse GGM |
| `BayesLogit_PG_DualNet_SparseGGM.cpp` | PG | Dual | Sparse GGM + Fixed |

### Shared headers

- **`BayesLogit_Numerics.h`** — `safe_mh_accept()`, `robust_chol_inplace()`, `clamp_finite()`, `sanitize_sym_mat_inplace()`, and other numerical utilities. Included by all dense backends.
- **`BayesLogit_Sparse_Helpers.h`** — CSC-format scatter matrix accessors (`ConstSparseS`, `ConstSparseAdj`), `column_ll_diff()`, `apply_column_update()`, Propp-Wilson helpers (`proppwilson_single_sparse`, `proppwilson_dual_sparse`), Möller update functions (`moller_update_single_sparse`, `moller_update_dual_sparse`), `ggm_column_sweep_sparse()`, and storage helpers. Includes `BayesLogit_Numerics.h` and is included by all sparse backends.

---

## 13. Complete Worked Examples

### Example A — Single fixed adjacency, PG sampler (recommended default)

```r
library(BVS.DAdj)
set.seed(2024)
n <- 300; p <- 80
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(rep(1.2, 6), rep(0, p - 6))
y <- rbinom(n, 1, plogis(X %*% beta_true))

# Build block adjacency (known pathway structure)
R <- matrix(0L, p, p)
R[1:6, 1:6] <- 1L; diag(R) <- 0L

# Fit
fit <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R,
              niter = 20000, burnin = 5000,
              mu = -log(1/0.3 - 1), eta1_sd = 0.5)

# Summarise
s <- summary(fit, pip_threshold = 0.5)
cat("Selected:", s$selected, "\n")
cat("Model size:", s$nselected, "\n")
plot(fit, type = "all", top_n = 6)
```

### Example B — MH sampler with inner thinning, dual fixed adjacency

```r
set.seed(2024)
R1 <- matrix(0L, p, p); R1[1:6, 1:6] <- 1L; diag(R1) <- 0L
R2 <- matrix(0L, p, p); R2[1:8, 1:8] <- 1L; diag(R2) <- 0L

fit_dual <- bvs_mh(X, y,
  adj_type   = "dual_fixed",
  adj_fixed  = R1, adj_fixed2 = R2,
  niter      = 20000, burnin = 5000,
  n_thin_gb  = 3L,
  eta1_sd    = 0.4, eta2_sd = 0.3)

s <- summary(fit_dual)
print(s$pip[1:10])
```

### Example C — High-dimensional sparse GGM backend (p > n)

```r
set.seed(2024)
n <- 100; p <- 500
X_hd <- matrix(rnorm(n * p), n, p)
beta_hd <- c(rep(1, 3), rep(0, p - 3))
y_hd <- rbinom(n, 1, plogis(X_hd %*% beta_hd))

# Pre-compute sparse scatter matrix for reuse
S_csc <- prepare_sparse_S(X_hd, threshold = 1e-4)

fit_hd <- bvs_pg(X_hd, y_hd,
  adj_type    = "ggm", sparse = TRUE,
  S_ggm       = S_csc,
  niter       = 10000, burnin = 2000,
  pii_ggm     = 30 / (p - 1),
  store_gamma = FALSE, store_beta = FALSE,
  store_Z_pip = TRUE)

s_hd <- summary(fit_hd)
cat("Selected (PIP > 0.5):", s_hd$selected, "\n")
```

### Example D — Convergence diagnostics across two chains

```r
fit_a <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R,
                niter = 15000, burnin = 3000)
fit_b <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R,
                niter = 15000, burnin = 3000)

diag <- bvs_gelman_diag(list(fit_a, fit_b), vars = 1:20)
print(diag$mpsrf)               # multivariate PSRF (target < 1.1)
print(diag$psrf[1:10, ])        # per-variable PSRF

# Also inspect ESS
summ_a <- summary(fit_a)
print(summ_a$summary_beta[1:6, "ESS"])
```

### Example E — Eta hyperparameter tuning workflow

```r
# Stage 1: profile phase transition
pt <- phase_transition(R, mu = -log(1/0.3 - 1),
                       min_eta = 0, max_eta = 1.5,
                       step_size = 0.05, num_rep = 8L)
matplot(pt, type = "l", xlab = "eta index", ylab = "model size",
        main = "Phase transition")

# Stage 2: build candidate grid
grid <- bvs_eta_grid_single(
  adj = R,
  eta1_frac  = seq(0.2, 0.8, by = 0.2),
  mu_tilde   = c(-4, -5),
  eta1_tilde = c(0.05, 0.10)
)

# Stage 3: select by FDR constraint
best <- bvs_tune_eta_fixed_pip(
  X, y,
  adj_type    = "fixed", adj_fixed = R,
  sampler     = "pg",
  eta_grid    = grid,
  niter       = 3000L, burnin = 500L,
  pip_threshold_fixed = 0.5,
  target_fdr          = 0.05
)
cat("Best eta_sd:", best$best_row$eta1_sd, "\n")
```

---

## 14. Parameter Reference Tables

### MCMC control

| Parameter | Default | Function | Description |
|---|---|---|---|
| `niter` | `60000` | both | Post-burn-in MCMC iterations stored |
| `burnin` | `10000` | both | Burn-in iterations (discarded) |
| `thin` | `1` | both | Storage thinning interval |
| `n_thin_gb` | `3` | `bvs_mh` only | Inner MH sub-iterations for γ + β (dense backends) |
| `n_mh_gamma` | `3` | both | γ flip proposals per outer iteration (GGM/sparse) |

### Variable selection priors

| Parameter | Default | Description |
|---|---|---|
| `nu0` | `2` | IG(ν₀/2, ν₀σ₀²/2) shape for σ² |
| `sigmasq0` | `1.5` | IG scale for σ² |
| `h` | `1.5` | Intercept variance inflation (α ~ N(α₀, h·σ²)) |
| `mu` | `-log(1/0.3 − 1)` | Ising external field (≈ 0.847; prior expected model size ≈ 30% of p) |
| `alpha0` | `0` | Prior mean for intercept α |
| `beta0` | `0` | Prior mean for regression coefficients β_j |

### Ising / Möller hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `eta1_sd` | `0.5` | Upper bound Uniform prior on η₁ |
| `eta2_sd` | `0.5` | Upper bound Uniform prior on η₂ (dual models) |
| `e_eta`, `f_eta` | `2`, `1` | Beta(e, f) prior shape on η |
| `mu_tilde` | `-4` | Möller auxiliary MRF external field |
| `eta1_tilde` | `0.075` | Möller auxiliary η₁ coupling |
| `eta2_tilde` | `0.065` | Möller auxiliary η₂ coupling (dual models) |
| `Tmax` | `64` | Propp-Wilson maximum doubling horizon |
| `proposal_type` | `1` | η proposal: `0` = Uniform, `1` = truncated Normal |

### GGM SSVS priors

| Parameter | Default | Description |
|---|---|---|
| `v0_ggm` | `0.015²` | Spike variance for off-diagonal precision entries |
| `v1_ggm` | `50² × v0_ggm` | Slab variance |
| `pii_ggm` | `30 / (p − 1)` | Prior edge inclusion probability |
| `lambda_ggm` | `1` | Exponential prior scale for diagonal precision entries |

### Adjacency type routing

| `adj_type` | Sampler | `sparse` | C++ backend used |
|---|---|---|---|
| `"fixed"` | MH | — | `BayesLogit_SingleNet_FixedAdj.cpp` |
| `"dual_fixed"` | MH | — | `BayesLogit_DualNet_FixedAdj.cpp` |
| `"glasso"` | MH | — | `BayesLogit_SingleNet_FixedAdj.cpp` (glasso adj pre-computed) |
| `"glasso_fixed"` | MH | — | `BayesLogit_DualNet_FixedAdj.cpp` (glasso adj pre-computed) |
| `"ggm"` | MH | FALSE | `BayesLogit_SingleNet_GGM.cpp` |
| `"ggm"` | MH | TRUE | `BayesLogit_SingleNet_SparseGGM.cpp` |
| `"ggm_fixed"` | MH | FALSE | `BayesLogit_DualNet_GGM.cpp` |
| `"ggm_fixed"` | MH | TRUE | `BayesLogit_DualNet_SparseGGM.cpp` |
| `"fixed"` | PG | — | `BayesLogit_PG_SingleAdj.cpp` |
| `"dual_fixed"` | PG | — | `BayesLogit_PG_DualAdj.cpp` |
| `"glasso"` | PG | — | `BayesLogit_PG_SingleAdj.cpp` (glasso adj pre-computed) |
| `"glasso_fixed"` | PG | — | `BayesLogit_PG_DualAdj.cpp` (glasso adj pre-computed) |
| `"ggm"` | PG | FALSE | `BayesLogit_PG_SingleAdj_GGM_Moller.cpp` |
| `"ggm"` | PG | TRUE | `BayesLogit_PG_SingleAdj_SparseGGM.cpp` |
| `"ggm_fixed"` | PG | FALSE | `BayesLogit_PG_GGM_Moller.cpp` |
| `"ggm_fixed"` | PG | TRUE | `BayesLogit_PG_DualNet_SparseGGM.cpp` |

---

*BVS.DAdj version 0.1.0 · GPL (≥ 2) · Maintained by Yubing Yao <yyao@umass.edu>*
