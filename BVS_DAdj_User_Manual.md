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
15. [MCMC Mixing and Convergence Improvements](#15-mcmc-mixing-and-convergence-improvements)

---

## 1. Overview

`BVS.DAdj` is an R package for Bayesian variable selection with network-informed Ising priors on the binary inclusion indicators γ. The `bvs_mh` sampler uses Metropolis-Hastings (MH) by default and also provides Hamiltonian Monte Carlo (HMC) and the No-U-Turn Sampler (NUTS) for active parameter-block updates in binary and TTE models; it supports binary logistic outcomes (`outcome_type = "binary"`, default), continuous Gaussian outcomes (`outcome_type = "continuous"`), right-censored time-to-event outcomes (`outcome_type = "TTE"`) via Cox's model, and overdispersed count outcomes (`outcome_type = "count"`) via a negative-binomial (Poisson-Gamma) representation. The PG sampler (`bvs_pg`) targets binary logistic outcomes.

The package is designed for:

- **Single-network** settings where one source of structural information is available (e.g., a biological pathway graph or a graphical model learned from the predictor matrix).
- **Dual-network** settings where two sources of structural information are combined: one fixed external network and one estimated from the data.
- **High-dimensional settings** (`p ≫ n`) via sparse C++ backends that operate entirely in compressed-sparse-column (CSC) format and use O(|E|) neighbourhood sums instead of O(p²) dense operations.

Two primary MCMC sampling algorithms are provided:

| Function | Algorithm | Beta/alpha update | Typical use |
|---|---|---|---|
| `bvs_mh()` | Metropolis-Hastings (MH) by default; optional HMC/NUTS | Binary/TTE: default MH, or joint HMC/NUTS; Continuous: conjugate Gibbs | Flexible; any adjacency type |
| `bvs_pg()` | Pólya-Gamma Gibbs | Exact Gibbs via data augmentation | Improved mixing for beta/alpha |

---

## 2. Statistical Model

The outcome likelihood depends on `outcome_type`:

```
Binary (`outcome_type = "binary"`):
y_i | X, β, α, z_i, τ  ~  Bernoulli( logistic(α + x_i'β + z_i'τ) ),   i = 1, …, n

Continuous (`outcome_type = "continuous"`, MH only):
y_i | X, β, α, z_i, τ, σ²  ~  N( α + x_i'β + z_i'τ, σ² ),   i = 1, …, n

Time-to-event (`outcome_type = "TTE"`, MH/HMC/NUTS in `bvs_mh`):
log L(β, τ) = Σ_{i:δ_i=1} [ η_i − log Σ_{j∈R(t_i)} exp(η_j) ],
η_i = x_i'β + z_i'τ

Count (`outcome_type = "count"`, MH only):
y_i | w_i, X, β, α, z_i, τ ~ Poisson( w_i · exp(α + x_i'β + z_i'τ) ),
w_i ~ Gamma(r, r)
```

The spike-and-slab prior on coefficients conditional on the inclusion indicator γ:

```
β_j | γ_j, σ²  ~  γ_j · N(β₀, σ²)  +  (1 − γ_j) · δ₀
τ_k | σ²       ~  N(τ₀, h_τ · σ²)
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

### `bvs_mh` (Default Metropolis-Hastings (MH), Optional Hamiltonian Monte Carlo (HMC) / No-U-Turn Sampler (NUTS))

Per-iteration steps:

1. **Gamma and beta** (inner loop, repeated `n_thin_gb` times):
   - Propose γ_j flip with β_j drawn from the spike-and-slab prior if activating, or set to 0 if deactivating.
   - MH accept/reject using the outcome likelihood ratio and Ising prior ratio.
   - If `alg_type = "MH"` (default):
     - Binary outcome: random-walk MH perturbation (MALA) of active β_j.
     - TTE outcome: random-walk MH perturbation (Fisher-scoring) of active β_j.
     - Continuous outcome: conjugate Gaussian Gibbs update for active β_j.
     - Count outcome: MALA block update under conditional Poisson likelihood.
   - If `alg_type = "HMC"` or `"NUTS"` (for Binary/TTE):
     - Binary outcome: joint proposal of active β, α, τ, and log(σ²).
     - TTE outcome: joint proposal of active β, τ, and log(σ²), with α fixed at 0.
2. **Alpha, tau, and σ²**
   - Binary/count outcomes with `alg_type = "MH"`: random-walk MH updates.
   - Continuous outcome: conjugate Gibbs updates.
   - TTE outcome with `alg_type = "MH"`: alpha is fixed at 0 (not identifiable under partial likelihood); tau is updated by MH.
   - Under `alg_type = "HMC"` or `"NUTS"` for binary/TTE, these parameters are already updated in the joint Hamiltonian step.
3. **η (coupling)** — Möller et al. (2006) auxiliary variable MH with Propp-Wilson perfect simulation to handle the intractable Ising normalising constant.
4. **GGM adjacency** *(when `adj_type ∈ {"ggm", "ggm_fixed"}`)* — Wang (2012) Bayesian GGM column sweep with spike-and-slab precision priors.

All MH acceptance steps use the robust `safe_mh_accept()` function (see [Section 6](#6-numerical-robustness-robust-mh-acceptance)).

### Pólya-Gamma Gibbs (`bvs_pg`)

Per-iteration steps:

1. **Omega (PG latent variables)** — Sample ω_i ~ PG(1, α + x_i'β) for each observation.
2. **Beta and alpha** — Exact closed-form Gibbs draw from a multivariate normal, conditioned on ω and γ.
3. **Gamma (variable selection)** — Reversible-jump MH update for each predictor in random order, using the augmented likelihood.
4. **σ²** — Exact Inverse-Gamma Gibbs sample (same conjugate update as in `bvs_mh` continuous mode).
5. **η** — Möller auxiliary variable MH (same as `bvs_mh`).
6. **GGM adjacency** *(if applicable)* — Wang (2012) column sweep.

The Pólya-Gamma augmentation makes the beta/alpha update *exact* (no MH), which substantially improves mixing for regression coefficients.

---

## 4. Core Functions: `bvs_mh` and `bvs_pg`

### 4.1 `bvs_mh` — Metropolis-Hastings Sampler

```r
bvs_mh(
  X, y,
  event = NULL,             # required for outcome_type = "TTE" (1=event, 0=censored), ignored otherwise
  outcome_type = c("binary", "continuous", "TTE", "count"),
  adj_type  = c("fixed", "dual_fixed", "glasso", "glasso_fixed", "ggm", "ggm_fixed"),
  adj_fixed  = NULL,        # p×p binary adjacency matrix (required for fixed types)
  adj_fixed2 = NULL,        # second p×p matrix (required for dual_fixed only)
  sparse     = FALSE,       # TRUE → sparse backend (ggm / ggm_fixed only)
  S_ggm      = NULL,        # optional pre-computed scatter matrix; required when sparse and p >= 10000
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
  mu       = -log(1/0.1 - 1), # Ising external field (sparsity)
  alpha0   = 0,             # prior mean for intercept
  beta0    = 0,             # prior mean for coefficients
  # --- Gamma / Ising MH ---
  n_mh_gamma  = 3L,         # gamma flip proposals per iteration (GGM/sparse only)
  use_lb_gamma = TRUE,      # sparse GGM: locally-balanced (Zanella-style) gamma coordinate proposals
  eta1_sd     = 0.5,        # upper bound Uniform prior on η₁
  eta2_sd     = 0.5,        # upper bound Uniform prior on η₂ (dual models)
  mu_tilde    = -4,         # Möller auxiliary external field
  eta1_tilde  = 0.5,        # Möller auxiliary η₁ coupling
  eta2_tilde  = 0.5,        # Möller auxiliary η₂ coupling (dual models)
  e_eta = 1, f_eta = 1,     # Beta(e, f) prior shape on η
  Tmax  = 64L,              # Propp-Wilson maximum doubling horizon
  proposal_type = 1L,       # η proposal: 0=Uniform, 1=truncated Normal
  # --- GGM SSVS priors ---
  v0_ggm    = 0.015^2,      # spike variance
  v1_ggm    = NULL,         # slab variance (default: 50² × v0_ggm)
  pii_ggm   = NULL,         # edge inclusion probability (default: 4/(p-1))
  lambda_ggm = 1,           # GGM diagonal prior scale
  # --- Initialisation ---
  beta_init  = NULL,        # initial beta (length-p numeric vector)
  gamma_init = NULL,        # initial gamma (length-p 0/1 integer vector)
  alpha_init = NULL,        # initial intercept (scalar)
  # --- Always-included covariates ---
  z_dat      = NULL,        # always-included covariates (n×q)
  tau0       = 0,           # prior mean for tau
  htau       = 1.5,         # tau variance multiplier relative to sigma^2
  tau_init   = NULL,        # initial tau (length q)
  # --- Algorithm ---
  alg_type   = "MH",        # active-parameter-block update algorithm ("MH"=Metropolis-Hastings, "HMC"=Hamiltonian Monte Carlo, "NUTS"=No-U-Turn Sampler)
  hmc_step_size = 0.1,      # step size epsilon for HMC/NUTS
  hmc_n_leapfrog = 10L,     # number of leapfrog steps for HMC
  nuts_max_treedepth = 10L  # maximum tree depth for NUTS
)
```

**Key difference from `bvs_pg`:** The `n_thin_gb` parameter (default `3`) runs `n_thin_gb` inner gamma/beta update sub-iterations per MCMC step in dense MH backends. Sparse GGM backends ignore `n_thin_gb` for computational efficiency.

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
  mu = -log(1/0.1 - 1), alpha0 = 0, beta0 = 0,
  # --- Gamma / Ising MH ---
  n_mh_gamma = 3L,
  use_lb_gamma = TRUE,      # sparse GGM: locally-balanced (Zanella-style) gamma coordinate proposals
  eta1_sd = 0.5, eta2_sd = 0.5,
  mu_tilde = -4, eta1_tilde = 0.5, eta2_tilde = 0.5,
  e_eta = 1, f_eta = 1,
  Tmax = 64L, proposal_type = 1L,
  # --- GGM SSVS priors ---
  v0_ggm = 0.015^2, v1_ggm = NULL, pii_ggm = NULL, lambda_ggm = 1,
  # --- Block update ---
  block_size = 1L, pcg_threshold = 500L,
  # --- Initialisation ---
  beta_init = NULL, gamma_init = NULL, alpha_init = NULL,
  # --- Always-included covariates ---
  z_dat = NULL, tau0 = 0, htau = 1.5, tau_init = NULL
)
```

`bvs_pg` does not have `n_thin_gb` because the Pólya-Gamma augmentation gives exact Gibbs draws for beta and alpha — repeated proposals are not needed for mixing.
`bvs_pg` currently models binary outcomes only.

### 4.3 Return Value

Both functions return an S3 object of class `"bvs"`:

| Field | Type | Present when |
|---|---|---|
| `beta` | `niter × p` matrix or list | always (dense) / `store_beta = TRUE` (sparse) |
| `gamma` | `niter × p` matrix or list | always (dense) / `store_gamma = TRUE` (sparse) |
| `gamma_pip` | numeric vector, length `p`, or `NULL` | available when recoverable from stored gamma samples (dense runs and sparse runs with `store_gamma = TRUE`) |
| `alpha` | numeric vector, length `niter` | always |
| `sigmasq` | numeric vector, length `niter` | always |
| `tau` | matrix (`niter × q`) or `NULL` | `z_dat` supplied |
| `eta1` | numeric vector | single- and dual-network |
| `eta2` | numeric vector | dual-network models only |
| `Z_list` | list of sparse adjacency snapshots | `store_Z_list = TRUE` |
| `Z_pip` | sparse matrix (`p × p`) | `store_Z_pip = TRUE` (GGM types) |
| `call` | matched call | always |
| `outcome_type` | character | always for `bvs_mh` |
| `adj_type`, `sampler` | character | always; `sampler` is `"mh"`, `"hmc"`, `"nuts"`, or `"pg"` |
| `niter`, `burnin`, `p`, `n`, `ntau` | integer | always |

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

**Toy Example — `bvs_mh` with continuous outcome:**
```r
y_cont <- as.numeric(X %*% beta_true + rnorm(n, sd = 0.5))
fit_cont <- bvs_mh(
  X, y_cont,
  outcome_type = "continuous",
  adj_type = "fixed", adj_fixed = R,
  niter = 5, burnin = 2
)
fit_cont$outcome_type
```

**Toy Example — `bvs_mh` with overdispersed count outcome:**
```r
y_count <- rnbinom(n, size = 3, mu = exp(0.4 + as.numeric(X %*% beta_true) / 4))
fit_count <- bvs_mh(
  X, y_count,
  outcome_type = "count",
  adj_type = "fixed", adj_fixed = R,
  niter = 5, burnin = 2
)
fit_count$outcome_type
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

This robustness mechanism is applied to MH accept/reject steps across all 12 C++ backends:

- Gamma (variable selection) MH acceptance
- Beta/alpha/σ² proposal MH acceptance in binary-outcome MH backends
- η₁ and η₂ (coupling) Möller auxiliary MH acceptance

---

## 7. Inner Gamma/Beta Thinning (`n_thin_gb`)

**What it does:** In `bvs_mh()`, the `n_thin_gb` parameter (default `3`) wraps the inner gamma/beta update blocks in an inner `for` loop that runs `n_thin_gb` times per recorded MCMC iteration. Only every `n_thin_gb`-th state of γ and β is kept for inference.

**Why it helps:** Discrete inclusion indicators γ can mix slowly. Running multiple inner gamma/beta updates per stored sample improves exploration of the variable-selection space without inflating output size. In binary and count modes, these are MH moves; in continuous mode, beta updates are conjugate Gibbs draws.

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
- **ESS** — bulk effective sample size for β_j (rank-normalised, Vehtari et al. 2021).
- **ESS_tail** — tail effective sample size (min of ESS at 5th and 95th percentiles).

Returns a named list including `pip`, `selected` (integer indices with PIP > `pip_threshold`), `nselected`, `summary_beta` (data frame with columns `Mean`, `SD`, `PIP`, `Q_low`, `Q_up`, `HPD_low`, `HPD_up`, `ESS`, `ESS_tail`), `summary_alpha`, `sampler`, `adj_type`.

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

### 8.5 Modern Convergence Diagnostics (Vehtari et al. 2021)

Three exported functions implement rank-normalised diagnostics following Vehtari, Gelman, Simpson, Carpenter, and Bürkner (2021). These improve on classical R-hat and ESS by detecting non-Gaussianity and heavy tails in the posterior.

```r
# Single-chain usage (chain automatically split in half):
rhat_eta  <- bvs_rhat(fit$eta1)        # rank-normalised split-R-hat; target < 1.01
ess_bulk  <- bvs_ess_bulk(fit$eta1)    # bulk ESS (body of posterior)
ess_tail  <- bvs_ess_tail(fit$eta1)    # tail ESS (min ESS at 5th and 95th pct)

# Multi-chain usage (matrix with one column per chain):
eta_mat <- cbind(fit1$eta1, fit2$eta1)
bvs_rhat(eta_mat)
bvs_ess_bulk(eta_mat)
bvs_ess_tail(eta_mat)
```

These functions delegate to the `posterior` package when installed (`posterior::rhat()`, `posterior::ess_bulk()`, `posterior::ess_tail()`); otherwise a pure-R fallback implementing the same algorithm is used.

The `summary.bvs()` function uses `bvs_ess_bulk` and `bvs_ess_tail` internally, so the `ESS` and `ESS_tail` columns of `summary_beta` already reflect the improved diagnostics.

**Reference:** Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner P-C (2021). Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC (with discussion). *Bayesian Analysis*, 16(2), 667–718. DOI: 10.1214/20-BA1221.

---

## 9. Eta Hyperparameter Tuning

The Ising coupling parameter η governs how strongly neighbouring predictors co-select. Its prior is Uniform(0, `eta_sd`) combined with the Möller auxiliary variable construction. Choosing `eta_sd` and the Möller hyperparameters (`mu_tilde`, `eta1_tilde`, `eta2_tilde`) is crucial for valid inference.

The package provides a three-stage tuning workflow:

### Stage 1 — Phase transition profiling

Locate the **critical coupling strength** η* at which the Ising prior transitions from sparse to dense models for your specific adjacency matrix.

```r
pt <- phase_transition(R, mu = -log(1/0.1 - 1),
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
  e_eta       = 1, f_eta = 1
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

Pre-computes the scatter matrix `S = X'X` in CSC format for the sparse GGM backends. Entries with `|S_ij| < threshold` are treated as structural zeros. Use this when the same `X` is used across many MCMC runs to avoid recomputing the scatter matrix repeatedly.

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

For sparse backends, set `store_beta = FALSE` to reduce memory. If you need `gamma_pip`, keep `store_gamma = TRUE` so PIPs can be reconstructed from stored sparse gamma samples.

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

- **`BayesLogit_Numerics.h`** — `safe_mh_accept()`, `robust_chol_inplace()`, `clamp_finite()`, `sanitize_sym_mat_inplace()`, and other numerical utilities. Included by all backends.
- **`BayesLogit_Sparse_Helpers.h`** — CSC-format scatter matrix accessors (`ConstSparseS`, `ConstSparseAdj`), `column_ll_diff()`, `apply_column_update()`, Propp-Wilson helpers (`proppwilson_single_sparse`, `proppwilson_dual_sparse`), Möller update functions (`moller_update_single_sparse`, `moller_update_dual_sparse`), `ggm_column_sweep_sparse()`, storage helpers, and:
  - **`EtaAdapter`** struct (Vihola 2012 RAM): 1D Robbins-Monro log-σ update with n^{-2/3} step schedule targeting 44% acceptance; used in all 12 backend Möller η updates.
  - **Locally-balanced gamma helpers** (Zanella 2020): `init_lb_single_scores_dense`, `init_lb_dual_scores_dense`, `init_lb_single_scores_ggm`, `init_lb_dual_scores_ggm` (dense); `init_lb_single_scores`, `init_lb_dual_scores` (sparse). Plus matching `build_lb_*_delta_*` and `apply_lb_delta` functions. Used by all 12 backends when `use_lb_gamma = TRUE`.
  - Included by all 12 backends.

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
              mu = -log(1/0.1 - 1), eta1_sd = 0.5)

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
  pii_ggm     = 4 / (p - 1),
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
pt <- phase_transition(R, mu = -log(1/0.1 - 1),
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
| `outcome_type` | `"binary"` | `bvs_mh` only | Outcome model: `"binary"` (logistic), `"continuous"` (Gaussian), `"TTE"` (Cox partial likelihood), or `"count"` (negative-binomial via Poisson-Gamma augmentation) |
| `event` | `NULL` | `bvs_mh` only | Event indicator for `outcome_type = "TTE"` (`1`=event, `0`=censored; `{-1,1}` accepted), ignored otherwise |
| `alg_type` | `"MH"` | `bvs_mh` only | Active-parameter-block update algorithm: `"MH"` = Metropolis-Hastings (default), `"HMC"` = Hamiltonian Monte Carlo, or `"NUTS"` = No-U-Turn Sampler; HMC/NUTS are supported for binary and TTE outcomes |
| `hmc_step_size` | `0.1` | `bvs_mh` only | Initial HMC/NUTS step size; NUTS adapts during burn-in |
| `hmc_n_leapfrog` | `10` | `bvs_mh` only | Number of leapfrog steps for HMC |
| `nuts_max_treedepth` | `10` | `bvs_mh` only | Maximum tree depth for NUTS |
| `niter` | `60000` | both | Post-burn-in MCMC iterations stored |
| `burnin` | `10000` | both | Burn-in iterations (discarded) |
| `thin` | `1` | both | Storage thinning interval |
| `n_thin_gb` | `3` | `bvs_mh` only | Inner MH sub-iterations for γ + β (dense backends) |
| `n_mh_gamma` | `3` | both | γ flip proposals per outer iteration (GGM/sparse) |
| `use_lb_gamma` | `TRUE` | both | Use Zanella (2020) locally-balanced `g(t)=√t` coordinate proposals for γ updates across **all 12 backends** (sparse and dense); set `FALSE` for uniform-coordinate baseline |

### Variable selection priors

| Parameter | Default | Description |
|---|---|---|
| `nu0` | `2` | IG(ν₀/2, ν₀σ₀²/2) shape for σ² |
| `sigmasq0` | `1.5` | IG scale for σ² |
| `h` | `1.5` | Intercept variance inflation (α ~ N(α₀, h·σ²)) |
| `mu` | `-log(1/0.1 − 1)` | Ising external field (≈ -2.197; prior expected model size ≈ 10% of p) |
| `alpha0` | `0` | Prior mean for intercept α |
| `beta0` | `0` | Prior mean for regression coefficients β_j |
| `tau0` | `0` | Prior mean for regression coefficients τ associated with `z_dat` |
| `htau` | `1.5` | τ variance multiplier relative to σ² |

### Ising / Möller hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `eta1_sd` | `0.5` | Upper bound Uniform prior on η₁ |
| `eta2_sd` | `0.5` | Upper bound Uniform prior on η₂ (dual models) |
| `e_eta`, `f_eta` | `1`, `1` | Beta(e, f) prior shape on η |
| `mu_tilde` | `-4` | Möller auxiliary MRF external field |
| `eta1_tilde` | `0.5` | Möller auxiliary η₁ coupling |
| `eta2_tilde` | `0.5` | Möller auxiliary η₂ coupling (dual models) |
| `Tmax` | `64` | Propp-Wilson maximum doubling horizon |
| `proposal_type` | `1` | η proposal: `0` = Uniform, `1` = truncated Normal |

### GGM SSVS priors

| Parameter | Default | Description |
|---|---|---|
| `v0_ggm` | `0.015²` | Spike variance for off-diagonal precision entries |
| `v1_ggm` | `50² × v0_ggm` | Slab variance |
| `pii_ggm` | `4 / (p − 1)` | Prior edge inclusion probability |
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

## 15. MCMC Mixing and Convergence Improvements

Three complementary improvements have been implemented across all 12 C++ backends and the R post-processing layer.

### 15.1 IMP-6: Locally-Balanced γ Proposals (Zanella 2020)

**Motivation.** The standard MH update for the binary inclusion indicators γ proposes a coordinate to flip uniformly at random, which can be inefficient when most coordinates have low acceptance probability. Zanella (2020) introduced *locally balanced* proposals that weight each candidate coordinate by the square-root of the local MH acceptance probability.

**Implementation.** For each predictor j, the unnormalised proposal weight is

```
w_j = exp( 0.5 × clip[ (1 − 2γ_j)(μ + η·N_j) ] )
```

where N_j = Σ_k γ_k R_{jk} is the active-neighbour count and `clip` bounds the argument to [−60, 60]. The balancing function g(t) = √t corresponds to the optimal "Barker" kernel for discrete targets. Variable j is selected with probability proportional to w_j. The MH ratio is corrected by the log proposal ratio log(q_rev / q_fwd).

**Files modified.** `src/BayesLogit_Sparse_Helpers.h` (helper functions), all 12 backend `.cpp` files. Controlled by `use_lb_gamma` parameter (default `TRUE`).

**Expected gain.** Locally balanced proposals increase ESS for γ indicators by 2–10× in sparse high-dimensional settings compared to uniform-coordinate proposals, at the cost of O(p) weight computation per flip.

**Reference:** Zanella, G. (2020). Informed proposals for local MCMC in discrete spaces. *Journal of the American Statistical Association*, 115(530), 852–865. DOI: 10.1080/01621459.2019.1585255.

---

### 15.2 IMP-7: Vihola (2012) Robust Adaptive Metropolis for η

**Motivation.** The Ising coupling parameters η₁ and η₂ are updated with a scalar random-walk MH proposal. A fixed proposal standard deviation performs poorly unless carefully tuned. The Robust Adaptive Metropolis (RAM) algorithm of Vihola (2012) automates this by adapting the log-scale proposal SD to coerce the empirical acceptance rate toward a target.

**Implementation.** Each backend declares one or two `EtaAdapter` objects (defined in `BayesLogit_Sparse_Helpers.h`):

```cpp
struct EtaAdapter {
  double log_sigma;    // log of current proposal SD
  double alpha_star;   // target acceptance rate (0.44 for 1D)
  double gamma_decay;  // Robbins-Monro exponent (2/3)
  int    n_adapt;      // adaptation counter

  void update(double accept_prob) {
    double step = std::min(1.0, std::pow((double)n_adapt, -gamma_decay));
    log_sigma += 0.5 * step * (accept_prob - alpha_star);
    ++n_adapt;
  }
  double sigma() const { return std::exp(std::max(-10.0, std::min(5.0, log_sigma))); }
};
```

After each Möller MH step, `adapter.update(accept_prob)` adjusts log_sigma. The n^{-2/3} diminishing step size schedule satisfies the diminishing-adaptation condition (Theorem 2 of Vihola 2012), ensuring ergodicity of the combined chain.

**Files modified.** `src/BayesLogit_Sparse_Helpers.h` (`EtaAdapter` struct, `moller_update_single_sparse`, `moller_update_dual_sparse`), all 12 backend `.cpp` files.

**Expected gain.** Automatic tuning of η acceptance rates; typical acceptance rates converge to 44% (optimal for 1D scalar targets per Roberts, Gelman & Gilks 1997) without manual grid search.

**References:**
- Vihola, M. (2012). Robust adaptive Metropolis algorithm with coerced acceptance rate. *Statistics and Computing*, 22(5), 997–1008. DOI: 10.1007/s11222-011-9269-5.
- Andrieu, C., & Thoms, J. (2008). A tutorial on adaptive MCMC. *Statistics and Computing*, 18(4), 343–373.

---

### 15.3 IMP-8: Rank-Normalised R̂ and Bulk/Tail ESS (Vehtari et al. 2021)

**Motivation.** The classical Gelman-Rubin R̂ and the `coda::effectiveSize` ESS assume approximately Gaussian posteriors. For heavy-tailed or multimodal chains these diagnostics are unreliable. Vehtari et al. (2021) propose (i) splitting each chain in half to double the chain count before computing R̂, (ii) rank-normalising the draws to assess Gaussianity, and (iii) separately computing bulk ESS (via Geyer's monotone sequence estimator) and tail ESS (minimum of ESS at the 5th and 95th percentiles).

**Implementation.** Three exported R functions in `R/bvs_summary.R`:

| Function | Description |
|---|---|
| `bvs_rhat(x)` | Rank-normalised split-R̂: max(R̂_bulk, R̂_tail). Values < 1.01 indicate convergence. |
| `bvs_ess_bulk(x)` | Bulk ESS on rank-normalised split chains. Measures sampling efficiency for the body of the posterior. |
| `bvs_ess_tail(x)` | Tail ESS = min(ESS at 5th pct, ESS at 95th pct). Measures sampling efficiency in the tails. |

Each function accepts a numeric vector (single chain, automatically split in half) or a matrix (columns = independent chains). When the `posterior` package is installed it delegates to `posterior::rhat()`, `posterior::ess_bulk()`, and `posterior::ess_tail()` for speed and additional features.

The `summary.bvs()` function uses these internally: the `ESS` and `ESS_tail` columns of `summary_beta` reflect bulk and tail ESS respectively.

```r
# Access directly:
s <- summary(fit)
s$summary_beta[, c("ESS", "ESS_tail")]   # per-variable bulk and tail ESS

# Or use directly on any parameter chain:
bvs_rhat(fit$eta1)      # R-hat for coupling parameter
bvs_ess_bulk(fit$eta1)  # bulk ESS
bvs_ess_tail(fit$eta1)  # tail ESS
```

**Files modified.** `R/bvs_summary.R` (new functions `bvs_rhat`, `bvs_ess_bulk`, `bvs_ess_tail`; updated `summary.bvs()`). The `posterior` package added to `Suggests` in `DESCRIPTION`.

**Reference:** Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner P-C (2021). Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC (with discussion). *Bayesian Analysis*, 16(2), 667–718. DOI: 10.1214/20-BA1221.

---

## 16. References

- Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. *arXiv preprint arXiv:1701.02434*.
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. *J. Mach. Learn. Res.*, 15(1), 1593–1623.
- Li, F., & Zhang, N. R. (2010). Bayesian variable selection in structured high-dimensional covariate spaces. *JASA*, 105, 1202–1214.
- Möller, J., Pettitt, A. N., Reeves, R., & Berthelsen, K. K. (2006). An efficient MCMC method for distributions with intractable normalising constants. *Biometrika*, 93(2), 451–458.
- Murray, I., Ghahramani, Z., & MacKay, D. J. C. (2006). MCMC for doubly-intractable distributions. *UAI 2006*.
- Nishimura, A., & Suchard, M. A. (2022). Prior-preconditioned conjugate gradient for accelerated Gibbs sampling in large n, large p Bayesian sparse regression. *JASA*. DOI: 10.1080/01621459.2022.2057859.
- Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference for logistic models using Pólya-Gamma latent variables. *JASA*, 108(504), 1339–1349.
- Propp, J. G., & Wilson, D. B. (1996). Exact sampling with coupled Markov chains. *Random Structures and Algorithms*, 9, 223–252.
- Stingo, F. C., Chen, Y. A., Tadesse, M. G., & Vannucci, M. (2011). Incorporating biological information into linear models: A Bayesian approach. *Annals of Applied Statistics*, 5, 1978–2002.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved R̂. *Bayesian Analysis*, 16(2), 667–718.
- Vihola, M. (2012). Robust adaptive Metropolis algorithm with coerced acceptance rate. *Statistics and Computing*, 22(5), 997–1008.
- Andrieu, C., & Thoms, J. (2008). A tutorial on adaptive MCMC. *Statistics and Computing*, 18(4), 343–373.
- Wang, H. (2012). Bayesian graphical lasso models and efficient posterior computation. *Bayesian Analysis*, 7(4), 867–886.
- Wang, H. (2015). Scaling it up: Stochastic search structure learning in graphical models. *Bayesian Analysis*, 10(2), 351–377.
- Zanella, G. (2020). Informed proposals for local MCMC in discrete spaces. *JASA*, 115(530), 852–865.

---

*BVS.DAdj version 0.1.0 · GPL (≥ 2) · Maintained by Yubing Yao <yyao@umass.edu>*
