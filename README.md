# BVS.DAdj

**Bayesian Variable Selection with Dual-Adjacency Network-Informed Ising Priors**

MCMC methods for Bayesian variable selection with Ising/MRF priors informed by graphical model adjacency matrices. `bvs_mh()` supports binary logistic outcomes (default), continuous Gaussian outcomes, right-censored time-to-event outcomes via Cox partial likelihood, and overdispersed count outcomes via a negative-binomial (Poisson-Gamma) representation; `bvs_pg()` targets binary logistic outcomes. Both samplers support six adjacency-structure types with dense and sparse high-dimensional backends.

## Features

| Feature | Description |
|---|---|
| **Two MCMC samplers** | Metropolis-Hastings (`bvs_mh`) and Polya-Gamma Gibbs (`bvs_pg`) |
| **Outcome support** | `bvs_mh`: `outcome_type = "binary"`, `"continuous"`, `"TTE"`, or `"count"`; `bvs_pg`: binary logistic |
| **Six adjacency types** | `"fixed"`, `"dual_fixed"`, `"glasso"`, `"glasso_fixed"`, `"ggm"`, `"ggm_fixed"` |
| **Dense & sparse backends** | 12 optimised C++ backends for moderate and high-dimensional settings |
| **Robust MH acceptance** | `safe_mh_accept()` guards against `log(0)`, NaN, and ±Inf in all MH steps |
| **Inner gamma/beta thinning** | `n_thin_gb` sub-iterations per MCMC step for better mixing (MH dense backends) |
| **Ising/MRF coupling estimation** | Möller (2006) auxiliary variable MH with Propp-Wilson perfect simulation |
| **Phase transition detection** | Sweep eta to locate the critical coupling strength and calibrate priors |
| **Eta hyperparameter tuning** | Grid-search functions to select optimal Möller hyperparameters |
| **Rich diagnostics** | PIP, trace, ACF plots; Gelman-Rubin PSRF via `coda` |

## Installation

```r
# Install from GitHub
devtools::install_github("yuy113/BVS.DAdj")
```

### Dependencies

- **Required:** `Rcpp`, `RcppArmadillo`, `Matrix`, `methods`, `stats`, `graphics`, `coda`
- **Suggested:** `huge` (graphical lasso adjacency), `MASS`, `testthat`, `knitr`, `rmarkdown`

## Quick Start

```r
library(BVS.DAdj)

# Simulate data
set.seed(42)
n <- 200; p <- 50
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(rep(1, 5), rep(0, p - 5))
y <- rbinom(n, 1, plogis(X %*% beta_true))

# Build adjacency matrix (block structure encoding prior beliefs)
R <- matrix(0L, p, p)
R[1:5, 1:5] <- 1L; diag(R) <- 0L
```

### Example 1 — MH sampler with fixed adjacency

```r
fit <- bvs_mh(X, y, adj_type = "fixed", adj_fixed = R,
              niter = 10000, burnin = 2000, n_thin_gb = 3L)
summary(fit)
plot(fit, type = "pip")
```

### Example 1b — MH sampler with continuous outcome

```r
y_cont <- as.numeric(X %*% beta_true + rnorm(n, sd = 0.5))
fit_cont <- bvs_mh(
  X, y_cont,
  outcome_type = "continuous",
  adj_type = "fixed", adj_fixed = R,
  niter = 10000, burnin = 2000
)
fit_cont$outcome_type  # "continuous"
```

### Example 1c — MH sampler with right-censored time-to-event outcome

```r
# Simulate simple TTE data with random censoring
linpred <- as.numeric(X %*% beta_true)
tte_time <- rexp(n, rate = exp(linpred / 4))
tte_event <- rbinom(n, 1, 0.7)  # 1=event, 0=censored

fit_tte <- bvs_mh(
  X, tte_time,
  event = tte_event,
  outcome_type = "TTE",
  adj_type = "fixed", adj_fixed = R,
  niter = 10000, burnin = 2000
)
fit_tte$outcome_type  # "TTE"
```

### Example 1d — MH sampler with overdispersed count outcome

```r
mu_count <- exp(0.4 + as.numeric(X %*% beta_true) / 4)
y_count <- rnbinom(n, size = 3, mu = mu_count)

fit_count <- bvs_mh(
  X, y_count,
  outcome_type = "count",
  adj_type = "fixed", adj_fixed = R,
  niter = 10000, burnin = 2000
)
fit_count$outcome_type  # "count"
```

### Example 2 — PG Gibbs sampler with graphical lasso adjacency (EBIC)

```r
fit <- bvs_pg(X, y, adj_type = "glasso", glasso_criterion = "ebic",
              niter = 10000, burnin = 2000)
summary(fit)
```

### Example 3 — PG with sparse Bayesian GGM (high-dimensional, p >> n)

```r
fit <- bvs_pg(X, y, adj_type = "ggm", sparse = TRUE,
              niter = 10000, burnin = 2000)
s <- summary(fit)
s$selected   # indices of selected variables at PIP > 0.5
s$pip        # full posterior inclusion probability vector
# For sparse MH backends with p >= 10000, provide S_ggm explicitly.
```

### Example 4 — MH with GGM + fixed adjacency (dual-eta)

```r
fit <- bvs_mh(X, y, adj_type = "ggm_fixed", adj_fixed = R,
              sparse = FALSE, niter = 10000, burnin = 2000,
              n_thin_gb = 3L)
summary(fit)
```

### Example 5 — Phase transition detection and eta calibration

```r
# Profile the phase transition to find the critical coupling strength
pt <- phase_transition(R, max_eta = 1.5, step_size = 0.05, num_rep = 10)
matplot(pt, type = "l", xlab = "eta index", ylab = "model size",
        main = "Ising Phase Transition")

# Build a tuning grid anchored at the phase-transition point
grid <- bvs_eta_grid_single(adj = R, eta1_frac = seq(0.2, 1.0, by = 0.2))
head(grid)
```

### Example 6 — Multi-chain Gelman-Rubin convergence check

```r
fit1 <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R,
               niter = 10000, burnin = 2000)
fit2 <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R,
               niter = 10000, burnin = 2000)
bvs_gelman_diag(list(fit1, fit2))
```

## Key Parameters

### MCMC control

| Parameter | Default | Description |
|---|---|---|
| `outcome_type` | `"binary"` | Outcome model for `bvs_mh`: `"binary"` (logistic), `"continuous"` (Gaussian), `"TTE"` (Cox partial likelihood), or `"count"` (negative-binomial via Poisson-Gamma augmentation) |
| `event` | `NULL` | Event indicator for `outcome_type = "TTE"` (`1`=event, `0`=censored; `{-1,1}` also accepted); ignored for other outcomes |
| `niter` | `60000` | Post-burn-in MCMC iterations |
| `burnin` | `10000` | Burn-in iterations (discarded) |
| `thin` | `1` | Storage thinning interval |
| `n_thin_gb` | `3` | Inner MH sub-iterations for gamma and beta per MCMC step *(MH dense backends only)* |
| `n_mh_gamma` | `3` | Gamma flip proposals per iteration *(GGM/sparse backends)* |

### Ising prior

| Parameter | Default | Description |
|---|---|---|
| `mu` | `-log(1/0.1 − 1)` | External field (controls baseline sparsity) |
| `eta1_sd` | `0.5` | Upper bound of Uniform prior on η₁ |
| `eta2_sd` | `0.5` | Upper bound of Uniform prior on η₂ *(dual models)* |
| `e_eta`, `f_eta` | `1`, `1` | Beta(e, f) prior shape on η |

### Möller auxiliary MH

| Parameter | Default | Description |
|---|---|---|
| `mu_tilde` | `-4` | Auxiliary MRF external field |
| `eta1_tilde` | `0.5` | Auxiliary η₁ coupling |
| `eta2_tilde` | `0.5` | Auxiliary η₂ coupling *(dual models)* |
| `Tmax` | `64` | Propp-Wilson maximum doubling horizon |
| `proposal_type` | `1` | η proposal kernel: `0` = Uniform, `1` = truncated Normal |

### GGM SSVS priors (when `adj_type` includes `"ggm"`)

| Parameter | Default | Description |
|---|---|---|
| `v0_ggm` | `0.015²` | Spike variance |
| `v1_ggm` | `50² × 0.015²` | Slab variance |
| `pii_ggm` | `4 / (p − 1)` | Edge inclusion prior probability |
| `lambda_ggm` | `1` | Prior scale for diagonal precision entries |

### Regression priors

| Parameter | Default | Description |
|---|---|---|
| `nu0`, `sigmasq0` | `2`, `1.5` | IG(ν₀/2, ν₀σ₀²/2) prior on σ² |
| `h` | `1.5` | Intercept variance inflation factor |
| `alpha0`, `beta0` | `0`, `0` | Prior means for intercept and regression coefficients `beta` |
| `tau0` | `0` | Prior mean for regression coefficients `tau` associated with `z_dat` |
| `htau` | `1.5` | `tau` variance multiplier relative to `sigma^2` |

## Adjacency Types

| `adj_type` | Etas | Adjacency source | Sparse backend |
|---|---|---|---|
| `"fixed"` | 1 | External matrix (`adj_fixed`) | No |
| `"dual_fixed"` | 2 | Two external matrices | No |
| `"glasso"` | 1 | Frequentist graphical lasso (requires `huge`) | No |
| `"glasso_fixed"` | 2 | Glasso + external fixed | No |
| `"ggm"` | 1 | Bayesian GGM SSVS (Wang 2012) | Yes (set `sparse = TRUE`) |
| `"ggm_fixed"` | 2 | Bayesian GGM + external fixed | Yes (set `sparse = TRUE`) |

## C++ Backends

12 hand-optimised C++ translation units in `src/` cover all sampler × adjacency × dimension combinations. Every MH acceptance step uses the numerically robust `bvs_dadj::safe_mh_accept()` helper (handles `log(0)`, NaN, and ±Inf without deadlock).

| File | Sampler | Adjacency | Etas | Dimension |
|---|---|---|---|---|
| `BayesLogit_SingleNet_FixedAdj.cpp` | MH | 1 fixed | 1 | dense |
| `BayesLogit_DualNet_FixedAdj.cpp` | MH | 2 fixed | 2 | dense |
| `BayesLogit_SingleNet_GGM.cpp` | MH | 1 GGM | 1 | dense |
| `BayesLogit_DualNet_GGM.cpp` | MH | 1 GGM + 1 fixed | 2 | dense |
| `BayesLogit_PG_SingleAdj.cpp` | PG | 1 fixed | 1 | dense |
| `BayesLogit_PG_DualAdj.cpp` | PG | 2 fixed | 2 | dense |
| `BayesLogit_PG_SingleAdj_GGM_Moller.cpp` | PG | 1 GGM | 1 | dense |
| `BayesLogit_PG_GGM_Moller.cpp` | PG | 1 GGM + 1 fixed | 2 | dense |
| `BayesLogit_SingleNet_SparseGGM.cpp` | MH | 1 sparse GGM | 1 | sparse |
| `BayesLogit_DualNet_SparseGGM.cpp` | MH | 1 sparse GGM + 1 fixed | 2 | sparse |
| `BayesLogit_PG_SingleAdj_SparseGGM.cpp` | PG | 1 sparse GGM | 1 | sparse |
| `BayesLogit_PG_DualNet_SparseGGM.cpp` | PG | 1 sparse GGM + 1 fixed | 2 | sparse |

Shared headers:
- `BayesLogit_Numerics.h` — numerical utilities including `safe_mh_accept()`
- `BayesLogit_Sparse_Helpers.h` — CSC sparse-matrix operations, Möller/Propp-Wilson helpers

## Return Value

Both `bvs_mh()` and `bvs_pg()` return an S3 object of class `"bvs"`:

| Field | Type | Description |
|---|---|---|
| `beta` | matrix or list | Posterior beta samples (`niter × p`, or sparse list) |
| `gamma` | matrix or list | Posterior gamma samples (`niter × p`, or sparse list) |
| `gamma_pip` | numeric vector or `NULL` | Posterior inclusion probabilities (length `p`), or `NULL` when unavailable (for example sparse runs with `store_gamma = FALSE`) |
| `alpha` | numeric vector | Posterior intercept samples |
| `sigmasq` | numeric vector | Posterior σ² samples |
| `tau` | matrix or `NULL` | Posterior samples for always-included covariates (`z_dat`) |
| `eta1` | numeric vector | Posterior η₁ samples |
| `eta2` | numeric vector | Posterior η₂ samples *(dual models)* |
| `Z_list` | list | GGM adjacency snapshots *(if `store_Z_list = TRUE`)* |
| `Z_pip` | sparse matrix | GGM edge PIPs *(if `store_Z_pip = TRUE`)* |
| `call` | call | Matched function call |
| `outcome_type` | character | Present for `bvs_mh` (`"binary"`, `"continuous"`, `"TTE"`, or `"count"`) |
| `adj_type`, `sampler` | character | Model identifiers |
| `niter`, `burnin`, `p`, `n`, `ntau` | integer | Stored run dimensions and counts |

Use `summary()`, `print()`, and `plot()` S3 methods for post-processing.

## Authors

- **Yubing Yao** — yyao@umass.edu
- **Mahlet Tadesse** — mgt26@georgetown.edu
- **Raji Balasubramanian** — rbalasub@schoolph.umass.edu

## References

- Wang, H. (2015). Scaling It Up: Stochastic Search Structure Learning in Graphical Models *Bayesian Analysis*, 10(2), 351–377.
- Möller, J., Pettitt, A. N., Reeves, R., & Berthelsen, K. K. (2006). An efficient Markov chain Monte Carlo method for distributions with intractable normalising constants. *Biometrika*, 93(2), 451–458.
- Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference for logistic models using Pólya–Gamma latent variables. *Journal of the American Statistical Association*, 108(504), 1339–1349.

## License

GPL (>= 2)
