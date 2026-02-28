# BVS.DAdj

**Bayesian Variable Selection with Dual-Adjacency Network-Informed Ising Priors**

MCMC methods for Bayesian variable selection in logistic regression with Ising/MRF priors informed by graphical model adjacency matrices. Supports Metropolis-Hastings and Polya-Gamma augmented Gibbs sampling across six adjacency-structure types, with dense and sparse high-dimensional backends.

## Features

| Feature | Description |
|---|---|
| **Two MCMC samplers** | Metropolis-Hastings (`bvs_mh`) and Polya-Gamma Gibbs (`bvs_pg`) |
| **Six adjacency types** | `"fixed"`, `"dual_fixed"`, `"glasso"`, `"glasso_fixed"`, `"ggm"`, `"ggm_fixed"` |
| **Dense & sparse backends** | 12 optimised C++ backends for moderate and high-dimensional settings |
| **Robust MH acceptance** | `safe_mh_accept()` guards against `log(0)`, NaN, and ±Inf in all MH steps |
| **Inner gamma/beta thinning** | `n_thin_gb` sub-iterations per MCMC step for better mixing (MH dense backends) |
| **Ising coupling estimation** | Möller (2006) auxiliary variable MH with Propp-Wilson perfect simulation |
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
grid <- bvs_eta_grid_single(adj = R, eta_frac = seq(0.2, 1.0, by = 0.2))
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
| `niter` | `60000` | Post-burn-in MCMC iterations |
| `burnin` | `10000` | Burn-in iterations (discarded) |
| `thin` | `1` | Storage thinning interval |
| `n_thin_gb` | `3` | Inner MH sub-iterations for gamma and beta per MCMC step *(MH dense backends only)* |
| `n_mh_gamma` | `3` | Gamma flip proposals per iteration *(GGM/sparse backends)* |

### Ising prior

| Parameter | Default | Description |
|---|---|---|
| `mu` | `-log(1/0.3 − 1)` | External field (controls baseline sparsity) |
| `eta1_sd` | `0.5` | Upper bound of Uniform prior on η₁ |
| `eta2_sd` | `0.5` | Upper bound of Uniform prior on η₂ *(dual models)* |
| `e_eta`, `f_eta` | `2`, `1` | Beta(e, f) prior shape on η |

### Möller auxiliary MH

| Parameter | Default | Description |
|---|---|---|
| `mu_tilde` | `-4` | Auxiliary MRF external field |
| `eta1_tilde` | `0.075` | Auxiliary η₁ coupling |
| `eta2_tilde` | `0.065` | Auxiliary η₂ coupling *(dual models)* |
| `Tmax` | `64` | Propp-Wilson maximum doubling horizon |
| `proposal_type` | `1` | η proposal kernel: `0` = Uniform, `1` = truncated Normal |

### GGM SSVS priors (when `adj_type` includes `"ggm"`)

| Parameter | Default | Description |
|---|---|---|
| `v0_ggm` | `0.015²` | Spike variance |
| `v1_ggm` | `50² × 0.015²` | Slab variance |
| `pii_ggm` | `30 / (p − 1)` | Edge inclusion prior probability |
| `lambda_ggm` | `1` | Prior scale for diagonal precision entries |

### Regression priors

| Parameter | Default | Description |
|---|---|---|
| `nu0`, `sigmasq0` | `2`, `1.5` | IG(ν₀/2, ν₀σ₀²/2) prior on σ² |
| `h` | `1.5` | Intercept variance inflation factor |
| `alpha0`, `beta0` | `0`, `0` | Prior means for intercept and coefficients |

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
| `gamma_pip` | numeric vector | Posterior inclusion probabilities (length `p`) |
| `alpha` | numeric vector | Posterior intercept samples |
| `sigmasq` | numeric vector | Posterior σ² samples |
| `eta1` | numeric vector | Posterior η₁ samples |
| `eta2` | numeric vector | Posterior η₂ samples *(dual models)* |
| `Z_list` | list | GGM adjacency snapshots *(if `store_Z_list = TRUE`)* |
| `Z_pip` | sparse matrix | GGM edge PIPs *(if `store_Z_pip = TRUE`)* |
| `call` | call | Matched function call |
| `adj_type`, `sampler` | character | Model identifiers |

Use `summary()`, `print()`, and `plot()` S3 methods for post-processing.

## Authors

- **Yubing Yao** — yyao@umass.edu
- **Mahlet Tadesse** — mgt26@georgetown.edu
- **Raji Balasubramanian** — rbalasub@schoolph.umass.edu

## References

- Wang, H. (2012). Bayesian Graphical Lasso Models and Efficient Posterior Computation. *Bayesian Analysis*, 7(4), 867–886.
- Möller, J., Pettitt, A. N., Reeves, R., & Berthelsen, K. K. (2006). An efficient Markov chain Monte Carlo method for distributions with intractable normalising constants. *Biometrika*, 93(2), 451–458.
- Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference for logistic models using Pólya–Gamma latent variables. *Journal of the American Statistical Association*, 108(504), 1339–1349.
- Liu, F., & Chakraborty, S. (2022). Bayesian variable selection with network-structured priors. *(Related methodology reference.)*

## License

GPL (>= 2)
