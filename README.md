# BVS.DAdj

**Bayesian Variable Selection with Dual-Adjacency Network-Informed Ising Priors**

MCMC methods for Bayesian variable selection in logistic regression with Ising/MRF priors informed by graphical model adjacency matrices. Supports Metropolis-Hastings and Polya-Gamma augmented Gibbs sampling.

## Features

| Feature | Description |
|---|---|
| **Two MCMC samplers** | Metropolis-Hastings (`bvs_mh`) and Polya-Gamma Gibbs (`bvs_pg`) |
| **Five adjacency types** | `"fixed"`, `"dual_fixed"`, `"glasso"`, `"ggm"`, `"ggm_fixed"` |
| **Dense & sparse backends** | Optimized C++ for both moderate and high-dimensional settings |
| **Ising coupling estimation** | Möller (2006) auxiliary variable MH with Propp-Wilson perfect simulation |
| **Phase transition detection** | Sweep eta to find the critical coupling strength |

## Installation

```r
# Install from GitHub
devtools::install_github("yuy113/BVS.DAdj")
```

### Dependencies

- **Required:** `Rcpp`, `RcppArmadillo`, `Matrix`
- **Suggested:** `huge` (for graphical lasso adjacency), `MASS`

## Quick Start

```r
library(BVS.DAdj)

# Simulate data
set.seed(42)
n <- 200; p <- 50
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(rep(1, 5), rep(0, p - 5))
y <- rbinom(n, 1, plogis(X %*% beta_true))

# Build adjacency matrix (block structure)
R <- matrix(0L, p, p)
R[1:5, 1:5] <- 1L; diag(R) <- 0L
```

### Example 1: MH with Fixed Adjacency

```r
fit <- bvs_mh(X, y, adj_type = "fixed", adj_fixed = R, niter = 10000)
summary(fit)
plot(fit)
```

### Example 2: PG with Glasso Adjacency (EBIC)

```r
fit <- bvs_pg(X, y, adj_type = "glasso", glasso_criterion = "ebic",
              niter = 10000)
summary(fit)
```

### Example 3: PG with Sparse Bayesian GGM Adjacency

```r
fit <- bvs_pg(X, y, adj_type = "ggm", sparse = TRUE, niter = 10000)
s <- summary(fit)
s$selected   # selected variable indices
s$pip        # posterior inclusion probabilities
```

### Example 4: MH with Dense GGM + Fixed Adjacency (Dual Eta)

```r
fit <- bvs_mh(X, y, adj_type = "ggm_fixed",
              adj_fixed = R, sparse = FALSE, niter = 10000)
summary(fit)
```

### Phase Transition Detection

```r
pt <- phase_transition(R, max_eta = 1.5, num_rep = 10)
matplot(pt, type = "l", xlab = "eta index", ylab = "model size",
        main = "Phase Transition")
```

## C++ Backends (12 files in `src/`)

| File | Sampler | Adjacency | Eta | Dim |
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

## Authors

- **Yubing Yao** — yyao@umass.edu
- **Raji Balasubramanian** — rbalasub@schoolph.umass.edu
- **Mahlet Tadesse** — mgt26@georgetown.edu

## References

- Wang, H. (2012). Bayesian Graphical Lasso Models and Efficient Posterior Computation. *Bayesian Analysis*.
- Möller, J. et al. (2006). An efficient Markov chain Monte Carlo method for distributions with intractable normalising constants.
- Polson, N.G., Scott, J.G., & Windle, J. (2013). Bayesian inference for logistic models using Pólya–Gamma latent variables.

## License

GPL (>= 2)
