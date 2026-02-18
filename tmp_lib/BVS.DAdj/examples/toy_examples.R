# ========================================================================
# BayesVarSel: Toy Examples
# ========================================================================
# These examples use small n, p, niter to run quickly for demonstration.
# For real analysis, increase niter (e.g. 60,000+) and use appropriate
# adjacency structures.
# ========================================================================

library(BayesVarSel)

## ---- Setup: simulate logistic regression data ----
set.seed(42)
n <- 200; p <- 50
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(rep(1, 5), rep(0.5, 5), rep(0, p - 10))
prob <- plogis(X %*% beta_true)
y <- rbinom(n, 1, prob)

# Block adjacency: first 10 variables form a clique
R_block <- matrix(0L, p, p)
R_block[1:10, 1:10] <- 1L
diag(R_block) <- 0L


## ---- Example 1: MH with single fixed adjacency ----
cat("\n=== Example 1: bvs_mh, fixed adjacency ===\n")
fit1 <- bvs_mh(X, y, adj_type = "fixed", adj_fixed = R_block,
                niter = 2000, burnin = 500)
s1 <- summary(fit1)
cat("Selected variables:", paste(s1$selected, collapse = ", "), "\n")
cat("PIPs (first 15):", round(s1$pip[1:15], 2), "\n")
plot(fit1, main = "MH + Fixed Adj")


## ---- Example 2: PG with single fixed adjacency ----
cat("\n=== Example 2: bvs_pg, fixed adjacency ===\n")
fit2 <- bvs_pg(X, y, adj_type = "fixed", adj_fixed = R_block,
                niter = 2000, burnin = 500)
s2 <- summary(fit2)
cat("Selected variables:", paste(s2$selected, collapse = ", "), "\n")


## ---- Example 3: PG with graphical lasso adjacency (EBIC) ----
cat("\n=== Example 3: bvs_pg, glasso (EBIC) ===\n")
if (requireNamespace("huge", quietly = TRUE)) {
  fit3 <- bvs_pg(X, y, adj_type = "glasso", glasso_criterion = "ebic",
                  niter = 2000, burnin = 500)
  s3 <- summary(fit3)
  cat("Selected variables:", paste(s3$selected, collapse = ", "), "\n")
} else {
  cat("Skipping: 'huge' package not installed.\n")
}


## ---- Example 4: MH with sparse Bayesian GGM adjacency ----
cat("\n=== Example 4: bvs_mh, sparse GGM ===\n")
fit4 <- bvs_mh(X, y, adj_type = "ggm", sparse = TRUE,
                niter = 2000, burnin = 500)
s4 <- summary(fit4)
cat("Selected variables:", paste(s4$selected, collapse = ", "), "\n")


## ---- Example 5: PG with dual adjacency (GGM + fixed) ----
cat("\n=== Example 5: bvs_pg, GGM + fixed, sparse ===\n")
fit5 <- bvs_pg(X, y, adj_type = "ggm_fixed",
                adj_fixed = R_block, sparse = TRUE,
                niter = 2000, burnin = 500)
s5 <- summary(fit5)
cat("Selected variables:", paste(s5$selected, collapse = ", "), "\n")


## ---- Example 6: Phase transition detection ----
cat("\n=== Example 6: Phase transition detection ===\n")
pt <- phase_transition(R_block, max_eta = 1.5, num_rep = 5)
matplot(pt, type = "l", lty = 1, col = "steelblue",
        xlab = "eta index", ylab = "model size",
        main = "Phase Transition (Single Eta)")
