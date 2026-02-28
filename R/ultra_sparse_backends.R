# Ultra-sparse wrappers for high-dimensional sparse-GGM backends.

.as_dgC <- function(M) {
  if (inherits(M, "dgCMatrix")) {
    return(M)
  }
  if (!inherits(M, "Matrix")) {
    M <- Matrix::Matrix(M, sparse = TRUE)
  }
  # Avoid deprecated direct coercions (e.g., dsCMatrix -> dgCMatrix) by:
  # 1) forcing a general sparse form; 2) rebuilding a numeric dgC matrix.
  M <- methods::as(M, "generalMatrix")
  M <- methods::as(M, "CsparseMatrix")
  if (!inherits(M, "dgCMatrix")) {
    ii <- M@i + 1L
    xx <- if ("x" %in% slotNames(M)) as.numeric(M@x) else rep(1, length(ii))
    M <- Matrix::sparseMatrix(
      i = ii,
      p = M@p,
      x = xx,
      dims = dim(M),
      index1 = TRUE,
      repr = "C"
    )
  }
  M
}

.init_ultra_sparse_state <- function(y, p, beta_init, gamma_init, alpha_init) {
  if (is.null(beta_init)) {
    beta_init <- numeric(p)
  }
  if (is.null(gamma_init)) {
    gamma_init <- integer(p)
  }
  if (is.null(alpha_init)) {
    y01 <- as.numeric(y)
    if (all(y01 %in% c(-1, 1))) {
      y01 <- 0.5 * (y01 + 1)
    }
    ybar <- mean(y01, na.rm = TRUE)
    ybar <- min(max(ybar, 1e-4), 1 - 1e-4)
    alpha_init <- qlogis(ybar)
  }
  list(beta_init = beta_init, gamma_init = gamma_init, alpha_init = alpha_init)
}

.sparse_diag_drop <- function(M) {
  d <- Matrix::diag(M)
  if (length(d) == 0 || all(d == 0)) {
    return(M)
  }
  M - Matrix::Diagonal(n = nrow(M), x = d)
}

.prepare_sparse_S_triplet <- function(X, S_ggm = NULL, p_max_crossprod = 1e5) {
  if (is.null(S_ggm)) {
    p <- ncol(X)
    if (p >= p_max_crossprod) {
      stop("For large p, provide sparse S_ggm explicitly; crossprod(X) is disabled.")
    }
    S_ggm <- Matrix::crossprod(X)
  }

  S_ggm <- .as_dgC(S_ggm)
  p <- ncol(S_ggm)
  if (nrow(S_ggm) != p) {
    stop("S_ggm must be square.")
  }

  S_diag <- as.numeric(Matrix::diag(S_ggm))

  # Fast symmetric diagonal drop natively leveraging dgCMatrix internals
  S_off <- S_ggm
  Matrix::diag(S_off) <- 0
  S_off <- Matrix::drop0(S_off)

  list(
    S_i = as.integer(S_off@i),
    S_p = as.integer(S_off@p),
    S_x = as.numeric(S_off@x),
    S_diag = as.numeric(S_diag),
    p = as.integer(p)
  )
}

.prepare_sparse_adj_triplet <- function(adj_fixed) {
  A <- .as_dgC(adj_fixed)
  if (nrow(A) != ncol(A)) {
    stop("adj_fixed must be square.")
  }

  A <- A + Matrix::t(A)
  A@x[] <- 1
  A <- Matrix::drop0(A)
  A <- .sparse_diag_drop(A)
  A <- Matrix::drop0(A)

  list(
    R_i = as.integer(A@i),
    R_p = as.integer(A@p),
    p = as.integer(ncol(A))
  )
}

.reconstruct_sparse_pip <- function(result, p, store_Z_pip) {
  if (!isTRUE(store_Z_pip)) {
    result$Z_pip <- NULL
    result$R_ggm_median <- NULL
    result$Z_pip_row <- NULL
    result$Z_pip_col <- NULL
    result$Z_pip_val <- NULL
    return(result)
  }

  pr <- result$Z_pip_row
  pc <- result$Z_pip_col
  pv <- result$Z_pip_val

  if (length(pr) > 0L) {
    Z_pip <- Matrix::sparseMatrix(
      i = c(pr + 1L, pc + 1L),
      j = c(pc + 1L, pr + 1L),
      x = c(pv, pv),
      dims = c(p, p),
      repr = "C"
    )
  } else {
    Z_pip <- Matrix::sparseMatrix(
      i = integer(0), j = integer(0), x = numeric(0),
      dims = c(p, p), repr = "C"
    )
  }

  result$Z_pip <- Z_pip
  result$R_ggm_median <- (Z_pip > 0.5) * 1
  result$Z_pip_row <- NULL
  result$Z_pip_col <- NULL
  result$Z_pip_val <- NULL
  result
}

.reconstruct_gamma_pip <- function(result, p, n_save, store_gamma) {
  if (!isTRUE(store_gamma) || !is.list(result$gamma)) {
    result$gamma_pip <- NULL
    return(result)
  }

  pip_counts <- numeric(p)
  for (s in seq_along(result$gamma)) {
    idx1 <- .normalize_sparse_idx(result$gamma[[s]], p)
    if (length(idx1) > 0L) {
      pip_counts[idx1] <- pip_counts[idx1] + 1
    }
  }
  result$gamma_pip <- pip_counts / n_save
  result
}
