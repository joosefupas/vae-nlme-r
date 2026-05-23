# =============================================================================
# pop_parameter.R  —  Population parameter update (BICc-based)
# R translation of pop_parameter.py (Jan Rohleff, CPT:PSP 2025)
#
# Implements the M-step of the VAE-NLME algorithm:
#   1.  BICc-ELBO minimisation via per-parameter subset enumeration with
#       alpha_pen annealing — faithful to original Gurobi-based MIQP.
#   2.  Closed-form update of diagonal random-effects variance omega_pop
#       using exponentially smoothed sufficient statistics (s1, s2, s3).
#   3.  Smoothed update of additive error parameter a via s4.
# =============================================================================
library(torch)
library(R6)

pop_parameter <- R6Class(
  "pop_parameter",

  public = list(

    # ----- current estimates --------------------------------------------------
    z_pop     = NULL,   # [z_dim] (burn-in) or [z_dim + M] (training)
    omega_pop = NULL,   # [z_dim]  diagonal variance
    a         = NULL,   # scalar   additive error
    b         = NULL,   # scalar   proportional error (= 0)
    mu_smooth = NULL,   # [N, z_dim]  running posterior mean (= s1)

    # ----- smoothed sufficient statistics (EMA) ------------------------------
    s1    = NULL,   # [N, z_dim]       EMA of mu
    s2    = NULL,   # [z_dim, z_dim]   EMA of sum_i mu_i mu_i^T
    s3    = NULL,   # [z_dim, z_dim]   EMA of sum_i L_i L_i^T
    s4    = NULL,   # scalar (R numeric) EMA of residual sum-of-squares
    gamma = 1.0,    # smoothing rate (1 until smoothing phase)

    # ----- configuration ------------------------------------------------------
    z_dim           = NULL,
    nbatch          = NULL,
    n_cov           = NULL,
    M               = NULL,   # = z_dim * n_cov
    gamma_iter      = NULL,
    kl_iter         = NULL,
    penalized_indices = NULL,
    lengths         = NULL,
    C               = NULL,   # [N, z_dim, z_dim + M]
    C_regression    = NULL,   # [z_dim, N, 1 + n_cov]
    C_diag          = NULL,   # [N, z_dim, z_dim]   (= identity block for burn-in)
    data            = NULL,
    alpha_pen       = NULL,   # [kl_iter] torch vector, linspace(alpha, 1, kl_iter)

    initialize = function(z_dim, nbatch, gamma_iter, data, C, C_regression,
                          C_diag, penalized_indices, n_cov, kl_iter, lengths,
                          alpha = 2.0) {
      self$z_dim             <- z_dim
      self$nbatch            <- nbatch
      self$n_cov             <- n_cov
      self$M                 <- z_dim * n_cov
      self$gamma_iter        <- gamma_iter
      self$kl_iter           <- kl_iter
      self$penalized_indices <- penalized_indices
      self$lengths           <- lengths
      self$C                 <- C
      self$C_regression      <- C_regression
      self$C_diag            <- C_diag
      self$data              <- data

      # Penalty annealing schedule: alpha -> 1 over kl_iter iterations
      self$alpha_pen <- torch_linspace(alpha, 1.0, kl_iter)

      # Initialise population parameters
      self$z_pop     <- torch_zeros(z_dim)        # grows to [z_dim+M] after 1st cov_sel call
      self$omega_pop <- torch_ones(z_dim)
      self$a         <- torch_tensor(1.0)
      self$b         <- torch_tensor(0.0)

      # Initialise sufficient statistics
      self$gamma <- 1.0          # Python: self.gamma = 1  (full replacement during training)
      self$s1 <- torch_zeros(nbatch, z_dim)
      self$s2 <- torch_zeros(z_dim, z_dim)
      self$s3 <- torch_zeros(z_dim, z_dim)
      self$s4 <- 0.0
    },

    # -------------------------------------------------------------------------
    # Main M-step update — faithful translation of Python update_pop()
    #
    # mu       : [N, z_dim]        posterior mean   (detached)
    # L        : [N, z_dim, z_dim] Cholesky factor  (detached)
    # pred_x   : [N, T, 1]         model predictions (detached)
    # iter     : integer            current iteration (0 = burn-in)
    # covariate_selection : logical
    # smoothing: logical            if TRUE use slow EMA (gamma << 1)
    # update_pop: logical
    # -------------------------------------------------------------------------
    update_pop = function(mu, L, pred_x, iter,
                          covariate_selection = FALSE,
                          smoothing           = FALSE,
                          update_pop          = TRUE) {

      # ---- 1. Update smoothing rate ----------------------------------------
      if (smoothing) self$gamma <- 1.0 / (iter - self$gamma_iter)

      # ---- 2. Update sufficient statistics (EMA) ---------------------------
      self$s1 <- self$s1 + self$gamma * (mu - self$s1)

      # Python uses raw mu (not the smoothed s1) to compute sum2 — faithful to
      # the EM sufficient-statistic:  s2 ≈ EMA( sum_i mu_i mu_i^T )
      # Using self$s1 here would create a "double-smoothed" statistic that
      # diverges from Python as soon as gamma < 1, causing omega_pop to shrink
      # artificially at the start of the smoothing phase (k_beta).
      sum2 <- torch_zeros(self$z_dim, self$z_dim)
      for (i in seq_len(self$nbatch)) {
        mi   <- mu[i, ]$view(c(self$z_dim, 1L))   # raw encoder mu, not s1
        sum2 <- sum2 + torch_matmul(mi, mi$t())
      }
      self$s2 <- self$s2 + self$gamma * (sum2 - self$s2)

      sum3 <- torch_matmul(L, L$transpose(2L, 3L))$sum(dim = 1L)  # [z_dim, z_dim]
      self$s3 <- self$s3 + self$gamma * (sum3 - self$s3)

      # ---- 3. Update s4 (smoothed RSS) and error parameter a ---------------
      len_vec  <- as.integer(as.numeric(self$lengths$detach()))
      new_s4   <- 0.0
      for (i in seq_len(self$nbatch)) {
        ni     <- len_vec[i]
        obs_i  <- as.numeric(self$data[i, 1:ni, 2]$detach())
        pred_i <- as.numeric(pred_x[i, 1:ni, 1]$detach())
        new_s4 <- new_s4 + sum((obs_i - pred_i)^2)
      }
      self$s4 <- self$s4 + self$gamma * (new_s4 - self$s4)
      a <- torch_clamp(torch_sqrt(torch_tensor(self$s4 / sum(len_vec))), min = 1e-6)

      # ---- 4. Update z_pop and omega_pop -----------------------------------
      if (covariate_selection) {
        # Per-parameter MIQP with BICc penalty annealing
        z_pop <- private$solve_miqp_per_param(iter)

        # omega update using s2, s3 and smoothed s1
        omega_mat <- torch_zeros(self$z_dim, self$z_dim)
        for (i in seq_len(self$nbatch)) {
          Cz      <- torch_matmul(self$C[i, , ], z_pop)$view(c(self$z_dim, 1L))
          s1i     <- self$s1[i, ]$view(c(self$z_dim, 1L))
          omega_mat <- omega_mat -
            torch_matmul(Cz, s1i$t()) - torch_matmul(s1i, Cz$t()) +
            torch_matmul(Cz, Cz$t())
        }
        omega_diag <- ((1.0 / self$nbatch) * (self$s2 + omega_mat + self$s3))$diagonal()
        omega_pop  <- torch_clamp(omega_diag$detach(), min = 0.01)

      } else {
        # Burn-in / no covariate selection: solve intercepts-only via C_diag
        A   <- torch_zeros(self$z_dim, self$z_dim)
        rhs <- torch_zeros(self$z_dim)
        inv_om <- torch_diag(1.0 / self$omega_pop)
        for (i in seq_len(self$nbatch)) {
          Ci  <- self$C_diag[i, , ]    # [z_dim, z_dim] identity
          A   <- A   + torch_matmul(Ci$t(), torch_matmul(inv_om, Ci))
          rhs <- rhs + torch_matmul(Ci$t(), torch_matmul(inv_om, self$s1[i, ]))
        }
        z_pop <- linalg_solve(A, rhs$unsqueeze(-1L))$squeeze(-1L)

        omega_mat <- torch_zeros(self$z_dim, self$z_dim)
        for (i in seq_len(self$nbatch)) {
          Cz  <- torch_matmul(self$C_diag[i, , ], z_pop)$view(c(self$z_dim, 1L))
          s1i <- self$s1[i, ]$view(c(self$z_dim, 1L))
          omega_mat <- omega_mat -
            torch_matmul(Cz, s1i$t()) - torch_matmul(s1i, Cz$t()) +
            torch_matmul(Cz, Cz$t())
        }
        omega_diag <- ((1.0 / self$nbatch) * (self$s2 + omega_mat + self$s3))$diagonal()
        omega_pop  <- torch_clamp(omega_diag$detach(), min = 0.01)
      }

      # ---- 5. Store updated state ------------------------------------------
      if (update_pop) {
        self$z_pop     <- z_pop$detach()
        self$omega_pop <- omega_pop
        self$a         <- a$detach()
      }
      self$mu_smooth <- self$s1$clone()$detach()

      list(
        z_pop     = z_pop$detach(),
        omega_pop = omega_pop,
        a         = a$detach(),
        mu_smooth = self$mu_smooth
      )
    }
  ),

  private = list(

    # -------------------------------------------------------------------------
    # Per-parameter MIQP with BICc penalty annealing.
    # Faithful translation of the Python per-k loop using C_regression[k].
    #
    # For each PK parameter k:
    #   X_k = C_regression[k,:,:] / sqrt(omega_k)   [N, 1+n_cov]
    #   y_k = s1[:,k]             / sqrt(omega_k)   [N]
    #   minimize ||y_k - X_k @ beta_k||^2 + alpha_pen * log(N) * |S_k|
    #   s.t.  beta_k[covariates] = 0  if covariate not in S_k
    #
    # With n_cov=2 there are only 4 subsets per parameter (exact enumeration).
    # -------------------------------------------------------------------------
    solve_miqp_per_param = function(iter) {
      n_theta   <- self$z_dim + self$M
      z_pop_vec <- numeric(n_theta)

      alpha_val <- if (iter < self$kl_iter) {
        as.numeric(self$alpha_pen[iter + 1L]$detach())
      } else {
        1.0
      }
      ln_N   <- log(self$nbatch)
      omega  <- as.numeric(self$omega_pop$detach())
      s1_mat <- as.matrix(self$s1$detach())          # [N, z_dim]
      C_reg  <- as.array(self$C_regression$detach()) # [z_dim, N, 1+n_cov]

      for (k in seq_len(self$z_dim)) {
        sqrt_om_k <- sqrt(omega[k])
        X <- C_reg[k, , ] / sqrt_om_k   # [N, 1+n_cov]
        y <- s1_mat[, k]  / sqrt_om_k   # [N]

        best_obj  <- Inf
        best_beta <- numeric(1L + self$n_cov)

        # Enumerate all 2^n_cov subsets of covariates
        for (s_int in 0L:(2L^self$n_cov - 1L)) {
          s_bits <- as.logical(intToBits(s_int)[seq_len(self$n_cov)])
          # Column indices: intercept (1) always active, plus selected covariates
          active_cols <- c(1L, 1L + which(s_bits))

          X_sub    <- X[, active_cols, drop = FALSE]
          XtX      <- crossprod(X_sub)
          Xty      <- as.numeric(t(X_sub) %*% y)
          beta_sub <- tryCatch(
            as.numeric(base::solve(XtX, Xty)),
            error = function(e) NULL
          )
          if (is.null(beta_sub)) next

          beta_full <- numeric(1L + self$n_cov)
          beta_full[active_cols] <- beta_sub

          resid <- y - X %*% beta_full
          obj   <- sum(resid^2) + alpha_val * ln_N * sum(s_bits)

          if (obj < best_obj) {
            best_obj  <- obj
            best_beta <- beta_full
          }
        }

        # Store intercept and covariate effects for param k
        z_pop_vec[k]                                           <- best_beta[1L]
        z_pop_vec[self$z_dim + (k - 1L) * self$n_cov + seq_len(self$n_cov)] <-
          best_beta[2L:(1L + self$n_cov)]
      }

      torch_tensor(z_pop_vec, dtype = torch_float())
    }
  )
)
