# =============================================================================
# functions.R  —  Core VAE-NLME loss functions
# R translation of functions.py (Jan Rohleff, CPT:PSP 2025)
# =============================================================================
library(torch)

# -----------------------------------------------------------------------------
# KL divergence  KL( N(mean1, var1) || N(mean2, diag(var2)) )
# var1 : full covariance matrix  (z_dim × z_dim)
# var2 : diagonal variance vector (z_dim)
# log_diag : log of diagonal elements of the Cholesky factor L
# -----------------------------------------------------------------------------
kldiv_normal_normal <- function(mean1, var1, mean2, var2, log_diag) {
  k <- mean1$shape[1]
  0.5 * (torch_trace(var1 / var2) - k +
           torch_sum((mean2 - mean1)^2 / var2) +
           torch_log(var2)$sum() - 2 * log_diag$sum())
}

# -----------------------------------------------------------------------------
# p(x|z) — negative log-likelihood, combined additive+proportional error model
#   sigma_ij = a + b * f(z)_ij
# data   : [N, T, 2]  (col 1 = time, col 2 = observation)
# x_mean : [N, T, 1]  (model predictions)
# res    : list(a, b)
# lengths: [N] integer tensor
# -----------------------------------------------------------------------------
p_x_z_compute <- function(data, x_mean, res, lengths) {
  a <- res[[1]]
  b <- res[[2]]
  sigma <- a + b * x_mean

  nbatch <- data$shape[1]
  err_sq_sum    <- torch_tensor(0.0)
  log_sigma_sum <- torch_tensor(0.0)
  N_tot <- lengths$sum()

  for (i in seq_len(nbatch)) {
    ni <- lengths[i]$item()
    err_i     <- (data[i, 1:ni, 1] - x_mean[i, 1:ni, 1]) / sigma[i, 1:ni, 1]
    err_sq_sum    <- err_sq_sum    + torch_sum(err_i^2)
    log_sigma_sum <- log_sigma_sum + torch_log(sigma[i, 1:ni, 1])$sum()
  }

  ln_pi <- torch_log(torch_tensor(2 * pi))
  0.5 * err_sq_sum + 0.5 * N_tot * ln_pi + log_sigma_sum
}

# -----------------------------------------------------------------------------
# p(z) — negative log prior  (log-normal population distribution)
# z       : [N, z_dim]
# z_pop   : [z_dim]  population mean (in log space)
# omega_pop: [z_dim] diagonal variance
# -----------------------------------------------------------------------------
p_z_compute <- function(z, z_pop, omega_pop) {
  ln_pi <- torch_log(torch_tensor(2 * pi))
  0.5 * torch_sum((z - z_pop)^2 / omega_pop + torch_log(omega_pop) + ln_pi)
}

# -----------------------------------------------------------------------------
# q(z|x) — negative log variational posterior  (diagonal Gaussian encoder)
# eps   : [N, z_dim]  standard normal samples
# sigma : [N, z_dim]  posterior standard deviations
# -----------------------------------------------------------------------------
q_z_x_compute <- function(eps, sigma) {
  ln_pi <- torch_log(torch_tensor(2 * pi))
  0.5 * torch_sum(eps^2 + ln_pi + 2 * torch_log(sigma))
}

# --- Batch versions -----------------------------------------------------------

p_z_compute_batch <- function(z, z_pop, omega_pop) {
  ln_pi <- torch_log(torch_tensor(2 * pi))
  0.5 * torch_sum(((z - z_pop)^2 / omega_pop + torch_log(omega_pop) + ln_pi),
                  dim = 2)
}

q_z_x_compute_batch <- function(eps, sigma) {
  ln_pi <- torch_log(torch_tensor(2 * pi))
  0.5 * torch_sum(eps^2 + ln_pi + 2 * torch_log(sigma), dim = 2)
}

p_x_z_compute_lengths_batch <- function(data, x_mean, res, lengths) {
  a <- res[[1]]
  b <- res[[2]]
  sigma <- a + b * x_mean
  nbatch <- data$shape[1]
  p_x_z <- torch_zeros(nbatch)
  for (i in seq_len(nbatch)) {
    ni <- lengths[i]$item()
    err_i <- (data[i, 1:ni] - x_mean[i, 1:ni]) / sigma[i, 1:ni]
    p_x_z[i] <- 0.5 * torch_sum(err_i^2) +
      0.5 * ni * torch_log(torch_tensor(2 * pi)) +
      torch_log(sigma[i, 1:ni])$sum()
  }
  p_x_z
}

# -----------------------------------------------------------------------------
# Log-likelihood via linearisation around a given parameter point phi
# (Laplace approximation / first-order linearisation)
# phi : [N, z_dim]  linearisation point (e.g. posterior mean or EBE)
# -----------------------------------------------------------------------------
LogLikelihood_linearization <- function(z_pop_hat, omega_pop, res, data,
                                        phi, C, h, time, lengths, Decoder) {
  N     <- data$shape[1]
  z_dim <- omega_pop$shape[1]

  z_pop <- torch_zeros(N, z_dim)
  for (i in seq_len(N)) {
    z_pop[i] <- torch_matmul(C[i, , ], z_pop_hat)
  }

  phi_normal <- phi$clone()$detach()
  phi0 <- phi$clone()$detach()$requires_grad_(TRUE)
  pred_x <- Decoder(phi0, time, h)

  mu     <- torch_zeros(N, data$shape[2])
  df_dphi <- torch_zeros(N, data$shape[2], z_dim)
  ff     <- torch_zeros(N, data$shape[2])

  for (t in seq_len(data$shape[2])) {
    ff[, t] <- pred_x[, t, 1]
    ff[, t]$sum()$backward(retain_graph = TRUE)
    df_dphi[, t, ] <- phi0$grad
    mu[, t] <- ff[, t]
    for (zz in seq_len(z_dim)) {
      mu[, t] <- mu[, t] + phi0$grad[, zz] * (z_pop[, zz] - phi_normal[, zz])
    }
    phi0$grad$zero_()
  }

  LL <- torch_tensor(0.0)
  for (i in seq_len(N)) {
    ni <- lengths[i]$item()
    g   <- res[[1]] + res[[2]] * ff[i, 1:ni]
    Sig <- torch_eye(ni)
    tmp <- torch_matmul(torch_diag(g), torch_matmul(Sig, torch_diag(g)))
    dzf <- df_dphi[i, 1:ni, ]
    Gam <- torch_matmul(dzf, torch_matmul(torch_diag(omega_pop), dzf$t())) + tmp
    resid <- data[i, 1:ni, 2] - mu[i, 1:ni]
    LL <- LL + (ni / 2) * torch_log(torch_tensor(2 * pi)) +
      0.5 * linalg_slogdet(Gam)[[2]] +
      0.5 * torch_matmul(resid, linalg_solve(Gam, resid$unsqueeze(-1))$squeeze(-1))
  }
  LL
}

# -----------------------------------------------------------------------------
# Log-likelihood via importance sampling
# M : number of importance samples
# -----------------------------------------------------------------------------
LogLikelihood_sample <- function(M, z_pop_hat, omega_pop, res, data,
                                 mu, L, C, h, time, lengths, Decoder) {
  nbatch <- data$shape[1]
  z_dim  <- omega_pop$shape[1]

  z_pop <- torch_zeros(nbatch, z_dim)
  for (i in seq_len(nbatch)) {
    z_pop[i] <- torch_matmul(C[i, , ], z_pop_hat)
  }

  # ---- approximate true posterior with K = 100 weighted samples ----
  K <- 100L
  z_samples   <- torch_zeros(K, nbatch, z_dim)
  log_weights <- torch_zeros(K, nbatch)

  with_no_grad({
    for (k in seq_len(K)) {
      eps      <- torch_randn(nbatch, z_dim)
      z_k      <- mu + torch_matmul(L, eps$unsqueeze(-1))$squeeze(-1)
      z_samples[k, ,] <- z_k
      pred_x <- Decoder(z_k, time, h)
      p_xz   <- p_x_z_compute_lengths_batch(
        data[,, 2, drop = FALSE]$view(c(nbatch, data$shape[2], 1)),
        pred_x, res, lengths)
      pz   <- p_z_compute_batch(z_k, z_pop, omega_pop)
      qz   <- q_z_x_compute_batch(eps, torch_diagonal(L, dim1 = 2, dim2 = 3))
      log_weights[k, ] <- -(p_xz + pz - qz)
    }
    lw_norm   <- log_weights - torch_logsumexp(log_weights, dim = 1)$unsqueeze(1)
    w         <- torch_exp(lw_norm)
    mu_true   <- torch_sum(w$unsqueeze(-1) * z_samples, dim = 1)
    diff      <- z_samples - mu_true
    outer_    <- diff$unsqueeze(-1) * diff$unsqueeze(-2)
    cov_true  <- torch_sum(w$unsqueeze(-1)$unsqueeze(-1) * outer_, dim = 1)
    L_true    <- linalg_cholesky(cov_true)
  })

  # ---- importance sampling log-likelihood ----
  log_w <- torch_zeros(M, nbatch)
  for (m in seq_len(M)) {
    eps      <- torch_randn(nbatch, z_dim)
    z_m      <- mu_true + torch_matmul(L_true, eps$unsqueeze(-1))$squeeze(-1)
    pred_x <- Decoder(z_m, time, h)
    p_xz <- p_x_z_compute_lengths_batch(
      data[,, 2, drop = FALSE]$view(c(nbatch, data$shape[2], 1)),
      pred_x, res, lengths)
    pz   <- p_z_compute_batch(z_m, z_pop, omega_pop)
    qz   <- q_z_x_compute_batch(eps, torch_diagonal(L_true, dim1 = 2, dim2 = 3))
    log_w[m, ] <- -(p_xz + pz - qz)
    log_px <- torch_logsumexp(log_w[1:m, ], dim = 1) - log(m)
  }
  list(LL = -log_px$sum(), log_px_individual = -log_px)
}

# -----------------------------------------------------------------------------
# Burn-in: initialise encoder with frozen population parameters
# -----------------------------------------------------------------------------
initialize_encoder <- function(iters, L_iter, Encoder, Decoder,
                                data, data_in, z_dim, covariates_in,
                                lengths, h, pop) {
  cat("#############################################\n")
  cat("BURN IN phase\n")

  optimizer <- optim_adam(Encoder$parameters, lr = 8e-3)
  b <- torch_tensor(0.0)

  z_pop_iter_bi     <- torch_zeros(iters, z_dim)
  omega_pop_iter_bi <- torch_zeros(iters, z_dim)
  a_iter_bi         <- torch_zeros(iters)
  elbo_iter_bi      <- torch_zeros(iters)
  count <- 0L

  for (iter in seq_len(iters * L_iter)) {
    # Use $forward() directly instead of Encoder(...) to bypass R torch's
    # __call__ dispatch, which corrupts for larger nn_modules (h_dim=100, z_dim=5).
    enc  <- Encoder$forward(data_in, covariates_in, lengths)
    z_normal <- enc$z_normal; mu <- enc$mu
    L <- enc$L; log_sigma <- enc$log_sigma; eps <- enc$eps

    pred_x <- Decoder(z_normal, data[,, 1], h)
    upd    <- pop$update_pop(mu$detach(), L$detach(), pred_x$detach(), 0L)
    z_pop  <- upd$z_pop; omega_pop <- upd$omega_pop; a <- upd$a

    p_xz <- p_x_z_compute(data[,, 2, drop = FALSE]$view(c(data$shape[1], data$shape[2], 1)),
                           pred_x, list(a, b), lengths)
    # During burn-in, covariate betas are zero so C_i @ z_pop == z_pop[1:z_dim]
    pz   <- p_z_compute(z_normal, z_pop[1:z_dim], omega_pop)
    qz   <- q_z_x_compute(eps, torch_diagonal(L, dim1 = 2, dim2 = 3))
    DKL  <- pz - qz
    elbo <- p_xz + 0.001 * DKL

    elbo$backward()
    optimizer$step()
    optimizer$zero_grad()

    if (iter %% L_iter == 0L) {
      count <- count + 1L
      cat(sprintf("Iteration %d/%d\r", count, iters))
      z_pop_iter_bi[count, ]     <- h(z_pop[1:z_dim])$detach()
      omega_pop_iter_bi[count, ] <- omega_pop$sqrt()$detach()
      a_iter_bi[count]           <- a$detach()
      elbo_iter_bi[count]        <- (p_xz + DKL)$detach()
    }
  }

  list(Encoder = Encoder, optimizer = optimizer,
       pred_x = pred_x$detach(), mu = mu, L = L,
       a = a$detach(), b = b,
       z_pop_iter_bi = z_pop_iter_bi,
       omega_pop_iter_bi = omega_pop_iter_bi,
       a_iter_bi = a_iter_bi,
       elbo_iter_bi = elbo_iter_bi)
}

# =============================================================================
# save_vae_fit() / load_vae_fit()
# Save/load a complete VAE fit (population params + encoder + convergence)
# so that tables and plots can be regenerated without re-training.
#
# save_vae_fit() writes:
#   {save_dir}/{dataset_name}_fit.rds    — pure-R list (all tensors as arrays)
#   {save_dir}/{dataset_name}_encoder.pt — Encoder state dict (torch binary)
#
# load_vae_fit() returns a named list with torch tensors reconstructed.
# All datasets use h(x)=exp(x), h_inverse(x)=log(x).
# =============================================================================

save_vae_fit <- function(fit_list, save_dir, dataset_name) {
  if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)

  # Helper: safely convert a torch tensor to base-R; pass through if already R
  .to_r <- function(x) {
    if (is.null(x)) return(NULL)
    if (inherits(x, "torch_tensor")) return(as.array(x$detach()$cpu()))
    x   # already base R (numeric, matrix, array, etc.)
  }

  # Save encoder state dict separately
  enc_path <- file.path(save_dir, paste0(dataset_name, "_encoder.pt"))
  torch_save(fit_list$Encoder$state_dict(), enc_path)

  rds_obj <- list(
    dataset_name   = dataset_name,
    # Architecture scalars
    x_dim          = fit_list$x_dim,
    h_dim          = fit_list$h_dim,
    z_dim          = fit_list$z_dim,
    n_cov          = fit_list$n_cov,
    nbatch         = fit_list$nbatch,
    mu0_vec        = .to_r(fit_list$mu0),
    sigma0_vec     = .to_r(fit_list$sigma0),
    # Iteration settings
    iters          = fit_list$iters,
    kl_iter        = fit_list$kl_iter,
    gamma_iter     = fit_list$gamma_iter,
    iters_burn_in  = fit_list$iters_burn_in,
    # Population parameter estimates
    z_pop_vec      = as.numeric(.to_r(fit_list$z_pop)),
    omega_pop_vec  = as.numeric(.to_r(fit_list$omega_pop)),
    a_val          = as.numeric(.to_r(fit_list$a)),
    b_val          = as.numeric(.to_r(fit_list$b)),
    names_co       = fit_list$names_co,
    param_names    = fit_list$param_names,
    omega_names    = fit_list$omega_names,
    LL_lin_mu      = as.numeric(.to_r(fit_list$LL_lin_mu)),
    LL_is          = as.numeric(.to_r(fit_list$LL_is)),
    # Data arrays
    data_arr       = .to_r(fit_list$data),
    lengths_vec    = as.integer(as.numeric(.to_r(fit_list$lengths))),
    C_arr          = .to_r(fit_list$C),
    data_in_arr    = .to_r(fit_list$data_in),
    covariates_arr = .to_r(fit_list$covariates_in),
    # Convergence traces (combined burn-in + training)
    z_pop_iter_mat     = .to_r(fit_list$z_pop_iter),
    omega_pop_iter_mat = .to_r(fit_list$omega_pop_iter),
    Elbo_iter_vec      = as.numeric(.to_r(fit_list$Elbo_iter)),
    a_iter_vec         = as.numeric(.to_r(fit_list$a_iter)),
    # Dataset-specific (NULL if not applicable)
    dose       = .to_r(fit_list$dose),
    rate       = .to_r(fit_list$rate),
    t_inf      = .to_r(fit_list$t_inf),
    dose_times = .to_r(fit_list$dose_times),
    dose_amts  = .to_r(fit_list$dose_amts)
  )

  rds_path <- file.path(save_dir, paste0(dataset_name, "_fit.rds"))
  saveRDS(rds_obj, rds_path)
  cat(sprintf("  [save_vae_fit] RDS  -> %s\n", rds_path))
  cat(sprintf("  [save_vae_fit] Enc  -> %s\n", enc_path))
  invisible(rds_path)
}

load_vae_fit <- function(save_dir, dataset_name) {
  rds_path <- file.path(save_dir, paste0(dataset_name, "_fit.rds"))
  enc_path <- file.path(save_dir, paste0(dataset_name, "_encoder.pt"))
  if (!file.exists(rds_path)) stop("RDS not found: ", rds_path)
  if (!file.exists(enc_path)) stop("Encoder .pt not found: ", enc_path)

  obj <- readRDS(rds_path)

  # All datasets use exp/log link
  h         <- function(x) torch_exp(x)
  h_inverse <- function(x) torch_log(x)

  # Reconstruct encoder with saved architecture
  mu0    <- torch_tensor(obj$mu0_vec)
  sigma0 <- torch_tensor(obj$sigma0_vec)
  Encoder <- lstm_encoder(obj$x_dim, obj$h_dim, obj$z_dim, obj$n_cov,
                          mu0, sigma0, h_inverse)
  Encoder$load_state_dict(torch_load(enc_path))
  Encoder$eval()

  # Helper: reconstruct torch tensor (handle NULL)
  .to_t <- function(x, dtype = NULL) {
    if (is.null(x)) return(NULL)
    if (is.null(dtype)) torch_tensor(x) else torch_tensor(x, dtype = dtype)
  }

  list(
    dataset_name   = obj$dataset_name,
    x_dim          = obj$x_dim,
    h_dim          = obj$h_dim,
    z_dim          = obj$z_dim,
    n_cov          = obj$n_cov,
    nbatch         = obj$nbatch,
    iters          = obj$iters,
    kl_iter        = obj$kl_iter,
    gamma_iter     = obj$gamma_iter,
    iters_burn_in  = obj$iters_burn_in,
    names_co       = obj$names_co,
    param_names    = obj$param_names,
    omega_names    = obj$omega_names,
    LL_lin_mu      = .to_t(obj$LL_lin_mu),
    LL_is          = .to_t(obj$LL_is),
    h              = h,
    h_inverse      = h_inverse,
    z_pop          = .to_t(obj$z_pop_vec),
    omega_pop      = .to_t(obj$omega_pop_vec),
    a              = .to_t(obj$a_val),
    b              = .to_t(obj$b_val),
    data           = .to_t(obj$data_arr),
    lengths        = .to_t(obj$lengths_vec, dtype = torch_int64()),
    C              = .to_t(obj$C_arr),
    data_in        = .to_t(obj$data_in_arr),
    covariates_in  = .to_t(obj$covariates_arr),
    Encoder        = Encoder,
    z_pop_iter     = .to_t(obj$z_pop_iter_mat),
    omega_pop_iter = .to_t(obj$omega_pop_iter_mat),
    Elbo_iter      = .to_t(obj$Elbo_iter_vec),
    a_iter         = .to_t(obj$a_iter_vec),
    dose           = .to_t(obj$dose),
    rate           = .to_t(obj$rate),
    t_inf          = .to_t(obj$t_inf),
    dose_times     = .to_t(obj$dose_times),
    dose_amts      = .to_t(obj$dose_amts)
  )
}
