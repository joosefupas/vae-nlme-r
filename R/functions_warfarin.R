# =============================================================================
# functions_warfarin.R  —  Data loading and helpers for warfarin PK
# Dataset: nlmixr2data::warfarin
#
# PK model: 1-compartment oral (ka, ke, V)
#   C(t) = D * ka / (V * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))
# DV column used: dvid == "cp"  (concentration, mg/L)
# Covariates: wt (log-ratio), age (log-ratio), sex (0=male, 1=female)
# =============================================================================
library(torch)

# -----------------------------------------------------------------------------
# load_data_warfarin()
#
# Returns:
#   data         : [N, T_max, 2]   (obs_time, DV_cp)
#   data_in      : [N, T_max, 2]   standardised (time, DV)
#   lengths      : [N]             int, obs per individual
#   dose         : [N]             single oral dose per individual
#   covariates   : [N, 3]          (log(wt/wt_pop), log(age/age_pop), sex_binary)
#   covariates_in: [N, 3]          (same — all transforms done here, no further scaling)
# -----------------------------------------------------------------------------
load_data_warfarin <- function() {
  if (!requireNamespace("nlmixr2data", quietly = TRUE))
    stop("Package 'nlmixr2data' is required. Install with: install.packages('nlmixr2data')")
  df <- nlmixr2data::warfarin

  # Keep PK (cp) observations and dose rows only
  df_pk   <- df[df$dvid == "cp", ]
  df_obs  <- df_pk[df_pk$evid == 0, ]          # concentration observations
  df_dose <- df_pk[df_pk$evid == 1, ]          # dose events

  ids   <- sort(unique(df_obs$id))
  N     <- length(ids)
  T_max <- max(table(df_obs$id))

  data_arr <- array(0, dim = c(N, T_max, 2L))
  len_vec  <- integer(N)
  dose_vec <- numeric(N)

  for (i in seq_along(ids)) {
    pid  <- ids[i]
    obs  <- df_obs[df_obs$id == pid, ]
    obs  <- obs[order(obs$time), ]
    ni   <- nrow(obs)
    len_vec[i]   <- ni
    data_arr[i, 1:ni, 1] <- obs$time
    data_arr[i, 1:ni, 2] <- obs$dv

    dose_row <- df_dose[df_dose$id == pid, ]
    dose_vec[i] <- if (nrow(dose_row) > 0) dose_row$amt[1] else 0.0
  }

  data    <- torch_tensor(data_arr, dtype = torch_float())
  lengths <- torch_tensor(len_vec,  dtype = torch_int())
  dose    <- torch_tensor(dose_vec, dtype = torch_float())

  # --- covariates (one row per individual from any row of that id in df_pk) ---
  cov_raw <- df_pk[!duplicated(df_pk$id), c("id", "wt", "age", "sex")]
  cov_raw <- cov_raw[match(ids, cov_raw$id), ]
  wt_pop  <- mean(cov_raw$wt,  na.rm = TRUE)
  age_pop <- mean(cov_raw$age, na.rm = TRUE)
  sex_bin <- as.integer(cov_raw$sex != "male")   # 0=male, 1=female

  log_wt  <- log(cov_raw$wt  / wt_pop)
  log_age <- log(cov_raw$age / age_pop)

  cov_mat <- cbind(log_wt, log_age, sex_bin)
  covariates <- torch_tensor(cov_mat, dtype = torch_float())   # [N, 3]
  # All transformations already applied; covariates == covariates_in
  covariates_in <- covariates$clone()

  # --- standardise data_in ---
  data_in       <- data$clone()
  data_in[,, 1] <- data_in[,, 1] / data_in[,, 1]$max()
  dv_mean       <- data[,, 2]$mean()
  dv_std        <- data[,, 2]$std()
  data_in[,, 2] <- (data_in[,, 2] - dv_mean) / dv_std

  list(
    data          = data,
    data_in       = data_in,
    lengths       = lengths,
    dose          = dose,
    covariates    = covariates,
    covariates_in = covariates_in,
    wt_pop        = wt_pop,
    age_pop       = age_pop
  )
}

# -----------------------------------------------------------------------------
# initalize_C_warfarin()
# Covariates are already in their final transformed form in `covariates`.
# C matrix: same structure as neonates (all covariates applied uniformly).
# -----------------------------------------------------------------------------
initalize_C_warfarin <- function(nbatch, z_dim, n_cov, covariates) {
  C            <- torch_zeros(nbatch, z_dim, z_dim + z_dim * n_cov)
  C_regression <- torch_zeros(z_dim, nbatch, 1L + n_cov)

  for (i in seq_len(nbatch)) {
    C[i,, 1:z_dim] <- torch_eye(z_dim)
    count <- 0L
    for (k in seq_len(z_dim)) {
      for (j in seq_len(n_cov)) {
        C[i, k, j + z_dim + count] <- covariates[i, j]
      }
      count <- count + n_cov
    }
  }

  for (k in seq_len(z_dim)) {
    for (i in seq_len(nbatch)) {
      C_regression[k, i, 1] <- 1.0
      for (j in seq_len(n_cov)) {
        C_regression[k, i, j + 1L] <- covariates[i, j]
      }
    }
  }

  list(C = C, C_regression = C_regression)
}

# -----------------------------------------------------------------------------
# EmpiricalBayesEstimate_warfarin()
# Same 1-cmt oral formula as theophylline.
# data_i : [T, 2] R matrix  (time, DV)
# dose_i : scalar R numeric
# z_pop  : [z_dim] torch tensor
# omega_pop : [z_dim] torch tensor
# mu_i   : [z_dim] torch tensor
# res    : list(a_tensor, b_tensor)
# C_i    : [z_dim, z_dim+M] torch tensor
# h      : function (torch exp)
# -----------------------------------------------------------------------------
EmpiricalBayesEstimate_warfarin <- function(data_i, z_pop, omega_pop, mu_i,
                                             res, C_i, h, dose_i) {
  Cz     <- as.numeric(torch_matmul(C_i, z_pop)$detach()[1:3])
  data_m <- as.matrix(data_i$detach())
  om     <- as.numeric(omega_pop$detach())
  mu_v   <- as.numeric(mu_i$detach())
  a_v    <- as.numeric(res[[1]]$detach())
  b_v    <- as.numeric(res[[2]]$detach())
  dose_r <- as.numeric(dose_i$detach())

  ebe_loss <- function(phi) {
    phi_h <- exp(phi)           # (ka, ke, V)
    ka <- phi_h[1]; ke <- phi_h[2]; V <- phi_h[3]
    t_obs <- data_m[, 1]
    if (abs(ka - ke) < 1e-8) ke <- ke + 1e-7
    pred  <- dose_r * ka / (V * (ka - ke)) *
             (exp(-ke * t_obs) - exp(-ka * t_obs))
    pred  <- pmax(pred, 0)
    sigma <- a_v + b_v * pred
    eps   <- (data_m[, 2] - pred) / sigma
    sum(0.5 * eps^2 + log(sigma)) + 0.5 * sum((phi - Cz)^2 / om)
  }

  opt <- optim(mu_v, ebe_loss, method = "Nelder-Mead",
               control = list(reltol = 1e-6, maxit = 500))
  torch_tensor(opt$par, dtype = torch_float())
}
