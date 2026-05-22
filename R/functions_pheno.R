# =============================================================================
# functions_pheno.R  —  Data loading and helpers for phenobarbital PK
# Dataset: nlmixr2data::pheno_sd
#
# PK model: 1-compartment IV bolus with multiple doses (superposition)
#   C(t) = (1/V) * sum_{k : t_k <= t} AMT_k * exp(-CL/V * (t - t_k))
# Parameters: (CL, V) in log-space
# Covariates: WT (log-ratio), APGR (log-ratio)
# =============================================================================
library(torch)

# -----------------------------------------------------------------------------
# load_data_pheno()
#
# Returns:
#   data       : [N, T_max, 2]     (obs_time, DV)
#   data_in    : [N, T_max, 2]     standardised (time, DV)
#   lengths    : [N]               int, number of obs per individual
#   dose_times : [N, D_max]        dose event times (padded with large value)
#   dose_amts  : [N, D_max]        dose amounts     (padded with 0)
#   covariates : [N, 2]            (log(WT/WT_pop), log(APGR/APGR_pop))
#   covariates_in : [N, 2]         same
# -----------------------------------------------------------------------------
load_data_pheno <- function() {
  if (!requireNamespace("nlmixr2data", quietly = TRUE))
    stop("Package 'nlmixr2data' is required.")
  df <- nlmixr2data::pheno_sd

  ids   <- sort(unique(df$ID))
  N     <- length(ids)

  # Separate dose and observation rows
  df_obs  <- df[df$EVID == 0, ]
  df_dose <- df[df$EVID == 1, ]

  T_max <- max(table(df_obs$ID))
  D_max <- max(table(df_dose$ID))

  data_arr  <- array(0, dim = c(N, T_max, 2L))
  len_vec   <- integer(N)
  dt_arr    <- array(1e9, dim = c(N, D_max))   # pad with large time
  da_arr    <- array(0,   dim = c(N, D_max))

  cov_arr <- matrix(0, N, 2)    # (WT, APGR) — raw, will log-ratio transform below

  for (i in seq_along(ids)) {
    pid  <- ids[i]
    obs  <- df_obs[df_obs$ID == pid, ]
    obs  <- obs[order(obs$TIME), ]
    dse  <- df_dose[df_dose$ID == pid, ]
    dse  <- dse[order(dse$TIME), ]

    ni <- nrow(obs); nd <- nrow(dse)
    len_vec[i] <- ni

    data_arr[i, 1:ni, 1] <- obs$TIME
    data_arr[i, 1:ni, 2] <- obs$DV

    if (nd > 0) {
      dt_arr[i, 1:nd] <- dse$TIME
      da_arr[i, 1:nd] <- dse$AMT
    }
    # Covariates from first row of this individual
    row1 <- df[df$ID == pid, ][1, ]
    cov_arr[i, 1] <- row1$WT
    cov_arr[i, 2] <- row1$APGR
  }

  WT_pop   <- mean(cov_arr[, 1], na.rm = TRUE)
  APGR_pop <- mean(cov_arr[, 2], na.rm = TRUE)

  cov_trans        <- cov_arr
  cov_trans[, 1]   <- log(cov_arr[, 1] / WT_pop)
  cov_trans[, 2]   <- log(cov_arr[, 2] / APGR_pop)

  data      <- torch_tensor(data_arr, dtype = torch_float())
  lengths   <- torch_tensor(len_vec,  dtype = torch_int())
  dose_times <- torch_tensor(dt_arr,  dtype = torch_float())
  dose_amts  <- torch_tensor(da_arr,  dtype = torch_float())
  covariates <- torch_tensor(cov_trans, dtype = torch_float())    # [N, 2]
  covariates_in <- covariates$clone()

  # Standardise data_in
  data_in       <- data$clone()
  data_in[,, 1] <- data_in[,, 1] / data_in[,, 1]$max()
  dv_mean       <- data[,, 2]$mean()
  dv_std        <- data[,, 2]$std()
  data_in[,, 2] <- (data_in[,, 2] - dv_mean) / dv_std

  list(
    data          = data,
    data_in       = data_in,
    lengths       = lengths,
    dose_times    = dose_times,
    dose_amts     = dose_amts,
    covariates    = covariates,
    covariates_in = covariates_in,
    WT_pop        = WT_pop,
    APGR_pop      = APGR_pop
  )
}

# -----------------------------------------------------------------------------
# initalize_C_pheno()  —  same structure as neonates (covariates pre-transformed)
# -----------------------------------------------------------------------------
initalize_C_pheno <- function(nbatch, z_dim, n_cov, covariates) {
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
# EmpiricalBayesEstimate_pheno()
# 1-cmt IV multi-dose superposition in pure R.
# data_i      : [T, 2] R matrix (obs_time, DV)
# dose_times_i: R numeric vector of dose times for individual i
# dose_amts_i : R numeric vector of dose amounts
# -----------------------------------------------------------------------------
EmpiricalBayesEstimate_pheno <- function(data_i, z_pop, omega_pop, mu_i,
                                          res, C_i, h, dose_times_i, dose_amts_i) {
  Cz   <- as.numeric(torch_matmul(C_i, z_pop)$detach()[1:2])
  data_m <- as.matrix(data_i$detach())
  om   <- as.numeric(omega_pop$detach())
  mu_v <- as.numeric(mu_i$detach())
  a_v  <- as.numeric(res[[1]]$detach())
  b_v  <- as.numeric(res[[2]]$detach())

  # Remove padded doses (large time values)
  valid <- dose_times_i < 1e8
  dt_v  <- dose_times_i[valid]
  da_v  <- dose_amts_i[valid]

  ebe_loss <- function(phi) {
    phi_h <- exp(phi)       # (CL, V)
    CL <- phi_h[1]; V <- phi_h[2]; ke <- CL / V
    t_obs <- data_m[, 1]
    pred <- sapply(t_obs, function(t) {
      k_before <- which(dt_v <= t)
      if (length(k_before) == 0) return(0)
      sum(da_v[k_before] / V * exp(-ke * (t - dt_v[k_before])))
    })
    pred  <- pmax(pred, 0)
    sigma <- a_v + b_v * pred
    eps   <- (data_m[, 2] - pred) / sigma
    sum(0.5 * eps^2 + log(sigma)) + 0.5 * sum((phi - Cz)^2 / om)
  }

  opt <- optim(mu_v, ebe_loss, method = "Nelder-Mead",
               control = list(reltol = 1e-6, maxit = 500))
  torch_tensor(opt$par, dtype = torch_float())
}
