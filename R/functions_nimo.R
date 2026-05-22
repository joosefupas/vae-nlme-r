# =============================================================================
# functions_nimo.R  —  Data loading and helpers for nimoData PK
# Dataset: nlmixr2data::nimoData
#
# PK model: 1-compartment IV infusion (analytical solution)
#   C(t) = (R/CL)*(1-exp(-CL/V*t))               for t <= T_inf
#   C(t) = (R/CL)*(1-exp(-CL/V*T_inf))*exp(-CL/V*(t-T_inf))  for t > T_inf
# Parameters: (CL, V) in log-space
# Each (ID, OCC) pair treated as an independent profile.
# Covariates: WGT (log-ratio), AGE (log-ratio), HGT (log-ratio)
# =============================================================================
library(torch)

# -----------------------------------------------------------------------------
# load_data_nimo()
#
# Returns:
#   data         : [N_eff, T_max, 2]  (TAD, DV)
#   data_in      : [N_eff, T_max, 2]  standardised
#   lengths      : [N_eff]            int, obs per profile
#   rate         : [N_eff]            infusion rate  (AMT units / h)
#   t_inf        : [N_eff]            infusion duration = AMT/RATE  (h)
#   covariates   : [N_eff, 3]         (log_WGT, log_AGE, log_HGT)
#   covariates_in: [N_eff, 3]         same
# -----------------------------------------------------------------------------
load_data_nimo <- function() {
  if (!requireNamespace("nlmixr2data", quietly = TRUE))
    stop("Package 'nlmixr2data' is required.")
  df <- nlmixr2data::nimoData

  df_dose <- df[df$EVID == 1, ]
  df_obs  <- df[df$EVID == 0, ]

  prof_keys <- unique(df_dose[, c("ID", "OCC")])
  prof_keys <- prof_keys[order(prof_keys$ID, prof_keys$OCC), ]
  N_eff <- nrow(prof_keys)

  T_max <- max(table(paste(df_obs$ID, df_obs$OCC)))

  data_arr <- array(0, dim = c(N_eff, T_max, 2L))
  len_vec  <- integer(N_eff)
  rate_vec <- numeric(N_eff)
  tinf_vec <- numeric(N_eff)
  cov_arr  <- matrix(0, N_eff, 3L)   # (WGT, AGE, HGT)

  for (k in seq_len(N_eff)) {
    pid <- prof_keys$ID[k]
    occ <- prof_keys$OCC[k]

    dose_row <- df_dose[df_dose$ID == pid & df_dose$OCC == occ, ]
    obs_k    <- df_obs[df_obs$ID   == pid & df_obs$OCC  == occ, ]
    obs_k    <- obs_k[order(obs_k$TAD), ]

    ni <- nrow(obs_k)
    len_vec[k]  <- ni

    data_arr[k, 1:ni, 1] <- obs_k$TAD
    data_arr[k, 1:ni, 2] <- obs_k$DV

    RATE_k      <- dose_row$RATE[1]
    AMT_k       <- dose_row$AMT[1]
    rate_vec[k] <- RATE_k
    tinf_vec[k] <- AMT_k / RATE_k

    cov_arr[k, 1] <- dose_row$WGT[1]
    cov_arr[k, 2] <- dose_row$AGE[1]
    cov_arr[k, 3] <- dose_row$HGT[1]
  }

  WGT_pop <- mean(cov_arr[, 1])
  AGE_pop <- mean(cov_arr[, 2])
  HGT_pop <- mean(cov_arr[, 3])

  cov_trans      <- cov_arr
  cov_trans[, 1] <- log(cov_arr[, 1] / WGT_pop)
  cov_trans[, 2] <- log(cov_arr[, 2] / AGE_pop)
  cov_trans[, 3] <- log(cov_arr[, 3] / HGT_pop)

  data       <- torch_tensor(data_arr,  dtype = torch_float())
  lengths    <- torch_tensor(len_vec,   dtype = torch_int())
  rate       <- torch_tensor(rate_vec,  dtype = torch_float())
  t_inf      <- torch_tensor(tinf_vec,  dtype = torch_float())
  covariates <- torch_tensor(cov_trans, dtype = torch_float())
  covariates_in <- covariates$clone()

  data_in       <- data$clone()
  data_in[,, 1] <- data_in[,, 1] / data_in[,, 1]$max()
  dv_mean       <- data[,, 2]$mean()
  dv_std        <- data[,, 2]$std()
  data_in[,, 2] <- (data_in[,, 2] - dv_mean) / dv_std

  list(
    data          = data,
    data_in       = data_in,
    lengths       = lengths,
    rate          = rate,
    t_inf         = t_inf,
    covariates    = covariates,
    covariates_in = covariates_in,
    WGT_pop       = WGT_pop,
    AGE_pop       = AGE_pop,
    HGT_pop       = HGT_pop
  )
}

# -----------------------------------------------------------------------------
# initalize_C_nimo()
# -----------------------------------------------------------------------------
initalize_C_nimo <- function(nbatch, z_dim, n_cov, covariates) {
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
# EmpiricalBayesEstimate_nimo()
# Pure-R 1-cmt IV infusion analytical solution.
# rate_i, t_inf_i: R numeric scalars for profile i
# data_i: [T, 2] R matrix (TAD, DV)
# -----------------------------------------------------------------------------
EmpiricalBayesEstimate_nimo <- function(data_i, z_pop, omega_pop, mu_i,
                                         res, C_i, h, rate_i, t_inf_i) {
  Cz    <- as.numeric(torch_matmul(C_i, z_pop)$detach()[1:2])
  data_m <- as.matrix(data_i$detach())
  om    <- as.numeric(omega_pop$detach())
  mu_v  <- as.numeric(mu_i$detach())
  a_v   <- as.numeric(res[[1]]$detach())
  b_v   <- as.numeric(res[[2]]$detach())

  pred_1cmt_inf <- function(t_vec, CL, V, R0, tinf) {
    ke      <- CL / V
    c_dur   <- (R0 / CL) * (1 - exp(-ke * t_vec))
    dt2     <- pmax(t_vec - tinf, 0)
    c_at_tinf <- (R0 / CL) * (1 - exp(-ke * tinf))
    c_aft   <- c_at_tinf * exp(-ke * dt2)
    ifelse(t_vec <= tinf, c_dur, c_aft)
  }

  ebe_loss <- function(phi) {
    phi_h <- exp(phi)
    CL <- phi_h[1]; V <- phi_h[2]
    pred  <- pred_1cmt_inf(data_m[, 1], CL, V, rate_i, t_inf_i)
    pred  <- pmax(pred, 0)
    sigma <- a_v + b_v * pred
    eps   <- (data_m[, 2] - pred) / sigma
    sum(0.5 * eps^2 + log(sigma)) + 0.5 * sum((phi - Cz)^2 / om)
  }

  opt <- optim(mu_v, ebe_loss, method = "Nelder-Mead",
               control = list(reltol = 1e-6, maxit = 500))
  torch_tensor(opt$par, dtype = torch_float())
}
