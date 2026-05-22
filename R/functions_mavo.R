# =============================================================================
# functions_mavo.R  —  Data loading and helpers for mavoglurant PK
# Dataset: nlmixr2data::mavoglurant
#
# PK model: 2-compartment IV infusion (analytical solution)
#   C1(t) = sum of 2 exponential terms; rate and duration vary per profile
# Parameters: (CL, V1, Q, V2) in log-space   [units: L/h and L]
# Dose scaling: AMT mg × 1000 → μg;  RATE mg/h × 1000 → μg/h
#   so DV (ng/mL = μg/L) is consistent with μg doses and L volumes
# Each OCC treated as an independent profile (occasion-split approach).
# Covariates: AGE (log-ratio), SEX (0/1), WT (log-ratio), HT (log-ratio)
# =============================================================================
library(torch)

# -----------------------------------------------------------------------------
# load_data_mavo()
#
# Returns:
#   data         : [N_eff, T_max, 2]  (TAD, DV)
#   data_in      : [N_eff, T_max, 2]  standardised
#   lengths      : [N_eff]            int, obs per profile
#   rate         : [N_eff]            infusion rate in μg/h (= RATE_mg_per_h × 1000)
#   t_inf        : [N_eff]            infusion duration in h  (= AMT_mg/RATE_mg_per_h)
#   covariates   : [N_eff, 4]         (log_age, sex_bin, log_wt, log_ht)
#   covariates_in: [N_eff, 4]         same
# -----------------------------------------------------------------------------
load_data_mavo <- function() {
  if (!requireNamespace("nlmixr2data", quietly = TRUE))
    stop("Package 'nlmixr2data' is required.")
  df <- nlmixr2data::mavoglurant

  df_dose <- df[df$EVID == 1, ]                          # infusion dose rows
  df_obs  <- df[df$EVID == 0 & df$CMT == 2, ]           # concentration obs

  # Build profile list: one row per (ID, OCC)
  prof_keys <- unique(df_dose[, c("ID", "OCC")])
  prof_keys <- prof_keys[order(prof_keys$ID, prof_keys$OCC), ]
  N_eff <- nrow(prof_keys)

  T_max <- max(table(paste(df_obs$ID, df_obs$OCC)))

  data_arr <- array(0,   dim = c(N_eff, T_max, 2L))
  len_vec  <- integer(N_eff)
  rate_vec <- numeric(N_eff)
  tinf_vec <- numeric(N_eff)
  cov_arr  <- matrix(0,  N_eff, 4L)    # (AGE, SEX, WT, HT) raw

  for (k in seq_len(N_eff)) {
    pid <- prof_keys$ID[k]
    occ <- prof_keys$OCC[k]

    dose_row <- df_dose[df_dose$ID == pid & df_dose$OCC == occ, ]
    obs_k    <- df_obs[df_obs$ID   == pid & df_obs$OCC  == occ, ]
    obs_k    <- obs_k[order(obs_k$TIME), ]

    time_dose <- dose_row$TIME[1]
    ni        <- nrow(obs_k)
    len_vec[k] <- ni

    data_arr[k, 1:ni, 1] <- obs_k$TIME - time_dose   # TAD
    data_arr[k, 1:ni, 2] <- obs_k$DV

    rate_vec[k] <- dose_row$RATE[1] * 1000            # mg/h → μg/h
    tinf_vec[k] <- dose_row$AMT[1]  / dose_row$RATE[1]  # h

    cov_arr[k, 1] <- dose_row$AGE[1]
    cov_arr[k, 2] <- as.integer(dose_row$SEX[1])   # already 0/1
    cov_arr[k, 3] <- dose_row$WT[1]
    cov_arr[k, 4] <- dose_row$HT[1]
  }

  # Log-ratio transform continuous covariates; SEX stays as 0/1
  AGE_pop <- mean(cov_arr[, 1])
  WT_pop  <- mean(cov_arr[, 3])
  HT_pop  <- mean(cov_arr[, 4])

  cov_trans        <- cov_arr
  cov_trans[, 1]   <- log(cov_arr[, 1] / AGE_pop)  # log age
  # col 2 = SEX (binary, no transform)
  cov_trans[, 3]   <- log(cov_arr[, 3] / WT_pop)   # log WT
  cov_trans[, 4]   <- log(cov_arr[, 4] / HT_pop)   # log HT

  data       <- torch_tensor(data_arr,  dtype = torch_float())
  lengths    <- torch_tensor(len_vec,   dtype = torch_int())
  rate       <- torch_tensor(rate_vec,  dtype = torch_float())
  t_inf      <- torch_tensor(tinf_vec,  dtype = torch_float())
  covariates <- torch_tensor(cov_trans, dtype = torch_float())   # [N_eff, 4]
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
    AGE_pop       = AGE_pop,
    WT_pop        = WT_pop,
    HT_pop        = HT_pop
  )
}

# -----------------------------------------------------------------------------
# initalize_C_mavo()
# -----------------------------------------------------------------------------
initalize_C_mavo <- function(nbatch, z_dim, n_cov, covariates) {
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
# EmpiricalBayesEstimate_mavo()
# Pure-R 2-compartment analytical IV infusion.
# rate_i, t_inf_i: R numeric scalars for profile i
# data_i: [T, 2] R matrix (TAD, DV)
# -----------------------------------------------------------------------------
EmpiricalBayesEstimate_mavo <- function(data_i, z_pop, omega_pop, mu_i,
                                         res, C_i, h, rate_i, t_inf_i) {
  Cz    <- as.numeric(torch_matmul(C_i, z_pop)$detach()[1:4])
  data_m <- as.matrix(data_i$detach())
  om    <- as.numeric(omega_pop$detach())
  mu_v  <- as.numeric(mu_i$detach())
  a_v   <- as.numeric(res[[1]]$detach())
  b_v   <- as.numeric(res[[2]]$detach())

  pred_2cmt <- function(t_vec, CL, V1, Q, V2, R0, tinf) {
    k10 <- CL / V1; k12 <- Q / V1; k21 <- Q / V2
    S   <- k10 + k12 + k21
    L   <- sqrt(pmax((k10 + k12 - k21)^2 + 4 * k12 * k21, 1e-20))
    al  <- (S + L) / 2;  be <- (S - L) / 2
    dab <- max(al - be, 1e-8)
    A   <- R0 * (al - k21) / (V1 * dab * max(al, 1e-10))
    B   <- -R0 * (be - k21) / (V1 * dab * max(be, 1e-10))
    # During infusion
    c_dur <- A * (1 - exp(-al * t_vec)) + B * (1 - exp(-be * t_vec))
    # After infusion
    dt2   <- pmax(t_vec - tinf, 0)
    c_aft <- A * (1 - exp(-al * tinf)) * exp(-al * dt2) +
             B * (1 - exp(-be * tinf)) * exp(-be * dt2)
    ifelse(t_vec <= tinf, c_dur, c_aft)
  }

  ebe_loss <- function(phi) {
    phi_h <- exp(phi)
    CL <- phi_h[1]; V1 <- phi_h[2]; Q <- phi_h[3]; V2 <- phi_h[4]
    pred  <- pred_2cmt(data_m[, 1], CL, V1, Q, V2, rate_i, t_inf_i)
    pred  <- pmax(pred, 0)
    sigma <- a_v + b_v * pred
    eps   <- (data_m[, 2] - pred) / sigma
    sum(0.5 * eps^2 + log(sigma)) + 0.5 * sum((phi - Cz)^2 / om)
  }

  opt <- optim(mu_v, ebe_loss, method = "Nelder-Mead",
               control = list(reltol = 1e-6, maxit = 500))
  torch_tensor(opt$par, dtype = torch_float())
}
