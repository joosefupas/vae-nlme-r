# =============================================================================
# functions_neonates.R  —  Data loading and helpers for Case Study 2 (Neonates)
# R translation of functions_neonates.py (Jan Rohleff, CPT:PSP 2025)
# =============================================================================
library(torch)

# -----------------------------------------------------------------------------
# Load neonates data from CSV + compute per-individual lengths
# Columns: ID, Time, DV, Sex, DelM, GA, Mage, Para2
#
# Returns:
#   data         : [N, T_max, 8]  raw tensor (time, DV, Sex, DelM, GA_tr, Mage_tr, Para2, pad)
#   data_in      : [N, T_max, 2]  standardised (time, DV)
#   lengths      : [N]            number of observations per individual
#   covariates   : [N, n_cov]     log-transformed covariates (padded into data rows)
# -----------------------------------------------------------------------------
load_data_neonates <- function(path_data) {
  df  <- read.csv(path_data, stringsAsFactors = FALSE)
  ids <- sort(unique(df$ID))
  N   <- length(ids)
  T_max <- max(table(df$ID))

  # Feature layout: [time, DV, Sex, DelM, GA, Mage, Para2]
  n_feat   <- 7L
  data_arr <- array(0, dim = c(N, T_max, n_feat))
  len_vec  <- integer(N)

  for (i in seq_along(ids)) {
    sub        <- df[df$ID == ids[i], ]
    ni         <- nrow(sub)
    len_vec[i] <- ni
    data_arr[i, 1:ni, 1] <- sub$Time
    data_arr[i, 1:ni, 2] <- sub$DV
    data_arr[i, 1:ni, 3] <- sub$Sex
    data_arr[i, 1:ni, 4] <- sub$DelM
    data_arr[i, 1:ni, 5] <- sub$GA
    data_arr[i, 1:ni, 6] <- sub$Mage
    data_arr[i, 1:ni, 7] <- sub$Para2
  }

  data    <- torch_tensor(data_arr, dtype = torch_float())
  lengths <- torch_tensor(len_vec,  dtype = torch_int())

  # ---- log-transform GA and Mage (centred by population mean) -------------
  GA    <- data[, 1, 5]$clone()
  Mage  <- data[, 1, 6]$clone()
  mask  <- torch_arange(T_max)$unsqueeze(1) < lengths$unsqueeze(1)$t()
  mask  <- mask$t()   # [N, T_max]

  # GA_tr and Mage_tr stored back into data tensor (only at observed rows)
  for (i in seq_len(N)) {
    ni <- len_vec[i]
    data[i, 1:ni, 5] <- torch_log(GA[i] / GA$mean())
    data[i, 1:ni, 6] <- torch_log(Mage[i] / Mage$mean())
  }

  # ---- covariates: one row per individual ---------------------------------
  # Covariates: Sex, DelM, GA (log-centred), Mage (log-centred), Para2
  covariates <- data[, 1, 3:7]   # [N, 5]

  # ---- standardise inputs -------------------------------------------------
  data_in         <- data[,, 1:2]$clone()
  data_in[,, 1]   <- data_in[,, 1] / data_in[,, 1]$max()
  data_mean       <- data[,, 2]$mean()
  data_std        <- data[,, 2]$std()
  data_in[,, 2]   <- (data_in[,, 2] - data_mean) / data_std

  list(data = data, data_in = data_in, lengths = lengths,
       covariates = covariates)
}

# -----------------------------------------------------------------------------
# Build covariate design matrix C for neonates
# All covariates are already log-centred (Sex/DelM/Para2 are binary/raw).
# C[i] is [z_dim, z_dim + z_dim*n_cov]
# -----------------------------------------------------------------------------
initalize_C_neonates <- function(nbatch, z_dim, n_cov, covariates) {
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
# Build per-individual t_eval tensor for the RK4 ODE decoder
# t_eval[i, ] = actual timepoints for individual i, padded with last timepoint
# -----------------------------------------------------------------------------
build_t_eval_neonates <- function(data, lengths) {
  nbatch <- data$shape[1]
  T_max  <- lengths$max()$item()
  t_eval <- torch_zeros(nbatch, T_max)
  for (i in seq_len(nbatch)) {
    ni        <- lengths[i]$item()
    t_last    <- data[i, ni, 1]
    t_eval[i, 1:ni]            <- data[i, 1:ni, 1]
    if (ni < T_max) {
      t_eval[i, (ni + 1):T_max] <- t_last$expand(c(T_max - ni))
    }
  }
  t_eval
}
