# =============================================================================
# functions_theo.R  —  Data loading and helpers for Case Study 1 (Theophylline)
# R translation of functions_theo.py (Jan Rohleff, CPT:PSP 2025)
# =============================================================================
library(torch)

# -----------------------------------------------------------------------------
# Load theophylline single-dose data from the .tab file
# Returns a list matching the Python output:
#   data         : [N, T, 5]  (time, conc, dose, weight, sex)
#   data_in      : [N, T, 2]  standardised (time, conc)
#   lengths      : [N]        integer (number of observations per individual)
#   dose         : [N]        individual dose
#   weight_pop   : scalar     mean weight
#   covariates   : [N, n_cov] raw covariates (weight, sex)
#   covariates_in: [N, n_cov] standardised covariates
# -----------------------------------------------------------------------------
load_data_theo <- function(path) {
  df <- read.table(path, header = TRUE, sep = "", stringsAsFactors = FALSE)
  # Columns: Id  Dose  Time  Concentration  Weight  Sex

  ids       <- unique(df$Id)
  N         <- length(ids)
  T_max     <- max(table(df$Id))
  n_feat    <- 5L  # time, conc, dose, weight, sex

  data_arr  <- array(0, dim = c(N, T_max, n_feat))
  len_vec   <- integer(N)

  for (i in seq_along(ids)) {
    sub          <- df[df$Id == ids[i], ]
    ni           <- nrow(sub)
    len_vec[i]   <- ni
    data_arr[i, 1:ni, 1] <- sub$Time
    data_arr[i, 1:ni, 2] <- sub$Concentration
    data_arr[i, 1:ni, 3] <- sub$Dose       # non-zero only at row 1
    data_arr[i, 1:ni, 4] <- sub$Weight
    data_arr[i, 1:ni, 5] <- sub$Sex
  }

  data    <- torch_tensor(data_arr, dtype = torch_float())
  lengths <- torch_tensor(len_vec,  dtype = torch_int())

  dose       <- data[, 1, 3]                    # [N]  dose at t=0
  weight_pop <- data[, 1, 4]$mean()
  covariates <- data[, 1, 4:5]                  # [N, 2]  weight, sex

  # Standardise inputs
  data_in <- data[,, 1:2]$clone()
  data_in[,, 1] <- data_in[,, 1] / data_in[,, 1]$max()
  data_mean <- data[,, 2]$mean()
  data_std  <- data[,, 2]$std()
  data_in[,, 2] <- (data_in[,, 2] - data_mean) / data_std

  covariates_in        <- data[, 1, 4:5]$clone()
  covariates_in[, 1]   <- (covariates_in[, 1] - covariates_in[, 1]$mean()) /
                            covariates_in[, 1]$std()

  list(data = data, data_in = data_in, lengths = lengths, dose = dose,
       weight_pop = weight_pop, covariates = covariates,
       covariates_in = covariates_in)
}

# -----------------------------------------------------------------------------
# Load theophylline multiple-dose data from a CSV file
# Columns: ID, Dose, Time, Concentration, Weight, Sex
# -----------------------------------------------------------------------------
load_data_multiple_theo <- function(path) {
  df    <- read.csv(path, stringsAsFactors = FALSE)
  ids   <- sort(unique(df$ID))
  N     <- length(ids)
  T_max <- max(table(df$ID))

  data_arr <- array(0, dim = c(N, T_max, 5L))
  len_vec  <- integer(N)

  for (i in seq_along(ids)) {
    sub          <- df[df$ID == ids[i], ]
    ni           <- nrow(sub)
    len_vec[i]   <- ni
    data_arr[i, 1:ni, 1] <- sub$Time
    data_arr[i, 1:ni, 2] <- sub$Concentration
    data_arr[i, 1:ni, 3] <- sub$Dose
    data_arr[i, 1:ni, 4] <- sub$Weight
    data_arr[i, 1:ni, 5] <- sub$Sex
  }

  data    <- torch_tensor(data_arr, dtype = torch_float())
  lengths <- torch_tensor(len_vec,  dtype = torch_int())

  weight_pop <- data[, 1, 4]$mean()
  covariates <- data[, 1, 4:5]

  # Build multi-dose table: [N, n_doses, 2]  (dose_time, dose_amount)
  dose_list <- lapply(seq_along(ids), function(i) {
    sub    <- df[df$ID == ids[i], ]
    d_rows <- sub[sub$Dose != 0, c("Time", "Dose")]
    as.matrix(d_rows)
  })
  n_doses_max <- max(sapply(dose_list, nrow))
  dose_arr    <- array(0, dim = c(N, n_doses_max, 2L))
  for (i in seq_along(ids)) {
    dm                   <- dose_list[[i]]
    dose_arr[i, 1:nrow(dm), ] <- dm
  }
  dose <- torch_tensor(dose_arr, dtype = torch_float())

  # Standardise inputs
  data_in <- data[,, 1:2]$clone()
  data_in[,, 1] <- data_in[,, 1] / data_in[,, 1]$max()
  data_mean     <- data[,, 2]$mean()
  data_std      <- data[,, 2]$std()
  data_in[,, 2] <- (data_in[,, 2] - data_mean) / data_std

  covariates_in      <- data[, 1, 4:5]$clone()
  covariates_in[, 1] <- (covariates_in[, 1] - covariates_in[, 1]$mean()) /
                         covariates_in[, 1]$std()

  list(data = data, data_in = data_in, lengths = lengths, dose = dose,
       weight_pop = weight_pop, covariates = covariates,
       covariates_in = covariates_in)
}

# -----------------------------------------------------------------------------
# Build the covariate design matrix C and C_regression for theophylline
# Covariate model:
#   log(theta_k,i) = log(theta_k,pop) + beta_k_w * log(w_i/w_pop)
#                                      + beta_k_sex * sex_i
# C[i] is [z_dim, z_dim + z_dim*n_cov] such that C[i] * z_pop = z_pop_i
# -----------------------------------------------------------------------------
initalize_C_theo <- function(nbatch, z_dim, n_cov, covariates, weight_pop) {
  C   <- torch_zeros(nbatch, z_dim, z_dim + z_dim * n_cov)
  C_regression <- torch_zeros(z_dim, nbatch, 1L + n_cov)

  for (i in seq_len(nbatch)) {
    C[i,, 1:z_dim] <- torch_eye(z_dim)
    count <- 0L
    for (k in seq_len(z_dim)) {
      for (j in seq_len(n_cov)) {
        col_idx <- j + z_dim + count
        if (j == 2L) {
          # sex: binary covariate (no log-transform)
          C[i, k, col_idx] <- covariates[i, j]
        } else {
          # continuous covariate: log-ratio normalisation
          C[i, k, col_idx] <- torch_log(covariates[i, j] / weight_pop)
        }
      }
      count <- count + n_cov
    }
  }

  for (k in seq_len(z_dim)) {
    for (i in seq_len(nbatch)) {
      C_regression[k, i, 1] <- 1.0
      for (j in seq_len(n_cov)) {
        if (j == 2L) {
          C_regression[k, i, j + 1L] <- covariates[i, j]
        } else {
          C_regression[k, i, j + 1L] <- torch_log(covariates[i, j] / weight_pop)
        }
      }
    }
  }

  list(C = C, C_regression = C_regression)
}

# -----------------------------------------------------------------------------
# Empirical Bayes Estimate (EBE) for individual parameters
# Minimises the individual posterior objective with Nelder-Mead
# -----------------------------------------------------------------------------
EmpiricalBayesEstimate_theo <- function(data_i, z_pop, omega_pop, mu_i, res, C_i, h) {
  a      <- as.numeric(res[[1]]$detach())
  b      <- as.numeric(res[[2]]$detach())
  Cz     <- as.numeric(torch_matmul(C_i, z_pop)$detach()[1:length(mu_i)])
  data_m <- as.matrix(data_i$detach())
  om     <- as.numeric(omega_pop$detach())
  mu_v   <- as.numeric(mu_i$detach())
  C_mat  <- as.matrix(C_i$detach())

  ebe_loss <- function(phi) {
    phi_h  <- as.numeric(h(torch_tensor(phi, dtype = torch_float()))$detach())
    dose   <- data_m[1, 3]
    t_obs  <- data_m[, 1]
    pred   <- dose * phi_h[1] / (phi_h[3] * (phi_h[1] - phi_h[2])) *
              (exp(-phi_h[2] * t_obs) - exp(-phi_h[1] * t_obs))
    sigma  <- a + b * pred
    eps    <- (data_m[, 2] - pred) / sigma
    sum(0.5 * eps^2 + log(sigma)) +
      0.5 * sum((phi - Cz)^2 / om)
  }

  opt <- optim(mu_v, ebe_loss, method = "Nelder-Mead",
               control = list(reltol = 1e-6, maxit = 500))
  torch_tensor(opt$par, dtype = torch_float())
}
