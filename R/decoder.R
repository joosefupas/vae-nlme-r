# =============================================================================
# decoder.R  —  Decoder functions for all case studies
# R translation of decoder.py (Jan Rohleff, CPT:PSP 2025)
# =============================================================================
library(torch)

# -----------------------------------------------------------------------------
# Case Study 1 — Theophylline (single dose)
#   1-compartment PK with first-order absorption and elimination
#   C(t) = D * ka / (V * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))
#
# z_normal : [N, z_dim]   latent parameters in log-space
# time     : [N, T]       observation times
# h        : function     inverse-link  (exp for log-normal params)
# dose     : [N]          dose per individual
# -----------------------------------------------------------------------------
decoder_theophylline <- function(z_normal, time, h, dose) {
  z  <- h(z_normal)           # [N, z_dim] in natural scale
  ka <- z[, 1]                # absorption rate constant
  ke <- z[, 2]                # elimination rate constant
  V  <- z[, 3]                # volume of distribution

  nbatch <- dose$shape[1]
  n_time <- time$shape[2]

  pred_x <- torch_zeros(nbatch, n_time, 1L)
  for (t in seq_len(n_time)) {
    pred_x[, t, 1] <- dose * ka / (V * (ka - ke)) *
      (torch_exp(-ke * time[, t]) - torch_exp(-ka * time[, t]))
  }
  pred_x
}

# -----------------------------------------------------------------------------
# Case Study 1 — Theophylline (multiple dosing)
#   Superposition of 1-compartment contributions from each dose event
#
# dose : [N, n_doses, 2]   col1 = dose time, col2 = dose amount
# -----------------------------------------------------------------------------
decoder_theophylline_multiple <- function(z_normal, time, h, dose) {
  z  <- h(z_normal)
  ka <- z[, 1, drop = FALSE]   # [N, 1]
  ke <- z[, 2, drop = FALSE]
  V  <- z[, 3, drop = FALSE]

  nbatch <- dose$shape[1]
  n_time <- time$shape[2]

  # Superposition principle: sum contributions from all prior doses
  sol <- function(t) {
    t_dose <- dose[,, 1]   # [N, n_doses]  times of dose events
    AMT    <- dose[,, 2]   # [N, n_doses]  dose amounts

    mask    <- t$unsqueeze(2) >= t_dose   # [N, n_doses]
    delta_t <- (t$unsqueeze(2) - t_dose) * mask

    expo  <- torch_exp(-ke * delta_t) - torch_exp(-ka * delta_t)
    terms <- AMT * ka / (V * (ka - ke)) * expo * mask   # [N, n_doses]
    terms$sum(dim = 2)   # [N]
  }

  pred_x <- torch_zeros(nbatch, n_time, 1L)
  for (t in seq_len(n_time)) {
    pred_x[, t, 1] <- sol(time[, t])
  }
  pred_x
}

# -----------------------------------------------------------------------------
# Case Study 2 — Neonates (ODE model)
#   Weight progression ODE (Bräm et al.):
#     dW/dt = kin * sigmoid(2*(t - T_lag)) - kout_max * (1 - t/(T50+t)) * W
#     W(0)  = W0
#
# Solved with a fixed-step batched RK4 in torch (fully differentiable).
#
# z_normal  : [N, z_dim=5]   latent parameters  (W0, kin, Tlag, koutmax, T50)
# t_eval    : [N, T_max]     per-individual evaluation times (padded)
# h         : function        inverse-link (exp)
# step_size : float           RK4 step size in days (default 0.05)
# -----------------------------------------------------------------------------
decoder_neonates <- function(z_normal, t_eval, h, step_size = 0.5) {
  z       <- h(z_normal)         # [N, 5] in natural scale
  W0      <- z[, 1]
  kin     <- z[, 2]
  TL      <- z[, 3]
  koutmax <- z[, 4]
  T50     <- z[, 5]

  nbatch <- z_normal$shape[1]
  n_eval <- t_eval$shape[2]

  t_max     <- as.numeric(t_eval[, n_eval]$max()$item())
  n_steps   <- as.integer(ceiling(t_max / step_size))
  actual_dt <- t_max / n_steps
  t_grid    <- seq(0, t_max, length.out = n_steps + 1L)

  # ODE right-hand side (vectorised over N individuals)
  f_ode <- function(t_scalar, W) {
    t_vec <- torch_full(list(nbatch), t_scalar)
    kprod <- kin * torch_sigmoid(2.0 * (t_vec - TL))
    kelim <- koutmax * (1.0 - t_vec / (T50 + t_vec))
    kprod - kelim * W
  }

  # RK4: accumulate W at each grid step in a list (no in-place tensor ops,
  # preserves autograd graph through W_store → pred_x)
  W      <- W0$clone()
  W_list <- vector("list", n_steps + 1L)
  W_list[[1L]] <- W

  for (step in seq_len(n_steps)) {
    t0 <- t_grid[step]
    k1 <- f_ode(t0,                  W)
    k2 <- f_ode(t0 + actual_dt / 2,  W + (actual_dt / 2) * k1)
    k3 <- f_ode(t0 + actual_dt / 2,  W + (actual_dt / 2) * k2)
    k4 <- f_ode(t0 + actual_dt,      W + actual_dt * k3)
    W  <- W + (actual_dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    W_list[[step + 1L]] <- W
  }

  # Stack list of [N] tensors → W_store [N, n_steps+1]
  W_store <- torch_stack(W_list, dim = 2L)

  # Map each observation time to the nearest RK4 grid index (1-indexed for R torch gather)
  eval_idx <- torch_clamp(
    torch_round(t_eval / actual_dt)$to(dtype = torch_long()) + 1L,
    min = 1L, max = n_steps + 1L
  )  # [N, n_eval], 1-indexed

  # Vectorised gather: pred_x[i,j] = W_store[i, eval_idx[i,j]]
  pred_x <- torch_gather(W_store, 2L, eval_idx)$unsqueeze(3L)  # [N, n_eval, 1]

  pred_x
}

# -----------------------------------------------------------------------------
# nlmixr2 Case A — Warfarin (single oral dose, PK only)
#   Identical formula to theophylline; provided as a named alias for clarity.
#   z_dim = 3: (ka, ke, V)
# -----------------------------------------------------------------------------
decoder_warfarin <- function(z_normal, time, h, dose) {
  decoder_theophylline(z_normal, time, h, dose)
}

# -----------------------------------------------------------------------------
# nlmixr2 Case B — Phenobarbital (multiple IV bolus doses, superposition)
#   C(t) = (1/V) * sum_{k: t_k <= t} AMT_k * exp(-CL/V * (t - t_k))
#   z_dim = 2: (CL, V)
#
# z_normal  : [N, 2]   latent (log CL, log V)
# obs_times : [N, T]   observation times (padded)
# h         : function  exp
# dose_times: [N, D]   dose event times  (padded with 1e9)
# dose_amts : [N, D]   dose amounts      (padded with 0)
# -----------------------------------------------------------------------------
decoder_pheno_1cmt <- function(z_normal, obs_times, h, dose_times, dose_amts) {
  z  <- h(z_normal)             # [N, 2]
  CL <- z[, 1]                  # [N]
  V  <- z[, 2]                  # [N]
  ke <- CL / V                  # [N]

  N <- z_normal$shape[1]

  # Reshape for [N, T, D] broadcasting
  ke3    <- ke$view(c(N, 1L, 1L))        # [N, 1, 1]
  V2     <- V$view(c(N, 1L))             # [N, 1]
  obs_t  <- obs_times$unsqueeze(3L)      # [N, T, 1]
  dose_t <- dose_times$unsqueeze(2L)     # [N, 1, D]
  dose_a <- dose_amts$unsqueeze(2L)      # [N, 1, D]

  dt   <- obs_t - dose_t                 # [N, T, D]
  mask <- (dt >= 0)$to(dtype = torch_float())

  # Contribution: AMT_k/V * exp(-ke*(t - t_k)) for each dose k
  contrib <- dose_a * torch_exp(-ke3 * torch_clamp(dt, min = 0)) * mask
  conc    <- contrib$sum(dim = 3L) / V2  # [N, T]

  conc$unsqueeze(3L)                     # [N, T, 1]
}

# -----------------------------------------------------------------------------
# nlmixr2 Case C — Mavoglurant (2-compartment IV infusion, analytical)
#   During infusion: C1(t) = A*(1-exp(-α*t)) + B*(1-exp(-β*t))
#   After  infusion: C1(t) = A*(1-exp(-α*Tinf))*exp(-α*(t-Tinf))
#                           + B*(1-exp(-β*Tinf))*exp(-β*(t-Tinf))
#   z_dim = 4: (CL, V1, Q, V2)  in log-space
#
# z_normal : [N, 4]   latent parameters
# time     : [N, T]   TAD (time after dose), observation times
# h        : function  exp
# rate     : [N]      infusion rate in μg/h
# t_inf    : [N]      infusion duration in h
# -----------------------------------------------------------------------------
decoder_mavo_2cmt <- function(z_normal, time, h, rate, t_inf) {
  z  <- h(z_normal)             # [N, 4]
  CL <- z[, 1];  V1 <- z[, 2]
  Q  <- z[, 3];  V2 <- z[, 4]

  k10 <- CL / V1
  k12 <- Q  / V1
  k21 <- Q  / V2

  # Macro hybrid rate constants
  S    <- k10 + k12 + k21
  Ld   <- torch_sqrt(torch_clamp((k10 + k12 - k21)^2 + 4 * k12 * k21, min = 1e-12))
  al   <- (S + Ld) / 2           # [N]  larger root
  be   <- (S - Ld) / 2           # [N]  smaller root
  dab  <- torch_clamp(al - be, min = 1e-8)

  # Coefficients for IV infusion (rate R0)
  R0 <- rate
  A  <- R0 * (al - k21) / (V1 * dab * torch_clamp(al, min = 1e-10))  # [N]
  B  <- -R0 * (be - k21) / (V1 * dab * torch_clamp(be, min = 1e-10)) # [N]

  # Expand to [N, T]
  A2    <- A$unsqueeze(2L);     B2    <- B$unsqueeze(2L)
  al2   <- al$unsqueeze(2L);    be2   <- be$unsqueeze(2L)
  tinf2 <- t_inf$unsqueeze(2L)

  # During infusion (time <= t_inf)
  c_dur <- A2 * (1 - torch_exp(-al2 * time)) +
           B2 * (1 - torch_exp(-be2 * time))

  # After infusion
  dt2   <- torch_clamp(time - tinf2, min = 0.0)
  c_aft <- A2 * (1 - torch_exp(-al2 * tinf2)) * torch_exp(-al2 * dt2) +
           B2 * (1 - torch_exp(-be2 * tinf2)) * torch_exp(-be2 * dt2)

  during_mask <- (time <= tinf2)$to(dtype = torch_float())
  conc <- c_dur * during_mask + c_aft * (1 - during_mask)

  conc$unsqueeze(3L)             # [N, T, 1]
}

# -----------------------------------------------------------------------------
# nlmixr2 Case D — NimoData (1-compartment IV infusion, analytical)
#   During infusion: C(t) = (R/CL)*(1 - exp(-CL/V * t))
#   After  infusion: C(t) = (R/CL)*(1 - exp(-CL/V * Tinf)) * exp(-CL/V*(t-Tinf))
#   z_dim = 2: (CL, V)
#
# z_normal : [N, 2]  latent (log CL, log V)
# time     : [N, T]  TAD (time after dose)
# h        : function exp
# rate     : [N]     infusion rate (same units as AMT/h)
# t_inf    : [N]     infusion duration = AMT/RATE (h)
# -----------------------------------------------------------------------------
decoder_nimo_1cmt <- function(z_normal, time, h, rate, t_inf) {
  z  <- h(z_normal)             # [N, 2]
  CL <- z[, 1]                  # [N]
  V  <- z[, 2]                  # [N]
  ke <- CL / V                  # [N]

  # Expand to [N, T]
  CL2   <- CL$unsqueeze(2L)
  ke2   <- ke$unsqueeze(2L)
  R2    <- rate$unsqueeze(2L)
  tinf2 <- t_inf$unsqueeze(2L)

  # During infusion
  c_dur <- (R2 / CL2) * (1 - torch_exp(-ke2 * time))

  # After infusion
  c_at_tinf <- (R2 / CL2) * (1 - torch_exp(-ke2 * tinf2))
  dt2   <- torch_clamp(time - tinf2, min = 0.0)
  c_aft <- c_at_tinf * torch_exp(-ke2 * dt2)

  during_mask <- (time <= tinf2)$to(dtype = torch_float())
  conc <- c_dur * during_mask + c_aft * (1 - during_mask)

  conc$unsqueeze(3L)             # [N, T, 1]
}
