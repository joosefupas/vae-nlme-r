# =============================================================================
# Main/neonates.R  —  VAE-NLME  Case Study 2: Neonatal weight progression
# R translation of Main/neonates.py  (Jan Rohleff, CPT:PSP 2025)
#
# ODE model (Bräm et al.):
#   dW/dt = kin * sigmoid(2*(t - T_lag))  -  kout_max*(1 - t/(T50+t))*W
#   W(0)  = W0
# Parameters: (W0, kin, T_lag, kout_max, T50)
# Covariates: Sex, DelM, GA (log-centred), Mage (log-centred), Para2
#
# The ODE is solved with a differentiable batched RK4 implemented in
# R torch (decoder_neonates in R/decoder.R).  rxode2 is NOT required
# for the training loop (no closed-form gradient needed from rxode2).
# rxode2 can optionally be used *after* training for post-hoc simulation.
# =============================================================================

# ---- resolve project root (works interactively, via source(), and in RStudio) -
.resolve_root <- function() {
  for (i in seq_len(sys.nframe())) {
    f <- sys.frame(i)$ofile
    if (!is.null(f) && nchar(f)) return(normalizePath(file.path(dirname(f), ".."), winslash = "/"))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- tryCatch(rstudioapi::getActiveDocumentContext()$path, error = function(e) "")
    if (nchar(p)) return(normalizePath(file.path(dirname(p), ".."), winslash = "/"))
  }
  wd <- getwd()
  if (basename(wd) == "Main") return(normalizePath("..", winslash = "/"))
  wd
}
root_dir <- .resolve_root()

source(file.path(root_dir, "R", "functions.R"))
source(file.path(root_dir, "R", "encoder.R"))
source(file.path(root_dir, "R", "decoder.R"))
source(file.path(root_dir, "R", "pop_parameter.R"))
source(file.path(root_dir, "R", "functions_neonates.R"))
source(file.path(root_dir, "R", "visualization.R"))

library(torch)
torch_manual_seed(1L)

# =============================================================================
# 1. Load data
# =============================================================================
data_path <- file.path(root_dir, "Data", "neonates_data.csv")
loaded    <- load_data_neonates(data_path)
data      <- loaded$data
data_in   <- loaded$data_in
lengths   <- loaded$lengths
covariates <- loaded$covariates   # [N, 5]  (Sex, DelM, GA_tr, Mage_tr, Para2)
covariates_in <- covariates       # alias used by save_vae_fit / PPC

# =============================================================================
# 2. Dimensions
# =============================================================================
nbatch <- data$shape[1]    # N = 189 neonates
x_dim  <- 2L               # (time, weight)
z_dim  <- 5L               # (W0, kin, Tlag, koutmax, T50)
n_cov  <- covariates$shape[2]   # 5

# =============================================================================
# 3. Build per-individual t_eval for the RK4 decoder
# =============================================================================
t_eval <- build_t_eval_neonates(data, lengths)   # [N, T_max]

# =============================================================================
# 4. Prior / link function
# =============================================================================
h         <- function(x) torch_exp(x)
h_inverse <- function(x) torch_log(x)

# =============================================================================
# 5. Encoder
# =============================================================================
h_dim  <- 100L                                                        # Table S2: n_h = 100
sigma0 <- torch_log(torch_tensor(c(1e-4, 1e-3, 1e-2, 1e-2, 1e-2)))  # Table S2: L_0 diag
mu0    <- torch_tensor(c(3000.0, 30.0, 2.0, 0.05, 1.0))

Encoder <- lstm_encoder(x_dim, h_dim, z_dim, n_cov, mu0, sigma0, h_inverse)

# =============================================================================
# 6. Decoder  (batched RK4)
# =============================================================================
Decoder <- function(z_normal, time, h) {
  decoder_neonates(z_normal, t_eval, h, step_size = 0.5)
}

# =============================================================================
# 7. Covariate design matrix
# =============================================================================
C_list       <- initalize_C_neonates(nbatch, z_dim, n_cov, covariates)
C            <- C_list$C
C_regression <- C_list$C_regression

names_co <- c(
  "Sex_W0",    "DelM_W0",    "GAexact_W0",    "Mage_W0",    "Para2_W0",
  "Sex_kin",   "DelM_kin",   "GAexact_kin",   "Mage_kin",   "Para2_kin",
  "Sex_TL",    "DelM_TL",    "GAexact_TL",    "Mage_TL",    "Para2_TL",
  "Sex_kout",  "DelM_kout",  "GAexact_kout",  "Mage_kout",  "Para2_kout",
  "Sex_T50",   "DelM_T50",   "GAexact_T50",   "Mage_T50",   "Para2_T50"
)
penalized_indices <- seq_len(n_cov)
M <- z_dim * n_cov   # 25 covariate effects

# =============================================================================
# 8. Hyper-parameters
# =============================================================================
iters_burn_in <- 25L    # Table S2: K_burn = 25
kl_iter       <- 50L    # Table S2: K_KL   = 50
gamma_iter    <- 200L   # Table S2: K_2    = 200
iters         <- 250L   # Table S2: K_iter = 250
L_iter        <- 10L    # Table S2: L_iter = 10
alpha_KL      <- torch_linspace(0.01, 1.0, kl_iter)

# =============================================================================
# 9. Population parameter object
# =============================================================================
pop <- pop_parameter$new(z_dim, nbatch, gamma_iter, data, C, C_regression,
                         C[, 1:z_dim, 1:z_dim], penalized_indices,
                         n_cov, kl_iter, lengths, alpha = 2.0)

# =============================================================================
# 10. Burn-in
# =============================================================================
bi <- initialize_encoder(iters_burn_in, L_iter, Encoder, Decoder,
                          data, data_in, z_dim, covariates, lengths, h, pop)
Encoder <- bi$Encoder; optimizer <- bi$optimizer
pred_x_mean <- bi$pred_x; mu <- bi$mu; L <- bi$L
a <- bi$a; b <- bi$b
z_pop_iter_bi <- bi$z_pop_iter_bi; omega_pop_iter_bi <- bi$omega_pop_iter_bi
a_iter_bi <- bi$a_iter_bi; elbo_iter_bi <- bi$elbo_iter_bi

optimizer$param_groups[[1]]$lr <- 5e-3

# =============================================================================
# 11. Main training loop
# =============================================================================
z_pop_iter     <- torch_zeros(iters, z_dim + M)
omega_pop_iter <- torch_zeros(iters, z_dim)
a_iter         <- torch_zeros(iters)
Elbo_iter      <- torch_zeros(iters)

cat("\n#############################################\n")
cat("Training VAE (Neonates)\n")

for (iter in seq_len(iters)) {

  if (iter > 1L) {
    pred_x_mean <- data_matrix[(L_iter - 1):L_iter]$mean(dim = 1L)
  }

  if (iter > gamma_iter) {
    upd <- pop$update_pop(mu$detach(), L$detach(), pred_x_mean, iter,
                          covariate_selection = TRUE,
                          smoothing = TRUE, update_pop = TRUE)
  } else {
    upd <- pop$update_pop(mu$detach(), L$detach(), pred_x_mean, iter,
                          covariate_selection = TRUE, update_pop = TRUE)
  }
  z_pop <- upd$z_pop; omega_pop <- upd$omega_pop
  a <- upd$a; mu_smooth <- upd$mu_smooth

  data_matrix <- torch_zeros(L_iter, nbatch, data$shape[2], 1L)
  ELBO        <- torch_zeros(L_iter)

  for (l in seq_len(L_iter)) {
    enc      <- Encoder$forward(data_in, covariates, lengths)
    z_normal <- enc$z_normal; mu <- enc$mu; L <- enc$L; eps <- enc$eps

    pred_x <- Decoder(z_normal, data[,, 1], h)
    data_matrix[l, , ,] <- pred_x$clone()$detach()

    z_pop_batch <- torch_zeros(nbatch, z_dim)
    for (i in seq_len(nbatch)) z_pop_batch[i,] <- torch_matmul(C[i, ,], z_pop)

    p_xz <- p_x_z_compute(data[,, 2, drop = FALSE]$view(c(nbatch, data$shape[2], 1L)),
                           pred_x, list(a, b), lengths)
    pz   <- p_z_compute(z_normal, z_pop_batch, omega_pop)
    qz   <- q_z_x_compute(eps, torch_diagonal(L, dim1 = 2, dim2 = 3))
    DKL  <- pz - qz

    elbo <- if (iter < kl_iter) p_xz + alpha_KL[iter] * DKL else p_xz + DKL
    ELBO[l] <- (p_xz + DKL)$detach()

    elbo$backward(); optimizer$step(); optimizer$zero_grad()
  }

  if (iter < gamma_iter) {
    cat(sprintf("Iteration %d/%d\r", iter, gamma_iter))
  } else {
    cat(sprintf("Iteration %d/%d (smoothing)\r", iter - gamma_iter, iters - gamma_iter))
  }

  z_pop_iter[iter,]     <- torch_cat(list(h(z_pop[1:z_dim])$detach(),
                                          z_pop[(z_dim+1):(z_dim+M)]$detach()))
  omega_pop_iter[iter,] <- omega_pop$sqrt()
  a_iter[iter]          <- a
  Elbo_iter[iter]       <- ELBO$mean()
}

# =============================================================================
# 12. Log-likelihood
# =============================================================================
cat("\n#############################################\n")
cat("Computing log-likelihood\n")

LL_lin_mu <- LogLikelihood_linearization(z_pop, omega_pop, list(a, b),
                                          data, mu_smooth, C, h,
                                          data[,, 1], lengths, Decoder)
LL_is_out <- LogLikelihood_sample(100L, z_pop, omega_pop, list(a, b),
                                   data, mu, L, C, h,
                                   data[,, 1], lengths, Decoder)
LL_is <- LL_is_out$LL
cat("#############################################\n")

# =============================================================================
# 13. Output
# =============================================================================
z_pop_iter <- torch_vstack(list(
  torch_hstack(list(z_pop_iter_bi, torch_zeros(iters_burn_in, M))), z_pop_iter))
omega_pop_iter <- torch_vstack(list(omega_pop_iter_bi, omega_pop_iter))
a_iter         <- torch_hstack(list(a_iter_bi, a_iter))

printOutput_neonates(z_pop, omega_pop, a, b, z_dim, nbatch,
                      lengths$sum(), h, names_co, LL_lin_mu, LL_is)
saveOutput_neonates(z_pop, omega_pop, a, b, z_dim, nbatch,
                    lengths$sum(), h, names_co, LL_lin_mu, LL_is,
                    save_dir = file.path(root_dir, "Results"))

save_vae_fit(list(
  Encoder = Encoder, x_dim = x_dim, h_dim = h_dim, z_dim = z_dim, n_cov = n_cov,
  nbatch = nbatch, mu0 = mu0, sigma0 = sigma0,
  iters = iters, kl_iter = kl_iter, gamma_iter = gamma_iter, iters_burn_in = iters_burn_in,
  z_pop = z_pop, omega_pop = omega_pop, a = a, b = b,
  names_co = names_co,
  param_names = c("W0_pop", "kin_pop", "Tlag_pop", "koutmax_pop", "T50_pop"),
  omega_names = c("omega_W0", "omega_kin", "omega_Tlag", "omega_koutmax", "omega_T50"),
  LL_lin_mu = LL_lin_mu, LL_is = LL_is,
  data = data, lengths = lengths, C = C, data_in = data_in, covariates_in = covariates_in,
  z_pop_iter = z_pop_iter, omega_pop_iter = omega_pop_iter,
  Elbo_iter = Elbo_iter, a_iter = a_iter,
  dose = NULL, rate = NULL, t_inf = NULL, dose_times = NULL, dose_amts = NULL
), save_dir = file.path(root_dir, "Results"), dataset_name = "neonates")

setwd(root_dir)
plotConvergence_pop_neonates(Elbo_iter, a_iter, z_pop_iter, omega_pop_iter,
                              iters, kl_iter, gamma_iter, iters_burn_in)
plotConvergence_covariate_neonates(z_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in)

# =============================================================================
# VPC + PPC
# =============================================================================
plot_vpc_neonates(data, lengths, z_pop, omega_pop, a, b, C, z_dim,
                  n_sim=300L, nbins=8,
                  save_path=file.path(root_dir,"Plots","neonates_vpc.pdf"))
plot_ppc_neonates(data, lengths, Encoder, data_in, covariates_in,
                  a, b, z_dim,
                  n_sim=100L, nbins=8,
                  save_path=file.path(root_dir,"Plots","neonates_ppc.pdf"))
