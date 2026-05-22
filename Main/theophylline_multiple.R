# =============================================================================
# Main/theophylline_multiple.R  —  VAE-NLME  Case Study 1: multiple dosing
# R translation of Main/theophylline_multiple.py  (Jan Rohleff, CPT:PSP 2025)
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
source(file.path(root_dir, "R", "functions_theo.R"))
source(file.path(root_dir, "R", "visualization.R"))

library(torch)
torch_manual_seed(1L)

# =============================================================================
# 1. Load data
# =============================================================================
data_path <- file.path(root_dir, "Data", "theophylline_multiple_data.csv")
loaded    <- load_data_multiple_theo(data_path)
data      <- loaded$data
data_in   <- loaded$data_in
lengths   <- loaded$lengths
dose      <- loaded$dose          # [N, n_doses, 2]
weight_pop     <- loaded$weight_pop
covariates     <- loaded$covariates
covariates_in  <- loaded$covariates_in

# =============================================================================
# 2. Dimensions
# =============================================================================
nbatch <- data$shape[1]
x_dim  <- 2L
z_dim  <- 3L              # (ka, ke, V)
n_cov  <- covariates$shape[2]

# =============================================================================
# 3. Prior / link function
# =============================================================================
h         <- function(x) torch_exp(x)
h_inverse <- function(x) torch_log(x)

# =============================================================================
# 4. Encoder
# =============================================================================
h_dim  <- 25L
sigma0 <- torch_log(torch_tensor(c(1e-2, 5e-3, 1e-1)))
mu0    <- torch_tensor(c(1.0, 0.5, 15.0))

Encoder <- lstm_encoder(x_dim, h_dim, z_dim, n_cov, mu0, sigma0, h_inverse)

# =============================================================================
# 5. Decoder  (multiple-dosing analytical solution)
# =============================================================================
Decoder <- function(z_normal, time, h) {
  decoder_theophylline_multiple(z_normal, time, h, dose)
}

# =============================================================================
# 6. Covariate design matrix
# =============================================================================
C_list       <- initalize_C_theo(nbatch, z_dim, n_cov, covariates, weight_pop)
C            <- C_list$C
C_regression <- C_list$C_regression
names_co     <- c("beta_ka_weight", "beta_ka_sex",
                  "beta_ke_weight", "beta_ke_sex",
                  "beta_V_weight",  "beta_V_sex")
penalized_indices <- seq_len(n_cov)
M <- z_dim * n_cov

# =============================================================================
# 7. Hyper-parameters
# =============================================================================
iters_burn_in <- 100L
kl_iter       <- 50L
gamma_iter    <- 250L
iters         <- 300L
L_iter        <- 5L
alpha_KL      <- torch_linspace(0.01, 1.0, kl_iter)

# =============================================================================
# 8. Population parameter object
# =============================================================================
pop <- pop_parameter$new(z_dim, nbatch, gamma_iter, data, C, C_regression,
                         C[, 1:z_dim, 1:z_dim], penalized_indices,
                         n_cov, kl_iter, lengths, alpha = 2.0)

# =============================================================================
# 9. Burn-in
# =============================================================================
bi <- initialize_encoder(iters_burn_in, L_iter, Encoder, Decoder,
                          data, data_in, z_dim, covariates_in, lengths, h, pop)
Encoder <- bi$Encoder; optimizer <- bi$optimizer
pred_x_mean <- bi$pred_x; mu <- bi$mu; L <- bi$L
a <- bi$a; b <- bi$b
z_pop_iter_bi <- bi$z_pop_iter_bi; omega_pop_iter_bi <- bi$omega_pop_iter_bi
a_iter_bi <- bi$a_iter_bi; elbo_iter_bi <- bi$elbo_iter_bi

optimizer$param_groups[[1]]$lr <- 5e-3

# =============================================================================
# 10. Main training loop  (identical structure to single-dose case)
# =============================================================================
z_pop_iter     <- torch_zeros(iters, z_dim + M)
omega_pop_iter <- torch_zeros(iters, z_dim)
a_iter         <- torch_zeros(iters)
Elbo_iter      <- torch_zeros(iters)

cat("\n#############################################\n")
cat("Training VAE (multiple dosing)\n")

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
    enc      <- Encoder(data_in, covariates_in, lengths)
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
# 11. Log-likelihood
# =============================================================================
cat("\n#############################################\n")
cat("Computing log-likelihood\n")

LL_lin_mu <- LogLikelihood_linearization(z_pop, omega_pop, list(a, b),
                                          data, mu_smooth, C, h,
                                          data[,, 1], lengths, Decoder)

phi_opt <- torch_zeros(nbatch, z_dim)
for (i in seq_len(nbatch)) {
  phi_opt[i,] <- EmpiricalBayesEstimate_theo(data[i,,], z_pop, omega_pop,
                                              mu[i,], list(a, b), C[i,,], h)
}
LL_lin    <- LogLikelihood_linearization(z_pop, omega_pop, list(a, b),
                                          data, phi_opt, C, h,
                                          data[,, 1], lengths, Decoder)
LL_is_out <- LogLikelihood_sample(3000L, z_pop, omega_pop, list(a, b),
                                   data, mu, L, C, h,
                                   data[,, 1], lengths, Decoder)
LL_is <- LL_is_out$LL
cat("#############################################\n")

# =============================================================================
# 12. Output
# =============================================================================
z_pop_iter <- torch_vstack(list(
  torch_hstack(list(z_pop_iter_bi, torch_zeros(iters_burn_in, M))), z_pop_iter))
omega_pop_iter <- torch_vstack(list(omega_pop_iter_bi, omega_pop_iter))
a_iter         <- torch_hstack(list(a_iter_bi, a_iter))

printOutput_theo(z_pop, omega_pop, a, b, z_dim, nbatch,
                  lengths$sum(), h, names_co, LL_lin_mu, LL_is)

saveOutput_results(z_pop, omega_pop, a, b, z_dim, nbatch,
                   lengths$sum(), h,
                   param_names  = c("ka_pop", "ke_pop", "V_pop"),
                   omega_names  = c("omega_ka", "omega_ke", "omega_V"),
                   names_co     = names_co,
                   LL_lin_mu    = LL_lin_mu, LL_is = LL_is,
                   dataset_name = "theophylline_multiple",
                   save_dir     = file.path(root_dir, "Results"))

save_vae_fit(list(
  Encoder        = Encoder,
  x_dim          = x_dim,        h_dim     = h_dim,
  z_dim          = z_dim,        n_cov     = n_cov,
  nbatch         = nbatch,
  mu0            = mu0,          sigma0    = sigma0,
  iters          = iters,        kl_iter   = kl_iter,
  gamma_iter     = gamma_iter,   iters_burn_in = iters_burn_in,
  z_pop          = z_pop,        omega_pop = omega_pop,
  a              = a,            b         = b,
  names_co       = names_co,
  param_names    = c("ka_pop", "ke_pop", "V_pop"),
  omega_names    = c("omega_ka", "omega_ke", "omega_V"),
  LL_lin_mu      = LL_lin_mu,    LL_is     = LL_is,
  data           = data,         lengths   = lengths,
  C              = C,
  data_in        = data_in,      covariates_in = covariates_in,
  z_pop_iter     = z_pop_iter,   omega_pop_iter = omega_pop_iter,
  Elbo_iter      = Elbo_iter,    a_iter    = a_iter,
  dose           = dose,
  rate = NULL, t_inf = NULL, dose_times = NULL, dose_amts = NULL
), save_dir = file.path(root_dir, "Results"), dataset_name = "theophylline_multiple")

setwd(root_dir)


plotConvergence_pop_theo(Elbo_iter, a_iter, z_pop_iter, omega_pop_iter,
                          iters, kl_iter, gamma_iter, iters_burn_in,
                          save_path = "Plots/theophylline_multiple_convergence_popParam.pdf")
plotConvergence_covariate_theo(z_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in,
                                save_path = "Plots/theophylline_multiple_convergence_covariate.pdf")

# =============================================================================
# VPC + PPC
# =============================================================================
plot_vpc_theo_mult(data, lengths, z_pop, omega_pop, a, b, C, z_dim, dose,
                   n_sim=500L, nbins=10, tald=TRUE,
                   save_path=file.path(root_dir,"Plots","theophylline_multiple_vpc.pdf"))
plot_ppc_theo_mult(data, lengths, Encoder, data_in, covariates_in,
                   a, b, dose, z_dim,
                   n_sim=200L, nbins=10, tald=TRUE,
                   save_path=file.path(root_dir,"Plots","theophylline_multiple_ppc.pdf"))
