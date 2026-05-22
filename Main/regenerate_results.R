# =============================================================================
# Main/regenerate_results.R
#
# Standalone script to regenerate all result tables and plots from saved RDS
# files WITHOUT re-running training.
#
# Prerequisites:
#   - All datasets have been run at least once (*.rds + *_encoder.pt exist
#     in the Results/ folder)
#
# Usage:  source this file, or run from the R console:
#   source("Main/regenerate_results.R")
# =============================================================================

root_dir <- "C:/Users/PASCUGI1/OneDrive - Novartis Pharma AG/Desktop/Projects/VAE_R"
setwd(root_dir)

suppressPackageStartupMessages({
  library(torch)
  library(ggplot2)
  library(patchwork)
})

source("R/encoder.R")
source("R/decoder.R")
source("R/functions.R")
source("R/pop_parameter.R")
source("R/visualization.R")
source("R/functions_theo.R")
source("R/functions_neonates.R")
source("R/functions_warfarin.R")
source("R/functions_pheno.R")
source("R/functions_mavo.R")
source("R/functions_nimo.R")

results_dir <- file.path(root_dir, "Results")
plots_dir   <- file.path(root_dir, "Plots")
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# Helper: regenerate tables + convergence plots + VPC + PPC for one dataset
# =============================================================================

.regen_generic <- function(ds_name, vpc_fn, ppc_fn,
                            vpc_extra = list(), ppc_extra = list(),
                            n_sim_vpc = 500L, n_sim_ppc = 200L, nbins = 10,
                            convergence_type = c("generic", "theo", "neonates"),
                            param_titles = NULL, omega_titles = NULL,
                            cov_titles_fn = NULL) {

  convergence_type <- match.arg(convergence_type)
  cat(sprintf("\n========== %s ==========" , ds_name))
  cat("\n")

  fit <- load_vae_fit(results_dir, ds_name)
  h   <- fit$h

  # -- Tables ------------------------------------------------------------------
  cat("  Regenerating tables...\n")
  if (ds_name %in% c("theophylline", "theophylline_multiple")) {
    printOutput_theo(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                     fit$z_dim, fit$nbatch, fit$lengths$sum(),
                     h, fit$names_co, fit$LL_lin_mu, fit$LL_is)
    saveOutput_theo(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                    fit$z_dim, fit$nbatch, fit$lengths$sum(),
                    h, fit$names_co, fit$LL_lin_mu, fit$LL_is,
                    save_dir = results_dir)
  } else if (ds_name == "neonates") {
    printOutput_neonates(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                         fit$z_dim, fit$nbatch, fit$lengths$sum(),
                         h, fit$names_co, fit$LL_lin_mu, fit$LL_is)
    saveOutput_neonates(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                        fit$z_dim, fit$nbatch, fit$lengths$sum(),
                        h, fit$names_co, fit$LL_lin_mu, fit$LL_is,
                        save_dir = results_dir)
  } else {
    printOutput_generic(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                        fit$z_dim, fit$nbatch, fit$lengths$sum(), h,
                        param_names  = fit$param_names,
                        omega_names  = fit$omega_names,
                        names_co     = fit$names_co,
                        LL_lin_mu    = fit$LL_lin_mu,
                        LL_is        = fit$LL_is)
    saveOutput_results(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                       fit$z_dim, fit$nbatch, fit$lengths$sum(), h,
                       param_names  = fit$param_names,
                       omega_names  = fit$omega_names,
                       names_co     = fit$names_co,
                       LL_lin_mu    = fit$LL_lin_mu,
                       LL_is        = fit$LL_is,
                       dataset_name = ds_name,
                       save_dir     = results_dir)
  }

  # -- Convergence plots -------------------------------------------------------
  cat("  Regenerating convergence plots...\n")
  setwd(root_dir)
  if (convergence_type == "theo") {
    plotConvergence_pop_theo(
      fit$Elbo_iter, fit$a_iter, fit$z_pop_iter, fit$omega_pop_iter,
      fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
      save_path = file.path(plots_dir, paste0(ds_name, "_convergence_popParam.pdf")))
    plotConvergence_covariate_theo(
      fit$z_pop_iter, fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
      save_path = file.path(plots_dir, paste0(ds_name, "_convergence_covariate.pdf")))
  } else if (convergence_type == "neonates") {
    plotConvergence_pop_neonates(
      fit$Elbo_iter, fit$a_iter, fit$z_pop_iter, fit$omega_pop_iter,
      fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
      save_path = file.path(plots_dir, paste0(ds_name, "_convergence_popParam.pdf")))
    plotConvergence_covariate_neonates(
      fit$z_pop_iter, fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
      save_path = file.path(plots_dir, paste0(ds_name, "_convergence_covariate.pdf")))
  } else {
    ct <- if (!is.null(cov_titles_fn)) cov_titles_fn(fit) else
            lapply(fit$names_co, function(x) x)
    plotConvergence_pop_generic(
      fit$Elbo_iter, fit$a_iter, fit$z_pop_iter, fit$omega_pop_iter,
      fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
      param_titles, omega_titles,
      save_path = file.path(plots_dir, paste0(ds_name, "_convergence_popParam.pdf")))
    plotConvergence_covariate_generic(
      fit$z_pop_iter, fit$z_dim, fit$iters, fit$kl_iter, fit$gamma_iter,
      fit$iters_burn_in, ct,
      save_path = file.path(plots_dir, paste0(ds_name, "_convergence_covariate.pdf")))
  }

  # -- VPC ---------------------------------------------------------------------
  cat("  Regenerating VPC...\n")
  vpc_args <- c(
    list(data = fit$data, lengths = fit$lengths,
         z_pop = fit$z_pop, omega_pop = fit$omega_pop,
         a = fit$a, b = fit$b, C = fit$C, z_dim = fit$z_dim,
         n_sim = n_sim_vpc, nbins = nbins,
         save_path = file.path(plots_dir, paste0(ds_name, "_vpc.pdf"))),
    vpc_extra
  )
  do.call(vpc_fn, vpc_args)

  # -- PPC ---------------------------------------------------------------------
  cat("  Regenerating PPC...\n")
  ppc_args <- c(
    list(data = fit$data, lengths = fit$lengths,
         Encoder = fit$Encoder, data_in = fit$data_in,
         covariates_in = fit$covariates_in,
         a = fit$a, b = fit$b, z_dim = fit$z_dim,
         n_sim = n_sim_ppc, nbins = nbins,
         save_path = file.path(plots_dir, paste0(ds_name, "_ppc.pdf"))),
    ppc_extra
  )
  do.call(ppc_fn, ppc_args)

  cat(sprintf("  Done: %s\n", ds_name))
  invisible(fit)
}

# =============================================================================
# 1. Theophylline (single dose)
# =============================================================================
fit_theo <- .regen_generic(
  "theophylline",
  vpc_fn = function(data, lengths, z_pop, omega_pop, a, b, C, z_dim,
                    n_sim, nbins, save_path, dose, h) {
    plot_vpc_theo(data, lengths, z_pop, omega_pop, a, b, h, dose, C, z_dim,
                  n_sim = n_sim, nbins = nbins, save_path = save_path)
  },
  ppc_fn = function(data, lengths, Encoder, data_in, covariates_in,
                    a, b, z_dim, n_sim, nbins, save_path, dose, h) {
    plot_ppc_theo(data, lengths, Encoder, data_in, covariates_in,
                  a, b, dose, h, z_dim,
                  n_sim = n_sim, nbins = nbins, save_path = save_path)
  },
  vpc_extra = list(dose = NULL, h = NULL),  # will be set below
  convergence_type = "theo"
)
# Re-run with h from fit
{
  fit <- fit_theo
  h <- fit$h
  plot_vpc_theo(fit$data, fit$lengths, fit$z_pop, fit$omega_pop, fit$a, fit$b,
                h, fit$dose, fit$C, fit$z_dim, n_sim = 500L, nbins = 10,
                save_path = file.path(plots_dir, "theophylline_vpc.pdf"))
  plot_ppc_theo(fit$data, fit$lengths, fit$Encoder, fit$data_in, fit$covariates_in,
                fit$a, fit$b, fit$dose, h, fit$z_dim, n_sim = 200L, nbins = 10,
                save_path = file.path(plots_dir, "theophylline_ppc.pdf"))
}

# =============================================================================
# 2. Theophylline Multiple Dose
# =============================================================================
{
  ds <- "theophylline_multiple"
  cat(sprintf("\n========== %s ==========" , ds))
  cat("\n")
  fit <- load_vae_fit(results_dir, ds)
  h   <- fit$h
  printOutput_theo(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                   fit$z_dim, fit$nbatch, fit$lengths$sum(),
                   h, fit$names_co, fit$LL_lin_mu, fit$LL_is)
  saveOutput_theo(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                  fit$z_dim, fit$nbatch, fit$lengths$sum(),
                  h, fit$names_co, fit$LL_lin_mu, fit$LL_is,
                  save_dir = results_dir)
  setwd(root_dir)
  plotConvergence_pop_theo(
    fit$Elbo_iter, fit$a_iter, fit$z_pop_iter, fit$omega_pop_iter,
    fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
    save_path = file.path(plots_dir, paste0(ds, "_convergence_popParam.pdf")))
  plotConvergence_covariate_theo(
    fit$z_pop_iter, fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
    save_path = file.path(plots_dir, paste0(ds, "_convergence_covariate.pdf")))
  plot_vpc_theo_mult(fit$data, fit$lengths, fit$z_pop, fit$omega_pop,
                     fit$a, fit$b, fit$C, fit$z_dim, fit$dose,
                     n_sim = 500L, nbins = 10, tald = TRUE,
                     save_path = file.path(plots_dir, paste0(ds, "_vpc.pdf")))
  plot_ppc_theo_mult(fit$data, fit$lengths, fit$Encoder, fit$data_in, fit$covariates_in,
                     fit$a, fit$b, fit$dose, fit$z_dim,
                     n_sim = 200L, nbins = 10, tald = TRUE,
                     save_path = file.path(plots_dir, paste0(ds, "_ppc.pdf")))
  cat(sprintf("  Done: %s\n", ds))
}

# =============================================================================
# 3. Neonates
# =============================================================================
{
  ds <- "neonates"
  cat(sprintf("\n========== %s ==========" , ds))
  cat("\n")
  fit <- load_vae_fit(results_dir, ds)
  h   <- fit$h
  printOutput_neonates(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                       fit$z_dim, fit$nbatch, fit$lengths$sum(),
                       h, fit$names_co, fit$LL_lin_mu, fit$LL_is)
  saveOutput_neonates(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                      fit$z_dim, fit$nbatch, fit$lengths$sum(),
                      h, fit$names_co, fit$LL_lin_mu, fit$LL_is,
                      save_dir = results_dir)
  setwd(root_dir)
  plotConvergence_pop_neonates(
    fit$Elbo_iter, fit$a_iter, fit$z_pop_iter, fit$omega_pop_iter,
    fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
    save_path = file.path(plots_dir, paste0(ds, "_convergence_popParam.pdf")))
  plotConvergence_covariate_neonates(
    fit$z_pop_iter, fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
    save_path = file.path(plots_dir, paste0(ds, "_convergence_covariate.pdf")))
  plot_vpc_neonates(fit$data, fit$lengths, fit$z_pop, fit$omega_pop,
                    fit$a, fit$b, fit$C, fit$z_dim,
                    n_sim = 300L, nbins = 8,
                    save_path = file.path(plots_dir, paste0(ds, "_vpc.pdf")))
  plot_ppc_neonates(fit$data, fit$lengths, fit$Encoder, fit$data_in, fit$covariates_in,
                    fit$a, fit$b, fit$z_dim,
                    n_sim = 100L, nbins = 8,
                    save_path = file.path(plots_dir, paste0(ds, "_ppc.pdf")))
  cat(sprintf("  Done: %s\n", ds))
}

# =============================================================================
# 4. Warfarin
# =============================================================================
{
  ds <- "warfarin"
  cat(sprintf("\n========== %s ==========" , ds))
  cat("\n")
  fit <- load_vae_fit(results_dir, ds)
  h   <- fit$h
  printOutput_generic(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                      fit$z_dim, fit$nbatch, fit$lengths$sum(), h,
                      param_names = fit$param_names, omega_names = fit$omega_names,
                      names_co = fit$names_co, LL_lin_mu = fit$LL_lin_mu, LL_is = fit$LL_is)
  saveOutput_results(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                     fit$z_dim, fit$nbatch, fit$lengths$sum(), h,
                     param_names = fit$param_names, omega_names = fit$omega_names,
                     names_co = fit$names_co, LL_lin_mu = fit$LL_lin_mu, LL_is = fit$LL_is,
                     dataset_name = ds, save_dir = results_dir)
  setwd(root_dir)
  plotConvergence_pop_generic(
    fit$Elbo_iter, fit$a_iter, fit$z_pop_iter, fit$omega_pop_iter,
    fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
    param_titles = list(expression(k[a*","*pop]), expression(k[e*","*pop]), expression(V[pop])),
    omega_titles = list(expression(omega[k[a]]), expression(omega[k[e]]), expression(omega[V])),
    save_path = file.path(plots_dir, paste0(ds, "_convergence_popParam.pdf")))
  plotConvergence_covariate_generic(
    fit$z_pop_iter, fit$z_dim, fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
    cov_titles = lapply(fit$names_co, function(x) x),
    save_path = file.path(plots_dir, paste0(ds, "_convergence_covariate.pdf")))
  plot_vpc_warfarin(fit$data, fit$lengths, fit$z_pop, fit$omega_pop,
                    fit$a, fit$b, fit$C, fit$z_dim, fit$dose,
                    n_sim = 500L, nbins = 10,
                    save_path = file.path(plots_dir, paste0(ds, "_vpc.pdf")))
  plot_ppc_warfarin(fit$data, fit$lengths, fit$Encoder, fit$data_in, fit$covariates_in,
                    fit$a, fit$b, fit$dose, fit$z_dim,
                    n_sim = 200L, nbins = 10,
                    save_path = file.path(plots_dir, paste0(ds, "_ppc.pdf")))
  cat(sprintf("  Done: %s\n", ds))
}

# =============================================================================
# 5. Pheno SD
# =============================================================================
{
  ds <- "pheno_sd"
  cat(sprintf("\n========== %s ==========" , ds))
  cat("\n")
  fit <- load_vae_fit(results_dir, ds)
  h   <- fit$h
  printOutput_generic(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                      fit$z_dim, fit$nbatch, fit$lengths$sum(), h,
                      param_names = fit$param_names, omega_names = fit$omega_names,
                      names_co = fit$names_co, LL_lin_mu = fit$LL_lin_mu, LL_is = fit$LL_is)
  saveOutput_results(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                     fit$z_dim, fit$nbatch, fit$lengths$sum(), h,
                     param_names = fit$param_names, omega_names = fit$omega_names,
                     names_co = fit$names_co, LL_lin_mu = fit$LL_lin_mu, LL_is = fit$LL_is,
                     dataset_name = ds, save_dir = results_dir)
  setwd(root_dir)
  plotConvergence_pop_generic(
    fit$Elbo_iter, fit$a_iter, fit$z_pop_iter, fit$omega_pop_iter,
    fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
    param_titles = list(expression(CL[pop]), expression(V[pop])),
    omega_titles = list(expression(omega[CL]), expression(omega[V])),
    save_path = file.path(plots_dir, paste0(ds, "_convergence_popParam.pdf")))
  plotConvergence_covariate_generic(
    fit$z_pop_iter, fit$z_dim, fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
    cov_titles = lapply(fit$names_co, function(x) x),
    save_path = file.path(plots_dir, paste0(ds, "_convergence_covariate.pdf")))
  plot_vpc_pheno(fit$data, fit$lengths, fit$z_pop, fit$omega_pop,
                 fit$a, fit$b, fit$C, fit$z_dim, fit$dose_times, fit$dose_amts,
                 n_sim = 500L, nbins = 8, tald = TRUE,
                 save_path = file.path(plots_dir, paste0(ds, "_vpc.pdf")))
  plot_ppc_pheno(fit$data, fit$lengths, fit$Encoder, fit$data_in, fit$covariates_in,
                 fit$a, fit$b, fit$dose_times, fit$dose_amts, fit$z_dim,
                 n_sim = 200L, nbins = 8, tald = TRUE,
                 save_path = file.path(plots_dir, paste0(ds, "_ppc.pdf")))
  cat(sprintf("  Done: %s\n", ds))
}

# =============================================================================
# 6. Mavoglurant
# =============================================================================
{
  ds <- "mavoglurant"
  cat(sprintf("\n========== %s ==========" , ds))
  cat("\n")
  fit <- load_vae_fit(results_dir, ds)
  h   <- fit$h
  printOutput_generic(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                      fit$z_dim, fit$nbatch, fit$lengths$sum(), h,
                      param_names = fit$param_names, omega_names = fit$omega_names,
                      names_co = fit$names_co, LL_lin_mu = fit$LL_lin_mu, LL_is = fit$LL_is)
  saveOutput_results(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                     fit$z_dim, fit$nbatch, fit$lengths$sum(), h,
                     param_names = fit$param_names, omega_names = fit$omega_names,
                     names_co = fit$names_co, LL_lin_mu = fit$LL_lin_mu, LL_is = fit$LL_is,
                     dataset_name = ds, save_dir = results_dir)
  setwd(root_dir)
  plotConvergence_pop_generic(
    fit$Elbo_iter, fit$a_iter, fit$z_pop_iter, fit$omega_pop_iter,
    fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
    param_titles = list(expression(CL[pop]), expression(V[1*","*pop]),
                        expression(Q[pop]),  expression(V[2*","*pop])),
    omega_titles = list(expression(omega[CL]), expression(omega[V[1]]),
                        expression(omega[Q]),  expression(omega[V[2]])),
    save_path = file.path(plots_dir, paste0(ds, "_convergence_popParam.pdf")))
  plotConvergence_covariate_generic(
    fit$z_pop_iter, fit$z_dim, fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
    cov_titles = lapply(fit$names_co, function(x) x),
    save_path = file.path(plots_dir, paste0(ds, "_convergence_covariate.pdf")))
  plot_vpc_mavo(fit$data, fit$lengths, fit$z_pop, fit$omega_pop,
                fit$a, fit$b, fit$C, fit$z_dim, fit$rate, fit$t_inf,
                n_sim = 500L, nbins = 10,
                save_path = file.path(plots_dir, paste0(ds, "_vpc.pdf")))
  plot_ppc_mavo(fit$data, fit$lengths, fit$Encoder, fit$data_in, fit$covariates_in,
                fit$a, fit$b, fit$rate, fit$t_inf, fit$z_dim,
                n_sim = 200L, nbins = 10,
                save_path = file.path(plots_dir, paste0(ds, "_ppc.pdf")))
  cat(sprintf("  Done: %s\n", ds))
}

# =============================================================================
# 7. NimoData
# =============================================================================
{
  ds <- "nimoData"
  cat(sprintf("\n========== %s ==========" , ds))
  cat("\n")
  fit <- load_vae_fit(results_dir, ds)
  h   <- fit$h
  printOutput_generic(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                      fit$z_dim, fit$nbatch, fit$lengths$sum(), h,
                      param_names = fit$param_names, omega_names = fit$omega_names,
                      names_co = fit$names_co, LL_lin_mu = fit$LL_lin_mu, LL_is = fit$LL_is)
  saveOutput_results(fit$z_pop, fit$omega_pop, fit$a, fit$b,
                     fit$z_dim, fit$nbatch, fit$lengths$sum(), h,
                     param_names = fit$param_names, omega_names = fit$omega_names,
                     names_co = fit$names_co, LL_lin_mu = fit$LL_lin_mu, LL_is = fit$LL_is,
                     dataset_name = ds, save_dir = results_dir)
  setwd(root_dir)
  plotConvergence_pop_generic(
    fit$Elbo_iter, fit$a_iter, fit$z_pop_iter, fit$omega_pop_iter,
    fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
    param_titles = list(expression(CL[pop]), expression(V[pop])),
    omega_titles = list(expression(omega[CL]), expression(omega[V])),
    save_path = file.path(plots_dir, paste0(ds, "_convergence_popParam.pdf")))
  plotConvergence_covariate_generic(
    fit$z_pop_iter, fit$z_dim, fit$iters, fit$kl_iter, fit$gamma_iter, fit$iters_burn_in,
    cov_titles = lapply(fit$names_co, function(x) x),
    save_path = file.path(plots_dir, paste0(ds, "_convergence_covariate.pdf")))
  plot_vpc_nimo(fit$data, fit$lengths, fit$z_pop, fit$omega_pop,
                fit$a, fit$b, fit$C, fit$z_dim, fit$rate, fit$t_inf,
                n_sim = 500L, nbins = 10,
                save_path = file.path(plots_dir, paste0(ds, "_vpc.pdf")))
  plot_ppc_nimo(fit$data, fit$lengths, fit$Encoder, fit$data_in, fit$covariates_in,
                fit$a, fit$b, fit$rate, fit$t_inf, fit$z_dim,
                n_sim = 200L, nbins = 10,
                save_path = file.path(plots_dir, paste0(ds, "_ppc.pdf")))
  cat(sprintf("  Done: %s\n", ds))
}

cat("\n=== All datasets regenerated successfully ===\n")
