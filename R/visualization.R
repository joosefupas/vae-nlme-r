# =============================================================================
# visualization.R  —  Convergence plots and result summaries
# R translation of visualization.py (Jan Rohleff, CPT:PSP 2025)
# Uses ggplot2 + patchwork
# =============================================================================
library(ggplot2)
library(patchwork)

# Helper to extract a numeric vector from a torch tensor
.t2v <- function(x) as.numeric(x$detach())

# Shared theme for convergence plots
.vae_theme <- function() {
  theme_bw(base_size = 11) +
    theme(
      plot.title    = element_text(size = 12, face = "bold", hjust = 0.5),
      axis.title.x  = element_blank(),
      panel.grid.minor = element_blank()
    )
}

# Helper to save a ggplot as PDF + JPG using explicit device calls.
# Uses normalizePath() for absolute path (avoids grDevices::pdf relative-path
# issues on Windows OneDrive paths with spaces).
.save_plot <- function(p, save_path, width, height) {
  abs_path <- normalizePath(save_path, winslash = "/", mustWork = FALSE)
  dir.create(dirname(abs_path), recursive = TRUE, showWarnings = FALSE)

  .write_device <- function(path, open_fn) {
    opened <- tryCatch({ open_fn(path); TRUE },
                       error = function(e) {
                         message("Save failed [", path, "]: ", conditionMessage(e))
                         FALSE
                       })
    if (!opened) return(invisible(NULL))
    tryCatch(print(p), error = function(e) NULL)
    grDevices::dev.off()
    message("Saved: ", path)
  }

  # PDF
  .write_device(abs_path,
    function(f) grDevices::pdf(file = f, width = width, height = height))
  # JPG
  jpg_path <- sub("\\.pdf$", ".jpg", abs_path, ignore.case = TRUE)
  .write_device(jpg_path,
    function(f) grDevices::jpeg(filename = f,
                                width  = round(width  * 150),
                                height = round(height * 150),
                                res = 150, quality = 95))
}

# Add phase annotations to a ggplot (burn-in shading + phase lines)
.add_phases <- function(p, iters_burn_in, kl_iter, gamma_iter, iters) {
  x_bi  <- iters_burn_in
  x_kl  <- iters_burn_in + kl_iter
  x_gam <- iters_burn_in + gamma_iter
  x_end <- iters_burn_in + iters

  p +
    annotate("rect", xmin = 0, xmax = x_bi, ymin = -Inf, ymax = Inf,
             fill = "grey70", alpha = 0.25) +
    geom_vline(xintercept = x_kl,  linetype = "dashed", color = "darkgreen") +
    geom_vline(xintercept = x_gam, linetype = "dashed", color = "red") +
    coord_cartesian(xlim = c(0, x_end))
}

# =============================================================================
# Case Study 1 — Theophylline
# =============================================================================

printOutput_theo <- function(z_pop, omega_pop, a, b, z_dim, nbatch,
                             n_tot, h, names_co, LL_lin, LL_is) {
  zp   <- .t2v(h(z_pop[1:z_dim]))
  om   <- .t2v(omega_pop$sqrt())
  ln_N <- log(nbatch)
  ln_n <- log(as.numeric(n_tot$detach()))

  cat("\n#############################################\n")
  cat("ESTIMATION OF THE POPULATION PARAMETERS\n")
  cat("#############################################\n\n")
  cat("Fixed Effects:\n")
  cat(sprintf("  %-15s %10.4f\n", "ka_pop:", zp[1]))
  cat(sprintf("  %-15s %10.4f\n", "ke_pop:", zp[2]))
  cat(sprintf("  %-15s %10.4f\n", "V_pop:",  zp[3]))

  z_pop_v <- .t2v(z_pop)
  count   <- 0L
  for (k in seq(z_dim + 1, length(z_pop_v))) {
    if (z_pop_v[k] != 0) {
      cat(sprintf("  %-15s %10.4f\n", paste0(names_co[k - z_dim], ":"), z_pop_v[k]))
      count <- count + 1L
    }
  }

  cat("\nStandard Deviations of Random Effects:\n")
  cat(sprintf("  %-15s %10.4f\n", "omega_ka:", om[1]))
  cat(sprintf("  %-15s %10.4f\n", "omega_ke:", om[2]))
  cat(sprintf("  %-15s %10.4f\n", "omega_V:",  om[3]))

  a_v <- as.numeric(a$detach())
  b_v <- as.numeric(b$detach())
  cat("\nError Model Parameters:\n")
  if (a_v != 0) cat(sprintf("  %-15s %10.4f\n", "a:", a_v))
  if (b_v != 0) cat(sprintf("  %-15s %10.4f\n", "b:", b_v))

  LL_lin_v <- as.numeric(LL_lin$detach())
  LL_is_v  <- as.numeric(LL_is$detach())
  p_est    <- 2 * z_dim + 1 + count

  cat("\n#########################################################\n")
  cat("LOG-LIKELIHOOD AND INFORMATION CRITERIA\n")
  cat("#########################################################\n")
  cat(sprintf("  %-45s %20s  %20s\n", "", "Linearisation:", "Importance Sampling:"))
  cat(strrep("-", 90), "\n")
  cat(sprintf("  %-45s %20.2f  %20.2f\n", "-2 log-lik (OFV):", 2*LL_lin_v, 2*LL_is_v))
  cat(sprintf("  %-45s %20.2f  %20.2f\n", "AIC:",
              2*LL_lin_v + 2*p_est, 2*LL_is_v + 2*p_est))
  cat(sprintf("  %-45s %20.2f  %20.2f\n", "BIC:",
              2*LL_lin_v + ln_N*p_est, 2*LL_is_v + ln_N*p_est))
  cat(sprintf("  %-45s %20.2f  %20.2f\n", "BICc:",
              2*LL_lin_v + ln_N*(z_dim+count) + ln_n*(z_dim+1),
              2*LL_is_v  + ln_N*(z_dim+count) + ln_n*(z_dim+1)))
}

plotConvergence_pop_theo <- function(elbo_iter, a_iter, z_pop_iter,
                                     omega_pop_iter, iters, kl_iter,
                                     gamma_iter, iters_burn_in,
                                     save_path = "Plots/theophylline_convergence_popParam.pdf") {
  n_total <- iters_burn_in + iters
  x_axis  <- seq_len(n_total)

  elbo_v  <- c(rep(NA_real_, iters_burn_in), .t2v(elbo_iter))
  a_v     <- .t2v(a_iter)
  zpop    <- as.matrix(.t2v(z_pop_iter))  # will be N × (z_dim+M)
  # z_pop_iter: rows = iterations (incl. burn-in), cols = z_dim + M
  zpop_mat  <- matrix(.t2v(z_pop_iter), ncol = dim(z_pop_iter)[2])
  omega_mat <- matrix(.t2v(omega_pop_iter), ncol = dim(omega_pop_iter)[2])

  make_panel <- function(y, title) {
    df <- data.frame(x = x_axis, y = y[seq_len(n_total)])
    p  <- ggplot(df, aes(x, y)) + geom_line(colour = "steelblue", linewidth = 0.6) +
      ggtitle(title) + ylab("") + .vae_theme()
    .add_phases(p, iters_burn_in, kl_iter, gamma_iter, iters)
  }

  panels <- list(
    make_panel(zpop_mat[, 1],  expression(k[a*","*pop])),
    make_panel(zpop_mat[, 2],  expression(k[e*","*pop])),
    make_panel(zpop_mat[, 3],  expression(V[pop])),
    make_panel(omega_mat[, 1], expression(omega[k[a]])),
    make_panel(omega_mat[, 2], expression(omega[k[e]])),
    make_panel(omega_mat[, 3], expression(omega[V])),
    make_panel(a_v,            expression(a)),
    make_panel(elbo_v,         expression(-italic(L)[psi](x)))
  )

  p_out <- wrap_plots(panels, ncol = 3)
  .save_plot(p_out, save_path, width = 10, height = 6)
}

plotConvergence_covariate_theo <- function(z_pop_iter, iters, kl_iter,
                                            gamma_iter, iters_burn_in,
                                            save_path = "Plots/theophylline_convergence_covariate.pdf") {
  zpop_mat <- matrix(.t2v(z_pop_iter), ncol = dim(z_pop_iter)[2])
  x_axis   <- seq_len(iters)
  # z_pop_iter contains burn-in rows prepended to main-loop rows (total = iters_burn_in + iters).
  # Always plot the last `iters` rows (main-loop data only).
  zpop_sub <- zpop_mat[tail(seq_len(nrow(zpop_mat)), iters), , drop = FALSE]

  # Columns 4,6,8 = weight effects; 5,7,9 = sex effects (0-indexed in Python → +1 in R)
  panels_info <- list(
    list(col = 4,  title = expression(beta[k[a]]^w)),
    list(col = 6,  title = expression(beta[k[e]]^w)),
    list(col = 8,  title = expression(beta[V]^w)),
    list(col = 5,  title = expression(beta[k[a]]^sex)),
    list(col = 7,  title = expression(beta[k[e]]^sex)),
    list(col = 9,  title = expression(beta[V]^sex))
  )

  make_panel <- function(y, title) {
    df <- data.frame(x = x_axis, y = y[seq_len(iters)])
    p  <- ggplot(df, aes(x, y)) + geom_line(colour = "steelblue", linewidth = 0.6) +
      ggtitle(title) + ylab("") + .vae_theme()
    if (tail(y[!is.na(y)], 1) == 0) {
      p <- p + annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
                        fill = "grey80", alpha = 0.4)
    }
    p + geom_vline(xintercept = kl_iter,   linetype = "dashed", color = "darkgreen") +
        geom_vline(xintercept = gamma_iter, linetype = "dashed", color = "red") +
        coord_cartesian(xlim = c(0, iters))
  }

  panels <- lapply(panels_info, function(pi) {
    make_panel(zpop_sub[, pi$col], pi$title)
  })

  p_out <- wrap_plots(panels, ncol = 3)
  .save_plot(p_out, save_path, width = 10, height = 4)
}

# =============================================================================
# Case Study 2 — Neonates
# =============================================================================

printOutput_neonates <- function(z_pop, omega_pop, a, b, z_dim, nbatch,
                                  n_tot, h, names_co, LL_lin_mu, LL_is) {
  zp  <- .t2v(h(z_pop[1:z_dim]))
  om  <- .t2v(omega_pop$sqrt())
  ln_N <- log(nbatch)
  ln_n <- log(as.numeric(n_tot$detach()))

  cat("\n#############################################\n")
  cat("ESTIMATION OF THE POPULATION PARAMETERS\n")
  cat("#############################################\n\n")
  param_names <- c("W0_pop:", "kin_pop:", "Tlag_pop:", "koutmax_pop:", "T50_pop:")
  for (k in seq_len(z_dim)) cat(sprintf("  %-15s %10.4f\n", param_names[k], zp[k]))

  z_pop_v <- .t2v(z_pop)
  count   <- 0L
  for (k in seq(z_dim + 1, length(z_pop_v))) {
    if (z_pop_v[k] != 0) {
      cat(sprintf("  %-15s %10.4f\n", paste0(names_co[k - z_dim], ":"), z_pop_v[k]))
      count <- count + 1L
    }
  }

  cat("\nStandard Deviations of Random Effects:\n")
  om_names <- c("omega_W0:", "omega_kin:", "omega_Tlag:", "omega_koutmax:", "omega_T50:")
  for (k in seq_len(z_dim)) cat(sprintf("  %-15s %10.4f\n", om_names[k], om[k]))

  a_v <- as.numeric(a$detach())
  b_v <- as.numeric(b$detach())
  cat("\nError Model Parameters:\n")
  if (a_v != 0) cat(sprintf("  %-15s %10.4f\n", "a:", a_v))
  if (b_v != 0) cat(sprintf("  %-15s %10.4f\n", "b:", b_v))

  LL_lin_v <- as.numeric(LL_lin_mu$detach())
  LL_is_v  <- as.numeric(LL_is$detach())
  p_est    <- 2 * z_dim + 1 + count

  cat("\n#########################################################\n")
  cat("LOG-LIKELIHOOD AND INFORMATION CRITERIA\n")
  cat("#########################################################\n")
  cat(sprintf("  %-45s %20s  %20s\n", "", "Linearisation:", "Importance Sampling:"))
  cat(strrep("-", 90), "\n")
  cat(sprintf("  %-45s %20.2f  %20.2f\n", "-2 log-lik (OFV):", 2*LL_lin_v, 2*LL_is_v))
  cat(sprintf("  %-45s %20.2f  %20.2f\n", "AIC:",
              2*LL_lin_v + 2*p_est, 2*LL_is_v + 2*p_est))
  cat(sprintf("  %-45s %20.2f  %20.2f\n", "BIC:",
              2*LL_lin_v + ln_N*p_est, 2*LL_is_v + ln_N*p_est))
  cat(sprintf("  %-45s %20.2f  %20.2f\n", "BICc:",
              2*LL_lin_v + ln_N*(z_dim+count) + ln_n*(z_dim+1),
              2*LL_is_v  + ln_N*(z_dim+count) + ln_n*(z_dim+1)))
}

plotConvergence_pop_neonates <- function(elbo_iter, a_iter, z_pop_iter,
                                          omega_pop_iter, iters, kl_iter,
                                          gamma_iter, iters_burn_in,
                                          save_path = "Plots/neonates_convergence_popParam.pdf") {
  n_total  <- iters_burn_in + iters
  x_axis   <- seq_len(n_total)
  elbo_v   <- c(rep(NA_real_, iters_burn_in), .t2v(elbo_iter))
  a_v      <- .t2v(a_iter)
  zpop_mat  <- matrix(.t2v(z_pop_iter),  ncol = dim(z_pop_iter)[2])
  omega_mat <- matrix(.t2v(omega_pop_iter), ncol = dim(omega_pop_iter)[2])

  make_panel <- function(y, title) {
    df <- data.frame(x = x_axis, y = y[seq_len(n_total)])
    p  <- ggplot(df, aes(x, y)) + geom_line(colour = "steelblue", linewidth = 0.6) +
      ggtitle(title) + ylab("") + .vae_theme()
    .add_phases(p, iters_burn_in, kl_iter, gamma_iter, iters)
  }

  param_titles <- list(expression(W[0*","*pop]), expression(k["in"*","*pop]),
                       expression(T[lag*","*pop]), expression(k[out*","*pop]),
                       expression(T[50*","*pop]))
  omega_titles <- list(expression(omega[W[0]]), expression(omega[k["in"]]),
                       expression(omega[T[lag]]), expression(omega[k[out]]),
                       expression(omega[T[50]]))

  panels <- c(
    lapply(seq_len(5), function(k) make_panel(zpop_mat[, k],  param_titles[[k]])),
    lapply(seq_len(5), function(k) make_panel(omega_mat[, k], omega_titles[[k]])),
    list(make_panel(a_v,    expression(a)),
         make_panel(elbo_v, expression(-italic(L)[psi](x))))
  )

  p_out <- wrap_plots(panels, ncol = 4)
  .save_plot(p_out, save_path, width = 12, height = 8)
}

# =============================================================================
# Generic functions — reusable for any dataset
# =============================================================================

# -----------------------------------------------------------------------------
# printOutput_generic()
#   param_names  : character vector, length z_dim  (fixed-effect labels)
#   omega_names  : character vector, length z_dim  (random-effect labels)
# -----------------------------------------------------------------------------
printOutput_generic <- function(z_pop, omega_pop, a, b, z_dim, nbatch,
                                 n_tot, h, param_names, omega_names,
                                 names_co, LL_lin_mu, LL_is) {
  zp   <- .t2v(h(z_pop[1:z_dim]))
  om   <- .t2v(omega_pop$sqrt())
  ln_N <- log(nbatch)
  ln_n <- log(as.numeric(n_tot$detach()))

  cat("\n#############################################\n")
  cat("ESTIMATION OF THE POPULATION PARAMETERS\n")
  cat("#############################################\n\n")
  cat("Fixed Effects:\n")
  for (k in seq_len(z_dim))
    cat(sprintf("  %-20s %10.4f\n", paste0(param_names[k], ":"), zp[k]))

  z_pop_v <- .t2v(z_pop)
  count   <- 0L
  for (k in seq(z_dim + 1, length(z_pop_v))) {
    if (z_pop_v[k] != 0) {
      cat(sprintf("  %-20s %10.4f\n", paste0(names_co[k - z_dim], ":"), z_pop_v[k]))
      count <- count + 1L
    }
  }

  cat("\nStandard Deviations of Random Effects:\n")
  for (k in seq_len(z_dim))
    cat(sprintf("  %-20s %10.4f\n", paste0(omega_names[k], ":"), om[k]))

  a_v <- as.numeric(a$detach())
  b_v <- as.numeric(b$detach())
  cat("\nError Model Parameters:\n")
  if (a_v != 0) cat(sprintf("  %-20s %10.4f\n", "a:", a_v))
  if (b_v != 0) cat(sprintf("  %-20s %10.4f\n", "b:", b_v))

  LL_lin_v <- as.numeric(LL_lin_mu$detach())
  LL_is_v  <- as.numeric(LL_is$detach())
  p_est    <- 2 * z_dim + 1 + count

  cat("\n#########################################################\n")
  cat("LOG-LIKELIHOOD AND INFORMATION CRITERIA\n")
  cat("#########################################################\n")
  cat(sprintf("  %-45s %20s  %20s\n", "", "Linearisation:", "Importance Sampling:"))
  cat(strrep("-", 90), "\n")
  cat(sprintf("  %-45s %20.2f  %20.2f\n", "-2 log-lik (OFV):", 2*LL_lin_v, 2*LL_is_v))
  cat(sprintf("  %-45s %20.2f  %20.2f\n", "AIC:",
              2*LL_lin_v + 2*p_est, 2*LL_is_v + 2*p_est))
  cat(sprintf("  %-45s %20.2f  %20.2f\n", "BIC:",
              2*LL_lin_v + ln_N*p_est, 2*LL_is_v + ln_N*p_est))
  cat(sprintf("  %-45s %20.2f  %20.2f\n", "BICc:",
              2*LL_lin_v + ln_N*(z_dim+count) + ln_n*(z_dim+1),
              2*LL_is_v  + ln_N*(z_dim+count) + ln_n*(z_dim+1)))
}

# -----------------------------------------------------------------------------
# plotConvergence_pop_generic()
#   param_titles : list of plot titles (expression or character) for z_dim params
#   omega_titles : list of plot titles for z_dim omegas
# -----------------------------------------------------------------------------
plotConvergence_pop_generic <- function(elbo_iter, a_iter, z_pop_iter,
                                         omega_pop_iter, iters, kl_iter,
                                         gamma_iter, iters_burn_in,
                                         param_titles, omega_titles,
                                         save_path = "Plots/convergence_popParam.pdf") {
  n_total  <- iters_burn_in + iters
  z_dim    <- length(param_titles)
  x_axis   <- seq_len(n_total)
  elbo_v   <- c(rep(NA_real_, iters_burn_in), .t2v(elbo_iter))
  a_v      <- .t2v(a_iter)
  zpop_mat  <- matrix(.t2v(z_pop_iter),  ncol = dim(z_pop_iter)[2])
  omega_mat <- matrix(.t2v(omega_pop_iter), ncol = dim(omega_pop_iter)[2])

  make_panel <- function(y, title) {
    df <- data.frame(x = x_axis, y = y[seq_len(n_total)])
    p  <- ggplot(df, aes(x, y)) + geom_line(colour = "steelblue", linewidth = 0.6) +
      ggtitle(title) + ylab("") + .vae_theme()
    .add_phases(p, iters_burn_in, kl_iter, gamma_iter, iters)
  }

  panels <- c(
    lapply(seq_len(z_dim), function(k) make_panel(zpop_mat[, k],  param_titles[[k]])),
    lapply(seq_len(z_dim), function(k) make_panel(omega_mat[, k], omega_titles[[k]])),
    list(make_panel(a_v,    expression(a)),
         make_panel(elbo_v, expression(-italic(L)[psi](x))))
  )

  ncols <- min(4L, ceiling(sqrt(length(panels))))
  p_out <- wrap_plots(panels, ncol = ncols)
  ht    <- ceiling(length(panels) / ncols) * 2.2
  .save_plot(p_out, save_path, width = ncols * 2.8, height = ht)
}

# -----------------------------------------------------------------------------
# plotConvergence_covariate_generic()
#   cov_titles : list of expression/character, length M = z_dim * n_cov
# -----------------------------------------------------------------------------
plotConvergence_covariate_generic <- function(z_pop_iter, z_dim, iters, kl_iter,
                                               gamma_iter, iters_burn_in,
                                               cov_titles,
                                               save_path = "Plots/convergence_covariate.pdf") {
  zpop_mat <- matrix(.t2v(z_pop_iter), ncol = dim(z_pop_iter)[2])
  x_axis   <- seq_len(iters)
  zpop_sub <- zpop_mat[(iters_burn_in + 1):nrow(zpop_mat), , drop = FALSE]
  M        <- ncol(zpop_sub) - z_dim

  make_panel <- function(y, title) {
    df <- data.frame(x = x_axis, y = y[seq_len(iters)])
    p  <- ggplot(df, aes(x, y)) + geom_line(colour = "steelblue", linewidth = 0.5) +
      ggtitle(title) + ylab("") + .vae_theme()
    if (tail(y[!is.na(y)], 1) == 0) {
      p <- p + annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
                        fill = "grey80", alpha = 0.4)
    }
    p + geom_vline(xintercept = kl_iter,   linetype = "dashed", color = "darkgreen") +
        geom_vline(xintercept = gamma_iter, linetype = "dashed", color = "red") +
        coord_cartesian(xlim = c(0, iters))
  }

  panels <- lapply(seq_len(M), function(k) {
    make_panel(zpop_sub[, k + z_dim], cov_titles[[k]])
  })

  n_cov  <- M / z_dim
  ncols  <- min(n_cov, 5L)
  nrows  <- ceiling(M / ncols)
  p_out  <- wrap_plots(panels, ncol = ncols)
  .save_plot(p_out, save_path, width = ncols * 2.8, height = nrows * 2.2)
}

plotConvergence_covariate_neonates <- function(z_pop_iter, iters, kl_iter,
                                                gamma_iter, iters_burn_in,
                                                save_path = "Plots/neonates_convergence_covariate.pdf") {
  zpop_mat <- matrix(.t2v(z_pop_iter), ncol = dim(z_pop_iter)[2])
  x_axis   <- seq_len(iters)
  zpop_sub <- zpop_mat[(iters_burn_in + 1):nrow(zpop_mat), , drop = FALSE]

  # Columns 6:30 are the 25 covariate effects (z_dim=5, n_cov=5)
  cov_names <- c(
    expression(beta[W[0]]^sex),   expression(beta[W[0]]^DelM),
    expression(beta[W[0]]^GA),    expression(beta[W[0]]^Mage),
    expression(beta[W[0]]^Para2),
    expression(beta[k["in"]]^sex),  expression(beta[k["in"]]^DelM),
    expression(beta[k["in"]]^GA),   expression(beta[k["in"]]^Mage),
    expression(beta[k["in"]]^Para2),
    expression(beta[T[lag]]^sex), expression(beta[T[lag]]^DelM),
    expression(beta[T[lag]]^GA),  expression(beta[T[lag]]^Mage),
    expression(beta[T[lag]]^Para2),
    expression(beta[k[out]]^sex), expression(beta[k[out]]^DelM),
    expression(beta[k[out]]^GA),  expression(beta[k[out]]^Mage),
    expression(beta[k[out]]^Para2),
    expression(beta[T[50]]^sex),  expression(beta[T[50]]^DelM),
    expression(beta[T[50]]^GA),   expression(beta[T[50]]^Mage),
    expression(beta[T[50]]^Para2)
  )

  make_panel <- function(y, title) {
    df <- data.frame(x = x_axis, y = y[seq_len(iters)])
    p  <- ggplot(df, aes(x, y)) + geom_line(colour = "steelblue", linewidth = 0.5) +
      ggtitle(title) + ylab("") + .vae_theme()
    if (tail(y[!is.na(y)], 1) == 0) {
      p <- p + annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
                        fill = "grey80", alpha = 0.4)
    }
    p + geom_vline(xintercept = kl_iter,   linetype = "dashed", color = "darkgreen") +
        geom_vline(xintercept = gamma_iter, linetype = "dashed", color = "red") +
        coord_cartesian(xlim = c(0, iters))
  }

  panels <- lapply(seq_len(25), function(k) {
    make_panel(zpop_sub[, k + 5], cov_names[[k]])
  })

  p_out <- wrap_plots(panels, ncol = 5)
  .save_plot(p_out, save_path, width = 18, height = 10)
}

# =============================================================================
# saveOutput_results()  — Save estimation results to CSV, LaTeX, and Markdown
# =============================================================================
#
# Arguments mirror printOutput_generic() plus:
#   dataset_name : character  e.g. "theophylline" — used as file prefix & caption
#   save_dir     : directory for output files (created if absent)
#
# Creates three files in save_dir/:
#   {dataset_name}_results.csv   — flat table (one row per param + IC rows)
#   {dataset_name}_results.tex   — LaTeX booktabs table (\usepackage{booktabs})
#   {dataset_name}_results.md    — GitHub-Flavoured Markdown pipe table
# =============================================================================
saveOutput_results <- function(z_pop, omega_pop, a, b, z_dim, nbatch,
                                n_tot, h,
                                param_names, omega_names, names_co,
                                LL_lin_mu, LL_is,
                                dataset_name = "model",
                                save_dir     = "Results") {

  # ---- extract numerics -------------------------------------------------------
  zp       <- .t2v(h(z_pop[1:z_dim]))
  om       <- .t2v(omega_pop$sqrt())
  z_pop_v  <- .t2v(z_pop)
  a_v      <- as.numeric(a$detach())
  b_v      <- as.numeric(b$detach())
  LL_lin_v <- as.numeric(LL_lin_mu$detach())
  LL_is_v  <- as.numeric(LL_is$detach())
  ln_N     <- log(nbatch)
  ln_n     <- log(as.numeric(n_tot$detach()))

  # selected covariate effects (non-zero entries beyond z_dim)
  cov_names_sel <- character(0)
  cov_vals_sel  <- numeric(0)
  if (length(z_pop_v) > z_dim) {
    for (k in seq(z_dim + 1L, length(z_pop_v))) {
      if (z_pop_v[k] != 0) {
        cov_names_sel <- c(cov_names_sel, names_co[k - z_dim])
        cov_vals_sel  <- c(cov_vals_sel,  z_pop_v[k])
      }
    }
  }
  count <- length(cov_names_sel)
  p_est <- 2 * z_dim + 1 + count

  OFV_lin  <- 2 * LL_lin_v;   OFV_is   <- 2 * LL_is_v
  AIC_lin  <- OFV_lin + 2 * p_est;     AIC_is  <- OFV_is + 2 * p_est
  BIC_lin  <- OFV_lin + ln_N * p_est;  BIC_is  <- OFV_is + ln_N * p_est
  BICc_lin <- OFV_lin + ln_N*(z_dim+count) + ln_n*(z_dim+1)
  BICc_is  <- OFV_is  + ln_N*(z_dim+count) + ln_n*(z_dim+1)

  ic_list <- list(c("OFV",  OFV_lin,  OFV_is),  c("AIC",  AIC_lin,  AIC_is),
                  c("BIC",  BIC_lin,  BIC_is),   c("BICc", BICc_lin, BICc_is))

  abs_dir <- normalizePath(save_dir, winslash = "/", mustWork = FALSE)
  dir.create(abs_dir, recursive = TRUE, showWarnings = FALSE)

  # ===========================================================================
  # 1. CSV
  # ===========================================================================
  rows <- list()
  add_row <- function(section, param, est_lin, est_is = NA_real_)
    rows[[length(rows) + 1L]] <<- data.frame(
      Section = section, Parameter = param,
      Estimate_Lin = est_lin, Estimate_IS = est_is,
      stringsAsFactors = FALSE)

  for (k in seq_len(z_dim))
    add_row("Fixed Effects", param_names[k], round(zp[k], 4))
  for (k in seq_along(cov_names_sel))
    add_row("Covariate Effects", cov_names_sel[k], round(cov_vals_sel[k], 4))
  for (k in seq_len(z_dim))
    add_row("Random Effects (SD)", omega_names[k], round(om[k], 4))
  if (a_v != 0) add_row("Error Model", "a", round(a_v, 4))
  if (b_v != 0) add_row("Error Model", "b", round(b_v, 4))
  for (ic in ic_list)
    add_row("IC", ic[1], round(as.numeric(ic[2]), 2), round(as.numeric(ic[3]), 2))

  df       <- do.call(rbind, rows)
  csv_path <- file.path(abs_dir, paste0(dataset_name, "_results.csv"))
  write.csv(df, csv_path, row.names = FALSE)
  message("Saved: ", csv_path)

  # ===========================================================================
  # 2. LaTeX  (requires \usepackage{booktabs})
  # NOTE: use \_ (not \\_) inside \texttt{} for a literal underscore
  # ===========================================================================
  .esc_tex <- function(x) gsub("_", "\\_", x, fixed = TRUE)  # → \_ in file

  lt <- c(
    "% Requires: \\usepackage{booktabs}",
    "\\begin{table}[ht]",
    "\\centering",
    sprintf("\\caption{VAE-NLME Parameter Estimates: %s}", .esc_tex(dataset_name)),
    sprintf("\\label{tab:%s}", gsub("_", "-", dataset_name, fixed = TRUE)),
    "\\begin{tabular}{@{}lrr@{}}",
    "\\toprule",
    "Parameter & Linear. & IS \\\\",
    "\\midrule"
  )

  tex_sec <- function(title)
    lt <<- c(lt, sprintf("\\multicolumn{3}{@{}l}{\\textit{%s}} \\\\", title))
  tex_par <- function(nm, val)
    lt <<- c(lt, sprintf("\\quad \\texttt{%s} & %.4f & \\\\", .esc_tex(nm), val))

  tex_sec("Fixed Effects")
  for (k in seq_len(z_dim))           tex_par(param_names[k], zp[k])
  if (count > 0) {
    tex_sec("Covariate Effects (selected)")
    for (k in seq_along(cov_names_sel)) tex_par(cov_names_sel[k], cov_vals_sel[k])
  }
  tex_sec("Random Effects (SD)")
  for (k in seq_len(z_dim))           tex_par(omega_names[k], om[k])
  tex_sec("Error Model")
  if (a_v != 0) lt <- c(lt, sprintf("\\quad $a$ & %.4f & \\\\", a_v))
  if (b_v != 0) lt <- c(lt, sprintf("\\quad $b$ & %.4f & \\\\", b_v))
  lt <- c(lt, "\\midrule",
    "\\multicolumn{3}{@{}l}{\\textit{Information Criteria}} \\\\",
    "\\quad Criterion & Linearisation & Imp.\\ Sampling \\\\",
    "\\midrule")
  for (ic in ic_list)
    lt <- c(lt, sprintf("\\quad %s & %.2f & %.2f \\\\",
                        ic[1], as.numeric(ic[2]), as.numeric(ic[3])))
  lt <- c(lt, "\\bottomrule", "\\end{tabular}", "\\end{table}")

  tex_path <- file.path(abs_dir, paste0(dataset_name, "_results.tex"))
  writeLines(lt, tex_path)
  message("Saved: ", tex_path)

  # ===========================================================================
  # 3. Markdown (GitHub-Flavoured Markdown)
  # ===========================================================================
  md <- c(sprintf("## %s — VAE-NLME Parameter Estimates", dataset_name), "",
          "### Fixed Effects", "", "| Parameter | Estimate |", "|:----------|--------:|")
  for (k in seq_len(z_dim))
    md <- c(md, sprintf("| `%s` | %.4f |", param_names[k], zp[k]))
  if (count > 0) {
    md <- c(md, "", "### Covariate Effects (selected)", "",
            "| Parameter | Estimate |", "|:----------|--------:|")
    for (k in seq_along(cov_names_sel))
      md <- c(md, sprintf("| `%s` | %.4f |", cov_names_sel[k], cov_vals_sel[k]))
  }
  md <- c(md, "", "### Random Effects (SD)", "",
          "| Parameter | Estimate |", "|:----------|--------:|")
  for (k in seq_len(z_dim))
    md <- c(md, sprintf("| `%s` | %.4f |", omega_names[k], om[k]))
  md <- c(md, "", "### Error Model", "",
          "| Parameter | Estimate |", "|:----------|--------:|")
  if (a_v != 0) md <- c(md, sprintf("| `a` | %.4f |", a_v))
  if (b_v != 0) md <- c(md, sprintf("| `b` | %.4f |", b_v))
  md <- c(md, "", "### Information Criteria", "",
          "| Criterion | Linearisation | Imp. Sampling |",
          "|:----------|-------------:|--------------:|")
  for (ic in ic_list)
    md <- c(md, sprintf("| %s | %.2f | %.2f |",
                        ic[1], as.numeric(ic[2]), as.numeric(ic[3])))
  md_path <- file.path(abs_dir, paste0(dataset_name, "_results.md"))
  writeLines(md, md_path)
  message("Saved: ", md_path)

  # ===========================================================================
  # 4. Word (.docx) via officer + flextable
  # ===========================================================================
  if (requireNamespace("officer",   quietly = TRUE) &&
      requireNamespace("flextable", quietly = TRUE)) {

    # Build flat display data frame; track section-header row indices
    sec_idx <- integer(0)
    d_rows  <- list()
    add_sec <- function(title) {
      d_rows[[length(d_rows)+1L]] <<- data.frame(
        Parameter = title, Linearisation = "", Imp_Sampling = "",
        stringsAsFactors = FALSE)
      sec_idx <<- c(sec_idx, length(d_rows))
    }
    add_par <- function(nm, lin, is_ = "")
      d_rows[[length(d_rows)+1L]] <<- data.frame(
        Parameter = nm, Linearisation = lin, Imp_Sampling = is_,
        stringsAsFactors = FALSE)

    add_sec("Fixed Effects")
    for (k in seq_len(z_dim))           add_par(param_names[k], sprintf("%.4f", zp[k]))
    if (count > 0) {
      add_sec("Covariate Effects (selected)")
      for (k in seq_along(cov_names_sel))
        add_par(cov_names_sel[k], sprintf("%.4f", cov_vals_sel[k]))
    }
    add_sec("Random Effects (SD)")
    for (k in seq_len(z_dim))           add_par(omega_names[k], sprintf("%.4f", om[k]))
    add_sec("Error Model")
    if (a_v != 0) add_par("a", sprintf("%.4f", a_v))
    if (b_v != 0) add_par("b", sprintf("%.4f", b_v))
    add_sec("Information Criteria")
    # sub-header row inside IC section
    d_rows[[length(d_rows)+1L]] <- data.frame(
      Parameter = "Criterion", Linearisation = "Linearisation",
      Imp_Sampling = "Imp. Sampling", stringsAsFactors = FALSE)
    for (ic in ic_list)
      add_par(ic[1], sprintf("%.2f", as.numeric(ic[2])),
                     sprintf("%.2f", as.numeric(ic[3])))

    ft_df <- do.call(rbind, d_rows)
    colnames(ft_df) <- c("Parameter", "Linearisation", "Imp. Sampling")

    ft <- flextable::flextable(ft_df)
    ft <- flextable::set_header_labels(ft,
            Parameter = "Parameter", Linearisation = "Linear.",
            `Imp. Sampling` = "Imp. Sampling")
    ft <- flextable::bold(ft, part = "header")
    ft <- flextable::bold(ft,   i = sec_idx, part = "body")
    ft <- flextable::italic(ft, i = sec_idx, part = "body")
    ft <- flextable::bg(ft, i = sec_idx, bg = "#E8E8E8", part = "body")
    ft <- flextable::align(ft, j = c(2, 3), align = "right",  part = "all")
    ft <- flextable::align(ft, j = 1,       align = "left",   part = "all")
    ft <- flextable::hline_top(ft, part = "body",
            border = officer::fp_border(width = 1.5))
    ft <- flextable::hline_bottom(ft, part = "body",
            border = officer::fp_border(width = 1.5))
    ft <- flextable::autofit(ft)

    doc <- officer::read_docx()
    doc <- officer::body_add_par(doc,
             sprintf("VAE-NLME Parameter Estimates: %s", dataset_name),
             style = "heading 2")
    doc <- flextable::body_add_flextable(doc, ft)

    docx_path <- file.path(abs_dir, paste0(dataset_name, "_results.docx"))
    print(doc, target = docx_path)
    message("Saved: ", docx_path)
  } else {
    message("Note: install 'officer' and 'flextable' for Word (.docx) export")
  }

  invisible(df)
}

# Convenience wrappers with hard-coded parameter labels
saveOutput_theo <- function(z_pop, omega_pop, a, b, z_dim, nbatch,
                             n_tot, h, names_co, LL_lin_mu, LL_is,
                             save_dir = "Results") {
  saveOutput_results(z_pop, omega_pop, a, b, z_dim, nbatch, n_tot, h,
                     param_names  = c("ka_pop", "ke_pop", "V_pop"),
                     omega_names  = c("omega_ka", "omega_ke", "omega_V"),
                     names_co     = names_co,
                     LL_lin_mu    = LL_lin_mu, LL_is = LL_is,
                     dataset_name = "theophylline", save_dir = save_dir)
}

saveOutput_neonates <- function(z_pop, omega_pop, a, b, z_dim, nbatch,
                                 n_tot, h, names_co, LL_lin_mu, LL_is,
                                 save_dir = "Results") {
  saveOutput_results(z_pop, omega_pop, a, b, z_dim, nbatch, n_tot, h,
                     param_names  = c("W0_pop", "kin_pop", "Tlag_pop",
                                      "koutmax_pop", "T50_pop"),
                     omega_names  = c("omega_W0", "omega_kin", "omega_Tlag",
                                      "omega_koutmax", "omega_T50"),
                     names_co     = names_co,
                     LL_lin_mu    = LL_lin_mu, LL_is = LL_is,
                     dataset_name = "neonates", save_dir = save_dir)
}

# =============================================================================
# plot_vpc_theo()  —  Visual Predictive Check for Theophylline (ggPMX style)
#
# Simulates n_sim replicates from the fitted VAE-NLME model (population params
# + individual random effects), computes prediction intervals, and overlays them
# on the observed data (spaghetti) with observed empirical quantiles.
#
# Arguments:
#   data      : torch tensor [N, T_max, 5]  (time, conc, dose, wt, sex)
#   lengths   : torch tensor [N]            observed length per individual
#   z_pop     : torch tensor [z_dim + M]    fitted population parameters
#   omega_pop : torch tensor [z_dim]        fitted variance of random effects
#   a, b      : scalars (torch tensors)     additive/proportional error
#   h         : function                    inverse-link (exp)
#   dose      : torch tensor [N]            individual dose
#   C         : torch tensor [N, z_dim, z_dim+M] covariate design matrix
#   z_dim     : integer                     number of PK parameters
#   n_sim     : integer                     number of simulation replicates
#   pi_lo, pi_hi : numerics                 prediction interval quantiles
#   seed      : integer                     RNG seed for reproducibility
#   save_path : character                   output PDF path (JPG auto-created)
# =============================================================================
plot_vpc_theo <- function(data, lengths, z_pop, omega_pop, a, b, h,
                           dose, C, z_dim,
                           n_sim     = 500L,
                           nbins     = NULL,    # NULL = exact time points; integer = N time bins
                           pi_lo     = 0.05,    # lower PI quantile (default 5th)
                           pi_hi     = 0.95,    # upper PI quantile (default 95th)
                           ci_level  = 0.95,    # CI around each simulated percentile band
                           show_obs  = TRUE,    # overlay individual observed data (GOF)
                           seed      = 42L,
                           save_path = "Plots/theophylline_vpc.pdf") {

  set.seed(seed)

  # ---- extract numeric values from torch tensors ----------------------------
  z_pop_v  <- as.numeric(z_pop$detach())
  omega_v  <- as.numeric(omega_pop$detach())   # variances (omega^2)
  a_v      <- as.numeric(a$detach())
  b_v      <- as.numeric(b$detach())
  N        <- data$shape[1]
  len_v    <- as.integer(lengths$detach())
  dose_v   <- as.numeric(dose$detach())
  C_arr    <- as.array(C$detach())             # [N, z_dim, z_dim+M]
  data_arr <- as.array(data$detach())          # [N, T_max, 5]

  # ---- observed data in long format ----------------------------------------
  n_obs <- sum(len_v)
  obs_id   <- integer(n_obs)
  obs_time <- numeric(n_obs)
  obs_dv   <- numeric(n_obs)
  pos <- 1L
  for (i in seq_len(N)) {
    ni  <- len_v[i]
    end <- pos + ni - 1L
    obs_id[pos:end]   <- i
    obs_time[pos:end] <- data_arr[i, seq_len(ni), 1L]
    obs_dv[pos:end]   <- data_arr[i, seq_len(ni), 2L]
    pos <- end + 1L
  }
  obs_df <- data.frame(id = obs_id, time = obs_time, dv = obs_dv)

  # ---- 1-compartment oral PK -----------------------------------------------
  pk1_oral <- function(ka, ke, V, dose_i, t_vec) {
    if (abs(ka - ke) < 1e-7) ka <- ka * (1 + 1e-6)
    dose_i * ka / (V * (ka - ke)) * (exp(-ke * t_vec) - exp(-ka * t_vec))
  }

  # ---- simulate n_sim replicates (keep replicate index for CI bands) --------
  total_rows <- n_sim * n_obs
  sim_rep  <- integer(total_rows)
  sim_time <- numeric(total_rows)
  sim_dv   <- numeric(total_rows)
  pos <- 1L
  time_list <- lapply(seq_len(N), function(i) data_arr[i, seq_len(len_v[i]), 1L])

  message("Simulating ", n_sim, " replicates for VPC ...")
  for (s in seq_len(n_sim)) {
    for (i in seq_len(N)) {
      ni    <- len_v[i]
      C_i   <- C_arr[i, , ]                         # [z_dim, z_dim+M]
      z_ind <- as.numeric(C_i %*% z_pop_v)          # [z_dim] individual log-params
      eta   <- rnorm(z_dim, 0, sqrt(pmax(omega_v, 0)))
      phi   <- z_ind + eta
      ka_i  <- exp(phi[1L]); ke_i <- exp(phi[2L]); V_i <- exp(phi[3L])
      t_vec <- time_list[[i]]
      pred  <- pk1_oral(ka_i, ke_i, V_i, dose_v[i], t_vec)
      sigma <- pmax(a_v + b_v * pmax(pred, 0), 1e-8)
      dv_s  <- pmax(pred + rnorm(ni, 0, sigma), 0)
      end_pos <- pos + ni - 1L
      sim_rep[pos:end_pos]  <- s
      sim_time[pos:end_pos] <- t_vec
      sim_dv[pos:end_pos]   <- dv_s
      pos <- end_pos + 1L
    }
  }
  sim_df <- data.frame(rep = sim_rep, time = sim_time, dv = sim_dv)

  # ---- time binning --------------------------------------------------------
  if (is.null(nbins)) {
    all_t       <- sort(unique(round(obs_df$time, 4)))
    obs_df$bin  <- match(round(obs_df$time, 4), all_t)
    sim_df$bin  <- match(round(sim_df$time, 4), all_t)
    bin_centers <- all_t
  } else {
    brks        <- seq(min(obs_df$time) - 1e-6, max(obs_df$time) + 1e-6,
                       length.out = nbins + 1L)
    obs_df$bin  <- cut(obs_df$time, breaks = brks, labels = FALSE,
                       include.lowest = TRUE)
    sim_df$bin  <- cut(sim_df$time, breaks = brks, labels = FALSE,
                       include.lowest = TRUE)
    all_bins_raw <- sort(unique(obs_df$bin))
    bin_centers  <- sapply(all_bins_raw,
                           function(b) median(obs_df$time[obs_df$bin == b]))
  }
  all_bins <- sort(unique(obs_df$bin))

  # ---- CI bands + observed percentiles via shared helper -------------------
  bands  <- .sim_bands(sim_df, obs_df, bin_centers, pi_lo, pi_hi, ci_level)
  sim_ci <- bands$sim_ci; obs_qi <- bands$obs_qi

  # ---- build two-panel plot ------------------------------------------------
  p_lin <- .sim_panel(sim_ci, obs_qi, obs_df, show_obs, pi_lo, pi_hi,
                       log_scale = FALSE)
  p_log <- .sim_panel(sim_ci, obs_qi, obs_df, show_obs, pi_lo, pi_hi,
                       log_scale = TRUE)

  bins_lbl <- if (is.null(nbins)) paste0(length(all_bins), " time points") else
                                  paste0(nbins, " bins")
  pct_pi   <- round((pi_hi - pi_lo) * 100)
  ci_pct   <- round(ci_level * 100)
  subtitle <- paste0(
    "sim ", pct_pi, "% PI (blue)  |  sim 50th (gold)  |  ",
    "obs ", round(pi_lo*100), "th/", round(pi_hi*100), "th (red)  |  obs 50th (black)",
    if (show_obs) "  |  obs data (grey)" else "")
  title_str <- sprintf(
    "VPC - Theophylline  (N=%d,  n_sim=%d,  %d%% PI,  %d%% sim CI,  %s)\n%s",
    N, n_sim, pct_pi, ci_pct, bins_lbl, subtitle)

  p_out <- p_lin / p_log +
    plot_annotation(
      title = title_str,
      theme = theme(
        plot.title = element_text(size = 9.5, hjust = 0.5, lineheight = 1.45))
    )

  .save_plot(p_out, save_path, width = 8, height = 10)
  invisible(p_out)
}

# =============================================================================
# Internal helpers shared by plot_vpc_theo and plot_ppc_theo
# =============================================================================

# Compute per-bin CI ribbon data (sim) + observed quantiles.
# sim_df must have columns: rep (replicate index), time, dv, bin
# obs_df must have columns: time, dv, bin
.sim_bands <- function(sim_df, obs_df, bin_centers, pi_lo, pi_hi, ci_level) {
  all_bins <- sort(unique(obs_df$bin))
  probs    <- c(pi_lo, 0.5, pi_hi)
  ci_lo_p  <- (1 - ci_level) / 2
  ci_hi_p  <- 1 - ci_lo_p
  n_sim    <- max(sim_df$rep)

  ci_list <- vector("list", length(all_bins))
  for (bi in seq_along(all_bins)) {
    b   <- all_bins[bi]
    sub <- sim_df[sim_df$bin == b, c("rep", "dv")]
    pct_by_rep <- tapply(sub$dv, sub$rep, function(x)
      quantile(x, probs = probs, names = FALSE))
    pct_mat <- do.call(rbind, pct_by_rep)   # [n_sim, 3]
    ci_list[[bi]] <- data.frame(
      bin    = b, time   = bin_centers[bi],
      lo_lo  = quantile(pct_mat[, 1], ci_lo_p, names = FALSE),
      lo_med = median(pct_mat[, 1]),
      lo_hi  = quantile(pct_mat[, 1], ci_hi_p, names = FALSE),
      md_lo  = quantile(pct_mat[, 2], ci_lo_p, names = FALSE),
      md_med = median(pct_mat[, 2]),
      md_hi  = quantile(pct_mat[, 2], ci_hi_p, names = FALSE),
      hi_lo  = quantile(pct_mat[, 3], ci_lo_p, names = FALSE),
      hi_med = median(pct_mat[, 3]),
      hi_hi  = quantile(pct_mat[, 3], ci_hi_p, names = FALSE),
      stringsAsFactors = FALSE)
  }
  sim_ci <- do.call(rbind, ci_list)

  obs_qi <- do.call(rbind, lapply(seq_along(all_bins), function(bi) {
    b    <- all_bins[bi]; vals <- obs_df$dv[obs_df$bin == b]
    data.frame(bin = b, time = bin_centers[bi],
               q05 = quantile(vals, pi_lo, names = FALSE),
               q50 = quantile(vals, 0.5,   names = FALSE),
               q95 = quantile(vals, pi_hi, names = FALSE),
               stringsAsFactors = FALSE)
  }))
  list(sim_ci = sim_ci, obs_qi = obs_qi)
}

# Build a single VPC/PPC ggplot panel (linear or log scale).
.sim_panel <- function(sim_ci, obs_qi, obs_df, show_obs, pi_lo, pi_hi,
                        log_scale = FALSE,
                        panel_title = if (log_scale) "Log\u2080 scale" else "Linear scale",
                        y_label = "Concentration (mg/L)",
                        x_label = "Time (h)") {
  col_pi_fill  <- "#AED6F1"; col_pi_line  <- "#1A5276"
  col_med_fill <- "#FCF3CF"; col_med_line <- "#D35400"
  col_obs_pi   <- "#C0392B"; col_obs_med  <- "#1C2833"

  p <- ggplot() +
    geom_ribbon(data = sim_ci, aes(x = time, ymin = lo_lo, ymax = lo_hi),
                fill = col_pi_fill, alpha = 0.60) +
    geom_ribbon(data = sim_ci, aes(x = time, ymin = hi_lo, ymax = hi_hi),
                fill = col_pi_fill, alpha = 0.60) +
    geom_ribbon(data = sim_ci, aes(x = time, ymin = md_lo, ymax = md_hi),
                fill = col_med_fill, alpha = 0.75) +
    geom_line(data = sim_ci, aes(x = time, y = lo_med),
              colour = col_pi_line,  linetype = "dashed", linewidth = 0.65) +
    geom_line(data = sim_ci, aes(x = time, y = hi_med),
              colour = col_pi_line,  linetype = "dashed", linewidth = 0.65) +
    geom_line(data = sim_ci, aes(x = time, y = md_med),
              colour = col_med_line, linetype = "dashed", linewidth = 0.65)

  if (show_obs) {
    p <- p +
      geom_line(data  = obs_df, aes(x = time, y = dv, group = id),
                colour = "grey60", linewidth = 0.30, alpha = 0.70) +
      geom_point(data = obs_df, aes(x = time, y = dv),
                 colour = "grey25", size = 0.9, alpha = 0.80)
  }

  p <- p +
    geom_line(data = obs_qi, aes(x = time, y = q05),
              colour = col_obs_pi,  linewidth = 0.85) +
    geom_line(data = obs_qi, aes(x = time, y = q95),
              colour = col_obs_pi,  linewidth = 0.85) +
    geom_line(data = obs_qi, aes(x = time, y = q50),
              colour = col_obs_med, linewidth = 1.05) +
    labs(x = x_label, y = y_label, title = panel_title) +
    theme_bw(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          plot.title = element_text(size = 11, face = "bold", hjust = 0.5))

  if (log_scale) {
    y_min <- max(0.01, min(obs_df$dv[obs_df$dv > 0]) * 0.5)
    p     <- p + scale_y_log10(limits = c(y_min, NA))
  }
  p
}

# =============================================================================
# plot_vpc()    Generic VPC for any VAE-NLME model
#
# sim_fn : function(i, z_log, t_vec)  numeric vector of predicted DV
#           i = subject index (1-indexed), z_log = individual params in log-space,
#           t_vec = observation times for subject i
#           The function should capture any extra dose/infusion data via closure.
# dataset_name : character  used in plot title
# y_label      : character  y-axis label (default "Concentration (mg/L)")
# =============================================================================
plot_vpc <- function(data, lengths, z_pop, omega_pop, a, b, C, z_dim,
                     sim_fn,
                     n_sim        = 500L,
                     nbins        = NULL,
                     pi_lo        = 0.05,
                     pi_hi        = 0.95,
                     ci_level     = 0.95,
                     show_obs     = TRUE,
                     seed         = 42L,
                     dataset_name = "model",
                     y_label      = "Concentration (mg/L)",
                     tald_fn      = NULL,
                     save_path    = NULL) {

  set.seed(seed)
  z_pop_v  <- as.numeric(z_pop$detach())
  omega_v  <- as.numeric(omega_pop$detach())
  a_v      <- as.numeric(a$detach()); b_v <- as.numeric(b$detach())
  N        <- data$shape[1]; len_v <- as.integer(lengths$detach())
  C_arr    <- as.array(C$detach()); data_arr <- as.array(data$detach())
  n_obs    <- sum(len_v)

  obs_id <- integer(n_obs); obs_time <- numeric(n_obs); obs_dv <- numeric(n_obs)
  pos <- 1L
  for (i in seq_len(N)) {
    ni <- len_v[i]; end <- pos + ni - 1L
    obs_id[pos:end]   <- i
    obs_time[pos:end] <- data_arr[i, seq_len(ni), 1L]
    obs_dv[pos:end]   <- data_arr[i, seq_len(ni), 2L]
    pos <- end + 1L
  }
  obs_df    <- data.frame(id = obs_id, time = obs_time, dv = obs_dv)
  time_list <- lapply(seq_len(N), function(i) data_arr[i, seq_len(len_v[i]), 1L])

  total_rows <- n_sim * n_obs
  sim_rep  <- integer(total_rows); sim_time <- numeric(total_rows); sim_dv <- numeric(total_rows)
  pos <- 1L
  message(sprintf("VPC: simulating %d replicates for %s ...", n_sim, dataset_name))
  for (s in seq_len(n_sim)) {
    for (i in seq_len(N)) {
      ni    <- len_v[i]
      C_i   <- C_arr[i, , ]
      z_ind <- as.numeric(C_i %*% z_pop_v)
      eta   <- rnorm(z_dim, 0, sqrt(pmax(omega_v, 0)))
      phi   <- z_ind + eta
      t_vec <- time_list[[i]]
      pred  <- sim_fn(i, phi, t_vec)
      sigma <- pmax(a_v + b_v * pmax(pred, 0), 1e-8)
      dv_s  <- pmax(pred + rnorm(ni, 0, sigma), 0)
      end_pos <- pos + ni - 1L
      sim_rep[pos:end_pos]  <- s; sim_time[pos:end_pos] <- t_vec; sim_dv[pos:end_pos] <- dv_s
      pos <- end_pos + 1L
    }
  }
  sim_df <- data.frame(rep = sim_rep, time = sim_time, dv = sim_dv)

  # ---- Optional: replace absolute time with time-after-last-dose (TALD) -----
  x_label <- "Time (h)"
  if (!is.null(tald_fn)) {
    tald_obs <- numeric(n_obs); pos2 <- 1L
    for (i in seq_len(N)) {
      ni <- len_v[i]
      tald_obs[pos2:(pos2 + ni - 1L)] <- tald_fn(i, time_list[[i]])
      pos2 <- pos2 + ni
    }
    obs_df$time <- tald_obs
    sim_df$time <- rep(tald_obs, n_sim)
    x_label <- "Time after last dose (h)"
  }

  if (is.null(nbins)) {
    all_t        <- sort(unique(round(obs_df$time, 4)))
    obs_df$bin   <- match(round(obs_df$time, 4), all_t)
    sim_df$bin   <- match(round(sim_df$time, 4), all_t)
    bin_centers  <- all_t
  } else {
    brks         <- seq(min(obs_df$time) - 1e-6, max(obs_df$time) + 1e-6, length.out = nbins + 1L)
    obs_df$bin   <- cut(obs_df$time, breaks = brks, labels = FALSE, include.lowest = TRUE)
    sim_df$bin   <- cut(sim_df$time, breaks = brks, labels = FALSE, include.lowest = TRUE)
    all_bins_raw <- sort(unique(obs_df$bin))
    bin_centers  <- sapply(all_bins_raw, function(b) median(obs_df$time[obs_df$bin == b]))
  }
  all_bins <- sort(unique(obs_df$bin))
  bands    <- .sim_bands(sim_df, obs_df, bin_centers, pi_lo, pi_hi, ci_level)
  sim_ci   <- bands$sim_ci; obs_qi <- bands$obs_qi

  p_lin <- .sim_panel(sim_ci, obs_qi, obs_df, show_obs, pi_lo, pi_hi,
                      log_scale = FALSE, y_label = y_label, x_label = x_label)
  p_log <- .sim_panel(sim_ci, obs_qi, obs_df, show_obs, pi_lo, pi_hi,
                      log_scale = TRUE,  y_label = y_label, x_label = x_label)

  bins_lbl  <- if (is.null(nbins)) paste0(length(all_bins), " time points") else
                                   paste0(nbins, " bins")
  pct_pi    <- round((pi_hi - pi_lo) * 100); ci_pct <- round(ci_level * 100)
  tald_note <- if (!is.null(tald_fn)) "  |  x = TALD" else ""
  subtitle  <- paste0(
    "sim ", pct_pi, "% PI (blue)  |  sim 50th (gold)  |  ",
    "obs ", round(pi_lo*100), "th/", round(pi_hi*100), "th (red)  |  obs 50th (black)",
    if (show_obs) "  |  obs data (grey)" else "", tald_note)
  title_str <- sprintf(
    "VPC - %s  (N=%d,  n_sim=%d,  %d%% PI,  %d%% CI,  %s)\n%s",
    dataset_name, N, n_sim, pct_pi, ci_pct, bins_lbl, subtitle)

  p_out <- p_lin / p_log +
    plot_annotation(title = title_str,
                    theme = theme(plot.title = element_text(size = 9.5, hjust = 0.5,
                                                            lineheight = 1.45)))
  if (!is.null(save_path)) .save_plot(p_out, save_path, width = 8, height = 10)
  invisible(p_out)
}

# =============================================================================
# plot_ppc()    Generic Posterior Predictive Check for any VAE-NLME model
#
# Uses encoder posterior q(z|x_i) = N(mu_i, L_i L_i^T) for simulation.
# sim_fn : same signature as for plot_vpc()
# =============================================================================
plot_ppc <- function(data, lengths, Encoder, data_in, covariates_in,
                     a, b, z_dim,
                     sim_fn,
                     n_sim        = 200L,
                     nbins        = NULL,
                     pi_lo        = 0.05,
                     pi_hi        = 0.95,
                     ci_level     = 0.95,
                     show_obs     = TRUE,
                     seed         = 42L,
                     dataset_name = "model",
                     y_label      = "Concentration (mg/L)",
                     tald_fn      = NULL,
                     save_path    = NULL) {

  set.seed(seed)
  enc_out <- with_no_grad(Encoder$forward(data_in, covariates_in, lengths))
  mu_arr  <- as.array(enc_out$mu$detach())
  L_arr   <- as.array(enc_out$L$detach())

  a_v      <- as.numeric(a$detach()); b_v <- as.numeric(b$detach())
  N        <- data$shape[1]; len_v <- as.integer(lengths$detach())
  data_arr <- as.array(data$detach())
  n_obs    <- sum(len_v)

  obs_id <- integer(n_obs); obs_time <- numeric(n_obs); obs_dv <- numeric(n_obs)
  pos <- 1L
  for (i in seq_len(N)) {
    ni <- len_v[i]; end <- pos + ni - 1L
    obs_id[pos:end]   <- i
    obs_time[pos:end] <- data_arr[i, seq_len(ni), 1L]
    obs_dv[pos:end]   <- data_arr[i, seq_len(ni), 2L]
    pos <- end + 1L
  }
  obs_df    <- data.frame(id = obs_id, time = obs_time, dv = obs_dv)
  time_list <- lapply(seq_len(N), function(i) data_arr[i, seq_len(len_v[i]), 1L])

  total_rows <- n_sim * n_obs
  sim_rep  <- integer(total_rows); sim_time <- numeric(total_rows); sim_dv <- numeric(total_rows)
  pos <- 1L
  message(sprintf("PPC: sampling encoder posterior for %s (n_sim=%d) ...", dataset_name, n_sim))
  for (s in seq_len(n_sim)) {
    for (i in seq_len(N)) {
      ni    <- len_v[i]
      mu_i  <- mu_arr[i, ]; L_i <- L_arr[i, , ]
      eps_i <- rnorm(z_dim, 0, 1)
      z_i   <- mu_i + as.numeric(L_i %*% eps_i)
      t_vec <- time_list[[i]]
      pred  <- sim_fn(i, z_i, t_vec)
      sigma <- pmax(a_v + b_v * pmax(pred, 0), 1e-8)
      dv_s  <- pmax(pred + rnorm(ni, 0, sigma), 0)
      end_pos <- pos + ni - 1L
      sim_rep[pos:end_pos]  <- s; sim_time[pos:end_pos] <- t_vec; sim_dv[pos:end_pos] <- dv_s
      pos <- end_pos + 1L
    }
  }
  sim_df <- data.frame(rep = sim_rep, time = sim_time, dv = sim_dv)

  # ---- Optional: replace absolute time with time-after-last-dose (TALD) -----
  x_label <- "Time (h)"
  if (!is.null(tald_fn)) {
    tald_obs <- numeric(n_obs); pos2 <- 1L
    for (i in seq_len(N)) {
      ni <- len_v[i]
      tald_obs[pos2:(pos2 + ni - 1L)] <- tald_fn(i, time_list[[i]])
      pos2 <- pos2 + ni
    }
    obs_df$time <- tald_obs
    sim_df$time <- rep(tald_obs, n_sim)
    x_label <- "Time after last dose (h)"
  }

  if (is.null(nbins)) {
    all_t        <- sort(unique(round(obs_df$time, 4)))
    obs_df$bin   <- match(round(obs_df$time, 4), all_t)
    sim_df$bin   <- match(round(sim_df$time, 4), all_t)
    bin_centers  <- all_t
  } else {
    brks         <- seq(min(obs_df$time) - 1e-6, max(obs_df$time) + 1e-6, length.out = nbins + 1L)
    obs_df$bin   <- cut(obs_df$time, breaks = brks, labels = FALSE, include.lowest = TRUE)
    sim_df$bin   <- cut(sim_df$time, breaks = brks, labels = FALSE, include.lowest = TRUE)
    all_bins_raw <- sort(unique(obs_df$bin))
    bin_centers  <- sapply(all_bins_raw, function(b) median(obs_df$time[obs_df$bin == b]))
  }
  all_bins <- sort(unique(obs_df$bin))
  bands    <- .sim_bands(sim_df, obs_df, bin_centers, pi_lo, pi_hi, ci_level)
  sim_ci   <- bands$sim_ci; obs_qi <- bands$obs_qi

  p_lin <- .sim_panel(sim_ci, obs_qi, obs_df, show_obs, pi_lo, pi_hi,
                      log_scale = FALSE, y_label = y_label, x_label = x_label)
  p_log <- .sim_panel(sim_ci, obs_qi, obs_df, show_obs, pi_lo, pi_hi,
                      log_scale = TRUE,  y_label = y_label, x_label = x_label)

  bins_lbl  <- if (is.null(nbins)) paste0(length(all_bins), " time points") else
                                   paste0(nbins, " bins")
  pct_pi    <- round((pi_hi - pi_lo) * 100); ci_pct <- round(ci_level * 100)
  tald_note <- if (!is.null(tald_fn)) "  |  x = TALD" else ""
  subtitle  <- paste0(
    "post. pred. ", pct_pi, "% PI (blue)  |  post. pred. 50th (gold)  |  ",
    "obs ", round(pi_lo*100), "th/", round(pi_hi*100), "th (red)  |  obs 50th (black)",
    if (show_obs) "  |  obs data (grey)" else "", tald_note)
  title_str <- sprintf(
    "PPC - %s  (N=%d,  n_sim=%d,  %d%% PI,  %d%% CI,  %s)\n[Encoder posterior q(z|x)]\n%s",
    dataset_name, N, n_sim, pct_pi, ci_pct, bins_lbl, subtitle)

  p_out <- p_lin / p_log +
    plot_annotation(title = title_str,
                    theme = theme(plot.title = element_text(size = 9.0, hjust = 0.5,
                                                            lineheight = 1.45)))
  if (!is.null(save_path)) .save_plot(p_out, save_path, width = 8, height = 10)
  invisible(p_out)
}
# =============================================================================
# plot_ppc_theo()  —  Posterior Predictive Check (PPC) for Theophylline
#
# Unlike the VPC (which samples from the marginal population distribution),
# the PPC samples from the ENCODER POSTERIOR q(z | x_i) = N(mu_i, L_i L_i^T):
#   eps ~ N(0, I);  z_i = mu_i + L_i %*% eps
# This conditions on each subject's observed data, so bands are tighter.
# Use to check individual-level fit; use VPC to check population-level fit.
#
# Extra arguments vs plot_vpc_theo:
#   Encoder       : trained lstm_encoder R6 module
#   data_in       : torch tensor [N, T, 2]   standardised (time, conc) input
#   covariates_in : torch tensor [N, n_cov]  standardised covariates
# =============================================================================
plot_ppc_theo <- function(data, lengths, Encoder, data_in, covariates_in,
                           a, b, dose, h, z_dim,
                           n_sim    = 200L,
                           nbins    = NULL,
                           pi_lo    = 0.05,
                           pi_hi    = 0.95,
                           ci_level = 0.95,
                           show_obs = TRUE,
                           seed     = 42L,
                           save_path = "Plots/theophylline_ppc.pdf") {

  set.seed(seed)

  # ---- encoder forward pass (no gradient needed) ---------------------------
  enc_out <- with_no_grad(Encoder$forward(data_in, covariates_in, lengths))
  mu_arr  <- as.array(enc_out$mu$detach())   # [N, z_dim]  posterior mean
  L_arr   <- as.array(enc_out$L$detach())    # [N, z_dim, z_dim]  Cholesky

  # ---- extract scalar values -----------------------------------------------
  a_v      <- as.numeric(a$detach())
  b_v      <- as.numeric(b$detach())
  N        <- data$shape[1]
  len_v    <- as.integer(lengths$detach())
  dose_v   <- as.numeric(dose$detach())
  data_arr <- as.array(data$detach())

  # ---- observed data in long format ----------------------------------------
  n_obs <- sum(len_v)
  obs_id <- integer(n_obs); obs_time <- numeric(n_obs); obs_dv <- numeric(n_obs)
  pos <- 1L
  for (i in seq_len(N)) {
    ni <- len_v[i]; end <- pos + ni - 1L
    obs_id[pos:end]   <- i
    obs_time[pos:end] <- data_arr[i, seq_len(ni), 1L]
    obs_dv[pos:end]   <- data_arr[i, seq_len(ni), 2L]
    pos <- end + 1L
  }
  obs_df <- data.frame(id = obs_id, time = obs_time, dv = obs_dv)

  # ---- 1-cmt oral PK -------------------------------------------------------
  pk1_oral <- function(ka, ke, V, dose_i, t_vec) {
    if (abs(ka - ke) < 1e-7) ka <- ka * (1 + 1e-6)
    dose_i * ka / (V * (ka - ke)) * (exp(-ke * t_vec) - exp(-ka * t_vec))
  }

  # ---- simulate from encoder posterior q(z | x_obs) -----------------------
  # z_i = mu_i + L_i %*% eps_i,  eps_i ~ N(0, I_{z_dim})
  time_list  <- lapply(seq_len(N), function(i) data_arr[i, seq_len(len_v[i]), 1L])
  total_rows <- n_sim * n_obs
  sim_rep    <- integer(total_rows)
  sim_time   <- numeric(total_rows)
  sim_dv     <- numeric(total_rows)
  pos <- 1L

  message("PPC: sampling from encoder posterior (n_sim=", n_sim, ") ...")
  for (s in seq_len(n_sim)) {
    for (i in seq_len(N)) {
      ni    <- len_v[i]
      mu_i  <- mu_arr[i, ]             # [z_dim]  posterior mean in log-space
      L_i   <- L_arr[i, , ]            # [z_dim, z_dim]  Cholesky
      eps_i <- rnorm(z_dim, 0, 1)
      z_i   <- mu_i + as.numeric(L_i %*% eps_i)   # [z_dim] sample from q(z|x)
      ka_i  <- exp(z_i[1L]); ke_i <- exp(z_i[2L]); V_i <- exp(z_i[3L])
      t_vec <- time_list[[i]]
      pred  <- pk1_oral(ka_i, ke_i, V_i, dose_v[i], t_vec)
      sigma <- pmax(a_v + b_v * pmax(pred, 0), 1e-8)
      dv_s  <- pmax(pred + rnorm(ni, 0, sigma), 0)
      end_pos <- pos + ni - 1L
      sim_rep[pos:end_pos]  <- s
      sim_time[pos:end_pos] <- t_vec
      sim_dv[pos:end_pos]   <- dv_s
      pos <- end_pos + 1L
    }
  }
  sim_df <- data.frame(rep = sim_rep, time = sim_time, dv = sim_dv)

  # ---- time binning --------------------------------------------------------
  if (is.null(nbins)) {
    all_t        <- sort(unique(round(obs_df$time, 4)))
    obs_df$bin   <- match(round(obs_df$time, 4), all_t)
    sim_df$bin   <- match(round(sim_df$time, 4), all_t)
    bin_centers  <- all_t
  } else {
    brks         <- seq(min(obs_df$time) - 1e-6, max(obs_df$time) + 1e-6,
                        length.out = nbins + 1L)
    obs_df$bin   <- cut(obs_df$time, breaks = brks, labels = FALSE,
                        include.lowest = TRUE)
    sim_df$bin   <- cut(sim_df$time, breaks = brks, labels = FALSE,
                        include.lowest = TRUE)
    all_bins_raw <- sort(unique(obs_df$bin))
    bin_centers  <- sapply(all_bins_raw,
                           function(b) median(obs_df$time[obs_df$bin == b]))
  }

  # ---- CI bands + observed percentiles via shared helper ------------------
  bands  <- .sim_bands(sim_df, obs_df, bin_centers, pi_lo, pi_hi, ci_level)
  sim_ci <- bands$sim_ci; obs_qi <- bands$obs_qi

  # ---- build two-panel plot ------------------------------------------------
  p_lin <- .sim_panel(sim_ci, obs_qi, obs_df, show_obs, pi_lo, pi_hi,
                       log_scale   = FALSE,
                       panel_title = "Linear scale")
  p_log <- .sim_panel(sim_ci, obs_qi, obs_df, show_obs, pi_lo, pi_hi,
                       log_scale   = TRUE,
                       panel_title = "Log\u2080 scale")

  all_bins <- sort(unique(obs_df$bin))
  bins_lbl <- if (is.null(nbins)) paste0(length(all_bins), " time points") else
                                  paste0(nbins, " bins")
  pct_pi   <- round((pi_hi - pi_lo) * 100)
  ci_pct   <- round(ci_level * 100)

  subtitle <- paste0(
    "post. pred. ", pct_pi, "% PI (blue)  |  post. pred. 50th (gold)  |  ",
    "obs ", round(pi_lo*100), "th/", round(pi_hi*100), "th (red)  |  obs 50th (black)",
    if (show_obs) "  |  obs data (grey)" else "")
  title_str <- sprintf(
    "PPC - Theophylline  (N=%d,  n_sim=%d,  %d%% PI,  %d%% CI,  %s)\n[Posterior Predictive: encoder q(z|x)]\n%s",
    N, n_sim, pct_pi, ci_pct, bins_lbl, subtitle)

  p_out <- p_lin / p_log +
    plot_annotation(
      title = title_str,
      theme = theme(
        plot.title = element_text(size = 9.0, hjust = 0.5, lineheight = 1.45))
    )

  .save_plot(p_out, save_path, width = 8, height = 10)
  invisible(p_out)
}

plot_vpc_warfarin <- function(data, lengths, z_pop, omega_pop, a, b, C, z_dim, dose,
                               n_sim=500L, nbins=NULL, pi_lo=0.05, pi_hi=0.95,
                               ci_level=0.95, show_obs=TRUE, seed=42L,
                               save_path="Plots/warfarin_vpc.pdf") {
  dose_v <- as.numeric(dose$detach())
  sim_fn <- function(i, z_log, t_vec) {
    ka <- exp(z_log[1]); ke <- exp(z_log[2]); V <- exp(z_log[3])
    if (abs(ka - ke) < 1e-7) ka <- ka * (1 + 1e-6)
    pmax(dose_v[i] * ka / (V * (ka - ke)) * (exp(-ke * t_vec) - exp(-ka * t_vec)), 0)
  }
  plot_vpc(data, lengths, z_pop, omega_pop, a, b, C, z_dim, sim_fn,
           n_sim=n_sim, nbins=nbins, pi_lo=pi_lo, pi_hi=pi_hi, ci_level=ci_level,
           show_obs=show_obs, seed=seed, dataset_name="warfarin",
           y_label="Concentration (mg/L)", save_path=save_path)
}

plot_ppc_warfarin <- function(data, lengths, Encoder, data_in, covariates_in,
                               a, b, dose, z_dim,
                               n_sim=200L, nbins=NULL, pi_lo=0.05, pi_hi=0.95,
                               ci_level=0.95, show_obs=TRUE, seed=42L,
                               save_path="Plots/warfarin_ppc.pdf") {
  dose_v <- as.numeric(dose$detach())
  sim_fn <- function(i, z_log, t_vec) {
    ka <- exp(z_log[1]); ke <- exp(z_log[2]); V <- exp(z_log[3])
    if (abs(ka - ke) < 1e-7) ka <- ka * (1 + 1e-6)
    pmax(dose_v[i] * ka / (V * (ka - ke)) * (exp(-ke * t_vec) - exp(-ka * t_vec)), 0)
  }
  plot_ppc(data, lengths, Encoder, data_in, covariates_in, a, b, z_dim, sim_fn,
           n_sim=n_sim, nbins=nbins, pi_lo=pi_lo, pi_hi=pi_hi, ci_level=ci_level,
           show_obs=show_obs, seed=seed, dataset_name="warfarin",
           y_label="Concentration (mg/L)", save_path=save_path)
}

plot_vpc_neonates <- function(data, lengths, z_pop, omega_pop, a, b, C, z_dim,
                               n_sim=500L, nbins=NULL, pi_lo=0.05, pi_hi=0.95,
                               ci_level=0.95, show_obs=TRUE, seed=42L,
                               save_path="Plots/neonates_vpc.pdf") {
  sim_fn <- function(i, z_log, t_vec) {
    W0 <- exp(z_log[1]); kin <- exp(z_log[2]); TL  <- exp(z_log[3])
    koutmax <- exp(z_log[4]); T50 <- exp(z_log[5])
    t_max <- max(t_vec, 0.001)
    n_s   <- max(1L, as.integer(ceiling(t_max / 0.5)))
    dt    <- t_max / n_s
    tg    <- seq(0, t_max, length.out = n_s + 1L)
    f_ode <- function(t, W) kin * plogis(2*(t - TL)) - koutmax * (1 - t/(T50 + t)) * W
    W_store <- numeric(n_s + 1L); W <- W0; W_store[1] <- W
    for (j in seq_len(n_s)) {
      t0 <- tg[j]
      k1 <- f_ode(t0, W); k2 <- f_ode(t0+dt/2, W+dt/2*k1)
      k3 <- f_ode(t0+dt/2, W+dt/2*k2); k4 <- f_ode(t0+dt, W+dt*k3)
      W  <- W + dt/6*(k1+2*k2+2*k3+k4)
      W_store[j+1L] <- W
    }
    pmax(approx(tg, W_store, xout = pmax(t_vec, 0))$y, 0)
  }
  plot_vpc(data, lengths, z_pop, omega_pop, a, b, C, z_dim, sim_fn,
           n_sim=n_sim, nbins=nbins, pi_lo=pi_lo, pi_hi=pi_hi, ci_level=ci_level,
           show_obs=show_obs, seed=seed, dataset_name="neonates",
           y_label="Weight (g)", save_path=save_path)
}

plot_ppc_neonates <- function(data, lengths, Encoder, data_in, covariates_in,
                               a, b, z_dim,
                               n_sim=200L, nbins=NULL, pi_lo=0.05, pi_hi=0.95,
                               ci_level=0.95, show_obs=TRUE, seed=42L,
                               save_path="Plots/neonates_ppc.pdf") {
  sim_fn <- function(i, z_log, t_vec) {
    W0 <- exp(z_log[1]); kin <- exp(z_log[2]); TL  <- exp(z_log[3])
    koutmax <- exp(z_log[4]); T50 <- exp(z_log[5])
    t_max <- max(t_vec, 0.001)
    n_s   <- max(1L, as.integer(ceiling(t_max / 0.5)))
    dt    <- t_max / n_s
    tg    <- seq(0, t_max, length.out = n_s + 1L)
    f_ode <- function(t, W) kin * plogis(2*(t - TL)) - koutmax * (1 - t/(T50 + t)) * W
    W_store <- numeric(n_s + 1L); W <- W0; W_store[1] <- W
    for (j in seq_len(n_s)) {
      t0 <- tg[j]
      k1 <- f_ode(t0, W); k2 <- f_ode(t0+dt/2, W+dt/2*k1)
      k3 <- f_ode(t0+dt/2, W+dt/2*k2); k4 <- f_ode(t0+dt, W+dt*k3)
      W  <- W + dt/6*(k1+2*k2+2*k3+k4)
      W_store[j+1L] <- W
    }
    pmax(approx(tg, W_store, xout = pmax(t_vec, 0))$y, 0)
  }
  plot_ppc(data, lengths, Encoder, data_in, covariates_in, a, b, z_dim, sim_fn,
           n_sim=n_sim, nbins=nbins, pi_lo=pi_lo, pi_hi=pi_hi, ci_level=ci_level,
           show_obs=show_obs, seed=seed, dataset_name="neonates",
           y_label="Weight (g)", save_path=save_path)
}

plot_vpc_pheno <- function(data, lengths, z_pop, omega_pop, a, b, C, z_dim,
                            dose_times, dose_amts,
                            n_sim=500L, nbins=NULL, pi_lo=0.05, pi_hi=0.95,
                            ci_level=0.95, show_obs=TRUE, seed=42L,
                            tald=FALSE,
                            save_path="Plots/pheno_sd_vpc.pdf") {
  dt_r <- as.matrix(dose_times$detach())
  da_r <- as.matrix(dose_amts$detach())
  sim_fn <- function(i, z_log, t_vec) {
    CL <- exp(z_log[1]); V <- exp(z_log[2]); ke <- CL / V
    dt_i <- dt_r[i, ]; da_i <- da_r[i, ]
    sapply(t_vec, function(t) {
      valid <- da_i > 0 & dt_i <= t
      if (!any(valid)) return(0)
      sum(da_i[valid] / V * exp(-ke * (t - dt_i[valid])))
    })
  }
  tald_fn <- if (tald) {
    function(i, t_vec) {
      dt_i <- dt_r[i, ]; da_i <- da_r[i, ]
      active_dt <- dt_i[da_i > 0 & dt_i < 1e8]
      sapply(t_vec, function(t) {
        prior <- active_dt[active_dt <= t]
        if (length(prior) == 0L) return(t)
        t - max(prior)
      })
    }
  } else NULL
  plot_vpc(data, lengths, z_pop, omega_pop, a, b, C, z_dim, sim_fn,
           n_sim=n_sim, nbins=nbins, pi_lo=pi_lo, pi_hi=pi_hi, ci_level=ci_level,
           show_obs=show_obs, seed=seed, dataset_name="pheno_sd",
           y_label="Concentration (mg/L)", tald_fn=tald_fn, save_path=save_path)
}

plot_ppc_pheno <- function(data, lengths, Encoder, data_in, covariates_in,
                            a, b, dose_times, dose_amts, z_dim,
                            n_sim=200L, nbins=NULL, pi_lo=0.05, pi_hi=0.95,
                            ci_level=0.95, show_obs=TRUE, seed=42L,
                            tald=FALSE,
                            save_path="Plots/pheno_sd_ppc.pdf") {
  dt_r <- as.matrix(dose_times$detach())
  da_r <- as.matrix(dose_amts$detach())
  sim_fn <- function(i, z_log, t_vec) {
    CL <- exp(z_log[1]); V <- exp(z_log[2]); ke <- CL / V
    dt_i <- dt_r[i, ]; da_i <- da_r[i, ]
    sapply(t_vec, function(t) {
      valid <- da_i > 0 & dt_i <= t
      if (!any(valid)) return(0)
      sum(da_i[valid] / V * exp(-ke * (t - dt_i[valid])))
    })
  }
  tald_fn <- if (tald) {
    function(i, t_vec) {
      dt_i <- dt_r[i, ]; da_i <- da_r[i, ]
      active_dt <- dt_i[da_i > 0 & dt_i < 1e8]
      sapply(t_vec, function(t) {
        prior <- active_dt[active_dt <= t]
        if (length(prior) == 0L) return(t)
        t - max(prior)
      })
    }
  } else NULL
  plot_ppc(data, lengths, Encoder, data_in, covariates_in, a, b, z_dim, sim_fn,
           n_sim=n_sim, nbins=nbins, pi_lo=pi_lo, pi_hi=pi_hi, ci_level=ci_level,
           show_obs=show_obs, seed=seed, dataset_name="pheno_sd",
           y_label="Concentration (mg/L)", tald_fn=tald_fn, save_path=save_path)
}

plot_vpc_mavo <- function(data, lengths, z_pop, omega_pop, a, b, C, z_dim,
                           rate, t_inf,
                           n_sim=500L, nbins=NULL, pi_lo=0.05, pi_hi=0.95,
                           ci_level=0.95, show_obs=TRUE, seed=42L,
                           save_path="Plots/mavoglurant_vpc.pdf") {
  rate_v  <- as.numeric(rate$detach())
  t_inf_v <- as.numeric(t_inf$detach())
  sim_fn <- function(i, z_log, t_vec) {
    CL <- exp(z_log[1]); V1 <- exp(z_log[2]); Q <- exp(z_log[3]); V2 <- exp(z_log[4])
    k10 <- CL/V1; k12 <- Q/V1; k21 <- Q/V2
    S   <- k10 + k12 + k21
    disc <- sqrt(max((k10+k12-k21)^2 + 4*k12*k21, 1e-12))
    al  <- (S+disc)/2; be <- (S-disc)/2; dab <- max(al-be, 1e-8)
    A   <-  rate_v[i]*(al-k21)/(V1*dab*max(al,1e-10))
    B   <- -rate_v[i]*(be-k21)/(V1*dab*max(abs(be),1e-10))
    ti  <- t_inf_v[i]
    sapply(t_vec, function(t) {
      if (t <= ti) A*(1-exp(-al*t)) + B*(1-exp(-be*t))
      else { dt <- t-ti; A*(1-exp(-al*ti))*exp(-al*dt) + B*(1-exp(-be*ti))*exp(-be*dt) }
    })
  }
  plot_vpc(data, lengths, z_pop, omega_pop, a, b, C, z_dim, sim_fn,
           n_sim=n_sim, nbins=nbins, pi_lo=pi_lo, pi_hi=pi_hi, ci_level=ci_level,
           show_obs=show_obs, seed=seed, dataset_name="mavoglurant",
           y_label="Concentration (ng/mL)", save_path=save_path)
}

plot_ppc_mavo <- function(data, lengths, Encoder, data_in, covariates_in,
                           a, b, rate, t_inf, z_dim,
                           n_sim=200L, nbins=NULL, pi_lo=0.05, pi_hi=0.95,
                           ci_level=0.95, show_obs=TRUE, seed=42L,
                           save_path="Plots/mavoglurant_ppc.pdf") {
  rate_v  <- as.numeric(rate$detach())
  t_inf_v <- as.numeric(t_inf$detach())
  sim_fn <- function(i, z_log, t_vec) {
    CL <- exp(z_log[1]); V1 <- exp(z_log[2]); Q <- exp(z_log[3]); V2 <- exp(z_log[4])
    k10 <- CL/V1; k12 <- Q/V1; k21 <- Q/V2
    S   <- k10 + k12 + k21
    disc <- sqrt(max((k10+k12-k21)^2 + 4*k12*k21, 1e-12))
    al  <- (S+disc)/2; be <- (S-disc)/2; dab <- max(al-be, 1e-8)
    A   <-  rate_v[i]*(al-k21)/(V1*dab*max(al,1e-10))
    B   <- -rate_v[i]*(be-k21)/(V1*dab*max(abs(be),1e-10))
    ti  <- t_inf_v[i]
    sapply(t_vec, function(t) {
      if (t <= ti) A*(1-exp(-al*t)) + B*(1-exp(-be*t))
      else { dt <- t-ti; A*(1-exp(-al*ti))*exp(-al*dt) + B*(1-exp(-be*ti))*exp(-be*dt) }
    })
  }
  plot_ppc(data, lengths, Encoder, data_in, covariates_in, a, b, z_dim, sim_fn,
           n_sim=n_sim, nbins=nbins, pi_lo=pi_lo, pi_hi=pi_hi, ci_level=ci_level,
           show_obs=show_obs, seed=seed, dataset_name="mavoglurant",
           y_label="Concentration (ng/mL)", save_path=save_path)
}

plot_vpc_nimo <- function(data, lengths, z_pop, omega_pop, a, b, C, z_dim,
                           rate, t_inf,
                           n_sim=500L, nbins=NULL, pi_lo=0.05, pi_hi=0.95,
                           ci_level=0.95, show_obs=TRUE, seed=42L,
                           save_path="Plots/nimoData_vpc.pdf") {
  rate_v  <- as.numeric(rate$detach())
  t_inf_v <- as.numeric(t_inf$detach())
  sim_fn <- function(i, z_log, t_vec) {
    CL <- exp(z_log[1]); V <- exp(z_log[2]); ke <- CL/V
    ti <- t_inf_v[i]; R <- rate_v[i]
    sapply(t_vec, function(t) {
      if (t <= ti) (R/CL)*(1-exp(-ke*t))
      else { c_tinf <- (R/CL)*(1-exp(-ke*ti)); c_tinf*exp(-ke*(t-ti)) }
    })
  }
  plot_vpc(data, lengths, z_pop, omega_pop, a, b, C, z_dim, sim_fn,
           n_sim=n_sim, nbins=nbins, pi_lo=pi_lo, pi_hi=pi_hi, ci_level=ci_level,
           show_obs=show_obs, seed=seed, dataset_name="nimoData",
           y_label="Concentration", save_path=save_path)
}

plot_ppc_nimo <- function(data, lengths, Encoder, data_in, covariates_in,
                           a, b, rate, t_inf, z_dim,
                           n_sim=200L, nbins=NULL, pi_lo=0.05, pi_hi=0.95,
                           ci_level=0.95, show_obs=TRUE, seed=42L,
                           save_path="Plots/nimoData_ppc.pdf") {
  rate_v  <- as.numeric(rate$detach())
  t_inf_v <- as.numeric(t_inf$detach())
  sim_fn <- function(i, z_log, t_vec) {
    CL <- exp(z_log[1]); V <- exp(z_log[2]); ke <- CL/V
    ti <- t_inf_v[i]; R <- rate_v[i]
    sapply(t_vec, function(t) {
      if (t <= ti) (R/CL)*(1-exp(-ke*t))
      else { c_tinf <- (R/CL)*(1-exp(-ke*ti)); c_tinf*exp(-ke*(t-ti)) }
    })
  }
  plot_ppc(data, lengths, Encoder, data_in, covariates_in, a, b, z_dim, sim_fn,
           n_sim=n_sim, nbins=nbins, pi_lo=pi_lo, pi_hi=pi_hi, ci_level=ci_level,
           show_obs=show_obs, seed=seed, dataset_name="nimoData",
           y_label="Concentration", save_path=save_path)
}

plot_vpc_theo_mult <- function(data, lengths, z_pop, omega_pop, a, b, C, z_dim, dose,
                                n_sim=500L, nbins=NULL, pi_lo=0.05, pi_hi=0.95,
                                ci_level=0.95, show_obs=TRUE, seed=42L,
                                tald=TRUE,
                                save_path="Plots/theophylline_multiple_vpc.pdf") {
  dose_arr_r <- as.array(dose$detach())
  sim_fn <- function(i, z_log, t_vec) {
    ka <- exp(z_log[1]); ke <- exp(z_log[2]); V <- exp(z_log[3])
    if (abs(ka-ke) < 1e-7) ka <- ka*(1+1e-6)
    dt_i <- dose_arr_r[i, , 1]; da_i <- dose_arr_r[i, , 2]
    sapply(t_vec, function(t) {
      valid <- da_i > 0 & dt_i <= t
      if (!any(valid)) return(0)
      sum(da_i[valid]*ka/(V*(ka-ke))*(exp(-ke*(t-dt_i[valid]))-exp(-ka*(t-dt_i[valid]))))
    })
  }
  tald_fn <- if (tald) {
    function(i, t_vec) {
      dt_i <- dose_arr_r[i, , 1]; da_i <- dose_arr_r[i, , 2]
      active_dt <- dt_i[da_i > 0]
      sapply(t_vec, function(t) {
        prior <- active_dt[active_dt <= t]
        if (length(prior) == 0L) return(t)
        t - max(prior)
      })
    }
  } else NULL
  plot_vpc(data, lengths, z_pop, omega_pop, a, b, C, z_dim, sim_fn,
           n_sim=n_sim, nbins=nbins, pi_lo=pi_lo, pi_hi=pi_hi, ci_level=ci_level,
           show_obs=show_obs, seed=seed, dataset_name="theophylline_multiple",
           y_label="Concentration (mg/L)", tald_fn=tald_fn, save_path=save_path)
}

plot_ppc_theo_mult <- function(data, lengths, Encoder, data_in, covariates_in,
                                a, b, dose, z_dim,
                                n_sim=200L, nbins=NULL, pi_lo=0.05, pi_hi=0.95,
                                ci_level=0.95, show_obs=TRUE, seed=42L,
                                tald=TRUE,
                                save_path="Plots/theophylline_multiple_ppc.pdf") {
  dose_arr_r <- as.array(dose$detach())
  sim_fn <- function(i, z_log, t_vec) {
    ka <- exp(z_log[1]); ke <- exp(z_log[2]); V <- exp(z_log[3])
    if (abs(ka-ke) < 1e-7) ka <- ka*(1+1e-6)
    dt_i <- dose_arr_r[i, , 1]; da_i <- dose_arr_r[i, , 2]
    sapply(t_vec, function(t) {
      valid <- da_i > 0 & dt_i <= t
      if (!any(valid)) return(0)
      sum(da_i[valid]*ka/(V*(ka-ke))*(exp(-ke*(t-dt_i[valid]))-exp(-ka*(t-dt_i[valid]))))
    })
  }
  tald_fn <- if (tald) {
    function(i, t_vec) {
      dt_i <- dose_arr_r[i, , 1]; da_i <- dose_arr_r[i, , 2]
      active_dt <- dt_i[da_i > 0]
      sapply(t_vec, function(t) {
        prior <- active_dt[active_dt <= t]
        if (length(prior) == 0L) return(t)
        t - max(prior)
      })
    }
  } else NULL
  plot_ppc(data, lengths, Encoder, data_in, covariates_in, a, b, z_dim, sim_fn,
           n_sim=n_sim, nbins=nbins, pi_lo=pi_lo, pi_hi=pi_hi, ci_level=ci_level,
           show_obs=show_obs, seed=seed, dataset_name="theophylline_multiple",
           y_label="Concentration (mg/L)", tald_fn=tald_fn, save_path=save_path)
}

