# =============================================================================
# encoder.R  —  LSTM Encoder module
# R translation of encoder.py (Jan Rohleff, CPT:PSP 2025)
#
# Architecture (faithful to Python Cython source):
#   - Single LSTM layer with DEFAULT PyTorch init (no special forget-gate or
#     orthogonal init — Python uses none)
#   - Single FC head: [h_last | covariates] -> [mu | log_sigma | L_mask]
#     where L_mask = strictly-lower-triangular off-diagonal elements of L
#     Weight: normal(mean=0, std=1e-3).  Bias: [h_inverse(mu0), sigma0, 0...]
#   - Full lower-triangular Cholesky L: L_kk = exp(log_sigma_k),
#     L_jk (j>k) = L_mask elements  [N, z_dim, z_dim]
#   - Reparameterisation: z = mu + (L @ eps)  (matrix multiply, not elementwise)
#   - Returns L (full lower-triangular), log_sigma, eps
# =============================================================================
library(torch)

lstm_encoder <- nn_module(
  "LSTMEncoder",

  initialize = function(x_dim, h_dim, z_dim, n_cov, mu0, sigma0, h_inverse) {
    self$z_dim     <- z_dim
    self$h_dim     <- h_dim
    self$h_inverse <- h_inverse

    # LSTM — default PyTorch init (no special forget-gate or orthogonal init)
    self$lstm <- nn_lstm(
      input_size  = x_dim,
      hidden_size = h_dim,
      batch_first = TRUE
    )

    # Single FC head: [h_last | covariates] → [mu(z_dim) | log_sigma(z_dim) | L_mask(n_off)]
    n_off    <- as.integer(z_dim * (z_dim - 1L) / 2L)   # strictly-lower-triangular count
    n_out_fc <- 2L * z_dim + n_off                       # total output = z_dim + z_dim*(z_dim+1)/2
    self$n_off    <- n_off
    self$n_out_fc <- n_out_fc

    self$fc <- nn_linear(h_dim + n_cov, n_out_fc)

    # FC weight: normal(0, 1e-3); bias: [h_inverse(mu0), sigma0, zeros(n_off)]
    with_no_grad({
      self$fc$weight$normal_(mean = 0, std = 1e-3)
      bias_init <- torch_cat(list(
        h_inverse(mu0)$detach()$clone(),
        sigma0$detach()$clone(),
        torch_zeros(n_off)
      ))
      self$fc$bias$copy_(bias_init)
    })

    # Pre-compute tril_mask [n_off, z_dim, z_dim] for vectorised L construction.
    # Storing as a plain field (no grad needed — it's a constant indicator matrix).
    # Using matmul in forward avoids all in-place tensor ops inside nn_module,
    # which can corrupt R6 dispatch state in R torch for larger z_dim values.
    if (n_off > 0L) {
      tril_rows <- integer(0); tril_cols <- integer(0)
      for (r in 2L:z_dim) {
        for (cc in 1L:(r - 1L)) {
          tril_rows <- c(tril_rows, r)
          tril_cols <- c(tril_cols, cc)
        }
      }
      tm <- torch_zeros(n_off, z_dim, z_dim)
      for (idx in seq_len(n_off)) {
        tm[idx, tril_rows[idx], tril_cols[idx]] <- 1.0
      }
      self$tril_mask <- tm$detach()   # constant, no gradient
    } else {
      self$tril_mask <- NULL
    }
  },

  forward = function(data_in, covariates_in, lengths) {
    # data_in      : [N, T, x_dim]  standardised (time, conc)
    # covariates_in: [N, n_cov]     standardised covariates
    # lengths      : [N]            number of valid observations per individual

    nbatch   <- data_in$shape[1]
    z_dim    <- self$z_dim
    n_off    <- self$n_off
    n_out_fc <- self$n_out_fc

    # --- run LSTM (default init matches Python) ---------------------------
    lstm_out <- self$lstm(data_in)[[1]]   # [N, T, h_dim]

    # Extract the last valid hidden state via gather (no in-place ops).
    # lengths [N] are 1-indexed (R convention). R torch gather is also 1-indexed,
    # so use lengths directly (do NOT subtract 1 as Python 0-indexed code does).
    # idx_t: [N, 1, h_dim] — same time-step index repeated across h_dim.
    # In-place loops over nbatch cause R torch R6 dispatch corruption for
    # large N (>~50), manifesting as "could not find function 'fn'".
    idx_t  <- lengths$to(dtype = torch_long())$view(c(nbatch, 1L, 1L))$expand(c(nbatch, 1L, self$h_dim))
    h_last <- lstm_out$gather(2L, idx_t)$squeeze(2L)   # [N, h_dim]

    # --- single FC pass ---------------------------------------------------
    combined <- torch_cat(list(h_last, covariates_in), dim = 2)   # [N, h_dim+n_cov]
    out      <- self$fc(combined)                                  # [N, n_out_fc]

    mu        <- out[, 1L:z_dim]                                   # [N, z_dim]
    log_sigma <- out[, (z_dim + 1L):(2L * z_dim)]                  # [N, z_dim]
    L_mask    <- if (n_off > 0L) out[, (2L * z_dim + 1L):n_out_fc] else NULL

    # --- build full lower-triangular Cholesky L (fully out-of-place) ----------
    # [N, n_off] @ [n_off, z_dim^2]  →  [N, z_dim^2]  →  [N, z_dim, z_dim]
    # then add diagonal exp(log_sigma) via diag_embed.
    # This avoids any in-place tensor ops inside nn_module$forward, which
    # can corrupt R6 dispatch in R torch when n_off is large (z_dim ≥ 4).
    sigma <- torch_exp(log_sigma)                                   # [N, z_dim]
    if (!is.null(self$tril_mask) && n_off > 0L) {
      tril_flat <- self$tril_mask$view(c(n_off, z_dim * z_dim))    # [n_off, z_dim^2]
      L_flat    <- torch_matmul(L_mask, tril_flat)                  # [N, z_dim^2]
      L         <- L_flat$view(c(nbatch, z_dim, z_dim)) +
                   torch_diag_embed(sigma)
    } else {
      L <- torch_diag_embed(sigma)
    }

    # --- reparameterisation: z = mu + L @ eps  (matrix multiply) ---------
    eps      <- torch_randn_like(mu)                               # [N, z_dim]
    z_normal <- (mu + torch_matmul(L, eps$unsqueeze(-1L))$squeeze(-1L))

    list(z_normal = z_normal, mu = mu, L = L, log_sigma = log_sigma, eps = eps)
  }
)
