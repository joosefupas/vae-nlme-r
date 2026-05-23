# =============================================================================
# run_all.R  —  Batch runner for all VAE-NLME case studies
#
# Runs every main script sequentially inside an isolated environment so that
# one failing case study does not abort the rest.  Results are saved to
# Results/ and plots to Plots/ as usual.
#
# Usage (from R):
#   source("run_all.R")
#
# Usage (Rscript CLI):
#   Rscript run_all.R
#   Rscript run_all.R theophylline warfarin   # run a subset by name
# =============================================================================

# ---- resolve project root (same logic as individual scripts) ----------------
.resolve_root <- function() {
  for (i in seq_len(sys.nframe())) {
    f <- sys.frame(i)$ofile
    if (!is.null(f) && nchar(f))
      return(normalizePath(file.path(dirname(f), "."), winslash = "/"))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- tryCatch(rstudioapi::getActiveDocumentContext()$path, error = function(e) "")
    if (nchar(p))
      return(normalizePath(dirname(p), winslash = "/"))
  }
  wd <- getwd()
  if (basename(wd) == "Main") return(normalizePath("..", winslash = "/"))
  wd
}
root_dir <- .resolve_root()
setwd(root_dir)

# ---- ordered list of case studies -------------------------------------------
all_studies <- c(
  #"theophylline",
  "theophylline_multiple",
  "neonates",
  "warfarin",
  "pheno_sd",
  "mavoglurant",
  "nimoData"
)

# Allow CLI subset:  Rscript run_all.R theophylline warfarin
args <- commandArgs(trailingOnly = TRUE)
studies <- if (length(args) > 0) intersect(args, all_studies) else all_studies

if (length(studies) == 0) stop("No recognised study names in arguments.")

cat(sprintf("\n=== VAE-NLME batch run: %d case stud%s ===\n\n",
            length(studies), if (length(studies) == 1) "y" else "ies"))

# ---- helpers ----------------------------------------------------------------
.hline <- function() cat(paste0(rep("=", 60), collapse = ""), "\n")

results <- data.frame(
  study   = character(),
  status  = character(),
  elapsed = numeric(),
  message = character(),
  stringsAsFactors = FALSE
)

# ---- locate Rscript executable -----------------------------------------------
# Prefer the Rscript that launched this session; fall back to PATH.
.rscript <- function() {
  r_home <- R.home("bin")
  candidates <- c(
    file.path(r_home, "Rscript.exe"),   # Windows
    file.path(r_home, "Rscript")        # Unix
  )
  found <- candidates[file.exists(candidates)]
  if (length(found)) return(normalizePath(found[1], winslash = "/"))
  "Rscript"   # rely on PATH
}
RSCRIPT <- .rscript()

# ---- run each script in a subprocess ----------------------------------------
#
# Why subprocess and not source() / eval(parse())?
#
# source("run_all.R") pushes an ofile frame onto the call stack for run_all.R
# (located in VAE_R/).  Each individual script's .resolve_root() scans frames
# oldest-first and finds *this* frame before its own, then computes:
#   dirname("…/VAE_R/run_all.R") + ".."  →  "…/Projects/"   ← wrong!
#
# A fresh Rscript subprocess has no inherited frame stack.  R sets the
# subprocess's ofile to the script being run (e.g., "…/VAE_R/Main/foo.R"),
# so .resolve_root() correctly returns dirname + ".."  →  "…/VAE_R/".
for (study in studies) {
  script <- file.path(root_dir, "Main", paste0(study, ".R"))

  if (!file.exists(script)) {
    cat(sprintf("[SKIP]  %s — script not found: %s\n", study, script))
    results <- rbind(results, data.frame(
      study = study, status = "SKIP", elapsed = 0,
      message = "script not found", stringsAsFactors = FALSE))
    next
  }

  .hline()
  cat(sprintf(">>> Starting: %s\n", study))
  cat(sprintf("    Script : %s\n", script))
  cat(sprintf("    Time   : %s\n\n", format(Sys.time(), "%H:%M:%S")))

  t0 <- proc.time()["elapsed"]

  # The subprocess's sys.nframe() == 0 (top-level execution), so the ofile
  # frame loop in .resolve_root() never iterates.  We set cwd to Main/ before
  # launching so the getwd() fallback fires:
  #   basename(getwd()) == "Main"  →  normalizePath("..")  →  VAE_R/  ✓
  old_wd <- getwd()
  setwd(dirname(script))   # → .../VAE_R/Main/

  # Capture stdout+stderr so they're visible in RStudio's console.
  # (system2 with stdout/stderr="" can go silent inside RStudio.)
  out <- tryCatch(
    system2(
      RSCRIPT,
      args   = c("--no-save", "--no-restore",
                 shQuote(normalizePath(script, winslash = "/"))),
      stdout = TRUE,   # capture as character vector
      stderr = TRUE    # merge stderr into the same vector
    ),
    error   = function(e) { structure(conditionMessage(e), status = 1L) },
    finally = setwd(old_wd)
  )

  ret <- attr(out, "status")
  if (is.null(ret)) ret <- 0L      # exit code (0 = success)
  if (length(out)) cat(paste(out, collapse = "\n"), "\n")

  ok <- (ret == 0)

  elapsed <- round(proc.time()["elapsed"] - t0, 1)

  status  <- if (isTRUE(ok)) "OK" else "FAILED"
  msg     <- if (isTRUE(ok)) "" else "see console for error"

  cat(sprintf("\n<<< Finished: %s  [%s]  %.0f s\n\n", study, status, elapsed))

  results <- rbind(results, data.frame(
    study = study, status = status, elapsed = elapsed,
    message = msg, stringsAsFactors = FALSE))
}

# ---- summary ----------------------------------------------------------------
.hline()
cat("\nBATCH RUN SUMMARY\n\n")
cat(sprintf("  %-30s  %-8s  %8s\n", "Study", "Status", "Elapsed"))
cat(sprintf("  %s\n", paste(rep("-", 50), collapse = "")))
for (i in seq_len(nrow(results))) {
  r <- results[i, ]
  cat(sprintf("  %-30s  %-8s  %7.0f s%s\n",
              r$study, r$status, r$elapsed,
              if (nchar(r$message) > 0) paste0("  [", r$message, "]") else ""))
}

n_ok     <- sum(results$status == "OK")
n_failed <- sum(results$status == "FAILED")
n_skip   <- sum(results$status == "SKIP")
total    <- round(sum(results$elapsed))

cat(sprintf("\n  Total: %d OK, %d failed, %d skipped — %.0f s (%.1f min)\n\n",
            n_ok, n_failed, n_skip, total, total / 60))
.hline()

invisible(results)
