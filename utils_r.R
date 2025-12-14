fuzzy_jump_cpp <- function(Y, 
                           K, 
                           lambda = 1e-5, 
                           m = 1,
                           max_iter = 5, 
                           n_init = 10, 
                           tol = 1e-16, 
                           verbose = FALSE,
                           parallel = FALSE,
                           n_cores = NULL,
                           pen_exp = 2,
                           s_init = NULL) {
  # Fit fuzzy jump model for mixed‐type time series with Gower dissimilarity and L^{pen_exp} jump penalty
  #
  # Arguments:
  #   Y        - data.frame or matrix with mixed data types (categorical variables as factor/character)
  #   K        - number of states
  #   lambda   - jump penalty hyperparameter
  #   m        - fuzziness exponent for soft memberships
  #   max_iter - maximum number of iterations per initialization
  #   n_init   - number of initializations
  #   tol      - convergence tolerance (set NULL to disable)
  #   verbose  - logical; print progress and diagnostics
  #   parallel - logical; use mclapply for multiple initializations
  #   n_cores  - number of cores when parallel = TRUE; if NULL uses detectCores() - 1
  #   pen_exp  - exponent of the L^{pen_exp} penalty on the temporal total variation of S
  #   s_init   - optional vector of initial hard states of length TT
  #
  # Value:
  #   A list with components:
  #     best_S        - TT × K soft‐membership matrix from best initialization
  #     best_mu       - K × P state‐conditional prototypes (weighted medians/modes on scaled data)
  #     loss          - best objective value
  #     MAP           - re‐ordered state sequence (1..K)
  #     Y             - processed data matrix used in fitting
  #     feature_types - integer vector (0 = continuous, 1 = categorical)
  #     all_losses    - objective values along the iterations of the selected run
  
  lambda <- lambda / (2 ^ pen_exp)
  Rcpp::sourceCpp("utils_c.cpp")
  
  K  <- as.integer(K)
  TT <- nrow(Y)
  P  <- ncol(Y)
  
  feature_types <- sapply(Y, class)
  feature_types <- as.integer(feature_types == "factor" | feature_types == "character")
  
  for (j in seq_len(P)) {
    if (feature_types[j] == 0) {
      Y[[j]] <- scale(Y[[j]], center = TRUE, scale = TRUE)
    }
  }
  
  if (!is.matrix(Y)) {
    Y <- as.matrix(
      data.frame(
        lapply(Y, function(col) {
          if (is.factor(col) || is.character(col)) as.integer(as.character(col))
          else                                     as.numeric(col)
        })
      )
    )
  }
  
  run_one <- function(init) {
    all_losses <- NULL
    all_tv     <- NULL
    
    if (sum(feature_types) == 0) {
      temp <- cont_jump(
        Y           = as.matrix(Y),
        K           = K,
        jump_penalty= lambda,
        alpha       = pen_exp,
        mode_loss   = FALSE,
        n_init      = 5,
        grid_size   = 0.1
      )
      S <- temp$best_S
    } else {
      if (!is.null(s_init)) {
        s <- s_init
      } else {
        s <- initialize_states(Y, K)
      }
      S <- matrix(0, nrow = TT, ncol = K)
      S[cbind(seq_len(TT), s)] <- 1L
    }
    
    mu <- matrix(NA_real_, nrow = K, ncol = P, dimnames = list(NULL, colnames(Y)))
    
    global_median <- numeric(P)
    global_mode   <- numeric(P)
    
    for (p in seq_len(P)) {
      x <- Y[, p]
      if (feature_types[p] == 0L) {
        global_median[p] <- stats::median(x, na.rm = TRUE)
        global_mode[p]   <- NA_real_
      } else {
        tab_x <- table(x)
        if (length(tab_x) == 0L) {
          global_mode[p] <- NA_real_
        } else {
          global_mode[p] <- as.numeric(names(which.max(tab_x)))
        }
        global_median[p] <- NA_real_
      }
    }
    
    for (k in seq_len(K)) {
      w <- S[, k]^m
      w[is.na(w)] <- 0
      sw <- sum(w)
      
      for (p in seq_len(P)) {
        x <- Y[, p]
        
        if (feature_types[p] == 0L) {
          if (is.na(sw) || sw < 1e-12 || all(is.na(x))) {
            mu[k, p] <- global_median[p]
          } else {
            mu[k, p] <- as.numeric(
              suppressWarnings(poliscidata::wtd.median(x, weights = w))
            )
          }
        } else {
          if (is.na(sw) || sw < 1e-12 || all(is.na(x))) {
            mu[k, p] <- global_mode[p]
          } else {
            mu[k, p] <- as.numeric(
              suppressWarnings(poliscidata::wtd.mode(x, weights = w))
            )
          }
        }
      }
    }
    
    S_old <- S
    V <- gower_dist(Y, mu, feat_type = feature_types)
    
    if (TT > 1) {
      dS <- S[2:TT, , drop = FALSE] - S[1:(TT - 1), , drop = FALSE]
      tv_old <- sum(rowSums(abs(dS))^pen_exp)
    } else {
      tv_old <- 0
    }
    loss_old <- sum(V * (S^m)) + lambda * tv_old
    all_losses <- c(all_losses, loss_old)
    all_tv     <- c(all_tv, tv_old)
    
    for (it in seq_len(max_iter)) {
      if (TT > 1) {
        S[1, ] <- optimize_pgd_1T(
          init    = S[1, ],
          g       = V[1, ],
          s_t1    = S[2, ],
          lambda  = lambda,
          m       = m,
          pen_exp = pen_exp
        )$par
      } else {
        S[1, ] <- optimize_pgd_1T(
          init    = S[1, ],
          g       = V[1, ],
          s_t1    = S[1, ],
          lambda  = 0,
          m       = m,
          pen_exp = pen_exp
        )$par
      }
      
      if (TT > 2) {
        for (t in 2:(TT - 1)) {
          S[t, ] <- optimize_pgd_2T(
            init    = S[t, ],
            g       = V[t, ],
            s_prec  = S[t - 1, ],
            s_succ  = S[t + 1, ],
            lambda  = lambda,
            m       = m,
            pen_exp = pen_exp
          )$par
        }
      }
      
      if (TT > 1) {
        S[TT, ] <- optimize_pgd_1T(
          init    = S[TT, ],
          g       = V[TT, ],
          s_t1    = S[TT - 1, ],
          lambda  = lambda,
          m       = m,
          pen_exp = pen_exp
        )$par
      }
      
      S <- S / rowSums(S)
      
      for (k in seq_len(K)) {
        w <- S[, k]^m
        w[is.na(w)] <- 0
        
        for (p in seq_len(P)) {
          x <- Y[, p]
          
          if (feature_types[p] == 0L) {
            if (sum(w) < 1e-12 || all(is.na(x))) {
              mu[k, p] <- stats::median(x, na.rm = TRUE)
            } else {
              mu[k, p] <- as.numeric(
                suppressWarnings(poliscidata::wtd.median(x, weights = w))
              )
            }
          } else {
            if (sum(w) < 1e-12 || all(is.na(x))) {
              tab_x <- table(x)
              if (length(tab_x) == 0L) {
                mu[k, p] <- NA_real_
              } else {
                mu[k, p] <- as.numeric(names(which.max(tab_x)))
              }
            } else {
              mu[k, p] <- as.numeric(
                suppressWarnings(poliscidata::wtd.mode(x, weights = w))
              )
            }
          }
        }
      }
      
      V  <- gower_dist(Y, mu, feat_type = feature_types)
      if (TT > 1) {
        dS <- S[2:TT, , drop = FALSE] - S[1:(TT - 1), , drop = FALSE]
        tv <- sum(rowSums(abs(dS))^pen_exp)
      } else {
        tv <- 0
      }
      loss <- sum(V * (S^m)) + lambda * tv
      
      all_losses <- c(all_losses, loss)
      all_tv     <- c(all_tv, tv)
      if (verbose)
        cat(sprintf("Init %d, Iter %d: loss = %.6e, TV = %.6e\n",
                    init, it, loss, tv))
      
      improv <- loss_old - loss
      if (!is.null(tol)) {
        if (improv >= 0 && improv < tol) break
        if (sum(abs(S - S_old)^pen_exp) < tol) {
          break
        }
      }
      
      loss_old <- loss
      tv_old   <- tv
      S_old    <- S
    }
    
    list(
      S          = S,
      mu         = mu,
      loss       = loss_old,
      all_losses = all_losses,
      all_tv     = all_tv
    )
  }
  
  if (parallel) {
    library(parallel)
    if (is.null(n_cores)) {
      n_cores <- max(detectCores() - 1, 1)
    }
    res_list <- mclapply(seq_len(n_init), run_one, mc.cores = n_cores)
  } else {
    res_list <- lapply(seq_len(n_init), run_one)
  }
  
  losses   <- vapply(res_list, function(x) x$loss, numeric(1))
  best_idx <- which.min(losses)
  best_run <- res_list[[best_idx]]
  best_S   <- best_run$S
  best_loss<- best_run$loss
  best_mu  <- best_run$mu
  all_losses <- best_run$all_losses
  
  V_best <- gower_dist(Y, best_mu, feat_type = feature_types)
  
  if (TT > 1) {
    dS <- best_S[2:TT, , drop = FALSE] - best_S[1:(TT - 1), , drop = FALSE]
    tv_best <- sum(rowSums(abs(dS))^pen_exp)
  } else {
    tv_best <- 0
  }
  loss_best_check <- sum(V_best * (best_S^m)) + lambda * tv_best
  
  Gk      <- colSums(V_best)
  k_star  <- which.min(Gk)
  S_const <- matrix(0, nrow = TT, ncol = K)
  S_const[, k_star] <- 1
  
  if (TT > 1) {
    dS <- S_const[2:TT, , drop = FALSE] - S_const[1:(TT - 1), , drop = FALSE]
    tv_const <- sum(rowSums(abs(dS))^pen_exp)
  } else {
    tv_const <- 0
  }
  loss_const <- sum(V_best * S_const) + lambda * tv_const
  
  if (verbose) {
    cat("---- CHECK percorso costante ----\n")
    cat(sprintf("  best_idx      = %d\n", best_idx))
    cat(sprintf("  loss_best     = %.6e (stored)\n", best_loss))
  }
  
  old_MAP <- apply(best_S, 1, which.max)
  MAP <- order_states_condMed(Y[, 1], old_MAP)
  tab <- table(MAP, old_MAP)
  new_order <- apply(tab, 1, which.max)
  best_S  <- best_S[, new_order, drop = FALSE]
  best_mu <- best_mu[new_order, , drop = FALSE]
  
  best_S <- best_S / rowSums(best_S)
  
  return(list(
    best_S        = best_S,
    best_mu       = best_mu,
    loss          = best_loss,
    MAP           = MAP,
    Y             = Y,
    feature_types = feature_types,
    all_losses    = all_losses
  ))
}



order_states_condMed=function(y,s){
  
  # This function organizes states by assigning 1 to the state with the smallest conditional median for vector y
  # and sequentially numbering each new state as 2, 3, etc., incrementing by 1 for each newly observed state.
  
  condMed=sort(tapply(y,s,median,na.rm=T))
  
  states_temp=match(s,names(condMed))
  
  return(states_temp)
}

simulate_fuzzy_mixture_mv<- function(
    TT = 1000,
    P  = 2,
    K  = 2,
    mu = 1,
    Sigma_rho = 0,
    ar_rho    = 0.9,
    tau       = 0.5,
    seed      = NULL
) {
  
  # This function generates multivariate Gaussian mixture data with fuzzy (soft) cluster memberships
  # that evolve over time following a transformed autoregressive process.
  #
  # Arguments:
  # TT        - Integer. Number of time points (default: 1000)
  # P         - Integer. Number of features (default: 2)
  # K         - Integer. Number of clusters (default: 2)
  # mu        - Numeric. Controls the spacing of the K cluster means from -mu to +mu (default: 1)
  # Sigma_rho - Numeric. Off-diagonal value of the shared covariance matrix (default: 0)
  # ar_rho    - Numeric. Autoregressive coefficient for the latent score dynamics (default: 0.9)
  # tau       - Numeric. Standard deviation of innovations in the AR(1) process (default: 0.5)
  # seed      - Integer or NULL. Optional random seed for reproducibility (default: NULL)
  #
  # Returns:
  # A data.frame with TT rows and the following columns:
  # - time: time index
  # - Y1, ..., YP: observed multivariate data
  # - alpha_1, ..., alpha_K: latent AR(1) scores before softmax
  # - pi_1, ..., pi_K: softmax-transformed cluster probabilities
  # - MAP: most likely cluster assignment at each time point (argmax of pi)
  
  
  if (!is.null(seed)) set.seed(seed)
  if (!requireNamespace("MASS", quietly = TRUE)) {
    stop("Package 'MASS' is required but not installed.")
  }
  
  # --- Generate K centroids spaced from -mu to +mu ---
  mu_vals <- seq(-mu, mu, length.out = K)
  mus     <- matrix(mu_vals, nrow = K, ncol = P, byrow = FALSE)
  
  # --- Common covariance ---
  Sigma <- matrix(Sigma_rho, nrow = P, ncol = P)
  diag(Sigma) <- 1
  
  # --- Storage ---
  alpha_mat <- matrix(0, nrow = TT, ncol = K)
  pi_mat    <- matrix(0, nrow = TT, ncol = K)
  y_mat     <- matrix(0, nrow = TT, ncol = P)
  
  # --- Initialize latent scores ---
  alpha_mat[1, ] <- rnorm(K, 0, tau)
  pi_mat[1, ]    <- exp(alpha_mat[1, ]) / sum(exp(alpha_mat[1, ]))
  
  # --- Simulate AR(1) + softmax weights ---
  for (t in 2:TT) {
    alpha_mat[t, ] <- ar_rho * alpha_mat[t - 1, ] + rnorm(K, 0, tau)
    ealpha <- exp(alpha_mat[t, ])
    pi_mat[t, ] <- ealpha / sum(ealpha)
  }
  
  # --- Draw from mixture ---
  for (t in 1:TT) {
    k_t       <- which.max(rmultinom(1, size = 1, prob = pi_mat[t, ]))
    y_mat[t,] <- MASS::mvrnorm(1, mu = mus[k_t, ], Sigma = Sigma)
  }
  
  # --- Build output ---
  MAP <- max.col(pi_mat)
  df <- data.frame(
    time      = seq_len(TT),
    y_mat,
    alpha_mat,
    pi_mat,
    MAP       = MAP
  )
  
  # Name columns
  names(df)[2:(1+P)]          <- paste0("Y",   1:P)
  names(df)[(2+P):(1+P+K)]    <- paste0("alpha_",1:K)
  names(df)[(2+P+K):(1+P+2*K)]<- paste0("pi_",   1:K)
  
  return(df)
}
