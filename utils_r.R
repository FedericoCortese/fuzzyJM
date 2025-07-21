fuzzy_jump_cpp <- function(Y, 
                           K, 
                           lambda = 1e-5, 
                           m = 1,
                           max_iter = 5, 
                           n_init = 10, 
                           tol = 1e-16, 
                           verbose = FALSE,
                           parallel = FALSE,
                           n_cores = NULL
) {
  # Fit jump model for mixed‐type data 
  #
  # Arguments:
  #   Y            - data.frame with mixed data types (categorical vars must be factors)
  #   K            - number of states
  #   lambda       - penalty for the number of jumps
  #   m            - fuzziness exponent (for soft membership)
  #   max_iter     - maximum number of iterations per initialization
  #   n_init       - number of random initializations
  #   tol          - convergence tolerance
  #   verbose      - print progress per iteration (TRUE/FALSE)
  #   parallel     - if TRUE, use mclapply for parallel initializations
  #   n_cores      - number of cores for mclapply; if NULL, uses detectCores() - 1
  #
  # Value:
  #   List with:
  #     best_S      - TT×K soft‐membership matrix from best initialization
  #     best_mu     - K×P state‐conditional weighted medians/modes(computed on scaled data)
  #     loss        - best objective value
  #     MAP         - re‐ordered state sequence (1..K)
  #     feature_types - vector indicating feature types (0 for continuous, 1 for categorical)
  
  K <- as.integer(K)
  TT <- nrow(Y)
  P  <- ncol(Y)
  
  Rcpp::sourceCpp("utils_c.cpp")
  
  # Get feature types vector
  feature_types <- sapply(Y, class)
  # Convert feature_types to integer (0 for continuous, 1 for categorical)
  feature_types <- as.integer(feature_types == "factor" | feature_types == "character")
  
  # Standardize only continuous features
  for (j in seq_len(P)) {
    if (feature_types[j] == 0) {  # Continuous feature
      Y[[j]] <- scale(Y[[j]], center = TRUE, scale = TRUE)
    } 
  }
  
  # Transform into a matrix
  if(!is.matrix(Y)){
    Y=as.matrix(
      data.frame(
        lapply(Y, function(col) {
          if (is.factor(col)||is.character(col)) as.integer(as.character(col))
          else              as.numeric(col)
        })
      )
    )
  }
  
  
  run_one <- function(init) {
    # Single initialization
    # 1) Initialize hard states via k‐prototypes++ 
    s <- initialize_states(Y, K)
    S <- matrix(0, nrow = TT, ncol = K)
    S[cbind(seq_len(TT), s)] <- 1L
    
    # 2) Initialize mu: state‐conditional medians/modes
    mu <- matrix(NA_real_, nrow = K, ncol = P, 
                 dimnames = list(NULL, colnames(Y)))
    
    for (k in seq_len(K)) {
      w <- S[, k]^m
      for (p in seq_len(P)) {
        if (feature_types[p] == 0L) {
          mu[k, p] <- as.numeric(poliscidata::wtd.median(Y[, p], weights = w))
        } else {
          mu[k, p] <- as.numeric(poliscidata::wtd.mode(Y[, p], weights = w))
        }
      }
    }
    
    S_old <- S
    V <- gower_dist(Y, mu)
    loss_old <- sum(V * (S^m)) + lambda * sum(abs(S[1:(TT-1), ] - S[2:TT, ])^2)
    
    for (it in seq_len(max_iter)) {
      
      # Update S[1, ]
      S[1, ] <- optimize_pgd_1T(
        init   = rep(1/K, K),
        g      = V[1, ],
        s_t1   = S[2, ],
        lambda = lambda,
        m      = m
      )$par
      
      # Update S[2:(TT-1), ]
      if (TT > 2) {
        for (t in 2:(TT - 1)) {
          S[t, ] <- optimize_pgd_2T(
            init    = rep(1/K, K),
            g       = V[t, ],
            s_prec  = S[t - 1, ],
            s_succ  = S[t + 1, ],
            lambda  = lambda,
            m       = m
          )$par
        }
      }
      
      # Update S[TT, ]
      S[TT, ] <- optimize_pgd_1T(
        init   = rep(1/K, K),
        g      = V[TT, ],
        s_t1   = S[TT - 1, ],
        lambda = lambda,
        m      = m
      )$par
      
      # Recompute mu
      for (k in seq_len(K)) {
        w <- S[, k]^m
        for (p in seq_len(P)) {
          if (feature_types[p] == 0L) {
            mu[k, p] <- as.numeric(poliscidata::wtd.median(Y[, p], weights = w))
          } else {
            mu[k, p] <- as.numeric(poliscidata::wtd.mode(Y[, p], weights = w))
          }
        }
      }
      
      # Recompute distances and loss
      
      V <- gower_dist(Y, mu)
      loss <- sum(V * (S^m)) + lambda * sum(abs(S[1:(TT-1), ] - S[2:TT, ])^2)
      
      if (verbose) cat(sprintf("Initialization %d, Iteration %d: loss = %.6e\n", init, it, loss))
      
      # Check convergence
      if (!is.null(tol)) {
        if ((loss_old - loss) < tol) break
      } else if (all(S == S_old)) {
        break
      }
      loss_old <- loss
      S_old <- S
    }
    
    list(S = S, loss = loss_old, mu = mu)
  }
  
  # Choose apply function based on parallel flag
  if (parallel) {
    library(parallel)
    if (is.null(n_cores)) {
      n_cores <- max(detectCores() - 1, 1)
    }
    res_list <- mclapply(seq_len(n_init), run_one, mc.cores = n_cores)
  } else {
    res_list <- lapply(seq_len(n_init), run_one)
  }
  
  # Find best initialization
  losses <- vapply(res_list, function(x) x$loss, numeric(1))
  best_idx <- which.min(losses)
  best_run <- res_list[[best_idx]]
  best_S   <- best_run$S
  best_loss<- best_run$loss
  best_mu  <- best_run$mu
  
  # Compute MAP and re‐order states
  old_MAP <- apply(best_S, 1, which.max)
  MAP <- order_states_condMed(Y[, 1], old_MAP)
  tab <- table(MAP, old_MAP)
  new_order <- apply(tab, 1, which.max)
  best_S <- best_S[, new_order]
  
  
  return(list(
    best_S    = best_S,
    best_mu = best_mu,
    loss      = best_loss,
    MAP       = MAP,
    feature_types = feature_types
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
