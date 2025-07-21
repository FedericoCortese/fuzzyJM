# Source utils file (remember to save 'utils_c.cpp' in the same folder)
source("utils_r.R")

# Simulate data with high uncertainty in cluster assignment and high persistence
TT=1000
P=10
simDat=simulate_fuzzy_mixture_mv(
  TT = TT,
  P  = P,
  K  = 3,
  mu = 2,
  Sigma_rho = 0,
  ar_rho    = 0.99,
  tau       = 0.5,
  seed      = 123
)

Y=simDat[,2:11]  # Data

# Transform some features into categorical variables

Y$Y9=round(Y$Y9)
Y$Y9=as.factor(Y$Y9)
Y$Y10=round(Y$Y10)
Y$Y10=as.factor(Y$Y10)

str(Y)

# Fit fuzzy JM with soft assignment (m=1.25)

m=1.25
K=3
lambda=.5
fit=fuzzy_jump_cpp(Y=Y,
                   K=K,
                   m=m,
                   lambda=lambda,
                   n_init=5,
                   max_iter=10,
                   tol=1e-8,
                   verbose=T)

# Analyze results

# State conditional probabilities
S=fit$best_S
x11()
par(mfrow=c(2,1))
matplot(S,type='l',ylab=" ",xlab="Time",main="fuzzy JM state probabilities")
matplot(simDat[,c("pi_1","pi_2","pi_3")],type='l',ylab=" ",xlab="Time",
        main="True state probabilities")

# State conditional weighted prototypes

feature_types <- fit$feature_types
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

mu