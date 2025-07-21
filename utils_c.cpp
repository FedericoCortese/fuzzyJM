// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
using namespace Rcpp;

// Helper: project v onto the probability simplex { x >= 0, sum x = 1 }
NumericVector proj_simplex(const NumericVector& v) {
  int n = v.size();
  NumericVector u = clone(v);
  std::sort(u.begin(), u.end(), std::greater<double>());
  NumericVector css(n);
  css[0] = u[0];
  for (int i = 1; i < n; ++i) {
    css[i] = css[i - 1] + u[i];
  }
  double rho = 0, theta = 0;
  for (int i = n - 1; i >= 0; --i) {
    double t = (css[i] - 1.0) / (i + 1);
    if (u[i] > t) {
      rho = i;
      theta = t;
      break;
    }
  }
  NumericVector w(n);
  for (int i = 0; i < n; ++i) {
    w[i] = std::max(v[i] - theta, 0.0);
  }
  return w;
}

// Compute objective for the 1T case: sum(s^m * g) + lambda * sum((s_t_1 - s)^2)
double eval_obj_1T(const NumericVector& s,
                   const NumericVector& g,
                   const NumericVector& s_t1,
                   double lambda,
                   double m) {
  int K = s.size();
  double val = 0.0, pen = 0.0;
  for (int i = 0; i < K; ++i) {
    val += std::pow(s[i], m) * g[i];
    double diff = s_t1[i] - s[i];
    pen += diff * diff;
  }
  return val + lambda * pen;
}

// Compute objective for the 2T case: sum(s^m * g) + lambda*(sum((s_t_prec - s)^2) + sum((s_t_succ - s)^2))
double eval_obj_2T(const NumericVector& s,
                   const NumericVector& g,
                   const NumericVector& s_prec,
                   const NumericVector& s_succ,
                   double lambda,
                   double m) {
  int K = s.size();
  double val = 0.0, pen = 0.0;
  for (int i = 0; i < K; ++i) {
    val += std::pow(s[i], m) * g[i];
    double d1 = s_prec[i] - s[i];
    double d2 = s_succ[i] - s[i];
    pen += d1 * d1 + d2 * d2;
  }
  return val + lambda * pen;
}

// [[Rcpp::export]]
List optimize_pgd_1T(NumericVector init,
                     NumericVector g,
                     NumericVector s_t1,
                     double lambda,
                     double m,
                     int max_iter = 1000,
                     double alpha    = 1e-2,
                     double tol      = 1e-8) {
  int K = init.size();
  NumericVector s = clone(init);
  for (int iter = 0; iter < max_iter; ++iter) {
    // gradient: m*s^(m-1)*g - 2*lambda*(s_t1 - s)
    NumericVector grad(K);
    for (int i = 0; i < K; ++i) {
      double diff = s_t1[i] - s[i];
      grad[i] = m * std::pow(s[i], m - 1) * g[i] - 2.0 * lambda * diff;
    }
    NumericVector s_new = proj_simplex(s - alpha * grad);
    if (max(abs(s_new - s)) < tol) {
      s = s_new;
      break;
    }
    s = s_new;
  }
  double obj = eval_obj_1T(s, g, s_t1, lambda, m);
  return List::create(
    _["par"]   = s,
    _["value"] = obj
  );
}

// [[Rcpp::export]]
List optimize_pgd_2T(NumericVector init,
                     NumericVector g,
                     NumericVector s_prec,
                     NumericVector s_succ,
                     double lambda,
                     double m,
                     int max_iter = 1000,
                     double alpha    = 1e-2,
                     double tol      = 1e-8) {
  int K = init.size();
  NumericVector s = clone(init);
  for (int iter = 0; iter < max_iter; ++iter) {
    // gradient: m*s^(m-1)*g - 2*lambda*((s_prec - s) + (s_succ - s))
    NumericVector grad(K);
    for (int i = 0; i < K; ++i) {
      double d1 = s_prec[i] - s[i];
      double d2 = s_succ[i] - s[i];
      grad[i] = m * std::pow(s[i], m - 1) * g[i] - 2.0 * lambda * (d1 + d2);
    }
    NumericVector s_new = proj_simplex(s - alpha * grad);
    if (max(abs(s_new - s)) < tol) {
      s = s_new;
      break;
    }
    s = s_new;
  }
  double obj = eval_obj_2T(s, g, s_prec, s_succ, lambda, m);
  return List::create(
    _["par"]   = s,
    _["value"] = obj
  );
}


// [[Rcpp::export]]
NumericMatrix gower_dist(const NumericMatrix& Y,
                         const NumericMatrix& mu,
                         Nullable<IntegerVector> feat_type = R_NilValue,
                         std::string scale = "m" // default = "m" (max-min), "i"=IQR/1.35, "s"=std-dev
) {
  int n = Y.nrow();            // observations
  int p = Y.ncol();            // features
  int m = mu.nrow();           // prototypes
  
  // 1. Handle NULL feat_type -> all continuous
  IntegerVector ft;
  if (feat_type.isNotNull()) {
    ft = feat_type.get();
    if (ft.size() != p)
      stop("feat_type must have length = ncol(Y)");
  } else {
    ft = IntegerVector(p, 0);
  }
  
  // 2. Compute s_p: scale for continuous features per 'scale' flag
  std::vector<double> s_p(p);
  for (int j = 0; j < p; ++j) {
    if (ft[j] == 0) {
      // continuous feature
      std::vector<double> col(n);
      for (int i = 0; i < n; ++i) col[i] = Y(i, j);
      if (scale == "m") {
        // max-min
        double mn = col[0], mx = col[0];
        for (double v: col) {
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
        s_p[j] = mx - mn;
      } else if (scale == "i") {
        // IQR / 1.35
        std::sort(col.begin(), col.end());
        double q1 = col[(int)std::floor((n - 1) * 0.25)];
        double q3 = col[(int)std::floor((n - 1) * 0.75)];
        s_p[j] = (q3 - q1) / 1.35;
      } else if (scale == "s") {
        // standard deviation
        double sum = 0.0;
        for (double v: col) sum += v;
        double mu_col = sum / n;
        double ss = 0.0;
        for (double v: col) ss += (v - mu_col) * (v - mu_col);
        s_p[j] = std::sqrt(ss / (n - 1));
      } else {
        stop("Invalid scale flag: must be 'm', 'i', or 's'");
      }
      if (s_p[j] == 0.0) s_p[j] = 1.0;
    } else {
      // categorical or ordinal
      s_p[j] = 1.0;
    }
  }
  
  // 3. Precompute ordinal levels and ranks
  std::vector< std::vector<int> > ord_rank_Y(p);
  std::vector< std::vector<int> > ord_rank_mu(p);
  std::vector<int> M(p, 1);
  for (int j = 0; j < p; ++j) {
    if (ft[j] == 2) {
      std::vector<double> vals;
      vals.reserve(n + m);
      for (int i = 0; i < n; ++i) vals.push_back(Y(i, j));
      for (int u = 0; u < m; ++u) vals.push_back(mu(u, j));
      std::sort(vals.begin(), vals.end());
      vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
      int levels = vals.size();
      M[j] = levels > 1 ? levels : 1;
      ord_rank_Y[j].resize(n);
      ord_rank_mu[j].resize(m);
      for (int i = 0; i < n; ++i) {
        ord_rank_Y[j][i] = std::lower_bound(vals.begin(), vals.end(), Y(i, j)) - vals.begin();
      }
      for (int u = 0; u < m; ++u) {
        ord_rank_mu[j][u] = std::lower_bound(vals.begin(), vals.end(), mu(u, j)) - vals.begin();
      }
    }
  }
  
  // 4. Compute Gower distances
  NumericMatrix V(n, m);
  for (int i = 0; i < n; ++i) {
    for (int u = 0; u < m; ++u) {
      double acc = 0.0;
      for (int j = 0; j < p; ++j) {
        double diff;
        if (ft[j] == 0) {
          // continuous
          diff = std::abs(Y(i, j) - mu(u, j)) / s_p[j];
        } else if (ft[j] == 1) {
          // categorical
          diff = (Y(i, j) != mu(u, j)) ? 1.0 : 0.0;
        } else {
          // ordinal
          double denom = double(M[j] - 1);
          diff = denom > 0.0 ? std::abs(ord_rank_Y[j][i] - ord_rank_mu[j][u]) / denom : 0.0;
        }
        acc += diff;
      }
      V(i, u) = acc / p;  // mean over features
    }
  }
  
  return V;
}

// [[Rcpp::export]]
IntegerVector initialize_states(const NumericMatrix& Y,
                                int K,
                                Nullable<IntegerVector> feat_type = R_NilValue,
                                int reps = 10,
                                std::string scale = "m") {
  int TT = Y.nrow();
  int P  = Y.ncol();
  
  // handle feat_type
  IntegerVector ft;
  if (feat_type.isNotNull()) {
    ft = feat_type.get();
    if ((int)ft.size() != P) stop("feat_type must have length = ncol(Y)");
  } else {
    ft = IntegerVector(P, 0);
  }
  
  // Precompute full Gower distance matrix Y->Y
  NumericMatrix Dall = gower_dist(Y, Y, ft, scale);
  
  double best_sum = R_PosInf;
  IntegerVector best_assign(TT);
  
  // Repeat multiple times
  for (int rep = 0; rep < reps; ++rep) {
    // 1) Initialize centroids indices via kmeans++
    std::vector<int> centIdx;
    centIdx.reserve(K);
    // first centroid random
    int idx0 = std::floor(R::runif(0, TT));
    centIdx.push_back(idx0);
    
    // distances to nearest centroid
    std::vector<double> closestDist(TT);
    for (int j = 0; j < TT; ++j) closestDist[j] = Dall(idx0, j);
    
    // choose remaining centroids
    for (int k = 1; k < K; ++k) {
      // sample next centroid with prob proportional to closestDist
      double sumd = std::accumulate(closestDist.begin(), closestDist.end(), 0.0);
      if (sumd <= 0) {
        idx0 = std::floor(R::runif(0, TT));
      } else {
        double u = R::runif(0, sumd);
        double cum = 0;
        int idx = 0;
        for (; idx < TT; ++idx) {
          cum += closestDist[idx];
          if (cum >= u) break;
        }
        if (idx >= TT) idx = TT - 1;
        idx0 = idx;
      }
      centIdx.push_back(idx0);
      // update closestDist
      for (int j = 0; j < TT; ++j) {
        closestDist[j] = std::min(closestDist[j], Dall(centIdx[k], j));
      }
    }
    
    // 2) Assign each point to nearest centroid
    // Compute distance Y->centroids via Dall
    double sum_intra = 0.0;
    IntegerVector assign(TT);
    for (int i = 0; i < TT; ++i) {
      int best_k = 0;
      double best_d = Dall(centIdx[0], i);
      for (int k = 1; k < K; ++k) {
        double d = Dall(centIdx[k], i);
        if (d < best_d) {
          best_d = d;
          best_k = k;
        }
      }
      assign[i] = best_k + 1;  // 1-based cluster
      sum_intra += best_d;
    }
    
    // 3) Keep the best initialization
    if (sum_intra < best_sum) {
      best_sum = sum_intra;
      best_assign = assign;
    }
  }
  
  return best_assign;
}