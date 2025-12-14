// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
using namespace Rcpp;

/*** ---------------------------
 *  Simplex projections
 *  ---------------------------
 */

// Standard projection onto { x >= 0, sum(x) = 1 } (Duchi et al. 2008)
NumericVector proj_simplex(const NumericVector& v) {
  int n = v.size();
  NumericVector u = clone(v);
  std::sort(u.begin(), u.end(), std::greater<double>());
  NumericVector css(n);
  css[0] = u[0];
  for (int i = 1; i < n; ++i) css[i] = css[i - 1] + u[i];
  
  double theta = 0.0;
  for (int i = n - 1; i >= 0; --i) {
    double t = (css[i] - 1.0) / (i + 1);
    if (u[i] > t) { theta = t; break; }
  }
  NumericVector w(n);
  for (int i = 0; i < n; ++i) w[i] = std::max(v[i] - theta, 0.0);
  return w;
}

// Projection onto { x >= eps, sum(x) = 1 } via affine map:
// x = eps + c * y,  y = (v - eps)/c,  c = 1 - K*eps, then project y to standard simplex.
// Requires 0 < eps < 1/K.
// [[Rcpp::export]]
NumericVector proj_simplex_lb(const NumericVector& v, double eps) {
  int K = v.size();
  double eps_max = 1.0 / (double)K - 1e-15;
  if (eps <= 0.0) eps = 1e-12;
  if (eps >= eps_max) eps = eps_max;
  
  double c = 1.0 - K * eps;
  NumericVector y(K);
  for (int i = 0; i < K; ++i) y[i] = (v[i] - eps) / c;
  
  NumericVector yproj = proj_simplex(y);
  
  NumericVector x(K);
  for (int i = 0; i < K; ++i) {
    x[i] = eps + c * yproj[i];
    if (x[i] < eps) x[i] = eps; // safeguard
  }
  return x;
}

// sign with sign(0) = 0 (valid subgradient choice)
inline double sgn(double x){ return (x > 0.0) - (x < 0.0); }

/*** ---------------------------
 *  Objectives with L1^2 penalty
 *  ---------------------------
 */

// sum(s^m * g) + lambda * ( || s - s_t1 ||_1 )^pen_exp
// [[Rcpp::export]]
double eval_obj_1T(const NumericVector& s,
                   const NumericVector& g,
                   const NumericVector& s_t1,
                   double lambda,
                   double m,
                   double pen_exp) {
  int K = s.size();
  double fit = 0.0, l1 = 0.0;
  for (int i = 0; i < K; ++i) {
    fit += std::pow(s[i], m) * g[i];
    l1  += std::abs(s[i] - s_t1[i]);
  }
  double pen = 0.0;
  if (lambda > 0.0 && pen_exp > 0.0 && l1 > 0.0) {
    pen = lambda * std::pow(l1, pen_exp);
  }
  return fit + pen;
}

// sum(s^m * g) + lambda * ( || s - s_prec ||_1 )^pen_exp
//               + lambda * ( || s - s_succ ||_1 )^pen_exp
// [[Rcpp::export]]
double eval_obj_2T(const NumericVector& s,
                   const NumericVector& g,
                   const NumericVector& s_prec,
                   const NumericVector& s_succ,
                   double lambda,
                   double m,
                   double pen_exp) {
  int K = s.size();
  double fit = 0.0, l1p = 0.0, l1n = 0.0;
  for (int i = 0; i < K; ++i) {
    fit  += std::pow(s[i], m) * g[i];
    l1p  += std::abs(s[i] - s_prec[i]);
    l1n  += std::abs(s[i] - s_succ[i]);
  }
  double pen = 0.0;
  if (lambda > 0.0 && pen_exp > 0.0) {
    if (l1p > 0.0) pen += lambda * std::pow(l1p, pen_exp);
    if (l1n > 0.0) pen += lambda * std::pow(l1n, pen_exp);
  }
  return fit + pen;
}

/*** ---------------------------
 *  PGD optimizers with L1^2 penalty and (0,1) constraint
 *  ---------------------------
 */

// [[Rcpp::export]]
List optimize_pgd_1T(NumericVector init,
                     NumericVector g,
                     NumericVector s_t1,
                     double lambda,
                     double m,
                     double pen_exp = 2.0,   // <--- nuovo: esponente della L1
                     int max_iter   = 1000,
                     double alpha   = 1e-2,  // passo iniziale
                     double tol     = 1e-8) {
  int K = init.size();
  NumericVector s = proj_simplex(init);
  
  // Parametri Armijo / backtracking
  double alpha_cur    = alpha;
  const double alpha_min = 1e-12;
  const double alpha_max = 1.0;
  const double c_armijo  = 1e-4;
  const double shrink    = 0.5;
  const double expand    = 1.2;
  
  double f_old = eval_obj_1T(s, g, s_t1, lambda, m, pen_exp);
  
  for (int it = 0; it < max_iter; ++it) {
    // 1) Gradiente in s
    double L1 = 0.0;
    for (int i = 0; i < K; ++i)
      L1 += std::abs(s[i] - s_t1[i]);
    
    NumericVector grad(K);
    double grad_norm2 = 0.0;
    
    double coeff = 0.0;
    if (lambda > 0.0 && pen_exp > 0.0 && L1 > 0.0) {
      coeff = lambda * pen_exp * std::pow(L1, pen_exp - 1.0);
    }
    
    for (int i = 0; i < K; ++i) {
      double data_grad = m * std::pow(s[i], m - 1.0) * g[i];
      double pen_grad  = (coeff != 0.0) ? coeff * sgn(s[i] - s_t1[i]) : 0.0;
      grad[i] = data_grad + pen_grad;
      grad_norm2 += grad[i] * grad[i];
    }
    
    if (grad_norm2 < 1e-20) break;
    
    // 2) Backtracking line search
    double alpha_try = alpha_cur;
    double f_new = NA_REAL;
    NumericVector s_new(K);
    bool accepted = false;
    
    for (int ls = 0; ls < 20; ++ls) {
      for (int i = 0; i < K; ++i)
        s_new[i] = s[i] - alpha_try * grad[i];
      s_new = proj_simplex(s_new);
      
      f_new = eval_obj_1T(s_new, g, s_t1, lambda, m, pen_exp);
      
      if (f_new <= f_old - c_armijo * alpha_try * grad_norm2) {
        accepted = true;
        break;
      } else {
        alpha_try *= shrink;
        if (alpha_try < alpha_min)
          break;
      }
    }
    
    if (!accepted) break;
    
    double max_diff = 0.0;
    for (int i = 0; i < K; ++i) {
      double d = std::abs(s_new[i] - s[i]);
      if (d > max_diff) max_diff = d;
    }
    s = s_new;
    f_old = f_new;
    
    if (max_diff < tol) break;
    
    alpha_cur = std::min(alpha_max,
                         std::max(alpha_min, alpha_try * expand));
  }
  
  double obj = eval_obj_1T(s, g, s_t1, lambda, m, pen_exp);
  return List::create(_["par"] = s, _["value"] = obj);
}


// [[Rcpp::export]]
List optimize_pgd_2T(NumericVector init,
                     NumericVector g,
                     NumericVector s_prec,
                     NumericVector s_succ,
                     double lambda,
                     double m,
                     double pen_exp = 2.0,   // <--- nuovo
                     int max_iter   = 1000,
                     double alpha   = 1e-2,
                     double tol     = 1e-8) {
  int K = init.size();
  NumericVector s = proj_simplex(init);
  
  // Parametri Armijo / backtracking
  double alpha_cur    = alpha;
  const double alpha_min = 1e-12;
  const double alpha_max = 1.0;
  const double c_armijo  = 1e-4;
  const double shrink    = 0.5;
  const double expand    = 1.2;
  
  double f_old = eval_obj_2T(s, g, s_prec, s_succ, lambda, m, pen_exp);
  
  for (int it = 0; it < max_iter; ++it) {
    double L1p = 0.0, L1n = 0.0;
    for (int i = 0; i < K; ++i) {
      L1p += std::abs(s[i] - s_prec[i]);
      L1n += std::abs(s[i] - s_succ[i]);
    }
    
    NumericVector grad(K);
    double grad_norm2 = 0.0;
    
    double coeff_p = 0.0, coeff_n = 0.0;
    if (lambda > 0.0 && pen_exp > 0.0) {
      if (L1p > 0.0) coeff_p = lambda * pen_exp * std::pow(L1p, pen_exp - 1.0);
      if (L1n > 0.0) coeff_n = lambda * pen_exp * std::pow(L1n, pen_exp - 1.0);
    }
    
    for (int i = 0; i < K; ++i) {
      double data_grad = m * std::pow(s[i], m - 1.0) * g[i];
      double pen_grad  = 0.0;
      if (coeff_p != 0.0) pen_grad += coeff_p * sgn(s[i] - s_prec[i]);
      if (coeff_n != 0.0) pen_grad += coeff_n * sgn(s[i] - s_succ[i]);
      grad[i] = data_grad + pen_grad;
      grad_norm2 += grad[i] * grad[i];
    }
    
    if (grad_norm2 < 1e-20) break;
    
    double alpha_try = alpha_cur;
    double f_new = NA_REAL;
    NumericVector s_new(K);
    bool accepted = false;
    
    for (int ls = 0; ls < 20; ++ls) {
      for (int i = 0; i < K; ++i)
        s_new[i] = s[i] - alpha_try * grad[i];
      s_new = proj_simplex(s_new);
      
      f_new = eval_obj_2T(s_new, g, s_prec, s_succ, lambda, m, pen_exp);
      
      if (f_new <= f_old - c_armijo * alpha_try * grad_norm2) {
        accepted = true;
        break;
      } else {
        alpha_try *= shrink;
        if (alpha_try < alpha_min)
          break;
      }
    }
    
    if (!accepted) break;
    
    double max_diff = 0.0;
    for (int i = 0; i < K; ++i) {
      double d = std::abs(s_new[i] - s[i]);
      if (d > max_diff) max_diff = d;
    }
    s = s_new;
    f_old = f_new;
    
    if (max_diff < tol) break;
    
    alpha_cur = std::min(alpha_max,
                         std::max(alpha_min, alpha_try * expand));
  }
  
  double obj = eval_obj_2T(s, g, s_prec, s_succ, lambda, m, pen_exp);
  return List::create(_["par"] = s, _["value"] = obj);
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


//------------------------------------------------------------------------------
// Helper: generate all integer combinations of length n_c that sum to N
//------------------------------------------------------------------------------
static void generate_combinations(int n_c, int N, 
                                  std::vector< std::vector<int> > &out,
                                  std::vector<int> &current,
                                  int idx, int remaining) 
{
  if (idx == n_c - 1) {
    current[idx] = remaining;
    out.push_back(current);
    return;
  }
  for (int i = 0; i <= remaining; ++i) {
    current[idx] = i;
    generate_combinations(n_c, N, out, current, idx + 1, remaining - i);
  }
}

//------------------------------------------------------------------------------
// [[Rcpp::export]]
NumericMatrix discretize_prob_simplex(int n_c, double grid_size) {
  // Sample a grid on the n_c-dimensional probability simplex.
  int N = (int) std::round(1.0 / grid_size);
  if (N <= 0) {
    stop("grid_size must be a positive fraction <= 1");
  }
  
  // Generate all integer tuples of length n_c that sum to N
  std::vector< std::vector<int> > int_tuples;
  std::vector<int> current(n_c, 0);
  generate_combinations(n_c, N, int_tuples, current, 0, N);
  
  int M = (int) int_tuples.size();
  NumericMatrix simplex(M, n_c);
  
  // Fill in reverse order to match R code’s reverse indexing
  for (int i = 0; i < M; ++i) {
    int rev_i = M - 1 - i;
    for (int j = 0; j < n_c; ++j) {
      simplex(i, j) = double(int_tuples[rev_i][j]) / double(N);
    }
  }
  
  return simplex;
}

//------------------------------------------------------------------------------
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <numeric>
using namespace Rcpp;

// Forward declarations
NumericMatrix discretize_prob_simplex(int K, double grid_size);
NumericMatrix gower_dist(const NumericMatrix& Y, const NumericMatrix& mu,
                         Nullable<IntegerVector> feat_type, std::string scale);

// helper: median of a vector
static double median_vec(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  int n = v.size();
  if(n == 0) return NA_REAL;
  return (n % 2 == 1) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

// [[Rcpp::export]]
List cont_jump(const NumericMatrix &Y_in,
               int K,
               double jump_penalty = 1e-5,   // questo è λ_CJM nel paper
               double alpha = 2.0,           // deve essere 2 per avere λ/4 * || · ||_1^2
               Nullable<IntegerVector> initial_states_ = R_NilValue,
               int max_iter = 10,
               int n_init = 10,
               Nullable<double> tol_ = R_NilValue,
               bool mode_loss = true,
               double grid_size = 0.05,
               bool verbose = false,
               bool stop_when_path_unchanged = true)
{
  if (verbose) {
    Rcout << "Starting cont_jump with " << n_init << " initializations\n";
    if (alpha != 2.0)
      Rcout << "WARNING: alpha != 2.0; non coincide più con λ/4 * ||·||_1^2 del paper.\n";
  }
  
  // --- Preprocessing: imputazione con media di colonna ---
  NumericMatrix Y_orig = clone(Y_in);
  int TT = Y_orig.nrow();
  int P  = Y_orig.ncol();
  
  LogicalMatrix M_mask(TT, P);
  NumericVector colMean(P);
  
  for (int j = 0; j < P; ++j) {
    double sumj = 0.0;
    int cntj = 0;
    for (int t = 0; t < TT; ++t) {
      if (NumericVector::is_na(Y_orig(t, j))) {
        M_mask(t, j) = true;
      } else {
        sumj += Y_orig(t, j);
        cntj++;
        M_mask(t, j) = false;
      }
    }
    colMean[j] = (cntj > 0) ? (sumj / cntj) : 0.0;
  }
  for (int t = 0; t < TT; ++t)
    for (int j = 0; j < P; ++j)
      if (M_mask(t, j)) Y_orig(t, j) = colMean[j];
      
      // --- Griglia sul simplesso e distanza L1 tra nodi ---
      NumericMatrix prob_vecs = discretize_prob_simplex(K, grid_size); // N x K
      int N = prob_vecs.nrow();
      
      NumericMatrix pairwise_l1(N, N);
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          double acc = 0.0;
          for (int c = 0; c < K; ++c)
            acc += std::abs(prob_vecs(i, c) - prob_vecs(j, c));
          pairwise_l1(i, j) = acc / 2.0;   // => (||·||_1 / 2)
        }
      }
      
      // --- Array per salvare tutte le run ---
      std::vector<List> all_runs;
      all_runs.reserve(n_init);
      
      // ======================================================================
      // LOOP SU n_init
      // ======================================================================
      for (int init_idx = 0; init_idx < n_init; ++init_idx) {
        if (verbose) Rcout << " init " << (init_idx + 1) << "/" << n_init << "\n";
        
        NumericMatrix Y = clone(Y_orig);
        LogicalMatrix M = clone(M_mask);
        
        // Parametri di tolleranza
        bool use_tol = false;
        double tol   = 0.0;
        if (tol_.isNotNull()) {
          tol = as<double>(tol_);
          use_tol = true;
        }
        
        // Inizializzazione degli stati "discreti" (solo per init prototipi)
        IntegerVector s_disc(TT);
        if (initial_states_.isNotNull()) {
          IntegerVector init_s = initial_states_.get();
          if ((int)init_s.size() != TT) stop("initial_states must match rows (T).");
          for (int t = 0; t < TT; ++t) s_disc[t] = init_s[t] - 1; // 0-based
        } else {
          RNGScope scope;
          for (int t = 0; t < TT; ++t)
            s_disc[t] = (int)std::floor(R::runif(0.0, (double)K));
        }
        
        // Prototipi mu (K x P): inizializzazione come medie per stato
        NumericMatrix mu(K, P);
        for (int k = 0; k < K; ++k) {
          int cnt = 0;
          for (int j = 0; j < P; ++j) mu(k, j) = 0.0;
          for (int t = 0; t < TT; ++t) if (s_disc[t] == k) {
            cnt++;
            for (int j = 0; j < P; ++j) mu(k, j) += Y(t, j);
          }
          if (cnt > 0) {
            for (int j = 0; j < P; ++j) mu(k, j) /= cnt;
          } else {
            for (int j = 0; j < P; ++j) mu(k, j) = colMean[j];
          }
        }
        
        // Matrici di lavoro
        NumericMatrix S(TT, K), S_old(TT, K), loss_mx(TT, N), values(TT, N);
        NumericVector assign(TT);
        double loss_old  = R_PosInf;
        double value_opt = R_PosInf;
        int    iter_count = 0;
        
        std::vector<double> loss_history;
        loss_history.reserve(max_iter);
        
        // ====================================================================
        // LOOP ESTERNO SU max_iter
        // ====================================================================
        for (int iter = 0; iter < max_iter; ++iter) {
          iter_count = iter + 1;
          
          // 1) Loss rispetto ai K prototipi (scaled squared L2)
          //    l(y_t, mu_k) = 0.5 * ||y_t - mu_k||_2^2   (metrica del paper)
          NumericMatrix loss_k(TT, K);
          for (int t = 0; t < TT; ++t) {
            for (int k = 0; k < K; ++k) {
              double ss = 0.0;
              for (int j = 0; j < P; ++j) {
                double d = Y(t, j) - mu(k, j);
                ss += d * d;
              }
              loss_k(t, k) = 0.5 * ss;   // *** QUI CAMBIATO: prima era 0.5 * sqrt(ss) ***
            }
          }
          
          // 2) Matrice delle penalità di salto tra nodi della griglia
          //    Con alpha = 2:
          //    jump_penalty_mx(i,j) = jump_penalty * (pairwise_l1(i,j))^2
          //                         = (jump_penalty/4)*||s^i - s^j||_1^2
          //    => λ_CJM = jump_penalty, fattore 1/4 come nel paper.
          NumericMatrix jump_penalty_mx(N, N);
          for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
              jump_penalty_mx(i, j) =
                jump_penalty * std::pow(pairwise_l1(i, j), alpha);
            }
          }
          
          // 3) (Opzionale) "mode_loss" smoothing
          if (mode_loss) {
            NumericVector m_loss(N);
            for (int i = 0; i < N; ++i) {
              double mx = -R_PosInf;
              for (int j = 0; j < N; ++j)
                mx = std::max(mx, -jump_penalty_mx(i, j));
              double se = 0.0;
              for (int j = 0; j < N; ++j)
                se += std::exp(-jump_penalty_mx(i, j) - mx);
              m_loss[i] = mx + std::log(se);
            }
            double off = m_loss[0];
            for (int i = 0; i < N; ++i) m_loss[i] -= off;
            for (int i = 0; i < N; ++i)
              for (int j = 0; j < N; ++j)
                jump_penalty_mx(i, j) += m_loss[i];
          }
          
          // 4) Loss dei nodi della griglia:
          //    L_t(i) = sum_k s^{(i)}_k * l(y_t, mu_k)
          for (int t = 0; t < TT; ++t) {
            for (int i = 0; i < N; ++i) {
              double ac = 0.0;
              for (int k = 0; k < K; ++k)
                ac += loss_k(t, k) * prob_vecs(i, k);
              loss_mx(t, i) = ac;
            }
          }
          for (int t = 0; t < TT; ++t)
            for (int i = 0; i < N; ++i)
              if (std::isnan(loss_mx(t, i))) loss_mx(t, i) = R_PosInf;
              
              // 5) Viterbi-like DP sui nodi della griglia
              // forward
              for (int i = 0; i < N; ++i) values(0, i) = loss_mx(0, i);
              for (int t = 1; t < TT; ++t) {
                for (int i = 0; i < N; ++i) {
                  double bv = R_PosInf;
                  for (int j = 0; j < N; ++j) {
                    double cand = values(t - 1, j) + jump_penalty_mx(j, i);
                    if (cand < bv) bv = cand;
                  }
                  values(t, i) = loss_mx(t, i) + bv;
                }
              }
              // backtrack
              double best = R_PosInf;
              for (int i = 0; i < N; ++i) {
                if (values(TT - 1, i) < best) {
                  best = values(TT - 1, i);
                  assign[TT - 1] = i;
                }
              }
              value_opt = best;
              
              for (int k = 0; k < K; ++k)
                S(TT - 1, k) = prob_vecs(assign[TT - 1], k);
              
              for (int t = TT - 2; t >= 0; --t) {
                int nxt = assign[t + 1];
                double bv = R_PosInf;
                int bi = 0;
                for (int i = 0; i < N; ++i) {
                  double cnd = values(t, i) + jump_penalty_mx(i, nxt);
                  if (cnd < bv) { bv = cnd; bi = i; }
                }
                assign[t] = bi;
                for (int k = 0; k < K; ++k)
                  S(t, k) = prob_vecs(bi, k);
              }
              
              // registra loss
              loss_history.push_back(value_opt);
              
              // 6) Aggiorna prototipi mu come medie pesate da S
              NumericVector sum_prob(K);
              for (int k = 0; k < K; ++k) {
                sum_prob[k] = 0.0;
                for (int j = 0; j < P; ++j) mu(k, j) = 0.0;
              }
              
              for (int t = 0; t < TT; ++t) {
                for (int k = 0; k < K; ++k) {
                  sum_prob[k] += S(t, k);
                  for (int j = 0; j < P; ++j)
                    mu(k, j) += S(t, k) * Y(t, j);
                }
              }
              for (int k = 0; k < K; ++k)
                if (sum_prob[k] > 0.0)
                  for (int j = 0; j < P; ++j)
                    mu(k, j) /= sum_prob[k];
                
                // 7) Re-imputazione missing con prototipo più probabile (hard)
                for (int t = 0; t < TT; ++t) {
                  int hk = 0; double mp = S(t, 0);
                  for (int k = 1; k < K; ++k)
                    if (S(t, k) > mp) { mp = S(t, k); hk = k; }
                    for (int j = 0; j < P; ++j)
                      if (M(t, j)) Y(t, j) = mu(hk, j);
                }
                
                // 8) Criteri di arresto
                bool converged = false;
                
                if (use_tol) {
                  double eps = loss_old - value_opt; // >0 se migliora
                  if (verbose)
                    Rcout << " iter=" << iter_count
                          << " loss=" << value_opt
                          << " dloss=" << eps << "\n";
                    if (eps >= 0.0 && eps < tol) converged = true;
                }
                
                if (!use_tol && stop_when_path_unchanged) {
                  converged = true;
                  for (int t = 0; t < TT && converged; ++t) {
                    for (int k = 0; k < K; ++k) {
                      if (S(t, k) != S_old(t, k)) { converged = false; break; }
                    }
                  }
                }
                
                // prepara per prossimo giro
                loss_old = value_opt;
                for (int t = 0; t < TT; ++t)
                  for (int k = 0; k < K; ++k)
                    S_old(t, k) = S(t, k);
                
                if (converged) {
                  if (verbose)
                    Rcout << "  converged at iter=" << iter_count
                          << " loss=" << value_opt << "\n";
                    break;
                }
        } // fine loop iter
        
        if (verbose)
          Rcout << " end init " << (init_idx + 1)
                << " loss=" << value_opt
                << " iters=" << iter_count << "\n";
          
          NumericVector loss_hist_R(loss_history.begin(), loss_history.end());
          all_runs.push_back(List::create(
              Named("S")            = S,
              Named("value_opt")    = value_opt,
              Named("mu")           = mu,
              Named("loss_history") = loss_hist_R
          ));
      } // fine loop su n_init
      
      // --- scegli il best ---
      double best_loss = R_PosInf;
      int best_idx = 0;
      for (int i = 0; i < n_init; ++i) {
        double v = as<double>(all_runs[i]["value_opt"]);
        if (v < best_loss) { best_loss = v; best_idx = i; }
      }
      if (verbose)
        Rcout << "Best init = " << (best_idx + 1)
              << " loss = " << best_loss << "\n";
        
        List best = all_runs[best_idx];
        
        return List::create(
          Named("best_S")            = best["S"],
                                           Named("best_loss")         = best["value_opt"],
                                                                            Named("best_mu")           = best["mu"],
                                                                                                             Named("best_loss_history") = best["loss_history"],
                                                                                                                                              Named("all_runs")          = all_runs
        );
}
