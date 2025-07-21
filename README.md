# Fuzzy Jump Model

This repository provides an R implementation of the **fuzzy jump model**: a flexible approach for clustering multivariate time series data with mixed data types. The model estimates time-varying fuzzy (soft) cluster memberships while discouraging frequent transitions between states.

## 🔍 Model Overview

Our **fuzzy jump model** sequentially estimates time-varying state probabilities. It supports both **soft and hard clustering** via a fuzziness parameter `m`, and it accommodates **mixed-type multivariate time series**. Transitions between clusters are penalized to favor persistent regime behavior, making the method suitable for segmentation tasks with temporal coherence.

## 📁 Repository Structure

- `utils_r.R` — Main R functions:
  - `fuzzy_jump_cpp(...)`: fits the fuzzy jump model on a mixed-type time series.
  - `simulate_fuzzy_mixture_mv(...)`: generates toy data from a fuzzy time-varying Gaussian mixture.
  - `order_states_condMed(...)`: helper to reorder clusters based on conditional medians.

- `utils_c.cpp` — C++ code used internally for optimization (required by `fuzzy_jump_cpp`).  
  **Must be saved in the same folder as `utils_r.R`.**

- `example.R` — A minimal example demonstrating how to simulate data and fit the model.

## 🚀 How to Use

1. Place `utils_r.R` and `utils_c.cpp` in the same directory.
2. Source the R functions and run the toy example from `example.R`.
3. Dependencies:  
   - R packages: `MASS`, `parallel`, `poliscidata`  
   - C++ compilation via `Rcpp`

## 📦 Example

```r
source("utils_r.R")

# Simulate data
sim_data <- simulate_fuzzy_mixture_mv(TT = 300, P = 3, K = 2)

# Fit fuzzy jump model
res <- fuzzy_jump_cpp(sim_data[, paste0("Y", 1:3)], 
                      K = 2, lambda = .25, m = 1.25, 
                      max_iter = 10, n_init = 5, parallel = F)
