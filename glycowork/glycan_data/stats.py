import pandas as pd
import numpy as np
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import wilcoxon, rankdata, norm, chi2, t, f, entropy, gmean, f_oneway, combine_pvalues, dirichlet, spearmanr, ttest_rel, ttest_ind
from scipy.stats.mstats import winsorize
from scipy.spatial import procrustes
from scipy.spatial.distance import squareform
import scipy.integrate as integrate
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttost_ind, ttost_paired
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import statsmodels.api as sm
import statsmodels.formula.api as smf
import copy
rng = np.random.default_rng(42)
np.random.seed(0)


def cohen_d(x: Union[np.ndarray, List[float]], # comparison group containing numerical data
           y: Union[np.ndarray, List[float]], # comparison group containing numerical data
           paired: bool = False # whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient)
          ) -> Tuple[float, float]: # (Cohen's d, variance) where d: 0.2 small; 0.5 medium; 0.8 large effect size
  "calculates effect size between two groups"
  if paired:
    assert len(x) == len(y), "For paired samples, the size of x and y should be the same"
    diff = np.array(x) - np.array(y)
    diff_std = np.std(diff, ddof = 1)
    if diff_std == 0:
      d = np.inf if np.mean(diff) > 0 else -np.inf
      return d, 0
    n = len(diff)
    d = np.mean(diff) / diff_std
    var_d = 1 / n + d**2 / (2 * n)
  else:
    nx = len(x)
    ny = len(y)
    sx = max(np.std(x, ddof = 1), 1e-6)
    sy = max(np.std(y, ddof = 1), 1e-6)
    dof = nx + ny - 2
    d = (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1) * sx ** 2 + (ny-1) * sy ** 2) / dof)
    var_d = (nx + ny) / (nx * ny) + d**2 / (2 * (nx + ny))
  return d, var_d


def mahalanobis_distance(x: Union[np.ndarray, pd.DataFrame], # comparison group containing numerical data
                        y: Union[np.ndarray, pd.DataFrame], # comparison group containing numerical data
                        paired: bool = False # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                       ) -> float: # Mahalanobis distance effect size
  "calculates effect size between two groups in a multivariate comparison"
  if paired:
    assert x.shape == y.shape, "For paired samples, the size of x and y should be the same"
    x = np.array(x) - np.array(y)
    y = np.zeros_like(x)
  if isinstance(x, pd.DataFrame):
    x = x.values
  if isinstance(y, pd.DataFrame):
    y = y.values
  pooled_cov_inv = np.linalg.pinv((np.cov(x) + np.cov(y)) / 2)
  diff_means = (np.mean(y, axis = 1) - np.mean(x, axis = 1)).reshape(-1, 1)
  mahalanobis_d = np.sqrt(np.clip(diff_means.T @ pooled_cov_inv @ diff_means, 0, None))
  return mahalanobis_d[0][0]


def mahalanobis_variance(x: Union[np.ndarray, pd.DataFrame], # comparison group containing numerical data
                        y: Union[np.ndarray, pd.DataFrame], # comparison group containing numerical data
                        paired: bool = False # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                       ) -> float: # variance of Mahalanobis distance
  "Estimates variance of Mahalanobis distance via bootstrapping"
  # Combine gp1 and gp2 into a single matrix
  data = np.concatenate((x.T, y.T), axis = 0)
  # Perform bootstrap resampling
  n_iterations = 1000
  # Initialize an empty array to store the bootstrap samples
  bootstrap_samples = np.empty(n_iterations)
  size_x = x.shape[1]
  for i in range(n_iterations):
    # Generate a random bootstrap sample
    sample = data[rng.choice(range(data.shape[0]), size = data.shape[0], replace = True)]
    # Split the bootstrap sample into two groups
    x_sample = sample[:size_x]
    y_sample = sample[size_x:]
    # Calculate the Mahalanobis distance for the bootstrap sample
    bootstrap_samples[i] = mahalanobis_distance(x_sample.T, y_sample.T, paired = paired)
  # Estimate the variance of the Mahalanobis distance
  return np.var(bootstrap_samples)


def variance_stabilization(data: pd.DataFrame, # dataframe with glycans/motifs as indices and samples as columns
                         groups: Union[List[List[str]], None] = None # list containing lists of column names of samples from same group for group-specific normalization; otherwise global
                        ) -> pd.DataFrame: # normalized dataframe in same format as input
  "performs variance stabilization normalization"
  # Apply log1p transformation
  data = np.log1p(data)
  # Scale data to have zero mean and unit variance
  if groups is None:
    data = (data - data.mean(axis = 0)) / data.std(axis = 0, ddof = 1)
  else:
    for group in groups:
      group_data = data[group]
      data[group] = (group_data - group_data.mean(axis = 0)) / group_data.std(axis = 0, ddof = 1)
  return data


class MissForest:
  def __init__(self, regressor: RandomForestRegressor = RandomForestRegressor(n_jobs = -1), # estimator object for each imputation
                 max_iter: int = 5, # number of iterations for imputation process
                 tol: float = 1e-5 # convergence tolerance
                ) -> None:
    "A class to perform MissForest imputation adapted from https://github.com/yuenshingyan/MissForest"
    self.regressor = regressor
    self.max_iter = max_iter
    self.tol = tol

  def fit_transform(self, X: pd.DataFrame # input dataframe with missing values
                    ) -> pd.DataFrame: # imputed dataframe
    "Replace missing values using the MissForest algorithm"
    # Step 1: Initialization
    # Keep track of where NaNs are in the original dataset
    X_nan = X.isnull()
    # Replace NaNs with median of the column in a new dataset that will be transformed
    X_transform = X.fillna(X.median())
    # Sort columns by the number of NaNs (ascending)
    sorted_columns = X_nan.sum().sort_values().index
    for _ in range(self.max_iter):
      total_change = 0
      # Step 2: Imputation
      for column in sorted_columns:
        missing_idx = X_nan[column]
        if missing_idx.any():  # if column has missing values in original dataset
          # Split data into observed and missing for the current column
          observed = X_transform.loc[~missing_idx]
          missing = X_transform.loc[missing_idx]
          features = observed.drop(columns = column)
          if features.notna().any().any():
            # Use other columns to predict the current column
            self.regressor.fit(observed.drop(columns = column), observed[column])
            y_missing_pred = self.regressor.predict(missing.drop(columns = column))
            # Replace missing values in the current column with predictions
            total_change += np.sum(np.abs(X_transform.loc[missing_idx, column] - y_missing_pred))
            X_transform.loc[missing_idx, column] = y_missing_pred
      # Check for convergence
      if total_change < self.tol:
        break  # Break out of the loop if converged
    # Avoiding zeros
    X_transform += 1e-6
    return X_transform


def impute_and_normalize(df_in: pd.DataFrame, # dataframe with glycan sequences in first col and abundances in subsequent cols
                        groups: List[List[str]], # nested list of column name lists, one list per group
                        impute: bool = True, # replaces zeroes with predictions from MissForest
                        min_samples: float = 0.1 # percent of samples that need non-zero values for glycan to be kept
                       ) -> pd.DataFrame: # normalized dataframe in same style as input
    "discards rows with too many missings, imputes the rest, and normalizes"
    df = df_in.copy()
    if min_samples:
      min_count = max(np.floor(df.shape[1] * min_samples), 1) + 1
      mask = (df != 0).sum(axis = 1) >= min_count
      df = df[mask].reset_index(drop = True)
    colname = df.columns[0]
    glycans = df[colname]
    df = df.iloc[:, 1:]
    df = df.astype(float)
    for group in groups:
      group_data = df[group]
      all_zero_mask = (group_data == 0).all(axis = 1)
      df.loc[all_zero_mask, group] = 1e-5
    old_cols = []
    if isinstance(colname, int):
      old_cols = df.columns
      df.columns = df.columns.astype(str)
    if impute:
      mf = MissForest()
      df = df.replace(0, np.nan)
      df = mf.fit_transform(df)
    df = (df / df.sum(axis = 0)) * 100
    if len(old_cols) > 0:
      df.columns = old_cols
    df.insert(loc = 0, column = colname, value = glycans)
    return df


def variance_based_filtering(df: pd.DataFrame, # dataframe with glycans as index and samples in columns
                           min_feature_variance: float = 0.02 # minimum variance to include a feature
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]: # (filtered df with variance > min, discarded df with variance <= min)
  "Variance-based filtering of features"
  variances = df.var(axis = 1)
  filtered_df = df.loc[variances > min_feature_variance]
  discarded_df = df.loc[variances <= min_feature_variance]
  return filtered_df, discarded_df


def jtkdist(timepoints: Union[int, np.ndarray], # number/array of timepoints within experiment
           param_dic: Dict, # dictionary carrying parameter values
           reps: int = 1 # number of replicates within each timepoint
          ) -> Dict: # updated param_dic with statistical values
  "Precalculates all possible JT test statistic permutation probabilities for reference using the Harding algorithm"
  tim = np.full(timepoints, reps)
  nn = sum(tim)  # Number of data values (Independent of period and lag)
  M = (nn ** 2 - np.sum(np.square(tim))) * 0.5 if nn > 0 else 0  # Max possible jtk statistic
  param_dic.update({"GRP_SIZE": tim, "NUM_GRPS": len(tim), "NUM_VALS": nn,
                    "MAX": M, "DIMS": [int(nn * (nn - 1) * 0.5), 1 if nn > 1 else [0, 0]]})
  squared_terms = tim * tim * (2 * tim + 3)
  var = (nn ** 2 * (2 * nn + 3) - np.sum(squared_terms)) / 72
  param_dic["VAR"] = var  # Variance of JTK
  param_dic["SDV"] = np.sqrt(max(var, 0.0))  # Standard deviation of JTK
  param_dic["EXV"] = M * 0.5  # Expected value of JTK
  param_dic["EXACT"] = False
  MM = int(M // 2)  # Mode of this possible alternative to JTK distribution
  cf = np.ones(MM + 1)  # Initial lower half cumulative frequency (cf) distribution
  size = np.sort(tim)  # Sizes of each group of known replicate values, in ascending order for fastest calculation
  N = np.cumsum(size[::-1])[::-1][1:] # Count permutations using the Harding algorithm
  for m, n in zip(size[:-1], N): # Calculate cumulative frequencies; cf now contains the lower half cumulative frequency distribution
    P = min(m + n, MM)
    for q in range(n + 1, P + 1):
      cf[q:MM + 1] -= cf[:MM + 1 - q]
    Q = min(m, MM)
    for s in range(1, Q + 1):
      cf[s:MM + 1] += cf[:MM + 1 - s]
  # append the symmetric upper half cumulative frequency distribution to cf
  if M % 2:   # jtkcf = upper-tail cumulative frequencies for all integer jtk
    jtkcf = np.concatenate((cf, 2 * cf[MM] - cf[:MM][::-1], [2 * cf[MM]]))[::-1]
  else:
    jtkcf = np.concatenate((cf, cf[MM - 1] + cf[MM] - cf[:MM-1][::-1], [cf[MM - 1] + cf[MM]]))[::-1]
  ajtkcf = [(jtkcf[i - 1] + jtkcf[i]) / 2 for i in range(1, len(jtkcf))]  # Interpolated values
  cf = [ajtkcf[(j - 1) // 2] if j % 2 == 0 else jtkcf[j // 2] for j in range(1, 2 * int(M) + 2)]
  param_dic["CP"] = [c / jtkcf[0] if jtkcf[0] != 0 else 1 for c in cf]  # All upper-tail p-values
  return param_dic


def jtkinit(periods: List[int], # possible periods of rhythmicity in biological data (valued as 'number of timepoints')
           param_dic: Dict, # dictionary carrying parameter values
           interval: int = 1, # units of time between experimental timepoints
           replicates: int = 1 # number of replicates within each group
          ) -> Dict: # updated param_dic with waveform parameters
  "Defines the parameters of the simulated sine waves for reference later"
  param_dic["INTERVAL"] = interval
  param_dic["PERIODS"] = list(periods)
  tim = np.array(param_dic["GRP_SIZE"])
  timepoints = int(param_dic["NUM_GRPS"])
  param_dic["PERFACTOR"] = np.repeat(np.arange(1, len(periods) + 1), timepoints)
  timerange = np.arange(timepoints)  # Zero-based time indices
  max_period = max(periods)
  signcos_length = ((math.floor(timepoints / max_period) * max_period) * replicates)
  param_dic["SIGNCOS"] = np.zeros((max_period, signcos_length), dtype = int)
  param_dic["CGOOSV"] = []
  PI2 = 2 * round(math.pi, 4)
  max_tim = np.max(tim) if np.max(tim) > 0 else 1
  for i, period in enumerate(periods):
    theta = (PI2 * timerange) / period  # Zero-based angular values across time indices
    cos_v = np.cos(theta)  # Unique cosine values at each time point
    ranked = rankdata(cos_v)
    cos_r = np.repeat(ranked, max_tim) # replicated ranks of unique cosine values
    cgoos = np.sign(np.subtract.outer(cos_r, cos_r)).astype(int)
    mask = np.tril_indices(len(cgoos), k = -1)
    cgoos = cgoos[mask]  # Lower triangular calculation
    period_array = np.zeros((len(cgoos), period))
    period_array[:, 0] = cgoos.reshape(param_dic["DIMS"])[:, 0]
    param_dic["CGOOSV"].append(period_array)  # Period array
    cycles = math.floor(timepoints / period)
    jrange = np.arange(cycles * period)
    cos_s = np.repeat(np.sign(cos_v)[jrange], tim[jrange])  # Cosine signs
    slice_idx = slice(len(cos_s))
    if replicates == 1:
      param_dic["SIGNCOS"][slice_idx, i] = cos_s
    else:
      param_dic["SIGNCOS"][i, slice_idx] = cos_s
    for j in range(1, period):  # One-based half-integer lag index j
      delta_theta = j * theta / 2  # Angles of half-integer lags
      cos_v = np.cos(theta + delta_theta)  # Cycle left
      cos_r = np.repeat(rankdata(cos_v), tim) # Phase-shifted replicated ranks
      cgoos = np.sign(np.subtract.outer(cos_r, cos_r))
      mask = np.triu_indices(len(cgoos), k = 1)
      cgoos = cgoos[mask]  #Upper triangular calculation
      param_dic["CGOOSV"][i][:, j] = cgoos.reshape(param_dic["DIMS"]).flatten()
      cos_s = np.repeat(np.sign(cos_v.flatten())[jrange], tim[jrange])
      if replicates == 1:
        param_dic["SIGNCOS"][slice_idx, j] = cos_s
      else:
        param_dic["SIGNCOS"][j, slice_idx] = cos_s
  return param_dic


def jtkstat(z: pd.DataFrame, # expression data for a molecule ordered in groups by timepoint
           param_dic: Dict # dictionary containing parameters defining model waveforms
          ) -> Dict: # updated param_dic with appropriate model waveform assigned
  "Determines the JTK statistic and p-values for all model phases, compared to expression data"
  param_dic["CJTK"] = []
  M = param_dic["MAX"]
  z = np.array(z).flatten()
  valid_mask = ~np.isnan(z)
  z_valid = z[valid_mask]
  foosv = np.sign(np.subtract.outer(z_valid, z_valid)).T  # Select upper triangle rather than the lower triangle
  mask = np.triu(np.ones(foosv.shape), k = 1).astype(bool) # Remove middle diagonal from the tri index
  mask[np.diag_indices(mask.shape[0])] = False
  foosv = foosv[mask]
  expected_dims = param_dic["DIMS"][0] * param_dic["DIMS"][1]
  if len(foosv) != expected_dims:
    temp_foosv = np.zeros(expected_dims)
    temp_foosv[:len(foosv)] = foosv[:expected_dims]
    foosv = temp_foosv
  foosv = foosv.reshape(param_dic["DIMS"])
  num_periods = param_dic["PERIODS"][0]
  EXV = param_dic["EXV"]
  SDV = param_dic["SDV"]
  is_exact = param_dic.get("EXACT", False)
  CP = param_dic["CP"]
  for i in range(num_periods):
    if i >= len(param_dic["CGOOSV"][0]):
      param_dic["CJTK"].append([1, 0, 0])
      continue
    cgoosv = param_dic["CGOOSV"][0][i]
    if foosv.shape != cgoosv.shape:
      param_dic["CJTK"].append([1, 0, 0])
      continue
    S = np.nansum(np.diag(foosv * cgoosv))
    if S == 0:
      param_dic["CJTK"].append([1, 0, 0])
      continue
    jtk = (abs(S) + M) / 2  # Two-tailed JTK statistic for this lag and distribution
    tau = S / M if M != 0 else 0
    if is_exact:
      jtki = min(1 + 2 * int(jtk), len(CP))  # index into the exact upper-tail distribution
      p = 2 * CP[jtki-1] if jtki > 0 else 1
    else:
      p = 2 * norm.cdf(-(jtk - 0.5), -EXV, SDV)
    param_dic["CJTK"].append([p, S, tau])  # include tau = s/M for this lag and distribution
  return param_dic


def jtkx(z: pd.DataFrame, # expression data ordered in groups by timepoint
        param_dic: Dict, # dictionary containing parameters defining model waveforms
        ampci: bool = False # whether to calculate amplitude confidence interval
       ) -> pd.Series: # optimal waveform parameters for each molecular species
  "Deployment of jtkstat for repeated use, and parameter extraction"
  param_dic = jtkstat(z, param_dic)  # Calculate p and S for all phases
  padj = np.array([cjtk[0] for cjtk in param_dic["CJTK"]])  # Exact two-tailed p values for period/phase combos
  JTK_ADJP = np.min(padj)  # Global minimum adjusted p-value
  minpadj = [np.min(padj[param_dic["PERFACTOR"] == p]) for p in range(1, len(param_dic["PERIODS"]) + 1)] # Minimum adjusted p-values for each period
  pers = param_dic["PERIODS"][np.argmin(minpadj)] if len(param_dic["PERIODS"]) > 1 else param_dic["PERIODS"][0]
  lagis = np.where(np.abs(padj - JTK_ADJP) < 1e-10)[0]  # list of optimal lag indices for each optimal period
  if len(lagis) == 0:
    lagis = np.array([np.nanargmin(np.abs(padj - JTK_ADJP))])
  best_results = {'bestper': 0, 'bestlag': 0, 'besttau': 0, 'maxamp': 0, 'maxamp_ci': 2, 'maxamp_pval': 0}
  sc = param_dic["SIGNCOS"].T
  z_trim = z[:len(sc)]
  hlm_z = float(np.nanmedian(z_trim)) if len(z_trim) > 0 else 0.0
  if np.isnan(hlm_z).all():
    hlm_z = np.zeros_like(z_trim)  # Fallback if all values are NaN
  w = (z_trim - hlm_z) * np.sqrt(2)
  for _ in range(abs(pers)):
    for lagi in lagis:
      S = param_dic["CJTK"][lagi][1]
      s = np.sign(S) if S != 0 else 1
      lag = (pers + (1 - s) * pers / 4 - lagi / 2) % pers
      tmp = s * w * sc[:, lagi]
      tmp_clean = tmp[np.isfinite(tmp)]
      if len(tmp_clean) > 0:
        if ampci:
          jtkwt = pd.DataFrame(wilcoxon(tmp_clean, zero_method = 'wilcox', correction = False,
                                                alternatives = 'two-sided', mode = 'exact'))
          amp = jtkwt['confidence_interval'].median()  # Extract estimate (median) from the conf. interval
          best_results['maxamp_ci'] = jtkwt['confidence_interval'].values
          best_results['maxamp_pval'] = jtkwt['pvalue'].values
        else:
          amp = float(np.nanmedian(tmp_clean)) if len(tmp_clean) > 0 else 0.0
        if amp > best_results['maxamp']:
          best_results.update({'bestper': pers, 'bestlag': lag, 'besttau': [abs(param_dic["CJTK"][lagi][2])], 'maxamp': amp})
  JTK_PERIOD = param_dic["INTERVAL"] * best_results['bestper']
  JTK_LAG = param_dic["INTERVAL"] * best_results['bestlag']
  JTK_AMP = float(max(0, best_results['maxamp']))
  return pd.Series([JTK_ADJP, JTK_PERIOD, JTK_LAG, JTK_AMP])


def get_BF(n: int, # sample size
          p: float, # p-value
          z: bool = False, # True if p-value from z-statistic, False if t-statistic
          method: str = "robust", # method for choice of 'b': "JAB", "min", "robust", "balanced"
          upper: float = 10 # upper limit for range of realistic effect sizes
         ) -> float: # Bayes factor in favor of H1
  "Transforms a p-value into Jeffreys' approximate Bayes factor (BF)"
  method_dict = {"JAB": lambda n: 1/n, "min": lambda n: 2/n, "robust": lambda n: max(2/n, 1/np.sqrt(n))}
  if method == "balanced":
    integrand = lambda x: np.exp(-n * x**2 / 4)
    method_dict["balanced"] = lambda n: max(2/n, min(0.5, integrate.quad(integrand, 0, upper)[0]))
  t_statistic = norm.ppf(1 - p/2) if z else t.ppf(1 - p/2, n - 2)
  b = method_dict.get(method, lambda n: 1/n)(n)
  BF = np.exp(0.5 * t_statistic**2) * np.sqrt(b)
  return BF


def get_alphaN(n: int, # sample size
              BF: float = 3, # Bayes factor you would like to match
              method: str = "robust", # method for choice of 'b': "JAB", "min", "robust", "balanced"
              upper: float = 10 # upper limit for range of realistic effect sizes
             ) -> float: # alpha level required to achieve desired evidence
  "Set the alpha level based on sample size via Bayesian-Adaptive Alpha Adjustment"
  method_dict = {"JAB": lambda n: 1/n, "min": lambda n: 2/n, "robust": lambda n: max(2/n, 1/np.sqrt(n))}
  if method == "balanced":
    integrand = lambda x: np.exp(-n * x**2 / 4)
    method_dict["balanced"] = lambda n: max(2/n, min(0.5, integrate.quad(integrand, 0, upper)[0]))
  b = method_dict.get(method, lambda n: 1/n)(n)
  alpha = 1 - chi2.cdf(2 * np.log(BF / np.sqrt(b)), 1)
  print(f"You're working with an alpha of {alpha} that has been adjusted for your sample size of {n}.")
  return alpha


def pi0_tst(p_values: np.ndarray, # array of p-values
           alpha: float = 0.05 # significance threshold for testing
          ) -> float: # estimate of π0, proportion of true null hypotheses
  "estimate the proportion of true null hypotheses in a set of p-values"
  alpha_prime = alpha / (1 + alpha)
  n = len(p_values)
  # Apply the BH procedure at level α'
  sorted_indices = np.argsort(p_values)
  sorted_p_values = p_values[sorted_indices]
  bh_values = (n / rankdata(sorted_p_values)) * sorted_p_values
  corrected_p_values = np.minimum.accumulate(bh_values[::-1])[::-1]
  corrected_p_values_sorted_indices = np.argsort(sorted_indices)
  corrected_p_values = corrected_p_values[corrected_p_values_sorted_indices]
  # Estimate π0
  rejected = corrected_p_values < alpha_prime
  n_rejected = np.sum(rejected)
  pi0_estimate = (n - n_rejected) / n
  return pi0_estimate


def TST_grouped_benjamini_hochberg(identifiers_grouped: Dict[str, List], # dictionary of group : list of glycans
                                 p_values_grouped: Dict[str, List[float]], # dictionary of group : list of p-values
                                 alpha: float # significance threshold for testing
                                ) -> Tuple[Dict[str, float], Dict[str, bool]]: # (glycan:corrected p-value dict, glycan:significant dict)
  "perform the two-stage adaptive Benjamini-Hochberg procedure for multiple testing correction"
  # Initialize results
  adjusted_p_values = {}
  significance_dict = {}
  for group, group_p_values in p_values_grouped.items():
    group_p_values = np.array(group_p_values)
    # Estimate π0 for the group within the Two-Stage method
    pi0_estimate = pi0_tst(group_p_values, alpha)
    if pi0_estimate == 1:
      group_adjusted_p_values = [1.0] * len(group_p_values)
      for identifier, corrected_pval in zip(identifiers_grouped[group], group_adjusted_p_values):
        adjusted_p_values[identifier] = corrected_pval
        significance_dict[identifier] = False
      continue
    n = len(group_p_values)
    sorted_indices = np.argsort(group_p_values)
    sorted_p_values = group_p_values[sorted_indices]
    # Weight the alpha value by π0 estimate
    adjusted_alpha = alpha / max(pi0_estimate, 0.3)
    # Calculate the BH adjusted p-values
    ecdffactor = (np.arange(1, n + 1) / n)
    pvals_corrected_raw = sorted_p_values / (ecdffactor)
    group_adjusted_p_values = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    group_adjusted_p_values_sorted_indices = np.argsort(sorted_indices)
    group_adjusted_p_values = group_adjusted_p_values[group_adjusted_p_values_sorted_indices]
    group_adjusted_p_values = np.minimum(group_adjusted_p_values, 1)
    group_adjusted_p_values = np.maximum(group_adjusted_p_values, group_p_values)
    for identifier, corrected_pval in zip(identifiers_grouped[group], group_adjusted_p_values):
      adjusted_p_values[identifier] = corrected_pval
      significance_dict[identifier] = bool(corrected_pval < adjusted_alpha)
  return adjusted_p_values, significance_dict


def compare_inter_vs_intra_group(cohort_b: pd.DataFrame, # dataframe of glycans as rows and samples as columns of case samples
                            cohort_a: pd.DataFrame, # dataframe of glycans as rows and samples as columns of control samples
                            glycans: List[str], # list of glycans in IUPAC-condensed nomenclature
                            grouped_glycans: Dict[str, List[str]], # dictionary of type group : glycans
                            paired: bool = False # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                           ) -> Tuple[float, float]: # (intra-group correlation, inter-group correlation)
  "estimates intra- and inter-group correlation of a given grouping of glycans via a mixed-effects model"
  reverse_lookup = {k: v for v, l in grouped_glycans.items() for k in l}
  if paired:
    temp = pd.DataFrame(np.log2(abs((cohort_b.values + 1e-8) / (cohort_a.values + 1e-8))))
  else:
    mean_cohort_a = np.mean(cohort_a, axis = 1).values[:, np.newaxis] + 1e-8
    temp = pd.DataFrame(np.log2((cohort_b.values + 1e-8) / mean_cohort_a))
  temp.index = glycans
  temp = temp.reset_index()
  # Melt the dataframe to long format
  temp = temp.melt(id_vars = 'index', var_name = 'glycan', value_name = 'measurement')
  # Rename the columns appropriately
  temp.columns= ["glycan", "sample_id", "diffs"]
  temp["group_id"] = [reverse_lookup[g] for g in temp.glycan]
  # Define the model
  md = smf.mixedlm("diffs ~ C(group_id)", temp,
                     groups = temp["sample_id"],
                     re_formula = "~1",  # Random intercept for glycans
                     vc_formula = {"glycan": "0 + C(glycan)"}) # Variance component for glycans
  # Fit the model
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category = ConvergenceWarning)
    mdf = md.fit()
  # Extract variance components
  var_samples = mdf.cov_re.iloc[0, 0]  # Variance due to differences among groups of glycans (inter-group)
  var_glycans_within_group = mdf.vcomp[0] # Variance due to differences among glycans within the same group (intra-group)
  residual_var = mdf.scale  # Residual variance
  # Total variance
  total_var = var_samples + var_glycans_within_group + residual_var
  # Calculate Intra-group Correlation (ICC)
  icc = var_glycans_within_group / total_var
  # Calculate Inter-group Correlation
  inter_group_corr = var_samples / total_var
  return icc, inter_group_corr


def replace_outliers_with_IQR_bounds(full_row: pd.Series, # row from dataframe, with all but possibly first value numerical
                                   cap_side: str = 'both' # which side(s) to cap outliers on: 'both', 'lower', or 'upper'
                                  ) -> pd.Series: # row with replaced outliers
  "replaces outlier values with row median"
  row = full_row.iloc[1:] if isinstance(full_row.iloc[0], str) else full_row
  # Calculate Q1, Q3, and IQR for each row
  Q1 = row.quantile(0.25)
  Q3 = row.quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5*IQR
  upper_bound = Q3 + 1.5*IQR
  def cap_value(x):
    if cap_side in ['both', 'lower'] and x < lower_bound and x != 0:
      return lower_bound
    elif cap_side in ['both', 'upper'] and x > upper_bound and x != 0:
      return upper_bound
    else:
      return x
  # Define outliers as values outside of Q1 - 1.5*IQR and Q3 + 1.5*IQR
  capped_values = row.apply(cap_value)
  # Replace outliers with row median
  if isinstance(full_row.iloc[0], str):
    full_row.iloc[1:] = capped_values
  else:
    full_row = capped_values
  return full_row


def replace_outliers_winsorization(full_row: pd.Series, # row from dataframe, with all but possibly first value numerical
                                 cap_side: str = 'both' # which side(s) to cap outliers on: 'both', 'lower', or 'upper'
                                ) -> pd.Series: # row with outliers replaced by Winsorization
  "Replaces outlier values using Winsorization"
  row = full_row.iloc[1:] if isinstance(full_row.iloc[0], str) else full_row
  # Apply Winsorization - limits set to match typical IQR outlier detection
  nan_placeholder = row.min() - 1
  row = row.astype(float).fillna(nan_placeholder)
  limit_value = max(0.05, 1/len(row))
  if cap_side == 'both':
    limits = [limit_value, limit_value]
  elif cap_side == 'lower':
    limits = [limit_value, 0]
  elif cap_side == 'upper':
    limits = [0, limit_value]
  else:
    raise ValueError("cap_side must be 'both', 'lower', or 'upper'")
  winsorized_values = winsorize(row, limits = limits)
  winsorized_values = pd.Series(winsorized_values, index = row.index)
  winsorized_values = winsorized_values.replace(nan_placeholder, np.nan)
  if isinstance(full_row.iloc[0], str):
    full_row.iloc[1:] = winsorized_values
  else:
    full_row = winsorized_values
  return full_row


def hotellings_t2(group1: np.ndarray, # comparison group containing numerical data
                 group2: np.ndarray, # comparison group containing numerical data
                 paired: bool = False # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                ) -> Tuple[float, float]: # (F statistic, p-value)
  "Hotelling's T^2 test (the t-test for multivariate comparisons)"
  if paired:
    assert group1.shape == group2.shape, "For paired samples, the size of group1 and group2 should be the same"
    group1 -= group2
    group2 = None
  # Calculate the means and covariances of each group
  n1, p = group1.shape
  mean1 = np.mean(group1, axis = 0)
  cov1 = np.cov(group1, rowvar = False)
  if group2 is not None:
    n2, _ = group2.shape
    mean2 = np.mean(group2, axis = 0)
    cov2 = np.cov(group2, rowvar = False)
  else:
    n2 = 0
    mean2 = np.zeros_like(mean1)
    cov2 = np.zeros_like(cov1)
  # Calculate the difference between the means
  diff = mean1 - mean2
  # Calculate the pooled covariance matrix
  denom = (n1 + n2 - 2)
  pooled_cov = cov1 if denom < 1 else ((n1 - 1) * cov1 + (n2 - 1) * cov2) / denom
  pooled_cov += np.eye(p) * 1e-6
  # Calculate the Hotelling's T^2 statistic
  T2 = (n1 * n2) / (n1 + n2) * diff @ np.linalg.pinv(pooled_cov) @ diff.T
  # Convert the T^2 statistic to an F statistic
  F = 0 if denom < 1 else T2 * (denom + 1 - p) / (denom * p)
  if F == 0:
    return F, 1.0
  # Calculate the p-value of the F statistic
  p_value = f.sf(F, p, n1 + n2 - p - 1)
  return F, p_value


def sequence_richness(counts: np.ndarray # array of counts per feature
                    ) -> int: # number of non-zero features
  "counts number of features with non-zero abundance"
  return (counts != 0).sum()


def shannon_diversity_index(counts: np.ndarray # array of counts
                         ) -> float: # Shannon diversity index value
  "calculates Shannon diversity index"
  proportions = counts / counts.sum()
  return entropy(proportions)


def simpson_diversity_index(counts: np.ndarray # array of counts
                         ) -> float: # Simpson diversity index value
  "calculates Simpson diversity index"
  proportions = counts / counts.sum()
  return 1 - np.sum(proportions**2)


def get_equivalence_test(row_a: np.ndarray, # array of control samples for one glycan/motif
                        row_b: np.ndarray, # array of case samples for one glycan/motif
                        paired: bool = False # whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient)
                       ) -> float: # p-value for equivalence test
  "performs equivalence test (two one-sided t-tests) to test whether differences between group means are considered practically equivalent"
  pooled_std = np.sqrt(((len(row_a) - 1) * np.var(row_a, ddof = 1) + (len(row_b) - 1) * np.var(row_b, ddof = 1)) / (len(row_a) + len(row_b) - 2))
  delta = 0.2 * pooled_std
  low, up = -delta, delta
  return ttost_paired(row_a, row_b, low, up)[0] if paired else ttost_ind(row_a, row_b, low, up)[0]


def clr_transformation(df: pd.DataFrame, # dataframe with features as rows and samples as columns
                      group1: List[Union[str, int]], # column indices/names for first group of samples, usually control
                      group2: List[Union[str, int]], # column indices/names for second group of samples
                      gamma: float = 0.1, # degree of uncertainty that CLR assumption holds
                      custom_scale: Union[float, Dict] = 0 # ratio total signal group2/group1 for scale model (or group_idx:mean/min dict for multivariate)
                     ) -> pd.DataFrame: # CLR-transformed dataframe
  "performs the Center Log-Ratio (CLR) Transformation with scale model adjustment"
  geometric_mean = gmean(df.replace(0, np.nan), axis = 0, nan_policy = 'omit')
  clr_adjusted = np.zeros_like(df.values)
  if gamma and not isinstance(custom_scale, dict):
    group1i = [df.columns.get_loc(c) for c in group1]
    group2i = [df.columns.get_loc(c) for c in group2] if group2 else group1i
    geometric_mean = -np.log2(geometric_mean)
    if group2:
      clr_adjusted[:, group1i] = np.log2(df[group1]) + (geometric_mean[group1i] if not custom_scale else norm.rvs(loc = np.log2(1), scale = gamma, random_state = rng, size = (df.shape[0], len(group1))))
      condition = norm.rvs(loc = geometric_mean[group2i], scale = gamma, random_state = rng, size = (df.shape[0], len(group2))) if not custom_scale else \
                  norm.rvs(loc = np.log2(custom_scale), scale = gamma, random_state = rng, size = (df.shape[0], len(group2)))
      clr_adjusted[:, group2i] = np.log2(df[group2]) + condition
    else:
      clr_adjusted[:, group1i] = np.log2(df[group1]) + norm.rvs(loc = geometric_mean[group1i], scale = gamma, random_state = rng, size = (df.shape[0], len(group1)))
  elif not group2 and isinstance(custom_scale, dict):
    gamma = max(gamma, 0.1)
    for idx in range(df.shape[1]):
      group_id = group1[idx] if isinstance(group1[0], int) else group1[idx].split('_')[1]
      scale_factor = custom_scale.get(group_id, 1)
      clr_adjusted[:, idx] = np.log2(df.iloc[:, idx]) + norm.rvs(loc = np.log2(scale_factor), scale = gamma, random_state = rng, size = df.shape[0])
  else:
    clr_adjusted = np.log2(df) - np.log2(geometric_mean)
  return pd.DataFrame(clr_adjusted, index = df.index, columns = df.columns)


def anosim(df: pd.DataFrame, # square distance matrix
          group_labels_in: List[str], # list of group membership for each sample
          permutations: int = 999 # number of permutations to perform in ANOSIM test
         ) -> Tuple[float, float]: # (ANOSIM R statistic [-1 to 1], p-value)
  "Performs analysis of similarity (ANOSIM) statistical test"
  group_labels = copy.deepcopy(group_labels_in)
  n = df.shape[0]
  condensed_dist = df.values[np.tril_indices(n, k = -1)]
  ranks = rankdata(condensed_dist, method = 'average')
  # Boolean array for within and between group comparisons
  group_matrix = np.equal.outer(group_labels, group_labels)
  within_group_indices = group_matrix[np.tril_indices(n, k = -1)]
  # Mean ranks for within and between groups
  mean_rank_within = np.mean(ranks[within_group_indices])
  mean_rank_between = np.mean(ranks[~within_group_indices])
  # R statistic
  divisor = n * (n - 1) / 4
  R = (mean_rank_between - mean_rank_within) / divisor
  # Permutation test
  permuted_Rs = np.zeros(permutations)
  for i in range(permutations):
    np.random.shuffle(group_labels)
    permuted_group_matrix = np.equal.outer(group_labels, group_labels)
    permuted_within_group_indices = permuted_group_matrix[np.tril_indices(n, k = -1)]
    perm_mean_rank_within = np.mean(ranks[permuted_within_group_indices])
    perm_mean_rank_between = np.mean(ranks[~permuted_within_group_indices])
    permuted_Rs[i] = (perm_mean_rank_between - perm_mean_rank_within) / divisor
  # Calculate the p-value
  p_value = np.sum(permuted_Rs >= R) / permutations
  return R, p_value


def alpha_biodiversity_stats(df: pd.DataFrame, # square distance matrix
                           group_labels: List[str] # list of group membership for each sample
                          ) -> Optional[Tuple[float, float]]: # F statistic and p-value if groups have >1 sample, None otherwise
  "Performs an ANOVA on the respective alpha diversity distance"
  group_counts = Counter(group_labels)
  if all(count > 1 for count in group_counts.values()):
    stat_outputs = pd.DataFrame({'group': group_labels, 'diversity': df.squeeze()})
    grouped_diversity = stat_outputs.groupby('group')['diversity'].apply(list).tolist()
    stats = f_oneway(*grouped_diversity)
    return stats


def calculate_permanova_stat(df: pd.DataFrame, # square distance matrix
                           group_labels: List[str] # list of group membership for each sample
                          ) -> float: # F statistic - higher means effect more likely
  "Performs multivariate analysis of variance"
  unique_groups = np.unique(group_labels)
  n = len(group_labels)
  # Between-group and within-group sums of squares
  ss_total = np.sum(squareform(df)) / 2
  ss_within = 0
  for group in unique_groups:
    group_mask = np.array(group_labels) == group
    group_indices = np.arange(len(group_labels))[group_mask]
    group_matrix = df.values[np.ix_(group_indices, group_indices)]
    ss_within += np.sum(squareform(group_matrix)) / 2
  ss_between = ss_total - ss_within
  # Calculate the PERMANOVA test statistic: pseudo-F
  ms_between = ss_between / max(len(unique_groups) - 1, 1e-10)
  ms_within = ss_within / max(n - len(unique_groups), 1e-10)
  f_stat = ms_between / ms_within
  return f_stat


def permanova_with_permutation(df: pd.DataFrame, # square distance matrix
                             group_labels: List[str], # list of group membership for each sample
                             permutations: int = 999 # number of permutations for test
                            ) -> Tuple[float, float]: # (F statistic, p-value)
  "Performs permutational multivariate analysis of variance (PERMANOVA)"
  observed_f = calculate_permanova_stat(df, group_labels)
  permuted_fs = np.zeros(permutations)
  for i in range(permutations):
    permuted_labels = np.random.permutation(group_labels)
    permuted_fs[i] = calculate_permanova_stat(df, permuted_labels)
  p_value = np.sum(permuted_fs >= observed_f) / permutations
  return observed_f, p_value


def alr_transformation(df: pd.DataFrame, # dataframe with features as rows and samples as columns
                      reference_component_index: int, # row index of feature to be used as reference
                      group1: List[Union[str, int]], # column indices/names for first group of samples, usually control
                      group2: List[Union[str, int]], # column indices/names for second group of samples
                      gamma: float = 0.1, # degree of uncertainty that CLR assumption holds
                      custom_scale: Union[float, Dict] = 0 # ratio total signal group2/group1 for scale model (or group_idx:mean/min dict for multivariate)
                     ) -> pd.DataFrame: # ALR-transformed dataframe
  "Given a reference feature, performs additive log-ratio transformation (ALR) on the data"
  reference_values = df.iloc[reference_component_index, :]
  alr_transformed = np.zeros_like(df.values)
  group1i = [df.columns.get_loc(c) for c in group1]
  group2i = [df.columns.get_loc(c) for c in group2] if group2 else group1i
  if not isinstance(custom_scale, dict):
    if custom_scale:
      alr_transformed[:, group1i] = df.iloc[:, group1i].subtract(reference_values.iloc[group1i] - norm.rvs(loc = np.log2(1), scale = gamma, random_state = rng, size = len(group1i)), axis = 1)
    else:
      alr_transformed[:, group1i] = df.iloc[:, group1i].subtract(reference_values.iloc[group1i])
    scale_adjustment = np.log2(custom_scale) if custom_scale else 0
    alr_transformed[:, group2i] = df.iloc[:, group2i].subtract(reference_values.iloc[group2i] - norm.rvs(loc = scale_adjustment, scale = gamma, random_state = rng, size = len(group2i)), axis = 1)
  else:
    gamma = max(gamma, 0.1)
    for idx in range(df.shape[1]):
      group_id = group1[idx] if isinstance(group1[0], int) else group1[idx].split('_')[1]
      scale_factor = custom_scale.get(group_id, 1)
      reference_adjusted = reference_values[idx] - norm.rvs(loc = np.log2(scale_factor), scale = gamma, random_state = rng)
      alr_transformed[:, idx] = df.iloc[:, idx] - reference_adjusted
  alr_transformed = pd.DataFrame(alr_transformed, index = df.index, columns = df.columns)
  alr_transformed = alr_transformed.drop(index = reference_values.name)
  alr_transformed = alr_transformed.reset_index(drop=True)
  return alr_transformed


def get_procrustes_scores(df: pd.DataFrame, # dataframe with features as rows and samples as columns
                         group1: List[Union[str, int]], # column indices/names for first group of samples, usually control
                         group2: List[Union[str, int]], # column indices/names for second group of samples
                         paired: bool = False, # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                         custom_scale: Union[float, Dict] = 0 # ratio total signal group2/group1 for scale model (or group_idx:mean/min dict)
                        ) -> Tuple[List[float], List[float], List[float]]: # (Procrustes scores, correlations, variances)
  "For each feature, estimates its value as ALR reference component"
  if isinstance(group1[0], int):
    group1 = [df.columns.tolist()[k] for k in group1]
    group2 = [df.columns.tolist()[k] for k in group2]
  df = df.iloc[:, 1:].astype(float)
  ref_matrix = clr_transformation(df, group1, group2, gamma = 0.01, custom_scale = custom_scale)
  df = np.log2(df)
  if group2:
    if paired:
      differences = df[group1].values - df[group2].values
      variances = np.var(differences, axis = 1, ddof = 1)
    else:
      var_group1 = df[group1].var(axis = 1)
      var_group2 = df[group2].var(axis = 1)
      variances = abs(var_group1 - var_group2)
  else:
    variances = abs(df[group1].var(axis = 1))
  procrustes_corr = [1 - procrustes(ref_matrix.drop(ref_matrix.index[i]),
                                    alr_transformation(df, i, group1, group2, gamma = 0.01, custom_scale = custom_scale))[2] for i in range(df.shape[0])]
  return [a * (1/b) for a, b in zip(procrustes_corr, variances)], procrustes_corr, variances


def get_additive_logratio_transformation(df: pd.DataFrame, # dataframe with features as rows and samples as columns
                                       group1: List[Union[str, int]], # column indices/names for first group of samples
                                       group2: List[Union[str, int]], # column indices/names for second group of samples
                                       paired: bool = False, # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                                       gamma: float = 0.1, # degree of uncertainty that CLR assumption holds
                                       custom_scale: Union[float, Dict] = 0 # ratio total signal group2/group1 for scale model
                                      ) -> pd.DataFrame: # ALR-transformed dataframe
  "Identifies ALR reference component and transforms data according to ALR"
  scores, procrustes_corr, variances = get_procrustes_scores(df, group1, group2, paired = paired, custom_scale = custom_scale)
  ref_component = np.argmax(scores)
  ref_component_string = df.iloc[:, 0].values[ref_component]
  print(f"Reference component for ALR is {ref_component_string}, with Procrustes correlation of {procrustes_corr[ref_component]} and variance of {variances[ref_component]}")
  if procrustes_corr[ref_component] < 0.9 or variances[ref_component] > 0.1:
    print("Metrics of chosen reference component not good enough for ALR; switching to CLR instead.")
    df.iloc[:, 1:] = clr_transformation(df.iloc[:, 1:], group1, group2, gamma = gamma, custom_scale = custom_scale)
    return df
  glycans = df.iloc[:, 0].values.tolist()
  glycans = glycans[:ref_component] + glycans[ref_component+1:]
  alr = alr_transformation(np.log2(df.iloc[:, 1:]), ref_component, group1, group2, gamma = gamma, custom_scale = custom_scale)
  alr.insert(loc = 0, column = 'glycan', value = glycans)
  return alr


def correct_multiple_testing(pvals: Union[List[float], np.ndarray], # list of raw p-values
                           alpha: float, # p-value threshold for statistical significance
                           correction_method: str = "two-stage" # "two-stage" or "one-stage" Benjamini-Hochberg
                          ) -> Tuple[List[float], List[bool]]: # (corrected p-values, significance True/False)
  "Corrects p-values for multiple testing, by default with the two-stage Benjamini-Hochberg procedure"
  if not isinstance(pvals, list):
    pvals = pvals.tolist()
  corrpvals = multipletests(pvals, method = 'fdr_tsbh' if correction_method == "two-stage" else 'fdr_bh')[1]
  corrpvals = [p if p >= pvals[i] else pvals[i] for i, p in enumerate(corrpvals)]
  significance = [bool(p < alpha) for p in corrpvals]
  if sum(significance) > 0.9*len(significance):
    print("Significance inflation detected. The CLR/ALR transformation possibly cannot handle this dataset. Consider running again with a higher gamma value.\
             Proceed with caution; for now switching to Bonferroni correction to be conservative about this.")
    res = multipletests(pvals, method = 'bonferroni')
    corrpvals, alpha = res[1], res[3]
    significance = [bool(p < alpha) for p in corrpvals]
  return corrpvals, significance


def omega_squared(row: Union[pd.Series, np.ndarray], # values for one feature
                 groups: List[str] # list indicating group membership with indices per column
                ) -> float: # effect size as omega squared
  "Calculates Omega squared, as an effect size in an ANOVA setting"
  long_df = pd.DataFrame({'value': row, 'group': groups})
  model = ols('value ~ C(group)', data = long_df).fit()
  anova_results = anova_lm(model, typ = 2)
  ss_total = sum(model.resid ** 2) + anova_results['sum_sq'].sum()
  omega_squared = (anova_results.at['C(group)', 'sum_sq'] - (anova_results.at['C(group)', 'df'] * model.mse_resid)) / (ss_total + model.mse_resid)
  return omega_squared


def get_glycoform_diff(df_res: pd.DataFrame, # result from .motif.analysis.get_differential_expression
                      alpha: float = 0.05, # significance threshold for testing
                      level: str = 'peptide' # analyze at 'peptide' or 'protein' level
                     ) -> pd.DataFrame: # df with differential expression results, p-vals (Fisher’s Combined Probability Test), significance, effect sizes (Cohen's d)
  "Calculates differential expression of glycoforms from either a peptide or a whole protein"
  label_col = 'Glycosite' if 'Glycosite' in df_res.columns else 'Glycan'
  if level == 'protein':
    df_res[label_col] = [k.split('_')[0] for k in df_res[label_col]]
  else:
    df_res[label_col] = ['_'.join(k.split('_')[:-1]) for k in df_res[label_col]]
  grouped = df_res.groupby(label_col)['corr p-val'].apply(lambda p: combine_pvalues(p)[1]) # Fisher’s Combined Probability Test
  mean_effect_size = df_res.groupby(label_col)['Effect size'].mean()
  pvals, sig = correct_multiple_testing(grouped, alpha)
  df_out = pd.DataFrame({'Glycosite': grouped.index, 'corr p-val': pvals, 'significant': sig, 'Effect size': mean_effect_size.values})
  return df_out.sort_values(by = 'corr p-val')


def get_glm(group: pd.DataFrame, # longform data of glycoform abundances for a glycosite
           glycan_features: List[str] = ['H', 'N', 'A', 'F', 'G'] # extracted glycan features to consider as variables
          ) -> Tuple[Union[str, str], List[str]]: # (fitted GLM or failure message, list of variables)
  "given glycoform data from a glycosite, constructs & fits a GLM formula for main+interaction effects"
  retained_vars = [c for c in glycan_features if c in group.columns and max(group[c]) > 0]
  if not retained_vars:
    return ("No variables retained", [])
  base_formula = 'Abundance ~ '
  formula_parts = ['Condition']
  formula_parts += [f'{col} + {col}_Condition' for col in retained_vars] # Main and interaction effects
  for col in retained_vars:
    group[f'{col}_Condition'] = group[col] * group['Condition']
  formula = base_formula + ' + '.join(formula_parts)
  try:
    with np.errstate(divide = 'ignore'):
      model = smf.glm(formula = formula, data = group, family = sm.families.Gaussian()).fit()
    return model, retained_vars
  except Exception as e:
    return (f"GLM fitting failed: {str(e)}", [])


def process_glm_results(df: pd.DataFrame, # CLR-transformed glycoproteomics data, rows glycoforms, columns samples
                       alpha: float, # significance threshold
                       glycan_features: List[str] # extracted glycan features to consider as variables
                      ) -> pd.DataFrame: # regression coefficients, p-values, and significance for each condition/interaction
  "tests for interaction effects of glycan features and the condition on glycoform abundance via a GLM"
  results = df.groupby('Glycosite', group_keys = False)[df.columns].apply(lambda x: get_glm(x.reset_index(drop = True), glycan_features = glycan_features))
  all_retained_vars = set()
  for _, retained_vars in results:
    all_retained_vars.update(retained_vars)
  int_terms = ['Condition'] + [f'{v}_Condition' for v in all_retained_vars]
  out = {idx: [v.pvalues.get(term, 1.0) for term in int_terms] if not isinstance(v, str) else [1.0] * len(int_terms) for idx, (v, _) in results.items()}
  out2 = {idx: [v.params.get(term, 0.0) for term in int_terms] if not isinstance(v, str) else [0.0] * len(int_terms) for idx, (v, _) in results.items()}
  df_pvals = pd.DataFrame(out).T
  df_coefs = pd.DataFrame(out2).T
  df_pvals.columns = int_terms
  df_coefs.columns = int_terms
  df_out = pd.DataFrame(index = df_pvals.index)
  for term in int_terms:
    corrpvals, significance = correct_multiple_testing(df_pvals[term], alpha)
    df_out[f'{term}_coefficient'] = df_coefs[term]
    df_out[f'{term}_corr_pval'] = corrpvals
    df_out[f'{term}_significant'] = significance
  return df_out.sort_values(by = 'Condition_corr_pval')


def partial_corr(x: np.ndarray, # typically values from a column or row
                y: np.ndarray, # typically values from a column or row
                controls: np.ndarray, # variables correlated with x or y
                motifs: bool = False # whether to analyze full sequences or motifs
               ) -> Tuple[float, float]: # (regularized partial correlation coefficient, p-value from Spearman correlation of residuals)
  "Compute regularized partial correlation of x and y, controlling for multiple other variables in controls"
  # Fit regression models
  alpha = 0.1 if motifs else 0.25
  beta_x = Ridge(alpha = alpha).fit(controls, x).coef_
  beta_y = Ridge(alpha = alpha).fit(controls, y).coef_
  # Compute residuals
  res_x = x - controls.dot(beta_x)
  res_y = y - controls.dot(beta_y)
  # Compute correlation of residuals
  corr, pval = spearmanr(res_x, res_y)
  return corr, pval


def estimate_technical_variance(df: pd.DataFrame, # dataframe with abundances in cols
                              group1: List[Union[str, int]], # column indices/names for first group of samples
                              group2: List[Union[str, int]], # column indices/names for second group of samples
                              num_instances: int = 128, # number of Monte Carlo instances to sample
                              gamma: float = 0.1, # uncertainty parameter for CLR transformation scale
                              custom_scale: Union[float, Dict] = 0 # ratio total signal group2/group1 for scale model
                             ) -> pd.DataFrame: # transformed df (features, samples*num_instances) with CLR-transformed Monte Carlo instances
  "Monte Carlo sampling from Dirichlet distribution with relative abundances as concentration, followed by CLR transformation"
  df = df.apply(lambda col: (col / col.sum())*5000, axis = 0)
  features, samples = df.shape
  transformed_data = np.zeros((features, samples, num_instances))
  for j in range(samples):
    dirichlet_samples = dirichlet.rvs(alpha = df.iloc[:, j], random_state = rng, size = num_instances)
    # CLR Transformation for each Monte Carlo instance
    for n in range(num_instances):
      sample_instance = pd.DataFrame(dirichlet_samples[n, :])
      transformed_data[:, j, n] = clr_transformation(sample_instance, sample_instance.columns.tolist(), [],
                                                     gamma = gamma, custom_scale = custom_scale).squeeze()
  columns = [col for col in df.columns for _ in range(num_instances)]
  transformed_data_2d = transformed_data.reshape((features, samples* num_instances))
  transformed_df = pd.DataFrame(transformed_data_2d, columns = columns)
  return transformed_df


def perform_tests_monte_carlo(group_a: pd.DataFrame, # rows as features, columns as sample instances from one condition
                            group_b: pd.DataFrame, # rows as features, columns as sample instances from one condition
                            num_instances: int = 128, # number of Monte Carlo instances to sample
                            paired: bool = False # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                           ) -> Tuple[List[float], List[float], List[float]]: # (uncorrected p-vals, corrected p-vals, effect sizes)
  "Perform tests on each Monte Carlo instance, apply Benjamini-Hochberg correction, calculate effect sizes"
  num_features, _ = group_a.shape
  avg_uncorrected_p_values, avg_corrected_p_values, avg_effect_sizes = np.zeros(num_features), np.zeros(num_features), np.zeros(num_features)
  for instance in range(num_instances):
    instance_p_values = []
    instance_effect_sizes = []
    for feature in range(num_features):
      sample_a = group_a.iloc[feature, instance::num_instances].values
      sample_b = group_b.iloc[feature, instance::num_instances].values
      p_value = ttest_rel(sample_b, sample_a)[1] if paired else ttest_ind(sample_b, sample_a, equal_var = False)[1]
      effect_size, _ = cohen_d(sample_b, sample_a, paired = paired)
      instance_p_values.append(p_value)
      instance_effect_sizes.append(effect_size)
    # Apply Benjamini-Hochberg correction for multiple testing within the instance
    avg_uncorrected_p_values += instance_p_values
    corrected_p_values = multipletests(instance_p_values, method = 'fdr_tsbh')[1]
    avg_corrected_p_values += corrected_p_values
    avg_effect_sizes += instance_effect_sizes
  avg_uncorrected_p_values /= num_instances
  avg_corrected_p_values /= num_instances
  avg_corrected_p_values = [p if p >= avg_uncorrected_p_values[i] else avg_uncorrected_p_values[i] for i, p in enumerate(avg_corrected_p_values)]
  avg_effect_sizes /= num_instances
  return avg_uncorrected_p_values, avg_corrected_p_values, avg_effect_sizes
