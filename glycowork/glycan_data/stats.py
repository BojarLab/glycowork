import pandas as pd
import numpy as np
import math
import warnings
from collections import defaultdict, Counter
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from scipy.special import gammaln
from scipy.stats import wilcoxon, rankdata, norm, chi2, t, f, entropy, gmean, f_oneway, combine_pvalues
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


def fast_two_sum(a, b):
  """Assume abs(a) >= abs(b)"""
  x = int(a) + int(b)
  y = b - (x - int(a))
  return [x] if y == 0 else [x, y]


def two_sum(a, b):
  """For unknown order of a and b"""
  x = int(a) + int(b)
  y = (a - (x - int(b))) + (b - (x - int(a)))
  return [x] if y == 0 else [x, y]


def expansion_sum(*args):
  """For the expansion sum of floating points"""
  g = sorted(args, reverse = True)
  q, *h = fast_two_sum(np.array(g[0]), np.array(g[1]))
  for val in g[2:]:
    z = two_sum(q, np.array(val))
    q, *extra = z
    if extra:
      h += extra
  return [h, q] if h else q


def hlm(z):
  """Hodges-Lehmann estimator of the median"""
  z = np.array(z)
  zz = np.add.outer(z, z)
  zz = zz[np.tril_indices(len(z))]
  return np.median(zz) / 2


def update_cf_for_m_n(m, n, MM, cf):
  """Constructs cumulative frequency table for experimental parameters defined in the function 'jtkinit'"""
  P = min(m + n, MM)
  for t_temp in range(n + 1, P + 1):  # Zero-based offset t_temp
    for u in range(MM, t_temp - 1, -1):  # One-based descending index u
      cf[u] = expansion_sum(cf[u], -cf[u - t_temp])  # Shewchuk algorithm
  Q = min(m, MM)
  for s in range(1, Q + 1): # Zero-based offset s
    for u in range(s, MM + 1):  # One-based descending index u
      cf[u] = expansion_sum(cf[u], cf[u - s])  # Shewchuk algorithm


def cohen_d(x, y, paired = False):
  """calculates effect size between two groups\n
  | Arguments:
  | :-
  | x (list or 1D-array): comparison group containing numerical data
  | y (list or 1D-array): comparison group containing numerical data
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False\n
  | Returns:
  | :-
  | Returns Cohen's d (and its variance) as a measure of effect size (0.2 small; 0.5 medium; 0.8 large)
  """
  if paired:
    assert len(x) == len(y), "For paired samples, the size of x and y should be the same"
    diff = np.array(x) - np.array(y)
    diff_std = np.std(diff, ddof = 1)
    if diff_std == 0:
      return 0, 0
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


def mahalanobis_distance(x, y, paired = False):
  """calculates effect size between two groups in a multivariate comparison\n
  | Arguments:
  | :-
  | x (list or 1D-array or dataframe): comparison group containing numerical data
  | y (list or 1D-array or dataframe): comparison group containing numerical data
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False\n
  | Returns:
  | :-
  | Returns Mahalanobis distance as a measure of effect size
  """
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


def mahalanobis_variance(x, y, paired = False):
  """Estimates variance of Mahalanobis distance via bootstrapping\n
  | Arguments:
  | :-
  | x (list or 1D-array or dataframe): comparison group containing numerical data
  | y (list or 1D-array or dataframe): comparison group containing numerical data
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False\n
  | Returns:
  | :-
  | Returns Mahalanobis distance as a measure of effect size
  """
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


def variance_stabilization(data, groups = None):
  """Variance stabilization normalization\n
  | Arguments:
  | :-
  | data (dataframe): pandas dataframe with glycans/motifs as indices and samples as columns
  | groups (nested list): list containing lists of column names of samples from same group for group-specific normalization; otherwise global; default:None\n
  | Returns:
  | :-
  | Returns a dataframe in the same style as the input
  """
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
  """Parameters
  (adapted from https://github.com/yuenshingyan/MissForest)
  ----------
  regressor : estimator object.
  A object of that type is instantiated for each imputation.
  This object is assumed to implement the scikit-learn estimator API.

  n_iter : int
  Determines the number of iterations for the imputation process.
  """
  def __init__(self, regressor = RandomForestRegressor(n_jobs = -1), max_iter = 5, tol = 1e-5):
    self.regressor = regressor
    self.max_iter = max_iter
    self.tol = tol

  def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
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


def impute_and_normalize(df, groups, impute = True, min_samples = 0.1):
    """given a dataframe, discards rows with too many missings, imputes the rest, and normalizes\n
    | Arguments:
    | :-
    | df (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns
    | groups (list): nested list of column name lists, one list per group
    | impute (bool): replaces zeroes with predictions from MissForest; default:True
    | min_samples (float): Percent of the samples that need to have non-zero values for glycan to be kept; default: 10%\n
    | Returns:
    | :-
    | Returns a dataframe in the same style as the input 
    """
    if min_samples:
      min_count = max(np.floor(df.shape[1] * min_samples), 2) + 1
      mask = (df != 0).sum(axis = 1) >= min_count
      df = df[mask].reset_index(drop = True)
    colname = df.columns[0]
    glycans = df[colname]
    df = df.iloc[:, 1:]
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


def variance_based_filtering(df, min_feature_variance = 0.02):
    """Variance-based filtering of features\n
    | Arguments:
    | :-
    | df (dataframe): dataframe containing glycan sequences in index and samples in columns
    | min_feature_variance (float): Minimum variance to include a feature in the analysis; default: 2%\n
    | Returns:
    | :-
    | filtered_df (DataFrame): DataFrame with remaining glycans (variance > min_feature_variance) as indices and samples in columns.
    | discarded_df (DataFrame): DataFrame with discarded glycans (variance <= min_feature_variance) as indices and samples in columns.
    """
    variances = df.var(axis = 1)
    filtered_df = df.loc[variances > min_feature_variance]
    discarded_df = df.loc[variances <= min_feature_variance]
    return filtered_df, discarded_df


def jtkdist(timepoints, param_dic, reps = 1, normal = False):
  """Precalculates all possible JT test statistic permutation probabilities for reference later, speeding up the
  | analysis. Calculates the exact null distribution using the Harding algorithm.\n
  | Arguments:
  | :-
  | timepoints (int): number of timepoints within the experiment.
  | param_dic (dict): dictionary carrying around the parameter values
  | reps (int): number of replicates within each timepoint.
  | normal (bool): a flag for normal approximation if maximum possible negative log p-value is too large.\n
  | Returns:
  | :-
  | Returns statistical values, added to 'param_dic'.
  """
  timepoints = timepoints if isinstance(timepoints, int) else timepoints.sum()
  tim = np.full(timepoints, reps) if reps != timepoints else reps  # Support for unbalanced replication (unequal replicates in all groups)
  maxnlp = gammaln(np.sum(tim)) - np.sum(np.log(np.arange(1, np.max(tim)+1)))
  limit = math.log(float('inf'))
  normal = normal or (maxnlp > limit - 1)  # Switch to normal approximation if maxnlp is too large
  nn = sum(tim)  # Number of data values (Independent of period and lag)
  M = (nn ** 2 - np.sum(np.square(tim))) / 2  # Max possible jtk statistic
  param_dic.update({"GRP_SIZE": tim, "NUM_GRPS": len(tim), "NUM_VALS": nn,
                    "MAX": M, "DIMS": [int(nn * (nn - 1) / 2), 1]})
  if normal:
    param_dic["VAR"] = (nn ** 2 * (2 * nn + 3) - np.sum(np.square(tim) * (2 * t + 3) for t in tim)) / 72  # Variance of JTK
    param_dic["SDV"] = math.sqrt(param_dic["VAR"])  # Standard deviation of JTK
    param_dic["EXV"] = M / 2  # Expected value of JTK
    param_dic["EXACT"] = False
  MM = int(M // 2)  # Mode of this possible alternative to JTK distribution
  cf = [1] * (MM + 1)  # Initial lower half cumulative frequency (cf) distribution
  size = sorted(tim)  # Sizes of each group of known replicate values, in ascending order for fastest calculation
  k = len(tim)  # Number of groups of replicates
  N = [size[k-1]]
  if k > 2:
    for i in range(k - 1, 1, -1):  # Count permutations using the Harding algorithm
      N.insert(0, (size[i] + N[0]))
  for m, n in zip(size[:-1], N):
    update_cf_for_m_n(m, n, MM, cf)
  cf = np.array(cf)
  # cf now contains the lower half cumulative frequency distribution
  # append the symmetric upper half cumulative frequency distribution to cf
  if M % 2:   # jtkcf = upper-tail cumulative frequencies for all integer jtk
    jtkcf = np.concatenate((cf, 2 * cf[MM] - cf[:MM][::-1], [2 * cf[MM]]))[::-1]
  else:
    jtkcf = np.concatenate((cf, cf[MM - 1] + cf[MM] - cf[:MM-1][::-1], [cf[MM - 1] + cf[MM]]))[::-1]
  ajtkcf = list((jtkcf[i - 1] + jtkcf[i]) / 2 for i in range(1, len(jtkcf)))  # interpolated cumulative frequency values for all half-intgeger jtk
  cf = [ajtkcf[(j - 1) // 2] if j % 2 == 0 else jtkcf[j // 2] for j in [i for i in range(1, 2 * int(M) + 2)]]
  param_dic["CP"] = [c / jtkcf[0] for c in cf]  # all upper-tail p-values
  return param_dic


def jtkinit(periods, param_dic, interval = 1, replicates = 1):
  """Defines the parameters of the simulated sine waves for reference later.\n
  | Each molecular species within the analysis is matched to the optimal wave defined here, and the parameters
  | describing that wave are attributed to the molecular species.\n
  | Arguments:
  | :-
  | periods (list): the possible periods of rhytmicity in the biological data (valued as 'number of timepoints').
  | (note: periods can accept multiple values (ie, you can define circadian rhythms as between 22, 24, 26 hours))
  | param_dic (dict): dictionary carrying around the parameter values
  | interval (int): the number of units of time (arbitrary) between each experimental timepoint.
  | replicates (int): number of replicates within each group.\n
  | Returns:
  | :-
  | Returns values describing waveforms, added to 'param_dic'.
  """
  param_dic["INTERVAL"] = interval
  if len(periods) > 1:
    param_dic["PERIODS"] = list(periods)
  else:
    param_dic["PERIODS"] = list(periods)
  param_dic["PERFACTOR"] = np.concatenate([np.repeat(i, ti) for i, ti in enumerate(periods, start = 1)])
  tim = np.array(param_dic["GRP_SIZE"])
  timepoints = int(param_dic["NUM_GRPS"])
  timerange = np.arange(timepoints)  # Zero-based time indices
  param_dic["SIGNCOS"] = np.zeros((periods[0], ((math.floor(timepoints / (periods[0]))*int(periods[0]))* replicates)), dtype = int)
  for i, period in enumerate(periods):
    time2angle = np.array([(2*round(math.pi, 4))/period])  # convert time to angle using an ~pi value
    theta = timerange*time2angle  # zero-based angular values across time indices
    cos_v = np.cos(theta)  # unique cosine values at each time point
    cos_r = np.repeat(rankdata(cos_v), np.max(tim))  # replicated ranks of unique cosine values
    cgoos = np.sign(np.subtract.outer(cos_r, cos_r)).astype(int)
    lower_tri = []
    for col in range(len(cgoos)):
      for row in range(col + 1, len(cgoos)):
        lower_tri.append(cgoos[row, col])
    cgoos = np.array(lower_tri)
    cgoosv = np.array(cgoos).reshape(param_dic["DIMS"])
    param_dic["CGOOSV"] = []
    param_dic["CGOOSV"].append(np.zeros((cgoos.shape[0], period)))
    param_dic["CGOOSV"][i][:, 0] = cgoosv[:, 0]
    cycles = math.floor(timepoints / period)
    jrange = np.arange(cycles * period)
    cos_s = np.sign(cos_v)[jrange]
    cos_s = np.repeat(cos_s, (tim[jrange]))
    if replicates == 1:
      param_dic["SIGNCOS"][:, i] = cos_s
    else:
      param_dic["SIGNCOS"][i] = cos_s
    for j in range(1, period):  # One-based half-integer lag index j
      delta_theta = j * time2angle / 2  # Angles of half-integer lags
      cos_v = np.cos(theta + delta_theta)  # Cycle left
      cos_r = np.concatenate([np.repeat(val, num) for val, num in zip(rankdata(cos_v), tim)]) # Phase-shifted replicated ranks
      cgoos = np.sign(np.subtract.outer(cos_r, cos_r)).T
      mask = np.triu(np.ones(cgoos.shape), k = 1).astype(bool)
      mask[np.diag_indices(mask.shape[0])] = False
      cgoos = cgoos[mask]
      cgoosv = cgoos.reshape(param_dic["DIMS"])
      matrix_i = param_dic["CGOOSV"][i]
      matrix_i[:, j] = cgoosv.flatten()
      param_dic["CGOOSV[i]"] = matrix_i
      cos_v = cos_v.flatten()
      cos_s = np.sign(cos_v)[jrange]
      cos_s = np.repeat(cos_s, (tim[jrange]))
      if replicates == 1:
        param_dic["SIGNCOS"][:, j] = cos_s
      else:
        param_dic["SIGNCOS"][j] = cos_s
  return param_dic


def jtkstat(z, param_dic):
  """Determines the JTK statistic and p-values for all model phases, compared to expression data.\n
  | Arguments:
  | :-
  | z (pd.DataFrame): expression data for a molecule ordered in groups, by timepoint.
  | param_dic (dict): a dictionary containing parameters defining model waveforms.\n
  | Returns:
  | :-
  | Returns an updated parameter dictionary where the appropriate model waveform has been assigned to the
  | molecules in the analysis.
  """
  param_dic["CJTK"] = []
  M = param_dic["MAX"]
  z = np.array(z)
  foosv = np.sign(np.subtract.outer(z, z)).T  # Due to differences in the triangle indexing of R / Python we need to transpose and select upper triangle rather than the lower triangle
  mask = np.triu(np.ones(foosv.shape), k = 1).astype(bool) # Additionally, we need to remove the middle diagonal from the tri index
  mask[np.diag_indices(mask.shape[0])] = False
  foosv = foosv[mask].reshape(param_dic["DIMS"])
  for i in range(param_dic["PERIODS"][0]):
    cgoosv = param_dic["CGOOSV"][0][:, i]
    S = np.nansum(np.diag(foosv * cgoosv))
    jtk = (abs(S) + M) / 2  # Two-tailed JTK statistic for this lag and distribution
    if S == 0:
      param_dic["CJTK"].append([1, 0, 0])
    elif param_dic.get("EXACT", False):
      jtki = 1 + 2 * int(jtk)  # index into the exact upper-tail distribution
      p = 2 * param_dic["CP"][jtki-1]
      param_dic["CJTK"].append([p, S, S / M])
    else:
      p = 2 * norm.cdf(-(jtk - 0.5), -param_dic["EXV"], param_dic["SDV"])
      param_dic["CJTK"].append([p, S, S / M])  # include tau = s/M for this lag and distribution
  return param_dic


def jtkx(z, param_dic, ampci = False):
  """Deployment of jtkstat for repeated use, and parameter extraction\n
  | Arguments:
  | :-
  | z (pd.dataframe): expression data ordered in groups, by timepoint.
  | param_dic (dict): a dictionary containing parameters defining model waveforms.
  | ampci (bool): flag for calculating amplitude confidence interval (TRUE = compute); default=False.\n
  | Returns:
  | :-
  | Returns an updated parameter dictionary containing the optimal waveform parameters for each molecular species.
  """
  param_dic = jtkstat(z, param_dic)  # Calculate p and S for all phases
  pvals = [cjtk[0] for cjtk in param_dic["CJTK"]]  # Exact two-tailed p values for period/phase combos
  padj = multipletests(pvals, method = 'fdr_bh')[1]
  JTK_ADJP = min(padj)  # Global minimum adjusted p-value
  def groupings(padj, param_dic):
    d = defaultdict(list)
    for i, value in enumerate(padj):
      key = param_dic["PERFACTOR"][i]
      d[key].append(value)
    return dict(d)
  dpadj = groupings(padj, param_dic)
  padj = np.array(pd.DataFrame(dpadj.values()).T)
  minpadj = [padj[i].min() for i in range(0, np.shape(padj)[1])]  # Minimum adjusted p-values for each period
  if len(param_dic["PERIODS"]) > 1:
    pers_index = np.where(JTK_ADJP == minpadj)[0]  # indices of all optimal periods
  else:
    pers_index = 0
  pers = param_dic["PERIODS"][int(pers_index)]    # all optimal periods
  lagis = np.where(padj == JTK_ADJP)[0]  # list of optimal lag indice for each optimal period
  best_results = {'bestper': 0, 'bestlag': 0, 'besttau': 0, 'maxamp': 0, 'maxamp_ci': 2, 'maxamp_pval': 0}
  sc = np.transpose(param_dic["SIGNCOS"])
  w = (z[:len(sc)] - hlm(z[:len(sc)])) * math.sqrt(2)
  for i in range(abs(pers)):
    for lagi in lagis:
      S = param_dic["CJTK"][lagi][1]
      s = np.sign(S) if S != 0 else 1
      lag = (pers + (1 - s) * pers / 4 - lagi / 2) % pers
      tmp = s * w * sc[:, lagi]
      amp = hlm(tmp)  # Allows missing values
      if ampci:
        jtkwt = pd.DataFrame(wilcoxon(tmp[np.isfinite(tmp)], zero_method = 'wilcox', correction = False,
                                              alternatives = 'two-sided', mode = 'exact'))
        amp = jtkwt['confidence_interval'].median()  # Extract estimate (median) from the conf. interval
        best_results['maxamp_ci'] = jtkwt['confidence_interval'].values
        best_results['maxamp_pval'] = jtkwt['pvalue'].values
      if amp > best_results['maxamp']:
        best_results.update({'bestper': pers, 'bestlag': lag, 'besttau': [abs(param_dic["CJTK"][lagi][2])], 'maxamp': amp})
  JTK_PERIOD = param_dic["INTERVAL"] * best_results['bestper']
  JTK_LAG = param_dic["INTERVAL"] * best_results['bestlag']
  JTK_AMP = float(max(0, best_results['maxamp']))
  return pd.Series([JTK_ADJP, JTK_PERIOD, JTK_LAG, JTK_AMP])


def get_BF(n, p, z = False, method = "robust", upper = 10):
  """Transforms a p-value into Jeffreys' approximate Bayes factor (BF)\n
  | Arguments:
  | :-
  | n (int): Sample size.
  | p (float): The p-value.
  | z (bool): True if the p-value is based on a z-statistic, False if t-statistic; default:False
  | method (str): Method used for the choice of 'b'. Options are "JAB", "min", "robust", "balanced"; default:'robust'
  | upper (float): The upper limit for the range of realistic effect sizes. Only relevant when method="balanced"; default:10\n
  | Returns:
  | :-
  | float: A numeric value for the BF in favour of H1.
  """
  method_dict = {
    "JAB": lambda n: 1/n,
    "min": lambda n: 2/n,
    "robust": lambda n: max(2/n, 1/np.sqrt(n)),
    }
  if method == "balanced":
    integrand = lambda x: np.exp(-n * x**2 / 4)
    method_dict["balanced"] = lambda n: max(2/n, min(0.5, integrate.quad(integrand, 0, upper)[0]))
  t_statistic = norm.ppf(1 - p/2) if z else t.ppf(1 - p/2, n - 2)
  b = method_dict.get(method, lambda n: 1/n)(n)
  BF = np.exp(0.5 * t_statistic**2) * np.sqrt(b)
  return BF


def get_alphaN(n, BF = 3, method = "robust", upper = 10):
  """Set the alpha level based on sample size via Bayesian-Adaptive Alpha Adjustment.\n
  | Arguments:
  | :-
  | n (int): Sample size.
  | BF (float): Bayes factor you would like to match; default:3
  | method (str): Method used for the choice of 'b'. Options are "JAB", "min", "robust", "balanced"; default:"robust"
  | upper (float): The upper limit for the range of realistic effect sizes. Only relevant when method="balanced"; default:10\n
  | Returns:
  | :-
  | float: Numeric alpha level required to achieve the desired level of evidence.
  """
  method_dict = {
    "JAB": lambda n: 1/n,
    "min": lambda n: 2/n,
    "robust": lambda n: max(2/n, 1/np.sqrt(n)),
    }
  if method == "balanced":
    integrand = lambda x: np.exp(-n * x**2 / 4)
    method_dict["balanced"] = lambda n: max(2/n, min(0.5, integrate.quad(integrand, 0, upper)[0]))
  b = method_dict.get(method, lambda n: 1/n)(n)
  alpha = 1 - chi2.cdf(2 * np.log(BF / np.sqrt(b)), 1)
  print(f"You're working with an alpha of {alpha} that has been adjusted for your sample size of {n}.")
  return alpha


def pi0_tst(p_values, alpha = 0.05):
  """estimate the proportion of true null hypotheses in a set of p-values\n
  | Arguments:
  | :-
  | p_values (array): array of p-values
  | alpha (float): significance threshold for testing; default:0.05\n
  | Returns:
  | :-
  | Returns an estimate of π0, the proportion of true null hypotheses in that dataset
  """
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


def TST_grouped_benjamini_hochberg(identifiers_grouped, p_values_grouped, alpha):
  """perform the two-stage adaptive Benjamini-Hochberg procedure for multiple testing correction\n
  | Arguments:
  | :-
  | identifiers_grouped (dict): dictionary of group : list of glycans
  | p_values_grouped (dict): dictionary of group : list of p-values
  | alpha (float): significance threshold for testing\n
  | Returns:
  | :-
  | Returns dictionaries of glycan : corrected p-value and glycan : significant?
  """
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
      significance_dict[identifier] = corrected_pval < adjusted_alpha
  return adjusted_p_values, significance_dict


def test_inter_vs_intra_group(cohort_b, cohort_a, glycans, grouped_glycans, paired = False):
  """estimates intra- and inter-group correlation of a given grouping of glycans via a mixed-effects model\n
  | Arguments:
  | :-
  | cohort_b (dataframe): dataframe of glycans as rows and samples as columns of the case samples
  | cohort_a (dataframe): dataframe of glycans as rows and samples as columns of the control samples
  | glycans (list): list of glycans in IUPAC-condensed nomenclature
  | grouped_glycans (dict): dictionary of type group : glycans
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False\n
  | Returns:
  | :-
  | Returns floats for the intra-group and inter-group correlation
  """
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


def replace_outliers_with_IQR_bounds(full_row):
  """replaces outlier values with row median\n
  | Arguments:
  | :-
  | full_row (pd.DataFrame row): row from a pandas dataframe, with all but possibly the first value being numerical\n
  | Returns:
  | :-
  | Returns row with replaced outliers
  """
  row = full_row.iloc[1:] if isinstance(full_row.iloc[0], str) else full_row
  # Calculate Q1, Q3, and IQR for each row
  Q1 = row.quantile(0.25)
  Q3 = row.quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5*IQR
  upper_bound = Q3 + 1.5*IQR
  # Define outliers as values outside of Q1 - 1.5*IQR and Q3 + 1.5*IQR
  capped_values = row.apply(lambda x: lower_bound if (x < lower_bound and x != 0) else (upper_bound if (x > upper_bound and x != 0) else x))
  # Replace outliers with row median
  if isinstance(full_row.iloc[0], str):
    full_row.iloc[1:] = capped_values
  else:
    full_row = capped_values
  return full_row


def replace_outliers_winsorization(full_row):
  """Replaces outlier values using Winsorization.\n
  | Arguments:
  | :-
  | full_row (pd.DataFrame row): row from a pandas dataframe, with all but possibly the first value being numerical\n
  | Returns:
  | :-
  | Returns row with outliers replaced by Winsorization.
  """
  row = full_row.iloc[1:] if isinstance(full_row.iloc[0], str) else full_row
  # Apply Winsorization - limits set to match typical IQR outlier detection
  nan_placeholder = row.min() - 1
  row = row.astype(float).fillna(nan_placeholder)
  winsorized_values = winsorize(row, limits = [0.05, 0.05])
  winsorized_values = pd.Series(winsorized_values, index = row.index)
  winsorized_values = winsorized_values.replace(nan_placeholder, np.nan)
  if isinstance(full_row.iloc[0], str):
    full_row.iloc[1:] = winsorized_values
  else:
    full_row = winsorized_values
  return full_row


def hotellings_t2(group1, group2, paired = False):
  """Hotelling's T^2 test (the t-test for multivariate comparisons)\n
  """
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


def sequence_richness(counts):
  return (counts != 0).sum()


def shannon_diversity_index(counts):
  proportions = counts / counts.sum()
  return entropy(proportions)


def simpson_diversity_index(counts):
  proportions = counts / counts.sum()
  return 1 - np.sum(proportions**2)


def get_equivalence_test(row_a, row_b, paired = False):
  """performs an equivalence test (two one-sided t-tests) to test whether differences between group means are considered practically equivalent\n
  | Arguments:
  | :-
  | row_a (array-like): basically a row of the control samples for one glycan/motif
  | row_b (array-like): basically a row of the case samples for one glycan/motif
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False\n
  | Returns:
  | :-
  | Returns a p-value of whether the two group means can be considered equivalent
  """
  pooled_std = np.sqrt(((len(row_a) - 1) * np.var(row_a, ddof = 1) + (len(row_b) - 1) * np.var(row_b, ddof = 1)) / (len(row_a) + len(row_b) - 2))
  delta = 0.2 * pooled_std
  low, up = -delta, delta
  return ttost_paired(row_a, row_b, low, up)[0] if paired else ttost_ind(row_a, row_b, low, up)[0]


def clr_transformation(df, group1, group2, gamma = 0.1, custom_scale = 0):
  """performs the Center Log-Ratio (CLR) Transformation with scale model adjustment\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing features in rows and samples in columns
  | group1 (list): list of column indices or names for the first group of samples, usually the control
  | group2 (list): list of column indices or names for the second group of samples
  | gamma (float): the degree of uncertainty that the CLR assumption holds; default: 0.1
  | custom_scale (float or dict): Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)\n
  | Returns:
  | :-
  | Returns a dataframe that is CLR-transformed with scale model adjustment
  """
  geometric_mean = gmean(df.replace(0, np.nan), axis = 0)
  clr_adjusted = np.zeros_like(df.values)
  if gamma and not isinstance(custom_scale, dict):
    group1i = [df.columns.get_loc(c) for c in group1]
    group2i = [df.columns.get_loc(c) for c in group2] if group2 else group1i
    geometric_mean = -np.log2(geometric_mean)
    if group2:
      clr_adjusted[:, group1i] = np.log2(df[group1]) + (geometric_mean[group1i] if not custom_scale else norm.rvs(loc = np.log2(1), scale = gamma, size = (df.shape[0], len(group1))))
      condition = norm.rvs(loc = geometric_mean[group2i], scale = gamma, size = (df.shape[0], len(group2))) if not custom_scale else norm.rvs(loc = np.log2(custom_scale), scale = gamma, size = (df.shape[0], len(group2)))
      clr_adjusted[:, group2i] = np.log2(df[group2]) + condition
    else:
      clr_adjusted[:, group1i] = np.log2(df[group1]) + norm.rvs(loc = geometric_mean[group1i], scale = gamma, size = (df.shape[0], len(group1)))
  elif not group2 and isinstance(custom_scale, dict):
    gamma = max(gamma, 0.1)
    for idx in range(df.shape[1]):
      group_id = group1[idx] if isinstance(group1[0], int) else group1[idx].split('_')[1]
      scale_factor = custom_scale.get(group_id, 1)
      clr_adjusted[:, idx] = np.log2(df.iloc[:, idx]) + norm.rvs(loc = np.log2(scale_factor), scale = gamma, size = df.shape[0])
  else:
    clr_adjusted = np.log2(df) - np.log2(geometric_mean)
  return pd.DataFrame(clr_adjusted, index = df.index, columns = df.columns)


def anosim(df, group_labels_in, permutations = 999):
  """Performs analysis of similarity statistical test\n
  | Arguments:
  | :-
  | df (dataframe): square distance matrix
  | group_labels_in (list): list of group membership for each sample
  | permutations (int): number of permutations to perform in ANOSIM statistical test; default:999\n
  | Returns:
  | :-
  | (i) ANOSIM R statistic - ranges between -1 to 1.
  | (ii) p-value of the R statistic
  """
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


def alpha_biodiversity_stats(df, group_labels):
  """Performs an ANOVA on the respective alpha diversity distance\n
  | Arguments:
  | :-
  | df (dataframe): square distance matrix
  | group_labels (list): list of group membership for each sample\n
  | Returns:
  | :-
  | F statistic and p-value
  """
  group_counts = Counter(group_labels)
  if all(count > 1 for count in group_counts.values()):
    stat_outputs = pd.DataFrame({'group': group_labels, 'diversity': df.squeeze()})
    grouped_diversity = stat_outputs.groupby('group')['diversity'].apply(list).tolist()
    stats = f_oneway(*grouped_diversity)
    return stats


def calculate_permanova_stat(df, group_labels):
  """Performs multivariate analysis of variance\n
  | Arguments:
  | :-
  | df (dataframe): square distance matrix
  | group_labels (list): list of group membership for each sample\n
  | Returns:
  | :-
  | F statistic - higher means effect more likely.
  """
  unique_groups = np.unique(group_labels)
  n = len(group_labels)
  # Between-group and within-group sums of squares
  ss_total = np.sum(squareform(df)) / 2
  ss_within = 0
  for group in unique_groups:
    group_indices = np.where(group_labels == group)[0]
    group_matrix = df.values[np.ix_(group_indices, group_indices)]
    ss_within += np.sum(squareform(group_matrix)) / 2
  ss_between = ss_total - ss_within
  # Calculate the PERMANOVA test statistic: pseudo-F
  ms_between = ss_between / (len(unique_groups) - 1)
  ms_within = ss_within / (n - len(unique_groups))
  f_stat = ms_between / ms_within
  return f_stat


def permanova_with_permutation(df, group_labels, permutations = 999):
  """Performs permutational multivariate analysis of variance\n
  | Arguments:
  | :-
  | df (dataframe): square distance matrix
  | group_labels (list): list of group membership for each sample
  | permutations (int): number of permutations to perform in PERMANOVA statistical test; default:999\n
  | Returns:
  | :-
  | (i) F statistic - higher means effect more likely.
  | (ii) p-value of the F statistic
  """
  observed_f = calculate_permanova_stat(df, group_labels)
  permuted_fs = np.zeros(permutations)
  for i in range(permutations):
    permuted_labels = np.random.permutation(group_labels)
    permuted_fs[i] = calculate_permanova_stat(df, permuted_labels)
  p_value = np.sum(permuted_fs >= observed_f) / permutations
  return observed_f, p_value


def alr_transformation(df, reference_component_index, group1, group2, gamma = 0.1, custom_scale = 0):
  """Given a reference feature, performs additive log-ratio transformation (ALR) on the data\n
  | Arguments:
  | :-
  | df (dataframe): log2-transformed dataframe with features as rows and samples as columns
  | reference_component_index (int): row index of feature to be used as reference
  | group1 (list): list of column indices or names for the first group of samples, usually the control
  | group2 (list): list of column indices or names for the second group of samples
  | gamma (float): the degree of uncertainty that the CLR assumption holds; default: 0.1
  | custom_scale (float or dict): Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)\n
  | Returns:
  | :-
  | ALR-transformed dataframe
  """
  reference_values = df.iloc[reference_component_index, :]
  alr_transformed = np.zeros_like(df.values)
  group1i = [df.columns.get_loc(c) for c in group1]
  group2i = [df.columns.get_loc(c) for c in group2] if group2 else group1i
  if not isinstance(custom_scale, dict):
    if custom_scale:
      alr_transformed[:, group1i] = df.iloc[:, group1i].subtract(reference_values[group1i] - norm.rvs(loc = np.log2(1), scale = gamma, size = len(group1i)), axis = 1)
    else:
      alr_transformed[:, group1i] = df.iloc[:, group1i].subtract(reference_values[group1i])
    scale_adjustment = np.log2(custom_scale) if custom_scale else 0
    alr_transformed[:, group2i] = df.iloc[:, group2i].subtract(reference_values[group2i] - norm.rvs(loc = scale_adjustment, scale = gamma, size = len(group2i)), axis = 1)
  else:
    gamma = max(gamma, 0.1)
    for idx in range(df.shape[1]):
      group_id = group1[idx] if isinstance(group1[0], int) else group1[idx].split('_')[1]
      scale_factor = custom_scale.get(group_id, 1)
      reference_adjusted = reference_values[idx] - norm.rvs(loc = np.log2(scale_factor), scale = gamma)
      alr_transformed[:, idx] = df.iloc[:, idx] - reference_adjusted
  alr_transformed = pd.DataFrame(alr_transformed, index = df.index, columns = df.columns)
  alr_transformed = alr_transformed.drop(index = reference_values.name)
  return alr_transformed


def get_procrustes_scores(df, group1, group2, paired = False, custom_scale = 0):
  """For each feature, estimates it value as ALR reference component\n
  | Arguments:
  | :-
  | df (dataframe): dataframe with features as rows and samples as columns
  | group1 (list): list of column indices or names for the first group of samples, usually the control
  | group2 (list): list of column indices or names for the second group of samples
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False
  | custom_scale (float or dict): Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)\n
  | Returns:
  | :-
  | List of Procrustes scores (Procrustes correlation * inverse of feature variance)
  """
  if isinstance(group1[0], int):
    group1 = [df.columns.tolist()[k] for k in group1]
    group2 = [df.columns.tolist()[k] for k in group2]
  df = df.iloc[:, 1:]
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


def get_additive_logratio_transformation(df, group1, group2, paired = False, gamma = 0.1, custom_scale = 0):
  """Identifies ALR reference component and transforms data according to ALR\n
  | Arguments:
  | :-
  | df (dataframe): dataframe with features as rows and samples as columns
  | group1 (list): list of column indices or names for the first group of samples, usually the control
  | group2 (list): list of column indices or names for the second group of samples
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False
  | gamma (float): the degree of uncertainty that the CLR assumption holds; default: 0.1
  | custom_scale (float or dict): Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)\n
  | Returns:
  | :-
  | ALR-transformed dataframe
  """
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


def correct_multiple_testing(pvals, alpha):
  """Corrects p-values for multiple testing, by default with the two-stage Benjamini-Hochberg procedure\n
  | Arguments:
  | :-
  | pvals (list): list of raw p-values
  | alpha (float): p-value threshold for statistical significance\n
  | Returns:
  | :-
  | (i) list of corrected p-values
  | (ii) list of True/False of whether corrected p-value reaches statistical significance
  """
  corrpvals = multipletests(pvals, method = 'fdr_tsbh')[1]
  corrpvals = [p if p >= pvals[i] else pvals[i] for i, p in enumerate(corrpvals)]
  significance = [p < alpha for p in corrpvals]
  if sum(significance) > 0.9*len(significance):
    print("Significance inflation detected. The CLR/ALR transformation possibly cannot handle this dataset. Consider running again with a higher gamma value.\
             Proceed with caution; for now switching to Bonferroni correction to be conservative about this.")
    res = multipletests(pvals, method = 'bonferroni')
    corrpvals, alpha = res[1], res[3]
    significance = [p < alpha for p in corrpvals]
  return corrpvals, significance


def omega_squared(row, groups):
  """Calculates Omega squared, as an effect size in an ANOVA setting\n
  | Arguments:
  | :-
  | row (pd.Series or array-like): values for one feature
  | groups (list): list indicating group membership with indices per column\n
  | Returns:
  | :-
  | Returns effect size as omega squared (float)
  """
  long_df = pd.DataFrame({'value': row, 'group': groups})
  model = ols('value ~ C(group)', data=long_df).fit()
  anova_results = anova_lm(model, typ = 2)
  ss_total = sum(model.resid ** 2) + anova_results['sum_sq'].sum()
  omega_squared = (anova_results.at['C(group)', 'sum_sq'] - (anova_results.at['C(group)', 'df'] * model.mse_resid)) / (ss_total + model.mse_resid)
  return omega_squared


def get_glycoform_diff(df_res, alpha = 0.05, level = 'peptide'):
  """Calculates differential expression of glycoforms of either a peptide or a whole protein\n
  | Arguments:
  | :-
  | df_res (pd.DataFrame): result from .motif.analysis.get_differential_expression
  | alpha (float): significance threshold for testing; default:0.05
  | level (string): whether to analyze glycoform differential expression at the level of 'peptide' or 'protein'; default:'peptide'\n
  | Returns:
  | :-
  | Returns a dataframe with:
  | (i) Differentially expressed glycopeptides/glycoproteins
  | (ii) Corrected p-values (Welch's t-test with two-stage Benjamini-Hochberg correction aggregated with Fisher’s Combined Probability Test) for difference in mean
  | (iii) Significance: True/False of whether the corrected p-value lies below the sample size-appropriate significance threshold
  | (iv) Effect size as the average Cohen's d across glycoforms
  """
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


def get_glm(group, glycan_features = ['H', 'N', 'A', 'F', 'G']):
  """given glycoform data from a glycosite, constructs & fits a GLM formula for main+interaction effects of explaining glycoform abundance\n
  | Arguments:
  | :-
  | group (dataframe): longform data of glycoform abundances for a particular glycosite
  | glycan_features (list): list of extracted glycan features to consider as variables; default:['H', 'N', 'A', 'F', 'G']\n
  | Returns:
  | :-
  | Returns fitted GLM (or string with failure message if unsuccessful) and a list of the variables that it contains
  """
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


def process_glm_results(df, alpha, glycan_features):
  """tests for interaction effects of glycan features and the condition on glycoform abundance via a GLM\n
  | Arguments:
  | :-
  | df (dataframe): CLR-transformed glycoproteomics data (processed grouped by site), rows glycoforms, columns samples
  | glycan_features (list): list of extracted glycan features to consider as variables\n
  | Returns:
  | :-
  | (for each condition/interaction feature)
  | (i) Regression coefficient from the GLM (indicating direction of change in the treatment condition)
  | (ii) Corrected p-values (two-tailed t-test with two-stage Benjamini-Hochberg correction) for testing the coefficient against zero
  | (iii) Significance: True/False of whether the corrected p-value lies below the sample size-appropriate significance threshold
  """
  results = df.groupby('Glycosite').apply(get_glm, glycan_features = glycan_features)
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


def partial_corr(x, y, controls, motifs = False):
  """Compute regularized partial correlation of x and y, controlling for multiple other variables in controls\n
  | Arguments:
  | :-
  | x (array-like): typically the values from a column or row
  | y (array-like): typically the values from a column or row
  | controls (array-like): variables that are correlated with x or y
  | motifs (bool): whether to analyze full sequences (False) or motifs (True); default:False\n
  | Returns:
  | :-
  | float: regularized partial correlation coefficient.
  | float: p-value associated with the Spearman correlation of residuals.\n
  """
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
