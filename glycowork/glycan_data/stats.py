import pandas as pd
import numpy as np
import warnings
from functools import lru_cache
from itertools import permutations as iter_permutations
from math import factorial
from collections import Counter
from scipy.stats import rankdata, norm, chi2, t, f, entropy, f_oneway, combine_pvalues, dirichlet, spearmanr, ttest_rel, ttest_ind, gamma as gamma_dist
from scipy.spatial import procrustes
from scipy.special import digamma, polygamma
import scipy.integrate as integrate
rng = np.random.default_rng(42)
np.random.seed(0)


def cohen_d(x: np.ndarray | list[float], # comparison group containing numerical data
            y: np.ndarray | list[float], # comparison group containing numerical data
            paired: bool = False # whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient)
            ) -> tuple[float, float]: # (Cohen's d, variance) where d: 0.2 small; 0.5 medium; 0.8 large effect size
    "calculates effect size between two groups, for one feature or for a whole feature x sample frame"
    X, Y = np.atleast_2d(np.asarray(x, dtype = float)), np.atleast_2d(np.asarray(y, dtype = float))
    d, var_d = np.full(X.shape[0], np.nan), np.full(X.shape[0],
                                                    np.nan)  # a feature without two usable observations per group has no standardized effect; leaving it NaN is what the dense code produced anyway, minus the empty-slice warnings
    if paired:
        assert X.shape == Y.shape, "For paired samples, the size of x and y should be the same"
        diff = X - Y
        n = np.isfinite(diff).sum(axis = 1)
        ok = n > 1
        if ok.any():
            n, mean_diff = n[ok], np.nanmean(diff[ok], axis = 1)
            diff_std = np.nanstd(diff[ok], axis = 1, ddof = 1)
            # A degenerate difference has an unbounded standardized effect and no sampling variance left to report
            degenerate = diff_std == 0
            d[ok] = np.where(degenerate, np.where(mean_diff == 0, 0.0, np.where(mean_diff > 0, np.inf, -np.inf)),
                             mean_diff / np.where(degenerate, 1, diff_std))
            var_d[ok] = np.where(degenerate, 0.0, 1 / n + np.where(degenerate, 0, d[ok]) ** 2 / (2 * n))
    else:
        nx, ny = np.isfinite(X).sum(axis = 1), np.isfinite(Y).sum(axis = 1)
        ok = (nx > 1) & (ny > 1)
        if ok.any():
            nx, ny = nx[ok], ny[ok]
            sx, sy = np.maximum(np.nanstd(X[ok], axis = 1, ddof = 1), 1e-6), np.maximum(
                np.nanstd(Y[ok], axis = 1, ddof = 1), 1e-6)
            d[ok] = (np.nanmean(X[ok], axis = 1) - np.nanmean(Y[ok], axis = 1)) / np.sqrt(
                ((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / (nx + ny - 2))
            var_d[ok] = (nx + ny) / (nx * ny) + d[ok] ** 2 / (2 * (nx + ny))
    return (d, var_d) if np.ndim(x) > 1 else (d[0], var_d[0])


def mahalanobis_distance(x: np.ndarray | pd.DataFrame, # comparison group containing numerical data
                         y: np.ndarray | pd.DataFrame, # comparison group containing numerical data
                         paired: bool = False # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                         ) -> float: # Mahalanobis distance effect size
    "calculates effect size between two groups in a multivariate comparison"
    if paired:
        assert x.shape == y.shape, "For paired samples, the size of x and y should be the same"
        x = np.array(x) - np.array(y)
        y = np.zeros_like(x)
    x, y = np.asarray(x), np.asarray(y)
    pooled_cov_inv = np.linalg.pinv(np.cov(x) if paired else (np.cov(x) + np.cov(y)) / 2)
    diff_means = (np.mean(y, axis = 1) - np.mean(x, axis = 1)).reshape(-1, 1)
    mahalanobis_d = np.sqrt(np.clip(diff_means.T @ pooled_cov_inv @ diff_means, 0, None))
    return mahalanobis_d[0][0]


def mahalanobis_variance(x: np.ndarray | pd.DataFrame, # comparison group containing numerical data
                         y: np.ndarray | pd.DataFrame, # comparison group containing numerical data
                         paired: bool = False,  # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                         random_state: int | np.random.Generator | None = None
                         # optional random state for reproducibility
                         ) -> float:  # variance of Mahalanobis distance
    "Estimates variance of Mahalanobis distance via bootstrapping"
    local_rng = np.random.default_rng(random_state) if random_state is not None else rng
    # Combine gp1 and gp2 into a single matrix
    data = np.concatenate((x.T, y.T), axis = 0)
    # Perform bootstrap resampling
    n_iterations = 1000
    size_x = x.shape[1]
    # Generate all bootstrap indices at once
    boot_indices = local_rng.choice(data.shape[0], size = (n_iterations, data.shape[0]), replace = True)
    bootstrap_samples = np.array([mahalanobis_distance(data[idx[:size_x]].T, data[idx[size_x:]].T, paired = paired)
                                  for idx in boot_indices])
    # Estimate the variance of the Mahalanobis distance
    return np.var(bootstrap_samples)


def variance_stabilization(data: pd.DataFrame, # dataframe with glycans/motifs as indices and samples as columns
                           groups: list[list[str]] | None = None # list containing lists of column names of samples from same group for group-specific normalization; otherwise global
                           ) -> pd.DataFrame: # normalized dataframe in same format as input
    "performs variance stabilization normalization"
    # Apply log1p transformation
    data = np.log1p(data)
    # Scale data to have zero mean and unit variance
    if groups is None:
        data = (data - data.mean(axis = 0)) / data.std(axis = 0, ddof = 1).replace(0, 1)
    else:
        for group in groups:
            group_data = data[group]
            data[group] = (group_data - group_data.mean(axis = 0)) / group_data.std(axis = 0, ddof = 1).replace(0, 1)
    return data


class MissForest:
    def __init__(self, regressor: 'RandomForestRegressor | None' = None, # estimator object for each imputation
                 max_iter: int = 5, # number of iterations for imputation process
                 tol: float = 1e-5, # convergence tolerance
                 circadian: bool = False, # initialize missing values from the same feature's median at the same circadian phase
                 timepoints: int | list | np.ndarray | None = None, # number of timepoints, or explicit time values per column (required if circadian)
                 periods: list[int] | None = None,  # cycle lengths to encode (e.g., [12, 24]) (only relevant if circadian)
                 interval: int = 1,  # time units between experimental timepoints (only relevant if circadian)
                 replicates: int = 1,  # replicates per timepoint (only relevant if circadian)
                 random_state: int | np.random.Generator | None = None  # optional random state for reproducibility
                 ) -> None:
        "A class to perform MissForest imputation adapted from https://github.com/yuenshingyan/MissForest"
        from sklearn.ensemble import RandomForestRegressor
        seed = random_state if random_state is None or isinstance(random_state, (int, np.integer)) else int(np.random.default_rng(random_state).integers(2 ** 32))
        self.regressor = regressor if regressor is not None else RandomForestRegressor(random_state = seed)
        self._auto_n_jobs = regressor is None
        self.max_iter = max_iter
        self.tol = tol
        self.circadian = circadian
        self.timepoints = timepoints
        self.periods = periods if periods is not None else [24]
        self.interval = interval
        self.replicates = replicates

    def fit_transform(self, X: pd.DataFrame # input dataframe with missing values
                      ) -> pd.DataFrame: # imputed dataframe
        "Replace missing values using the MissForest algorithm, blended with left-censored draws wherever missingness is intensity-dependent (MNAR)"
        from sklearn.linear_model import LogisticRegression
        # Step 1: Initialization
        # Keep track of where NaNs are in the original dataset
        X_nan = X.isnull()
        if self._auto_n_jobs:
            self.regressor.n_jobs = -1 if X.shape[0] >= 150 else 1
        # Replace NaNs with row medians (each glycan's median across its observed samples)
        row_medians = X.median(axis = 1)
        # Intensity-dependent missingness: features sitting near the detection limit are missing not at random, which random forest (an MAR method) systematically over-imputes towards the feature median
        logX = np.log2(X.mask(X <= 0))
        feat_mu = logX.median(axis = 1)
        w, fit_idx = pd.Series(1.0, index = X.index), feat_mu.notna()
        y = X_nan.loc[fit_idx].values.ravel().astype(int)
        if fit_idx.sum() > 5 and len(y) and 0 < y.mean() < 1 and feat_mu[fit_idx].std() > 0:
            z = ((feat_mu[fit_idx] - feat_mu[fit_idx].mean()) / feat_mu[fit_idx].std()).values.reshape(-1, 1)
            lr = LogisticRegression().fit(np.repeat(z, X.shape[1], axis = 0), y)
            p = lr.predict_proba(z)[:, 1]
            # Missingness that persists at the high-intensity end is the intensity-independent (MAR) baseline; only the excess over it is attributed to censoring
            p0 = lr.predict_proba(np.array([[np.quantile(z, 0.95)]]))[0, 1]
            w[fit_idx] = np.clip((p - p0) / np.maximum(p, 1e-9), 0, 1)
        else:
            w[fit_idx] = 0.0
        # Per-sample detection limit from robust column statistics, so a heavy left tail of sentinel values cannot drag it down
        med_c = logX.median(axis = 0)
        mad_c = ((logX - med_c).abs().median(axis = 0) * 1.4826).replace(0, np.nan).fillna(logX.std(axis = 0, ddof = 1)).fillna(1.0)
        b = norm.cdf(-1.6)
        mnar = pd.DataFrame(np.nan, index = X.index, columns = X.columns, dtype = float)
        for col in X.columns:
            idx = X_nan.index[X_nan[col].values]
            if not len(idx):
                continue
            # Spread the censored draws across the truncated left tail by feature intensity, so they keep rank order and variance instead of collapsing onto a single constant
            r = rankdata(feat_mu.reindex(idx).fillna(-np.inf).values, method = 'ordinal')
            mnar.loc[idx, col] = norm.ppf((r - 0.5) / len(idx) * b) * mad_c[col] + med_c[col]
        mnar, wv = np.exp2(mnar), w.values[:, None]
        if self.circadian and self.timepoints is None:
            raise ValueError(
                "circadian = True requires timepoints: either the number of timepoints or the explicit time value of each column.")
        if self.circadian:
            time_values = np.array(self.timepoints) if isinstance(self.timepoints, (list, np.ndarray)) \
                else np.repeat(np.arange(self.timepoints) * self.interval, self.replicates)[:X.shape[1]]
            phases = time_values % max(self.periods)
            X_transform = X.copy()
            # Phase-aware fill: use same glycan's median at same circadian phase
            for phase in np.unique(phases):
                phase_cols = X.columns[phases == phase]
                phase_medians = X[phase_cols].median(axis = 1)
                for col in phase_cols:
                    X_transform[col] = X[col].fillna(phase_medians)
            # Fall back to global row median if all same-phase values are also NaN
            X_transform = X_transform.apply(lambda col: col.fillna(row_medians))
        else:
            X_transform = X.apply(lambda col: col.fillna(row_medians))
        # Start the iterations from the MNAR-aware prior, so the forest is not anchored at the far-too-high feature median for censored values
        X_transform = X_transform.mask(X_nan, pd.DataFrame(
            np.exp2(wv * np.log2(mnar.values) + (1 - wv) * np.log2(np.maximum(X_transform.values, 1e-9))),
            index = X.index, columns = X.columns))
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
                        self.regressor.fit(features, observed[column])
                        y_missing_pred = self.regressor.predict(missing.drop(columns = column))
                        # Pull the prediction back towards the left-censored draw in proportion to how much of this feature's missingness is intensity-driven
                        wm, mn = w[missing_idx].values, mnar.loc[missing_idx, column].values
                        y_missing_pred = np.exp2(
                            wm * np.log2(mn) + (1 - wm) * np.log2(np.maximum(y_missing_pred, 1e-9)))
                        # Replace missing values in the current column with predictions
                        total_change += np.sum(np.abs(X_transform.loc[missing_idx, column] - y_missing_pred))
                        X_transform.loc[missing_idx, column] = y_missing_pred
            # Convergence has to be judged against the scale of the values being imputed, or an absolute threshold on a sum of abundances is never met and every pass is always paid for
            if total_change < self.tol * max(np.abs(X_transform.values[X_nan.values]).sum(), 1.0):
                break  # Break out of the loop if converged
        # Avoiding zeros
        X_transform += 1e-6
        return X_transform


def impute_and_normalize(df_in: pd.DataFrame, # dataframe with glycan sequences in first col and abundances in subsequent cols
                         groups: list[list[str]], # nested list of column name lists, one list per group
                         impute: bool = True, # replaces zeroes with predictions from MissForest
                         min_samples: float = 0.1,  # percent of samples that need non-zero values for glycan to be kept
                         protect: pd.DataFrame | None = None,  # boolean frame in the shape/order of the abundance block, marking cells that were never measured and must stay NaN
                         circadian: bool = False,  # inject sin/cos time features into MissForest
                         timepoints: int | list | np.ndarray | None = None, # number of timepoints, or explicit time values per column (only relevant if circadian)
                         periods: list[int] | None = None, # cycle lengths to encode (e.g., [12, 24]) (only relevant if circadian)
                         interval: int = 1, # time units between experimental timepoints (only relevant if circadian)
                         replicates: int = 1,  # replicates per timepoint (only relevant if circadian)
                         random_state: int | np.random.Generator | None = None # optional random state for reproducibility
                         ) -> pd.DataFrame:  # normalized dataframe in same style as input
    "discards rows with too many missings, imputes the rest, and normalizes"
    df = df_in.copy()
    if min_samples:
        min_count = max(np.floor((df.shape[1] - 1) * min_samples), 1)
        mask = (df.iloc[:, 1:] != 0).sum(axis = 1) >= min_count
        df = df[mask].reset_index(drop = True)
        if protect is not None:
            protect = protect[mask.values].reset_index(drop = True)
    colname = df.columns[0]
    glycans = df[colname]
    df = df.iloc[:, 1:]
    df = df.astype(float)
    if protect is not None:
        protect = protect.set_axis(df.index).set_axis(df.columns, axis = 1).astype(bool)
        df = df.mask(
            protect)  # a cell that was never measured is not a zero; NaN keeps it out of floors, imputation targets, column totals, and geometric means
    floor = 1e-7 if len(groups) == 2 else 1e-5
    for group in groups:
        group_data = df[group]
        all_zero_mask = (group_data.fillna(0) == 0).all(axis = 1) & group_data.notna().any(
            axis = 1)  # only a group that was measured as all-zero earns a floor, not one that was never measured
        df.loc[all_zero_mask, group] = df.loc[
                                           all_zero_mask, group] + floor  # observed cells here are exactly 0 so this is the old assignment, but NaN + floor stays NaN
    old_cols = df.columns if isinstance(colname, int) else []
    if len(old_cols):
        df.columns = df.columns.astype(str)
    if impute:
        mf = MissForest(circadian = circadian, timepoints = timepoints, periods = periods,
                        interval = interval, replicates = replicates, random_state = random_state)
        df = df.replace(0, np.nan)
        df = mf.fit_transform(df)
        if protect is not None:
            df = df.mask(
                protect)  # re-blank afterwards; these cells still steer the iterative fit, which is the residual approximation of this approach
    df = (df / df.sum(
        axis = 0)) * 100  # pandas sums skip NaN, so an unmeasured cell no longer inflates its sample's total
    if len(old_cols) > 0:
        df.columns = old_cols
    df.insert(loc = 0, column = colname, value = glycans)
    return df


def variance_based_filtering(df: pd.DataFrame, # dataframe with glycans as index and samples in columns
                             min_feature_variance: float = 0.02 # minimum variance to include a feature
                             ) -> tuple[pd.DataFrame, pd.DataFrame]: # (filtered df with variance > min, discarded df with variance <= min)
    "Variance-based filtering of features"
    keep = df.var(axis = 1) > min_feature_variance  # a NaN variance is False in both directions, so the two frames have to partition on one mask or the feature disappears from the output altogether
    return df.loc[keep], df.loc[~keep]


class JTKTest:
    def __init__(self, timepoints: int, periods: list[int], interval: int = 1, replicates: int = 1):
        self.group_sizes = np.full(timepoints, replicates)
        self.n = self.group_sizes.sum()
        squared_sizes = np.square(self.group_sizes)
        self.max_stat = (self.n**2 - squared_sizes.sum()) * 0.5 if self.n > 0 else 0
        self.interval = interval
        self.periods = periods  # Keep original periods
        self.timepoint_periods = np.array(periods) / interval
        self.variance = (self.n**2 * (2 * self.n + 3) - (squared_sizes * (2 * self.group_sizes + 3)).sum()) / 72
        self.waveforms = self._generate_reference_waveforms(timepoints)

    def _generate_reference_waveforms(self, timepoints: int) -> dict[int, list[np.ndarray]]:
        timerange = np.arange(timepoints) * self.interval
        waveforms = {}
        for period in self.periods:
            theta = 2 * np.pi * timerange / period
            waveforms[period] = [
                self._generate_phase_waveform(theta + 2 * np.pi * j * self.interval / period)
                for j in range(timepoints)
            ]
        return waveforms

    def _generate_phase_waveform(self, theta: np.ndarray) -> np.ndarray:
        cos_r = np.repeat(rankdata(np.cos(theta)), self.group_sizes[0])
        matrix = np.sign(np.subtract.outer(cos_r, cos_r))
        return matrix[np.tril_indices(len(cos_r), k = -1)]

    def test(self, values: np.ndarray) -> tuple[float, int, int, float]:
        signs = np.sign(np.subtract.outer(values, values))[np.tril_indices(len(values), k = -1)]
        best_stats = (1.0, self.periods[0], 0, 0)
        for period in self.periods:
            waveforms_period = self.waveforms[period]
            for phase, waveform_phase in enumerate(waveforms_period):
                if phase > 0 and (period+(phase*self.interval) in self.periods or period-(phase*self.interval) in self.periods):
                    continue
                S = (signs * waveform_phase).sum()
                if S == 0:
                    continue
                jtk = (abs(S) + self.max_stat) / 2
                p_val = 2 * norm.cdf(-(jtk - 0.5), -self.max_stat/2, np.sqrt(self.variance))
                if p_val < best_stats[0]:
                    # The reference waveform is shifted backwards, and |S| makes an antiphase match score identically, so the reported lag has to be un-mirrored and offset by half a period when S is negative
                    best_stats = (p_val, period,
                                  (period - phase * self.interval - (0 if S > 0 else period / 2)) % period,
                                  S / self.max_stat)
        return best_stats


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
    if method not in method_dict:
        raise ValueError(f"'{method}' is not a valid method; choose from 'JAB', 'min', 'robust', or 'balanced'.")
    b = method_dict[method](n)
    BF = np.exp(0.5 * t_statistic ** 2) * np.sqrt(b)
    return BF


def get_alphaN(n: int, # sample size
               BF: float = 3, # Bayes factor you would like to match
               method: str = "robust", # method for choice of 'b': "JAB", "min", "robust", "balanced"
               upper: float = 10, # upper limit for range of realistic effect sizes
               verbose: bool = False # whether to print the adjusted alpha
               ) -> float: # alpha level required to achieve desired evidence
    "Set the alpha level based on sample size via Bayesian-Adaptive Alpha Adjustment"
    method_dict = {"JAB": lambda n: 1/n, "min": lambda n: 2/n, "robust": lambda n: max(2/n, 1/np.sqrt(n))}
    if method == "balanced":
        integrand = lambda x: np.exp(-n * x**2 / 4)
        method_dict["balanced"] = lambda n: max(2/n, min(0.5, integrate.quad(integrand, 0, upper)[0]))
    if method not in method_dict:
        raise ValueError(f"'{method}' is not a valid method; choose from 'JAB', 'min', 'robust', or 'balanced'.")
    b = method_dict[method](n)
    alpha = 1 - chi2.cdf(2 * np.log(BF / np.sqrt(b)), 1)
    if verbose:
        print(f"You're working with an alpha of {alpha} that has been adjusted for your sample size of {n}.")
    return alpha


def pi0_tst(p_values: np.ndarray, # array of p-values
            alpha: float = 0.05 # significance threshold for testing
            ) -> float: # estimate of π0, proportion of true null hypotheses
    "estimate the proportion of true null hypotheses in a set of p-values"
    alpha_prime = alpha / (1 + alpha)
    n = len(p_values)
    if not n:
        return 1.0  # an empty family has nothing to reject, so no signal is the only defensible estimate
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


def TST_grouped_benjamini_hochberg(identifiers_grouped: dict[str, list], # dictionary of group : list of glycans
                                   p_values_grouped: dict[str, list[float]], # dictionary of group : list of p-values
                                   alpha: float # significance threshold for testing
                                   ) -> tuple[dict[str, float], dict[str, bool]]: # (glycan:corrected p-value dict, glycan:significant dict)
    "perform the two-stage adaptive Benjamini-Hochberg procedure for multiple testing correction"
    # Initialize results
    adjusted_p_values = {}
    significance_dict = {}
    for group, group_p_values in p_values_grouped.items():
        group_p_values = np.array(group_p_values, dtype = float)
        if not len(group_p_values):
            continue
        # Estimate π0 for the group within the Two-Stage method
        pi0_estimate = pi0_tst(group_p_values, alpha = alpha)
        # π0 = 1 just means stage 1 found no signal in this family; standard TST then falls back to ordinary within-group BH (adjusted_alpha = alpha below), instead of discarding the whole family, which silently wipes out sparse-signal conditions
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
                                 glycans: list[str], # list of glycans in IUPAC-condensed nomenclature
                                 grouped_glycans: dict[str, list[str]], # dictionary of type group : glycans
                                 paired: bool = False # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                                 ) -> tuple[float, float]: # (intra-group correlation, inter-group correlation)
    "estimates intra- and inter-group correlation of a given grouping of glycans via a two-way variance decomposition"
    # With no features or no samples left there is no variance to decompose, and every mean below would reduce over an empty axis
    if not len(glycans) or not cohort_b.shape[0] or not cohort_b.shape[1]:
        return (0.0, 0.0)
    reverse_lookup = {k: v for v, l in grouped_glycans.items() for k in l}
    if paired:
        Y = np.log2(abs((cohort_b.values + 1e-8) / (cohort_a.values + 1e-8)))
    else:
        Y = np.log2(abs((cohort_b.values + 1e-8) / (np.mean(cohort_a, axis = 1).values[:, np.newaxis] + 1e-8)))
    # Every glycan is observed in every sample, so the glycans-nested-in-families components are the classic ANOVA contrasts, which need no iterative fit and do not leave the family, glycan, and residual terms confounded
    codes = pd.factorize(np.asarray([reverse_lookup[g] for g in glycans], dtype = object))[0]
    G, S = Y.shape
    K = codes.max() + 1 if G else 1
    row_mean, col_mean, grand = Y.mean(axis = 1), Y.mean(axis = 0), Y.mean()
    counts = np.bincount(codes, minlength = K)
    fam_mean = np.array([row_mean[codes == k].mean() for k in range(K)])
    ms_resid = ((Y - row_mean[:, None] - col_mean[None, :] + grand) ** 2).sum() / max((G - 1) * (S - 1), 1)
    ms_within = S * ((row_mean - fam_mean[codes]) ** 2).sum() / max(G - K, 1)
    var_glycans_within_group = max((ms_within - ms_resid) / S, 0.0)
    # A grouping earns its own pi0 estimate when membership explains variance, so the criterion is the between-family component, expected over the nested within-family term rather than over the residual
    ms_family = S * (counts * (fam_mean - grand) ** 2).sum() / max(K - 1, 1)
    n0 = (G - (counts ** 2).sum() / G) / max(K - 1, 1) if G else 1.0
    var_between_families = max((ms_family - ms_within) / max(S * n0, 1e-12), 0.0)
    # Sample and residual variance are identical for every candidate grouping, so the shares that discriminate between them are the glycan-level ones
    total_var = var_between_families + var_glycans_within_group
    return (var_between_families / total_var, var_glycans_within_group / total_var) if total_var else (0.0, 0.0)


def replace_outliers_winsorization(df: pd.DataFrame, # features as rows, all but possibly first column numerical
                                   cap_side: str = 'both' # which side(s) to cap outliers on: 'both', 'lower', or 'upper'
                                   ) -> pd.DataFrame: # dataframe with outliers replaced by Winsorization
    "Replaces outlier values using Winsorization"
    if cap_side not in ('both', 'lower', 'upper'):
        raise ValueError("cap_side must be 'both', 'lower', or 'upper'")
    num = df.select_dtypes('number')
    V = num.to_numpy(float)
    n = V.shape[1]
    nan_mask = np.isnan(V)
    # NaNs are pushed to +inf so they sort to the back, which leaves the k-th slot as the k-th real order statistic; the upper rank still has to be counted from the observed count per row, not from n
    obs = n - nan_mask.sum(axis = 1)
    V = np.where(nan_mask, np.inf, V)
    # Limits set to match typical IQR outlier detection
    kk = np.minimum(np.floor(np.maximum(0.05, 1 / np.maximum(obs, 1)) * obs).astype(int), np.maximum((obs - 3) // 2, 0))
    S = np.sort(V, axis = 1)
    rows = np.arange(V.shape[0])
    lower = S[rows, kk][:, None] if cap_side in ('both', 'lower') else -np.inf
    upper = S[rows, np.maximum(obs - 1 - kk, 0)][:, None] if cap_side in ('both', 'upper') else np.inf
    out = np.clip(V, lower, upper)
    out[nan_mask] = np.nan
    res = df.copy()
    res[num.columns] = out
    return res


def hotellings_t2(group1: np.ndarray, # comparison group containing numerical data
                  group2: np.ndarray, # comparison group containing numerical data
                  paired: bool = False # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                  ) -> tuple[float, float]: # (F statistic, p-value)
    "Hotelling's T^2 test (the t-test for multivariate comparisons)"
    if paired:
        assert group1.shape == group2.shape, "For paired samples, the size of group1 and group2 should be the same"
        group1 = group1 - group2  # one-sample test of mean difference vs zero
        group2 = None
        # Calculate the means and covariances of each group
    n1, p = group1.shape
    mean1 = np.mean(group1, axis = 0)
    cov1 = np.atleast_2d(np.cov(group1,
                                rowvar = False))  # np.cov returns a 0-d array for a single variable, which breaks the ridge below
    if group2 is not None:  # two-sample case
        n2, _ = group2.shape
        diff = mean1 - np.mean(group2, axis = 0)
        cov2 = np.atleast_2d(np.cov(group2, rowvar = False))
        denom = n1 + n2 - 2
        pooled_cov = cov1 if denom < 1 else ((n1 - 1) * cov1 + (n2 - 1) * cov2) / denom
        scale, df2 = (n1 * n2) / (n1 + n2), n1 + n2 - p - 1
    else:  # one-sample case (incl. paired)
        diff = mean1
        denom, pooled_cov = n1 - 1, cov1
        scale, df2 = n1, n1 - p
    pooled_cov += np.eye(p) * 1e-6
    # Calculate the Hotelling's T^2 statistic and convert to F
    T2 = scale * diff @ np.linalg.pinv(pooled_cov) @ diff.T
    F = 0 if denom < 1 or df2 < 1 else T2 * df2 / (denom * p)
    if F == 0:
        return F, 1.0
    p_value = f.sf(F, p, df2)
    return F, p_value


def sequence_richness(counts: np.ndarray # array of counts per feature
                      ) -> int: # number of non-zero features
    "counts number of features with non-zero abundance"
    return (counts != 0).sum()


def shannon_diversity_index(counts: np.ndarray # array of counts
                            ) -> float: # Shannon diversity index value
    "calculates Shannon diversity index"
    total = counts.sum()
    if total == 0:
        warnings.warn("A sample has no non-zero abundances at all; its Shannon diversity is undefined and is reported as 0.")
        return 0.0
    return entropy(counts / total)


def simpson_diversity_index(counts: np.ndarray # array of counts
                            ) -> float: # Simpson diversity index value
    "calculates Simpson diversity index"
    total = counts.sum()
    if total == 0:
        warnings.warn("A sample has no non-zero abundances at all; its Simpson diversity is undefined and is reported as 0.")
        return 0.0
    proportions = counts / total
    return 1 - np.sum(proportions ** 2)


def get_equivalence_test(row_a: np.ndarray, # array of control samples for one glycan/motif
                         row_b: np.ndarray, # array of case samples for one glycan/motif
                         paired: bool = False # whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient)
                         ) -> float: # p-value for equivalence test
    "performs equivalence test (two one-sided t-tests) to test whether differences between group means are considered practically equivalent, for one feature or for a whole feature x sample frame"
    A, B = np.atleast_2d(np.asarray(row_a, dtype = float)), np.atleast_2d(np.asarray(row_b, dtype = float))
    na, nb = np.isfinite(A).sum(axis = 1), np.isfinite(B).sum(
        axis = 1)  # per-feature counts, so a structurally unmeasured sample drops out instead of turning the whole row into NaN
    pooled_std = np.sqrt(
        ((na - 1) * np.nanvar(A, axis = 1, ddof = 1) + (nb - 1) * np.nanvar(B, axis = 1, ddof = 1)) / (na + nb - 2))
    delta = 0.2 * pooled_std
    if paired:
        assert A.shape[1] == B.shape[1], "For paired samples, the size of row_a and row_b should be the same"
        diff = A - B
        nd = np.isfinite(diff).sum(axis = 1)
        mdiff, se, dof = np.nanmean(diff, axis = 1), np.nanstd(diff, axis = 1, ddof = 1) / np.sqrt(nd), nd - 1
    else:
        mdiff, se, dof = np.nanmean(A, axis = 1) - np.nanmean(B, axis = 1), pooled_std * np.sqrt(
            1 / na + 1 / nb), na + nb - 2
    # TOST: the equivalence p-value is the larger of the two one-sided t-tests against the -delta and +delta bounds
    se = np.maximum(se, 1e-300)
    p = np.maximum(t.sf((mdiff + delta) / se, dof), t.cdf((mdiff - delta) / se, dof))
    return p if np.ndim(row_a) > 1 else p[0]


def clr_transformation(df: pd.DataFrame, # dataframe with features as rows and samples as columns
                       group1: list[str | int], # column indices/names for first group of samples, usually control
                       group2: list[str | int], # column indices/names for second group of samples
                       gamma: float = 0.1, # degree of uncertainty that CLR assumption holds
                       custom_scale: float | dict = 0, # ratio total signal group2/group1 for scale model (or group_idx:mean/min dict for multivariate)
                       random_state: int | np.random.Generator | None = None, # optional random state for reproducibility
                       reference: list | None = None # subset of feature rows defining the log-ratio reference; defaults to all rows
                       ) -> pd.DataFrame: # CLR-transformed dataframe
    "performs the Center Log-Ratio (CLR) Transformation with scale model adjustment"
    if df.shape[1] and not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):  # tolerate being handed the frame with its glycan/feature column still in front
        id_col = df.columns[0]
        out = clr_transformation(df.drop(columns = id_col), [c for c in (group1 or []) if c != id_col], [c for c in (group2 or []) if c != id_col], gamma = gamma, custom_scale = custom_scale, random_state = random_state, reference = reference)
        out.insert(0, id_col, df[id_col])
        return out
    local_rng = np.random.default_rng(random_state) if random_state is not None else rng
    ref = (df if reference is None else df.loc[reference]).to_numpy(dtype = float)
    logs = np.log(np.where(ref > 0, ref,
                           np.nan))  # a column whose reference rows are all missing has no geometric mean; computing it by hand keeps that case silent instead of warning, since nansum of an all-NaN column is 0 with a count of 0
    cnt = np.isfinite(logs).sum(axis = 0)
    geometric_mean = np.where((cnt > 0) & ~(ref < 0).any(axis = 0),
                              np.exp(np.nansum(logs, axis = 0) / np.maximum(cnt, 1)),
                              np.nan)  # a negative abundance has no log, so the column stays undefined exactly as gmean left it
    clr_adjusted = np.zeros(df.shape, dtype = float)
    if gamma and not isinstance(custom_scale, dict):
        group1i = [df.columns.get_loc(c) for c in group1]
        group2i = [df.columns.get_loc(c) for c in group2] if group2 else group1i
        geometric_mean = -np.log2(geometric_mean)
        if group2:
            clr_adjusted[:, group1i] = np.log2(df[group1]) + (geometric_mean[group1i] if not custom_scale else norm.rvs(loc = np.log2(1), scale = gamma, random_state = local_rng, size = (df.shape[0], len(group1))))
            condition = norm.rvs(loc = geometric_mean[group2i], scale = gamma, random_state = local_rng, size = (df.shape[0], len(group2))) if not custom_scale else \
                norm.rvs(loc = np.log2(custom_scale), scale = gamma, random_state = local_rng, size = (df.shape[0], len(group2)))
            clr_adjusted[:, group2i] = np.log2(df[group2]) + condition
        else:
            clr_adjusted[:, group1i] = np.log2(df[group1]) + norm.rvs(loc = geometric_mean[group1i], scale = gamma, random_state = local_rng, size = (df.shape[0], len(group1)))
    elif not group2 and isinstance(custom_scale, dict):
        gamma = max(gamma, 0.1)
        for idx, group_id in enumerate(group1):
            scale_factor = custom_scale.get(group_id, 1)
            clr_adjusted[:, idx] = np.log2(df.iloc[:, idx]) + norm.rvs(loc = np.log2(scale_factor), scale = gamma, random_state = local_rng, size = df.shape[0])
    else:
        clr_adjusted = np.log2(df) - np.log2(geometric_mean)
    return pd.DataFrame(clr_adjusted, index = df.index, columns = df.columns)


def anosim(df: pd.DataFrame, # square distance matrix
           group_labels_in: list[str], # list of group membership for each sample
           permutations: int = 999,  # number of permutations to perform in ANOSIM test
           random_state: int | np.random.Generator | None = None  # optional random state for reproducibility
           ) -> tuple[float, float]:  # (ANOSIM R statistic [-1 to 1], p-value)
    "Performs analysis of similarity (ANOSIM) statistical test"
    local_rng = np.random.default_rng(random_state) if random_state is not None else rng
    group_labels = list(group_labels_in)
    n = df.shape[0]
    if len(group_labels) != n:
        raise ValueError(
            f"anosim got {len(group_labels)} group labels for a {n}x{n} distance matrix; exactly one label per sample is required.")
    if len(set(group_labels)) < 2 or max(Counter(group_labels).values()) < 2:
        raise ValueError(
            f"anosim needs at least two groups and at least one group with more than one sample, otherwise within- or between-group distances are empty; got {dict(Counter(group_labels))}.")
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
        local_rng.shuffle(group_labels)
        permuted_group_matrix = np.equal.outer(group_labels, group_labels)
        permuted_within_group_indices = permuted_group_matrix[np.tril_indices(n, k = -1)]
        perm_mean_rank_within = np.mean(ranks[permuted_within_group_indices])
        perm_mean_rank_between = np.mean(ranks[~permuted_within_group_indices])
        permuted_Rs[i] = (perm_mean_rank_between - perm_mean_rank_within) / divisor
    # Calculate the p-value
    p_value = (np.sum(permuted_Rs >= R) + 1) / (permutations + 1)
    return R, p_value


def alpha_biodiversity_stats(df: pd.DataFrame, # square distance matrix
                             group_labels: list[str] # list of group membership for each sample
                             ) -> tuple[float, float] | None: # F statistic and p-value if groups have >1 sample, None otherwise
    "Performs Welch's ANOVA on the respective alpha diversity distance"
    group_counts = Counter(group_labels)
    if all(count > 1 for count in group_counts.values()):
        stat_outputs = pd.DataFrame({'group': group_labels, 'diversity': df.squeeze()})
        grouped_diversity = stat_outputs.groupby('group')['diversity'].apply(list).tolist()
        return f_oneway(*grouped_diversity, equal_var = False)


@lru_cache(maxsize = 16)
def _permutation_labels(codes: tuple, # integer group code per sample
                        permutations: int, # number of random draws requested
                        seed: int | None = None # optional seed, so that the draws can be reproduced
                        ) -> np.ndarray: # label matrix whose first row is the observed labelling
    "Builds the label matrix of a permutation test once per design, enumerated exactly where the design has fewer distinct labellings than requested draws"
    n = len(codes)
    distinct = factorial(n) // int(np.prod([factorial(v) for v in Counter(codes).values()]))
    if n <= 8 and distinct <= permutations + 1:
        # A small design has very few distinct labellings, so enumerating them is both cheaper than sampling and free of Monte Carlo error
        return np.array([list(codes)] + [list(r) for r in sorted(set(iter_permutations(codes)) - {codes})])
    return np.vstack([np.asarray(codes), (np.random.default_rng(seed) if seed is not None else rng).permuted(np.tile(np.asarray(codes), (permutations, 1)), axis = 1)])


def permanova_with_permutation(df: pd.DataFrame, # square distance matrix
                               group_labels: list[str], # list of group membership for each sample
                               permutations: int = 999, # number of permutations for test
                               random_state: int | np.random.Generator | None = None # optional random state for reproducibility
                               ) -> tuple[float, float]: # (F statistic, p-value)
    "Performs permutational multivariate analysis of variance (PERMANOVA)"
    # The label matrix is cached, so the key has to be hashable and a Generator is drawn from once, as MissForest does
    seed = random_state if random_state is None or isinstance(random_state, (int, np.integer)) else int(np.random.default_rng(random_state).integers(2 ** 32))
    D2 = np.square(np.asarray(df, dtype = float))
    codes, ug = pd.factorize(np.asarray(group_labels))
    n = len(codes)
    # The labelling depends only on the design, so it is built once and reused by every feature tested against it; the observed labelling rides along as row 0 so that a draw reproducing it stays a bitwise tie
    P = _permutation_labels(tuple(codes.tolist()), permutations, seed)
    ss_within = np.zeros(len(P))
    for g in range(len(ug)):
        M = (P == g).astype(float)
        ss_within += ((M @ D2) * M).sum(axis = 1) / (2 * M.sum(axis = 1))
    ss_between = D2.sum() / (2 * n) - ss_within
    fs = (ss_between / max(len(ug) - 1, 1e-10)) / (ss_within / max(n - len(ug), 1e-10))
    return fs[0], (np.sum(fs[1:] >= fs[0]) + 1) / len(P)


def alr_transformation(df: pd.DataFrame, # dataframe with features as rows and samples as columns
                       reference_component_index: int, # row index of feature to be used as reference
                       group1: list[str | int], # column indices/names for first group of samples, usually control
                       group2: list[str | int], # column indices/names for second group of samples
                       gamma: float = 0.1, # degree of uncertainty that CLR assumption holds
                       custom_scale: float | dict = 0, # ratio total signal group2/group1 for scale model (or group_idx:mean/min dict for multivariate)
                       random_state: int | np.random.Generator | None = None # optional random state for reproducibility
                       ) -> pd.DataFrame: # ALR-transformed dataframe
    "Given a reference feature, performs additive log-ratio transformation (ALR) on the data"
    local_rng = np.random.default_rng(random_state) if random_state is not None else rng
    reference_values = df.iloc[reference_component_index, :]
    alr_transformed = np.zeros(df.shape, dtype = float)
    group1i = [df.columns.get_loc(c) for c in group1]
    group2i = [df.columns.get_loc(c) for c in group2] if group2 else group1i
    if not isinstance(custom_scale, dict):
        if custom_scale:
            alr_transformed[:, group1i] = df.iloc[:, group1i].subtract(reference_values.iloc[group1i] - norm.rvs(loc = np.log2(1), scale = gamma, random_state = local_rng, size = len(group1i)), axis = 1)
        else:
            alr_transformed[:, group1i] = df.iloc[:, group1i].subtract(reference_values.iloc[group1i])
        scale_adjustment = np.log2(custom_scale) if custom_scale else 0
        alr_transformed[:, group2i] = df.iloc[:, group2i].subtract(reference_values.iloc[group2i] - norm.rvs(loc = scale_adjustment, scale = gamma, random_state = local_rng, size = len(group2i)), axis = 1)
    else:
        gamma = max(gamma, 0.1)
        for idx in range(df.shape[1]):
            group_id = group1[idx] if isinstance(group1[0], int) else group1[idx].split('_')[1]
            scale_factor = custom_scale.get(group_id, 1)
            reference_adjusted = reference_values.iloc[idx] - norm.rvs(loc = np.log2(scale_factor), scale = gamma, random_state = local_rng)
            alr_transformed[:, idx] = df.iloc[:, idx] - reference_adjusted
    alr_transformed = pd.DataFrame(alr_transformed, index = df.index, columns = df.columns)
    alr_transformed = alr_transformed.drop(index = reference_values.name)
    alr_transformed = alr_transformed.reset_index(drop = True)
    return alr_transformed


def get_procrustes_scores(df: pd.DataFrame, # dataframe with features as rows and samples as columns
                          group1: list[str | int], # column indices/names for first group of samples, usually control
                          group2: list[str | int], # column indices/names for second group of samples
                          paired: bool = False, # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                          custom_scale: float | dict = 0, # ratio total signal group2/group1 for scale model (or group_idx:mean/min dict)
                          random_state: int | np.random.Generator | None = None # optional random state for reproducibility
                          ) -> tuple[list[float], list[float], list[float]]: # (Procrustes scores, correlations, variances)
    "For each feature, estimates its value as ALR reference component"
    local_rng = np.random.default_rng(random_state) if random_state is not None else rng
    if isinstance(group1[0], int):
        group1 = [df.columns.tolist()[k] for k in group1]
        group2 = [df.columns.tolist()[k] for k in group2]
    df = df.iloc[:, 1:].astype(float)
    ref_matrix = clr_transformation(df, group1, group2, gamma = 0.01, custom_scale = custom_scale, random_state = local_rng)
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
                                      alr_transformation(df, i, group1, group2, gamma = 0.01, custom_scale = custom_scale, random_state = local_rng))[2] for i in range(df.shape[0])]
    return [a / max(b, 1e-8) for a, b in zip(procrustes_corr, variances)], procrustes_corr, variances


def get_additive_logratio_transformation(df: pd.DataFrame, # dataframe with features as rows and samples as columns
                                         group1: list[str | int], # column indices/names for first group of samples
                                         group2: list[str | int], # column indices/names for second group of samples
                                         paired: bool = False, # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                                         gamma: float = 0.1, # degree of uncertainty that CLR assumption holds
                                         custom_scale: float | dict = 0, # ratio total signal group2/group1 for scale model
                                         random_state: int | np.random.Generator | None = None # optional random state for reproducibility
                                         ) -> pd.DataFrame: # ALR-transformed dataframe
    "Identifies ALR reference component and transforms data according to ALR"
    local_rng = np.random.default_rng(random_state) if random_state is not None else rng
    scores, procrustes_corr, variances = get_procrustes_scores(df, group1, group2, paired = paired, custom_scale = custom_scale, random_state = local_rng)
    ref_component = np.argmax(scores)
    ref_component_string = df.iloc[:, 0].values[ref_component]
    print(f"Reference component for ALR is {ref_component_string}, with Procrustes correlation of {procrustes_corr[ref_component]} and variance of {variances[ref_component]}")
    if procrustes_corr[ref_component] < 0.9 or variances[ref_component] > 0.1:
        print("Metrics of chosen reference component not good enough for ALR; switching to CLR instead.")
        df = df.astype({c: float for c in df.columns[1:]})
        df.iloc[:, 1:] = clr_transformation(df.iloc[:, 1:], group1, group2, gamma = gamma, custom_scale = custom_scale,
                                            random_state = local_rng)
        return df
    glycans = df.iloc[:, 0].values.tolist()
    glycans = glycans[:ref_component] + glycans[ref_component+1:]
    alr = alr_transformation(np.log2(df.iloc[:, 1:]), ref_component, group1, group2, gamma = gamma, custom_scale = custom_scale, random_state = local_rng)
    alr.insert(loc = 0, column = 'glycan', value = glycans)
    return alr


def correct_multiple_testing(pvals: list[float] | np.ndarray, # list of raw p-values
                             alpha: float, # p-value threshold for statistical significance
                             correction_method: str = "two-stage" # "two-stage" or "one-stage" Benjamini-Hochberg
                             ) -> tuple[list[float], list[bool]]: # (corrected p-values, significance True/False)
    "Corrects p-values for multiple testing, by default with the two-stage Benjamini-Hochberg procedure"
    pvals = list(pvals)
    if not pvals:
        return [], []
    corrpvals = bh_adjust(pvals, alpha, two_stage = correction_method == "two-stage")
    corrpvals = [p if p >= pvals[i] else pvals[i] for i, p in enumerate(corrpvals)]
    significance = [bool(p < alpha) for p in corrpvals]
    if len(significance) >= 10 and sum(significance) > 0.9 * len(significance):
        print("Significance inflation detected. The CLR/ALR transformation possibly cannot handle this dataset. Consider running again with a higher gamma value.\
             Proceed with caution; for now switching to Bonferroni correction to be conservative about this.")
        corrpvals = np.minimum(np.asarray(pvals) * len(pvals), 1)
        significance = [bool(p < alpha) for p in corrpvals]
    return corrpvals, significance


def moderated_variance(residual_var: np.ndarray, # per-feature within-group variance
                       df_resid: float | np.ndarray,
                       # residual degrees of freedom of the design; per-feature when missingness makes it vary
                       neighbors: list[list[int]] | None = None
                       # per-feature indices of containment neighbors, for a local prior
                       ) -> tuple[
    np.ndarray, np.ndarray]:  # (posterior variance per feature, posterior degrees of freedom per feature)
    "Empirical-Bayes moderation of feature variances; the prior is the geometric mean over each feature's containment neighborhood, or over all features when no graph is given"
    s2 = np.maximum(np.asarray(residual_var, dtype = float), 1e-12)
    d = np.maximum(np.broadcast_to(np.asarray(df_resid, dtype = float), s2.shape),
                   1e-6)  # a feature measured in fewer samples carries less information and must shrink harder, so the design df is kept per feature
    ls2 = np.log(s2)
    # Smyth's moment estimator for the prior degrees of freedom, so the amount of shrinkage is set by the data rather than chosen
    z = ls2 - digamma(d / 2) + np.log(d / 2)
    v = (np.var(z, ddof = 1) if len(z) > 1 else 0.0) - np.mean(polygamma(1, d / 2))
    if v <= 0 or not np.isfinite(v):
        d0 = float(np.median(d))  # variances look homogeneous, so shrink as hard as the cap allows
    else:
        x = 0.5 / v + 0.5
        for _ in range(50):  # Newton inversion of the trigamma function
            tri = polygamma(1, x)
            dx = tri * (1 - tri / v) / polygamma(2, x)
            x += dx
            if abs(dx / x) < 1e-8:
                break
        # the prior may contribute at most as much information as the typical feature's data, which stops a chance-homogeneous variance set from producing absurdly small p-values in tiny cohorts
        d0 = float(np.clip(2 * x, 0.1, np.median(d)))
    # Motifs that contain one another are measured on overlapping structures and so share measurement noise, which makes them a better variance reference than unrelated motifs
    prior = np.array([np.exp(np.mean(ls2[nb + [i]])) if neighbors and nb else np.exp(np.mean(ls2))
                      for i, nb in enumerate(neighbors if neighbors else [[]] * len(s2))])
    return (d0 * prior + d * s2) / (d0 + d), d + d0


def dag_neighbors(index: list[str], # feature labels in the order they are tested
                   dag # containment DAG, or None for a global prior
                   ) -> list[list[int]] | None: # per-feature positions of parents and children
    "Positions of each feature's parents and children in the containment DAG, for use as a local variance prior"
    if dag is None:
        return None
    pos = {g: i for i, g in enumerate(index)}
    return [sorted({pos[x] for x in list(dag.predecessors(g)) + list(dag.successors(g)) if x in pos}) if g in dag else []
            for g in index]


def meta_analysis(effect_sizes: np.ndarray | list[float], # per-study effect sizes
                  variances: np.ndarray | list[float], # variance of each effect size
                  model: str = 'random', # 'fixed' or 'random' effects
                  leave_one_out: bool = False # also pool with each study dropped in turn
                  ) -> dict: # pooled effect, CI, p-value, tau2/Q/I2, per-study weights, optional leave-one-out
    "Fixed/random-effects pooling (DerSimonian-Laird) with heterogeneity statistics and optional leave-one-out sensitivity"
    if model not in ('fixed', 'random'):
        raise ValueError(f"meta_analysis got model='{model}'; must be 'fixed' or 'random'.")
    eff, var = np.asarray(effect_sizes, dtype = float), np.asarray(variances, dtype = float)
    if len(eff) != len(var):
        raise ValueError(f"meta_analysis got {len(eff)} effect sizes for {len(var)} variances; exactly one variance per effect size is required.")
    var = np.maximum(var, 1e-12)
    w = 1 / var
    fixed = float(np.dot(w, eff) / w.sum())
    # Cochran's Q is defined under fixed-effect weights, which is what the DerSimonian-Laird moment estimator of tau2 assumes
    q, dfree = float(np.dot(w, (eff - fixed) ** 2)), len(eff) - 1
    c = w.sum() - np.sum(w ** 2) / w.sum()
    tau2 = max((q - dfree) / c, 0.0) if dfree > 0 and c > 0 else 0.0
    if model == 'random':
        w = 1 / (var + tau2)
    pooled = float(np.dot(w, eff) / w.sum())
    se = float(np.sqrt(1 / w.sum()))
    z = pooled / se if se > 0 else 0.0
    out = {'effect': pooled, 'se': se, 'ci_low': pooled - 1.96 * se, 'ci_high': pooled + 1.96 * se, 'z': float(z),
           'p_val': float(2 * norm.sf(abs(z))), 'tau2': tau2, 'Q': q, 'Q_df': dfree,
           'Q_p_val': float(chi2.sf(q, dfree)) if dfree > 0 else np.nan,
           'I2': float(max(0.0, (q - dfree) / q) * 100) if q > 0 and dfree > 0 else 0.0,
           'weights': (w / w.sum()).tolist(), 'model': model, 'k': len(eff)}
    if leave_one_out:
        out['leave_one_out'] = [meta_analysis(np.delete(eff, i), np.delete(var, i), model = model) for i in
                                range(len(eff))] if len(eff) > 2 else []  # dropping one of two studies leaves a single-study pool, which is the input effect back again and carries no sensitivity information
    return out


def omega_squared(row: pd.Series | np.ndarray | pd.DataFrame, # values for one feature, or a whole feature x sample frame
                  groups: list[str] # list indicating group membership with indices per column
                  ) -> float | pd.Series: # effect size as omega squared, per feature
    "Calculates Omega squared, as an effect size in an ANOVA setting"
    X = np.atleast_2d(np.asarray(row, dtype = float))
    g = np.asarray(groups)
    if len(g) != X.shape[1]:
        raise ValueError(
            f"omega_squared got {len(g)} group labels for {X.shape[1]} samples; exactly one label per sample is required.")
    ug = np.unique(g)
    if X.shape[1] <= len(ug):
        raise ValueError(
            f"omega_squared needs more samples than groups, got {X.shape[1]} samples for {len(ug)} groups; with one sample per group there is no within-group variance and the effect size is undefined.")
    ns = np.stack([np.isfinite(X[:, g == u]).sum(axis = 1) for u in ug],
                  axis = 1)  # per-feature group sizes, so an unmeasured sample drops out instead of turning the whole row into NaN
    group_means = np.stack([np.nanmean(X[:, g == u], 1) for u in ug], 1)
    n_tot = ns.sum(axis = 1)
    grand_mean = np.nansum(X, axis = 1) / n_tot
    ss_between = (((group_means - grand_mean[:, None]) ** 2) * ns).sum(1)
    ss_total = np.nansum((X - grand_mean[:, None]) ** 2, axis = 1)
    mse_resid = (ss_total - ss_between) / (n_tot - len(ug))
    out = (ss_between - (len(ug) - 1) * mse_resid) / (ss_total + mse_resid)
    return pd.Series(out, index = row.index) if isinstance(row, pd.DataFrame) else out[0]


def get_glycoform_diff(df_res: pd.DataFrame, # result from .motif.analysis.get_differential_expression
                       alpha: float = 0.05, # significance threshold for testing
                       level: str = 'peptide' # analyze at 'peptide' or 'protein' level
                       ) -> pd.DataFrame: # df with differential expression results, p-vals (Fisher’s Combined Probability Test), significance, effect sizes (Cohen's d)
    "Calculates differential expression of glycoforms from either a peptide or a whole protein"
    label_col = 'Glycosite' if 'Glycosite' in df_res.columns else 'Glycan'
    labels = [k.split('_')[0] for k in df_res[label_col]] if level == 'protein' else ['_'.join(k.split('_')[:-1]) for k
                                                                                      in df_res[label_col]]
    grouped = df_res['p-val'].groupby(labels).agg(lambda p: combine_pvalues(p)[1])  # Fisher’s Combined Probability Test
    mean_effect_size = df_res['Effect size'].groupby(labels).mean()
    pvals, sig = correct_multiple_testing(grouped, alpha)
    df_out = type(df_res)(
        {'Glycosite': grouped.index, 'corr p-val': pvals, 'significant': sig, 'Effect size': mean_effect_size.values})
    for attr in ('_contrasts', '_paired', '_glyco_name', '_provenance'):
        if hasattr(df_res, attr):
            object.__setattr__(df_out, attr, getattr(df_res, attr))
    df_out.attrs.update({**df_res.attrs, 'alpha': alpha, 'test': "Fisher's combined probability test", 'level': level})
    return df_out.sort_values(by = 'corr p-val')


def get_glm(group: pd.DataFrame, # longform data of glycoform abundances for a glycosite
            glycan_features: list[str] = ['H', 'N', 'A', 'F', 'G'] # extracted glycan features to consider as variables
            ) -> tuple[str | str, list[str]]: # (fitted GLM or failure message, list of variables)
    "given glycoform data from a glycosite, constructs & fits a GLM formula for main+interaction effects"
    retained_vars = [c for c in glycan_features if c in group.columns and max(group[c]) > 0]
    if not retained_vars:
        return ("No variables retained", [])
    terms = ['Condition']
    for col in retained_vars:  # Main and interaction effects
        terms += [col, f'{col}_Condition']
    group = group.copy()  # the interaction columns below are scratch for the fit and must not widen the caller's frame
    for col in retained_vars:
        group[f'{col}_Condition'] = group[col] * group['Condition']
    try:
        X = np.column_stack([np.ones(len(group))] + [group[c].values.astype(float) for c in terms])
        y = group['Abundance'].values.astype(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond = None)
        dof = X.shape[0] - np.linalg.matrix_rank(X)
        if dof < 1:
            return ("GLM fitting failed: insufficient residual degrees of freedom", [])
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (((y - X @ beta) ** 2).sum() / dof))
        names = ['Intercept'] + terms
        return (pd.Series(beta, index = names), pd.Series(2 * norm.sf(np.abs(beta / se)), index = names)), retained_vars
    except Exception as e:
        return (f"GLM fitting failed: {str(e)}", [])


def process_glm_results(df: pd.DataFrame, # CLR-transformed glycoproteomics data, rows glycoforms, columns samples
                        alpha: float, # significance threshold
                        glycan_features: list[str] # extracted glycan features to consider as variables
                        ) -> pd.DataFrame: # regression coefficients, p-values, and significance for each condition/interaction
    "tests for interaction effects of glycan features and the condition on glycoform abundance via a GLM"
    results = df.groupby('Glycosite', group_keys = False)[df.columns].apply(lambda x: get_glm(x.reset_index(drop = True), glycan_features = glycan_features))
    all_retained_vars = set()
    for _, retained_vars in results:
        all_retained_vars.update(retained_vars)
    int_terms = ['Condition'] + [f'{v}_Condition' for v in sorted(all_retained_vars)]
    out = {idx: [v[1].get(term, 1.0) for term in int_terms] if not isinstance(v, str) else [1.0] * len(int_terms) for
           idx, (v, _) in results.items()}
    out2 = {idx: [v[0].get(term, 0.0) for term in int_terms] if not isinstance(v, str) else [0.0] * len(int_terms) for
            idx, (v, _) in results.items()}
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
                 ) -> tuple[float, float]: # (regularized partial correlation coefficient, p-value from Spearman correlation of residuals)
    "Compute regularized partial correlation of x and y, controlling for multiple other variables in controls"
    from sklearn.linear_model import Ridge
    # Check if we have any controls
    if controls.size == 0 or controls.shape[1] == 0:
        return spearmanr(x, y)
    # Fit regression models
    alpha = 0.1 if motifs else 0.25
    beta_x = Ridge(alpha = alpha).fit(controls, x).coef_
    beta_y = Ridge(alpha = alpha).fit(controls, y).coef_
    # Compute residuals
    res_x = x - controls @ beta_x
    res_y = y - controls @ beta_y
    # Compute correlation of residuals
    return spearmanr(res_x, res_y)


def estimate_technical_variance(df: pd.DataFrame, # dataframe with abundances in cols
                                group1: list[str | int], # column indices/names for first group of samples
                                group2: list[str | int], # column indices/names for second group of samples
                                num_instances: int = 128, # number of Monte Carlo instances to sample
                                gamma: float = 0.1, # uncertainty parameter for CLR transformation scale
                                custom_scale: float | dict = 0,  # ratio total signal group2/group1 for scale model
                                random_state: int | np.random.Generator | None = None
                                # optional random state for reproducibility
                                ) -> pd.DataFrame:  # transformed df (features, samples*num_instances) with CLR-transformed Monte Carlo instances
    "Monte Carlo sampling from Dirichlet distribution with relative abundances as concentration, followed by CLR transformation"
    local_rng = np.random.default_rng(random_state) if random_state is not None else rng
    df = df.apply(lambda col: (col / col.sum()) * 5000, axis = 0)
    features, samples = df.shape
    transformed_data = np.zeros((features, samples, num_instances))
    for j in range(samples):
        dirichlet_samples = dirichlet.rvs(alpha = np.maximum(df.iloc[:, j].to_numpy(float), 1e-6), random_state = local_rng, size = num_instances).T
        if isinstance(custom_scale, dict) or custom_scale:
            for n in range(num_instances):
                sample_instance = pd.DataFrame(dirichlet_samples[:, n])
                transformed_data[:, j, n] = clr_transformation(sample_instance, sample_instance.columns.tolist(), [],
                                                               gamma = gamma, custom_scale = custom_scale,
                                                               random_state = local_rng).squeeze()
        else:
            # CLR on a single column is just log2(x) minus the log2 geometric mean, plus the gamma uncertainty term
            log_samples = np.log2(np.where(dirichlet_samples > 0, dirichlet_samples, np.nan))
            log_gmean = np.nanmean(log_samples, axis = 0)
            transformed_data[:, j, :] = log_samples + norm.rvs(loc = -log_gmean, scale = gamma,
                                                               random_state = local_rng,
                                                               size = (features, num_instances))
    columns = [col for col in df.columns for _ in range(num_instances)]
    transformed_data_2d = transformed_data.reshape((features, samples* num_instances))
    transformed_df = pd.DataFrame(transformed_data_2d, columns = columns)
    return transformed_df


def perform_tests_monte_carlo(group_a: pd.DataFrame, # rows as features, columns as sample instances from one condition
                              group_b: pd.DataFrame, # rows as features, columns as sample instances from one condition
                              num_instances: int = 128,  # number of Monte Carlo instances to sample
                              paired: bool = False,  # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                              alpha: float = 0.05  # error rate the within-instance FDR correction is calibrated to
                              ) -> tuple[
    list[float], list[float], list[float]]:  # (uncorrected p-vals, corrected p-vals, effect sizes)
    "Perform tests on each Monte Carlo instance, apply Benjamini-Hochberg correction, calculate effect sizes"
    num_features, _ = group_a.shape
    avg_uncorrected_p_values, avg_corrected_p_values, avg_effect_sizes = np.zeros(num_features), np.zeros(num_features), np.zeros(num_features)
    n_a, n_b = group_a.shape[1] // num_instances, group_b.shape[1] // num_instances
    arr_a = group_a.values.reshape(num_features, n_a, num_instances)
    arr_b = group_b.values.reshape(num_features, n_b, num_instances)
    for instance in range(num_instances):
        sample_a, sample_b = arr_a[:, :, instance], arr_b[:, :, instance]
        instance_p_values = np.nan_to_num((
            ttest_rel(sample_b, sample_a, axis = 1) if paired else ttest_ind(sample_b, sample_a, equal_var = False,
                                                                             axis = 1))[1], nan = 1.0)
        instance_effect_sizes = cohen_d(sample_b, sample_a, paired = paired)[0]
        # Apply Benjamini-Hochberg correction for multiple testing within the instance
        avg_uncorrected_p_values += instance_p_values
        avg_corrected_p_values += bh_adjust(instance_p_values, alpha)
        avg_effect_sizes += instance_effect_sizes
    avg_uncorrected_p_values /= num_instances
    avg_corrected_p_values /= num_instances
    avg_corrected_p_values = [p if p >= avg_uncorrected_p_values[i] else avg_uncorrected_p_values[i] for i, p in enumerate(avg_corrected_p_values)]
    avg_effect_sizes /= num_instances
    return avg_uncorrected_p_values, avg_corrected_p_values, avg_effect_sizes


def hsic(x: np.ndarray, # first variable; 1-D or (n_samples, n_features)
         y: np.ndarray, # second variable; same n_samples as x
         sigma: float | None = None # RBF bandwidth; per-variable median heuristic if None
         ) -> tuple[float, float]: # (HSIC statistic, analytical p-value via gamma approximation)
    "Hilbert-Schmidt Independence Criterion with analytical p-value (Gretton et al. 2005) to measure dependency between variables"
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim == 1: x = x[:, None]
    if y.ndim == 1: y = y[:, None]
    n = x.shape[0]
    sq_x = np.sum((x[:, None] - x[None, :]) ** 2, axis = -1)
    sq_y = np.sum((y[:, None] - y[None, :]) ** 2, axis = -1)
    sx = np.sqrt(np.median(sq_x[sq_x > 0]) + 1e-10) if sigma is None else sigma
    sy = np.sqrt(np.median(sq_y[sq_y > 0]) + 1e-10) if sigma is None else sigma
    H = np.eye(n) - 1.0 / n
    Kc = H @ np.exp(-sq_x / (2 * sx ** 2)) @ H
    Lc = H @ np.exp(-sq_y / (2 * sy ** 2)) @ H
    stat = np.trace(Kc @ Lc) / (n - 1) ** 2
    ev_K = np.linalg.eigvalsh(Kc)
    ev_L = np.linalg.eigvalsh(Lc)
    ev_K, ev_L = ev_K[ev_K > 1e-12] / n, ev_L[ev_L > 1e-12] / n
    # Under H0 the statistic is a weighted sum of chi2(1) terms with weights lambda_i * mu_j, so matching its first two moments gives a gamma tail that is symmetric in x and y
    w = np.outer(ev_K, ev_L).ravel()
    mean_T, var_T = w.sum(), 2 * (w ** 2).sum()
    return stat, (float(
        gamma_dist.sf(stat * (n - 1) ** 2 / n, mean_T ** 2 / var_T, scale = var_T / mean_T)) if var_T > 0 else 1.0)


def _bh(p_sorted, alpha):
    n = len(p_sorted)
    ecdf = np.arange(1, n + 1)/n
    corr = np.minimum.accumulate((p_sorted/ecdf)[::-1])[::-1].clip(max = 1)
    rej = p_sorted <= ecdf*alpha
    if rej.any():
        rej[:np.nonzero(rej)[0].max()] = True
    return rej, corr


def bh_adjust(pvals: list[float] | np.ndarray, # raw p-values
              alpha: float, # error rate the correction is calibrated to; two-stage output is only valid at this alpha
              two_stage: bool = True # add the Benjamini-Krieger-Yekutieli pi0 estimation step
              ) -> np.ndarray: # Benjamini-Hochberg adjusted p-values
    "Benjamini-Hochberg adjusted p-values, optionally with two-stage pi0 estimation"
    p = np.asarray(pvals, dtype = float)
    order = np.argsort(p)
    ps, n = p[order], len(p)
    rej, corr = _bh(ps, alpha)
    if two_stage and 0 < (r1 := int(rej.sum())) < n:
        n0 = float(n - r1)
        _, corr = _bh(ps, alpha * n / n0)
        corr = (corr * (n0 / n)).clip(max = 1)
    out = np.empty_like(corr)
    out[order] = corr
    return out