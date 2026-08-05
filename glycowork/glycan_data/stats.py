import pandas as pd
import numpy as np
import warnings
from collections import Counter
from scipy.stats import rankdata, norm, chi2, t, f, entropy, gmean, f_oneway, combine_pvalues, dirichlet, spearmanr, ttest_rel, ttest_ind
from scipy.spatial import procrustes
from scipy.spatial.distance import squareform
import scipy.integrate as integrate
rng = np.random.default_rng(42)
np.random.seed(0)


def cohen_d(x: np.ndarray | list[float], # comparison group containing numerical data
            y: np.ndarray | list[float], # comparison group containing numerical data
            paired: bool = False # whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient)
            ) -> tuple[float, float]: # (Cohen's d, variance) where d: 0.2 small; 0.5 medium; 0.8 large effect size
    "calculates effect size between two groups"
    if paired:
        assert len(x) == len(y), "For paired samples, the size of x and y should be the same"
        diff = np.asarray(x) - np.asarray(y)
        diff_std = np.std(diff, ddof = 1)
        if diff_std == 0:
            return (np.inf if np.mean(diff) > 0 else -np.inf), 0
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
    pooled_cov_inv = np.linalg.pinv((np.cov(x) + np.cov(y)) / 2)
    diff_means = (np.mean(y, axis = 1) - np.mean(x, axis = 1)).reshape(-1, 1)
    mahalanobis_d = np.sqrt(np.clip(diff_means.T @ pooled_cov_inv @ diff_means, 0, None))
    return mahalanobis_d[0][0]


def mahalanobis_variance(x: np.ndarray | pd.DataFrame, # comparison group containing numerical data
                         y: np.ndarray | pd.DataFrame, # comparison group containing numerical data
                         paired: bool = False # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                         ) -> float: # variance of Mahalanobis distance
    "Estimates variance of Mahalanobis distance via bootstrapping"
    # Combine gp1 and gp2 into a single matrix
    data = np.concatenate((x.T, y.T), axis = 0)
    # Perform bootstrap resampling
    n_iterations = 1000
    size_x = x.shape[1]
    # Generate all bootstrap indices at once
    boot_indices = rng.choice(data.shape[0], size = (n_iterations, data.shape[0]), replace = True)
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
                 circadian: bool = False,  # inject sin/cos time features to exploit periodic structure
                 timepoints: int | list | np.ndarray | None = None,  # number of timepoints, or explicit time values per column (only relevant if circadian)
                 periods: list[int] | None = None,  # cycle lengths to encode (e.g., [12, 24]) (only relevant if circadian)
                 interval: int = 1,  # time units between experimental timepoints (only relevant if circadian)
                 replicates: int = 1  # replicates per timepoint (only relevant if circadian)
                 ) -> None:
        "A class to perform MissForest imputation adapted from https://github.com/yuenshingyan/MissForest"
        from sklearn.ensemble import RandomForestRegressor
        self.regressor = regressor if regressor is not None else RandomForestRegressor(n_jobs = -1)
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
        if self.circadian and self.timepoints is not None:
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
                        self.regressor.fit(observed.drop(columns = column), observed[column])
                        y_missing_pred = self.regressor.predict(missing.drop(columns = column))
                        # Pull the prediction back towards the left-censored draw in proportion to how much of this feature's missingness is intensity-driven
                        wm, mn = w[missing_idx].values, mnar.loc[missing_idx, column].values
                        y_missing_pred = np.exp2(
                            wm * np.log2(mn) + (1 - wm) * np.log2(np.maximum(y_missing_pred, 1e-9)))
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
                         groups: list[list[str]], # nested list of column name lists, one list per group
                         impute: bool = True, # replaces zeroes with predictions from MissForest
                         min_samples: float = 0.1, # percent of samples that need non-zero values for glycan to be kept
                         circadian: bool = False, # inject sin/cos time features into MissForest
                         timepoints: int | list | np.ndarray | None = None, # number of timepoints, or explicit time values per column (only relevant if circadian)
                         periods: list[int] | None = None, # cycle lengths to encode (e.g., [12, 24]) (only relevant if circadian)
                         interval: int = 1, # time units between experimental timepoints (only relevant if circadian)
                         replicates: int = 1 # replicates per timepoint (only relevant if circadian)
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
    floor = 1e-7 if len(groups) == 2 else 1e-5
    for group in groups:
        group_data = df[group]
        all_zero_mask = (group_data == 0).all(axis = 1)
        df.loc[all_zero_mask, group] = floor
    old_cols = df.columns if isinstance(colname, int) else []
    if len(old_cols):
        df.columns = df.columns.astype(str)
    if impute:
        mf = MissForest(circadian = circadian, timepoints = timepoints, periods = periods,
                        interval = interval, replicates = replicates)
        df = df.replace(0, np.nan)
        df = mf.fit_transform(df)
    df = (df / df.sum(axis = 0)) * 100
    if len(old_cols) > 0:
        df.columns = old_cols
    df.insert(loc = 0, column = colname, value = glycans)
    return df


def variance_based_filtering(df: pd.DataFrame, # dataframe with glycans as index and samples in columns
                             min_feature_variance: float = 0.02 # minimum variance to include a feature
                             ) -> tuple[pd.DataFrame, pd.DataFrame]: # (filtered df with variance > min, discarded df with variance <= min)
    "Variance-based filtering of features"
    variances = df.var(axis = 1)
    filtered_df = df.loc[variances > min_feature_variance]
    discarded_df = df.loc[variances <= min_feature_variance]
    return filtered_df, discarded_df


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
                    best_stats = (p_val, period, phase * self.interval, S/self.max_stat)
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
    b = method_dict.get(method, lambda n: 1/n)(n)
    BF = np.exp(0.5 * t_statistic**2) * np.sqrt(b)
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
    b = method_dict.get(method, lambda n: 1/n)(n)
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
        group_p_values = np.array(group_p_values)
        # Estimate π0 for the group within the Two-Stage method
        pi0_estimate = pi0_tst(group_p_values, alpha)
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
    "estimates intra- and inter-group correlation of a given grouping of glycans via a mixed-effects model"
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import statsmodels.formula.api as smf
    reverse_lookup = {k: v for v, l in grouped_glycans.items() for k in l}
    if paired:
        temp = pd.DataFrame(np.log2(abs((cohort_b.values + 1e-8) / (cohort_a.values + 1e-8))))
    else:
        mean_cohort_a = np.mean(cohort_a, axis = 1).values[:, np.newaxis] + 1e-8
        temp = pd.DataFrame(np.log2(abs((cohort_b.values + 1e-8) / mean_cohort_a)))
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
    "caps outlier values at the IQR fences"
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


def replace_outliers_winsorization(df: pd.DataFrame, # features as rows, all but possibly first column numerical
                                   cap_side: str = 'both' # which side(s) to cap outliers on: 'both', 'lower', or 'upper'
                                   ) -> pd.DataFrame: # dataframe with outliers replaced by Winsorization
    "Replaces outlier values using Winsorization"
    if cap_side not in ('both', 'lower', 'upper'):
        raise ValueError("cap_side must be 'both', 'lower', or 'upper'")
    num = df.select_dtypes('number')
    V = num.to_numpy(float)
    n = V.shape[1]
    # Park NaNs below the minimum so they never affect the order statistics, then restore them
    placeholder = np.nanmin(V, axis = 1) - 1
    V = np.where(np.isnan(V), placeholder[:, None], V)
    # Limits set to match typical IQR outlier detection
    k = min(int(np.floor(max(0.05, 1 / n) * n)), max((n - 3) // 2, 0))
    S = np.sort(V, axis = 1)
    lower = S[:, k][:, None] if cap_side in ('both', 'lower') else -np.inf
    upper = S[:, n - 1 - k][:, None] if cap_side in ('both', 'upper') else np.inf
    out = np.clip(V, lower, upper)
    out[out == placeholder[:, None]] = np.nan
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
    from statsmodels.stats.weightstats import ttost_ind, ttost_paired
    na, nb = len(row_a), len(row_b)
    pooled_std = np.sqrt(((na - 1) * np.var(row_a, ddof = 1) + (nb - 1) * np.var(row_b, ddof = 1)) / (na + nb - 2))
    delta = 0.2 * pooled_std
    low, up = -delta, delta
    return ttost_paired(row_a, row_b, low, up)[0] if paired else ttost_ind(row_a, row_b, low, up)[0]


def clr_transformation(df: pd.DataFrame, # dataframe with features as rows and samples as columns
                       group1: list[str | int], # column indices/names for first group of samples, usually control
                       group2: list[str | int], # column indices/names for second group of samples
                       gamma: float = 0.1, # degree of uncertainty that CLR assumption holds
                       custom_scale: float | dict = 0, # ratio total signal group2/group1 for scale model (or group_idx:mean/min dict for multivariate)
                       random_state: int | np.random.Generator | None = None, # optional random state for reproducibility
                       reference: list | None = None # subset of feature rows defining the log-ratio reference; defaults to all rows
                       ) -> pd.DataFrame: # CLR-transformed dataframe
    "performs the Center Log-Ratio (CLR) Transformation with scale model adjustment"
    local_rng = np.random.default_rng(random_state) if random_state is not None else rng
    geometric_mean = gmean((df if reference is None else df.loc[reference]).replace(0, np.nan), axis = 0, nan_policy = 'omit')
    clr_adjusted = np.zeros_like(df.values)
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
           permutations: int = 999 # number of permutations to perform in ANOSIM test
           ) -> tuple[float, float]: # (ANOSIM R statistic [-1 to 1], p-value)
    "Performs analysis of similarity (ANOSIM) statistical test"
    group_labels = list(group_labels_in)
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


def calculate_permanova_stat(df: pd.DataFrame, # square distance matrix
                             group_labels: list[str] # list of group membership for each sample
                             ) -> float: # F statistic - higher means effect more likely
    "Performs multivariate analysis of variance"
    unique_groups = np.unique(group_labels)
    n = len(group_labels)
    # Between-group and within-group sums of squares
    ss_total = np.sum(squareform(df) ** 2) / n
    ss_within = 0
    for group in unique_groups:
        group_mask = np.array(group_labels) == group
        group_matrix = df.values[np.ix_(group_mask, group_mask)]
        ss_within += np.sum(squareform(group_matrix) ** 2) / group_mask.sum()
    ss_between = ss_total - ss_within
    # Calculate the PERMANOVA test statistic: pseudo-F
    ms_between = ss_between / max(len(unique_groups) - 1, 1e-10)
    ms_within = ss_within / max(n - len(unique_groups), 1e-10)
    f_stat = ms_between / ms_within
    return f_stat


def permanova_with_permutation(df: pd.DataFrame, # square distance matrix
                               group_labels: list[str], # list of group membership for each sample
                               permutations: int = 999 # number of permutations for test
                               ) -> tuple[float, float]: # (F statistic, p-value)
    "Performs permutational multivariate analysis of variance (PERMANOVA)"
    observed_f = calculate_permanova_stat(df, group_labels)
    permuted_fs = np.zeros(permutations)
    for i in range(permutations):
        permuted_labels = np.random.permutation(group_labels)
        permuted_fs[i] = calculate_permanova_stat(df, permuted_labels)
    p_value = (np.sum(permuted_fs >= observed_f) + 1) / (permutations + 1)
    return observed_f, p_value


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
    alr_transformed = np.zeros_like(df.values)
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
    return [a * (1/b) for a, b in zip(procrustes_corr, variances)], procrustes_corr, variances


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
        df.iloc[:, 1:] = clr_transformation(df.iloc[:, 1:], group1, group2, gamma = gamma, custom_scale = custom_scale, random_state = local_rng)
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
    from statsmodels.stats.multitest import multipletests
    pvals = list(pvals)
    if not pvals:
        return [], []
    corrpvals = multipletests(pvals, method = 'fdr_tsbh' if correction_method == "two-stage" else 'fdr_bh')[1]
    corrpvals = [p if p >= pvals[i] else pvals[i] for i, p in enumerate(corrpvals)]
    significance = [bool(p < alpha) for p in corrpvals]
    if sum(significance) > 0.9*len(significance):
        print("Significance inflation detected. The CLR/ALR transformation possibly cannot handle this dataset. Consider running again with a higher gamma value.\
             Proceed with caution; for now switching to Bonferroni correction to be conservative about this.")
        corrpvals = multipletests(pvals, method = 'bonferroni')[1]
        significance = [bool(p < alpha) for p in corrpvals]
    return corrpvals, significance


def omega_squared(row: pd.Series | np.ndarray | pd.DataFrame, # values for one feature, or a whole feature x sample frame
                  groups: list[str] # list indicating group membership with indices per column
                  ) -> float | pd.Series: # effect size as omega squared, per feature
    "Calculates Omega squared, as an effect size in an ANOVA setting"
    X = np.atleast_2d(np.asarray(row, dtype = float))
    g = np.asarray(groups)
    ug = np.unique(g)
    ns = np.array([(g == u).sum() for u in ug])
    group_means = np.stack([X[:, g == u].mean(1) for u in ug], 1)
    grand_mean = X.mean(1)
    ss_between = (((group_means - grand_mean[:, None]) ** 2) * ns).sum(1)
    ss_total = ((X - grand_mean[:, None]) ** 2).sum(1)
    mse_resid = (ss_total - ss_between) / (X.shape[1] - len(ug))
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
    df_out = pd.DataFrame({'Glycosite': grouped.index, 'corr p-val': pvals, 'significant': sig, 'Effect size': mean_effect_size.values})
    return df_out.sort_values(by = 'corr p-val')


def get_glm(group: pd.DataFrame, # longform data of glycoform abundances for a glycosite
            glycan_features: list[str] = ['H', 'N', 'A', 'F', 'G'] # extracted glycan features to consider as variables
            ) -> tuple[str | str, list[str]]: # (fitted GLM or failure message, list of variables)
    "given glycoform data from a glycosite, constructs & fits a GLM formula for main+interaction effects"
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
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
                        glycan_features: list[str] # extracted glycan features to consider as variables
                        ) -> pd.DataFrame: # regression coefficients, p-values, and significance for each condition/interaction
    "tests for interaction effects of glycan features and the condition on glycoform abundance via a GLM"
    results = df.groupby('Glycosite', group_keys = False)[df.columns].apply(lambda x: get_glm(x.reset_index(drop = True), glycan_features = glycan_features))
    all_retained_vars = set()
    for _, retained_vars in results:
        all_retained_vars.update(retained_vars)
    int_terms = ['Condition'] + [f'{v}_Condition' for v in sorted(all_retained_vars)]
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
                                custom_scale: float | dict = 0 # ratio total signal group2/group1 for scale model
                                ) -> pd.DataFrame: # transformed df (features, samples*num_instances) with CLR-transformed Monte Carlo instances
    "Monte Carlo sampling from Dirichlet distribution with relative abundances as concentration, followed by CLR transformation"
    df = df.apply(lambda col: (col / col.sum())*5000, axis = 0)
    features, samples = df.shape
    transformed_data = np.zeros((features, samples, num_instances))
    for j in range(samples):
        dirichlet_samples = dirichlet.rvs(alpha = df.iloc[:, j], random_state = rng, size = num_instances).T
        if isinstance(custom_scale, dict) or custom_scale:
            for n in range(num_instances):
                sample_instance = pd.DataFrame(dirichlet_samples[:, n])
                transformed_data[:, j, n] = clr_transformation(sample_instance, sample_instance.columns.tolist(), [],
                                                               gamma = gamma, custom_scale = custom_scale).squeeze()
        else:
            # CLR on a single column is just log2(x) minus the log2 geometric mean, plus the gamma uncertainty term
            log_samples = np.log2(np.where(dirichlet_samples > 0, dirichlet_samples, np.nan))
            log_gmean = np.nanmean(log_samples, axis = 0)
            transformed_data[:, j, :] = log_samples + norm.rvs(loc = -log_gmean, scale = gamma, random_state = rng,
                                                               size = (features, num_instances))
    columns = [col for col in df.columns for _ in range(num_instances)]
    transformed_data_2d = transformed_data.reshape((features, samples* num_instances))
    transformed_df = pd.DataFrame(transformed_data_2d, columns = columns)
    return transformed_df


def perform_tests_monte_carlo(group_a: pd.DataFrame, # rows as features, columns as sample instances from one condition
                              group_b: pd.DataFrame, # rows as features, columns as sample instances from one condition
                              num_instances: int = 128, # number of Monte Carlo instances to sample
                              paired: bool = False # whether samples are paired (e.g. tumor & tumor-adjacent tissue)
                              ) -> tuple[list[float], list[float], list[float]]: # (uncorrected p-vals, corrected p-vals, effect sizes)
    "Perform tests on each Monte Carlo instance, apply Benjamini-Hochberg correction, calculate effect sizes"
    from statsmodels.stats.multitest import multipletests
    num_features, _ = group_a.shape
    avg_uncorrected_p_values, avg_corrected_p_values, avg_effect_sizes = np.zeros(num_features), np.zeros(num_features), np.zeros(num_features)
    n_samples = group_a.shape[1] // num_instances
    arr_a = group_a.values.reshape(num_features, n_samples, num_instances)
    arr_b = group_b.values.reshape(num_features, n_samples, num_instances)
    for instance in range(num_instances):
        sample_a, sample_b = arr_a[:, :, instance], arr_b[:, :, instance]
        instance_p_values = (
            ttest_rel(sample_b, sample_a, axis = 1) if paired else ttest_ind(sample_b, sample_a, equal_var = False,
                                                                             axis = 1))[1]
        var_a, var_b = sample_a.var(axis = 1, ddof = 1), sample_b.var(axis = 1, ddof = 1)
        instance_effect_sizes = (sample_b.mean(1) - sample_a.mean(1)) / np.sqrt(
            ((n_samples - 1) * np.maximum(var_b, 1e-12) + (n_samples - 1) * np.maximum(var_a, 1e-12)) / (
                        2 * n_samples - 2))
        # Apply Benjamini-Hochberg correction for multiple testing within the instance
        avg_uncorrected_p_values += instance_p_values
        avg_corrected_p_values += multipletests(instance_p_values, method = 'fdr_tsbh')[1]
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
    ev_K, ev_L = ev_K[ev_K > 1e-12], ev_L[ev_L > 1e-12]
    theta = np.mean(ev_K) * np.mean(ev_L)
    df = 4 * np.mean(ev_K) ** 2 / np.var(ev_K) if np.var(ev_K) > 0 else 1.0
    return stat, 1 - chi2.cdf(stat * (n - 1) ** 2 / theta, df)