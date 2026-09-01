from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.linewidth': 0.8,
    'xtick.major.size': 4, 'ytick.major.size': 4, 'xtick.major.width': 0.8,
    'ytick.major.width': 0.8, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.prop_cycle': plt.cycler('color',
                                  ['#2D6A9F', '#C84B55', '#3A9268', '#E8863A', '#7B5EA7', '#C4843A', '#4AADA8'])
})
from collections import Counter
from typing import Any
from scipy.stats import ttest_ind, ttest_rel, levene, f, f_oneway, spearmanr, t as t_dist
from scipy.spatial.distance import squareform, pdist

from glycowork.glycan_data import loader
from glycowork.glycan_data.loader import strip_suffixes, download_model, GlycoDataFrame
from glycowork.glycan_data.stats import (cohen_d, mahalanobis_distance, mahalanobis_variance,
                                         impute_and_normalize, variance_based_filtering, JTKTest,
                                         MissForest, get_alphaN, TST_grouped_benjamini_hochberg,
                                         compare_inter_vs_intra_group, replace_outliers_winsorization, hotellings_t2,
                                         sequence_richness, shannon_diversity_index, simpson_diversity_index,
                                         get_equivalence_test, clr_transformation, anosim, permanova_with_permutation,
                                         alpha_biodiversity_stats, get_additive_logratio_transformation,
                                         correct_multiple_testing, meta_analysis, bh_adjust,
                                         omega_squared, moderated_variance, dag_neighbors,
                                         get_glycoform_diff, process_glm_results, partial_corr,
                                         estimate_technical_variance,
                                         perform_tests_monte_carlo)
from glycowork.motif.processing import enforce_class, process_for_glycoshift
from glycowork.motif.annotate import (annotate_dataset, quantify_motifs, create_correlation_network,
                                      group_glycans_core, group_glycans_sia_fuc, group_glycans_N_glycan_type,
                                      load_lectin_lib, get_motif_dag, get_composition_dag, _motif_sequence,
                                      create_lectin_and_motif_mappings, lectin_motif_scoring, deduplicate_motifs)
from glycowork.motif.graph import subgraph_isomorphism, glycan_to_nxGraph


def preprocess_data(
        df: pd.DataFrame | str | Path,  # Input dataframe or filepath (.csv/.xlsx)
        group1: list[str | int] | None = None,
        # Column indices/names for first group; default: from the frame's contrasts
        group2: list[str | int] | None = None,
        # Column indices/names for second group; default: from the frame's contrasts
        experiment: str = "diff",  # Type of experiment: "diff" or "anova"
        motifs: bool = False,  # Analyze motifs instead of sequences
        glycoproteomics: bool = False, # Whether rows are glycoforms, ordered by composition containment instead of substructure containment
        feature_set: list[str] = ['exhaustive', 'known'],
        # Feature sets to use; exhaustive, known, terminal1, terminal2, terminal3, chemical, graph, custom, size_branch
        paired: bool | None = None,  # Whether samples are paired; default: from the frame
        impute: bool = True,  # Replace zeros with Random Forest model
        min_samples: float = 0.1,  # Min percent of non-zero samples required
        transform: str | None = None,  # Transformation type: "CLR" or "ALR"
        gamma: float = 0.1,  # Uncertainty parameter for CLR transform
        custom_scale: float | dict = 0,
        # Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
        custom_motifs: list[str] = [],  # Custom motifs if using 'custom' feature set
        monte_carlo: bool = False,
        # Use Monte Carlo simulation to control for technical variation (will take longer to run)
        random_state: int | np.random.Generator | None = None,  # optional random state for reproducibility
        circadian: bool = False,  # initialize missing values from the same feature's median at the same circadian phase
        circadian_timepoints: int | list | np.ndarray | None = None,  # number of timepoints or explicit time values (only relevant if circadian)
        circadian_periods: list[int] | None = None,  # cycle lengths to encode (only relevant if circadian)
        circadian_interval: int = 1,  # time units between timepoints (only relevant if circadian)
        circadian_replicates: int = 1,  # replicates per timepoint (only relevant if circadian)
        motif_dag: bool = True # Build the containment DAG; only worth its n^2 isomorphism sweep for callers that read it
) -> tuple[pd.DataFrame, pd.DataFrame, list[str | int], list[
    str | int]]:  # (transformed df, untransformed df, group1 labels, group2 labels)
    "Preprocesses glycomics data by handling missing values with Random Forest imputation, applying CLR/ALR transformations to escape compositional bias, and optionally quantifying glycan motifs"
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df) if Path(df).suffix.lower() == ".csv" else pd.read_csv(df, sep = "\t") if Path(
            df).suffix.lower() == ".tsv" else pd.read_excel(df)
    if group1 is None and isinstance(df, GlycoDataFrame) and df._contrasts:
        group1, group2 = list(df.group1), list(df.group2)
    if group2 is None:
        group2 = []
    if not group1:
        raise ValueError(
            "No groups given: pass group1 (and group2 for a two-group comparison) as lists of column names or column indices; groups are only inferred automatically from a GlycoDataFrame that carries contrasts.")
    if paired is None:
        paired = df.paired if isinstance(df, GlycoDataFrame) else False
    if glycoproteomics and gamma == 0.1:
        gamma = 0.25  # a glycosite subcomposition has few parts, so the CLR-is-a-valid-reference assumption is much weaker than for a whole glycome; only raised when the caller left the glycomics default
    prov = (getattr(df, '_glyco_name', ''), getattr(df, '_provenance', {}))
    if not isinstance(group1[0], str) and experiment == "diff":
        columns_list = df.columns.tolist()
        group1 = [columns_list[k] for k in group1]
        group2 = [columns_list[k] for k in group2]
    gcol = GlycoDataFrame(df)._glycan_col or df.columns[0]
    df = df[[gcol] + [c for c in df.columns if c != gcol]].iloc[:, :len(group1) + 1].fillna(
        0) if experiment == "anova" else df.loc[
        :, [gcol] + group1 + group2].fillna(0)
    # Drop rows with all zero, followed by outlier removal and imputation & normalization
    df = df.loc[~(df.iloc[:, 1:] == 0).all(axis = 1)].reset_index(drop = True)
    protect = None
    if glycoproteomics:
        sites = pd.Series(['_'.join(str(k).split('_')[:-1]) for k in df.iloc[:, 0]])
        seen = (df.iloc[:, 1:] > 0).groupby(
            sites.values).max()  # a site counts as measured in a sample if any of its glycoforms was seen there
        if min_samples:
            df = df[sites.map(seen.mean(axis = 1) >= min_samples).values].reset_index(drop = True)
            sites = pd.Series(['_'.join(str(k).split('_')[:-1]) for k in df.iloc[:, 0]])
        protect = pd.DataFrame(~seen.loc[sites.values].to_numpy(), index = df.index, columns = df.columns[
            1:])  # every glycoform of a site that was never identified in a sample is structurally missing, not below detection
    df = replace_outliers_winsorization(df)
    if experiment == "diff":
        df = impute_and_normalize(df, [group1, group2], impute = impute, min_samples = min_samples, protect = protect,
                                  circadian = circadian, timepoints = circadian_timepoints, periods = circadian_periods,
                                  interval = circadian_interval, replicates = circadian_replicates,
                                  random_state = random_state)
    elif experiment == "anova":
        groups_unq = sorted(set(group1))
        df = impute_and_normalize(df, [[df.columns[i + 1] for i, x in enumerate(group1) if x == g] for g in groups_unq],
                                  impute = impute, min_samples = min_samples, protect = protect,
                                  circadian = circadian, timepoints = circadian_timepoints, periods = circadian_periods,
                                  interval = circadian_interval, replicates = circadian_replicates,
                                  random_state = random_state)
    df_org = df.copy(deep = True)
    if transform is None:
        transform = "CLR" if glycoproteomics else "ALR" if (isinstance(df.iloc[0, 0], str) and enforce_class(
            df.iloc[0, 0], "N")) and len(df) > 50 else "CLR"
    if transform not in ("ALR", "CLR", "Nothing"):
        raise ValueError("Only ALR and CLR are valid transforms for now.")
    if motifs and glycoproteomics:
        raise ValueError(
            "motifs and glycoproteomics cannot be combined: the motif path replaces df with run-wide motif abundances and never applies the per-glycosite closure, so glycoproteomics would be silently ignored. For now, quantify motifs per glycosite yourself and pass the result with glycoproteomics = True instead.")
    if motifs or glycoproteomics:
        pass  # both cases overwrite df below with their own transform, so running the run-wide one here only pays for ALR's O(features) Procrustes search and, for glycoproteomics, would close over a simplex that does not exist
    elif transform == "ALR":
        df = get_additive_logratio_transformation(df, df.columns[1:].tolist() if experiment == "anova" else group1,
                                                  group2, paired = paired, gamma = gamma, custom_scale = custom_scale,
                                                  random_state = random_state)
    elif transform == "CLR":
        if monte_carlo:
            df = GlycoDataFrame(pd.concat([df.iloc[:, 0], estimate_technical_variance(df.iloc[:, 1:], group1, group2,
                                                                                      gamma = gamma,
                                                                                      custom_scale = custom_scale,
                                                                                      random_state = random_state)],
                                          axis = 1),
                                contrasts = getattr(df, '_contrasts', {}), paired = paired,
                                name = getattr(df, '_glyco_name', ''))
        else:
            df.iloc[:, 1:] = df.iloc[:, 1:] + 0.0000001
            clr_group1 = (group1 + group2) if paired else group1
            df.iloc[:, 1:] = clr_transformation(df.iloc[:, 1:],
                                                clr_group1 if experiment == "diff" else df.columns[1:].tolist(),
                                                [] if paired else group2, gamma = gamma,
                                                custom_scale = 0 if paired else custom_scale,
                                                random_state = random_state)
    if motifs:
        # Motif extraction and quantification
        df_org = quantify_motifs(df_org, feature_set = feature_set, custom_motifs = custom_motifs)
        # Re-normalization
        df_org = df_org.apply(lambda col: col / col.sum() * 100, axis = 0)
        if motif_dag:
            df_org.attrs['motif_dag'] = get_motif_dag(df_org.index.tolist(), abundances = df_org)
        df = df_org + 0.0000001
        if transform == "CLR":
            df = clr_transformation(df, (
                (group1 + group2) if paired else group1) if experiment == "diff" else df.columns.tolist(),
                                    [] if paired else group2,
                                    gamma = gamma, custom_scale = 0 if paired else custom_scale,
                                    random_state = random_state)
        elif transform == "ALR":
            df = get_additive_logratio_transformation(df.reset_index(), group1 if experiment == "diff" else df.columns.tolist(), group2, paired = paired, gamma = gamma,
                                                      custom_scale = custom_scale, random_state = random_state)
            df = df.set_index(df.columns[0])
    else:
        df = df.set_index(df.columns[0])
        df = df.groupby(df.index).sum() if glycoproteomics else df.groupby(
            df.index).mean()  # duplicate glycoproteomics rows are charge states/repeat identifications of one part, so they amalgamate by summation before closure
        df_org = df_org.set_index(df_org.columns[0])
        df_org = df_org.groupby(df_org.index).sum() if glycoproteomics else df_org.groupby(df_org.index).mean()
        if glycoproteomics:
            # Component-wise composition containment forces the same abundance inequality as substructure containment, so glycoforms admit the same balances and residuals as motifs
            sites = pd.Series(['_'.join(str(k).split('_')[:-1]) for k in df_org.index], index = df_org.index)
            keep = sites.groupby(sites).transform(
                'size') > 1  # a one-part subcomposition carries no log-ratio and would CLR to an all-zero row
            df_org, sites = df_org[keep], sites[keep]
            df_org = df_org.div(df_org.groupby(sites).transform(
                'sum')) * 100  # close within the glycosite: glycoforms compete for one site, glycoforms on different proteins do not
            gsets = [group1, group2] if experiment == "diff" and group2 else (
                [[c for c, x in zip(df_org.columns, group1) if x == g] for g in
                 sorted(set(group1))] if experiment == "anova" else [df_org.columns.tolist()])
            ok = pd.concat([df_org[g].notna().sum(axis = 1) >= 2 for g in gsets], axis = 1).all(axis = 1)
            if paired and group2:
                ok &= (df_org[group1].notna().values & df_org[group2].notna().values).sum(
                    axis = 1) >= 2  # a paired test needs complete pairs, which both groups being observed twice does not guarantee
            df_org, sites = df_org[ok], sites[
                ok]  # a group needs two measured samples before it has a variance; the closure above already ran over the observed cells only, so dropping here changes no site total
            if transform == "Nothing":
                df = df.loc[df_org.index]
            else:
                cols = df_org.columns.tolist() if experiment == "anova" else ((group1 + group2) if paired else group1)
                grp2 = [] if (paired or experiment == "anova") else group2
                parts = []
                for _, g in df_org.groupby(sites):
                    ref = [g.mean(axis = 1).idxmax()] if len(
                        g) < 4 else None  # with 2-3 parts the geometric mean is dominated by the very feature being tested, so we pin the denominator to the site's dominant glycoform instead
                    parts.append(clr_transformation(g + 0.0000001, cols, grp2, gamma = gamma,
                                                    custom_scale = 0 if paired else custom_scale,
                                                    random_state = random_state, reference = ref))
                df = pd.concat(parts).loc[df_org.index]
            if motif_dag:
                df_org.attrs['motif_dag'] = get_composition_dag(df_org.index.tolist(), abundances = df_org)
    df_org.attrs['dataset'], df_org.attrs['provenance'] = prov
    return df, df_org, group1, group2


def get_pvals_motifs(
        df: pd.DataFrame | str,  # Input dataframe or filepath (.csv/.xlsx)
        label_col_name: str = 'target',  # Column name for labels
        zscores: bool = True,  # Whether data are z-scores
        thresh: float = 1.645,  # Threshold to separate positive/negative
        sorting: bool = True,  # Sort p-value dataframe
        feature_set: list[str] = ['exhaustive'],
        # Feature sets to use; exhaustive, known, terminal1, terminal2, terminal3, chemical, graph, custom, size_branch
        multiple_samples: bool = False,  # Multiple samples with glycan columns
        motifs: pd.DataFrame | None = None,  # Modified motif_list
        custom_motifs: list[str] = [],  # Custom motifs if using 'custom' feature set
        grouped_BH: bool = True,  # Two-stage adaptive Benjamini-Hochberg within DAG-grouped motif families
        moderate_variance: bool = True
        # Empirical-Bayes variance moderation, with the containment DAG as the prior neighborhood
) -> GlycoDataFrame:  # DataFrame with p-values, FDR-corrected p-values, significance, Cohen's d effect sizes, and equivalence p-values for glycan motifs
    "Identifies significantly enriched glycan motifs using a moderated t-test with DAG-grouped FDR correction and Cohen's d effect size calculation, comparing samples above/below threshold"
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df) if Path(df).suffix.lower() == ".csv" else pd.read_csv(df, sep = "\t") if Path(
            df).suffix.lower() == ".tsv" else pd.read_excel(df)
    in_name = getattr(df, '_glyco_name', '')
    glycan_col_name = GlycoDataFrame(df)._glycan_col or df.columns[0]
    if not multiple_samples and label_col_name not in df.columns:
        raise ValueError(
            f"No column '{label_col_name}' in the input; set label_col_name to the name of your label column (available: {', '.join(map(str, df.columns))}), or set multiple_samples = True if every column after the glycans is a sample.")
    # Reformat to allow for proper annotation in all samples
    df = df.copy()
    value_cols = [c for c in df.columns if c != glycan_col_name]
    if not zscores:
        df[value_cols] = (df[value_cols] - df[value_cols].mean()) / (df[value_cols].std() + 1e-6)
    if multiple_samples:
        df.columns = [glycan_col_name if c == glycan_col_name else label_col_name for c in df.columns]
    # Annotate glycan motifs in dataset
    df_motif = annotate_dataset(df[glycan_col_name].values.tolist(),
                                motifs = motifs, feature_set = feature_set, condense = True,
                                custom_motifs = custom_motifs)
    # Motifs with identical presence across all glycans are one hypothesis, not several, and would otherwise inflate their own family during correction
    df_motif = deduplicate_motifs(df_motif.T).T
    # Broadcast the dataframe to the correct size given the number of samples
    if multiple_samples:
        df = df.set_index(glycan_col_name)
        df_motif = pd.concat([pd.concat([df.iloc[:, k], df_motif], axis = 1).dropna() for k in range(len(df.columns))],
                             axis = 0)
        cols = df_motif.columns.tolist()[1:] + [df_motif.columns.tolist()[0]]
        df_motif = df_motif[cols]
    else:
        df_motif[label_col_name] = df[label_col_name].values.tolist()
    motif_names = df_motif.columns.tolist()[:-1]
    labels = df_motif.iloc[:, -1].values.astype(float)
    X = df_motif.iloc[:, :-1].values.astype(float)
    # Divide into motifs with expression above threshold & below, weighting each motif count by the binding strength it was observed at
    pos, neg = labels > thresh, labels <= thresh
    B, A = (X[pos] * labels[pos, None]).T, (X[neg] * labels[neg, None]).T
    na, nb = A.shape[1], B.shape[1]
    # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
    alpha = get_alphaN(na + nb)
    dag = get_motif_dag(motif_names, abundances = pd.DataFrame(X.T,
                                                               index = motif_names)) if grouped_BH or moderate_variance else None
    # Test statistical enrichment for motifs in above vs below
    live = (A.var(axis = 1, ddof = 1) > 1e-12) | (B.var(axis = 1, ddof = 1) > 1e-12)
    Al, Bl = A[live], B[live]
    ttests, effect_sizes = np.ones(len(motif_names)), np.zeros(len(motif_names))
    if moderate_variance and len(Al) > 1 and na > 1 and nb > 1:
        # Shrinking each motif's variance toward its containment neighborhood stabilizes the many rare motifs an exhaustive feature set produces
        resid = ((na - 1) * Al.var(axis = 1, ddof = 1) + (nb - 1) * Bl.var(axis = 1, ddof = 1)) / (na + nb - 2)
        s2, dfp = moderated_variance(resid, df_resid = na + nb - 2,
                                     neighbors = dag_neighbors([m for m, k in zip(motif_names, live) if k], dag))
        se = np.maximum(np.sqrt(s2 * (1 / na + 1 / nb)), 1e-300)
        ttests[live] = 2 * t_dist.sf(np.abs((Bl.mean(axis = 1) - Al.mean(axis = 1)) / se), dfp)
    elif len(Al):
        ttests[live] = ttest_ind(Bl, Al, axis = 1, equal_var = False)[1]
    ttests = [1.0 if not np.isfinite(p) else float(p) for p in ttests]
    if len(Al):
        effect_sizes[live] = cohen_d(Bl, Al, paired = False)[0]
    equivalence_pvals = np.full(len(motif_names), np.nan)
    todo = (np.array(ttests) > alpha) & live
    if todo.any() and na > 1 and nb > 1:
        equivalence_pvals[todo] = get_equivalence_test(A[todo], B[todo], paired = False)
        valid = ~np.isnan(equivalence_pvals)
        equivalence_pvals[valid] = correct_multiple_testing(equivalence_pvals[valid], alpha)[0]
    equivalence_pvals[np.isnan(equivalence_pvals)] = 1.0
    # Multiple testing correction
    if grouped_BH:
        grouped_motifs, grouped_pvals = select_grouping(pd.DataFrame(B, index = motif_names),
                                                        pd.DataFrame(A, index = motif_names), motif_names, ttests,
                                                        grouped_BH = True, dag = dag)
        corrpvals, significance_dict = TST_grouped_benjamini_hochberg(grouped_motifs, grouped_pvals, alpha)
        ttests_corr = [max(corrpvals[m], ttests[i]) for i, m in enumerate(motif_names)]
        significance = [significance_dict[m] for m in motif_names]
    else:
        ttests_corr, significance = correct_multiple_testing(ttests, alpha)
    rows = {}
    if dag is not None:
        # A parent's count in a glycan is its children's counts plus whatever sits in a context no child covers, so the decomposition the differential functions report is available here too
        F, pos_i, neg_i = X.T, np.where(pos)[0], np.where(neg)[0]
        idx, eff = {m: i for i, m in enumerate(motif_names)}, dict(zip(motif_names, effect_sizes))
        for p in [m for m in motif_names if m in dag and dag.out_degree(m)]:
            kids = [c for c in dag.successors(p) if c in idx]
            kv = F[[idx[c] for c in kids]]
            resid = F[idx[p]] - kv.sum(axis = 0)
            parts = np.vstack([kv, np.clip(resid, 0, None)])
            parts = parts[(parts > 1e-6).any(
                axis = 1)]  # a part that never occurs is not in the sub-composition and cannot redistribute
            # Children and residual sum to the parent, so subtracting one part gives the additive logratio of a genuine sub-composition
            bal = np.log2(parts + 0.0000001)
            bal = bal[1:] - bal[0] if len(bal) > 1 else bal[:0]
            bal = bal[bal.std(axis = 1, ddof = 1) > 1e-9]
            bal_p = hotellings_t2(bal[:, neg_i].T, bal[:, pos_i].T)[1] if 0 < len(bal) < min(len(pos_i),
                                                                                             len(neg_i)) else np.nan
            explained = ', '.join(f'{c} ({eff[c] - eff[p]:+.2f})' for c in kids)
            if (resid <= 1e-6).all():
                rows[p] = (explained, 1.0, 0.0,
                           bal_p)  # parent occurs only inside its children: no context of its own left to test
                continue
            r = np.log2(np.clip(resid, 0.0000001, None))
            rows[p] = (explained, ttest_ind(r[pos_i], r[neg_i], equal_var = False)[1], cohen_d(r[pos_i], r[neg_i])[0],
                       bal_p)
    # Residuals and balances answer different questions than the marginals, so each is corrected as its own, much smaller family
    cp = dict(zip(rows, correct_multiple_testing([v[1] for v in rows.values()], alpha)[0])) if rows else {}
    bk = [m for m in rows if not np.isnan(rows[m][3])]
    bp = dict(zip(bk, correct_multiple_testing([rows[m][3] for m in bk], alpha)[0])) if bk else {}
    out = GlycoDataFrame(pd.DataFrame({
        'motif': motif_names,
        'pval': ttests,
        'corr_pval': ttests_corr,
        'significant': significance,
        'effect_size': effect_sizes,
        'equivalence_pval': equivalence_pvals,
        'Explained by': [rows[m][0] if m in rows else '' for m in motif_names],
        'Redistribution p-val': [bp.get(m, np.nan) for m in motif_names],
        'Residual p-val': [cp.get(m, np.nan) for m in motif_names],
        'Residual effect size': [rows[m][2] if m in rows else np.nan for m in motif_names]
    }), name = in_name)
    out['significant'] = out['significant'].astype('bool')
    if sorting:
        out['abs_effect_size'] = out['effect_size'].abs()
        out = out.sort_values(by = ['abs_effect_size', 'corr_pval', 'pval'], ascending = [False, True, True])
        out = out.drop('abs_effect_size', axis = 1)
    out.attrs.update({'alpha': alpha, 'n': na + nb,
                      'test': "moderated t-test" if moderate_variance else "Welch's t-test"})
    return out


def get_representative_substructures(
        enrichment_df: pd.DataFrame  # Output from get_pvals_motifs
) -> list[str]:  # Up to 10 minimal glycans containing enriched motifs
    "Constructs minimal glycan structures that represent significantly enriched motifs by optimizing for motif content while minimizing structure size using subgraph isomorphism"
    glycans = sorted(set(loader.df_species.glycan))
    # Only consider motifs that are significantly enriched
    filtered_df = (enrichment_df[enrichment_df.significant] if 'significant' in enrichment_df else
                   enrichment_df[enrichment_df.corr_pval < 0.05]).reset_index(drop = True)
    if filtered_df.empty:
        return []
    log_pvals = -np.log10(filtered_df.pval.values)
    max_log_pval = np.max(log_pvals) or 1
    weights = log_pvals / max_log_pval
    motifs = filtered_df.motif.values.tolist()
    weight_of = dict(zip(motifs, weights))
    # A parent motif is present in every glycan its children are, so scoring both counts one piece of evidence twice
    dag = get_motif_dag(motifs)
    descendants = {m: nx.descendants(dag, m) for m in dag}
    # Rows carry motif labels, not sequences, so they need the same resolution get_motif_dag applies to them above
    gmotifs = {}
    for m in motifs:
        s, sp = _motif_sequence(m)
        if s.startswith('r'):
            gmotifs[m] = (s, [])  # subgraph_isomorphism routes a glyco-regex to the regex engine itself
            continue
        try:
            gmotifs[m] = (glycan_to_nxGraph(s, termini = 'provided', termini_list = sp), sp)
        except Exception:
            continue  # non-structural features (graph/chemical/size_branch) cannot be searched for in a sequence
    motif_scores = []
    for k in glycans:
        ggraph = glycan_to_nxGraph(k, termini = 'calc')
        hit = {m for m, (p, sp) in gmotifs.items() if subgraph_isomorphism(ggraph, p, termini_list = sp)}
        motif_scores.append(sum(weight_of[m] for m in hit if not (descendants.get(m, set()) & hit)))
    # For each glycan, get their motif score, normalized by glycan length
    length_scores = [len(g) for g in glycans]
    combined_scores = np.divide(motif_scores, length_scores)
    df_score = pd.DataFrame({'glycan': glycans, 'motif_score': motif_scores, 'length_score': length_scores,
                             'combined_score': combined_scores})
    df_score = df_score.sort_values(by = 'combined_score', ascending = False)
    # Take the 10 glycans with the highest score
    rep_motifs = df_score.glycan.values.tolist()[:10]
    rep_motifs.sort(key = len)
    # Make sure that the list only contains the minimum number of representative glycans
    return [k for k in rep_motifs if sum(subgraph_isomorphism(j, k) for j in rep_motifs) <= 1]


def get_heatmap(
        df: pd.DataFrame | str | Path,  # Input dataframe or filepath (.csv/.xlsx)
        motifs: bool = False,  # Analyze motifs instead of sequences
        feature_set: list[str] = ['known'],
        # Feature sets to use; exhaustive, known, terminal1, terminal2, terminal3, chemical, graph, custom, size_branch
        transform: str = '',  # Transform data before plotting
        datatype: str = 'response',  # Data type: 'response' for quantitative values or 'presence' for presence/absence
        rarity_filter: float = 0.05,  # Min proportion for non-zero values
        filepath: str | Path = '',  # Path to save plot
        index_col: str = 'glycan',  # Column to use as index
        custom_motifs: list[str] = [],  # Custom motifs if using 'custom' feature set
        return_plot: bool = False,  # Return plot object
        show_all: bool = False,  # Show all tick labels
        **kwargs: Any  # Keyword args passed to seaborn clustermap
) -> tuple[Any, list[
    str], pd.DataFrame] | None:  # None or (plot object, column names, transformed dataframe) if return_plot=True
    "Creates hierarchically clustered heatmap visualization of glycan/motif abundances"
    import seaborn as sns
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df) if Path(df).suffix.lower() == ".csv" else pd.read_csv(df, sep = "\t") if Path(
            df).suffix.lower() == ".tsv" else pd.read_excel(df)
    gcol = index_col if index_col in df.columns else GlycoDataFrame(df)._glycan_col
    if gcol:
        df = df.set_index(gcol)
    elif isinstance(df.iloc[0, 0], str):
        df = df.set_index(df.columns[0])
    if not isinstance(df.index[0], str) or (
            isinstance(df.index[0], str) and ('(' not in df.index[0] or '-' not in df.index[0])):
        df = df.T
    df = df.fillna(0)
    if transform:
        df = df.replace(0, np.nan).dropna(thresh = np.max([np.round(rarity_filter * df.shape[0]), 1]), axis = 1).fillna(
            1e-6)
    if motifs:
        if 'custom' in feature_set and len(feature_set) == 1 and len(custom_motifs) < 2:
            raise ValueError("A heatmap needs to have at least two motifs.")
        if datatype == 'response':
            df = quantify_motifs(df, glycans = df.index.tolist(), feature_set = feature_set,
                                 custom_motifs = custom_motifs)
        elif datatype == 'presence':
            # Count glycan motifs and remove rare motifs from the result
            df_motif = annotate_dataset(df.index.tolist(), feature_set = feature_set, condense = True,
                                        custom_motifs = custom_motifs)
            df_motif = df_motif.replace(0, np.nan).dropna(
                thresh = np.max([np.round(rarity_filter * df_motif.shape[0]), 1]), axis = 1)
            df = df_motif.T.fillna(0) @ df
            df = df.apply(lambda col: col / col.sum()).T
            df = deduplicate_motifs(df.T)
    # Quantify on raw abundances and transform the motif composition, as get_pca/preprocess_data do
    if transform == "CLR":
        df = clr_transformation(df + 1e-7, [], [], gamma = 0)
    elif transform == "ALR":
        df = get_additive_logratio_transformation(df.reset_index(), df.columns.tolist(), [], paired = False, gamma = 0)
        df = df.set_index(df.columns[0])
    df = df.dropna(axis = 1)
    if not (df < 0).any().any():
        df /= df.sum()
        df *= 100
        center = None
    else:
        center = 0
    # Cluster the abundances
    ticklabels = {'yticklabels': True, 'xticklabels': True} if show_all else {}
    combined_kwargs = {**ticklabels, **kwargs}
    g = sns.clustermap(df, center = center, **combined_kwargs)
    if max(len(str(label)) for label in df.index) > 100:
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize = 6)
    plt.xlabel('Samples')
    plt.ylabel('Glycans' if not motifs else 'Motifs')
    plt.tight_layout()
    if filepath and not return_plot:
        plt.savefig(filepath, format = Path(filepath).suffix[1:], dpi = 300, bbox_inches = 'tight')
        plt.close(g.fig)
    elif return_plot:
        return g, df.columns.tolist(), df
    else:
        plt.show()


def plot_embeddings(
        glycans: list[str],  # List of IUPAC-condensed glycan sequences
        emb: dict[str, np.ndarray] | pd.DataFrame | None = None,
        # Glycan embeddings dict/DataFrame; defaults to SweetNet embeddings
        label_list: list[Any] | None = None,  # Labels for coloring points
        shape_feature: str | None = None,  # Monosaccharide/bond for point shapes
        filepath: str | Path = '',  # Path to save plot
        alpha: float = 0.8,  # Point transparency
        palette: str = 'colorblind',  # Color palette for groups
        **kwargs: Any  # Keyword args passed to seaborn scatterplot
) -> None:
    "Visualizes learned glycan embeddings using t-SNE dimensionality reduction with optional group coloring"
    import seaborn as sns
    from sklearn.manifold import TSNE
    idx = [i for i, g in enumerate(glycans) if '{' not in g]
    glycans = [glycans[i] for i in idx]
    if label_list is not None:
        label_list = [label_list[i] for i in idx]
    # Get all glycan embeddings
    if emb is None:
        model_path = download_model("glycan_representations.pkl")
        emb = pickle.load(open(model_path, 'rb'))
    # Get the subset of embeddings corresponding to 'glycans'
    embs = emb.iloc[idx].values if isinstance(emb, pd.DataFrame) else np.vstack([emb[g] for g in glycans])
    # Calculate t-SNE of embeddings
    n_samples = embs.shape[0]
    perplexity = min(30, n_samples - 1)
    embs = TSNE(random_state = 42, perplexity = perplexity,
                init = 'pca', learning_rate = 'auto').fit_transform(embs)
    # Plot the t-SNE
    markers = None
    if shape_feature is not None:
        markers = {shape_feature: "X", "Absent": "o"}
        shape_feature = [shape_feature if shape_feature in g else 'Absent' for g in glycans]
    sns.scatterplot(x = embs[:, 0], y = embs[:, 1], hue = label_list if label_list is not None else None,
                    palette = palette if label_list is not None else None, style = shape_feature,
                    markers = markers, alpha = alpha, **kwargs)
    sns.despine(left = True, bottom = True)
    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    if label_list is not None:
        plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, format = Path(filepath).suffix[1:], dpi = 300, bbox_inches = 'tight')
    plt.show()


def characterize_monosaccharide(
        sugar: str,  # Monosaccharide or linkage to analyze
        df: pd.DataFrame | None = None,  # DataFrame with glycan column 'glycan'; defaults to df_species
        mode: str = 'sugar',  # Analysis mode: 'sugar', 'bond', 'sugarbond'
        rank: str | None = None,  # Column name for group filtering
        focus: str | None = None,  # Row value for group filtering
        modifications: bool = False,  # Consider modified monosaccharides
        filepath: str | Path = '',  # Path to save plot
        thresh: int = 10  # Minimum count threshold for inclusion
) -> None:
    "Analyzes connectivity and modification patterns of specified monosaccharides/linkages in glycan sequences"
    import seaborn as sns
    if mode not in ('sugar', 'bond', 'sugarbond'):
        raise ValueError(f"mode has to be 'sugar', 'bond', or 'sugarbond', got '{mode}'.")
    if modifications and mode == 'bond':
        print("Modifications currently only work in mode == 'sugar' and mode == 'sugarbond'; continuing without them.")
        modifications = False
    if (rank is None) != (focus is None):
        raise ValueError("rank and focus have to be given together: rank is the column to filter on (e.g., 'Kingdom'), focus the value to keep (e.g., 'Animalia').")
    if df is None:
        df = loader.df_species
    glycan_col_name = GlycoDataFrame(df)._glycan_col
    if rank is not None and focus is not None:
        df = df[df[rank] == focus]
    # Get all disaccharides by extracting adjacent pairs from graph structure
    pool_in = []
    for glycan in df[glycan_col_name].tolist():
        graph = glycan_to_nxGraph(glycan)
        node_labels = nx.get_node_attributes(graph, 'string_labels')
        for parent_mono in graph.nodes():
            if parent_mono % 2 == 0:
                for linkage in graph.successors(parent_mono):
                    for child_mono in graph.successors(linkage):
                        pool_in.append(f"{node_labels[child_mono]}*{node_labels[linkage]}*{node_labels[parent_mono]}")
    pool_in_split = [k.split('*') for k in pool_in]
    pool, sugars = [], []
    if mode == 'bond':
        # Get upstream monosaccharides for a specific linkage
        pool = [k[0] for k in pool_in_split if k[1] == sugar]
    elif mode in ['sugar', 'sugarbond']:
        for k in pool_in_split:
            # Get downstream monosaccharides or downstream linkages for a specific monosaccharide
            k0, k2 = k[0], k[2] if len(k) > 2 else None
            if modifications and sugar in k0:
                sugars.append(k0)
                pool.append(k2 if mode == 'sugar' else k[1])
            elif k0 == sugar:
                pool.append(k2 if mode == 'sugar' else k[1])
    # Count objects in pool, filter by rarity, and calculate proportion
    cou = Counter(pool).most_common()
    filtered_items = [(item, count) for item, count in cou if count > thresh]
    cou_k, cou_v = zip(*filtered_items) if filtered_items else ((), ())
    cou_v = [v / len(pool) for v in cou_v]
    # Start plotting
    fig, (a0, a1) = plt.subplots(1, 2, figsize = (8, 4), gridspec_kw = {'width_ratios': [1, 1]})
    if modifications:
        # Get counts and proportions for the input monosaccharide + its modifications
        cou2 = Counter(sugars).most_common()
        filtered_items = [(item, count) for item, count in cou2 if count > thresh]
        cou_k2, cou_v2 = zip(*filtered_items) if filtered_items else ((), ())
        cou_v2 = [v / len(sugars) for v in cou_v2]
        # Map the input monosaccharide + its modifications to colors
        color_list = plt.get_cmap('tab20')
        color_map = {key: color_list(idx / len(cou_k2)) for idx, key in enumerate(cou_k2)}
        palette = [color_map[k] for k in cou_k2]
        # Start linking downstream monosaccharides / linkages to the input monosaccharide + its modifications
        pos, cou_k_set, by_key = 2 if mode == 'sugar' else 1, set(cou_k), {}
        for k in pool_in_split:
            if k[pos] in cou_k_set:
                by_key.setdefault(k[0], []).append(k[pos])
        cou_for_df = []
        for key in cou_k2:
            counts = Counter(by_key.get(key, []))
            cou_v_t = [counts.get(item, 0) for item in cou_k]
            if len(cou_k2) > 1:
                cou_for_df.append(
                    pd.DataFrame({'monosaccharides': cou_k, 'counts': cou_v_t, 'colors': [key] * len(cou_k)}))
            else:
                sns.barplot(x = cou_k, y = cou_v_t, ax = a1, color = "#2D6A9F")
        if len(cou_k2) > 1:
            cou_df = pd.concat(cou_for_df).reset_index(drop = True)
            sns.histplot(data = cou_df, x = 'monosaccharides', hue = 'colors', weights = 'counts',
                         multiple = 'stack', palette = palette, ax = a1, legend = False, shrink = 0.8)
        a1.set_ylabel('Absolute Occurrence')
    else:
        sns.barplot(x = cou_k, y = cou_v, ax = a1, color = "#2D6A9F")
        a1.set_ylabel('Relative Proportion')
    sns.despine(left = True, bottom = True)
    a1.set_xlabel('')
    a1.set_title(f'{sugar} and variants are connected to')
    plt.setp(a0.get_xticklabels(), rotation = 'vertical')
    # Confusingly, this second plot block refers to the *first* plot, depicting the input monosaccharide + its modifications
    if modifications:
        if len(cou_k2) > 1:
            cou_df2 = pd.DataFrame({'monosaccharides': cou_k2, 'counts': cou_v2})
            sns.histplot(data = cou_df2, x = 'monosaccharides', weights = 'counts',
                         hue = 'monosaccharides', shrink = 0.8, legend = False, ax = a0, palette = palette,
                         alpha = 0.75)
        else:
            sns.barplot(x = cou_k2, y = cou_v2, ax = a0, color = "#2D6A9F")
        sns.despine(left = True, bottom = True)
        a0.set_ylabel('Relative Proportion')
        a0.set_xlabel('')
        a0.set_title(f'Observed Modifications of {sugar}')
        plt.setp(a1.get_xticklabels(), rotation = 'vertical')
    fig.suptitle(f'Characterizing {sugar}')
    fig.tight_layout()
    if filepath:
        plt.savefig(filepath, format = Path(filepath).suffix[1:], dpi = 300, bbox_inches = 'tight')
    plt.show()


def get_coverage(
        df: pd.DataFrame | str | Path,  # DataFrame with glycans in rows (col 1), abundances in columns
        filepath: str = ''  # Path to save plot
) -> None:
    "Visualizes glycan detection frequency across samples with intensity-based ordering"
    import seaborn as sns
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df) if Path(df).suffix.lower() == ".csv" else pd.read_csv(df, sep = "\t") if Path(
            df).suffix.lower() == ".tsv" else pd.read_excel(df)
    d = df.iloc[:, 1:]
    # arrange by mean intensity across all samples
    order = d.mean(axis = 1).sort_values().index
    # plot figure
    ax = sns.heatmap(d.loc[order], cmap = sns.color_palette("mako", as_cmap = True),
                     cbar_kws = {'label': 'Relative Intensity', 'shrink': 0.5},
                     cbar = True, mask = d.loc[order] == 0, linewidths = 0, rasterized = True)
    ax.set(xlabel = 'Samples', ylabel = 'Glycan ID', title = '')
    # save figure
    if filepath:
        plt.savefig(filepath, format = Path(filepath).suffix[1:], dpi = 300, bbox_inches = 'tight')
    plt.show()


def get_pca(
        df: pd.DataFrame | str | Path,  # DataFrame with glycans in rows (col 1), abundances in columns
        groups: list[int] | pd.DataFrame | None = None,
        # Group labels (e.g., [1,1,1,2,2,2,3,3,3]) or metadata DataFrame with 'id' column
        motifs: bool = False,  # Analyze motifs instead of sequences
        feature_set: list[str] = ['known', 'exhaustive'],
        # Feature sets to use; exhaustive, known, terminal1, terminal2, terminal3, chemical, graph, custom, size_branch
        pc_x: int = 1,  # Principal component for x-axis
        pc_y: int = 2,  # Principal component for y-axis
        color: str | None = None,  # Column in metadata for color grouping; recommended to be categorical
        shape: str | None = None,  # Column in metadata for shape grouping; recommended to be categorical
        size: str | None = None,  # Column in metadata for point size control; recommended to be scalar
        filepath: str | Path = '',  # Path to save plot
        custom_motifs: list[str] = [],  # Custom motifs if using 'custom' feature set
        transform: str | None = None,  # Transformation type: "CLR" or "ALR"
        rarity_filter: float = 0.05  # Min proportion for non-zero values
) -> None:
    "Performs PCA on glycan/motif abundance data with group-based visualization"
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df) if Path(df).suffix.lower() == ".csv" else pd.read_csv(df, sep = "\t") if Path(
            df).suffix.lower() == ".tsv" else pd.read_excel(df)
    if groups is None and isinstance(df, GlycoDataFrame) and df._contrasts:
        groups = list(df.groups)
    if transform and not motifs:
        df = df.replace(0, np.nan).dropna(thresh = np.max([np.round(rarity_filter * df.shape[0]), 1]), axis = 1).fillna(
            1e-6)
        if transform == "ALR":
            df = get_additive_logratio_transformation(df, df.columns.tolist()[1:], [], paired = False, gamma = 0)
        elif transform == "CLR":
            df.iloc[:, 1:] = clr_transformation(df.iloc[:, 1:], [], [], gamma = 0)
    if motifs:
        # Motif extraction and quantification
        df_motif = (
            df.replace(0, np.nan).dropna(thresh = np.max([np.round(rarity_filter * df.shape[0]), 1]), axis = 1).fillna(
                1e-6) if transform else df)
        raw = quantify_motifs(df_motif, feature_set = feature_set, custom_motifs = custom_motifs)
        if transform == "CLR":
            raw = clr_transformation(raw + 1e-7, raw.columns.tolist(), [], gamma = 0)
        elif transform == "ALR":
            # ALR drops the reference component, so the surviving names must be read off the result; both that path and the CLR fallback keep them in column 0
            alr = get_additive_logratio_transformation(raw.reset_index(), raw.columns.tolist(), [], paired = False,
                                                       gamma = 0)
            raw = alr.select_dtypes(include = 'number').set_axis(alr.iloc[:, 0].values)
        df = raw.reset_index()
    X = np.array(df.iloc[:, 1:len(groups) + 1].T) if isinstance(groups, list) and groups else np.array(df.iloc[:, 1:].T)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_std)
    percent_var = np.round(pca.explained_variance_ratio_ * 100)
    df_pca = pd.DataFrame(X_pca)
    # merge with metadata
    if isinstance(groups, pd.DataFrame):
        df_pca['id'] = groups.id.values.tolist()
        df_pca = df_pca.merge(groups)
    # manual grouping assignment
    if isinstance(groups, list):
        color = groups
    # make plot
    ax = sns.scatterplot(x = pc_x - 1, y = pc_y - 1, data = df_pca, hue = color, style = shape, size = size)
    if color or shape or size:
        plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left', borderaxespad = 0)
    ax.set(xlabel = f'PC{pc_x}: {percent_var[pc_x - 1]}% variance',
           ylabel = f'PC{pc_y}: {percent_var[pc_y - 1]}% variance')
    sns.despine()
    # save to file
    if filepath:
        plt.savefig(filepath, format = Path(filepath).suffix[1:], dpi = 300, bbox_inches = 'tight')
    plt.show()


def select_grouping(
        cohort_b: pd.DataFrame,  # Case samples dataframe
        cohort_a: pd.DataFrame,  # Control samples dataframe
        glycans: list[str],  # List of glycans in IUPAC-condensed nomenclature
        p_values: list[float],  # Associated p-values from statistical tests
        paired: bool = False,  # Whether samples are paired
        grouped_BH: bool = False,  # Use two-stage adaptive Benjamini-Hochberg
        dag: nx.DiGraph | None = None  # Motif containment DAG; groups motifs by rarest ancestral family instead of by glycan class
) -> tuple[dict[str, list[str]], dict[str, list[float]]]:  # (group:glycans dict, group:p-values dict)
    "Evaluates optimal glycan grouping strategies (by core type, Sia/Fuc content, or N-glycan type) based on intraclass correlation coefficient, enabling group-aware multiple testing correction"
    if not grouped_BH:
        return {"group1": glycans}, {"group1": p_values}
    if dag is not None:
        # Family = rarest root ancestor, so that branches with different null proportions each get their own pi0 estimate instead of sharing a global one
        desc = {r: nx.descendants(dag, r) for r in dag if not dag.in_degree(r)}
        breadth = {r: len(d) for r, d in desc.items()}
        # a node's root ancestors are exactly the roots whose descendant set already contains it, so no per-node ancestor walk is needed
        anc_roots = {}
        for r, d in desc.items():
            for g in d:
                anc_roots.setdefault(g, []).append(r)

        def group_by_family(gs, ps):
            gg, gp = {}, {}
            for g, p in zip(gs, ps):
                anc = anc_roots.get(g, []) + ([g] if g in breadth else [])
                grp = min(anc, key = lambda a: (breadth[a], a)) if anc else "rest"
                gg.setdefault(grp, []).append(g)
                gp.setdefault(grp, []).append(p)
            for k in [k for k, v in gg.items() if len(v) < 2 and k != "rest"]:
                gg.setdefault("rest", []).extend(gg.pop(k))
                gp.setdefault("rest", []).extend(gp.pop(k))
            return gg, gp

        grouped_glycans, grouped_p_values = group_by_family(glycans, p_values)
        if all(len(g) > 1 for g in grouped_glycans.values()):
            print("Chosen grouping: by_motif_family")
            return grouped_glycans, grouped_p_values
        return {"group1": glycans}, {"group1": p_values}
    funcs = {"by_Sia/Fuc": group_glycans_sia_fuc}
    if any([g.endswith("GalNAc") for g in glycans]):
        funcs["by_core"] = group_glycans_core
    elif any([g.endswith("GlcNAc(b1-4)GlcNAc") for g in glycans]):
        funcs["by_Ntype"] = group_glycans_N_glycan_type
    out = {}
    for desc, func in funcs.items():
        grouped_glycans, grouped_p_values = func(glycans, p_values)
        if any([len(g) < 2 for g in grouped_glycans.values()]):
            continue
        intra, inter = compare_inter_vs_intra_group(cohort_b, cohort_a, glycans, grouped_glycans, paired = paired)
        out[desc] = ((intra, inter), (grouped_glycans, grouped_p_values))
    if not out:
        return {"group1": glycans}, {"group1": p_values}
    desc = list(out.keys())[np.argmax([v[0][0] - v[0][1] for k, v in out.items()])]
    intra, inter = out[desc][0]
    grouped_glycans, grouped_p_values = out[desc][1]
    if intra > inter:
        print("Chosen grouping: " + desc)
        print("ICC of grouping: " + str(intra))
        print("Inter-group correlation of grouping: " + str(inter))
        return grouped_glycans, grouped_p_values
    return {"group1": glycans}, {"group1": p_values}


def get_differential_expression(
        df: pd.DataFrame | str | Path,
        # DataFrame with glycans in rows (col 1) and abundance values in subsequent columns
        group1: list[str | int] | None = None,
        # Column indices/names for first group; default: from the frame's contrasts
        group2: list[str | int] | None = None,
        # Column indices/names for second group; default: from the frame's contrasts
        motifs: bool = False,  # Analyze motifs instead of sequences
        feature_set: list[str] = ['exhaustive', 'known'],
        # Feature sets to use; exhaustive, known, terminal1, terminal2, terminal3, chemical, graph, custom, size_branch
        paired: bool | None = None,  # Whether samples are paired; default: from the frame
        impute: bool = True,  # Replace zeros with Random Forest model
        sets: bool = False,  # Identify clusters of correlated glycans
        set_thresh: float = 0.9,  # Correlation threshold for clusters
        effect_size_variance: bool = False,  # Calculate effect size variance
        min_samples: float = 0.1,  # Min percent of non-zero samples required
        grouped_BH: bool | None = None,  # Use two-stage adaptive Benjamini-Hochberg; None infers True for motifs (DAG-grouped families) and False for sequences
        custom_motifs: list[str] = [],  # Custom motifs if using 'custom' feature set
        transform: str | None = None,  # Transformation type: "CLR" or "ALR"; None auto-decides
        gamma: float = 0.1,  # Uncertainty parameter for CLR transform
        custom_scale: float | dict = 0,
        # Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
        moderate_variance: bool = True,
        # Empirical-Bayes variance moderation, with the containment DAG as the prior neighborhood
        glycoproteomics: bool = False,  # Whether data is from glycoproteomics
        level: str = 'peptide',  # Analysis level for glycoproteomics
        monte_carlo: bool = False,  # Use Monte Carlo for technical variation
        random_state: int | np.random.Generator | None = None  # optional random state for reproducibility
) -> GlycoDataFrame:  # DataFrame with log2FC, p-values, FDR-corrected p-values, and Cohen's d/Mahalanobis distance effect sizes
    "Performs differential expression analysis using Welch's t-test (or Hotelling's T2 for sets) with multiple testing correction on glycomics abundance data"
    grouped_BH = ((motifs or glycoproteomics) and not sets) if grouped_BH is None else grouped_BH
    if glycoproteomics and monte_carlo:
        raise ValueError(
            "monte_carlo is not available for glycoproteomics: the per-glycosite closure replaces the run-wide CLR that estimate_technical_variance simulates over, so no Monte Carlo frame is ever built.")
    in_contrasts, in_name, in_prov = getattr(df, '_contrasts', {}), getattr(df, '_glyco_name', ''), getattr(df,
                                                                                                            '_provenance',
                                                                                                            {})
    paired = df.paired if paired is None and isinstance(df, GlycoDataFrame) else bool(paired)
    df, df_org, group1, group2 = preprocess_data(df, group1 = group1, group2 = group2, experiment = "diff", motifs = motifs,
                                                 glycoproteomics = glycoproteomics, impute = impute,
                                                 min_samples = min_samples, transform = transform,
                                                 feature_set = feature_set,
                                                 paired = paired, gamma = gamma, custom_scale = custom_scale,
                                                 custom_motifs = custom_motifs,
                                                 monte_carlo = monte_carlo, random_state = random_state)
    # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
    alpha = get_alphaN(len(group1 + group2))
    # Variance-based filtering of features
    if not monte_carlo:
        df, df_prison = variance_based_filtering(df)
        df_org, df_org_prison = df_org.loc[df.index], df_org.loc[df_prison.index]
    else:
        df_prison, df_org_prison = pd.DataFrame(), pd.DataFrame()
    glycans = df.index.tolist()
    variances = [0] * len(glycans)
    mean_abundance = df_org.mean(axis = 1)
    df_a, df_b = df[group1], df[group2]
    if sets:
        # Motif/sequence set enrichment
        clusters = create_correlation_network(df.T, set_thresh)
        glycans, mean_abundance_c, pvals, log2fc, levene_pvals, effect_sizes, variances, equivalence_pvals = [], [], [], [], [], [], [], []
        # Testing differential expression of each set/cluster
        for cluster in clusters:
            if len(cluster) > 1:
                cluster = list(cluster)
                glycans.append(cluster)
                gp1, gp2 = df_a.loc[cluster, :], df_b.loc[cluster, :]
                mean_abundance_c.append(mean_abundance.loc[cluster].mean())
                log2fc.append(((gp2.values - gp1.values).mean(axis = 1)).mean() if paired else (
                            gp2.mean(axis = 1) - gp1.mean(axis = 1)).mean())
                # Hotelling's T^2 test for multivariate comparisons
                pvals.append(hotellings_t2(gp1.T.values, gp2.T.values, paired = paired)[1])
                levene_pvals.append(
                    np.mean([levene(gp1.loc[variable, :], gp2.loc[variable, :])[1] for variable in cluster]))
                # Calculate Mahalanobis distance as measure of effect size for multivariate comparisons
                effect_sizes.append(mahalanobis_distance(gp1, gp2, paired = paired))
                equivalence_pvals.append(np.nan)
                if effect_size_variance:
                    variances.append(mahalanobis_variance(gp1, gp2, paired = paired, random_state = random_state))
        mean_abundance = mean_abundance_c
    else:
        log2fc = np.nanmean(df_b.values - df_a.values, axis = 1) if paired else (
                    df_b.mean(axis = 1) - df_a.mean(axis = 1))
        if paired:
            assert len(group1) == len(group2), "For paired samples, the size of group1 and group2 should be the same"
        if monte_carlo:
            pvals, corrpvals, effect_sizes = perform_tests_monte_carlo(df_a, df_b, paired = paired, alpha = alpha)
            significance = [cp < alpha for cp in corrpvals]
            equivalence_pvals = [1.0] * len(pvals)
            levene_pvals = [1.0] * len(pvals)
        else:
            A, B = df_a.values, df_b.values
            if moderate_variance and len(A) > 1:
                # Shrinking each feature's variance toward its containment neighborhood stabilizes the small samples typical of glycomics, without touching the reported effect sizes
                D = B - A if paired else None
                na, nb = np.isfinite(A).sum(axis = 1), np.isfinite(B).sum(
                    axis = 1)  # per-feature counts, because glycoproteomics leaves structurally unmeasured cells as NaN; on dense input these are constant and everything below reduces to the old expressions
                nd = np.isfinite(D).sum(axis = 1) if paired else None
                resid = (np.nanvar(D, axis = 1, ddof = 1) if paired else
                         ((na - 1) * np.nanvar(A, axis = 1, ddof = 1) + (nb - 1) * np.nanvar(B, axis = 1, ddof = 1)) / (
                                     na + nb - 2))
                dfr = (nd - 1) if paired else (na + nb - 2)
                s2, dfp = moderated_variance(resid, df_resid = dfr,
                                             neighbors = dag_neighbors(glycans, df_org.attrs.get('motif_dag')))
                se = np.sqrt(s2 / nd) if paired else np.sqrt(s2 * (1 / na + 1 / nb))
                delta = np.nanmean(D, axis = 1) if paired else (np.nanmean(B, axis = 1) - np.nanmean(A, axis = 1))
                pvals = list(2 * t_dist.sf(np.abs(delta / se), dfp))
            else:
                pvals = list((ttest_rel(B, A, axis = 1, nan_policy = 'omit') if paired else ttest_ind(B, A, axis = 1,
                                                                                                      equal_var = False,
                                                                                                      nan_policy = 'omit'))[
                                 1])
            equivalence_pvals = np.full(len(A), np.nan)
            todo = np.array(pvals) > alpha
            if todo.any():
                equivalence_pvals[todo] = get_equivalence_test(A[todo], B[todo], paired = paired)
            valid_equivalence_pvals = equivalence_pvals[~np.isnan(equivalence_pvals)]
            corrected_equivalence_pvals = bh_adjust(valid_equivalence_pvals, alpha) if len(
                valid_equivalence_pvals) else []
            equivalence_pvals[~np.isnan(equivalence_pvals)] = corrected_equivalence_pvals
            equivalence_pvals[np.isnan(equivalence_pvals)] = 1.0
            # Levene with the default median center is a one-way ANOVA on absolute deviations from the group medians, which reduces along the sample axis in one call
            U, V = np.abs(B - np.nanmedian(B, axis = 1, keepdims = True)), np.abs(
                A - np.nanmedian(A, axis = 1, keepdims = True))
            hiU, loU = np.where(np.isfinite(U), U, -np.inf).max(axis = 1), np.where(np.isfinite(U), U, np.inf).min(
                axis = 1)
            hiV, loV = np.where(np.isfinite(V), V, -np.inf).max(axis = 1), np.where(np.isfinite(V), V, np.inf).min(
                axis = 1)
            tol = 1e-9 * np.maximum(np.maximum(np.where(np.isfinite(B), np.abs(B), 0.0).max(axis = 1),
                                               np.where(np.isfinite(A), np.abs(A), 0.0).max(axis = 1)), 1.0)
            vary = ~(((hiU - loU) <= tol) & ((hiV - loV) <= tol)) & (np.isfinite(U).sum(axis = 1) > 1) & (
                        np.isfinite(V).sum(
                            axis = 1) > 1)  # deviations that are flat, or differ only by floating-point noise, make F infinite and p exactly 0: a spurious variance difference, and what scipy warns about
            levene_pvals = np.ones(len(df_a))
            if vary.any() and df_a.shape[1] > 2 and df_b.shape[1] > 2:
                levene_pvals[vary] = f_oneway(U[vary], V[vary], axis = 1, nan_policy = 'omit')[1]
            effect_sizes, variances = cohen_d(B, A, paired = paired) if len(A) else ([0] * len(glycans),
                                                                                     [0] * len(glycans))
    # Multiple testing correction
    if not monte_carlo and pvals:
        if grouped_BH:
            grouped_glycans, grouped_pvals = select_grouping(df_b, df_a, glycans, pvals, paired = paired,
                                                             grouped_BH = grouped_BH, dag = df_org.attrs.get('motif_dag') if motifs or glycoproteomics else None)
            corrpvals, significance_dict = TST_grouped_benjamini_hochberg(grouped_glycans, grouped_pvals, alpha)
            corrpvals = [corrpvals[g] for g in glycans]
            corrpvals = [p if p >= pvals[i] else pvals[i] for i, p in enumerate(corrpvals)]
            significance = [significance_dict[g] for g in glycans]
        else:
            corrpvals, significance = correct_multiple_testing(pvals, alpha)
        levene_pvals = bh_adjust(levene_pvals, alpha)
    elif monte_carlo:
        pass
    else:
        corrpvals, significance = [1] * len(glycans), [False] * len(glycans)
    df_out = GlycoDataFrame(pd.DataFrame(list(
        zip(glycans, mean_abundance, log2fc, pvals, corrpvals, significance, levene_pvals, effect_sizes,
            equivalence_pvals)),
        columns = ['Glycan', 'Mean abundance', 'Log2FC', 'p-val', 'corr p-val',
                   'significant', 'corr Levene p-val', 'Effect size',
                   'Equivalence p-val']),
        contrasts = in_contrasts, paired = paired, name = in_name, provenance = in_prov)
    if not monte_carlo:
        prison_rows = pd.DataFrame({
            'Glycan': df_prison.index,
            'Mean abundance': df_org_prison.mean(axis = 1),
            'Log2FC': (df_prison[group2].values - df_prison[group1].values).mean(axis = 1) if paired else (
                        df_prison[group2].mean(axis = 1) - df_prison[group1].mean(axis = 1)),
            'p-val': [1.0] * len(df_prison),
            'corr p-val': [1.0] * len(df_prison),
            'significant': [False] * len(df_prison),
            'corr Levene p-val': [1.0] * len(df_prison),
            'Effect size': [0] * len(df_prison),
            'Equivalence p-val': [1.0] * len(df_prison)})
        prison_rows = prison_rows.astype({'significant': 'bool'})
        if len(prison_rows) > 0:
            df_out = GlycoDataFrame(
                prison_rows if df_out.empty else pd.concat([df_out, prison_rows], ignore_index = True),
                contrasts = in_contrasts, paired = paired, name = in_name, provenance = in_prov)
    df_out['significant'] = df_out['significant'].astype('bool')
    if effect_size_variance:
        df_out['Effect size variance'] = list(variances) + [0] * len(df_prison)
    if (motifs or glycoproteomics) and not sets and not monte_carlo and df_org.attrs.get(
            'motif_dag') is not None and not df.empty:
        dag, full = df_org.attrs['motif_dag'], pd.concat([df_org, df_org_prison])
        # Recover the per-sample logratio offset from the transform itself, so residual features land in the frame everything else was tested in
        ref = np.nanmedian(np.log2(df_org.values + 0.0000001) - df.values, axis = 0)
        fc, rows = dict(zip(df_out['Glycan'], df_out['Log2FC'])), {}
        pos, F = {m: i for i, m in enumerate(full.index)}, full.values
        g1i = [full.columns.get_loc(c) for c in group1]
        g2i = [full.columns.get_loc(c) for c in group2]
        for p in [m for m in full.index if m in dag and dag.out_degree(m)]:
            kids = [c for c in dag.successors(p) if c in pos]
            kv = F[[pos[c] for c in kids]]
            resid = F[pos[p]] - kv.sum(axis = 0)
            # A per-sample scalar cancels from a parent/child logratio, so these balances carry no reference frame and no scale model; alone among the outputs they are pure data
            parts = np.vstack([kv, np.clip(resid, 0, None)])
            parts = parts[(parts > 1e-6).any(
                axis = 1)]  # a part that never occurs is not in the sub-composition and cannot redistribute
            # A balance is only defined in a sample where every part was measured; a structurally missing cell is not a zero, and leaving it in makes the whole row's std NaN, fails the variance filter, and empties bal
            obs = ~np.isnan(parts).any(axis = 0)
            cols_b = [c for k in range(len(g1i)) if obs[g1i[k]] and obs[g2i[k]] for c in
                      (g1i[k], g2i[k])] if paired else [c for c in g1i + g2i if obs[c]]
            b1, b2 = [k for k, c in enumerate(cols_b) if c in g1i], [k for k, c in enumerate(cols_b) if c in g2i]
            # Children and residual sum to the parent, so they are a genuine sub-composition; subtracting one part gives its additive logratio, which is the isometric test in disguise (Hotelling's T2 is affine-invariant) and drops the evenness direction that dividing by the parent leaves behind
            bal = np.log2(parts[:, cols_b] + 0.0000001)
            bal = bal[1:] - bal[0] if len(bal) > 1 else bal[:0]
            bal = bal[bal.std(axis = 1, ddof = 1) > 1e-9] if len(bal) and len(cols_b) > 1 else bal[:0]
            bal_p = hotellings_t2(bal[:, b1].T, bal[:, b2].T, paired = paired)[1] if 0 < len(bal) < min(len(b1),
                                                                                                        len(b2)) else np.nan
            explained = ', '.join((f'{c} ({fc[c] - fc[p]:+.2f})' if p in fc else c) for c in kids if c in fc)
            if (resid <= 1e-6).all():
                rows[p] = (explained, 1.0, 0.0,
                           bal_p)  # parent occurs only inside its children: no context of its own left to test
                continue
            r = np.log2(np.clip(resid, 0.0000001, None)) - ref
            r_a, r_b = r[g1i], r[g2i]
            rows[p] = (explained, ttest_rel(r_b, r_a, nan_policy = 'omit')[1] if paired else
            ttest_ind(r_b, r_a, equal_var = False, nan_policy = 'omit')[1],
                       cohen_d(r_b, r_a, paired = paired)[0], bal_p)
        # Residuals and balances answer different questions than the marginals, so each is corrected as its own, much smaller family
        cp = dict(zip(rows, correct_multiple_testing([v[1] for v in rows.values()], alpha)[0])) if rows else {}
        bk = [m for m in rows if not np.isnan(rows[m][3])]
        bp = dict(zip(bk, correct_multiple_testing([rows[m][3] for m in bk], alpha)[0])) if bk else {}
        df_out['Explained by'] = [rows[m][0] if m in rows else '' for m in df_out['Glycan']]
        df_out['Redistribution p-val'] = [bp.get(m, np.nan) for m in df_out['Glycan']]
        df_out['Residual p-val'] = [cp.get(m, np.nan) for m in df_out['Glycan']]
        df_out['Residual effect size'] = [rows[m][2] if m in rows else np.nan for m in df_out['Glycan']]
    df_out.attrs.update(
        {'alpha': alpha, 'n': len(group1) + len(group2), 'test': "Welch's t-test" if not paired else "paired t-test",
         'transform': transform, 'paired': paired, 'dataset': in_name, 'provenance': in_prov})
    if glycoproteomics:
        df_site = get_glycoform_diff(df_out, alpha = alpha, level = level)
        df_site.attrs[
            'glycoforms'] = df_out  # aggregating to glycosites drops the per-glycoform decomposition, so we keep it reachable
        return df_site
    else:
        return df_out.dropna(subset = ['Log2FC', 'p-val']).sort_values(by = ['corr p-val', 'p-val'])


def get_pval_distribution(
        df_res: pd.DataFrame | str | Path,  # Output DataFrame from get_differential_expression
        filepath: str | Path = ''  # Path to save plot
) -> None:
    "Creates histogram of p-values from differential expression analysis"
    import seaborn as sns
    if isinstance(df_res, (str, Path)):
        df_res = pd.read_csv(df_res) if Path(df_res).suffix.lower() == ".csv" else pd.read_csv(df_res,
                                                                                               sep = "\t") if Path(
            df_res).suffix.lower() == ".tsv" else pd.read_excel(df_res)
    # make plot
    ax = sns.histplot(x = 'p-val', data = df_res, stat = 'frequency')
    ax.set(xlabel = 'p-values', ylabel = 'Frequency', title = '')
    sns.despine(left = True, bottom = True)
    # save to file
    if filepath:
        plt.savefig(filepath, format = Path(filepath).suffix[1:], dpi = 300, bbox_inches = 'tight')
    plt.show()


def get_ma(
        df_res: pd.DataFrame | str | Path,  # Output DataFrame from get_differential_expression
        log2fc_thresh: int = 1,  # Log2FC threshold for highlighting
        sig_thresh: float | None = None,  # Significance threshold for highlighting; defaults to the sample-size-adjusted alpha stored on the results
        filepath: str | Path = ''  # Path to save plot
) -> None:
    "Generates MA plot (mean abundance vs log2 fold change) from differential expression results"
    import seaborn as sns
    if isinstance(df_res, (str, Path)):
        df_res = pd.read_csv(df_res) if Path(df_res).suffix.lower() == ".csv" else pd.read_csv(df_res,
                                                                                               sep = "\t") if Path(
            df_res).suffix.lower() == ".tsv" else pd.read_excel(df_res)
    if sig_thresh is None:
        sig_thresh = df_res.attrs.get('alpha', 0.05)
    # Create masks for significant and non-significant points
    sig_mask = (abs(df_res['Log2FC']) > log2fc_thresh) & (df_res['corr p-val'] < sig_thresh)
    ax = sns.scatterplot(x = 'Mean abundance', y = 'Log2FC', data = df_res[~sig_mask],
                         color = '#CCCCCC', alpha = 0.5, s = 20, linewidth = 0)
    sns.scatterplot(x = 'Mean abundance', y = 'Log2FC', data = df_res[sig_mask & (df_res['Log2FC'] > 0)],
                    color = '#C84B55', alpha = 0.9, s = 30, linewidth = 0, ax = ax)
    sns.scatterplot(x = 'Mean abundance', y = 'Log2FC', data = df_res[sig_mask & (df_res['Log2FC'] <= 0)],
                    color = '#2D6A9F', alpha = 0.9, s = 30, linewidth = 0, ax = ax)
    ax.axhline(0, color = '#888888', ls = '--', lw = 0.8, alpha = 0.5)
    ax.set(xlabel = 'Mean Abundance', ylabel = 'Log2FC', title = df_res.attrs.get('dataset', ''))
    sns.despine(left = True, bottom = True)
    # save to file
    if filepath:
        plt.savefig(filepath, format = Path(filepath).suffix[1:], dpi = 300, bbox_inches = 'tight')
    plt.show()


def get_volcano(
        df_res: pd.DataFrame | str | Path,
        # DataFrame from get_differential_expression with columns [Glycan, Log2FC, p-val, corr p-val]
        y_thresh: float | None = None,  # Corrected p threshold for labeling; default: alpha stamped by the analysis function, else 0.05
        x_thresh: float = 0,  # Absolute x metric threshold for labeling
        n: int | None = None,  # Sample size for Bayesian-Adaptive Alpha
        label_changed: bool = True,  # Add text labels to significant points
        x_metric: str = 'Log2FC',  # x-axis metric: 'Log2FC' or 'Effect size'
        annotate_volcano: bool = False,  # Annotate dots with SNFG images
        filepath: str = '',  # Path to save plot
        **kwargs: Any  # Keyword args passed to seaborn scatterplot
) -> None:  # Displays volcano plot
    "Creates volcano plot showing -log10(FDR-corrected p-values) vs Log2FC or effect size"
    import seaborn as sns
    if annotate_volcano and not filepath:
        raise ValueError("annotate_volcano = True draws the SNFG annotations into a saved figure and therefore needs a filepath, e.g., filepath = 'volcano.svg'.")
    if isinstance(df_res, (str, Path)):
        df_res = pd.read_csv(df_res) if Path(df_res).suffix.lower() == ".csv" else pd.read_csv(df_res,
                                                                                               sep = "\t") if Path(
            df_res).suffix.lower() == ".tsv" else pd.read_excel(df_res)
    df_res = df_res.copy()
    df_res['log_p'] = -np.log10(df_res['corr p-val'].values)
    x = df_res[x_metric].values
    y = df_res['log_p'].values
    if df_res.index.name == "Glycan": df_res.reset_index(inplace = True)
    labels = df_res['Glycan'].values if 'Glycan' in df_res.columns else df_res['Glycosite'].values
    # set y_thresh based on sample size via Bayesian-Adaptive Alpha Adjustment
    if y_thresh is None:
        y_thresh = get_alphaN(n) if n else df_res.attrs.get('alpha', 0.05)
    # Make plot
    kwargs.pop('color', None)
    kwargs.pop('hue', None)
    sig = df_res['log_p'] > -np.log10(y_thresh)
    df_res['_cat'] = np.where(sig & (df_res[x_metric] > 0), 'up', np.where(sig & (df_res[x_metric] <= 0), 'down', 'ns'))
    ax = sns.scatterplot(x = x_metric, y = 'log_p', data = df_res, hue = '_cat',
                         palette = {'up': '#C84B55', 'down': '#2D6A9F', 'ns': '#BBBBBB'},
                         alpha = 0.85, s = 25, linewidth = 0, legend = False, **kwargs)
    df_res.drop('_cat', axis = 1, inplace = True)
    ax.set(xlabel = x_metric, ylabel = '-log10(corr p-val)', title = df_res.attrs.get('dataset', ''))
    plt.axhline(y = -np.log10(y_thresh), c = '#888888', ls = '--', lw = 0.8, alpha = 0.5)
    plt.axvline(x = x_thresh, c = '#888888', ls = '--', lw = 0.8, alpha = 0.5)
    plt.axvline(x = -x_thresh, c = '#888888', ls = '--', lw = 0.8, alpha = 0.5)
    sns.despine(bottom = True, left = True)
    # Text labels
    if label_changed:
        for i in np.where((y > -np.log10(y_thresh)) & (np.abs(x) > x_thresh))[0]:
            plt.text(x[i], y[i], labels[i], fontsize = 8, alpha = 0.85,
                     bbox = dict(boxstyle = 'round,pad=0.15', facecolor = 'white', alpha = 0.5, linewidth = 0))
    # Save to file
    if filepath:
        plt.savefig(filepath, format = Path(filepath).suffix[1:], dpi = 300, bbox_inches = 'tight')
        if annotate_volcano:
            from glycowork.motif.draw import annotate_figure
            svg_temp = filepath.rsplit('.', 1)[0] + '_temp.svg'
            plt.savefig(svg_temp, format = 'svg', bbox_inches = 'tight')
            annotate_figure(svg_temp, filepath = filepath, scale_by_DE_res = df_res, y_thresh = y_thresh,
                            x_thresh = x_thresh, x_metric = x_metric)
            import os
            os.remove(svg_temp)
    plt.show()


def get_glycanova(
        df: pd.DataFrame | str | Path,  # DataFrame with glycans in rows (col 1) and abundance values in columns
        groups: list[Any] | None = None,  # Group labels for samples (e.g., [1,1,1,2,2,2,3,3,3]); inferred from a GlycoDataFrame's contrasts if omitted
        impute: bool = True,  # Replace zeros with Random Forest model
        motifs: bool = False,  # Analyze motifs instead of sequences
        feature_set: list[str] = ['exhaustive', 'known'],
        # Feature sets to use; exhaustive, known, terminal1, terminal2, terminal3, chemical, graph, custom, size_branch
        min_samples: float = 0.1,  # Min percent of non-zero samples required
        posthoc: bool = True,  # Perform Tukey's HSD test post-hoc
        grouped_BH: bool | None = None,  # Use two-stage adaptive Benjamini-Hochberg; None infers True for motifs (DAG-grouped families) and False for sequences
        custom_motifs: list[str] = [],  # Custom motifs if using 'custom' feature set
        transform: str | None = None,  # Transformation type: "CLR" or "ALR"; None auto-decides
        gamma: float = 0.1,  # Uncertainty parameter for CLR transform
        custom_scale: float = 0,
        # Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
        moderate_variance: bool = True,
        # Empirical-Bayes variance moderation, with the containment DAG as the prior neighborhood
        glycoproteomics: bool = False,  # Whether rows are glycoforms from glycoproteomics instead of glycans
        random_state: int | np.random.Generator | None = None  # optional random state for reproducibility
) -> tuple[GlycoDataFrame, dict[
    str, pd.DataFrame]]:  # (ANOVA results with F-stats and omega-squared effect sizes, post-hoc results)
    "Performs one-way ANOVA with omega-squared effect size calculation and optional Tukey's HSD post-hoc testing on glycomics data across multiple groups"
    from scipy.stats import tukey_hsd
    grouped_BH = (motifs or glycoproteomics) if grouped_BH is None else grouped_BH
    if groups is None and isinstance(df, GlycoDataFrame) and df._contrasts:
        groups = list(df.groups)
    if not groups:
        raise ValueError(
            "No groups given: pass groups as a list of per-sample labels; groups are only inferred automatically from a GlycoDataFrame that carries contrasts.")
    if len(set(groups)) < 3:
        raise ValueError(
            "You have fewer than three groups. We suggest get_differential_expression for those cases. ANOVA is for >= three groups.")
    df, df_org, groups, _ = preprocess_data(df, group1 = groups, group2 = [], experiment = "anova", motifs = motifs, impute = impute,
                                       glycoproteomics = glycoproteomics,
                                       min_samples = min_samples, transform = transform, feature_set = feature_set,
                                       gamma = gamma, custom_scale = custom_scale, custom_motifs = custom_motifs,
                                       random_state = random_state)
    results, posthoc_results = [], {}
    # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
    alpha = get_alphaN(len(groups))
    effect_sizes = omega_squared(df, groups)
    # Variance-based filtering of features
    df, df_prison = variance_based_filtering(df)
    garr, X = np.asarray(groups), df.values
    # One-way ANOVA on a fixed design is the same F for every feature, so all features go through one vectorized call instead of one formula parse and OLS fit each
    f_values, p_values = f_oneway(*[X[:, garr == g] for g in np.unique(garr)], axis = 1, nan_policy = 'omit')
    if moderate_variance and len(X) > 1:
        # Shrinking each feature's residual variance toward its containment neighborhood stabilizes the F test without touching the reported effect sizes
        ug = np.unique(garr)
        ng = np.stack([np.isfinite(X[:, garr == g]).sum(axis = 1) for g in ug],
                      axis = 1)  # per-feature group sizes, because glycoproteomics leaves structurally unmeasured cells as NaN; on dense input these are constant and everything below reduces to the old expressions
        gm, dfr = (np.nansum(X, axis = 1) / ng.sum(axis = 1))[:, None], ng.sum(axis = 1) - len(ug)
        ssb = sum(((np.nanmean(X[:, garr == g], axis = 1, keepdims = True) - gm) ** 2).ravel() * ng[:, i] for i, g in
                  enumerate(ug))
        ssw = sum(
            np.nansum((X[:, garr == g] - np.nanmean(X[:, garr == g], axis = 1, keepdims = True)) ** 2, axis = 1) for g
            in ug)
        s2, dfp = moderated_variance(ssw / dfr, df_resid = dfr,
                                     neighbors = dag_neighbors(df.index.tolist(), df_org.attrs.get('motif_dag')))
        f_values = (ssb / (len(ug) - 1)) / s2
        p_values = f.sf(f_values, len(ug) - 1, dfp)
    results = list(zip(df.index, f_values, p_values))
    if posthoc:
        for i, glycan in enumerate(df.index):
            if p_values[i] < alpha:
                ug_ph = np.unique(garr)
                res_ph = tukey_hsd(*[X[i][garr == g] for g in ug_ph])
                ci_ph = res_ph.confidence_interval(1 - alpha)
                posthoc_results[glycan] = pd.DataFrame(
                    [{'group1': ug_ph[a], 'group2': ug_ph[b], 'meandiff': -res_ph.statistic[a, b],
                      'p-adj': res_ph.pvalue[a, b], 'lower': -ci_ph.high[a, b], 'upper': -ci_ph.low[a, b],
                      'reject': res_ph.pvalue[a, b] < alpha}
                     for a in range(len(ug_ph)) for b in range(a + 1, len(ug_ph))])
    df_out = GlycoDataFrame(results, columns = ["Glycan", "F statistic", "p-val"])
    dag = df_org.attrs.get('motif_dag') if motifs or glycoproteomics else None
    if grouped_BH and dag is not None:
        grouped_glycans, grouped_pvals = select_grouping(df, df, df_out['Glycan'].tolist(), df_out['p-val'].tolist(),
                                                         grouped_BH = grouped_BH, dag = dag)
        corrpvals, significance_dict = TST_grouped_benjamini_hochberg(grouped_glycans, grouped_pvals, alpha)
        corrpvals = [corrpvals[g] for g in df_out['Glycan']]
        corrpvals = [p if p >= df_out['p-val'].iloc[i] else df_out['p-val'].iloc[i] for i, p in enumerate(corrpvals)]
        significance = [significance_dict[g] for g in df_out['Glycan']]
    else:
        corrpvals, significance = correct_multiple_testing(df_out['p-val'], alpha)
    df_out['corr p-val'] = corrpvals
    df_out['significant'] = significance
    prison_rows = pd.DataFrame({
        'Glycan': df_prison.index,
        'F statistic': [0] * len(df_prison),
        'p-val': [1.0] * len(df_prison),
        'corr p-val': [1.0] * len(df_prison),
        'significant': [False] * len(df_prison)})
    prison_rows = prison_rows.astype({'significant': 'bool'})
    if len(prison_rows) > 0:
        # An all-prison run leaves df_out empty with object dtypes, and inferring result dtypes from that is what pandas is deprecating
        df_out = GlycoDataFrame(prison_rows) if df_out.empty else pd.concat([df_out, prison_rows], ignore_index = True)
    df_out['significant'] = df_out['significant'].astype('bool')
    df_out['Effect size'] = effect_sizes.reindex(df_out['Glycan']).values
    # With no feature surviving the variance filter there is no transformed frame to read the per-sample offset off, so the residuals have no reference frame to land in
    if (motifs or glycoproteomics) and df_org.attrs.get('motif_dag') is not None and not df.empty:
        dag, full = df_org.attrs['motif_dag'], df_org
        # Recover the per-sample logratio offset from the transform itself, so residual features land in the frame everything else was tested in
        ref = np.nanmedian(np.log2(full.loc[df.index].values + 0.0000001) - df.values, axis = 0)
        levels, garr = sorted(set(groups)), np.asarray(groups)
        eff, rows = dict(zip(df_out['Glycan'], df_out['Effect size'])), {}
        pos, F = {m: i for i, m in enumerate(full.index)}, full.values
        for p in [m for m in full.index if m in dag and dag.out_degree(m)]:
            kids = [c for c in dag.successors(p) if c in pos]
            kv = F[[pos[c] for c in kids]]
            resid = F[pos[p]] - kv.sum(axis = 0)
            # Children and residual sum to the parent, so subtracting one part's log gives the additive logratios of a genuine sub-composition; a one-way PERMANOVA on those balances asks whether the parent redistributes across contexts, free of any reference frame or scale model
            parts = np.vstack([kv, np.clip(resid, 0, None)])
            parts = parts[(parts > 1e-6).any(
                axis = 1)]  # a part that never occurs is not in the sub-composition and cannot redistribute
            # A balance is only defined in a sample where every part was measured; a structurally missing cell is not a zero, and leaving it in makes the whole row's std NaN, fails the variance filter, and empties bal
            obs = ~np.isnan(parts).any(axis = 0)
            grp_b = garr[obs].tolist()
            bal = np.log2(parts[:, obs] + 0.0000001)
            bal = bal[1:] - bal[0] if len(bal) > 1 else bal[:0]
            bal = bal[bal.std(axis = 1, ddof = 1) > 1e-9] if len(bal) and obs.sum() > 1 else bal[:0]
            bal_p = permanova_with_permutation(squareform(pdist(bal.T, metric = 'euclidean')), group_labels = grp_b,
                                               permutations = 999)[1] if len(
                bal) and len(set(grp_b)) > 1 else np.nan
            explained = ', '.join((f'{c} ({eff[c] - eff[p]:+.2f})' if p in eff else c) for c in kids if c in eff)
            if (resid <= 1e-6).all():
                rows[p] = (explained, 1.0, 0.0,
                           bal_p)  # parent occurs only inside its children: no context of its own left to test
                continue
            r = np.log2(np.clip(resid, 0.0000001, None)) - ref
            rows[p] = (explained, f_oneway(*[r[garr == g] for g in levels])[1], omega_squared(r, groups), bal_p)
        # Residuals and balances answer different questions than the marginals, so each is corrected as its own, much smaller family
        cp = dict(zip(rows, correct_multiple_testing([v[1] for v in rows.values()], alpha)[0])) if rows else {}
        bk = [m for m in rows if not np.isnan(rows[m][3])]
        bp = dict(zip(bk, correct_multiple_testing([rows[m][3] for m in bk], alpha)[0])) if bk else {}
        df_out['Explained by'] = [rows[m][0] if m in rows else '' for m in df_out['Glycan']]
        df_out['Redistribution p-val'] = [bp.get(m, np.nan) for m in df_out['Glycan']]
        df_out['Residual p-val'] = [cp.get(m, np.nan) for m in df_out['Glycan']]
        df_out['Residual effect size'] = [rows[m][2] if m in rows else np.nan for m in df_out['Glycan']]
    df_out.attrs.update({'alpha': alpha, 'n': len(groups), 'test': 'ANOVA', 'transform': transform, 'paired': False,
                         'dataset': df_org.attrs.get('dataset', ''), 'provenance': df_org.attrs.get('provenance', {})})
    return df_out.sort_values(by = 'corr p-val'), posthoc_results


def get_meta_analysis(
        effect_sizes: np.ndarray | list[float],  # List of Cohen's d/other effect sizes
        variances: np.ndarray | list[float],  # Associated variance estimates
        model: str = 'fixed',  # 'fixed' or 'random' effects model
        filepath: str = '',  # Path to save Forest plot
        study_names: list[str] = [],  # Names corresponding to each effect size
        full_output: bool = False  # Return heterogeneity statistics (tau2, Q, I2) and leave-one-out pooling instead of just (effect, p-value)
) -> tuple[
         float, float] | dict:  # (combined effect size, two-tailed p-value), or the full result dict when full_output=True
    "Performs fixed/random effects meta-analysis using DerSimonian-Laird method for between-study variance estimation, with optional Forest plot visualization"
    res = meta_analysis(effect_sizes, variances, model = model, leave_one_out = full_output)
    effect_sizes, variances = np.array(effect_sizes), np.array(variances)
    combined_effect_size, p_value = res['effect'], res['p_val']
    # Check whether Forest plot should be constructed and saved
    if filepath:
        df_temp = pd.DataFrame({'Study': study_names, 'EffectSize': effect_sizes, 'EffectSizeVariance': variances})
        # sort studies by effect size
        df_temp = df_temp.sort_values(by = 'EffectSize', key = abs, ascending = False)
        # calculate standard error
        standard_error = df_temp['EffectSizeVariance'].pow(.5)
        # calculate the confidence interval
        df_temp['lower'] = df_temp['EffectSize'] - 1.96 * standard_error
        df_temp['upper'] = df_temp['EffectSize'] + 1.96 * standard_error
        # Create a new figure and a axes to plot on
        _, ax = plt.subplots(figsize = (7, max(len(df_temp) * 0.55, 3)))
        y_pos = np.arange(len(df_temp))
        ax.hlines(y_pos, df_temp['lower'], df_temp['upper'], color = '#2D6A9F', lw = 2.5, alpha = 0.6)
        ax.scatter(df_temp['EffectSize'], y_pos, color = '#2D6A9F', s = 55, zorder = 3)
        ax.axvline(0, color = '#888888', ls = '--', lw = 0.9, alpha = 0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_temp['Study'])
        ax.invert_yaxis()
        ax.set_xlabel('Effect size')
        for spine in ['right', 'top', 'left']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(left = False)
        plt.tight_layout()
        plt.savefig(filepath, format = Path(filepath).suffix[1:], dpi = 300, bbox_inches = 'tight')
    return res if full_output else (combined_effect_size, p_value)


def get_glycan_change_over_time(
        data: np.ndarray,  # 2D array with columns [timepoint, abundance]
        degree: int = 1  # Polynomial degree for regression
) -> tuple[float | np.ndarray, float]:  # (regression coefficients, t-test/F-test p-value)
    "Fits polynomial regression (default: linear) to glycan abundance time series data using OLS, testing significance of temporal changes"
    from scipy.stats import linregress
    # Extract arrays for time and glycan abundance from the 2D input array
    time, glycan_abundance = data[:, 0], data[:, 1]
    if degree == 1:
        results = linregress(time, glycan_abundance)
        coefficients, p_value = results.slope, results.pvalue
    else:
        # Polynomial Regression
        coefficients = np.polyfit(time, glycan_abundance, degree)
        residuals = glycan_abundance - np.polyval(coefficients, time)
        n = len(time)
        ss_res, ss_tot = float(np.sum(residuals ** 2)), float(np.sum((glycan_abundance - glycan_abundance.mean()) ** 2))
        df_resid = n - degree - 1
        p_value = 1.0 if (df_resid < 1 or ss_res <= 0 or ss_tot <= ss_res) else float(
            f.sf(((ss_tot - ss_res) / degree) / (ss_res / df_resid), degree, df_resid))
    return coefficients, p_value


def get_time_series(
        df: pd.DataFrame | str | Path,
        # DataFrame with sample IDs as 'sampleID_timepoint_replicate' in col 1 (e.g., T1_h5_r1)
        impute: bool = True,  # Replace zeros with Random Forest model
        motifs: bool = False,  # Analyze motifs instead of sequences
        feature_set: list[str] = ['known', 'exhaustive'],
        # Feature sets to use; exhaustive, known, terminal1, terminal2, terminal3, chemical, graph, custom, size_branch
        degree: int = 1,  # Polynomial degree for regression
        min_samples: float = 0.1,  # Min percent of non-zero samples required
        grouped_BH: bool | None = None,  # Family-grouped two-stage Benjamini-Hochberg via the motif DAG; None infers True for motifs and False for sequences
        custom_motifs: list[str] = [],  # Custom motifs if using 'custom' feature set
        transform: str | None = None,  # Transformation type: "CLR" or "ALR"; None auto-decides
        gamma: float = 0.1,  # Uncertainty parameter for CLR transform
        custom_scale: float | dict = 0,
        # Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
        glycoproteomics: bool = False, # Whether rows are glycoforms, ordered by composition containment instead of substructure containment
        random_state: int | np.random.Generator | None = None  # optional random state for reproducibility
) -> GlycoDataFrame:  # DataFrame with regression coefficients and FDR-corrected p-values
    "Analyzes time series glycomics data using polynomial regression"
    grouped_BH = (motifs or glycoproteomics) if grouped_BH is None else grouped_BH
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df) if Path(df).suffix.lower() == ".csv" else pd.read_csv(df, sep = "\t") if Path(
            df).suffix.lower() == ".tsv" else pd.read_excel(df)
    df = df.fillna(0)
    if isinstance(df.iloc[0, 0], str):
        df = df.set_index(df.columns[0])
    if not glycoproteomics and '-' not in df.index[0]:
        df = df.T  # glycan IUPAC labels always carry a linkage dash; glycoform IDs do not, so the orientation heuristic must not fire on them
    df = replace_outliers_winsorization(df).reset_index(names = 'glycan')
    df = impute_and_normalize(df, [df.columns[1:].tolist()], impute = impute, min_samples = min_samples, random_state = random_state)
    if transform is None:
        transform = "ALR" if enforce_class(df.iloc[0, 0], "N") and len(df) > 50 else "CLR"
    # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
    alpha = get_alphaN(df.shape[1] - 1)
    dag = None
    if motifs:
        # Quantify on raw abundances, then transform the motif composition (as preprocess_data/get_pca do); transforming glycans first centers motifs by the glycan geometric mean instead of the motif one
        glycans = strip_suffixes(df.iloc[:, 0])
        df = quantify_motifs(df.iloc[:, 1:], glycans = glycans, feature_set = feature_set,
                             custom_motifs = custom_motifs) + 0.0000001
        # Containment DAG off the raw motif frame, so its abundance-dominance prefilter stays valid (built pre-transform)
        dag = get_motif_dag(df.index.tolist(), abundances = df) if grouped_BH else None
        if transform == "ALR":
            df = get_additive_logratio_transformation(df.reset_index(), df.columns.tolist(), [], paired = False,
                                                      gamma = gamma, custom_scale = custom_scale, random_state = random_state)
            df = df.set_index(df.columns[0])
        elif transform == "CLR":
            df = clr_transformation(df, df.columns.tolist(), [], gamma = gamma, custom_scale = custom_scale, random_state = random_state)
        elif transform != "Nothing":
            raise ValueError("Only ALR and CLR are valid transforms for now.")
    else:
        if glycoproteomics and grouped_BH:
            # Composition containment off the raw glycoform frame (pre-transform), keeping the abundance-dominance prefilter valid
            raw = df.iloc[:, 1:].copy()
            raw.index = strip_suffixes(df.iloc[:, 0])
            raw = raw.groupby(level = 0).mean()
            dag = get_composition_dag(raw.index.tolist(), abundances = raw)
        if transform == "ALR":
            df = get_additive_logratio_transformation(df, df.columns[1:].tolist(), [], paired = False,
                                                      gamma = gamma,
                                                      custom_scale = custom_scale, random_state = random_state)
        elif transform == "CLR":
            df.iloc[:, 1:] = clr_transformation(df.iloc[:, 1:], df.columns[1:].tolist(), [], gamma = gamma,
                                                custom_scale = custom_scale, random_state = random_state)
        elif transform != "Nothing":
            raise ValueError("Only ALR and CLR are valid transforms for now.")
        df.index = strip_suffixes(df.iloc[:, 0])
        df = df.drop([df.columns[0]], axis = 1)
        df = df.groupby(df.index).mean()
    df = df.T.reset_index()
    try:
        df[df.columns[0]] = df.iloc[:, 0].apply(lambda x: float(x.split('_')[1][1:]))
    except (IndexError, ValueError) as e:
        raise ValueError(
            f"Sample columns have to be named 'sampleID_timepoint_replicate' (e.g., 'T1_h5_r1'); could not read a timepoint from {df.iloc[:, 0].tolist()[:3]}") from e
    df = df.sort_values(by = df.columns[0])
    time = df.iloc[:, 0].to_numpy()  # Time points
    df_out = [(c, *get_glycan_change_over_time(np.column_stack((time, df[c].to_numpy())), degree = degree)) for c in
              df.columns[1:]]
    df_out = GlycoDataFrame(df_out, columns = ['Glycan', 'Change', 'p-val'])
    if grouped_BH and dag is not None:
        df_num = df.iloc[:, 1:].T.astype(float)
        grouped_glycans, grouped_pvals = select_grouping(df_num, df_num, df_out['Glycan'].tolist(),
                                                         df_out['p-val'].tolist(),
                                                         grouped_BH = grouped_BH, dag = dag)
        corrpvals, significance_dict = TST_grouped_benjamini_hochberg(grouped_glycans, grouped_pvals, alpha)
        corrpvals = [corrpvals[g] for g in df_out['Glycan']]
        corrpvals = [p if p >= df_out['p-val'].iloc[i] else df_out['p-val'].iloc[i] for i, p in enumerate(corrpvals)]
        significance = [significance_dict[g] for g in df_out['Glycan']]
    else:
        corrpvals, significance = correct_multiple_testing(df_out['p-val'], alpha)
    df_out['corr p-val'] = corrpvals
    df_out['significant'] = significance
    df_out.attrs.update(
        {'alpha': alpha, 'n': df.shape[0], 'test': 'OLS trend', 'transform': transform, 'paired': False})
    return df_out.sort_values(by = 'corr p-val')


def get_jtk(
        df_in: pd.DataFrame | str | Path,
        # DataFrame with glycans in rows (first column), then groups arranged by ascending timepoints
        timepoints: int,  # Number of timepoints (each must have same number of replicates)
        interval: int,  # Time units between experimental timepoints
        periods: list[int] = [12, 24],  # Timepoints per cycle to test
        motifs: bool = False,  # Analyze motifs instead of sequences
        feature_set: list[str] = ['known', 'exhaustive', 'terminal'],
        # Feature sets to use; exhaustive, known, terminal1, terminal2, terminal3, chemical, graph, custom, size_branch
        custom_motifs: list[str] = [],  # Custom motifs if using 'custom' feature set
        transform: str | None = None,  # Transformation type: "CLR" or "ALR"; None auto-decides
        gamma: float = 0.1,  # Uncertainty parameter for CLR transform
        correction_method: str = "two-stage",  # Multiple testing correction method
        grouped_BH: bool | None = None,  # Family-grouped two-stage Benjamini-Hochberg via the motif DAG; None infers True for motifs and False for sequences
        glycoproteomics: bool = False,  # Whether rows are glycoforms, ordered by composition containment instead of substructure containment
        random_state: int | np.random.Generator | None = None  # optional random state for reproducibility
) -> GlycoDataFrame:  # DataFrame with JTK results: adjusted p-values, period length, lag phase, amplitude
    "Identifies rhythmically expressed glycans using Jonckheere-Terpstra-Kendall algorithm for time series analysis"
    grouped_BH = (motifs or glycoproteomics) if grouped_BH is None else grouped_BH
    if isinstance(df_in, (str, Path)):
        df = pd.read_csv(df_in) if Path(df_in).suffix.lower() == ".csv" else pd.read_csv(df_in, sep = "\t") if Path(
            df_in).suffix.lower() == ".tsv" else pd.read_excel(df_in)
    else:
        df = df_in.copy(deep = True)
    replicates = (df.shape[1] - 1) // timepoints
    alpha = get_alphaN(df.shape[1] - 1)
    jtk = JTKTest(timepoints, periods, interval, replicates)
    df = replace_outliers_winsorization(df)
    mf = MissForest(circadian = True, timepoints = timepoints, periods = periods, interval = interval, replicates = replicates,
                    random_state = random_state)
    df = df.replace(0, np.nan)
    annot = df.pop(df.columns[0])
    df = mf.fit_transform(df)
    df.insert(0, 'Molecule_Name', annot)
    if transform is None:
        transform = "ALR" if enforce_class(df.iloc[0, 0], "N") and len(df) > 50 else "CLR"
    dag = None
    if motifs:
        df = quantify_motifs(df, feature_set = feature_set, custom_motifs = custom_motifs) + 0.0000001
        # Containment DAG off the raw motif frame (pre-transform), keeping the abundance-dominance prefilter valid
        dag = get_motif_dag(df.index.tolist(), abundances = df) if grouped_BH else None
        if transform == "CLR":
            df = clr_transformation(df, df.columns.tolist(), [], gamma = gamma, random_state = random_state).reset_index()
        elif transform == "ALR":
            df = get_additive_logratio_transformation(df.reset_index(), df.columns.tolist(), [], paired = False,
                                                      gamma = gamma, random_state = random_state)
        elif transform == "Nothing":
            df = df.reset_index()
        else:
            raise ValueError("Only ALR and CLR are valid transforms for now.")
    else:
        if glycoproteomics and grouped_BH:
            # Composition containment off the raw glycoform frame (pre-transform), keeping the abundance-dominance prefilter valid
            raw = df.set_index(df.columns[0]).astype(float)
            dag = get_composition_dag(raw.index.tolist(), abundances = raw)
        if transform == "ALR":
            df = get_additive_logratio_transformation(df, df.columns[1:].tolist(), [], paired = False,
                                                      gamma = gamma, random_state = random_state)
        elif transform == "CLR":
            df.iloc[:, 1:] = clr_transformation(df.iloc[:, 1:], df.columns[1:].tolist(), [], gamma = gamma, random_state = random_state)
        elif transform != "Nothing":
            raise ValueError("Only ALR and CLR are valid transforms for now.")
    results = []
    for _, row in df.iterrows():
        p_val, period, phase, tau = jtk.test(row.iloc[1:].values.astype(float))
        results.append([row.iloc[0], p_val, period, phase, abs(tau)])
    df_out = GlycoDataFrame(results,
                            columns = ['Molecule_Name', 'Adjusted_P_value', 'Period_Length', 'Lag_Phase', 'Amplitude'])
    if grouped_BH and dag is not None:
        df_num = df.iloc[:, 1:].astype(float)
        grouped_glycans, grouped_pvals = select_grouping(df_num, df_num, df_out['Molecule_Name'].tolist(),
                                                         df_out['Adjusted_P_value'].tolist(), grouped_BH = grouped_BH,
                                                         dag = dag)
        corrpvals, significance_dict = TST_grouped_benjamini_hochberg(grouped_glycans, grouped_pvals, alpha)
        corrpvals = [corrpvals[g] for g in df_out['Molecule_Name']]
        corrpvals = [p if p >= df_out['Adjusted_P_value'].iloc[i] else df_out['Adjusted_P_value'].iloc[i] for i, p in
                     enumerate(corrpvals)]
        significance = [significance_dict[g] for g in df_out['Molecule_Name']]
    else:
        corrpvals, significance = correct_multiple_testing(df_out.iloc[:, 1].tolist(), alpha,
                                                           correction_method = correction_method)
    df_out['Adjusted_P_value'] = corrpvals
    df_out['significant'] = significance
    df_out.attrs.update({'alpha': alpha, 'n': df.shape[1] - 1, 'test': 'JTK_CYCLE', 'transform': None, 'paired': False})
    return df_out.sort_values("Adjusted_P_value").reset_index(drop = True)


def get_biodiversity(
        df: pd.DataFrame | str | Path,  # DataFrame with glycans in rows (col 1), abundances in columns
        group1: list[str | int] | None = None,
        # First group column indices or group labels; default: from the frame's contrasts
        group2: list[str | int] | None = None,
        # Second group indices or additional group labels; default: from the frame's contrasts
        metrics: list[str] = ['alpha', 'beta'],  # Diversity metrics to calculate
        motifs: bool = False,  # Analyze motifs instead of sequences
        feature_set: list[str] = ['exhaustive', 'known'],
        # Feature sets to use; exhaustive, known, terminal1, terminal2, terminal3, chemical, graph, custom, size_branch
        custom_motifs: list[str] = [],  # Custom motifs if using 'custom' feature set
        paired: bool | None = None,  # Whether samples are paired; default: from the frame
        permutations: int = 999,  # Number of permutations for ANOSIM/PERMANOVA
        transform: str | None = None,  # Transformation type: "CLR" or "ALR"
        gamma: float = 0.1,  # Uncertainty parameter for CLR transform
        custom_scale: float | dict = 0,
        # Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
        random_state: int | np.random.Generator | None = None,  # optional random state for reproducibility
        circadian: bool = False,  # test whether diversity changes rhythmically over time via JTK
        timepoints: int | None = None,  # number of timepoints, columns ordered by ascending timepoint (required if circadian)
        interval: int = 1,  # time units between timepoints (only relevant if circadian)
        periods: list[int] = [12, 24],  # cycle lengths to test (only relevant if circadian)
) -> tuple:  # First DataFrame with diversity indices and test statistics, second with beta-diversity distance matrix
    "Calculates alpha (Shannon/Simpson) and beta (ANOSIM/PERMANOVA) diversity measures from glycomics data"
    if group1 is None and isinstance(df, GlycoDataFrame) and df._contrasts:
        group1, group2 = list(df.group1), list(df.group2)
    paired = df.paired if paired is None and isinstance(df, GlycoDataFrame) else bool(paired)
    experiment = "diff" if group2 else "anova"
    df, df_org, group1, group2 = preprocess_data(df, group1 = group1, group2 = group2, experiment = experiment, motifs = motifs,
                                                 impute = False, transform = transform, feature_set = feature_set, paired = paired,
                                                 gamma = gamma, custom_scale = custom_scale, custom_motifs = custom_motifs,
                                                 random_state = random_state,
                                                 motif_dag = False)  # rows are diversity metrics, not motifs, so a containment DAG has nothing to group here
    shopping_cart = []
    distance_matrix = pd.DataFrame()
    group_sizes = group1 if not group2 else len(group1) * [1] + len(group2) * [2]
    group_counts = Counter(group_sizes)
    # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
    alpha = get_alphaN(len(group_sizes))
    if 'alpha' in metrics:
        unique_counts = df_org.apply(sequence_richness)
        shan_div = df_org.apply(shannon_diversity_index)
        simp_div = df_org.apply(simpson_diversity_index)
        a_df = pd.DataFrame({'species_richness': unique_counts,
                             'shannon_diversity': shan_div, 'simpson_diversity': simp_div}).T
        if circadian:
            jtk = JTKTest(timepoints, periods, interval, df_org.shape[1] // timepoints)
            for metric_name, series in (('Species richness', unique_counts), ('Shannon diversity', shan_div),
                                        ('Simpson diversity', simp_div)):
                p_val, period, phase, tau = jtk.test(series.values.astype(float))
                shopping_cart.append(pd.DataFrame(
                    {'Metric': f'{metric_name} (JTK)', 'p-val': p_val, 'Period length': period, 'Lag phase': phase,
                     'Amplitude': abs(tau)}, index = [0]))
        elif len(group_counts) == 2 and group2:
            df_a, df_b = a_df[group1], a_df[group2]
            mean_a, mean_b = [np.mean(row_a) for row_a in df_a.values], [np.mean(row_b) for row_b in df_b.values]
            if paired and len(group1) != len(group2):
                raise ValueError(
                    f"For paired samples, group1 and group2 have to be the same size; got {len(group1)} and {len(group2)}.")
            pvals = []
            effect_sizes = []
            for row_a, row_b in zip(df_a.values, df_b.values):
                if np.allclose(row_a, row_b, rtol = 1e-5, atol = 1e-8):
                    pvals.append(1.0)
                    effect_sizes.append(0.0)
                else:
                    pval = ttest_rel(row_b, row_a)[1] if paired else ttest_ind(row_b, row_a, equal_var = False)[1]
                    pvals.append(pval if (pval > 0 and pval < 1) else 1.0)
                    effect, _ = cohen_d(row_b, row_a, paired = paired)
                    effect_sizes.append(effect)
            a_df_stats = pd.DataFrame(list(zip(a_df.index.tolist(), mean_a, mean_b, pvals, effect_sizes)),
                                      columns = ["Metric", "Group1 mean", "Group2 mean", "p-val", "Effect size"])
            shopping_cart.append(a_df_stats)
        elif all(count > 2 for count in group_counts.values()) and len(group_counts) > 2:
            sh_stats = alpha_biodiversity_stats(shan_div, group_sizes)
            shopping_cart.append(
                pd.DataFrame({'Metric': 'Shannon diversity (ANOVA)', 'p-val': sh_stats[1], 'Effect size': sh_stats[0]},
                             index = [0]))
            si_stats = alpha_biodiversity_stats(simp_div, group_sizes)
            shopping_cart.append(
                pd.DataFrame({'Metric': 'Simpson diversity (ANOVA)', 'p-val': si_stats[1], 'Effect size': si_stats[0]},
                             index = [0]))
    if 'beta' in metrics:
        if not isinstance(df.index[0], str):
            df = df.set_index(df.columns[0])
        distance_matrix = squareform(
            pdist(df.values.T, metric = 'braycurtis' if transform == "Nothing" else 'euclidean'))
        if circadian:
            n = distance_matrix.shape[0]
            tvec = np.repeat(np.arange(timepoints) * interval, n // timepoints)[:n].astype(float)
            J = np.eye(n) - np.ones((n, n)) / n
            G = -0.5 * J @ (distance_matrix ** 2) @ J  # Gower-centered distances for db-RDA
            rng = np.random.default_rng(random_state) if not isinstance(random_state,
                                                                        np.random.Generator) else random_state
            for period in periods:
                w = 2 * np.pi / period
                X = np.column_stack([np.ones(n), np.cos(w * tvec), np.sin(w * tvec)])
                p = X.shape[1]
                pseudo_f = lambda design: (lambda H: np.trace(H @ G @ H) / (p - 1) / (
                            np.trace((np.eye(n) - H) @ G @ (np.eye(n) - H)) / (n - p)))(
                    design @ np.linalg.pinv(design.T @ design) @ design.T)
                f_obs = pseudo_f(X)
                perm_f = np.array([pseudo_f(X[rng.permutation(n)]) for _ in range(
                    permutations)])  # free permutation of cosinor design against fixed turnover structure
                p_val = (np.sum(perm_f >= f_obs) + 1) / (permutations + 1)
                shopping_cart.append(pd.DataFrame(
                    {'Metric': f'Beta diversity rhythm {period}h (db-RDA)', 'p-val': p_val, 'Period length': period,
                     'Effect size': f_obs}, index = [0]))
            distance_matrix = pd.DataFrame(distance_matrix, index = range(n), columns = range(n))
        elif all(count > 1 for count in group_counts.values()):
            beta_df_out = pd.DataFrame(distance_matrix, index = range(len(df.columns)),
                                       columns = range(len(df.columns)))
            r, p = anosim(beta_df_out, group_labels_in = group_sizes, permutations = permutations,
                          random_state = random_state)
            b_test_stats = pd.DataFrame({'Metric': 'Beta diversity (ANOSIM)', 'p-val': p, 'Effect size': r},
                                        index = [0])
            shopping_cart.append(b_test_stats)
            f, p = permanova_with_permutation(beta_df_out, group_labels = group_sizes, permutations = permutations,
                                              random_state = random_state)
            b_test_stats = pd.DataFrame({'Metric': 'Beta diversity (PERMANOVA)', 'p-val': p, 'Effect size': f},
                                        index = [0])
            shopping_cart.append(b_test_stats)
    df_out = pd.concat(shopping_cart, axis = 0).reset_index(drop = True)
    corrpvals, significance = correct_multiple_testing(df_out['p-val'], alpha)
    df_out["corr p-val"] = corrpvals
    df_out["significant"] = significance
    df_out.attrs.update(
        {'alpha': alpha, 'n': len(group_sizes), 'test': 'ANOVA/t-test per metric', 'transform': None, 'paired': paired})
    return df_out.sort_values(by = ['corr p-val', 'p-val']).reset_index(drop = True), distance_matrix


def get_SparCC(
        df1: pd.DataFrame | str | Path,  # First DataFrame with glycans in rows (col 1) and abundances in columns
        df2: pd.DataFrame | str | Path,  # Second DataFrame with same format as df1
        motifs: bool = False,  # Analyze motifs instead of sequences
        feature_set: list[str] = ["known", "exhaustive"],
        # Feature sets to use; exhaustive, known, terminal1, terminal2, terminal3, chemical, graph, custom, size_branch
        custom_motifs: list[str] = [],  # Custom motifs if using 'custom' feature set
        transform: str | None = None,  # Transformation type: "CLR" or "ALR"
        gamma: float = 0.1,  # Uncertainty parameter for CLR transform
        partial_correlations: bool = False,  # Use regularized partial correlations
        random_state: int | np.random.Generator | None = None  # optional random state for reproducibility
) -> tuple[pd.DataFrame, pd.DataFrame]:  # (Spearman correlation matrix, FDR-corrected p-value matrix)
    "Calculates SparCC (Sparse Correlations for Compositional Data) between two matching datasets (e.g., glycomics)"
    if isinstance(df1, (str, Path)):
        df1 = pd.read_csv(df1) if Path(df1).suffix.lower() == ".csv" else pd.read_csv(df1, sep = "\t") if Path(
            df1).suffix.lower() == ".tsv" else pd.read_excel(df1)
    if isinstance(df2, (str, Path)):
        df2 = pd.read_csv(df2) if Path(df2).suffix.lower() == ".csv" else pd.read_csv(df2, sep = "\t") if Path(
            df2).suffix.lower() == ".tsv" else pd.read_excel(df2)
    if df1.columns.tolist()[0] != df2.columns.tolist()[0] and df1.columns.tolist()[0] in df2.columns.tolist():
        common_columns = df1.columns.intersection(df2.columns)
        df1 = df1[common_columns]
        df2 = df2[common_columns]
    df1, df2 = df1.copy(), df2.copy()
    df1.iloc[:, 0] = strip_suffixes(df1.iloc[:, 0])
    df2.iloc[:, 0] = strip_suffixes(df2.iloc[:, 0])
    # Drop rows with all zero, followed by outlier removal and imputation & normalization
    df1 = df1.loc[~(df1.iloc[:, 1:] == 0).all(axis = 1)]
    df1 = replace_outliers_winsorization(df1)
    df1 = impute_and_normalize(df1, [df1.columns.tolist()[1:]], random_state = random_state)
    df2 = df2.loc[~(df2.iloc[:, 1:] == 0).all(axis = 1)]
    df2 = replace_outliers_winsorization(df2)
    df2 = impute_and_normalize(df2, [df2.columns.tolist()[1:]], random_state = random_state)
    # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
    alpha = get_alphaN(df1.shape[1] - 1)
    if transform is None:
        transform = "ALR" if (enforce_class(df1.iloc[0, 0], "N") and len(df1) > 50) and (
                    enforce_class(df2.iloc[0, 0], "N") and len(df2) > 50) else "CLR"
    if transform not in ("ALR", "CLR", "Nothing"):
        raise ValueError("Only ALR and CLR are valid transforms for now.")
    # Quantify on raw abundances, then transform the motif composition, as preprocess_data/get_pca/get_time_series do; transforming glycans first centers motifs by the glycan geometric mean instead of the motif one
    if motifs:
        df1 = quantify_motifs(df1, feature_set = feature_set, custom_motifs = custom_motifs)
        df2 = quantify_motifs(df2, feature_set = feature_set, custom_motifs = custom_motifs) if '(' in df2.iloc[
            :, 0].values.tolist()[0] else df2.set_index(df2.columns.tolist()[0])
    else:
        df1 = df1.set_index(df1.columns.tolist()[0])
        df2 = df2.set_index(df2.columns.tolist()[0])
    if transform == "ALR":
        df1 = get_additive_logratio_transformation(df1.reset_index(), df1.columns.tolist(), [], paired = False,
                                                   gamma = gamma, random_state = random_state)
        df1 = df1.set_index(df1.columns[0])
        df2 = get_additive_logratio_transformation(df2.reset_index(), df2.columns.tolist(), [], paired = False,
                                                   gamma = gamma, random_state = random_state)
        df2 = df2.set_index(df2.columns[0])
    elif transform == "CLR":
        df1 = clr_transformation(df1 + 0.0000001, df1.columns.tolist(), [], gamma = gamma, random_state = random_state)
        df2 = clr_transformation(df2 + 0.0000001, df2.columns.tolist(), [], gamma = gamma, random_state = random_state)
    df1, df2 = df1.T, df2.T
    correlation_matrix = np.zeros((df1.shape[1], df2.shape[1]))
    p_value_matrix = np.zeros((df1.shape[1], df2.shape[1]))
    if partial_correlations:
        correlations_df1 = np.abs(np.corrcoef(df1.transpose()))
        correlations_df2 = np.abs(np.corrcoef(df2.transpose()))
        threshold = 0.5 if motifs else 0.2
    # Compute Spearman correlation for each pair of columns between transformed df1 and df2
    if partial_correlations:
        max_controls = min(df1.shape[0] // 5, 5)
        controls_i = [df1.values[
                          :, [k for k in range(df1.shape[1]) if correlations_df1[i, k] > threshold and k != i][
                              :max_controls]] for i in range(df1.shape[1])]
        controls_j = [df2.values[
                          :, [k for k in range(df2.shape[1]) if correlations_df2[j, k] > threshold and k != j][
                              :max_controls]] for j in range(df2.shape[1])]
        values1, values2 = df1.values, df2.values
        for i in range(df1.shape[1]):
            for j in range(df2.shape[1]):
                corr, p_val = partial_corr(values1[:, i], values2[:, j], np.hstack([controls_i[i], controls_j[j]]),
                                           motifs = motifs)
                correlation_matrix[i, j] = corr
                p_value_matrix[i, j] = p_val
    else:
        corrs, pvals = spearmanr(df1.values, df2.values)
        # spearmanr returns bare scalars, not matrices, whenever the stacked input holds only two variables; broadcasting keeps the slicing below well-defined
        n = df1.shape[1] + df2.shape[1]
        corrs, pvals = np.broadcast_to(corrs, (n, n)), np.broadcast_to(pvals, (n, n))
        correlation_matrix, p_value_matrix = corrs[:df1.shape[1], df1.shape[1]:], pvals[
            :df1.shape[1], df1.shape[1]:]
    if motifs:
        # Each motif's correlations are one family, grouped by rarest ancestral motif family so that branches with different null proportions each get their own pi0
        dag = get_motif_dag(df1.columns.tolist())
        desc = {r: nx.descendants(dag, r) for r in dag if not dag.in_degree(r)}
        breadth = {r: len(d) for r, d in desc.items()}
        anc_roots = {}
        for r, d in desc.items():
            for g in d:
                anc_roots.setdefault(g, []).append(r)
        grouped_cells, grouped_pvals = {}, {}
        for i, m in enumerate(df1.columns):
            anc = anc_roots.get(m, []) + ([m] if m in breadth else [])
            grp = min(anc, key = lambda a: (breadth[a], a)) if anc else "rest"
            grouped_cells.setdefault(grp, []).extend((i, j) for j in range(df2.shape[1]))
            grouped_pvals.setdefault(grp, []).extend(p_value_matrix[i, :].tolist())
        corrected, _ = TST_grouped_benjamini_hochberg(grouped_cells, grouped_pvals, alpha)
        p_value_matrix = np.array([[corrected[(i, j)] for j in range(df2.shape[1])] for i in range(df1.shape[1])])
    else:
        p_value_matrix = np.reshape(correct_multiple_testing(p_value_matrix.flatten(), alpha)[0], p_value_matrix.shape)
    correlation_df = pd.DataFrame(correlation_matrix, index = df1.columns, columns = df2.columns)
    p_value_df = pd.DataFrame(p_value_matrix, index = df1.columns, columns = df2.columns)
    correlation_df.attrs.update(
        {'alpha': alpha, 'n': df1.shape[0], 'test': 'partial Spearman' if partial_correlations else 'Spearman',
         'transform': transform, 'paired': False})
    return correlation_df, p_value_df


def multi_feature_scoring(
        df: pd.DataFrame,  # Transformed dataframe with glycans in rows, abundances in columns
        group1: list[str | int],  # First group indices/names
        group2: list[str | int],  # Second group indices/names
        filepath: str = '',  # Path to save ROC plot
        random_state: int | np.random.Generator | None = None,  # optional random state for reproducibility
        dag: nx.DiGraph | None = None  # Motif containment DAG; collapses collinear parent/child motifs before selection
) -> tuple['LogisticRegression', float, list[str]]:  # (L1-regularized logistic regression model, ROC AUC score, selected features)
    "Identifies minimal glycan feature set for group classification using L1-regularized logistic regression"
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn import __version__ as _sklearn_version
    # sklearn 1.8 deprecated LogisticRegression's 'penalty' in favor of 'l1_ratio'; pick the right kwarg once
    _LR_L1 = {'l1_ratio': 1} if tuple(map(int, _sklearn_version.split('.')[:2])) >= (1, 8) else {'penalty': 'l1'}
    _LR_L2 = {'l1_ratio': 0} if 'l1_ratio' in _LR_L1 else {'penalty': 'l2'}
    if group2:
        y = [0] * len(group1) + [1] * len(group2)
    else:
        y = group1
    X = df.T
    if dag is not None:
        # A parent motif and a child it always travels with are one signal; L1 would break that tie arbitrarily, so the more specific form is kept
        redundant = {p for p, c in dag.edges() if p in X.columns and c in X.columns
                     and abs(np.corrcoef(X[p].values, X[c].values)[0, 1]) > 0.99}
        X = X.drop(columns = list(redundant))
    model = LogisticRegression(**_LR_L1, solver = 'liblinear', random_state = random_state)
    model.fit(X.values, y)
    model = SelectFromModel(model, prefit = True)
    selected_features = X.columns[model.get_support()].tolist()
    if dag is not None:
        # A parent is present wherever a selected child is, so keeping both puts one signal in the model twice
        picked = set(selected_features)
        selected_features = [m for m in selected_features if not (m in dag and nx.descendants(dag, m) & picked)]
    X_selected = X[selected_features].values
    model = LogisticRegression(**_LR_L2, solver = 'liblinear', random_state = random_state)
    model.fit(X_selected, y)
    # Evaluate ROC AUC on the selected features
    y_scores = model.predict_proba(X_selected)[:, 1]
    roc_auc = roc_auc_score(y, y_scores)
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label = f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Optimal Features')
    plt.legend(loc = "lower right")
    if filepath:
        plt.savefig(filepath, format = Path(filepath).suffix[1:], dpi = 300, bbox_inches = 'tight')
    return model, roc_auc, selected_features


def get_roc(
        df: pd.DataFrame | str | Path,  # DataFrame with glycans in rows (col 1), abundances in columns
        group1: list[str | int] | None = None,  # First group indices/names; default: from the frame's contrasts
        group2: list[str | int] | None = None,  # Second group indices/names; default: from the frame's contrasts
        motifs: bool = False,  # Analyze motifs instead of sequences
        feature_set: list[str] = ["known", "exhaustive"],
        # Feature sets to use; exhaustive, known, terminal1, terminal2, terminal3, chemical, graph, custom, size_branch
        paired: bool | None = None,  # Whether samples are paired; default: from the frame
        impute: bool = True,  # Replace zeros with Random Forest model
        min_samples: float = 0.1,  # Min percent of non-zero samples required
        custom_motifs: list[str] = [],  # Custom motifs if using 'custom' feature set
        transform: str | None = None,  # Transformation type: "CLR" or "ALR"
        gamma: float = 0.1,  # Uncertainty parameter for CLR transform
        custom_scale: float | dict = 0,
        # Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
        filepath: str | Path = '',  # Path to save ROC plot
        multi_score: bool = False,  # Find best multi-glycan score
        random_state: int | np.random.Generator | None = None  # optional random state for reproducibility
) -> list[tuple[str, float]] | dict[Any, tuple[str, float]] | tuple[
    'LogisticRegression', float, list[str]]:  # (Feature scores with ROC AUC values)
    "Calculates ROC curves and AUC scores for glycans/motifs or multi-glycan classifiers"
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import auc, roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import label_binarize
    if group1 is None and isinstance(df, GlycoDataFrame) and df._contrasts:
        group1, group2 = list(df.group1), list(df.group2)
    paired = df.paired if paired is None and isinstance(df, GlycoDataFrame) else bool(paired)
    experiment = "diff" if group2 else "anova"
    df, df_org, group1, group2 = preprocess_data(df, group1 = group1, group2 = group2, experiment = experiment,
                                                 motifs = motifs, impute = impute,
                                                 transform = transform, feature_set = feature_set, paired = paired, gamma = gamma,
                                                 custom_scale = custom_scale, custom_motifs = custom_motifs, random_state = random_state)
    if multi_score:
        return multi_feature_scoring(df, group1, group2, filepath = filepath, random_state = random_state,
                                     dag = df_org.attrs.get('motif_dag'))
    auc_scores = {}
    if group2:  # binary comparison
        for feature, values in df.iterrows():
            values_group1 = values[group1]
            values_group2 = values[group2]
            y_true = [0] * len(values_group1) + [1] * len(values_group2)
            y_scores = np.concatenate([values_group1, values_group2])
            auc_scores[feature] = roc_auc_score(y_true, y_scores)
        sorted_auc_scores = sorted(auc_scores.items(), key = lambda item: item[1], reverse = True)
        best, res = sorted_auc_scores[0]
        values = df.loc[best, :]
        values_group1 = values[group1]
        values_group2 = values[group2]
        y_true = [0] * len(values_group1) + [1] * len(values_group2)
        y_scores = np.concatenate([values_group1, values_group2])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure()
        plt.plot(fpr, tpr, label = f'ROC Curve (area = {res:.2f})')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {best}')
        plt.legend(loc = 'lower right')
        if filepath:
            plt.savefig(filepath, format = Path(filepath).suffix[1:], dpi = 300, bbox_inches = 'tight')
    else:  # multi-group comparison
        classes = list(set(group1))
        df = df.groupby(df.index).mean()
        df = df.T  # Ensure features are columns
        df['group'] = group1
        y = label_binarize(df['group'], classes = classes)
        n_classes = y.shape[1]
        sorted_auc_scores, best_fpr, best_tpr = {}, {}, {}
        # sklearn takes a seed rather than a Generator, so one is drawn from it, as MissForest does
        seed = 42 if random_state is None else random_state if isinstance(random_state, (int, np.integer)) else int(
            np.random.default_rng(random_state).integers(2 ** 32))
        for feature in df.columns[:-1]:  # exclude the 'group' label column
            X = df[feature].values.reshape(-1, 1)  # Feature matrix needs to be column-wise
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed)
            classifier = OneVsRestClassifier(LogisticRegression(solver = 'lbfgs'))
            classifier.fit(X_train, y_train)
            for i in range(n_classes):
                if len(np.unique(y_test[:, i])) < 2:  # Check if the test set is not degenerate
                    continue
                y_score = classifier.predict_proba(X_test)[:, i]
                fpr, tpr, _ = roc_curve(y_test[:, i], y_score)
                roc_auc = auc(fpr, tpr)
                # Store the best feature for each class based on AUC
                if classes[i] not in sorted_auc_scores or roc_auc > sorted_auc_scores[classes[i]][1]:
                    sorted_auc_scores[classes[i]] = (feature, roc_auc)
                    best_fpr[classes[i]] = fpr
                    best_tpr[classes[i]] = tpr
        # Plot average ROC curves for the best features
        for classy, (best_feature, best_auc) in sorted_auc_scores.items():
            plt.figure()
            plt.plot(best_fpr[classy], best_tpr[classy], label = f'ROC curve (area = {best_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Best Feature ROC for {classy}: {best_feature}')
            plt.legend(loc = "lower right")
            if filepath:
                filepath = Path(filepath)
                plt.savefig(f"{filepath.stem}_{classy}{filepath.suffix}", format = filepath.suffix[1:], dpi = 300,
                            bbox_inches = 'tight')
    plt.show()
    return sorted_auc_scores


def get_lectin_array(
        df: pd.DataFrame | str | Path,
        # DataFrame with samples as rows and lectins as columns, first column containing sample IDs
        group1: list[str | int] | None = None,
        # First group indices/names; inferred from a GlycoDataFrame's contrasts if omitted
        group2: list[str | int] | None = None,
        # Second group indices/names; inferred from a GlycoDataFrame's contrasts if omitted
        paired: bool | None = None,  # Whether samples are paired; inferred from a GlycoDataFrame if omitted
        transform: str = ''  # Optional log2 transformation
) -> pd.DataFrame:  # DataFrame with altered glycan motifs, supporting lectins, and effect sizes
    "Analyzes lectin microarray data by mapping lectin binding patterns to glycan motifs, calculating Cohen's d effect sizes between groups and clustering results by significance"
    from sklearn.cluster import KMeans
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df) if Path(df).suffix.lower() == ".csv" else pd.read_csv(df, sep = "\t") if Path(
            df).suffix.lower() == ".tsv" else pd.read_excel(df)
    in_name, in_prov, contrasts = getattr(df, '_glyco_name', ''), getattr(df, '_provenance', {}), getattr(df,
                                                                                                          '_contrasts',
                                                                                                          {})
    df = df.set_index(df.columns[0])
    if group1 is None and contrasts:
        # A lectin array has its samples in the rows, so the contrasts are read off the index rather than off the columns
        names = list(dict.fromkeys(contrasts.values()))
        group1 = [s for s in df.index if contrasts.get(s) == names[0]]
        group2 = [s for s in df.index if len(names) > 1 and contrasts.get(s) == names[1]]
    if paired is None:
        paired = getattr(df, '_paired', False)
    if not group1:
        raise ValueError(
            "No groups given: pass group1 (and group2) as lists of sample names or indices; groups are only inferred automatically from a GlycoDataFrame that carries contrasts.")
    alpha = get_alphaN(df.shape[0])
    duplicated_cols = set(df.columns[df.columns.duplicated()])
    if duplicated_cols:
        raise ValueError(
            f'Analysis aborted due to:\nDuplicates found for the following lectin(s): {", ".join(duplicated_cols)}.\nIf you have multiple copies of the same lectin, rename them by adding a suffix in the form of "_<identifier>" (underscore + an identifier).\nFor example, "SNA" may be renamed "SNA_1", "SNA_batch1", etc. ')
    lectin_list = df.columns.tolist()
    df = np.log2(df) if transform == "log2" else df
    df = df.T
    if not isinstance(group1[0], str):
        if group1[0] == 1 or (group2 and group2[0] == 1):
            group1 = [k - 1 for k in group1]
            group2 = [k - 1 for k in group2]
        columns_list = df.columns.tolist()
        group1 = [columns_list[k] for k in group1]
        group2 = [columns_list[k] for k in group2]
    df = replace_outliers_winsorization(df)
    lectin_lib = load_lectin_lib()
    useable_lectin_mapping, motif_mapping = create_lectin_and_motif_mappings(lectin_list, lectin_lib)
    if group2:
        mean_scores_per_condition = df[group1 + group2].T.groupby([0] * len(group1) + [1] * len(group2)).mean().T
    else:
        mean_scores_per_condition = df.T.groupby(group1).mean().T
    lectin_variance = mean_scores_per_condition.var(axis = 1)
    idf = np.sqrt(lectin_variance)
    if group2:
        df_a, df_b = df[group1], df[group2]
        effect_sizes = cohen_d(df_b.values, df_a.values, paired = paired)[0] if len(df) else [0] * len(df)
    else:
        effect_sizes = omega_squared(df, group1)
    lectin_score_dict = {lec: effect_sizes[i] if group2 else effect_sizes.iloc[i] for i, lec in enumerate(lectin_list)}
    df_out = lectin_motif_scoring(useable_lectin_mapping, motif_mapping, lectin_score_dict, lectin_lib, idf)
    df_out = df_out.sort_values(by = "score", ascending = False)
    scores = df_out['score'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters = 3, random_state = 0, n_init = 'auto').fit(scores)
    df_out['significance'] = kmeans.labels_
    centroids = kmeans.cluster_centers_.flatten()
    sorted_centroid_indices = centroids.argsort()
    significance_mapping = {sorted_centroid_indices[0]: 'low significance',
                            sorted_centroid_indices[1]: 'moderately significant',
                            sorted_centroid_indices[2]: 'highly significant'}
    df_out['significance'] = df_out['significance'].apply(lambda x: significance_mapping[x])
    temp = annotate_dataset(df_out.iloc[:, 0], condense = True)
    occurring_motifs = [temp.columns[temp.iloc[idx].astype(bool)].tolist() for idx in range(len(temp))]
    df_out.insert(1, "named_motifs", occurring_motifs)
    if not group2:
        df_out["change"] = ["different"] * len(df_out)
    df_out.attrs.update(
        {'alpha': alpha, 'n': len(group1) + (len(group2) if group2 else 0),
         'test': "Cohen's d" if group2 else 'omega squared',
         'transform': None, 'paired': paired, 'dataset': in_name, 'provenance': in_prov})
    return df_out


def get_glycoshift_per_site(
        df: pd.DataFrame | str | Path,
        # DataFrame with rows formatted as 'protein_site_composition' in col 1, abundances in remaining cols
        group1: list[str | int] | None = None,
        # First group indices/names or group labels for multi-group; default: from the frame's contrasts
        group2: list[str | int] | None = None,  # Second group indices/names; default: from the frame's contrasts
        paired: bool | None = None,  # Whether samples are paired; default: from the frame
        impute: bool = True,  # Replace zeros with Random Forest model
        min_samples: float = 0.2,  # Min percent of non-zero samples required
        gamma: float = 0.1,  # Uncertainty parameter for CLR transform
        custom_scale: float | dict = 0,
        # Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
        random_state: int | np.random.Generator | None = None  # optional random state for reproducibility
) -> pd.DataFrame:  # DataFrame with GLM coefficients and FDR-corrected p-values
    "Analyzes site-specific glycosylation changes in glycoproteomics data using generalized linear models (GLM) with compositional data normalization"
    paired = df.paired if paired is None and isinstance(df, GlycoDataFrame) else bool(paired)
    df, _, group1, group2 = preprocess_data(df, group1 = group1, group2 = group2, experiment = "diff", motifs = False, impute = impute,
                                            min_samples = min_samples, transform = "Nothing", paired = paired,
                                            random_state = random_state)
    alpha = get_alphaN(len(group1 + group2))
    df, glycan_features = process_for_glycoshift(df)
    necessary_columns = ['Glycoform'] + glycan_features
    preserved_data = df[necessary_columns]
    df = df.drop(necessary_columns, axis = 1)
    df = df.set_index('Glycosite')
    df = df.div(df.sum(axis = 0), axis = 1) * 100
    df = df.reset_index()
    results = [
        clr_transformation(group_df[group1 + group2], group1, group2, gamma = gamma, custom_scale = custom_scale, random_state = random_state)
        .assign(Glycosite = glycosite) for glycosite, group_df in df.groupby('Glycosite')
    ]
    df = pd.concat(results).sort_index()
    df = pd.concat([df, preserved_data.reset_index(drop = True)], axis = 1)
    df_long = pd.melt(df, id_vars = ['Glycosite', 'Glycoform'] + glycan_features, var_name = 'Sample',
                      value_name = 'Abundance')
    df_long['Condition'] = (~df_long['Sample'].isin(set(group1))).astype(int)
    return process_glm_results(df_long, alpha, glycan_features)
