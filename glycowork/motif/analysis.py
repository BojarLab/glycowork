import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('default')
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.stats import ttest_ind, ttest_rel, norm, levene, f_oneway, spearmanr
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from glycowork.glycan_data.loader import df_species, unwrap, strip_suffixes, download_model
from glycowork.glycan_data.stats import (cohen_d, mahalanobis_distance, mahalanobis_variance,
                                         impute_and_normalize, variance_based_filtering,
                                         jtkdist, jtkinit, MissForest, jtkx, get_alphaN, TST_grouped_benjamini_hochberg,
                                         compare_inter_vs_intra_group, replace_outliers_winsorization, hotellings_t2,
                                         sequence_richness, shannon_diversity_index, simpson_diversity_index,
                                         get_equivalence_test, clr_transformation, anosim, permanova_with_permutation,
                                         alpha_biodiversity_stats, get_additive_logratio_transformation, correct_multiple_testing,
                                         omega_squared, get_glycoform_diff, process_glm_results, partial_corr, estimate_technical_variance,
                                         perform_tests_monte_carlo)
from glycowork.motif.processing import enforce_class, process_for_glycoshift
from glycowork.motif.annotate import (annotate_dataset, quantify_motifs, link_find, create_correlation_network,
                                      group_glycans_core, group_glycans_sia_fuc, group_glycans_N_glycan_type, load_lectin_lib,
                                      create_lectin_and_motif_mappings, lectin_motif_scoring, clean_up_heatmap)
from glycowork.motif.graph import subgraph_isomorphism


def preprocess_data(
    df: Union[pd.DataFrame, str], # Input dataframe or filepath (.csv/.xlsx)
    group1: List[Union[str, int]], # Column indices/names for first group
    group2: List[Union[str, int]], # Column indices/names for second group
    experiment: str = "diff", # Type of experiment: "diff" or "anova"
    motifs: bool = False, # Analyze motifs instead of sequences
    feature_set: List[str] = ['exhaustive', 'known'], # Feature sets to use; exhaustive/known/terminal1/terminal2/terminal3/chemical/graph/custom
    paired: bool = False, # Whether samples are paired
    impute: bool = True, # Replace zeros with Random Forest model
    min_samples: float = 0.1, # Min percent of non-zero samples required
    transform: Optional[str] = None, # Transformation type: "CLR" or "ALR"
    gamma: float = 0.1, # Uncertainty parameter for CLR transform
    custom_scale: Union[float, Dict] = 0, # Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
    custom_motifs: List[str] = [], # Custom motifs if using 'custom' feature set
    monte_carlo: bool = False # Use Monte Carlo for technical variation
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Union[str, int]], List[Union[str, int]]]: # (transformed df, untransformed df, group1 labels, group2 labels)
  "Preprocesses glycomics data by handling missing values with Random Forest imputation, applying CLR/ALR transformations to escape compositional bias, and optionally quantifying glycan motifs"
  if isinstance(df, str):
    df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
  if not isinstance(group1[0], str) and experiment == "diff":
    columns_list = df.columns.tolist()
    group1 = [columns_list[k] for k in group1]
    group2 = [columns_list[k] for k in group2]
  df = df.iloc[:, :len(group1)+1].fillna(0) if experiment == "anova" else df.loc[:, [df.columns[0]]+group1+group2].fillna(0)
  # Drop rows with all zero, followed by outlier removal and imputation & normalization
  df = df.loc[~(df.iloc[:, 1:] == 0).all(axis = 1)]
  df = df.apply(replace_outliers_winsorization, axis = 1)
  if experiment == "diff":
    df = impute_and_normalize(df, [group1, group2], impute = impute, min_samples = min_samples)
  elif experiment == "anova":
    groups_unq = sorted(set(group1))
    df = impute_and_normalize(df, [[df.columns[i+1] for i, x in enumerate(group1) if x == g] for g in groups_unq], impute = impute,
                              min_samples = min_samples)
  df_org = df.copy(deep = True)
  if transform is None:
    transform = "ALR" if enforce_class(df.iloc[0, 0], "N") and len(df) > 50 else "CLR"
  if transform == "ALR":
    df = get_additive_logratio_transformation(df, group1 if experiment == "diff" else df.columns[1:], group2, paired = paired, gamma = gamma, custom_scale = custom_scale)
  elif transform == "CLR":
    if monte_carlo and not motifs:
      df = pd.concat([df.iloc[:, 0], estimate_technical_variance(df.iloc[:, 1:], group1, group2,
                                                                 gamma = gamma, custom_scale = custom_scale)], axis = 1)
    else:
      df.iloc[:, 1:] = clr_transformation(df.iloc[:, 1:], group1 if experiment == "diff" else df.columns[1:], group2, gamma = gamma, custom_scale = custom_scale)
  elif transform == "Nothing":
    pass
  else:
    raise ValueError("Only ALR and CLR are valid transforms for now.")
  if motifs:
    # Motif extraction and quantification
    df = quantify_motifs(df.iloc[:, 1:], df.iloc[:, 0].values.tolist(), feature_set, custom_motifs = custom_motifs)
    df_org = quantify_motifs(df_org.iloc[:, 1:], df_org.iloc[:, 0].values.tolist(), feature_set, custom_motifs = custom_motifs)
    # Re-normalization
    df_org = df_org.apply(lambda col: col / col.sum() * 100, axis = 0)
  else:
    df = df.set_index(df.columns.tolist()[0])
    df = df.groupby(df.index).mean()
    df_org = df_org.set_index(df_org.columns.tolist()[0])
    df_org = df_org.groupby(df_org.index).mean()
  return df, df_org, group1, group2


def get_pvals_motifs(
    df: Union[pd.DataFrame, str], # Input dataframe or filepath (.csv/.xlsx)
    glycan_col_name: str = 'glycan', # Column name for glycan sequences
    label_col_name: str = 'target', # Column name for labels
    zscores: bool = True, # Whether data are z-scores
    thresh: float = 1.645, # Threshold to separate positive/negative
    sorting: bool = True, # Sort p-value dataframe
    feature_set: List[str] = ['exhaustive'], # Feature sets to use; exhaustive/known/terminal1/terminal2/terminal3/chemical/graph/custom
    multiple_samples: bool = False, # Multiple samples with glycan columns
    motifs: Optional[pd.DataFrame] = None, # Modified motif_list
    custom_motifs: List[str] = [] # Custom motifs if using 'custom' feature set
    ) -> pd.DataFrame: # DataFrame with p-values, FDR-corrected p-values, and Cohen's d effect sizes for glycan motifs
    "Identifies significantly enriched glycan motifs using Welch's t-test with FDR correction and Cohen's d effect size calculation, comparing samples above/below threshold"
    if isinstance(df, str):
      df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
    # Reformat to allow for proper annotation in all samples
    if multiple_samples:
      df = df.drop('target', axis = 1, errors = 'ignore').T.reset_index()
      df.columns = [glycan_col_name] + [label_col_name] * (len(df.columns) - 1)
    if not zscores:
      means = df.iloc[:, 1:].mean()
      std_devs = df.iloc[:, 1:].std()
      df.iloc[:, 1:] = (df.iloc[:, 1:] - means) / std_devs
    # Annotate glycan motifs in dataset
    df_motif = annotate_dataset(df[glycan_col_name].values.tolist(),
                                motifs = motifs, feature_set = feature_set, condense = True,
                                custom_motifs = custom_motifs)
    # Broadcast the dataframe to the correct size given the number of samples
    if multiple_samples:
      df = df.set_index(glycan_col_name)
      df_motif = pd.concat([pd.concat([df.iloc[:, k],
                                   df_motif], axis = 1).dropna() for k in range(len(df.columns))], axis = 0)
      cols = df_motif.columns.values.tolist()[1:] + [df_motif.columns.values.tolist()[0]]
      df_motif = df_motif[cols]
    else:
      df_motif[label_col_name] = df[label_col_name].values.tolist()
    # Divide into motifs with expression above threshold & below
    df_pos = df_motif[df_motif[label_col_name] > thresh]
    df_neg = df_motif[df_motif[label_col_name] <= thresh]
    # Test statistical enrichment for motifs in above vs below
    ttests = [ttest_ind(np.append(df_pos.iloc[:, k] * df_pos[label_col_name], [1]),
                        np.append(df_neg.iloc[:, k] * df_neg[label_col_name], [1]),
                        equal_var = False)[1] for k in range(df_motif.shape[1]-1)]
    ttests_corr = multipletests(ttests, method = 'fdr_tsbh')[1].tolist()
    effect_sizes, _ = zip(*[cohen_d(np.append(df_pos.iloc[:, k].values, [1, 0]),
                                            np.append(df_neg.iloc[:, k].values, [1, 0]), paired = False) for k in range(df_motif.shape[1]-1)])
    out = pd.DataFrame({
        'motif': df_motif.columns.tolist()[:-1],
        'pval': ttests,
        'corr_pval': ttests_corr,
        'effect_size': effect_sizes
        })
    if sorting:
      out['abs_effect_size'] = out['effect_size'].abs()
      out = out.sort_values(by = ['abs_effect_size', 'corr_pval', 'pval'], ascending = [False, True, True])
      out = out.drop('abs_effect_size', axis = 1)
    return out


def get_representative_substructures(
    enrichment_df: pd.DataFrame # Output from get_pvals_motifs
    ) -> List[str]: # Up to 10 minimal glycans containing enriched motifs
    "Constructs minimal glycan structures that represent significantly enriched motifs by optimizing for motif content while minimizing structure size using subgraph isomorphism"
    glycans = list(set(df_species.glycan))
    # Only consider motifs that are significantly enriched
    filtered_df = enrichment_df[enrichment_df.corr_pval < 0.05].reset_index(drop = True)
    log_pvals = -np.log10(filtered_df.pval.values)
    max_log_pval = np.max(log_pvals)
    weights = log_pvals / max_log_pval
    motifs = filtered_df.motif.values.tolist()
    # Pair glycoletters & disaccharides with their pvalue-based weight
    mono, mono_weights = list(zip(*[(motifs[k], weights[k]) for k in range(len(motifs)) if '(' not in motifs[k]]))
    di, di_weights = list(zip(*[(motifs[k], weights[k]) for k in range(len(motifs)) if '(' in motifs[k]]))
    mono_scores = [sum([mono_weights[j] for j in range(len(mono)) if mono[j] in k]) for k in glycans]
    di_scores = [sum([di_weights[j] for j in range(len(di)) if subgraph_isomorphism(k, di[j])]) for k in glycans]
    # For each glycan, get their glycoletter & disaccharide scores, normalized by glycan length
    motif_scores = np.add(mono_scores, di_scores)
    length_scores = [len(g) for g in glycans]
    combined_scores = np.divide(motif_scores, length_scores)
    df_score = pd.DataFrame({'glycan': glycans, 'motif_score': motif_scores, 'length_score': length_scores, 'combined_score': combined_scores})
    df_score = df_score.sort_values(by = 'combined_score', ascending = False)
    # Take the 10 glycans with the highest score
    rep_motifs = df_score.glycan.values.tolist()[:10]
    rep_motifs.sort(key = len)
    # Make sure that the list only contains the minimum number of representative glycans
    return [k for k in rep_motifs if sum(subgraph_isomorphism(j, k) for j in rep_motifs) <= 1]


def get_heatmap(
    df: Union[pd.DataFrame, str], # Input dataframe or filepath (.csv/.xlsx)
    motifs: bool = False, # Analyze motifs instead of sequences
    feature_set: List[str] = ['known'], # Feature sets to use; exhaustive/known/terminal1/terminal2/terminal3/chemical/graph/custom
    transform: str = '', # Transform data before plotting
    datatype: str = 'response', # Data type: 'response' or 'presence'
    rarity_filter: float = 0.05, # Min proportion for non-zero values
    filepath: str = '', # Path to save plot
    index_col: str = 'glycan', # Column to use as index
    custom_motifs: List[str] = [], # Custom motifs if using 'custom' feature set
    return_plot: bool = False, # Return plot object
    show_all: bool = False, # Show all tick labels
    **kwargs: Any # Keyword args passed to seaborn clustermap
    ) -> Optional[Any]: # None or plot object if return_plot=True
  "Creates hierarchically clustered heatmap visualization of glycan/motif abundances"
  if isinstance(df, str):
    df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
  if index_col in df.columns:
    df = df.set_index(index_col)
  elif isinstance(df.iloc[0,0], str):
    df = df.set_index(df.columns.tolist()[0])
  if not isinstance(df.index[0], str) or (isinstance(df.index[0], str) and ('(' not in df.index[0] or '-' not in df.index[0])):
    df = df.T
  df = df.fillna(0)
  if transform == "CLR":
    df = df.replace(0, np.nan).dropna(thresh = np.max([np.round(rarity_filter * df.shape[0]), 1]), axis = 1).fillna(1e-6)
    df = clr_transformation(df, [], [], gamma = 0)
  elif transform == "ALR":
    df = df.replace(0, np.nan).dropna(thresh = np.max([np.round(rarity_filter * df.shape[0]), 1]), axis = 1).fillna(1e-6)
    df = get_additive_logratio_transformation(df.reset_index(), df.columns.tolist(), [], paired = False, gamma = 0)
    df = df.set_index(df.columns[0])
  if motifs:
    if 'custom' in feature_set and len(feature_set) == 1 and len(custom_motifs) < 2:
      raise ValueError("A heatmap needs to have at least two motifs.")
    if datatype == 'response':
      df = quantify_motifs(df, df.index.tolist(), feature_set, custom_motifs = custom_motifs)
    elif datatype == 'presence':
      # Count glycan motifs and remove rare motifs from the result
      df_motif = annotate_dataset(df.index.tolist(), feature_set = feature_set, condense = True, custom_motifs = custom_motifs)
      df_motif = df_motif.replace(0, np.nan).dropna(thresh = np.max([np.round(rarity_filter * df_motif.shape[0]), 1]), axis = 1)
      df = df_motif.T.fillna(0) @ df
      df = df.apply(lambda col: col / col.sum()).T
      df = clean_up_heatmap(df.T)
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
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300,
                bbox_inches = 'tight')
    plt.close(g.fig)
  elif return_plot:
    return g
  else:
    plt.show()


def plot_embeddings(
   glycans: List[str], # List of IUPAC-condensed glycan sequences
   emb: Optional[Union[Dict[str, np.ndarray], pd.DataFrame]] = None, # Glycan embeddings dict/DataFrame; defaults to SweetNet embeddings
   label_list: Optional[List[Any]] = None, # Labels for coloring points
   shape_feature: Optional[str] = None, # Monosaccharide/bond for point shapes
   filepath: str = '', # Path to save plot
   alpha: float = 0.8, # Point transparency
   palette: str = 'colorblind', # Color palette for groups
   **kwargs: Any # Keyword args passed to seaborn scatterplot
   ) -> None:
    "Visualizes learned glycan embeddings using t-SNE dimensionality reduction with optional group coloring"
    idx = [i for i, g in enumerate(glycans) if '{' not in g]
    glycans = [glycans[i] for i in idx]
    if label_list is not None:
      label_list = [label_list[i] for i in idx]
    # Get all glycan embeddings
    if emb is None:
      if not os.path.exists('glycan_representations_v1_4.pkl'):
          download_model("https://drive.google.com/file/d/1--tf0kyea9jFLfffUtICKkyIw36E9hJ3/view?usp=sharing", local_path = 'glycan_representations_v1_4.pkl')
      emb = pickle.load(open('glycan_representations_v1_4.pkl', 'rb'))
    # Get the subset of embeddings corresponding to 'glycans'
    embs = emb.values if isinstance(emb, pd.DataFrame) else np.vstack([emb[g] for g in glycans])
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
      plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300,
                  bbox_inches = 'tight')
    plt.show()


def characterize_monosaccharide(
    sugar: str, # Monosaccharide or linkage to analyze
    df: Optional[pd.DataFrame] = None, # DataFrame with glycan column 'glycan'; defaults to df_species
    mode: str = 'sugar', # Analysis mode: 'sugar', 'bond', 'sugarbond'
    glycan_col_name: str = 'glycan', # Column name for glycan sequences
    rank: Optional[str] = None, # Column name for group filtering
    focus: Optional[str] = None, # Row value for group filtering
    modifications: bool = False, # Consider modified monosaccharides
    filepath: str = '', # Path to save plot
    thresh: int = 10 # Minimum count threshold for inclusion
    ) -> None:
  "Analyzes connectivity and modification patterns of specified monosaccharides/linkages in glycan sequences"
  if df is None:
    df = df_species
  if rank is not None and focus is not None:
    df = df[df[rank] == focus]
  # Get all disaccharides for linkage analysis
  pool_in = unwrap([link_find(k) for k in df[glycan_col_name]])
  pool_in = [k.replace('(', '*').replace(')', '*') for k in pool_in]
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
  cou_k, cou_v = zip(*filtered_items)
  cou_v = [v / len(pool) for v in cou_v]
  # Start plotting
  fig, (a0, a1) = plt.subplots(1, 2, figsize = (8, 4), gridspec_kw = {'width_ratios': [1, 1]})
  if modifications:
    if mode == 'bond':
      print("Modifications currently only work in mode == 'sugar' and mode == 'sugarbond'.")
    # Get counts and proportions for the input monosaccharide + its modifications
    cou2 = Counter(sugars).most_common()
    filtered_items = [(item, count) for item, count in cou2 if count > thresh]
    cou_k2, cou_v2 = zip(*filtered_items)
    cou_v2 = [v / len(sugars) for v in cou_v2]
    # Map the input monosaccharide + its modifications to colors
    color_list = plt.get_cmap('tab20')
    color_map = {key: color_list(idx / len(cou_k2)) for idx, key in enumerate(cou_k2)}
    palette = [color_map[k] for k in cou_k2]
    # Start linking downstream monosaccharides / linkages to the input monosaccharide + its modifications
    pool_in2 = [k for k in pool_in if k.split('*')[2 if mode == 'sugar' else 1] in cou_k]
    cou_for_df = []
    for key in cou_k2:
      idx = [item.split('*')[2 if mode == 'sugar' else 1] for item in pool_in2 if item.split('*')[0] == key]
      cou_t = Counter(idx).most_common()
      cou_v_t = {item[0]: item[1] for item in cou_t}
      cou_v_t = [cou_v_t.get(item, 0) for item in cou_k]
      if len(cou_k2) > 1:
        cou_for_df.append(pd.DataFrame({'monosaccharides': cou_k, 'counts': cou_v_t, 'colors': [key] * len(cou_k)}))
      else:
        sns.barplot(x = cou_k, y = cou_v_t, ax = a1, color = "cornflowerblue")
    if len(cou_k2) > 1:
      cou_df = pd.concat(cou_for_df).reset_index(drop = True)
      sns.histplot(data = cou_df, x = 'monosaccharides', hue = 'colors', weights = 'counts',
                       multiple = 'stack', palette = palette, ax = a1, legend = False, shrink = 0.8)
    a1.set_ylabel('Absolute Occurrence')
  else:
    sns.barplot(x = cou_k, y = cou_v, ax = a1, color = "cornflowerblue")
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
                     hue = 'monosaccharides', shrink = 0.8, legend = False,
                     ax = a0, palette = palette, alpha = 0.75)
    else:
      sns.barplot(x = cou_k2, y = cou_v2, ax = a0, color = "cornflowerblue")
    sns.despine(left = True, bottom = True)
    a0.set_ylabel('Relative Proportion')
    a0.set_xlabel('')
    a0.set_title(f'Observed Modifications of {sugar}')
    plt.setp(a1.get_xticklabels(), rotation = 'vertical')
  fig.suptitle(f'Characterizing {sugar}')
  fig.tight_layout()
  if filepath:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300,
                  bbox_inches = 'tight')
  plt.show()


def get_coverage(
    df: Union[pd.DataFrame, str], # DataFrame with glycans in rows (col 1), abundances in columns
    filepath: str = '' # Path to save plot
    ) -> None:
  "Visualizes glycan detection frequency across samples with intensity-based ordering"
  if isinstance(df, str):
    df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
  d = df.iloc[:,1:]
  # arrange by mean intensity across all samples
  order = d.mean(axis = 1).sort_values().index
  # plot figure
  ax = sns.heatmap(d.loc[order], cmap = sns.cubehelix_palette(as_cmap=True, start = 3, dark = 0.50, light = 0.90),
                   cbar_kws = {'label': 'Relative Intensity'}, cbar = True, mask = d.loc[order] == 0)
  ax.set(xlabel = 'Samples', ylabel =  'Glycan ID', title = '')
  # save figure
  if filepath:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300, bbox_inches = 'tight')
  plt.show()


def get_pca(
    df: Union[pd.DataFrame, str], # DataFrame with glycans in rows (col 1), abundances in columns
    groups: Optional[Union[List[int], pd.DataFrame]] = None, # Group labels (e.g., [1,1,1,2,2,2,3,3,3]) or metadata DataFrame with 'id' column
    motifs: bool = False, # Analyze motifs instead of sequences
    feature_set: List[str] = ['known', 'exhaustive'], # Feature sets to use; exhaustive/known/terminal1/terminal2/terminal3/chemical/graph/custom
    pc_x: int = 1, # Principal component for x-axis
    pc_y: int = 2, # Principal component for y-axis
    color: Optional[str] = None, # Column in metadata for color grouping
    shape: Optional[str] = None, # Column in metadata for shape grouping
    filepath: str = '', # Path to save plot
    custom_motifs: List[str] = [], # Custom motifs if using 'custom' feature set
    transform: Optional[str] = None, # Transformation type: "CLR" or "ALR"
    rarity_filter: float = 0.05 # Min proportion for non-zero values
    ) -> None:
  "Performs PCA on glycan/motif abundance data with group-based visualization"
  if isinstance(df, str):
    df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
  if transform == "ALR":
    df = df.replace(0, np.nan).dropna(thresh = np.max([np.round(rarity_filter * df.shape[0]), 1]), axis = 1).fillna(1e-6)
    df = get_additive_logratio_transformation(df, df.columns.tolist()[1:], [], paired = False, gamma = 0)
  elif transform == "CLR":
    df = df.replace(0, np.nan).dropna(thresh = np.max([np.round(rarity_filter * df.shape[0]), 1]), axis = 1).fillna(1e-6)
    df.iloc[:, 1:] = clr_transformation(df.iloc[:, 1:], [], [], gamma = 0)
  # get pca
  if motifs:
    # Motif extraction and quantification
    df = quantify_motifs(df.iloc[:, 1:], df.iloc[:, 0].values.tolist(), feature_set, custom_motifs = custom_motifs, remove_redundant = False).T.reset_index()
  X = np.array(df.iloc[:, 1:len(groups)+1].T) if isinstance(groups, list) and groups else np.array(df.iloc[:, 1:].T)
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
  ax = sns.scatterplot(x = pc_x-1, y = pc_y-1, data = df_pca, hue = color, style = shape)
  if color or shape:
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left', borderaxespad = 0)
  ax.set(xlabel = f'PC{pc_x}: {percent_var[pc_x - 1]}% variance',
           ylabel = f'PC{pc_y}: {percent_var[pc_y - 1]}% variance')
  # save to file
  if filepath:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300, bbox_inches = 'tight')
  plt.show()


def select_grouping(
    cohort_b: pd.DataFrame, # Case samples dataframe
    cohort_a: pd.DataFrame, # Control samples dataframe
    glycans: List[str], # List of glycans in IUPAC-condensed nomenclature
    p_values: List[float], # Associated p-values from statistical tests
    paired: bool = False, # Whether samples are paired
    grouped_BH: bool = False # Use two-stage adaptive Benjamini-Hochberg
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[float]]]: # (group:glycans dict, group:p-values dict)
  "Evaluates optimal glycan grouping strategies (by core type, Sia/Fuc content, or N-glycan type) based on intraclass correlation coefficient, enabling group-aware multiple testing correction"
  if not grouped_BH:
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
  if (intra > 3*inter) or (intra > inter and intra > .1):
    print("Chosen grouping: " + desc)
    print("ICC of grouping: " + str(intra))
    print("Inter-group correlation of grouping: " + str(inter))
    return grouped_glycans, grouped_p_values
  return {"group1": glycans}, {"group1": p_values}


def get_differential_expression(
    df: Union[pd.DataFrame, str], # DataFrame with glycans in rows (col 1) and abundance values in subsequent columns
    group1: List[Union[str, int]], # Column indices/names for first group
    group2: List[Union[str, int]], # Column indices/names for second group
    motifs: bool = False, # Analyze motifs instead of sequences
    feature_set: List[str] = ['exhaustive', 'known'], # Feature sets to use; exhaustive/known/terminal1/terminal2/terminal3/chemical/graph/custom
    paired: bool = False, # Whether samples are paired
    impute: bool = True, # Replace zeros with Random Forest model
    sets: bool = False, # Identify clusters of correlated glycans
    set_thresh: float = 0.9, # Correlation threshold for clusters
    effect_size_variance: bool = False, # Calculate effect size variance
    min_samples: float = 0.1, # Min percent of non-zero samples required
    grouped_BH: bool = False, # Use two-stage adaptive Benjamini-Hochberg
    custom_motifs: List[str] = [], # Custom motifs if using 'custom' feature set
    transform: Optional[str] = None, # Transformation type: "CLR" or "ALR"
    gamma: float = 0.1, # Uncertainty parameter for CLR transform
    custom_scale: Union[float, Dict] = 0, # Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
    glycoproteomics: bool = False, # Whether data is from glycoproteomics
    level: str = 'peptide', # Analysis level for glycoproteomics
    monte_carlo: bool = False # Use Monte Carlo for technical variation
    ) -> pd.DataFrame: # DataFrame with log2FC, p-values, FDR-corrected p-values, and Cohen's d/Mahalanobis distance effect sizes
  "Performs differential expression analysis using Welch's t-test (or Hotelling's T2 for sets) with multiple testing correction on glycomics abundance data"
  df, df_org, group1, group2 = preprocess_data(df, group1, group2, experiment = "diff", motifs = motifs, impute = impute,
                                               min_samples = min_samples, transform = transform, feature_set = feature_set,
                                               paired = paired, gamma = gamma, custom_scale = custom_scale, custom_motifs = custom_motifs,
                                               monte_carlo = monte_carlo)
  # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
  alpha = get_alphaN(len(group1+group2))
  # Variance-based filtering of features
  if not monte_carlo:
    df, df_prison = variance_based_filtering(df)
    df_org, df_org_prison = df_org.loc[df.index], df_org.loc[df_prison.index]
  else:
    df_prison, df_org_prison = pd.DataFrame(), pd.DataFrame()
  glycans = df.index.tolist()
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
        log2fc.append(((gp2.values - gp1.values).mean(axis = 1)).mean() if paired else (gp2.mean(axis = 1) - gp1.mean(axis = 1)).mean())
        # Hotelling's T^2 test for multivariate comparisons
        pvals.append(hotellings_t2(gp1.T.values, gp2.T.values, paired = paired)[1])
        levene_pvals.append(np.mean([levene(gp1.loc[variable, :], gp2.loc[variable, :])[1] for variable in cluster]))
        # Calculate Mahalanobis distance as measure of effect size for multivariate comparisons
        effect_sizes.append(mahalanobis_distance(gp1, gp2, paired = paired))
        if effect_size_variance:
          variances.append(mahalanobis_variance(gp1, gp2, paired = paired))
    mean_abundance = mean_abundance_c
  else:
    log2fc = (df_b.values - df_a.values).mean(axis = 1) if paired else (df_b.mean(axis = 1) - df_a.mean(axis = 1))
    if paired:
      assert len(group1) == len(group2), "For paired samples, the size of group1 and group2 should be the same"
    if monte_carlo:
      pvals, corrpvals, effect_sizes = perform_tests_monte_carlo(df_a, df_b, paired = paired)
      significance = [cp < alpha for cp in corrpvals]
      equivalence_pvals = [1.0]*len(pvals)
      levene_pvals = [1.0]*len(pvals)
    else:
      pvals = [ttest_rel(row_b, row_a)[1] if paired else ttest_ind(row_b, row_a, equal_var = False)[1] for row_a, row_b in zip(df_a.values, df_b.values)]
      equivalence_pvals = np.array([get_equivalence_test(row_a, row_b, paired = paired) if pvals[i] > 0.05 else np.nan for i, (row_a, row_b) in enumerate(zip(df_a.values, df_b.values))])
      valid_equivalence_pvals = equivalence_pvals[~np.isnan(equivalence_pvals)]
      corrected_equivalence_pvals = multipletests(valid_equivalence_pvals, method = 'fdr_tsbh')[1] if len(valid_equivalence_pvals) else []
      equivalence_pvals[~np.isnan(equivalence_pvals)] = corrected_equivalence_pvals
      equivalence_pvals[np.isnan(equivalence_pvals)] = 1.0
      levene_pvals = [levene(row_b, row_a)[1] for row_a, row_b in zip(df_a.values, df_b.values)] if (df_a.shape[1] > 2 and df_b.shape[1] > 2) else [1.0]*len(df_a)
      effects = [cohen_d(row_b, row_a, paired = paired) for row_a, row_b in zip(df_a.values, df_b.values)]
      effect_sizes, variances = list(zip(*effects)) if effects else [[0]*len(glycans), [0]*len(glycans)]
  # Multiple testing correction
  if not monte_carlo and pvals:
    if not motifs and grouped_BH:
      grouped_glycans, grouped_pvals = select_grouping(df_b, df_a, glycans, pvals, paired = paired, grouped_BH = grouped_BH)
      corrpvals, significance_dict = TST_grouped_benjamini_hochberg(grouped_glycans, grouped_pvals, alpha)
      corrpvals = [corrpvals[g] for g in glycans]
      corrpvals = [p if p >= pvals[i] else pvals[i] for i, p in enumerate(corrpvals)]
      significance = [significance_dict[g] for g in glycans]
    else:
      corrpvals, significance = correct_multiple_testing(pvals, alpha)
    levene_pvals = multipletests(levene_pvals, method = 'fdr_tsbh')[1]
  elif monte_carlo:
    pass
  else:
    corrpvals, significance = [1]*len(glycans), [False]*len(glycans)
  df_out = pd.DataFrame(list(zip(glycans, mean_abundance, log2fc, pvals, corrpvals, significance, levene_pvals, effect_sizes, equivalence_pvals)),
                     columns = ['Glycan', 'Mean abundance', 'Log2FC', 'p-val', 'corr p-val', 'significant', 'corr Levene p-val', 'Effect size', 'Equivalence p-val'])
  if not monte_carlo:
    prison_rows = pd.DataFrame({
        'Glycan': df_prison.index,
        'Mean abundance': df_org_prison.mean(axis = 1),
        'Log2FC': (df_prison[group2].values - df_prison[group1].values).mean(axis = 1) if paired else (df_prison[group2].mean(axis = 1) - df_prison[group1].mean(axis = 1)),
        'p-val': [1.0] * len(df_prison),
        'corr p-val': [1.0] * len(df_prison),
        'significant': [False] * len(df_prison),
        'corr Levene p-val': [1.0] * len(df_prison),
        'Effect size': [0] * len(df_prison),
        'Equivalence p-val': [1.0] * len(df_prison)})
    prison_rows = prison_rows.astype({'significant': 'bool'})
    if len(prison_rows) > 0:
      df_out = pd.concat([df_out, prison_rows], ignore_index = True)
  df_out['significant'] = df_out['significant'].astype('bool')
  if effect_size_variance:
    df_out['Effect size variance'] = list(variances) + [0]*len(df_prison)
  if glycoproteomics:
    return get_glycoform_diff(df_out, alpha = alpha, level = level)
  else:
    return df_out.dropna().sort_values(by = 'p-val').sort_values(by = 'corr p-val')


def get_pval_distribution(
    df_res: Union[pd.DataFrame, str], # Output DataFrame from get_differential_expression
    filepath: str = '' # Path to save plot
    ) -> None:
  "Creates histogram of p-values from differential expression analysis"
  if isinstance(df_res, str):
    df_res = pd.read_csv(df_res) if df_res.endswith(".csv") else pd.read_excel(df_res)
  # make plot
  ax = sns.histplot(x = 'p-val', data = df_res, stat = 'frequency')
  ax.set(xlabel = 'p-values', ylabel =  'Frequency', title = '')
  # save to file
  if filepath:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300, bbox_inches = 'tight')
  plt.show()


def get_ma(
    df_res: Union[pd.DataFrame, str], # Output DataFrame from get_differential_expression
    log2fc_thresh: int = 1, # Log2FC threshold for highlighting
    sig_thresh: float = 0.05, # Significance threshold for highlighting
    filepath: str = '' # Path to save plot
    ) -> None:
  "Generates MA plot (mean abundance vs log2 fold change) from differential expression results"
  if isinstance(df_res, str):
    df_res = pd.read_csv(df_res) if df_res.endswith(".csv") else pd.read_excel(df_res)
  # Create masks for significant and non-significant points
  sig_mask = (abs(df_res['Log2FC']) > log2fc_thresh) & (df_res['corr p-val'] < sig_thresh)
  # Plot non-significant points first
  ax = sns.scatterplot(x = 'Mean abundance', y = 'Log2FC',
                       data = df_res[~sig_mask], color = 'grey', alpha = 0.5)
  # Overlay significant points
  sns.scatterplot(x = 'Mean abundance', y = 'Log2FC',
                    data = df_res[sig_mask], color = '#CC4446', alpha = 0.95, ax = ax)
  ax.set(xlabel = 'Mean Abundance', ylabel = 'Log2FC', title = '')
  # save to file
  if filepath:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300, bbox_inches = 'tight')
  plt.show()


def get_volcano(
    df_res: Union[pd.DataFrame, str], # DataFrame from get_differential_expression with columns [Glycan, Log2FC, p-val, corr p-val]
    y_thresh: float = 0.05, # Corrected p threshold for labeling
    x_thresh: float = 0, # Absolute x metric threshold for labeling
    n: Optional[int] = None, # Sample size for Bayesian-Adaptive Alpha
    label_changed: bool = True, # Add text labels to significant points
    x_metric: str = 'Log2FC', # x-axis metric: 'Log2FC' or 'Effect size'
    annotate_volcano: bool = False, # Annotate dots with SNFG images
    filepath: str = '', # Path to save plot
    **kwargs: Any # Keyword args passed to seaborn scatterplot
    ) -> None: # Displays volcano plot
  "Creates volcano plot showing -log10(FDR-corrected p-values) vs Log2FC or effect size"
  if isinstance(df_res, str):
    df_res = pd.read_csv(df_res) if df_res.endswith(".csv") else pd.read_excel(df_res)
  df_res['log_p'] = -np.log10(df_res['corr p-val'].values)
  x = df_res[x_metric].values
  y = df_res['log_p'].values
  labels = df_res['Glycan'].values if 'Glycan' in df_res.columns else df_res['Glycosite'].values
  # set y_thresh based on sample size via Bayesian-Adaptive Alpha Adjustment
  if n:
    y_thresh = get_alphaN(n)
  else:
    print("You're working with a default alpha of 0.05. Set sample size (n = ...) for Bayesian-Adaptive Alpha Adjustment")
  # Make plot
  color = kwargs.pop('color', '#3E3E3E')
  ax = sns.scatterplot(x = x_metric, y = 'log_p', data = df_res, color = color, alpha = 0.8, **kwargs)
  ax.set(xlabel = x_metric, ylabel = '-log10(corr p-val)', title = '')
  plt.axhline(y = -np.log10(y_thresh), c = 'k', ls = ':', lw = 0.5, alpha = 0.3)
  plt.axvline(x = x_thresh, c = 'k', ls = ':', lw = 0.5, alpha = 0.3)
  plt.axvline(x = -x_thresh, c = 'k', ls = ':', lw = 0.5, alpha = 0.3)
  sns.despine(bottom = True, left = True)
  # Text labels
  if label_changed:
    for i in np.where((y > -np.log10(y_thresh)) & (np.abs(x) > x_thresh))[0]:
      plt.text(x[i], y[i], labels[i])
  # Save to file
  if filepath:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300,
                  bbox_inches = 'tight')
    if annotate_volcano:
      from glycowork.motif.draw import annotate_figure
      annotate_figure(filepath, filepath = filepath.split('.')[0]+'.pdf', scale_by_DE_res = df_res,
                          y_thresh = y_thresh, x_thresh = x_thresh, x_metric = x_metric)
  plt.show()


def get_glycanova(
    df: Union[pd.DataFrame, str], # DataFrame with glycans in rows (col 1) and abundance values in columns
    groups: List[Any], # Group labels for samples (e.g., [1,1,1,2,2,2,3,3,3])
    impute: bool = True, # Replace zeros with Random Forest model
    motifs: bool = False, # Analyze motifs instead of sequences
    feature_set: List[str] = ['exhaustive', 'known'], # Feature sets to use; exhaustive/known/terminal1/terminal2/terminal3/chemical/graph/custom
    min_samples: float = 0.1, # Min percent of non-zero samples required
    posthoc: bool = True, # Perform Tukey's HSD test post-hoc
    custom_motifs: List[str] = [], # Custom motifs if using 'custom' feature set
    transform: Optional[str] = None, # Transformation type: "CLR" or "ALR"
    gamma: float = 0.1, # Uncertainty parameter for CLR transform
    custom_scale: float = 0 # Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]: # (ANOVA results with F-stats and omega-squared effect sizes, post-hoc results)
    "Performs one-way ANOVA with omega-squared effect size calculation and optional Tukey's HSD post-hoc testing on glycomics data across multiple groups"
    if len(set(groups)) < 3:
      raise ValueError("You have fewer than three groups. We suggest get_differential_expression for those cases. ANOVA is for >= three groups.")
    df, _, groups, _ = preprocess_data(df, groups, [], experiment = "anova", motifs = motifs, impute = impute,
                                      min_samples = min_samples, transform = transform, feature_set = feature_set,
                                      gamma = gamma, custom_scale = custom_scale, custom_motifs = custom_motifs)
    results, posthoc_results = [], {}
    # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
    alpha = get_alphaN(len(groups))
    effect_sizes = df.apply(omega_squared, axis = 1, args = (groups,))
    # Variance-based filtering of features
    df, df_prison = variance_based_filtering(df)
    for glycan in df.index:
      # Create a DataFrame with the glycan abundance and group identifier for each sample
      data = pd.DataFrame({"Abundance": df.loc[glycan], "Group": groups})
      # Run an ANOVA
      model = ols("Abundance ~ C(Group)", data = data).fit()
      anova_table = sm.stats.anova_lm(model, typ = 2)
      f_value = anova_table["F"]["C(Group)"]
      p_value = anova_table["PR(>F)"]["C(Group)"]
      results.append((glycan, f_value, p_value))
      if p_value < alpha and posthoc:
        posthoc = pairwise_tukeyhsd(endog = data['Abundance'], groups = data['Group'], alpha = alpha)
        posthoc_results[glycan] = pd.DataFrame(data = posthoc._results_table.data[1:], columns = posthoc._results_table.data[0])
    df_out = pd.DataFrame(results, columns = ["Glycan", "F statistic", "p-val"])
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
      df_out = pd.concat([df_out, prison_rows], ignore_index = True)
    df_out['significant'] = df_out['significant'].astype('bool')
    df_out['Effect size'] = effect_sizes.values
    return df_out.sort_values(by = 'corr p-val'), posthoc_results


def get_meta_analysis(
    effect_sizes: Union[np.ndarray, List[float]], # List of Cohen's d/other effect sizes
    variances: Union[np.ndarray, List[float]], # Associated variance estimates
    model: str = 'fixed', # 'fixed' or 'random' effects model
    filepath: str = '', # Path to save Forest plot
    study_names: List[str] = [] # Names corresponding to each effect size
    ) -> Tuple[float, float]: # (combined effect size, two-tailed p-value)
    "Performs fixed/random effects meta-analysis using DerSimonian-Laird method for between-study variance estimation, with optional Forest plot visualization"
    if model not in ['fixed', 'random']:
      raise ValueError("Model must be 'fixed' or 'random'")
    variances = np.array(variances)
    weights = 1 / variances
    total_weight = np.sum(weights)
    combined_effect_size = np.dot(weights, effect_sizes) / total_weight
    if model == 'random':
      # Estimate between-study variance (tau squared) using DerSimonian-Laird method
      q = np.dot(weights, (effect_sizes - combined_effect_size) ** 2)
      df = len(effect_sizes) - 1
      c = total_weight - np.sum(weights**2) / total_weight
      tau_squared = max((q - df) / (c + 1e-8), 0)
      # Update weights for tau_squared
      weights = 1 / (variances + tau_squared)
      total_weight = np.sum(weights)
      # Recalculate combined effect size
      combined_effect_size = np.dot(weights, effect_sizes) / (total_weight + 1e-8)
    # Calculate standard error and z-score
    se = np.sqrt(1 / (total_weight + 1e-8))
    z = combined_effect_size / (se + 1e-8)
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))
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
      _, ax = plt.subplots(figsize = (8, len(df_temp)*0.6))  # adjust the size as needed
      y_pos = np.arange(len(df_temp))
      # Draw a horizontal line for each study, with x-values between the lower and upper confidence bounds
      ax.hlines(y_pos, df_temp['lower'], df_temp['upper'], color = 'skyblue')
      # Draw a marker at the effect size
      ax.plot(df_temp['EffectSize'], y_pos, 'o', color = 'skyblue')
      # Draw a vertical line at x=0 (or change to the desired reference line)
      ax.vlines(0, -1, len(df_temp), colors = 'gray', linestyles = 'dashed')
      ax.set_yticks(y_pos)
      ax.set_yticklabels(df_temp['Study'])
      ax.invert_yaxis()
      ax.set_xlabel('Effect size')
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      plt.tight_layout()
      plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300,
                    bbox_inches = 'tight')
    return combined_effect_size, p_value


def get_glycan_change_over_time(
    data: np.ndarray, # 2D array with columns [timepoint, abundance]
    degree: int = 1 # Polynomial degree for regression
    ) -> Tuple[Union[float, np.ndarray], float]: # (regression coefficients, t-test/F-test p-value)
    "Fits polynomial regression (default: linear) to glycan abundance time series data using OLS, testing significance of temporal changes"
    # Extract arrays for time and glycan abundance from the 2D input array
    time, glycan_abundance = data[:, 0], data[:, 1]
    if degree == 1:
      # Add a constant (for the intercept term)
      time_with_intercept = sm.add_constant(time)
      # Fit the OLS model
      results = sm.OLS(glycan_abundance, time_with_intercept).fit()
      # Get the slope & the p-value for the slope from the model summary
      coefficients, p_value = results.params[1], results.pvalues[1]
    else:
      # Polynomial Regression
      coefficients = np.polyfit(time, glycan_abundance, degree)
      # Calculate the residuals
      residuals = glycan_abundance - np.polyval(coefficients, time)
      # Perform F-test to get p_value
      _, p_value = f_oneway(glycan_abundance, residuals)
    return coefficients, p_value


def get_time_series(
    df: Union[pd.DataFrame, str], # DataFrame with sample IDs as 'sampleID_timepoint_replicate' in col 1 (e.g., T1_h5_r1)
    impute: bool = True, # Replace zeros with Random Forest model
    motifs: bool = False, # Analyze motifs instead of sequences
    feature_set: List[str] = ['known', 'exhaustive'], # Feature sets to use; exhaustive/known/terminal1/terminal2/terminal3/chemical/graph/custom
    degree: int = 1, # Polynomial degree for regression
    min_samples: float = 0.1, # Min percent of non-zero samples required
    custom_motifs: List[str] = [], # Custom motifs if using 'custom' feature set
    transform: Optional[str] = None, # Transformation type: "CLR" or "ALR"
    gamma: float = 0.1, # Uncertainty parameter for CLR transform
    custom_scale: Union[float, Dict] = 0 # Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
    ) -> pd.DataFrame: # DataFrame with regression coefficients and FDR-corrected p-values
    "Analyzes time series glycomics data using polynomial regression"
    if isinstance(df, str):
      df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
    df = df.fillna(0)
    if isinstance(df.iloc[0, 0], str):
      df = df.set_index(df.columns[0])
    if '-' not in df.index[0]:
      df = df.T
    df = df.apply(replace_outliers_winsorization, axis = 0).reset_index(names = 'glycan')
    df = impute_and_normalize(df, [df.columns[1:].tolist()], impute = impute, min_samples = min_samples)
    if transform is None:
      transform = "ALR" if enforce_class(df.iloc[0, 0], "N") and len(df) > 50 else "CLR"
    if transform == "ALR":
      df = get_additive_logratio_transformation(df, df.columns[1:].tolist(), [], paired = False, gamma = gamma, custom_scale = custom_scale)
    elif transform == "CLR":
      df.iloc[:, 1:] = clr_transformation(df.iloc[:, 1:], df.columns[1:].tolist(), [], gamma = gamma, custom_scale = custom_scale)
    elif transform == "Nothing":
      pass
    else:
      raise ValueError("Only ALR and CLR are valid transforms for now.")
    # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
    alpha = get_alphaN(df.shape[1] - 1)
    glycans = [k.split('.')[0] for k in df.iloc[:, 0]]
    if motifs:
      df = quantify_motifs(df.iloc[:, 1:], glycans, feature_set, custom_motifs = custom_motifs)
    else:
      df.index = glycans
      df = df.drop([df.columns[0]], axis = 1)
      df = df.groupby(df.index).mean()
    df = df.T.reset_index()
    df[df.columns[0]] = df.iloc[:, 0].apply(lambda x: float(x.split('_')[1][1:]))
    df = df.sort_values(by = df.columns[0])
    time = df.iloc[:, 0].to_numpy()  # Time points
    df_out = [(c, *get_glycan_change_over_time(np.column_stack((time, df[c].to_numpy())), degree = degree)) for c in df.columns[1:]]
    df_out = pd.DataFrame(df_out, columns = ['Glycan', 'Change', 'p-val'])
    corrpvals, significance = correct_multiple_testing(df_out['p-val'], alpha)
    df_out['corr p-val'] = corrpvals
    df_out['significant'] = significance
    return df_out.sort_values(by = 'corr p-val')


def get_jtk(
   df_in: Union[pd.DataFrame, str], # DataFrame with molecules in rows (col 0), then groups arranged by ascending timepoints
   timepoints: int, # Number of timepoints (each must have same number of replicates)
   periods: List[int], # Timepoints per cycle to test
   interval: int, # Time units between experimental timepoints
   motifs: bool = False, # Analyze motifs instead of sequences
   feature_set: List[str] = ['known', 'exhaustive', 'terminal'], # Feature sets to use; exhaustive/known/terminal1/terminal2/terminal3/chemical/graph/custom
   custom_motifs: List[str] = [], # Custom motifs if using 'custom' feature set
   transform: Optional[str] = None, # Transformation type: "CLR" or "ALR"
   gamma: float = 0.1, # Uncertainty parameter for CLR transform
   correction_method: str = "two-stage" # Multiple testing correction method
   ) -> pd.DataFrame: # DataFrame with JTK results: adjusted p-values, period length, lag phase, amplitude
    "Identifies rhythmically expressed glycans using Jonckheere-Terpstra-Kendall algorithm for time series analysis"
    if isinstance(df_in, str):
      df = pd.read_csv(df_in) if df_in.endswith(".csv") else pd.read_excel(df_in)
    else:
      df = df_in.copy(deep = True)
    replicates = (df.shape[1] - 1) // timepoints
    alpha = get_alphaN(replicates)
    param_dic = {"GRP_SIZE": [], "NUM_GRPS": [], "MAX": [], "DIMS": [], "EXACT": bool(True),
                 "VAR": [], "EXV": [], "SDV": [], "CGOOSV": []}
    param_dic = jtkdist(timepoints, param_dic, replicates)
    param_dic = jtkinit(periods, param_dic, interval, replicates)
    df = df.apply(replace_outliers_winsorization, axis = 1)
    mf = MissForest()
    df = df.replace(0, np.nan)
    annot = df.pop(df.columns.tolist()[0])
    df = mf.fit_transform(df)
    df.insert(0, 'Molecule_Name', annot)
    if transform is None:
      transform = "ALR" if enforce_class(df.iloc[0, 0], "N") and len(df) > 50 else "CLR"
    if transform == "ALR":
      df = get_additive_logratio_transformation(df, df.columns[1:].tolist(), [], paired = False, gamma = gamma)
    elif transform == "CLR":
      df.iloc[:, 1:] = clr_transformation(df.iloc[:, 1:], df.columns[1:].tolist(), [], gamma = gamma)
    elif transform == "Nothing":
      pass
    else:
      raise ValueError("Only ALR and CLR are valid transforms for now.")
    if motifs:
      df = quantify_motifs(df.iloc[:, 1:], df.iloc[:, 0].values.tolist(), feature_set, custom_motifs = custom_motifs).reset_index()
    res = df.iloc[:, 1:].apply(jtkx, param_dic = param_dic, axis = 1)
    corrpvals, significance = correct_multiple_testing(res[0], alpha, correction_method = correction_method)
    df_out = pd.concat([df.iloc[:, 0], pd.DataFrame(corrpvals), res], axis = 1)
    df_out.columns = ['Molecule_Name', 'BH_Q_Value', 'Adjusted_P_value', 'Period_Length', 'Lag_Phase', 'Amplitude']
    df_out['significant'] = significance
    return df_out.sort_values("Adjusted_P_value")


def get_biodiversity(
    df: Union[pd.DataFrame, str], # DataFrame with glycans in rows (col 1), abundances in columns
    group1: List[Union[str, int]], # First group column indices or group labels
    group2: List[Union[str, int]], # Second group indices or additional group labels
    metrics: List[str] = ['alpha', 'beta'], # Diversity metrics to calculate
    motifs: bool = False, # Analyze motifs instead of sequences
    feature_set: List[str] = ['exhaustive', 'known'], # Feature sets to use; exhaustive/known/terminal1/terminal2/terminal3/chemical/graph/custom
    custom_motifs: List[str] = [], # Custom motifs if using 'custom' feature set
    paired: bool = False, # Whether samples are paired
    permutations: int = 999, # Number of permutations for ANOSIM/PERMANOVA
    transform: Optional[str] = None, # Transformation type: "CLR" or "ALR"
    gamma: float = 0.1, # Uncertainty parameter for CLR transform
    custom_scale: Union[float, Dict] = 0 # Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
    ) -> pd.DataFrame: # DataFrame with diversity indices and test statistics
  "Calculates alpha (Shannon/Simpson) and beta (ANOSIM/PERMANOVA) diversity measures from glycomics data"
  experiment = "diff" if group2 else "anova"
  df, df_org, group1, group2 = preprocess_data(df, group1, group2, experiment = experiment, motifs = motifs, impute = False,
                                               transform = transform, feature_set = feature_set, paired = paired, gamma = gamma,
                                               custom_scale = custom_scale, custom_motifs = custom_motifs)
  shopping_cart = []
  group_sizes = group1 if not group2 else len(group1)*[1]+len(group2)*[2]
  group_counts = Counter(group_sizes)
  # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
  alpha = get_alphaN(len(group_sizes))
  if 'alpha' in metrics:
    unique_counts = df_org.apply(sequence_richness)
    shan_div = df_org.apply(shannon_diversity_index)
    simp_div = df_org.apply(simpson_diversity_index)
    a_df = pd.DataFrame({'species_richness': unique_counts,
                               'shannon_diversity': shan_div, 'simpson_diversity': simp_div}).T
    if len(group_counts) == 2:
      df_a, df_b = a_df[group1], a_df[group2]
      mean_a, mean_b = [np.mean(row_a) for row_a in df_a.values], [np.mean(row_b) for row_b in df_b.values]
      if paired:
        assert len(df_a) == len(df_b), "For paired samples, the size of group1 and group2 should be the same"
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
      shopping_cart.append(pd.DataFrame({'Metric': 'Shannon diversity (ANOVA)', 'p-val': sh_stats[1], 'Effect size': sh_stats[0]}, index = [0]))
      si_stats = alpha_biodiversity_stats(simp_div, group_sizes)
      shopping_cart.append(pd.DataFrame({'Metric': 'Simpson diversity (ANOVA)', 'p-val': si_stats[1], 'Effect size': si_stats[0]}, index = [0]))
  if 'beta' in metrics:
    if not isinstance(df.index[0], str):
      df = df.set_index(df.columns[0])
    bc_diversity = {}  # Calculating pair-wise indices
    for index_1 in range(0, len(df.columns)):
      for index_2 in range(0, len(df.columns)):
        bc_pair = np.sqrt(np.sum((df.iloc[:, index_1] - df.iloc[:, index_2]) ** 2))
        bc_diversity[index_1, index_2] = bc_pair
    b_df_out = pd.DataFrame.from_dict(bc_diversity, orient = 'index')
    out_len = int(np.sqrt(len(b_df_out)))
    b_df_out_values = b_df_out.values.reshape(out_len, out_len)
    beta_df_out = pd.DataFrame(data = b_df_out_values, index = range(out_len), columns = range(out_len))
    if all(count > 1 for count in group_counts.values()):
      r, p = anosim(beta_df_out, group_sizes, permutations)
      b_test_stats = pd.DataFrame({'Metric': 'Beta diversity (ANOSIM)', 'p-val': p, 'Effect size': r}, index = [0])
      shopping_cart.append(b_test_stats)
      f, p = permanova_with_permutation(beta_df_out, group_sizes, permutations)
      b_test_stats = pd.DataFrame({'Metric': 'Beta diversity (PERMANOVA)', 'p-val': p, 'Effect size': f}, index = [0])
      shopping_cart.append(b_test_stats)
  df_out = pd.concat(shopping_cart, axis = 0).reset_index(drop = True)
  corrpvals, significance = correct_multiple_testing(df_out['p-val'], alpha)
  df_out["corr p-val"] = corrpvals
  df_out["significant"] = significance
  return df_out.sort_values(by = 'p-val').sort_values(by = 'corr p-val').reset_index(drop = True)


def get_SparCC(
    df1: Union[pd.DataFrame, str], # First DataFrame with glycans in rows (col 1) and abundances in columns
    df2: Union[pd.DataFrame, str], # Second DataFrame with same format as df1
    motifs: bool = False, # Analyze motifs instead of sequences
    feature_set: List[str] = ["known", "exhaustive"], # Feature sets to use; exhaustive/known/terminal1/terminal2/terminal3/chemical/graph/custom
    custom_motifs: List[str] = [], # Custom motifs if using 'custom' feature set
    transform: Optional[str] = None, # Transformation type: "CLR" or "ALR"
    gamma: float = 0.1, # Uncertainty parameter for CLR transform
    partial_correlations: bool = False # Use regularized partial correlations
    ) -> Tuple[pd.DataFrame, pd.DataFrame]: # (Spearman correlation matrix, FDR-corrected p-value matrix)
  "Calculates SparCC (Sparse Correlations for Compositional Data) between two matching datasets (e.g., glycomics)"
  if isinstance(df1, str):
    df1 = pd.read_csv(df1) if df1.endswith(".csv") else pd.read_excel(df1)
    df2 = pd.read_csv(df2) if df2.endswith(".csv") else pd.read_excel(df2)
  df1.iloc[:, 0] = strip_suffixes(df1.iloc[:, 0])
  df2.iloc[:, 0] = strip_suffixes(df2.iloc[:, 0])
  if df1.columns.tolist()[0] != df2.columns.tolist()[0] and df1.columns.tolist()[0] in df2.columns.tolist():
    common_columns = df1.columns.intersection(df2.columns)
    df1 = df1[common_columns]
    df2 = df2[common_columns]
  # Drop rows with all zero, followed by outlier removal and imputation & normalization
  df1 = df1.loc[~(df1.iloc[:, 1:] == 0).all(axis = 1)]
  df1 = df1.apply(replace_outliers_winsorization, axis = 1)
  df1 = impute_and_normalize(df1, [df1.columns.tolist()[1:]])
  df2 = df2.loc[~(df2.iloc[:, 1:] == 0).all(axis = 1)]
  df2 = df2.apply(replace_outliers_winsorization, axis = 1)
  df2 = impute_and_normalize(df2, [df2.columns.tolist()[1:]])
  # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
  _ = get_alphaN(df1.shape[1] - 1)
  if transform is None:
    transform = "ALR" if (enforce_class(df1.iloc[0, 0], "N") and len(df1) > 50) and (enforce_class(df2.iloc[0, 0], "N") and len(df2) > 50) else "CLR"
  if transform == "ALR":
    df1 = get_additive_logratio_transformation(df1, df1.columns[1:].tolist(), [], paired = False, gamma = gamma)
    df2 = get_additive_logratio_transformation(df2, df2.columns[1:].tolist(), [], paired = False, gamma = gamma)
  elif transform == "CLR":
    df1.iloc[:, 1:] = clr_transformation(df1.iloc[:, 1:], df1.columns.tolist()[1:], [], gamma = gamma)
    df2.iloc[:, 1:] = clr_transformation(df2.iloc[:, 1:], df2.columns.tolist()[1:], [], gamma = gamma)
  elif transform == "Nothing":
    pass
  else:
    raise ValueError("Only ALR and CLR are valid transforms for now.")
  if motifs:
    df1 = quantify_motifs(df1.iloc[:, 1:], df1.iloc[:, 0].values.tolist(), feature_set, custom_motifs = custom_motifs)
    if '(' in df2.iloc[:, 0].values.tolist()[0]:
      df2 = quantify_motifs(df2.iloc[:, 1:], df2.iloc[:, 0].values.tolist(), feature_set, custom_motifs = custom_motifs)
    else:
      df2 = df2.set_index(df2.columns.tolist()[0])
  else:
    df1 = df1.set_index(df1.columns.tolist()[0])
    df2 = df2.set_index(df2.columns.tolist()[0])
  df1, df2 = df1.T, df2.T
  correlation_matrix = np.zeros((df1.shape[1], df2.shape[1]))
  p_value_matrix = np.zeros((df1.shape[1], df2.shape[1]))
  if partial_correlations:
    correlations_df1 = np.abs(np.corrcoef(df1.transpose()))
    correlations_df2 = np.abs(np.corrcoef(df2.transpose()))
    threshold = 0.5 if motifs else 0.2
  # Compute Spearman correlation for each pair of columns between transformed df1 and df2
  for i in range(df1.shape[1]):
    for j in range(df2.shape[1]):
      if partial_correlations:
        valid_controls_i = [k for k in range(df1.shape[1]) if correlations_df1[i, k] > threshold and k != i]
        valid_controls_j = [k for k in range(df2.shape[1]) if correlations_df2[j, k] > threshold and k != j]
        controls_i = df1.iloc[:, valid_controls_i].values
        controls_j = df2.iloc[:, valid_controls_j].values
        controls = np.hstack([controls_i, controls_j])
        corr, p_val = partial_corr(df1.iloc[:, i].values, df2.iloc[:, j].values, controls, motifs = motifs)
      else:
        corr, p_val = spearmanr(df1.iloc[:, i], df2.iloc[:, j])
      correlation_matrix[i, j] = corr
      p_value_matrix[i, j] = p_val
  p_value_matrix = multipletests(p_value_matrix.flatten(), method = 'fdr_tsbh')[1].reshape(p_value_matrix.shape)
  correlation_df = pd.DataFrame(correlation_matrix, index = df1.columns, columns = df2.columns)
  p_value_df = pd.DataFrame(p_value_matrix, index = df1.columns, columns = df2.columns)
  return correlation_df, p_value_df


def multi_feature_scoring(
    df: pd.DataFrame, # Transformed dataframe with glycans in rows, abundances in columns
    group1: List[Union[str, int]], # First group indices/names
    group2: List[Union[str, int]], # Second group indices/names
    filepath: str = '' # Path to save ROC plot
    ) -> Tuple[LogisticRegression, float]: # (L1-regularized logistic regression model, ROC AUC score)
  "Identifies minimal glycan feature set for group classification using L1-regularized logistic regression"
  if group2:
    y = [0] * len(group1) + [1] * len(group2)
  else:
    y = group1
  X = df.T
  model = LogisticRegression(penalty = 'l1', solver = 'liblinear', random_state = 42)
  model.fit(X.values, y)
  model = SelectFromModel(model, prefit = True)
  X_selected = model.transform(X.values)
  selected_features = X.columns[model.get_support()]
  print("Optimal features:", selected_features)
  model = LogisticRegression(penalty = 'l2', solver = 'liblinear', random_state = 42)
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
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300, bbox_inches = 'tight')
  return model, roc_auc


def get_roc(
    df: Union[pd.DataFrame, str], # DataFrame with glycans in rows (col 1), abundances in columns
    group1: List[Union[str, int]], # First group indices/names
    group2: List[Union[str, int]], # Second group indices/names
    motifs: bool = False, # Analyze motifs instead of sequences
    feature_set: List[str] = ["known", "exhaustive"], # Feature sets to use; exhaustive/known/terminal1/terminal2/terminal3/chemical/graph/custom
    paired: bool = False, # Whether samples are paired
    impute: bool = True, # Replace zeros with Random Forest model
    min_samples: float = 0.1, # Min percent of non-zero samples required
    custom_motifs: List[str] = [], # Custom motifs if using 'custom' feature set
    transform: Optional[str] = None, # Transformation type: "CLR" or "ALR"
    gamma: float = 0.1, # Uncertainty parameter for CLR transform
    custom_scale: Union[float, Dict] = 0, # Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
    filepath: str = '', # Path to save ROC plot
    multi_score: bool = False # Find best multi-glycan score
    ) -> Union[List[Tuple[str, float]], Dict[Any, Tuple[str, float]], Tuple[LogisticRegression, float]]: # (Feature scores with ROC AUC values)
  "Calculates ROC curves and AUC scores for glycans/motifs or multi-glycan classifiers"
  experiment = "diff" if group2 else "anova"
  df, _, group1, group2 = preprocess_data(df, group1, group2, experiment = experiment, motifs = motifs, impute = impute,
                                               transform = transform, feature_set = feature_set, paired = paired, gamma = gamma,
                                               custom_scale = custom_scale, custom_motifs = custom_motifs)
  if multi_score:
    return multi_feature_scoring(df, group1, group2, filepath = filepath)
  auc_scores = {}
  if group2: # binary comparison
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
      plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300, bbox_inches = 'tight')
  else: # multi-group comparison
    classes = list(set(group1))
    df = df.groupby(df.index).mean()
    df = df.T  # Ensure features are columns
    df['group'] = group1
    y = label_binarize(df['group'], classes = classes)
    n_classes = y.shape[1]
    sorted_auc_scores, best_fpr, best_tpr = {}, {}, {}
    for feature in df.columns[:-1]:  # exclude the 'group' label column
      X = df[feature].values.reshape(-1, 1)  # Feature matrix needs to be column-wise
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
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
      plt.savefig(filepath.split('.')[0] + "_" + str(classy) + filepath.split('.')[-1], format = filepath.split('.')[-1], dpi = 300, bbox_inches = 'tight')
  return sorted_auc_scores


def get_lectin_array(
    df: Union[pd.DataFrame, str], # DataFrame with samples as rows and lectins as columns, first column containing sample IDs
    group1: List[Union[str, int]], # First group indices/names
    group2: List[Union[str, int]], # Second group indices/names
    paired: bool = False, # Whether samples are paired
    transform: str = '' # Optional log2 transformation
    ) -> pd.DataFrame: # DataFrame with altered glycan motifs, supporting lectins, and effect sizes
  "Analyzes lectin microarray data by mapping lectin binding patterns to glycan motifs, calculating Cohen's d effect sizes between groups and clustering results by significance"
  if isinstance(df, str):
    df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
  df = df.set_index(df.columns[0])
  duplicated_cols = set(df.columns[df.columns.duplicated()])
  if duplicated_cols:
    raise ValueError(f'Analysis aborted due to:\nDuplicates found for the following lectin(s): {", ".join(duplicated_cols)}.\nIf you have multiple copies of the same lectin, rename them by adding a suffix in the form of "_<identifier>" (underscore + an identifier).\nFor example, "SNA" may be renamed "SNA_1", "SNA_batch1", etc. ')
  lectin_list = df.columns.tolist()
  df = np.log2(df) if transform == "log2" else df
  df = df.T
  if not isinstance(group1[0], str):
    if group1[0] == 1 or group2[0] == 1:
      group1 = [k-1 for k in group1]
      group2 = [k-1 for k in group2]
    columns_list = df.columns.tolist()
    group1 = [columns_list[k] for k in group1]
    group2 = [columns_list[k] for k in group2]
  df = df.apply(replace_outliers_winsorization, axis = 1)
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
    effects = [cohen_d(row_b, row_a, paired = paired) for row_a, row_b in zip(df_a.values, df_b.values)]
    effect_sizes, _ = list(zip(*effects)) if effects else [[0]*len(df), [0]*len(df)]
  else:
    effect_sizes = df.apply(omega_squared, axis = 1, args = (group1,))
  lectin_score_dict = {lec: effect_sizes[i] for i, lec in enumerate(lectin_list)}
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
  return df_out


def get_glycoshift_per_site(
    df: Union[pd.DataFrame, str], # DataFrame with rows formatted as 'protein_site_composition' in col 1, abundances in remaining cols
    group1: List[Union[str, int]], # First group indices/names
    group2: List[Union[str, int]], # Second group indices/names
    paired: bool = False, # Whether samples are paired
    impute: bool = True, # Replace zeros with Random Forest model
    min_samples: float = 0.2, # Min percent of non-zero samples required
    gamma: float = 0.1, # Uncertainty parameter for CLR transform
    custom_scale: Union[float, Dict] = 0 # Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
    ) -> pd.DataFrame: # DataFrame with GLM coefficients and FDR-corrected p-values
  "Analyzes site-specific glycosylation changes in glycoproteomics data using generalized linear models (GLM) with compositional data normalization"
  df, _, group1, group2 = preprocess_data(df, group1, group2, experiment = "diff", motifs = False, impute = impute,
                                               min_samples = min_samples, transform = "Nothing", paired = paired)
  alpha = get_alphaN(len(group1 + group2))
  df, glycan_features = process_for_glycoshift(df)
  necessary_columns = ['Glycoform'] + glycan_features
  preserved_data = df[necessary_columns]
  df = df.drop(necessary_columns, axis = 1)
  df = df.set_index('Glycosite')
  df = df.div(df.sum(axis = 0), axis = 1) * 100
  df = df.reset_index()
  results = [
        clr_transformation(group_df[group1 + group2], group1, group2, gamma = gamma, custom_scale = custom_scale)
        .assign(Glycosite = glycosite)
        for glycosite, group_df in df.groupby('Glycosite')
    ]
  df = pd.concat(results, ignore_index = True)
  df = pd.concat([df, preserved_data.reset_index(drop = True)], axis = 1)
  df_long = pd.melt(df, id_vars = ['Glycosite', 'Glycoform'] + glycan_features,
                  var_name = 'Sample', value_name = 'Abundance')
  df_long['Condition'] = df_long['Sample'].apply(lambda x: 0 if x in group1 else 1)
  return process_glm_results(df_long, alpha, glycan_features)
