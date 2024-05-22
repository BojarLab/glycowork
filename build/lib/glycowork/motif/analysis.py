import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('default')
from os import path
from collections import Counter
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from glycowork.glycan_data.loader import df_species, unwrap, motif_list, strip_suffixes
from glycowork.glycan_data.stats import (cohen_d, mahalanobis_distance, mahalanobis_variance,
                                         variance_stabilization, impute_and_normalize, variance_based_filtering,
                                         jtkdist, jtkinit, MissForest, jtkx, get_alphaN, TST_grouped_benjamini_hochberg,
                                         test_inter_vs_intra_group, replace_outliers_winsorization, hotellings_t2,
                                         sequence_richness, shannon_diversity_index, simpson_diversity_index,
                                         get_equivalence_test, clr_transformation, anosim, permanova_with_permutation,
                                         alpha_biodiversity_stats, get_additive_logratio_transformation, correct_multiple_testing,
                                         omega_squared)
from glycowork.motif.processing import enforce_class
from glycowork.motif.annotate import (annotate_dataset, quantify_motifs, link_find, create_correlation_network,
                                      group_glycans_core, group_glycans_sia_fuc, group_glycans_N_glycan_type, load_lectin_lib,
                                      create_lectin_and_motif_mappings, lectin_motif_scoring)
from glycowork.motif.graph import subgraph_isomorphism


def get_pvals_motifs(df, glycan_col_name = 'glycan', label_col_name = 'target',
                     zscores = True, thresh = 1.645, sorting = True, feature_set = ['exhaustive'],
                     multiple_samples = False, motifs = None, custom_motifs = []):
    """returns enriched motifs based on label data or predicted data\n
    | Arguments:
    | :-
    | df (dataframe): dataframe containing glycan sequences and labels [alternative: filepath to .csv or .xlsx]
    | glycan_col_name (string): column name for glycan sequences; arbitrary if multiple_samples = True; default:'glycan'
    | label_col_name (string): column name for labels; arbitrary if multiple_samples = True; default:'target'
    | zscores (bool): whether data are presented as z-scores or not, will be z-score transformed if False; default:True
    | thresh (float): threshold value to separate positive/negative; default is 1.645 for Z-scores
    | sorting (bool): whether p-value dataframe should be sorted ascendingly; default: True
    | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), \
    |   'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), \
    |   'terminal2' (non-reducing end motifs of size 2), 'terminal3' (non-reducing end motifs of size 3), 'custom' (specify your own motifs in custom_motifs), \
    |   and 'chemical' (molecular properties of glycan)
    | multiple_samples (bool): set to True if you have multiple samples (rows) with glycan information (columns); default:False
    | motifs (dataframe): can be used to pass a modified motif_list to the function; default:None
    | custom_motifs (list): list of glycan motifs, used if feature_set includes 'custom'; default:empty\n
    | Returns:
    | :-
    | Returns dataframe with p-values, corrected p-values, and Cohen's d as effect size for every glycan motif
    """
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
    effect_sizes, variances = zip(*[cohen_d(np.append(df_pos.iloc[:, k].values, [1, 0]),
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


def get_representative_substructures(enrichment_df):
    """builds minimal glycans that contain enriched motifs from get_pvals_motifs\n
    | Arguments:
    | :-
    | enrichment_df (dataframe): output from get_pvals_motifs\n
    | Returns:
    | :-
    | Returns up to 10 minimal glycans in a list
    """
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


def clean_up_heatmap(df):
  """removes redundant motifs from the dataframe\n
  | Arguments:
  | :-
  | df (dataframe): dataframe with glycan data, rows are glycan motifs and columns are samples\n
  | Returns:
  | :-
  | Returns a cleaned up dataframe of the same format as the input
  """
  df.index = df.index.to_series().apply(lambda x: x + ' '*20 if x in motif_list.motif_name else x)
  # Group the DataFrame by identical rows
  grouped = df.groupby(list(df.columns))
  # Find the row with the longest string index within each group and return a new DataFrame
  max_idx_series = grouped.apply(lambda group: group.index.to_series().str.len().idxmax())
  result = df.loc[max_idx_series].drop_duplicates()
  result.index = result.index.str.strip()
  return result


def get_heatmap(df, motifs = False, feature_set = ['known'], transform = '',
                 datatype = 'response', rarity_filter = 0.05, filepath = '', index_col = 'glycan',
                custom_motifs = [], **kwargs):
  """clusters samples based on glycan data (for instance glycan binding etc.)\n
  | Arguments:
  | :-
  | df (dataframe): dataframe with glycan data, rows are samples and columns are glycans [alternative: filepath to .csv or .xlsx]
  | motifs (bool): whether to analyze full sequences (False) or motifs (True); default:False
  | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), \
  |   'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), \
  |   'terminal2' (non-reducing end motifs of size 2), 'terminal3' (non-reducing end motifs of size 3), 'custom' (specify your own motifs in custom_motifs), \
  |   and 'chemical' (molecular properties of glycan)
  | transform (string): whether to transform the data before plotting, currently the only option is "CLR", recommended for glycomics data; default: no transformation
  | datatype (string): whether df comes from a dataset with quantitative variable ('response') or from presence_to_matrix ('presence')
  | rarity_filter (float): proportion of samples that need to have a non-zero value for a variable to be included; default:0.05
  | filepath (string): absolute path including full filename allows for saving the plot
  | index_col (string): default column to convert to dataframe index; default:'glycan'
  | custom_motifs (list): list of glycan motifs, used if feature_set includes 'custom'; default:empty
  | **kwargs: keyword arguments that are directly passed on to seaborn clustermap\n                      
  | Returns:
  | :-
  | Prints clustermap              
  """
  if isinstance(df, str):
    df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
  if index_col in df.columns:
    df = df.set_index(index_col)
  elif isinstance(df.iloc[0,0], str):
    df = df.set_index(df.columns.tolist()[0])
  if not isinstance(df.index.tolist()[0], str) or (isinstance(df.index.tolist()[0], str) and '(' not in df.index.tolist()[0] and '-' not in df.index.tolist()[0]):
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
  df = df.dropna(axis = 1)
  if motifs:
    df = clean_up_heatmap(df.T)
  if not (df < 0).any().any():
    df /= df.sum()
    df *= 100
    center = None
  else:
    center = 0
  # Cluster the abundances
  sns.clustermap(df, center = center, **kwargs)
  plt.xlabel('Samples')
  plt.ylabel('Glycans' if not motifs else 'Motifs')
  plt.tight_layout()
  if filepath:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300,
                bbox_inches = 'tight')
  plt.show()


def plot_embeddings(glycans, emb = None, label_list = None,
                    shape_feature = None, filepath = '', alpha = 0.8,
                    palette = 'colorblind', **kwargs):
    """plots glycan representations for a list of glycans\n
    | Arguments:
    | :-
    | glycans (list): list of IUPAC-condensed glycan sequences as strings
    | emb (dictionary): stored glycan representations; default takes them from trained species-level SweetNet model
    | label_list (list): list of same length as glycans if coloring of the plot is desired
    | shape_feature (string): monosaccharide/bond used to display alternative shapes for dots on the plot
    | filepath (string): absolute path including full filename allows for saving the plot
    | alpha (float): transparency of points in plot; default:0.8
    | palette (string): color palette to color different classes; default:'colorblind'
    | **kwargs: keyword arguments that are directly passed on to matplotlib\n
    """
    idx = [i for i, g in enumerate(glycans) if '{' not in g]
    glycans = [glycans[i] for i in idx]
    label_list = [label_list[i] for i in idx]
    # Get all glycan embeddings
    if emb is None:
      this_dir, this_filename = path.split(__file__)
      data_path = path.join(this_dir, 'glycan_representations.pkl')
      emb = pickle.load(open(data_path, 'rb'))
    # Get the subset of embeddings corresponding to 'glycans'
    if isinstance(emb, pd.DataFrame):
      emb = {g: emb.iloc[i, :] for i, g in enumerate(glycans)}
    embs = np.vstack([emb[g] for g in glycans])
    # Calculate t-SNE of embeddings
    embs = TSNE(random_state = 42,
                init = 'pca', learning_rate = 'auto').fit_transform(embs)
    # Plot the t-SNE
    markers = None
    if shape_feature is not None:
      markers = {shape_feature: "X", "Absent": "o"}
      shape_feature = [shape_feature if shape_feature in g else 'Absent' for g in glycans]
    sns.scatterplot(x = embs[:, 0], y = embs[:, 1], hue = label_list,
                    palette = palette, style = shape_feature, markers = markers,
                    alpha = alpha, **kwargs)
    sns.despine(left = True, bottom = True)
    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.tight_layout()
    if filepath:
      plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300,
                  bbox_inches = 'tight')
    plt.show()


def characterize_monosaccharide(sugar, df = None, mode = 'sugar', glycan_col_name = 'glycan',
                                rank = None, focus = None, modifications = False, filepath = '', thresh = 10):
  """for a given monosaccharide/linkage, return typical neighboring linkage/monosaccharide\n
  | Arguments:
  | :-
  | sugar (string): monosaccharide or linkage
  | df (dataframe): dataframe to use for analysis; default:df_species
  | mode (string): either 'sugar' (connected monosaccharides), 'bond' (monosaccharides making a provided linkage), or 'sugarbond' (linkages that a provided monosaccharides makes); default:'sugar'
  | glycan_col_name (string): column name under which glycans can be found; default:'glycan'
  | rank (string): add column name as string if you want to filter for a group
  | focus (string): add row value as string if you want to filter for a group
  | modifications (bool): set to True if you want to consider modified versions of a monosaccharide; default:False
  | filepath (string): absolute path including full filename allows for saving the plot
  | thresh (int): threshold count of when to include motifs in plot; default:10 occurrences\n
  | Returns:
  | :-
  | Plots modification distribution and typical neighboring bond/monosaccharide
  """
  if df is None:
    df = df_species
  if rank is not None and focus is not None:
    df = df[df[rank] == focus]
  # Get all disaccharides for linkage analysis
  pool_in = unwrap([link_find(k) for k in df[glycan_col_name]])
  pool_in = [k.replace('(', '*').replace(')', '*') for k in pool_in]
  pool_in_split = [k.split('*') for k in pool_in]
  pool, lab, sugars = [], '', []
  if mode == 'bond':
    # Get upstream monosaccharides for a specific linkage
    pool = [k[0] for k in pool_in_split if k[1] == sugar]
    lab = f'Observed Monosaccharides Making Linkage {sugar}'
  elif mode in ['sugar', 'sugarbond']:
    for k in pool_in_split:
      # Get downstream monosaccharides or downstream linkages for a specific monosaccharide
      k0, k2 = k[0], k[2] if len(k) > 2 else None
      if modifications and sugar in k0:
        sugars.append(k0)
        pool.append(k2 if mode == 'sugar' else k[1])
      elif k0 == sugar:
        pool.append(k2 if mode == 'sugar' else k[1])
    lab = f'Observed Monosaccharides Paired with {sugar}' if mode == 'sugar' else f'Observed Linkages Made by {sugar}'
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


def get_coverage(df, filepath = ''):
  """ Plot glycan coverage across samples, ordered by average intensity\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns [alternative: filepath to .csv or .xlsx]
  | filepath (string): absolute path including full filename allows for saving the plot\n
  | Returns:
  | :-
  | Prints the heatmap
  """
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
  

def get_pca(df, groups = None, motifs = False, feature_set = ['known', 'exhaustive'],
            pc_x = 1, pc_y = 2, color = None, shape = None, filepath = '', custom_motifs = [],
            transform = None, rarity_filter = 0.05):
  """ PCA plot from glycomics abundance dataframe\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns [alternative: filepath to .csv or .xlsx]
  | groups (list): a list of group identifiers for each sample (e.g., [1,1,1,2,2,2,3,3,3]); default:None
  |                     alternatively: design dataframe with 'id' column of samples names and additional columns with meta information
  | motifs (bool): whether to analyze full sequences (False) or motifs (True); default:False
  | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), \
  |   'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), \
  |   'terminal2' (non-reducing end motifs of size 2), 'terminal3' (non-reducing end motifs of size 3), 'custom' (specify your own motifs in custom_motifs), \
  |   and 'chemical' (molecular properties of glycan)
  | pc_x (int): principal component to plot on x axis; default:1
  | pc_y (int): principal component to plot on y axis; default:2
  | color (string): if design dataframe is provided: column name for color grouping; default:None
  | shape (string): if design dataframe is provided: column name for shape grouping; default:None
  | filepath (string): absolute path including full filename allows for saving the plot
  | custom_motifs (list): list of glycan motifs, used if feature_set includes 'custom'; default:empty
  | transform (string): whether to transform the data before plotting, options are "CLR" and "ALR", recommended for glycomics data; default: no transformation
  | rarity_filter (float): proportion of samples that need to have a non-zero value for a variable to be included; default:0.05\n
  | Returns:
  | :-
  | Prints PCA plot
  """
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
    df = quantify_motifs(df.iloc[:, 1:], df.iloc[:, 0].values.tolist(), feature_set, custom_motifs = custom_motifs).T.reset_index()
  X = np.array(df.iloc[:, 1:len(groups)+1].T) if groups and isinstance(groups, list) else np.array(df.iloc[:, 1:].T)
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


def select_grouping(cohort_b, cohort_a, glycans, p_values, paired = False, grouped_BH = False):
  """test various means of grouping glycans by domain knowledge, to obtain high intra-group correlation\n
  | Arguments:
  | :-
  | cohort_b (dataframe): dataframe of glycans as rows and samples as columns of the case samples
  | cohort_a (dataframe): dataframe of glycans as rows and samples as columns of the control samples
  | glycans (list): list of glycans in IUPAC-condensed nomenclature
  | p_values (list): list of associated p-values
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False
  | grouped_BH (bool): whether to perform two-stage adaptive Benjamini-Hochberg as a grouped multiple testing correction; will SIGNIFICANTLY increase runtime; default:False\n 
  | Returns:
  | :-
  | Returns dictionaries of group : glycans and group : p-values (just one big group if no grouping can be found)
  """
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
    intra, inter = test_inter_vs_intra_group(cohort_b, cohort_a, glycans, grouped_glycans, paired = paired)
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


def get_differential_expression(df, group1, group2,
                                motifs = False, feature_set = ['exhaustive', 'known'], paired = False,
                                impute = True, sets = False, set_thresh = 0.9, effect_size_variance = False,
                                min_samples = 0.1, grouped_BH = False, custom_motifs = [], transform = None,
                                gamma = 0.1, custom_scale = 0):
  """Calculates differentially expressed glycans or motifs from glycomics data\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns [alternative: filepath to .csv or .xlsx]
  | group1 (list): list of column indices or names for the first group of samples, usually the control
  | group2 (list): list of column indices or names for the second group of samples
  | motifs (bool): whether to analyze full sequences (False) or motifs (True); default:False
  | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), \
  |   'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), \
  |   'terminal2' (non-reducing end motifs of size 2), 'terminal3' (non-reducing end motifs of size 3), 'custom' (specify your own motifs in custom_motifs), \
  |   and 'chemical' (molecular properties of glycan)
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False
  | impute (bool): replaces zeroes with a Random Forest based model; default:True
  | sets (bool): whether to identify clusters of highly correlated glycans/motifs to test for differential expression; default:False
  | set_thresh (float): correlation value used as a threshold for clusters; only used when sets=True; default:0.9
  | effect_size_variance (bool): whether effect size variance should also be calculated/estimated; default:False
  | min_samples (float): Percent of the samples that need to have non-zero values for glycan to be kept; default: 10%
  | grouped_BH (bool): whether to perform two-stage adaptive Benjamini-Hochberg as a grouped multiple testing correction; will SIGNIFICANTLY increase runtime; default:False
  | custom_motifs (list): list of glycan motifs, used if feature_set includes 'custom'; default:empty
  | transform (str): transformation to escape Aitchison space; options are CLR and ALR (use ALR if you have many glycans (>100) with low values); default:will be inferred
  | gamma (float): uncertainty parameter to estimate scale uncertainty for CLR transformation; default: 0.1
  | custom_scale (float or dict): Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)\n
  | Returns:
  | :-
  | Returns a dataframe with:
  | (i) Differentially expressed glycans/motifs/sets
  | (ii) Their mean abundance across all samples in group1 + group2
  | (iii) Log2-transformed fold change of group2 vs group1 (i.e., negative = lower in group2)
  | (iv) Uncorrected p-values (Welch's t-test) for difference in mean
  | (v) Corrected p-values (Welch's t-test with Benjamini-Hochberg correction) for difference in mean
  | (vi) Significance: True/False of whether the corrected p-value lies below the sample size-appropriate significance threshold
  | (vii) Corrected p-values (Levene's test for equality of variances with Benjamini-Hochberg correction) for difference in variance
  | (viii) Effect size as Cohen's d (sets=False) or Mahalanobis distance (sets=True)
  | (xi) Corrected p-values of equivalence test to test whether means are significantly equivalent; only done for p-values > 0.05 from (iv)
  | (x) [only if effect_size_variance=True] Effect size variance
  """
  if isinstance(df, str):
    df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
  if not isinstance(group1[0], str):
    columns_list = df.columns.tolist()
    group1 = [columns_list[k] for k in group1]
    group2 = [columns_list[k] for k in group2]
  df = df.loc[:, [df.columns.tolist()[0]]+group1+group2].fillna(0)
  # Drop rows with all zero, followed by outlier removal and imputation & normalization
  df = df.loc[~(df.iloc[:, 1:] == 0).all(axis = 1)]
  df = df.apply(replace_outliers_winsorization, axis = 1)
  df = impute_and_normalize(df, [group1, group2], impute = impute, min_samples = min_samples)
  df_org = df.copy(deep = True)
  if transform is None:
    transform = "ALR" if enforce_class(df.iloc[0, 0], "N") and len(df) > 50 else "CLR"
  if transform == "ALR":
    df = get_additive_logratio_transformation(df, group1, group2, paired = paired, gamma = gamma, custom_scale = custom_scale)
  elif transform == "CLR":
    df.iloc[:, 1:] = clr_transformation(df.iloc[:, 1:], group1, group2, gamma = gamma, custom_scale = custom_scale)
  elif transform == "Nothing":
    pass
  else:
    raise ValueError("Only ALR and CLR are valid transforms for now.")
  # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
  alpha = get_alphaN(df.shape[1] - 1)
  if motifs:
    # Motif extraction and quantification
    df = quantify_motifs(df.iloc[:, 1:], df.iloc[:, 0].values.tolist(), feature_set, custom_motifs = custom_motifs)
    df_org = quantify_motifs(df_org.iloc[:, 1:], df_org.iloc[:, 0].values.tolist(), feature_set, custom_motifs = custom_motifs)
    # Deduplication
    df = clean_up_heatmap(df.T)
    df_org = clean_up_heatmap(df_org.T)
    # Re-normalization
    df_org = df_org.apply(lambda col: col / col.sum() * 100, axis = 0)
  else:
    df = df.set_index(df.columns.tolist()[0])
    df = df.groupby(df.index).mean()
    df_org = df_org.set_index(df_org.columns.tolist()[0])
    df_org = df_org.groupby(df_org.index).mean()
  # Variance-based filtering of features
  df = variance_based_filtering(df)
  df_org = df_org.loc[df.index]
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
        glycans.append(cluster)
        cluster = list(cluster)
        gp1, gp2 = df_a.loc[cluster, :], df_b.loc[cluster, :]
        mean_abundance_c.append(mean_abundance.loc[cluster].mean())
        log2fc.append(((gp2.values - gp1.values).mean(axis = 1)).mean() if paired else (gp2.mean(axis = 1) - gp1.mean(axis = 1)).mean())
      gp1, gp2 = df.loc[cluster, group1], df.loc[cluster, group2]
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
    pvals = [ttest_rel(row_b, row_a)[1] if paired else ttest_ind(row_b, row_a, equal_var = False)[1] for row_a, row_b in zip(df_a.values, df_b.values)]
    equivalence_pvals = np.array([get_equivalence_test(row_a, row_b, paired = paired) if pvals[i] > 0.05 else np.nan for i, (row_a, row_b) in enumerate(zip(df_a.values, df_b.values))])
    valid_equivalence_pvals = equivalence_pvals[~np.isnan(equivalence_pvals)]
    corrected_equivalence_pvals = multipletests(valid_equivalence_pvals, method = 'fdr_tsbh')[1] if len(valid_equivalence_pvals) else []
    equivalence_pvals[~np.isnan(equivalence_pvals)] = corrected_equivalence_pvals
    equivalence_pvals[np.isnan(equivalence_pvals)] = 1.0
    levene_pvals = [levene(row_b, row_a)[1] for row_a, row_b in zip(df_a.values, df_b.values)]
    effects = [cohen_d(row_b, row_a, paired = paired) for row_a, row_b in zip(df_a.values, df_b.values)]
    effect_sizes, variances = list(zip(*effects)) if effects else [[0]*len(glycans), [0]*len(glycans)]
  # Multiple testing correction
  if pvals:
    if not motifs and grouped_BH:
      grouped_glycans, grouped_pvals = select_grouping(df_b, df_a, glycans, pvals, paired = paired, grouped_BH = grouped_BH)
      corrpvals, significance_dict = TST_grouped_benjamini_hochberg(grouped_glycans, grouped_pvals, alpha)
      corrpvals = [corrpvals[g] for g in glycans]
      corrpvals = [p if p >= pvals[i] else pvals[i] for i, p in enumerate(corrpvals)]
      significance = [significance_dict[g] for g in glycans]
    else:
      corrpvals, significance = correct_multiple_testing(pvals, alpha)
    levene_pvals = multipletests(levene_pvals, method = 'fdr_tsbh')[1]
  else:
    corrpvals, significance = [1]*len(glycans), [False]*len(glycans)
  out = pd.DataFrame(list(zip(glycans, mean_abundance, log2fc, pvals, corrpvals, significance, levene_pvals, effect_sizes, equivalence_pvals)),
                     columns = ['Glycan', 'Mean abundance', 'Log2FC', 'p-val', 'corr p-val', 'significant', 'corr Levene p-val', 'Effect size', 'Equivalence p-val'])
  if effect_size_variance:
    out['Effect size variance'] = variances
  return out.dropna().sort_values(by = 'p-val').sort_values(by = 'corr p-val')


def get_pval_distribution(df_res, filepath = ''):
  """ p-value distribution plot of glycan differential expression result\n
  | Arguments:
  | :-
  | df_res (dataframe): output from get_differential_expression [alternative: filepath to .csv]
  | filepath (string): absolute path including full filename allows for saving the plot\n
  | Returns:
  | :-
  | prints p-value distribution plot
  """
  if isinstance(df_res, str):
    df_res = pd.read_csv(df_res) if df_res.endswith(".csv") else pd.read_excel(df_res)
  # make plot
  ax = sns.histplot(x = 'p-val', data = df_res, stat = 'frequency')
  ax.set(xlabel = 'p-values', ylabel =  'Frequency', title = '')
  # save to file
  if filepath:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300, bbox_inches = 'tight')
  plt.show()  


def get_ma(df_res, log2fc_thresh = 1, sig_thresh = 0.05, filepath = ''):
  """ MA plot of glycan differential expression result\n
  | Arguments:
  | :-
  | df_res (dataframe): output from get_differential_expression [alternative: filepath to .csv or .xlsx]
  | log2fc_thresh (int): absolute Log2FC threshold for highlighting datapoints
  | sig_thresh (int): significance threshold for highlighting datapoints
  | filepath (string): absolute path including full filename allows for saving the plot\n
  | Returns:
  | :-
  | prints MA plot
  """
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


def get_volcano(df_res, y_thresh = 0.05, x_thresh = 0, n = None, label_changed = True,
                x_metric = 'Log2FC', annotate_volcano = False, filepath = '', **kwargs):
  """Plots glycan differential expression results in a volcano plot\n
  | Arguments:
  | :-
  | df_res (dataframe): output from get_differential_expression [alternative: filepath to .csv or .xlsx]
  | y_thresh (float): corr p threshhold for labeling datapoints; default:0.05
  | x_thresh (float): absolute x metric threshold for labeling datapoints; default:0
  | n (float): sample size for Bayesian-Adaptive Alpha Adjustment; default = None
  | label_changed (bool): if True, add text labels to significantly up- and downregulated datapoints; default:True
  | x_metric (string): x-axis metric; default:'Log2FC'; options are 'Log2FC', 'Effect size'
  | annotate_volcano (bool): whether to annotate the dots in the plot with SNFG images; default: False
  | filepath (string): absolute path including full filename allows for saving the plot
  | **kwargs: keyword arguments that are directly passed on to seaborn scatterplot\n
  | Returns:
  | :-
  | Prints volcano plot
  """
  if isinstance(df_res, str):
    df_res = pd.read_csv(df_res) if df_res.endswith(".csv") else pd.read_excel(df_res)
  df_res['log_p'] = -np.log10(df_res['corr p-val'].values)
  x = df_res[x_metric].values
  y = df_res['log_p'].values
  labels = df_res['Glycan'].values
  # set y_thresh based on sample size via Bayesian-Adaptive Alpha Adjustment
  if n:
    y_thresh = get_alphaN(n)
  else:
    print(f"You're working with a default alpha of 0.05. Set sample size (n = ...) for Bayesian-Adaptive Alpha Adjustment")
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


def get_glycanova(df, groups, impute = True, motifs = False, feature_set = ['exhaustive', 'known'],
                  min_samples = 0.1, posthoc = True, custom_motifs = [], transform = None, gamma = 0.1):
    """Calculate an ANOVA for each glycan (or motif) in the DataFrame\n
    | Arguments:
    | :-
    | df (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns [alternative: filepath to .csv or .xlsx]
    | groups (list): a list of group identifiers for each sample (e.g., [1,1,1,2,2,2,3,3,3])
    | impute (bool): replaces zeroes with with a Random Forest based model; default:True
    | motifs (bool): whether to analyze full sequences (False) or motifs (True); default:False
    | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), \
    |   'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), \
    |   'terminal2' (non-reducing end motifs of size 2), 'terminal3' (non-reducing end motifs of size 3), 'custom' (specify your own motifs in custom_motifs), \
    |   and 'chemical' (molecular properties of glycan)
    | min_samples (float): Percent of the samples that need to have non-zero values for glycan to be kept; default: 10%
    | posthoc (bool): whether to do Tukey's HSD test post-hoc to find out which differences were significant; default:True
    | custom_motifs (list): list of glycan motifs, used if feature_set includes 'custom'; default:empty
    | transform (str): transformation to escape Aitchison space; options are CLR and ALR (use ALR if you have many glycans (>100) with low values); default:will be inferred
    | gamma (float): uncertainty parameter to estimate scale uncertainty for CLR transformation; default: 0.1\n
    | Returns:
    | :-
    | (i) a pandas DataFrame with an F statistic, corrected p-value, and indication of its significance for each glycan.
    | (ii) a dictionary of type glycan : pandas DataFrame, with post-hoc results for each glycan with a significant ANOVA.
    """
    if isinstance(df, str):
      df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
    results, posthoc_results = [], {}
    df = df.iloc[:, :len(groups)+1].fillna(0)
    df = df.loc[~(df.iloc[:, 1:] == 0).all(axis = 1)]
    df = df.apply(replace_outliers_winsorization, axis = 1)
    groups_unq = sorted(set(groups))
    df = impute_and_normalize(df, [[df.columns[i+1] for i, x in enumerate(groups) if x == g] for g in groups_unq], impute = impute,
                              min_samples = min_samples)
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
    # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
    alpha = get_alphaN(len(groups))
    if motifs:
      df = quantify_motifs(df.iloc[:, 1:], df.iloc[:, 0].values.tolist(), feature_set, custom_motifs = custom_motifs)
      # Deduplication
      df = clean_up_heatmap(df.T)
    else:
      df = df.set_index(df.columns.tolist()[0])
      df = df.groupby(df.index).mean()
    # Variance-based filtering of features
    df = variance_based_filtering(df)
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
    return df_out.sort_values(by = 'corr p-val'), posthoc_results


def get_meta_analysis(effect_sizes, variances, model = 'fixed', filepath = '',
                      study_names = []):
    """Fixed-effects model or random-effects model for meta-analysis of glycan effect sizes\n
    | Arguments:
    | :-
    | effect_sizes (array-like): Effect sizes (e.g., Cohen's d) from each study
    | variances (array-like): Corresponding effect size variances from each study
    | model (string): Whether to use 'fixed' or 'random' effects model
    | filepath (string): absolute path including full filename allows for saving the Forest plot
    | study_names (list): list of strings indicating the name of each study\n
    | Returns:
    | :-
    | (1) The combined effect size 
    | (2) The p-value for the combined effect size
    """
    if model not in ['fixed', 'random']:
      raise ValueError("Model must be 'fixed' or 'random'")
    weights = 1 / np.array(variances)
    total_weight = np.sum(weights)
    combined_effect_size = np.dot(weights, effect_sizes) / total_weight
    if model == 'random':
      # Estimate between-study variance (tau squared) using DerSimonian-Laird method
      q = np.dot(weights, (effect_sizes - combined_effect_size) ** 2)
      df = len(effect_sizes) - 1
      c = total_weight - np.sum(weights**2) / total_weight
      tau_squared = max((q - df) / (c + 1e-8), 0)
      # Update weights for tau_squared
      weights /= (variances + tau_squared)
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
      fig, ax = plt.subplots(figsize = (8, len(df_temp)*0.6))  # adjust the size as needed
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


def get_glycan_change_over_time(data, degree = 1):
    """Tests if the abundance of a glycan changes significantly over time using an OLS model\n
    | Arguments:
    | :-
    | data (numpy array): a 2D numpy array with two columns (time and glycan abundance) and one row per observation
    | degree (int): degree of the polynomial for regression, default:1 for linear regression\n
    | Returns:
    | :-
    | (i) slope -- the slope of the regression line (i.e., the rate of change of glycan expression over time)
    | (ii) p_value -- the p-value of the t-test for the slope
    """
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


def get_time_series(df, impute = True, motifs = False, feature_set = ['known', 'exhaustive'], degree = 1,
                    min_samples = 0.1, custom_motifs = [], transform = None, gamma = 0.1):
    """Analyzes time series data of glycans using an OLS model\n
    | Arguments:
    | :-
    | df (dataframe): dataframe containing sample IDs of style sampleID_UnitTimepoint_replicate (e.g., T1_h5_r1) in first column and glycan relative abundances in subsequent columns [alternative: filepath to .csv or .xlsx]
    | impute (bool): replaces zeroes with a Random Forest based model; default:True
    | motifs (bool): whether to analyze full sequences (False) or motifs (True); default:False
    | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), \
    |   'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), \
    |   'terminal2' (non-reducing end motifs of size 2), 'terminal3' (non-reducing end motifs of size 3), 'custom' (specify your own motifs in custom_motifs), \
    |   and 'chemical' (molecular properties of glycan)
    | degree (int): degree of the polynomial for regression, default:1 for linear regression
    | min_samples (float): Percent of the samples that need to have non-zero values for glycan to be kept; default: 10%
    | custom_motifs (list): list of glycan motifs, used if feature_set includes 'custom'; default:empty
    | transform (str): transformation to escape Aitchison space; options are CLR and ALR (use ALR if you have many glycans (>100) with low values); default:will be inferred
    | gamma (float): uncertainty parameter to estimate scale uncertainty for CLR transformation; default: 0.1\n
    | Returns:
    | :-
    | Returns a dataframe with:
    | (i) Glycans/motifs potentially exhibiting significant changes over time
    | (ii) The slope of their expression curve over time
    | (iii) Uncorrected p-values (t-test) for testing whether slope is significantly different from zero
    | (iv) Corrected p-values (t-test with Benjamini-Hochberg correction) for testing whether slope is significantly different from zero
    | (v) Significance: True/False whether the corrected p-value lies below the sample size-appropriate significance threshold
    """
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
      df = get_additive_logratio_transformation(df, df.columns[1:].tolist(), [], paired = False, gamma = gamma)
    elif transform == "CLR":
      df.iloc[:, 1:] = clr_transformation(df.iloc[:, 1:], df.columns[1:].tolist(), [], gamma = gamma)
    elif transform == "Nothing":
      pass
    else:
      raise ValueError("Only ALR and CLR are valid transforms for now.")
    # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
    alpha = get_alphaN(df.shape[1] - 1)
    glycans = [k.split('.')[0] for k in df.iloc[:, 0]]
    if motifs:
      df = quantify_motifs(df.iloc[:, 1:], glycans, feature_set, custom_motifs = custom_motifs)
      # Deduplication
      df = clean_up_heatmap(df.T)
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


def get_jtk(df_in, timepoints, periods, interval, motifs = False, feature_set = ['known', 'exhaustive', 'terminal'],
            custom_motifs = [], transform = None, gamma = 0.1):
    """Detecting rhythmically expressed glycans via the Jonckheere–Terpstra–Kendall (JTK) algorithm\n
    | Arguments:
    | :-
    | df_in (pd.DataFrame): A dataframe containing data for analysis. [alternative: filepath to .csv or .xlsx]
    |   (column 0 = molecule IDs, then arranged in groups and by ascending timepoints)
    | timepoints (int): number of timepoints in the experiment (each timepoint must have the same number of replicates).
    | periods (list): number of timepoints (as int) per cycle.
    | interval (int): units of time (Arbitrary units) between experimental timepoints.
    | motifs (bool): a flag for running structural of motif-based analysis (True = run motif analysis); default:False.
    | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), \
    |   'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), \
    |   'terminal2' (non-reducing end motifs of size 2), 'terminal3' (non-reducing end motifs of size 3), 'custom' (specify your own motifs in custom_motifs), \
    |   and 'chemical' (molecular properties of glycan)
    | custom_motifs (list): list of glycan motifs, used if feature_set includes 'custom'; default:empty
    | transform (str): transformation to escape Aitchison space; options are CLR and ALR (use ALR if you have many glycans (>100) with low values); default:will be inferred
    | gamma (float): uncertainty parameter to estimate scale uncertainty for CLR transformation; default: 0.1\n
    | Returns:
    | :-
    | Returns a pandas dataframe containing the adjusted p-values, and most important waveform parameters for each
    | molecule in the analysis.
    """
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
      df = quantify_motifs(df.iloc[:, 1:], df.iloc[:, 0].values.tolist(), feature_set, custom_motifs = custom_motifs).T
      df = clean_up_heatmap(df).reset_index()
    res = df.iloc[:, 1:].apply(jtkx, param_dic = param_dic, axis = 1)
    corrpvals, significance = correct_multiple_testing(res[0], alpha)
    df_out = pd.concat([df.iloc[:, 0], pd.DataFrame(corrpvals), res], axis = 1)
    df_out.columns = ['Molecule_Name', 'BH_Q_Value', 'Adjusted_P_value', 'Period_Length', 'Lag_Phase', 'Amplitude']
    df_out['significant'] = significance
    return df_out.sort_values("Adjusted_P_value")


def get_biodiversity(df, group1, group2, metrics = ['alpha', 'beta'], motifs = False, feature_set = ['exhaustive', 'known'],
                     custom_motifs = [], paired = False, permutations = 999, transform = None, gamma = 0.1):
  """Calculates diversity indices from glycomics data, similar to alpha/beta diversity etc in microbiome data\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns [alternative: filepath to .csv or .xlsx]
  | group1 (list): a list of column identifiers corresponding to samples in group 1
  | group2 (list): a list of column identifiers corresponding to samples in group 2 (note, if an empty list is provided, group 1 can be used a list of group identifiers for each column - e.g., [1,1,2,2,3,3...])
  | metrics (list): which diversity metrics to calculate (alpha, beta); default:['alpha', 'beta']
  | motifs (bool): whether to analyze full sequences (False) or motifs (True); default:False
  | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), \
  |   'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), \
  |   'terminal2' (non-reducing end motifs of size 2), 'terminal3' (non-reducing end motifs of size 3), 'custom' (specify your own motifs in custom_motifs), \
  |   and 'chemical' (molecular properties of glycan)
  | custom_motifs (list): list of glycan motifs, used if feature_set includes 'custom'; default:empty
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False
  | permutations (int): number of permutations to perform in ANOSIM and PERMANOVA statistical test; default:999
  | transform (str): transformation to escape Aitchison space; options are CLR and ALR (use ALR if you have many glycans (>100) with low values); default:will be inferred
  | gamma (float): uncertainty parameter to estimate scale uncertainty for CLR transformation; default: 0.1\n
  | Returns:
  | :-
  | Returns a dataframe with:
  | (i) Diversity indices/metrics
  | (ii) Mean value of diversity metrics in group 1 (only alpha)
  | (iii) Mean value of diversity metrics in group 2 (only alpha)
  | (iv) Uncorrected p-values (Welch's t-test) for difference in mean
  | (v) Corrected p-values (Welch's t-test with two-stage Benjamini-Hochberg correction) for difference in mean
  | (vi) Significance: True/False of whether the corrected p-value lies below the sample size-appropriate significance threshold
  | (vii) Effect size as Cohen's d (ANOSIM R for beta; F statistics for PERMANOVA and Shannon/Simpson (ANOVA))
  """
  if isinstance(df, str):
    df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
  df = df.fillna(0)
  # Drop rows with all zero, followed by outlier removal and imputation & normalization
  df = df.loc[~(df.iloc[:, 1:] == 0).all(axis = 1)]
  df = df.apply(replace_outliers_winsorization, axis = 1)
  shopping_cart = []
  if not group2:
    group_sizes = group1
    # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
    alpha = get_alphaN(df.shape[1] - 1)
  else:
    group_sizes = len(group1)*[1]+len(group2)*[2]
    if not isinstance(group1[0], str):
      columns_list = df.columns.tolist()
      group1 = [columns_list[k] for k in group1]
      group2 = [columns_list[k] for k in group2]
    df = df.loc[:, [df.columns.tolist()[0]] + group1 + group2]
    # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
    alpha = get_alphaN(df.shape[1] - 1)
  group_counts = Counter(group_sizes)
  if motifs:
    if 'beta' in metrics:
      df_org = df.copy(deep = True)
    # Motif extraction and quantification
    df = quantify_motifs(df.iloc[:, 1:], df.iloc[:, 0].values.tolist(), feature_set, custom_motifs = custom_motifs)
    # Deduplication
    df = clean_up_heatmap(df.T)
    # Re-normalization
    df = df.apply(lambda col: col / col.sum() * 100, axis = 0)
  else:
    df = df.set_index(df.columns[0])
  if 'alpha' in metrics:
    unique_counts = df.apply(sequence_richness)
    shan_div = df.apply(shannon_diversity_index)
    simp_div = df.apply(simpson_diversity_index)
    a_df = pd.DataFrame({'species_richness': unique_counts,
                               'shannon_diversity': shan_div, 'simpson_diversity': simp_div}).T
    if len(group_counts) == 2:
      df_a, df_b = a_df[group1], a_df[group2]
      mean_a, mean_b = [np.mean(row_a) for row_a in df_a.values], [np.mean(row_b) for row_b in df_b.values]
      if paired:
        assert len(df_a) == len(df_b), "For paired samples, the size of group1 and group2 should be the same"
      pvals = [ttest_rel(row_b, row_a)[1] if paired else ttest_ind(row_b, row_a, equal_var = False)[1] for
                     row_a, row_b in zip(df_a.values, df_b.values)]
      pvals = [p if p > 0 and p < 1 else 1.0 for p in pvals]
      effect_sizes, variances = zip(*[cohen_d(row_b, row_a, paired = paired) for row_a, row_b in zip(df_a.values, df_b.values)])
      a_df_stats = pd.DataFrame(list(zip(a_df.index.tolist(), mean_a, mean_b, pvals, effect_sizes)),
                               columns = ["Metric", "Group1 mean", "Group2 mean", "p-val", "Effect size"])
      shopping_cart.append(a_df_stats)
    elif all(count > 2 for count in group_counts.values()) and len(group_counts) > 2:
      sh_stats = alpha_biodiversity_stats(shan_div, group_sizes)
      shopping_cart.append(pd.DataFrame({'Metric': 'Shannon diversity (ANOVA)', 'p-val': sh_stats[1], 'Effect size': sh_stats[0]}, index = [0]))
      si_stats = alpha_biodiversity_stats(simp_div, group_sizes)
      shopping_cart.append(pd.DataFrame({'Metric': 'Simpson diversity (ANOVA)', 'p-val': si_stats[1], 'Effect size': si_stats[0]}, index = [0]))
  if 'beta' in metrics:
    if not motifs:
      df_org = df.reset_index()
    if transform is None:
      transform = "ALR" if enforce_class(df_org.iloc[0, 0], "N") and len(df_org) > 50 else "CLR"
    if transform == "ALR":
      df_org = get_additive_logratio_transformation(df_org, group1, group2, paired = paired, gamma = gamma)
    elif transform == "CLR":
      df_org.iloc[:, 1:] = clr_transformation(df_org.iloc[:, 1:], group1, group2, gamma = gamma)
    elif transform == "Nothing":
      pass
    else:
      raise ValueError("Only ALR and CLR are valid transforms for now.")
    if motifs:
      b_df = quantify_motifs(df_org.iloc[:, 1:], df_org.iloc[:, 0].values.tolist(), feature_set, custom_motifs = custom_motifs)
      b_df = clean_up_heatmap(b_df.T)
    else:
      b_df = df_org
    if not isinstance(b_df.index[0], str):
      b_df = b_df.set_index(b_df.columns[0])
    bc_diversity = {}  # Calculating pair-wise indices
    for index_1 in range(0, len(b_df.columns)):
      for index_2 in range(0, len(b_df.columns)):
        bc_pair = np.sqrt(np.sum((b_df.iloc[:, index_1] - b_df.iloc[:, index_2]) ** 2))
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


def get_SparCC(df1, df2, motifs = False, feature_set = ["known", "exhaustive"], custom_motifs = [],
               transform = None, gamma = 0.1):
  """Performs SparCC (Sparse Correlations for Compositional Data) on two (glycomics) datasets. Samples should be in the same order.\n
  | Arguments:
  | :-
  | df1 (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns [alternative: filepath to .csv or .xlsx]
  | df2 (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns [alternative: filepath to .csv or .xlsx]
  | motifs (bool): whether to analyze full sequences (False) or motifs (True); default:False
  | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), \
  |   'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), \
  |   'terminal2' (non-reducing end motifs of size 2), 'terminal3' (non-reducing end motifs of size 3), 'custom' (specify your own motifs in custom_motifs), \
  |   and 'chemical' (molecular properties of glycan)
  | custom_motifs (list): list of glycan motifs, used if feature_set includes 'custom'; default:empty
  | transform (str): transformation to escape Aitchison space; options are CLR and ALR (use ALR if you have many glycans (>100) with low values); default:will be inferred
  | gamma (float): uncertainty parameter to estimate scale uncertainty for CLR transformation; default: 0.1\n
  | Returns:
  | :-
  | Returns (i) a dataframe of pairwise correlations (Spearman's rho)
  | and (ii) a dataframe with corrected p-values (two-stage Benjamini-Hochberg)
  """
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
  alpha = get_alphaN(df1.shape[1] - 1)
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
    df1 = clean_up_heatmap(df1.T)
    if '(' in df2.iloc[:, 0].values.tolist()[0]:
      df2 = quantify_motifs(df2.iloc[:, 1:], df2.iloc[:, 0].values.tolist(), feature_set, custom_motifs = custom_motifs)
      df2 = clean_up_heatmap(df2.T)
    else:
      df2 = df2.set_index(df2.columns.tolist()[0])
  else:
    df1 = df1.set_index(df1.columns.tolist()[0])
    df2 = df2.set_index(df2.columns.tolist()[0])
  df1, df2 = df1.T, df2.T
  correlation_matrix = np.zeros((df1.shape[1], df2.shape[1]))
  p_value_matrix = np.zeros((df1.shape[1], df2.shape[1]))
  # Compute Spearman correlation for each pair of columns between transformed df1 and df2
  for i in range(df1.shape[1]):
    for j in range(df2.shape[1]):
      corr, p_val = spearmanr(df1.iloc[:, i], df2.iloc[:, j])
      correlation_matrix[i, j] = corr
      p_value_matrix[i, j] = p_val
  p_value_matrix = multipletests(p_value_matrix.flatten(), method = 'fdr_tsbh')[1].reshape(p_value_matrix.shape)
  correlation_df = pd.DataFrame(correlation_matrix, index = df1.columns, columns = df2.columns)
  p_value_df = pd.DataFrame(p_value_matrix, index = df1.columns, columns = df2.columns)
  return correlation_df, p_value_df


def get_roc(df, group1, group2, plot = False, motifs = False, feature_set = ["known", "exhaustive"], paired = False, impute = True,
            min_samples = 0.1, custom_motifs = [], transform = None, gamma = 0.1, custom_scale = 0, filepath = ''):
  """Calculates ROC AUC for every feature and, optionally, plots the best\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns [alternative: filepath to .csv or .xlsx]
  | group1 (list): list of column indices or names for the first group of samples, usually the control
  | group2 (list): list of column indices or names for the second group of samples (note, if an empty list is provided, group 1 can be used a list of group identifiers for each column - e.g., [1,1,2,2,3,3...])
  | plot (bool): whether to plot the ROC curve for the best feature; default: False
  | motifs (bool): whether to analyze full sequences (False) or motifs (True); default:False
  | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), \
  |   'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), \
  |   'terminal2' (non-reducing end motifs of size 2), 'terminal3' (non-reducing end motifs of size 3), 'custom' (specify your own motifs in custom_motifs), \
  |   and 'chemical' (molecular properties of glycan)
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False
  | impute (bool): replaces zeroes with a Random Forest based model; default:True
  | min_samples (float): Percent of the samples that need to have non-zero values for glycan to be kept; default: 10%
  | custom_motifs (list): list of glycan motifs, used if feature_set includes 'custom'; default:empty
  | transform (str): transformation to escape Aitchison space; options are CLR and ALR (use ALR if you have many glycans (>100) with low values); default:will be inferred
  | gamma (float): uncertainty parameter to estimate scale uncertainty for CLR transformation; default: 0.1
  | custom_scale (float or dict): Ratio of total signal in group2/group1 for an informed scale model (or group_idx: mean(group)/min(mean(groups)) signal dict for multivariate)
  | filepath (string): absolute path including full filename allows for saving the plot, if plot=True\n
  | Returns:
  | :-
  | Returns a sorted list of tuples of type (glycan, AUC score) and, optionally, ROC curve for best feature
  """
  if isinstance(df, str):
    df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
  if not isinstance(group1[0], str) and group2:
    columns_list = df.columns.tolist()
    group1 = [columns_list[k] for k in group1]
    group2 = [columns_list[k] for k in group2]
  df = df.loc[~(df.iloc[:, 1:] == 0).all(axis = 1)]
  df = df.apply(replace_outliers_winsorization, axis = 1)
  if group2:
    df = impute_and_normalize(df, [group1, group2], impute = impute, min_samples = min_samples)
  else:
    classes = list(set(group1))
    df = impute_and_normalize(df, [[df.columns[i+1] for i, x in enumerate(group1) if x == g] for g in classes], impute = impute, min_samples = min_samples)
  if transform is None:
    transform = "ALR" if enforce_class(df.iloc[0, 0], "N") and len(df) > 50 else "CLR"
  if transform == "ALR":
    if group2:
      df = get_additive_logratio_transformation(df, group1, group2, paired = paired, gamma = gamma, custom_scale = custom_scale)
    else:
      df = get_additive_logratio_transformation(df, df.columns[1:].tolist(), [], paired = paired, gamma = gamma, custom_scale = custom_scale)
  elif transform == "CLR":
    df.iloc[:, 1:] = clr_transformation(df.iloc[:, 1:], group1, group2, gamma = gamma, custom_scale = custom_scale)
  elif transform == "Nothing":
    pass
  else:
    raise ValueError("Only ALR and CLR are valid transforms for now.")
  if not isinstance(df.index[0], str):
    df = df.set_index(df.columns[0])
  if motifs:
    df = quantify_motifs(df, df.index.tolist(), feature_set, custom_motifs = custom_motifs)
    df = clean_up_heatmap(df.T)
  auc_scores = {}
  if group2: # binary comparison
    for feature, values in df.iterrows():
      values_group1 = values[group1]
      values_group2 = values[group2]
      y_true = [0] * len(values_group1) + [1] * len(values_group2)
      y_scores = np.concatenate([values_group1, values_group2])
      auc_scores[feature] = roc_auc_score(y_true, y_scores)
    sorted_auc_scores = sorted(auc_scores.items(), key = lambda item: item[1], reverse = True)
    if plot:
      best, res = sorted_auc_scores[0]
      values = df.loc[best, :]
      values_group1 = values[group1]
      values_group2 = values[group2]
      y_true = [0] * len(values_group1) + [1] * len(values_group2)
      y_scores = np.concatenate([values_group1, values_group2])
      fpr, tpr, _ = roc_curve(y_true, y_scores)
      plt.figure()
      plt.plot(fpr, tpr, label=f'ROC Curve (area = {res:.2f})')
      plt.plot([0, 1], [0, 1], 'r--')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title(f'ROC Curve for {best}')
      plt.legend(loc = 'lower right')
      if filepath:
        plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300, bbox_inches = 'tight')
      plt.show()
  else: # multi-group comparison
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
    if plot:
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


def get_lectin_array(df, group1, group2, paired = False, transform = ''):
  """Function for analyzing lectin array data for two or more groups.\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing samples as rows and lectins as columns [alternative: filepath to .csv or .xlsx]
  | group1 (list): list of indices or names for the first group of samples, usually the control
  | group2 (list): list of indices or names for the second group of samples (note, if an empty list is provided, group 1 can be used a list of group identifiers for each column - e.g., [1,1,2,2,3,3...])
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False
  | transform (string): optional data-processing, "log2" transforms the with np.log2; default:nothing\n
  | Returns:
  | :-
  | Returns an output dataframe with:
  | (i) Deduced glycan motifs altered between groups
  | (ii) human names for features identified in the motifs from (i)
  | (iii) Lectins supporting the change in (i)
  | (iv) Direction of the change (e.g., "up" means higher in group2)
  | (v) Score/Magnitude of the change (remember, if you have more than two groups this reports on any pairwise combination, like an ANOVA)
  | (vi) Clustering of the scores into highly/moderate/low significance findings
  """
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
    mean_scores_per_condition = df[group1 + group2].groupby([0] * len(group1) + [1] * len(group2), axis = 1).mean()
  else:
    mean_scores_per_condition = df.groupby(group1, axis = 1).mean()
  lectin_variance = mean_scores_per_condition.var(axis = 1)
  idf = np.sqrt(lectin_variance)
  if group2:
    df_a, df_b = df[group1], df[group2]
    effects = [cohen_d(row_b, row_a, paired = paired) for row_a, row_b in zip(df_a.values, df_b.values)]
    effect_sizes, variances = list(zip(*effects)) if effects else [[0]*len(df), [0]*len(df)]
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
