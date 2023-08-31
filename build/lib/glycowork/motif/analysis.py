import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('default')
from collections import Counter
from scipy.stats import ttest_ind, ttest_rel, f, norm, levene, f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from glycowork.glycan_data.loader import lib, df_species, unwrap, motif_list
from glycowork.motif.processing import cohen_d, mahalanobis_distance, mahalanobis_variance, variance_stabilization, impute_and_normalize, variance_based_filtering
from glycowork.motif.annotate import annotate_dataset, quantify_motifs, link_find, create_correlation_network
from glycowork.motif.graph import subgraph_isomorphism


def get_pvals_motifs(df, glycan_col_name = 'glycan', label_col_name = 'target',
                     thresh = 1.645, sorting = True, feature_set = ['exhaustive'], extra = 'termini',
                     wildcard_list = [], multiple_samples = False, motifs = None):
    """returns enriched motifs based on label data or predicted data\n
    | Arguments:
    | :-
    | df (dataframe): dataframe containing glycan sequences and labels
    | glycan_col_name (string): column name for glycan sequences; arbitrary if multiple_samples = True; default:'glycan'
    | label_col_name (string): column name for labels; arbitrary if multiple_samples = True; default:'target'
    | thresh (float): threshold value to separate positive/negative; default is 1.645 for Z-scores
    | sorting (bool): whether p-value dataframe should be sorted ascendingly; default: True
    | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'exhaustive'; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), and 'chemical' (molecular properties of glycan)
    | extra (string): 'ignore' skips this, 'wildcards' allows for wildcard matching', and 'termini' allows for positional matching; default:'termini'
    | wildcard_list (list): list of wildcard names (such as '?1-?', 'Hex', 'HexNAc', 'Sia')
    | multiple_samples (bool): set to True if you have multiple samples (rows) with glycan information (columns); default:False
    | motifs (dataframe): can be used to pass a modified motif_list to the function; default:None\n
    | Returns:
    | :-
    | Returns dataframe with p-values and corrected p-values for every glycan motif
    """
    # Reformat to allow for proper annotation in all samples
    if multiple_samples:
        if 'target' in df.columns:
          df.drop(['target'], axis = 1, inplace = True)
        df = df.T
        df = df.reset_index()
        df.columns = [glycan_col_name] + df.columns.values.tolist()[1:]
    # Annotate glycan motifs in dataset
    df_motif = annotate_dataset(df[glycan_col_name].values.tolist(),
                                motifs = motifs, feature_set = feature_set,
                                extra = extra, wildcard_list = wildcard_list,
                                condense = True)
    # Broadcast the dataframe to the correct size given the number of samples
    if multiple_samples:
        df.set_index(glycan_col_name, inplace = True)
        df.columns = [label_col_name]*len(df.columns)
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
    ttests = [ttest_ind(df_pos.iloc[:, k].values.tolist()+[1],
                        df_neg.iloc[:, k].values.tolist()+[1],
                        equal_var = False)[1]/2 if np.mean(df_pos.iloc[:, k]) > np.mean(df_neg.iloc[:, k]) else 1.0 for k in range(0,
                                                                                                                                   df_motif.shape[1]-1)]
    ttests_corr = multipletests(ttests, method = 'fdr_bh')[1].tolist()
    out = pd.DataFrame(list(zip(df_motif.columns.tolist()[:-1], ttests, ttests_corr)))
    out.columns = ['motif', 'pval', 'corr_pval']
    if sorting:
        return out.sort_values(by = ['corr_pval', 'pval'])
    else:
        return out


def get_representative_substructures(enrichment_df, libr = None):
    """builds minimal glycans that contain enriched motifs from get_pvals_motifs\n
    | Arguments:
    | :-
    | enrichment_df (dataframe): output from get_pvals_motifs
    | libr (dict): dictionary of form glycoletter:index\n
    | Returns:
    | :-
    | Returns up to 10 minimal glycans in a list
    """
    if libr is None:
        libr = lib
    glycans = list(set(df_species.target))
    # Only consider motifs that are significantly enriched
    filtered_df = enrichment_df[enrichment_df.corr_pval < 0.05].reset_index(drop = True)
    pvals = filtered_df.pval.values.tolist()
    weights = -np.log10(pvals) / max(-np.log10(pvals))
    motifs = filtered_df.motif.values.tolist()

    # Pair glycoletters & disaccharides with their pvalue-based weight
    mono, mono_weights = list(zip(*[(motifs[k], weights[k]) for k in range(len(motifs)) if '(' not in motifs[k]]))
    di, di_weights = list(zip(*[(motifs[k], weights[k]) for k in range(len(motifs)) if '(' in motifs[k]]))
    mono_scores = [sum([mono_weights[j] for j in range(len(mono)) if mono[j] in k]) for k in glycans]
    di_scores = [sum([di_weights[j] for j in range(len(di)) if subgraph_isomorphism(k, di[j],
                                                                                    libr = libr)]) for k in glycans]
    # For each glycan, get their glycoletter & disaccharide scores, normalized by glycan length
    motif_scores = [a + b for a, b in zip(mono_scores, di_scores)]
    length_scores = [len(k) for k in glycans]

    combined_scores = [motif_scores[k] / length_scores[k] for k in range(len(motif_scores))]
    df_score = pd.DataFrame(list(zip(glycans, motif_scores, length_scores, combined_scores)),
                        columns = ['glycan', 'motif_score', 'length_score', 'combined_score'])
    df_score = df_score.sort_values(by = 'combined_score',
                                    ascending = False)
    # Take the 10 glycans with the highest score
    rep_motifs = df_score.glycan.values.tolist()[:10]
    rep_motifs.sort(key = len)

    clean_list = []
    # Make sure that the list only contains the minimum number of representative glycans
    for k in rep_motifs:
        if sum([subgraph_isomorphism(j, k, libr = libr) for j in rep_motifs]) > 1:
            rep_motifs.remove(k)
        else:
            clean_list.append(k)
    return clean_list


def clean_up_heatmap(df):
  """removes redundant motifs from the dataframe\n
  | Arguments:
  | :-
  | df (dataframe): dataframe with glycan data, rows are glycan motifs and columns are samples\n
  | Returns:
  | :-
  | Returns a cleaned up dataframe of the same format as the input
  """
  df.index = [k+' '*20 if k in motif_list.motif_name else k for k in df.index]
  # Group the DataFrame by identical rows
  grouped = df.groupby(list(df.columns))
  # Find the row with the longest string index within each group and return a new DataFrame
  result = pd.concat(
        [
            group.loc[group.index.to_series().str.len().idxmax()]
            for _, group in grouped
        ],
        axis = 1,
    ).T
  result.index = [k.strip() for k in result.index]
  return result


def get_heatmap(df, mode = 'sequence', feature_set = ['known'],
                 extra = 'termini', wildcard_list = [], datatype = 'response',
                 rarity_filter = 0.05, filepath = '', index_col = 'target',
                 **kwargs):
  """clusters samples based on glycan data (for instance glycan binding etc.)\n
  | Arguments:
  | :-
  | df (dataframe): dataframe with glycan data, rows are samples and columns are glycans
  | mode (string): whether glycan 'sequence' or 'motif' should be used for clustering; default:sequence
  | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'exhaustive'; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), and 'chemical' (molecular properties of glycan)
  | extra (string): 'ignore' skips this, 'wildcards' allows for wildcard matching', and 'termini' allows for positional matching; default:'termini'
  | wildcard_list (list): list of wildcard names (such as '?1-?', 'Hex', 'HexNAc', 'Sia')
  | datatype (string): whether df comes from a dataset with quantitative variable ('response') or from presence_to_matrix ('presence')
  | rarity_filter (float): proportion of samples that need to have a non-zero value for a variable to be included; default:0.05
  | filepath (string): absolute path including full filename allows for saving the plot
  | index_col (string): default column to convert to dataframe index; default:'target'
  | **kwargs: keyword arguments that are directly passed on to seaborn clustermap\n                      
  | Returns:
  | :-
  | Prints clustermap              
  """
  if index_col in df.columns:
      df.set_index(index_col, inplace = True)
  df.fillna(0, inplace = True)
  if mode == 'motif':
      # Count glycan motifs and remove rare motifs from the result
      df_motif = annotate_dataset(df.columns.tolist(),
                                  feature_set = feature_set,
                                  extra = extra, wildcard_list = wildcard_list,
                                  condense = True)
      df_motif = df_motif.replace(0, np.nan).dropna(thresh = np.max([np.round(rarity_filter * df_motif.shape[0]), 1]), axis = 1)
      collect_dic = {}
      # Distinguish the case where the motif abundance is paired to a quantitative value or a qualitative variable
      if datatype == 'response':
        for c, col in enumerate(df_motif.columns):
          indices = [i for i, x in enumerate(df_motif[col]) if x >= 1]
          temp = df.iloc[:, indices]
          temp.columns = range(temp.columns.size)
          collect_dic[col] = (temp * df_motif.iloc[indices, c].reset_index(drop = True)).sum(axis = 1)
        df = pd.DataFrame(collect_dic)
      elif datatype == 'presence':
        idx = df.index.tolist()
        collecty = [[np.sum(df.iloc[row, [i for i, x in enumerate(df_motif[col]) if x >= 1]])/df.iloc[row, :].values.sum() for col in df_motif.columns] for row in range(df.shape[0])]
        df = pd.DataFrame(collecty)
        df.columns = df_motif.columns
        df.index = idx
  df.dropna(axis = 1, inplace = True)
  df = clean_up_heatmap(df.T)
  for col in df.columns:
      col_sum = max([sum(df.loc[:, col]), 0.1])
      df[col] = [k/col_sum*100 for k in df.loc[:, col]]
  # Cluster the motif abundances
  sns.clustermap(df, **kwargs)
  plt.xlabel('Samples')
  if mode == 'sequence':
      plt.ylabel('Glycans')
  else:
      plt.ylabel('Motifs')
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
    idx = [k for k in range(len(glycans)) if '{' not in glycans[k]]
    glycans = [glycans[k] for k in idx]
    label_list = [label_list[k] for k in idx]
    # Get all glycan embeddings
    if emb is None:
        this_dir, this_filename = os.path.split(__file__)
        data_path = os.path.join(this_dir, 'glycan_representations.pkl')
        emb = pickle.load(open(data_path, 'rb'))
    # Get the subset of embeddings corresponding to 'glycans'
    if isinstance(emb, pd.DataFrame):
        emb = {glycans[k]: emb.iloc[k, :] for k in range(len(glycans))}
    embs = np.array([emb[k] for k in glycans])
    # Calculate t-SNE of embeddings
    embs = TSNE(random_state = 42,
                init = 'pca', learning_rate = 'auto').fit_transform(embs)
    # Plot the t-SNE
    markers = None
    if shape_feature is not None:
        markers = {shape_feature: "X", "Absent": "o"}
        shape_feature = [shape_feature if shape_feature in k else 'Absent' for k in glycans]
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


def characterize_monosaccharide(sugar, df = None, mode = 'sugar', glycan_col_name = 'target',
                                rank = None, focus = None, modifications = False,
                                filepath = '', thresh = 10):
  """for a given monosaccharide/linkage, return typical neighboring linkage/monosaccharide\n
  | Arguments:
  | :-
  | sugar (string): monosaccharide or linkage
  | df (dataframe): dataframe to use for analysis; default:df_species
  | mode (string): either 'sugar' (connected monosaccharides), 'bond' (monosaccharides making a provided linkage), or 'sugarbond' (linkages that a provided monosaccharides makes); default:'sugar'
  | glycan_col_name (string): column name under which glycans can be found; default:'target'
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
  if rank is not None:
    df = df[df[rank] == focus]
  # Get all disaccharides for linkage analysis
  pool_in = unwrap([link_find(k) for k in df[glycan_col_name]])
  pool_in = [k.replace('(', '*').replace(')', '*') for k in pool_in]

  if mode == 'bond':
    # Get upstream monosaccharides for a specific linkage
    pool = [k.split('*')[0] for k in pool_in if k.split('*')[1] == sugar]
    lab = 'Observed Monosaccharides Making Linkage %s' % sugar
  elif mode == 'sugar':
    # Get downstream monosaccharides for a specific monosaccharide
    if modifications:
      sugars = [k.split('*')[0] for k in pool_in if sugar in k.split('*')[0]]
      pool = [k.split('*')[2] for k in pool_in if sugar in k.split('*')[0]]
    else:
      pool = [k.split('*')[2] for k in pool_in if k.split('*')[0] == sugar]
    lab = 'Observed Monosaccharides Paired with %s' % sugar
  elif mode == 'sugarbond':
    # Get downstream linkages for a specific monosaccharide
    if modifications:
      sugars = [k.split('*')[0] for k in pool_in if sugar in k.split('*')[0]]
      pool = [k.split('*')[1] for k in pool_in if sugar in k.split('*')[0]]
    else:
      pool = [k.split('*')[1] for k in pool_in if k.split('*')[0] == sugar]
    lab = 'Observed Linkages Made by %s' % sugar

  # Count objects in pool, filter by rarity, and calculate proportion
  cou = Counter(pool).most_common()
  cou_k = [k[0] for k in cou if k[1] > thresh]
  cou_v_in = [k[1] for k in cou if k[1] > thresh]
  cou_v = [v / len(pool) for v in cou_v_in]

  # Start plotting
  fig, (a0, a1) = plt.subplots(1, 2, figsize = (8, 4), gridspec_kw = {'width_ratios': [1, 1]})
  if modifications:
      if mode == 'bond':
          print("Modifications currently only work in mode == 'sugar' and mode == 'sugarbond'.")
      # Get counts and proportions for the input monosaccharide + its modifications
      cou2 = Counter(sugars).most_common()
      cou_k2 = [k[0] for k in cou2 if k[1] > thresh]
      cou_v2_in = [k[1] for k in cou2 if k[1] > thresh]
      cou_v2 = [v / len(sugars) for v in cou_v2_in]
      # Map the input monosaccharide + its modifications to colors
      color_list = plt.cm.get_cmap('tab20')
      color_map = {cou_k2[k]: color_list(k/len(cou_k2)) for k in range(len(cou_k2))}
      palette = [color_map[k] for k in cou_k2]
      # Start linking downstream monosaccharides / linkages to the input monosaccharide + its modifications
      if mode == 'sugar':
          pool_in2 = [k for k in pool_in if k.split('*')[2] in cou_k]
      elif mode == 'sugarbond':
          pool_in2 = [k for k in pool_in if k.split('*')[1] in cou_k]
      cou_for_df = []
      for k in range(len(cou_k2)):
          if mode == 'sugar':
              idx = [j.split('*')[2] for j in pool_in2 if j.split('*')[0] == cou_k2[k]]
          elif mode == 'sugarbond':
              idx = [j.split('*')[1] for j in pool_in2 if j.split('*')[0] == cou_k2[k]]
          cou_t = Counter(idx).most_common()
          cou_v_t = {cou_t[j][0]: cou_t[j][1] for j in range(len(cou_t))}
          cou_v_t = [cou_v_t[j] if j in list(cou_v_t.keys()) else 0 for j in cou_k]
          if len(cou_k2) > 1:
              cou_for_df.append(pd.DataFrame({'monosaccharides': cou_k, 'counts': cou_v_t, 'colors': [cou_k2[k]]*len(cou_k)}))
          else:
              sns.barplot(x = cou_k, y = cou_v_t, ax = a1, color = "cornflowerblue")
      if len(cou_k2) > 1:
          cou_df = pd.concat(cou_for_df).reset_index(drop = True)
          sns.histplot(data = cou_df, x = 'monosaccharides', hue = 'colors', weights = 'counts',
                       multiple = 'stack', palette = palette, ax = a1, legend = False,
                   shrink = 0.8)
      a1.set_ylabel('Absolute Occurrence')
  else:
      sns.barplot(x = cou_k, y = cou_v, ax = a1, color = "cornflowerblue")
      a1.set_ylabel('Relative Proportion')
  sns.despine(left = True, bottom = True)
  a1.set_xlabel('')
  a1.set_title(str(sugar) + ' and variants are connected to')
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
    a0.set_title('Observed Modifications of ' + str(sugar))
    plt.setp(a1.get_xticklabels(), rotation = 'vertical')

  fig.suptitle('Characterizing ' + str(sugar))
  fig.tight_layout()
  if filepath:
      plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300,
                  bbox_inches = 'tight')
  plt.show()


def hotellings_t2(group1, group2, paired = False):
  """Hotelling's T^2 test (the t-test for multivariate comparisons)\n
  """
  if paired:
    assert group1.shape == group2.shape, "For paired samples, the size of group1 and group2 should be the same"
    group1 = np.array(group1) - np.array(group2)
    group2 = None
  # Calculate the means and covariances of each group
  n1, p = group1.shape
  mean1 = np.mean(group1, axis = 0)
  cov1 = np.cov(group1.T)
  if group2 is not None:
    n2, _ = group2.shape
    mean2 = np.mean(group2, axis = 0)
    cov2 = np.cov(group2.T)
  else:
    n2 = 0
    mean2 = np.zeros_like(mean1)
    cov2 = np.zeros_like(cov1)
  # Calculate the difference between the means
  diff = mean1 - mean2
  # Calculate the pooled covariance matrix
  denom = (n1 + n2 - 2)
  if denom < 1:
      pooled_cov = cov1
  else:
      pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / denom
      pooled_cov += np.eye(p) * 1e-6
  # Calculate the Hotelling's T^2 statistic
  T2 = (n1 * n2) / (n1 + n2) * diff @ np.linalg.pinv(pooled_cov) @ diff.T
  # Convert the T^2 statistic to an F statistic
  if denom < 1:
      F = 0
  else:
      F = T2 * (n1 + n2 - p - 1) / (denom * p)
  if F == 0:
    return F, 1.0
  # Calculate the p-value of the F statistic
  p_value = f.sf(F, p, n1 + n2 - p - 1)
  return F, p_value


def get_coverage(df, filepath = ''):
  """ Plot glycan coverage across samples, ordered by average intensity\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns
  | filepath (string): absolute path including full filename allows for saving the plot\n
  | Returns:
  | :-
  | Prints the heatmap
  """

  d = df.iloc[:,1:]

  # arrange by mean intensity across all samples
  order = d.mean(axis = 1).rank(ascending = True).sort_values().index

  # plot figure
  ax = sns.heatmap(d.loc[order], cmap = sns.cubehelix_palette(as_cmap=True, start = 3, dark = 0.50, light = 0.90),
                   cbar_kws={'label': 'Relative Intensity'}, cbar = True, mask = d.loc[order] == 0)

  ax.set(xlabel = 'Samples', ylabel =  'Glycan ID', title = '')

  # save figure
  if filepath:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300, bbox_inches = 'tight')

  plt.show()
  

def get_pca(df, groups = None, motifs = False, feature_set = ['known', 'exhaustive'],
            pc_x = 1, pc_y = 2, color = None, shape = None, filepath = ''):
  """ PCA plot from glycomics abundance dataframe\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns
  | groups (list): a list of group identifiers for each sample (e.g., [1,1,1,2,2,2,3,3,3]); default:None
  |                     alternatively: design dataframe with 'id' column of samples names and additional columns with meta information
  | motifs (bool): whether to analyze full sequences (False) or motifs (True); default:False
  | feature_set (list): which feature set to use for annotations, add more to list to expand; default is ['exhaustive','known']; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), and 'chemical' (molecular properties of glycan)
  | pc_x (int): principal component to plot on x axis; default:1
  | pc_y (int): principal component to plot on y axis; default:2
  | color (string): if design dataframe is provided: column name for color grouping; default:None
  | shape (string): if design dataframe is provided: column name for shape grouping; default:None
  | filepath (string): absolute path including full filename allows for saving the plot\n
  | Returns:
  | :-
  | Prints PCA plot
  """
  # get pca
  if motifs:
      # Motif extraction and quantification
      df = quantify_motifs(df.iloc[:, 1:], df.iloc[:, 0].values.tolist(), feature_set).T.reset_index()
  X = np.array(df.iloc[:, 1:].T)
  scaler = StandardScaler()
  X_std = scaler.fit_transform(X)
  pca = PCA()
  X_pca = pca.fit_transform(X_std)
  percent_var = list(pca.explained_variance_ratio_*100)
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0)
  ax.set(xlabel = 'PC'+ str(pc_x) + ': ' + str(round(percent_var[pc_x-1])) + '% variance' , 
        ylabel =  'PC'+ str(pc_y) + ': ' + str(round(percent_var[pc_y-1])) + '% variance' , 
        title = '' )
  
  # save to file
  if filepath:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300, bbox_inches = 'tight')
  
  plt.show()


def get_differential_expression(df, group1, group2,
                                motifs = False, feature_set = ['exhaustive', 'known'], paired = False,
                                impute = True, sets = False, set_thresh = 0.9, effect_size_variance = False,
                                min_samples = None):
  """Calculates differentially expressed glycans or motifs from glycomics data\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns
  | group1 (list): list of column indices or names for the first group of samples, usually the control
  | group2 (list): list of column indices or names for the second group of samples
  | motifs (bool): whether to analyze full sequences (False) or motifs (True); default:False
  | feature_set (list): which feature set to use for annotations, add more to list to expand; default is ['exhaustive','known']; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), and 'chemical' (molecular properties of glycan)
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False
  | impute (bool): replaces zeroes with draws from left-shifted distribution or KNN-Imputer; default:True
  | sets (bool): whether to identify clusters of highly correlated glycans/motifs to test for differential expression; default:False
  | set_thresh (float): correlation value used as a threshold for clusters; only used when sets=True; default:0.9
  | effect_size_variance (bool): whether effect size variance should also be calculated/estimated; default:False
  | min_samples (int): How many samples per group need to have non-zero values for glycan to be kept; default: at least half per group\n
  | Returns:
  | :-
  | Returns a dataframe with:
  | (i) Differentially expressed glycans/motifs/sets
  | (ii) Their mean abundance across all samples in group1 + group2
  | (iii) Log2-transformed fold change of group2 vs group1 (i.e., negative = lower in group2)
  | (iv) Uncorrected p-values (Welch's t-test) for difference in mean
  | (v) Corrected p-values (Welch's t-test with Benjamini-Hochberg correction) for difference in mean
  | (vi) Corrected p-values (Levene's test for equality of variances with Benjamini-Hochberg correction) for difference in variance
  | (vii) Effect size as Cohen's d (sets=False) or Mahalanobis distance (sets=True)
  | (viii) [only if effect_size_variance=True] Effect size variance
  """
  if isinstance(group1[0], str):
    pass
  else:
    group1 = [df.columns.tolist()[k] for k in group1]
    group2 = [df.columns.tolist()[k] for k in group2]
  df = df.loc[:, [df.columns.tolist()[0]]+group1+group2]
  df.fillna(0, inplace = True)
  df = impute_and_normalize(df, [group1, group2], impute = impute, min_samples = min_samples)
  glycans = df.iloc[:, 0].values.tolist()
  if motifs:
    # Motif extraction and quantification
    df = quantify_motifs(df.iloc[:, 1:], glycans, feature_set)
    # Deduplication
    df = clean_up_heatmap(df.T)
    # Re-normalization
    for col in df.columns:
      df[col] = [k/sum(df.loc[:, col])*100 for k in df.loc[:, col]]
  else:
    df.set_index(df.columns.tolist()[0], inplace = True)
    df = df.groupby(df.index).mean()
      
  # Variance-based filtering of features
  df = variance_based_filtering(df)
  glycans = df.index.tolist()
  mean_abundance = df.mean(axis = 1)
    
  df_a = df.loc[:, group1]
  df_b = df.loc[:, group2]
  if sets:
    # Motif/sequence set enrichment
    df2 = variance_stabilization(df, groups = [group1, group2])
    clusters = create_correlation_network(df2.T, set_thresh)
    glycans, mean_abundance_c, pvals, fc, levene_pvals, effect_sizes, variances = [], [], [], [], [], [], []
    # Testing differential expression of each set/cluster
    for cluster in clusters:
      if len(cluster) > 1:
        glycans.append(cluster)
        gp1 = df_a.loc[list(cluster), :]
        gp2 = df_b.loc[list(cluster), :]
        mean_abundance_c.append(mean_abundance.loc[list(cluster)].mean())
        if paired:
            fc.append(np.log2(((gp2.values + 1e-8) / (gp1.values + 1e-8)).mean(axis = 1)).mean())
        else:
            fc.append(np.log2((gp2.mean(axis = 1) + 1e-8) / (gp1.mean(axis = 1) + 1e-8)).mean())
        gp1 = df2.loc[list(cluster), group1]
        gp2 = df2.loc[list(cluster), group2]
        # Hotelling's T^2 test for multivariate comparisons
        pvals.append(hotellings_t2(gp1.T.values, gp2.T.values, paired = paired)[1])
        levene_pvals.append(np.mean([levene(gp1.loc[variable, :], gp2.loc[variable, :])[1] for variable in cluster]))
        # Calculate Mahalanobis distance as measure of effect size for multivariate comparisons
        effect_sizes.append(mahalanobis_distance(gp1, gp2, paired = paired))
        if effect_size_variance:
          variances.append(mahalanobis_variance(gp1, gp2, paired = paired))
    mean_abundance = mean_abundance_c
  else:
    if paired:
        fc = np.log2(((df_b.values + 1e-8)/ (df_a.values + 1e-8)).mean(axis = 1)).tolist()
    else:
        fc = np.log2(df_b.mean(axis = 1) / df_a.mean(axis = 1)).tolist()
    df_a = variance_stabilization(df_a)
    df_b = variance_stabilization(df_b)
    if paired:
        assert len(group1) == len(group2), "For paired samples, the size of group1 and group2 should be the same"
        pvals = [ttest_rel(df_a.iloc[k, :], df_b.iloc[k, :])[1] for k in range(len(df_a))]
    else:
        pvals = [ttest_ind(df_a.iloc[k, :], df_b.iloc[k, :], equal_var = False)[1] for k in range(len(df_a))]
    levene_pvals = [levene(df_a.iloc[k, :], df_b.iloc[k, :])[1] for k in range(len(df_a))]
    effect_sizes, variances = zip(*[cohen_d(df_b.iloc[k, :], df_a.iloc[k, :], paired = paired) for k in range(len(df_a))])
  # Multiple testing correction
  if pvals:
      corrpvals = multipletests(pvals, method = 'fdr_bh')[1]
      levene_pvals = multipletests(levene_pvals, method = 'fdr_bh')[1]
  else:
      corrpvals = []
  out = [(glycans[k], mean_abundance[k], fc[k], pvals[k], corrpvals[k], levene_pvals[k], effect_sizes[k]) for k in range(len(glycans))]
  out = pd.DataFrame(out)
  out.columns = ['Glycan', 'Mean abundance', 'Log2FC', 'p-val', 'corr p-val', 'corr Levene p-val', 'Effect size']
  if effect_size_variance:
      out['Effect size variance'] = variances
  out = out.dropna()
  return out.sort_values(by = 'corr p-val')


def get_pval_distribution(df_res, filepath = ''):
  """ p-value distribution plot of glycan differential expression result\n
  | Arguments:
  | :-
  | df_res (dataframe): output from get_differential_expression
  | filepath (string): absolute path including full filename allows for saving the plot\n
  | Returns:
  | :-
  | prints p-value distribution plot
  """  
  # make plot
  ax = sns.histplot(x = 'p-val', data = df_res, stat = 'frequency')
  ax.set(xlabel = 'p-values', 
        ylabel =  'Frequency', 
        title = '')
  
  # save to file
  if filepath:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300, bbox_inches = 'tight')
  
  plt.show()  


def get_ma(df_res, log2fc_thresh = 1, sig_thresh = 0.05, filepath = ''):
  """ MA plot of glycan differential expression result\n
  | Arguments:
  | :-
  | df_res (dataframe): output from get_differential_expression
  | log2fc_thresh (int): absolute Log2FC threshold for highlighting datapoints
  | sig_thresh (int): significance threshold for highlighting datapoints
  | filepath (string): absolute path including full filename allows for saving the plot\n
  | Returns:
  | :-
  | prints MA plot
  """
  # reorder to draw significant glycans/motifs last
  df_res = df_res.sort_values('corr p-val', ascending = False)
  
  # color points by log2fc/pval
  custom_palette = []
  alphas = []
  for k in range(len(df_res)):
    if abs(df_res['Log2FC'].values.tolist()[k]) > log2fc_thresh and df_res['corr p-val'].values.tolist()[k] < sig_thresh:
      custom_palette.append('#CC4446')
      alphas.append(0.95)
    else:
      custom_palette.append('grey')
      alphas.append(0.5)  
  
  # make plot
  ax = sns.scatterplot(x = 'Mean abundance', y = 'Log2FC', data = df_res, c = custom_palette, alpha = alphas)
  ax.set(xlabel = 'Mean Abundance',
        ylabel =  'Log2FC',
        title = '')
  
  # save to file
  if filepath:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300, bbox_inches = 'tight')
  
  plt.show()  


def get_volcano(df_res, y_thresh = 0.05, x_thresh = 1.0,
                 label_changed = True, x_metric = 'Log2FC', filepath = ''):
  """Plots glycan differential expression results in a volcano plot\n
  | Arguments:
  | :-
  | df_res (dataframe): output from get_differential_expression
  | y_thresh (float): corr p threshhold for labeling datapoints; default:0.05
  | x_thresh (float): absolute x metric threshold for labeling datapoints; default:1.0
  | label_changed (bool): if True, add text labels to significantly up- and downregulated datapoints; default:True
  | x_metric (string): x-axis metric; default:'Log2FC'; options are 'Log2FC', 'Effect size'
  | filepath (string): absolute path including full filename allows for saving the plot\n
  | Returns:
  | :-
  | Prints volcano plot
  """

  df_res['log_p'] = -np.log10(df_res['corr p-val'].values.tolist())
  x = df_res[x_metric].values.tolist()
  y = df_res['log_p'].values.tolist()
  label = df_res['Glycan'].values.tolist()

  # Make plot
  ax = sns.scatterplot(x = x_metric, y = 'log_p', data = df_res, color = '#3E3E3E', alpha = 0.8)
  ax.set(xlabel = x_metric, ylabel = '-log10(corr p-val)', title = '')
  plt.axhline(y = -np.log10(y_thresh), c = 'k', ls = ':', lw = 0.5, alpha = 0.3)
  plt.axvline(x = x_thresh, c = 'k', ls = ':', lw = 0.5, alpha = 0.3)
  plt.axvline(x = -x_thresh, c = 'k', ls = ':', lw = 0.5, alpha = 0.3)
  sns.despine(bottom = True, left = True)

  # Text labels
  if label_changed:
    [plt.text(x[i], y[i], label[i]) for i in range(len(x)) if y[i] > -np.log10(y_thresh) and abs(x[i]) > x_thresh]

  # Save to file
  if filepath:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300,
                bbox_inches = 'tight')

  plt.show()


def get_glycanova(df, groups, impute = True, motifs = False, feature_set = ['exhaustive', 'known'], min_samples = None, posthoc = True):
    """Calculate an ANOVA for each glycan (or motif) in the DataFrame\n
    | Arguments:
    | :-
    | df (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns
    | group_sizes (list): a list of group identifiers for each sample (e.g., [1,1,1,2,2,2,3,3,3])
    | impute (bool): replaces zeroes with draws from left-shifted distribution or KNN-Imputer; default:True
    | motifs (bool): whether to analyze full sequences (False) or motifs (True); default:False
    | feature_set (list): which feature set to use for annotations, add more to list to expand; default is ['exhaustive','known']; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), and 'chemical' (molecular properties of glycan)
    | min_samples (int): How many samples per group need to have non-zero values for glycan to be kept; default: at least half per group
    | posthoc (bool): whether to do Tukey's HSD test post-hoc to find out which differences were significant; default:True\n
    | Returns:
    | :-
    | (i) a pandas DataFrame with an F statistic and corrected p-value for each glycan.
    | (ii) a dictionary of type glycan : pandas DataFrame, with post-hoc results for each glycan with a significant ANOVA.
    """
    results = []
    posthoc_results = {}
    df.fillna(0, inplace = True)
    groups_unq = sorted(set(groups))
    df = impute_and_normalize(df, [[df.columns.tolist()[i+1] for i, x in enumerate(groups) if x == g] for g in groups_unq], impute = impute,
                              min_samples = min_samples)
    glycans = df.iloc[:,0].values.tolist()
    if motifs:
        df = quantify_motifs(df.iloc[:, 1:], glycans, feature_set)
        # Deduplication
        df = clean_up_heatmap(df.T)
        # Re-normalization
        for col in df.columns:
            df[col] = [k/sum(df.loc[:, col])*100 for k in df.loc[:, col]]
    else:
        df.set_index(df.columns.tolist()[0], inplace = True)
        df = df.groupby(df.index).mean()
    # Variance-based filtering of features
    df = variance_based_filtering(df)
    col_names = [[col for group, col in zip(groups, df.columns) if group == unique_group]
                 for unique_group in sorted(set(groups))]
    df = variance_stabilization(df, groups = col_names)
    glycans = df.index.tolist()
    for glycan in glycans:
        # Create a DataFrame with the glycan abundance and group identifier for each sample
        data = pd.DataFrame({"Abundance": df.loc[glycan], "Group": groups})
        # Run an ANOVA
        model = ols("Abundance ~ C(Group)", data = data).fit()
        anova_table = sm.stats.anova_lm(model, typ = 2)
        f_value = anova_table["F"]["C(Group)"]
        p_value = anova_table["PR(>F)"]["C(Group)"]
        results.append((glycan, f_value, p_value))
        if p_value < 0.05 and posthoc:
            posthoc = pairwise_tukeyhsd(endog = data['Abundance'], groups = data['Group'], alpha = 0.05)
            posthoc_results[glycan] = pd.DataFrame(data = posthoc._results_table.data[1:], columns = posthoc._results_table.data[0])

    results_df = pd.DataFrame(results, columns = ["Glycan", "F statistic", "corr p-val"])
    results_df['corr p-val'] = multipletests(results_df['corr p-val'].values.tolist(),
                                             method = 'fdr_bh')[1]
    return results_df.sort_values(by = 'corr p-val'), posthoc_results


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
    combined_effect_size = np.sum(weights * effect_sizes) / np.sum(weights)

    if model == 'random':
        # Estimate between-study variance (tau squared) using DerSimonian-Laird method
        q = np.sum(weights * (effect_sizes - combined_effect_size)**2)
        df = len(effect_sizes) - 1
        c = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
        tau_squared = max((q - df) / (c + 1e-8), 0)
        # Update weights for tau_squared
        weights = 1 / (variances + tau_squared)
        # Recalculate combined effect size
        combined_effect_size = np.sum(weights * effect_sizes) / (np.sum(weights) + 1e-8)

    # Calculate standard error and z-score
    se = np.sqrt(1 / (np.sum(weights) + 1e-8))
    z = combined_effect_size / (se + 1e-8)

    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))

    # Check whether Forest plot should be constructed and saved
    if len(filepath) > 1:
        df_temp = pd.DataFrame([study_names, effect_sizes, variances]).T
        df_temp.columns = ['Study', 'EffectSize', 'EffectSizeVariance']
        # sort studies by effect size
        df_temp = df_temp.sort_values(by = 'EffectSize', key = abs, ascending = False)
        # calculate standard error
        standard_error = np.sqrt(df_temp.EffectSizeVariance.values.tolist())
        # calculate the confidence interval
        df_temp['lower'] = df_temp.EffectSize.values - 1.96 * standard_error
        df_temp['upper'] = df_temp.EffectSize.values + 1.96 * standard_error
        # Create a new figure and a axes to plot on
        fig, ax = plt.subplots(figsize = (8, df_temp.shape[0]*0.6))  # adjust the size as needed
        y_pos = np.arange(df_temp.shape[0])
        # Draw a horizontal line for each study, with x-values between the lower and upper confidence bounds
        ax.hlines(y_pos, df_temp['lower'], df_temp['upper'], color = 'skyblue')
        # Draw a marker at the effect size
        ax.plot(df_temp['EffectSize'], y_pos, 'o', color = 'skyblue')
        # Draw a vertical line at x=0 (or change to the desired reference line)
        ax.vlines(0, -1, len(df_temp), colors = 'gray', linestyles = 'dashed')
        # Label the y-axis with the study names
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_temp['Study'])
        # Invert y-axis so that studies are displayed top-down
        ax.invert_yaxis()
        # Label the x-axis and give the plot a title
        ax.set_xlabel('Effect size')
        # Remove plot borders
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
    time = data[:, 0]
    glycan_abundance = data[:, 1]
    
    if degree == 1:
        # Add a constant (for the intercept term)
        time_with_intercept = sm.add_constant(time)

        # Fit the OLS model
        model = sm.OLS(glycan_abundance, time_with_intercept)
        results = model.fit()

        # Get the slope & the p-value for the slope from the model summary
        coefficients = results.params[1]
        p_value = results.pvalues[1]

    else:
        # Polynomial Regression
        coeffs = np.polyfit(time, glycan_abundance, degree)
        p = np.poly1d(coeffs)

        # Calculate the residuals
        residuals = glycan_abundance - p(time)
        
        # Perform F-test to get p_value
        _, p_value = f_oneway(glycan_abundance, residuals)
        coefficients = coeffs

    return coefficients, p_value


def get_time_series(df, impute = True, motifs = False, feature_set = ['known', 'exhaustive'], degree = 1, min_samples = None):
    """Analyzes time series data of glycans using an OLS model\n
    | Arguments:
    | :-
    | df (dataframe): dataframe containing sample IDs of style sampleID_UnitTimepoint_replicate (e.g., T1_h5_r1) in first column and glycan relative abundances in subsequent columns
    | impute (bool): replaces zeroes with draws from left-shifted distribution or KNN-Imputer; default:True
    | motifs (bool): whether to analyze full sequences (False) or motifs (True); default:False
    | feature_set (list): which feature set to use for annotations, add more to list to expand; default is ['exhaustive','known']; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), and 'chemical' (molecular properties of glycan)
    | degree (int): degree of the polynomial for regression, default:1 for linear regression
    | min_samples (int): How many samples per group need to have non-zero values for glycan to be kept; default: at least half per group\n
    | Returns:
    | :-
    | Returns a dataframe with:
    | (i) Glycans/motifs potentially exhibiting significant changes over time
    | (ii) The slope of their expression curve over time
    | (iii) Uncorrected p-values (t-test) for testing whether slope is significantly different from zero
    | (iv) Corrected p-values (t-test with Benjamini-Hochberg correction) for testing whether slope is significantly different from zero
    """
    df.fillna(0, inplace = True)
    df = df.set_index(df.columns.tolist()[0]).T
    df = impute_and_normalize(df, [df.columns.tolist()[1:]], impute = impute, min_samples = min_samples)
    df = df.reset_index()
    glycans = df.iloc[:, 0].values.tolist()
    glycans = [k.split('.')[0] for k in glycans]
    if motifs:
        df = quantify_motifs(df.iloc[:, 1:], glycans, feature_set)
        # Deduplication
        df = clean_up_heatmap(df.T)
        # Re-normalization
        for col in df.columns:
            df[col] = [k/sum(df.loc[:, col])*100 for k in df.loc[:, col]]
    else:
        df.index = glycans
        df.drop([df.columns.tolist()[0]], axis = 1, inplace = True)
        df = df.groupby(df.index).mean()
    df = df.T.reset_index()
    df[df.columns.tolist()[0]] = [float(k.split('_')[1][1:]) for k in df.iloc[:, 0].values.tolist()]
    df = df.sort_values(by = df.columns.tolist()[0])
    time = np.array(df.iloc[:, 0].values.tolist())  # Time points
    res = []
    for c in df.columns.tolist()[1:]:
        glycan_abundance = np.array(df.loc[:, c].values.tolist())  # Glycan abundances for each time point and replicate
        data = np.column_stack((time, glycan_abundance))
        s, p = get_glycan_change_over_time(data, degree = degree)
        res.append((c, s, p))
    res = pd.DataFrame(res)
    res.columns = ['Glycan', 'Change', 'p-val']
    res['corr p-val'] = multipletests(res['p-val'].values.tolist(), method = 'fdr_bh')[1]
    return res.sort_values(by = 'corr p-val')
