import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('default')
from collections import Counter
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.manifold import TSNE

from glycowork.glycan_data.loader import lib, glycan_emb, df_species
from glycowork.motif.annotate import annotate_dataset
from glycowork.motif.tokenization import link_find


def get_pvals_motifs(df, glycan_col_name = 'glycan', label_col_name = 'target',
                     libr = None, thresh = 1.645, sorting = True,
                     feature_set = ['exhaustive'], extra = 'termini',
                     wildcard_list = [], multiple_samples = False):
    """returns enriched motifs based on label data or predicted data\n
    | Arguments:
    | :-
    | df (dataframe): dataframe containing glycan sequences and labels
    | glycan_col_name (string): column name for glycan sequences; arbitrary if multiple_samples = True; default:'glycan'
    | label_col_name (string): column name for labels; arbitrary if multiple_samples = True; default:'target'
    | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset
    | thresh (float): threshold value to separate positive/negative; default is 1.645 for Z-scores
    | sorting (bool): whether p-value dataframe should be sorted ascendingly; default: True
    | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'exhaustive'; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans) and 'exhaustive' (all mono- and disaccharide features)
    | extra (string): 'ignore' skips this, 'wildcards' allows for wildcard matching', and 'termini' allows for positional matching; default:'termini'
    | wildcard_list (list): list of wildcard names (such as 'bond', 'Hex', 'HexNAc', 'Sia')
    | multiple_samples (bool): set to True if you have multiple samples (rows) with glycan information (columns); default:False\n
    | Returns:
    | :-
    | Returns dataframe with p-values and corrected p-values for every glycan motif
    """
    if libr is None:
        libr = lib
    if multiple_samples:
        if 'target' in df.columns.values.tolist():
          df.drop(['target'], axis = 1, inplace = True)
        df = df.T
        samples = df.shape[1]
        df = df.reset_index()
        df.columns = [glycan_col_name] + df.columns.values.tolist()[1:]
    df_motif = annotate_dataset(df[glycan_col_name].values.tolist(),
                                libr = lib, feature_set = feature_set,
                               extra = extra, wildcard_list = wildcard_list)
    if multiple_samples:
        df.index = df[glycan_col_name].values.tolist()
        df = df.drop([glycan_col_name], axis = 1)
        df = pd.concat([df.iloc[:,k] for k in range(len(df.columns.values.tolist()))], axis = 0)
        df = df.reset_index()
        df.columns = [glycan_col_name, label_col_name]
        df_motif = pd.concat([df_motif]*samples, axis = 0)
    df_motif[label_col_name] = df[label_col_name].values.tolist()
    df_motif = df_motif.loc[:, (df_motif != 0).any(axis = 0)]
    df_pos = df_motif[df_motif[label_col_name] > thresh]
    df_neg = df_motif[df_motif[label_col_name] <= thresh]
    ttests = [ttest_ind(df_pos.iloc[:,k].values.tolist()+[1],
                        df_neg.iloc[:,k].values.tolist()+[1],
                        equal_var = False)[1]/2 if np.mean(df_pos.iloc[:,k])>np.mean(df_neg.iloc[:,k]) else 1.0 for k in range(0,
                                                                                                                               df_motif.shape[1]-1)]
    ttests_corr = multipletests(ttests, method = 'hs')[1].tolist()
    out = pd.DataFrame(list(zip(df_motif.columns.values.tolist()[:-1], ttests, ttests_corr)))
    out.columns = ['motif', 'pval', 'corr_pval']
    if sorting:
        return out.sort_values(by = ['corr_pval', 'pval'])
    else:
        return out

def make_heatmap(df, mode = 'sequence', libr = None, feature_set = ['known'],
                 extra = 'termini', wildcard_list = [], datatype = 'response',
                 rarity_filter = 0.05, filepath = '', index_col = 'target',
                 **kwargs):
    """clusters samples based on glycan data (for instance glycan binding etc.)\n
    | Arguments:
    | :-
    | df (dataframe): dataframe with glycan data, rows are samples and columns are glycans
    | mode (string): whether glycan 'sequence' or 'motif' should be used for clustering; default:sequence
    | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset
    | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'exhaustive'; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans) and 'exhaustive' (all mono- and disaccharide features)
    | extra (string): 'ignore' skips this, 'wildcards' allows for wildcard matching', and 'termini' allows for positional matching; default:'termini'
    | wildcard_list (list): list of wildcard names (such as 'bond', 'Hex', 'HexNAc', 'Sia')
    | datatype (string): whether df comes from a dataset with quantitative variable ('response') or from presence_to_matrix ('presence')
    | rarity_filter (float): proportion of samples that need to have a non-zero value for a variable to be included; default:0.05
    | filepath (string): absolute path including full filename allows for saving the plot
    | index_col (string): default column to convert to dataframe index; default:'target'
    | **kwargs: keyword arguments that are directly passed on to seaborn clustermap\n                          
    | Returns:
    | :-
    | Prints clustermap                         
    """
    if libr is None:
        libr = lib
    if index_col in df.columns.values.tolist():
        df.index = df[index_col]
        df.drop([index_col], axis = 1, inplace = True)
    df = df.fillna(0)
    if mode == 'motif':
        df_motif = annotate_dataset(df.columns.values.tolist(),
                                libr = libr, feature_set = feature_set,
                                    extra = extra, wildcard_list = wildcard_list)
        df_motif = df_motif.replace(0,np.nan).dropna(thresh = np.max([np.round(rarity_filter * df_motif.shape[0]), 1]), axis = 1)
        collect_dic = {}
        if datatype == 'response':
          for col in df_motif.columns.values.tolist():
            indices = [i for i, x in enumerate(df_motif[col].values.tolist()) if x >= 1]
            temp = np.mean(df.iloc[:, indices], axis = 1)
            collect_dic[col] = temp
          df = pd.DataFrame(collect_dic)
        elif datatype == 'presence':
          idx = df.index.values.tolist()
          collecty = [[np.sum(df.iloc[row, [i for i, x in enumerate(df_motif[col].values.tolist()) if x >= 1]])/df.iloc[row, :].values.sum() for col in df_motif.columns.values.tolist()] for row in range(df.shape[0])]
          df = pd.DataFrame(collecty)
          df.columns = df_motif.columns.values.tolist()
          df.index = idx
    df.dropna(axis = 1, inplace = True)
    sns.clustermap(df.T, **kwargs)
    plt.xlabel('Samples')
    if mode == 'sequence':
        plt.ylabel('Glycans')
    else:
        plt.ylabel('Motifs')
    plt.tight_layout()
    if len(filepath) > 1:
      plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300,
                  bbox_inches = 'tight')
    plt.show()

def plot_embeddings(glycans, emb = None, label_list = None,
                    shape_feature = None, filepath = '',
                    **kwargs):
    """plots glycan representations for a list of glycans\n
    | Arguments:
    | :-
    | glycans (list): list of IUPAC-condensed glycan sequences as strings
    | emb (list): stored glycan representations; default takes them from trained species-level SweetNet model
    | label_list (list): list of same length as glycans if coloring of the plot is desired
    | shape_feature (string): monosaccharide/bond used to display alternative shapes for dots on the plot
    | filepath (string): absolute path including full filename allows for saving the plot
    | **kwargs: keyword arguments that are directly passed on to matplotlib\n
    """
    if emb is None:
        emb = glycan_emb
    embs = np.array([emb[k] for k in glycans])
    embs = TSNE(random_state = 42).fit_transform(embs)
    markers = None
    if shape_feature is not None:
        markers = {shape_feature: "X", "Absent": "o"}
        shape_feature = [shape_feature if shape_feature in k else 'Absent' for k in glycans]
    sns.scatterplot(x = embs[:,0], y = embs[:,1], hue = label_list,
                    palette = 'colorblind', style = shape_feature, markers = markers,
                    alpha = 0.8, **kwargs)
    sns.despine(left = True, bottom = True)
    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.tight_layout()
    if len(filepath) > 1:
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
  | Plots typical neighboring bond/monosaccharide and modification distribution
  """
  if df is None:
    df = df_species
  if rank is not None:
    df = df[df[rank] == focus]
  pool_in = [link_find(k) for k in df[glycan_col_name].values.tolist()]
  pool_in = [item for sublist in pool_in for item in sublist]
  pool_in = [k.replace('(', '*') for k in pool_in]
  pool_in = [k.replace(')', '*') for k in pool_in]

  if mode == 'bond':
    pool = [k.split('*')[0] for k in pool_in if k.split('*')[1] == sugar]
    lab = 'Observed Monosaccharides Making Linkage %s' % sugar
  elif mode == 'sugar':
    if modifications:
      sugars = [k.split('*')[0] for k in pool_in if sugar in k.split('*')[0]]
      pool = [k.split('*')[2] for k in pool_in if sugar in k.split('*')[0]]
    else:
      pool = [k.split('*')[2] for k in pool_in if k.split('*')[0] == sugar]
    lab = 'Observed Monosaccharides Paired with %s' % sugar
  elif mode == 'sugarbond':
    if modifications:
      sugars = [k.split('*')[0] for k in pool_in if sugar in k.split('*')[0]]
      pool = [k.split('*')[1] for k in pool_in if sugar in k.split('*')[0]]
    else:
      pool = [k.split('*')[1] for k in pool_in if k.split('*')[0] == sugar]
    lab = 'Observed Linkages Made by %s' % sugar
 
  cou = Counter(pool).most_common()
  cou_k = [k[0] for k in cou if k[1] > thresh]
  cou_v_in = [k[1] for k in cou if k[1] > thresh]
  cou_v = [v / len(pool) for v in cou_v_in]

  fig, (a0,a1) = plt.subplots(1, 2 , figsize = (8, 4), gridspec_kw = {'width_ratios': [1, 1]})
  if modifications:
      cou2 = Counter(sugars).most_common()
      cou_k2 = [k[0] for k in cou2 if k[1] > thresh]
      cou_v2_in = [k[1] for k in cou2 if k[1] > thresh]
      cou_v2 = [v / len(sugars) for v in cou_v2_in]
      color_list =  plt.cm.get_cmap('tab20')
      color_map = {cou_k2[k]:color_list(k/len(cou_k2)) for k in range(len(cou_k2))}
      palette = [color_map[k] for k in cou_k2]
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
          cou_v_t = {cou_t[j][0]:cou_t[j][1] for j in range(len(cou_t))}
          cou_v_t = [cou_v_t[j] if j in list(cou_v_t.keys()) else 0 for j in cou_k]
          if len(cou_k2) > 1:
              cou_for_df.append(pd.DataFrame({'monosaccharides':cou_k, 'counts':cou_v_t, 'colors':[cou_k2[k]]*len(cou_k)}))
          else:
              sns.barplot(x = cou_k, y = cou_v_t, ax = a0, color = "cornflowerblue")
      if len(cou_k2) > 1:
          cou_df = pd.concat(cou_for_df)
          sns.histplot(cou_df, x = 'monosaccharides', hue = 'colors', weights = 'counts',
                       multiple = 'stack', palette = palette, ax = a0, legend = False,
                   shrink = 0.8)
      a0.set_ylabel('Absolute Occurrence')
  else:
      sns.barplot(x = cou_k, y = cou_v, ax = a0, color = "cornflowerblue")
      a0.set_ylabel('Relative Proportion')
  sns.despine(left = True, bottom = True)
  a0.set_xlabel(lab)
  a0.set_title('Pairings')
  plt.setp(a0.get_xticklabels(), rotation = 'vertical')
  
  if modifications:
    if len(cou_k2) > 1:
        cou_df2 = pd.DataFrame({'monosaccharides': cou_k2, 'counts': cou_v2})
        sns.histplot(cou_df2, x = 'monosaccharides', weights = 'counts',
                     hue = 'monosaccharides', shrink = 0.8, legend = False,
                     ax = a1, palette = palette, alpha = 0.75)
    else:
        sns.barplot(x = cou_k2, y = cou_v2, ax = a1, color = "cornflowerblue")
    sns.despine(left = True, bottom = True)
    a1.set_ylabel('Relative Proportion')
    a1.set_xlabel(sugar + " Variants")
    a1.set_title('Observed Modifications')
    plt.setp(a1.get_xticklabels(), rotation = 'vertical')

  fig.suptitle('Sequence context of ' + str(sugar))
  fig.tight_layout()
  if len(filepath) > 1:
      plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300,
                  bbox_inches = 'tight')
  plt.show()
