import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.manifold import TSNE

from glycowork.glycan_data.loader import lib, glycan_emb, df_species
from glycowork.motif.annotate import annotate_dataset
from glycowork.motif.tokenization import link_find

def get_pvals_motifs(df, glycan_col_name, label_col_name,
                     libr = None, thresh = 1.645, sorting = True,
                     feature_set = ['exhaustive'], wildcards = False,
                     wildcard_list = []):
    """returns enriched motifs based on label data or predicted data\n
    df -- dataframe containing glycan sequences and labels\n
    glycan_col_name -- column name for glycan sequences; string\n
    label_col_name -- column name for labels; string\n
    libr -- sorted list of unique glycoletters observed in the glycans of our dataset\n
    thresh -- threshold value to separate positive/negative; default is 1.645 for Z-scores\n
    sorting -- whether p-value dataframe should be sorted ascendingly; default: True\n
    feature_set -- which feature set to use for annotations, add more to list to expand; default is 'exhaustive'\n
                 options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans)
                               and 'exhaustive' (all mono- and disaccharide features)\n
    wildcards -- set to True to allow wildcards (e.g., 'bond', 'monosaccharide'); default is False\n
    wildcard_list -- list of wildcard names (such as 'bond', 'Hex', 'HexNAc', 'Sia')\n

    returns dataframe with p-values and corrected p-values for every glycan motif
    """
    if libr is None:
        libr = lib
    df_motif = annotate_dataset(df[glycan_col_name].values.tolist(),
                                libr = lib, feature_set = feature_set,
                                wildcards = wildcards, wildcard_list = wildcard_list)
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
        return out.sort_values(by = 'corr_pval')
    else:
        return out

def make_heatmap(df, mode = 'sequence', libr = None, feature_set = ['known'],
                 wildcards = False, wildcard_list = [], datatype = 'response',
                 rarity_filter = 0.05,
                 **kwargs):
    """clusters samples based on glycan data (for instance glycan binding etc.)\n
    df -- dataframe with glycan data, rows are samples and columns are glycans\n
    mode -- whether glycan 'sequence' or 'motif' should be used for clustering; default:sequence\n
    libr -- sorted list of unique glycoletters observed in the glycans of our dataset\n
    feature_set -- which feature set to use for annotations, add more to list to expand; default is 'exhaustive'\n
                 options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans)
                               and 'exhaustive' (all mono- and disaccharide features)\n
    wildcards -- set to True to allow wildcards (e.g., 'bond', 'monosaccharide'); default is False\n
    wildcard_list -- list of wildcard names (such as 'bond', 'Hex', 'HexNAc', 'Sia')\n
    datatype -- whether df comes from a dataset with quantitative variable ('response') or from presence_to_matrix ('presence')\n
    rarity_filter -- proportion of samples that need to have a non-zero value for a variable to be included; default:0.05\n
    **kwargs -- keyword arguments that are directly passed on to seaborn clustermap\n                          

    prints clustermap                         
    """
    if libr is None:
        libr = lib
    df = df.fillna(0)
    if mode == 'motif':
        df_motif = annotate_dataset(df.columns.values.tolist(),
                                libr = libr, feature_set = feature_set,
                                    wildcards = wildcards, wildcard_list = wildcard_list)
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
    plt.show()

def plot_embeddings(glycans, emb = None, label_list = None):
    """plots glycan representations for a list of glycans
    glycans -- list of IUPACcondensed glycan sequences (string)\n
    emb -- stored glycan representations; default takes them from trained species-level SweetNet model\n
    label_list -- list of same length as glycans if coloring of the plot is desired\n
    """
    if emb is None:
        emb = glycan_emb
    embs = np.array([emb[k] for k in glycans])
    embs = TSNE(random_state = 42).fit_transform(embs)
    sns.scatterplot(x = embs[:,0], y = embs[:,1], hue = label_list,
                    palette = 'colorblind')
    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    plt.tight_layout()
    plt.show()

def characterize_monosaccharide(sugar, df = None, mode = 'sugar', glycan_col_name = 'target',
                                rank = None, focus = None, modifications = False):
  """for a given monosaccharide/bond, return typical neighboring bond/monosaccharide\n
  sugar -- monosaccharide or linkage as string\n
  df -- dataframe to use for analysis; default:df_species\n
  mode -- either 'sugar' (connected monosaccharides),\n
          'bond' (monosaccharides making a provided linkage),\n
          or 'sugarbond' (linkages that a provided monosaccharides makes); default:'sugar'\n
  glycan_col_name -- column name under which glycans can be found; default:'target'\n
  rank -- add column name as string if you want to filter for a group\n
  focus -- add row value as string if you want to filter for a group\n
  modifications -- set to True if you want to consider modified versions of a monosaccharide; default:False\n

  plots typical neighboring bond/monosaccharide and modification distribution
  """
  if df is None:
    df = df_species
  if rank is not None:
    df = df[df[rank] == focus]
  pool = [link_find(k) for k in df[glycan_col_name].values.tolist()]
  pool = [item for sublist in pool for item in sublist]

  if mode == 'bond':
    pool = [k.split('*')[0] for k in pool if k.split('*')[1] == sugar]
    lab = 'Observed Monosaccharides Making Bond %s' % sugar
  elif mode == 'sugar':
    if modifications:
      sugars = [k.split('*')[0] for k in pool if sugar in k.split('*')[0]]
      pool = [k.split('*')[2] for k in pool if sugar in k.split('*')[0]]
    else:
      pool = [k.split('*')[2] for k in pool if k.split('*')[0] == sugar]
    lab = 'Observed Monosaccharides Paired with %s' % sugar
  elif mode == 'sugarbond':
    if modifications:
      sugars = [k.split('*')[0] for k in pool if sugar in k.split('*')[0]]
      pool = [k.split('*')[1] for k in pool if sugar in k.split('*')[0]]
    else:
      pool = [k.split('*')[1] for k in pool if k.split('*')[0] == sugar]
    lab = 'Observed Bonds Made by %s' % sugar
    
  cou = Counter(pool).most_common()
  cou_k = [k[0] for k in cou if k[1] > 10]
  cou_v = [k[1] for k in cou if k[1] > 10]
  cou_v = [v / len(pool) for v in cou_v]

  fig, (a0,a1) = plt.subplots(1, 2 , figsize = (8, 4), gridspec_kw = {'width_ratios': [1, 1]})
  a0.bar(cou_k, cou_v)
  a0.set_ylabel('Relative Proportion')
  a0.set_xlabel(lab)
  a0.set_title('Pairings')
  if mode == 'sugar' or mode == 'bond':
    plt.setp(a0.get_xticklabels(), rotation = 'vertical')
  
  if modifications:
    cou = Counter(sugars).most_common()
    cou_k = [k[0] for k in cou if k[1] > 10]
    cou_v = [k[1] for k in cou if k[1] > 10]
    cou_v = [v / len(sugars) for v in cou_v]
    a1.bar(cou_k, cou_v)
    a1.set_ylabel('Relative Proportion')
    a1.set_xlabel(sugar + " Variants")
    plt.setp(a1.get_xticklabels(), rotation = 'vertical')

  fig.tight_layout()
  plt.show()
