import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.manifold import TSNE

from glycowork.glycan_data.loader import lib, glycan_emb
from glycowork.motif.annotate import annotate_dataset

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
                 wildcards = False, wildcard_list = [],
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
        df_motif = df_motif.loc[:, (df_motif != 0).any(axis = 0)]
        collect_dic = {}
        for col in df_motif.columns.values.tolist():
            indices = [i for i, x in enumerate(df_motif[col].values.tolist()) if x >= 1]
            temp = np.mean(df.iloc[:, indices], axis = 1)
            collect_dic[col] = temp
        df = pd.DataFrame(collect_dic)
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
