import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from glycowork.glycan_data.loader import lib
from glycowork.motif.tokenization import motif_matrix

def get_pvals_motifs(df, glycan_col_name, label_col_name,
                     libr = None, thresh = 1.645, sorting = True):
    """returns enriched motifs based on label data or predicted data\n
    df -- dataframe containing glycan sequences and labels\n
    glycan_col_name -- column name for glycan sequences; string\n
    label_col_name -- column name for labels; string\n
    libr -- sorted list of unique glycoletters observed in the glycans of our dataset\n
    thresh -- threshold value to separate positive/negative; default is 1.645 for Z-scores\n
    sorting -- whether p-value dataframe should be sorted ascendingly; default: True\n

    returns dataframe with p-values and corrected p-values for every glycan motif
    """
    if libr is None:
        libr = lib
    df_motif = motif_matrix(df, glycan_col_name, label_col_name, libr = libr)
    df_motif = df_motif.loc[:,~df_motif.columns.duplicated()]
    df_motif.reset_index(drop = True, inplace = True)
    df_pos = df_motif[df_motif[label_col_name] > thresh]
    df_neg = df_motif[df_motif[label_col_name] <= thresh]
    ttests = [ttest_ind(df_pos.iloc[:,k], df_neg.iloc[:,k], equal_var = False)[1]/2 if np.mean(df_pos.iloc[:,k])>np.mean(df_neg.iloc[:,k]) else 1.0 for k in range(0, df_motif.shape[1]-1)]
    ttests_corr = multipletests(ttests, method = 'hs')[1].tolist()
    out = pd.DataFrame(list(zip(df_motif.columns.values.tolist()[:-1], ttests, ttests_corr)))
    out.columns = ['motif', 'pval', 'corr_pval']
    if sorting:
        return out.sort_values(by = 'corr_pval')
    else:
        return out
