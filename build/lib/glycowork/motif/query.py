import numpy as np
import pandas as pd

from glycowork.glycan_data.loader import lib, motif_list, df_glycan
from glycowork.motif.graph import glycan_to_nxGraph, fast_compare_glycans
from glycowork.motif.annotate import annotate_glycan

def get_insight(glycan, libr = lib, motifs = motif_list):
    """prints out meta-information about a glycan
    glycan -- glycan in string format (IUPACcondensed)
    libr -- library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
    motifs -- motifs -- dataframe of glycan motifs (name + sequence)
    """
    print("Let's get rolling! Give us a few moments to crunch some numbers.")
    ggraph = glycan_to_nxGraph(glycan, libr = libr)
    df_glycan['graph'] = [glycan_to_nxGraph(k, libr = libr) for k in df_glycan.glycan.values.tolist()]
    idx = np.where([fast_compare_glycans(ggraph, k, libr = lib) for k in df_glycan.graph.values.tolist()])[0]
    species = df_glycan.species.values.tolist()[idx[0]]
    if len(species) > 0:
        print("\nThis glycan occurs in the following species: " + str(species))
    found_motifs = annotate_glycan(glycan, motifs = motifs, libr = libr)
    found_motifs = found_motifs.loc[:, (found_motifs != 0).any(axis = 0)].columns.values.tolist()
    if len(found_motifs) > 0:
        print("\nThis glycan contains the following motifs: " + str(found_motifs))
    print("\nThat's all we can do for you at this point!")
    
