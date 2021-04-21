import numpy as np
import pandas as pd

from glycowork.glycan_data.loader import lib, motif_list, df_glycan
from glycowork.motif.graph import glycan_to_nxGraph, fast_compare_glycans
from glycowork.motif.annotate import annotate_glycan

def get_insight(glycan, libr = None, motifs = None):
    """prints out meta-information about a glycan\n
    | Arguments:
    | :-
    | glycan (string): glycan in IUPAC-condensed format
    | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
    | motifs (dataframe): dataframe of glycan motifs (name + sequence); default:motif_list
    """
    if libr is None:
        libr = lib
    if motifs is None:
        motifs = motif_list
    print("Let's get rolling! Give us a few moments to crunch some numbers.")
    ggraph = glycan_to_nxGraph(glycan, libr = libr)
    df_glycan['graph'] = [glycan_to_nxGraph(k, libr = libr) for k in df_glycan.glycan.values.tolist()]
    idx = np.where([fast_compare_glycans(ggraph, k, libr = lib) for k in df_glycan.graph.values.tolist()])[0]
    species = df_glycan.Species.values.tolist()[idx[0]]
    if len(species) > 0:
        print("\nThis glycan occurs in the following species: " + str(species))
    if len(eval(species)) > 5:
        phyla = df_glycan.Phylum.values.tolist()[idx[0]]
        print("\nPuh, that's quite a lot! Here are the phyla of those species: " + str(phyla))
    found_motifs = annotate_glycan(glycan, motifs = motifs, libr = libr)
    found_motifs = found_motifs.loc[:, (found_motifs != 0).any(axis = 0)].columns.values.tolist()
    if len(found_motifs) > 0:
        print("\nThis glycan contains the following motifs: " + str(found_motifs))
    if isinstance(df_glycan.immunogenicity.values.tolist()[idx[0]], float):
        if df_glycan.immunogenicity.values.tolist()[idx[0]] > 0:
            print("\nThis glycan is likely to be immunogenic to humans.")
        elif df_glycan.immunogenicity.values.tolist()[idx[0]] < 1:
            print("\nThis glycan is likely to be non-immunogenic to humans.")
    if isinstance(df_glycan.ecoli_pathogenicity.values.tolist()[idx[0]], float):
        if df_glycan.ecoli_pathogenicity.values.tolist()[idx[0]] > 0:
            print("\nThis glycan is likely to contribute to E.coli pathogenicity.")
        elif df_glycan.ecoli_pathogenicity.values.tolist()[idx[0]] < 1:
            print("\nThis glycan is found in non-pathogenic E.coli strains.")
    print("\nThat's all we can do for you at this point!")
    
