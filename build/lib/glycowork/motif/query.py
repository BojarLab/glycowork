import ast
import numpy as np
import pandas as pd

from glycowork.glycan_data.loader import lib, motif_list, df_glycan
from glycowork.motif.graph import compare_glycans
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
    if glycan in df_glycan.glycan.values.tolist():
        idx = df_glycan.glycan.values.tolist().index(glycan)
    else:       
        idx = np.where([compare_glycans(glycan, k, libr = libr) for k in df_glycan.glycan.values.tolist()])[0][0]
    species = ast.literal_eval(df_glycan.Species.values.tolist()[idx])
    if len(species) > 0:
        print("\nThis glycan occurs in the following species: " + str(list(sorted(species))))
    else:
        try:
            species = df_glycan.predicted_taxonomy.values.tolist()[idx]
            print("\nNo definitive information in our database but this glycan is predicted to occur here: " + str(species))
        except:
            pass
    if len(species) > 5:
        phyla = list(sorted(list(set(eval(df_glycan.Phylum.values.tolist()[idx])))))
        print("\nPuh, that's quite a lot! Here are the phyla of those species: " + str(phyla))
    found_motifs = annotate_glycan(glycan, motifs = motifs, libr = libr)
    found_motifs = found_motifs.loc[:, (found_motifs != 0).any(axis = 0)].columns.values.tolist()
    if len(found_motifs) > 0:
        print("\nThis glycan contains the following motifs: " + str(found_motifs))
    if isinstance(df_glycan.glytoucan_id.values.tolist()[idx], str):
        print("\nThis is the GlyTouCan ID for this glycan: " + str(df_glycan.glytoucan_id.values.tolist()[idx]))
    if isinstance(df_glycan.immunogenicity.values.tolist()[idx], float):
        if df_glycan.immunogenicity.values.tolist()[idx] > 0:
            print("\nThis glycan is likely to be immunogenic to humans.")
        elif df_glycan.immunogenicity.values.tolist()[idx] < 1:
            print("\nThis glycan is likely to be non-immunogenic to humans.")
    if len(df_glycan.tissue_sample.values.tolist()[idx]) > 2:
        tissue = ast.literal_eval(df_glycan.tissue_sample.values.tolist()[idx])
        print("\nThis glycan has been reported to be expressed in: " + str(list(sorted(tissue))))
    if len(df_glycan.disease_association.values.tolist()[idx]) > 2:
        disease = ast.literal_eval(df_glycan.disease_association.values.tolist()[idx])
        direction = ast.literal_eval(df_glycan.disease_direction.values.tolist()[idx])
        disease_sample = ast.literal_eval(df_glycan.disease_sample.values.tolist()[idx])
        print("\nThis glycan has been reported to be dysregulated in (disease, direction, sample): " \
              + str([(disease[k],
                     direction[k],
                     disease_sample[k]) for k in range(len(disease))]))
    print("\nThat's all we can do for you at this point!")

def glytoucan_to_glycan(ids, revert = False):
    """interconverts GlyTouCan IDs and glycans in IUPAC-condensed\n
    | Arguments:
    | :-
    | ids (list): list of GlyTouCan IDs as strings (if using glycans instead, change 'revert' to True
    | revert (bool): whether glycans should be mapped to GlyTouCan IDs or vice versa; default:False\n
    | Returns:
    | :-
    | Returns list of either GlyTouCan IDs or glycans in IUPAC-condensed
    """
    if revert:
        ids = [df_glycan[df_glycan.glycan == k].glytoucan_id.values.tolist()[0] for k in ids]
        if any([k not in df_glycan.glycan.values.tolist() for k in ids]):
            print('These glycans are not in our database: ' + str([k for k in ids if k not in df_glycan.glycan.values.tolist()]))
        return ids
    else:
        glycans = [df_glycan[df_glycan.glytoucan_id == k].glycan.values.tolist()[0] if k in df_glycan.glytoucan_id.values.tolist() else k for k in ids]
        if any([k not in df_glycan.glytoucan_id.values.tolist() for k in ids]):
            print('These IDs are not in our database: ' + str([k for k in ids if k not in df_glycan.glytoucan_id.values.tolist()]))
        return glycans
