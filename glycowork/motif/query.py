import numpy as np
import pandas as pd

from glycowork.glycan_data import loader
from glycowork.glycan_data.loader import motif_list
from glycowork.motif.graph import compare_glycans
from glycowork.motif.annotate import annotate_glycan


def get_insight(glycan: str, # Glycan in IUPAC-condensed format
                motifs: pd.DataFrame | None = None # DataFrame of glycan motifs; default:motif_list
               ) -> None: # Prints glycan meta-information
    "Print meta-information about a glycan"
    if motifs is None:
        motifs = motif_list
    print("Let's get rolling! Give us a few moments to crunch some numbers.")
    # Find glycan as string or as graph in df_glycan
    if glycan in loader.df_glycan.glycan.values:
        idx = loader.df_glycan.glycan.values.tolist().index(glycan)
    else:
        hits = np.where([compare_glycans(glycan, k) for k in loader.df_glycan.glycan.values.tolist()])[0]
        if not len(hits):
            print(
                f"\nWe don't have {glycan} in our database yet. Please double-check the sequence or try a related structure.")
            return
        idx = hits[0]
    row = loader.df_glycan.iloc[idx]
    species = row.Species
    if len(species) > 0:
        print("\nThis glycan occurs in the following species: " + str(sorted(species)))
    if len(species) > 5:
        phyla = sorted(set(row.Phylum))
        print("\nPuh, that's quite a lot! Here are the phyla of those species: " + str(phyla))
    found_motifs = annotate_glycan(glycan, motifs = motifs)
    found_motifs = found_motifs.loc[:, (found_motifs != 0).any(axis = 0)].columns.values.tolist()
    if len(found_motifs) > 0:
        print("\nThis glycan contains the following motifs: " + str(found_motifs))
    if isinstance(row.glytoucan_id, str):
        print("\nThis is the GlyTouCan ID for this glycan: " + str(row.glytoucan_id))
    if len(row.tissue_sample) > 0:
        tissue = row.tissue_sample
        print("\nThis glycan has been reported to be expressed in: " + str(sorted(tissue)))
    if len(row.disease_association) > 0:
        disease = row.disease_association
        direction = row.disease_direction
        disease_sample = row.disease_sample
        print(
            "\nThis glycan has been reported to be dysregulated in (disease, direction, sample): "
            + str([(disease[k],
                    direction[k],
                    disease_sample[k]) for k in range(len(disease))])
            )
    print("\nThat's all we can do for you at this point!")
