import glycowork

from glycowork.motif.processing import *
from glycowork.glycan_data.loader import *

print("Length Glyco-Vocabulary")
vocab = get_lib(df_glycan.glycan.values.tolist())
print(len(vocab))

print("Seed dataframe with wildcards")
print(seed_wildcard(df_species, linkages, 'bond')[-10:])


print("Convert Presence to Matrix")
print(presence_to_matrix(df_species[df_species.order == 'Fabales'].reset_index(drop = True),
                         label_col_name = 'genus'))
