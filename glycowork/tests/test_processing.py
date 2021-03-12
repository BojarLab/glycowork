import glycowork

from glycowork.motif.processing import *
from glycowork.glycan_data.loader import *

print("Length Glyco-Vocabulary")
vocab = get_lib(df_glycan.glycan.values.tolist())
print(len(vocab))

print("Seed dataframe with wildcards")
print(seed_wildcard(df_species, linkages, 'bond')[-10:])
