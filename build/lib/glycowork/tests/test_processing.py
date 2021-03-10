import glycowork

from glycowork.motif.processing import *
from glycowork.helper.func import *

df_glycan = load_file("v3_sugarbase.csv")

print("Length Glyco-Vocabulary")
vocab = get_lib(df_glycan.glycan.values.tolist())
print(len(vocab))

print("Seed dataframe with wildcards")
print(seed_wildcard(df_species, linkages, 'bond')[-10:])
