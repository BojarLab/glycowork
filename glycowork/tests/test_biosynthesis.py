import glycowork

from glycowork.network.biosynthesis import *
from glycowork.glycan_data.loader import df_species

df_human = df_species[df_species.Species == 'Homo_sapiens']
df_human = df_human[df_human.target.str.endswith('Gal(b1-4)Glc')]

print("Human milk oligosaccharides - adjacency matrix")
print(create_adjacency_matrix(df_human.target.values.tolist(),
                              df_human))


