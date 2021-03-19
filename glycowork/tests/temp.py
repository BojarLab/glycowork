import glycowork
import copy

from glycowork.motif.graph import glycan_to_nxGraph
from glycowork.glycan_data.loader import df_species, df_glycan

df_glycan['graph'] = [glycan_to_nxGraph(k) for k in df_glycan.glycan.values.tolist()]
df_out = copy.deepcopy(df_glycan)
df_out.to_csv('C:/Users/danie/Downloads/v3_sugarbase.csv', index=False)
