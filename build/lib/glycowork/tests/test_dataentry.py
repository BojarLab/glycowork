import glycowork

from glycowork.glycan_data.data_entry import *
from glycowork.glycan_data.loader import df_species
from glycowork.motif.graph import glycan_to_nxGraph

df_species['graph'] = [glycan_to_nxGraph(k) for k in df_species.target.values.tolist()]


print("Check presence of Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc")
check_presence('Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc', df_species, fast = True)

print("Check presence of Fuc(b1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc")
check_presence('Fuc(b1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc', df_species, fast = True)

print("Check presence of Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc in the species Danielus Bojarum")
check_presence('Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc', df_species,
               name = 'Danielus Bojarum', fast = True)
