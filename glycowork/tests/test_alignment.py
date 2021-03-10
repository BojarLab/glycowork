import glycowork

from glycowork.alignment.glysum import *

glycan = 'Fuc(a1-4)[GlcNAc(b1-3)]GlcNAc(b1-6)[Fuc(a1-2)Gal(b1-3)]GalNAc'

print("Test Alignment")
print(pairwiseAlign(glycan))
