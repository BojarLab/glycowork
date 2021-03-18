import glycowork

from glycowork.alignment.glysum import *

glycan = 'Man(a1-3)Man(a1-4)Glc(b1-4)Man(a1-5)Kdo'

print("Test Alignment")
print(pairwiseAlign(glycan))
