import glycowork

from glycowork.motif.tokenization import *

glycans = ['Man(a1-3)[Man(a1-6)][Xyl(b1-2)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-3)]GlcNAc',
           'Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-3)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
           'GalNAc(a1-4)GlcNAcA(a1-4)[GlcN(b1-7)]Kdo(a2-5)[Kdo(a2-4)]Kdo(a2-6)GlcOPN(b1-6)GlcOPN']
label = [1,1,0]
test_df = pd.DataFrame({'glycan':glycans, 'eukaryotic':label})
print("Glyco-Motif Test")
print(motif_matrix(test_df, 'glycan', 'eukaryotic'))
print("Annotate Test")
print(annotate_dataset(glycans))
