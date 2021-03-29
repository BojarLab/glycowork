import glycowork

from glycowork.motif.analysis import *
from glycowork.glycan_data.loader import influenza_binding

glycans = ['Man(a1-3)[Man(a1-6)][Xyl(b1-2)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-3)]GlcNAc',
           'Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-3)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
           'GalNAc(a1-4)GlcNAcA(a1-4)[GlcN(b1-7)]Kdo(a2-5)[Kdo(a2-4)]Kdo(a2-6)GlcOPN(b1-6)GlcOPN',
           'Man(a1-2)Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
           'Glc(b1-3)Glc(b1-3)Glc']
label = [3.234, 2.423, 0.733, 3.102, 0.108]
test_df = pd.DataFrame({'glycan':glycans, 'binding':label})

print("Glyco-Motif enrichment p-value test")
print(get_pvals_motifs(test_df, 'glycan', 'binding'))

print("Glyco-Motif enrichment p-value test expanded")
print(get_pvals_motifs(test_df, 'glycan', 'binding', feature_set = ['known',
                                                                    'exhaustive']))
print("Real Glyco-Motif enrichment test")
print(get_pvals_motifs(influenza_binding, 'glycan', 'binding',
                       multiple_samples = True))

label2 = [0.134, 0.345, 1.15, 0.233, 2.981]
label3 = [0.334, 0.245, 1.55, 0.133, 2.581]
test_df2 = pd.DataFrame([label, label2, label3], columns = glycans)
print("Test Glyco-Heatmap")
make_heatmap(test_df2, mode = 'motif', feature_set = ['known', 'exhaustive'])



