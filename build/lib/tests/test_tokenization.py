import glycowork

from glycowork.motif.tokenization import *
from glycowork.motif.graph import glycan_to_nxGraph

glycans = ['Man(a1-3)[Man(a1-6)][Xyl(b1-2)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-3)]GlcNAc',
           'Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-3)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc',
           'GalNAc(a1-4)GlcNAcA(a1-4)[GlcN(b1-7)]Kdo(a2-5)[Kdo(a2-4)]Kdo(a2-6)GlcOPN(b1-6)GlcOPN']
label = [1,1,0]
test_df = pd.DataFrame({'glycan':glycans, 'eukaryotic':label})
print("Glyco-Motif Test")
print(motif_matrix(test_df, 'glycan', 'eukaryotic'))

glycans_test = [
    (
        "Xyla1-6Glcb1-4(Xyla1-6)Glcb1-4[Xyla1-6]Glcb1-4Glc-ol",
        "Xyl(a1-6)Glc(b1-4)[Xyl(a1-6)]Glc(b1-4)[Xyl(a1-6)]Glc(b1-4)Glc-ol",
    ),
    ("GalNAcb1-4(Neu5Aca2-8Neu5Aca2-3)Galb1-4Glcb-Sp", "GalNAc(b1-4)[Neu5Ac(a2-8)Neu5Ac(a2-3)]Gal(b1-4)Glc"),
    ("GlcN(Gc)b-Sp8", "GlcNGc"),
    ("Rhaa-Sp8", "Rha"),
    (
        "Galb1-4GlcNAcb1-2Mana1-3(Galb1-4GlcNAcb1-2Mana1-6)Manb1-4GlcNAcb1-4GlcNAcb-Asn",
        "Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    ),
    ("Rha-Sp", "Rha"),
    ("Neu5Aca2-3Galb1-4(6S)GlcNAcb-Sp", "Neu5Ac(a2-3)Gal(b1-4)GlcNAc6S"),
    ("(4P)GlcNAcb1-4Manb-Sp", "GlcNAc4P(b1-4)Man"),
    ("Gal(3S,6S)b1-4GlcNAc(6S)", "Gal3S6S(b1-4)GlcNAc6S"),
    ("6-H2PO3Glcb-Sp10", "Glc6P"),
    ("[3OSO3]Galb1-3(Fuca1-4)GlcNAcb-Sp8", "Gal3S(b1-3)[Fuc(a1-4)]GlcNAc"),
    ("GlcNAcb1-", "GlcNAc"),
    ("GlcNAcb1", "GlcNAc"),
    ("GlcNAcb", "GlcNAc"),
    ("a-L-Fuc-Sp8", "Fuc"),
    ("b-D-Gal-Sp8", "Gal"),
    ("b-GlcN(Gc)-Sp8", "GlcNGc"),
    ("KDNa2-3Galb1-3GlcNAcb-Sp0", "Kdn(a2-3)Gal(b1-3)GlcNAc"),
    ("9NAcNeu5Aca-Sp8", "Neu5Ac9NAc"),
    ("Mana1-3(Mana1-6)Manb1-4GlcNAcb1-4GlcNAcb-Gly", "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAcGly"),
    ("b-NeuAc-Sp8", "Neu5Ac"),
    ("Galb1-3(GlcNacb1-6)GalNAc-Sp14", "Gal(b1-3)[GlcNAc(b1-6)]GalNAc"),
    ("NeuAc(9Ac)a2-3Galb1-4GlcNAcb-Sp0", "Neu5Ac9Ac(a2-3)Gal(b1-4)GlcNAc")
]


def test_cfg_to_iupac():
    for input_glycan, expected_glycan in glycans_test:
        ouput_glycan = cfg_to_iupac(glycan=input_glycan)
        # Test to make sure it can be parsed by glycowork
        glycan_to_nxGraph(glycan=ouput_glycan)
        assert ouput_glycan == expected_glycan
