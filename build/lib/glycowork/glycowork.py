import pandas as pd
import numpy as np
import re
import time, copy
import random
import torch
import networkx as nx
from sklearn.model_selection import StratifiedShuffleSplit
try:
  from torch_geometric.data import Data, DataLoader
except ImportError:
    raise ImportError('<torch_geometric missing; cannot do deep learning>')

lib = ['', '1,4-Anhydro-Gal', '1,4-Anhydro-Kdo', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1dAlt-ol', '1dEry-ol', '2,3-Anhydro-All', '2,3-Anhydro-Man', '2,3-Anhydro-Rib', '2,5-Anhydro-D-Alt', '2,5-Anhydro-D-AltOS', '2,5-Anhydro-L-Man', '2,5-Anhydro-Man', '2,5-Anhydro-Man-ol', '2,5-Anhydro-ManOS', '2,5-Anhydro-Tal-ol', '2,5-Anhydro-TalOP', '2,7-Anhydro-Kdo', '2,7-Anhydro-Kdof', '2-4', '2-5', '2-6', '3', '3,6-Anhydro-Fruf', '3,6-Anhydro-Gal', '3,6-Anhydro-GalOS', '3,6-Anhydro-Glc', '3,6-Anhydro-L-Gal', '3,6-Anhydro-L-GalOMe', '3-3', '3-5', '3-6', '3dLyxHepUlosaric', '4', '4,7-Anhydro-KdoOPEtn', '4,8-Anhydro-DDGlcOct', '4,8-Anhydro-Kdo', '4,8-Anhydro-LDGlcOct', '4-5', '4dAraHex', '4dEry-ol', '4eLegNAcNAc', '5-2', '5-3', '5-5', '5-6', '6dAlt', '6dAltNAc', '6dAltOAc', '6dAltf', '6dAltfOAc', '6dGul', '6dManHep', '6dTal', '6dTalNAc', '6dTalNAcOAc', '6dTalOAc', '6dTalOAcOAc', '6dTalOAcOMe', '6dTalOMe', '6dTalOMe-ol', '6dTalf', '8eAciNAcNAc', '8eLeg', '8eLegNAc', '8eLegNAcNAc', '8eLegNAcNAcGro', '8eLegNAcNBut', 'Abe', 'AbeOAc', 'AcefA', 'AciNAcNAc', 'Aco', 'AcoNAc', 'AllN', 'AllOAc', 'AllOMe', 'Alt', 'AltA', 'AltAN', 'AltNAcA', 'AltOMeA', 'Altf', 'AltfOAc', 'Ami', 'ApiOAc', 'ApiOMe-ol', 'Apif', 'Ara', 'Ara-ol', 'AraHepUloNAc-onic', 'AraHepUloNAcN-onic', 'AraHepUloNGc-onic', 'AraHexA', 'AraN', 'AraNMeOMe', 'AraOAc', 'AraOAcOP-ol', 'AraOMe', 'AraOPN', 'Araf', 'ArafGro', 'ArafOCoum', 'ArafOFer', 'ArafOMe', 'ArafOS', 'Asc', 'Bac', 'BacNAc', 'BoiOMe', 'Col', 'D-2dAraHex', 'D-2dAraHexA', 'D-3dAraHepUlosonic', 'D-3dLyxHepUlosaric', 'D-3dThrHexUlosonic', 'D-3dThrPen', 'D-3dXylHexOMe', 'D-4dAraHex', 'D-4dEryHexOAcN4en', 'D-4dLyxHex', 'D-4dLyxHexOMe', 'D-4dThrHexA4en', 'D-4dThrHexAN4en', 'D-4dThrHexOAcN4en', 'D-4dXylHex', 'D-6dAllOMe', 'D-6dAlt', 'D-6dAltHep', 'D-6dAltHepOMe', 'D-6dAltHepf', 'D-6dAraHex', 'D-6dAraHexN', 'D-6dAraHexNAc', 'D-6dAraHexOMe', 'D-6dLyxHexOMe', 'D-6dManHep', 'D-6dManHepOAc', 'D-6dManHepOP', 'D-6dTal', 'D-6dTalHep', 'D-6dTalOAc', 'D-6dTalOAcOMe', 'D-6dTalOMe', 'D-6dXylHex', 'D-6dXylHexN4Ulo', 'D-6dXylHexNAc4Ulo', 'D-6dXylHexOMe', 'D-7dLyxOctUlosonic', 'D-9dThrAltNon-onic', 'D-Alt', 'D-Apif', 'D-ApifOAc', 'D-ApifOMe', 'D-Ara', 'D-Ara-ol', 'D-AraHepUlo-onic', 'D-AraHex', 'D-AraHexUloOMe', 'D-AraN', 'D-AraOS', 'D-Araf', 'D-ArafN', 'D-Fuc', 'D-Fuc-ol', 'D-FucN', 'D-FucNAc', 'D-FucNAc-ol', 'D-FucNAcN', 'D-FucNAcNMe', 'D-FucNAcNMeN', 'D-FucNAcOAc', 'D-FucNAcOMe', 'D-FucNAcOP', 'D-FucNAcOPEtn', 'D-FucNAlaAc', 'D-FucNAsp', 'D-FucNBut', 'D-FucNButGro', 'D-FucNFo', 'D-FucNLac', 'D-FucNMeN', 'D-FucNN', 'D-FucNThrAc', 'D-FucOAc', 'D-FucOAcN', 'D-FucOAcNBut', 'D-FucOAcNGroA', 'D-FucOAcOBut', 'D-FucOAcOMe', 'D-FucOBut', 'D-FucOEtn', 'D-FucOMe', 'D-FucOMeN', 'D-FucOMeOCoum', 'D-FucOMeOFer', 'D-FucOMeOSin', 'D-FucOS', 'D-Fucf', 'D-FucfNAc', 'D-FucfOAc', 'D-Ido', 'D-IdoA', 'D-IdoOSA', 'D-Rha', 'D-Rha-ol', 'D-RhaCMe', 'D-RhaGro', 'D-RhaN', 'D-RhaNAc', 'D-RhaNAcOAc', 'D-RhaNBut', 'D-RhaNButOMe', 'D-RhaNFo', 'D-RhaOFoN', 'D-RhaOMe', 'D-RhaOMeN', 'D-RhaOP', 'D-RhaOS', 'D-RibHex', 'D-RibHexNAc', 'D-Sor', 'D-ThrHexA4en', 'D-ThrHexAN4en', 'D-ThrHexfNAc2en', 'D-ThrPen', 'D-Thre-ol', 'DDAltHep', 'DDAltHepOMe', 'DDGalHep', 'DDGalHepOMe', 'DDGlcHep', 'DDManHep', 'DDManHepOBut', 'DDManHepOEtn', 'DDManHepOMe', 'DDManHepOP', 'DDManHepOPEtn', 'DDManHepOPGroA', 'DDManNonUloNAcOFoN-onic', 'DLAltNonUloNAc-onic', 'DLGalNonUloNAc-onic', 'DLGalNonUloNAcN', 'DLGalNonUloNAcN-onic', 'DLGlcHepOMe', 'DLHepGlcOMe', 'DLManHep', 'DLManHepOPEtn', 'Dha', 'Dig', 'DigCMe', 'DigOAc', 'DigOFo', 'DigOMe', 'Ery', 'Ery-L-GlcNonUloNAcOAcOMeSH-onic', 'Ery-ol', 'Ery-onic', 'EryHex', 'EryHex2en', 'EryHexA3en', 'EryOMe-onic', 'Fru', 'Fruf', 'FrufF', 'FrufI', 'FrufN', 'FrufNAc', 'FrufOAc', 'FrufOAcOBzOCoum', 'FrufOAcOFer', 'FrufOBzOCin', 'FrufOBzOCoum', 'FrufOBzOFer', 'FrufOFer', 'FrufOLau', 'Fuc', 'Fuc-ol', 'FucN', 'FucNAc', 'FucNAcA', 'FucNAcN', 'FucNAcNMe', 'FucNAcOAc', 'FucNAcOMe', 'FucNAcOPGro', 'FucNAla', 'FucNAm', 'FucNBut', 'FucNFo', 'FucNProp', 'FucNThrAc', 'FucOAc', 'FucOAcNAm', 'FucOAcNBut', 'FucOAcOMe', 'FucOAcOSOMe', 'FucOMe', 'FucOMeOPam', 'FucOMeOVac', 'FucOP', 'FucOPOMe', 'FucOS', 'FucOSOMe', 'Fucf', 'Gal', 'Gal-ol', 'Gal6Ulo', 'GalA', 'GalA-ol', 'GalAAla', 'GalAAlaLys', 'GalAGroN', 'GalALys', 'GalAN', 'GalANCys', 'GalANCysAc', 'GalANSerAc', 'GalAOLac', 'GalAOPyr', 'GalASer', 'GalAThr', 'GalAThrAc', 'GalCl', 'GalF', 'GalGro', 'GalGroN', 'GalN', 'GalNAc', 'GalNAc-ol', 'GalNAc-onic', 'GalNAcA', 'GalNAcAAla', 'GalNAcAN', 'GalNAcASer', 'GalNAcGro', 'GalNAcN', 'GalNAcOAc', 'GalNAcOAcA', 'GalNAcOAcAN', 'GalNAcOAcOMeA', 'GalNAcOAcOP', 'GalNAcOAcOPGro', 'GalNAcOMe', 'GalNAcOP', 'GalNAcOPCho', 'GalNAcOPEtn', 'GalNAcOPGro', 'GalNAcOPGroAN', 'GalNAcOPyr', 'GalNAcOS', 'GalNAla', 'GalNAmA', 'GalNCysGly', 'GalNFoA', 'GalNFoAN', 'GalNOPCho', 'GalNSuc', 'GalNonUloNAc-onic', 'GalOAc', 'GalOAcA', 'GalOAcAGroN', 'GalOAcAOLac', 'GalOAcAThr', 'GalOAcN', 'GalOAcNAla', 'GalOAcNAmA', 'GalOAcNFoA', 'GalOAcNFoAN', 'GalOAcOFoA', 'GalOAcOMe', 'GalOAcOP', 'GalOAcOPGro', 'GalOAcOPyr', 'GalOFoAN', 'GalOFoNAN', 'GalOLac', 'GalOLac-ol', 'GalOMe', 'GalOMeA', 'GalOMeCl', 'GalOMeF', 'GalOMeNAla', 'GalOP', 'GalOPA', 'GalOPAEtn', 'GalOPAN', 'GalOPCho', 'GalOPEtn', 'GalOPEtnA', 'GalOPEtnN', 'GalOPGro', 'GalOPy', 'GalOPyr', 'GalOS', 'GalOSA', 'GalOSOEt', 'GalOSOMeA', 'GalOctUloNAc-onic', 'Galf', 'GalfGro', 'GalfNAc', 'GalfOAc', 'GalfOAcGro', 'GalfOAcOLac', 'GalfOAcOPGro', 'GalfOLac', 'GalfOMe', 'GalfOP', 'GalfOPCho', 'GalfOPGro', 'GalfOPyr', 'Gl', 'Glc', 'Glc-ol', 'Glc6Ulo', 'GlcA', 'GlcAAla', 'GlcAAlaLys', 'GlcAGlu', 'GlcAGly', 'GlcAGro', 'GlcAGroN', 'GlcALys', 'GlcAN', 'GlcAOLac', 'GlcAOPy', 'GlcAOPyr', 'GlcASer', 'GlcAThr', 'GlcAThrAc', 'GlcCho', 'GlcF', 'GlcGro', 'GlcGroA', 'GlcI', 'GlcN', 'GlcN-ol', 'GlcNAc', 'GlcNAc-ol', 'GlcNAcA', 'GlcNAcAAla', 'GlcNAcAN', 'GlcNAcANAla', 'GlcNAcANAlaAc', 'GlcNAcANAlaFo', 'GlcNAcAla', 'GlcNAcCl', 'GlcNAcGlu', 'GlcNAcGly', 'GlcNAcGro', 'GlcNAcI', 'GlcNAcN', 'GlcNAcN-ol', 'GlcNAcNAla', 'GlcNAcNAlaFo', 'GlcNAcNAmA', 'GlcNAcNButA', 'GlcNAcOAc', 'GlcNAcOAcA', 'GlcNAcOAcN', 'GlcNAcOAcNAla', 'GlcNAcOAcOCmOOle', 'GlcNAcOAcOCmOPam', 'GlcNAcOAcOCmOVac', 'GlcNAcOAcOLac', 'GlcNAcOAcOOle', 'GlcNAcOAcOPam', 'GlcNAcOAcOPyr', 'GlcNAcOAcOS-ol', 'GlcNAcOAcOVac', 'GlcNAcOGc', 'GlcNAcOLac', 'GlcNAcOLacAla', 'GlcNAcOLacGro', 'GlcNAcOMe', 'GlcNAcOMeA', 'GlcNAcOP', 'GlcNAcOPCho', 'GlcNAcOPEtg', 'GlcNAcOPEtn', 'GlcNAcOPGro', 'GlcNAcOPGroA', 'GlcNAcOPOAch', 'GlcNAcOPyr', 'GlcNAcOS', 'GlcNAcOS-ol', 'GlcNAcOSA', 'GlcNAm', 'GlcNAmA', 'GlcNBut', 'GlcNButAN', 'GlcNButOAc', 'GlcNCmOCm', 'GlcNCmOCmOOle', 'GlcNCmOCmOVac', 'GlcNCmOVac', 'GlcNGc', 'GlcNGly', 'GlcNMe', 'GlcNMeOCm', 'GlcNMeOCmOPam', 'GlcNMeOCmOSte', 'GlcNMeOCmOVac', 'GlcNMeOSte', 'GlcNMeOVac', 'GlcNN', 'GlcNOAep', 'GlcNOCmOAch', 'GlcNOCmOVac', 'GlcNOMar', 'GlcNOMe', 'GlcNOMyr', 'GlcNOOle', 'GlcNOPam', 'GlcNOPyr', 'GlcNOSte', 'GlcNOVac', 'GlcNS', 'GlcNSOS', 'GlcNSOSOMe', 'GlcNSuc', 'GlcOAc', 'GlcOAcA', 'GlcOAcGro', 'GlcOAcGroA', 'GlcOAcN', 'GlcOAcNBut', 'GlcOAcNCmOOle', 'GlcOAcNCmOPam', 'GlcOAcNCmOVac', 'GlcOAcNMeOCm', 'GlcOAcNMeOCmOVac', 'GlcOAcNMeOVac', 'GlcOAcNOCmOVac', 'GlcOAcNOOle', 'GlcOAcNOPam', 'GlcOAcNOVac', 'GlcOAcOCoum', 'GlcOAcOFer', 'GlcOAcOOle', 'GlcOAcOP', 'GlcOAcOPGro', 'GlcOAcOPam', 'GlcOAcOS', 'GlcOAcOSA', 'GlcOAcOSte', 'GlcOButA', 'GlcOBz', 'GlcOCoum', 'GlcOEt', 'GlcOEtn', 'GlcOEtnA', 'GlcOEtnN', 'GlcOFer', 'GlcOFoN', 'GlcOGc', 'GlcOLac', 'GlcOMal', 'GlcOMe', 'GlcOMe-ol', 'GlcOMeA', 'GlcOMeAN', 'GlcOMeN', 'GlcOMeNOMyr', 'GlcOMeOFoA', 'GlcOMeOPyr', 'GlcOOle', 'GlcOP', 'GlcOP-ol', 'GlcOPA', 'GlcOPCho', 'GlcOPChoGro', 'GlcOPEtn', 'GlcOPEtnGro', 'GlcOPEtnN', 'GlcOPGro', 'GlcOPGroA', 'GlcOPN', 'GlcOPNOMyr', 'GlcOPNOPam', 'GlcOPOOle', 'GlcOPOPGro', 'GlcOPPEtn', 'GlcOPPEtnN', 'GlcOPam', 'GlcOPyr', 'GlcOS', 'GlcOSA', 'GlcOSN', 'GlcOSNMeOCm', 'GlcOSOEt', 'GlcOSOMe', 'GlcOSOMeA', 'GlcOSin', 'GlcS', 'GlcSH', 'GlcThr', 'Glcf', 'Gro', 'Gro-ol', 'Gul', 'GulAN', 'GulNAcA', 'GulNAcAN', 'GulNAcNAmA', 'GulNAcOAcA', 'Hep', 'HepOP', 'HepOPEtn', 'HepOPPEtn', 'Hex', 'HexA', 'HexN', 'HexNAc', 'HexOMeOFo', 'Hexf', 'Ido', 'IdoA', 'IdoN', 'IdoNAc', 'IdoOAcA', 'IdoOAcOSA', 'IdoOMeA', 'IdoOS', 'IdoOSA', 'IdoOSOEtA', 'IdoOSOMeA', 'Kdn', 'KdnOAc', 'KdnOMe', 'KdnOPyr', 'Kdo', 'Kdo-ol', 'KdoN', 'KdoOAc', 'KdoOAcOS', 'KdoOMe', 'KdoOP', 'KdoOPEtn', 'KdoOPGro', 'KdoOPN', 'KdoOPOEtn', 'KdoOPOPEtn', 'KdoOPPEtn', 'KdoOPPEtnN', 'KdoOPyr', 'KdoOS', 'Kdof', 'Ko', 'KoOMe', 'KoOPEtn', 'L-4dEryHexAN4en', 'L-4dThrHex4en', 'L-4dThrHexA4en', 'L-4dThrHexA4enAla', 'L-4dThrHexAN4en', 'L-4dThre-ol', 'L-6dAraHex', 'L-6dAraHexOMe', 'L-6dGalHep', 'L-6dGalHepOP', 'L-6dGulHep', 'L-6dGulHepOMe', 'L-6dGulHepOP', 'L-6dXylHexNAc4Ulo', 'L-Aco', 'L-AcoOMe', 'L-AcoOMeOFo', 'L-BoiOMe', 'L-Cym', 'L-CymOAc', 'L-DigOMe', 'L-Ery', 'L-EryCMeOH', 'L-EryHexA4en', 'L-Fru', 'L-Fruf', 'L-Gal', 'L-GalAN', 'L-GalNAc', 'L-GalNAc-onic', 'L-GalNAcA', 'L-GalNAcAN', 'L-GalNAcOAcA', 'L-GalNAmA', 'L-GalOAcNAmA', 'L-GalOS', 'L-Glc', 'L-GlcA', 'L-GlcNAc', 'L-GlcOMe', 'L-Gro-onic', 'L-GroHexUlo', 'L-Gul', 'L-Gul-onic', 'L-GulA', 'L-GulAN', 'L-GulHep', 'L-GulNAc', 'L-GulNAcA', 'L-GulNAcAGly', 'L-GulNAcAN', 'L-GulNAcANEtn', 'L-GulNAcNAmA', 'L-GulNAcNEtnA', 'L-GulNAcOAc', 'L-GulNAcOAcA', 'L-GulNAcOAcAN', 'L-GulNAcOEtA', 'L-GulNAcOEtnA', 'L-GulOAcA', 'L-Lyx', 'L-LyxHex', 'L-LyxHexNMe', 'L-LyxHexOMe', 'L-Man', 'L-ManOMe', 'L-ManOctUlo-onic', 'L-Ole', 'L-OleOAc', 'L-Oli', 'L-OliOMe', 'L-Qui', 'L-QuiN', 'L-QuiNAc', 'L-QuiNAcOMe', 'L-QuiNAcOP', 'L-QuiOMeN', 'L-RibHex', 'L-Ribf', 'L-Tal', 'L-The', 'L-TheOAc', 'L-Thr', 'L-ThrHexA4en', 'L-ThrHexAN4en', 'L-ThrHexOMe4en', 'L-ThrHexOMeA4en', 'L-ThrHexOSA4en', 'L-Xyl', 'L-XylHex', 'L-XylOMe', 'LDGalHep', 'LDGalNonUloNAc-onic', 'LDGlcHep', 'LDIdoHep', 'LDIdoHepPro', 'LDManHep', 'LDManHepGroN', 'LDManHepOAc', 'LDManHepOCm', 'LDManHepOEtn', 'LDManHepOMe', 'LDManHepOP', 'LDManHepOPEtn', 'LDManHepOPEtnOEtn', 'LDManHepOPGroA', 'LDManHepOPOCm', 'LDManHepOPOMe', 'LDManHepOPOPEtn', 'LDManHepOPOPPEtn', 'LDManHepOPPEtn', 'LDManHepOPPEtnOPyrP', 'LDManHepOPyrP', 'LDManNonUloNAcOFoN-onic', 'LDManNonUloOFoNN-onic', 'LLManNonUloOFoN-onic', 'Leg', 'LegNAc', 'LegNAcAla', 'LegNAcNAc', 'LegNAcNAla', 'LegNAcNAm', 'LegNAcNBut', 'LegNFo', 'Lyx', 'LyxHex', 'LyxHexOMe', 'LyxOMe', 'LyxOctUlo-onic', 'Lyxf', 'M', 'Man', 'Man-ol', 'ManA', 'ManCMe', 'ManF', 'ManN', 'ManNAc', 'ManNAcA', 'ManNAcAAla', 'ManNAcAGro', 'ManNAcAN', 'ManNAcANOOrn', 'ManNAcASer', 'ManNAcAThr', 'ManNAcGroA', 'ManNAcNAmA', 'ManNAcNEtnA', 'ManNAcOAc', 'ManNAcOAcA', 'ManNAcOLac', 'ManNAcOMe', 'ManNAcOMeAN', 'ManNAcOPEtn', 'ManNAcOPGro', 'ManNAcOPGroA', 'ManNAcOPyr', 'ManNBut', 'ManNOPGro', 'ManNonUloNAc-onic', 'ManOAc', 'ManOAcA', 'ManOAcN', 'ManOAcOMe', 'ManOAcOPyr', 'ManOAep', 'ManOBut', 'ManOEtn', 'ManOLac', 'ManOMe', 'ManOMeA', 'ManOP', 'ManOP-ol', 'ManOPCho', 'ManOPEtn', 'ManOPGro', 'ManOPOMe', 'ManOPOPyr-ol', 'ManOPy', 'ManOPyr', 'ManOS', 'ManOctUlo', 'ManSH', 'Manf', 'Mur', 'MurNAc', 'MurNAcAla', 'MurNAcOP', 'MurNAcSer', 'Neu', 'NeuNAc', 'NeuNAcN', 'NeuNAcNAc', 'NeuNAcNMe', 'NeuNAcOAc', 'NeuNAcOAcOMe', 'NeuNAcOGc', 'NeuNAcOMe', 'NeuNAcOS', 'NeuNGc', 'NeuNGcA', 'NeuNGcN', 'NeuNGcOMe', 'NeuNGcOS', 'NeuOFo', 'NeuOMe', 'OLac', 'Ole', 'Oli', 'OliN', 'OliNAc', 'OliOMe', 'Par', 'Parf', 'PerNAc', 'Pse', 'PseNAc', 'PseNAcNAc', 'PseNAcNAcNBut', 'PseNAcNAcOBut', 'PseNAcNAm', 'PseNAcNBut', 'PseNAcNFo', 'PseNAcNGro', 'PseNAcOAcNBut', 'PseNAcOBut', 'PseNButNFo', 'PseNGcNAm', 'PseOAc', 'PseOAcOFo', 'PseOFo', 'Qui', 'QuiN', 'QuiNAc', 'QuiNAc-ol', 'QuiNAcGro', 'QuiNAcN', 'QuiNAcNAlaAc', 'QuiNAcNAm', 'QuiNAcNAspAc', 'QuiNAcNBut', 'QuiNAcNButGro', 'QuiNAcNGroA', 'QuiNAcOAc', 'QuiNAcOBut', 'QuiNAcOMe', 'QuiNAcOP', 'QuiNAcOPGro', 'QuiNAla', 'QuiNAlaAc', 'QuiNAlaAcGro', 'QuiNAlaBut', 'QuiNAlaButGro', 'QuiNAspAc', 'QuiNBut', 'QuiNButAla', 'QuiNButOMe', 'QuiNFo', 'QuiNGlyAc', 'QuiNHse', 'QuiNHseGro', 'QuiNLac', 'QuiNMal', 'QuiNSerAc', 'QuiNThrAc', 'QuiOMe', 'QuiOMeN', 'QuiOS', 'QuiOSN', 'QuiOSNBut', 'Rha', 'Rha-ol', 'RhaCMe', 'RhaCl', 'RhaGro', 'RhaGroA', 'RhaNAc', 'RhaNAcNBut', 'RhaNAcNFo', 'RhaNAcOAc', 'RhaNPro', 'RhaOAc', 'RhaOAcOLac', 'RhaOAcOMe', 'RhaOBut', 'RhaOFer', 'RhaOLac', 'RhaOMe', 'RhaOMeCMeNLac', 'RhaOMeCMeOFo', 'RhaOP', 'RhaOPEtn', 'RhaOPGro', 'RhaOPOMe', 'RhaOProp', 'RhaOPyr', 'RhaOS', 'Rhaf', 'Rib', 'Rib-ol', 'RibOAc', 'RibOAcOP-ol', 'RibOP-ol', 'RibOPEtn-ol', 'RibOPGro-ol', 'RibOPOPGro-ol', 'Ribf', 'Ribf-uronic', 'RibfOAc', 'Sed', 'Sedf', 'Sor', 'Sorf', 'Suc', 'Sug', 'SugOAc', 'Tag', 'Tal', 'The', 'Thr', 'Thre-ol', 'Thre-onic', 'Tyv', 'VioNAc', 'Xluf', 'XlufOMe', 'Xyl', 'Xyl-ol', 'Xyl-onic', 'XylHex', 'XylHexNAc', 'XylHexUlo', 'XylHexUloN', 'XylHexUloNAc', 'XylNAc', 'XylNMe', 'XylOAc', 'XylOBz', 'XylOMe', 'XylOP', 'XylOS', 'Xylf', 'Yer', 'YerOAc', 'a-Tri-ol', 'a-Tri-onic', 'a1-1', 'a1-2', 'a1-3', 'a1-4', 'a1-5', 'a1-6', 'a1-7', 'a1-8', 'a2-1', 'a2-2', 'a2-3', 'a2-4', 'a2-5', 'a2-6', 'a2-7', 'a2-8', 'a2-9', 'a6-6', 'aldehyde-2,5-Anhydro-L-Man', 'aldehyde-2,5-Anhydro-Tal', 'aldehyde-Gro', 'aldehyde-Hex', 'aldehyde-L-Gro', 'aldehyde-L-GroN', 'aldehyde-QuiNAc', 'aldehyde-Rib', 'aldehyde-a-Tri-ol', 'aldehyde-b-Tri-ol', 'b-Tri-N-ol', 'b-Tri-OP-ol', 'b-Tri-ol', 'b-Tri-onic', 'b1-1', 'b1-2', 'b1-3', 'b1-3FucNAc', 'b1-4', 'b1-4Glc', 'b1-5', 'b1-6', 'b1-7', 'b1-8', 'b1-9', 'b2-1', 'b2-2', 'b2-3', 'b2-4', 'b2-5', 'b2-6', 'b2-7', 'b2-8', 'b3-3', 'cNeuNAc']

def get_namespace():
  """returns list of function names in this package"""
  return ['unwrap, find_nth, small_motif_find, min_process_glycans, motif_find, \
process_glycans, character_to_label, string_to_labels, pad_sequence, get_lib, \
glycan_to_graph, dataset_to_graphs, hierarchy_filter, compare_glycans']

def unwrap(nested_list):
  """converts a nested list into a flat list"""
  out = [item for sublist in nested_list for item in sublist]
  return out

def find_nth(haystack, needle, n):
  """finds n-th instance of motif
  haystack -- string to search for motif
  needle -- motif
  n -- n-th occurrence in string

  returns starting index of n-th occurrence in string 
  """
  start = haystack.find(needle)
  while start >= 0 and n > 1:
    start = haystack.find(needle, start+len(needle))
    n -= 1
  return start

def small_motif_find(s):
  """processes IUPACcondensed glycan sequence (string) without splitting it into glycowords"""
  b = s.split('(')
  b = [k.split(')') for k in b]
  b = [item for sublist in b for item in sublist]
  b = [k.strip('[') for k in b]
  b = [k.strip(']') for k in b]
  b = [k.replace('[', '') for k in b]
  b = [k.replace(']', '') for k in b]
  b = '*'.join(b)
  return b

def min_process_glycans(glycan_list):
  """converts list of glycans into a nested lists of glycowords"""
  glycan_motifs = [small_motif_find(k) for k in glycan_list]
  glycan_motifs = [i.split('*') for i in glycan_motifs]
  return glycan_motifs

def motif_find(s, exhaustive = False):
  """processes IUPACcondensed glycan sequence (string) into glycowords
  s -- glycan string
  exhaustive -- True for processing glycans shorter than one glycoword

  returns list of glycowords
  """
  b = s.split('(')
  b = [k.split(')') for k in b]
  b = [item for sublist in b for item in sublist]
  b = [k.strip('[') for k in b]
  b = [k.strip(']') for k in b]
  b = [k.replace('[', '') for k in b]
  b = [k.replace(']', '') for k in b]
  if exhaustive:
    if len(b) >= 5:
      b = ['*'.join(b[i:i+5]) for i in range(0, len(b)-4, 2)]
    else:
      b = ['*'.join(b)]
  else:
    b = ['*'.join(b[i:i+5]) for i in range(0, len(b)-4, 2)]
  return b

def process_glycans(glycan_list, exhaustive = False):
  """wrapper function to process list of glycans into glycowords
  glycan_list -- list of IUPACcondensed glycan sequences (string)
  exhaustive -- True for processing glycans shorter than one glycoword

  returns nested list of glycowords for every glycan
  """
  glycan_motifs = [motif_find(k, exhaustive = exhaustive) for k in glycan_list]
  glycan_motifs = [[i.split('*') for i in k] for k in glycan_motifs]
  return glycan_motifs

def character_to_label(character, libr):
  """tokenizes character by indexing passed library
  character -- character to index
  libr -- list of library items

  returns index of character in library
  """
  character_label = libr.index(character)
  return character_label

def string_to_labels(character_string, libr):
  """tokenizes word by indexing characters in passed library
  character_string -- string of characters to index
  libr -- list of library items

  returns indexes of characters in library
  """
  return list(map(lambda character: character_to_label(character, libr), character_string))

def pad_sequence(seq, max_length, pad_label = len(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T',
     'V','W','Y','X'])):
  """brings all sequences to same length by adding padding token
  seq -- sequence to pad
  max_length -- sequence length to pad to
  pad_label -- which padding label to use

  returns padded sequence
  """
  seq += [pad_label for i in range(max_length-len(seq))]
  return seq

def get_lib(glycan_list, mode='letter', exhaustive = True):
  """returns sorted list of unique glycoletters in list of glycans
  mode -- default is letter for glycoletters; change to obtain glycowords
  exhaustive -- if True, processes glycans shorted than 1 glycoword; default is True"""
  proc = process_glycans(glycan_list, exhaustive = exhaustive)
  lib = unwrap(proc)
  lib = list(set([tuple(k) for k in lib]))
  lib = [list(k) for k in lib]
  if mode=='letter':
    lib = list(sorted(list(set(unwrap(lib)))))
  else:
    lib = list(sorted(list(set([tuple(k) for k in lib]))))
  return lib

def glycan_to_graph(glycan, libr = lib):
  """the monumental function for converting glycans into graphs
  glycan -- IUPACcondensed glycan sequence (string)
  lib -- sorted list of unique glycoletters observed in the glycans of our dataset

  returns (1) a list of labeled glycoletters from the glycan / node list
          (2) two lists to indicate which glycoletters are connected in the glycan graph / edge list
  """
  bracket_count = glycan.count('[')
  parts = []
  branchbranch = []
  branchbranch2 = []
  position_bb = []
  b_counts = []
  bb_count = 0
  #checks for branches-within-branches and handles them
  if bool(re.search('\[[^\]]+\[', glycan)):
    double_pos = [(k.start(),k.end()) for k in re.finditer('\[[^\]]+\[', glycan)]
    for spos, pos in double_pos:
      bracket_count -= 1
      glycan_part = glycan[spos+1:]
      glycan_part = glycan_part[glycan_part.find('['):]
      idx = [k.end() for k in re.finditer('\][^\(]+\(', glycan_part)][0]
      idx2 = [k.start() for k in re.finditer('\][^\(]+\(', glycan_part)][0]
      branchbranch.append(glycan_part[:idx-1].replace(']','').replace('[',''))
      branchbranch2.append(glycan[pos-1:])
      glycan_part = glycan[:pos-1]
      b_counts.append(glycan_part.count('[')-bb_count)
      glycan_part = glycan_part[glycan_part.rfind('[')+1:]
      position_bb.append(glycan_part.count('(')*2)
      bb_count += 1
    for b in branchbranch2:
      glycan =  glycan.replace(b, ']'.join(b.split(']')[1:]))
  main = re.sub("[\[].*?[\]]", "", glycan)
  position = []
  branch_points = [x.start() for x in re.finditer('\]', glycan)]
  for i in branch_points:
    glycan_part = glycan[:i+1]
    glycan_part = re.sub("[\[].*?[\]]", "", glycan_part)
    position.append(glycan_part.count('(')*2)
  parts.append(main)

  for k in range(1,bracket_count+1):
    start = find_nth(glycan, '[', k) + 1
    #checks whether glycan continues after branch
    if bool(re.search("[\]][^\[]+[\(]", glycan[start:])):
      #checks for double branches and removes second branch
      if bool(re.search('\]\[', glycan[start:])):
        glycan_part = re.sub("[\[].*?[\]]", "", glycan[start:])
        end = re.search("[\]].*?[\(]", glycan_part).span()[1] - 1
        parts.append(glycan_part[:end].replace(']',''))
      else:
        end = re.search("[\]].*?[\(]", glycan[start:]).span()[1] + start -1
        parts.append(glycan[start:end].replace(']',''))
    else:
      if bool(re.search('\]\[', glycan[start:])):
        glycan_part = re.sub("[\[].*?[\]]", "", glycan[start:])
        end = len(glycan_part)
        parts.append(glycan_part[:end].replace(']',''))
      else:
        end = len(glycan)
        parts.append(glycan[start:end].replace(']',''))
  
  try:
    for bb in branchbranch:
      parts.append(bb)
  except:
    pass

  parts = min_process_glycans(parts)
  parts_lengths = [len(j) for j in parts]
  parts_tokenized = [string_to_labels(k, libr) for k in parts]
  parts_tokenized = [parts_tokenized[0]] + [parts_tokenized[k][:-1] for k in range(1,len(parts_tokenized))]
  parts_tokenized = [item for sublist in parts_tokenized for item in sublist]
  
  range_list = list(range(len([item for sublist in parts for item in sublist])))
  init = 0
  parts_positions = []
  for k in parts_lengths:
    parts_positions.append(range_list[init:init+k])
    init += k

  for j in range(1,len(parts_positions)-len(branchbranch)):
    parts_positions[j][-1] = position[j-1]
  for j in range(1, len(parts_positions)):
    try:
      for z in range(j+1,len(parts_positions)):
        parts_positions[z][:-1] = [o-1 for o in parts_positions[z][:-1]]
    except:
      pass
  try:
    for i,j in enumerate(range(len(parts_positions)-len(branchbranch), len(parts_positions))):
      parts_positions[j][-1] = parts_positions[b_counts[i]][position_bb[i]]
  except:
    pass

  pairs = []
  for i in parts_positions:
    pairs.append([(i[m],i[m+1]) for m in range(0,len(i)-1)])
  pairs = list(zip(*[item for sublist in pairs for item in sublist]))
  return parts_tokenized, pairs



def dataset_to_graphs(glycan_list, labels, libr = lib, label_type = torch.long, separate = False,
                      context = False, error_catch = False, wo_labels = False):
  """wrapper function to convert a whole list of glycans into a graph dataset
  glycan_list -- list of IUPACcondensed glycan sequences (string)
  label_type -- which tensor type for label, default is torch.long for binary labels, change to torch.float for continuous
  separate -- True returns node list / edge list / label list as separate files; False returns list of data tuples; default is False
  lib -- sorted list of unique glycoletters observed in the glycans of our dataset
  context -- legacy-ish; used for generating graph context dataset for pre-training; keep at False
  error_catch -- troubleshooting option, True will print glycans that cannot be converted into graphs; default is False
  wo_labels -- change to True if you do not want to pass and receive labels; default is False

  returns list of node list / edge list / label list data tuples
  """
  if error_catch:
    glycan_graphs = []
    for k in glycan_list:
      try:
        glycan_graphs.append(glycan_to_graph(k, libr))
      except:
        print(k)
  else:
    glycan_graphs = [glycan_to_graph(k, libr) for k in glycan_list]
  if separate:
    glycan_nodes, glycan_edges = zip(*glycan_graphs)
    return list(glycan_nodes), list(glycan_edges), labels
  else:
    if context:
      contexts = [ggraph_to_context(k, lib=lib) for k in glycan_graphs]
      labels = [k[1] for k in contexts]
      labels = [item for sublist in labels for item in sublist]
      contexts = [k[0] for k in contexts]
      contexts = [item for sublist in contexts for item in sublist]
      data = [Data(x = torch.tensor(contexts[k][0], dtype = torch.long),
                 y = torch.tensor(labels[k], dtype = label_type),
                 edge_index = torch.tensor([contexts[k][1][0],contexts[k][1][1]], dtype = torch.long)) for k in range(len(contexts))]
      return data
    else:
      if wo_labels:
        glycan_nodes, glycan_edges = zip(*glycan_graphs)
        glycan_graphs = list(zip(glycan_nodes, glycan_edges))
        data = [Data(x = torch.tensor(k[0], dtype = torch.long),
                     edge_index = torch.tensor([k[1][0],k[1][1]], dtype = torch.long)) for k in glycan_graphs]
        return data
      else:
        glycan_nodes, glycan_edges = zip(*glycan_graphs)
        glycan_graphs = list(zip(glycan_nodes, glycan_edges, labels))
        data = [Data(x = torch.tensor(k[0], dtype = torch.long),
                 y = torch.tensor([k[2]], dtype = label_type),
                 edge_index = torch.tensor([k[1][0],k[1][1]], dtype = torch.long)) for k in glycan_graphs]
        return data

def hierarchy_filter(df_in, rank = 'domain', min_seq = 5):
  """stratified data split in train/test at the taxonomic level, removing duplicate glycans and infrequent classes"""
  df = copy.deepcopy(df_in)
  rank_list = ['species','genus','family','order','class','phylum','kingdom','domain']
  rank_list.remove(rank)
  df.drop(rank_list, axis = 1, inplace = True)
  class_list = list(set(df[rank].values.tolist()))
  temp = []

  for i in range(len(class_list)):
    t = df[df[rank] == class_list[i]]
    t = t.drop_duplicates('target', keep = 'first')
    temp.append(t)
  df = pd.concat(temp).reset_index(drop = True)

  counts = df[rank].value_counts()
  allowed_classes = [counts.index.tolist()[k] for k in range(len(counts.index.tolist())) if (counts >= min_seq).values.tolist()[k]]
  df = df[df[rank].isin(allowed_classes)]

  class_list = list(sorted(list(set(df[rank].values.tolist()))))
  class_converter = {class_list[k]:k for k in range(len(class_list))}
  df[rank] = [class_converter[k] for k in df[rank].values.tolist()]

  sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2)
  sss.get_n_splits(df.target.values.tolist(), df[rank].values.tolist())
  for i, j in sss.split(df.target.values.tolist(), df[rank].values.tolist()):
    train_x = [df.target.values.tolist()[k] for k in i]
    len_train_x = [len(k) for k in train_x]

    val_x = [df.target.values.tolist()[k] for k in j]
    id_val = list(range(len(val_x)))
    len_val_x = [len(k) for k in val_x]

    train_y = [df[rank].values.tolist()[k] for k in i]

    val_y = [df[rank].values.tolist()[k] for k in j]
    id_val = [[id_val[k]] * len_val_x[k] for k in range(len(len_val_x))]
    id_val = [item for sublist in id_val for item in sublist]

  return train_x, val_x, train_y, val_y, id_val, class_list, class_converter

def compare_glycans(glycan_a, glycan_b, libr = lib):
  """returns True if glycans are the same and False if not
  glycan_a -- glycan in string format (IUPACcondensed)"""
  glycan_graph_a = glycan_to_graph(glycan_a, libr)
  edgelist = list(zip(glycan_graph_a[1][0], glycan_graph_a[1][1]))
  g1 = nx.from_edgelist(edgelist)
  nx.set_node_attributes(g1, glycan_graph_a[0],'labels')
  
  glycan_graph_b = glycan_to_graph(glycan_b, libr)
  edgelist = list(zip(glycan_graph_b[1][0], glycan_graph_b[1][1]))
  g2 = nx.from_edgelist(edgelist)
  nx.set_node_attributes(g2, glycan_graph_b[0],'labels')

  return nx.is_isomorphic(g1, g2)
  
