"""Isomeric SMILES for glycans, straight from IUPAC-condensed sequences.

Every monosaccharide is stored once as a SMILES template rooted at its anomeric oxygen, with a
named slot at every substitutable position. Attaching a child residue or a modification is then a
string substitution, so building a glycan needs no cheminformatics toolkit at all: the ring
perception, carbon numbering and stereo-perception that a toolkit would redo per call are baked
into the templates by bin/build_smiles_templates.py, which is the only place RDKit is used.

Templates carry three placeholders: '{a}' is the anomeric tag ('' or '@', taken from the skeleton's
alpha entry; the beta anomer is the other one and an undefined anomer drops the tag), '{r}' is the
ring-closure digit, and '{pN}' is the substituent at carbon N. An unfilled slot falls back to the
skeleton's own heteroatom, 'O' for a hydroxyl and 'N' for an amine.
"""

import re
from typing import NamedTuple
import networkx as nx
from glycowork.motif.graph import glycan_to_nxGraph, graph_to_string, glycan_graph_memoize

CERAMIDE = 'OC[C@@H](NC(=O)CCCCCCCCCCCCCCC)[C@H](O)/C=C/CCCCCCCCCCCCC'  # d18:1/16:0, the placeholder used whenever a sequence just says 'Cer'
UNKNOWN_POSITION = 'lowest'  # where a modification without a position number goes; 'lowest' free slot reproduces what GlyLES did

SKELETONS = {  # monosaccharide: (template, alpha tag, {position: heteroatom})
    '4eLeg': ('O[C@{a}]{r}(C(=O)O)C[C@@H]({p4})[C@@H]({p5})[C@H]([C@H]({p7})[C@H]({p8})C)O{r}', '', {4: 'O', 5: 'N', 7: 'N', 8: 'O'}),
    '6dAlt': ('O[C@{a}H]{r}O[C@@H](C)[C@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 4: 'O'}),
    '6dAltf': ('O[C@{a}H]{r}O[C@@H]([C@@H]({p5})C)[C@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 5: 'O'}),
    '6dGul': ('O[C@{a}H]{r}O[C@H](C)[C@H]({p4})[C@@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O'}),
    '6dTal': ('O[C@{a}H]{r}O[C@H](C)[C@H]({p4})[C@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O'}),
    'Abe': ('O[C@{a}H]{r}O[C@H](C)[C@H]({p4})C[C@H]{r}{p2}', '', {2: 'O', 4: 'O'}),
    'Abef': ('O[C@{a}H]{r}O[C@@H]([C@H]({p5})C)C[C@H]{r}{p2}', '@', {2: 'O', 5: 'O'}),
    'Acef': ('O[C@{a}H]{r}O[C@@H](C)[C@]({p3})(C(=O)O)[C@@H]{r}{p2}', '@', {2: 'O', 3: 'O'}),
    'Aci': ('O[C@{a}]{r}(C(=O)O)C[C@H]({p4})[C@@H]({p5})[C@H]([C@@H]({p7})[C@@H]({p8})C)O{r}', '', {4: 'O', 5: 'N', 7: 'N', 8: 'O'}),
    'Aco': ('O[C@{a}H]{r}O[C@@H](C)[C@H]({p4})[C@@H]({p3}C)[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 4: 'O'}),
    'All': ('O[C@{a}H]{r}O[C@@H](C{p6})[C@H]({p4})[C@H]({p3})[C@@H]{r}{p2}', '@', {2: 'O', 3: 'O', 4: 'O', 6: 'O'}),
    'Allf': ('O[C@{a}H]{r}O[C@@H]([C@@H]({p5})C{p6})[C@H]({p3})[C@@H]{r}{p2}', '@', {2: 'O', 3: 'O', 5: 'O', 6: 'O'}),
    'Alt': ('O[C@{a}H]{r}O[C@@H](C{p6})[C@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 4: 'O', 6: 'O'}),
    'Altf': ('O[C@{a}H]{r}O[C@@H]([C@@H]({p5})C{p6})[C@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 5: 'O', 6: 'O'}),
    'Api': ('O[C@{a}H]{r}OC[C@]({p3})(CO)[C@H]{r}{p2}', '@', {2: 'O', 3: 'O'}),
    'Apif': ('O[C@{a}H]{r}OC[C@]({p3})(CO)[C@H]{r}{p2}', '@', {2: 'O', 3: 'O'}),
    'Ara': ('O[C@{a}H]{r}OC[C@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 4: 'O'}),
    'Araf': ('O[C@{a}H]{r}O[C@@H](C{p5})[C@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 5: 'O'}),
    'Asc': ('O[C@{a}H]{r}O[C@@H](C)[C@H]({p4})C[C@H]{r}{p2}', '@', {2: 'O', 4: 'O'}),
    'Bac': ('O[C@{a}H]{r}O[C@H](C)[C@@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '', {2: 'N', 3: 'O', 4: 'N'}),
    'Col': ('O[C@{a}H]{r}O[C@@H](C)[C@@H]({p4})C[C@@H]{r}{p2}', '@', {2: 'O', 4: 'O'}),
    'DDAltHep': ('O[C@{a}H]{r}O[C@@H]([C@H]({p6})C{p7})[C@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
    'DDGlcHep': ('O[C@{a}H]{r}O[C@H]([C@H]({p6})C{p7})[C@@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
    'DDManHep': ('O[C@{a}H]{r}O[C@H]([C@H]({p6})C{p7})[C@@H]({p4})[C@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
    'DLGlcHep': ('O[C@{a}H]{r}O[C@@H]([C@H]({p6})C{p7})[C@H]({p4})[C@@H]({p3})[C@@H]{r}{p2}', '@', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
    'Dha': ('O[C@{a}]{r}(C(=O)O)C[C@@H]({p4})[C@@H]({p5})[C@@H](C(=O)O)O{r}', '', {4: 'O', 5: 'O'}),
    'Dig': ('O[C@{a}H]{r}C[C@H]({p3})[C@H]({p4})[C@@H](C)O{r}', '@', {3: 'O', 4: 'O'}),
    'Erwiniose': ('O[C@{a}H]{r}O[C@H](C)[C@@]({p4})([C@@H](O)C[C@@H](C)O)C[C@H]{r}{p2}', '', {2: 'O', 4: 'O'}),
    'Eryf': ('O[C@{a}H]{r}OC[C@@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O'}),
    'Fru': ('O[C@{a}]{r}(C{p1})OC[C@@H]({p5})[C@@H]({p4})[C@@H]{r}{p3}', '@', {1: 'O', 3: 'O', 4: 'O', 5: 'O'}),
    'Fruf': ('O[C@{a}]{r}(C{p1})O[C@H](C{p6})[C@@H]({p4})[C@@H]{r}{p3}', '@', {1: 'O', 3: 'O', 4: 'O', 6: 'O'}),
    'Fuc': ('O[C@{a}H]{r}O[C@@H](C)[C@@H]({p4})[C@@H]({p3})[C@@H]{r}{p2}', '@', {2: 'O', 3: 'O', 4: 'O'}),
    'Fucf': ('O[C@{a}H]{r}O[C@H]([C@@H]({p5})C)[C@@H]({p3})[C@@H]{r}{p2}', '@', {2: 'O', 3: 'O', 5: 'O'}),
    'Fus': ('O[C@{a}]{r}(C(=O)O)C[C@@H]({p4})[C@H]({p5})[C@H]([C@@H]({p7})[C@@H]({p8})C)O{r}', '', {4: 'O', 5: 'N', 7: 'O', 8: 'O'}),
    'Gal': ('O[C@{a}H]{r}O[C@H](C{p6})[C@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O'}),
    'GalHep': ('O[C@{a}H]{r}O[C@H](C({p6})C{p7})[C@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
    'Galf': ('O[C@{a}H]{r}O[C@@H]([C@H]({p5})C{p6})[C@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 5: 'O', 6: 'O'}),
    'Glc': ('O[C@{a}H]{r}O[C@H](C{p6})[C@@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O'}),
    'GlcHep': ('O[C@{a}H]{r}O[C@H](C({p6})C{p7})[C@@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
    'Glcf': ('O[C@{a}H]{r}O[C@H]([C@H]({p5})C{p6})[C@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 5: 'O', 6: 'O'}),
    'Gul': ('O[C@{a}H]{r}O[C@H](C{p6})[C@H]({p4})[C@@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O'}),
    'Gulf': ('O[C@{a}H]{r}O[C@@H]([C@H]({p5})C{p6})[C@@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 5: 'O', 6: 'O'}),
    'Ins': ('O[C@@H]{r}[C@H]({p2})[C@H]({p3})[C@@H]({p4})[C@H]({p5})[C@H]{r}{p6}', '', {2: 'O', 3: 'O', 4: 'O', 5: 'O', 6: 'O'}),  # 1D-myo, numbered off 1D-myo-inositol 1,4,5-trisphosphate
    'Hex': ('OC{r}OC(C{p6})C({p4})C({p3})C{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O'}),
    'dHex': ('OC{r}OC(C)C({p4})C({p3})C{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O'}),
    'Pen': ('OC{r}OCC({p4})C({p3})C{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O'}),
    'Ido': ('O[C@{a}H]{r}O[C@@H](C{p6})[C@@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 4: 'O', 6: 'O'}),
    'IdoHep': ('O[C@{a}H]{r}O[C@@H](C({p6})C{p7})[C@@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
    'Idof': ('O[C@{a}H]{r}O[C@H]([C@@H]({p5})C{p6})[C@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 5: 'O', 6: 'O'}),
    'Kdn': ('O[C@{a}]{r}(C(=O)O)C[C@H]({p4})[C@@H]({p5})[C@H]([C@H]({p7})[C@H]({p8})C{p9})O{r}', '', {4: 'O', 5: 'O', 7: 'O', 8: 'O', 9: 'O'}),
    'Kdo': ('O[C@{a}]{r}(C(=O)O)C[C@@H]({p4})[C@@H]({p5})[C@@H]([C@H]({p7})C{p8})O{r}', '', {4: 'O', 5: 'O', 7: 'O', 8: 'O'}),
    'Kdof': ('O[C@{a}]{r}(C(=O)O)C[C@@H]({p4})[C@H]([C@H]({p6})[C@H]({p7})C{p8})O{r}', '', {4: 'O', 6: 'O', 7: 'O', 8: 'O'}),
    'Ko': ('O[C@{a}]{r}(C(=O)O)O[C@H]([C@H]({p7})C{p8})[C@H]({p5})[C@H]({p4})[C@@H]{r}{p3}', '@', {3: 'O', 4: 'O', 5: 'O', 7: 'O', 8: 'O'}),
    'LDAltHep': ('O[C@{a}H]{r}O[C@@H]([C@@H]({p6})C{p7})[C@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
    'LDGlcHep': ('O[C@{a}H]{r}O[C@H]([C@@H]({p6})C{p7})[C@@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
    'LDManHep': ('O[C@{a}H]{r}O[C@H]([C@@H]({p6})C{p7})[C@@H]({p4})[C@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
    'LLGlcHep': ('O[C@{a}H]{r}O[C@@H]([C@@H]({p6})C{p7})[C@H]({p4})[C@@H]({p3})[C@@H]{r}{p2}', '@', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
    'Leg': ('O[C@{a}]{r}(C(=O)O)C[C@H]({p4})[C@@H]({p5})[C@H]([C@H]({p7})[C@H]({p8})C)O{r}', '', {4: 'O', 5: 'N', 7: 'N', 8: 'O'}),
    'Lyx': ('O[C@{a}H]{r}OC[C@@H]({p4})[C@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O'}),
    'Lyxf': ('O[C@{a}H]{r}O[C@H](C{p5})[C@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O', 5: 'O'}),
    'Man': ('O[C@{a}H]{r}O[C@H](C{p6})[C@@H]({p4})[C@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O'}),
    'ManHep': ('O[C@{a}H]{r}O[C@H](C({p6})C{p7})[C@@H]({p4})[C@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
    'Manf': ('O[C@{a}H]{r}O[C@H]([C@H]({p5})C{p6})[C@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O', 5: 'O', 6: 'O'}),
    'Mur': ('O[C@{a}H]{r}O[C@H](C{p6})[C@@H]({p4})[C@H]({p3}[C@H](C)C(=O)O)[C@H]{r}{p2}', '', {2: 'N', 3: 'O', 4: 'O', 6: 'O'}),
    'Neu': ('O[C@{a}]{r}(C(=O)O)C[C@H]({p4})[C@@H]({p5})[C@H]([C@H]({p7})[C@H]({p8})C{p9})O{r}', '', {4: 'O', 5: 'N', 7: 'O', 8: 'O', 9: 'O'}),
    'Oli': ('O[C@{a}H]{r}C[C@@H]({p3})[C@H]({p4})[C@@H](C)O{r}', '@', {3: 'O', 4: 'O'}),
    'Par': ('O[C@{a}H]{r}O[C@H](C)[C@@H]({p4})C[C@H]{r}{p2}', '', {2: 'O', 4: 'O'}),
    'Pau': ('O[C@{a}H]{r}C[C@@H]({p3})[C@@H]({p4})[C@@H](C)O{r}', '@', {3: 'O', 4: 'O'}),
    'Per': ('O[C@{a}H]{r}O[C@H](C)[C@@H]({p4})[C@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'N'}),
    'Pse': ('O[C@{a}]{r}(C(=O)O)C[C@H]({p4})[C@H]({p5})[C@H]([C@@H]({p7})[C@@H]({p8})C)O{r}', '', {4: 'O', 5: 'N', 7: 'N', 8: 'O'}),
    'Psi': ('O[C@{a}]{r}(C{p1})OC[C@@H]({p5})[C@@H]({p4})[C@H]{r}{p3}', '@', {1: 'O', 3: 'O', 4: 'O', 5: 'O'}),
    'Qui': ('O[C@{a}H]{r}O[C@H](C)[C@@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O'}),
    'Quif': ('O[C@{a}H]{r}O[C@H]([C@H]({p5})C)[C@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 5: 'O'}),
    'Rha': ('O[C@{a}H]{r}O[C@@H](C)[C@H]({p4})[C@@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 4: 'O'}),
    'Rhaf': ('O[C@{a}H]{r}O[C@@H]([C@@H]({p5})C)[C@@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 5: 'O'}),
    'Rib': ('O[C@{a}H]{r}OC[C@@H]({p4})[C@@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O'}),
    'Ribf': ('O[C@{a}H]{r}O[C@H](C{p5})[C@@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 5: 'O'}),
    'Rulf': ('O[C@{a}]{r}(C{p1})OC[C@@H]({p4})[C@H]{r}{p3}', '@', {1: 'O', 3: 'O', 4: 'O'}),
    'Sed': ('O[C@{a}]{r}(C{p1})O[C@H](C{p7})[C@@H]({p5})[C@@H]({p4})[C@@H]{r}{p3}', '@', {1: 'O', 3: 'O', 4: 'O', 5: 'O', 7: 'O'}),
    'Sedf': ('O[C@{a}]{r}(C{p1})O[C@H]([C@H]({p6})C{p7})[C@@H]({p4})[C@@H]{r}{p3}', '@', {1: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
    'Sor': ('O[C@{a}]{r}(C{p1})OC[C@H]({p5})[C@@H]({p4})[C@@H]{r}{p3}', '', {1: 'O', 3: 'O', 4: 'O', 5: 'O'}),
    'Tag': ('O[C@{a}]{r}(C{p1})OC[C@@H]({p5})[C@H]({p4})[C@@H]{r}{p3}', '@', {1: 'O', 3: 'O', 4: 'O', 5: 'O'}),
    'Tal': ('O[C@{a}H]{r}O[C@H](C{p6})[C@H]({p4})[C@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O'}),
    'Talf': ('O[C@{a}H]{r}O[C@@H]([C@H]({p5})C{p6})[C@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O', 5: 'O', 6: 'O'}),
    'Thref': ('O[C@{a}H]{r}OC[C@@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O'}),
    'Tyv': ('O[C@{a}H]{r}O[C@H](C)[C@@H]({p4})C[C@@H]{r}{p2}', '', {2: 'O', 4: 'O'}),
    'Vio': ('O[C@{a}H]{r}O[C@H](C)[C@@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'N'}),
    'Xluf': ('O[C@{a}]{r}(C{p1})OC[C@@H]({p4})[C@@H]{r}{p3}', '@', {1: 'O', 3: 'O', 4: 'O'}),
    'Xyl': ('O[C@{a}H]{r}OC[C@@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O'}),
    'Xylf': ('O[C@{a}H]{r}O[C@H](C{p5})[C@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 5: 'O'}),
    'Yer': ('O[C@{a}H]{r}O[C@H](C)[C@@]({p4})([C@H](C)O)C[C@H]{r}{p2}', '', {2: 'O', 4: 'O'}),
}

ALDITOLS = {  # open-chain form of a reduced sugar ('-ol'), numbered like the parent: (template, {position: heteroatom})
    '6dTal': ('OC[C@@H]({p2})[C@@H]({p3})[C@@H]({p4})[C@H]({p5})C', {2: 'O', 3: 'O', 4: 'O', 5: 'O'}),
    'All': ('OC[C@@H]({p2})[C@@H]({p3})[C@@H]({p4})[C@@H]({p5})C{p6}', {2: 'O', 3: 'O', 4: 'O', 5: 'O', 6: 'O'}),
    'Alt': ('OC[C@H]({p2})[C@@H]({p3})[C@@H]({p4})[C@@H]({p5})C{p6}', {2: 'O', 3: 'O', 4: 'O', 5: 'O', 6: 'O'}),
    'Api': ('OC[C@H]({p2})[C@]({p3})(C{p4})CO', {2: 'O', 3: 'O', 4: 'O'}),
    'Ara': ('OC[C@H]({p2})[C@@H]({p3})[C@@H]({p4})C{p5}', {2: 'O', 3: 'O', 4: 'O', 5: 'O'}),
    'Ery': ('OC[C@H]({p2})[C@H]({p3})C{p4}', {2: 'O', 3: 'O', 4: 'O'}),
    'Fru': ('OC(C{p1})[C@@H]({p3})[C@H]({p4})[C@H]({p5})C{p6}', {1: 'O', 3: 'O', 4: 'O', 5: 'O', 6: 'O'}),
    'Fuc': ('OC[C@@H]({p2})[C@H]({p3})[C@H]({p4})[C@@H]({p5})C', {2: 'O', 3: 'O', 4: 'O', 5: 'O'}),
    'Gal': ('OC[C@H]({p2})[C@@H]({p3})[C@@H]({p4})[C@H]({p5})C{p6}', {2: 'O', 3: 'O', 4: 'O', 5: 'O', 6: 'O'}),
    'Galf': ('OC[C@H]({p2})[C@@H]({p3})[C@@H]({p4})[C@H]({p5})C{p6}', {2: 'O', 3: 'O', 4: 'O', 5: 'O', 6: 'O'}),
    'Glc': ('OC[C@H]({p2})[C@@H]({p3})[C@H]({p4})[C@H]({p5})C{p6}', {2: 'O', 3: 'O', 4: 'O', 5: 'O', 6: 'O'}),
    'Glcf': ('OC[C@H]({p2})[C@@H]({p3})[C@H]({p4})[C@H]({p5})C{p6}', {2: 'O', 3: 'O', 4: 'O', 5: 'O', 6: 'O'}),
    'Gul': ('OC[C@H]({p2})[C@H]({p3})[C@@H]({p4})[C@H]({p5})C{p6}', {2: 'O', 3: 'O', 4: 'O', 5: 'O', 6: 'O'}),
    'Ido': ('OC[C@H]({p2})[C@@H]({p3})[C@H]({p4})[C@@H]({p5})C{p6}', {2: 'O', 3: 'O', 4: 'O', 5: 'O', 6: 'O'}),
    'Lyx': ('OC[C@@H]({p2})[C@@H]({p3})[C@H]({p4})C{p5}', {2: 'O', 3: 'O', 4: 'O', 5: 'O'}),
    'Man': ('OC[C@@H]({p2})[C@@H]({p3})[C@H]({p4})[C@H]({p5})C{p6}', {2: 'O', 3: 'O', 4: 'O', 5: 'O', 6: 'O'}),
    'Qui': ('OC[C@H]({p2})[C@@H]({p3})[C@H]({p4})[C@H]({p5})C', {2: 'O', 3: 'O', 4: 'O', 5: 'O'}),
    'Rha': ('OC[C@H]({p2})[C@H]({p3})[C@@H]({p4})[C@@H]({p5})C', {2: 'O', 3: 'O', 4: 'O', 5: 'O'}),
    'Rib': ('OC[C@H]({p2})[C@H]({p3})[C@H]({p4})C{p5}', {2: 'O', 3: 'O', 4: 'O', 5: 'O'}),
    'Tal': ('OC[C@@H]({p2})[C@@H]({p3})[C@@H]({p4})[C@H]({p5})C{p6}', {2: 'O', 3: 'O', 4: 'O', 5: 'O', 6: 'O'}),
    'Thre': ('OC[C@@H]({p2})[C@H]({p3})C{p4}', {2: 'O', 3: 'O', 4: 'O'}),
    'Xyl': ('OC[C@H]({p2})[C@@H]({p3})[C@H]({p4})C{p5}', {2: 'O', 3: 'O', 4: 'O', 5: 'O'}),
}

# default enantiomer of each skeleton, so that an explicit D-/L- prefix knows when to mirror
ENANTIOMER = {'4eLeg': 'D', '6dAlt': 'L', '6dAltf':
              'L', '6dGul': 'D', '6dTal': 'D', 'Abe': 'D', 'Abef': 'D', 'Acef': 'L', 'Aci': 'D', 'Aco': 'D', 'All': 'L', 'Allf': 'L', 'Alt': 'L',
              'Altf': 'L', 'Api': 'L', 'Apif': 'L', 'Ara': 'L', 'Araf': 'L', 'Asc': 'L', 'Bac': 'D', 'Col': 'L', 'DDAltHep': 'L', 'DDGlcHep': 'D',
              'DDManHep': 'D', 'DLGlcHep': 'L', 'Dha': 'D', 'Dig': 'D', 'Erwiniose': 'D', 'Eryf': 'D', 'Fru': 'D', 'Fruf': 'D', 'Fuc': 'L', 'Fucf':
              'L', 'Fus': 'L', 'Gal': 'D', 'GalHep': 'D', 'Galf': 'D', 'Glc': 'D', 'GlcHep': 'D', 'Glcf': 'D', 'Gul': 'D', 'Gulf': 'D', 'Ido': 'L',
              'IdoHep': 'L', 'Idof': 'L', 'Kdn': 'D', 'Kdo': 'D', 'Kdof': 'D', 'Ko': 'D', 'LDAltHep': 'L', 'LDGlcHep': 'D', 'LDManHep': 'D',
              'LLGlcHep': 'L', 'Leg': 'D', 'Lyx': 'D', 'Lyxf': 'D', 'Man': 'D', 'ManHep': 'D', 'Manf': 'D', 'Mur': 'D', 'Neu': 'D', 'Oli': 'D',
              'Par': 'D', 'Pau': 'L', 'Per': 'D', 'Pse': 'L', 'Psi': 'D', 'Qui': 'D', 'Quif': 'D', 'Rha': 'L', 'Rhaf': 'L', 'Rib': 'D', 'Ribf': 'D',
              'Rulf': 'D', 'Sed': 'D', 'Sedf': 'D', 'Sor': 'L', 'Tag': 'D', 'Tal': 'D', 'Talf': 'D', 'Thref': 'D', 'Tyv': 'D', 'Vio': 'D', 'Xluf':
              'D', 'Xyl': 'D', 'Xylf': 'D', 'Yer': 'L'}

SUBSTITUENTS = {  # modification: (form replacing a hydroxyl, form replacing an amine)
    'S': ('OS(=O)(=O)O', 'NS(=O)(=O)O'), 'P': ('OP(=O)(O)O', 'NP(=O)(O)O'), 'PP': ('OP(=O)(O)OP(=O)(O)O', None),
    'Me': ('OC', 'NC'), 'Ac': ('OC(C)=O', 'NC(C)=O'), 'Gc': ('OC(=O)CO', 'NC(=O)CO'), 'Fo': ('OC=O', 'NC=O'),
    'Et': ('OCC', 'NCC'), 'Am': ('OC(C)=N', 'NC(C)=N'), 'Gro': ('OCC(O)CO', None), 'SH': ('S', None),
    'PEtN': ('OP(=O)(O)OCCN', None), 'PCho': ('OP(=O)([O-])OCC[N+](C)(C)C', None), 'PGro': ('OP(=O)(O)OCC(O)CO', None),
    'PPEtN': ('OP(=O)(O)OP(=O)(O)OCCN', None), 'Aep': ('OP(=O)(O)CCN', None), 'Cm': ('OCC(=O)O', 'NCC(=O)O'),
    'Lac': ('O[C@@H](C)C(=O)O', 'N[C@@H](C)C(=O)O'), 'Pyr': ('OC(=O)C(C)=O', None), 'Cer': (CERAMIDE, None),
    'N': ('N', 'N'), 'NAc': ('NC(C)=O', 'NC(C)=O'), 'NGc': ('NC(=O)CO', 'NC(=O)CO'), 'NS': ('NS(=O)(=O)O', 'NS(=O)(=O)O'),
    'NMe': ('NC', 'NC'), 'NFo': ('NC=O', 'NC=O'), 'NAm': ('NC(C)=N', 'NC(C)=N'), 'NP': ('NP(=O)(O)O', 'NP(=O)(O)O'),
    'NBut': ('NC(=O)CCC', 'NC(=O)CCC'), 'NSuc': ('NC(=O)CCC(=O)O', 'NC(=O)CCC(=O)O'), 'NCm': ('NCC(=O)O', 'NCC(=O)O'),
    'NAla': ('NC(=O)[C@@H](C)N', 'NC(=O)[C@@H](C)N'), 'NGly': ('NC(=O)CN', 'NC(=O)CN'),
    'NSer': ('NC(=O)[C@@H](CO)N', 'NC(=O)[C@@H](CO)N'), 'NThr': ('NC(=O)[C@@H]([C@@H](C)O)N', 'NC(=O)[C@@H]([C@@H](C)O)N'),
}
ANOMERIC = {  # what a modification on the anomeric position replaces that whole anomeric oxygen with
    'S': 'OS(=O)(=O)O', 'P': 'OP(=O)(O)O', 'PP': 'OP(=O)(O)OP(=O)(O)O', 'Me': 'CO', 'Ac': 'CC(=O)O', 'Et': 'CCO',
    'Gc': 'OCC(=O)O', 'Fo': 'C(=O)O', 'Gro': 'OCC(O)CO', 'PEtN': 'NCCOP(=O)(O)O', 'PCho': 'C[N+](C)(C)CCOP(=O)([O-])O',
    'PGro': 'OCC(O)COP(=O)(O)O', 'PPEtN': 'NCCOP(=O)(O)OP(=O)(O)O', 'Cer': 'CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](NC(=O)CCCCCCCCCCCCCCC)CO',
    'N': 'N', 'NAc': 'CC(=O)N', 'NMe': 'CN', 'NGc': 'OCC(=O)N', 'NS': 'OS(=O)(=O)N', 'NFo': 'C(=O)N',
}
ESTERIFIED = {'Me': 'C', 'Et': 'CC'}  # what a modification on the carbon of a uronic acid does to that acid
URONIC = 'C(=O)O'  # what 'A' does to the primary alcohol: oxidize it to a carboxylic acid
AMIDATED = 'C(=O)N'  # what 'Am', 'NAm' or 'N' then does to that acid
WILDCARDS = {'Sia', 'dNon', 'ddNon', 'ddHex', 'Monosaccharide', 'Unknown', 'Assigned'}  # 'Hex', 'dHex' and 'Pen' are stereochemistry-free skeletons instead
_MOD_NAMES = set(SUBSTITUENTS) | {'A'} | {'O' + m for m in SUBSTITUENTS if m[0] not in 'ON'}  # an 'O' prefix means the position is unknown
_SKELETON_RE = re.compile('|'.join(sorted(set(SKELETONS) | set(ALDITOLS), key = len, reverse = True)))
_MOD_RE = re.compile(r'(\d*)(' + '|'.join(sorted(_MOD_NAMES, key = len, reverse = True)) + r')')
_ATOM_RE = re.compile(r'\[[^\]]*\]|Br|Cl|[BCNOPSFI]')
_SLOT_RE = re.compile(r'\{p(\d)\}')


class GlycanSMILESError(ValueError):
    "Raised for a residue, modification or linkage that has no defined structure"
    pass


def _free_slots(template: str, # Residue template, possibly already partly substituted
                hetero: dict # Remaining {position: heteroatom} map
                ) -> list: # Positions still open for a substituent or a linkage
    "A slot only counts as free if nothing already hangs off it, as the lactyl ether of muramic acid does"
    return sorted(p for p in hetero if re.search(r'\{p%d\}(\)|$)' % p, template))


def _anomeric_position(token: str # Monosaccharide token
                       ) -> int: # 1 for an aldose, 2 for a ketose or ulosonic acid, None if there is no anomeric centre
    "Which carbon carries the anomeric oxygen, i.e. the one a glycosidic bond may leave from"
    skeleton, _, _, reduced = _split_token(token)
    if reduced or skeleton == 'Ins':
        return None
    return 2 if '[C@{a}]' in SKELETONS[skeleton][0] else 1


def _split_token(token: str # Monosaccharide token such as 'GlcNAc6S' or 'LDManHepOPEtN'
                 ) -> tuple: # Skeleton, [(position or None, modification)], enantiomer prefix, reduced flag
    "Split a monosaccharide token into its skeleton and its modifications"
    enantiomer, reduced = '', token.endswith('-ol')
    if reduced:
        token = token[:-3]
    if token[:2] in ('D-', 'L-'):
        enantiomer, token = token[0], token[2:]
    if token in WILDCARDS:
        raise GlycanSMILESError(f"'{token}' is a wildcard without a defined structure")
    match = _SKELETON_RE.match(token)
    if not match:
        raise GlycanSMILESError(f"no skeleton for '{token}'")
    skeleton, rest, mods = match.group(), token[match.end():], []
    while rest:
        m = _MOD_RE.match(rest)
        if not m:
            raise GlycanSMILESError(f"cannot read modification '{rest}' of '{token}'")
        mods.append((int(m.group(1)) if m.group(1) else None, m.group(2)))
        rest = rest[m.end():]
    return skeleton, mods, enantiomer, reduced


def _residue(token: str, # Monosaccharide token
             anomer: str, # 'a', 'b', or anything else for an undefined anomeric centre
             ring: str, # Ring-closure digit for this residue
             bridge: str, # Second ring-closure digit, for a pyruvate ketal
             taken: set # Positions the linkages of this residue will need
             ) -> tuple: # Template with linkage slots still open, its {position: heteroatom} map, and modifications whose position is unknown
    "Build the SMILES template of one monosaccharide, with every positioned modification applied"
    skeleton, mods, enantiomer, reduced = _split_token(token)
    if reduced and skeleton not in ALDITOLS:
        raise GlycanSMILESError(f"no open-chain form for '{skeleton}-ol'")
    if not reduced and skeleton not in SKELETONS:
        raise GlycanSMILESError(f"'{skeleton}' only exists as an alditol; write it as '{skeleton}-ol'")
    (template, hetero), alpha = (ALDITOLS[skeleton], '') if reduced else (SKELETONS[skeleton][::2], SKELETONS[skeleton][1])
    hetero, pending, acid = dict(hetero), [], None
    if enantiomer and enantiomer != ENANTIOMER.get(skeleton, 'D'):
        anomeric = '[C@{a}H]' if '[C@{a}H]' in template else '[C@{a}]'
        template = template.replace(anomeric, '\x01').replace('@@', '\x00').replace('@', '@@').replace('\x00', '@').replace('\x01', anomeric)
        alpha = '@' if alpha == '' else ''
    pyruvates = sorted(p for p, mod in mods if mod == 'Pyr' and p)
    for position, mod in sorted(mods, key = lambda m: not (m[0] is None and m[1][0] == 'N')):  # an N-acyl claims its position before anything can sit on it
        if mod not in SUBSTITUENTS and mod != 'A' and mod.startswith('O'):
            mod, position = mod[1:], None
        if mod == 'A':
            acid = max((p for p in hetero if 'C{p%d}' % p in template), default = None)
            if acid is None:
                raise GlycanSMILESError(f"'{token}' has no primary alcohol to oxidize")
            template = template.replace('C{p%d}' % acid, URONIC)
            hetero.pop(acid)
            continue
        if acid is not None and (position == acid or (position is None and mod in ('Am', 'NAm'))):
            if mod in ('Am', 'NAm', 'N'):
                template = template.replace(URONIC, AMIDATED)
            elif mod in ESTERIFIED:
                template = template.replace(URONIC, URONIC + ESTERIFIED[mod])
            else:
                raise GlycanSMILESError(f"'{mod}' is not defined on the carboxyl of '{token}'")
            continue
        if mod == 'Pyr' and len(pyruvates) == 2 and position in pyruvates:
            template = template.replace('{p%d}' % position, 'OC%s(C)C(=O)O' % bridge if position == pyruvates[0] else 'O%s' % bridge)
            hetero.pop(position, None)
            continue
        group = SUBSTITUENTS.get(mod)
        if group is None:
            raise GlycanSMILESError(f"unsupported modification '{mod}' in '{token}'")
        if position is None and mod[0] != 'N':  # where a sulfate or phosphate goes is only decided once the linkages are known
            pending.append((mod, group))
            continue
        if position is None:  # an N-acyl belongs on the amine of the sugar, or on its lowest free position
            free = [p for p in _free_slots(template, hetero) if p != 1 and p not in taken]
            position = next((p for p in free if hetero[p] == 'N'), free[0] if free else None)
            if position is None:
                raise GlycanSMILESError(f"'{token}' has nowhere to put '{mod}'")
        if position == 1 and '{p1}' not in template:
            replacement = ANOMERIC.get(mod)
            if replacement is None:
                raise GlycanSMILESError(f"'{mod}' cannot sit on the anomeric oxygen of '{token}'")
            template = replacement + template[1:]
            continue
        if mod == 'N':  # a bare amine only states which heteroatom sits there; whatever comes next attaches to it
            if position not in _free_slots(template, hetero):
                raise GlycanSMILESError(f"'{token}' has no free position {position} for '{mod}'")
            hetero[position] = 'N'
            continue
        template, hetero = _place(token, template, hetero, position, mod, group)
    if anomer == 'a':
        template = template.replace('{a}', alpha)
    elif anomer == 'b':
        template = template.replace('{a}', '@' if alpha == '' else '')
    else:
        template = template.replace('[C@{a}H]', 'C').replace('[C@{a}]', 'C')
    return template.replace('{r}', ring), hetero, pending


def _place(token: str, # Monosaccharide token, for error messages
           template: str, # Residue template
           hetero: dict, # Remaining {position: heteroatom} map
           position: int, # Carbon to modify
           mod: str, # Modification name
           group: tuple # Its (hydroxyl form, amine form)
           ) -> tuple: # Template and heteroatom map after the substitution
    "Drop one substituent into its slot, in the form that suits the heteroatom already sitting there"
    if position not in _free_slots(template, hetero):
        raise GlycanSMILESError(f"'{token}' has no free position {position} for '{mod}'")
    form = group[1] if hetero[position] == 'N' else group[0]
    if form is None:
        raise GlycanSMILESError(f"'{mod}' is not defined for the amine at position {position} of '{token}'")
    hetero.pop(position)
    return template.replace('{p%d}' % position, form), hetero


def _ring_digit(n: int # Ring-closure index
                ) -> str: # SMILES ring-closure token
    "Ring-closure tokens above 9 need the '%' form"
    if n > 99:
        raise GlycanSMILESError('glycan is nested too deeply for SMILES ring-closure digits')
    return str(n) if n < 10 else '%%%d' % n


@glycan_graph_memoize(maxsize = 4096)
def graph_to_smiles(graph: nx.DiGraph, # Glycan graph, as produced by glycan_to_nxGraph
                    mapping: bool = False, # Also return, for every atom of the SMILES, the graph node it came from
                    strict: bool = False # Raise on an unknown linkage or modification position instead of taking the lowest free one
                    ) -> str | tuple: # Isomeric SMILES, or (SMILES, atom-to-node list) when mapping
    "Assemble the isomeric SMILES of a glycan graph by splicing monosaccharide templates into each other"
    if not nx.is_weakly_connected(graph):
        raise GlycanSMILESError('glycan graph is disconnected; a floating substituent has no defined attachment point')
    labels = {n: graph.nodes[n]['string_labels'] for n in graph.nodes}

    def splice(fragment, owners, slot, piece, piece_owners):
        start = fragment.index(slot)
        return fragment[:start] + piece + fragment[start + len(slot):], owners[:start] + piece_owners + owners[start + len(slot):]

    def build(node, anomer, depth):
        taken = {int(p) for p in (labels[link].split('-')[-1] for link in graph.successors(node)) if p.isdigit()}
        fragment, hetero, pending = _residue(labels[node], anomer, _ring_digit(depth + 1), _ring_digit(depth + 51), taken)
        owners, first = [node] * len(fragment), _anomeric_position(labels[node])
        children = sorted(((graph.successors(link), labels[link]) for link in graph.successors(node)), key = lambda x: not x[1].split('-')[-1].isdigit())
        for successors, link in children:
            child = next(iter(successors), None)
            if child is None:
                raise GlycanSMILESError(f"linkage '{link}' has no child residue")
            donor, anomeric = link.split('-')[0].lstrip('ab?'), _anomeric_position(labels[child])
            if donor.isdigit() and anomeric is not None and int(donor) != anomeric:
                raise GlycanSMILESError(f"linkage '{link}' leaves '{labels[child]}' from position {donor} rather than its anomeric carbon")
            if anomeric is None:
                raise GlycanSMILESError(f"'{labels[child]}' has no anomeric carbon to form linkage '{link}' with")
            target, free = link.split('-')[-1], _free_slots(fragment, hetero)
            if not target.isdigit():
                options = [int(p) for p in target.split('/') if p.isdigit()]
                if strict or not (free or options):
                    raise GlycanSMILESError(f"linkage '{link}' onto '{labels[node]}' has no known position")
                target = str(next((p for p in options if p in free), options[0] if options else free[0]))
            onto_anomeric = int(target) == first and '{p%s}' % target not in fragment
            if onto_anomeric and depth:
                raise GlycanSMILESError(f"'{labels[node]}' cannot use its anomeric oxygen for both its own linkage and '{link}'")
            piece, piece_owners = build(child, link[0], depth + 1)
            if onto_anomeric:  # onto the anomeric centre: the two residues share that one oxygen
                fragment = 'O(' + piece[1:] + ')' + fragment[1:]
                owners = owners[:1] * 2 + piece_owners[1:] + owners[:1] + owners[1:]
                continue
            if '{p%s}' % target not in fragment:
                raise GlycanSMILESError(f"'{labels[node]}' has no free position {target} for linkage '{link}'")
            fragment, owners = splice(fragment, owners, '{p%s}' % target, piece, piece_owners)
            hetero.pop(int(target), None)
        for mod, group in pending:
            free = [p for p in _free_slots(fragment, hetero) if p != first]
            if strict or not free:
                raise GlycanSMILESError(f"'{labels[node]}' has nowhere left to put '{mod}'")
            position = next((p for p in free if hetero[p] == 'N'), None) if mod[0] == 'N' else None
            position = position if position is not None else (free[0] if UNKNOWN_POSITION == 'lowest' else free[-1])
            if mod == 'N':
                hetero[position] = 'N'
                continue
            piece, hetero = _place(labels[node], '{p%d}' % position, hetero, position, mod, group)
            fragment, owners = splice(fragment, owners, '{p%d}' % position, piece, [node] * len(piece))
        for slot in [m.group() for m in _SLOT_RE.finditer(fragment)]:
            start = fragment.index(slot)
            fragment = fragment[:start] + hetero[int(slot[2])] + fragment[start + len(slot):]
            owners = owners[:start] + owners[start:start + 1] + owners[start + len(slot):]
        return fragment, owners

    smiles, owners = build(max(graph.nodes), '?', 0)
    if not mapping:
        return smiles
    return smiles, [owners[m.start()] for m in _ATOM_RE.finditer(smiles)]


def glycan_to_smiles(glycan: str | nx.DiGraph, # Glycan in IUPAC-condensed format or as a networkx graph
                     mapping: bool = False, # Also return, for every atom of the SMILES, the graph node it came from
                     strict: bool = False # Raise on an unknown linkage or modification position instead of taking the lowest free one
                     ) -> str | tuple: # Isomeric SMILES, or (SMILES, atom-to-node list) when mapping
    "Convert a glycan in IUPAC-condensed format into an isomeric SMILES string"
    return graph_to_smiles(glycan if isinstance(glycan, nx.Graph) else glycan_to_nxGraph(glycan), mapping = mapping, strict = strict)


_TOKEN_RE = re.compile(r'\[[^\]]*\]|Br|Cl|[BCNOPSFI]|%\d\d|\d|[()=#/\\.\-+]')
_BRACKET_RE = re.compile(r'^\[(\d*)([A-Z][a-z]?|\*)(@{0,2})(?:H(\d*))?([+-]\d*)?\]$')


def parse_smiles(smiles: str # SMILES string
                 ) -> tuple: # Atoms as (element, charge, chirality), bonds as (first, second, order), rings as tuples of bond indices, and each atom's neighbors in the order the string writes them
    "Read a SMILES into atoms, bonds, rings and written neighbor order, which is what the chirality tags refer to"
    atoms, bonds, rings, parent, branches, pending_rings, neighbors = [], [], [], [], [], {}, []
    previous, order = None, 1
    tokens = _TOKEN_RE.findall(smiles)
    if ''.join(tokens) != smiles:
        raise GlycanSMILESError('this is not a SMILES string this module can read')
    for token in tokens:
        if token == '(':
            branches.append(previous)
        elif token == ')':
            previous = branches.pop()
        elif token in ('=', '#'):
            order = 2 if token == '=' else 3
        elif token in ('/', '\\', '-', '.'):
            previous = None if token == '.' else previous
        elif token[0].isdigit() or token[0] == '%':
            label = token.lstrip('%')
            if label not in pending_rings:
                pending_rings[label] = (previous, len(neighbors[previous]))
                neighbors[previous].append(None)
            else:
                opened, slot = pending_rings.pop(label)
                neighbors[opened][slot], _ = previous, neighbors[previous].append(opened)
                up = {}  # ancestors of the closing atom, and the bond that leads to each
                walk, trail = previous, []
                while walk is not None:
                    up[walk] = tuple(trail)
                    trail.append(parent[walk][1])
                    walk = parent[walk][0]
                walk, path = opened, []
                while walk not in up:  # the two atoms meet at their lowest common ancestor, which a branch can sit above
                    path.append(parent[walk][1])
                    walk = parent[walk][0]
                bonds.append((opened, previous, order))
                rings.append(tuple(path + list(up[walk]) + [len(bonds) - 1]))
                order = 1
        else:
            match = _BRACKET_RE.match(token) if token[0] == '[' else None
            if token[0] == '[' and not match:
                raise GlycanSMILESError(f"cannot read atom '{token}'")
            element = match.group(2) if match else token
            charge = 0 if not match or not match.group(5) else int(match.group(5) + '1' if len(match.group(5)) == 1 else match.group(5))
            atoms.append((element, charge, match.group(3) if match else ''))
            index = len(atoms) - 1
            parent.append((previous, len(bonds) if previous is not None else None))
            neighbors.append([] if previous is None else [previous])
            if match and match.group(3) and match.group(4) != '0' and 'H' in token:
                neighbors[index].append('H')  # a hydrogen inside the brackets counts where it is written
            if previous is not None:
                neighbors[previous].append(index)
                bonds.append((previous, index, order))
            previous, order = index, 1
    if pending_rings:
        raise GlycanSMILESError(f'unclosed ring bond {sorted(pending_rings)}')
    return atoms, bonds, rings, neighbors


class Molecule(NamedTuple):
    "A glycan's molecular graph; a tuple rather than a dict so that torch_geometric stores it as one object"
    smiles: str  # the SMILES it was read from
    atoms: list  # (element, charge, chirality) per atom, in the order the SMILES writes them
    bonds: list  # (first atom, second atom, order) per bond
    rings: list  # tuples of bond indices, one per ring
    atom_monos: list  # the monosaccharide every atom came from
    bond_monos: list  # the monosaccharide every bond came from, taken from its first atom


def glycan_to_molecule(glycan: str | nx.DiGraph, # Glycan in IUPAC-condensed format or as a networkx graph
                       strict: bool = False # Raise on an unknown linkage or modification position instead of taking the lowest free one
                       ) -> Molecule: # Atoms, bonds, rings, and the graph node every atom and bond came from
    "Build a glycan's molecular graph without a cheminformatics toolkit, keeping every atom's monosaccharide of origin"
    smiles, owners = graph_to_smiles(glycan if isinstance(glycan, nx.Graph) else glycan_to_nxGraph(glycan), mapping = True, strict = strict)
    atoms, bonds, rings, neighbors = parse_smiles(smiles)
    if len(atoms) != len(owners):
        raise GlycanSMILESError('atom mapping does not line up with the parsed SMILES')
    return Molecule(smiles, atoms, bonds, rings, owners, [owners[first] for first, second, order in bonds])


def _parity(written: list, # Neighbors in the order the SMILES writes them
            target: list # The same neighbors in the canonical order
            ) -> int: # 1 if the reordering is even, -1 if it is odd
    "Sign of the permutation that takes one neighbor ordering to another, which is what flips a chirality tag"
    order, sign = [written.index(atom) for atom in target], 1
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if order[i] > order[j]:
                sign = -sign
    return sign


def _neighbor_key(atom: int | str, # Neighbor to sort, or 'H'
                   number: dict, # {atom: carbon number} for this residue
                   ring_oxygen: int, # The ring oxygen of this residue
                   elements: list # Element of every atom
                   ) -> tuple: # Sort key placing ring atoms first, then substituents, then hydrogen
    "Canonical order of a stereocentre's neighbors: around the ring first, then what hangs off it"
    if atom == 'H':
        return (3, 0)
    if atom == ring_oxygen:
        return (0, 0)
    if atom in number:
        return (1, number[atom])
    return (2, {'O': 0, 'N': 1}.get(elements[atom], 2))


def _chirality(centre: int, # Atom to describe
               atoms: list, # Atoms of the molecule
               neighbors: list, # Written neighbor order per atom
               number: dict, # {atom: carbon number} for this residue
               ring_oxygen: int # The ring oxygen of this residue
               ) -> str: # '@', '@@' or '' when the centre carries no tag
    "Rewrite an atom's chirality tag in terms of a canonical neighbor order, so it can be compared between molecules"
    tag = atoms[centre][2]
    if not tag:
        return ''
    elements = [element for element, charge, chirality in atoms]
    written = list(neighbors[centre])
    if len(written) != 4:
        raise GlycanSMILESError(f'stereocentre at atom {centre} has {len(written)} neighbors instead of 4')
    target = sorted(written, key = lambda atom: _neighbor_key(atom, number, ring_oxygen, elements))
    return tag if _parity(written, target) == 1 else ('@@' if tag == '@' else '@')


def _rings_of_atoms(bonds: list, # Bonds of the molecule
                    rings: list # Rings as tuples of bond indices
                    ) -> list: # Rings as lists of atom indices, in ring order
    "Turn each ring's bonds into the cycle of atoms it runs through"
    out = []
    for ring in rings:
        adjacency = {}
        for bond in ring:
            first, second, order = bonds[bond]
            adjacency.setdefault(first, []).append(second)
            adjacency.setdefault(second, []).append(first)
        start = min(adjacency)
        cycle, previous, current = [start], None, start
        while True:
            following = [atom for atom in adjacency[current] if atom != previous]
            if not following or following[0] == start:
                break
            previous, current = current, following[0]
            cycle.append(current)
        out.append(cycle)
    return out


def _number_residue(cycle: list, # Ring atoms in ring order
                    atoms: list, # Atoms of the molecule
                    adjacency: dict # {atom: set of bonded atoms}
                    ) -> tuple: # {atom: carbon number}, the ring oxygen, and the anomeric carbon
    "Number a sugar ring the way a chemist would, starting from the anomeric carbon"
    elements = [element for element, charge, chirality in atoms]
    oxygens = [atom for atom in cycle if elements[atom] == 'O']
    if len(oxygens) != 1 or any(elements[atom] != 'C' for atom in cycle if atom not in oxygens):
        raise GlycanSMILESError('not a sugar ring')
    ring_oxygen, ring = oxygens[0], set(cycle)
    candidates = [atom for atom in adjacency[ring_oxygen]
                  if any(elements[other] in 'ON' and other not in ring for other in adjacency[atom])]
    if len(candidates) != 1:
        raise GlycanSMILESError('anomeric carbon is ambiguous')
    anomeric = candidates[0]
    head = [other for other in adjacency[anomeric] if elements[other] == 'C' and other not in ring]
    number, index = {}, 1
    if len(head) == 1:  # a ketose or ulosonic acid carries C1 outside the ring
        number[head[0]], index = 1, 2
    elif len(head) > 1:
        raise GlycanSMILESError('branched anomeric carbon')
    number[anomeric], previous, current = index, ring_oxygen, anomeric
    while True:
        following = [other for other in adjacency[current] if other in ring and other not in (previous, ring_oxygen)]
        if not following:
            break
        index += 1
        previous, current = current, following[0]
        number[current] = index
    while True:  # walk out along the exocyclic chain, as in a heptose or a sialic acid
        following = [other for other in adjacency[current] if elements[other] == 'C' and other not in number and other not in ring]
        if len(following) != 1:
            break
        index += 1
        current = following[0]
        number[current] = index
    return number, ring_oxygen, anomeric


def _slot_kind(carbon: int, # Numbered carbon of a residue
               atoms: list, # Atoms of the molecule
               adjacency: dict, # {atom: set of bonded atoms}
               ring: set, # Ring atoms of this residue
               root_oxygen: int # The anomeric oxygen, which is a linkage rather than a slot
               ) -> tuple: # What sits at this position ('O', 'N', 'acid' or '-') and the atom carrying it
    "What a carbon carries besides the skeleton: a hydroxyl, an amine, a carboxyl, or nothing"
    elements = [element for element, charge, chirality in atoms]
    if any(elements[other] == 'O' and len(adjacency[other]) == 1 and other != root_oxygen
           for other in adjacency[carbon]) and sum(elements[other] == 'O' for other in adjacency[carbon]) > 1:
        return 'acid', None
    exocyclic = [other for other in adjacency[carbon]
                 if elements[other] in 'ON' and other not in ring and other != root_oxygen]
    if len(exocyclic) == 1:
        return elements[exocyclic[0]], exocyclic[0]
    return '-', None


def _key(ring_size: int, # Number of atoms in the ring
         positions: list, # (number, slot, chirality, branches) per numbered carbon
         anomeric_index: int # Which number the anomeric carbon has
         ) -> tuple: # Lookup key into the signature table
    "The part of a residue's description that identifies its skeleton, leaving the anomeric configuration aside"
    anomeric = positions[anomeric_index - 1]
    return (ring_size, tuple(entry for entry in positions if entry[0] != anomeric_index) + ((anomeric_index, anomeric[1], anomeric[3]),))


def _branch_size(start: int, # First atom of the branch
                 came_from: int, # Atom the branch hangs off
                 adjacency: dict # {atom: set of bonded atoms}
                 ) -> int: # Number of heavy atoms in the branch
    "How big a carbon branch is, so that two sugars differing only in what hangs off a ring carbon stay distinguishable"
    seen, stack = {came_from, start}, [start]
    while stack:
        for other in adjacency[stack.pop()]:
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return len(seen) - 1


def _signature(cycle: list, # Ring atoms in ring order
               atoms: list, # Atoms of the molecule
               bonds: list, # Bonds of the molecule
               neighbors: list, # Written neighbor order per atom
               adjacency: dict # {atom: set of bonded atoms}
               ) -> tuple: # Lookup key, anomeric descriptor, numbering, ring oxygen, anomeric oxygen, anomeric carbon, positions and ring size
    "Describe a sugar ring in a way that is independent of how the SMILES was written"
    number, ring_oxygen, anomeric = _number_residue(cycle, atoms, adjacency)
    elements = [element for element, charge, chirality in atoms]
    ring = set(cycle)
    root = next(other for other in adjacency[anomeric] if elements[other] in 'ON' and other not in ring)
    positions = []
    for carbon, index in sorted(number.items(), key = lambda item: item[1]):
        slot, holder = _slot_kind(carbon, atoms, adjacency, ring, root)
        branches = tuple(sorted(_branch_size(other, carbon, adjacency) for other in adjacency[carbon]
                                if elements[other] == 'C' and other not in number))
        positions.append((index, slot, _chirality(carbon, atoms, neighbors, number, ring_oxygen), branches))
    anomeric_index = number[anomeric]
    return _key(len(cycle), positions, anomeric_index), positions[anomeric_index - 1][2], number, ring_oxygen, root, anomeric, positions, len(cycle)


_SIGNATURES = {}


def _signature_table() -> dict: # {signature: {chirality: (skeleton, anomer)}}
    "Fingerprint every skeleton by writing it out and reading it back, so perception is the exact inverse of generation"
    if _SIGNATURES:
        return _SIGNATURES
    best = {}
    tokens = [(name, name) for name in SKELETONS]
    tokens += [(f"{'D' if ENANTIOMER[name] == 'L' else 'L'}-{name}", name) for name in SKELETONS if name in ENANTIOMER]
    for token, name in tokens:
        for anomer in ('a', 'b'):
            try:
                template, hetero, pending = _residue(token, anomer, '1', '51', set())
                free = len(_free_slots(template, hetero))
                for position, element in hetero.items():
                    template = template.replace('{p%d}' % position, element)
                atoms, bonds, rings, neighbors = parse_smiles(template)
                adjacency = _adjacency(atoms, bonds)
                key, chirality, number, ring_oxygen, root, anomeric, positions, ring_size = _signature(_rings_of_atoms(bonds, rings)[0], atoms, bonds, neighbors, adjacency)
                orders = {}
                for first, second, order in bonds:
                    orders[(first, second)] = orders[(second, first)] = order
                holders, carbons = _holders(number, set(_rings_of_atoms(bonds, rings)[0]), root, atoms, adjacency)
                fixed = {position: fragment for position, fragment in _fragments(holders, atoms, adjacency, orders, carbons).items()
                         if fragment not in ('O', 'N')}  # groups the skeleton itself carries, such as the lactyl ether of muramic acid
            except (GlycanSMILESError, IndexError, StopIteration):
                continue
            previous = best.get((key, chirality))
            if previous and previous[0] == token:  # a skeleton without stereocentres cannot tell its anomers apart
                best[(key, chirality)] = (token, '?', free, fixed)
            elif not previous or free > previous[2]:  # prefer the plain skeleton over one that bakes a substituent in
                best[(key, chirality)] = (token, anomer, free, fixed)
    for (key, chirality), (name, anomer, free, fixed) in best.items():
        _SIGNATURES.setdefault(key, {})[chirality] = (name, anomer, fixed)
    return _SIGNATURES


def _adjacency(atoms: list, # Atoms of the molecule
               bonds: list # Bonds of the molecule
               ) -> dict: # {atom: set of bonded atoms}
    "Neighbor sets, which is how everything downstream walks the molecule"
    adjacency = {index: set() for index in range(len(atoms))}
    for first, second, order in bonds:
        adjacency[first].add(second)
        adjacency[second].add(first)
    return adjacency


def _fragment_key(atom: int, # Atom to serialize
                  came_from: int | None, # Where the walk arrived from
                  atoms: list, # Atoms of the molecule
                  adjacency: dict, # {atom: set of bonded atoms}
                  orders: dict, # {(first, second): bond order}
                  seen: set, # Atoms already visited, shared so that a ketal ring cannot be walked twice
                  budget: list = None # Single-element list counting down the atoms left to spend
                  ) -> str: # Canonical string for this fragment
    "Serialize a substituent into a canonical string, so it can be looked up whatever order the SMILES wrote it in"
    budget = [60] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0:
        return '...'  # too big to be a substituent worth naming, and not worth walking further
    element, charge, chirality = atoms[atom]
    seen.add(atom)
    children = sorted(('=' if orders[(atom, other)] == 2 else '') + (_fragment_key(other, atom, atoms, adjacency, orders, seen, budget = budget) if other not in seen else 'R')
                      for other in adjacency[atom] if other != came_from)
    return element + ('%+d' % charge if charge else '') + ('(' + ')('.join(children) + ')' if children else '')


_MODIFICATIONS = {}


def _modification_table() -> dict: # {heteroatom: {canonical fragment: modification}}
    "Invert the substituent table by serializing every group the way perception will see it"
    if _MODIFICATIONS:
        return _MODIFICATIONS
    _MODIFICATIONS.update({'O': {}, 'N': {}})
    for name, forms in SUBSTITUENTS.items():
        short = name[1:] if name.startswith('N') and len(name) > 1 else name  # the leading N is put back only where the skeleton lacks one
        for element, form in zip('ON', forms):
            if form is None or form[0] != element:
                continue
            atoms, bonds, rings, neighbors = parse_smiles(form)
            adjacency, orders = _adjacency(atoms, bonds), {}
            for first, second, order in bonds:
                orders[(first, second)] = orders[(second, first)] = order
            _MODIFICATIONS[element].setdefault(_fragment_key(0, None, atoms, adjacency, orders, set()), short)
    return _MODIFICATIONS


def _fragments(holders: dict, # {position: holder atom}
               atoms: list, # Atoms of the molecule
               adjacency: dict, # {atom: set of bonded atoms}
               orders: dict, # {(first, second): bond order}
               carbons: dict # {position: the carbon the holder hangs off}
               ) -> dict: # {position: canonical fragment string}
    "Serialize whatever sits at each position, so a skeleton's own groups can be told from added ones"
    return {position: _fragment_key(holder, carbons[position], atoms, adjacency, orders, set()) for position, holder in holders.items()}


def _holders(number: dict, # {atom: carbon number}
             ring: set, # Ring atoms of this residue
             root: int, # The anomeric oxygen
             atoms: list, # Atoms of the molecule
             adjacency: dict # {atom: set of bonded atoms}
             ) -> tuple: # {position: holder atom} and {position: the carbon it hangs off}
    "Find the oxygen or nitrogen sitting at every numbered position of a residue"
    elements, holders, carbons = [element for element, charge, chirality in atoms], {}, {}
    for carbon, position in number.items():
        holder = next((other for other in adjacency[carbon] if elements[other] in 'ON' and other not in ring and other != root), None)
        if holder is not None:
            holders[position], carbons[position] = holder, carbon
    return holders, carbons


def _match_residue(positions: list, # (number, slot, chirality, branches) per numbered carbon
                   ring_size: int, # Number of atoms in the ring
                   anomeric_index: int, # Which number the anomeric carbon has
                   chirality: str, # Canonical descriptor of the anomeric carbon
                   fragments: dict # {position: canonical fragment string} of what sits at each position
                   ) -> tuple: # Skeleton, anomer, implied modifications, and the groups the skeleton brings itself
    "Look a perceived ring up in the skeleton table, allowing for an oxidized or aminated position"
    table = _signature_table()
    acid = max((entry[0] for entry in positions if entry[1] == 'acid'), default = None)
    variants = [(positions, [])]
    if acid is not None and acid == max(entry[0] for entry in positions):  # a uronic acid is a modified sugar, not a skeleton of its own
        variants.append(([(n, 'O' if n == acid else s, c, b) for n, s, c, b in positions], [(None, 'A')]))
    for base, implied in list(variants):
        if any(slot == 'N' for n, slot, c, b in base):  # an amine the skeleton does not have is a modification
            variants.append(([(n, 'O' if slot == 'N' else slot, c, b) for n, slot, c, b in base], implied))
    for base, implied in variants:
        entry = table.get(_key(ring_size, base, anomeric_index))
        if not entry:
            continue
        for descriptor, (skeleton, anomer, fixed) in sorted(entry.items(), key = lambda item: item[0] != chirality):
            if all(fragments.get(position) == fragment for position, fragment in fixed.items()):
                return skeleton, anomer if descriptor == chirality else '?', implied, fixed
    raise GlycanSMILESError('no monosaccharide matches this ring')


def _name_modification(residue: dict, # Perceived residue
                       position: int, # Carbon the group sits on
                       atoms: list, # Atoms of the molecule
                       adjacency: dict # {atom: set of bonded atoms}
                       ) -> tuple: # Position (None where the forward reading would put it back) and modification name
    "Name what sits at one position, dropping the position number wherever reading the token back would restore it"
    template, tag, hetero = SKELETONS[residue['skeleton'].split('-')[-1]]
    holder, elements = residue['holders'][position], [element for element, charge, chirality in atoms]
    carbon = next(atom for atom, spot in residue['number'].items() if spot == position)
    default = next((spot for spot in _free_slots(template, hetero) if spot != 1), None)
    if len([other for other in adjacency[holder] if other != carbon]) == 0:  # nothing hangs off it
        if elements[holder] != 'N' or hetero.get(position) == 'N':
            return position, None
        return (None if position == default else position), 'N'
    name = _modification_table()[elements[holder]].get(residue['fragments'][position])
    if name is None or elements[holder] != 'N':
        return position, name
    if position == default:  # written without a number, as in GlcNAc or MurNAc, so the N has to be spelt out
        return None, 'N' + name
    return position, name if hetero.get(position) == 'N' else 'N' + name  # a numbered position on a sugar that already has the nitrogen, as in Neu5Ac


def _residue_graph(residues: list, # Perceived residues
                   root: int, # Index of the reducing-end residue
                   strict: bool # Raise on a group that cannot be named
                   ) -> nx.DiGraph: # Graph in the same shape glycan_to_nxGraph produces
    "Build the glycan graph so that glycowork's own writer can put the string together"
    graph, counter = nx.DiGraph(), [0]

    def add(label):
        graph.add_node(counter[0], string_labels = label, labels = 0)
        counter[0] += 1
        return counter[0] - 1

    def build(index):
        residue = residues[index]
        if strict and any(name is None for position, name in residue['mods']):
            raise GlycanSMILESError(f"unrecognized substituent on '{residue['skeleton']}'")
        edges = []
        for child in residue['children']:
            other, built = residues[child], build(child)  # children first, so that indices grow towards the reducing end
            edges.append((add(f"{other['anomer']}{other['anomeric']}-{other['parent'][1]}"), built))
        token = residue['skeleton'] + ''.join(('' if position is None else str(position)) + name
                                              for position, name in sorted(residue['mods'], key = lambda mod: (mod[0] is not None, mod[0] or 0)) if name)
        here = add(token)
        for linkage, child in edges:
            graph.add_edge(here, linkage)
            graph.add_edge(linkage, child)
        return here

    build(root)
    return graph


def smiles_to_iupac(smiles: str, # SMILES string of a glycan
                    strict: bool = False # Raise on a substituent that cannot be named instead of dropping it
                    ) -> str: # Glycan in IUPAC-condensed format
    "Read a glycan's SMILES back into IUPAC-condensed format, the inverse of glycan_to_smiles"
    atoms, bonds, rings, neighbors = parse_smiles(smiles)
    adjacency, orders = _adjacency(atoms, bonds), {}
    for first, second, order in bonds:
        orders[(first, second)] = orders[(second, first)] = order
    residues, of_root = [], {}
    for cycle in _rings_of_atoms(bonds, rings):  # first find every sugar ring and how its carbons are numbered
        try:
            key, chirality, number, ring_oxygen, root, anomeric, positions, ring_size = _signature(cycle, atoms, bonds, neighbors, adjacency)
        except (GlycanSMILESError, StopIteration):
            continue
        holders, carbons = _holders(number, set(cycle), root, atoms, adjacency)
        of_root[root] = len(residues)
        residues.append({'chirality': chirality, 'number': number, 'ring': set(cycle), 'root': root, 'positions': positions,
                         'ring_size': ring_size, 'anomeric': number[anomeric], 'holders': holders, 'carbons': carbons,
                         'mods': [], 'children': []})
    if not residues:
        raise GlycanSMILESError('no monosaccharide ring found in this SMILES')
    for index, residue in enumerate(residues):  # then split the oxygens into glycosidic bonds and substituents
        residue['links'] = {position: of_root[holder] for position, holder in residue['holders'].items()
                            if holder in of_root and of_root[holder] != index}
        for position, child in residue['links'].items():
            residues[child]['parent'] = (index, position)
            residue['children'].append(child)
    for index, residue in enumerate(residues):  # only now is it safe to serialize what hangs off a position
        residue['fragments'] = _fragments({position: holder for position, holder in residue['holders'].items() if position not in residue['links']},
                                          atoms, adjacency, orders, residue['carbons'])
        residue['skeleton'], residue['anomer'], implied, residue['fixed'] = _match_residue(
            residue['positions'], residue['ring_size'], residue['anomeric'], residue['chirality'], residue['fragments'])
        residue['mods'] += implied
        for position in sorted(residue['fragments']):
            if position not in residue['fixed']:  # a group the skeleton's own name already covers, such as the lactyl of muramic acid
                residue['mods'].append(_name_modification(residue, position, atoms, adjacency))
        _link_through_root(residues, index, of_root, atoms, adjacency, orders)
    for index, residue in enumerate(residues):  # two residues sharing one anomeric oxygen are linked to each other, as in trehalose
        twin = next((other for other, host in enumerate(residues) if other != index and host['root'] == residue['root']), None)
        if twin is not None and 'parent' not in residue and 'parent' not in residues[twin] and residue['chirality'] != '':
            residue['parent'] = (twin, residues[twin]['anomeric'])
            residues[twin]['children'].append(index)
    roots = [index for index, residue in enumerate(residues) if 'parent' not in residue]
    if len(roots) != 1:
        raise GlycanSMILESError(f'found {len(roots)} reducing ends; this does not look like a single glycan')
    _check_coverage(residues, atoms, adjacency, orders)
    return graph_to_string(_residue_graph(residues, roots[0], strict))


def _check_coverage(residues: list, # Perceived residues
                    atoms: list, # Atoms of the molecule
                    adjacency: dict, # {atom: set of bonded atoms}
                    orders: dict # {(first, second): bond order}
                    ) -> None:
    "Refuse to return a glycan that quietly leaves a sugar-like piece of the molecule out of the name"
    covered = set()
    for residue in residues:
        covered |= set(residue['number']) | residue['ring'] | set(residue['holders'].values()) | {residue['root']}
        named = any(name for position, name in residue['mods'] if position == residue['anomeric'])
        for holder in list(residue['holders'].values()) + ([residue['root']] if named else []):
            _fragment_key(holder, None, atoms, adjacency, orders, covered)
    elements = [element for element, charge, chirality in atoms]
    left = [atom for atom in range(len(atoms)) if atom not in covered]
    sugary = [atom for atom in left if elements[atom] == 'C' and any(elements[other] == 'O' for other in adjacency[atom])]
    if len(sugary) >= 3:
        raise GlycanSMILESError('part of this molecule looks like a residue but could not be named; an open-chain sugar perhaps')


def _link_through_root(residues: list, # Perceived residues
                       index: int, # Residue to look at
                       of_root: dict, # {anomeric oxygen: residue}
                       atoms: list, # Atoms of the molecule
                       adjacency: dict, # {atom: set of bonded atoms}
                       orders: dict # {(first, second): bond order}
                       ) -> None:
    "Read what hangs off a residue's own anomeric oxygen: an aglycon, a phosphate, or a phosphodiester to its parent"
    residue = residues[index]
    carbon = next(atom for atom, position in residue['number'].items() if position == residue['anomeric'])
    beyond = [other for other in adjacency[residue['root']] if other != carbon]
    if not beyond:
        return
    elements = [element for element, charge, chirality in atoms]
    bridge = beyond[0]
    if elements[bridge] == 'P':  # a phosphodiester: the phosphate belongs to this residue and the bond to its parent
        for oxygen in adjacency[bridge]:
            for other in adjacency[oxygen]:
                if oxygen == residue['root'] or elements[oxygen] != 'O':
                    continue
                for parent, host in enumerate(residues):
                    if parent != index and other in host['number']:
                        residue['parent'] = (parent, host['number'][other])
                        host['children'].append(index)
                        residue['mods'].append((residue['anomeric'], 'P'))
                        return
    residue['mods'].append((residue['anomeric'], _modification_table()['O'].get(_fragment_key(residue['root'], carbon, atoms, adjacency, orders, set()))))


_IUPAC_LINKAGE = re.compile(r'\([ab?][0-9?]')


def looks_like_smiles(text: str # String to classify
                      ) -> bool: # Whether it reads as SMILES rather than as a glycan sequence
    "Tell a SMILES string apart from the glycan formats canonicalize_iupac already understands"
    if not text or _IUPAC_LINKAGE.search(text):
        return False
    if '@' in text:  # no other format glycowork reads uses it, so this is SMILES even if it turns out to be malformed
        return True
    if '-' in text.replace('[O-]', '').replace('[N+]', ''):
        return False
    if not any(character.isdigit() for character in text) or 'O' not in text:
        return False
    try:
        atoms, bonds, rings, neighbors = parse_smiles(text)
    except GlycanSMILESError:
        return False
    return len(atoms) > 4 and bool(rings)
