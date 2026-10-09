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
from functools import lru_cache
from itertools import combinations, product
from typing import NamedTuple
import networkx as nx
from glycowork.motif.graph import graph_to_string, glycan_graph_memoize, ensure_graph

CERAMIDE = 'OC[C@H](NC(=O)CCCCCCCCCCCCCCC)[C@H](O)/C=C/CCCCCCCCCCCCC'  # d18:1/16:0, the placeholder used whenever a sequence just says 'Cer'
UNKNOWN_POSITION = 'lowest'  # where a modification without a position number goes; 'lowest' free slot reproduces what GlyLES did

SKELETONS = {  # monosaccharide: (template, alpha tag, {position: heteroatom})
    '4eLeg': ('O[C@{a}]{r}(C(=O)O)C[C@@H]({p4})[C@@H]({p5})[C@H]([C@H]({p7})[C@H]({p8})C)O{r}', '', {4: 'O', 5: 'N', 7: 'N', 8: 'O'}),
    '6dAlt': ('O[C@{a}H]{r}O[C@@H](C)[C@H]({p4})[C@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 4: 'O'}),
    '6dAltf': ('O[C@{a}H]{r}O[C@@H]([C@@H]({p5})C)[C@H]({p3})[C@H]{r}{p2}', '@', {2: 'O', 3: 'O', 5: 'O'}),
    '6dGul': ('O[C@{a}H]{r}O[C@H](C)[C@H]({p4})[C@@H]({p3})[C@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O'}),
    '6dTal': ('O[C@{a}H]{r}O[C@H](C)[C@H]({p4})[C@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O'}),
    'Abe': ('O[C@{a}H]{r}O[C@H](C)[C@H]({p4})C[C@H]{r}{p2}', '', {2: 'O', 4: 'O'}),
    'Abef': ('O[C@{a}H]{r}O[C@@H]([C@H]({p5})C)C[C@H]{r}{p2}', '', {2: 'O', 5: 'O'}),
    'Acef': ('O[C@{a}H]{r}O[C@@H](C)[C@]({p3})(C(=O)O)[C@@H]{r}{p2}', '@', {2: 'O', 3: 'O'}),
    'Aci': ('O[C@{a}]{r}(C(=O)O)C[C@H]({p4})[C@@H]({p5})[C@H]([C@@H]({p7})[C@@H]({p8})C)O{r}', '@', {4: 'O', 5: 'N', 7: 'N', 8: 'O'}),
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
    'DDAltHep': ('O[C@{a}H]{r}O[C@H]([C@H]({p6})C{p7})[C@@H]({p4})[C@@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
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
    'Fus': ('O[C@{a}]{r}(C(=O)O)C[C@@H]({p4})[C@H]({p5})[C@H]([C@@H]({p7})[C@@H]({p8})C)O{r}', '@', {4: 'O', 5: 'N', 7: 'O', 8: 'O'}),
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
    'LDAltHep': ('O[C@{a}H]{r}O[C@H]([C@@H]({p6})C{p7})[C@@H]({p4})[C@@H]({p3})[C@@H]{r}{p2}', '', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'}),
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
    'Pse': ('O[C@{a}]{r}(C(=O)O)C[C@H]({p4})[C@H]({p5})[C@H]([C@@H]({p7})[C@@H]({p8})C)O{r}', '@', {4: 'O', 5: 'N', 7: 'N', 8: 'O'}),
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
    'Hex': ('OCC({p2})C({p3})C({p4})C({p5})C{p6}', {2: 'O', 3: 'O', 4: 'O', 5: 'O', 6: 'O'}),
    'dHex': ('OCC({p2})C({p3})C({p4})C({p5})C', {2: 'O', 3: 'O', 4: 'O', 5: 'O'}),
    'Pen': ('OCC({p2})C({p3})C({p4})C{p5}', {2: 'O', 3: 'O', 4: 'O', 5: 'O'}),
}

# default enantiomer of each skeleton, so that an explicit D-/L- prefix knows when to mirror
ENANTIOMER = {'4eLeg': 'D', '6dAlt': 'L', '6dAltf':
              'L', '6dGul': 'D', '6dTal': 'D', 'Abe': 'D', 'Abef': 'D', 'Acef': 'L', 'Aci': 'L', 'Aco': 'D', 'All': 'L', 'Allf': 'L', 'Alt': 'L',
              'Altf': 'L', 'Api': 'L', 'Apif': 'L', 'Ara': 'L', 'Araf': 'L', 'Asc': 'L', 'Bac': 'D', 'Col': 'L', 'DDAltHep': 'D', 'DDGlcHep': 'D',
              'DDManHep': 'D', 'DLGlcHep': 'L', 'Dha': 'D', 'Dig': 'D', 'Erwiniose': 'D', 'Eryf': 'D', 'Fru': 'D', 'Fruf': 'D', 'Fuc': 'L', 'Fucf':
              'L', 'Fus': 'L', 'Gal': 'D', 'GalHep': 'D', 'Galf': 'D', 'Glc': 'D', 'GlcHep': 'D', 'Glcf': 'D', 'Gul': 'D', 'Gulf': 'D', 'Ido': 'L',
              'IdoHep': 'L', 'Idof': 'L', 'Kdn': 'D', 'Kdo': 'D', 'Kdof': 'D', 'Ko': 'D', 'LDAltHep': 'D', 'LDGlcHep': 'D', 'LDManHep': 'D',
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
    'Lac': ('O[C@H](C)C(=O)O', 'N[C@H](C)C(=O)O'), 'Pyr': ('OC(=O)C(C)=O', None), 'Cer': (CERAMIDE, None),
    'N': ('N', 'N'), 'NAc': ('NC(C)=O', 'NC(C)=O'), 'NGc': ('NC(=O)CO', 'NC(=O)CO'), 'NS': ('NS(=O)(=O)O', 'NS(=O)(=O)O'),
    'NMe': ('NC', 'NC'), 'NFo': ('NC=O', 'NC=O'), 'NAm': ('NC(C)=N', 'NC(C)=N'), 'NP': ('NP(=O)(O)O', 'NP(=O)(O)O'),
    'NBut': ('NC(=O)CCC', 'NC(=O)CCC'), 'NSuc': ('NC(=O)CCC(=O)O', 'NC(=O)CCC(=O)O'), 'NCm': ('NCC(=O)O', 'NCC(=O)O'),
    'NAla': ('NC(=O)[C@H](C)N', 'NC(=O)[C@H](C)N'), 'NGly': ('NC(=O)CN', 'NC(=O)CN'),
    'NSer': ('NC(=O)[C@H](CO)N', 'NC(=O)[C@H](CO)N'), 'NThr': ('NC(=O)[C@H]([C@@H](C)O)N', 'NC(=O)[C@H]([C@@H](C)O)N'),
}
SUBSTITUENTS |= {  # acyl chains as in lipid A or Glc1Ole6Ac, and halogens that replace a hydroxyl outright as in Fuc2F
    'But': ('OC(=O)CCC', 'NC(=O)CCC'), 'Suc': ('OC(=O)CCC(=O)O', 'NC(=O)CCC(=O)O'), 'Dco': ('OC(=O)' + 'C' * 9, 'NC(=O)' + 'C' * 9),
    'Lau': ('OC(=O)' + 'C' * 11, 'NC(=O)' + 'C' * 11), 'Myr': ('OC(=O)' + 'C' * 13, 'NC(=O)' + 'C' * 13), 'Pam': ('OC(=O)' + 'C' * 15, 'NC(=O)' + 'C' * 15),
    'Mar': ('OC(=O)' + 'C' * 16, 'NC(=O)' + 'C' * 16), 'Ste': ('OC(=O)' + 'C' * 17, 'NC(=O)' + 'C' * 17),
    'Ole': ('OC(=O)CCCCCCC/C=C\\CCCCCCCC', 'NC(=O)CCCCCCC/C=C\\CCCCCCCC'), 'F': ('F', None), 'Cl': ('Cl', None), 'Br': ('Br', None), 'I': ('I', None),
}
SUBSTITUENTS |= {'N' + name: (SUBSTITUENTS[name][1],) * 2 for name in ('Dco', 'Lau', 'Myr', 'Pam', 'Mar', 'Ste', 'Ole')}  # also written without a position, as in GlcNPam
ANOMERIC = {  # what a modification on the anomeric position replaces that whole anomeric oxygen with
    'S': 'OS(=O)(=O)O', 'P': 'OP(=O)(O)O', 'PP': 'OP(=O)(O)OP(=O)(O)O', 'Me': 'CO', 'Ac': 'CC(=O)O', 'Et': 'CCO',
    'Gc': 'OCC(=O)O', 'Fo': 'C(=O)O', 'Gro': 'OCC(O)CO', 'PEtN': 'NCCOP(=O)(O)O', 'PCho': 'C[N+](C)(C)CCOP(=O)([O-])O',
    'PGro': 'OCC(O)COP(=O)(O)O', 'PPEtN': 'NCCOP(=O)(O)OP(=O)(O)O', 'Cer': 'CCCCCCCCCCCCC/C=C/[C@@H](O)[C@@H](NC(=O)CCCCCCCCCCCCCCC)CO',
    'N': 'N', 'NAc': 'CC(=O)N', 'NMe': 'CN', 'NGc': 'OCC(=O)N', 'NS': 'OS(=O)(=O)N', 'NFo': 'C(=O)N',
    'F': 'F', 'Cl': 'Cl', 'Br': 'Br', 'I': 'I', 'But': 'CCCC(=O)O', 'Lau': 'C' * 11 + 'C(=O)O', 'Myr': 'C' * 13 + 'C(=O)O',
    'Pam': 'C' * 15 + 'C(=O)O', 'Ste': 'C' * 17 + 'C(=O)O', 'Ole': 'CCCCCCCC/C=C\\CCCCCCCC(=O)O',
}
ESTERIFIED = {'Me': 'OC', 'Et': 'OCC'}  # what a modification on the carbon of a uronic acid does to that acid's hydroxyl
URONIC = 'C(=O){A}'  # what 'A' does to the primary alcohol: oxidize it to a carboxylic acid, whose hydroxyl stays a slot until the end
ULOSONIC = '{r}(C(=O)O)'  # C1 of an ulosonic acid, which is its carboxyl rather than its anomeric center
WILDCARDS = {'Sia', 'dNon', 'ddNon', 'ddHex', 'Monosaccharide', 'Unknown', 'Assigned', 'Sug'}  # 'Hex', 'dHex' and 'Pen' are stereochemistry-free skeletons instead
READ_ALIASES = {'Api', 'Per', 'Vio', 'Bac'}  # skeletons the reader leaves to their usual spelling in glycowork: D-Apif, D-Rha4N, Qui4N, QuiNAc4NAc
NUMBERED_N = {'P', 'Me', 'Suc', 'Dco', 'Lau', 'Myr', 'Pam', 'Mar', 'Ste', 'Ole'}  # groups glycowork writes on a numbered amine (GlcN2Me, the GlcN2Myr of lipid A), whereas N-acyls and heparin's N-sulfate sit on an unnumbered one (GlcNAc, GlcNS)
SUFFIXES = {'-ulosonic': ('-ulosonic', False), '-ulosaric': ('-ulosonic', True), '-uronic': ('', True), '-onic': ('-onic', False), '-aric': ('-onic', True),
            '-ol': ('-ol', False)}  # residue-name ending: (form of the skeleton, whether the last carbon is a carboxyl as well)
FISCHER = {'Gro': 'R', 'Ery': 'RR', 'Thr': 'LR', 'Rib': 'RRR', 'Ara': 'LRR', 'Xyl': 'RLR', 'Lyx': 'LLR', 'All': 'RRRR', 'Alt': 'LRRR', 'Glc': 'RLRR', 'Man': 'LLRR', 'Gul': 'RRLR',
           'Ido': 'LRLR', 'Gal': 'RLLR', 'Tal': 'LLLR'}  # configurational prefix: side of each hydroxyl in the Fischer projection of the D form, from the carbonyl end ('R' right)
CARBONS = {'Tet': 4, 'Pen': 5, 'Hex': 6, 'Hep': 7, 'Oct': 8, 'Non': 9}
BASES = {  # ring forms the systematic names are built from, (carbons, furanose, ulosonic acid): (D template with each stereocentre marked by its <position>, its slots, the Fischer side of each stereocentre, anomeric reference position)
    (4, True, False): ('O[C@{a}H]{r}OC[C@@H]<3>({p3})[C@H]<2>{r}{p2}', {2: 'O', 3: 'O'}, {2: 'R', 3: 'R'}, 3),
    (5, False, False): ('O[C@{a}H]{r}OC[C@@H]<4>({p4})[C@H]<3>({p3})[C@H]<2>{r}{p2}', {2: 'O', 3: 'O', 4: 'O'}, {2: 'R', 3: 'L', 4: 'R'}, 4),
    (5, True, False): ('O[C@{a}H]{r}O[C@H]<4>(C{p5})[C@H]<3>({p3})[C@H]<2>{r}{p2}', {2: 'O', 3: 'O', 5: 'O'}, {2: 'R', 3: 'L', 4: 'R'}, 4),
    (6, False, False): ('O[C@{a}H]{r}O[C@H]<5>(C{p6})[C@@H]<4>({p4})[C@H]<3>({p3})[C@H]<2>{r}{p2}', {2: 'O', 3: 'O', 4: 'O', 6: 'O'}, {2: 'R', 3: 'L', 4: 'R', 5: 'R'}, 5),
    (6, True, False): ('O[C@{a}H]{r}O[C@H]<4>([C@H]<5>({p5})C{p6})[C@H]<3>({p3})[C@H]<2>{r}{p2}', {2: 'O', 3: 'O', 5: 'O', 6: 'O'}, {2: 'R', 3: 'L', 4: 'R', 5: 'R'}, 5),
    (7, False, False): ('O[C@{a}H]{r}O[C@H]<5>([C@H]<6>({p6})C{p7})[C@@H]<4>({p4})[C@H]<3>({p3})[C@H]<2>{r}{p2}', {2: 'O', 3: 'O', 4: 'O', 6: 'O', 7: 'O'},
                        {2: 'R', 3: 'L', 4: 'R', 5: 'R', 6: 'R'}, 5),
    (7, True, False): ('O[C@{a}H]{r}O[C@H]<4>([C@H]<5>({p5})[C@H]<6>({p6})C{p7})[C@H]<3>({p3})[C@H]<2>{r}{p2}', {2: 'O', 3: 'O', 5: 'O', 6: 'O', 7: 'O'},
                       {2: 'R', 3: 'L', 4: 'R', 5: 'R', 6: 'R'}, 5),
    (6, False, True): ('O[C@{a}]{r}(C(=O)O)C[C@@H]<4>({p4})[C@@H]<5>({p5})CO{r}', {4: 'O', 5: 'O'}, {4: 'L', 5: 'L'}, 5),
    (7, False, True): ('O[C@{a}]{r}(C(=O)O)C[C@@H]<4>({p4})[C@@H]<5>({p5})[C@@H]<6>(C{p7})O{r}', {4: 'O', 5: 'O', 7: 'O'}, {4: 'L', 5: 'L', 6: 'R'}, 6),
    (8, False, True): ('O[C@{a}]{r}(C(=O)O)C[C@@H]<4>({p4})[C@@H]<5>({p5})[C@@H]<6>([C@H]<7>({p7})C{p8})O{r}', {4: 'O', 5: 'O', 7: 'O', 8: 'O'},
                       {4: 'L', 5: 'L', 6: 'R', 7: 'R'}, 7),
    (9, False, True): ('O[C@{a}]{r}(C(=O)O)C[C@H]<4>({p4})[C@@H]<5>({p5})[C@H]<6>([C@H]<7>({p7})[C@H]<8>({p8})C{p9})O{r}', {4: 'O', 5: 'O', 7: 'O', 8: 'O', 9: 'O'},
                       {4: 'R', 5: 'L', 6: 'L', 7: 'R', 8: 'R'}, 7),
}
_MOD_NAMES = set(SUBSTITUENTS) | {'A'} | {'O' + m for m in SUBSTITUENTS if m[0] not in 'ON'}  # an 'O' prefix means the position is unknown
_SKELETON_RE = re.compile(r'(?:([DL]{2})-?)?((?:\d+,)*\d+d)?(?:(\d+)e)?(?:((?:' + '|'.join(FISCHER) + r')*)(' + '|'.join(CARBONS) + r')|(' + '|'.join(sorted(
    set(SKELETONS) | set(ALDITOLS), key = len, reverse = True)) + r'))(f?)((?:\d+(?:Ulo|en))*)')  # glycero letters, deoxy and epimer positions, then configurations and chain length or a named skeleton, ring, keto and ene positions
_PREFIX_RE = re.compile(r'([DL])-|(\d+),(\d+)-Anhydro-')
_MOD_RE = re.compile(r'(\d*)(' + '|'.join(sorted(_MOD_NAMES, key = len, reverse = True)) + r')')
_ATOM_RE = re.compile(r'\[[^\]]*\]|Br|Cl|[BCNOPSFI]')
_SLOT_RE = re.compile(r'\{p(\d)\}')
_HOLDERS = ('O', 'N', 'S', 'F', 'Cl', 'Br', 'I')  # elements that can sit on a numbered carbon in place of its hydroxyl


class GlycanSMILESError(ValueError):
    "Raised for a residue, modification or linkage that has no defined structure"
    pass


def _free_slots(template: str, # Residue template, possibly already partly substituted
                hetero: dict # Remaining {position: heteroatom} map
                ) -> list: # Positions still open for a substituent or a linkage
    "A slot only counts as free if nothing already hangs off it, as the lactyl ether of muramic acid does"
    return sorted(p for p in hetero if '{p%d})' % p in template or template.endswith('{p%d}' % p))


@lru_cache(maxsize = 4096)
def _anomeric_position(token: str  # Monosaccharide token
                       ) -> int:  # 1 for an aldose, an alditol or an aldonic acid, 2 for a ketose or ulosonic acid, None if there is no anomeric center
    "Which carbon carries the oxygen a template is rooted at, i.e. the one a glycosidic or phosphodiester bond may leave from"
    skeleton, _, _, form, _ = _split_token(token)
    if form in ('-ol', '-onic'):  # the O1 of an alditol is written first just like an anomeric oxygen, as in Gal(b1-1)Rib-ol or Rib-ol(1-5)Rib5P-ol
        return 1
    if skeleton == 'Ins':
        return None
    return 2 if '[C@{a}]' in _skeleton(skeleton, form)[0] else 1


@lru_cache(maxsize = 4096)
def _split_token(token: str  # Monosaccharide token such as 'GlcNAc6S', 'LDManHepOPEtN' or '3,6-Anhydro-L-Gal2S'
                 ) -> tuple: # Skeleton, [(position or None, modification)], enantiomer prefix, form ('', '-ol', '-onic', '-ulosonic'), anhydro bridges
    "Split a monosaccharide token into its skeleton and its modifications"
    suffix = next((ending for ending in SUFFIXES if token.endswith(ending)), '')
    (form, acid), token, enantiomer, anhydro = SUFFIXES.get(suffix, ('', False)), token[:len(token) - len(suffix)], '', []
    while prefix := _PREFIX_RE.match(token):  # D-, L- and Anhydro- prefixes come in either order, as in 2,5-Anhydro-D-Alt-ol and D-2,7-Anhydro-3dManHep-ulosonic
        enantiomer, anhydro, token = prefix.group(1) or enantiomer, anhydro + ([(int(prefix.group(2)), int(prefix.group(3)))] if prefix.group(2) else []), token[prefix.end():]
    if token in WILDCARDS or (not _SKELETON_RE.match(token) and any(token.startswith(wildcard) for wildcard in WILDCARDS)):
        raise GlycanSMILESError(f"'{token}' is a wildcard without a defined structure")
    states = ''.join(re.findall(r'\d+(?:Ulo|en)', token.replace('ulo', 'Ulo')))  # a keto or ene position changes the skeleton itself, wherever the token puts it (D-6dXylHexNAc4Ulo, D-4dEryHexOAcN4en)
    token = re.sub(r'\d+(?:Ulo|ulo|en)', '', token)
    match = _SKELETON_RE.match(token)
    if not match or not match.group() or (match.group(5) and not match.group(4) and match.group(5) not in ('Pen', 'Hex', 'Hep')):  # a bare chain length is a skeleton only where a stereochemistry-free ring exists
        raise GlycanSMILESError(f"no skeleton for '{token}'")
    skeleton, rest, mods = match.group() + states, token[match.end():], []
    while rest:
        m = _MOD_RE.match(rest)
        if not m:
            raise GlycanSMILESError(f"cannot read modification '{rest}' of '{token}'")
        mods.append((int(m.group(1)) if m.group(1) else None, m.group(2)))
        rest = rest[m.end():]
    return skeleton, mods + ([(None, 'A')] if acid else []), enantiomer, form, tuple(anhydro)


@lru_cache(maxsize = 1024)
def _skeleton(name: str, # Skeleton as _split_token returns it, such as 'Gal', '6dAltHep', 'LDIdoHep' or '3dLyxHep'
              form: str # '' for the ring, '-ol' for the alditol, '-onic' for the aldonic acid, '-ulosonic' for an ulosonic acid
              ) -> tuple: # Template, alpha tag, {position: heteroatom}, and the enantiomer the template is
    "The template of a skeleton: from the tables, or built from them for deoxy, keto, ene and epimer positions and for systematic names"
    if form in ('-ol', '-onic') and name in ALDITOLS:
        template, hetero = ALDITOLS[name]
        return ('OC(=O)' + template[2:] if form == '-onic' else template), '', hetero, ENANTIOMER.get(name, 'D')
    if not form and name in SKELETONS:
        return *SKELETONS[name], ENANTIOMER.get(name, 'D')
    match = _SKELETON_RE.fullmatch(name)
    if not match:
        raise GlycanSMILESError(f"no skeleton for '{name}'")
    letters, deoxy, epimer, configs, length, named, furanose, states = match.groups()
    deoxy, keto, ene = {int(p) for p in deoxy[:-1].split(',')} if deoxy else set(), {int(p) for p in re.findall(r'(\d+)Ulo', states)}, {int(p) for p in re.findall(r'(\d+)en', states)}
    if named:  # a named skeleton with deoxygenated, oxidized or inverted positions, as in 6dAll, 9dNeu5Ac, 1dEry-ol or 8eLeg5Ac7Ac
        if not (deoxy or keto or epimer) or letters or ene:
            if not form and named + furanose in ALDITOLS:
                raise GlycanSMILESError(f"'{named}' only exists as an alditol; write it as '{named}-ol'")
            raise GlycanSMILESError(f"no {'open-chain form' if form in ('-ol', '-onic') else form[1:] + ' form' if form else 'skeleton'} for '{name}{form}'")
        template, alpha, hetero, default = _skeleton(named + furanose, form)
        hetero = dict(hetero)
        for position in sorted(deoxy | keto | ({int(epimer)} if epimer else set())):
            if form == '-ol' and position == 1 and position in deoxy:  # the alditol's O1 is the first atom of its template
                template = template[1:]
                continue
            slot = re.search(r'(\[C@@?H\]|C)(\{r\})?(\(\{p%d\}\)|\{p%d\})' % (position, position), template)
            if not slot or position not in hetero or (position not in deoxy | keto and '@' not in slot.group(1)):
                raise GlycanSMILESError(f"'{name}' has no position {position} to {'invert' if position not in deoxy | keto else 'deoxygenate' if position in deoxy else 'oxidize'}")
            if position not in deoxy | keto:
                template = template[:slot.start()] + slot.group(1).replace('@@', '\x00').replace('@', '@@').replace('\x00', '@') + template[slot.start() + len(slot.group(1)):]
                continue
            template = template[:slot.start()] + 'C' + (slot.group(2) or '') + (('(=O)' if slot.group(3)[0] == '(' else '=O') if position in keto else '') + template[slot.end():]
            hetero.pop(position)
        wanted = ENANTIOMER.get(name, 'D' if deoxy and named in ('All', 'Allf') else default)  # glycowork's bare All is L-allose, but its 6-deoxy sugar is D
        return (_mirror(template), '@' if alpha == '' else '') + (hetero, wanted) if wanted != default else (template, alpha, hetero, wanted)
    carbons, ulosonic, configs = CARBONS[length], form == '-ulosonic', re.findall('|'.join(FISCHER), configs)
    letters = list(letters or '')
    configs = ['Gro'] + configs if len(letters) == 2 and len(configs) == 1 else configs  # LDIdoHep: L-glycero-D-ido
    if letters and len(letters) != len(configs):
        raise GlycanSMILESError(f"'{name}' has {len(letters)} D/L letters for {len(configs)} configurational prefixes")
    centers = [p for p in range(4 if ulosonic else 2, carbons) if p not in deoxy | keto | ene | {p + 1 for p in ene}]
    total, unspecified = sum(len(FISCHER[c]) for c in configs), set()
    if configs and len(centers) == total + 1 and carbons == 7 and 6 in centers and 'Gro' not in configs and not ulosonic:
        unspecified = {6}  # GalHep: a heptose with a hexose prefix leaves C6 open
    elif configs and len(centers) == total + 1 and not ulosonic and 2 in centers:
        deoxy, centers = deoxy | {2}, centers[1:]  # D-6dAraHex: a three-centre prefix on a 6-deoxyhexose, i.e. 2,6-dideoxy-D-arabino-hexose
    if configs and len(centers) - len(unspecified) != total:
        raise GlycanSMILESError(f"'{name}' configures {total} stereocentres but has {len(centers) - len(unspecified)}")
    sides, open_centers = {}, [p for p in centers if p not in unspecified]
    for config, letter in zip(reversed(configs), reversed(letters or ['D'] * len(configs))):  # prefixes are written from the far end, so the last one covers the centres next to the carbonyl
        for side in FISCHER[config]:
            sides[open_centers.pop(0)] = side if letter == 'D' else {'R': 'L', 'L': 'R'}[side]
    reference = sorted(sides)[len(FISCHER[configs[-1]]) - 1] if configs else None  # the anomeric reference is the highest centre of the prefix next to the carbonyl
    if form in ('-ol', '-onic'):  # an open chain is written from C1 on, where '@' puts the hydroxyl on the right of the Fischer projection
        if ene or 1 in keto:
            raise GlycanSMILESError(f"no open-chain form for '{name}{form}'")
        template = ('' if 1 in deoxy else 'OC(=O)' if form == '-onic' else 'OC') + ''.join(
            'C' if p in deoxy else 'C(=O)' if p in keto else '[C@%sH]({p%d})' % ('' if sides[p] == 'R' else '@', p) if p in sides else 'C({p%d})' % p
            for p in range(2, carbons)) + ('C' if carbons in deoxy else 'C{p%d}' % carbons)
        return template, '', {p: 'O' for p in range(2, carbons + 1) if p not in deoxy | keto}, letters[-1] if letters else 'D'
    if (carbons, bool(furanose), ulosonic) not in BASES or 1 in deoxy | keto:
        raise GlycanSMILESError(f"no {'furanose' if furanose else 'pyranose'} form for '{name}{form}'")
    template, hetero, base, base_reference = BASES[(carbons, bool(furanose), ulosonic)]
    hetero = dict(hetero)
    for position, side in base.items():
        if position not in sides:  # no longer a stereocentre, or left open
            template = re.sub(r'\[C@@?H\]<%d>' % position, 'C<%d>' % position, template)
        elif side != sides[position]:
            template = re.sub(r'\[C(@@?)H\]<%d>' % position, lambda m: '[C%sH]<%d>' % ('@' if m.group(1) == '@@' else '@@', position), template)
    for position in sorted(deoxy | keto):
        if ulosonic and position == 3:  # every ulosonic acid here is a 3-deoxy one already
            continue
        slot = re.search(r'(<%d>)?(\{r\})?(\(\{p%d\}\)|\{p%d\})' % ((position,) * 3), template)
        if not slot or position not in hetero:
            raise GlycanSMILESError(f"'{name}' has no position {position} to {'deoxygenate' if position in deoxy else 'oxidize'}")
        template = template[:slot.start()] + (slot.group(1) or '') + (slot.group(2) or '') + (('(=O)' if slot.group(3)[0] == '(' else '=O') if position in keto else '') + template[slot.end():]
        hetero.pop(position)
    for position in ene:  # a double bond between this carbon and the next, as in the 4-deoxy-hex-4-enuronic acid heparinases leave
        template, count = re.subn(r'(C<%d>(?:\((?:[^()]|\([^()]*\))*\))?)(?=C<%d>)' % (position + 1, position), r'\1=', template)
        if not count:
            raise GlycanSMILESError(f"no double bond from position {position} in '{name}'")
    if not configs:  # Hep, Penf: a chain length and a ring size, but no stereochemistry
        return re.sub(r'<\d>', '', template).replace('[C@{a}H]', 'C').replace('[C@{a}]', 'C'), '', hetero, 'D'
    return re.sub(r'<\d>', '', template), '' if sides[reference] == base[base_reference] else '@', hetero, letters[-1] if letters else 'D'


def _mirror(template: str # Residue template
            ) -> str: # Template of the enantiomer
    "Invert every stereocentre but the anomeric one, whose tag the alpha entry sets"
    anomeric = '[C@{a}H]' if '[C@{a}H]' in template else '[C@{a}]'
    return template.replace(anomeric, '\x01').replace('@@', '\x00').replace('@', '@@').replace('\x00', '@').replace('\x01', anomeric)


def _residue(token: str, # Monosaccharide token
             anomer: str, # 'a', 'b', or anything else for an undefined anomeric center
             ring: str, # Ring-closure digit for this residue
             bridge: str, # Second ring-closure digit, for a pyruvate ketal or an anhydro bridge
             taken: set # Positions the linkages of this residue will need
             ) -> tuple: # Template with linkage slots still open, its {position: heteroatom} map, and modifications whose position is unknown
    "Build the SMILES template of one monosaccharide, with every positioned modification applied"
    skeleton, mods, enantiomer, form, anhydro = _split_token(token)
    template, alpha, hetero, default = _skeleton(skeleton, form)
    if anhydro and not form and any(p not in hetero and p != _anomeric_position(token) for bridged in anhydro for p in bridged):  # 2,5-Anhydro-Man: a bridge to the ring-closing carbon leaves the open-chain aldehyde
        template, alpha, hetero, default = _skeleton(skeleton, '-ol')
        if not any(_anomeric_position(token) in bridged for bridged in anhydro):  # 1,5-Anhydro-Glc bridges C1 itself and is the anhydroalditol 1,5-Anhydro-Glc-ol, not an aldehyde whose oxygen also closes the ring
            template = 'O=' + template[1:]
    hetero, pending, acid = dict(hetero), [], None
    if enantiomer and enantiomer != default:
        template, alpha = _mirror(template), '@' if alpha == '' else ''
    if len(anhydro) + (len([p for p, mod in mods if mod == 'Pyr' and p]) == 2) > 1:
        raise GlycanSMILESError(f"'{token}' needs more than one bridge across the residue")
    for bridged in anhydro:  # an anhydro bridge is one oxygen shared by two carbons: the first slot written gets it, the second a ring bond to it
        ends = sorted((p for p in bridged if '{p%d}' % p in template), key = lambda p: template.index('{p%d}' % p))
        if len(ends) + (template[0] == 'O' and min(bridged) == _anomeric_position(token)) != 2 or any(p in taken for p in ends):
            raise GlycanSMILESError(f"'{token}' has no free positions {bridged[0]} and {bridged[1]} to bridge")
        template = template.replace('{p%d}' % ends[0], 'O' + bridge) if len(ends) == 2 else 'O' + bridge + template[1:]
        template = template.replace('({p%d})' % ends[-1], bridge) if '({p%d})' % ends[-1] in template else template.replace('{p%d}' % ends[-1], bridge)
        for p in bridged:
            hetero.pop(p, None)
    pyruvates = sorted(p for p, mod in mods if mod == 'Pyr' and p)
    carboxyl = next((n for n, (p, mod) in enumerate(mods) if mod == 'A'), len(mods))
    mods = [(p, 'Am') if p is None and mod == 'N' and n > carboxyl else (p, mod) for n, (p, mod) in enumerate(
        mods)]  # the trailing 'N' of a uronamide, as in GalNAcAN, sits on the carboxyl and not on a ring carbon
    for position, mod in sorted(mods, key = lambda m: not (m[0] is None and m[1][0] == 'N')):  # an N-acyl claims its position before anything can sit on it
        prefixed = mod not in SUBSTITUENTS and mod != 'A' and mod.startswith('O')
        if prefixed:
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
                template = template.replace('{A}', 'N')
            elif mod in ESTERIFIED:
                template = template.replace('{A}', ESTERIFIED[mod])
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
            pending.append((mod, group, prefixed))
            continue
        if position is None:  # an N-acyl belongs on the amine of the sugar, or on its lowest position that no linkage or numbered acyl needs
            free = [p for p in _free_slots(template, hetero) if p != 1 and p not in taken and (mod == 'N' or all(q != p for q, m in mods))]
            position = next((p for p in free if hetero[p] == 'N'), free[0] if free else None)
            if position is None:
                raise GlycanSMILESError(f"'{token}' has nowhere to put '{mod}'")
        anomeric = 2 if '[C@{a}]' in template else 1
        if position == 1 and anomeric == 2 and ULOSONIC in template:
            if mod in ('Am', 'NAm', 'N'):
                template = template.replace(ULOSONIC, '{r}(C(=O)N)')
            elif mod in ESTERIFIED:
                template = template.replace(ULOSONIC, '{r}(C(=O)%s)' % ESTERIFIED[mod])
            else:
                raise GlycanSMILESError(f"'{mod}' is not defined on the carboxyl of '{token}'")
            continue
        if position == anomeric and '{p%d}' % position not in template:
            replacement = ANOMERIC.get(mod)
            if replacement is None or template.startswith('O='):  # the aldehyde of 2,5-Anhydro-Man has no oxygen at C1 to carry a group
                raise GlycanSMILESError(f"'{mod}' cannot sit on the anomeric oxygen of '{token}'")
            template = replacement + template[1:]
            continue
        if mod == 'N':  # a bare amine only states which heteroatom sits there; whatever comes next attaches to it
            if position not in _free_slots(template, hetero):
                raise GlycanSMILESError(f"'{token}' has no free position {position} for '{mod}'")
            hetero[position] = 'N'
            continue
        if mod in ('P',
                   'S') and position in taken:  # a phospho- or sulfodiester: the child leaves from the group's own oxygen, not from the ring carbon
            group = (group[0][:-1] + '{p%d}' % position,
                     None if group[1] is None else group[1][:-1] + '{p%d}' % position)
        template, hetero = _place(token, template, hetero, position, mod, group)
    if anomer == 'a':
        template = template.replace('{a}', alpha)
    elif anomer == 'b':
        template = template.replace('{a}', '@' if alpha == '' else '')
    else:
        template = template.replace('[C@{a}H]', 'C').replace('[C@{a}]', 'C')
    return template.replace('{r}', ring).replace('{A}', 'O'), hetero, pending


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
def graph_to_smiles(graph: nx.DiGraph,  # Glycan graph, as produced by glycan_to_nxGraph
                    mapping: bool = False,  # Also return, for every atom of the SMILES, the graph node it came from
                    strict: bool = False  # Raise on an unknown linkage or modification position instead of taking the lowest free one
                    ) -> str | tuple:  # Isomeric SMILES, or (SMILES, atom-to-node tuple) when mapping
    "Assemble the isomeric SMILES of a glycan graph by splicing monosaccharide templates into each other"
    if not len(graph):  # an empty string or None became an empty graph, on which networkx raised 'Connectivity is undefined for the null graph'
        raise GlycanSMILESError('glycan graph is empty')
    if not nx.is_weakly_connected(graph):
        raise GlycanSMILESError('glycan graph is disconnected; a floating substituent has no defined attachment point')
    labels = {n: graph.nodes[n]['string_labels'] for n in graph.nodes}

    def splice(fragment, owners, slot, piece, piece_owners):
        start = fragment.index(slot)
        return fragment[:start] + piece + fragment[start + len(slot):], owners[:start] + piece_owners + owners[start + len(slot):]

    def build(node, anomer, depth):
        taken = {int(p) for p in (labels[link].split('-')[-1] for link in graph.successors(node)) if p.isdigit()}
        fragment, hetero, pending = _residue(labels[node], anomer, _ring_digit(depth + 1), _ring_digit(depth + 51) if 'Pyr' in labels[node] or 'Anhydro' in labels[node] else '', taken)
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
                target = str(next((p for p in options if p in free), options[0] if options else next((p for p in free if hetero[p] == 'O'), free[0])))  # a glycosidic bond at an unknown position takes a hydroxyl, never the amine of an amino sugar
            onto_anomeric = int(target) == first and '{p%s}' % target not in fragment
            if onto_anomeric and depth:
                raise GlycanSMILESError(f"'{labels[node]}' cannot use its anomeric oxygen for both its own linkage and '{link}'")
            piece, piece_owners = build(child, link[0], depth + 1)
            if piece[0] not in 'ON' or piece[1] in '=%0123456789':  # an ester, halide, aglycon, or anhydro bridge on the anomeric oxygen leaves nothing to bond to the parent with
                raise GlycanSMILESError(f"'{labels[child]}' already carries a group on its anomeric oxygen and cannot also form linkage '{link}'")
            if onto_anomeric:  # onto the anomeric center: the two residues share that one oxygen
                fragment = 'O(' + piece[1:] + ')' + fragment[1:]
                owners = owners[:1] * 2 + piece_owners[1:] + owners[:1] + owners[1:]
                continue
            if '{p%s}' % target not in fragment:
                raise GlycanSMILESError(f"'{labels[node]}' has no free position {target} for linkage '{link}'")
            fragment, owners = splice(fragment, owners, '{p%s}' % target, piece, piece_owners)
            hetero.pop(int(target), None)
        for mod, group, prefixed in pending:
            free = [p for p in _free_slots(fragment, hetero) if p != first]
            if strict or not free:
                raise GlycanSMILESError(f"'{labels[node]}' has nowhere left to put '{mod}'")
            position = next((p for p in free if hetero[p] == ('N' if mod[0] == 'N' else 'O')), None) if mod[0] == 'N' or prefixed else None  # GlcNOMyr: an O-prefixed group never lands on the amine
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
    return smiles, tuple(owners[m.start()] for m in _ATOM_RE.finditer(
        smiles))  # a tuple, since the result is memoized and a caller editing a list would change it for every later call on that glycan


def glycan_to_smiles(glycan: str | nx.DiGraph,  # Glycan in IUPAC-condensed format or as a networkx graph
                     mapping: bool = False,  # Also return, for every atom of the SMILES, the graph node it came from
                     strict: bool = False  # Raise on an unknown linkage or modification position instead of taking the lowest free one
                     ) -> str | tuple:  # Isomeric SMILES, or (SMILES, atom-to-node tuple) when mapping
    "Convert a glycan in IUPAC-condensed format into an isomeric SMILES string"
    return graph_to_smiles(ensure_graph(glycan), mapping = mapping, strict = strict)


_TOKEN_RE = re.compile(r'\[[^\]]*\]|Br|Cl|[BCNOPSFI]|%\d\d|\d|[()=#/\\.\-+]')
_BRACKET_RE = re.compile(r'^\[(\d*)([A-Z][a-z]?|\*)(@{0,2})(?:H(\d*))?([+-]\d*)?\]$')


def parse_smiles(smiles: str # SMILES string
                 ) -> tuple: # Atoms as (element, charge, chirality), bonds as (first, second, order), rings as tuples of bond indices, and each atom's neighbors in the order the string writes them
    "Read a SMILES into atoms, bonds, rings, and written neighbor order, which is what the chirality tags refer to"
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
    shrunk = True
    while shrunk:  # a ring closed around a fused neighbor, such as a pyruvate ketal, runs round both; trade it for the smaller cycle
        shrunk = False
        for i, j in ((i, j) for i in range(len(rings)) for j in range(len(rings)) if len(rings[i]) < len(rings[j])):
            merged = set(rings[i]) ^ set(rings[j])
            if len(merged) < len(rings[j]) and set(rings[i]) & set(rings[j]):
                rings[j], shrunk = tuple(merged), True
                break
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
    smiles, owners = graph_to_smiles(ensure_graph(glycan), mapping = True, strict = strict)
    atoms, bonds, rings, neighbors = parse_smiles(smiles)
    if len(atoms) != len(owners):
        raise GlycanSMILESError('atom mapping does not line up with the parsed SMILES')
    return Molecule(smiles, atoms, bonds, rings, list(owners), [owners[first] for first, second, order in bonds])


def _parity(written: list, # Neighbors in the order the SMILES writes them
            target: list # The same neighbors in the canonical order
            ) -> int: # 1 if the reordering is even, -1 if it is odd
    "Sign of the permutation that takes one neighbor ordering to another, which is what flips a chirality tag"
    order, sign = [written.index(atom) for atom in target], 1
    for i, oi in enumerate(order):
        for j in range(i + 1, len(order)):
            if oi > order[j]:
                sign = -sign
    return sign


def _neighbor_key(atom: int | str, # Neighbor to sort, or 'H'
                   number: dict, # {atom: carbon number} for this residue
                   ring_oxygen: int, # The ring oxygen of this residue
                   elements: list # Element of every atom
                   ) -> tuple: # Sort key placing ring atoms first, then substituents, then hydrogen
    "Canonical order of a stereocenter's neighbors: around the ring first, then what hangs off it"
    if atom == 'H':
        return (3, 0)
    if atom == ring_oxygen:
        return (0, 0)
    if atom in number:
        return (1, number[atom])
    return (2, {'O': 0, 'N': 1}.get(elements[atom], 2))


def _chirality(center: int, # Atom to describe
               atoms: list, # Atoms of the molecule
               elements: list,  # Element of every atom
               neighbors: list,  # Written neighbor order per atom
               number: dict,  # {atom: carbon number} for this residue
               ring_oxygen: int  # The ring oxygen of this residue
               ) -> str:  # '@', '@@' or '' when the center carries no tag
    "Rewrite an atom's chirality tag in terms of a canonical neighbor order, so it can be compared between molecules"
    tag = atoms[center][2]
    if not tag:
        return ''
    written = list(neighbors[center])
    if len(written) != 4:
        raise GlycanSMILESError(f'stereocenter at atom {center} has {len(written)} neighbors instead of 4')
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
                    elements: list,  # Element of every atom
                    adjacency: dict  # {atom: set of bonded atoms}
                    ) -> tuple:  # {atom: carbon number}, the ring oxygen, and the anomeric carbon
    "Number a sugar ring the way a chemist would, starting from the anomeric carbon"
    oxygens = [atom for atom in cycle if elements[atom] == 'O']
    if len(oxygens) != 1 or any(elements[atom] != 'C' for atom in cycle if atom not in oxygens):
        raise GlycanSMILESError('not a sugar ring')
    ring_oxygen, ring = oxygens[0], set(cycle)
    candidates = [atom for atom in adjacency[ring_oxygen]
                  if any(elements[other] in _HOLDERS and other not in ring for other in adjacency[atom])]  # a glycosyl fluoride (Glc1F) has its anomeric carbon too
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
               elements: list,  # Element of every atom
               adjacency: dict,  # {atom: set of bonded atoms}
               ring: set,  # Ring atoms of this residue
               root_oxygen: int  # The anomeric oxygen, which is a linkage rather than a slot
               ) -> tuple:  # What sits at this position ('O', 'N', 'acid' or '-') and the atom carrying it
    "What a carbon carries besides the skeleton: a hydroxyl, an amine, a carboxyl, or nothing"
    if any(elements[other] == 'O' and len(adjacency[other]) == 1 and other != root_oxygen
           for other in adjacency[carbon]) and sum(elements[other] in 'ON' for other in adjacency[carbon]) > 1:
        return 'acid', None
    exocyclic = [other for other in adjacency[carbon]
                 if elements[other] in _HOLDERS and other not in ring and (other != root_oxygen or carbon not in ring)]  # the anomeric oxygen is a slot of an exocyclic carbon it bridges to, as in 2,7-Anhydro-Kdo
    if len(exocyclic) == 1:  # a thiol or a halogen takes the place of a hydroxyl, so the skeleton still has an 'O' there
        return 'N' if elements[exocyclic[0]] == 'N' else 'O', exocyclic[0]
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
               elements: list,  # Element of every atom
               bonds: list,  # Bonds of the molecule
               neighbors: list,  # Written neighbor order per atom
               adjacency: dict  # {atom: set of bonded atoms}
               ) -> tuple:  # Lookup key, anomeric descriptor, numbering, ring oxygen, anomeric oxygen, anomeric carbon, positions and ring size
    "Describe a sugar ring in a way that is independent of how the SMILES was written"
    number, ring_oxygen, anomeric = _number_residue(cycle, elements, adjacency)
    ring = set(cycle)
    root = next(other for other in adjacency[anomeric] if elements[other] in _HOLDERS and other not in ring)
    if any(order == 2 and {first, second} == {anomeric, root} for first, second, order in bonds):  # a lactone such as 1,5-Anhydro-Glc-onic, which the open chains take
        raise GlycanSMILESError('a lactone, not a sugar ring')
    positions = []
    for carbon, index in sorted(number.items(), key = lambda item: item[1]):
        slot, holder = _slot_kind(carbon, elements, adjacency, ring, root)
        branches = tuple(sorted(_branch_size(other, carbon, adjacency) for other in adjacency[carbon]
                                if elements[other] == 'C' and other not in number))
        positions.append((index, slot, _chirality(carbon, atoms, elements, neighbors, number, ring_oxygen), branches))
    anomeric_index = number[anomeric]
    return _key(len(cycle), positions, anomeric_index), positions[anomeric_index - 1][2], number, ring_oxygen, root, anomeric, positions, len(cycle)


_SIGNATURES = {}


def _signature_table() -> dict: # {signature: {chirality: (skeleton, anomer)}}
    "Fingerprint every skeleton by writing it out and reading it back, so perception is the exact inverse of generation"
    if _SIGNATURES:
        return _SIGNATURES
    from glycowork.glycan_data.loader import lib  # loaded lazily, as loader sits above this module
    best, names, derived = {}, [name for name in SKELETONS if name not in READ_ALIASES], set()
    for token in lib:  # systematic, deoxy and epimer skeletons glycowork knows, such as 6dAltHep or D-6dAraHex
        try:
            skeleton, mods, enantiomer, form, anhydro = _split_token(token)
            if form in ('', '-ulosonic') and not anhydro and (form or skeleton not in SKELETONS) and _skeleton(skeleton, form):
                derived.add(skeleton + form)
        except GlycanSMILESError:
            continue
    derived = sorted(derived)
    tokens = [(name, name) for name in names] + [(f"{'D' if ENANTIOMER[name] == 'L' else 'L'}-{name}", name) for name in names if name in ENANTIOMER]
    tokens += [(name, name) for name in derived] + [(f"{'L' if _skeleton(*_split_token(name)[::3])[3] == 'D' else 'D'}-{name}", name) for name in derived]
    for token, name in tokens:
        for anomer in ('a', 'b'):
            try:
                template, hetero, pending = _residue(token, anomer, '1', '51', set())
                free = len(_free_slots(template, hetero))
                for position, element in hetero.items():
                    template = template.replace('{p%d}' % position, element)
                atoms, bonds, rings, neighbors = parse_smiles(template)
                adjacency, elements = _adjacency(atoms, bonds), [element for element, charge, chirality in atoms]
                key, chirality, number, ring_oxygen, root, anomeric, positions, ring_size = _signature(_rings_of_atoms(bonds, rings)[0], atoms, elements, bonds, neighbors, adjacency)
                orders = {}
                for first, second, order in bonds:
                    orders[(first, second)] = orders[(second, first)] = order
                holders, carbons = _holders(number, set(_rings_of_atoms(bonds, rings)[0]), root, elements, adjacency)
                fixed = {position: fragment for position, fragment in _fragments(holders, atoms, adjacency, orders, carbons).items()
                         if fragment not in ('O', 'N')}  # groups the skeleton itself carries, such as the lactyl ether of muramic acid
            except (GlycanSMILESError, IndexError, StopIteration):
                continue
            previous = best.get((key, chirality))
            if previous and previous[0] == token:  # a skeleton without stereocenters cannot tell its anomers apart
                best[(key, chirality)] = (token, '?', free, fixed)
            elif not previous or free > previous[2]:  # prefer the plain skeleton over one that bakes a substituent in
                best[(key, chirality)] = (token, anomer, free, fixed)
    for (key, chirality), (name, anomer, free, fixed) in best.items():
        _SIGNATURES.setdefault(key, {})[chirality] = (name, anomer, fixed)
    return _SIGNATURES


_CHAINS = {}


def _chain_table() -> dict: # {(carbons, chirality of each carbon): [(alditol, positions carrying a heteroatom)]}
    "Fingerprint every alditol by writing it out and reading the chain back, in both enantiomers, as an open chain is perceived"
    if _CHAINS:
        return _CHAINS
    for name in ALDITOLS:
        if name in ('Fru', 'Api', 'Glcf', 'Galf'):  # a ketose and a branched chain number differently, and the furanose names repeat their pyranoses
            continue
        for token in (name, f"{'L' if ENANTIOMER.get(name, 'D') == 'D' else 'D'}-{name}"):
            template, hetero, pending = _residue(token + '-ol', '?', '1', '', set())
            for position, element in hetero.items():
                template = template.replace('{p%d}' % position, element)
            atoms, bonds, rings, neighbors = parse_smiles(template)
            adjacency, order = _adjacency(atoms, bonds), [1]  # C1 bonds to O1, the first atom of the template
            while following := [other for other in adjacency[order[-1]] if atoms[other][0] == 'C' and other not in order]:
                order.append(following[0])
            number = {atom: index + 1 for index, atom in enumerate(order)}
            _CHAINS.setdefault((len(order), tuple(_chirality(atom, atoms, [a[0] for a in atoms], neighbors, number, None) for atom in order)), []).append((token, set(hetero) | {1}))
    atoms, bonds, rings, neighbors = parse_smiles(re.sub(r'\{p\d\}', 'O', _residue('Ins', '?', '1', '', set())[0]))  # 1D-myo-inositol, its carbons written C1 to C6
    carbons = [atom for atom, (element, charge, chirality) in enumerate(atoms) if element == 'C']
    _CHAINS['Ins'] = tuple(_chirality(atom, atoms, [a[0] for a in atoms], neighbors, {atom: index + 1 for index, atom in enumerate(carbons)}, None) for atom in carbons)
    return _CHAINS


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
    _MODIFICATIONS.update({element: {} for element in _HOLDERS})
    for name, forms in SUBSTITUENTS.items():
        short = name[1:] if name.startswith('N') and len(name) > 1 else name  # the leading N is put back only where the skeleton lacks one
        for form in forms:
            if form is None:
                continue
            atoms, bonds, rings, neighbors = parse_smiles(form)
            adjacency, orders = _adjacency(atoms, bonds), {}
            for first, second, order in bonds:
                orders[(first, second)] = orders[(second, first)] = order
            _MODIFICATIONS[atoms[0][0]].setdefault(_fragment_key(0, None, atoms, adjacency, orders, set()), short)
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
             elements: list,  # Element of every atom
             adjacency: dict  # {atom: set of bonded atoms}
             ) -> tuple:  # {position: holder atom} and {position: the carbon it hangs off}
    "Find the oxygen or nitrogen sitting at every numbered position of a residue"
    holders, carbons = {}, {}
    for carbon, position in number.items():
        holder = max((other for other in adjacency[carbon] if elements[other] in _HOLDERS and other not in ring and (other != root or carbon not in ring)),
                     key = lambda other: (len(adjacency[other]), elements[other] == 'N', -other), default = None)
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
    amines = [n for n, slot, c, b in positions if slot == 'N']
    variants = [([(n, 'O' if n in turned else s, c, b) for n, s, c, b in base], implied) for size in range(len(amines) + 1)
                for turned in combinations(amines, size) for base, implied in variants]  # an amine the skeleton does not have is a modification, and the fewest such keep Neu5Ac9NAc from reading as Kdn5NAc9NAc
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
                       elements: list,  # Element of every atom
                       adjacency: dict  # {atom: set of bonded atoms}
                       ) -> tuple:  # Position (None where the forward reading would put it back) and modification name
    "Name what sits at one position, dropping the position number wherever reading the token back would restore it"
    template, tag, hetero, default = _skeleton(_split_token(residue['skeleton'])[0], residue['form'])
    holder = residue['holders'][position]
    carbon = next(atom for atom, spot in residue['number'].items() if spot == position)
    default = next((spot for spot in _free_slots(template, hetero) if spot != 1), None)
    if len([other for other in adjacency[holder] if other != carbon]) == 0:  # nothing hangs off it
        if elements[holder] not in 'ON':  # a thiol or a halogen, as in Gal6F
            return position, _modification_table()[elements[holder]].get(elements[holder])
        if elements[holder] != 'N' or hetero.get(
                position) == 'N':  # a plain hydroxyl or the skeleton's own amine: nothing to name, and nothing strict should object to
            return position, ''
        return (None if position == default else position), 'N'
    name = _modification_table()[elements[holder]].get(residue['fragments'][position])
    if name is None or elements[holder] != 'N':
        return position, name
    if position == default:  # written without a number, as in GlcNAc or MurNAc, so the N has to be spelt out
        return None, 'N' + name
    return position, name if hetero.get(position) == 'N' and name in SUBSTITUENTS else 'N' + name  # a numbered position on a sugar that already has the nitrogen, as in Neu5Ac


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
        if strict and any(name is None and position not in {residues[child]['parent'][1] for child in residue['children']} for position, name in residue['mods']):  # a phosphodiester to a child reads as an unnamed group on this side
            raise GlycanSMILESError(f"unrecognized substituent on '{residue['skeleton']}'")
        edges = []
        for child in residue['children']:
            other, built = residues[child], build(child)  # children first, so that indices grow towards the reducing end
            edges.append((add(f"{'' if other['chain'] else other['anomer']}{other['anomeric']}-{other['parent'][1]}"), built))  # an alditol has no anomer: Rib-ol(1-5)
        template, tag, hetero, enantiomer = _skeleton(_split_token(residue['skeleton'])[0], residue['form'])
        default, acid = next((spot for spot in _free_slots(template, hetero) if spot != 1), None), max((p for p in hetero if 'C{p%d}' % p in template), default = None) if (None, 'A') in residue['mods'] else None
        choices = []
        for position, group in [(position, group) for position, group in residue['mods'] if group]:  # the ways glycowork writes the same group: GlcNAc, GlcN2S, Oli3NAc on an amine, GlcAN or GlcA6N on a uronamide
            if position is None and group[0] == 'N' and group[1:] and default:
                spelled = [[(None, group)], [(None, 'N'), (default, group[1:])], [(default, group)]]
                choices.append([spelled[1], spelled[0], spelled[2]] if group[1:] in NUMBERED_N else spelled)
            else:
                choices.append([[(None, 'N@')], [(position, group)]] if position == acid is not None and group == 'N' else [[(position, group)]])
        core, states = re.fullmatch(r'(.*?)((?:\d+(?:Ulo|en))*)', residue['skeleton']).groups()  # D-6dXylHexNAc4Ulo and L-4dThrHexA4en name their keto and ene positions last
        systematic = core[1:2] != '-' and core not in SKELETONS and not residue['chain'] and (match := _SKELETON_RE.fullmatch(core)) is not None and match.group(5) and not match.group(1)  # a systematic name states its enantiomer (D-6dAltHep), which glycowork leaves out of Gal, 8eLeg or LDManHep unless lib only knows it that way (2,5-Anhydro-D-Tal)
        from glycowork.glycan_data.loader import lib  # loaded lazily, as loader sits above this module
        tokens = [''.join(f'{p},{q}-Anhydro-' for p, q in residue['anhydro']) + skeleton + ''.join(('' if position is None else str(position)) + group.rstrip('@') for position, group in sorted(
                  sum(mods, []), key = lambda mod: (mod[0] is not None, mod[0] or 0, mod[1] in ('A', 'N@'), mod[1] == 'N@'))) + states + residue['suffix']
                  for skeleton in ([core] + [f'{enantiomer}-{core}'] * (core[1:2] != '-'))[::-1 if systematic else 1] for mods in product(*choices)]
        here = add(next((token for token in tokens if token in lib), tokens[0]))  # of the spellings, the one glycowork knows
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
    adjacency, orders, elements = _adjacency(atoms, bonds), {}, [element for element, charge, chirality in atoms]
    for first, second, order in bonds:
        orders[(first, second)] = orders[(second, first)] = order
    residues, of_root = [], {}
    for cycle in _rings_of_atoms(bonds, rings):  # first find every sugar ring and how its carbons are numbered
        try:
            key, chirality, number, ring_oxygen, root, anomeric, positions, ring_size = _signature(cycle, atoms, elements, bonds, neighbors, adjacency)
        except (GlycanSMILESError, StopIteration):
            continue
        holders, carbons = _holders(number, set(cycle), root, elements, adjacency)
        anhydro = sorted((p, q) for p in holders for q in holders if p < q and holders[p] == holders[q]) + [(number[anomeric], p) for p in holders if holders[p] == root]  # one oxygen on two carbons is an anhydro bridge, as in 3,6-Anhydro-Gal or 2,7-Anhydro-Kdo
        of_root[root] = len(residues)
        residues.append({'chirality': chirality, 'number': number, 'ring': set(cycle), 'root': root, 'positions': positions, 'chain': False, 'anhydro': anhydro, 'form': '',
                         'ring_size': ring_size, 'anomeric': number[anomeric], 'holders': {p: h for p, h in holders.items() if not any(p in pair for pair in anhydro)},
                         'carbons': carbons, 'extra': {holders[q] for p, q in anhydro}, 'mods': [], 'children': [], 'suffix': ''})
    seen, best, taken = {atom for residue in residues for atom in residue['number']}, {}, {holder for residue in residues for holder in residue['holders'].values()}
    for start in [atom for atom, element in enumerate(elements) if element == 'C' and atom not in seen]:  # then the open chains no ring numbered: alditols, aldonic acids, anhydro alditols
        if start in seen:
            continue
        chain, stack = {start}, [start]
        while stack:
            stack += [other for other in adjacency[stack.pop()] if elements[other] == 'C' and other not in seen | chain]
            chain |= set(stack)
        seen |= chain
        if len(chain) == 6 and all(len(adjacency[atom] & chain) == 2 for atom in chain):  # a carbocycle: inositol, numbered like the 1D-myo template wherever that fits with the lowest locants
            ring, inositol = [start], _chain_table()['Ins']
            while len(ring) < 6:
                ring.append(next(other for other in adjacency[ring[-1]] & chain if other not in ring))
            for order in [ring[i:] + ring[:i] for i in range(6)] + [(ring[i::-1] + ring[:i:-1]) for i in range(6)]:
                number = {atom: index + 1 for index, atom in enumerate(order)}
                hetero = {number[atom]: other for atom in order for other in adjacency[atom] if elements[other] in _HOLDERS}
                if len(hetero) == 6 and tuple(_chirality(atom, atoms, elements, neighbors, number, None) for atom in order) == inositol:
                    locants = tuple(sorted(p for p, h in hetero.items() if len(adjacency[h]) > 1 or elements[h] != 'O'))
                    if start not in best or (locants,) < best[start][0]:
                        best[start] = ((locants,), 'Ins', [], number, hetero, {}, [])
            continue
        if not 4 <= len(chain) <= 9 or any(len(adjacency[atom] & chain) > 2 for atom in chain) or sum(len(adjacency[atom] & chain) == 1 for atom in chain) != 2:
            continue
        for end in sorted(atom for atom in chain if len(adjacency[atom] & chain) == 1):  # both ends could be C1; the name decides
            order = [end]
            while len(order) < len(chain):
                order.append(next(other for other in adjacency[order[-1]] & chain if other not in order))
            number, hetero, oxo = {atom: index + 1 for index, atom in enumerate(order)}, {}, {}
            for atom, position in number.items():
                single = [other for other in adjacency[atom] if elements[other] in _HOLDERS and orders[(atom, other)] == 1]
                double = [other for other in adjacency[atom] if elements[other] == 'O' and orders[(atom, other)] == 2]
                if len(single) > 1 or len(double) > 1 or (double and position not in (1, len(order))) or (set(single) & taken and (position != 1 or double or elements[single[0]] != 'O')):
                    break  # an alditol reaches a ring's hydroxyl only from its O1, as in Rib-ol(1-3)Ribf, whereas an amino acid on a ring's amine (QuiNThrAc) is no aldonic acid
                hetero.update({position: single[0]} if single else {})
                oxo.update({position: double[0]} if double else {})
            else:
                anhydro = sorted((p, q) for p in hetero for q in hetero if p < q and hetero[p] == hetero[q])
                key = (len(order), tuple(_chirality(atom, atoms, elements, neighbors, number, None) for atom in order))
                for rank, (name, slots) in enumerate(_chain_table().get(key, [])):
                    present = set(hetero) | set(oxo)
                    deoxy = sorted(slots - present)
                    if present - slots or any(p not in (1, len(order)) for p in deoxy) or (1 in oxo and 1 not in hetero and not anhydro) or (len(order) in oxo and len(order) not in hetero):
                        continue
                    locants = tuple(sorted(p for p, h in hetero.items() if len(adjacency[h]) > 1 or elements[h] != 'O'))
                    score = (len(deoxy), len(order) in oxo and 1 not in oxo, name[1] == '-', locants, rank)  # an aldonic acid is numbered from its carboxyl, and IUPAC's lowest locants settle a symmetric chain
                    if start not in best or score < best[start][0]:
                        best[start] = (score, name, deoxy, number, hetero, oxo, anhydro)
    for score, name, deoxy, number, hetero, oxo, anhydro in best.values():
        bridged, last, (prefix, core) = {p for pair in anhydro for p in pair}, len(number), (name[:2], name[2:]) if name[1] == '-' else ('', name)
        root = hetero[1] if 1 in hetero and 1 not in bridged else None
        if root is not None and root not in of_root:
            of_root[root] = len(residues)
        residues.append({'chirality': '', 'number': number, 'ring': set(number) if name == 'Ins' else set(), 'root': root, 'positions': None, 'chain': name != 'Ins', 'anhydro': anhydro, 'ring_size': 0, 'anomeric': 1,
                         'form': '' if name == 'Ins' else '-ol',
                         'holders': {p: h for p, h in hetero.items() if p != 1 and p not in bridged}, 'carbons': {p: atom for atom, p in number.items() if p in hetero and p != 1 and p not in bridged},
                         'extra': set(oxo.values()) | {hetero[q] for p, q in anhydro}, 'mods': [(None, 'A')] if last in oxo and not (1 in oxo and 1 in hetero) else [], 'children': [],
                         'anomer': '?', 'fixed': {}, 'skeleton': prefix + (','.join(map(str, deoxy)) + 'd' if deoxy else '') + core,
                         'suffix': ('-aric' if last in oxo else '-onic') if 1 in oxo and 1 in hetero else '' if 1 in oxo or name == 'Ins' else '-ol'})
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
        if 'skeleton' not in residue:
            residue['skeleton'], residue['anomer'], implied, residue['fixed'] = _match_residue(
                residue['positions'], residue['ring_size'], residue['anomeric'], residue['chirality'], residue['fragments'])
            residue['mods'] += implied
            if residue['skeleton'].endswith('-ulosonic'):  # D-3dThrHex-ulosonic: the suffix follows the modifications
                residue['skeleton'], residue['form'], residue['suffix'] = residue['skeleton'][:-9], '-ulosonic', '-ulosonic'
        own = {residue['holders'][position] for position in residue['fragments']}
        ketal = {position: atom for position in residue['fragments'] for atom in adjacency[residue['holders'][position]] if atom != residue['carbons'][position]
                 and _fragment_key(atom, None, atoms, adjacency, orders, set(own)) == 'C(C)(C(=O)(O))(R)(R)'}  # a pyruvate ketal bridges two positions (Gal4Pyr6Pyr), so neither side alone reads as a named group
        slots = _skeleton(_split_token(residue['skeleton'])[0], residue['form'])[2]
        for position in sorted(residue['fragments']):
            if position not in residue['fixed']:  # a group the skeleton's own name already covers, such as the lactyl of muramic acid
                carbon = residue['carbons'][position]
                oxo = orders[(carbon, residue['holders'][position])] == 2 and position in slots and sum(elements[other] in 'ON' for other in adjacency[carbon]) == 1  # a carbonyl where the skeleton has a hydroxyl, as in the aldehyde of Gal6ulo
                residue['mods'].append((position, 'Pyr') if list(ketal.values()).count(ketal.get(position)) == 2 else (position, 'ulo') if oxo else _name_modification(residue, position, elements, adjacency))
                if not residue['mods'][-1][
                    1]:  # whatever hangs off an unnamed position has to pass the coverage check on its own
                    residue['holders'].pop(position)
        if residue['root'] is not None:
            _link_through_root(residues, index, of_root, atoms, elements, adjacency, orders)
    size = lambda index: 1 + sum(size(child) for child in residues[index]['children'])
    for index, residue in enumerate(residues):  # of two residues sharing one anomeric diester, the one with a stated anomer, else the one with less hanging off it, is the branch
        if 'mutual' in residue and 'parent' not in residue and 'parent' not in residues[residue['mutual'][0]]:
            child, host = sorted((index, residue['mutual'][0]), key = lambda side: (residues[side]['anomer'] == '?', size(side), residues[side]['skeleton']))  # a reducing end has no stated anomer
            residues[child]['parent'] = (host, residues[host]['anomeric'])
            residues[child]['mods'].append((residues[child]['anomeric'], residues[child]['mutual'][1]))
            residues[host]['children'].append(child)
    for index, residue in enumerate(residues):  # two residues sharing one anomeric oxygen are linked to each other, as in trehalose
        twin = next((other for other, host in enumerate(residues) if other != index and host['root'] == residue['root'] and residue['root'] is not None), None)
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
        covered |= set(residue['number']) | residue['ring'] | set(residue['holders'].values()) | {residue['root']} | residue['extra']
    for residue in residues:  # only once every residue is in, so that walking a named group cannot run on into another residue
        named = any(name for position, name in residue['mods'] if position == residue['anomeric'])
        for holder in list(residue['holders'].values()) + ([residue['root']] if named else []):
            _fragment_key(holder, None, atoms, adjacency, orders, covered)
    elements, left = [element for element, charge, chirality in atoms], set(range(len(atoms))) - covered
    while left:  # each leftover piece on its own, so that several small unnamed groups never add up to a residue
        piece, stack = set(), [left.pop()]
        while stack:
            atom = stack.pop()
            piece.add(atom)
            stack += adjacency[atom] & left
            left -= adjacency[atom]
        if sum(elements[atom] == 'C' and any(elements[other] == 'O' for other in adjacency[atom]) for atom in
               piece) >= 3:
            raise GlycanSMILESError(
                'part of this molecule looks like a residue but could not be named; an open-chain sugar perhaps')


def _link_through_root(residues: list, # Perceived residues
                       index: int, # Residue to look at
                       of_root: dict, # {anomeric oxygen: residue}
                       atoms: list, # Atoms of the molecule
                       elements: list,  # Element of every atom
                       adjacency: dict,  # {atom: set of bonded atoms}
                       orders: dict  # {(first, second): bond order}
                       ) -> None:
    "Read what hangs off a residue's own anomeric oxygen: an aglycon, a phosphate, or a phosphodiester to its parent"
    residue = residues[index]
    carbon = next(atom for atom, position in residue['number'].items() if position == residue['anomeric'])
    beyond = [other for other in adjacency[residue['root']] if other != carbon]
    if not beyond:
        if elements[residue['root']] != 'O':  # a glycosylamine (GlcNAc1N) or a glycosyl halide (Glc1F)
            residue['mods'].append((residue['anomeric'], elements[residue['root']]))
        return
    if any(beyond[0] in host['number'] for host in
           residues):  # bonded straight to a residue, as every glycosidic bond is: nothing to name but the nitrogen of an N-glycoside (Glc1N(a1-4)), and walking on would only serialize that residue
        residue['mods'].append((residue['anomeric'], 'N' if elements[residue['root']] == 'N' else ''))
        return
    group, stack, hosts = {residue['root']}, [residue['root']], set()
    while stack:  # walk the group on the anomeric oxygen up to whatever residue it reaches: a phospho-, pyrophospho- or glycerol phosphodiester
        for other in adjacency[stack.pop()] - group - {carbon} - set(residue['number']):
            owner = next((parent for parent, host in enumerate(residues) if parent != index and other in host['number']), None)
            hosts |= {(owner, other)} if owner is not None else set()
            if owner is None:
                group.add(other)
                stack.append(other)
    name = (_bridge_table() if hosts else _modification_table()[elements[residue['root']]]).get(_fragment_key(residue['root'], carbon, atoms, adjacency, orders, {other for owner, other in hosts}))
    if len(hosts) == 1 and name:  # the group belongs to this residue and the bond to its parent, as glycan_to_smiles writes Glc1PGro(a1-6)
        (owner, other), = hosts
        if residues[owner]['root'] in group:  # both anomeric oxygens in one diester, as in Ara1P4N(b1-1)GlcN: which side is the branch is settled once every other link is known
            residue['mutual'] = (owner, name)
            return
        residue['parent'] = (owner, residues[owner]['number'][other])
        residues[owner]['children'].append(index)
        residue['mods'].append((residue['anomeric'], name))
        return
    residue['mods'].append((residue['anomeric'], 'N' + name if name and elements[residue['root']] == 'N' and not hosts else None if hosts else name))


_BRIDGES = {}


def _bridge_table() -> dict: # {canonical group from the anomeric oxygen to the parent's carbon: modification}
    "Serialize every anomeric group a child can link through (Man1P(a1-6), Glc1PGro(a1-6)) the way the reader walks it"
    if _BRIDGES:
        return _BRIDGES
    for name, form in ANOMERIC.items():
        if form[0] in 'ON':  # glycan_to_smiles splices the group into the parent's slot, so it starts at the parent and ends at the anomeric carbon
            atoms, bonds, rings, neighbors = parse_smiles('C' + form)  # the leading C stands in for the parent's carbon
            orders = {}
            for first, second, order in bonds:
                orders[(first, second)] = orders[(second, first)] = order
            _BRIDGES.setdefault(_fragment_key(len(atoms) - 1, None, atoms, _adjacency(atoms, bonds), orders, {0}), name)
    return _BRIDGES


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
