import pandas as pd
import numpy as np
import networkx as nx
import re
from random import sample
from importlib import resources
from collections import Counter, defaultdict
from functools import lru_cache, reduce

from glycowork.glycan_data import loader
from glycowork.glycan_data.loader import lib, unwrap, Hex, dHex, HexA, HexN, HexNAc, Pen, linkages, multireplace
from glycowork.motif.processing import min_process_glycans, rescue_glycans, rescue_compositions, parse_floating_bit, FLOATY_ALT, is_composition, canonicalize_composition
from glycowork.motif.graph import compare_glycans, glycan_to_nxGraph, graph_to_string, IS_LINKAGE
from glycowork.motif.smiles import glycan_to_molecule

chars = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11,
         'L': 12, 'M': 13, 'N': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'V': 20, 'W': 21, 'Y': 22, 'X': 23, 'Z': 24, 'z': 25}
_CORES = ['dHexNAc', 'GlcNAc', 'GalNAc', 'ManNAc', 'FucNAc', 'QuiNAc', 'RhaNAc', 'GulNAc', 'IdoNAc', 'Ins', 'MurNAc',
          '6dAltNAc', 'AcoNAc', 'HexA', 'GlcA', 'AltA', 'GalA', 'ManA', 'Tyv', 'Yer', 'Abe', 'GlcfNAc', 'GalfNAc',
          'ManfNAc', 'FucfNAc', 'IdoA', 'GulA', 'LDManHep', 'DDManHep', 'DDGlcHep', 'LyxHep', 'ManHep', 'DDAltHep',
          'IdoHep', 'DLGlcHep', 'GalHep', 'ddHex', 'ddNon', 'Unknown', 'Assigned', 'MurNGc', '6dTalNAc', '6dGul',
          'AllA', 'TalA', 'AllNAc', 'TalNAc', 'Kdn', 'Pen', 'GlcN', 'GalN', 'ManN', 'FucN', 'QuiN', 'RhaN', 'AraN',
          'IdoN', 'Glcf', 'Galf', 'Manf', 'Fucf', 'Araf', 'Lyxf', 'Xylf', '6dAltf', 'Ribf', 'Fruf', 'Apif', 'Kdof',
          'Sedf', '6dTal', 'AltNAc', '6dAlt', 'dHex', 'HexNAc', 'dNon', '4eLeg', 'GulN', 'AltN', 'AllN', 'TalN',
          'HexN', 'Glc', 'Gal', 'Man', 'Fuc', 'Qui', 'Rha', 'Ara', 'Oli', 'Gul', 'Lyx', 'Xyl', 'Dha', 'Rib', 'Kdo',
          'Tal', 'All', 'Pse', 'Leg', 'Asc', 'Hex', 'Fru', 'Alt', 'Xluf', 'Api', 'Ko', 'Pau', 'Fus', 'Erwiniose',
          'Aco', 'Bac', 'Dig', 'Thre-ol', 'Ery-ol', 'Tag', 'Sor', 'Psi', 'Mur', 'Aci', 'Sia', 'Par', 'Col', 'Ido']
_SPECIAL_MODS = {
    '1,7lactone': {'replacement': '', 'diff_moiety': ('-H2O',)},
    'Az': {'replacement': 'Ac', 'diff_moiety': ('+N3', '-H')}  # azidoacetyl is acetyl with one H swapped for N3
    # Add other special modifications here in the format:
    # 'modification': {'replacement': 'what to replace with', 'diff_moiety': tuple of chemical formulas, sign indicating loss/gain}
}
_VALID_COMPONENTS = {'Hex', 'dHex', 'HexNAc', 'HexN', 'HexA', 'Neu5Ac', 'Neu5Gc', 'Kdn', 'Pen', 'Me', 'S', 'P', 'PCho', 'PEtN', 'Ac', '-H2O', '+N3', '-H',
                     'Neu5Ac_a2-6', 'Neu5Ac_a2-3', 'Neu5Gc_a2-6', 'Neu5Gc_a2-3', 'Kdn_a2-6', 'Kdn_a2-3'}
_COUNTED_MODS = re.compile(r'^(?:[DL]-)?f?|-ol$|1(?:Ser|Thr|Asn|Cer)|[\dO]*(?:/\d+)*(?:PCho|PEtN|Ac|Me|S|P)')  # what glycan_to_composition accounts for beyond a residue's core: enantiomer, furanose, alditol, the dropped aglycones, and the substituents it counts
_COMPOSITION_INDEX = {}
_MZ_POOL = {}  # (kingdom, glycan_class) or ('custom', glycan_class) -> (source frame, unique composition item-tuples, {(mass_value, sample_prep, modification, filter_out): (compositions, masses)}) for mz_to_composition
_MASS_CACHE = {}  # (composition item-tuple, mass_value, sample_prep, modification) -> composition_to_mass result

with resources.files("glycowork.motif").joinpath("mz_to_composition.csv").open(encoding = 'utf-8-sig') as f:
    mapping_file = pd.read_csv(f)
mass_dict = dict(zip(mapping_file.composition, mapping_file["underivatized_monoisotopic"]))
_MASS_DICTS = {col: dict(zip(mapping_file.composition, mapping_file[col])) for col in mapping_file.columns[1:]} | {"underivatized_monoisotopic": mass_dict}
HYDROGEN_MASS = 1.007825
ELECTRON_MASS = 0.000548580
PROTON_MASS = HYDROGEN_MASS - ELECTRON_MASS  # charge carrier; the H atom is 0.55 mDa heavier
NEUTRAL_ADDUCTS = {'Acetonitrile', 'Trifluoroacetic acid'}  # mz_to_composition.csv entries that are neutral molecules, so their adduct ions still need the protons of [M+H]+/[M-H]-
modification_formula_dict = {'reduced': 'H2', 'reduced_D': 'HD', '2AA': 'C7H7NO', '2AB': 'C7H8N2', 'procainamide': 'C13H21N3', 'PA': 'C5H6N2-O', 'APTS': 'C16H11NO9S3-O',
                           'aminoxyTMT': 'C15H30N4O3-H2O', 'GirardP': 'C7H9N3O-H2O'}  # net gain at the reducing end: H2 by reduction (HD with NaBD4), label minus O by reductive amination (2-aminopyridine has no O, so PA loses one), label minus H2O by oxime (aminoxyTMT0, C15H30N4O3) or hydrazone (Girard's P) formation; Girard's P is a fixed-charge pyridinium (C7H10N3O+), counted as the reagent minus H so that the mode's proton makes its cation
SIALIC_PRODUCTS = {'ethyl ester': ('+C2H4',), 'methyl ester': ('+CH2',), 'lactone': ('-H2O',), 'amide': ('+NH3', '-H2O'), 'methylamide': ('+CH5N', '-H2O'),
                   'dimethylamide': ('+C2H7N', '-H2O'), 'isopropylamide': ('+C3H9N', '-H2O'), 'anilide': ('+C6H7N', '-H2O'), 'd5-anilide': ('+C6H2D5N', '-H2O')}  # what a derivatization makes of a carboxyl, as the formulas it gains and loses
# Linkage-specific sialic acid derivatizations, as (product of a2-6 sialic acids, product of the lactone-forming a2-3/8/9 ones): ethyl esterification (Reiding 2014),
# ethyl esterification and amidation (Lageveen-Kammeijer 2019), methyl esterification (Wheeler 2009), dimethylamidation (de Haan 2015), and SALSA with methylamine
# aminolysis (Nishikaze 2017, Hanamatsu 2018), glycoqueuing's aniline amides on every sialic acid (Jin 2019, GQ; usually with GirardP at the reducing end) and its improved form
# with d5-aniline on a2-6 and d0-aniline on a2-3 sialic acids (GQ-d5); any other one is passed as such a pair of SIALIC_PRODUCTS, e.g., ('dimethylamide', 'amide') for double
# amidation (Holst 2016)
SIALIC_DERIVATIZATIONS = {'EE': ('ethyl ester', 'lactone'), 'EEA': ('ethyl ester', 'amide'), 'ME': ('methyl ester', 'lactone'), 'DMA': ('dimethylamide', 'lactone'),
                          'SALSA': ('isopropylamide', 'methylamide'), 'GQ': ('anilide', 'anilide'), 'GQ-d5': ('d5-anilide', 'anilide')}
SIALIC_KEYS = {base: (f'{base}_a2-6', f'{base}_a2-3') for base in ('Neu5Ac', 'Neu5Gc', 'Kdn')}  # composition keys of sialic acids whose linkage a derivatization reads: a2-6, and a2-3 for every lactone-forming one (a2-3/8/9), as the field's E and L codes
_SIALIC_BASE = {key: (base, i) for base, keys in SIALIC_KEYS.items() for i, key in enumerate(keys)}
AMINO_ACID_FORMULAS = {'G': 'C2H3NO', 'A': 'C3H5NO', 'S': 'C3H5NO2', 'P': 'C5H7NO', 'V': 'C5H9NO', 'T': 'C4H7NO2', 'C': 'C3H5NOS', 'L': 'C6H11NO', 'I': 'C6H11NO', 'N': 'C4H6N2O2',
                       'D': 'C4H5NO3', 'Q': 'C5H8N2O2', 'K': 'C6H12N2O', 'E': 'C5H7NO3', 'M': 'C5H9NOS', 'H': 'C6H7N3O', 'F': 'C9H9NO', 'R': 'C6H12N4O', 'Y': 'C9H9NO2', 'W': 'C11H10N2O'}  # amino acid residues in a chain (free amino acid minus H2O)
_PEPTIDE = re.compile(r'(?:[A-Z](?:\[[+-][^\[\]]+\])*)+')  # one-letter peptide with signed mass or formula deltas after residues, e.g., 'NC[+C2H3NO]SK', 'M[+15.9949]NGTK' (named modifications are written as their deltas first)
PEPTIDE_MODIFICATIONS = {'Carbamidomethyl': '+C2H3NO', 'Propionamide': '+C3H5NO', 'Oxidation': '+O', 'Deamidated': '+O-HN', 'Amidated': '+HN-O', 'Acetyl': '+C2H2O', 'Phospho': '+HPO3',
                         'Methyl': '+CH2', 'Dimethyl': '+C2H4', 'Carbamyl': '+CHNO', 'Guanidinyl': '+CH2N2', 'Gln->pyro-Glu': '-NH3', 'Glu->pyro-Glu': '-H2O', 'Ammonia-loss': '-NH3',
                         'TMT': '+224.152478', 'TMT2plex': '+225.155833', 'TMT6plex': '+229.162932', 'TMTpro': '+304.207146', 'iTRAQ4plex': '+144.102063', 'iTRAQ8plex': '+304.20536'}  # Unimod (PSI-MS) names of peptide modifications, as written in brackets after a residue ('NC[Carbamidomethyl]SK'), with their signed formula deltas; isotope-coded tags as masses, which have no formula
RESIDUE_FORMULAS = {'Hex': ('C6H10O5', 3, 3), 'HexNAc': ('C8H13NO5', 3, 2), 'dHex': ('C6H10O4', 2, 2), 'Pen': ('C5H8O4', 2, 2), 'Neu5Ac': ('C11H17NO8', 5, 3), 'Neu5Gc': ('C11H17NO9', 6, 4),
                    'Kdn': ('C9H14O8', 5, 4), 'HexA': ('C6H8O6', 3, 2), 'HexN': ('C6H11NO4', 4, 3), 'S': ('SO3', -1, -1), 'P': ('HPO3', 1, -1), 'PCho': ('C5H12NO3P', 0, -1),
                    'PEtN': ('C2H6NO3P', 3, 0), 'Me': ('CH2', 0, 0), 'red_end': ('H2O', 2, 2)}  # composition key: (underivatized formula in a chain, caps it gains permethylated (CH2 each) and peracetylated (C2H2O each)), which mz_to_composition.csv's masses are computed from
_DERIV_CAP = {'permethylated': 'CH2', 'peracetylated': 'C2H2O'}  # what derivatization puts on each free OH (and N-H), shared by the mass and formula paths
_LABEL_CAPS = {'2AA': 2, '2AB': 3, 'procainamide': 2}  # caps permethylation adds to a ring-opened labelled glycan: the freed C5-OH plus the label's own acid and amides (1 for reduction and PA, whose pyridine nitrogen is not modelled as quaternized)


def constrain_prot(proteins: str | list[str], # Protein sequence(s)
                   libr: dict[str, int] | None = None # Dictionary mapping amino acids to indices
                   ) -> list[str]: # List of filtered protein sequences
    """Ensure only characters from library are present in proteins"""
    if libr is None:
        libr = chars
    # Check whether any character is not in libr and replace it with a 'z' placeholder character
    return [''.join(c if c in libr else 'z' for c in protein) for protein in ([proteins] if isinstance(proteins, str) else proteins)]


def prot_to_coded(proteins: str | list[str], # Protein sequence(s)
                  libr: dict[str, int] | None = None, # Dictionary mapping amino acids to indices
                  pad_len: int = 1000 # Length for padding sequences
                  ) -> list[list[int]]: # List of encoded protein sequences
    """Encode protein sequences for LectinOracle-flex"""
    if libr is None:
        libr = chars
    pad_label = len(libr) - 1
    # Cut off protein sequence above pad_len
    prots = [protein[:pad_len] for protein in ([proteins] if isinstance(proteins, str) else proteins)]
    # Replace forbidden characters with 'z'
    prots = constrain_prot([protein.upper() for protein in prots], libr = libr)
    # Pad up to a length of pad_len
    return [pad_sequence(string_to_labels(prot, libr = libr), max_length = pad_len, pad_label = pad_label) for prot in
            prots]


def string_to_labels(character_string: str, # String to tokenize
                     libr: dict[str, int] | None = None # Dictionary mapping characters to indices
                     ) -> list[int]: # List of character indices
    """Tokenize word by indexing characters in library"""
    if libr is None:
        libr = chars
    return list(map(libr.get, character_string))


def pad_sequence(seq: list[int], # Sequence to pad
                 max_length: int, # Target length
                 pad_label: int | None = None, # Padding token value
                 libr: dict[str, int] | None = None # Character library
                 ) -> list[int]: # Padded sequence
    """Pad sequences to same length using padding token"""
    if libr is None:
        libr = chars
    if pad_label is None:
        pad_label = len(libr)
    padding_needed = max_length - len(seq)
    return seq + [pad_label] * padding_needed if padding_needed > 0 else seq


def _expand_position_lists(sugar: str # Monosaccharide or linkage
                           ) -> str: # Monosaccharide with one position per substituent
    "Spell out position lists like Neu5,9Ac2 as Neu5Ac9Ac, the form lib and get_core use"
    return re.sub(r'(\d(?:,\d)+)([A-Z][a-z]*)\d+', lambda m: ''.join(p + m.group(2) for p in m.group(1).split(',')),
                  sugar) if ',' in sugar else sugar


def get_core(sugar: str # Monosaccharide or linkage
             ) -> str: # Core monosaccharide string
    """Retrieve core monosaccharide from modified monosaccharide"""
    sugar = _expand_position_lists(sugar)
    if (catch := next((c for c in _CORES if c in sugar), -1)) != -1:
        return catch
    if 'Neu' in sugar:
        if '5Ac' in sugar:
            return 'Neu5Ac'
        if '5Gc' in sugar:
            return 'Neu5Gc'
        if '4Ac' in sugar:
            return 'Neu4Ac'
        return 'Neu'
    if sugar.startswith(('a', 'b', '?')) or re.match('^[0-9]+(-[0-9]+)+$', sugar):
        return sugar
    return 'Monosaccharide'


@lru_cache(maxsize = 4096)
def get_modification(sugar: str # Monosaccharide or linkage
                     ) -> str: # Modification string
    """Retrieve modification from modified monosaccharide"""
    sugar = _expand_position_lists(sugar)
    core = get_core(sugar)
    # 5Ac and 5Gc are part of the core only on a sialic acid (Neu4Ac5Ac, whose core Neu5Ac is no substring of it); elsewhere they are substituents (Kdn5Ac, Galf3Ac5Ac, Pse5Ac7Ac)
    return multireplace(sugar, {core: '', 'Neu': '', '5Ac': '', '5Gc': ''} if core.startswith('Neu') else {core: ''})


def get_stem_lib(libr: dict[str, int] # Dictionary mapping glycoletters to indices
                 ) -> dict[str, str]: # Dictionary mapping modified to core monosaccharides
    """Create mapping from modified monosaccharides to core monosaccharides"""
    return {k: get_core(k) for k in libr}


stem_lib = get_stem_lib(lib)
_STEM_LIB = stem_lib
_STEM_LIB_VALUES = set(_STEM_LIB.values())
_STEM_LIB_RANK = {k: (-len(k), i) for i, k in enumerate(_STEM_LIB) if '-' not in k}  # application order of stemify_glycan: longest first, ties in dict order
_CUSTOM_STEM = [None, None, None]  # (items, rank, clean values) of the last custom stem_lib passed to stemify_glycan


def stemify_glycan(glycan: str, # Glycan in IUPAC-condensed format
                   stem_lib: dict[str, str] | None = None, # Modified to core monosaccharide mapping; default:created from lib
                   libr: dict[str, int] | None = None # Glycoletter to index mapping
                   ) -> str: # Stemmed glycan string
    """Remove modifications from all monosaccharides in glycan"""
    if libr is None:
        libr = lib
    stem_lib = (_STEM_LIB if libr is lib else get_stem_lib(libr)) if stem_lib is None else stem_lib
    if '(' not in glycan:
        return stem_lib.get(glycan, get_core(glycan))
    if stem_lib is _STEM_LIB:
        rank, clean_values = _STEM_LIB_RANK, _STEM_LIB_VALUES
    else:  # derived lookups of a custom stem_lib are kept for as long as its content is unchanged, as stemify_dataset passes the same one for every glycan
        if _CUSTOM_STEM[0] != (items := list(stem_lib.items())):
            _CUSTOM_STEM[:] = [items, {k: (-len(k), i) for i, (k, _) in enumerate(items) if '-' not in k},
                               set(stem_lib.values())]
        rank, clean_values = _CUSTOM_STEM[1], _CUSTOM_STEM[2]
    pos = (float('-inf'),)
    # Apply keys longest first, each time the next one occurring in the current glycan; glycoletters never span a bracket, so only in-token substrings are looked up
    while cands := [(r, s) for t in set(re.split(r'[()\[\]{}]', glycan)) for i in range(len(t)) for j in
                    range(i + 1, len(t) + 1) if (r := rank.get(s := t[i:j])) and r > pos]:
        pos, key = min(cands)
        glycan = glycan.replace(key, stem_lib[key])
        if set(min_process_glycans([glycan])[0]).issubset(clean_values):
            return glycan
    return glycan


def stemify_dataset(df: pd.DataFrame, # DataFrame with glycan column
                    stem_lib: dict[str, str] | None = None, # Modified to core monosaccharide mapping; default:created from lib
                    libr: dict[str, int] | None = None, # Glycoletter to index mapping
                    glycan_col_name: str = 'glycan', # Column name for glycans
                    rarity_filter: int = 1 # Modified monosaccharides occurring more often than this keep their modification
                    ) -> pd.DataFrame: # DataFrame with stemified glycans
    """Remove monosaccharide modifications from all glycans in dataset"""
    if libr is None:
        libr = lib
    if stem_lib is None:
        stem_lib = get_stem_lib(libr)
    # Get pool of monosaccharides, decide which one to stemify based on rarity
    pool = unwrap(min_process_glycans(df[glycan_col_name].tolist()))
    pool_count = Counter(pool)
    stem_lib = {**stem_lib, **{k: k for k, v in pool_count.items() if v > rarity_filter}}
    # Stemify all offending monosaccharides
    df_out = df.copy()
    df_out[glycan_col_name] = df_out[glycan_col_name].apply(lambda x: stemify_glycan(x, stem_lib = stem_lib, libr = libr))
    return df_out


def sialic_linkage_splits(composition: dict[str, int] # Composition dictionary, e.g., {'Hex': 5, 'HexNAc': 4, 'Neu5Ac': 2}
                          ) -> list[dict[str, int]]: # Every way to give its sialic acids a linkage, fewest a2-3 first, e.g., {..., 'Neu5Ac_a2-6': 2}, {..., 'Neu5Ac_a2-6': 1, 'Neu5Ac_a2-3': 1}, {..., 'Neu5Ac_a2-3': 2}; [composition] if it has none of unknown linkage
    "Spells out the linkage-resolved compositions a composition stands for, as a linkage-specific sialic acid derivatization tells them apart by mass"
    splits = [{}]
    for k, v in composition.items():
        splits = [d | {(a6 := SIALIC_KEYS[k][0]): d.get(a6, 0) + v - n, (a3 := SIALIC_KEYS[k][1]): d.get(a3, 0) + n} for d in splits for n in range(v + 1)] if k in SIALIC_KEYS else [
            d | {k: d.get(k, 0) + v} for d in splits]
    return sorted([{k: v for k, v in d.items() if v} for d in splits], key = lambda d: sum(v for k, v in d.items() if k in _SIALIC_BASE and _SIALIC_BASE[k][1]))


def get_ion_mzs(mass: float | np.ndarray, # Neutral mass(es), including any reducing-end modification or label
                max_charge: int = -2, # Signed charge ceiling: sign sets ion mode (negative/positive), magnitude the highest charge state z considered
                adducts: str | list[str] | None = None, # Adduct(s) as named in mz_to_composition.csv, e.g., ['Acetate', 'Formate'] or ['Na+', 'K+', 'NH4+']; neutral ones (NEUTRAL_ADDUCTS) ride on the protonated/deprotonated ion; default: protonated ions only
                min_mass: dict[int, float] | None = None, # Smallest neutral mass that can carry charge z, e.g., {2: 900, 3: 1500}; default: no limit
                mass_value: str = 'monoisotopic' # Mass type of mass: monoisotopic/average; proton and adduct masses follow it
                ) -> dict[str, float | np.ndarray]: # Ion name (e.g., '[M-H]-', '[M+Acetate-H]2-', '[M+2Na]2+', '[M+Na-2H]-') : theoretical m/z (NaN where min_mass excludes the ion); by charge, then z protons, one adduct with z-1 protons, z adducts (an adduct of the other charge: one adduct with z+1 protons the other way)
    "Theoretical m/z of every ion species of a neutral mass"
    m = np.asarray(mass, dtype = float)
    adducts = [adducts] if isinstance(adducts, str) else adducts
    s, sign, out = (1 if max_charge > 0 else -1), ('+' if max_charge > 0 else '-'), {}
    proton, adduct_masses = calculate_adduct_mass('H', mass_value = mass_value) - ELECTRON_MASS, _MASS_DICTS[f"underivatized_{mass_value}"]
    for z in range(1, abs(max_charge) + 1):
        ok, charge, h = m > (min_mass or {}).get(z, -np.inf), f"{z if z > 1 else ''}{sign}", (f"{sign}{z - 1 if z > 2 else ''}H" if z > 1 else '')
        out[f"[M{sign}{z if z > 1 else ''}H]{charge}"] = np.where(ok, (m + z * (s * proton)) / z, np.nan)
        for adduct in adducts or []:
            a, name = adduct_masses[adduct], adduct.rstrip('+-')
            if adduct in NEUTRAL_ADDUCTS:
                out[f"[M+{name}{sign}{z if z > 1 else ''}H]{charge}"] = np.where(ok, (m + a + z * (s * proton)) / z, np.nan)
                continue
            if adduct.endswith('+') != (s > 0):  # an ion of the other charge, as the salt adducts of sialylated glycans in negative mode ([M+Na-2H]-), takes z + 1 protons the other way
                out[f"[M+{name}{sign}{z + 1}H]{charge}"] = np.where(ok, (m + a + (z + 1) * (s * proton)) / z, np.nan)
                continue
            out[f"[M+{name}{h}]{charge}"] = np.where(ok, (m + (z - 1) * (s * proton) + a) / z, np.nan)
            if z > 1:
                out[f"[M+{z}{name}]{charge}"] = np.where(ok, (m + z * a) / z, np.nan)
    return {k: v.item() if v.ndim == 0 else v for k, v in out.items()}


def mz_to_composition(mz_value: float, # m/z value from mass spec
                      max_charge: int = -2, # Signed charge ceiling: sign sets ion mode (negative/positive), magnitude the highest charge state z considered
                      mass_value: str = 'monoisotopic', # Mass type: monoisotopic/average
                      modification: str | None = None, # Reducing end modification: reduced/reduced_D/2AA/2AB/procainamide/PA/APTS/aminoxyTMT/GirardP
                      sample_prep: str = 'underivatized', # Sample preparation method: underivatized/permethylated/deuteropermethylated/peracetylated
                      mass_tolerance: float = 0.5, # Mass tolerance for matching
                      tolerance_unit: str = "Da",  # Unit of mass_tolerance: "Da" (absolute) or "ppm"
                      kingdom: str = 'Animalia', # Taxonomic kingdom filter for choosing a subset of glycans to consider
                      glycan_class: str = 'all', # Glycan class: N/O/lipid/free/all
                      df_use: pd.DataFrame | None = None, # Custom glycan database
                      filter_out: str | set[str] | None = None, # Monosaccharide(s) to ignore during composition finding
                      deprioritized: str | set[str] | None = {"Me", "HexA", "PCho"}, # Monosaccharide(s) to use only as fallback if no other composition matches
                      extras: str | list[str] = [], # Additional operation(s): 'adduct' also considers the adduct_ions
                      adduct: str | None = None, # Chemical formula of a neutral adduct added to every candidate's mass, e.g., "C2H4O2"
                      adduct_ions: str | list[str] | None = None, # Adduct ion(s) considered with extras = ['adduct'], as named in mz_to_composition.csv (e.g., 'Na+', 'K+', 'NH4+', 'Acetate', 'Formate'); default: Acetate (negative) or Na+ (positive)
                      mass_tag: float | None = None, # Mass in Da a reducing-end label adds to the free glycan, i.e., the label minus O (e.g., 121.0528 for 2AA, 120.0687 for 2AB, 219.1735 for procainamide), added to every candidate's neutral mass
                      sialic_derivatization: str | tuple[str, str] | None = None, # Linkage-specific sialic acid derivatization ('EE', 'EEA', 'ME', 'DMA', 'SALSA', 'GQ', 'GQ-d5', or a pair of SIALIC_PRODUCTS for a2-6 and a2-3/8/9 sialic acids); compositions then come with linkage-resolved sialic acids (Neu5Ac_a2-6, Neu5Ac_a2-3)
                      return_all: bool = False # Whether to return every composition and ion within mass_tolerance, as GlycoMod lists them, instead of the first hit
                      ) -> list[dict[str, int]] | pd.DataFrame: # List of matching compositions; with return_all a DataFrame of every match (composition, ion, theoretical mz, error_ppm) in Occam order, deprioritized ones last
    """Map m/z value to matching monosaccharide composition"""
    # The pool is filtered once per kingdom/class from SugarBase, or once per class from the last custom database (one slot, so a caller making a
    # new frame per run does not pile them up), and re-filtered whenever the source frame has been swapped out
    source, pool_key = (loader.df_glycan, (kingdom, glycan_class)) if df_use is None else (df_use, ('custom', glycan_class))
    if (entry := _MZ_POOL.get(pool_key)) is None or entry[0] is not source:
        if df_use is None:
            keep = source.Kingdom.apply(lambda x: kingdom in x) if glycan_class == "all" else (source.glycan_type == glycan_class) & (source.Kingdom.apply(lambda x: kingdom in x))
            comp_pool = source.Composition[keep]
        else:
            comp_pool = df_use[df_use.glycan_type == glycan_class] if glycan_class != "all" and 'glycan_type' in df_use.columns else df_use
            comp_pool = comp_pool.Composition if 'Composition' in comp_pool.columns else comp_pool.composition.map(canonicalize_composition)  # a combinatorial_compositions space has its compositions as strings
        entry = _MZ_POOL[pool_key] = (source, list(dict.fromkeys(tuple(d.items()) for d in comp_pool)), {})
    filter_out = {filter_out} if isinstance(filter_out, str) else set(filter_out or ())
    deprioritized = {deprioritized} if isinstance(deprioritized, str) else set(deprioritized or ())
    extras = [extras] if isinstance(extras, str) else extras
    # adduct and label sit on the neutral molecule, so they shift every ion's m/z by their mass divided by its charge
    shift = (calculate_adduct_mass(adduct, mass_value = mass_value) if adduct else 0) + (mass_tag or 0)
    if adduct_ions is None:
        adduct_ions = ['Acetate'] if max_charge < 0 else ['Na+']
    tol = mass_tolerance if tolerance_unit.lower() == "da" else mz_value * mass_tolerance / 1e6
    # The compositions left after filter_out and their masses are kept with the pool, as repeated calls (one per fragment or peak) would rebuild them
    deriv = tuple(sialic_derivatization) if isinstance(sialic_derivatization, (list, tuple)) else sialic_derivatization
    if (cached := entry[2].get(mass_key := (mass_value, sample_prep, modification, frozenset(filter_out), deriv))) is None:
        comps = [t for t in entry[1] if filter_out.isdisjoint(k for k, _ in t)]
        if deriv is not None:  # each way of linking its sialic acids is a composition of its own mass
            comps = [tuple(d.items()) for t in comps for d in sialic_linkage_splits(dict(t))]
        cached = entry[2][mass_key] = (comps, np.array([_MASS_CACHE[key] if (key := (t, mass_value, sample_prep, modification, deriv)) in _MASS_CACHE else
                                                         _MASS_CACHE.setdefault(key, composition_to_mass(dict(t), mass_value = mass_value, sample_prep = sample_prep, modification = modification,
                                                                                                         sialic_derivatization = deriv))
                                                         for t in comps], dtype = float))
    comps, masses = cached
    fallback, hits = [], []
    # Compare each composition's theoretical m/z against the observed value, ion by ion in Occam order (lower charge first, protons before adducts); return first non-deprioritized hit, else first deprioritized fallback
    for ion, observed in get_ion_mzs(masses + shift, max_charge = max_charge, adducts = adduct_ions if "adduct" in extras else None, mass_value = mass_value).items():
        diffs = np.abs(observed - mz_value)
        for i in sorted(np.flatnonzero(diffs < tol), key = lambda i: diffs[i]):
            if not deprioritized.isdisjoint(k for k, _ in comps[i]):
                fallback.append((dict(comps[i]), ion, observed[i]))
            elif return_all:
                hits.append((dict(comps[i]), ion, observed[i]))
            else:
                return [dict(comps[i])]
    if return_all:
        return pd.DataFrame([(canonicalize_composition(c, as_string = True), ion, mz, (mz_value - mz) / mz * 1e6) for c, ion, mz in hits + fallback], columns = ['composition', 'ion', 'mz', 'error_ppm'])
    return [c for c, _, _ in fallback[:1]]


@rescue_compositions
def match_composition_relaxed(composition: dict[str, int] | str, # Composition as a dict (e.g., {"dHex": 1, "Hex": 1, "HexNAc": 1}) or in any format canonicalize_composition reads (e.g., "H1N1F1")
                              glycan_class: str = 'N', # Glycan class: N/O/lipid/free/all
                              kingdom: str = 'Animalia', # Taxonomic kingdom filter for choosing a subset of glycans to consider
                              df_use: pd.DataFrame | None = None # Custom glycan database
                              ) -> list[str]: # List of matching glycans
    """Map coarse-grained composition to matching glycans"""
    # sialic acids of known linkage (from a linkage-specific derivatization) are looked up as plain ones, and the structures then have to have those linkages
    linked, plain = any(k in _SIALIC_BASE for k in composition), Counter()
    for k, v in composition.items():
        plain[_SIALIC_BASE[k][0] if k in _SIALIC_BASE else k] += v
    key, source = ((glycan_class, kingdom), loader.df_glycan) if df_use is None else (('custom', glycan_class), df_use)  # one slot per class for the last custom database, as in mz_to_composition, so a caller making a new frame per run does not pile up their indices
    # Index the database by composition once; re-index whenever the source frame has been swapped out
    entry = _COMPOSITION_INDEX.get(key)
    if entry is None or entry[0] is not source:
        if df_use is None:  # SugarBase is only filtered when its index has to be (re)built, not on every lookup
            df_use = source[((source.glycan_type == glycan_class) | (glycan_class == 'all')) & (source.Kingdom.apply(lambda x: kingdom in x))]
        elif glycan_class != 'all' and 'glycan_type' in df_use.columns:  # a custom database is filtered by class as in mz_to_composition, which found the composition
            df_use = df_use[df_use.glycan_type == glycan_class]
        index = defaultdict(list)
        for glycan in df_use.glycan.values.tolist():
            index[frozenset(glycan_to_composition(glycan).items())].append(glycan)
        entry = (source, index)
        _COMPOSITION_INDEX[key] = entry
    hits = list(entry[1].get(frozenset(plain.items()), []))
    return [g for g in (resolve_sialic_linkages(g, composition) for g in hits) if g is not None] if linked else hits


def resolve_sialic_linkages(glycan: str, # Glycan in IUPAC-condensed format
                            composition: dict[str, int] | str # Composition with linkage-resolved sialic acids, e.g., {'Hex': 5, 'HexNAc': 4, 'Neu5Ac_a2-6': 1, 'Neu5Ac_a2-3': 1} or 'H5N4E1L1'
                            ) -> str | None: # The glycan with its sialic acids of unknown linkage set wherever the composition decides them, or None if its sialic acids cannot have the composition's linkages
    "Fits the sialic acid linkages of a glycan to a composition whose sialic acids carry their linkage, as a linkage-specific derivatization reads them off the mass"
    composition = canonicalize_composition(composition) if isinstance(composition, str) else composition
    sites, fill = _carboxyl_sites(glycan), []
    for base, keys in SIALIC_KEYS.items():
        mine = [s for s in sites if s[0] == base]
        known, open_ = [sum(i == j for _, i, _, _ in mine) for j in (0, 1)], [s for s in mine if s[1] is None]
        want, free = [composition.get(k, 0) for k in keys], composition.get(base, 0)  # free: sialic acids the composition gives no linkage
        need, excess = [max(w - k, 0) for w, k in zip(want, known)], sum(max(k - w, 0) for w, k in zip(want, known))
        if len(mine) != sum(want) + free or excess > free or sum(need) > len(open_):
            return None
        # the unknown linkages are decided when all of them have to be one kind, and the glycan says where they sit
        if open_ and sum(need) == len(open_) and 0 in need and all(span for _, _, _, span in open_):
            fill.extend((span, need.index(0) == 0) for _, _, _, span in open_)
    for (start, end), lactone in sorted(fill, reverse = True):
        donor, _, acceptor = glycan[start:end].partition('-')
        options = [o for o in acceptor.split('/') if o in ('3', '8', '9')] if lactone else ['6']
        if len(options) == 1:
            glycan = glycan[:start] + f'{donor}-{options[0]}' + glycan[end:]
    return glycan


def condense_composition_matching(matched_composition: list[str] # List of matching glycans
                                  ) -> list[str]: # Minimal list of representative glycans
    """Find minimum set of glycans characterizing matched composition"""
    # Keep the first copy of every glycan that no other glycan in the list refines (e.g., Gal(?1-4)GlcNAc goes when Gal(b1-4)GlcNAc is there); only pairs equal given the wildcards, a much cheaper check, are tested, which leaves floating-part glycans next to their placed versions
    n = len(matched_composition)
    equal = {(i, j) for i in range(n) for j in range(i + 1, n) if compare_glycans(matched_composition[i], matched_composition[j])}
    refines = lambda i, j: (min(i, j), max(i, j)) in equal and compare_glycans(matched_composition[i], matched_composition[j], subsumes = True)
    return [glycan for i, glycan in enumerate(matched_composition) if not any(refines(i, j) and (j < i or not refines(j, i)) for j in range(n) if j != i)]


@rescue_compositions
def compositions_to_structures(composition_list: str | dict[str, int] | list[str | dict[str, int]], # Composition(s) like {'Hex': 1, 'HexNAc': 1} or 'H1N1'
                               glycan_class: str = 'N', # Glycan class: N/O/lipid/free/all
                               kingdom: str = 'Animalia', # Taxonomic kingdom filter for choosing a subset of glycans to consider
                               abundances: pd.DataFrame | None = None, # Sample abundances matrix
                               df_use: pd.DataFrame | None = None, # Custom glycan database
                               verbose: bool = False # Whether to print non-matching compositions
                               ) -> pd.DataFrame: # DataFrame of structures x intensities
    """Map compositions to structures, supporting accompanying relative intensities"""
    composition_list = [composition_list] if isinstance(composition_list, (str, dict)) else composition_list
    if abundances is None:
        abundances = pd.DataFrame([range(len(composition_list))] * 2).T
    abundances_values = abundances.iloc[:, 1:].values.tolist()
    df_out = []
    not_matched_count = 0
    not_matched_list = []
    for k, comp in enumerate(composition_list):
        # For each composition, map it to potential structures
        matched = match_composition_relaxed(comp, glycan_class = glycan_class, kingdom = kingdom, df_use = df_use)
        # If multiple structure matches, try to condense them by wildcard clustering
        if matched:
            df_out.extend([glycan] + abundances_values[k] for glycan in condense_composition_matching(matched))
        else:
            not_matched_count += 1
            if verbose:
                not_matched_list.append(comp)
    if df_out:
        df_out = pd.DataFrame(df_out, columns = ['glycan'] + ['abundance'] * (abundances.shape[1] - 1))
    if not_matched_count:
        print(f"{not_matched_count} compositions could not be matched." + (
            f" Not matched: {not_matched_list}" if verbose else " Run with verbose = True to see which compositions."))
    return df_out if isinstance(df_out, pd.DataFrame) else pd.DataFrame()


def mz_to_structures(mz_list: float | list[float], # Precursor mass(es)
                     glycan_class: str, # Glycan class: N/O/lipid/free/all
                     kingdom: str = 'Animalia', # Taxonomic kingdom filter for choosing a subset of glycans to consider
                     abundances: pd.DataFrame | None = None, # Sample abundances matrix
                     max_charge: int = -2, # Signed charge ceiling: sign sets ion mode (negative/positive), magnitude the highest charge state z considered
                     mass_value: str = 'monoisotopic', # Mass type: monoisotopic/average
                     sample_prep: str = 'underivatized', # Sample prep: underivatized/permethylated/deuteropermethylated/peracetylated
                     mass_tolerance: float = 0.5, # Mass tolerance for matching
                     tolerance_unit: str = "Da",  # Unit of mass_tolerance: "Da" (absolute) or "ppm"
                     modification: str | None = None, # Reducing end modification: reduced/reduced_D/2AA/2AB/procainamide/PA/APTS/aminoxyTMT/GirardP
                     df_use: pd.DataFrame | None = None, # Custom glycan database
                     filter_out: str | set[str] | None = None,  # Monosaccharide(s) to ignore
                     deprioritized: str | set[str] | None = {"Me", "HexA", "PCho"},  # Monosaccharide(s) to use only as fallback if no other composition matches
                     verbose: bool = False, # Whether to print non-matching compositions
                     mass_tag: float | None = None, # Mass in Da a reducing-end label adds to the free glycan, i.e., the label minus O (e.g., 121.0528 for 2AA), added to every candidate's neutral mass
                     sialic_derivatization: str | tuple[str, str] | None = None # Linkage-specific sialic acid derivatization ('EE', 'EEA', 'ME', 'DMA', 'SALSA', 'GQ', 'GQ-d5', or a pair of SIALIC_PRODUCTS); the mass then gives each sialic acid's linkage, and only structures with these linkages are returned, unknown ones filled in where the mass decides them
                     ) -> pd.DataFrame | list: # DataFrame of structures x intensities or empty list
    """Map precursor masses to structures, supporting accompanying relative intensities"""
    mz_list = [mz_list] if isinstance(mz_list, (int, float)) else mz_list
    if abundances is None:
        abundances = pd.DataFrame([range(len(mz_list))] * 2).T
    # Check glycan class
    if glycan_class not in {'N', 'O', 'free', 'lipid', 'all'}:
        print("Not a valid class for mz_to_composition; currently N/O/free/lipid/all matching is supported. For everything else, run compositions_to_structures separately.")
    # Map each m/z value to potential compositions
    compositions = [mz_to_composition(mz, max_charge = max_charge, mass_value = mass_value, modification = modification, sample_prep = sample_prep,
                                      mass_tolerance = mass_tolerance, tolerance_unit = tolerance_unit, kingdom = kingdom, glycan_class = glycan_class,
                                      df_use = df_use, filter_out = filter_out, deprioritized = deprioritized, mass_tag = mass_tag,
                                      sialic_derivatization = sialic_derivatization) for mz in mz_list]
    # Map each of these potential compositions to potential structures
    out_structures = []
    for m, comp in enumerate(compositions):
        out_structures.append(compositions_to_structures(comp, glycan_class = glycan_class,
                                                         abundances = abundances.iloc[[m]], kingdom = kingdom, df_use = df_use, verbose = verbose))
    return pd.concat(out_structures, axis = 0).reset_index(drop = True) if out_structures else []


def mask_rare_glycoletters(glycans: list[str], # List of IUPAC-condensed glycans
                           thresh_monosaccharides: int | None = None, # Threshold for rare monosaccharides (default: 0.001*len(glycans))
                           thresh_linkages: int | None = None # Threshold for rare linkages (default: 0.03*len(glycans))
                           ) -> list[str]: # List of glycans with masked rare elements
    """Mask rare monosaccharides and linkages in glycans"""
    # Get rarity thresholds
    if thresh_monosaccharides is None:
        thresh_monosaccharides = int(np.ceil(0.001 * len(glycans)))
    if thresh_linkages is None:
        thresh_linkages = int(np.ceil(0.03 * len(glycans)))
    rares = unwrap(min_process_glycans(glycans))
    rare_linkages, rare_monosaccharides = [], []
    # Sort monosaccharides and linkages into different bins; uncertain linkages like b1-3/4 are linkages too
    for x in rares:
        (rare_monosaccharides, rare_linkages)[bool(IS_LINKAGE(x))].append(x)
    rare_elements = [rare_monosaccharides, rare_linkages]
    thresholds = [thresh_monosaccharides, thresh_linkages]
    # Establish which ones are considered to be rare
    rare_dict = [
        {x: 'Monosaccharide' if i == 0 else '?1-?' if x.lstrip('ab?')[0] == '1' else '?2-?'
         for x, count in Counter(rare_elements[i]).items() if count <= thresholds[i]}
        for i in range(2)
    ]
    out = []
    for glycan in glycans:
        # Mask rare monosaccharides only as whole glycoletters (from a bracket or the start up to a '(' or the end), so masking Alt leaves 6dAlt and 2,5-Anhydro-Alt intact
        glycan = re.sub(r'(?<![^()\[\]{}])[^()\[\]{}]+(?=\(|$)', lambda m: rare_dict[0].get(m.group(), m.group()), glycan)
        # Replace rare linkages
        for k, v in rare_dict[1].items():
            glycan = glycan.replace(f'({k})', f'({v})')
        out.append(glycan)
    return out


@lru_cache(maxsize = 4096)
def map_to_basic(glycoletter: str, # Monosaccharide or linkage
                 obfuscate_ptm: bool = True # Whether to remove PTM position specificity
                 ) -> str: # Base monosaccharide/linkage
    """Map monosaccharide/linkage to corresponding base form"""
    conditions = [(Hex, 'Hex'), (dHex, 'dHex'), (HexA, 'HexA'), (HexN, 'HexN'), (HexNAc, 'HexNAc'), (Pen, 'Pen'), (linkages, '?1-?')]
    for cond, ret in conditions:
        if glycoletter in cond:
            return ret
    if re.search(r"\d\-\d", glycoletter):
        return "?1-?"
    g2 = re.sub(r"\d", 'O', glycoletter)
    if 'S' in glycoletter:
        if g2 in {k + 'OS' for k in Hex}:
            return 'HexOS' if obfuscate_ptm else 'Hex' + glycoletter[-2:]
        elif g2 in {k + 'OS' for k in HexNAc}:
            return 'HexNAcOS' if obfuscate_ptm else 'HexNAc' + glycoletter[-2:]
        elif g2 in {k + 'OS' for k in HexA}:
            return 'HexAOS' if obfuscate_ptm else 'HexA' + glycoletter[-2:]
        elif g2 in {k + 'S' for k in HexN}:
            return 'HexNS'
    if 'P' in glycoletter:
        if g2 in {k + 'OP' for k in Hex}:
            return 'HexOP' if obfuscate_ptm else 'Hex' + glycoletter[-2:]
        elif g2 in {k + 'OP' for k in HexNAc}:
            return 'HexNAcOP' if obfuscate_ptm else 'HexNAc' + glycoletter[-2:]
    return glycoletter


def structure_to_basic(glycan: str # Glycan in IUPAC-condensed format
                       ) -> str: # Base topology string
    """Convert glycan structure to base topology"""
    if glycan.endswith('-ol'):
        glycan = glycan[:-3]
    if '(' not in glycan:
        return map_to_basic(glycan)
    ggraph = glycan_to_nxGraph(glycan).copy()
    node_dict = dict(ggraph.nodes(data = True))
    nx.set_node_attributes(ggraph, {k: map_to_basic(node_dict[k]['string_labels']) for k in ggraph.nodes},
                           'string_labels')
    for _, d in ggraph.nodes(data = True):
        if 'anchors' in d:  # an anchor that still names specific monosaccharides can no longer resolve against the mapped backbone
            for link, anchor in list(d['anchors'].items()):
                ag = glycan_to_nxGraph(anchor.replace('^', '')).copy()
                nx.set_node_attributes(ag, {k2: map_to_basic(v['string_labels']) for k2, v in ag.nodes(data = True)},
                                       'string_labels')
                ag.nodes[next(i for i, x in enumerate(min_process_glycans([anchor])[0]) if x.endswith('^'))][
                    'string_labels'] += '^'
                d['anchors'][link] = graph_to_string(ag)
    return graph_to_string(ggraph)


def _carboxyl_sites(glycan: str # Glycan in IUPAC-condensed format
                    ) -> list[tuple[str, int | None, tuple[int, int], tuple[int, int] | None]]: # (base residue, product index in a SIALIC_DERIVATIZATIONS pair or None if the linkage does not tell, span of the residue in glycan, span of its linkage)
    "Every carboxyl a linkage-specific sialic acid derivatization reaches: sialic acids by their linkage, uronic acids as open carboxyls like a2-6 sialic acids"
    sites = []
    for m in re.finditer(r'([^()\[\]{}]+)(?:\(([^()]*)\)|$)', glycan):
        try:
            base = map_to_basic(stem_lib[m[1]] if m[1] in stem_lib else get_core(m[1]))
        except Exception:
            continue
        if base == 'HexA' and not re.search(r'6[A-Z]|AN(?![a-z])', m[
            1]):  # a uronic acid carboxyl that already is an ester or amide (GalA6Me, GlcA6N, GlcAN, GlcA6GroN) reacts no further
            sites.append((base, 0, m.span(1), m.span(2) if m[2] else None))
        elif base in SIALIC_KEYS and not m[1].startswith('Neu1'):  # a carboxyl that already is an ester, amide or lactone (Neu1Me5Ac, Neu1N5Ac, Neu1,7lactone5Ac) reacts no further
            to = set(m[2].split('-')[-1].split('/')) if m[2] and '-' in m[2] else {'?'}  # the root, an unknown position, or a choice between a2-6 and a lactone-forming one
            sites.append((base, 0 if to == {'6'} else 1 if to <= {'3', '8', '9'} else None, m.span(1), m.span(2) if m[2] else None))
    return sites


@rescue_glycans
def glycan_to_composition(glycan: str, # Glycan in IUPAC-condensed format
                          stem_libr: dict[str, str] | None = None, # Modified to core monosaccharide mapping; default: created from lib
                          sialic_linkage: bool = False # Whether to count Neu5Ac, Neu5Gc, and Kdn by linkage, as a linkage-specific derivatization tells them apart: a2-6 (Neu5Ac_a2-6) or lactone-forming a2-3/8/9 (Neu5Ac_a2-3); one of unknown linkage (a2-3/6) stays Neu5Ac
                          ) -> dict[str, int]: # Dictionary of monosaccharide counts; empty if a residue or substituent has no composition key
    """Map glycan to its composition"""
    if not isinstance(glycan, str):
        raise TypeError(f"glycan_to_composition takes one glycan sequence, got {type(glycan).__name__} {glycan!r:.80}{'; a composition dict already is one, so pass it to composition_to_mass or canonicalize_composition' if isinstance(glycan, dict) else ''}.")
    if stem_libr is None:
        stem_libr = stem_lib
    floating = ''
    if '{' in glycan:
        if '^' in glycan:
            glycan = FLOATY_ALT.sub(lambda m: '{' + parse_floating_bit(m.group(1))[0] + '}', glycan)
        floating = ''.join(re.findall(r'\{((?:[\dO?/]*(?:PCho|PEtN|Ac|Me|S|P))+)\}', glycan))  # a floating substituent ({OS}, {6P}) is counted below, but is no residue
        glycan = re.sub(r'\{(?:[\dO?/]*(?:PCho|PEtN|Ac|Me|S|P))+\}', '', glycan).replace('{', '').replace('}', '')
    # the amino acid and ceramide aglycones canonicalize_iupac keeps (GalNAc1Ser, GlcNAc1Asn) are not part of the composition, as for the aglycone residues already in lib; it sorts them among the residue's modifications (GalNAc6S1Thr to GalNAc1Thr6S)
    glycan = re.sub(r'1(?:Ser|Thr|Asn|Cer)((?:\d+[A-Z][A-Za-z]*)*)$', lambda m: m[0] if m[1] and re.split(r'[)\]]', glycan)[-1] in stem_libr else m[1], glycan)
    diff_moieties = Counter()
    for mod, info in _SPECIAL_MODS.items():
        if n := glycan.count(mod):  # every occurrence counts, as replace swaps them all at once
            diff_moieties.update({moiety: n for moiety in info['diff_moiety']})
            glycan = glycan.replace(mod, info['replacement'])
    # an alditol of a lib residue (GalNAc-ol, the reduced end of most O-glycans, is not in lib) has the residue's composition
    letters = [k[:-3] if k.endswith('-ol') and k not in stem_libr and k[:-3] in stem_libr else k for k in min_process_glycans([glycan])[0]]
    if unknown := [k for k in letters if '/' not in k and k not in stem_libr and not IS_LINKAGE(k)]:  # a linkage needs no core, and canonicalize_iupac writes ones lib may lack (Kdo(?2-4), Glc(?1-1))
        raise ValueError(f"Cannot map the glycoletter(s) {unknown} of glycan '{glycan}' onto a core monosaccharide; " + (
            f"'{glycan}' is a composition, which canonicalize_composition reads and composition_to_mass weighs." if is_composition(glycan) else
            "check their spelling or pass an extended stem_libr via get_stem_lib(expand_lib(lib, [glycan]))."))
    # a residue holding more than its core and the substituents counted below (Gal6Aep, Rha4N, GlcNAcA, Glc-onic, 3,6-Anhydro-Gal, Gal4Pyr6Pyr) has a mass no composition key expresses
    if any(_COUNTED_MODS.sub('', get_modification(k)) for k in letters if not IS_LINKAGE(k) and not re.search(r'/[A-Z]', k)):
        return {}
    composition = Counter(sorted(
        [map_to_basic((stem_libr[key] if (key := re.sub(r"/\d", "", k)) in stem_libr else get_core(key)) if '/' in k else stem_libr.get(k, k)) for k in
         letters]))
    composition.update(diff_moieties)
    glycan += floating
    for mod in ('Me', 'PCho', 'PEtN'):
        if mod in glycan:
            composition[mod] = glycan.count(mod)
    if n_p := len(re.findall(r'P(?!en|yr|Cho|EtN|am|ro)', glycan)):
        composition['P'] = n_p
    if n_s := len(re.findall(r'S(?!ia|or|ed|er|te|uc)', glycan)):
        composition['S'] = n_s
    # every O-acetyl (OAc, 1Ac to 9Ac) and the N-acetyl of an N-acetylphosphoethanolamine (PEtNAc), except the N-acetyl that makes each Neu5Ac
    if n_ac := len(re.findall(r'(?:[\dO]|PEtN)Ac', glycan)) - composition.get('Neu5Ac', 0):
        composition['Ac'] = n_ac
    composition.pop('?1-?', None)
    if sialic_linkage:
        for base, i, _, _ in _carboxyl_sites(glycan):
            if base in SIALIC_KEYS and i is not None and composition[base]:
                composition[base] -= 1
                composition[SIALIC_KEYS[base][i]] += 1
        composition = +composition
    return dict(composition) if all(k in _VALID_COMPONENTS for k in composition) else {}


def calculate_adduct_mass(formula: str, # Chemical formula of adduct (e.g., "C2H4O2", "-H2O", "+Na"), or signed parts (e.g., "C5H6N2-O")
                          mass_value: str = 'monoisotopic', # Mass type: monoisotopic/average
                          enforce_sign: bool = False # If True, returns 0 for unsigned formulas
                          ) -> float: # Formula mass
    """Calculate mass of adduct from chemical formula, including signed formulas"""
    if enforce_sign and not formula.startswith(('+', '-')):
        return 0
    element_masses = {
        'monoisotopic': {'C': 12.0000, 'H': 1.007825, 'D': 2.014102, 'O': 15.994915, 'N': 14.003074, 'S': 31.972071, 'P': 30.973762, 'Na': 22.989769, 'K': 38.963707, 'Cl': 34.968853, 'F': 18.998403, 'Br': 78.918337, 'I': 126.904473},
        'average': {'C': 12.0107, 'H': 1.00794, 'D': 2.014102, 'O': 15.9994, 'N': 14.0067, 'S': 32.065, 'P': 30.973762, 'Na': 22.989769, 'K': 39.0983, 'Cl': 35.453, 'F': 18.998403, 'Br': 79.904, 'I': 126.904473}
    }
    if not re.fullmatch(r'[+-]?(?:[A-Z][a-z]?\d*)+(?:[+-](?:[A-Z][a-z]?\d*)+)*', formula):
        raise ValueError(f"Cannot parse the formula '{formula}'; write it as element symbols with counts, e.g., 'C2H4O2', or as signed parts, e.g., 'C5H6N2-O'.")
    elements = [(sign, el, n) for sign, part in re.findall(r'([+-]?)([^+-]+)', formula) for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', part)]  # a net change that loses atoms, as the label minus O of 2-aminopyridine (C5H6N2-O), is written as signed parts
    if unknown := sorted({el for _, el, _ in elements} - element_masses[mass_value].keys()):
        raise ValueError(f"Cannot calculate the mass of '{formula}': {unknown} are not element symbols with a known mass ({sorted(element_masses[mass_value])}).")
    return sum((-1 if sign == '-' else 1) * element_masses[mass_value][el] * (int(n) if n else 1) for sign, el, n in elements)


def _peptide_deltas(peptide: str # Peptide in one-letter code, with signed deltas ('C[+C2H3NO]') or named modifications ('C[Carbamidomethyl]') after residues
                    ) -> str: # The peptide with every named modification written as its signed delta from PEPTIDE_MODIFICATIONS
    "Writes the named modifications of a peptide as their signed deltas"
    if unknown := sorted(set(re.findall(r'\[([A-Za-z][^\[\]]*)\]', peptide)) - PEPTIDE_MODIFICATIONS.keys()):
        raise ValueError(f"Unknown peptide modification(s) {unknown} in {peptide!r}; name one of {sorted(PEPTIDE_MODIFICATIONS)}, or give its signed formula or mass, e.g., 'C[+C2H3NO]' or 'M[+15.9949]'.")
    return re.sub(r'\[([A-Za-z][^\[\]]*)\]', lambda m: f'[{PEPTIDE_MODIFICATIONS[m[1]]}]', peptide)


@rescue_compositions
def composition_to_mass(dict_comp_in: dict[str, int] | str, # Composition dictionary of monosaccharide:count, or a string in any format canonicalize_composition reads (e.g., "H5N4F1")
                        mass_value: str = 'monoisotopic', # Mass type: monoisotopic/average
                        sample_prep: str = 'underivatized', # Sample prep: underivatized/permethylated/deuteropermethylated/peracetylated
                        adduct: str | float | None = None, # Chemical formula of adduct (e.g., "C2H4O2") OR its exact mass in Da
                        modification: str | None = None, # Reducing end modification: reduced/reduced_D/2AA/2AB/procainamide/PA/APTS/aminoxyTMT/GirardP
                        peptide: str | None = None, # Peptide carrying the glycan(s), in one-letter code with optional signed mass or formula deltas, or named modifications (PEPTIDE_MODIFICATIONS), after a residue (e.g., 'EEQYNSTYR', 'NC[+C2H3NO]SK' or 'NC[Carbamidomethyl]SK' for carbamidomethylation, 'M[+15.9949]NGTK'), to get the glycopeptide mass; several glycans on one peptide are passed as their total composition
                        sialic_derivatization: str | tuple[str, str] | None = None # Linkage-specific sialic acid derivatization of an underivatized sample, a key of SIALIC_DERIVATIZATIONS ('EE', 'EEA', 'ME', 'DMA', 'SALSA', 'GQ', 'GQ-d5') or a pair of SIALIC_PRODUCTS (a2-6, a2-3/8/9); every sialic acid then needs its linkage ('H5N4E1L1', {'Neu5Ac_a2-6': 1, 'Neu5Ac_a2-3': 1}), and uronic acids take the a2-6 product; peptide carboxyls are left to peptide's deltas
                        ) -> float: # Theoretical mass
    """Calculate theoretical mass from composition, optionally as a glycopeptide"""
    if sample_prep == 'deuteropermethylated':  # CD3 for every CH3 permethylation adds: the permethylated mass plus D3-H3 per cap, the caps counted as the CH2s permethylation added
        mass = composition_to_mass(dict_comp_in, mass_value = mass_value, sample_prep = 'permethylated', adduct = adduct, modification = modification, peptide = peptide, sialic_derivatization = sialic_derivatization)
        return mass + round((mass - composition_to_mass(dict_comp_in, mass_value = mass_value, adduct = adduct, modification = modification)) / calculate_adduct_mass('CH2', mass_value = mass_value)) * calculate_adduct_mass('D3-H3', mass_value = mass_value)
    peptide = _peptide_deltas(peptide) if peptide else peptide
    if peptide is not None and (sample_prep != 'underivatized' or modification or not _PEPTIDE.fullmatch(peptide) or set(re.sub(r'\[[^\]]*\]', '', peptide)) - AMINO_ACID_FORMULAS.keys()):
        raise ValueError(f"Cannot calculate a glycopeptide mass for peptide {peptide!r}: it has to be underivatized and unlabeled, written in one-letter code from {''.join(AMINO_ACID_FORMULAS)}, with signed deltas such as 'C[+C2H3NO]' or 'M[+15.9949]', or named modifications such as 'C[Carbamidomethyl]', after a residue.")
    dict_comp, carboxyl_mass = dict_comp_in.copy(), 0
    if sialic_derivatization is not None:
        products = SIALIC_DERIVATIZATIONS.get(sialic_derivatization) if isinstance(sialic_derivatization, str) else tuple(sialic_derivatization)
        if products is None or len(products) != 2 or not set(products) <= SIALIC_PRODUCTS.keys():
            raise ValueError(f"Unknown sialic_derivatization {sialic_derivatization!r}; use one of {list(SIALIC_DERIVATIZATIONS)} or a pair of products for a2-6 and a2-3/8/9 sialic acids from {list(SIALIC_PRODUCTS)}.")
        if sample_prep != 'underivatized':
            raise ValueError(f"sialic_derivatization is a derivatization of underivatized glycans, not of {sample_prep} ones.")
        if unknown := [k for k in SIALIC_KEYS if dict_comp.get(k)]:
            raise ValueError(f"A linkage-specific derivatization weighs each sialic acid by its linkage, which {unknown} of {dict_comp_in} do not give; write them as a2-6 or a2-3 (e.g., 'H5N4E1L1' or {{'Neu5Ac_a2-6': 1, 'Neu5Ac_a2-3': 1}}).")
        deltas = [sum(calculate_adduct_mass(f, mass_value = mass_value) for f in SIALIC_PRODUCTS[p]) for p in products]
        carboxyl_mass = dict_comp.get('HexA', 0) * deltas[0] + sum(v * deltas[_SIALIC_BASE[k][1]] for k, v in dict_comp.items() if k in _SIALIC_BASE)
    # a sialic acid of known linkage weighs as its residue, plus what a derivatization made of its carboxyl
    for k in [k for k in dict_comp if k in _SIALIC_BASE]:
        base = _SIALIC_BASE[k][0]
        dict_comp[base] = dict_comp.get(base, 0) + dict_comp.pop(k)
    if (mass_key := f"{sample_prep}_{mass_value}") not in _MASS_DICTS:
        raise ValueError(f"No masses for sample_prep {sample_prep!r} with mass_value {mass_value!r}; sample_prep is underivatized/permethylated/deuteropermethylated/peracetylated, mass_value monoisotopic/average.")
    mass_dict_in = _MASS_DICTS[mass_key]
    for old_key, new_key in {'S': 'Sulphate', 'P': 'Phosphate', 'Me': 'Methyl'}.items():
        if old_key in dict_comp:
            dict_comp[new_key] = dict_comp.pop(old_key)
    # O-acetylation adds acetyl minus H (net +C2H2O = 42.0106 monoisotopic), not full acetate (59 Da); an O-acetyl or a methyl (ether or glycoside) also takes an OH that derivatization would otherwise have capped (CH2 permethylated, C2H2O peracetylated), the rarer methyl ester of a uronic acid aside
    cap = _DERIV_CAP.get(sample_prep)
    ac_count = dict_comp.pop('Ac', 0)
    if missing := [k for k in dict_comp if (k not in _VALID_COMPONENTS | {'Sulphate', 'Phosphate', 'Methyl'} or pd.isna(mass_dict_in.get(k))) and not k.startswith(('+', '-'))]:  # the ion and adduct rows of mz_to_composition.csv are no residues: a key 'H' (Hex in shorthand) was weighed as a hydrogen atom
        raise ValueError(
            f"{missing} have no {mass_key} mass as composition components ({sorted(_VALID_COMPONENTS)}), so no mass can be calculated for this composition")
    total_mass = sum(v * (mass_dict_in.get(k) or calculate_adduct_mass(k, mass_value = mass_value, enforce_sign = True))
                     for k, v in dict_comp.items()) + mass_dict_in['red_end'] + ac_count * calculate_adduct_mass('C2H2O', mass_value = mass_value) - (
        ac_count + dict_comp.get('Methyl', 0)) * (calculate_adduct_mass(cap, mass_value = mass_value) if cap else 0) + carboxyl_mass
    if adduct:
        total_mass += calculate_adduct_mass(adduct, mass_value = mass_value) if isinstance(adduct, str) else adduct
    if modification:
        if modification not in modification_formula_dict:
            raise ValueError(
                f"Unknown reducing-end modification {modification!r}; choose one of {list(modification_formula_dict)} or pass a label mass via adduct.")
        total_mass += calculate_adduct_mass(modification_formula_dict[modification], mass_value = mass_value)
        if cap:  # reduction and reductive amination open the ring, which frees one OH (at C5, previously the ring oxygen); permethylation also caps a label's own acid and amide (2AA's COOH, 2AB's CONH2 twice, procainamide's CONH), which peracetylation spares, while the label's NH takes the place of the C1-OH red_end already caps
            total_mass += (_LABEL_CAPS.get(modification, 1) if sample_prep == 'permethylated' else 1) * calculate_adduct_mass(cap, mass_value = mass_value)
    if peptide:  # each glycan loses its reducing-end water on attachment and the peptide keeps one, so residues plus the one water counted above are the whole glycopeptide, however many glycans it carries
        total_mass += sum(calculate_adduct_mass(AMINO_ACID_FORMULAS[aa], mass_value = mass_value) for aa in re.sub(r'\[[^\]]*\]', '', peptide)) + sum(
            float(d) if re.fullmatch(r'[+-]\d*\.?\d+', d) else calculate_adduct_mass(d, mass_value = mass_value) for d in re.findall(r'\[([^\]]+)\]', peptide))
    return total_mass


def _molecule_formula(glycan: str, # Glycan in IUPAC-condensed format
                      sample_prep: str = 'underivatized' # Sample prep: underivatized/permethylated/peracetylated
                      ) -> tuple[Counter, int]: # Element counts of the underivatized molecule, and how many caps (CH2 or C2H2O) derivatization adds
    """Atoms of a glycan's molecule and its derivatization sites, for glycans no composition expresses"""
    # the aglycone is stripped as glycan_to_composition does, with hydrogens from the standard valences as in get_molecular_properties; permethylation caps every
    # O-H and N-H but a sulfate's (a charged phosphate oxygen counts as its OH) and turns a free amine into a trimethylammonium (counted, like PCho, as the neutral molecule minus H), peracetylation every alcohol and free amine once
    mol = glycan_to_molecule(re.sub(r'1(?:Ser|Thr|Asn|Cer)(?![a-z])', '', glycan))
    valence, neighbors = [0] * len(mol.atoms), [[] for _ in mol.atoms]
    for i, j, o in mol.bonds:
        valence[i] += o
        valence[j] += o
        neighbors[i].append((j, o))
        neighbors[j].append((i, o))
    hydrogens = [max({'C': 4, 'N': 3, 'O': 2, 'S': 2, 'P': 3, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1}.get(el, 0) - valence[k] + charge, 0) for k, (el, charge, _) in enumerate(mol.atoms)]
    acyl = {k for k in range(len(mol.atoms)) if any(mol.atoms[j][0] in 'CSP' and any(mol.atoms[q][0] == 'O' and q != k and o == 2 for q, o in neighbors[j]) for j, _ in neighbors[k])}
    if sample_prep == 'permethylated':
        caps = sum(hydrogens[k] + (charge < 0) + (el == 'N' and hydrogens[k] > 0 and k not in acyl) for k, (el, charge, _) in enumerate(mol.atoms) if el in 'ON' and not (el == 'O' and any(mol.atoms[j][0] == 'S' for j, _ in neighbors[k])))
    else:
        caps = sum(min(hydrogens[k], 1) for k, (el, charge, _) in enumerate(mol.atoms) if el in 'ON' and k not in acyl and not charge)
    return Counter(atom[0] for atom in mol.atoms) + Counter({'H': sum(hydrogens)}), caps


def glycan_to_mass(glycan: str | dict[str, int], # Glycan in IUPAC-condensed format, or a composition (dict, or a string canonicalize_composition reads, such as "H5N4F1")
                   mass_value: str = 'monoisotopic', # Mass type: monoisotopic/average
                   sample_prep: str = 'underivatized', # Sample prep: underivatized/permethylated/deuteropermethylated/peracetylated
                   stem_libr: dict[str, str] | None = None, # Modified to core monosaccharide mapping
                   adduct: str | float | None = None, # Chemical formula of adduct (e.g., "C2H4O2") OR its exact mass in Da
                   modification: str | None = None, # Reducing end modification: reduced/reduced_D/2AA/2AB/procainamide/PA/APTS/aminoxyTMT/GirardP; a glycan ending in an alditol (-ol) is reduced already
                   peptide: str | None = None, # Peptide carrying the glycan, in one-letter code with optional signed mass or formula deltas, or named modifications (PEPTIDE_MODIFICATIONS), after a residue (e.g., 'EEQYNSTYR', 'NC[+C2H3NO]SK', 'NC[Carbamidomethyl]SK'), to get the glycopeptide mass
                   sialic_derivatization: str | tuple[str, str] | None = None # Linkage-specific sialic acid derivatization of an underivatized sample ('EE', 'EEA', 'ME', 'DMA', 'SALSA', 'GQ', 'GQ-d5', or a pair of SIALIC_PRODUCTS for a2-6 and a2-3/8/9 sialic acids); each sialic acid then needs a known linkage class, a2-3/6 or a2-? raises
                   ) -> float: # Theoretical mass
    """Calculate theoretical mass from glycan or composition, optionally as a glycopeptide"""
    if sample_prep == 'deuteropermethylated':  # as in composition_to_mass, from the permethylated mass, so that alditol, inositol and atom-by-atom caps count too
        mass = glycan_to_mass(glycan, mass_value = mass_value, sample_prep = 'permethylated', stem_libr = stem_libr, adduct = adduct, modification = modification, peptide = peptide, sialic_derivatization = sialic_derivatization)
        return mass + round((mass - glycan_to_mass(glycan, mass_value = mass_value, stem_libr = stem_libr, adduct = adduct, modification = modification)) / calculate_adduct_mass('CH2', mass_value = mass_value)) * calculate_adduct_mass('D3-H3', mass_value = mass_value)
    if isinstance(glycan, dict) or is_composition(glycan):  # a composition has no sequence to read, so it goes straight to composition_to_mass
        return composition_to_mass(glycan, mass_value = mass_value, sample_prep = sample_prep, adduct = adduct, modification = modification, peptide = peptide,
                                   sialic_derivatization = sialic_derivatization)
    if sialic_derivatization is not None:
        # each derivatized carboxyl adds what its product gains, whichever way the rest of the glycan is weighed
        sites = _carboxyl_sites(glycan)
        if unknown := [base for base, i, _, _ in sites if i is None]:
            raise ValueError(f"A linkage-specific derivatization weighs each sialic acid by its linkage, which is unknown for {len(unknown)} {unknown[0]} of '{glycan}' (a2-3/6, a2-?, or the reducing end).")
        carboxyls = Counter(base if base == 'HexA' else SIALIC_KEYS[base][i] for base, i, _, _ in sites)
        return glycan_to_mass(glycan, mass_value = mass_value, sample_prep = sample_prep, stem_libr = stem_libr, adduct = adduct, modification = modification, peptide = peptide) + composition_to_mass(
            carboxyls, mass_value = mass_value, sample_prep = sample_prep, sialic_derivatization = sialic_derivatization) - composition_to_mass(carboxyls, mass_value = mass_value, sample_prep = sample_prep)
    if stem_libr is None:
        stem_libr = stem_lib
    comp = glycan_to_composition(glycan, stem_libr = stem_libr)
    if comp:
        mass = composition_to_mass(comp, mass_value = mass_value, sample_prep = sample_prep, adduct = adduct,
                                   modification = modification or ('reduced' if glycan.endswith('-ol') else None), peptide = peptide)  # the composition of Gal(b1-4)Glc-ol is that of Gal(b1-4)Glc, but the alditol carries two more H (and one more OH to derivatize)
        if inner := glycan.count('-ol') - glycan.endswith('-ol'):  # so does every alditol inside the chain, as the ribitol phosphates of teichoic acids
            mass += inner * (composition_to_mass(comp, mass_value = mass_value, sample_prep = sample_prep, modification = 'reduced') - composition_to_mass(comp, mass_value = mass_value, sample_prep = sample_prep))
        if sample_prep != 'underivatized' and (n_ins := len(re.findall(r'(?:^|(?<=[\[\])]))Ins', glycan))):  # inositol has no ring oxygen, so it carries one OH more than the hexose its composition counts
            mass += n_ins * calculate_adduct_mass(_DERIV_CAP[sample_prep], mass_value = mass_value)
        # the composition takes every phosphate and methyl to sit on a hydroxyl that peracetylation would otherwise acetylate, which a pyrophosphate (Glc1PP), a phosphodiester between residues (Man1P(a1-6)Man, GlcNAc(b1-6)Man6P), a methyl ester (GlcA6Me, Neu1Me5Ac) or an N-methylamine (GlcN2Me) does not, so those are weighed from the molecule below
        if sample_prep != 'peracetylated' or {'P', 'Me'}.isdisjoint(comp):
            return mass
    # residues no composition expresses (Kdo, heptoses, furanoses, QuiNAc, ...) are weighed atom by atom from the glycan's molecule
    try:
        formula, caps = _molecule_formula(glycan, sample_prep)
    except ValueError:
        if comp:
            return mass
        raise ValueError(
            f"No valid composition could be derived from '{glycan}' (it contains components outside {sorted(_VALID_COMPONENTS)}), and it has no defined structure to weigh it by, so no mass can be calculated.") from None
    cap = _DERIV_CAP.get(sample_prep)
    modification = None if glycan.endswith('-ol') and modification == 'reduced' else modification  # the molecule of an alditol is reduced already
    weighed = calculate_adduct_mass(''.join(f'{el}{n}' for el, n in formula.items()), mass_value = mass_value) + (caps * calculate_adduct_mass(cap, mass_value = mass_value) if cap else 0) + composition_to_mass(
        {}, mass_value = mass_value, sample_prep = sample_prep, adduct = adduct, modification = modification, peptide = peptide) - composition_to_mass({}, mass_value = mass_value, sample_prep = sample_prep)
    return mass if comp and abs(weighed - mass) < 21 else weighed  # where both count the same acetyls (42 Da each), the composition's mass stands


@rescue_glycans
def get_formula(glycan: str | dict[str, int], # Glycan in any nomenclature canonicalize_iupac reads, composition (dict, or a string canonicalize_composition reads, e.g., "H5N4F1"), or elemental formula (e.g., 'C62H104N4O46')
                sample_prep: str = 'underivatized', # Sample prep: underivatized/permethylated/deuteropermethylated/peracetylated
                stem_libr: dict[str, str] | None = None, # Modified to core monosaccharide mapping
                adduct: str | None = None, # Chemical formula of a neutral adduct (e.g., "C2H4O2"); a signed formula ("-H2O") removes atoms
                modification: str | None = None, # Reducing end modification: reduced/reduced_D/2AA/2AB/procainamide/PA/APTS/aminoxyTMT/GirardP; a glycan ending in an alditol (-ol) is reduced already
                peptide: str | None = None, # Peptide carrying the glycan(s), in one-letter code with optional signed formula deltas, or named modifications (PEPTIDE_MODIFICATIONS) other than isotope-coded tags, after a residue (e.g., 'EEQYNSTYR', 'NC[+C2H3NO]SK', 'NC[Carbamidomethyl]SK'); several glycans on one peptide are passed as their total composition
                sialic_derivatization: str | tuple[str, str] | None = None # Linkage-specific sialic acid derivatization of an underivatized sample ('EE', 'EEA', 'ME', 'DMA', 'SALSA', 'GQ', 'GQ-d5', or a pair of SIALIC_PRODUCTS for a2-6 and a2-3/8/9 sialic acids); every sialic acid then needs its linkage ('H5N4E1L1', 'Neu5Ac(a2-6)...')
                ) -> str: # Elemental formula in Hill order (C, H, then alphabetical), e.g., 'C62H104N4O46' for H5N4
    """Elemental formula of a glycan, composition, or formula: the molecule glycan_to_mass weighs"""
    peptide = _peptide_deltas(peptide) if peptide else peptide
    # adds the formula f n times; a signed formula ('-H2O', '+N3') counts with its sign
    elements, add = Counter(), lambda c, f, n = 1: c.update({el: (-n if f.startswith('-') else n) * (int(k) if k else 1) for el, k in re.findall(r'([A-Z][a-z]?)(\d*)', f)})
    cap = _DERIV_CAP.get(sample_prep)
    formula = isinstance(glycan, str) and re.fullmatch(r'(?:(?:Cl|Br|Na|[CHDNOPSKFI])\d*)+', glycan)
    if formula:  # only the elements calculate_adduct_mass knows, and wherever a formula also reads as a composition, it is one (H5N4 is Hex5HexNAc4, not a nitrogen hydride)
        try:
            canonicalize_composition(glycan, strict = True)
            formula = False
        except ValueError:
            pass
    if sample_prep == 'deuteropermethylated':  # the permethylated molecule with CD3 in every cap, so three H of each cap become D, the caps counted as the carbons permethylation added (adduct and label carbons are in both)
        add(elements, get_formula(glycan, sample_prep = 'permethylated', stem_libr = stem_libr, adduct = adduct, modification = modification, peptide = peptide, sialic_derivatization = sialic_derivatization))
        caps = elements['C'] - int(re.match(r'C(\d*)', get_formula(glycan, stem_libr = stem_libr, adduct = adduct, modification = modification)).group(1) or 1)
        elements.update({'H': -3 * caps, 'D': 3 * caps})
    elif formula:
        if (sample_prep, modification, adduct, peptide, sialic_derivatization) != ('underivatized', None, None, None, None):
            raise ValueError(f"'{glycan}' is an elemental formula, which already is the whole molecule; derivatization, labels, adducts and peptides only apply to a glycan or composition.")
        add(elements, glycan)
    elif isinstance(glycan, dict) or is_composition(glycan):
        comp = canonicalize_composition(glycan)
        composition_to_mass(comp, sample_prep = sample_prep, adduct = adduct, modification = modification, peptide = peptide, sialic_derivatization = sialic_derivatization)  # raises for whatever has no mass
        if not isinstance(adduct, (str, type(None))) or (peptide and re.search(r'\[[+-]\d*\.?\d+\]', peptide)):
            raise ValueError(f"A mass in Da has no elemental formula; give the adduct ({adduct!r}) and the peptide deltas ({peptide!r}) as formulas, e.g., 'C2H4O2' or 'M[+O]'.")
        # each key's formula with the caps derivatization adds (RESIDUE_FORMULAS), mirroring composition_to_mass term by term
        i = ('underivatized', 'permethylated', 'peracetylated').index(sample_prep)
        products = (SIALIC_DERIVATIZATIONS.get(sialic_derivatization) if isinstance(sialic_derivatization, str) else tuple(sialic_derivatization)) if sialic_derivatization is not None else None
        add(elements, 'H2O')
        caps = RESIDUE_FORMULAS['red_end'][i] if i else 0
        for k, v in comp.items():
            if k.startswith(('+', '-')):
                add(elements, k, v)
                continue
            if k == 'Ac':  # an O-acetyl adds C2H2O and takes an OH derivatization would have capped, as a methyl does
                add(elements, 'C2H2O', v)
                caps -= v
                continue
            base, j = _SIALIC_BASE.get(k, (k, 0 if k == 'HexA' else None))
            add(elements, RESIDUE_FORMULAS[base][0], v)
            caps += v * RESIDUE_FORMULAS[base][i] - v * (k == 'Me') if i else 0
            if products and j is not None:  # what a linkage-specific derivatization made of the carboxyl of a sialic acid (by its linkage) or a uronic acid (as a2-6)
                for f in SIALIC_PRODUCTS[products[j]]:
                    add(elements, f, v)
        if adduct:
            add(elements, adduct)
        if modification:  # the label, and the caps composition_to_mass gives the OH a ring opening frees and the label's own acid or amides
            for part in re.findall(r'[+-]?[^+-]+', modification_formula_dict[modification]):  # a label without oxygen gains less than an O (PA is C5H6N2-O)
                add(elements, part)
            caps += (_LABEL_CAPS.get(modification, 1) if sample_prep == 'permethylated' else 1) if cap else 0
        for f in [AMINO_ACID_FORMULAS[aa] for aa in re.sub(r'\[[^\]]*\]', '', peptide or '')] + re.findall(r'\[([^\]]+)\]', peptide or ''):
            for part in re.findall(r'[+-]?[^+-]+', f):  # a delta that loses atoms is written as signed parts (deamidation, +O-HN), each added with its sign
                add(elements, part)
        if cap:
            add(elements, cap, caps)
    elif sialic_derivatization is not None:  # a sequence: every step mirrors glycan_to_mass, adding the formula of what it weighs
        sites = _carboxyl_sites(glycan)
        if unknown := [base for base, i, _, _ in sites if i is None]:
            raise ValueError(f"A linkage-specific derivatization changes each sialic acid by its linkage, which is unknown for {len(unknown)} {unknown[0]} of '{glycan}' (a2-3/6, a2-?, or the reducing end).")
        carboxyls = Counter(base if base == 'HexA' else SIALIC_KEYS[base][i] for base, i, _, _ in sites)
        for f, n in ((get_formula(glycan, sample_prep = sample_prep, stem_libr = stem_libr, adduct = adduct, modification = modification, peptide = peptide), 1),
                     (get_formula(carboxyls, sample_prep = sample_prep, sialic_derivatization = sialic_derivatization), 1), (get_formula(carboxyls, sample_prep = sample_prep), -1)):
            add(elements, f, n)
    else:
        comp = glycan_to_composition(glycan, stem_libr = stem_lib if stem_libr is None else stem_libr)
        if comp:
            add(elements, get_formula(comp, sample_prep = sample_prep, adduct = adduct, modification = modification or ('reduced' if glycan.endswith('-ol') else None), peptide = peptide))
            if inner := glycan.count('-ol') - glycan.endswith('-ol'):
                add(elements, get_formula(comp, sample_prep = sample_prep, modification = 'reduced'), inner)
                add(elements, get_formula(comp, sample_prep = sample_prep), -inner)
            if cap:
                add(elements, cap, len(re.findall(r'(?:^|(?<=[\[\])]))Ins', glycan)))
        if not comp or (sample_prep == 'peracetylated' and not {'P', 'Me'}.isdisjoint(comp)):
            try:
                atoms, caps = _molecule_formula(glycan, sample_prep)
            except ValueError:
                if not comp:
                    raise ValueError(f"No valid composition could be derived from '{glycan}' (it contains components outside {sorted(_VALID_COMPONENTS)}), and it has no defined structure to read its atoms from.") from None
                atoms = None
            if atoms is not None:
                weighed = Counter(atoms)
                if cap:
                    add(weighed, cap, caps)
                add(weighed, get_formula({}, sample_prep = sample_prep, adduct = adduct, modification = None if glycan.endswith('-ol') and modification == 'reduced' else modification, peptide = peptide))
                add(weighed, get_formula({}, sample_prep = sample_prep), -1)
                if not comp or abs(sum((weighed[el] - elements[el]) * calculate_adduct_mass(el) for el in weighed.keys() | elements.keys())) >= 21:  # where both count the same acetyls, the composition stands, as in glycan_to_mass
                    elements = weighed
    if any(v < 0 for v in elements.values()):
        raise ValueError(f"The adduct or peptide deltas remove atoms the molecule does not have: {dict(elements)}.")
    return ''.join(f'{el}{elements[el] if elements[el] > 1 else ""}' for el in ('C', 'H', *sorted(elements.keys() - {'C', 'H'})) if elements[el])


def get_unique_topologies(composition: dict[str, int] | str, # Composition dictionary of monosaccharide:count, or a string in any format canonicalize_composition reads (e.g., "H5N4F1")
                          glycan_type: str, # Glycan class: N/O/lipid/free/repeat
                          df_use: pd.DataFrame | None = None, # Custom glycan database to use for mapping
                          universal_replacers: dict[str, str] | None = None, # Base-to-specific monosaccharide mapping
                          taxonomy_rank: str = "Kingdom", # Taxonomic rank for filtering
                          taxonomy_value: str = "Animalia" # Value at taxonomy rank
                          ) -> list[str]: # List of unique base topologies
    """Get all observed unique base topologies for composition"""
    if df_use is None:
        df_use = loader.df_glycan
    if universal_replacers is None:
        universal_replacers = {}
    composition = canonicalize_composition(composition) if isinstance(composition, str) else composition  # the Composition column holds dicts, which a string never equals
    df_use = df_use[(df_use.Composition == composition) & (df_use.glycan_type == glycan_type)]
    topologies = dict.fromkeys(structure_to_basic(g) for g, taxa in zip(df_use.glycan, df_use[taxonomy_rank]) if taxonomy_value in taxa)
    return [reduce(lambda x, kv: x.replace(*kv), universal_replacers.items(), g) for g in topologies if '{' not in g]


def get_random_glycan(n: int = 1, # How many random glycans to sample
                      glycan_class: str = 'all', # Glycan class: N/O/lipid/free/repeat/all
                      kingdom: str = 'Animalia' # Taxonomic kingdom filter for choosing a subset of glycans to consider
                      ) -> str | list[str]: # Returns a random glycan or list of glycans if n > 1
    """Sample random glycans from the SugarBase database"""
    if glycan_class == "all":
        df_use = loader.df_glycan[loader.df_glycan.Kingdom.apply(lambda x: kingdom in x)].glycan.values.tolist()
    else:
        df_use = loader.df_glycan[(loader.df_glycan.glycan_type == glycan_class) & (loader.df_glycan.Kingdom.apply(lambda x: kingdom in x))].glycan.values.tolist()
    return sample(df_use, n)[0] if n == 1 else sample(df_use, n)
