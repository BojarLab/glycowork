import pandas as pd
import numpy as np
import networkx as nx
import re
import copy
from random import sample
from importlib import resources
from collections import Counter
from sklearn.cluster import DBSCAN
from functools import reduce
from typing import Dict, List, Set, Union, Optional

from glycowork.glycan_data.loader import lib, unwrap, df_glycan, Hex, dHex, HexA, HexN, HexNAc, Pen, linkages, multireplace
from glycowork.motif.processing import min_process_glycans, rescue_glycans, rescue_compositions
from glycowork.motif.graph import compare_glycans, glycan_to_nxGraph, graph_to_string

chars = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 'J':10, 'K':11,
         'L':12, 'M':13, 'N':14, 'P':15, 'Q':16, 'R':17, 'S':18, 'T':19,
         'V':20, 'W':21, 'Y':22, 'X':23, 'Z':24, 'z':25}

with resources.files("glycowork.motif").joinpath("mz_to_composition.csv").open(encoding = 'utf-8-sig') as f:
  mapping_file = pd.read_csv(f)
mass_dict = dict(zip(mapping_file.composition, mapping_file["underivatized_monoisotopic"]))


def constrain_prot(proteins: List[str], # List of protein sequences
                  libr: Optional[Dict[str, int]] = None # Dictionary mapping amino acids to indices
                 ) -> List[str]: # List of filtered protein sequences
  "Ensure only characters from library are present in proteins"
  if libr is None:
    libr = chars
  # Check whether any character is not in libr and replace it with a 'z' placeholder character
  libr_set = set(libr.keys())
  return [''.join(c if c in libr_set else 'z' for c in protein) for protein in proteins]


def prot_to_coded(proteins: List[str], # List of protein sequences
                  libr: Optional[Dict[str, int]] = None, # Dictionary mapping amino acids to indices
                  pad_len: int = 1000 # Length for padding sequences
                 ) -> List[List[int]]: # List of encoded protein sequences
  "Encode protein sequences for LectinOracle-flex"
  if libr is None:
    libr = chars
  pad_label = len(libr) - 1
  # Cut off protein sequence above pad_len
  prots = [protein[:pad_len] for protein in proteins]
  # Replace forbidden characters with 'z'
  prots = constrain_prot(prots, libr = libr)
  # Pad up to a length of pad_len
  encoded_prots = [pad_sequence(string_to_labels(prot.upper(), libr = libr),
                                max_length = pad_len, pad_label = pad_label) for prot in prots]
  return encoded_prots


def string_to_labels(character_string: str, # String to tokenize
                    libr: Optional[Dict[str, int]] = None # Dictionary mapping characters to indices
                   ) -> List[int]: # List of character indices
  "Tokenize word by indexing characters in library"
  if libr is None:
    libr = chars
  return list(map(libr.get, character_string))


def pad_sequence(seq: List[int], # Sequence to pad
                max_length: int, # Target length
                pad_label: Optional[int] = None, # Padding token value
                libr: Optional[Dict[str, int]] = None # Character library
               ) -> List[int]: # Padded sequence
  "Pad sequences to same length using padding token"
  if libr is None:
    libr = chars
  if pad_label is None:
    pad_label = len(libr)
  padding_needed = max_length - len(seq)
  if padding_needed > 0:
    seq.extend([pad_label] * padding_needed)
  return seq


def get_core(sugar: str # Monosaccharide or linkage
           ) -> str: # Core monosaccharide string
  "Retrieve core monosaccharide from modified monosaccharide"
  easy_cores = set(['dHexNAc', 'GlcNAc', 'GalNAc', 'ManNAc', 'FucNAc', 'QuiNAc', 'RhaNAc', 'GulNAc',
                'IdoNAc', 'Ins', 'MurNAc', '6dAltNAc', 'AcoNAc', 'HexA', 'GlcA', 'AltA',
                'GalA', 'ManA', 'Tyv', 'Yer', 'Abe', 'GlcfNAc', 'GalfNAc', 'ManfNAc',
                'FucfNAc', 'IdoA', 'GulA', 'LDManHep', 'DDManHep', 'DDGlcHep', 'LyxHep', 'ManHep',
                'DDAltHep', 'IdoHep', 'DLGlcHep', 'GalHep', 'ddHex', 'ddNon', 'Unknown', 'Assigned',
                'MurNGc', '6dTalNAc', '6dGul', 'AllA', 'TalA', 'AllNAc', 'TalNAc', 'Kdn', 'Pen'])
  next_cores = set(['GlcN', 'GalN', 'ManN', 'FucN', 'QuiN', 'RhaN', 'AraN', 'IdoN' 'Glcf', 'Galf', 'Manf',
                'Fucf', 'Araf', 'Lyxf', 'Xylf', '6dAltf', 'Ribf', 'Fruf', 'Apif', 'Kdof', 'Sedf',
                '6dTal', 'AltNAc', '6dAlt', 'dHex', 'HexNAc', 'dNon', '4eLeg', 'GulN', 'AltN', 'AllN', 'TalN'])
  hard_cores = set(['HexN', 'Glc', 'Gal', 'Man', 'Fuc', 'Qui', 'Rha', 'Ara', 'Oli', 'Gul', 'Lyx',
                'Xyl', 'Dha', 'Rib', 'Kdo', 'Tal', 'All', 'Pse', 'Leg', 'Asc', 'Hex',
                'Fru', 'Hex', 'Alt', 'Xluf', 'Api', 'Ko', 'Pau', 'Fus', 'Erwiniose',
                'Aco', 'Bac', 'Dig', 'Thre-ol', 'Ery-ol', 'Tag', 'Sor', 'Psi', 'Mur', 'Aci', 'Sia',
                'Par', 'Col', 'Ido'])
  for core_set in [easy_cores, next_cores, hard_cores]:
    if catch := [ele for ele in core_set if (ele in sugar)]:
      return next(iter(catch))
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


def get_modification(sugar: str # Monosaccharide or linkage
                   ) -> str: # Modification string
  "Retrieve modification from modified monosaccharide"
  core = get_core(sugar)
  modification = multireplace(sugar, {core: '', 'Neu': '', '5Ac': '', '5Gc': ''})
  return modification


def get_stem_lib(libr: Dict[str, int] # Dictionary mapping glycoletters to indices
                ) -> Dict[str, str]: # Dictionary mapping modified to core monosaccharides
  "Create mapping from modified monosaccharides to core monosaccharides"
  return {k: get_core(k) for k in libr}


stem_lib = get_stem_lib(lib)


def stemify_glycan(glycan: str, # Glycan in IUPAC-condensed format
                  stem_lib: Optional[Dict[str, str]] = None, # Modified to core monosaccharide mapping; default:created from lib
                  libr: Optional[Dict[str, int]] = None # Glycoletter to index mapping
                 ) -> str: # Stemmed glycan string
  "Remove modifications from all monosaccharides in glycan"
  if libr is None:
    libr = lib
  if stem_lib is None:
    stem_lib = get_stem_lib(libr)
  if '(' not in glycan:
    return get_core(glycan)
  clean_list = list(stem_lib.values())
  compiled_re = re.compile('^[0-9]+(-[0-9]+)+$')
  for k in reversed(list(stem_lib.keys())[:-1]):
    # For each monosaccharide, check whether it's modified
    if ((k not in clean_list) and (k in glycan) and not (k.startswith(('a', 'b', '?'))) and not (compiled_re.match(k))):
      county = 0
      # Go at it until all modifications are stemified
      while ((k in glycan) and (sum(1 for s in clean_list if k in s) <= 1)) and county < 5:
        county += 1
        rindex_pos = glycan.rindex('(')
        glycan_start = glycan[:rindex_pos]
        glycan_part = glycan_start
        # Narrow it down to the offending monosaccharide
        if k in glycan_start:
          index_pos = glycan_start.index(k)
          cut = glycan_start[index_pos:]
          try:
            cut = cut.split('(', 1)[0]
          except:
            pass
          # Replace offending monosaccharide with stemified monosaccharide
          if cut not in clean_list:
            glycan_part = glycan_start[:index_pos] + stem_lib[k]
        # Check to see whether there is anything after the modification that should be appended
        try:
          glycan_mid = glycan_start[index_pos + len(k):]
          if ((cut not in clean_list) and (len(glycan_mid) > 0)):
            glycan_part = glycan_part + glycan_mid
        except:
          pass
        # Handling the reducing end
        glycan_end = glycan[rindex_pos:]
        if k in glycan_end:
          filt = ']' if ']' in glycan_end else ')'
          cut = glycan_end[glycan_end.index(filt)+1:]
          if cut not in clean_list:
            glycan_end = glycan_end[:glycan_end.index(filt)+1] + stem_lib[k]
        glycan = glycan_part + glycan_end
  return glycan


def stemify_dataset(df: pd.DataFrame, # DataFrame with glycan column
                   stem_lib: Optional[Dict[str, str]] = None, # Modified to core monosaccharide mapping; default:created from lib
                   libr: Optional[Dict[str, int]] = None, # Glycoletter to index mapping
                   glycan_col_name: str = 'glycan', # Column name for glycans
                   rarity_filter: int = 1 # Minimum occurrences to keep modification
                  ) -> pd.DataFrame: # DataFrame with stemified glycans
  "Remove monosaccharide modifications from all glycans in dataset"
  if libr is None:
    libr = lib
  if stem_lib is None:
    stem_lib = get_stem_lib(libr)
  # Get pool of monosaccharides, decide which one to stemify based on rarity
  pool = unwrap(min_process_glycans(df[glycan_col_name].tolist()))
  pool_count = Counter(pool)
  stem_lib.update({k: k for k, v in pool_count.items() if v > rarity_filter})
  # Stemify all offending monosaccharides
  df_out = copy.deepcopy(df)
  df_out[glycan_col_name] = df_out[glycan_col_name].apply(lambda x: stemify_glycan(x, stem_lib = stem_lib, libr = libr))
  return df_out


def mz_to_composition(mz_value: float, # m/z value from mass spec
                     mode: str = 'negative', # MS mode: positive/negative
                     mass_value: str = 'monoisotopic', # Mass type: monoisotopic/average
                     reduced: bool = False, # Whether glycans are reduced
                     sample_prep: str = 'underivatized', # Sample preparation method: underivatized/permethylated/peracetylated
                     mass_tolerance: float = 0.5, # Mass tolerance for matching
                     kingdom: str = 'Animalia', # Taxonomic kingdom filter for choosing a subset of glycans to consider
                     glycan_class: str = 'all', # Glycan class: N/O/lipid/free/all
                     df_use: Optional[pd.DataFrame] = None, # Custom glycan database
                     filter_out: Optional[Set[str]] = None, # Monosaccharides to ignore during composition finding
                     extras: List[str] = ["doubly_charged"], # Additional operations: adduct/doubly_charged
                     adduct: Optional[str] = None # Chemical formula of adduct that contributes to m/z, e.g., "C2H4O2"
                    ) -> List[Dict[str, int]]: # List of matching compositions
  "Map m/z value to matching monosaccharide composition"
  if df_use is None:
    if glycan_class == "all":
      df_use = df_glycan[df_glycan.Kingdom.apply(lambda x: kingdom in x)]
    else:
      df_use = df_glycan[(df_glycan.glycan_type == glycan_class) & (df_glycan.Kingdom.apply(lambda x: kingdom in x))]
  if filter_out is None:
    filter_out = set()
  if adduct:
    mz_value -= calculate_adduct_mass(adduct, mass_value)
  adduct_mass = mass_dict['Acetate'] if mode == 'negative' else mass_dict['Na+']
  if reduced:
    mz_value -= 1.0078
  multiplier = 1 if mode == 'negative' else -1
  comp_pool = [dict(t) for t in {tuple(d.items()) for d in df_use.Composition}]
  out = []
  cache = {}
  # Iterate over the composition pool
  for comp in comp_pool:
    mass = composition_to_mass(comp, mass_value = mass_value, sample_prep = sample_prep)
    cache[mass] = comp
    if abs(mass - mz_value) < mass_tolerance:
      if not filter_out.intersection(comp.keys()):
        return [comp]
  if "adduct" in extras:
    # Check for matches including the adduct mass
    for mass, comp in cache.items():
      if abs(mass + adduct_mass - mz_value) < mass_tolerance:
        if not filter_out.intersection(comp.keys()):
          return [comp]
  if "doubly_charged" in extras:
    # If no matches are found, consider a double charge scenario
    mz_value = (mz_value + 0.5*multiplier)*2 + (1.0078 if reduced else 0)
    for mass, comp in cache.items():
      if abs(mass - mz_value) < mass_tolerance:
        if not filter_out.intersection(comp.keys()):
          return [comp]
  return out


@rescue_compositions
def match_composition_relaxed(composition: Dict[str, int], # Dictionary indicating composition (e.g. {"dHex": 1, "Hex": 1, "HexNAc": 1})
                            glycan_class: str = 'N', # Glycan class: N/O/lipid/free
                            kingdom: str = 'Animalia', # Taxonomic kingdom filter for choosing a subset of glycans to consider
                            df_use: Optional[pd.DataFrame] = None, # Custom glycan database
                            reducing_end: Optional[str] = None # Reducing end specification
                           ) -> List[str]: # List of matching glycans
  "Map coarse-grained composition to matching glycans"
  if df_use is None:
    df_use = df_glycan[(df_glycan.glycan_type == glycan_class) & (df_glycan.Kingdom.apply(lambda x: kingdom in x))]
  # Subset for glycans with the right number of monosaccharides
  comp_count = sum(composition.values())
  len_distr = [len(k) - (len(k)-1)/2 for k in min_process_glycans(df_use.glycan.values.tolist())]
  idx = [i for i, length in enumerate(len_distr) if length == comp_count]
  output_list = df_use.iloc[idx, :].glycan.values.tolist()
  output_compositions = [glycan_to_composition(k) for k in output_list]
  return [glycan for glycan, glycan_comp in zip(output_list, output_compositions) if glycan_comp == composition]


def condense_composition_matching(matched_composition: List[str] # List of matching glycans
                               ) -> List[str]: # Minimal list of representative glycans
  "Find minimum set of glycans characterizing matched composition"
  # Establish glycan equality given the wildcards
  match_matrix = pd.DataFrame(
    [[compare_glycans(k, j)
      for j in matched_composition] for k in matched_composition],
    columns = matched_composition
    )
  # Cluster glycans by pairwise equality (given the wildcards)
  clustering = DBSCAN(eps = 1, min_samples = 1).fit(match_matrix)
  num_clusters = len(set(clustering.labels_))
  sum_glycans = []
  # For each cluster, get the most well-defined glycan and return it
  for k in range(num_clusters):
    cluster_glycans = np.array(matched_composition)[np.where(clustering.labels_ == k)[0]].tolist()
    county = np.array([j.count("?") for j in cluster_glycans])
    idx = np.where(county == county.min())[0]
    if len(idx) == 1:
      sum_glycans.append(cluster_glycans[idx[0]])
    else:
      sum_glycans.extend([cluster_glycans[j] for j in idx])
  return sum_glycans


@rescue_compositions
def compositions_to_structures(composition_list: List[Dict[str, int]], # List of compositions like {'Hex': 1, 'HexNAc': 1}
                             glycan_class: str = 'N', # Glycan class: N/O/lipid/free
                             kingdom: str = 'Animalia', # Taxonomic kingdom filter for choosing a subset of glycans to consider
                             abundances: Optional[pd.DataFrame] = None, # Sample abundances matrix
                             df_use: Optional[pd.DataFrame] = None, # Custom glycan database
                             verbose: bool = False # Whether to print non-matching compositions
                            ) -> pd.DataFrame: # DataFrame of structures x intensities
  "Map compositions to structures, supporting accompanying relative intensities"
  if df_use is None:
    df_use = df_glycan[(df_glycan.glycan_type == glycan_class) & (df_glycan.Kingdom.apply(lambda x: kingdom in x))]
  if abundances is None:
    abundances = pd.DataFrame([range(len(composition_list))]*2).T
  abundances_values = abundances.iloc[:, 1:].values.tolist()
  df_out = []
  not_matched_count = 0
  not_matched_list = []
  for k, comp in enumerate(composition_list):
    # For each composition, map it to potential structures
    matched = match_composition_relaxed(comp, glycan_class = glycan_class,
                                        kingdom = kingdom, df_use = df_use)
    # If multiple structure matches, try to condense them by wildcard clustering
    if matched:
      condensed = condense_composition_matching(matched)
      matched_data = [abundances_values[k]]*len(condensed)
      df_out.extend([[condensed[ele]] + matched_data[ele] for ele in range(len(condensed))])
    else:
      not_matched_count += 1
      if verbose:
        not_matched_list.append(comp)
  if df_out:
    df_out = pd.DataFrame(df_out, columns = ['glycan'] + ['abundance'] * (abundances.shape[1] - 1))
  print(f"{not_matched_count} compositions could not be matched. Run with verbose = True to see which compositions.")
  if verbose:
    print(not_matched_list)
  return df_out if isinstance(df_out, pd.DataFrame) else pd.DataFrame()


def mz_to_structures(mz_list: List[float], # List of precursor masses
                    glycan_class: str, # Glycan class: N/O/lipid/free
                    kingdom: str = 'Animalia', # Taxonomic kingdom filter for choosing a subset of glycans to consider
                    abundances: Optional[pd.DataFrame] = None, # Sample abundances matrix
                    mode: str = 'negative', # MS mode: positive/negative
                    mass_value: str = 'monoisotopic', # Mass type: monoisotopic/average
                    sample_prep: str = 'underivatized', # Sample prep: underivatized/permethylated/peracetylated
                    mass_tolerance: float = 0.5, # Mass tolerance for matching
                    reduced: bool = False, # Whether glycans are reduced
                    df_use: Optional[pd.DataFrame] = None, # Custom glycan database
                    filter_out: Optional[Set[str]] = None, # Monosaccharides to ignore
                    verbose: bool = False # Whether to print non-matching compositions
                   ) -> Union[pd.DataFrame, List]: # DataFrame of structures x intensities or empty list
  "Map precursor masses to structures, supporting accompanying relative intensities"
  if df_use is None:
    df_use = df_glycan[(df_glycan.glycan_type == glycan_class) & (df_glycan.Kingdom.apply(lambda x: kingdom in x))]
  if filter_out is None:
    filter_out = set()
  if abundances is None:
    abundances = pd.DataFrame([range(len(mz_list))]*2).T
  # Check glycan class
  if glycan_class not in {'N', 'O', 'free', 'lipid'}:
    print("Not a valid class for mz_to_composition; currently N/O/free/lipid matching is supported. For everything else run compositions_to_structures separately.")
  # Map each m/z value to potential compositions
  compositions = [mz_to_composition(mz, mode = mode, mass_value = mass_value, reduced = reduced, sample_prep = sample_prep,
                                    mass_tolerance = mass_tolerance, kingdom = kingdom, glycan_class = glycan_class,
                                    df_use = df_use, filter_out = filter_out) for mz in mz_list]
  # Map each of these potential compositions to potential structures
  out_structures = []
  for m, comp in enumerate(compositions):
    out_structures.append(compositions_to_structures(comp, glycan_class = glycan_class,
                                              abundances = abundances.iloc[[m]], kingdom = kingdom, df_use = df_use, verbose = verbose))
  return pd.concat(out_structures, axis = 0).reset_index(drop = True) if out_structures else []


def mask_rare_glycoletters(glycans: List[str], # List of IUPAC-condensed glycans
                          thresh_monosaccharides: Optional[int] = None, # Threshold for rare monosaccharides (default: 0.001*len(glycans))
                          thresh_linkages: Optional[int] = None # Threshold for rare linkages (default: 0.03*len(glycans))
                         ) -> List[str]: # List of glycans with masked rare elements
  "Mask rare monosaccharides and linkages in glycans"
  # Get rarity thresholds
  if thresh_monosaccharides is None:
    thresh_monosaccharides = int(np.ceil(0.001*len(glycans)))
  if thresh_linkages is None:
    thresh_linkages = int(np.ceil(0.03*len(glycans)))
  rares = unwrap(min_process_glycans(glycans))
  rare_linkages, rare_monosaccharides = [], []
  # Sort monosaccharides and linkages into different bins
  for x in rares:
    (rare_monosaccharides, rare_linkages)[x in linkages].append(x)
  rare_elements = [rare_monosaccharides, rare_linkages]
  thresholds = [thresh_monosaccharides, thresh_linkages]
  # Establish which ones are considered to be rare
  rare_dict = [
    {x: 'Monosaccharide' if i == 0 else '?1-?' if x[1] == '1' else '?2-?'
     for x, count in Counter(rare_elements[i]).items() if count <= thresholds[i]}
    for i in range(2)
    ]
  out = []
  # For each glycan, check whether they have rare monosaccharides/linkages and mask them
  for glycan in glycans:
    for k, v in rare_dict[0].items():
      # Replace rare monosaccharides
      if k in glycan and f'-{k}' not in glycan:
        glycan = glycan.replace(f'{k}(', f'{v}(')
        if glycan.endswith(k):
          glycan = glycan[:-len(k)] + v
      # Replace rare linkages
      for k, v in rare_dict[1].items():
        glycan = glycan.replace(k, v)
    out.append(glycan)
  return out


def map_to_basic(glycoletter: str, # Monosaccharide or linkage
                 obfuscate_ptm: bool = True # Whether to remove PTM position specificity
                ) -> str: # Base monosaccharide/linkage
  "Map monosaccharide/linkage to corresponding base form"
  conditions = [(Hex, 'Hex'), (dHex, 'dHex'), (HexA, 'HexA'), (HexN, 'HexN'), (HexNAc, 'HexNAc'), (Pen, 'Pen'), (linkages, '?1-?')]
  for cond, ret in conditions:
    if glycoletter in cond:
      return ret
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
  "Convert glycan structure to base topology"
  if glycan.endswith('-ol'):
    glycan = glycan[:-3]
  if '(' not in glycan:
    return map_to_basic(glycan)
  ggraph = glycan_to_nxGraph(glycan)
  nodeDict = dict(ggraph.nodes(data = True))
  nx.set_node_attributes(ggraph, {k: map_to_basic(nodeDict[k]['string_labels']) for k in ggraph.nodes}, 'string_labels')
  return graph_to_string(ggraph)


@rescue_glycans
def glycan_to_composition(glycan: str, # Glycan in IUPAC-condensed format
                         stem_libr: Optional[Dict[str, str]] = None # Modified to core monosaccharide mapping; default: created from lib
                        ) -> Dict[str, int]: # Dictionary of monosaccharide counts
  "Map glycan to its composition"
  if stem_libr is None:
    stem_libr = stem_lib
  SPECIAL_MODS = {
    '1,7lactone': {'replacement': '', 'diff_moiety': '-H2O'},
    'Az': {'replacement': 'AcH2O', 'diff_moiety': '+N3'},
    'AcH2O': {'replacement': 'Ac', 'diff_moiety': '-OH'}
    # Add other special modifications here in the format:
    # 'modification': {'replacement': 'what to replace with', 'diff_moiety': chemical formula, sign indicating loss/gain}
  }
  VALID_COMPONENTS = {'Hex', 'dHex', 'HexNAc', 'HexN', 'HexA', 'Neu5Ac', 'Neu5Gc', 'Kdn',
                     'Pen', 'Me', 'S', 'P', 'PCho', 'PEtN', 'Ac', '-H2O', '+N3', '-OH'}
  glycan = glycan.replace('{', '').replace('}', '') if '{' in glycan else glycan
  diff_moieties = Counter()
  for mod, info in SPECIAL_MODS.items():
    while mod in glycan:
      diff_moieties[info['diff_moiety']] += 1
      glycan = glycan.replace(mod, info['replacement'])
  composition = Counter(sorted([map_to_basic(stem_libr[k]) for k in min_process_glycans([glycan])[0]]))
  composition.update(diff_moieties)
  for mod in ('Me', 'S', 'P', 'PCho', 'PEtN'):
    if mod in glycan:
      composition[mod] = glycan.count(mod)
  if 'PCho' in glycan or 'PEtN' in glycan:
    composition.pop('P', None)
  ac_mods = ('OAc', '2Ac', '3Ac', '4Ac', '6Ac', '7Ac', '9Ac')
  if any(mod in glycan for mod in ac_mods):
    composition['Ac'] = sum(glycan.count(mod) for mod in ac_mods)
  composition.pop('?1-?', None)
  return dict(composition) if all(k in VALID_COMPONENTS for k in composition) else {}


def calculate_adduct_mass(formula: str, # Chemical formula of adduct (e.g., "C2H4O2", "-H2O", "+Na")
                         mass_value: str = 'monoisotopic', # Mass type: monoisotopic/average
                         enforce_sign: bool = False # If True, returns 0 for unsigned formulas
                        ) -> float: # Formula mass
  "Calculate mass of adduct from chemical formula, including signed formulas"
  # Handle sign if present
  sign = 1
  if formula.startswith(('+', '-')):
    sign = -1 if formula.startswith('-') else 1
    formula = formula[1:]
  elif enforce_sign:
    return 0
  element_masses = {
    'monoisotopic': {'C': 12.0000, 'H': 1.0078, 'O': 15.9949, 'N': 14.0031},
    'average': {'C': 12.0107, 'H': 1.00794, 'O': 15.9994, 'N': 14.0067}
  }
  mass = 0
  element_count = {'C': 0, 'H': 0, 'O': 0, 'N': 0}
  current_element = ''
  current_count = ''
  for char in formula:
    if char.isalpha():
      if current_element:
        element_count[current_element] += int(current_count) if current_count else 1
      current_element = char
      current_count = ''
    elif char.isdigit():
      current_count += char
  if current_element:
    element_count[current_element] += int(current_count) if current_count else 1
  for element, count in element_count.items():
    mass += element_masses[mass_value][element] * count
  return sign * mass


@rescue_compositions
def composition_to_mass(dict_comp_in: Dict[str, int], # Composition dictionary of monosaccharide:count
                       mass_value: str = 'monoisotopic', # Mass type: monoisotopic/average
                       sample_prep: str = 'underivatized', # Sample prep: underivatized/permethylated/peracetylated
                       adduct: Optional[Union[str, float]] = None # Chemical formula of adduct (e.g., "C2H4O2") OR its exact mass in Da
                      ) -> float: # Theoretical mass
  "Calculate theoretical mass from composition"
  dict_comp = dict_comp_in.copy()
  mass_key = f"{sample_prep}_{mass_value}"
  mass_dict_in = mass_dict if mass_key == "underivatized_monoisotopic" else dict(zip(mapping_file.composition, mapping_file[mass_key]))
  for old_key, new_key in {'S': 'Sulphate', 'P': 'Phosphate', 'Me': 'Methyl', 'Ac': 'Acetate'}.items():
    if old_key in dict_comp:
      dict_comp[new_key] = dict_comp.pop(old_key)
  total_mass = sum(v * (mass_dict_in.get(k) or calculate_adduct_mass(k, mass_value, enforce_sign = True))
                   for k, v in dict_comp.items()) + mass_dict_in['red_end']
  if adduct:
    total_mass += calculate_adduct_mass(adduct, mass_value) if isinstance(adduct, str) else adduct
  return total_mass


def glycan_to_mass(glycan: str, # Glycan in IUPAC-condensed format
                   mass_value: str = 'monoisotopic', # Mass type: monoisotopic/average
                   sample_prep: str = 'underivatized', # Sample prep: underivatized/permethylated/peracetylated
                   stem_libr: Optional[Dict[str, str]] = None, # Modified to core monosaccharide mapping
                   adduct: Optional[Union[str, float]] = None # Chemical formula of adduct (e.g., "C2H4O2") OR its exact mass in Da
                  ) -> float: # Theoretical mass
  "Calculate theoretical mass from glycan"
  if stem_libr is None:
    stem_libr = stem_lib
  comp = glycan_to_composition(glycan, stem_libr = stem_libr)
  return composition_to_mass(comp, mass_value = mass_value, sample_prep = sample_prep, adduct = adduct)


@rescue_compositions
def get_unique_topologies(composition: Dict[str, int], # Composition dictionary of monosaccharide:count
                         glycan_type: str, # Glycan class: N/O/lipid/free/repeat
                         df_use: Optional[pd.DataFrame] = None, # Custom glycan database to use for mapping
                         universal_replacers: Optional[Dict[str, str]] = None, # Base-to-specific monosaccharide mapping
                         taxonomy_rank: str = "Kingdom", # Taxonomic rank for filtering
                         taxonomy_value: str = "Animalia" # Value at taxonomy rank
                        ) -> List[str]: # List of unique base topologies
  "Get all observed unique base topologies for composition"
  if df_use is None:
    df_use = df_glycan
  if universal_replacers is None:
    universal_replacers = {}
  df_use = df_use[df_use.Composition == composition]
  df_use = df_use[df_use.glycan_type == glycan_type]
  df_use = df_use[df_use[taxonomy_rank].apply(lambda x: taxonomy_value in x)].glycan.values
  df_use = list(set([structure_to_basic(k) for k in df_use]))
  return [reduce(lambda x, kv: x.replace(*kv), universal_replacers.items(), g) for g in df_use if '{' not in g]


def get_random_glycan(n: int = 1, # How many random glycans to sample
                      glycan_class: str = 'all', # Glycan class: N/O/lipid/free/repeat/all
                      kingdom: str = 'Animalia' # Taxonomic kingdom filter for choosing a subset of glycans to consider
                      ) -> Union[str, List[str]]: # Returns a random glycan or list of glycans if n > 1
  if glycan_class == "all":
    df_use = df_glycan[df_glycan.Kingdom.apply(lambda x: kingdom in x)].glycan.values.tolist()
  else:
    df_use = df_glycan[(df_glycan.glycan_type == glycan_class) & (df_glycan.Kingdom.apply(lambda x: kingdom in x))].glycan.values.tolist()
  return sample(df_use, n)[0] if n == 1 else sample(df_use, n)
