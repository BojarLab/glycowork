import pandas as pd
import copy
import json
import re
from random import choice
from functools import wraps
from collections import defaultdict
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Set, Union, Optional, Callable, Tuple, Generator
from glycowork.glycan_data.loader import (unwrap, multireplace, df_glycan,
                                          find_nth, find_nth_reverse, lib, HexOS, HexNAcOS,
                                          linkages, Hex, HexNAc, dHex, Sia, HexA, Pen)

mapping_path = Path(__file__).parent / "common_names.json"
with open(mapping_path) as f:
  GLYCAN_MAPPINGS = json.load(f)
mapping_path = Path(__file__).parent / "wurcs_tokens.json"
with open(mapping_path) as f:
  monosaccharide_mapping = json.load(f)

# for canonicalize_iupac
replace_dic = {'αα': 'a', 'Nac': 'NAc', 'AC': 'Ac', 'Nc': 'NAc', 'Nue': 'Neu', 'NeuAc': 'Neu5Ac', 'NeuNAc': 'Neu5Ac', 'NeuGc': 'Neu5Gc',
                  'α': 'a', 'β': 'b', 'N(Gc)': 'NGc', 'GL': 'Gl', 'GaN': 'GalN', '(9Ac)': '9Ac', '5,9Ac2': '5Ac9Ac', '4,5Ac2': '4Ac5Ac', 'Talp': 'Tal',
                 'KDN': 'Kdn', 'OSO3': 'S', '-O-Su-': 'S', '(S)': 'S', 'SO3-': 'S', 'SO3(-)': 'S', 'H2PO3': 'P', '(P)': 'P', 'L-6dGal': 'Fuc', 'Hepp': 'Hep',
                 '–': '-', ' ': '', 'ß': 'b', '.': '', '((': '(', '))': ')', '→': '-', '*': '', 'Ga(': 'Gal(', 'aa': 'a', 'bb': 'b', 'Pc': 'PCho', 'Rhap': 'Rha', 'Quip': 'Qui',
                 'Glcp': 'Glc', 'Galp': 'Gal', 'Manp': 'Man', 'Fucp': 'Fuc', 'Neup': 'Neu', 'a?': 'a1', 'Kdop': 'Kdo', 'Abep': 'Abe',
                 '5Ac4Ac': '4Ac5Ac', '(-)': '(?1-?)', '(?-?)': '(?1-?)', '?-?)': '1-?)', '5ac': '5Ac', '-_': '-?'}
CANONICALIZE = re.compile('|'.join(map(re.escape, list(replace_dic.keys()))))


def min_process_glycans(glycan_list: List[str] # List of glycans in IUPAC-condensed format
                      ) -> List[List[str]]: # List of glycoletter lists
  "Convert list of glycans into a nested lists of glycoletters"
  return [k.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace(')', '(').split('(') for k in glycan_list]


def get_lib(glycan_list: List[str] # List of IUPAC-condensed glycan sequences
           ) -> Dict[str, int]: # Dictionary of glycoletter:index mappings
  "Returns dictionary mapping glycoletters to indices"
  # Convert to glycoletters & flatten & get unique vocab
  lib = sorted(set(unwrap(min_process_glycans(set(glycan_list)))))
  # Convert to dict
  return {k: i for i, k in enumerate(lib)}


def expand_lib(libr_in: Dict[str, int], # Existing dictionary of glycoletter:index
              glycan_list: List[str] # List of IUPAC-condensed glycan sequences
             ) -> Dict[str, int]: # Updated dictionary with new glycoletters
  "Updates libr with newly introduced glycoletters"
  libr = dict(libr_in)
  new_libr = get_lib(glycan_list)
  offset = len(libr)
  for k, v in new_libr.items():
    if k not in libr:
      libr[k] = v + offset
  return libr


def in_lib(glycan: str, # Glycan in IUPAC-condensed nomenclature
           libr: Dict[str, int] # Dictionary of glycoletter:index
          ) -> bool: # True if all glycoletters are in libr
  "Checks whether all glycoletters of glycan are in libr"
  glycan = min_process_glycans([glycan])[0]
  return set(glycan).issubset(libr.keys())


def get_possible_linkages(wildcard: str, # Pattern to match, ? can be wildcard
                         linkage_list: List[str] = linkages # List of linkages to search
                        ) -> Set[str]: # Matching linkages
  "Retrieves all linkages that match a given wildcard pattern"
  if '/' in wildcard:
    prefix = wildcard[:wildcard.index('-')].replace('?', '[ab?]')
    numbers = re.search(r'-(\d+(?:/\d+)*)', wildcard).group(1).split('/')
    base_pattern = f"{prefix}-({('|'.join(numbers))}|\\?)"
    return {l for l in linkage_list if re.compile(f'^{base_pattern}$').fullmatch(l)} | \
           ({f"{wildcard[:wildcard.index('-')]}-{'/'.join(sorted(combo))}"
             for combo in combinations(numbers, r = 2)} if len(numbers) > 2 else set())
  pattern = f"^{wildcard.replace('?', '[ab1-9?]')}$"
  return {l for l in linkage_list if re.compile(pattern).fullmatch(l)}


def get_possible_monosaccharides(wildcard: str # Monosaccharide type; options: Hex, HexNAc, dHex, Sia, HexA, Pen, HexOS, HexNAcOS
                               ) -> Set[str]: # Matching monosaccharides
  "Retrieves all matching common monosaccharides of a type"
  wildcard_dict = {'Hex': Hex, 'HexNAc': HexNAc, 'dHex': dHex, 'Sia': Sia, 'HexA': HexA, 'Pen': Pen,
                   'HexOS': HexOS, 'HexNAcOS': HexNAcOS,
                   'Monosaccharide': set().union(*[Hex, HexOS, HexNAc, HexNAcOS, dHex, Sia, HexA, Pen])}
  return wildcard_dict.get(wildcard, {})


def de_wildcard_glycoletter(glycoletter: str # Monosaccharide or linkage with wildcards
                          ) -> str: # Specific glycoletter instance
  "Retrieves a random specified instance of a general type (e.g., 'Gal' for 'Hex')"
  if '?' in glycoletter or '/' in glycoletter:
    return choice(list(get_possible_linkages(glycoletter)))
  elif monos := get_possible_monosaccharides(glycoletter):
    return choice(list(monos))
  else:
    return glycoletter


def bracket_removal(glycan_part: str # Residual part of glycan from glycan_to_graph
                  ) -> str: # Glycan part without interfering branches
  "Iteratively removes (nested) branches between start and end of glycan_part"
  regex = re.compile(r'\[[^\[\]]+\]')
  while regex.search(glycan_part):
    glycan_part = regex.sub('', glycan_part)
  return glycan_part


def presence_to_matrix(df: pd.DataFrame, # DataFrame with glycan occurrence
                      glycan_col_name: str = 'glycan', # Column name for glycans
                      label_col_name: str = 'Species' # Column name for labels
                     ) -> pd.DataFrame: # Matrix with labels as rows and glycan occurrences as columns
  "Converts a dataframe with glycan occurrence to absence/presence matrix"
  # Create a grouped dataframe where we count the occurrences of each glycan in each species group
  grouped_df = df.groupby([label_col_name, glycan_col_name]).size().unstack(fill_value = 0)
  return grouped_df.sort_index().sort_index(axis = 1)


def get_matching_indices(
    line: str, # Input string to search
    opendelim: str = '(', # Opening delimiter
    closedelim: str = ')' # Closing delimiter
    ) -> Union[Generator[Tuple[int, int, int], None, None], Optional[List[Tuple[int, int]]]]: # Yields (start pos, end pos, nesting depth)
  "Finds matching pairs of delimiters in a string, handling nested pairs and returning positions and depth;ref: https://stackoverflow.com/questions/5454322/python-how-to-match-nested-parentheses-with-regex"""
  stack = []
  pattern = r'[\[\]]' if opendelim == '[' else f'[{re.escape(opendelim)}{re.escape(closedelim)}]'
  for m in re.finditer(pattern, line):
      pos = m.start()
      if pos > 0 and line[pos-1] == '\\':
          # Skip escape sequence
          continue
      c = line[pos]
      if c == opendelim:
          stack.append(pos)
      elif c == closedelim and stack:
        yield (stack.pop() + 1, pos, len(stack))
      elif c == closedelim:
          print(f"Encountered extraneous closing quote at pos {pos}: '{line[pos:]}'")
  if stack:
    print(f"Unmatched opening delimiters at positions: {[p for p in stack]}")


def enforce_class(glycan: str, # Glycan in IUPAC-condensed nomenclature
                 glycan_class: str, # Glycan class (O, N, free, or lipid)
                 conf: Optional[float] = None, # Prediction confidence to override class
                 extra_thresh: float = 0.3 # Threshold to override class
                ) -> bool: # True if glycan is in glycan class
  "Determines whether glycan belongs to a specified class"
  pools = {
    'O': 'GalNAc|GalNAcOS|GalNAc[46]S|Man|Fuc|Gal|GlcNAc|GlcNAcOS|GlcNAc6S',
    'N': 'GlcNAc',
    'free': 'Glc|GlcOS|Glc3S|GlcNAc|GlcNAcOS|Gal|GalOS|Gal3S|Ins',
    'lipid': 'Glc|GlcOS|Glc3S|GlcNAc|GlcNAcOS|Gal|GalOS|Gal3S|Ins'
    }
  if glycan_class not in pools:
    return False
  glycan = glycan[:-3] if glycan.endswith('-ol') else glycan[:-4] if glycan.endswith('1Cer') else glycan
  truth = bool(re.search(f"({pools[glycan_class]})$", glycan))
  if truth and glycan_class in {'free', 'lipid', 'O'}:
    truth = not re.search(r'(GlcNAc\(b1-4\)GlcNAc|\[Fuc\(a1-6\)]GlcNAc)$', glycan)
  return conf > extra_thresh if not truth and conf is not None else truth


def get_class(glycan: str # Glycan in IUPAC-condensed nomenclature
            ) -> str: # Glycan class (repeat, O, N, free, lipid, lipid/free, or empty)
  "Determines glycan class"
  if glycan.startswith('['):
    return 'repeat'
  if glycan.endswith('-ol'):
    return 'free'
  if glycan.endswith(('1Cer', 'Ins')):
    return 'lipid'
  if glycan.endswith(('GlcNAc(b1-4)GlcNAc', '[Fuc(a1-6)]GlcNAc', '[Fuc(a1-3)]GlcNAc')):
    return 'N'
  if re.search(r'(GalNAc|GalNAcOS|GalNAc[46]S|Man|Fuc|Gal|GlcNAc|GlcNAcOS|GlcNAc6S)$', glycan):
    return 'O'
  if re.search(r'(Gal\(b1-4\)Glc|Gal\(b1-4\)\[Fuc\(a1-3\)\]Glc|Gal[36O]S\(b1-4\)Glc|Gal[36O]S\(b1-4\)\[Fuc\(a1-3\)\]Glc|Gal\(b1-4\)Glc[36O]S)$', glycan):
    return 'lipid/free'
  return ''


def canonicalize_composition(comp: str # Composition in Hex5HexNAc4Fuc1Neu5Ac2 or H5N4F1A2 format
                          ) -> Dict[str, int]: # Dictionary of monosaccharide:count
  "Converts composition from any common format to standardized dictionary"
  if '_' in comp:
    values = comp.split('_')
    temp = {"Hex": int(values[0]), "HexNAc": int(values[1]), "Neu5Ac": int(values[2]), "dHex": int(values[3])}
    return {k: v for k, v in temp.items() if v}
  elif comp.isdigit():
    temp = {"Hex": int(comp[0]), "HexNAc": int(comp[1]), "Neu5Ac": int(comp[2]), "dHex": int(comp[3])}
    return {k: v for k, v in temp.items() if v}
  elif comp[0].isdigit():
    comp = comp.replace(' ', '')
    if len(comp) < 5:
      temp = {"Hex": int(comp[0]), "HexNAc": int(comp[1]), "Neu5Ac": int(comp[2]), "dHex": int(comp[3])}
    else:
      temp = {"Hex": int(comp[0]), "HexNAc": int(comp[1]), "Neu5Ac": int(comp[2]), "Neu5Gc": int(comp[3]), "dHex": int(comp[4])}
    return {k: v for k, v in temp.items() if v}
  comp_dict = {}
  i = 0
  replace_dic = {"Neu5Ac": "NeuAc", "Neu5Gc": "NeuGc", '(': '', ')': '', ' ': '', '+': ''}
  comp = multireplace(comp, replace_dic)
  n = len(comp)
  # Dictionary to map letter codes to full names
  code_to_name = {'H': 'Hex', 'N': 'HexNAc', 'F': 'dHex', 'A': 'Neu5Ac', 'G': 'Neu5Gc', 'NeuGc': 'Neu5Gc', 'Gc': 'Neu5Gc',
                  'Hex': 'Hex', 'HexNAc': 'HexNAc', 'HexAc': 'HexNAc', 'Fuc': 'dHex', 'dHex': 'dHex', 'deHex': 'dHex', 'HexA': 'HexA',
                  'Neu5Ac': 'Neu5Ac', 'NeuAc': 'Neu5Ac', 'NeuNAc': 'Neu5Ac', 'HexNac': 'HexNAc', 'HexNc': 'HexNAc',
                  'Su': 'S', 's': 'S', 'Sul': 'S', 'p': 'P', 'Pent': 'Pen', 'Xyl': 'Pen', 'Man': 'Hex', 'GlcNAc': 'HexNAc', 'Deoxyhexose': 'dHex'}
  while i < n:
    # Code initialization
    code = ''
    # Read until you hit a number or the end of the string
    while i < n and not comp[i].isdigit():
      code += comp[i]
      i += 1
    # Initialize a variable to hold the number of occurrences
    num = 0
    # Parse the number following the code
    while i < n and comp[i].isdigit():
      num = num * 10 + int(comp[i])
      i += 1
    # Map code to full name and store in dictionary
    name = code_to_name.get(code, code)
    if name in comp_dict:
      comp_dict[name] += num
    else:
      comp_dict[name] = num
  return comp_dict


def IUPAC_to_SMILES(glycan_list: List[str] # List of IUPAC-condensed glycans
                   ) -> List[str]: # List of corresponding SMILES strings
  "Convert list of IUPAC-condensed glycans to isomeric SMILES using GlyLES"
  try:
    from glyles import convert
  except ImportError:
    raise ImportError("You must install the 'chem' dependencies to use this feature. Try 'pip install glycowork[chem]'.")
  if not isinstance(glycan_list, list):
    raise TypeError("Input must be a list")
  return [convert(g)[0][1] for g in glycan_list]


iupac_to_smiles = IUPAC_to_SMILES


def linearcode_to_iupac(linearcode: str # Glycan in LinearCode format
                      ) -> str: # Basic IUPAC-condensed format
  "Convert glycan from LinearCode to barebones IUPAC-condensed format"
  replace_dic = {'G': 'Glc', 'ME': 'me', 'M': 'Man', 'A': 'Gal', 'NN': 'Neu5Ac', 'GlcN': 'GlcNAc', 'GN': 'GlcNAc',
                 'GalN': 'GalNAc', 'AN': 'GalNAc', 'F': 'Fuc', 'K': 'Kdn', 'W': 'Kdo', 'L': 'GalA', 'I': 'IdoA', 'PYR': 'Pyr', 'R': 'Araf', 'H': 'Rha',
                 'X': 'Xyl', 'B': 'Rib', 'U': 'GlcA', 'O': 'All', 'E': 'Fruf', '[': '', ']': '', 'me': 'Me', 'PC': 'PCho', 'T': 'Ac'}
  return multireplace(linearcode.split(';')[0], replace_dic)


def linearcode1d_to_iupac(linearcode: str # Glycan in LinearCode-1D format
                      ) -> str: # Basic IUPAC-condensed format
  replace_dic = {')': '[', '(': ']', 'G': 'Glc(a', 'A': 'Gal(b', 'Y': 'GlcNAc(b', 'M': 'Man(a', 'X': 'Xyl(b', 'F': 'Fuc(a', 'L': 'GlcA(b'}
  glycan = multireplace(linearcode[::-1], replace_dic)
  return '('.join(re.sub(r'([a-zA-Z])(\d)(\d)', r'\1\2-\3)', glycan).replace("Man(a1-4)GlcNAc", "Man(b1-4)GlcNAc").split('(')[:-1])


def iupac_extended_to_condensed(iupac_extended: str # Glycan in IUPAC-extended format
                             ) -> str: # Basic IUPAC-condensed format
  "Convert glycan from IUPAC-extended to barebones IUPAC-condensed format"
  # Find all occurrences of the pattern and apply the changes
  def replace_pattern(match):
    # Move the α or β after the next opening parenthesis
    return f"{match.group('after')}{match.group('alpha_beta')}"
  # The regular expression looks for α-D- or β-D- followed by any characters until an open parenthesis
  pattern = re.compile(r"(?P<alpha_beta>[αβßab\?])-[DL]-(?P<after>[^\)]*\()")
  # Substitute the pattern in the string with our replace_pattern function
  adjusted_string = pattern.sub(replace_pattern, iupac_extended)
  adjusted_string = re.sub(r"-\(", "(", adjusted_string)
  adjusted_string = re.sub(r"\)-", ")", adjusted_string)
  adjusted_string = re.sub(r"\]-", "]", adjusted_string)
  return adjusted_string[:adjusted_string.rindex('(')]


def glycoct_to_iupac_int(glycoct: str, # GlycoCT format string
                        mono_replace: Dict[str, str], # Monosaccharide replacement mappings
                        sub_replace: Dict[str, str] # Substituent replacement mappings
                       ) -> Tuple[Dict[int, str], Dict[int, List[Tuple[str, int]]], Dict[int, int]]: # (Residue dict, IUPAC parts, Degrees)
  "Internal function for GlycoCT conversion"
  # Dictionaries to hold the mapping of residues and linkages
  residue_dic = {}
  iupac_parts = defaultdict(list)
  degrees = defaultdict(lambda:1)
  glycoct = glycoct.replace("S1b:", "S\n1b:")
  for line in glycoct.split('\n'):
    if len(line) < 1:
      continue
    if line.startswith('RES'):
      residues = True
    elif line.startswith('LIN'):
      residues = False
    elif residues:
      parts = line.split(':')
      #monosaccharide
      if parts[0][-1] == 'b':
        res_id = int(parts[0][:-1])
        res_type = parts[1].split('-')[1] + parts[1].split('-')[0].replace('x', '?')
        suffix = 'f' if parts[2].startswith('4') else 'A' if (len(parts) == 4 and parts[3].startswith('a')) else ''
        clean_mono = multireplace(res_type, mono_replace)
        if suffix:
          clean_mono = clean_mono[:-1] + suffix + clean_mono[-1]
        residue_dic[res_id] = clean_mono
      #modification
      elif parts[0][-1] == 's':
        tgt = ')' + str(int(parts[0][:-1]))+'n'
        pattern = re.escape(tgt)
        matched = re.search(pattern, glycoct)
        if matched:
          start_index = matched.start()
          stretch = glycoct[:start_index]
          stretch = stretch[stretch.rindex(':'):]
          numerical_part = re.search(r'\d+', stretch)
          res_id = int(numerical_part.group())
        else:
          res_id = max(residue_dic.keys())
        res_type = multireplace(parts[1], sub_replace)
        residue_dic[res_id] = residue_dic[res_id][:-1] + res_type + residue_dic[res_id][-1]
    #linkage
    elif len(line) > 0:
      line = line.replace('-1', '?')
      line = re.sub(r"\d\|\d", "?", line)
      parts = re.findall(r'(\d+)[do]\(([\d\?]+)\+(\d+)\)(\d+)', line)[0]
      parent_id, child_id = int(parts[0]), int(parts[3])
      link_type = f"{residue_dic.get(child_id, 99)}({parts[2]}-{parts[1]})"
      if link_type.startswith('99') and parts[1] not in ['2', '5']:
        residue_dic[parent_id] = re.sub(r'(O)(?=S|P|Ac|Me)', parts[1], residue_dic[parent_id], count = 1)
      if not link_type.startswith('99'):
        iupac_parts[parent_id].append((f"{parts[2]}-{parts[1]}", child_id))
        degrees[parent_id] += 1
  for r in residue_dic:
    if r not in degrees:
      degrees[r] = 1
  return residue_dic, iupac_parts, degrees


def glycoct_build_iupac(iupac_parts: Dict[int, List[Tuple[str, int]]], # IUPAC format components
                       residue_dic: Dict[int, str], # Residue mappings
                       degrees: Dict[int, int] # Node degrees
                      ) -> str: # IUPAC-condensed format
  "Build IUPAC string from GlycoCT components"
  start = min(residue_dic.keys())
  iupac = residue_dic[start]
  inverted_residue_dic = {}
  inverted_residue_dic.setdefault(residue_dic[start], []).append(start)
  for parent, children in iupac_parts.items():
    child_strings, i, last_child = [], 0, 0
    children_degree = [degrees[c[1]] for c in children]
    children = [x for _, x in sorted(zip(children_degree, children), reverse = True)]
    for child in children:
      prefix = '[' if degrees[child[1]] == 1 else ''
      suffix = ']' if children.index(child) > 0 else ''
      child_strings.append(prefix + residue_dic[child[1]] + '(' + child[0] + ')' + suffix)
      idx = inverted_residue_dic[residue_dic[parent]].index(parent)+1 if residue_dic[parent] in inverted_residue_dic else 0
      pre = iupac[:find_nth_reverse(iupac, residue_dic[parent], idx, ignore_branches = True)] if idx else ''
      if i > 0 and residue_dic[child[1]] == residue_dic[last_child]:
        inverted_residue_dic.setdefault(residue_dic[child[1]], []).insert(-1, child[1])
      elif (residue_dic[child[1]] in pre):
        county = pre.count(residue_dic[child[1]])
        inverted_residue_dic.setdefault(residue_dic[child[1]], []).insert(-county, child[1])
      else:
        inverted_residue_dic.setdefault(residue_dic[child[1]], []).append(child[1])
      i += 1
      last_child = child[1]
    prefix = ']' if degrees[parent] > 2 and len(children) == 1 else ''
    nth = [k.index(parent) for k in inverted_residue_dic.values() if parent in k][0] + 1
    idx = find_nth_reverse(iupac, residue_dic[parent], nth, ignore_branches = True)
    iupac = iupac[:idx] + ''.join(child_strings) + prefix + iupac[idx:]
  return iupac.strip('[]')


def glycoct_to_iupac(glycoct: str # Glycan in GlycoCT format
                    ) -> str: # Basic IUPAC-condensed format
  "Convert glycan from GlycoCT to barebones IUPAC-condensed format"
  floating_part, floating_bits = '', []
  mono_replace = {'dglc': 'Glc', 'dgal': 'Gal', 'dman': 'Man', 'lgal': 'Fuc', 'dgro': 'Neu', 'lido': 'Ido',
                  'dxyl': 'Xyl', 'dara': 'D-Ara', 'lara': 'Ara', 'HEX': 'Hex', 'lman': 'Rha', 'lxyl': 'Col', 'dgul': 'Gul'}
  sub_replace = {'n-acetyl': 'NAc', 'sulfate': 'OS', 'phosphate': 'OP', 'n-glycolyl': '5Gc',
                 'acetyl': 'OAc', 'methyl': 'OMe'}
  global_replace = {'dman-OCT': 'Kdo'}
  glycoct = multireplace(glycoct, global_replace)
  if len(glycoct.split("UND")) > 1:
    floating_bits = glycoct.split("UND")[2:]
    floating_bits = ["RES" + f.split('RES')[1] for f in floating_bits]
    glycoct = glycoct.split("UND")[0]
  # Split the input by lines and iterate over them
  residue_dic, iupac_parts, degrees = glycoct_to_iupac_int(glycoct, mono_replace, sub_replace)
  if floating_bits:
    for f in floating_bits:
      residue_dic_f, iupac_parts_f, degrees_f = glycoct_to_iupac_int(f, mono_replace, sub_replace)
      expr = "(1-?)}"
      if len(residue_dic_f) == 1:
        floating_part += f"{'{'}{list(residue_dic_f.values())[0]}{expr}"
      else:
        part = glycoct_build_iupac(iupac_parts_f, residue_dic_f, degrees_f)
        floating_part += f"{'{'}{part}{expr}"
  # Build the IUPAC-condensed string
  iupac = glycoct_build_iupac(iupac_parts, residue_dic, degrees)
  iupac = floating_part + iupac[:-1]
  pattern = re.compile(r'([ab\?])\(')
  iupac = pattern.sub(lambda match: f"({match.group(1)}", iupac)
  iupac = re.sub(r'(\?)(?=S|P|Me)', 'O', iupac)
  iupac = re.sub(r'([1-9\?O](S|P|Ac|Me))NAc', r'NAc\1', iupac)
  if ']' in iupac and iupac.index(']') < iupac.index('['):
    iupac = iupac.replace(']', '', 1)
  return iupac.replace('[[', '[').replace(']]', ']').replace('Neu(', 'Kdn(')


def get_mono(token: str # WURCS monosaccharide token
           ) -> str: # Monosaccharide with anomeric state
  "Map WURCS token to monosaccharide with anomeric state"
  hex_or_dhex = 'h' if 'h' in token else 'm'
  token = 'a' + token[1:].replace(hex_or_dhex, f'{hex_or_dhex}-1x_1-5', 1) if token.startswith('u') else token
  token = f"{token.replace('U', 'a')}-2x_2-6" if 'U' in token else token
  anomer = token[token.index('_')-1] if '_' in token else ''
  if token in monosaccharide_mapping:
    mono = monosaccharide_mapping[token]
  else:
    mono = monosaccharide_mapping.get(f"{token}-1x_1-5", None)
    for a in ['a', 'b', 'x']:
      if anomer and a != anomer:
        token = token[:token.index('_')-1] + a + token[token.index('_'):]
        mono = monosaccharide_mapping.get(token, None)
        if mono:
          break
    if not mono:
      raise Exception(f"Token {token} not recognized.")
  mono += anomer if anomer and anomer in ['a', 'b'] else '?'
  return mono


def wurcs_to_iupac(wurcs: str # Glycan in WURCS format
                  ) -> str: # Basic IUPAC-condensed format
  "Convert glycan from WURCS to barebones IUPAC-condensed format"
  wurcs = wurcs[wurcs.index('/')+1:]
  pattern = r'\b([a-z])\d(?:\|\1\d)+\}?|\b[a-z](\d)(?:\|[a-z]\2)+\}?'
  additional_pattern = r'\b([a-z])\?(?:\|\w\?)+\}?'
  def replacement(match):
    text = match.group(0)
    if '|' in text and text[-1].isdigit():  # Case like r3|r6
      letter = text[0]
      nums = [c for c in text if c.isdigit()]
      return f'{letter}{nums[0]}*{nums[1]}'
    return f'{match.group(1)}?' if match.group(1) else f'?{match.group(2)}'
  wurcs = re.sub(pattern, replacement, wurcs)
  wurcs = re.sub(additional_pattern, '?', wurcs)
  wurcs = re.sub(r'([a-z][\d\?])\*OPO\*\/3O\/3\=O', r'\1P', wurcs)  # phospho-linkages
  floating_part, floating_parts = '', []
  parts = wurcs.split('/')
  topology = parts[-1].split('_')
  monosaccharides = '/'.join(parts[1:-2]).strip('[]').split('][')
  connectivity = parts[-2].split('-')
  connectivity = {chr(97 + i) if i < 26 else chr(65 + i - 26): int(num) for i, num in enumerate(connectivity)}
  degrees = {c: ''.join(topology).count(c) for c in connectivity}
  inverted_connectivity, iupac_parts = {}, []
  for link in topology:
    if '-' not in link:
      return get_mono(monosaccharides[0])
    source, target = link.split('-')
    source_index, source_carbon = connectivity[source[:-1]], source[-1]
    source_mono = get_mono(monosaccharides[int(source_index)-1])
    if target[0] == '?':
      floating_part += f"{'{'}{source_mono}(1-{target[1:]}){'}'}"
      floating_parts.append(source[0])
      continue
    target_index, target_carbon = connectivity[target[0]], target[1:]
    target_mono = get_mono(monosaccharides[int(target_index)-1])
    if '*' in target[1:]:  # Ultra-narrow wildcards
      target_carbon = '/'.join(target[1:].split('*'))
      iupac_parts.append((f"{source_mono}({source_carbon}-{target_carbon}){target_mono}", source[0], target[0]))
    elif '?' in target:
      iupac_parts.append((f"{source_mono}({source_carbon}-{target_carbon}){target_mono}", source[0], target[0]))
    else:
      iupac_parts.append((f"{target_mono}({target_carbon}-{source_carbon}){source_mono}", target[0], source[0]))
  degrees_for_brackets = copy.deepcopy(degrees)
  iupac_parts = sorted(iupac_parts, key = lambda x: x[2])
  iupac = iupac_parts[0][0]
  inverted_connectivity.setdefault(connectivity[iupac_parts[0][2]], []).append(iupac_parts[0][2])
  inverted_connectivity.setdefault(connectivity[iupac_parts[0][1]], []).append(iupac_parts[0][1])
  degrees_for_brackets[iupac_parts[0][2]] -= 1
  prefix = '[' if degrees[iupac_parts[0][1]] == 1 else ''
  suffix = ']' if prefix == '[' and iupac_parts[0][2] == 'a' else ''
  iupac = prefix + iupac[:iupac.index(')')+1] + suffix + iupac[iupac.index(')')+1:]
  iupac = floating_part + iupac
  for fp in floating_parts:
    inverted_connectivity.setdefault(connectivity[fp], []).append(fp)
  for parts, tgt, src in iupac_parts[1:]:
    indices = [k.index(src) for k in inverted_connectivity.values() if src in k]
    nth = (indices[0] if indices else 0) + 1
    overlap = parts.split(')')[-1]
    ignore = True if degrees[src] > 2 or (degrees[src] == 2 and src == 'a') else False
    idx = find_nth_reverse(iupac, overlap, nth, ignore_branches = ignore)
    prefix = '[' if degrees[tgt] == 1 else ''
    suffix = ']' if (degrees[src] > 2 and degrees_for_brackets[src] < degrees[src]) or (degrees[src] == 2 and degrees_for_brackets[src] < degrees[src] and src == 'a') or (degrees[src] > 3 and degrees[tgt] == 1) or (degrees[tgt] == 1 and src =='a')  else ''
    iupac = iupac[:idx] + prefix + parts.split(')')[0]+')' + suffix + iupac[idx:]
    degrees_for_brackets[src] -= 1
    insertion_idx = iupac[:idx].count(parts.split(')')[0][:-4])
    if insertion_idx > 0:
      inverted_connectivity.setdefault(connectivity[tgt], []).insert(-insertion_idx, tgt)
    else:
      inverted_connectivity.setdefault(connectivity[tgt], []).append(tgt)
  iupac = iupac[:-1].strip('[]').replace('}[', '}').replace('{[', '{')
  pattern = re.compile(r'([ab\?])\(')
  iupac = pattern.sub(lambda match: f"({match.group(1)}", iupac)
  # Define the pattern to find two ][ separated by a string with exactly one (
  pattern = r'(\]\[[^\[\]]*\([^][]*\)\][^\[\]]*)\]\['
  iupac = re.sub(pattern, r'\1[', iupac)
  if ']' in iupac and '[' in iupac and iupac.index(']') < iupac.index('['):
    iupac = iupac.replace(']', '', 1)
  if '[' in iupac and ']' not in iupac[iupac.index('['):]:
    iupac = iupac[:iupac.rfind(')')+1] + ']' + iupac[iupac.rfind(')')+1:]
  def remove_first_unmatched_opening_bracket(s):
    balance = 0
    for i, char in enumerate(s):
      balance += (char == '[') - (char == ']')
      if balance < 0:
        return s[:i] + s[i + 1:]
    return s
  iupac = remove_first_unmatched_opening_bracket(iupac)
  return re.sub(r'(\d)([P])\-', r'\1-\2-', iupac)


def oxford_to_iupac(oxford: str # Glycan in Oxford format
                   ) -> str: # Glycan in IUPAC-condensed format
  "Convert glycan from Oxford to IUPAC-condensed format"

  def parse_sialic_acid_bonds(glycan_string):
      "Extracts sialic acids and their linkages from the Oxford string"
      result = []
      pattern = r'(Sg?(?:)?)(\d*)((?:[\[\(][^\]\)]+[\]\)])*)?(\d*)'
      for match in re.finditer(pattern, glycan_string):
          residue = match.group(1)
          linkages = match.group(3) or ""
          count = int(match.group(4) or match.group(2) or 1)
          bonds = []
          for bracket in re.finditer(r'[\[\(]([^\]\)]+)[\]\)]', linkages):
              if len(bonds) >= count: break
              if bracket.group(1) in ['Ac','s']: break
              nums = [int(x.strip()) for x in bracket.group(1).split(',')]
              if 2 in nums:
                  for i, n in enumerate(nums):
                      if n == 2 and i+1 < len(nums):
                          bonds.append(f"{nums[i+1]}")
                      elif i > 0 and nums[i-1] != 2 and i+1 < len(nums) and nums[i+1] == 2:
                          bonds.append(f"{n}")
              else:
                  bonds.extend(f"{n}" for n in nums if n in [3, 6, 8])
          out_res = {'S':'Neu5Ac','Sg':'Neu5Gc'}[residue]
          bonds = bonds + ["3/6"] * (count - len(bonds))
          for bond in bonds:
              result.append(f"{out_res}(a2-{bond})")
      return result

  def parse_galactose_info(sequence):
      "Extracts antennary galactose residues and their linkages from the Oxford string"
      pattern = r'(?<!S)G(?![a-z])(?:\(([^)]+)\))?(?:\[[^\]]+\])?(\d*)(?:\(([^)]+)\))?'
      match = re.search(pattern, sequence)
      if match:
          bond_str = match.group(1) or match.group(3)
          count = int(match.group(2)) if match.group(2) else 1
          if bond_str:
              bonds = [f'Gal(b1-{x.strip()})' for x in bond_str.split(',')]
              return bonds + ['Gal(b1-3/4)'] * (count - len(bonds))
          else:
              return ['Gal(b1-3/4)' for x in range(count)]
      else:
          return []

  def balance_mannose_branch_linkages(iupac, antenna_number=None):
    """
    Checks whether a1-3 and a1-6 branches are identical, if not assigns a1-3/6.
    If antenna_number is provided, assigns that number to the longest branch.
    Args:
        iupac: IUPAC string notation
        antenna_number: Optional preferred antenna number (3 or 6)
    """
    pattern = r'^(.*?)((?:\[[^\]]*\])?)(Man\(a1-3\))\[(.*?)(Man\(a1-6\))\](\[[^\]]*\])*(Man\(b1-4\))'
    match = re.search(pattern, iupac)
    if match:
        prefix = match.group(1)
        bracket_before_a1_3 = match.group(2)
        branch_a1_3_content = re.split(r'[\[\]]', prefix)[-1] + bracket_before_a1_3
        branch_a1_6_content = match.group(4)

        if branch_a1_3_content != branch_a1_6_content:
            if antenna_number is None:
                iupac = prefix + bracket_before_a1_3 + 'Man(a1-3/6)' + '[' + match.group(4) + 'Man(a1-3/6)' + ']' + (match.group(6) or '') + match.group(7) + iupac[match.end():]
            elif antenna_number:
                antenna_number = int(antenna_number)
                long_num, short_num = (6, 3) if (len(branch_a1_3_content) >= len(branch_a1_6_content)) ^ (antenna_number == 3) else (3, 6)
                iupac = prefix + bracket_before_a1_3 + f'Man(a1-{long_num})' + '[' + match.group(4) + f'Man(a1-{short_num})' + ']' + (match.group(6) or '') + match.group(7) + iupac[match.end():]
    return iupac

  match = re.fullmatch(r'^(?:M|Man)[-]?(\d+)$', oxford, re.IGNORECASE)
  if match:
    oxford = f'M{match.group(1)}'
  oxford = oxford.replace("(s)", "Sulf")
  oxford = oxford.strip().split('/')[0]
  antennae = {}
  iupac = "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc"
  mapping_dict = {"A": "GlcNAc(b1-?)", "G": "Gal(b1-3/4)", "S": "Neu5Ac(a2-3/6)",
                  "Sg": "Neu5Gc(a2-3/6)", "Ga": "Gal(a1-?)", "Gal": "Gal(?1-?)", "GalNAc": "GalNAc(?1-?)",
                  "Lac": "Gal(b1-3/4)GlcNAc(b1-?)", "F": "Fuc(a1-3/4)", "LacDiNAc": "GalNAc(b1-4)GlcNAc(b1-?)"}
  hardcoded = {"M3": "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
               "M4": "Man(a1-2/3/6)Man(a1-3/6)[Man(a1-3/6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
               "M9": "Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
               "M9Gluc1": "Glc(a1-3)Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
               "M10": "Glc(a1-3)Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
               "M11": "Glc(a1-3)Glc(a1-3)Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
               "M12": "Glc(a1-2)Glc(a1-3)Glc(a1-3)Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc"}
  if oxford in hardcoded:
    return hardcoded[oxford]
  oxford = oxford.replace("[SO4-2]", "Sulf")
  if "Sulf" in oxford:
    sulf = oxford[oxford.index("Sulf")+4]
    sulf = int(sulf) if sulf.isdigit() else 1
    oxford = oxford.replace("Sulf", '')
  else:
    sulf = 0
  if 'B' in oxford:
    split = iupac.index(']')
    iupac = iupac[:split+1] + "[GlcNAc(b1-4)]" + iupac[split+1:]
  elif 'X' in oxford:
    split = iupac.index(']')
    iupac = iupac[:split+1] + "[Xyl(b1-2)]" + iupac[split+1:]
  if oxford.startswith('F'):
    split = iupac.rindex(')')
    fuc = "[Fuc(a1-3)]" if "X" in oxford else "[Fuc(a1-6)]"
    iupac = iupac[:split+1] + fuc + iupac[split+1:]
  if 'F' in oxford[1:]:
    nth = oxford.count('F')
    antennae["F"] = int(oxford[find_nth(oxford, "F", nth)+1])
  floaty = ''
  if 'M' in oxford:
    M_count = int(oxford[oxford.index("M")+1]) - 3
    for m in range(M_count):
      floaty += "{Man(a1-?)}"
  oxford_wo_branches = bracket_removal(oxford)
  branches = {"A": int(oxford_wo_branches[oxford_wo_branches.index("A")+1]) if "A" in oxford_wo_branches and oxford_wo_branches[oxford_wo_branches.index("A")+1] != "c" else 0}
  extras = {"Ga": int(oxford_wo_branches[oxford_wo_branches.index("Ga")+2]) if "Ga" in oxford_wo_branches and oxford_wo_branches[oxford_wo_branches.index("Ga")+2].isdigit() else 0,
            "Gal": int(oxford_wo_branches[oxford_wo_branches.index("Gal")+3]) if "Gal" in oxford_wo_branches and oxford_wo_branches[oxford_wo_branches.index("Gal")+3].isdigit() else 0,
            "Lac": int(oxford_wo_branches[oxford_wo_branches.index("Lac")+3]) if "Lac" in oxford_wo_branches and oxford_wo_branches[oxford_wo_branches.index("Lac")+3] != "D" else 0,
            "LacDiNAc": 1 if "LacDiN" in oxford_wo_branches else 0}
  branches['G'] = parse_galactose_info(oxford)
  branches['S'] = parse_sialic_acid_bonds(oxford)
  branches['A'] = ['GlcNAc(b1-?)' for x in range(branches['A'])]
  built_branches = []
  while sum([len(x) for x in branches.values()]) > 0:
    temp = ''
    for c in ["S", "G", "A"]:
      if len(branches[c]) > 0:
        temp += branches[c].pop(0)
    if temp:
      built_branches.append(temp)
  i = 0
  for b in built_branches:
    if i == 0:
      iupac = b + iupac
    elif i == 1:
      split = iupac.index("[Man")
      iupac = iupac[:split+1] + b + iupac[split+1:]
    elif i == 2:
      split = iupac.index("Man")
      iupac = iupac[:split] + "[" + b + "]" + iupac[split:]
    elif i == 3:
      split = find_nth(iupac, "Man", 2)
      iupac = iupac[:split] + "[" + b + "]" + iupac[split:]
    i += 1
  for e,v in extras.items():
    while v > 0:
      if iupac.startswith("Gal(b"):
        iupac = mapping_dict[e] + iupac
      elif "[Gal(b" in iupac:
        split = iupac.index("[Gal(b")
        iupac = iupac[:split+1] + mapping_dict[e] + iupac[split+1:]
      else:
        iupac = mapping_dict[e] + iupac
      v -= 1
  if antennae:
    for k, v in antennae.items():
      while v > 0:
        if "Gal(b1-3/4)Glc" in iupac:
          split = iupac.index("Gal(b1-3/4)Glc")
          iupac = iupac[:split+len("Gal(b1-3/4)")] + "[" + mapping_dict[k] + "]" + iupac[split+len("Gal(b1-3/4)"):]
        else:
          split =  iupac.index("GalNAc(b1-4)Glc")
          iupac = iupac[:split+len("GalNAc(b1-4)")] + "[" + mapping_dict[k] + "]" + iupac[split+len("GalNAc(b1-4)"):]
        v -= 1
  iupac = iupac.replace("GlcNAc(b1-?)[Neu5Ac(a2-3/6)]Man", "[Neu5Ac(a2-3/6)]GlcNAc(b1-?)Man")
  while "Neu5Ac(a2-8)G" in iupac:
    iupac = iupac.replace("Neu5Ac(a2-8)G", "G", 1)
    idx = [m.start() for m in re.finditer(r'(?<!8\))Neu5Ac\(a2-[3|6|\?]\)', iupac)][0]
    iupac = iupac[:idx] + "Neu5Ac(a2-8)" + iupac[idx:]
  while "[Neu5Ac(a2-8)]" in iupac:
    iupac = iupac.replace("[Neu5Ac(a2-8)]", "", 1)
    idx = [m.start() for m in re.finditer(r'(?<!8\))Neu5Ac\(a2-[3|6|\?]\)', iupac)][0]
    iupac = iupac[:idx] + "Neu5Ac(a2-8)" + iupac[idx:]
  while sulf > 0:
    iupac = iupac.replace("Gal(", "GalOS(", 1)
    sulf -= 1
  iupac = iupac.replace("GlcNAc(b1-?)Man", "GlcNAc(b1-2)Man")
  antenna_number = match.group(2) if (match := re.search(r'A\d+(B)?\[([36])\]', oxford)) else None
  iupac = balance_mannose_branch_linkages(iupac,antenna_number)
  return floaty + iupac.strip('[]')



def glycam_to_iupac(glycan: str # Glycan in GLYCAM nomenclature
                    ) -> str: # Basic IUPAC-condensed format
  "Convert glycan from GLYCAM to IUPAC-condensed format"
  pattern = r'(?:[DL])|(?:\[(\d+[SP]+)\])'
  glycan = '-'.join(glycan.split('-')[:-1])[:-2] if '-OH' in glycan else glycan
  glycan = re.sub(pattern, lambda m: m.group(1) if m.group(1) else '', glycan)
  return glycan.replace('[', '(').replace(']', ')')


def glycoworkbench_to_iupac(glycan: str # Glycan in GlycoWorkBench nomenclature
                            ) -> str: # Basic IUPAC-condensed format
  """Convert GlycoWorkBench nomenclature to IUPAC-condensed format."""
  glycan = glycan.replace('D-', '').replace('L-', '').replace("--?[", '')  # Remove D- and L- prefixes
  repeat_pattern = r'--(\d+)\[(.*?)\]'
  main_part = glycan.split('$MONO')[0]
  while re.search(repeat_pattern, main_part):
    match = re.search(repeat_pattern, main_part)
    repeat_count = int(match.group(1))
    repeat_unit = match.group(2)
    expanded = repeat_unit * repeat_count
    main_part = main_part[:match.start()] + expanded + main_part[match.end():]
  glycan = (main_part + glycan[glycan.index('$MONO'):] if '$MONO' in glycan else main_part).replace("--?]", '')
  # Handle floating parts if present
  floaty_parts = []
  if '}' in glycan:
    glycan, floaty_section = glycan.split('}')
    floaty_section = floaty_section.split('$MONO')[0]
    # Process floating sections into parts
    content = floaty_section.strip('()')
    content = re.sub(r'\(+', '(', content)
    parts = [part.strip('()') for part in re.split(r'\)\-\-', content) if part]
    # Convert each floating part to IUPAC format
    for part in parts:
      new_float = ''.join(part.split(',p'))
      processed_monos = [f"{x.split(',')[0][3:]}({x[1]}{x[2]}-{x[0]})".strip('-') for x in new_float.split('--') if x]
      floaty_parts.append(''.join(processed_monos[::-1]))
  # Process main glycan structure
  split_monos = [x for x in glycan.split('$MONO')[0].split('--')[1:] if x != '?']
  if ',' not in split_monos[-1]:
    split_monos[-1] = split_monos[-1] + ',p'
  # Convert monosaccharides to IUPAC format
  converted_monos = [f"{x[1:]}(?1-?)" if re.match(r"^\?[A-Z]", x) else
                     f"{x.split(',')[0][3:]}{'f' if ',f' in x else ''}({x[1]}{x[2]}-{x[0]})" +
                     "".join(re.findall("[()]+", y)).replace("(","]").replace(")","[")
                     for x, y in zip(split_monos, glycan.split('--'))]
  converted_glycan = ''.join(converted_monos[::-1])
  # Fix double branch notation if present
  if ']]' in converted_glycan:
    double_brack_idx = converted_glycan.index(']]')
    branch_str = converted_glycan[:double_brack_idx + 2]
    second_brack_end = [(m.start(0), m.end(0)) for m in re.finditer(re.escape('['), branch_str)][-1][0]
    converted_glycan = branch_str[:second_brack_end] + ']' + branch_str[second_brack_end:-1] + converted_glycan[double_brack_idx+2:]
  if floaty_parts:  # Add floating parts to final structure
    converted_glycan = ''.join(f"{{{part}}}" for part in floaty_parts) + converted_glycan
  converted_glycan = converted_glycan.replace('((', '(').replace('))', ')').replace(')(', '(')
  converted_glycan = re.sub(r'([SP])[\)\(]*\?1-([\?\d])\)\[(.*?)\]([^(]+)', r'\3\4\2\1', converted_glycan)  # sulfate/phosphate with intervening branch
  converted_glycan = re.sub(r'\[([SP])[\)\(]*\?1-([\?\d])\)([^(]+)', r'[\3\2\1', converted_glycan)  # sulfate/phosphate
  converted_glycan = converted_glycan.replace('((', '(').replace('))', ')')
  return f"{converted_glycan[:-6]}-ol" if 'freeEnd' in glycan else converted_glycan[:-6]


def glytoucan_to_glycan(ids: List[str], # List of GlyTouCan IDs or glycans
                       revert: bool = False # Whether to map glycans to IDs; default:False
                      ) -> List[str]: # List of glycans or IDs
  "Convert between GlyTouCan IDs and IUPAC-condensed glycans"
  if not hasattr(glytoucan_to_glycan, 'glycan_dict'):
    glytoucan_to_glycan.glycan_dict = dict(zip(df_glycan.glytoucan_id, df_glycan.glycan))
    glytoucan_to_glycan.id_dict = dict(zip(df_glycan.glycan, df_glycan.glytoucan_id))
  lookup = glytoucan_to_glycan.id_dict if revert else glytoucan_to_glycan.glycan_dict
  result , not_found = [], []
  for item in ids:
    if item in lookup:
      result.append(lookup[item])
    else:
      result.append(item)
      not_found.append(item)
  # Print missing items if any
  if not_found:
    msg = 'glycans' if revert else 'IDs'
    print(f'These {msg} are not in our database: {not_found}')
  return result


def GAG_disaccharide_to_iupac(input_dsc: str # Disaccharide structural code (DSC) for GAGs
                             ) -> str: # Basic IUPAC-condensed format
  "Convert disaccharide GAG codes like D2A6 into 4uHexA2S(?1-?)GlcNAc6S"
  non_red_end_map = {'U': 'HexA', 'D': '4uHexA', 'G': 'GlcA', 'I': 'IdoA', 'g': 'Gal'}
  non_red_end_sulf = {'0': '', '2': '2S'}
  hexosamine_map = {'A': 'GlcNAc', 'a': 'GalNAc', 'S': 'GlcNS', 'H': 'GlcN'}
  hexosamine_sulf = {'0': '', '3': '3S', '4': '4S', '6': '6S', '9': '3S6S', '10': '4S6S'}
  non_red_base = non_red_end_map.get(input_dsc[0])
  non_red_sulfation = non_red_end_sulf.get(input_dsc[1])
  hexosamine_base = hexosamine_map.get(input_dsc[2])
  hexosamine_sulfation = hexosamine_sulf.get(input_dsc[3:])
  linkage = '(?1-?)'
  return f"{non_red_base}{non_red_sulfation}{linkage}{hexosamine_base}{hexosamine_sulfation}"


def nglycan_stub_to_iupac(nglycan_stub: str # Glycan in a N-glycan stub format
                      ) -> str: # Basic IUPAC-condensed format
  "Convert glycan from N-glycan stub to barebones IUPAC-condensed format"
  nglycan_stub = nglycan_stub.replace("(Man)3(GlcNAc)2", "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc").replace("Deoxyhexose", "Fuc")
  parts_dict, nglycan_stub = nglycan_stub.split('+')
  parts_dict = {x.split(')')[0].replace('(', ''): int(x.split(')')[1]) for x in re.findall(r'\([^)]+\)\d+', parts_dict)}
  parts = ''.join([f"{{{p}(?1-?)}}" * v for p, v in parts_dict.items()])
  return f"{parts}{nglycan_stub}"


def check_nomenclature(glycan: str # Glycan string to check
                     ) -> None: # Prints reason if not convertible
  "Check whether glycan has correct nomenclature for glycowork"
  if not isinstance(glycan, str):
    raise TypeError("Glycan sequences must be formatted as strings")
  if '@' in glycan:
    raise ValueError("Seems like you're using SMILES. We currently can only convert IUPAC-->SMILES; not the other way around.")


def sanitize_iupac(glycan: str # Glycan string to check
                   ) -> str: # Sanitized glycan string
  """Sanitize IUPAC glycan sequence by identifying and correcting chemical impossibilities."""
  # Handle NAc special case (any sugar with NAc can't have linkage at position 2)
  glycan = re.sub(r'([A-Za-z]+)\(([ab?][1-2])-2\)([A-Za-z]+NAc)', r'\1(\2-?)\3', glycan)
  # Handle modifications (can't have a linkage to a position that's modified)
  glycan = re.sub(r'\(([ab?][1-2])-(\d)\)([A-Za-z]+\2[A-Z])', r'(\1-?)\3', glycan)
  # Handle branched cases with same linkage position
  for match in re.finditer(r'([A-Za-z]+)\(([ab?][1-2])-(\d)\)\[((?:[A-Za-z]+\([ab?][1-2]-\d\))*)([A-Za-z]+)\(([ab?][1-2])-(\3)\)\]', glycan):
    glycan = glycan.replace(match.group(0), f'{match.group(1)}({match.group(2)}-?)[{match.group(4)}{match.group(5)}({match.group(6)}-?)]')
  return glycan


def transform_repeat_glycan(glycan: str # Glycan string to check
                            ) -> Tuple[str, bool]: # Glycan string with converted repeat structure, if necessary, and whether glycan is repeat
  """Transform -1)Fruf(b2-3)Fruf(b2- repeat structure into Fruf(b2-3)Fruf(b2-1)Fruf"""
  if glycan.startswith("-"):
    match = re.match(r"(-P)?(-\d+\))(\[[^\]]*\])?([A-Za-z0-9]+)(.*)", glycan)
    if match:
      phospho, linkage, brackets, first_mono, remainder = match.groups()
      brackets = brackets or ""
      phospho = phospho or ""
      glycan = brackets + first_mono + remainder + phospho + linkage + first_mono
      glycan = re.sub(r'(\w+)\(([\w\?])(\d+)-P-(\d+)\)', lambda m: f"{m.group(1)}{m.group(3)}P({m.group(2)}{m.group(3)}-{m.group(4)})", glycan)
      return re.sub(r"-ol(\d+)P", r"\1P-ol", glycan), True
  return glycan, False


def canonicalize_iupac(glycan: str # Glycan sequence in any supported format
                     ) -> str: # Standardized IUPAC-condensed format
  "Convert glycan from IUPAC-extended, LinearCode, GlycoCT, WURCS, Oxford, GLYCAM, GlycoWorkBench, CSDB-linear, and GlyTouCanIDs to standardized IUPAC-condensed format"
  glycan = glycan.strip().replace('–', '-').replace(' ', '')
  mapped_glycan = GLYCAN_MAPPINGS.get(glycan.lower())
  if mapped_glycan:
    return mapped_glycan
  # Check for different nomenclatures: LinearCode, IUPAC-extended, GlycoCT, WURCS, Oxford, GLYCAM, GlycoWorkBench, GlyTouCanIDs
  if ';' in glycan:
    glycan = linearcode_to_iupac(glycan)
  elif glycan.startswith('0'):
    glycan = linearcode1d_to_iupac(glycan)
  elif bool(re.match('[^o]-[LD]-', glycan)):
    glycan = iupac_extended_to_condensed(glycan)
  elif 'RES' in glycan:
    glycan = glycoct_to_iupac(glycan)
  elif 'S=' in glycan:
    glycan = wurcs_to_iupac(glycan)
  elif glycan.endswith('-OH') or bool(re.search(r'\d[DL][A-Z]', glycan)):
    glycan = glycam_to_iupac(glycan)
  elif 'End--' in glycan:
    glycan = glycoworkbench_to_iupac(glycan)
  elif bool(re.fullmatch(r'^[UDGIg][02][AaSH](0|3|4|6|9|10)$', glycan)):
    glycan = GAG_disaccharide_to_iupac(glycan)
  elif bool(re.match(r'^G\d+', glycan)):
    glycan = glytoucan_to_glycan([glycan])[0]
  elif not isinstance(glycan, str) or '@' in glycan:
    check_nomenclature(glycan)
  elif "(Man)3(GlcNAc)2" in glycan:
    glycan = nglycan_stub_to_iupac(glycan)
  elif 'e' not in glycan and  (not re.search(r"[a-z1]\-", glycan) or len(glycan) < 6) and ((glycan[-1].isdigit() and re.search("[A-Zm]", glycan)) or (glycan[-2].isdigit() and glycan[-1] in ')]') or glycan.endswith(('B', 'Bi', 'LacDiNAc'))):
    glycan = oxford_to_iupac(glycan)
  # Canonicalize usage of monosaccharides and linkages
  # Anomeric indicator placed before parentheses
  if len(re.findall(r'\(', glycan)) == len(re.findall(r'[βα]\(', glycan)):
    glycan = re.sub(r'([βα])(\()', r'\2\1', glycan)
  glycan = CANONICALIZE.sub(lambda mo: replace_dic[mo.group()], glycan)
  glycan = re.sub(r'-([ab])-(\d+),(\d+\)?)-', r'\1\2-\3', glycan)  # Inconsistent usage of dashes and commas, like in Neu5Ac-a-2,6-Gal-b-1,3-GlcNAc
  glycan = re.sub(r'(\d),(\d)(?!l)', r'\1-\2', glycan)  # Replace commas between numbers unless followed by 'l' (for lactone)
  if '{' in glycan and '}' not in glycan:
    glycan = f'{{{glycan[:glycan.index("{")]}?1-?}}{glycan[glycan.index("{")+1:]}'
  if '{' in glycan and '(' not in glycan:
    glycan = glycan.replace('{', '(').replace('}', ')')
  # Trim linkers
  if '-' in glycan:
    last_dash = glycan.rindex('-')
    if bool(re.search(r'[a-z]\-[a-zA-Z]', glycan[last_dash-1:])) and 'ol' not in glycan and glycan[last_dash+1:] not in lib:
      glycan = glycan[:last_dash]
  # Anomeric and steric indicators placed before monosaccharide (e.g., "bDGal(1-4)bDGlcNAc")
  glycan = re.sub(r'([abx\?])([DLX\?])([A-Z1-9][A-Za-z2-9\-]*)\((\d)-(\d*\))', r'\3(\1\4-\5', glycan)
  glycan = re.sub(r'([abx\?])([DLX\?])([A-Z1-9][A-Za-z2-9\-]*)\((\d)', r'\3(\1\4', glycan)  # for repeats
  # Anomeric indicator placed before monosaccharide (e.g., "bGal14GlcNAc")
  glycan = re.sub(r'([ab])([A-Z][A-Za-z5]*)(\d)(\d*)', r'\2\1-\4', glycan)
  # Anomeric indicator placed behind monosaccharide (e.g., "Galb14GlcNAc")
  glycan = re.sub(r'([A-Z][A-Za-z5]*)([ab])([1-2])(\d)', r'\1\2\3-\4', glycan)
  # Canonicalize usage of brackets and parentheses
  if bool(re.search(r'\([A-Zd3-9]', glycan)) and not bool(re.search(r'\([ab?]', glycan)):
    glycan = glycan.replace('(', '[').replace(')', ']')
  # Canonicalize linkage uncertainty
  # Open linkages with anomeric config specified (e.g., "Mana-")
  glycan = re.sub(r'([A-Z][a-z]*)([a-b])\-([A-Z])', r'\1\g<2>1-?\3', glycan)
  # Open linkages (e.g., "c-")
  glycan = re.sub(r'([a-z])\-([A-Z][^\-])', r'\1?1-?\2', glycan)
  # Open linkages2 (e.g., "1-")
  glycan = re.sub(r'([1-2])\-(\))', r'\1-?\2', glycan)
  # Missing linkages (e.g., "c)")
  glycan = re.sub(r'(?<![hr])([a-b])([\(\)])', r'\1?1-?\2', glycan)
  # Open linkages in front of branches (e.g., "1-[")
  glycan = re.sub(r'([0-9])\-([\[\]])', r'\1-?\2', glycan)
  # Open linkages in front of branches (with missing information) (e.g., "c-[")
  glycan = re.sub(r'([a-z])\-([\[\]])', r'\1?1-?\2', glycan)
  # Branches without linkages (e.g., "[GalGlcNAc]")
  glycan = re.sub(r'(\[[a-zA-Z]+)(\])', r'\1?1-?\2', glycan)
  # Missing linkages in front of branches (e.g., "c[G")
  glycan = re.sub(r'([a-z])(\[[A-Z])', r'\1?1-?\2', glycan)
  # Missing anomer info (e.g., "(1")
  glycan = re.sub(r'(\()([1-2])', r'\1?\2', glycan)
  # Missing starting carbon (e.g., "b-4")
  glycan = re.sub(r'(a|b|\?)-(\d)', r'\g<1>1-\2', glycan)
  # If still no '-' in glycan, assume 'a3' type of linkage denomination
  if '-' not in glycan or bool(re.search(r'[ab][123456](?![\-PA])', glycan)):
    # Check whether linkages are recorded as b1 or as a3
    if bool(re.search(r"^[^2-6]*1?[^2-6]*$", glycan)):
      glycan = re.sub(r'(a|b)(\d)(?!\-)', r'\g<1>\g<2>-?', glycan)
    else:
      glycan = re.sub(r'(?<!h)(a|b)(\d)(?!\-)', r'\g<1>1-\g<2>', glycan)
  # Introduce parentheses for linkages
  if '(' not in glycan and len(glycan) > 6:
    for k in range(1, glycan.count('-')+1):
      idx = find_nth(glycan, '-', k)
      if (glycan[idx-1].isnumeric()) and (glycan[idx+1].isnumeric() or glycan[idx+1] == '?'):
        glycan = f'{glycan[:idx-2]}({glycan[idx-2:idx+2]}){glycan[idx+2:]}'
      elif (glycan[idx-1].isnumeric()) and bool(re.search(r'[A-Z]', glycan[idx+1])):
        glycan = f'{glycan[:idx-2]}({glycan[idx-2:idx+1]}?){glycan[idx+1:]}'
  # Canonicalize reducing end
  if bool(re.search(r'[a-z]ol', glycan)):
    glycan = glycan[:-2] if 'Glcol' not in glycan else f'{glycan[:-2]}-ol'
  if glycan[-1] in 'ab' and glycan[-3:] not in ['Rha', 'Ara']:
    glycan = glycan[:-1]
  # Remove anomeric and steric indicators at reducing end
  if '(' in glycan and bool(re.search(r'\)([ab][DLX\?][A-Z][A-Za-z5]*)', glycan)):
    glycan = re.sub(r'\)([ab][DLX\?])([A-Z][A-Za-z5]*)', r')\2', glycan)
  # Handle modifications
  glycan = re.sub(r'\d{,2}%', '', glycan)  # [50%Ac(a1-2)] into [Ac(a1-2)]
  glycan = re.sub(r'(?<!\d),(?!\d)', '][', glycan)  # Replace only commas not flanked by digits
  glycan = re.sub(r'<<([A-Za-z0-9]+)\(([ab\?])(\d+)-\d+\)\|([A-Za-z0-9]+)\([ab\?]\d+-\d+\)>>', r'\1(\2\3-?)', glycan)  # <<Rha(a1-3)|Rha(a1-4)>> to Rha(a1-?)
  old_glycan = ""
  while glycan != old_glycan:
    old_glycan = glycan
    glycan = re.sub(r'\[([^]]+\([?ab]?\d+-([\d\?]+)\))\]([A-Z][A-Za-z1-9]*)',
                 lambda m: f"{m.group(3)}{m.group(2)}{m.group(1).split('(')[0]}" if (m.group(1).split('(')[0] not in lib and m.group(1).count('(') == 1) else f"[{m.group(1)}]{m.group(3)}",
                 glycan)  # [Ac(?1-3)]Fruf to Fruf3Ac
  glycan = re.sub(r'\[([A-Za-z0-9]+)\(\?(\d+)-(\d+)\)([A-Za-z0-9]+)',
                lambda m: f"[{m.group(4)}{m.group(3)}{m.group(1)}" if m.group(1) not in lib else f"[{m.group(1)}(?{m.group(2)}-{m.group(3)}){m.group(4)}",
                glycan)  # [Ac(?1-2)Rha to [Rha2Ac
  glycan = re.sub(r'\[([1-9][SP])\]\[([1-9][SP])\]([A-Z][^\(^\[]+)',
                lambda m: f"{m.group(3)}{min(m.group(1), m.group(2))}{max(m.group(1), m.group(2))}",
                glycan)  # [6S][4S]Gal to Gal4S6S
  glycan = re.sub(r'\?([A-Z])', r'O\1', glycan)  # Kdo?Ac to KdoOAc
  glycan = re.sub(r'\[([1-9]?[SP])\]([A-Z][^\(^\[]+)', r'\2\1', glycan)  # [S]Gal to GalS
  glycan = re.sub(r'(\)|\]|^)([1-9]?[SP])([A-Z][^\(^\[]+)', r'\1\3\2', glycan)  # )SGal to )GalS
  glycan = re.sub(r'(\-ol)([0-9]?[SP])', r'\2\1', glycan)  # Gal-olS to GalS-ol
  glycan = re.sub(r'(\[|\)|\]|^)([1-9]?[SP])(?!en)([A-Za-z]+)', r'\1\3\2', glycan)  # SGalNAc to GalNAcS
  glycan = re.sub(r'([1-9]?[SP])-([A-Za-n]+)', r'\2\1', glycan)  # S-Gal to GalS
  # Handle malformed things like Gal-GlcNAc in an otherwise properly formatted string
  glycan = re.sub(r'([a-z])\?', r'\1(?', glycan)
  glycan = re.sub(r'(~\([c-z])([1-2])-', r'\1(?\2-', glycan)
  glycan = re.sub(r'-([\?2-9])([A-Z])', r'-\1)\2', glycan)
  glycan = re.sub(r'([\?2-9])([\[\]])', r'\1)\2', glycan)
  # Floating bits
  if '+' in glycan:
    prefix = glycan[:glycan.index('+')]
    if prefix == 'S':
      glycan = glycan.replace('S+', 'OS+')
    elif '-' not in prefix:
      glycan = glycan.replace('+', '(?1-?)+')
    glycan = '{'+glycan.replace('+', '}')
  post_process = {'5Ac(?': '5Ac(a', '5Gc(?': '5Gc(a', '5Ac(a1': '5Ac(a2', '5Gc(a1': '5Gc(a2', 'Fuc(?': 'Fuc(a',
                  'GalS': 'GalOS', 'GlcS': 'GlcOS', 'GlcNAcS': 'GlcNAcOS', 'GalNAcS': 'GalNAcOS', 'SGal': 'GalOS', 'Kdn(?': 'Kdn(a',
                  'Kdn(a1': 'Kdn(a2', 'N2Ac(': 'NAc(', 'N2Ac3': 'NAc3', '(x': '(?'}
  glycan = multireplace(glycan, post_process)
  glycan = re.sub(r'(?:[ab])?-+$', '', glycan)  # Remove endings like Glcb-
  glycan = sanitize_iupac(glycan)
  # Assume every non-lib "monosaccharide" at the reducing end is a modification and glue it to the preceding monosaccharide
  glycan = re.sub(r'\(([ab\?][1-2])-([1])\)([A-Z][A-Za-z\-]*$)', lambda m: f'{m.group(2)}{m.group(3)}' if m.group(3) not in lib else f'({m.group(1)}-{m.group(2)}){m.group(3)}', glycan)
  glycan = re.sub(r'([\w-]+)(?:-ol)?\(([\w\?])(\d+)-P-(\d+)\)', lambda m: f"{m.group(1)}{m.group(3)}P({m.group(2)}{m.group(3)}-{m.group(4)})", glycan)  # Rha(a1-P-4) into Rha1P(a1-4)
  glycan, repeat = transform_repeat_glycan(glycan)
  glycan = re.sub(r"n\=[\d\?\-]+\/", "", glycan)  # Strip out internal repeats such as n=?/
  glycan = re.sub(r"\/([A-Z])", r"\1", glycan)  # Strip out any remaining / from internal repeats
  # Canonicalize branch ordering
  if '[' in glycan and not glycan.startswith('[') and ']' in glycan and not repeat:
    from glycowork.motif.graph import glycan_to_nxGraph, graph_to_string
    glycan = graph_to_string(glycan_to_nxGraph.__wrapped__(glycan))
  if '{' in glycan:
    floating_bits = re.findall(r'\{.*?\}', glycan)
    sorted_floating_bits = ''.join(sorted(floating_bits, key = len, reverse = True))
    glycan = sorted_floating_bits + glycan[glycan.rfind('}')+1:]
  if glycan.count('[') != glycan.count(']'):
    raise ValueError(f"Mismatching brackets in formatted glycan string: {glycan}")
  return glycan


def rescue_glycans(func: Callable # Function to wrap
                 ) -> Callable: # Wrapped function handling formatting issues
  "Decorator for handling malformed glycan sequences"
  @wraps(func)
  def wrapper(*args, **kwargs):
    try:
      # Try running the original function
      return func(*args, **kwargs)
    except Exception:
      # If an error occurs, attempt to rescue the glycan sequences
      rescued_args = [canonicalize_iupac(arg) if isinstance(arg, str) else [canonicalize_iupac(a) for a in arg] if isinstance(arg, list) and arg and isinstance(arg[0], str) else arg for arg in args]
      # After rescuing, attempt to run the function again
      return func(*rescued_args, **kwargs)
  return wrapper


def rescue_compositions(func: Callable # Function to wrap
                      ) -> Callable: # Wrapped function handling composition format issues
  "Decorator for handling malformed glycan compositions"
  @wraps(func)
  def wrapper(*args, **kwargs):
    try:
      # Try running the original function
      return func(*args, **kwargs)
    except Exception:
      # If an error occurs, attempt to rescue the glycan compositions
      rescued_args = [canonicalize_composition(arg) if isinstance(arg, str) else arg for arg in args]
      # After rescuing, attempt to run the function again
      return func(*rescued_args, **kwargs)
  return wrapper


def equal_repeats(r1: str, # First glycan sequence
                 r2: str # Second glycan sequence
                ) -> bool: # True if repeats are shifted versions
  "Check whether two repeat units could stem from the same repeating structure"
  if r1 == r2:
    return True
  r1_long = r1[:r1.rindex(')')+1] * 2
  return any(r1_long[i:i + len(r2)] == r2 for i in range(len(r1)))


def infer_features_from_composition(comp: Dict[str, int] # Composition dictionary of monosaccharide:count
                                 ) -> Dict[str, int]: # Dictionary of features and presence/absence
  "Extract higher-order glycan features from a composition"
  feature_dic = {'complex': 0, 'high_Man': 0, 'hybrid': 0, 'antennary_Fuc': 0}
  if comp.get('A', 0) + comp.get('G', 0) > 1 or comp.get('Neu5Ac', 0) + comp.get('Neu5Gc', 0) > 1:
    feature_dic['complex'] = 1
  if (comp.get('H', 0) > 5 and comp.get('N', 0) == 2) or (comp.get('Hex', 0) > 5 and comp.get('HexNAc', 0) == 2):
    feature_dic['high_Man'] = 1
  if (comp.get('A', 0) + comp.get('G', 0) < 2 and comp.get('H', 0) > 4) or (comp.get('Neu5Ac', 0) + comp.get('Neu5Gc', 0) < 2 and comp.get('Hex', 0) > 4):
    feature_dic['hybrid'] = 1
  if comp.get('dHex', 0) > 1 or comp.get('F', 0) > 1:
    feature_dic['antennary_Fuc'] = 1
  return feature_dic


@rescue_compositions
def parse_glycoform(glycoform: Union[str, Dict[str, int]], # Composition in H5N4F1A2 format or dict
                   glycan_features: List[str] = ['H', 'N', 'A', 'F', 'G'] # Features to extract
                  ) -> Dict[str, int]: # Dictionary of feature counts
  "Convert composition like H5N4F1A2 into monosaccharide counts"
  if isinstance(glycoform, dict):
    if not any(f in glycoform.keys() for f in glycan_features):
      mapping = {'Hex': 'H', 'HexNAc': 'N', 'dHex': 'F', 'Neu5Ac': 'A', 'Neu5Gc': 'G'}
      glycoform = {mapping.get(k, k): v for k, v in glycoform.items()}
    components = {k: glycoform.get(k, 0) for k in glycan_features}
    return components | infer_features_from_composition(components)
  components = {c: 0 for c in glycan_features}
  matches = re.finditer(r'([HNAFG])(\d+)', glycoform)
  for match in matches:
    components[match.group(1)] = int(match.group(2))
  return components | infer_features_from_composition(components)


def process_for_glycoshift(df: pd.DataFrame # Dataset with protein_site_composition index
                         ) -> Tuple[pd.DataFrame, List[str]]: # (Modified dataset with new columns for protein_site, composition, and composition counts, glycan features)
  "Extract and format compositions in glycoproteomics dataset"
  df['Glycosite'] = [k.split('_')[0] + '_' + k.split('_')[1] for i, k in enumerate(df.index)]
  if '[' in df.index[0]:
    comps = ['['+k.split('[')[1] for k in df.index]
    comps = [list(map(int, re.findall(r'\d+', s))) for s in comps]
    df['Glycoform'] = [f'H{c[0]}N{c[1]}F{c[3]}A{c[2]}' for c in comps]
    glycan_features = ['H', 'N', 'A', 'F', 'G']
  else:
    df['Glycoform'] = [canonicalize_composition(k.split('_')[-1]) for k in df.index]
    glycan_features = set(unwrap([list(c.keys()) for c in df.Glycoform]))
  org_cols = df.columns.tolist()
  df = df.join(df['Glycoform'].apply(parse_glycoform, glycan_features = glycan_features).apply(pd.Series))
  return df, [c for c in df.columns if c not in org_cols]
