import pandas as pd
import numpy as np
import copy
import math
import re
from functools import wraps
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from scipy.special import gammaln
from scipy.stats import wilcoxon, rankdata, norm
from statsmodels.stats.multitest import multipletests
from glycowork.glycan_data.loader import unwrap, multireplace, find_nth, find_nth_reverse, linkages, Hex, HexNAc, dHex, Sia, HexA, Pen, hlm, update_cf_for_m_n
rng = np.random.default_rng(42)


def min_process_glycans(glycan_list):
  """converts list of glycans into a nested lists of glycoletters\n
  | Arguments:
  | :-
  | glycan_list (list): list of glycans in IUPAC-condensed format as strings\n
  | Returns:
  | :-
  | Returns list of glycoletter lists
  """
  return [[x for x in multireplace(k, {'[': '', ']': '', '{': '', '}': '', ')': '('}).split('(') if x] for k in glycan_list]


def get_lib(glycan_list):
  """returns dictionary of form glycoletter:index\n
  | Arguments:
  | :-
  | glycan_list (list): list of IUPAC-condensed glycan sequences as strings\n
  | Returns:
  | :-
  | Returns dictionary of form glycoletter:index
  """
  # Convert to glycoletters & flatten & get unique vocab
  lib = sorted(set(unwrap(min_process_glycans(set(glycan_list)))))
  # Convert to dict
  return {k: i for i, k in enumerate(lib)}


def expand_lib(libr, glycan_list):
  """updates libr with newly introduced glycoletters\n
  | Arguments:
  | :-
  | libr (dict): dictionary of form glycoletter:index
  | glycan_list (list): list of IUPAC-condensed glycan sequences as strings\n
  | Returns:
  | :-
  | Returns new lib
  """
  new_libr = get_lib(glycan_list)
  offset = len(libr)
  for k, v in new_libr.items():
    if k not in libr:
      libr[k] = v + offset
  return libr


def in_lib(glycan, libr):
  """checks whether all glycoletters of glycan are in libr\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed nomenclature
  | libr (dict): dictionary of form glycoletter:index\n
  | Returns:
  | :-
  | Returns True if all glycoletters are in libr and False if not
  """
  glycan = min_process_glycans([glycan])[0]
  return set(glycan).issubset(libr.keys())


def get_possible_linkages(wildcard, linkage_list = linkages, libr = None):
  """Retrieves all linkages that match a given wildcard pattern from a list of linkages\n
  | Arguments:
  | :-
  | wildcard (string): The pattern to match, where '?' can be used as a wildcard for any single character.
  | linkage_list (list): List of linkages as strings to search within; default:linkages
  | libr (dict): dictionary of form glycoletter:index\n
  | Returns:
  | :-
  | Returns a list of linkages that match the wildcard pattern.
  """
  pattern = wildcard.replace("?", "[a-zA-Z0-9\?]")
  possible_linkages = [linkage for linkage in linkage_list if re.fullmatch(pattern, linkage)]
  return possible_linkages if libr is None else list(possible_linkages & libr.keys())


def get_possible_monosaccharides(wildcard, libr = None):
  """Retrieves all matching common monosaccharides of a type, given the type\n
  | Arguments:
  | :-
  | wildcard (string): Monosaccharide type, from "HexNAc", "Hex", "dHex", "Sia", "HexA", "Pen"
  | libr (dict): dictionary of form glycoletter:index\n
  | Returns:
  | :-
  | Returns a list of specified monosaccharides of that type
  """
  wildcard_dict = {'Hex': Hex, 'HexNAc': HexNAc, 'dHex': dHex, 'Sia': Sia, 'HexA': HexA, 'Pen': Pen,
                   'Monosaccharide': set().union(*[Hex, HexNAc, dHex, Sia, HexA, Pen])}
  possible_monosaccharides = wildcard_dict.get(wildcard, [])
  return list(possible_monosaccharides) if libr is None else list(possible_monosaccharides & libr.keys())


def bracket_removal(glycan_part):
  """iteratively removes (nested) branches between start and end of glycan_part\n
  | Arguments:
  | :-
  | glycan_part (string): residual part of a glycan from within glycan_to_graph\n
  | Returns:
  | :-
  | Returns glycan_part without interfering branches
  """
  regex = re.compile(r'\[[^\[\]]+\]')
  while regex.search(glycan_part):
    glycan_part = regex.sub('', glycan_part)
  return glycan_part


def find_isomorphs(glycan):
  """returns a set of isomorphic glycans by swapping branches etc.\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format\n
  | Returns:
  | :-
  | Returns list of unique glycan notations (strings) for a glycan in IUPAC-condensed
  """
  floaty = False
  if '{' in glycan:
    floaty = glycan[:glycan.rindex('}')+1]
    glycan = glycan[glycan.rindex('}')+1:]
  out_list = {glycan}
  # Starting branch swapped with next side branch
  if '[' in glycan and glycan.index('[') > 0:
    if not bool(re.search(r'\[[^\]]+\[', glycan)):
      out_list.add(re.sub(r'^(.*?)\[(.*?)\]', r'\2[\1]', glycan, 1))
    elif not bool(re.search(r'\[[^\]]+\[', glycan[find_nth(glycan, ']', 2):])) and bool(re.search(r'\[[^\]]+\[', glycan[:find_nth(glycan, '[', 3)])):
      out_list.add(re.sub(r'^(.*?)\[(.*?)(\])((?:[^\[\]]|\[[^\[\]]*\])*)\]', r'\2\3\4[\1]', glycan, 1))
  # Double branch swap
  temp = {re.sub(r'\[([^[\]]+)\]\[([^[\]]+)\]', r'[\2][\1]', k) for k in out_list if '][' in k}
  out_list.update(temp)
  temp = set()
  # Starting branch swapped with next side branch again to also include double branch swapped isomorphs
  for k in out_list:
    if not bool(re.search(r'\[[^\]]+\[', k)):
      temp.add(re.sub(r'^(.*?)\[(.*?)\]', r'\2[\1]', k, 1))
    if k.count('[') > 1 and k.index('[') > 0 and find_nth(k, '[', 2) > k.index(']') and (find_nth(k, ']', 2) < find_nth(k, '[', 3) or k.count('[') == 2):
      temp.add(re.sub(r'^(.*?)\[(.*?)\](.*?)\[(.*?)\]', r'\4[\1[\2]\3]', k, 1))
  out_list.update(temp)
  out_list = {k for k in out_list if not any([j in k for j in ['[[', ']]']])}
  if floaty:
    out_list = {floaty+k for k in out_list}
  return list(out_list)


def presence_to_matrix(df, glycan_col_name = 'glycan', label_col_name = 'Species'):
  """converts a dataframe such as df_species to absence/presence matrix\n
  | Arguments:
  | :-
  | df (dataframe): dataframe with glycan occurrence, rows are glycan-label pairs
  | glycan_col_name (string): column name under which glycans are stored; default:glycan
  | label_col_name (string): column name under which labels are stored; default:Species\n
  | Returns:
  | :-
  | Returns pandas dataframe with labels as rows and glycan occurrences as columns
  """
  # Create a grouped dataframe where we count the occurrences of each glycan in each species group
  grouped_df = df.groupby([label_col_name, glycan_col_name]).size().unstack(fill_value = 0)
  # Sort the index and columns
  grouped_df = grouped_df.sort_index().sort_index(axis = 1)
  return grouped_df


def find_matching_brackets_indices(s):
  stack = []
  matching_indices = []
  for i, c in enumerate(s):
    if c == '[':
      stack.append(i)
    elif c == ']':
      if stack:
        opening_index = stack.pop()
        matching_indices.append((opening_index, i))
  if stack:
    print("Unmatched opening brackets:", [s[i] for i in stack])
    return None
  else:
    matching_indices.sort()
    return matching_indices


def choose_correct_isoform(glycans, reverse = False):
  """given a list of glycan branch isomers, this function returns the correct isomer\n
  | Arguments:
  | :-
  | glycans (list): glycans in IUPAC-condensed nomenclature
  | reverse (bool): whether to return the correct isomer (False) or everything except the correct isomer (True); default:False\n
  | Returns:
  | :-
  | Returns the correct isomer as a string (if reverse=False; otherwise it returns a list of strings)
  """
  glycans = list(set(glycans))
  if len(glycans) == 1:
    return glycans[0]
  if not any(['[' in g for g in glycans]):
    return [] if reverse else glycans
  floaty = False
  if '{' in glycans[0]:
    floaty = glycans[0][:glycans[0].rindex('}')+1]
    glycans = [k[k.rindex('}')+1:] for k in glycans]
  # Heuristic: main chain should contain the most monosaccharides of all chains
  mains = [bracket_removal(g).count('(') for g in glycans]
  max_mains = max(mains)
  glycans2 = [g for k, g in enumerate(glycans) if mains[k] == max_mains]
  # Handle neighboring branches
  kill_list = set()
  for g in glycans2:
    if '][' in g:
      try:
        match = re.search(r'\[([^[\]]+)\]\[([^[\]]+)\]', g)
        count1, count2 = match.group(1).count('('), match.group(2).count('(')
        if count1 < count2:
          kill_list.add(g)
        elif count1 == count2 and int(match.group(1)[-2]) > int(match.group(2)[-2]):
          kill_list.add(g)
      except:
        pass
  glycans2 = [k for k in glycans2 if k not in kill_list]
  # Choose the isoform with the longest main chain before the branch & or the branch ending in the smallest number if all lengths are equal
  if len(glycans2) > 1:
    candidates = {k: find_matching_brackets_indices(k) for k in glycans2}
    prefix = [min_process_glycans([k[j[0]+1:j[1]] for j in candidates[k]]) for k in candidates.keys()]
    prefix = [np.argmax([len(j) for j in k]) for k in prefix]
    prefix = min_process_glycans([k[:candidates[k][prefix[i]][0]] for i, k in enumerate(candidates.keys())])
    branch_endings = [k[-1][-1] if k[-1][-1].isdigit() else 10 for k in prefix]
    if len(set(branch_endings)) == 1:
      branch_endings = [ord(k[0][0]) for k in prefix]
    branch_endings = [int(k) for k in branch_endings]
    min_ending = min(branch_endings)
    glycans2 = [g for k, g in enumerate(glycans2) if branch_endings[k] == min_ending]
    if len(glycans2) > 1:
        preprefix = min_process_glycans([glyc[:glyc.index('[')] for glyc in glycans2])
        branch_endings = [k[-1][-1] if k[-1][-1].isdigit() else 10 for k in preprefix]
        branch_endings = [int(k) for k in branch_endings]
        min_ending = min(branch_endings)
        glycans2 = [g for k, g in enumerate(glycans2) if branch_endings[k] == min_ending]
        if len(glycans2) > 1:
          correct_isoform = sorted(glycans2)[0]
        else:
          correct_isoform = glycans2[0]
    else:
        correct_isoform = glycans2[0]
  else:
    correct_isoform = glycans2[0]
  if floaty:
    correct_isoform = floaty + correct_isoform
  if reverse:
    glycans.remove(correct_isoform)
    correct_isoform = glycans
  return correct_isoform


def enforce_class(glycan, glycan_class, conf = None, extra_thresh = 0.3):
  """given a glycan and glycan class, determines whether glycan is from this class\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed nomenclature
  | glycan_class (string): glycan class in form of "O", "N", "free", or "lipid"
  | conf (float): prediction confidence; can be used to override class
  | extra_thresh (float): threshold to override class; default:0.3\n
  | Returns:
  | :-
  | Returns True if glycan is in glycan class and False if not
  """
  pools = {
    'O': ['GalNAc', 'GalNAcOS', 'GalNAc4S', 'GalNAc6S', 'Man', 'Fuc', 'Gal', 'GlcNAc', 'GlcNAcOS', 'GlcNAc6S'],
    'N': ['GlcNAc'],
    'free': ['Glc', 'GlcOS', 'Glc3S', 'GlcNAc', 'GlcNAcOS', 'Gal', 'GalOS', 'Gal3S', 'Ins'],
    'lipid': ['Glc', 'GlcOS', 'Glc3S', 'GlcNAc', 'GlcNAcOS', 'Gal', 'GalOS', 'Gal3S', 'Ins'],
    }
  pool = pools.get(glycan_class, [])
  truth = any([glycan.endswith(k) for k in pool])
  if truth and glycan_class in {'free', 'lipid', 'O'}:
    truth = not any(glycan.endswith(k) for k in ['GlcNAc(b1-4)GlcNAc', '[Fuc(a1-6)]GlcNAc'])
  if not truth and conf:
    return conf > extra_thresh
  return truth


def canonicalize_composition(comp):
  """converts a composition from any common format into the dictionary that is optimized for glycowork\n
  | Arguments:
  | :-
  | comp (string): composition formatted either in the style of HexNAc2Hex1Fuc3Neu5Ac1 or N2H1F3A1\n
  | Returns:
  | :-
  | Returns composition as a dictionary of style monosaccharide : count
  """
  if '_' in comp:
    values = comp.split('_')
    temp = {"Hex": int(values[0]), "HexNAc": int(values[1]), "Neu5Ac": int(values[2]), "dHex": int(values[3])}
    return {k: v for k, v in temp.items() if v}
  elif comp.isdigit():
    temp = {"Hex": int(comp[0]), "HexNAc": int(comp[1]), "Neu5Ac": int(comp[2]), "dHex": int(comp[3])}
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


def IUPAC_to_SMILES(glycan_list):
  """given a list of IUPAC-condensed glycans, uses GlyLES to return a list of corresponding isomeric SMILES\n
  | Arguments:
  | :-
  | glycan_list (list): list of IUPAC-condensed glycans\n
  | Returns:
  | :-
  | Returns a list of corresponding isomeric SMILES
  """
  try:
    from glyles import convert
  except ImportError:
    raise ImportError("You must install the 'chem' dependencies to use this feature. Try 'pip install glycowork[chem]'.")
  if not isinstance(glycan_list, list):
    raise TypeError("Input must be a list")
  return [convert(g)[0][1] for g in glycan_list]


def linearcode_to_iupac(linearcode):
  """converts a glycan from LinearCode into a barebones IUPAC-condensed version that is cleaned up by canonicalize_iupac\n
  | Arguments:
  | :-
  | linearcode (string): glycan sequence in LinearCode format\n
  | Returns:
  | :-
  | Returns glycan as a string in a barebones IUPAC-condensed form
  """
  replace_dic = {'G': 'Glc', 'ME': 'me', 'M': 'Man', 'A': 'Gal', 'NN': 'Neu5Ac', 'GlcN': 'GlcNAc', 'GN': 'GlcNAc',
                 'GalN': 'GalNAc', 'AN': 'GalNAc', 'F': 'Fuc', 'K': 'Kdn', 'W': 'Kdo', 'L': 'GalA', 'I': 'IdoA', 'PYR': 'Pyr', 'R': 'Araf', 'H': 'Rha',
                 'X': 'Xyl', 'B': 'Rib', 'U': 'GlcA', 'O': 'All', 'E': 'Fruf', '[': '', ']': '', 'me': 'Me', 'PC': 'PCho', 'T': 'Ac'}
  glycan = multireplace(linearcode.split(';')[0], replace_dic)
  return glycan


def iupac_extended_to_condensed(iupac_extended):
  """converts a glycan from IUPAC-extended into a barebones IUPAC-condensed version that is cleaned up by canonicalize_iupac\n
  | Arguments:
  | :-
  | iupac_extended (string): glycan sequence in IUPAC-extended format\n
  | Returns:
  | :-
  | Returns glycan as a string in a barebones IUPAC-condensed form
  """
  # Find all occurrences of the pattern and apply the changes
  def replace_pattern(match):
    # Move the α or β after the next opening parenthesis
    return f"{match.group('after')}{match.group('alpha_beta')}"
  # The regular expression looks for α-D- or β-D- followed by any characters until an open parenthesis
  pattern = re.compile(r"(?P<alpha_beta>[αβab\?])-[DL]-(?P<after>[^\)]*\()")
  # Substitute the pattern in the string with our replace_pattern function
  adjusted_string = pattern.sub(replace_pattern, iupac_extended)
  adjusted_string = re.sub(r"-\(", "(", adjusted_string)
  adjusted_string = re.sub(r"\)-", ")", adjusted_string)
  adjusted_string = re.sub(r"\]-", "]", adjusted_string)
  return adjusted_string[:adjusted_string.rindex('(')]


def glycoct_to_iupac_int(glycoct, mono_replace, sub_replace):
  # Dictionaries to hold the mapping of residues and linkages
  residue_dic = {}
  iupac_parts = defaultdict(list)
  degrees = defaultdict(lambda:1)
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
        tgt = '\n' + str(int(parts[0][:-1])-1)+':'
        pattern = re.escape(tgt)
        matched = re.search(pattern, glycoct)
        if matched:
          start_index = matched.end()
          numerical_part = re.search(r'\d+', glycoct[start_index:])
          res_id = int(numerical_part.group())
        else:
          res_id = max(residue_dic.keys())
        res_type = multireplace(parts[1], sub_replace)
        residue_dic[res_id] = residue_dic[res_id][:-1] + res_type + residue_dic[res_id][-1]
    #linkage
    elif len(line) > 0:
      line = line.replace('-1', '?')
      line = re.sub("\d\|\d", "?", line)
      parts = re.findall(r'(\d+)[do]\(([\d\?]+)\+(\d+)\)(\d+)', line)[0]
      parent_id, child_id = int(parts[0]), int(parts[3])
      link_type = f"{residue_dic.get(child_id, 99)}({parts[2]}-{parts[1]})"
      if link_type.startswith('99') and parts[1] not in ['2', '5']:
        residue_dic[parent_id] = re.sub(r'(O)(?=S|P|Ac)', parts[1], residue_dic[parent_id], count = 1)
      if not link_type.startswith('99'):
        iupac_parts[parent_id].append((f"{parts[2]}-{parts[1]}", child_id))
        degrees[parent_id] += 1
  for r in residue_dic:
    if r not in degrees:
      degrees[r] = 1
  return residue_dic, iupac_parts, degrees


def glycoct_build_iupac(iupac_parts, residue_dic, degrees):
  start = min(residue_dic.keys())
  iupac = residue_dic[start]
  inverted_residue_dic = {}
  inverted_residue_dic.setdefault(residue_dic[start], []).append(start)
  for parent, children in iupac_parts.items():
    child_strings = []
    children_degree = [degrees[c[1]] for c in children]
    children = [x for _, x in sorted(zip(children_degree, children), reverse = True)]
    i = 0
    last_child = 0
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


def glycoct_to_iupac(glycoct):
  """converts a glycan from GlycoCT into a barebones IUPAC-condensed version that is cleaned up by canonicalize_iupac\n
  | Arguments:
  | :-
  | glycoct (string): glycan sequence in GlycoCT format\n
  | Returns:
  | :-
  | Returns glycan as a string in a barebones IUPAC-condensed form
  """
  floating_bits = []
  floating_part = ''
  mono_replace = {'dglc': 'Glc', 'dgal': 'Gal', 'dman': 'Man', 'lgal': 'Fuc', 'dgro': 'Neu',
                  'dxyl': 'Xyl', 'dara': 'Ara', 'HEX': 'Hex', 'lman': 'L-Man'}
  sub_replace = {'n-acetyl': 'NAc', 'sulfate': 'OS', 'phosphate': 'OP', 'n-glycolyl': '5Gc',
                 'acetyl': 'OAc'}
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
  iupac = re.sub(r'(\?)(?=S|P)', 'O', iupac)
  iupac = re.sub(r'([1-9\?O](S|P|Ac))NAc', r'NAc\1', iupac)
  if ']' in iupac and iupac.index(']') < iupac.index('['):
    iupac = iupac.replace(']', '', 1)
  iupac = iupac.replace('[[', '[').replace(']]', ']').replace('Neu(', 'Kdn(')
  return iupac


def wurcs_to_iupac(wurcs):
  """converts a glycan from WURCS into a barebones IUPAC-condensed version that is cleaned up by canonicalize_iupac\n
  | Arguments:
  | :-
  | wurcs (string): glycan sequence in WURCS format\n
  | Returns:
  | :-
  | Returns glycan as a string in a barebones IUPAC-condensed form
  """
  wurcs = wurcs[wurcs.index('/')+1:]
  pattern = r'\b([a-z])\d(?:\|\1\d)+\}?|\b[a-z](\d)(?:\|[a-z]\2)+\}?'
  additional_pattern = r'\b([a-z])\?(?:\|\w\?)+\}?'
  def replacement(match):
    return f'{match.group(1)}?' if match.group(1) else f'?{match.group(2)}'
  wurcs = re.sub(pattern, replacement, wurcs)
  wurcs = re.sub(additional_pattern, '?', wurcs)
  floating_part = ''
  floating_parts = []
  monosaccharide_mapping = {
    'a2122h-1b_1-5_2*NCC/3=O': 'GlcNAcb', 'a2112h-1a_1-5_2*NCC/3=O': 'GalNAca',
    'a1122h-1b_1-5': 'Manb', 'Aad21122h-2a_2-6_5*NCC/3=O': 'Neu5Aca',
    'a1122h-1a_1-5': 'Mana', 'a2112h-1b_1-5': 'Galb', 'Aad21122h-2a_2-6_5*NCCO/3=O': 'Neu5Gca',
    'a2112h-1b_1-5_2*NCC/3=O_?*OSO/3=O/3=O': 'GalNAcOSb', 'a2112h-1b_1-5_2*NCC/3=O': 'GalNAcb',
    'a1221m-1a_1-5': 'Fuca', 'a2122h-1b_1-5_2*NCC/3=O_6*OSO/3=O/3=O': 'GlcNAc6Sb', 'a212h-1b_1-5': 'Xylb',
    'axxxxh-1b_1-5_2*NCC/3=O': 'HexNAcb', 'a2122h-1x_1-5_2*NCC/3=O': 'GlcNAc?', 'a2112h-1x_1-5': 'Gal?',
    'Aad21122h-2a_2-6': 'Kdna', 'a2122h-1a_1-5_2*NCC/3=O': 'GlcNAca', 'a2112h-1a_1-5': 'Gala',
    'a1122h-1x_1-5': 'Man?', 'Aad21122h-2x_2-6_5*NCCO/3=O': 'Neu5Gca', 'Aad21122h-2x_2-6_5*NCC/3=O': 'Neu5Aca',
    'a1221m-1x_1-5': 'Fuca', 'a212h-1x_1-5': 'Xyl?', 'a122h-1x_1-5': 'Ara?', 'a2122A-1b_1-5': 'GlcAb',
    'a2112h-1b_1-5_3*OC': 'Gal3Meb', 'a1122h-1a_1-5_2*NCC/3=O': 'ManNAca', 'a2122h-1x_1-5': 'Glc?',
    'axxxxh-1x_1-5_2*NCC/3=O': 'HexNAc?', 'axxxxh-1x_1-5': 'Hex?', 'a2112h-1b_1-4': 'Galfb',
    'a2122h-1x_1-5_2*NCC/3=O_6*OSO/3=O/3=O': 'GlcNAc6Sb', 'a2112h-1x_1-5_2*NCC/3=O': 'GalNAc?',
    'axxxxh-1a_1-5_2*NCC/3=O': 'HexNAca', 'Aad21122h-2a_2-6_4*OCC/3=O_5*NCC/3=O': 'Neu4Ac5Aca',
    'a2112h-1b_1-5_4*OSO/3=O/3=O': 'Gal4Sb', 'a2122h-1b_1-5_2*NCC/3=O_3*OSO/3=O/3=O': 'GlcNAc3Sb',
    'a2112h-1b_1-5_2*NCC/3=O_4*OSO/3=O/3=O': 'GalNAc4Sb', 'a2122A-1x_1-5_?*OSO/3=O/3=O': 'GlcAOS?'
    }
  parts = wurcs.split('/')
  topology = parts[-1].split('_')
  monosaccharides = '/'.join(parts[1:-2]).strip('[]').split('][')
  connectivity = parts[-2].split('-')
  connectivity = {chr(97 + i): int(num) for i, num in enumerate(connectivity)}
  degrees = {c: ''.join(topology).count(c) for c in connectivity}
  inverted_connectivity = {}
  iupac_parts = []
  for link in topology:
    if '-' not in link:
      return monosaccharide_mapping[monosaccharides[0]]
    source, target = link.split('-')
    source_index, source_carbon = connectivity[source[:-1]], source[-1]
    source_mono = monosaccharide_mapping[monosaccharides[int(source_index)-1]]
    if target[0] == '?':
      floating_part += f"{'{'}{source_mono}(1-{target[1:]}){'}'}"
      floating_parts.append(source[0])
      continue
    target_index, target_carbon = connectivity[target[0]], target[1:]
    target_mono = monosaccharide_mapping[monosaccharides[int(target_index)-1]]
    if '?' in target:
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
  iupac = iupac[:-1]
  iupac = iupac.strip('[]')
  iupac = iupac.replace('}[', '}').replace('{[', '{')
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
  return iupac


def oxford_to_iupac(oxford):
  """converts a glycan from Oxford into a barebones IUPAC-condensed version that is cleaned up by canonicalize_iupac\n
  | Arguments:
  | :-
  | oxford (string): glycan sequence in Oxford format\n
  | Returns:
  | :-
  | Returns glycan as a string in a barebones IUPAC-condensed form
  """
  oxford = re.sub(r'\([^)]*\)', '', oxford)
  antennae = {}
  iupac = "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc"
  mapping_dict = {"A": "GlcNAc(b1-?)", "G": "Gal(b1-?)", "S": "Neu5Ac(a2-?)",
                  "Sg": "Neu5Gc(a2-?)", "Ga": "Gal(a1-?)", "GalNAc": "GalNAc(?1-?)",
                  "Lac": "Gal(b1-?)GlcNAc(b1-?)", "F": "Fuc(a1-?)"}
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
  elif 'F' in oxford:
    antennae["F"] = int(oxford[oxford.index("F")+1])
  if 'M' in oxford:
    M_count = int(oxford[oxford.index("M")+1]) - 3
    for m in range(M_count):
      iupac = "{Man(a1-?)}" + iupac
    return iupac
  branches = {"A": int(oxford[oxford.index("A")+1]) if "A" in oxford else 0,
              "G": int(oxford[oxford.index("G")+1]) if "G" in oxford and oxford[oxford.index("G")+1] != 'a' else 0,
              "S": int(oxford[oxford.index("S")+1]) if "S" in oxford and oxford[oxford.index("S")+1] != 'g' else 0}
  extras = {"Sg": int(oxford[oxford.index("Sg")+2]) if "Sg" in oxford else 0,
            "Ga": int(oxford[oxford.index("Ga")+2]) if "Ga" in oxford else 0,
            "Lac": int(oxford[oxford.index("Lac")+3]) if "Lac" in oxford else 0}
  built_branches = []
  while sum(branches.values()) > 0:
    temp = ''
    for c in ["S", "G", "A"]:
      if branches[c] > 0:
        temp += mapping_dict[c]
        branches[c] -= 1
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
      v -= 1
  if antennae:
    for k, v in antennae.items():
      while v > 0:
        split = iupac.index("Gal(b1-?)Glc")
        iupac = iupac[:split+len("Gal(b1-?)")] + "[" + mapping_dict[k] + "]" + iupac[split+len("Gal(b1-?)"):]
        v -= 1
  return iupac.strip('[]')


def check_nomenclature(glycan):
  """checks whether the proposed glycan has the correct nomenclature for glycowork\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | Returns:
  | :-
  | Prints the reason why it's not convertable
  """
  if '@' in glycan:
    print("Seems like you're using SMILES. We currently can only convert IUPAC-->SMILES; not the other way around.")
    return
  if not isinstance(glycan, str):
    print("You need to format your glycan sequences as strings.")
    return
  return


def canonicalize_iupac(glycan):
  """converts a glycan from IUPAC-extended, LinearCode, GlycoCT, and WURCS into the exact IUPAC-condensed version that is optimized for glycowork\n
  | Arguments:
  | :-
  | glycan (string): glycan sequence; some rare post-biosynthetic modifications could still be an issue\n
  | Returns:
  | :-
  | Returns glycan as a string in canonicalized IUPAC-condensed
  """
  # Check for different nomenclatures: LinearCode, IUPAC-extended, GlycoCT, WURCS, Oxford
  if ';' in glycan:
    glycan = linearcode_to_iupac(glycan)
  elif '-D-' in glycan:
    glycan = iupac_extended_to_condensed(glycan)
  elif 'RES' in glycan:
    glycan = glycoct_to_iupac(glycan)
  elif '=' in glycan:
    glycan = wurcs_to_iupac(glycan)
  elif not isinstance(glycan, str) or any([k in glycan for k in ['@']]):
    check_nomenclature(glycan)
    return
  elif glycan[-1].isdigit() and 'e' not in glycan and '-' not in glycan:
    glycan = oxford_to_iupac(glycan)
  # Canonicalize usage of monosaccharides and linkages
  replace_dic = {'Nac': 'NAc', 'AC': 'Ac', 'Nc': 'NAc', 'NeuAc': 'Neu5Ac', 'NeuNAc': 'Neu5Ac', 'NeuGc': 'Neu5Gc',
                 '\u03B1': 'a', '\u03B2': 'b', 'N(Gc)': 'NGc', 'GL': 'Gl', 'GaN': 'GalN', '(9Ac)': '9Ac',
                 'KDN': 'Kdn', 'OSO3': 'S', '-O-Su-': 'S', '(S)': 'S', 'H2PO3': 'P', '(P)': 'P',
                 '–': '-', ' ': '', ',': '-', 'α': 'a', 'β': 'b', '.': '', '((': '(', '))': ')', '→': '-',
                 'Glcp': 'Glc', 'Galp': 'Gal', 'Manp': 'Man', 'Fucp': 'Fuc', 'Neup': 'Neu', 'a?': 'a1'}
  glycan = multireplace(glycan, replace_dic)
  # Trim linkers
  if '-' in glycan:
    if bool(re.search(r'[a-z]\-[a-zA-Z]', glycan[glycan.rindex('-')-1:])) and 'ol' not in glycan:
      glycan = glycan[:glycan.rindex('-')]
  # Canonicalize usage of brackets and parentheses
  if bool(re.search(r'\([A-Zd3-9]', glycan)):
    glycan = glycan.replace('(', '[').replace(')', ']')
  # Canonicalize linkage uncertainty
  # Open linkages (e.g., "c-")
  glycan = re.sub(r'([a-z])\-([A-Z])', r'\1?1-?\2', glycan)
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
  if '-' not in glycan:
      glycan = re.sub(r'(a|b)(\d)', r'\g<1>1-\g<2>', glycan)
  # Smudge uncertainty
  while '/' in glycan:
    glycan = glycan[:glycan.index('/')-1] + '?' + glycan[glycan.index('/')+2:]
  # Introduce parentheses for linkages
  if '(' not in glycan and len(glycan) > 6:
    for k in range(1, glycan.count('-')+1):
      idx = find_nth(glycan, '-', k)
      if (glycan[idx-1].isnumeric()) and (glycan[idx+1].isnumeric() or glycan[idx+1] == '?'):
        glycan = glycan[:idx-2] + '(' + glycan[idx-2:idx+2] + ')' + glycan[idx+2:]
      elif (glycan[idx-1].isnumeric()) and bool(re.search(r'[A-Z]', glycan[idx+1])):
        glycan = glycan[:idx-2] + '(' + glycan[idx-2:idx+1] + '?)' + glycan[idx+1:]
  # Canonicalize reducing end
  if bool(re.search(r'[a-z]ol', glycan)):
    if 'Glcol' not in glycan:
      glycan = glycan[:-2]
    else:
      glycan = glycan[:-2] + '-ol'
  if glycan[-1] in 'ab' and glycan[-3:] not in ['Rha', 'Ara']:
    glycan = glycan[:-1]
  # Handle modifications
  if bool(re.search(r'\[[1-9]?[SP]\][A-Z][^\(^\[]+', glycan)):
    glycan = re.sub(r'\[([1-9]?[SP])\]([A-Z][^\(^\[]+)', r'\2\1', glycan)
  if bool(re.search(r'(\)|\]|^)[1-9]?[SP][A-Z][^\(^\[]+', glycan)):
    glycan = re.sub(r'([1-9]?[SP])([A-Z][^\(^\[]+)', r'\2\1', glycan)
  if bool(re.search(r'\-ol[0-9]?[SP]', glycan)):
    glycan = re.sub(r'(\-ol)([0-9]?[SP])', r'\2\1', glycan)
  if bool(re.search(r'(\[|\)|\]|\^)[1-9]?[SP](?!en)[A-Za-z]+', glycan)):
    glycan = re.sub(r'([1-9]?[SP])(?!en)([A-Za-z]+)', r'\2\1', glycan)
  if bool(re.search(r'[1-9]?[SP]-[A-Za-z]+', glycan)):
    glycan = re.sub(r'([1-9]?[SP])-([A-Za-z]+)', r'\2\1', glycan)
  post_process = {'5Ac(?1': '5Ac(a2', '5Gc(?1': '5Gc(a2', '5Ac(a1': '5Ac(a2', '5Gc(a1': '5Gc(a2', 'Fuc(?': 'Fuc(a',
                  'GalS': 'GalOS', 'GlcNAcS': 'GlcNAcOS', 'GalNAcS': 'GalNAcOS', 'SGal': 'GalOS', 'Kdn(?1': 'Kdn(a2',
                  'Kdn(a1': 'Kdn(a2'}
  glycan = multireplace(glycan, post_process)
  # Canonicalize branch ordering
  if '[' in glycan:
    isos = find_isomorphs(glycan)
    glycan = choose_correct_isoform(isos)
  # Floating bits
  if '+' in glycan:
    glycan = '{'+glycan.replace('+', '}')
  if '{' in glycan:
    floating_bits = re.findall(r'\{.*?\}', glycan)
    sorted_floating_bits = ''.join(sorted(floating_bits, key = len, reverse = True))
    glycan = sorted_floating_bits + glycan[glycan.rfind('}')+1:]
  return glycan


def rescue_glycans(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    try:
      # Try running the original function
      return func(*args, **kwargs)
    except Exception as e:
      # If an error occurs, attempt to rescue the glycan sequences
      rescued_args = [canonicalize_iupac(arg) if isinstance(arg, str) else [canonicalize_iupac(a) for a in arg] if isinstance(arg, list) and isinstance(arg[0], str) else arg for arg in args]
      # After rescuing, attempt to run the function again
      return func(*rescued_args, **kwargs)
  return wrapper


def rescue_compositions(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    try:
      # Try running the original function
      return func(*args, **kwargs)
    except Exception as e:
      # If an error occurs, attempt to rescue the glycan compositions
      rescued_args = [canonicalize_composition(arg) if isinstance(arg, str) else arg for arg in args]
      # After rescuing, attempt to run the function again
      return func(*rescued_args, **kwargs)
  return wrapper


def cohen_d(x, y, paired = False):
  """calculates effect size between two groups\n
  | Arguments:
  | :-
  | x (list or 1D-array): comparison group containing numerical data
  | y (list or 1D-array): comparison group containing numerical data
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False\n
  | Returns:
  | :-
  | Returns Cohen's d (and its variance) as a measure of effect size (0.2 small; 0.5 medium; 0.8 large)
  """
  if paired:
    assert len(x) == len(y), "For paired samples, the size of x and y should be the same"
    diff = np.array(x) - np.array(y)
    n = len(diff)
    d = np.mean(diff) / np.std(diff, ddof = 1)
    var_d = 1 / n + d**2 / (2 * n)
  else:
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    d = (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof = 1) ** 2 + (ny-1)*np.std(y, ddof = 1) ** 2) / dof)
    var_d = (nx + ny) / (nx * ny) + d**2 / (2 * (nx + ny))
  return d, var_d


def mahalanobis_distance(x, y, paired = False):
  """calculates effect size between two groups in a multivariate comparison\n
  | Arguments:
  | :-
  | x (list or 1D-array or dataframe): comparison group containing numerical data
  | y (list or 1D-array or dataframe): comparison group containing numerical data
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False\n
  | Returns:
  | :-
  | Returns Mahalanobis distance as a measure of effect size
  """
  if paired:
    assert x.shape == y.shape, "For paired samples, the size of x and y should be the same"
    x = np.array(x) - np.array(y)
    y = np.zeros_like(x)
  if isinstance(x, pd.DataFrame):
    x = x.values
  if isinstance(y, pd.DataFrame):
    y = y.values
  pooled_cov_inv = np.linalg.pinv((np.cov(x) + np.cov(y)) / 2)
  diff_means = (np.mean(y, axis = 1) - np.mean(x, axis = 1)).reshape(-1, 1)
  mahalanobis_d = np.sqrt(np.clip(diff_means.T @ pooled_cov_inv @ diff_means, 0, None))
  return mahalanobis_d[0][0]


def mahalanobis_variance(x, y, paired = False):
  """Estimates variance of Mahalanobis distance via bootstrapping\n
  | Arguments:
  | :-
  | x (list or 1D-array or dataframe): comparison group containing numerical data
  | y (list or 1D-array or dataframe): comparison group containing numerical data
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False\n
  | Returns:
  | :-
  | Returns Mahalanobis distance as a measure of effect size
  """
  # Combine gp1 and gp2 into a single matrix
  data = np.concatenate((x.T, y.T), axis = 0)
  # Perform bootstrap resampling
  n_iterations = 1000
  # Initialize an empty array to store the bootstrap samples
  bootstrap_samples = np.empty(n_iterations)
  size_x = x.shape[1]
  for i in range(n_iterations):
      # Generate a random bootstrap sample
      sample = data[rng.choice(range(data.shape[0]), size = data.shape[0], replace = True)]
      # Split the bootstrap sample into two groups
      x_sample = sample[:size_x]
      y_sample = sample[size_x:]
      # Calculate the Mahalanobis distance for the bootstrap sample
      bootstrap_samples[i] = mahalanobis_distance(x_sample.T, y_sample.T, paired = paired)
  # Estimate the variance of the Mahalanobis distance
  return np.var(bootstrap_samples)


def variance_stabilization(data, groups = None):
  """Variance stabilization normalization\n
  | Arguments:
  | :-
  | data (dataframe): pandas dataframe with glycans/motifs as indices and samples as columns
  | groups (nested list): list containing lists of column names of samples from same group for group-specific normalization; otherwise global; default:None\n
  | Returns:
  | :-
  | Returns a dataframe in the same style as the input
  """
  # Apply log1p transformation
  data = np.log1p(data)
  # Scale data to have zero mean and unit variance
  if groups is None:
    data = (data - data.mean(axis = 0)) / data.std(axis = 0)
  else:
    for group in groups:
      group_data = data.loc[:, group]
      data.loc[:, group] = (group_data - group_data.mean(axis = 0)) / group_data.std(axis = 0)
  return data


class MissForest:
  """Parameters
  (adapted from https://github.com/yuenshingyan/MissForest)
  ----------
  regressor : estimator object.
  A object of that type is instantiated for each imputation.
  This object is assumed to implement the scikit-learn estimator API.

  n_iter : int
  Determines the number of iterations for the imputation process.
  """

  def __init__(self, regressor = RandomForestRegressor(n_jobs = -1), max_iter = 5, tol=1e-6):
    self.regressor = regressor
    self.max_iter = max_iter
    self.tol = tol

  def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Initialization 
    # Keep track of where NaNs are in the original dataset
    X_nan = X.isnull()

    # Replace NaNs with median of the column in a new dataset that will be transformed
    X_transform = X.copy()
    X_transform.fillna(X_transform.median(), inplace = True)
    # Sort columns by the number of NaNs (ascending)
    sorted_columns = X_nan.sum().sort_values().index.tolist()

    for _ in range(self.max_iter):
      X_old = X_transform.copy()
      # Step 2: Imputation
      for column in sorted_columns:
        if X_nan[column].any():  # if column has missing values in original dataset
          # Split data into observed and missing for the current column
          observed = X_transform.loc[~X_nan[column]]
          missing = X_transform.loc[X_nan[column]]
          
          # Use other columns to predict the current column
          X_other_columns = observed.drop(columns = column)
          y_observed = observed[column]

          self.regressor.fit(X_other_columns, y_observed)
          
          X_missing_other_columns = missing.drop(columns = column)
          y_missing_pred = self.regressor.predict(X_missing_other_columns)

          # Replace missing values in the current column with predictions
          X_transform.loc[X_nan[column], column] = y_missing_pred
      # Check for convergence
      if np.all(np.abs(X_old - X_transform) < self.tol):
        break  # Break out of the loop if converged
    # Avoiding zeros
    X_transform += 1e-6
    return X_transform


def impute_and_normalize(df, groups, impute = True, min_samples = None):
    """given a dataframe, discards rows with too many missings, imputes the rest, and normalizes\n
    | Arguments:
    | :-
    | df (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns
    | groups (list): nested list of column name lists, one list per group
    | impute (bool): replaces zeroes with draws from left-shifted distribution or KNN-Imputer; default:True
    | min_samples (int): How many samples per group need to have non-zero values for glycan to be kept; default: at least half per group\n
    | Returns:
    | :-
    | Returns a dataframe in the same style as the input 
    """
    if min_samples is None:
      min_samples = [len(group_cols) // 2 for group_cols in groups]
    else:
      min_samples = [min_samples] * len(groups)
    masks = [
      df[group_cols].apply(lambda row: (row != 0).sum(), axis=1) >= thresh
      for group_cols, thresh in zip(groups, min_samples)
      ]
    df = df[np.all(masks, axis = 0)].copy()
    colname = df.columns[0]
    glycans = df[colname].values
    df = df.iloc[:, 1:]
    old_cols = []
    if isinstance(df.columns[0], int):
      old_cols = df.columns
      df.columns = df.columns.astype(str)
    if impute:
      mf = MissForest()
      df.replace(0, np.nan, inplace = True)
      df = mf.fit_transform(df)
    df = (df / df.sum(axis = 0)) * 100
    if len(old_cols) > 0:
      df.columns = old_cols
    df.insert(loc = 0, column = colname, value = glycans)
    return df


def variance_based_filtering(df, min_feature_variance = 0.01):
    """Variance-based filtering of features\n
    | Arguments:
    | :-
    | df (dataframe): dataframe containing glycan sequences in index and samples in columns
    | min_feature_variance (float): Minimum variance to include a feature in the analysis\n
    | Returns:
    | :-
    | Returns a pandas DataFrame with remaining glycans as indices and samples in columns
    """
    feature_variances = df.var(axis = 1)
    variable_features = feature_variances[feature_variances > min_feature_variance].index
    # Subsetting df to only include features with enough variance
    return df.loc[variable_features]


def jtkdist(timepoints, param_dic, reps = 1, normal = False):
  """Precalculates all possible JT test statistic permutation probabilities for reference later, speeding up the
  | analysis. Calculates the exact null distribution using the Harding algorithm.\n
  | Arguments:
  | :-
  | timepoints (int): number of timepoints within the experiment.
  | param_dic (dict): dictionary carrying around the parameter values
  | reps (int): number of replicates within each timepoint.
  | normal (bool): a flag for normal approximation if maximum possible negative log p-value is too large.\n
  | Returns:
  | :-
  | Returns statistical values, added to 'param_dic'.
  """
  timepoints = timepoints if isinstance(timepoints, int) else timepoints.sum()
  tim = np.full(timepoints, reps) if reps != timepoints else reps  # Support for unbalanced replication (unequal replicates in all groups)
  maxnlp = gammaln(np.sum(tim)) - np.sum(np.log(np.arange(1, np.max(tim)+1)))
  limit = math.log(float('inf'))
  normal = normal or (maxnlp > limit - 1)  # Switch to normal approximation if maxnlp is too large
  lab = []
  nn = sum(tim)  # Number of data values (Independent of period and lag)
  M = (nn ** 2 - np.sum(np.square(tim))) / 2  # Max possible jtk statistic
  param_dic.update({"GRP_SIZE": tim, "NUM_GRPS": len(tim), "NUM_VALS": nn,
                    "MAX": M, "DIMS": [int(nn * (nn - 1) / 2), 1]})
  if normal:
    param_dic["VAR"] = (nn ** 2 * (2 * nn + 3) - np.sum(np.square(tim) * (2 * t + 3) for t in tim)) / 72  # Variance of JTK
    param_dic["SDV"] = math.sqrt(param_dic["VAR"])  # Standard deviation of JTK
    param_dic["EXV"] = M / 2  # Expected value of JTK
    param_dic["EXACT"] = False
  MM = int(M // 2)  # Mode of this possible alternative to JTK distribution
  cf = [1] * (MM + 1)  # Initial lower half cumulative frequency (cf) distribution
  size = sorted(tim)  # Sizes of each group of known replicate values, in ascending order for fastest calculation
  k = len(tim)  # Number of groups of replicates
  N = [size[k-1]]
  if k > 2:
    for i in range(k - 1, 1, -1):  # Count permutations using the Harding algorithm
      N.insert(0, (size[i] + N[0]))
  for m, n in zip(size[:-1], N):
    update_cf_for_m_n(m, n, MM, cf)
  cf = np.array(cf)
  # cf now contains the lower half cumulative frequency distribution
  # append the symmetric upper half cumulative frequency distribution to cf
  if M % 2:   # jtkcf = upper-tail cumulative frequencies for all integer jtk
    jtkcf = np.concatenate((cf, 2 * cf[MM] - cf[:MM][::-1], [2 * cf[MM]]))[::-1]
  else:
    jtkcf = np.concatenate((cf, cf[MM - 1] + cf[MM] - cf[:MM-1][::-1], [cf[MM - 1] + cf[MM]]))[::-1]
  ajtkcf = list((jtkcf[i - 1] + jtkcf[i]) / 2 for i in range(1, len(jtkcf)))  # interpolated cumulative frequency values for all half-intgeger jtk
  cf = [ajtkcf[(j - 1) // 2] if j % 2 == 0 else jtkcf[j // 2] for j in [i for i in range(1, 2 * int(M) + 2)]]
  param_dic["CP"] = [c / jtkcf[0] for c in cf]  # all upper-tail p-values
  return param_dic


def jtkinit(periods, param_dic, interval = 1, replicates = 1):
  """Defines the parameters of the simulated sine waves for reference later.\n
  | Each molecular species within the analysis is matched to the optimal wave defined here, and the parameters
  | describing that wave are attributed to the molecular species.\n
  | Arguments:
  | :-
  | periods (list): the possible periods of rhytmicity in the biological data (valued as 'number of timepoints').
  | (note: periods can accept multiple values (ie, you can define circadian rhythms as between 22, 24, 26 hours))
  | param_dic (dict): dictionary carrying around the parameter values
  | interval (int): the number of units of time (arbitrary) between each experimental timepoint.
  | replicates (int): number of replicates within each group.\n
  | Returns:
  | :-
  | Returns values describing waveforms, added to 'param_dic'.
  """
  param_dic["INTERVAL"] = interval
  if len(periods) > 1:
    param_dic["PERIODS"] = list(periods)
  else:
    param_dic["PERIODS"] = list(periods)
  param_dic["PERFACTOR"] = np.concatenate([np.repeat(i, ti) for i, ti in enumerate(periods, start = 1)])
  tim = np.array(param_dic["GRP_SIZE"])
  timepoints = int(param_dic["NUM_GRPS"])
  timerange = np.arange(timepoints)  # Zero-based time indices
  param_dic["SIGNCOS"] = np.zeros((periods[0], ((math.floor(timepoints / (periods[0]))*int(periods[0]))* replicates)), dtype = int)
  for i, period in enumerate(periods):
    time2angle = np.array([(2*round(math.pi, 4))/period])  # convert time to angle using an ~pi value
    theta = timerange*time2angle  # zero-based angular values across time indices
    cos_v = np.cos(theta)  # unique cosine values at each time point
    cos_r = np.repeat(rankdata(cos_v), np.max(tim))  # replicated ranks of unique cosine values
    cgoos = np.sign(np.subtract.outer(cos_r, cos_r)).astype(int)
    lower_tri = []
    for col in range(len(cgoos)):
      for row in range(col + 1, len(cgoos)):
        lower_tri.append(cgoos[row, col])
    cgoos = np.array(lower_tri)
    cgoosv = np.array(cgoos).reshape(param_dic["DIMS"])
    param_dic["CGOOSV"] = []
    param_dic["CGOOSV"].append(np.zeros((cgoos.shape[0], period)))
    param_dic["CGOOSV"][i][:, 0] = cgoosv[:, 0]
    cycles = math.floor(timepoints / period)
    jrange = np.arange(cycles * period)
    cos_s = np.sign(cos_v)[jrange]
    cos_s = np.repeat(cos_s, (tim[jrange]))
    if replicates == 1:
      param_dic["SIGNCOS"][:, i] = cos_s
    else:
      param_dic["SIGNCOS"][i] = cos_s
    for j in range(1, period):  # One-based half-integer lag index j
      delta_theta = j * time2angle / 2  # Angles of half-integer lags
      cos_v = np.cos(theta + delta_theta)  # Cycle left
      cos_r = np.concatenate([np.repeat(val, num) for val, num in zip(rankdata(cos_v), tim)]) # Phase-shifted replicated ranks
      cgoos = np.sign(np.subtract.outer(cos_r, cos_r)).T
      mask = np.triu(np.ones(cgoos.shape), k = 1).astype(bool)
      mask[np.diag_indices(mask.shape[0])] = False
      cgoos = cgoos[mask]
      cgoosv = cgoos.reshape(param_dic["DIMS"])
      matrix_i = param_dic["CGOOSV"][i]
      matrix_i[:, j] = cgoosv.flatten()
      param_dic["CGOOSV[i]"] = matrix_i
      cos_v = cos_v.flatten()
      cos_s = np.sign(cos_v)[jrange]
      cos_s = np.repeat(cos_s, (tim[jrange]))
      if replicates == 1:
        param_dic["SIGNCOS"][:, j] = cos_s
      else:
        param_dic["SIGNCOS"][j] = cos_s
  return param_dic


def jtkstat(z, param_dic):
  """Determines the JTK statistic and p-values for all model phases, compared to expression data.\n
  | Arguments:
  | :-
  | z (pd.DataFrame): expression data for a molecule ordered in groups, by timepoint.
  | param_dic (dict): a dictionary containing parameters defining model waveforms.\n
  | Returns:
  | :-
  | Returns an updated parameter dictionary where the appropriate model waveform has been assigned to the
  | molecules in the analysis.
  """
  param_dic["CJTK"] = []
  M = param_dic["MAX"]
  z = np.array(z)
  foosv = np.sign(np.subtract.outer(z, z)).T  # Due to differences in the triangle indexing of R / Python we need to transpose and select upper triangle rather than the lower triangle
  mask = np.triu(np.ones(foosv.shape), k = 1).astype(bool) # Additionally, we need to remove the middle diagonal from the tri index
  mask[np.diag_indices(mask.shape[0])] = False
  foosv = foosv[mask].reshape(param_dic["DIMS"])
  for i in range(param_dic["PERIODS"][0]):
    cgoosv = param_dic["CGOOSV"][0][:, i]
    S = np.nansum(np.diag(foosv * cgoosv))
    jtk = (abs(S) + M) / 2  # Two-tailed JTK statistic for this lag and distribution
    if S == 0:
      param_dic["CJTK"].append([1, 0, 0])
    elif param_dic.get("EXACT", False):
      jtki = 1 + 2 * int(jtk)  # index into the exact upper-tail distribution
      p = 2 * param_dic["CP"][jtki-1]
      param_dic["CJTK"].append([p, S, S / M])
    else:
      p = 2 * norm.cdf(-(jtk - 0.5), -param_dic["EXV"], param_dic["SDV"])
      param_dic["CJTK"].append([p, S, S / M])  # include tau = s/M for this lag and distribution
  return param_dic


def jtkx(z, param_dic, ampci = False):
  """Deployment of jtkstat for repeated use, and parameter extraction\n
  | Arguments:
  | :-
  | z (pd.dataframe): expression data ordered in groups, by timepoint.
  | param_dic (dict): a dictionary containing parameters defining model waveforms.
  | ampci (bool): flag for calculating amplitude confidence interval (TRUE = compute); default=False.\n
  | Returns:
  | :-
  | Returns an updated parameter dictionary containing the optimal waveform parameters for each molecular species.
  """
  param_dic = jtkstat(z, param_dic)  # Calculate p and S for all phases
  pvals = [cjtk[0] for cjtk in param_dic["CJTK"]]  # Exact two-tailed p values for period/phase combos
  padj = multipletests(pvals, method = 'fdr_bh')[1]
  JTK_ADJP = min(padj)  # Global minimum adjusted p-value
  def groupings(padj, param_dic):
    d = defaultdict(list)
    for i, value in enumerate(padj):
      key = param_dic["PERFACTOR"][i]
      d[key].append(value)
    return dict(d)
  dpadj = groupings(padj, param_dic)
  padj = np.array(pd.DataFrame(dpadj.values()).T)
  minpadj = [padj[i].min() for i in range(0, np.shape(padj)[1])]  # Minimum adjusted p-values for each period
  if len(param_dic["PERIODS"]) > 1:
    pers_index = np.where(JTK_ADJP == minpadj)[0]  # indices of all optimal periods
  else:
    pers_index = 0
  pers = param_dic["PERIODS"][int(pers_index)]    # all optimal periods
  padj_values = padj[pers_index]
  lagis = np.where(padj == JTK_ADJP)[0]  # list of optimal lag indice for each optimal period
  best_results = {'bestper': 0, 'bestlag': 0, 'besttau': 0, 'maxamp': 0, 'maxamp_ci': 2, 'maxamp_pval': 0}
  sc = np.transpose(param_dic["SIGNCOS"])
  w = (z[:len(sc)] - hlm(z[:len(sc)])) * math.sqrt(2)
  for i in range(abs(pers)):
    for lagi in lagis:
      S = param_dic["CJTK"][lagi][1]
      s = np.sign(S) if S != 0 else 1
      lag = (pers + (1 - s) * pers / 4 - lagi / 2) % pers
      signcos = sc[:, lagi]
      tmp = s * w * sc[:, lagi]
      amp = hlm(tmp)  # Allows missing values
      if ampci:
        jtkwt = pd.DataFrame(wilcoxon(tmp[np.isfinite(tmp)], zero_method = 'wilcox', correction = False,
                                              alternatives = 'two-sided', mode = 'exact'))
        amp = jtkwt['confidence_interval'].median()  # Extract estimate (median) from the conf. interval
        best_results['maxamp_ci'] = jtkwt['confidence_interval'].values
        best_results['maxamp_pval'] = jtkwt['pvalue'].values
      if amp > best_results['maxamp']:
        best_results.update({'bestper': pers, 'bestlag': lag, 'besttau': [abs(param_dic["CJTK"][lagi][2])], 'maxamp': amp})
  JTK_PERIOD = param_dic["INTERVAL"] * best_results['bestper']
  JTK_LAG = param_dic["INTERVAL"] * best_results['bestlag']
  JTK_AMP = float(max(0, best_results['maxamp']))
  JTK_TAU = best_results['besttau']
  JTK_AMP_CI = best_results['maxamp_ci']
  JTK_AMP_PVAL = best_results['maxamp_pval']
  return pd.Series([JTK_ADJP, JTK_PERIOD, JTK_LAG, JTK_AMP])
