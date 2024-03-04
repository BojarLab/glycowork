import pandas as pd
import numpy as np
import copy
import re
from functools import wraps
from collections import defaultdict
from glycowork.glycan_data.loader import (unwrap, multireplace,
                                          find_nth, find_nth_reverse,
                                          linkages, Hex, HexNAc, dHex, Sia, HexA, Pen)


# for WURCS mapping
monosaccharide_mapping = {
    'a2122h-1b_1-5_2*NCC/3=O': 'GlcNAc', 'a2112h-1a_1-5_2*NCC/3=O': 'GalNAc',
    'a1122h-1b_1-5': 'Man', 'Aad21122h-2a_2-6_5*NCC/3=O': 'Neu5Ac',
    'a1122h-1a_1-5': 'Man', 'a2112h-1b_1-5': 'Gal', 'Aad21122h-2a_2-6_5*NCCO/3=O': 'Neu5Gc',
    'a2112h-1b_1-5_2*NCC/3=O_?*OSO/3=O/3=O': 'GalNAcOS', 'a2112h-1b_1-5_2*NCC/3=O': 'GalNAc',
    'a1221m-1a_1-5': 'Fuc', 'a2122h-1b_1-5_2*NCC/3=O_6*OSO/3=O/3=O': 'GlcNAc6S', 'a212h-1b_1-5': 'Xyl',
    'axxxxh-1b_1-5_2*NCC/3=O': 'HexNAc', 'a2122h-1x_1-5_2*NCC/3=O': 'GlcNAc', 'a2112h-1x_1-5': 'Gal',
    'Aad21122h-2a_2-6': 'Kdn', 'a2122h-1a_1-5_2*NCC/3=O': 'GlcNAc', 'a2112h-1a_1-5': 'Gal',
    'a1122h-1x_1-5': 'Man', 'Aad21122h-2x_2-6_5*NCCO/3=O': 'Neu5Gc', 'Aad21122h-2x_2-6_5*NCC/3=O': 'Neu5Ac',
    'a1221m-1x_1-5': 'Fuc', 'a212h-1x_1-5': 'Xyl', 'a122h-1x_1-5': 'Ara', 'a2122A-1b_1-5': 'GlcA',
    'a2112h-1b_1-5_3*OC': 'Gal3Me', 'a1122h-1a_1-5_2*NCC/3=O': 'ManNAc', 'a2122h-1x_1-5': 'Glc',
    'axxxxh-1x_1-5_2*NCC/3=O': 'HexNAc', 'axxxxh-1x_1-5': 'Hex', 'a2112h-1b_1-4': 'Galf',
    'a2122h-1x_1-5_2*NCC/3=O_6*OSO/3=O/3=O': 'GlcNAc6S', 'a2112h-1x_1-5_2*NCC/3=O': 'GalNAc',
    'axxxxh-1a_1-5_2*NCC/3=O': 'HexNAc', 'Aad21122h-2a_2-6_4*OCC/3=O_5*NCC/3=O': 'Neu4Ac5Ac',
    'a2112h-1b_1-5_4*OSO/3=O/3=O': 'Gal4S', 'a2122h-1b_1-5_2*NCC/3=O_3*OSO/3=O/3=O': 'GlcNAc3S',
    'a2112h-1b_1-5_2*NCC/3=O_4*OSO/3=O/3=O': 'GalNAc4S', 'a2122A-1x_1-5_?*OSO/3=O/3=O': 'GlcAOS',
    'a2122A-1b_1-5_3*OSO/3=O/3=O': 'GlcA3S', 'a2211m-1x_1-5': 'Rha', 'a1122h-1b_1-5_2*NCC/3=O': 'ManNAc',
    'a1122h-1x_1-5_6*PO/2O/2=O': 'Man6P', 'a1122h-1a_1-5_6*OSO/3=O/3=O': 'Man6S', 'a2112h-1x_1-5_2*NCC/3=O_?*OSO/3=O/3=O': 'GalNAcOS'
    }


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


def get_possible_linkages(wildcard, linkage_list = linkages):
  """Retrieves all linkages that match a given wildcard pattern from a list of linkages\n
  | Arguments:
  | :-
  | wildcard (string): The pattern to match, where '?' can be used as a wildcard for any single character.
  | linkage_list (list): List of linkages as strings to search within; default:linkages\n
  | Returns:
  | :-
  | Returns a list of linkages that match the wildcard pattern.
  """
  pattern = wildcard.replace("?", "[a-zA-Z0-9\?]")
  return [linkage for linkage in linkage_list if re.fullmatch(pattern, linkage)]
  #return possible_linkages if libr is None else list(possible_linkages & libr.keys())


def get_possible_monosaccharides(wildcard):
  """Retrieves all matching common monosaccharides of a type, given the type\n
  | Arguments:
  | :-
  | wildcard (string): Monosaccharide type, from "HexNAc", "Hex", "dHex", "Sia", "HexA", "Pen"\n
  | Returns:
  | :-
  | Returns a list of specified monosaccharides of that type
  """
  wildcard_dict = {'Hex': Hex, 'HexNAc': HexNAc, 'dHex': dHex, 'Sia': Sia, 'HexA': HexA, 'Pen': Pen,
                   'Monosaccharide': set().union(*[Hex, HexNAc, dHex, Sia, HexA, Pen])}
  return wildcard_dict.get(wildcard, [])
  #return list(possible_monosaccharides) if libr is None else list(possible_monosaccharides & libr.keys())


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
  return grouped_df.sort_index().sort_index(axis = 1)


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
  glycan = glycan[:-3] if glycan.endswith('-ol') else glycan
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
  pattern = re.compile(r"(?P<alpha_beta>[αβßab\?])-[DL]-(?P<after>[^\)]*\()")
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
      line = re.sub("\d\|\d", "?", line)
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
                  'dxyl': 'Xyl', 'dara': 'D-Ara', 'lara': 'Ara', 'HEX': 'Hex', 'lman': 'Rha'}
  sub_replace = {'n-acetyl': 'NAc', 'sulfate': 'OS', 'phosphate': 'OP', 'n-glycolyl': '5Gc',
                 'acetyl': 'OAc', 'methyl': 'OMe'}
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
  iupac = iupac.replace('[[', '[').replace(']]', ']').replace('Neu(', 'Kdn(')
  return iupac


def get_mono(token):
  """maps WURCS token to monosaccharide with anomeric state; provides anomeric flexibility\n
  | Arguments:
  | :-
  | token (string): token indicating monosaccharide in WURCS format\n
  | Returns:
  | :-
  | Returns monosaccharide with anomeric state as string
  """
  anomer = token[token.index('_')-1]
  if token in monosaccharide_mapping:
    mono = monosaccharide_mapping[token]
  else:
    for a in ['a', 'b', 'x']:
      if a != anomer:
        token = token[:token.index('_')-1] + a + token[token.index('_'):]
        try:
          mono = monosaccharide_mapping[token]
          break
        except:
          raise Exception("Token " + token + " not recognized.")
  mono += anomer if anomer in ['a', 'b'] else '?'
  return mono


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
  oxford = re.sub(r'\([^)]*\)', '', oxford).strip().split('/')[0]
  antennae = {}
  iupac = "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc"
  mapping_dict = {"A": "GlcNAc(b1-?)", "G": "Gal(b1-?)", "S": "Neu5Ac(a2-?)",
                  "Sg": "Neu5Gc(a2-?)", "Ga": "Gal(a1-?)", "GalNAc": "GalNAc(?1-?)",
                  "Lac": "Gal(b1-?)GlcNAc(b1-?)", "F": "Fuc(a1-?)", "LacDiNAc": "GalNAc(b1-4)GlcNAc(b1-?)"}
  hardcoded = {"M3": "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
               "M4": "Man(a1-?)Man(a1-?)[Man(a1-?)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
               "M9": "Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
               "M10": "Glc(a1-3)Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
               "M11": "Glc(a1-3)Glc(a1-3)Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
               "M12": "Glc(a1-2)Glc(a1-3)Glc(a1-3)Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc"}
  if oxford in hardcoded:
    return hardcoded[oxford]
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
  branches = {"A": int(oxford_wo_branches[oxford_wo_branches.index("A")+1]) if "A" in oxford_wo_branches and oxford_wo_branches[oxford_wo_branches.index("A")+1] != "c" else 0,
              "G": int(oxford_wo_branches[oxford_wo_branches.index("G")+1]) if "G" in oxford_wo_branches and oxford_wo_branches[oxford_wo_branches.index("G")+1] != "a" else 0,
              "S": int(oxford_wo_branches[oxford_wo_branches.index("S")+1]) if "S" in oxford_wo_branches and oxford_wo_branches[oxford_wo_branches.index("S")+1] != "g" else 0}
  extras = {"Sg": int(oxford_wo_branches[oxford_wo_branches.index("Sg")+2]) if "Sg" in oxford_wo_branches else 0,
            "Ga": int(oxford_wo_branches[oxford_wo_branches.index("Ga")+2]) if "Ga" in oxford_wo_branches else 0,
            "Lac": int(oxford_wo_branches[oxford_wo_branches.index("Lac")+3]) if "Lac" in oxford_wo_branches and oxford_wo_branches[oxford_wo_branches.index("Lac")+3] != "D" else 0,
            "LacDiNAc": 1 if "LacDiN" in oxford_wo_branches else 0}
  specified_linkages = {'Neu5Ac(a2-?)': oxford[oxford.index("S")+2:] if branches['S'] else []}
  specified_linkages = {k: [int(n) for n in v[:v.index(']')].split(',')] for k, v in specified_linkages.items() if v}
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
      else:
        iupac = mapping_dict[e] + iupac
      v -= 1
  if antennae:
    for k, v in antennae.items():
      while v > 0:
        if "Gal(b1-?)Glc" in iupac:
          split = iupac.index("Gal(b1-?)Glc")
          iupac = iupac[:split+len("Gal(b1-?)")] + "[" + mapping_dict[k] + "]" + iupac[split+len("Gal(b1-?)"):]
        else:
          split =  iupac.index("GalNAc(b1-4)Glc")
          iupac = iupac[:split+len("GalNAc(b1-4)")] + "[" + mapping_dict[k] + "]" + iupac[split+len("GalNAc(b1-4)"):]
        v -= 1
  iupac = iupac.replace("GlcNAc(b1-?)[Neu5Ac(a2-?)]Man", "[Neu5Ac(a2-?)]GlcNAc(b1-?)Man")
  for k, v in specified_linkages.items():
    if v:
      for vv in v:
        iupac = iupac.replace(k, k[:-2]+str(vv)+')', 1)
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
  iupac = floaty + iupac.strip('[]')
  return iupac


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
  glycan = glycan.strip()
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
  elif ((glycan[-1].isdigit() and bool(re.search("[A-Z]", glycan))) or (glycan[-2].isdigit() and glycan[-1] == ']') or glycan.endswith('B') or glycan.endswith("LacDiNAc")) and 'e' not in glycan and '-' not in glycan:
    glycan = oxford_to_iupac(glycan)
  # Canonicalize usage of monosaccharides and linkages
  replace_dic = {'Nac': 'NAc', 'AC': 'Ac', 'Nc': 'NAc', 'NeuAc': 'Neu5Ac', 'NeuNAc': 'Neu5Ac', 'NeuGc': 'Neu5Gc',
                 '\u03B1': 'a', '\u03B2': 'b', 'N(Gc)': 'NGc', 'GL': 'Gl', 'GaN': 'GalN', '(9Ac)': '9Ac',
                 'KDN': 'Kdn', 'OSO3': 'S', '-O-Su-': 'S', '(S)': 'S', 'H2PO3': 'P', '(P)': 'P',
                 '–': '-', ' ': '', ',': '-', 'α': 'a', 'β': 'b', 'ß': 'b', '.': '', '((': '(', '))': ')', '→': '-',
                 'Glcp': 'Glc', 'Galp': 'Gal', 'Manp': 'Man', 'Fucp': 'Fuc', 'Neup': 'Neu', 'a?': 'a1',
                 '5Ac4Ac': '4Ac5Ac', '(-)': '(?1-?)'}
  glycan = multireplace(glycan, replace_dic)
  if '{' in glycan and '(' not in glycan:
    glycan = glycan.replace('{', '(').replace('}', ')')
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


def equal_repeats(r1, r2):
  """checks whether two repeat units could stem from the same repeating structure, just shifted\n
  | Arguments:
  | :-
  | r1 (string): glycan sequence in IUPAC-condensed nomenclature
  | r2 (string): glycan sequence in IUPAC-condensed nomenclature\n
  | Returns:
  | :-
  | Returns True if repeat structures are shifted versions of each other, else False
  """
  r1_long = r1[:r1.rindex(')')+1] * 2
  return any(r1_long[i:i + len(r2)] == r2 for i in range(len(r1)))
