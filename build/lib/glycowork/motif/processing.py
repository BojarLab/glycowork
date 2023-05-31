import pandas as pd
import numpy as np
import re
from glyles import convert
from glycowork.glycan_data.loader import unwrap, multireplace, find_nth


def min_process_glycans(glycan_list):
  """converts list of glycans into a nested lists of glycoletters\n
  | Arguments:
  | :-
  | glycan_list (list): list of glycans in IUPAC-condensed format as strings\n
  | Returns:
  | :-
  | Returns list of glycoletter lists
  """
  return [multireplace(k, {'[': '', ']': '', '{': '', '}': '', ')': '('}).split('(') for k in glycan_list]


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
  lib = unwrap(min_process_glycans(set(glycan_list)))
  lib = sorted(set(lib))
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
  new_libr = {k: v+len(libr) for k, v in new_libr.items() if k not in libr.keys()}
  return {**libr, **new_libr}


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
      glycan2 = re.sub(r'^(.*?)\[(.*?)\]', r'\2[\1]', glycan, 1)
    elif not bool(re.search(r'\[[^\]]+\[', glycan[find_nth(glycan, ']', 2):])) and bool(re.search(r'\[[^\]]+\[', glycan[:find_nth(glycan, '[', 3)])):
      glycan2 = re.sub(r'^(.*?)\[(.*?)(\]{1,1})(.*?)\]', r'\2\3\4[\1]', glycan, 1)
    try:
      out_list.add(glycan2)
    except:
      pass
  # Double branch swap
  temp = set()
  for k in out_list:
    if '][' in k:
      glycan2 = re.sub(r'\[([^[\]]+)\]\[([^[\]]+)\]', r'[\2][\1]', k)
      temp.add(glycan2)
  out_list.update(temp)
  temp = set()
  # Starting branch swapped with next side branch again to also include double branch swapped isomorphs
  for k in out_list:
    if k.count('[') > 1 and k.index('[') > 0 and find_nth(k, '[', 2) > k.index(']') and (find_nth(k, ']', 2) < find_nth(k, '[', 3) or k.count('[') == 2):
      glycan2 = re.sub(r'^(.*?)\[(.*?)\](.*?)\[(.*?)\]', r'\4[\1[\2]\3]', k, 1)
      temp.add(glycan2)
  out_list.update(temp)
  out_list = {k for k in out_list if not any([j in k for j in ['[[', ']]']])}
  if floaty:
    out_list = {floaty+k for k in out_list}
  return list(out_list)


def presence_to_matrix(df, glycan_col_name = 'target', label_col_name = 'Species'):
  """converts a dataframe such as df_species to absence/presence matrix\n
  | Arguments:
  | :-
  | df (dataframe): dataframe with glycan occurrence, rows are glycan-label pairs
  | glycan_col_name (string): column name under which glycans are stored; default:target
  | label_col_name (string): column name under which labels are stored; default:Species\n
  | Returns:
  | :-
  | Returns pandas dataframe with labels as rows and glycan occurrences as columns
  """
  glycans = sorted(set(df[glycan_col_name].values.tolist()))
  species = sorted(set(df[label_col_name].values.tolist()))
  # Get a count matrix for each rank - glycan combination
  mat_dic = {k: [df[df[label_col_name] == j][glycan_col_name].values.tolist().count(k) for j in species] for k in glycans}
  mat = pd.DataFrame(mat_dic)
  mat.index = species
  return mat


def find_matching_brackets_indices(s):
  stack = []
  opening_indices = {}
  matching_indices = []

  for i, c in enumerate(s):
    if c == '[':
      stack.append(i)
      opening_indices[i] = len(stack) - 1
    elif c == ']':
      if len(stack) > 0:
        opening_index = stack.pop()
        matching_indices.append((opening_index, i))
        del opening_indices[opening_index]

  if len(stack) > 0:
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
  if len(glycans) == 1:
    return glycans[0]
  floaty = False
  if '{' in glycans[0]:
    floaty = glycans[0][:glycans[0].rindex('}')+1]
    glycans = [k[k.rindex('}')+1:] for k in glycans]
  # Heuristic: main chain should contain the most monosaccharides of all chains
  mains = [bracket_removal(g) for g in glycans]
  mains = [len(k) for k in min_process_glycans(mains)]
  glycans2 = [g for k, g in enumerate(glycans) if mains[k] == max(mains)]
  # Handle neighboring branches
  kill_list = []
  for g in glycans2:
    if '][' in g:
      try:
        match = re.search(r'\[([^[\]]+)\]\[([^[\]]+)\]', g)
        if match.group(1).count('(') < match.group(2).count('('):
          kill_list.append(g)
        elif match.group(1).count('(') == match.group(2).count('('):
          if int(match.group(1)[-2]) > int(match.group(2)[-2]):
            kill_list.append(g)
      except:
        pass
  glycans2 = [k for k in glycans2 if k not in kill_list]
  # Choose the isoform with the longest main chain before the branch & or the branch ending in the smallest number if all lengths are equal
  if len(glycans2) > 1:
    candidates = {k: find_matching_brackets_indices(k) for k in glycans2}
    prefix = [min_process_glycans([k[j[0]+1:j[1]] for j in candidates[k]]) for k in candidates.keys()]
    prefix = [np.argmax([len(j) for j in k]) for k in prefix]
    prefix = min_process_glycans([k[:candidates[k][prefix[i]][0]] for i, k in enumerate(candidates.keys())])
    branch_endings = [k[-2][-1] if k[-2][-1] != 'd' and k[-2][-1] != '?' else 10 for k in prefix]
    if len(set(branch_endings)) == 1:
      branch_endings = [ord(k[0][0]) for k in prefix]
    branch_endings = [int(k) for k in branch_endings]
    glycans2 = [g for k,g in enumerate(glycans2) if branch_endings[k] == min(branch_endings)]
    if len(glycans2) > 1:
        preprefix = min_process_glycans([glyc[:glyc.index('[')] for glyc in glycans2])
        branch_endings = [k[-2][-1] if k[-2][-1] != 'd' and k[-2][-1] != '?' else 10 for k in preprefix]
        branch_endings = [int(k) for k in branch_endings]
        glycans2 = [g for k, g in enumerate(glycans2) if branch_endings[k] == min(branch_endings)]
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
  if glycan_class == 'O':
    pool = ['GalNAc', 'GalNAcOS', 'GalNAc6S' 'Man', 'Fuc', 'Gal', 'GlcNAc', 'GlcNAcOS', 'GlcNAc6S']
  elif glycan_class == 'N':
    pool = ['GlcNAc']
  elif glycan_class == 'free' or glycan_class == 'lipid':
    pool = ['Glc', 'GlcOS', 'Glc3S', 'GlcNAc', 'GlcNAcOS', 'Gal', 'GalOS', 'Gal3S', 'Ins']
  truth = any([glycan.endswith(k) for k in pool])
  if glycan_class == 'free' or glycan_class == 'lipid' or glycan_class == 'O':
    if any([glycan.endswith(k) for k in ['GlcNAc(b1-4)GlcNAc', '[Fuc(a1-6)]GlcNAc']]):
      truth = False
  if not truth and conf:
    if conf > extra_thresh:
      truth = True
  return truth


def IUPAC_to_SMILES(glycan_list):
  """given a list of IUPAC-condensed glycans, uses GlyLES to return a list of corresponding isomeric SMILES\n
  | Arguments:
  | :-
  | glycan_list (list): list of IUPAC-condensed glycans\n
  | Returns:
  | :-
  | Returns a list of corresponding isomeric SMILES
  """
  if not isinstance(glycan_list, list):
    raise TypeError("Input must be a list")
  return [convert(g)[0][1] for g in glycan_list]


def canonicalize_iupac(glycan):
  """converts a glycan from any IUPAC flavor into the exact IUPAC-condensed version that is optimized for glycowork\n
  | Arguments:
  | :-
  | glycan (string): glycan sequence in IUPAC; some post-biosynthetic modifications could still be an issue\n
  | Returns:
  | :-
  | Returns glycan as a string in canonicalized IUPAC-condensed
  """
  # Canonicalize usage of monosaccharides and linkages
  replace_dic = {'Nac': 'NAc', 'AC': 'Ac', 'NeuAc': 'Neu5Ac', 'NeuNAc': 'Neu5Ac', 'NeuGc': 'Neu5Gc',
                 '\u03B1': 'a', '\u03B2': 'b', 'N(Gc)': 'NGc', 'GL': 'Gl', '(9Ac)': '9Ac',
                 'KDN': 'Kdn', 'OSO3': 'S', '-O-Su-': 'S', '(S)': 'S', 'H2PO3': 'P', '(P)': 'P',
                 '–': '-', ' ': '', ',': '-', 'α': 'a', 'β': 'b', '.': '', '((': '(', '))': ')'}
  glycan = multireplace(glycan, replace_dic)
  # Trim linkers
  if '-' in glycan:
    if bool(re.search(r'[a-z]\-[a-zA-Z]', glycan[glycan.rindex('-')-1:])) and '-ol' not in glycan:
      glycan = glycan[:glycan.rindex('-')]
  # Canonicalize usage of brackets and parentheses
  if bool(re.search(r'\([A-Z3-9]', glycan)):
    glycan = glycan.replace('(', '[').replace(')', ']')
  # Canonicalize linkage uncertainty
  # Open linkages
  if bool(re.search(r'[a-z]\-[A-Z]', glycan)):
    glycan = re.sub(r'([a-z])\-([A-Z])', r'\1?1-?\2', glycan)
  # Open linkages2
  if bool(re.search(r'[1-2]\-\)', glycan)):
    glycan = re.sub(r'([1-2])\-(\))', r'\1-?\2', glycan)
  # Missing linkages
  if bool(re.search(r'[^hr][a-b][\(\)]', glycan)):
    glycan = re.sub(r'([a-b])([\(\)])', r'\1?1-?\2', glycan)
  # Open linkages in front of branches
  if bool(re.search(r'[0-9]\-[\[\]]', glycan)):
    glycan = re.sub(r'([0-9])\-([\[\]])', r'\1-?\2', glycan)
  # Open linkages in front of branches (with missing information)
  if bool(re.search(r'[a-z]\-[\[\]]', glycan)):
    glycan = re.sub(r'([a-z])\-([\[\]])', r'\1?1-?\2', glycan)
  # Branches without linkages
  if bool(re.search(r'\[([a-zA-Z])+\]', glycan)):
    glycan = re.sub(r'(\[[a-zA-Z]+)(\])', r'\1?1-?\2', glycan)
  # Missing linkages in front of branches
  if bool(re.search(r'[a-z]\[[A-Z]', glycan)):
    glycan = re.sub(r'([a-z])(\[[A-Z])', r'\1?1-?\2', glycan)
  # Missing anomer info
  if bool(re.search(r'\([1-2]', glycan)):
    glycan = re.sub(r'(\()([1-2])', r'\1?\2', glycan)
  # Smudge uncertainty
  while '/' in glycan:
    glycan = glycan[:glycan.index('/')-1] + '?' + glycan[glycan.index('/')+1:]
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
  if (glycan.endswith('a') or glycan.endswith('b')) and not glycan.endswith('Rha') and not glycan.endswith('Ara'):
    glycan = glycan[:-1]
  # Handle modifications
  if bool(re.search(r'\[[1-9]?[SP]\][A-Z][^\(^\[]+', glycan)):
    glycan = re.sub(r'\[([1-9]?[SP])\]([A-Z][^\(^\[]+)', r'\2\1', glycan)
  if bool(re.search(r'(\)|\]|^)[1-9]?[SP][A-Z][^\(^\[]+', glycan)):
    glycan = re.sub(r'([1-9]?[SP])([A-Z][^\(^\[]+)', r'\2\1', glycan)
  if bool(re.search(r'\-ol[0-9]?[SP]', glycan)):
    glycan = re.sub(r'(\-ol)([0-9]?[SP])', r'\2\1', glycan)
  post_process = {'5Ac(?1': '5Ac(a2', '5Gc(?1': '5Gc(a2', 'Fuc(?': 'Fuc(a', 'GalS': 'GalOS', 'GlcNAcS': 'GlcNAcOS',
                  'GalNAcS': 'GalNAcOS'}
  glycan = multireplace(glycan, post_process)
  # Canonicalize branch ordering
  if '[' in glycan:
    isos = find_isomorphs(glycan)
    glycan = choose_correct_isoform(isos)
  if '+' in glycan:
    glycan = '{'+glycan.replace('+', '}')
  return glycan


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
  data = np.concatenate((x, y), axis = 0)
  # Initialize an empty list to store the bootstrap samples
  bootstrap_samples = []
  # Perform bootstrap resampling
  n_iterations = 1000
  for _ in range(n_iterations):
      # Generate a random bootstrap sample
      sample = data[np.random.choice(range(data.shape[0]), size = data.shape[0], replace = True)]
      # Split the bootstrap sample into two groups
      x_sample = sample[:x.shape[0]]
      y_sample = sample[x.shape[0]:]
      # Calculate the Mahalanobis distance for the bootstrap sample
      bootstrap_samples.append(mahalanobis_distance(x_sample, y_sample, paired = paired))
  # Convert the list of bootstrap samples into a numpy array
  bootstrap_samples = np.array(bootstrap_samples)
  # Estimate the variance of the Mahalanobis distance
  return np.var(bootstrap_samples)


def variance_stabilization(data):
  """Variance stabilization normalization\n
  """
  # Apply log1p transformation
  data = np.log1p(data)
  # Scale data to have zero mean and unit variance
  data = (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)
  return data
