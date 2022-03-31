import pandas as pd
import numpy as np
import random

linkages = ['a1-1','a1-2','a1-3','a1-4','a1-5','a1-6','a1-7','a1-8','a1-9','a1-11','a1-z','a2-1','a2-2','a2-3','a2-4','a2-5','a2-6','a2-7','a2-8','a2-9','a2-11','b1-1','b1-2','b1-3','b1-4','b1-5','b1-6','b1-7','b1-8','b1-9','b1-z','b2-1','b2-2','b2-3','b2-4','b2-5','b2-6','b2-7','b2-8','z1-z','z2-z','z1-2','z1-3','z1-4','z1-6','z2-3','z2-6','z2-8']

def unwrap(nested_list):
  """converts a nested list into a flat list"""
  out = [item for sublist in nested_list for item in sublist]
  return out

def small_motif_find(glycan):
  """processes IUPACcondensed glycan sequence (string) without splitting it into glycowords\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format\n
  | Returns:
  | :-
  | Returns string in which glycoletters are separated by asterisks
  """
  b = glycan.split('(')
  b = [k.split(')') for k in b]
  b = [item for sublist in b for item in sublist]
  b = [k.strip('[') for k in b]
  b = [k.strip(']') for k in b]
  b = [k.replace('[', '') for k in b]
  b = [k.replace(']', '') for k in b]
  b = '*'.join(b)
  return b

def min_process_glycans(glycan_list):
  """converts list of glycans into a nested lists of glycoletters\n
  | Arguments:
  | :-
  | glycan_list (list): list of glycans in IUPAC-condensed format as strings\n
  | Returns:
  | :-
  | Returns list of glycoletter lists
  """
  glycan_motifs = [small_motif_find(k) for k in glycan_list]
  glycan_motifs = [i.split('*') for i in glycan_motifs]
  return glycan_motifs

def get_lib(glycan_list):
  """returns sorted list of unique glycoletters in list of glycans\n
  | Arguments:
  | :-
  | glycan_list (list): list of IUPAC-condensed glycan sequences as strings\n
  | Returns:
  | :-
  | Returns sorted list of unique glycoletters (strings) in glycan_list
  """
  proc = min_process_glycans(glycan_list)
  lib = unwrap(proc)
  lib = list(sorted(list(set(lib))))
  return lib

def expand_lib(libr, glycan_list):
  """updates libr with newly introduced glycoletters\n
  | Arguments:
  | :-
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset
  | glycan_list (list): list of IUPAC-condensed glycan sequences as strings\n
  | Returns:
  | :-
  | Returns new lib
  """
  new_lib = get_lib(glycan_list)
  return list(sorted(list(set(libr + new_lib))))

def seed_wildcard(df, wildcard_list, wildcard_name, r = 0.1, col = 'target'):
  """adds dataframe rows in which glycan parts have been replaced with the appropriate wildcards\n
  | Arguments:
  | :-
  | df (dataframe): dataframe in which the glycan column is called "target" and is the first column
  | wildcard_list (list): list which glycoletters a wildcard encompasses
  | wildcard_name (string): how the wildcard should be named in the IUPAC-condensed nomenclature
  | r (float): rate of replacement, default is 0.1 or 10%
  | col (string): column name for glycan sequences; default: target\n
  | Returns:
  | :-
  | Returns dataframe in which some glycoletters (from wildcard_list) have been replaced with wildcard_name
  """
  added_rows = []
  for k in range(len(df)):
    temp = df[col].values.tolist()[k]
    for j in wildcard_list:
      if j in temp:
        if random.uniform(0, 1) < r:
          added_rows.append([temp.replace(j, wildcard_name)] + df.iloc[k, 1:].values.tolist())
  added_rows = pd.DataFrame(added_rows, columns = df.columns.values.tolist())
  df_out = pd.concat([df, added_rows], axis = 0, ignore_index = True)
  return df_out

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
  glycans = list(sorted(list(set(df[glycan_col_name].values.tolist()))))
  species = list(sorted(list(set(df[label_col_name].values.tolist()))))
  mat_dic = {k:[df[df[label_col_name] == j][glycan_col_name].values.tolist().count(k) for j in species] for k in glycans}
  mat = pd.DataFrame(mat_dic)
  mat.index = species
  return mat

def check_nomenclature(glycan):
  """checks whether the proposed glycan has the correct nomenclature for glycowork\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  """
  if not isinstance(glycan, str):
    print("You need to format your glycan sequences as strings.")
    return
  if '?' in glycan:
    print("You're likely using ? somewhere, to indicate linkage uncertainty. Glycowork uses 'z' to indicate linkage uncertainty")
    return
  if '=' in glycan:
    print("Could it be that you're using WURCS? Please convert to IUPACcondensed for using glycowork.")
  if 'RES' in glycan:
    print("Could it be that you're using GlycoCT? Please convert to IUPACcondensed for using glycowork.")
    
  return True

def choose_correct_isoform(glycans, reverse = False):
  """given a list of glycan branch isomers, this function returns the correct isomer\n
  | Arguments:
  | :-
  | glycans (list): glycans in IUPAC-condensed nomenclature
  | reverse (bool): whether to return the correct isomer (False) or everything except the correct isomer (True); default:False\n
  | Returns:
  | :-
  | Returns the correct isomer as a string (if reverse=False, otherwise it returns a list of strings)
  """
  prefix = min_process_glycans([glyc[:glyc.index('[')] for glyc in glycans])
  prefix_len = [len(k) for k in prefix]
  if len(set(prefix_len)) == 1:
    branch_endings = [int(k[-2][-1]) if k[-2][-1] != 'z' and k[-2][-1] != 'd' else 10 for k in prefix]
    correct_isoform = glycans[np.argmin(branch_endings)]
  else:
    correct_isoform = glycans[np.argmax(prefix_len)]
  if reverse:
    glycans.remove(correct_isoform)
    correct_isoform = glycans
  return correct_isoform
