import pandas as pd
import random

linkages = ['a1-2','a1-3','a1-4','a1-6','a2-3','a2-6','a2-8','b1-2','b1-3','b1-4','b1-6']

def unwrap(nested_list):
  """converts a nested list into a flat list"""
  out = [item for sublist in nested_list for item in sublist]
  return out

def small_motif_find(s):
  """processes IUPACcondensed glycan sequence (string) without splitting it into glycowords
  s -- glycan in IUPACcondensed notation; string

  returns string in which glycoletters are separated by asterisks
  """
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
  """converts list of glycans into a nested lists of glycowords
  glycan_list -- list of glycans in IUPACcondensed notation; string

  returns list of glycowords (each as one list of glycoletters)
  """
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

def get_lib(glycan_list, mode = 'letter', exhaustive = True):
  """returns sorted list of unique glycoletters in list of glycans
  glycan_list -- list of IUPACcondensed glycan sequences (string)
  mode -- default is letter for glycoletters; change to obtain glycowords
  exhaustive -- if True, processes glycans shorted than 1 glycoword; default is True

  returns sorted list of unique glycoletters (strings) in glycan_list
  """
  proc = process_glycans(glycan_list, exhaustive = exhaustive)
  lib = unwrap(proc)
  lib = list(set([tuple(k) for k in lib]))
  lib = [list(k) for k in lib]
  if mode=='letter':
    lib = list(sorted(list(set(unwrap(lib)))))
  else:
    lib = list(sorted(list(set([tuple(k) for k in lib]))))
  return lib

def seed_wildcard(df, wildcard_list, wildcard_name, r = 0.1, col = 'target'):
  """adds dataframe rows in which glycan parts have been replaced with the appropriate wildcards
  df -- dataframe in which the glycan column is called "target" and is the first column
  wildcard_list -- list which glycoletters a wildcard encompasses
  wildcard_name -- how the wildcard should be named in the IUPACcondensed nomenclature
  r -- rate of replacement, default is 0.1 or 10%
  col -- column name for glycan sequences; default: target

  returns dataframe in which some glycoletters (from wildcard_list) have been replaced with wildcard_name
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

