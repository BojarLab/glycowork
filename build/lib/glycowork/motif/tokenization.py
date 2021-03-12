import pandas as pd
import re

from glycowork.glycan_data.loader import lib
from glycowork.motif.processing import small_motif_find

def character_to_label(character, libr = lib):
  """tokenizes character by indexing passed library
  character -- character to index
  libr -- list of library items

  returns index of character in library
  """
  character_label = libr.index(character)
  return character_label

def string_to_labels(character_string, libr = lib):
  """tokenizes word by indexing characters in passed library
  character_string -- string of characters to index
  libr -- list of library items

  returns indexes of characters in library
  """
  return list(map(lambda character: character_to_label(character, libr), character_string))

def pad_sequence(seq, max_length, pad_label = len(lib)):
  """brings all sequences to same length by adding padding token
  seq -- sequence to pad
  max_length -- sequence length to pad to
  pad_label -- which padding label to use

  returns padded sequence
  """
  seq += [pad_label for i in range(max_length-len(seq))]
  return seq

def convert_to_counts_glycoletter(glycan, libr = lib):
  """counts the occurrence of glycoletters in glycan
  glycan -- IUPACcondensed glycan sequence (string)
  libr -- sorted list of unique glycoletters observed in the glycans of our dataset

  returns dictionary with counts per glycoletter in a glycan
  """
  letter_dict = dict.fromkeys(libr, 0)
  glycan = small_motif_find(glycan).split('*')
  for i in libr:
    letter_dict[i] = glycan.count(i)
  return letter_dict

def glycoletter_count_matrix(glycans, target_col, target_col_name, libr = lib):
  """creates dataframe of counted glycoletters in glycan list
  glycans -- list of IUPACcondensed glycan sequences (strings)
  target_col -- label columns used for prediction; list or pd.Series
  target_col_name -- name for target_col; string
  libr -- sorted list of unique glycoletters observed in the glycans of our dataset

  returns dataframe with glycoletter counts (columns) for every glycan (rows)
  """
  counted_glycans = [convert_to_counts_glycoletter(i, libr) for i in glycans]
  out = pd.DataFrame(counted_glycans)
  out[target_col_name] = target_col
  return out

def find_isomorphs(glycan):
  """returns a set of isomorphic glycans by swapping branches etc.
  glycan -- IUPACcondensed glycan sequence (string)

  returns list of unique glycan notations (strings) for a glycan in IUPACcondensed
  """
  out_list = [glycan]
  #starting branch swapped with next side branch
  if '[' in glycan and glycan.index('[') > 0 and not bool(re.search('\[[^\]]+\[', glycan)):
    glycan2 = re.sub('^(.*?)\[(.*?)\]', r'\2[\1]', glycan, 1)
    out_list.append(glycan2)
  #double branch swap
  temp = []
  for k in out_list:
    if '][' in k:
      glycan3 = re.sub('(\[.*?\])(\[.*?\])', r'\2\1', k)
      temp.append(glycan3)
  #starting branch swapped with next side branch again to also include double branch swapped isomorphs
  temp2 = []
  for k in temp:
    if '[' in k and k.index('[') > 0 and not bool(re.search('\[[^\]]+\[', k)):
      glycan4 = re.sub('^(.*?)\[(.*?)\]', r'\2[\1]', k, 1)
      temp2.append(glycan4)
  return list(set(out_list + temp + temp2))

def link_find(glycan):
  """finds all disaccharide motifs in a glycan sequence using its isomorphs
  glycan -- IUPACcondensed glycan sequence (string)

  returns list of unique disaccharides (strings) for a glycan in IUPACcondensed
  """
  ss = find_isomorphs(glycan)
  coll = []
  for iso in ss:
    b_re = re.sub('\[[^\]]+\]', '', iso)
    for i in [iso, b_re]:
      b = i.split('(')
      b = [k.split(')') for k in b]
      b = [item for sublist in b for item in sublist]
      b = ['*'.join(b[i:i+3]) for i in range(0, len(b) - 2, 2)]
      b = [k for k in b if (re.search('\*\[', k) is None and re.search('\*\]\[', k) is None)]
      b = [k.strip('[') for k in b]
      b = [k.strip(']') for k in b]
      b = [k.replace('[', '') for k in b]
      b = [k.replace(']', '') for k in b]
      coll += b
  return list(set(coll))

def motif_matrix(df, glycan_col_name, label_col_name, libr = lib):
  """generates dataframe with counted glycoletters and disaccharides in glycans
  df -- dataframe containing glycan sequences and labels
  glycan_col_name -- column name for glycan sequences; string
  label_col_name -- column name for labels; string
  libr -- sorted list of unique glycoletters observed in the glycans of our dataset

  returns dataframe with glycoletter + disaccharide counts (columns) for each glycan (rows)
  """
  matrix_list = []
  di_dics = []
  wga_di = [link_find(i) for i in df[glycan_col_name].values.tolist()]
  lib_di = list(sorted(list(set([item for sublist in wga_di for item in sublist]))))
  for j in wga_di:
    di_dict = dict.fromkeys(lib_di, 0)
    for i in list(di_dict.keys()):
      di_dict[i] = j.count(i)
    di_dics.append(di_dict)
  wga_di_out = pd.DataFrame(di_dics)
  wga_letter = glycoletter_count_matrix(df[glycan_col_name].values.tolist(),
                                          df[label_col_name].values.tolist(),
                                   label_col_name, libr)
  out_matrix = pd.concat([wga_letter, wga_di_out], axis = 1)
  out_matrix = out_matrix.loc[:,~out_matrix.columns.duplicated()]
  temp = out_matrix.pop(label_col_name)
  out_matrix[label_col_name] = temp
  out_matrix.reset_index(drop = True, inplace = True)
  return out_matrix
