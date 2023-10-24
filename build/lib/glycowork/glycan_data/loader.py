import pandas as pd
import os
import ast
import pickle
import itertools
import pkg_resources

io = pkg_resources.resource_stream(__name__, "v8_sugarbase.csv")
df_glycan = pd.read_csv(io)
io = pkg_resources.resource_stream(__name__, "v8_df_species.csv")
df_species = pd.read_csv(io)
io = pkg_resources.resource_stream(__name__, "glycan_motifs.csv")
motif_list = pd.read_csv(io)
io = pkg_resources.resource_stream(__name__, "glycan_binding.csv")
glycan_binding = pd.read_csv(io)
this_dir, this_filename = os.path.split(__file__)  # Get path of data.pkl
data_path = os.path.join(this_dir, 'lib_v8.pkl')
lib = pickle.load(open(data_path, 'rb'))

linkages = {
  '1-4', '1-6', 'a1-1', 'a1-2', 'a1-3', 'a1-4', 'a1-5', 'a1-6', 'a1-7', 'a1-8', 'a1-9', 'a1-11', 'a1-?', 'a2-1', 'a2-2', 'a2-3', 'a2-4', 'a2-5', 'a2-6', 'a2-7', 'a2-8', 'a2-9',
  'a2-11', 'a2-?', 'b1-1', 'b1-2', 'b1-3', 'b1-4', 'b1-5', 'b1-6', 'b1-7', 'b1-8', 'b1-9', 'b1-?', 'b2-1', 'b2-2', 'b2-3', 'b2-4', 'b2-5', 'b2-6', 'b2-7', 'b2-8', '?1-?',
  '?2-?', '?1-2', '?1-3', '?1-4', '?1-6', '?2-3', '?2-6', '?2-8'
  }
Hex = {'Glc', 'Gal', 'Man', 'Ins', 'Galf', 'Hex'}
dHex = {'Fuc', 'Qui', 'Rha', 'dHex'}
HexA = {'GlcA', 'ManA', 'GalA', 'IdoA', 'HexA'}
HexN = {'GlcN', 'ManN', 'GalN', 'HexN'}
HexNAc = {'GlcNAc', 'GalNAc', 'ManNAc', 'HexNAc'}
Pen = {'Ara', 'Xyl', 'Rib', 'Lyx', 'Pen'}
Sia = {'Neu5Ac', 'Neu5Gc', 'Kdn', 'Sia'}


def unwrap(nested_list):
  """converts a nested list into a flat list"""
  return list(itertools.chain(*nested_list))


def find_nth(haystack, needle, n):
  """finds n-th instance of motif\n
  | Arguments:
  | :-
  | haystack (string): string to search for motif
  | needle (string): motif
  | n (int): n-th occurrence in string (not zero-indexed)\n
  | Returns:
  | :-
  | Returns starting index of n-th occurrence in string
  """
  start = haystack.find(needle)
  while start >= 0 and n > 1:
    start = haystack.find(needle, start+len(needle))
    n -= 1
  return start


def reindex(df_new, df_old, out_col, ind_col, inp_col):
  """Returns columns values in order of new dataframe rows\n
  | Arguments:
  | :-
  | df_new (pandas dataframe): dataframe with the new row order
  | df_old (pandas dataframe): dataframe with the old row order
  | out_col (string): column name of column in df_old that you want to reindex
  | ind_col (string): column name of column in df_old that will give the index
  | inp_col (string): column name of column in df_new that indicates the new order; ind_col and inp_col should match\n
  | Returns:
  | :-
  | Returns out_col from df_old in the same order of inp_col in df_new
  """
  if ind_col != inp_col:
    print("Mismatching column names for ind_col and inp_col. Doesn't mean it's wrong but pay attention.")
  return [df_old[out_col].values.tolist()[df_old[ind_col].values.tolist().index(k)] for k in df_new[inp_col].values.tolist()]


def stringify_dict(dicty):
  """Converts dictionary into a string\n
  | Arguments:
  | :-
  | dicty (dictionary): dictionary\n
  | Returns:
  | :-
  | Returns string of type key:value for sorted items
  """
  dicty = dict(sorted(dicty.items()))
  return ''.join(f"{key}{value}" for key, value in dicty.items())


def replace_every_second(string, old_char, new_char):
  """function to replace every second occurrence of old_char in string with new_char\n
  | Arguments:
  | :-
  | string (string): a string
  | old_char (string): a string character to be replaced (every second occurrence)
  | new_char (string): the string character to replace old_char with\n
  | Returns:
  | :-                           
  | Returns string with replaced characters
  """
  count = 0
  result = []
  for char in string:
    if char == old_char:
      count += 1
      result.append(new_char if count % 2 == 0 else char)
    else:
      result.append(char)
  return ''.join(result)


def multireplace(string, remove_dic):
  """
  Replaces all occurences of items in a set with a given string\n
  | Arguments:
  | :-
  | string (str): string to perform replacements on
  | remove_dic (set): dict of form to_replace:replace_with\n
  | Returns:
  | :-
  | (str) modified string
  """
  for k, v in remove_dic.items():
    string = string.replace(k, v)
  return string


def build_custom_df(df, kind = 'df_species'):
  """creates custom df from df_glycan\n
  | Arguments:
  | :-
  | df (dataframe): df_glycan / sugarbase
  | kind (string): whether to create 'df_species', 'df_tissue', or 'df_disease' from df_glycan; default:df_species\n
  | Returns:
  | :-
  | Returns custom df in the form of one glycan - species/tissue/disease association per row
  """
  kind_to_cols = {
        'df_species': ['glycan', 'Species', 'Genus', 'Family', 'Order', 'Class',
                       'Phylum', 'Kingdom', 'Domain', 'ref'],
        'df_tissue': ['glycan', 'tissue_sample', 'tissue_species', 'tissue_id', 'tissue_ref'],
        'df_disease': ['glycan', 'disease_association', 'disease_sample', 'disease_direction',
                       'disease_species', 'disease_id', 'disease_ref']
    }
  cols = kind_to_cols.get(kind, None)
  if cols is None:
    raise ValueError("Invalid value for 'kind' argument, only df_species, df_tissue, and df_disease are supported.")
  df = df.loc[df[cols[1]].str.len() > 2, cols]
  df.set_index('glycan', inplace = True)
  df.index.name = 'target'
  df = df.applymap(ast.literal_eval)
  df = df.explode(cols[1:]).reset_index()
  df.sort_values([cols[1], 'target'], ascending = [True, True], inplace = True)
  df.reset_index(drop = True, inplace = True)
  return df
