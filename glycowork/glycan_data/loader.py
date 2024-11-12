import re
import json
import gdown
import pandas as pd
from pickle import load
from os import path
from itertools import chain
from importlib import resources
from typing import Any, Dict

with resources.open_text("glycowork.glycan_data", "glycan_motifs.csv") as f:
  motif_list = pd.read_csv(f)
this_dir, this_filename = path.split(__file__)  # Get path of data.pkl
data_path = path.join(this_dir, 'lib_v11.pkl')
lib = load(open(data_path, 'rb'))


def __getattr__(name):
  if name == "glycan_binding":
    with resources.open_text("glycowork.glycan_data", "glycan_binding.csv") as f:
      glycan_binding = pd.read_csv(f)
    globals()[name] = glycan_binding  # Cache it to avoid reloading
    return glycan_binding
  elif name == "df_species":
    with resources.open_text("glycowork.glycan_data", "v11_df_species.csv") as f:
      df_species = pd.read_csv(f)
    globals()[name] = df_species  # Cache it to avoid reloading
    return df_species
  elif name == "df_glycan":
    data_path = path.join(this_dir, 'v11_sugarbase.json')
    df_glycan = serializer.deserialize(data_path)
    globals()[name] = df_glycan  # Cache it to avoid reloading
    return df_glycan
  elif name == "lectin_specificity":
    data_path = path.join(this_dir, 'lectin_specificity.pkl')
    lectin_specificity = load(open(data_path, 'rb'))
    globals()[name] = lectin_specificity  # Cache it to avoid reloading
    return lectin_specificity
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class LazyLoader:
  def __init__(self, package, directory, prefix = 'glycomics_'):
    self.package = package
    self.directory = directory
    self.prefix = prefix
    self._datasets = {}

  def __getattr__(self, name):
    if name not in self._datasets:
      filename = f"{self.prefix}{name}.csv"
      try:
        with resources.open_text(self.package + '.' + self.directory, filename) as f:
          self._datasets[name] = pd.read_csv(f)
      except FileNotFoundError:
        raise AttributeError(f"No dataset named {name} available under {self.directory} with prefix {self.prefix}.")
    return self._datasets[name]

  def __dir__(self):
    files = resources.contents(self.package + '.' + self.directory)
    dataset_names = [file[len(self.prefix):-4] for file in files if file.startswith(self.prefix) and file.endswith('.csv')]
    return dataset_names


glycomics_data_loader = LazyLoader("glycowork", "glycan_data")
lectin_array_data_loader = LazyLoader("glycowork", "glycan_data", prefix = 'lectin_array_')
glycoproteomics_data_loader = LazyLoader("glycowork", "glycan_data", prefix = 'glycoproteomics_')


linkages = {
  '1-4', '1-6', 'a1-1', 'a1-2', 'a1-3', 'a1-4', 'a1-5', 'a1-6', 'a1-7', 'a1-8', 'a1-9', 'a1-11', 'a1-?', 'a2-1', 'a2-2', 'a2-3', 'a2-4', 'a2-5', 'a2-6', 'a2-7', 'a2-8', 'a2-9',
  'a2-11', 'a2-?', 'b1-1', 'b1-2', 'b1-3', 'b1-4', 'b1-5', 'b1-6', 'b1-7', 'b1-8', 'b1-9', 'b1-?', 'b2-1', 'b2-2', 'b2-3', 'b2-4', 'b2-5', 'b2-6', 'b2-7', 'b2-8', '?1-?',
  '?2-?', '?1-2', '?1-3', '?1-4', '?1-6', '?2-3', '?2-6', '?2-8'
  }
Hex = {'Glc', 'Gal', 'Man', 'Ins', 'Galf', 'Hex'}
HexOS = [f"{h}OS" for h in Hex]
dHex = {'Fuc', 'Qui', 'Rha', 'dHex'}
HexA = {'GlcA', 'ManA', 'GalA', 'IdoA', 'HexA'}
HexN = {'GlcN', 'ManN', 'GalN', 'HexN'}
HexNAc = {'GlcNAc', 'GalNAc', 'ManNAc', 'HexNAc'}
HexNAcOS = [f"{h}OS" for h in HexNAc]
Pen = {'Ara', 'Xyl', 'Rib', 'Lyx', 'Pen'}
Sia = {'Neu5Ac', 'Neu5Gc', 'Kdn', 'Sia'}
modification_map = {'6S': {'GlcNAc', 'Gal'}, '3S': {'Gal'}, '4S': {'GalNAc'},
                    'OS': {'GlcNAc', 'Gal', 'GalNAc'}}


def unwrap(nested_list: list) -> list:
  """converts a nested list into a flat list"""
  return list(chain(*nested_list))


def find_nth(haystack: str, needle: str, n: int) -> int:
  """finds n-th instance of motif\n
  | Arguments:
  | :-
  | haystack (string): string to search for motif
  | needle (string): motif
  | n (int): n-th occurrence in string (not zero-indexed)\n
  | Returns:
  | :-
  | Returns starting index of n-th occurrence in string"""
  start = haystack.find(needle)
  while start >= 0 and n > 1:
    start = haystack.find(needle, start+len(needle))
    n -= 1
  return start


def find_nth_reverse(string: str, substring: str, n: int, ignore_branches: bool = False) -> int:
  # Reverse the string and the substring
  reversed_string = string[::-1]
  reversed_substring = substring[::-1]
  # Initialize the start index for the search
  start_index = 0
  # Loop to find the n-th occurrence
  for i in range(n):
    # Find the next occurrence index
    idx = reversed_string.find(reversed_substring, start_index)
    # If the substring is not found, return -1
    if idx == -1:
      return -1
    # Update the start index
    start_index = idx + len(substring)
  # Calculate and return the original starting index
  original_start_index = len(string) - start_index
  if ignore_branches:
    # Check if there is an immediate branch preceding the match
    branch_end_idx = original_start_index - 1
    if branch_end_idx > 0 and string[branch_end_idx] == ']' and string[branch_end_idx - 1] != '[':
      # Find the start of the immediate branch
      bracket_count = 1
      for i in range(branch_end_idx - 1, -1, -1):
        if string[i] == ']':
          bracket_count += 1
        elif string[i] == '[':
          bracket_count -= 1
        if bracket_count == 0:
          original_start_index = i
          break
  return original_start_index


def remove_unmatched_brackets(s: str) -> str:
  """Removes all unmatched brackets from the string s.\n
  | Arguments:
  | :-
  | s (string): glycan string in IUPAC-condensed\n
  | Returns:
  | :-
  | Returns glycan without unmatched brackets"""
  while True:
    # Keep track of the indexes of the brackets
    stack = []
    unmatched_open = set()
    unmatched_close = set()
    for i, char in enumerate(s):
      if char == '[':
        stack.append(i)
      elif char == ']':
        if stack:
          stack.pop()
        else:
          unmatched_close.add(i)
    unmatched_open.update(stack)
    # If there are no unmatched brackets, break the loop
    if not unmatched_open and not unmatched_close:
      break
    # Build a new string without the unmatched brackets
    s = ''.join([char for i, char in enumerate(s) if i not in unmatched_open and i not in unmatched_close])
  return s


def reindex(df_new: pd.DataFrame, df_old: pd.DataFrame, out_col: str, ind_col: str, inp_col: str) -> list:
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
  | Returns out_col from df_old in the same order of inp_col in df_new"""
  if ind_col != inp_col:
    print("Mismatching column names for ind_col and inp_col. Doesn't mean it's wrong but pay attention.")
  return [df_old[out_col].values.tolist()[df_old[ind_col].values.tolist().index(k)] for k in df_new[inp_col].values.tolist()]


def stringify_dict(dicty: dict) -> str:
  """Converts dictionary into a string\n
  | Arguments:
  | :-
  | dicty (dictionary): dictionary\n
  | Returns:
  | :-
  | Returns string of type key:value for sorted items"""
  dicty = dict(sorted(dicty.items()))
  return ''.join(f"{key}{value}" for key, value in dicty.items())


def replace_every_second(string: str, old_char: str, new_char: str) -> str:
  """function to replace every second occurrence of old_char in string with new_char\n
  | Arguments:
  | :-
  | string (string): a string
  | old_char (string): a string character to be replaced (every second occurrence)
  | new_char (string): the string character to replace old_char with\n
  | Returns:
  | :-
  | Returns string with replaced characters"""
  count = 0
  result = []
  for char in string:
    if char == old_char:
      count += 1
      result.append(new_char if count % 2 == 0 else char)
    else:
      result.append(char)
  return ''.join(result)


def multireplace(string: str, remove_dic: dict[str, str]) -> str:
  """Replaces all occurences of items in a set with a given string\n
  | Arguments:
  | :-
  | string (str): string to perform replacements on
  | remove_dic (set): dict of form to_replace:replace_with\n
  | Returns:
  | :-
  | (str) modified string"""
  for k, v in remove_dic.items():
    string = string.replace(k, v)
  return string


def strip_suffixes(columns: list) -> list[str]:
  """Strip numerical suffixes like .1, .2, etc., from column names."""
  return [re.sub(r"\.\d+$", "", str(name)) for name in columns]


def build_custom_df(df: pd.DataFrame, kind: str = 'df_species') -> pd.DataFrame:
  """creates custom df from df_glycan\n
  | Arguments:
  | :-
  | df (dataframe): df_glycan / sugarbase
  | kind (string): whether to create 'df_species', 'df_tissue', or 'df_disease' from df_glycan; default:df_species\n
  | Returns:
  | :-
  | Returns custom df in the form of one glycan - species/tissue/disease association per row"""
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
  df = df.loc[df[cols[1]].str.len() > 0, cols]
  df = df.set_index('glycan')
  df = df.explode(cols[1:]).reset_index()
  df = df.sort_values([cols[1], 'glycan'], ascending = [True, True])
  df = df.reset_index(drop = True)
  return df


def download_model(file_id: str, local_path: str = 'model_weights.pt') -> None:
  """Download the model weights file from Google Drive."""
  file_id = file_id.split('/d/')[1].split('/view')[0]
  url = f'https://drive.google.com/uc?id={file_id}'
  gdown.download(url, local_path, quiet = False)
  print("Download completed.")


class DataFrameSerializer:
  """A utility class for serializing and deserializing pandas DataFrames with complex data types
  in a version-independent manner."""

  @staticmethod
  def _serialize_cell(value: Any) -> Dict[str, Any]:
    """Convert a cell value to a serializable format with type information."""
    if isinstance(value, list):
      return {
        'type': 'list',
        'value': [str(item) for item in value]
      }
    elif isinstance(value, dict):
      # Check if it's a string:int dictionary
      if all(isinstance(k, str) and isinstance(v, int) for k, v in value.items()):
        return {
          'type': 'str_int_dict',
          'value': {str(k): int(v) for k, v in value.items()}
        }
      # Fall back to string:string for other dictionaries
      return {
        'type': 'dict',
        'value': {str(k): str(v) for k, v in value.items()}
      }
    elif pd.isna(value):
      return {
        'type': 'null',
        'value': None
      }
    else:
      return {
        'type': 'primitive',
        'value': str(value)
      }

  @staticmethod
  def _deserialize_cell(cell_data: Dict[str, Any]) -> Any:
    """Convert a serialized cell back to its original type."""
    if cell_data['type'] == 'list':
      return cell_data['value']
    elif cell_data['type'] == 'str_int_dict':
      return {str(k): int(v) for k, v in cell_data['value'].items()}
    elif cell_data['type'] == 'dict':
      return cell_data['value']
    elif cell_data['type'] == 'null':
      return None
    else:
      return cell_data['value']

  @classmethod
  def serialize(cls, df: pd.DataFrame, path: str) -> None:
    """Serialize a DataFrame to JSON with type information.

    Args:
      df: pandas DataFrame to serialize
      path: file path to save the serialized data"""
    data = {
      'columns': list(df.columns),
      'index': list(df.index),
      'data': []
    }

    for _, row in df.iterrows():
      serialized_row = [cls._serialize_cell(val) for val in row]
      data['data'].append(serialized_row)

    with open(path, 'w') as f:
      json.dump(data, f)

  @classmethod
  def deserialize(cls, path: str) -> pd.DataFrame:
    """Deserialize a DataFrame from JSON.

    Args:
      path: file path to load the serialized data from

    Returns:
      pandas DataFrame with restored data types"""
    with open(path, 'r') as f:
      data = json.load(f)

    deserialized_data = []
    for row in data['data']:
      deserialized_row = [cls._deserialize_cell(cell) for cell in row]
      deserialized_data.append(deserialized_row)

    return pd.DataFrame(
      data = deserialized_data,
      columns = data['columns'],
      index = data['index']
    )

serializer = DataFrameSerializer()
