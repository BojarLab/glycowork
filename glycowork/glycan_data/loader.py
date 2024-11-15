import re
import json
import gdown
import pandas as pd
from pickle import load
from os import path
from itertools import chain
from importlib import resources
from typing import Any, Dict, List, Optional

with resources.files("glycowork.glycan_data").joinpath("glycan_motifs.csv").open(encoding = 'utf-8-sig') as f:
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


def unwrap(nested_list: List[Any] # list to be flattened
         ) -> List[Any]: # flattened list
  "converts a nested list into a flat list"
  return list(chain(*nested_list))


def find_nth(haystack: str, # string to search for motif
            needle: str, # motif
            n: int # n-th occurrence in string (not zero-indexed)
           ) -> int: # starting index of n-th occurrence
  "finds n-th instance of motif"
  start = haystack.find(needle)
  while start >= 0 and n > 1:
    start = haystack.find(needle, start+len(needle))
    n -= 1
  return start


def find_nth_reverse(string: str, # string to search
                    substring: str, # substring to find
                    n: int, # n-th occurrence from end
                    ignore_branches: bool = False # whether to ignore branches when counting
                   ) -> int: # position of n-th occurrence from end
  "finds n-th instance of motif from end of string"
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


def remove_unmatched_brackets(s: str # glycan string in IUPAC-condensed
                           ) -> str: # glycan without unmatched brackets
  "Removes all unmatched brackets from the string s"
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


def reindex(df_new: pd.DataFrame, # dataframe with new row order
           df_old: pd.DataFrame, # dataframe with old row order
           out_col: str, # column name in df_old to reindex
           ind_col: str, # column name in df_old for index
           inp_col: str # column name in df_new for new order
          ) -> list: # out_col from df_old reordered to match inp_col in df_new
  "Returns columns values in order of new dataframe rows"
  if ind_col != inp_col:
    print("Mismatching column names for ind_col and inp_col. Doesn't mean it's wrong but pay attention.")
  return [df_old[out_col].values.tolist()[df_old[ind_col].values.tolist().index(k)] for k in df_new[inp_col].values.tolist()]


def stringify_dict(dicty: Dict[Any, Any] # dictionary to convert
                 ) -> str: # string of type key:value for sorted items
  "Converts dictionary into a string"
  dicty = dict(sorted(dicty.items()))
  return ''.join(f"{key}{value}" for key, value in dicty.items())


def replace_every_second(string: str, # input string
                        old_char: str, # character to replace
                        new_char: str # character to replace with
                       ) -> str: # modified string
  "function to replace every second occurrence of old_char in string with new_char"
  count = 0
  result = []
  for char in string:
    if char == old_char:
      count += 1
      result.append(new_char if count % 2 == 0 else char)
    else:
      result.append(char)
  return ''.join(result)


def multireplace(string: str, # string to perform replacements on
                remove_dic: Dict[str, str] # dict of form to_replace:replace_with
               ) -> str: # modified string
  "Replaces all occurences of items in a set with a given string"
  for k, v in remove_dic.items():
    string = string.replace(k, v)
  return string


def strip_suffixes(columns: List[Any] # column names
                 ) -> List[str]: # column names without numerical suffixes
  "Strip numerical suffixes like .1, .2, etc., from column names"
  return [re.sub(r"\.\d+$", "", str(name)) for name in columns]


def build_custom_df(df: pd.DataFrame, # df_glycan / sugarbase
                   kind: str = 'df_species' # whether to create 'df_species', 'df_tissue', or 'df_disease'
                  ) -> pd.DataFrame: # custom df with one glycan - species/tissue/disease association per row
  "creates custom df from df_glycan"
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


def download_model(file_id: str, # Google Drive file ID
                  local_path: str = 'model_weights.pt' # where to save model file
                 ) -> None:
  "Download the model weights file from Google Drive"
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
  def serialize(cls, df: pd.DataFrame, # DataFrame to serialize
                 path: str # file path to save serialized data
                ) -> None:
    "Serialize a DataFrame to JSON with type information"
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
  def deserialize(cls, path: str # file path to load serialized data
                  ) -> pd.DataFrame: # DataFrame with restored data types
    "Deserialize a DataFrame from JSON"
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
