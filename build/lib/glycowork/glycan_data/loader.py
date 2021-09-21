import pandas as pd
import os
import pickle
import pkg_resources

from glycowork.motif.processing import get_lib

io = pkg_resources.resource_stream(__name__, "glyco_targets_species_seq_all_V4.csv")
df_species = pd.read_csv(io)
io = pkg_resources.resource_stream(__name__, "v4_sugarbase.csv")
df_glycan = pd.read_csv(io)
io = pkg_resources.resource_stream(__name__, "df_glysum_v4.csv")
df_glysum = pd.read_csv(io)
df_glysum = df_glysum.iloc[:,1:]
io = pkg_resources.resource_stream(__name__, "glycan_motifs.csv")
motif_list = pd.read_csv(io)
io = pkg_resources.resource_stream(__name__, "influenza_glycan_binding.csv")
influenza_binding = pd.read_csv(io)
this_dir, this_filename = os.path.split(__file__)  # Get path of data.pkl
data_path = os.path.join(this_dir, 'glycan_representations_species.pkl')
glycan_emb = pickle.load(open(data_path, 'rb'))

lib = get_lib(list(set(df_glycan.glycan.values.tolist() +
                       df_species.target.values.tolist() +
                       motif_list.motif.values.tolist() +
                       influenza_binding.columns.values.tolist()[:-1] +
                       ['monosaccharide','Sia'])))

linkages = ['a1-2','a1-3','a1-4','a1-6','a2-3','a2-6','a2-8','b1-2','b1-3','b1-4','b1-6']
Hex = ['Glc', 'Gal', 'Man', 'Hex']
dHex = ['Fuc', 'Qui', 'Rha', 'dHex']
HexNAc = ['GlcNAc', 'GalNAc', 'ManNAc', 'HexNAc']
Sia = ['Neu5Ac', 'Neu5Gc', 'Kdn']

def unwrap(nested_list):
  """converts a nested list into a flat list"""
  out = [item for sublist in nested_list for item in sublist]
  return out

def find_nth(haystack, needle, n):
  """finds n-th instance of motif\n
  | Arguments:
  | :-
  | haystack (string): string to search for motif
  | needle (string): motif
  | n (int): n-th occurrence in string\n
  | Returns:
  | :-
  | Returns starting index of n-th occurrence in string 
  """
  start = haystack.find(needle)
  while start >= 0 and n > 1:
    start = haystack.find(needle, start+len(needle))
    n -= 1
  return start

def load_file(file):
  """loads .csv files from glycowork package\n
  | Arguments:
  | :-
  | file (string): name of the file to be loaded
  """
  try:
    temp = pd.read_csv("../glycan_data/" + file)
  except:
    temp = pd.read_csv("glycowork/glycan_data/" + file)
  return temp
