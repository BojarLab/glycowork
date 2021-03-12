import pandas as pd
import pkg_resources

from glycowork.motif.processing import get_lib

io = pkg_resources.resource_stream(__name__, "glyco_targets_species_seq_all_V3.csv")
df_species = pd.read_csv(io)
io = pkg_resources.resource_stream(__name__, "v3_sugarbase.csv")
df_glycan = pd.read_csv(io)
io = pkg_resources.resource_stream(__name__, "df_glyco_substitution_iso2.csv")
df_glysum = pd.read_csv(io)

lib = get_lib(list(set(df_glycan.glycan.values.tolist() + df_species.target.values.tolist())))

def unwrap(nested_list):
  """converts a nested list into a flat list"""
  out = [item for sublist in nested_list for item in sublist]
  return out

def find_nth(haystack, needle, n):
  """finds n-th instance of motif
  haystack -- string to search for motif
  needle -- motif
  n -- n-th occurrence in string

  returns starting index of n-th occurrence in string 
  """
  start = haystack.find(needle)
  while start >= 0 and n > 1:
    start = haystack.find(needle, start+len(needle))
    n -= 1
  return start

def load_file(file):
    """loads .csv files from glycowork package
    file -- name of the file to be loaded [string]
    """
    try:
      temp = pd.read_csv("../glycan_data/" + file)
    except:
      temp = pd.read_csv("glycowork/glycan_data/" + file)
    return temp
