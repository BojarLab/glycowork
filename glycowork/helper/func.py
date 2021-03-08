import pandas as pd

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
    return pd.read_csv('glycowork/glycan_data/' + file)
