import networkx as nx
from glycowork.glycan_data.loader import lib
from glycowork.motif.graph import glycan_to_graph, glycan_to_nxGraph, compare_glycans, fast_compare_glycans

def check_presence(glycan, df, colname = 'target', libr = lib,
                   name = None, rank = 'species', fast = False):
  """checks whether glycan (of that species) is already present in dataset
  glycan -- IUPACcondensed glycan sequence (string)
  df -- glycan dataframe where glycans are under colname and ideally taxonomic labels are columns
  libr -- sorted list of unique glycoletters observed in the glycans of our dataset
  name -- name of the species (etc.) of interest; string
  rank -- column name for filtering; default: species
  fast -- True uses precomputed glycan graphs, only use if df has column 'graph' with glycan graphs

  returns text output regarding whether the glycan is already in df
  """
  if name is not None:
    name = name.replace(" ", "_")
    df = df[df[rank] == name]
    if len(df) == 0:
      print("This is the best: %s is not in dataset" % name)
  if fast:
    ggraph = glycan_to_nxGraph(glycan, libr = libr)
    check_all = [fast_compare_glycans(ggraph, k) for k in df.graph.values.tolist()]
  else:
    check_all = [compare_glycans(glycan, k) for k in df[colname].values.tolist()]
  if any(check_all):
    print("Glycan already in dataset.")
  else:
    print("It's your lucky day, this glycan is new!")
