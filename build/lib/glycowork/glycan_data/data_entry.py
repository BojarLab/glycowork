import networkx as nx
from glycowork.glycan_data.loader import lib
from glycowork.motif.graph import glycan_to_graph, glycan_to_nxGraph, compare_glycans, fast_compare_glycans

def check_presence(glycan, df, colname = 'target', libr = None,
                   name = None, rank = 'Species', fast = False):
  """checks whether glycan (of that species) is already present in dataset\n
  | Arguments:
  | :-
  | glycan (string): IUPAC-condensed glycan sequence
  | df (dataframe): glycan dataframe where glycans are under colname and ideally taxonomic labels are columns
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset
  | name (string): name of the species (etc.) of interest
  | rank (string): column name for filtering; default: species
  | fast (bool): True uses precomputed glycan graphs, only use if df has column 'graph' with glycan graphs\n
  | Returns:
  | :-
  | Returns text output regarding whether the glycan is already in df
  """
  if libr is None:
    libr = lib
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
