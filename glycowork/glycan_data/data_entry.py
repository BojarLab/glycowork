import pandas as pd
from glycowork.motif.processing import check_nomenclature
from glycowork.motif.graph import glycan_to_nxGraph, compare_glycans


def check_presence(glycan: str, df: pd.DataFrame, colname: str = 'glycan', 
                  name: str | None = None, rank: str = 'Species', fast: bool = False) -> None:
  """checks whether glycan (of that species) is already present in dataset\n
  | Arguments:
  | :-
  | glycan (string): IUPAC-condensed glycan sequence
  | df (dataframe): glycan dataframe where glycans are under colname and ideally taxonomic labels are columns
  | name (string): name of the species (etc.) of interest
  | rank (string): column name for filtering; default: species
  | fast (bool): True uses precomputed glycan graphs, only use if df has column 'graph' with glycan graphs\n
  | Returns:
  | :-
  | Returns text output regarding whether the glycan is already in df"""
  if any([p in glycan for p in ['RES', '=']]) or not isinstance(glycan, str):
    check_nomenclature(glycan)
    return
  if name is not None:
    name = name.replace(" ", "_")
    df = df[df[rank] == name]
    if len(df) == 0:
      print("This is the best: %s is not in dataset" % name)
  if fast:
    ggraph = glycan_to_nxGraph(glycan)
    check_all = [compare_glycans(ggraph, k) for k in df.graph]
  else:
    check_all = [compare_glycans(glycan, k) for k in df[colname]]
  if any(check_all):
    print("Glycan already in dataset.")
  else:
    print("It's your lucky day, this glycan is new!")
