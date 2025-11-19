import pandas as pd
from glycowork.motif.processing import check_nomenclature
from glycowork.motif.graph import glycan_to_nxGraph, compare_glycans


def check_presence(glycan: str, # IUPAC-condensed glycan sequence
                  df: pd.DataFrame, # glycan dataframe where glycans are under colname
                  colname: str = 'glycan', # column name containing glycans
                  name: str | None = None, # name of species of interest
                  rank: str = 'Species', # column name for filtering
                  fast: bool = False # True uses precomputed glycan graphs
                 ) -> None:
  "checks whether glycan (of that species) is already present in dataset"
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
