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
    if not isinstance(glycan, str) or any(p in glycan for p in ['RES', '=']):
        check_nomenclature(glycan)
        return
    lookup_col = 'graph' if fast else colname
    if lookup_col not in df.columns:
        raise KeyError(f"'{lookup_col}' is not a column of df; available columns are {df.columns.tolist()}")
    if name is not None:
        if rank not in df.columns:
            raise KeyError(f"'{rank}' is not a column of df; available columns are {df.columns.tolist()}")
        name = name.replace(" ", "_")
        df = df[df[rank] == name]
        if len(df) == 0:
            print(f"This is the best: {name} is not in dataset")
    ggraph = glycan_to_nxGraph(glycan) if fast else glycan
    check_all = [compare_glycans(ggraph, k) for k in df[lookup_col]]
    if any(check_all):
        print("Glycan already in dataset.")
    else:
        print("It's your lucky day, this glycan is new!")
