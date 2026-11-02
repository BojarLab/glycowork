from glycowork.motif.processing import canonicalize_iupac
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import random
import re
import numpy as np
import pandas as pd


def seed_wildcard_hierarchy(glycans: list[str], # list of IUPAC-condensed glycans
                            labels: list[float |  int | str], # list of prediction labels
                            wildcard_list: list[str], # glycoletters covered by wildcard
                            wildcard_name: str, # name for wildcard in IUPAC nomenclature
                            r: float = 0.1,  # rate of replacement
                            random_state: int | None = 42
                            # optional random state for reproducibility, matching the splits this feeds
                            ) -> tuple[list[str], list[float | int | str]]:  # glycans and labels with wildcards
    "adds dataframe rows in which glycan parts have been replaced with the appropriate wildcards"
    local_rng = random.Random(random_state)
    # Only whole glycoletters are replaced (between brackets, braces, or the string ends), as substring replacement turned LDManHep into LDHexHep and Galf into Hexf
    patterns = {j: re.compile(rf'(?<![^()\[\]{{}}]){re.escape(j)}(?![^()\[\]{{}}])') for j in wildcard_list}
    added_glycans_labels = [(patterns[j].sub(wildcard_name, glycan), label) for glycan, label in zip(glycans, labels)
                            for j in wildcard_list if patterns[j].search(glycan) and local_rng.uniform(0, 1) < r]
    if added_glycans_labels:
        added_glycans, added_labels = zip(*added_glycans_labels)
        return glycans + list(added_glycans), labels + list(added_labels)
    return glycans, labels


def hierarchy_filter(df_in: pd.DataFrame, # dataframe of glycan sequences and taxonomic labels
                     rank: str = 'Domain', # taxonomic rank to filter
                     min_seq: int = 5, # minimum glycans per class
                     wildcard_seed: bool = False, # seed wildcard glycoletters
                     wildcard_list: list[str] | None = None, # glycoletters for wildcard
                     wildcard_name: str | None = None, # wildcard name in IUPAC
                     r: float = 0.1, # replacement rate
                     col: str = 'glycan' # column name for glycans
                     ) -> tuple[list[str], list[str], list[int], list[int], list[int], list[str], dict[str, int]]: # train/val splits and mappings
    "stratified data split in train/test at the taxonomic level, removing duplicate glycans and infrequent classes"
    if wildcard_seed and (not wildcard_list or not wildcard_name):
        raise ValueError("wildcard_seed = True requires wildcard_list and wildcard_name.")
    # Get unique classes in rank; only the rank and glycan columns are ever read, so the other ranks need no dropping
    class_list = sorted(set(df_in[rank].values.tolist()) - {'undetermined'})
    # For each class in rank, get unique set of glycans; one stable sort by class replaces a full-frame mask per class, which was quadratic at Genus/Species level
    order = {c: k for k, c in enumerate(class_list)}
    df = df_in[[c in order for c in df_in[rank].tolist()]].drop_duplicates([rank, col], keep = 'first')
    df = df.iloc[np.argsort([order[c] for c in df[rank].tolist()], kind = 'stable')]
    # Only keep classes in rank with minimum number of glycans, then map them to integers
    counts = df[rank].value_counts()
    df = df[df[rank].isin(counts.index[counts >= min_seq])]
    class_list = sorted(set(df[rank]))
    class_converter = {c: k for k, c in enumerate(class_list)}
    # Split each class proportionally
    all_x, all_y = df[col].tolist(), [class_converter[c] for c in df[rank]]
    train_idx, val_idx = next(StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42).split(all_x, all_y))
    train_x, train_y = [all_x[k] for k in train_idx], [all_y[k] for k in train_idx]
    val_x, val_y = [all_x[k] for k in val_idx], [all_y[k] for k in val_idx]
    if wildcard_seed:
        train_x, train_y = seed_wildcard_hierarchy(train_x, train_y, wildcard_list = wildcard_list, wildcard_name = wildcard_name, r = r)
        val_x, val_y = seed_wildcard_hierarchy(val_x, val_y, wildcard_list = wildcard_list, wildcard_name = wildcard_name, r = r)
    # Duplicates are only dropped within a class, so a glycan of two classes otherwise reaches both splits; wildcarding can additionally map distinct sequences onto the same string
    train_set = set(train_x)
    keep = [k for k, g in enumerate(val_x) if g not in train_set]
    val_x, val_y = [val_x[k] for k in keep], [val_y[k] for k in keep]
    id_val = [k for k, g in enumerate(val_x) for _ in g]
    return train_x, val_x, train_y, val_y, id_val, class_list, class_converter


def general_split(glycans: list[str], # list of IUPAC-condensed glycans
                  labels: list[float | int | str], # list of prediction labels
                  test_size: float = 0.2 # size of test set
                  ) -> tuple[list[str], list[str], list[float | int | str], list[float | int | str]]: # train/test splits
    "splits glycans and labels into train / test sets"
    return train_test_split(glycans, labels, shuffle = True, test_size = test_size, random_state = 42)


def prepare_multilabel(df: pd.DataFrame, # dataframe with one glycan-association per row
                       rank: str = 'Species', # label column to use
                       glycan_col: str = 'glycan' # column with glycan sequences
                       ) -> tuple[list[str], list[list[float]]]: # unique glycans and their label vectors
    "converts a one row per glycan-species/tissue/disease association file to a format of one glycan - all associations"
    df = df.copy()
    df[glycan_col] = canonicalize_iupac(df[glycan_col])
    glycans = list(dict.fromkeys(df[glycan_col].values.tolist()))
    class_list = sorted(list(set(df[rank].values.tolist())))
    labels = [[0.] * len(class_list) for k in range(len(glycans))]
    row_of, col_of = {g: k for k, g in enumerate(glycans)}, {c: k for k, c in enumerate(class_list)}
    # Get all class occurrences of glycan to construct multi-label
    for g, c in zip(df[glycan_col], df[rank]):
        labels[row_of[g]][col_of[c]] = 1.
    return glycans, labels
