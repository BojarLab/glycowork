from glycowork.motif.processing import canonicalize_iupac
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import copy
import random
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
    added_glycans_labels = [(glycan.replace(j, wildcard_name), label) for glycan, label in zip(glycans, labels)
                            for j in wildcard_list if j in glycan and local_rng.uniform(0, 1) < r]
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
    df = copy.deepcopy(df_in)
    # Get all non-selected ranks and drop from df
    rank_list = ['Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom', 'Domain']
    rank_list.remove(rank)
    df.drop([c for c in rank_list if c in df.columns], axis = 1, inplace = True)
    # Get unique classes in rank
    class_list = sorted(set(df[rank].values.tolist()) - {'undetermined'})
    temp = []
    # For each class in rank, get unique set of glycans
    for classy in class_list:
        t = df[df[rank] == classy]
        t = t.drop_duplicates(col, keep = 'first')
        temp.append(t)
    df = pd.concat(temp).reset_index(drop = True)
    # Only keep classes in rank with minimum number of glycans
    counts = df[rank].value_counts()
    allowed_classes = counts.index[counts >= min_seq].tolist()
    df = df[df[rank].isin(allowed_classes)]
    # Map string classes to integers
    class_list = list(sorted(list(set(df[rank].values.tolist()))))
    class_converter = {class_list[k]: k for k in range(len(class_list))}
    df[rank] = [class_converter[k] for k in df[rank]]
    # Split each class proportionally
    all_x, all_y = df[col].values.tolist(), df[rank].values.tolist()
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
    sss.get_n_splits(all_x, all_y)
    for i, j in sss.split(all_x, all_y):
        train_x = [all_x[k] for k in i]
        train_y = [all_y[k] for k in i]
        if wildcard_seed:
            train_x, train_y = seed_wildcard_hierarchy(train_x, train_y, wildcard_list = wildcard_list,
                                                       wildcard_name = wildcard_name, r = r)
        val_x = [all_x[k] for k in j]
        val_y = [all_y[k] for k in j]
        if wildcard_seed:
            val_x, val_y = seed_wildcard_hierarchy(val_x, val_y, wildcard_list = wildcard_list,
                                                   wildcard_name = wildcard_name, r = r)
        # Duplicates are only dropped within a class, so a glycan of two classes otherwise reaches both splits; wildcarding can additionally map distinct sequences onto the same string
        train_set = set(train_x)
        keep = [k for k, g in enumerate(val_x) if g not in train_set]
        val_x, val_y = [val_x[k] for k in keep], [val_y[k] for k in keep]
        id_val = list(range(len(val_x)))
        len_val_x = [len(k) for k in val_x]
        id_val = [[id_val[k]] * len_val_x[k] for k in range(len(len_val_x))]
        id_val = [item for sublist in id_val for item in sublist]
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
    df[glycan_col] = [canonicalize_iupac(g) for g in df[glycan_col]]
    glycans = list(dict.fromkeys(df[glycan_col].values.tolist()))
    class_list = sorted(list(set(df[rank].values.tolist())))
    labels = [[0.] * len(class_list) for k in range(len(glycans))]
    row_of, col_of = {g: k for k, g in enumerate(glycans)}, {c: k for k, c in enumerate(class_list)}
    # Get all class occurrences of glycan to construct multi-label
    for g, c in zip(df[glycan_col], df[rank]):
        labels[row_of[g]][col_of[c]] = 1.
    return glycans, labels
