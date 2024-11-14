from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from typing import Dict, List, Tuple, Union, Optional
import copy
import random
import pandas as pd


def seed_wildcard_hierarchy(glycans: List[str], # list of IUPAC-condensed glycans
                          labels: List[Union[float, int, str]], # list of prediction labels
                          wildcard_list: List[str], # glycoletters covered by wildcard
                          wildcard_name: str, # name for wildcard in IUPAC nomenclature
                          r: float = 0.1 # rate of replacement
                         ) -> Tuple[List[str], List[Union[float, int, str]]]: # glycans and labels with wildcards
  "adds dataframe rows in which glycan parts have been replaced with the appropriate wildcards"
  added_glycans_labels = [(glycan.replace(j, wildcard_name), label)
                             for glycan, label in zip(glycans, labels)
                             for j in wildcard_list if j in glycan and random.uniform(0, 1) < r]
  if added_glycans_labels:
    added_glycans, added_labels = zip(*added_glycans_labels)
    return glycans + list(added_glycans), labels + list(added_labels)
  return glycans, labels


def hierarchy_filter(df_in: pd.DataFrame, # dataframe of glycan sequences and taxonomic labels
                    rank: str = 'Domain', # taxonomic rank to filter
                    min_seq: int = 5, # minimum glycans per class
                    wildcard_seed: bool = False, # seed wildcard glycoletters
                    wildcard_list: Optional[List[str]] = None, # glycoletters for wildcard
                    wildcard_name: Optional[str] = None, # wildcard name in IUPAC
                    r: float = 0.1, # replacement rate
                    col: str = 'glycan' # column name for glycans
                   ) -> Tuple[List[str], List[str], List[int], List[int], List[int], List[str], Dict[str, int]]: # train/val splits and mappings
  "stratified data split in train/test at the taxonomic level, removing duplicate glycans and infrequent classes"
  df = copy.deepcopy(df_in)
  # Get all non-selected ranks and drop from df
  rank_list = ['Species', 'Genus', 'Family', 'Order',
               'Class', 'Phylum', 'Kingdom', 'Domain']
  rank_list.remove(rank)
  df.drop(rank_list, axis = 1, inplace = True)
  # Get unique classes in rank
  class_list = list(set(df[rank].values.tolist()))
  class_list = [k for k in class_list if k != 'undetermined']
  temp = []
  # For each class in rank, get unique set of glycans
  for classy in class_list:
    t = df[df[rank] == classy]
    t = t.drop_duplicates('glycan', keep = 'first')
    temp.append(t)
  df = pd.concat(temp).reset_index(drop = True)
  # Only keep classes in rank with minimum number of glycans
  counts = df[rank].value_counts()
  allowed_classes = [counts.index.tolist()[k] for k in range(len(counts.index.tolist())) if (counts >= min_seq).values.tolist()[k]]
  df = df[df[rank].isin(allowed_classes)]
  # Map string classes to integers
  class_list = list(sorted(list(set(df[rank].values.tolist()))))
  class_converter = {class_list[k]: k for k in range(len(class_list))}
  df[rank] = [class_converter[k] for k in df[rank]]
  # Split each class proportionally
  sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2)
  sss.get_n_splits(df[col].values.tolist(), df[rank].values.tolist())
  for i, j in sss.split(df[col].values.tolist(), df[rank].values.tolist()):
    train_x = [df[col].values.tolist()[k] for k in i]
    train_y = [df[rank].values.tolist()[k] for k in i]
    if wildcard_seed:
      train_x, train_y = seed_wildcard_hierarchy(train_x, train_y, wildcard_list = wildcard_list,
                                                 wildcard_name = wildcard_name, r = r)

    val_x = [df[col].values.tolist()[k] for k in j]
    val_y = [df[rank].values.tolist()[k] for k in j]
    if wildcard_seed:
      val_x, val_y = seed_wildcard_hierarchy(val_x, val_y, wildcard_list = wildcard_list,
                                             wildcard_name = wildcard_name, r = r)
    id_val = list(range(len(val_x)))
    len_val_x = [len(k) for k in val_x]
    id_val = [[id_val[k]] * len_val_x[k] for k in range(len(len_val_x))]
    id_val = [item for sublist in id_val for item in sublist]

  return train_x, val_x, train_y, val_y, id_val, class_list, class_converter


def general_split(glycans: List[str], # list of IUPAC-condensed glycans
                 labels: List[Union[float, int, str]], # list of prediction labels
                 test_size: float = 0.2 # size of test set
                ) -> Tuple[List[str], List[str], List[Union[float, int, str]], List[Union[float, int, str]]]: # train/test splits
  "splits glycans and labels into train / test sets"
  return train_test_split(glycans, labels, shuffle = True, test_size = test_size, random_state = 42)


def prepare_multilabel(df: pd.DataFrame, # dataframe with one glycan-association per row
                      rank: str = 'Species', # label column to use
                      glycan_col: str = 'glycan' # column with glycan sequences
                     ) -> Tuple[List[str], List[List[float]]]: # unique glycans and their label vectors
  "converts a one row per glycan-species/tissue/disease association file to a format of one glycan - all associations"
  glycans = list(set(df[glycan_col].values.tolist()))
  class_list = list(set(df[rank].values.tolist()))
  labels = [[0.]*len(class_list) for k in range(len(glycans))]
  # Get all class occurrences of glycan to construct multi-label
  for k, glyc in enumerate(glycans):
    sub_classes = df[df[glycan_col] == glyc][rank].values.tolist()
    for j in sub_classes:
      labels[k][class_list.index(j)] = 1.
  return glycans, labels
