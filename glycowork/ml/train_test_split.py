from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import copy
import random
import pandas as pd

def seed_wildcard_hierarchy(glycans, labels, wildcard_list,
                            wildcard_name, r = 0.1):
  """adds dataframe rows in which glycan parts have been replaced with the appropriate wildcards\n
  | Arguments:
  | :-
  | glycans (list): list of IUPAC-condensed glycan sequences as strings
  | labels (list): list of labels used for prediction
  | wildcard_list (list): list which glycoletters a wildcard encompasses
  | wildcard_name (string): how the wildcard should be named in the IUPAC-condensed nomenclature
  | r (float): rate of replacement, default:0.1 or 10%\n
  | Returns:
  | :-
  | Returns list of glycans (strings) and labels (flexible) where some glycan parts have been replaced with wildcard_name
  """
  added_glycans = []
  added_labels = []
  #each loop has the chance of exchanging glycan parts with each wildcard
  for k in range(len(glycans)):
    temp = glycans[k]
    for j in wildcard_list:
      if j in temp:
        if random.uniform(0, 1) < r:
          added_glycans.append(temp.replace(j, wildcard_name))
          added_labels.append(labels[k])
  glycans += added_glycans
  labels += added_labels
  return glycans, labels


def hierarchy_filter(df_in, rank = 'Domain', min_seq = 5, wildcard_seed = False, wildcard_list = None,
                     wildcard_name = None, r = 0.1, col = 'target'):
  """stratified data split in train/test at the taxonomic level, removing duplicate glycans and infrequent classes\n
  | Arguments:
  | :-
  | df_in (dataframe): dataframe of glycan sequences and taxonomic labels
  | rank (string): which rank should be filtered; default:'domain'
  | min_seq (int): how many glycans need to be present in class to keep it; default:5
  | wildcard_seed (bool): set to True if you want to seed wildcard glycoletters; default:False
  | wildcard_list (list): list which glycoletters a wildcard encompasses
  | wildcard_name (string): how the wildcard should be named in the IUPAC-condensed nomenclature
  | r (float): rate of replacement, default:0.1 or 10%
  | col (string): column name for glycan sequences; default:target\n
  | Returns:
  | :-
  | Returns train_x, val_x (lists of glycans (strings) after stratified shuffle split)
  | train_y, val_y (lists of taxonomic labels (mapped integers))
  | id_val (taxonomic labels in text form (strings))
  | class_list (list of unique taxonomic classes (strings))
  | class_converter (dictionary to map mapped integers back to text labels)
  """
  df = copy.deepcopy(df_in)
  #get all non-selected ranks and drop from df
  rank_list = ['Species','Genus','Family','Order',
               'Class','Phylum','Kingdom','Domain']
  rank_list.remove(rank)
  df.drop(rank_list, axis = 1, inplace = True)
  #get unique classes in rank
  class_list = list(set(df[rank].values.tolist()))
  class_list= [k for k in class_list if k != 'undetermined']
  temp = []

  #for each class in rank, get unique set of glycans
  for i in range(len(class_list)):
    t = df[df[rank] == class_list[i]]
    t = t.drop_duplicates('target', keep = 'first')
    temp.append(t)
  df = pd.concat(temp).reset_index(drop = True)

  #only keep classes in rank with minimum number of glycans
  counts = df[rank].value_counts()
  allowed_classes = [counts.index.tolist()[k] for k in range(len(counts.index.tolist())) if (counts >= min_seq).values.tolist()[k]]
  df = df[df[rank].isin(allowed_classes)]

  #map string classes to integers
  class_list = list(sorted(list(set(df[rank].values.tolist()))))
  class_converter = {class_list[k]:k for k in range(len(class_list))}
  df[rank] = [class_converter[k] for k in df[rank].values.tolist()]

  #split each class proportionally
  sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2)
  sss.get_n_splits(df[col].values.tolist(), df[rank].values.tolist())
  for i, j in sss.split(df[col].values.tolist(), df[rank].values.tolist()):
    train_x = [df[col].values.tolist()[k] for k in i]
    train_y = [df[rank].values.tolist()[k] for k in i]
    if wildcard_seed:
      train_x, train_y = seed_wildcard_hierarchy(train_x, train_y, wildcard_list = wildcard_list,
                                                 wildcard_name = wildcard_name, r = r)
    len_train_x = [len(k) for k in train_x]

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

def general_split(glycans, labels, test_size = 0.2):
  """splits glycans and labels into train / test sets\n
  | Arguments:
  | :-
  | glycans (list): list of IUPAC-condensed glycan sequences as strings
  | labels (list): list of labels used for prediction
  | test_size (float): % size of test set; default:0.2 / 20%\n
  | Returns:
  | :-
  | Returns X_train, X_test, y_train, y_test
  """
  return train_test_split(glycans, labels, shuffle = True,
                          test_size = test_size, random_state = 42)

def prepare_multilabel(df, rank = 'Species', glycan_col = 'target'):
  """converts a one row per glycan-species/tissue/disease association file to a format of one glycan - all associations\n
  | Arguments:
  | :-
  | df (dataframe): dataframe where each row is one glycan - species association
  | rank (string): which label column should be used; default:Species
  | glycan_col (string): column name of where the glycan sequences are stored; default:target\n
  | Returns:
  | :-
  | (1) list of unique glycans in df
  | (2) list of lists, where each inner list are all the labels of a glycan
  """
  glycans = list(set(df[glycan_col].values.tolist()))
  class_list = list(set(df[rank].values.tolist()))
  labels = [[0.]*len(class_list) for k in range(len(glycans))]
  #get all class occurrences of glycan to construct multi-label
  for k in range(len(glycans)):
    sub_classes = df[df.target==glycans[k]][rank].values.tolist()
    for j in sub_classes:
      labels[k][class_list.index(j)] = 1.
  return glycans, labels
