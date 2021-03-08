from sklearn.model_selection import StratifiedShuffleSplit
import copy
import random
from glycowork.helper.func import *

df_species = load_file("glyco_targets_species_seq_all_V3.csv")
linkages = ['a1-2','a1-3','a1-4','a1-6','a2-3','a2-6','a2-8','b1-2','b1-3','b1-4','b1-6']

def seed_wildcard_hierarchy(glycan_list, label_list, wildcard_list, wildcard_name, r = 0.1):
  """adds dataframe rows in which glycan parts have been replaced with the appropriate wildcards
  df_in -- dataframe in which the glycan column is called "target" and is the first column
  wildcard_list -- list which glycoletters a wildcard encompasses
  wildcard_name -- how the wildcard should be named in the IUPACcondensed nomenclature
  r -- rate of replacement, default is 0.1 or 10%"""
  added_glycans = []
  added_labels = []
  for k in range(len(glycan_list)):
    temp = glycan_list[k]
    for j in wildcard_list:
      if j in temp:
        if random.uniform(0, 1) < r:
          added_glycans.append(temp.replace(j, wildcard_name))
          added_labels.append(label_list[k])
  glycan_list += added_glycans
  label_list += added_labels
  return glycan_list, label_list


def hierarchy_filter(df_in, rank = 'domain', min_seq = 5, wildcard_seed = False, wildcard_list = None,
                     wildcard_name = None, r = 0.1, col = 'target'):
  """stratified data split in train/test at the taxonomic level, removing duplicate glycans and infrequent classes
  df_in -- dataframe of glycan sequences and taxonomic labels
  rank -- which rank should be filtered; default is 'domain'
  min_seq -- how many glycans need to be present in class to keep it; default is 5
  wildcard_seed -- set to True if you want to seed wildcard glycoletters; default is False
  wildcard_list -- list which glycoletters a wildcard encompasses
  wildcard_name -- how the wildcard should be named in the IUPACcondensed nomenclature
  r -- rate of replacement, default is 0.1 or 10%
  col -- column name for glycan sequences; default: target
  """
  df = copy.deepcopy(df_in)
  rank_list = ['species','genus','family','order','class','phylum','kingdom','domain']
  rank_list.remove(rank)
  df.drop(rank_list, axis = 1, inplace = True)
  class_list = list(set(df[rank].values.tolist()))
  temp = []

  for i in range(len(class_list)):
    t = df[df[rank] == class_list[i]]
    t = t.drop_duplicates('target', keep = 'first')
    temp.append(t)
  df = pd.concat(temp).reset_index(drop = True)

  counts = df[rank].value_counts()
  allowed_classes = [counts.index.tolist()[k] for k in range(len(counts.index.tolist())) if (counts >= min_seq).values.tolist()[k]]
  df = df[df[rank].isin(allowed_classes)]

  class_list = list(sorted(list(set(df[rank].values.tolist()))))
  class_converter = {class_list[k]:k for k in range(len(class_list))}
  df[rank] = [class_converter[k] for k in df[rank].values.tolist()]

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
