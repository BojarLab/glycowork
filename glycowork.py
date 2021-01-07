import pandas as pd
import numpy as np
import re
import time, copy
import random
import torch
import networkx as nx

def get_namespace():
  """returns list of function names in this package"""
  return ['unwrap, find_nth, small_motif_find, min_process_glycans, motif_find, \
process_glycans, character_to_label, string_to_labels, pad_sequence, get_lib, \
glycan_to_graph, dataset_to_graphs, hierarchy_filter, compare_glycans']

def unwrap(nested_list):
  """converts a nested list into a flat list"""
  out = [item for sublist in nested_list for item in sublist]
  return out

def find_nth(haystack, needle, n):
  """finds n-th instance of motif
  haystack -- string to search for motif
  needle -- motif
  n -- n-th occurrence in string

  returns starting index of n-th occurrence in string 
  """
  start = haystack.find(needle)
  while start >= 0 and n > 1:
    start = haystack.find(needle, start+len(needle))
    n -= 1
  return start

def small_motif_find(s):
  """processes IUPACcondensed glycan sequence (string) without splitting it into glycowords"""
  b = s.split('(')
  b = [k.split(')') for k in b]
  b = [item for sublist in b for item in sublist]
  b = [k.strip('[') for k in b]
  b = [k.strip(']') for k in b]
  b = [k.replace('[', '') for k in b]
  b = [k.replace(']', '') for k in b]
  b = '*'.join(b)
  return b

def min_process_glycans(glycan_list):
  """converts list of glycans into a nested lists of glycowords"""
  glycan_motifs = [small_motif_find(k) for k in glycan_list]
  glycan_motifs = [i.split('*') for i in glycan_motifs]
  return glycan_motifs

def motif_find(s, exhaustive = False):
  """processes IUPACcondensed glycan sequence (string) into glycowords
  s -- glycan string
  exhaustive -- True for processing glycans shorter than one glycoword

  returns list of glycowords
  """
  b = s.split('(')
  b = [k.split(')') for k in b]
  b = [item for sublist in b for item in sublist]
  b = [k.strip('[') for k in b]
  b = [k.strip(']') for k in b]
  b = [k.replace('[', '') for k in b]
  b = [k.replace(']', '') for k in b]
  if exhaustive:
    if len(b) >= 5:
      b = ['*'.join(b[i:i+5]) for i in range(0, len(b)-4, 2)]
    else:
      b = ['*'.join(b)]
  else:
    b = ['*'.join(b[i:i+5]) for i in range(0, len(b)-4, 2)]
  return b

def process_glycans(glycan_list, exhaustive = False):
  """wrapper function to process list of glycans into glycowords
  glycan_list -- list of IUPACcondensed glycan sequences (string)
  exhaustive -- True for processing glycans shorter than one glycoword

  returns nested list of glycowords for every glycan
  """
  glycan_motifs = [motif_find(k, exhaustive = exhaustive) for k in glycan_list]
  glycan_motifs = [[i.split('*') for i in k] for k in glycan_motifs]
  return glycan_motifs

def character_to_label(character, libr):
  """tokenizes character by indexing passed library
  character -- character to index
  libr -- list of library items

  returns index of character in library
  """
  character_label = libr.index(character)
  return character_label

def string_to_labels(character_string, libr):
  """tokenizes word by indexing characters in passed library
  character_string -- string of characters to index
  libr -- list of library items

  returns indexes of characters in library
  """
  return list(map(lambda character: character_to_label(character, libr), character_string))

def pad_sequence(seq, max_length, pad_label = len(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T',
     'V','W','Y','X'])):
  """brings all sequences to same length by adding padding token
  seq -- sequence to pad
  max_length -- sequence length to pad to
  pad_label -- which padding label to use

  returns padded sequence
  """
  seq += [pad_label for i in range(max_length-len(seq))]
  return seq

def get_lib(glycan_list, mode='letter'):
  """returns sorted list of unique glycoletters in list of glycans
  mode -- default is letter for glycoletters; change to obtain glycowords"""
  proc = process_glycans(glycan_list)
  lib = unwrap(proc)
  lib = list(set([tuple(k) for k in lib]))
  lib = [list(k) for k in lib]
  if mode=='letter':
    lib = list(sorted(list(set(unwrap(lib)))))
  else:
    lib = list(sorted(list(set([tuple(k) for k in lib]))))
  return lib

def glycan_to_graph(glycan, libr):
  """the monumental function for converting glycans into graphs
  glycan -- IUPACcondensed glycan sequence (string)
  lib -- sorted list of unique glycoletters observed in the glycans of our dataset

  returns (1) a list of labeled glycoletters from the glycan / node list
          (2) two lists to indicate which glycoletters are connected in the glycan graph / edge list
  """
  bracket_count = glycan.count('[')
  parts = []
  branchbranch = []
  branchbranch2 = []
  position_bb = []
  b_counts = []
  bb_count = 0
  #checks for branches-within-branches and handles them
  if bool(re.search('\[[^\]]+\[', glycan)):
    double_pos = [(k.start(),k.end()) for k in re.finditer('\[[^\]]+\[', glycan)]
    for spos, pos in double_pos:
      bracket_count -= 1
      glycan_part = glycan[spos+1:]
      glycan_part = glycan_part[glycan_part.find('['):]
      idx = [k.end() for k in re.finditer('\][^\(]+\(', glycan_part)][0]
      idx2 = [k.start() for k in re.finditer('\][^\(]+\(', glycan_part)][0]
      branchbranch.append(glycan_part[:idx-1].replace(']','').replace('[',''))
      branchbranch2.append(glycan[pos-1:])
      glycan_part = glycan[:pos-1]
      b_counts.append(glycan_part.count('[')-bb_count)
      glycan_part = glycan_part[glycan_part.rfind('[')+1:]
      position_bb.append(glycan_part.count('(')*2)
      bb_count += 1
    for b in branchbranch2:
      glycan =  glycan.replace(b, ']'.join(b.split(']')[1:]))
  main = re.sub("[\[].*?[\]]", "", glycan)
  position = []
  branch_points = [x.start() for x in re.finditer('\]', glycan)]
  for i in branch_points:
    glycan_part = glycan[:i+1]
    glycan_part = re.sub("[\[].*?[\]]", "", glycan_part)
    position.append(glycan_part.count('(')*2)
  parts.append(main)

  for k in range(1,bracket_count+1):
    start = find_nth(glycan, '[', k) + 1
    #checks whether glycan continues after branch
    if bool(re.search("[\]][^\[]+[\(]", glycan[start:])):
      #checks for double branches and removes second branch
      if bool(re.search('\]\[', glycan[start:])):
        glycan_part = re.sub("[\[].*?[\]]", "", glycan[start:])
        end = re.search("[\]].*?[\(]", glycan_part).span()[1] - 1
        parts.append(glycan_part[:end].replace(']',''))
      else:
        end = re.search("[\]].*?[\(]", glycan[start:]).span()[1] + start -1
        parts.append(glycan[start:end].replace(']',''))
    else:
      if bool(re.search('\]\[', glycan[start:])):
        glycan_part = re.sub("[\[].*?[\]]", "", glycan[start:])
        end = len(glycan_part)
        parts.append(glycan_part[:end].replace(']',''))
      else:
        end = len(glycan)
        parts.append(glycan[start:end].replace(']',''))
  
  try:
    for bb in branchbranch:
      parts.append(bb)
  except:
    pass

  parts = min_process_glycans(parts)
  parts_lengths = [len(j) for j in parts]
  parts_tokenized = [string_to_labels(k, lib) for k in parts]
  parts_tokenized = [parts_tokenized[0]] + [parts_tokenized[k][:-1] for k in range(1,len(parts_tokenized))]
  parts_tokenized = [item for sublist in parts_tokenized for item in sublist]
  
  range_list = list(range(len([item for sublist in parts for item in sublist])))
  init = 0
  parts_positions = []
  for k in parts_lengths:
    parts_positions.append(range_list[init:init+k])
    init += k

  for j in range(1,len(parts_positions)-len(branchbranch)):
    parts_positions[j][-1] = position[j-1]
  for j in range(1, len(parts_positions)):
    try:
      for z in range(j+1,len(parts_positions)):
        parts_positions[z][:-1] = [o-1 for o in parts_positions[z][:-1]]
    except:
      pass
  try:
    for i,j in enumerate(range(len(parts_positions)-len(branchbranch), len(parts_positions))):
      parts_positions[j][-1] = parts_positions[b_counts[i]][position_bb[i]]
  except:
    pass

  pairs = []
  for i in parts_positions:
    pairs.append([(i[m],i[m+1]) for m in range(0,len(i)-1)])
  pairs = list(zip(*[item for sublist in pairs for item in sublist]))
  return parts_tokenized, pairs



def dataset_to_graphs(glycan_list, libr, label_type = torch.long, separate = False,
                      context = False, error_catch = False):
  """wrapper function to convert a whole list of glycans into a graph dataset
  glycan_list -- list of IUPACcondensed glycan sequences (string)
  label_type -- which tensor type for label, default is torch.long for binary labels, change to torch.float for continuous
  separate -- True returns node list / edge list / label list as separate files; False returns list of data tuples; default is False
  lib -- sorted list of unique glycoletters observed in the glycans of our dataset
  context -- legacy-ish; used for generating graph context dataset for pre-training; keep at False
  error_catch -- troubleshooting option, True will print glycans that cannot be converted into graphs; default is False

  returns list of node list / edge list / label list data tuples
  """
  if error_catch:
    glycan_graphs = []
    for k in glycan_list:
      try:
        glycan_graphs.append(glycan_to_graph(k, libr))
      except:
        print(k)
  else:
    glycan_graphs = [glycan_to_graph(k, libr) for k in glycan_list]
  if separate:
    glycan_nodes, glycan_edges = zip(*glycan_graphs)
    return list(glycan_nodes), list(glycan_edges), labels
  else:
    if context:
      contexts = [ggraph_to_context(k, lib=lib) for k in glycan_graphs]
      labels = [k[1] for k in contexts]
      labels = [item for sublist in labels for item in sublist]
      contexts = [k[0] for k in contexts]
      contexts = [item for sublist in contexts for item in sublist]
      data = [Data(x = torch.tensor(contexts[k][0], dtype = torch.long),
                 y = torch.tensor(labels[k], dtype = label_type),
                 edge_index = torch.tensor([contexts[k][1][0],contexts[k][1][1]], dtype = torch.long)) for k in range(len(contexts))]
      return data
    else:
      glycan_nodes, glycan_edges = zip(*glycan_graphs)
      glycan_graphs = list(zip(glycan_nodes, glycan_edges))
      data = [Data(x = torch.tensor(k[0], dtype = torch.long),
                 edge_index = torch.tensor([k[1][0],k[1][1]], dtype = torch.long)) for k in glycan_graphs]
      return data

def hierarchy_filter(df_in, rank = 'domain', min_seq = 5):
  """stratified data split in train/test at the taxonomic level, removing duplicate glycans and infrequent classes"""
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
  sss.get_n_splits(df.target.values.tolist(), df[rank].values.tolist())
  for i, j in sss.split(df.target.values.tolist(), df[rank].values.tolist()):
    train_x = [df.target.values.tolist()[k] for k in i]
    len_train_x = [len(k) for k in train_x]

    val_x = [df.target.values.tolist()[k] for k in j]
    id_val = list(range(len(val_x)))
    len_val_x = [len(k) for k in val_x]

    train_y = [df[rank].values.tolist()[k] for k in i]

    val_y = [df[rank].values.tolist()[k] for k in j]
    id_val = [[id_val[k]] * len_val_x[k] for k in range(len(len_val_x))]
    id_val = [item for sublist in id_val for item in sublist]

  return train_x, val_x, train_y, val_y, id_val, class_list, class_converter

def compare_glycans(glycan_a, glycan_b, libr):
  """returns True if glycans are the same and False if not
  glycan_a -- glycan in string format (IUPACcondensed)"""
  glycan_graph_a = glycan_to_graph(glycan_a, libr)
  edgelist = list(zip(glycan_graph_a[1][0], glycan_graph_a[1][1]))
  g1 = nx.from_edgelist(edgelist)
  nx.set_node_attributes(g1, glycan_graph_a[0],'labels')
  
  glycan_graph_b = glycan_to_graph(glycan_b, libr)
  edgelist = list(zip(glycan_graph_b[1][0], glycan_graph_b[1][1]))
  g2 = nx.from_edgelist(edgelist)
  nx.set_node_attributes(g2, glycan_graph_b[0],'labels')

  return nx.is_isomorphic(g1, g2)
  
