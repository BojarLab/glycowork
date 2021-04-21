from glycowork.motif.graph import glycan_to_graph
from glycowork.glycan_data.loader import lib

try:
  import torch
  from torch_geometric.data import Data, DataLoader
except ImportError:
    raise ImportError('<torch or torch_geometric missing; cannot do deep learning>')

def dataset_to_graphs(glycan_list, labels, libr = None, label_type = torch.long, separate = False,
                      context = False, error_catch = False, wo_labels = False):
  """wrapper function to convert a whole list of glycans into a graph dataset\n
  | Arguments:
  | :-
  | glycan_list (list): list of IUPAC-condensed glycan sequences as strings
  | labels (list): list of labels
  | label_type (torch object): which tensor type for label, default is torch.long for binary labels, change to torch.float for continuous
  | separate (bool): True returns node list / edge list / label list as separate files; False returns list of data tuples; default is False
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset
  | context (bool): legacy-ish; used for generating graph context dataset for pre-training; keep at False
  | error_catch (bool): troubleshooting option, True will print glycans that cannot be converted into graphs; default is False
  | wo_labels (bool): change to True if you do not want to pass and receive labels; default is False\n
  | Returns:
  | :-
  | Returns list of node list / edge list / label list data tuples
  """
  if libr is None:
    libr = lib
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
      contexts = [ggraph_to_context(k, lib = lib) for k in glycan_graphs]
      labels = [k[1] for k in contexts]
      labels = [item for sublist in labels for item in sublist]
      contexts = [k[0] for k in contexts]
      contexts = [item for sublist in contexts for item in sublist]
      data = [Data(x = torch.tensor(contexts[k][0], dtype = torch.long),
                 y = torch.tensor(labels[k], dtype = label_type),
                 edge_index = torch.tensor([contexts[k][1][0],contexts[k][1][1]], dtype = torch.long)) for k in range(len(contexts))]
      return data
    else:
      if wo_labels:
        glycan_nodes, glycan_edges = zip(*glycan_graphs)
        glycan_graphs = list(zip(glycan_nodes, glycan_edges))
        data = [Data(x = torch.tensor(k[0], dtype = torch.long),
                     edge_index = torch.tensor([k[1][0],k[1][1]], dtype = torch.long)) for k in glycan_graphs]
        return data
      else:
        glycan_nodes, glycan_edges = zip(*glycan_graphs)
        glycan_graphs = list(zip(glycan_nodes, glycan_edges, labels))
        data = [Data(x = torch.tensor(k[0], dtype = torch.long),
                 y = torch.tensor([k[2]], dtype = label_type),
                 edge_index = torch.tensor([k[1][0],k[1][1]], dtype = torch.long)) for k in glycan_graphs]
        return data

def dataset_to_dataloader(glycan_list, labels, libr = None, batch_size = 32,
                          shuffle = True, drop_last = False):
  """wrapper function to convert glycans and labels to a torch_geometric DataLoader\n
  | Arguments:
  | :-
  | glycan_list (list): list of IUPAC-condensed glycan sequences as strings
  | labels (list): list of labels
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset
  | batch_size (int): how many samples should be in each batch; default:32
  | shuffle (bool): if samples should be shuffled when making dataloader; default:True
  | drop_last (bool): whether last batch is dropped; default:False\n
  | Returns:
  | :-
  | Returns a dataloader object used for training deep learning models
  """
  if libr is None:
    libr = lib
  glycan_graphs = dataset_to_graphs(glycan_list, labels, libr = libr)
  glycan_loader = DataLoader(glycan_graphs, batch_size = batch_size,
                             shuffle = shuffle, drop_last = drop_last)
  return glycan_loader

def split_data_to_train(glycan_list_train, glycan_list_val,
                        labels_train, labels_val, libr = None,
                        batch_size = 32, drop_last = False):
  """wrapper function to convert split training/test data into dictionary of dataloaders\n
  | Arguments:
  | :-
  | glycan_list_train (list): list of IUPAC-condensed glycan sequences as strings
  | glycan_list_val (list): list of IUPAC-condensed glycan sequences as strings
  | labels_train (list): list of labels
  | labels_val (list): list of labels
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset
  | batch_size (int): how many samples should be in each batch; default:32
  | drop_last (bool): whether last batch is dropped; default:False\n
  | Returns:
  | :-
  | Returns a dictionary of dataloaders for training and testing deep learning models
  """
  if libr is None:
    libr = lib
  train_loader = dataset_to_dataloader(glycan_list_train, labels_train, libr = libr,
                                       batch_size = batch_size, shuffle = True,
                                       drop_last = drop_last)
  val_loader = dataset_to_dataloader(glycan_list_val, labels_val, libr = libr,
                                       batch_size = batch_size, shuffle = False,
                                       drop_last = drop_last)
  return {'train':train_loader, 'val':val_loader}
