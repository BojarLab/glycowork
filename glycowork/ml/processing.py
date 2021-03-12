from glycowork.motif.graph import glycan_to_graph
from glycowork.glycan_data.loader import lib

try:
  import torch
  from torch_geometric.data import Data, DataLoader
except ImportError:
    raise ImportError('<torch or torch_geometric missing; cannot do deep learning>')

def dataset_to_graphs(glycan_list, labels, libr = lib, label_type = torch.long, separate = False,
                      context = False, error_catch = False, wo_labels = False):
  """wrapper function to convert a whole list of glycans into a graph dataset
  glycan_list -- list of IUPACcondensed glycan sequences (string)
  labels -- list of labels
  label_type -- which tensor type for label, default is torch.long for binary labels, change to torch.float for continuous
  separate -- True returns node list / edge list / label list as separate files; False returns list of data tuples; default is False
  libr -- sorted list of unique glycoletters observed in the glycans of our dataset
  context -- legacy-ish; used for generating graph context dataset for pre-training; keep at False
  error_catch -- troubleshooting option, True will print glycans that cannot be converted into graphs; default is False
  wo_labels -- change to True if you do not want to pass and receive labels; default is False

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

def dataset_to_dataloader(glycan_list, labels, libr = lib, batch_size = 32,
                          shuffle = True):
  """wrapper function to convert glycans and labels to a torch_geometric DataLoader
  glycan_list -- list of IUPACcondensed glycan sequences (string)
  labels -- list of labels
  libr -- sorted list of unique glycoletters observed in the glycans of our dataset
  batch_size -- how many samples should be in each batch; default:32
  shuffle -- if samples should be shuffled when making dataloader; default:True

  returns a dataloader object used for training deep learning models
  """
  glycan_graphs = dataset_to_graphs(glycan_list, labels, libr = libr)
  glycan_loader = DataLoader(glycan_graphs, batch_size = batch_size,
                             shuffle = shuffle)
  return glycan_loader

def split_data_to_train(glycan_list_train, glycan_list_val,
                        labels_train, labels_val, libr = lib,
                        batch_size = 32):
  """wrapper function to convert split training/test data into dictionary of dataloaders
  glycan_list_train -- list of IUPACcondensed glycan sequences (string)
  glycan_list_val -- list of IUPACcondensed glycan sequences (string)
  labels_train -- list of labels
  labels_val -- list of labels
  libr -- sorted list of unique glycoletters observed in the glycans of our dataset
  batch_size -- how many samples should be in each batch; default:32

  returns a dictionary of dataloaders for training and testing deep learning models
  """
  train_loader = dataset_to_dataloader(glycan_list_train, labels_train, libr = libr,
                                       batch_size = batch_size, shuffle = True)
  val_loader = dataset_to_dataloader(glycan_list_val, labels_val, libr = libr,
                                       batch_size = batch_size, shuffle = False)
  return {'train':train_loader, 'val':val_loader}
