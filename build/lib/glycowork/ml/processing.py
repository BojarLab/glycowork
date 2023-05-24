from glycowork.motif.graph import glycan_to_nxGraph
from glycowork.glycan_data.loader import lib

try:
  import torch
  from torch_geometric.loader import DataLoader
  from torch_geometric.utils.convert import from_networkx
except ImportError:
  raise ImportError("<torch_geometric missing; did you do 'pip install glycowork[ml]'?>")

def dataset_to_graphs(glycan_list, labels, libr = None, label_type = torch.long):
  """wrapper function to convert a whole list of glycans into a graph dataset\n
  | Arguments:
  | :-
  | glycan_list (list): list of IUPAC-condensed glycan sequences as strings
  | labels (list): list of labels
  | libr (dict): dictionary of form glycoletter:index
  | label_type (torch object): which tensor type for label, default is torch.long for binary labels, change to torch.float for continuous\n
  | Returns:
  | :-
  | Returns list of node list / edge list / label list data tuples
  """
  if libr is None:
    libr = lib
  #generating glycan graphs
  glycan_graphs = [glycan_to_nxGraph(k, libr = libr) for k in glycan_list]
  #converting graphs to Pytorch Geometric Data objects
  data = [from_networkx(k) for k in glycan_graphs]
  #adding graph labels
  for k in range(len(labels)):
    data[k].y = torch.tensor(labels[k])
  return data

def dataset_to_dataloader(glycan_list, labels, libr = None, batch_size = 32,
                          shuffle = True, drop_last = False,
                          extra_feature = None, label_type = torch.long):
  """wrapper function to convert glycans and labels to a torch_geometric DataLoader\n
  | Arguments:
  | :-
  | glycan_list (list): list of IUPAC-condensed glycan sequences as strings
  | labels (list): list of labels
  | libr (dict): dictionary of form glycoletter:index
  | batch_size (int): how many samples should be in each batch; default:32
  | shuffle (bool): if samples should be shuffled when making dataloader; default:True
  | drop_last (bool): whether last batch is dropped; default:False
  | extra_feature (list): can be used to feed another input to the dataloader; default:None
  | label_type (torch object): which tensor type for label, default is torch.long for binary labels, change to torch.float for continuous\n
  | Returns:
  | :-
  | Returns a dataloader object used for training deep learning models
  """
  if libr is None:
    libr = lib
  #converting glycans and labels to PyTorch Geometric Data objects
  glycan_graphs = dataset_to_graphs(glycan_list, labels,
                                    libr = libr, label_type = label_type)
  #adding (optional) extra feature to the Data objects
  if extra_feature is not None:
    for k in range(len(glycan_graphs)):
      glycan_graphs[k].train_idx = torch.tensor(extra_feature[k],
                                                dtype = torch.float)
  #generating the dataloader from the data objects
  glycan_loader = DataLoader(glycan_graphs, batch_size = batch_size,
                             shuffle = shuffle, drop_last = drop_last)
  return glycan_loader

def split_data_to_train(glycan_list_train, glycan_list_val,
                        labels_train, labels_val, libr = None,
                        batch_size = 32, drop_last = False,
                        extra_feature_train = None, extra_feature_val = None,
                        label_type = torch.long):
  """wrapper function to convert split training/test data into dictionary of dataloaders\n
  | Arguments:
  | :-
  | glycan_list_train (list): list of IUPAC-condensed glycan sequences as strings
  | glycan_list_val (list): list of IUPAC-condensed glycan sequences as strings
  | labels_train (list): list of labels
  | labels_val (list): list of labels
  | libr (dict): dictionary of form glycoletter:index
  | batch_size (int): how many samples should be in each batch; default:32
  | drop_last (bool): whether last batch is dropped; default:False
  | extra_feature_train (list): can be used to feed another input to the dataloader; default:None
  | extra_feature_val (list): can be used to feed another input to the dataloader; default:None
  | label_type (torch object): which tensor type for label, default is torch.long for binary labels, change to torch.float for continuous\n
  | Returns:
  | :-
  | Returns a dictionary of dataloaders for training and testing deep learning models
  """
  if libr is None:
    libr = lib
  #generating train and validation set loaders
  train_loader = dataset_to_dataloader(glycan_list_train, labels_train, libr = libr,
                                       batch_size = batch_size, shuffle = True,
                                       drop_last = drop_last, extra_feature = extra_feature_train,
                                       label_type = label_type)
  val_loader = dataset_to_dataloader(glycan_list_val, labels_val, libr = libr,
                                       batch_size = batch_size, shuffle = False,
                                       drop_last = drop_last, extra_feature = extra_feature_val,
                                     label_type = label_type)
  return {'train':train_loader, 'val':val_loader}
