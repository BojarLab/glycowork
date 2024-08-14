from copy import deepcopy
from random import getrandbits, random
from glycowork.motif.graph import glycan_to_nxGraph
from glycowork.motif.tokenization import map_to_basic
from glycowork.motif.processing import de_wildcard_glycoletter
from glycowork.glycan_data.loader import lib

try:
  import torch
  from torch_geometric.loader import DataLoader
  from torch_geometric.utils.convert import from_networkx
except ImportError:
  raise ImportError("<torch or torch_geometric missing; did you do 'pip install glycowork[ml]'?>")


def augment_glycan(glycan_data, libr, generalization_prob = 0.2):
  """Augment a single glycan by wildcarding some of its monosaccharides or linkages.\n
  | Arguments:
  | :-
  | glycan_data (torch Data object): contains the glycan as a networkx graph
  | libr (dict): dictionary of form glycoletter:index
  | generalization_prob (float): the probability (0-1) of wildcarding for every single monosaccharide/linkage; default: 0.2\n
  | Returns:
  | :-
  | Returns the data object with partially wildcarded glycan graph
  """
  augmented_data = deepcopy(glycan_data)
  # Augment node labels
  for i, label in enumerate(augmented_data.string_labels):
    if random() < generalization_prob:
      temp = de_wildcard_glycoletter(label) if '?' in label or 'x' in label else map_to_basic(label, obfuscate_ptm = getrandbits(1))
      augmented_data.string_labels[i] = temp
      augmented_data.labels[i] = libr.get(temp, len(libr))
  return augmented_data


class AugmentedGlycanDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, libr, augment_prob = 0.5, generalization_prob = 0.2):
    self.dataset = dataset
    self.augment_prob = augment_prob
    self.generalization_prob = generalization_prob
    self.libr = libr

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    glycan_data = self.dataset[idx]
    if random() < self.augment_prob:
      glycan_data = augment_glycan(glycan_data, self.libr, self.generalization_prob)
    return glycan_data


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
  glycan_cache = {}
  data = []
  for glycan, label in zip(glycan_list, labels):
    if glycan not in glycan_cache:
        nx_graph = glycan_to_nxGraph(glycan, libr = libr)
        pyg_data = from_networkx(nx_graph)
        glycan_cache[glycan] = pyg_data
    else:
        # Reuse cached data for duplicate glycan
        pyg_data = glycan_cache[glycan].clone()
    # Add label to the data object
    pyg_data.y = torch.tensor(label, dtype = label_type)
    data.append(pyg_data)
  return data


def dataset_to_dataloader(glycan_list, labels, libr = None, batch_size = 32,
                          shuffle = True, drop_last = False, extra_feature = None, label_type = torch.long,
                          augment_prob = 0., generalization_prob = 0.2):
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
  | label_type (torch object): which tensor type for label, default is torch.long for binary labels, change to torch.float for continuous
  | augment_prob (float): probability (0-1) of applying data augmentation to each datapoint; default:0.
  | generalization_prob (float): the probability (0-1) of wildcarding for every single monosaccharide/linkage; default: 0.2\n
  | Returns:
  | :-
  | Returns a dataloader object used for training deep learning models
  """
  if libr is None:
    libr = lib
  # Converting glycans and labels to PyTorch Geometric Data objects
  glycan_graphs = dataset_to_graphs(glycan_list, labels,
                                    libr = libr, label_type = label_type)
  # Adding (optional) extra feature to the Data objects
  if extra_feature is not None:
    for graph, feature in zip(glycan_graphs, extra_feature):
      graph.train_idx = torch.tensor(feature, dtype = torch.float)
  augmented_dataset = AugmentedGlycanDataset(glycan_graphs, libr, augment_prob = augment_prob, generalization_prob = generalization_prob)
  # Generating the dataloader from the data objects
  return DataLoader(augmented_dataset, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last)


def split_data_to_train(glycan_list_train, glycan_list_val,
                        labels_train, labels_val, libr = None,
                        batch_size = 32, drop_last = False,
                        extra_feature_train = None, extra_feature_val = None,
                        label_type = torch.long, augment_prob = 0., generalization_prob = 0.2):
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
  | label_type (torch object): which tensor type for label, default is torch.long for binary labels, change to torch.float for continuous
  | augment_prob (float): probability (0-1) of applying data augmentation to each datapoint; default:0.
  | generalization_prob (float): the probability (0-1) of wildcarding for every single monosaccharide/linkage; default: 0.2\n
  | Returns:
  | :-
  | Returns a dictionary of dataloaders for training and testing deep learning models
  """
  if libr is None:
    libr = lib
  # Generating train and validation set loaders
  train_loader = dataset_to_dataloader(glycan_list_train, labels_train, libr = libr,
                                       batch_size = batch_size, shuffle = True,
                                       drop_last = drop_last, extra_feature = extra_feature_train,
                                       label_type = label_type, augment_prob = augment_prob, generalization_prob = generalization_prob)
  val_loader = dataset_to_dataloader(glycan_list_val, labels_val, libr = libr,
                                     batch_size = batch_size, shuffle = False,
                                     drop_last = drop_last, extra_feature = extra_feature_val,
                                     label_type = label_type, augment_prob = 0., generalization_prob = 0.)
  return {'train': train_loader, 'val': val_loader}
