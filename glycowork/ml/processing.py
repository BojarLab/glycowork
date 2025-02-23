from copy import deepcopy
from random import getrandbits, random
from typing import Dict, List, Optional, Union
from glycowork.motif.graph import glycan_to_nxGraph
from glycowork.motif.tokenization import map_to_basic
from glycowork.motif.processing import de_wildcard_glycoletter
from glycowork.glycan_data.loader import lib, HashableDict

try:
  import torch
  from torch_geometric.loader import DataLoader
  from torch_geometric.utils.convert import from_networkx
except ImportError:
  raise ImportError("<torch or torch_geometric missing; did you do 'pip install glycowork[ml]'?>")


def augment_glycan(glycan_data: torch.utils.data.Dataset, # glycan as a networkx graph
                  libr: Dict[str, int], # dictionary of glycoletter:index
                  generalization_prob: float = 0.2 # probability of wildcarding monosaccharides/linkages
                 ) -> torch.utils.data.Dataset: # data object with partially wildcarded glycan
  "Augment a single glycan by wildcarding some of its monosaccharides or linkages"
  augmented_data = deepcopy(glycan_data)
  # Augment node labels
  for i, label in enumerate(augmented_data.string_labels):
    if random() < generalization_prob:
      temp = de_wildcard_glycoletter(label) if '?' in label or 'x' in label else map_to_basic(label, obfuscate_ptm = getrandbits(1))
      augmented_data.string_labels[i] = temp
      augmented_data.labels[i] = libr.get(temp, len(libr))
  return augmented_data


class AugmentedGlycanDataset(torch.utils.data.Dataset):
  def __init__(self, dataset: torch.utils.data.Dataset, # base dataset
                 libr: Dict[str, int], # dictionary of glycoletter:index
                 augment_prob: float = 0.5, # probability of augmentation
                 generalization_prob: float = 0.2 # probability of wildcarding
                ) -> None:
    "Dataset wrapper that augments glycans through wildcarding"
    self.dataset = dataset
    self.augment_prob = augment_prob
    self.generalization_prob = generalization_prob
    self.libr = libr

  def __len__(self) -> int:
    return len(self.dataset)

  def __getitem__(self, idx: int) -> torch.utils.data.Dataset:
    glycan_data = self.dataset[idx]
    if random() < self.augment_prob:
      glycan_data = augment_glycan(glycan_data, self.libr, self.generalization_prob)
    return glycan_data


def dataset_to_graphs(glycan_list: List[str], # list of IUPAC-condensed glycan sequences
                     labels: List[Union[float, int]], # list of labels
                     libr: Optional[Dict[str, int]] = None, # dictionary of glycoletter:index
                     label_type: torch.dtype = torch.long # tensor type for label
                    ) -> List[torch.utils.data.Dataset]: # list of node/edge/label data tuples
  "wrapper function to convert a whole list of glycans into a graph dataset"
  if libr is None:
    libr = lib
  libr = HashableDict(libr)
  glycan_cache, data = {}, []
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


def dataset_to_dataloader(glycan_list: List[str], # list of IUPAC-condensed glycans
                         labels: List[Union[float, int]], # list of labels
                         libr: Optional[Dict[str, int]] = None, # dictionary of glycoletter:index
                         batch_size: int = 32, # samples per batch
                         shuffle: bool = True, # shuffle samples in dataloader
                         drop_last: bool = False, # drop last batch
                         extra_feature: Optional[List[float]] = None, # additional input features
                         label_type: torch.dtype = torch.long, # tensor type for label
                         augment_prob: float = 0., # probability of data augmentation
                         generalization_prob: float = 0.2 # probability of wildcarding
                        ) -> torch.utils.data.DataLoader: # dataloader for training
  "wrapper function to convert glycans and labels to a torch_geometric DataLoader"
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


def split_data_to_train(glycan_list_train: List[str], # training glycans
                       glycan_list_val: List[str], # validation glycans
                       labels_train: List[Union[float, int]], # training labels
                       labels_val: List[Union[float, int]], # validation labels
                       libr: Optional[Dict[str, int]] = None, # dictionary of glycoletter:index
                       batch_size: int = 32, # samples per batch
                       drop_last: bool = False, # drop last batch
                       extra_feature_train: Optional[List[float]] = None, # additional training features
                       extra_feature_val: Optional[List[float]] = None, # additional validation features
                       label_type: torch.dtype = torch.long, # tensor type for label
                       augment_prob: float = 0., # probability of data augmentation
                       generalization_prob: float = 0.2 # probability of wildcarding
                      ) -> Dict[str, torch.utils.data.DataLoader]: # dictionary of train/val dataloaders
  "wrapper function to convert split training/test data into dictionary of dataloaders"
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
