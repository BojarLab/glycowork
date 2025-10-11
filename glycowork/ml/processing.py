import networkx as nx
from copy import deepcopy
from collections import defaultdict
from random import getrandbits, random
from typing import Any, Dict, List, Optional, Union
from glycowork.motif.graph import glycan_to_nxGraph
from glycowork.motif.tokenization import map_to_basic
from glycowork.motif.processing import de_wildcard_glycoletter
from glycowork.glycan_data.loader import lib, HashableDict
try:
  import torch
  from torch.utils.data import Dataset
  from torch_geometric.loader import DataLoader
  from torch_geometric.utils.convert import from_networkx
  from torch_geometric.data import HeteroData
  from torch_geometric.transforms.base_transform import BaseTransform
except ImportError:
  raise ImportError("<torch or torch_geometric missing; did you do 'pip install glycowork[ml]'?>")
try:
  import glyles
  from glyles.glycans.utils import smiles2mol
  from glyles.glycans.factory.factory import MonomerFactory
  from glyles.glycans.poly.merger import Merger
  from rdkit import Chem
  from rdkit.Chem import rdDepictor
except ImportError:
  raise ImportError("<rdkit missing; you need to do 'pip install glycowork[all]' to use the GIFFLAR model>")

atom_map = {6: 1, 7: 2, 8: 3, 15: 3, 16: 5}
bond_map = {Chem.BondDir.BEGINDASH: 1, Chem.BondDir.BEGINWEDGE: 2, Chem.BondDir.NONE: 3}


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
  glycan_graphs = dataset_to_graphs(glycan_list, labels, libr = libr, label_type = label_type)
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


class HeteroDataBatch:
  """Plain, dict-like object to store batches of HeteroData points"""
  def __init__(self, **kwargs: dict):
    """Initialize the object by setting each given argument as attribute of the object."""
    self.x_dict = {}
    self.edge_index_dict = {}
    self.batch_dict = {}
    for k, v in kwargs.items():
      setattr(self, k, v)

  def to(self,
         device: str  # Name of device to convert to, e.g., CPU, cuda:0, ...
         ) -> "HeteroDataBatch":  # Converted version of itself
    """Convert each field to the provided device by iteratively and recursively converting each field"""
    for k, v in self.__dict__.items():
      if hasattr(v, "to"):
        setattr(self, k, v.to(device))
      elif isinstance(v, dict):
        for key, value in v.items():
          if hasattr(value, "to"):
            v[key] = value.to(device)
    return self

  def __getitem__(self,
                  item: str  # Name of the queries attribute
                  ) -> Any:  # queries attribute
    """Get attribute from the object"""
    return getattr(self, item)


def determine_concat_dim(tensors: list[torch.Tensor]  # list of tensors to check
                         ) -> Optional[int]:  # dimension along which the tensors can be concatenated, or None if they cannot be concatenated
  """Determine the dimension along which a list of tensors can be concatenated"""
  if len(tensors[0].shape) == 1:
    return 0
  concat_dim = None
  if all(tensor.shape[1] == tensors[0].shape[1] for tensor in tensors):
    concat_dim = 0  # Concatenate along rows (dim 0)
  elif all(tensor.shape[0] == tensors[0].shape[0] for tensor in tensors):
    concat_dim = 1  # Concatenate along columns (dim 1)
  return concat_dim


def hetero_collate(data: Optional[Union[list[list[HeteroData]], list[HeteroData]]],  # list of HeteroData objects
                   *args, **kwargs
                   ) -> HeteroDataBatch:  # HeteroDataBatch object of the collated input samples
  """Collate a list of HeteroData objects to a batch thereof"""
  if data is None:
    raise ValueError("No data provided for collation.")
  # If, for whatever reason the input is a list of lists, fix that
  if isinstance(data[0], list):
    data = data[0]
  # Extract all valid node types and edge types
  node_types = ["atoms", "bonds", "monosacchs"]
  edge_types = [("atoms", "coboundary", "atoms"), ("atoms", "to", "bonds"), ("bonds", "to", "monosacchs"), ("bonds", "boundary", "bonds"), ("monosacchs", "boundary", "monosacchs")]
  # Setup empty fields for the most important attributes of the resulting batch
  x_dict, batch_dict, edge_index_dict, edge_attr_dict = {}, {}, {}, {}
  # Store the node counts to offset edge indices when collating
  node_counts = {node_type: [0] for node_type in node_types}
  kwargs = {key: [] for key in {"y", "ID"}}
  for d in data:
    for key in kwargs:  # Collect all length-queryable fields
      try:
        if not hasattr(d[key], "__len__") or len(d[key]) != 0:
          kwargs[key].append(d[key])
      except Exception as e:
        pass
    # Compute the offsets for each node type for sample identification after batching
    for node_type in node_types:
      node_counts[node_type].append(node_counts[node_type][-1] + d[node_type].num_nodes)
  # Collect the node features for each node type and store their assignment to the individual samples
  for node_type in node_types:
    x_dict[node_type] = torch.concat([d[node_type].x for d in data], dim = 0)
    batch_dict[node_type] = torch.cat([torch.full((d[node_type].num_nodes,), i, dtype = torch.long) for i, d in enumerate(data)], dim = 0)
  # Collect edge information for each edge type
  for edge_type in edge_types:
    tmp_edge_index, tmp_edge_attr = [], []
    for i, d in enumerate(data):
      # Collect the edge indices and offset them according to the offsets of their respective nodes
      if list(d[edge_type].edge_index.shape) == [0]:
        continue
      tmp_edge_index.append(torch.stack([d[edge_type].edge_index[0] + node_counts[edge_type[0]][i], d[edge_type].edge_index[1] + node_counts[edge_type[2]][i]]))
      # Also collect edge attributes if existent (NOT TESTED!)
      if hasattr(d[edge_type], "edge_attr"):
        tmp_edge_attr.append(d[edge_type].edge_attr)
    # Collate the edge information
    if len(tmp_edge_index) != 0:
      edge_index_dict[edge_type] = torch.cat(tmp_edge_index, dim = 1)
    if len(tmp_edge_attr) != 0:
      edge_attr_dict[edge_type] = torch.cat(tmp_edge_attr, dim = 0)
  # Remove all incompletely given data and concat lists of tensors into single tensors
  num_nodes = {node_type: x_dict[node_type].shape[0] for node_type in node_types}
  for key, value in list(kwargs.items()):
    if len(value) != len(data):
      del kwargs[key]
    elif isinstance(value[0], torch.Tensor):
      dim = determine_concat_dim(value)
      if dim is None:
        raise ValueError(f"Tensors for key {key} cannot be concatenated.")
      kwargs[key] = torch.cat(value, dim = dim)
  # Finally create and return the HeteroDataBatch
  return HeteroDataBatch(x_dict = x_dict, edge_index_dict = edge_index_dict, edge_attr_dict = edge_attr_dict, num_nodes = num_nodes, batch_dict = batch_dict, **kwargs)


class GIFFLARTransform(BaseTransform):
  """Transformation to bring data into a GIFFLAR format"""
  def __call__(self,
               data: HeteroData  # input data
               ) -> HeteroData:  # transformed data
    """Transform the data into a GIFFLAR format. This means to compute the simplex network and create a heterogenous graph from it"""
    # Set up the atom information
    data["atoms"].x = torch.tensor([atom_map.get(atom.GetAtomicNum(), 1) for atom in data["mol"].GetAtoms()])
    data["atoms"].num_nodes = len(data["atoms"].x)
    # Prepare all data that can be extracted from one iteration over all bonds
    data["bonds"].x = []
    data["atoms", "coboundary", "atoms"].edge_index = []
    data["atoms", "to", "bonds"].edge_index = []
    data["bonds", "to", "monosacchs"].edge_index = []
    # Fill all bond-related information
    for bond in data["mol"].GetBonds():
      data["bonds"].x.append(bond_map.get(bond.GetBondDir(), 1))
      data["atoms", "coboundary", "atoms"].edge_index += [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()), (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())]
      data["atoms", "to", "bonds"].edge_index += [(bond.GetBeginAtomIdx(), bond.GetIdx()), (bond.GetEndAtomIdx(), bond.GetIdx())]
      data["bonds", "to", "monosacchs"].edge_index.append((bond.GetIdx(), bond.GetIntProp("mono_id")))
    # Transform the data into tensors
    data["bonds"].x = torch.tensor(data["bonds"].x)
    data["bonds"].num_nodes = len(data["bonds"].x)
    data["atoms", "coboundary", "atoms"].edge_index = torch.tensor(data["atoms", "coboundary", "atoms"].edge_index, dtype = torch.long).t()
    data["atoms", "to", "bonds"].edge_index = torch.tensor(data["atoms", "to", "bonds"].edge_index, dtype = torch.long).t()
    data["bonds", "to", "monosacchs"].edge_index = torch.tensor(data["bonds", "to", "monosacchs"].edge_index, dtype = torch.long).t()
    # Compute both types of linkages between bonds
    data["bonds", "boundary", "bonds"].edge_index = torch.tensor([(bond1.GetIdx(), bond2.GetIdx()) for atom in data["mol"].GetAtoms() for bond1 in atom.GetBonds() for bond2 in atom.GetBonds() if bond1.GetIdx() != bond2.GetIdx()], dtype = torch.long).t()
    data["bonds", "coboundary", "bonds"].edge_index = torch.tensor([(bond1, bond2) for ring in data["mol"].GetRingInfo().BondRings() for bond1 in ring for bond2 in ring if bond1 != bond2], dtype = torch.long).t()
    # Set up the monosaccharide information; This does not make sense. The monomer-ids are categorical features
    data["monosacchs"].x = torch.tensor([lib.get(data["tree"].nodes[node]["name"], 1) for node in data["tree"].nodes])
    data["monosacchs"].num_nodes = len(data["monosacchs"].x)
    data["monosacchs", "boundary", "monosacchs"].edge_index = []
    for a, b in data["tree"].edges:
      data["monosacchs", "boundary", "monosacchs"].edge_index += [(a, b), (b, a)]
    data["monosacchs", "boundary", "monosacchs"].edge_index = torch.tensor(data["monosacchs", "boundary", "monosacchs"].edge_index, dtype = torch.long).T
    return data


class HeteroDataset(Dataset):
  def __init__(self,
               data_list: list[HeteroData]  # list of HeteroData objects to include in the dataset
               ):
    """Dataset for heterogenous GIFFLAR data"""
    super().__init__()
    self.pre_transform = GIFFLARTransform()
    self.data_list = list(filter(lambda x: x is not None, map(self.pre_transform, data_list)))

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    return self.data_list[idx]


def nx2mol(G: nx.Graph,  # graph representing a molecule
           sanitize = True  # bool flag indicating to sanitize the resulting molecule (should be True for "production mode" and False when debugging this function)
           ) -> Chem.Mol:  # converted, sanitized molecules in RDKit represented by the input graph
  """Convert a molecules from a networkx.Graph to RDKit"""
  # Create the molecule
  mol = Chem.RWMol()
  # Extract the node attributes
  atomic_nums = nx.get_node_attributes(G, 'atomic_num')
  chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
  formal_charges = nx.get_node_attributes(G, 'formal_charge')
  node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
  mono_ids = nx.get_node_attributes(G, 'mono_id')
  # Create all atoms based on their representing nodes
  node_to_idx = {}
  for node in G.nodes():
    a = Chem.Atom(atomic_nums[node])
    a.SetChiralTag(chiral_tags[node])
    a.SetFormalCharge(formal_charges[node])
    a.SetIsAromatic(node_is_aromatics[node])
    a.SetIntProp("mono_id", mono_ids[node])
    idx = mol.AddAtom(a)
    node_to_idx[node] = idx
  # Extract the edge attributes
  bond_types = nx.get_edge_attributes(G, 'bond_type')
  mono_bond_ids = nx.get_edge_attributes(G, 'mono_id')
  # Connect the atoms based on the edges from the graph
  for edge in G.edges():
    first, second = edge
    ifirst = node_to_idx[first]
    isecond = node_to_idx[second]
    bond_type = bond_types[first, second]
    mono_idx = mono_bond_ids[first, second]
    idx = mol.AddBond(ifirst, isecond, bond_type) - 1
    mol.GetBondWithIdx(idx).SetIntProp("mono_id", mono_idx)
  if sanitize:
    Chem.SanitizeMol(mol)
  return mol


def clean_tree(tree: nx.Graph  # tree to clean
               ) -> Optional[nx.Graph]:  # cleaned tree
  """Clean the tree from unnecessary node features and store only the IUPAC name"""
  for node in tree.nodes:
    attributes = deepcopy(tree.nodes[node])
    if "type" in attributes and isinstance(attributes["type"], glyles.glycans.mono.monomer.Monomer): # type: ignore
      tree.nodes[node].clear()
      tree.nodes[node].update({"iupac": "".join([x[0] for x in attributes["type"].recipe]), "name": attributes["type"].name, "recipe": attributes["type"].recipe})
    else:
      return None
  return tree


def iupac2mol(iupac: str  # IUPAC-condensed string of the glycan to convert
              ) -> Optional[HeteroData]:  # HeteroData object containing the IUPAC string, the SMILES representation, the RDKit molecule, and the monosaccharide tree
  """Convert a glycan stored given as IUPAC-condensed string into an RDKit molecule while keeping the information of which atom and which bond belongs to which monosaccharide"""
  if "{" in iupac or "?" in iupac:
    return None
  # Convert the IUPAC string using GlyLES
  glycan = glyles.Glycan(iupac)
  # Get its underlying monosaccharide-tree
  tree = glycan.parse_tree
  # Re-merge the monosaccharide tree using networkx graphs to keep the assignment of atoms and bonds to monosacchs.
  _, merged = Merger(MonomerFactory()).merge(tree, glycan.root_orientation, glycan.start, smiles_only = False)
  mol = nx2mol(merged)
  if not mol.GetNumConformers():
    rdDepictor.Compute2DCoords(mol)
  Chem.WedgeMolBonds(mol, mol.GetConformer())
  smiles = Chem.MolToSmiles(mol)
  if len(smiles) < 10 or not isinstance(tree, nx.Graph):
    return None
  tree = clean_tree(tree)
  data = HeteroData()
  data["IUPAC"] = iupac
  data["smiles"] = smiles
  data["mol"] = mol
  data["tree"] = tree
  return data
