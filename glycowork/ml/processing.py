import networkx as nx
from copy import deepcopy
from random import getrandbits, random
from typing import Any
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
  from glyles.glycans.factory.factory import MonomerFactory
  from glyles.glycans.poly.merger import Merger
  from rdkit import Chem
  from rdkit.Chem import rdDepictor
except ImportError:
  raise ImportError("<rdkit missing; you need to do 'pip install glycowork[all]' to use the GIFFLAR model>")

atom_map = {6: 1, 7: 2, 8: 3, 15: 3, 16: 5}
bond_map = {Chem.BondDir.BEGINDASH: 1, Chem.BondDir.BEGINWEDGE: 2, Chem.BondDir.NONE: 3}


def augment_glycan(glycan_data: torch.utils.data.Dataset, # glycan as a networkx graph
                  libr: dict[str, int], # dictionary of glycoletter:index
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
                 libr: dict[str, int], # dictionary of glycoletter:index
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


def dataset_to_graphs(glycan_list: list[str], # list of IUPAC-condensed glycan sequences
                     labels: list[float | int], # list of labels
                     libr: dict[str, int] | None = None, # dictionary of glycoletter:index
                     label_type: torch.dtype = torch.long # tensor type for label
                    ) -> list[torch.utils.data.Dataset]: # list of node/edge/label data tuples
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


def dataset_to_dataloader(glycan_list: list[str], # list of IUPAC-condensed glycans
                         labels: list[float | int], # list of labels
                         libr: dict[str, int] | None = None, # dictionary of glycoletter:index
                         batch_size: int = 32, # samples per batch
                         shuffle: bool = True, # shuffle samples in dataloader
                         drop_last: bool = False, # drop last batch
                         extra_feature: list[float] | None = None, # additional input features
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


def split_data_to_train(glycan_list_train: list[str], # training glycans
                       glycan_list_val: list[str], # validation glycans
                       labels_train: list[float | int], # training labels
                       labels_val: list[float | int], # validation labels
                       libr: dict[str, int] | None = None, # dictionary of glycoletter:index
                       batch_size: int = 32, # samples per batch
                       drop_last: bool = False, # drop last batch
                       extra_feature_train: list[float] | None = None, # additional training features
                       extra_feature_val: list[float] | None = None, # additional validation features
                       label_type: torch.dtype = torch.long, # tensor type for label
                       augment_prob: float = 0., # probability of data augmentation
                       generalization_prob: float = 0.2 # probability of wildcarding
                      ) -> dict[str, torch.utils.data.DataLoader]: # dictionary of train/val dataloaders
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

  def __getitem__(self, item: str) -> Any:
    """Get attribute from the object"""
    return getattr(self, item)


def hetero_collate(data: list[list[HeteroData]] | list[HeteroData] | None,  # list of HeteroData objects
                   *args, **kwargs
                   ) -> HeteroDataBatch:  # HeteroDataBatch object of the collated input samples
  """Collate a list of HeteroData objects to a batch thereof"""
  if data is None:
    raise ValueError("No data provided for collation.")
  if isinstance(data[0], list):
    data = data[0]
  # Extract all valid node types and edge types
  node_types = ["atoms", "bonds", "monosacchs"]
  edge_types = [("atoms", "coboundary", "atoms"), ("atoms", "to", "bonds"), ("bonds", "to", "monosacchs"), ("bonds", "boundary", "bonds"), ("monosacchs", "boundary", "monosacchs")]
  # Setup empty fields for the most important attributes of the resulting batch
  x_dict, batch_dict, edge_index_dict, edge_attr_dict = {}, {}, {}, {}
  # Store the node counts to offset edge indices when collating
  node_counts = {node_type: [0] for node_type in node_types}
  batch_kwargs = {"y": [], "ID": []}
  for d in data:
    for key in batch_kwargs:  # Collect all length-queryable fields
      try:
        if not hasattr(d[key], "__len__") or len(d[key]) != 0:
          batch_kwargs[key].append(d[key])
      except Exception:
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
    if tmp_edge_index:
      edge_index_dict[edge_type] = torch.cat(tmp_edge_index, dim = 1)
    if tmp_edge_attr:
      edge_attr_dict[edge_type] = torch.cat(tmp_edge_attr, dim = 0)
  # Remove all incompletely given data and concat lists of tensors into single tensors
  num_nodes = {node_type: x_dict[node_type].shape[0] for node_type in node_types}
  for key, value in list(batch_kwargs.items()):
    if len(value) != len(data):
      del batch_kwargs[key]
    elif isinstance(value[0], torch.Tensor):
      if len(value[0].shape) == 1:
        dim = 0
      elif all(t.shape[1] == value[0].shape[1] for t in value):
        dim = 0
      elif all(t.shape[0] == value[0].shape[0] for t in value):
        dim = 1
      else:
        raise ValueError(f"Tensors for key {key} cannot be concatenated.")
      batch_kwargs[key] = torch.cat(value, dim = dim)
  # Finally create and return the HeteroDataBatch
  return HeteroDataBatch(x_dict = x_dict, edge_index_dict = edge_index_dict, edge_attr_dict = edge_attr_dict, num_nodes = num_nodes, batch_dict = batch_dict, **batch_kwargs)


class GIFFLARTransform(BaseTransform):
  """Transformation to bring data into a GIFFLAR format"""
  def forward(self,
               data: HeteroData  # input data
               ) -> HeteroData:  # transformed data
    """Transform the data into a GIFFLAR format. This means to compute the simplex network and create a heterogenous graph from it"""
    # Set up the atom information
    data["atoms"].x = torch.tensor([atom_map.get(atom.GetAtomicNum(), 1) for atom in data["mol"].GetAtoms()])
    data["atoms"].num_nodes = len(data["atoms"].x)
    # Prepare all data that can be extracted from one iteration over all bonds
    bonds_x, atoms_coboundary, atoms_to_bonds, bonds_to_monosacchs = [], [], [], []
    # Fill all bond-related information
    for bond in data["mol"].GetBonds():
      bonds_x.append(bond_map.get(bond.GetBondDir(), 1))
      b_idx, e_idx, idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetIdx()
      atoms_coboundary.extend([(b_idx, e_idx), (e_idx, b_idx)])
      atoms_to_bonds.extend([(b_idx, idx), (e_idx, idx)])
      bonds_to_monosacchs.append((idx, bond.GetIntProp("mono_id")))
    # Transform the data into tensors
    data["bonds"].x = torch.tensor(bonds_x)
    data["bonds"].num_nodes = len(bonds_x)
    data["atoms", "coboundary", "atoms"].edge_index = torch.tensor(atoms_coboundary, dtype = torch.long).t()
    data["atoms", "to", "bonds"].edge_index = torch.tensor(atoms_to_bonds, dtype = torch.long).t()
    data["bonds", "to", "monosacchs"].edge_index = torch.tensor(bonds_to_monosacchs, dtype = torch.long).t()
    # Compute both types of linkages between bonds
    data["bonds", "boundary", "bonds"].edge_index = torch.tensor([(bond1.GetIdx(), bond2.GetIdx()) for atom in data["mol"].GetAtoms() for bond1 in atom.GetBonds() for bond2 in atom.GetBonds() if bond1.GetIdx() != bond2.GetIdx()], dtype = torch.long).t()
    data["bonds", "coboundary", "bonds"].edge_index = torch.tensor([(bond1, bond2) for ring in data["mol"].GetRingInfo().BondRings() for bond1 in ring for bond2 in ring if bond1 != bond2], dtype = torch.long).t()
    # Set up the monosaccharide information; This does not make sense. The monomer-ids are categorical features
    data["monosacchs"].x = torch.tensor([lib.get(data["tree"].nodes[node]["name"], 1) for node in data["tree"].nodes])
    data["monosacchs"].num_nodes = len(data["monosacchs"].x)
    monosacchs_boundary = [(a, b) for a, b in data["tree"].edges] + [(b, a) for a, b in data["tree"].edges]
    data["monosacchs", "boundary", "monosacchs"].edge_index = torch.tensor(monosacchs_boundary, dtype = torch.long).t()
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
           sanitize: bool = True  # bool flag indicating to sanitize the resulting molecule (should be True for "production mode" and False when debugging this function)
           ) -> Chem.Mol:  # converted, sanitized molecules in RDKit represented by the input graph
  """Convert a molecules from a networkx.Graph to RDKit"""
  # Create the molecule
  mol = Chem.RWMol()
  # Create all atoms based on their representing nodes
  node_to_idx = {}
  for node, attrs in G.nodes(data = True):
    a = Chem.Atom(attrs['atomic_num'])
    a.SetChiralTag(attrs['chiral_tag'])
    a.SetFormalCharge(attrs['formal_charge'])
    a.SetIsAromatic(attrs['is_aromatic'])
    a.SetIntProp("mono_id", attrs['mono_id'])
    node_to_idx[node] = mol.AddAtom(a)
  # Connect the atoms based on the edges from the graph
  for first, second, attrs in G.edges(data = True):
    idx = mol.AddBond(node_to_idx[first], node_to_idx[second], attrs['bond_type']) - 1
    mol.GetBondWithIdx(idx).SetIntProp("mono_id", attrs['mono_id'])
  if sanitize:
    Chem.SanitizeMol(mol)
  return mol


def clean_tree(tree: nx.Graph  # tree to clean
               ) -> nx.Graph | None:  # cleaned tree
  """Clean the tree from unnecessary node features and store only the IUPAC name"""
  for node in tree.nodes:
    attrs = deepcopy(tree.nodes[node])
    if "type" in attrs and isinstance(attrs["type"], glyles.glycans.mono.monomer.Monomer): # type: ignore
      tree.nodes[node].clear()
      tree.nodes[node].update({"iupac": "".join([x[0] for x in attrs["type"].recipe]), "name": attrs["type"].name, "recipe": attrs["type"].recipe})
    else:
      return None
  return tree


def iupac2mol(iupac: str  # IUPAC-condensed string of the glycan to convert
              ) -> HeteroData | None:  # HeteroData object containing the IUPAC string, the SMILES representation, the RDKit molecule, and the monosaccharide tree
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
  return HeteroData(IUPAC = iupac, smiles = smiles, mol = mol, tree = tree) if tree else None
