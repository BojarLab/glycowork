import copy
from collections import defaultdict

import torch
import glyles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdDepictor
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.transforms.base_transform import BaseTransform
from glyles.glycans.utils import smiles2mol
from glyles.glycans.factory.factory import MonomerFactory
from glyles.glycans.poly.merger import Merger
from glycowork.glycan_data.loader import lib

# unknown, values...
atom_map = {6: 1, 7: 2, 8: 3, 15: 3, 16: 5}
bond_map = {Chem.BondDir.BEGINDASH: 1, Chem.BondDir.BEGINWEDGE: 2, Chem.BondDir.NONE: 3}
lib_map = {n: i + 1 for i, n in enumerate(lib)}


class GIFFLARTransform(BaseTransform):
    """Transformation to bring data into a GIFFLAR format"""

    def __call__(self, data: HeteroData) -> HeteroData:
        """
        Transform the data into a GIFFLAR format. This means to compute the simplex network and create a heterogenous
        graph from it.

        Args:
            data: The input data.

        Returns:
            The transformed data.
        """
        # Set up the atom information
        data["atoms"].x = torch.tensor([
            atom_map.get(atom.GetAtomicNum(), 1) for atom in data["mol"].GetAtoms()
        ])
        data["atoms"].num_nodes = len(data["atoms"].x)

        # prepare all data that can be extracted from one iteration over all bonds
        data["bonds"].x = []
        data["atoms", "coboundary", "atoms"].edge_index = []
        data["atoms", "to", "bonds"].edge_index = []
        data["bonds", "to", "monosacchs"].edge_index = []

        # fill all bond-related information
        for bond in data["mol"].GetBonds():
            data["bonds"].x.append(bond_map.get(bond.GetBondDir(), 1))
            data["atoms", "coboundary", "atoms"].edge_index += [
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
            ]
            data["atoms", "to", "bonds"].edge_index += [
                (bond.GetBeginAtomIdx(), bond.GetIdx()),
                (bond.GetEndAtomIdx(), bond.GetIdx())
            ]
            data["bonds", "to", "monosacchs"].edge_index.append(
                (bond.GetIdx(), bond.GetIntProp("mono_id"))
            )

        # transform the data into tensors
        data["bonds"].x = torch.tensor(data["bonds"].x)
        data["bonds"].num_nodes = len(data["bonds"].x)
        data["atoms", "coboundary", "atoms"].edge_index = torch.tensor(data["atoms", "coboundary", "atoms"].edge_index,
                                                                       dtype=torch.long).T
        data["atoms", "to", "bonds"].edge_index = torch.tensor(data["atoms", "to", "bonds"].edge_index,
                                                               dtype=torch.long).T
        data["bonds", "to", "monosacchs"].edge_index = torch.tensor(data["bonds", "to", "monosacchs"].edge_index,
                                                                    dtype=torch.long).T

        # compute both types of linkages between bonds
        data["bonds", "boundary", "bonds"].edge_index = torch.tensor(
            [(bond1.GetIdx(), bond2.GetIdx()) for atom in data["mol"].GetAtoms() for bond1 in atom.GetBonds()
             for bond2 in atom.GetBonds() if bond1.GetIdx() != bond2.GetIdx()], dtype=torch.long).T
        data["bonds", "coboundary", "bonds"].edge_index = torch.tensor(
            [(bond1, bond2) for ring in data["mol"].GetRingInfo().BondRings() for bond1 in ring for bond2 in ring if
             bond1 != bond2], dtype=torch.long).T

        # Set up the monosaccharide information
        data["monosacchs"].x = torch.tensor([  # This does not make sense. The monomer-ids are categorical features
            lib_map.get(data["tree"].nodes[node]["name"], 1) for node in data["tree"].nodes
        ])
        data["monosacchs"].num_nodes = len(data["monosacchs"].x)
        data["monosacchs", "boundary", "monosacchs"].edge_index = []
        for a, b in data["tree"].edges:
            data["monosacchs", "boundary", "monosacchs"].edge_index += [(a, b), (b, a)]
        data["monosacchs", "boundary", "monosacchs"].edge_index = torch.tensor(
            data["monosacchs", "boundary", "monosacchs"].edge_index, dtype=torch.long).T

        return data


class HeteroDataset(Dataset):
    def __init__(self, data_list: list[HeteroData]):
        """
        Dataset for heterogenous GIFFLAR data.

        Args:
            data_list: A list of HeteroData objects to include in the dataset
        """
        super().__init__()
        self.pre_transform = GIFFLARTransform()
        self.data_list = list(filter(lambda x: x is not None, map(self.pre_transform, data_list)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def masked2nx(smiles: str, mask: list[int]) -> nx.Graph:
    """
    Convert a SMILES string with masked monosaccharide IDs into a networkx.Graph while keeping the information of which atom
    and which bond belongs to which monosaccharide.

    Args:
        smiles: The SMILES string of the molecule to convert
        mask: A list of integers indicating the monosaccharide ID for each atom in the SMILES string

    Returns:
        The converted molecule in networkx represented by the input SMILES string
    """
    mol = smiles2mol(smiles)
    tmp1, i, read_atom, read_atom_index = defaultdict(list), 0, "", -1
    for i in range(len(smiles)):
        # check for uppercase, i.e., atoms
        if smiles[i].isupper() or "c":
            # save if it's an element, ie starts with uppercase or is a aromatic c
            if read_atom != "" and (read_atom[0].isupper() or read_atom == "c"):
                tmp1[read_atom] += [mask[read_atom_index]]
            # read current atom
            read_atom = smiles[i]
            read_atom_index = i
        # extent current element
        elif smiles[i].isalpha():
            read_atom += smiles[i]
    if read_atom != "" and (read_atom[0].isupper() or read_atom == "c"):
        tmp1[read_atom] += [mask[read_atom_index]]

    atom_counter = defaultdict(int)
    G = nx.Graph()
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        G.add_node(
            atom.GetIdx(),
            atomic_num=elem,
            formal_charge=atom.GetFormalCharge(),
            chiral_tag=atom.GetChiralTag(),
            is_aromatic=atom.GetIsAromatic(),
            mono_id=tmp1[elem][atom_counter[elem]],
        )
        atom_counter[elem] += 1
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        G.add_edge(
            start,
            bond.GetEndAtomIdx(),
            bond_type=bond.GetBondType(),
            mono_id=G.nodes[start]["mono_id"],
        )
    return G


def nx2mol(G: nx.Graph, sanitize=True) -> Chem.Mol:
    """
    Convert a molecules from a networkx.Graph to RDKit.

    Args:
        G: The graph representing a molecule
        sanitize: A bool flag indicating to sanitize the resulting molecule (should be True for "production mode" and
            False when debugging this function)

    Returns:
        The converted, sanitized molecules in RDKit represented by the input graph
    """
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

    # print(Chem.MolToSmiles(mol))
    if sanitize:
        Chem.SanitizeMol(mol)
    return mol


def clean_tree(tree: nx.Graph) -> nx.Graph | None:
    """
    Clean the tree from unnecessary node features and store only the IUPAC name.

    Args:
        tree: The tree to clean

    Returns:
        The cleaned tree
    """
    for node in tree.nodes:
        attributes = copy.deepcopy(tree.nodes[node])
        if "type" in attributes and isinstance(attributes["type"], glyles.glycans.mono.monomer.Monomer): # type: ignore
            tree.nodes[node].clear()
            tree.nodes[node].update({"iupac": "".join([x[0] for x in attributes["type"].recipe]),
                                     "name": attributes["type"].name, "recipe": attributes["type"].recipe})
        else:
            return None
    return tree


def iupac2mol(iupac: str) -> HeteroData | None:
    """
    Convert a glycan stored given as IUPAC-condensed string into an RDKit molecule while keeping the information of
    which atom and which bond belongs to which monosaccharide.

    Args:
        iupac: The IUPAC-condensed string of the glycan to convert

    Returns:
        A HeteroData object containing the IUPAC string, the SMILES representation, the RDKit molecule, and the
        monosaccharide tree
    """
    if "{" in iupac or "?" in iupac:
        return None

    # convert the IUPAC string using GlyLES
    glycan = glyles.Glycan(iupac)

    # get it's underlying monosaccharide-tree
    tree = glycan.parse_tree

    # re-merge the monosaccharide tree using networkx graphs to keep the assignment of atoms and bonds to monosacchs.
    _, merged = Merger(MonomerFactory()).merge(tree, glycan.root_orientation, glycan.start, smiles_only=False)
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
