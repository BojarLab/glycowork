from typing import Any

import torch
from torch_geometric.data import HeteroData


class HeteroDataBatch:
    """Plain, dict-like object to store batches of HeteroData points"""

    def __init__(self, **kwargs: dict):
        """
        Initialize the object by setting each given argument as attribute of the object.
        """
        self.x_dict = {}
        self.edge_index_dict = {}
        self.batch_dict = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device: str) -> "HeteroDataBatch":
        """
        Convert each field to the provided device by iteratively and recursively converting each field.

        Args:
            device: Name of device to convert to, e.g., CPU, cuda:0, ...

        Returns:
            Converted version of itself
        """
        for k, v in self.__dict__.items():
            if hasattr(v, "to"):
                setattr(self, k, v.to(device))
            elif isinstance(v, dict):
                for key, value in v.items():
                    if hasattr(value, "to"):
                        v[key] = value.to(device)
        return self

    def __getitem__(self, item: str) -> Any:
        """
        Get attribute from the object.

        Args:
            item: Name of the queries attribute

        Returns:
            The queries attribute
        """
        return getattr(self, item)


def determine_concat_dim(tensors: list[torch.Tensor]) -> int | None:
    """
    Determine the dimension along which a list of tensors can be concatenated.

    Args:
        tensors: The list of tensors to check
    
    Returns:
        The dimension along which the tensors can be concatenated, or None if they cannot be concatenated
    """
    if len(tensors[0].shape) == 1:
        return 0
    concat_dim = None
    if all(tensor.shape[1] == tensors[0].shape[1] for tensor in tensors):
        concat_dim = 0  # Concatenate along rows (dim 0)
    elif all(tensor.shape[0] == tensors[0].shape[0] for tensor in tensors):
        concat_dim = 1  # Concatenate along columns (dim 1)
    return concat_dim


def hetero_collate(data: list[list[HeteroData]] | list[HeteroData] | None, *args, **kwargs) -> HeteroDataBatch:
    """
    Collate a list of HeteroData objects to a batch thereof.

    Args:
        data: The list of HeteroData objects

    Returns:
        A HeteroDataBatch object of the collated input samples
    """
    if data is None:
        raise ValueError("No data provided for collation.")
    
    # If, for whatever reason the input is a list of lists, fix that
    if isinstance(data[0], list):
        data = data[0]

    # Extract all valid node types and edge types
    node_types = ["atoms", "bonds", "monosacchs"]
    edge_types = [
        ("atoms", "coboundary", "atoms"),
        ("atoms", "to", "bonds"),
        ("bonds", "to", "monosacchs"),
        ("bonds", "boundary", "bonds"),
        ("monosacchs", "boundary", "monosacchs")
    ]

    # Setup empty fields for the most important attributes of the resulting batch
    x_dict = {}
    batch_dict = {}
    edge_index_dict = {}
    edge_attr_dict = {}

    # Store the node counts to offset edge indices when collating
    node_counts = {node_type: [0] for node_type in node_types}
    kwargs = {key: [] for key in {"y", "ID"}}
    for d in data:
        for key in kwargs:
            # Collect all length-queryable fields
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
        x_dict[node_type] = torch.concat([d[node_type].x for d in data], dim=0)
        batch_dict[node_type] = torch.cat([
            torch.full((d[node_type].num_nodes,), i, dtype=torch.long) for i, d in enumerate(data)
        ], dim=0)

    # Collect edge information for each edge type
    for edge_type in edge_types:
        tmp_edge_index = []
        tmp_edge_attr = []
        for i, d in enumerate(data):
            # Collect the edge indices and offset them according to the offsets of their respective nodes
            if list(d[edge_type].edge_index.shape) == [0]:
                continue
            tmp_edge_index.append(torch.stack([
                d[edge_type].edge_index[0] + node_counts[edge_type[0]][i],
                d[edge_type].edge_index[1] + node_counts[edge_type[2]][i]
            ]))

            # Also collect edge attributes if existent (NOT TESTED!)
            if hasattr(d[edge_type], "edge_attr"):
                tmp_edge_attr.append(d[edge_type].edge_attr)

        # Collate the edge information
        if len(tmp_edge_index) != 0:
            edge_index_dict[edge_type] = torch.cat(tmp_edge_index, dim=1)
        if len(tmp_edge_attr) != 0:
            edge_attr_dict[edge_type] = torch.cat(tmp_edge_attr, dim=0)

    # Remove all incompletely given data and concat lists of tensors into single tensors
    num_nodes = {node_type: x_dict[node_type].shape[0] for node_type in node_types}
    for key, value in list(kwargs.items()):
        if len(value) != len(data):
            del kwargs[key]
        elif isinstance(value[0], torch.Tensor):
            dim = determine_concat_dim(value)
            if dim is None:
                raise ValueError(f"Tensors for key {key} cannot be concatenated.")
            kwargs[key] = torch.cat(value, dim=dim)

    # Finally create and return the HeteroDataBatch
    return HeteroDataBatch(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict,
                           num_nodes=num_nodes, batch_dict=batch_dict, **kwargs)
