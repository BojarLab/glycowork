from typing import Union

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, GINConv, global_mean_pool

from glycowork.ml.gifflar.hetero import HeteroDataBatch
from glycowork.ml.gifflar.data import atom_map, bond_map, lib_map


def get_gin_layer(input_dim: int, output_dim: int) -> GINConv:
    """
    Get a GIN layer with the specified input and output dimensions

    Args:
        input_dim: The input dimension of the GIN layer
        output_dim: The output dimension of the GIN layer

    Returns:
        A GIN layer with the specified input and output dimensions
    """
    return GINConv(
        nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(output_dim),
        )
    )


class GIFFLAR(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            embed_dim: int,
            output_dim: int,
            num_layers: int,
    ):
        """
        Initialize the GIFFLAR model.
        
        Args:
            feat_dim: Dimension of initial atom/bond/monosacch features
            embed_dim: Dimension of embeddings of each GNN layer
            output_dim: Dimension of the output, e.g., num_classes for classification
            num_layers: Number of GNN layers
        """
        super(GIFFLAR, self).__init__()

        self.atom_embedding = nn.Embedding(len(atom_map) + 1, feat_dim)
        self.bond_embedding = nn.Embedding(len(bond_map) + 1, feat_dim)
        self.mono_embedding = nn.Embedding(len(lib_map) + 1, feat_dim)

        dims = [feat_dim] + [embed_dim] * num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(HeteroConv({
                key: get_gin_layer(dims[i], dims[i + 1]) for key in [
                    ("atoms", "coboundary", "atoms"),
                    ("atoms", "to", "bonds"),
                    ("bonds", "to", "monosacchs"),
                    ("bonds", "boundary", "bonds"),
                    ("monosacchs", "boundary", "monosacchs")
                ]
            }))
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, output_dim),
        )

    def forward(self, batch: HeteroDataBatch, embeddings: bool = False, *args, **kwargs) -> Union[torch.Tensor, dict]:
        """
        Compute the node embeddings.

        Args:
            batch: The batch of data to process
            embeddings: Whether to return embeddings or predictions

        Returns:
            node_embed: The node embeddings
        """
        batch.x_dict["atoms"] = self.atom_embedding.forward(batch.x_dict["atoms"])
        batch.x_dict["bonds"] = self.bond_embedding.forward(batch.x_dict["bonds"])
        batch.x_dict["monosacchs"] = self.mono_embedding.forward(batch.x_dict["monosacchs"])

        for conv in self.convs:
            batch.x_dict = conv(batch.x_dict, batch.edge_index_dict)
        
        graph_embed = self.pool(batch.x_dict, batch.batch_dict)
        pred = self.head(graph_embed).squeeze()

        if embeddings:
            return {
                "node_embed": batch.x_dict,
                "graph_embed": graph_embed,
                "pred": pred
            }
        return pred

    def pool(self, nodes: dict, batch_ids: dict) -> torch.Tensor:
        """
        Pool the node embeddings to get a graph-level embedding.

        Args:
            nodes: The node embeddings
            batch_ids: The batch IDs for each node

        Returns:
            graph_embed: The graph-level embedding
        """
        return global_mean_pool(
            torch.concat([nodes["atoms"], nodes["bonds"], nodes["monosacchs"]], dim=0),
            torch.concat([batch_ids["atoms"], batch_ids["bonds"], batch_ids["monosacchs"]], dim=0)
        )
