import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Callable
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage
from glycowork.motif.graph import subgraph_isomorphism

this_dir, this_filename = os.path.split(__file__)
data_path = os.path.join(this_dir, 'milk_networks_exhaustive.pkl')
net_dic = pickle.load(open(data_path, 'rb'))


def calculate_distance_matrix(to_compare: Union[Dict[str, List], List], # Objects to compare - dict values must be lists
                            dist_func: Callable[[List, List], float], # Distance function such as jaccard or cosine
                            label_list: Optional[List[str]] = None # Column names for distance matrix
                           ) -> pd.DataFrame: # Square distance matrix
  "Calculate pairwise distances between objects using provided metric"
  n = len(to_compare)
  dm = np.zeros((n, n))
  # Check whether comparison objects are in a dict or list
  if isinstance(to_compare, dict):
    items = list(to_compare.values())
    if label_list is None:
      label_list = list(to_compare.keys())
  else:
    items = to_compare
    if label_list is None:
      label_list = list(range(n))
  # Calculate pairwise distances
  for i in range(n):
    for j in range(i+1, n):
      distance = dist_func(items[i], items[j])
      dm[i, j] = distance
      dm[j, i] = distance
  dm = pd.DataFrame(dm, columns = label_list, index = label_list)
  return dm


def distance_from_embeddings(df: pd.DataFrame, # DataFrame with glycans (rows) and taxonomic info (columns)
                           embeddings: pd.DataFrame, # DataFrame with glycans (rows) and embeddings (columns) (e.g., from glycans_to_emb)
                           cut_off: int = 10, # Minimum glycans per rank to be included; default:10
                           rank: str = 'Species', # Taxonomic rank for grouping; default:Species
                           averaging: str = 'median' # How to average embeddings: median/mean
                          ) -> pd.DataFrame: # Rank x rank distance matrix
  "Calculate cosine distance matrix from learned embeddings"
  if averaging not in ['mean', 'median']:
    print("Only 'median' and 'mean' are permitted averaging choices.")
    return
  # Subset df to only contain ranks with a minimum number of data points
  value_counts = df[rank].value_counts()
  valid_ranks = value_counts.index[value_counts >= cut_off]
  df_filtered = df[df[rank].isin(valid_ranks)]
  # Retrieve and average embeddings for all glycans
  embeddings_filtered = embeddings.loc[df_filtered.index]
  grouped = df_filtered.groupby(rank)
  avg_embeddings = grouped.apply(lambda g: embeddings_filtered.loc[g.index].agg(averaging), include_groups = False).reset_index()
  if isinstance(avg_embeddings.iloc[0, 0], str):
    avg_embeddings = avg_embeddings.set_index(avg_embeddings.columns[0])
  avg_values = np.vstack(avg_embeddings.values)
  # Get the distance matrix
  return calculate_distance_matrix(avg_values, cosine, label_list = valid_ranks)


def jaccard(list1: Union[List, nx.Graph], # First list/network to compare
            list2: Union[List, nx.Graph] # Second list/network to compare
           ) -> float: # Jaccard distance
  "Calculate Jaccard distance between two lists/networks"
  intersection = len(set(list1).intersection(list2))
  union = (len(list1) + len(list2)) - intersection
  return 1 - float(intersection) / union


def distance_from_metric(df: pd.DataFrame, # DataFrame with glycans (rows) and taxonomic info (columns)
                        networks: List[nx.Graph], # List of networkx networks
                        metric: str = "Jaccard", # Distance metric to use
                        cut_off: int = 10, # Minimum glycans per rank to be included; default:10
                        rank: str = "Species" # Taxonomic rank for grouping; default:Species
                       ) -> pd.DataFrame: # Rank x rank distance matrix
  "Calculate distance matrix between networks using provided metric"
  # Choose distance metric
  metric_funcs = {"Jaccard": jaccard}
  dist_func = metric_funcs.get(metric)
  if dist_func is None:
    print("Not a defined metric. At the moment, only 'Jaccard' is available as a metric.")
    return
  # Get all objects to calculate distance between
  value_counts = df[rank].value_counts()
  valid_ranks = value_counts.index[value_counts >= cut_off]
  valid_networks = [net for spec, net in zip(value_counts.index, networks) if spec in valid_ranks]
  # Get distance matrix
  return calculate_distance_matrix(valid_networks, dist_func, label_list = valid_ranks.tolist())


def dendrogram_from_distance(dm: pd.DataFrame, # Rank x rank distance matrix (e.g., from distance_from_embeddings)
                           ylabel: str = 'Mammalia', # Y-axis label
                           filepath: str = '' # Path to save plot including filename
                          ) -> None: # Displays or saves dendrogram plot
  "Plot dendrogram from distance matrix"
  # Hierarchical clustering on the distance matrix
  Z = linkage(dm)
  plt.figure(figsize = (10, 10))
  plt.title('Hierarchical Clustering Dendrogram')
  plt.xlabel('Distance')
  plt.ylabel(ylabel)
  dendrogram(
      Z,
      truncate_mode = 'lastp',  # Show only the last p merged clusters
      orientation = 'left',
      p = 300,  # Show only the last p merged clusters
      show_leaf_counts = False,  # Otherwise numbers in brackets are counts
      leaf_rotation = 0.,
      labels = dm.columns.tolist(),
      leaf_font_size = 11.,
      show_contracted = True,  # To get a distribution impression in truncated branches
      )
  if len(filepath) > 1:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300,
                bbox_inches = 'tight')


def check_conservation(glycan: str, # Glycan or motif in IUPAC-condensed format
                      df: pd.DataFrame, # DataFrame with glycans (rows) and taxonomic levels (columns)
                      network_dic: Optional[Dict[str, nx.Graph]] = None, # Species:biosynthetic network mapping
                      rank: str = 'Order', # Taxonomic level to assess
                      threshold: int = 5, # Minimum glycans per species to be included
                      motif: bool = False # Whether glycan is a motif vs sequence
                     ) -> Dict[str, float]: # Taxonomic group-to-conservation mapping
  "Estimate evolutionary conservation of glycans via biosynthetic networks"
  if network_dic is None:
    network_dic = net_dic
  # Subset species with at least the threshold-number of glycans
  species_counts = df['Species'].value_counts()
  valid_species = species_counts.index[species_counts >= threshold]
  df_filtered = df[df['Species'].isin(valid_species)]
  # Subset members of rank with at least two species
  rank_counts = df_filtered.groupby(rank)['Species'].nunique()
  valid_ranks = rank_counts.index[rank_counts > 1]
  # Subset networks
  filtered_network_dic = {k: v for k, v in network_dic.items() if k in valid_species}
  # For each member of rank, get all species networks and search for motif
  conserved = {}
  for r in valid_ranks:
    rank_df = df_filtered[df_filtered[rank] == r]
    rank_species = rank_df['Species'].unique()
    rank_networks = [filtered_network_dic[spec] for spec in rank_species]
    rank_nodes = [list(net.nodes()) for net in rank_networks]
    if motif:
      if glycan[-1] == ')':
        conserved[r] = sum(glycan in "".join(nodes) for nodes in rank_nodes) / len(rank_nodes)
      else:
        conserved[r] =  sum(any(subgraph_isomorphism(node, glycan) for node in nodes) for nodes in rank_nodes) / len(rank_nodes)
    else:
      conserved[r] = sum(glycan in nodes for nodes in rank_nodes) / len(rank_nodes)
  return conserved


def get_communities(network_list: List[nx.Graph], # List of undirected biosynthetic networks
                   label_list: Optional[List[str]] = None # Labels for community names, running_number + _ + label_list[k]  for network_list[k]; default:range(len(graph_list))
                  ) -> Dict[str, List[str]]: # Community-to-glycan list mapping
  "Find communities for each graph in list of graphs"
  if label_list is None:
    label_list = list(range(len(network_list)))
  final_comm_dict = {}
  # Label the communities by species name and running number to distinguish them afterwards
  for i, network in enumerate(network_list):
    communities = nx.algorithms.community.louvain.louvain_communities(network)
    for comm_index, community in enumerate(communities):
      comm_name = f"{comm_index}_{label_list[i]}"
      final_comm_dict[comm_name] = list(community)
  return final_comm_dict
