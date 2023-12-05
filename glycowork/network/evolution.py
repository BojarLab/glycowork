import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from community import community_louvain
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage
from glycowork.glycan_data.loader import lib
from glycowork.motif.graph import subgraph_isomorphism

this_dir, this_filename = os.path.split(__file__)
data_path = os.path.join(this_dir, 'milk_networks_exhaustive.pkl')
net_dic = pickle.load(open(data_path, 'rb'))


def calculate_distance_matrix(to_compare, dist_func, label_list = None):
  """calculates pairwise distances based on objects and a metric\n
  | Arguments:
  | :-
  | to_compare (dict or list): objects to calculate pairwise distances for, if dict then values have to be lists
  | dist_func (function): function such as 'jaccard' or 'cosine' that calculates distance given two lists/elements
  | label_list (list): column names for the resulting distance matrix; default:range(len(to_compare))\n
  | Returns:
  | :-
  | Returns a len(to_compare) x len(to_compare) distance matrix
  """
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


def distance_from_embeddings(df, embeddings, cut_off = 10, rank = 'Species',
                             averaging = 'median'):
  """calculates a cosine distance matrix from learned embeddings\n
  | Arguments:
  | :-
  | df (dataframe): dataframe with glycans as rows and taxonomic information as columns
  | embeddings (dataframe): dataframe with glycans as rows and learned embeddings as columns (e.g., from glycans_to_emb)
  | cut_off (int): how many glycans a rank (e.g., species) needs to have at least to be included; default:10
  | rank (string): which taxonomic rank to use for grouping organisms; default:'Species'
  | averaging (string): how to average embeddings, by 'median' or 'mean'; default:'median'\n
  | Returns:
  | :-
  | Returns a rank x rank distance matrix
  """
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
  avg_embeddings = grouped.apply(lambda g: embeddings_filtered.loc[g.index].agg(averaging)).reset_index()
  avg_values = np.vstack(avg_embeddings[0].values)
  # Get the distance matrix
  return calculate_distance_matrix(avg_values, cosine, label_list = valid_ranks)


def jaccard(list1, list2):
  """calculates Jaccard distance from two networks\n
  | Arguments:
  | :-
  | list1 (list or networkx graph): list containing objects to compare
  | list2 (list or networkx graph): list containing objects to compare\n
  | Returns:
  | :-
  | Returns Jaccard distance between list1 and list2
  """
  intersection = len(set(list1).intersection(list2))
  union = (len(list1) + len(list2)) - intersection
  return 1 - float(intersection) / union


def distance_from_metric(df, networks, metric = "Jaccard", cut_off = 10, rank = "Species"):
  """calculates a distance matrix of generated networks based on provided metric\n
  | Arguments:
  | :-
  | df (dataframe): dataframe with glycans as rows and taxonomic information as columns
  | networks (list): list of networks in networkx format
  | metric (string): which metric to use, available: 'Jaccard'; default:'Jaccard'
  | cut_off (int): how many glycans a rank (e.g., species) needs to have at least to be included; default:10
  | rank (string): which taxonomic rank to use for grouping organisms; default:'Species'\n
  | Returns:
  | :-
  | Returns a rank x rank distance matrix
  """
  # Choose distance metric
  metric_funcs = {"Jaccard": jaccard}
  dist_func = metric_funcs.get(metric)
  if dist_func is None:
    print("Not a defined metric. At the moment, only 'Jaccard' is available as a metric.")
    return
  # Get all objects to calculate distance between
  value_counts = df[rank].value_counts()
  valid_ranks = value_counts.index[value_counts >= cut_off]
  valid_networks = [net for spec, net in zip(df[rank], networks) if spec in valid_ranks]
  # Get distance matrix
  return calculate_distance_matrix(valid_networks, dist_func, label_list = valid_ranks.tolist())


def dendrogram_from_distance(dm, ylabel = 'Mammalia', filepath = ''):
  """plots a dendrogram from distance matrix\n
  | Arguments:
  | :-
  | dm (dataframe): a rank x rank distance matrix (e.g., from distance_from_embeddings)
  | ylabel (string): how to label the y-axis of the dendrogram; default:'Mammalia'
  | filepath (string): absolute path including full filename allows for saving the plot\n
  """
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


def check_conservation(glycan, df, network_dic = None, libr = None, rank = 'Order', threshold = 5,
                       motif = False):
  """estimates evolutionary conservation of glycans and glycan motifs via biosynthetic networks\n
  | Arguments:
  | :-
  | glycan (string): full glycan or glycan motif in IUPAC-condensed nomenclature
  | df (dataframe): dataframe in the style of df_species, each row one glycan and columns are the taxonomic levels
  | network_dic (dict): dictionary of form species name : biosynthetic network (gained from construct_network); default:pre-computed milk networks
  | libr (dict): dictionary of form glycoletter:index
  | rank (string): at which taxonomic level to assess conservation; default:Order
  | threshold (int): threshold of how many glycans a species needs to have to consider the species;default:5
  | motif (bool): whether glycan is a motif (True) or a full sequence (False); default:False\n
  | Returns:
  | :-
  | Returns a dictionary of taxonomic group : degree of conservation
  """
  if libr is None:
    libr = lib
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
        conserved[r] =  sum(any(subgraph_isomorphism(node, glycan, libr = libr) for node in nodes) for nodes in rank_nodes) / len(rank_nodes)
    else:
      conserved[r] = sum(glycan in nodes for nodes in rank_nodes) / len(rank_nodes)
  return conserved


def get_communities(graph_list, label_list = None):
  """Find communities for each graph in a list of graphs\n
  | Arguments:
  | :-
  | graph_list (list): list of undirected biosynthetic networks, in the form of networkx objects
  | label_list (list): labels to create the community names, which are running_number + _ + label[k]  for graph_list[k]; default:range(len(graph_list))\n
  | Returns:
  | :-
  | Returns a merged dictionary of community : glycans in that community
  """
  # Perform community detection on each network graph
  comm_list = [community_louvain.best_partition(g) for g in graph_list]
  if label_list is None:
    label_list = list(range(len(comm_list)))
  final_comm_dict = {}
  # Label the communities by species name and running number to distinguish them afterwards
  for comm_dict, label in zip(comm_list, label_list):
    updated_dict = {
            f"{value}_{label}": [k for k, v in comm_dict.items() if v == value]
            for value in set(comm_dict.values())
        }
    final_comm_dict.update(updated_dict)
  return final_comm_dict
