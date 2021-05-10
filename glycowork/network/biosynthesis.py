import re
import itertools
import networkx as nx
import numpy as np
import pandas as pd
from glycowork.glycan_data.loader import lib
from glycowork.motif.graph import fast_compare_glycans, glycan_to_nxGraph

def safe_compare(g1, g2, libr = None):
  """fast_compare_glycans with try/except error catch\n
  | Arguments:
  | :-
  | g1 (networkx object): glycan graph from glycan_to_nxGraph
  | g2 (networkx object): glycan graph from glycan_to_nxGraph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-  
  | Returns True if two glycans are the same and False if not; returns False if 'except' is triggered
  """
  if libr is None:
    libr = lib
  try:
    return fast_compare_glycans(g1, g2, libr = lib)
  except:
    return False

def safe_max(diff_list):
  """returns max with try/except error catch to handle empty lists\n
  | Arguments:
  | :-
  | diff_list (list): list of length differences between glycan substructures\n
  | Returns:
  | :- 
  | Returns max length difference; returns 0 if 'except' is triggered
  """
  try:
    return max(diff_list)
  except:
    return 0

def subgraph_to_string(subgraph, libr = None):
  """converts glycan subgraph back to IUPAC-condensed format\n
  | Arguments:
  | :-
  | subgraph (networkx object): subgraph of one monosaccharide and its linkage
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns glycan motif in IUPAC-condensed format (string)
  """
  if libr is None:
    libr = lib
  glycan_motif = [libr[k] for k in list(nx.get_node_attributes(subgraph, 'labels').values())]
  glycan_motif = glycan_motif[0] + '(' + glycan_motif[1] + ')'
  return glycan_motif

def get_neighbors(glycan, df, libr = None, graphs = None,
                  glycan_col_name = 'target'):
  """find (observed) biosynthetic precursors of a glycan\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | df (dataframe): glycan dataframe in which every row contains one glycan (under column: 'target') and labels
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | graphs (list): list of glycans in df as graphs; optional if you call get_neighbors often with the same df and want to provide it precomputed
  | glycan_col_name (string): column name under which glycans are stored; default:target\n
  | Returns:
  | :-
  | (1) a list of direct glycan precursors in IUPAC-condensed
  | (2) dictionary of glycan motifs in IUPAC-condensed for how each precursor in (1) differs from input glycan
  | (3) a list of indices where each precursor from (1) can be found in df
  """
  if libr is None:
    libr = libr
  ggraph = glycan_to_nxGraph(glycan, libr = libr)
  ra = range(len(ggraph.nodes()))
  ggraph_nb = [ggraph.subgraph(k) for k in itertools.combinations(ra, len(ggraph.nodes())-2) if safe_max(np.diff(k)) in [1,3]]
  ggraph_nb = [k for k in ggraph_nb if nx.is_connected(k)]
  ggraph_nb = [j for j in ggraph_nb if sum([nx.is_isomorphic(j, i, node_match = nx.algorithms.isomorphism.categorical_node_match('labels', len(libr))) for i in ggraph_nb]) <= 1]
  diffs = [ggraph.subgraph([j for j in ggraph.nodes if j not in k.nodes]).copy() for k in ggraph_nb]
  if graphs is None:
    temp = [glycan_to_nxGraph(k, libr = libr) for k in df[glycan_col_name].values.tolist()]
  else:
    temp = graphs
  idx = [np.where([safe_compare(k, j, libr = libr) for k in temp])[0].tolist() for j in ggraph_nb]
  diffs = [diffs[k] if len(idx[k])>0 else 99 for k in range(len(idx))]
  diffs = [subgraph_to_string(k, libr = libr) if not isinstance(k, int) else np.nan for k in diffs]
  diffs = {idx[k][0]:diffs[k] for k in range(len(idx)) if len(idx[k])>0}
  nb = [df[glycan_col_name].values.tolist()[k[0]] for k in idx if len(k)>0]
  return nb, diffs, idx

def create_adjacency_matrix(glycans, df, libr = None,
                            glycan_col_name = 'target'):
  """creates a biosynthetic adjacency matrix from a list of glycans\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | df (dataframe): glycan dataframe in which every row contains one glycan (under column: 'target') and labels
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | glycan_col_name (string): column name under which glycans are stored; default:target\n
  | Returns:
  | :-
  | (1) adjacency matrix (glycan X glycan) denoting whether two glycans are connected by one biosynthetic step
  | (2) dictionary of glycan motifs in IUPAC-condensed for how each precursor differs from product glycan
  """
  if libr is None:
    libr = lib
  df_out = pd.DataFrame(0, index = glycans, columns = glycans)
  graphs = [glycan_to_nxGraph(k, libr = libr) for k in df[glycan_col_name].values.tolist()]
  neighbors_full = [get_neighbors(k, df, libr = libr,
                                  graphs = graphs, glycan_col_name = glycan_col_name) for k in glycans]
  neighbors = [k[0] for k in neighbors_full]
  edge_labels = [k[1] for k in neighbors_full]
  edge_labels = {k:edge_labels[k] for k in range(len(glycans))}
  idx = [k[2] for k in neighbors_full]
  for j in range(len(glycans)):
    if len(idx[j])>0:
      for i in range(len(idx[j])):
        if len(idx[j][i]) >= 1:
          df_out.iloc[idx[j][i], glycans.index(glycans[j])] = 1
  return df_out, edge_labels

def adjacencyMatrix_to_network(adjacency_matrix):
  """converts an adjacency matrix to a network\n
  | Arguments:
  | :-
  | adjacency_matrix (dataframe): denoting whether two glycans are connected by one biosynthetic step\n
  | Returns:
  | :-
  | Returns biosynthetic network as a networkx graph
  """
  return nx.convert_matrix.from_numpy_array(adjacency_matrix.values)

def generate_edgelabels(graph, edge_labels):
  """matches precursor differences with edge labels\n
  | Arguments:
  | :-
  | graph (networkx object): biosynthetic network made from adjacency matrix
  | edge_labels (dictionary): dictionary of glycan motifs in IUPAC-condensed for how each precursor differs from product glycan\n
  | Returns:
  | :-
  | Returns dictionary with formatted edge labels
  """
  out_dic = {k:'' for k in list(graph.edges)}
  for k in range(len(list(graph.edges))):
    edge = list(graph.edges)[k]
    inner_dic = edge_labels[edge[0]]
    if len(list(inner_dic.values())) > 0:
      if edge[1] in list(inner_dic.keys()):
        out_dic[edge] = inner_dic[edge[1]]
    inner_dic = edge_labels[edge[1]]
    if len(list(inner_dic.values())) > 0:
      if edge[0] in list(inner_dic.keys()):
        out_dic[edge] = inner_dic[edge[0]]
  return out_dic

##figure out whether to include mpld3
##def plot_network(adjacency_matrix, edge_labels):
##  """visualize biosynthetic network\n
##  | Arguments:
##  | :-
##  | adjacency_matrix (dataframe): denoting whether two glycans are connected by one biosynthetic step
##  | edge_labels (dictionary): dictionary of glycan motifs in IUPAC-condensed for how each precursor differs from product glycan\n
##  """
##  network = adjacencyMatrix_to_network(adjacency_matrix)
##  edge_labels = generate_edgelabels(network, edge_labels)
##  fig, ax = plt.subplots(figsize = (16,16))
##  #pos = nx.kamada_kawai_layout(network)
##  pos = nx.spring_layout(network, k = 1/4)
##  scatter = nx.draw_networkx_nodes(network, pos, node_size = 50, alpha = 0.7, ax = ax)
##  labels = adjacency_matrix.columns.values.tolist()
##  nx.draw_networkx_edges(network, pos, ax = ax)
##  nx.draw_networkx_edge_labels(network, pos, edge_labels, ax = ax)
##  tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels = labels)
##  mpld3.plugins.connect(fig, tooltip)
