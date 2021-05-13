import re
import itertools
import mpld3
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glycowork.glycan_data.loader import lib, unwrap
from glycowork.motif.graph import fast_compare_glycans, compare_glycans, glycan_to_nxGraph, graph_to_string

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
  glycan_motif = [libr[k] for k in list(sorted(list(nx.get_node_attributes(subgraph, 'labels').values())))]
  glycan_motif = glycan_motif[0] + '(' + glycan_motif[1] + ')'
  return glycan_motif

def find_diff(glycan_a, glycan_b, libr = None):
  """finds the subgraph that differs between glycans and returns it, will only work if the differing subgraph is connected\n
  | Arguments:
  | :-
  | glycan_a (string): glycan in IUPAC-condensed format
  | glycan_b (string): glycan in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns difference between glycan_a and glycan_b in IUPAC-condensed
  """
  if libr is None:
    libr = lib
  lens = [len(glycan_a), len(glycan_b)]
  glycans = [glycan_a, glycan_b]
  larger_graph = [libr[k] for k in list(sorted(list(nx.get_node_attributes(glycan_to_nxGraph(glycans[np.argmax(lens)], libr = libr), 'labels').values())))]
  smaller_graph = [libr[k] for k in list(sorted(list(nx.get_node_attributes(glycan_to_nxGraph(glycans[np.argmin(lens)], libr = libr), 'labels').values())))]
  for k in smaller_graph:
    larger_graph.remove(k)
  return "".join(larger_graph)

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

def get_virtual_nodes(glycan, df, libr = None, graphs = None,
                      glycan_col_name = 'target'):
  """find unobserved biosynthetic precursors of a glycan\n
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
    libr = lib
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
  idx = [k for k in range(len(diffs)) if not any([safe_compare(ggraph_nb[k], j, libr = libr) for j in temp])]
  diffs = [subgraph_to_string(k, libr = libr) for k in diffs]
  ggraph_nb = [ggraph_nb[k] for k in idx if diffs[k].split('(')[1][0] in ['a','b']]
  diffs = [diffs[k] for k in idx if diffs[k].split('(')[1][0] in ['a','b']]
  if len(ggraph_nb)>0:
    out = {}
    for k in range(len(diffs)):
      try:
        out[diffs[k]] = graph_to_string(ggraph_nb[k], libr = libr)
      except:
        pass
  else:
    out = {}
  return out

def find_shared_virtuals(glycan_a, glycan_b, df, libr = None, graphs = None,
                         glycan_col_name = 'target'):
  """finds virtual nodes that are shared between two glycans (i.e., that connect these two glycans)\n
  | Arguments:
  | :-
  | glycan_a (string): glycan in IUPAC-condensed format
  | glycan_b (string): glycan in IUPAC-condensed format
  | df (dataframe): glycan dataframe in which every row contains one glycan (under column: 'target') and labels
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | graphs (list): list of glycans in df as graphs; optional if you call get_neighbors often with the same df and want to provide it precomputed
  | glycan_col_name (string): column name under which glycans are stored; default:target\n
  | Returns:
  | :-
  | (1) list of edges between glycan and virtual node (if virtual node connects the two glycans)
  | (2) list of dictionaries describing how the virtual node differs from the glycans
  """
  if libr is None:
    libr = lib
  virtuals_a = get_virtual_nodes(glycan_a, df, libr = libr, graphs = graphs,
                                 glycan_col_name = glycan_col_name)
  virtuals_b = get_virtual_nodes(glycan_b, df, libr = libr, graphs = graphs,
                                 glycan_col_name = glycan_col_name)
  ggraph_nb_a = list(virtuals_a.values())
  ggraph_nb_b = list(virtuals_b.values())
  diffs_a = list(virtuals_a.keys())
  diffs_b = list(virtuals_b.keys())
  out = []
  out_diffs = []
  if len(ggraph_nb_a)>0:
    for k in range(len(ggraph_nb_a)):
      for j in range(len(ggraph_nb_b)):
        if compare_glycans(ggraph_nb_a[k], ggraph_nb_b[j], libr = libr):
          if [glycan_a, ggraph_nb_a[k]] not in [list(m) for m in out]:
            out.append((glycan_a, ggraph_nb_a[k]))
          if [glycan_b, ggraph_nb_b[k]] not in [list(m) for m in out]:
            out.append((glycan_b, ggraph_nb_a[k]))
          out_diffs.append({df[glycan_col_name].values.tolist().index(glycan_a): diffs_a[k],
                                    df[glycan_col_name].values.tolist().index(glycan_b): diffs_b[j]})
  return out, out_diffs

def fill_with_virtuals(glycans, df, libr = None, graphs = None,
                         glycan_col_name = 'target'):
  """for a list of glycans, identify virtual nodes connecting observed glycans and return their edges\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed
  | df (dataframe): glycan dataframe in which every row contains one glycan (under column: 'target') and labels
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | graphs (list): list of glycans in df as graphs; optional if you call get_neighbors often with the same df and want to provide it precomputed
  | glycan_col_name (string): column name under which glycans are stored; default:target\n
  | Returns:
  | :-
  | (1) list of edges that connect observed glycans to virtual nodes
  | (2) dictionary of edge labels detailing the difference between glycan and virtual node
  """
  if libr is None:
    libr = lib
  virtuals = [find_shared_virtuals(k[0],k[1], df, libr = libr,
                                   graphs = graphs, glycan_col_name = glycan_col_name) for k in list(itertools.combinations(glycans, 2))]
  v_edges = unwrap([k[0] for k in virtuals])
  v_labels = [k[1][0] for k in virtuals if len(k[1])>0]
  end_edges = [m[1] for m in v_edges]
  new_keys = [k for k in range(len(list(set(end_edges))))]
  idx = [list(set(end_edges)).index(k) for k in end_edges]
  v_labels_out = {new_keys[k]+len(df):[] for k in range(len(new_keys))}
  for k in range(len(v_labels)):
    v_labels_out[idx[k]+len(df)].append(v_labels[k])
  for k in list(v_labels_out.keys()):
    if len(v_labels_out[k])>0:
      v_labels_out[k] = v_labels_out[k][0]
    else:
      v_labels_out[k] = {}
  return v_edges, v_labels_out

def create_adjacency_matrix(glycans, df, libr = None, virtual_nodes = False,
                            glycan_col_name = 'target'):
  """creates a biosynthetic adjacency matrix from a list of glycans\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | df (dataframe): glycan dataframe in which every row contains one glycan (under column: 'target') and labels
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | virtual_nodes (bool): whether to include virtual nodes in network; default:False
  | glycan_col_name (string): column name under which glycans are stored; default:target\n
  | Returns:
  | :-
  | (1) adjacency matrix (glycan X glycan) denoting whether two glycans are connected by one biosynthetic step
  | (2) dictionary of glycan motifs in IUPAC-condensed for how each precursor differs from product glycan
  | (3) list of which nodes are virtual nodes (empty list if virtual_nodes is False)
  """
  if libr is None:
    libr = lib
  df_out = pd.DataFrame(0, index = glycans, columns = glycans)
  graphs = [glycan_to_nxGraph(k, libr = libr) for k in df[glycan_col_name].values.tolist()]
  neighbors_full = [get_neighbors(k, df, libr = libr, graphs = graphs) for k in glycans]
  neighbors = [k[0] for k in neighbors_full]
  edge_labels = [k[1] for k in neighbors_full]
  edge_labels = {k:edge_labels[k] for k in range(len(glycans))}
  idx = [k[2] for k in neighbors_full]
  for j in range(len(glycans)):
    if len(idx[j])>0:
      for i in range(len(idx[j])):
        if len(idx[j][i]) >= 1:
          inp = [idx[j][i], glycans.index(glycans[j])]
          df_out.iloc[inp[0], inp[1]] = 1
  new_nodes = []
  if virtual_nodes:
    virtual_edges, virtual_labels = fill_with_virtuals(glycans, libr = libr,
                                            df = df, glycan_col_name = glycan_col_name,
                                            graphs = graphs)
    edge_labels = {**edge_labels, **virtual_labels}
    unique_nodes = list(set([k[1] for k in virtual_edges]))
    new_nodes = [k+len(df_out) for k in range(len(unique_nodes))]
    for k in unique_nodes:
      df_out[k] = 0
      df_out.loc[len(df_out)] = 0
      df_out.index = df_out.index.values.tolist()[:-1] + [k]
    for k in virtual_edges:
      df_out.loc[k[0], k[1]] = 1
  return df_out + df_out.T, edge_labels, new_nodes

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

##LEGACY
##def generate_edgelabels(graph, edge_labels):
##  """matches precursor differences with edge labels\n
##  | Arguments:
##  | :-
##  | graph (networkx object): biosynthetic network made from adjacency matrix
##  | edge_labels (dictionary): dictionary of glycan motifs in IUPAC-condensed for how each precursor differs from product glycan\n
##  | Returns:
##  | :-
##  | Returns dictionary with formatted edge labels
##  """
##  out_dic = {k:'' for k in list(graph.edges)}
##  for k in range(len(list(graph.edges))):
##    edge = list(graph.edges)[k]
##    inner_dic = edge_labels[edge[0]]
##    if len(list(inner_dic.values())) > 0:
##      if edge[1] in list(inner_dic.keys()):
##        out_dic[edge] = inner_dic[edge[1]]
##    inner_dic = edge_labels[edge[1]]
##    if len(list(inner_dic.values())) > 0:
##      if edge[0] in list(inner_dic.keys()):
##        out_dic[edge] = inner_dic[edge[0]]
##  return out_dic

def plot_network(adjacency_matrix, virtual_nodes = None,
                 plot_format = 'kamada_kawai', plot = True, libr = None):
  """visualize biosynthetic network\n
  | Arguments:
  | :-
  | adjacency_matrix (dataframe): denoting whether two glycans are connected by one biosynthetic step
  | virtual_nodes (list): list of nodes that are virtual nodes
  | plot_format (string): how to layout network, either 'kamada_kawai' or 'spring'; default:'kamada_kawai'
  | plot (bool): whether to plot network or return as networkx object; default:True
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns either a plotted network or a networkx object of the network, depending on the state of 'plot'
  """
  mpld3.enable_notebook()
  if libr is None:
    libr = lib
  network = adjacencyMatrix_to_network(adjacency_matrix)
  edge_labels = {}
  for el in list(network.edges()):
    edge_labels[el] = find_diff(adjacency_matrix.columns.values.tolist()[el[0]],
                                  adjacency_matrix.columns.values.tolist()[el[1]],
                                  libr = libr)
  nx.set_edge_attributes(network, edge_labels, 'diffs')
  nx.set_node_attributes(network, adjacency_matrix.columns.values.tolist(), 'glycans')
  fig, ax = plt.subplots(figsize = (16,16))
  if plot_format == 'kamada_kawai':
    pos = nx.kamada_kawai_layout(network)
  elif plot_format == 'spring':
    pos = nx.spring_layout(network, k = 1/4)
  if virtual_nodes is None:
    scatter = nx.draw_networkx_nodes(network, pos, node_size = 50, alpha = 0.7, ax = ax)
  else:
    color_map = ['darkorange' if k in virtual_nodes else 'cornflowerblue' for k in range(len(list(network.nodes())))]
    scatter = nx.draw_networkx_nodes(network, pos, node_size = 50, alpha = 0.7,
                                     node_color = color_map, ax = ax)
  labels = adjacency_matrix.columns.values.tolist()
  nx.draw_networkx_edges(network, pos, ax = ax, edge_color = 'cornflowerblue')
  nx.draw_networkx_edge_labels(network, pos, edge_labels = nx.get_edge_attributes(network, 'diffs'), ax = ax)
  [sp.set_visible(False) for sp in ax.spines.values()]
  ax.set_xticks([])
  ax.set_yticks([])
  if plot:
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels = labels)
    mpld3.plugins.connect(fig, tooltip)
  else:
    return network
