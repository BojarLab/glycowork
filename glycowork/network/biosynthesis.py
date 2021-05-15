import re
import itertools
import mpld3
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glycowork.glycan_data.loader import lib, unwrap
from glycowork.motif.graph import fast_compare_glycans, compare_glycans, glycan_to_nxGraph, graph_to_string, subgraph_isomorphism

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

def create_neighbors(ggraph, libr = None):
  """creates biosynthetic precursor glycans\n
  | Arguments:
  | :-
  | ggraph (networkx object): glycan graph from glycan_to_nxGraph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns biosynthetic precursor glycans
  """
  if libr is None:
    libr = lib
  ra = range(len(ggraph.nodes()))
  ggraph_nb = [ggraph.subgraph(k) for k in itertools.combinations(ra, len(ggraph.nodes())-2) if safe_max(np.diff(k)) in [1,3]]
  for k in range(len(ggraph_nb)):
    if list(ggraph_nb[k].nodes())[0] == 2:
      ggraph_nb[k] = nx.relabel_nodes(ggraph_nb[k], {j:j-2 for j in list(ggraph_nb[k].nodes())})
  ggraph_nb = [k for k in ggraph_nb if nx.is_connected(k)]
  ggraph_nb = [j for j in ggraph_nb if sum([nx.is_isomorphic(j, i, node_match = nx.algorithms.isomorphism.categorical_node_match('labels', len(libr))) for i in ggraph_nb]) <= 1]
  return ggraph_nb

def get_neighbors(glycan, glycans, libr = None, graphs = None):
  """find (observed) biosynthetic precursors of a glycan\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | glycans (list): list of glycans in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | graphs (list): list of glycans in df as graphs; optional if you call get_neighbors often with the same df and want to provide it precomputed\n
  | Returns:
  | :-
  | (1) a list of direct glycan precursors in IUPAC-condensed
  | (2) a list of indices where each precursor from (1) can be found in glycans
  """
  if libr is None:
    libr = libr
  ggraph = glycan_to_nxGraph(glycan, libr = libr)
  ggraph_nb = create_neighbors(ggraph, libr = libr)
  if graphs is None:
    temp = [glycan_to_nxGraph(k, libr = libr) for k in glycans]
  else:
    temp = graphs
  idx = [np.where([safe_compare(k, j, libr = libr) for k in temp])[0].tolist() for j in ggraph_nb]
  nb = [glycans[k[0]] for k in idx if len(k)>0]
  return nb, idx

def get_virtual_nodes(glycan, libr = None, reducing_end = ['Glc', 'GlcNAc']):
  """find unobserved biosynthetic precursors of a glycan\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:['Glc','GlcNAc']\n
  | Returns:
  | :-
  | (1) list of virtual node graphs
  | (2) list of virtual nodes in IUPAC-condensed format
  """
  if libr is None:
    libr = lib
  try:
    ggraph = glycan_to_nxGraph(glycan, libr = libr)
  except:
    return [], []
  ggraph_nb = create_neighbors(ggraph, libr = libr)
  ggraph_nb_t = []
  for k in ggraph_nb:
    try:
      ggraph_nb_t.append(graph_to_string(k, libr = libr))
    except:
      pass
  ggraph_nb = []
  for k in ggraph_nb_t:
    try:
      ggraph_nb.append(glycan_to_nxGraph(k, libr = libr))
    except:
      pass
  ggraph_nb_t = [graph_to_string(k,libr=libr) for k in ggraph_nb]
  ggraph_nb_t = [k if k[0]!='[' else k.replace('[','',1).replace(']','',1) for k in ggraph_nb_t]
  
  ggraph_nb_t = [k for k in ggraph_nb_t if any([k[-len(j):] == j for j in reducing_end])]
  ggraph_nb = [glycan_to_nxGraph(k, libr = libr) for k in ggraph_nb_t]
  diffs = [find_diff(glycan, k) for k in ggraph_nb_t]
  idx = [k for k in range(len(ggraph_nb_t)) if ggraph_nb_t[k][0]!='(']
  return [ggraph_nb[i] for i in idx], [ggraph_nb_t[i] for i in idx]

def find_shared_virtuals(glycan_a, glycan_b, libr = None, reducing_end = ['Glc', 'GlcNAc']):
  """finds virtual nodes that are shared between two glycans (i.e., that connect these two glycans)\n
  | Arguments:
  | :-
  | glycan_a (string): glycan in IUPAC-condensed format
  | glycan_b (string): glycan in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:['Glc','GlcNAc']\n
  | Returns:
  | :-
  | Returns list of edges between glycan and virtual node (if virtual node connects the two glycans)
  """
  if libr is None:
    libr = lib
  virtuals_a = get_virtual_nodes(glycan_a, libr = libr, reducing_end = reducing_end)
                                 
  virtuals_b = get_virtual_nodes(glycan_b, libr = libr, reducing_end = reducing_end)
  ggraph_nb_a = virtuals_a[0]
  ggraph_nb_b = virtuals_b[0]
  glycans_a = virtuals_a[1]
  glycans_b = virtuals_b[1]
  out = []
  if len(ggraph_nb_a)>0:
    for k in range(len(ggraph_nb_a)):
      for j in range(len(ggraph_nb_b)):
        if fast_compare_glycans(ggraph_nb_a[k], ggraph_nb_b[j], libr = libr):
          if [glycan_a, glycans_a[k]] not in [list(m) for m in out]:
            out.append((glycan_a, glycans_a[k]))
          if [glycan_b, glycans_b[j]] not in [list(m) for m in out]:
            out.append((glycan_b, glycans_a[k]))
  return out

def fill_with_virtuals(glycans, libr = None, reducing_end = ['Glc', 'GlcNAc']):
  """for a list of glycans, identify virtual nodes connecting observed glycans and return their edges\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:['Glc','GlcNAc']\n
  | Returns:
  | :-
  | Returns list of edges that connect observed glycans to virtual nodes
  """
  if libr is None:
    libr = lib
  v_edges = [find_shared_virtuals(k[0],k[1], libr = libr, reducing_end = reducing_end) for k in list(itertools.combinations(glycans, 2))]
  v_edges = unwrap(v_edges)
  return v_edges

def create_adjacency_matrix(glycans, libr = None, virtual_nodes = False,
                            reducing_end = ['Glc', 'GlcNAc']):
  """creates a biosynthetic adjacency matrix from a list of glycans\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | virtual_nodes (bool): whether to include virtual nodes in network; default:False
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:['Glc','GlcNAc']\n
  | Returns:
  | :-
  | (1) adjacency matrix (glycan X glycan) denoting whether two glycans are connected by one biosynthetic step
  | (2) list of which nodes are virtual nodes (empty list if virtual_nodes is False)
  """
  if libr is None:
    libr = lib
  df_out = pd.DataFrame(0, index = glycans, columns = glycans)
  graphs = [glycan_to_nxGraph(k, libr = libr) for k in glycans]
  neighbors_full = [get_neighbors(k, glycans, libr = libr, graphs = graphs) for k in glycans]
  neighbors = [k[0] for k in neighbors_full]
  idx = [k[1] for k in neighbors_full]
  for j in range(len(glycans)):
    if len(idx[j])>0:
      for i in range(len(idx[j])):
        if len(idx[j][i]) >= 1:
          inp = [idx[j][i], glycans.index(glycans[j])]
          df_out.iloc[inp[0], inp[1]] = 1
  new_nodes = []
  if virtual_nodes:
    virtual_edges = fill_with_virtuals(glycans, libr = libr, reducing_end = reducing_end)
    new_nodes = list(set([k[1] for k in virtual_edges]))
    new_nodes = [k for k in new_nodes if k not in df_out.columns.values.tolist()]
    idx = np.where([any([compare_glycans(k,j, libr = lib) for j in df_out.columns.values.tolist()]) for k in new_nodes])[0].tolist()
    if len(idx)>0:
      virtual_edges = [j for j in virtual_edges if j[1] not in [new_nodes[k] for k in idx]]
      new_nodes = [new_nodes[k] for k in range(len(new_nodes)) if k not in idx]
    for k in new_nodes:
      df_out[k] = 0
      df_out.loc[len(df_out)] = 0
      df_out.index = df_out.index.values.tolist()[:-1] + [k]
    for k in virtual_edges:
      df_out.loc[k[0], k[1]] = 1
  return df_out + df_out.T, new_nodes

def adjacencyMatrix_to_network(adjacency_matrix):
  """converts an adjacency matrix to a network\n
  | Arguments:
  | :-
  | adjacency_matrix (dataframe): denoting whether two glycans are connected by one biosynthetic step\n
  | Returns:
  | :-
  | Returns biosynthetic network as a networkx graph
  """
  network = nx.convert_matrix.from_numpy_array(adjacency_matrix.values)
  network = nx.relabel_nodes(network, {k:adjacency_matrix.columns.values.tolist()[k] for k in range(len(adjacency_matrix))})
  return network

def propagate_virtuals(glycans, libr = None, reducing_end = ['Glc', 'GlcNAc']):
  """do one step of virtual node generation\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:['Glc','GlcNAc']\n
  | Returns:
  | :-
  | (1) list of virtual node graphs
  | (2) list of virtual nodes in IUPAC-condensed format
  """
  if libr is None:
    libr = lib
  virtuals = [get_virtual_nodes(k, libr = libr, reducing_end = reducing_end) for k in glycans if k.count('(')>1]
  virtuals_t = [k[1] for k in virtuals]
  virtuals = [k[0] for k in virtuals]
  return virtuals, virtuals_t

def shells_to_edges(prev_shell, next_shell):
  """map virtual node generations to edges\n
  | Arguments:
  | :-
  | prev_shell (list): virtual node generation from previous run of propagate_virtuals
  | next_shell (list): virtual node generation from next run of propagate_virtuals\n
  | Returns:
  | :-
  | Returns mapped edges between two virtual node generations
  """
  edges_out = []
  for m in range(len(prev_shell)):
    edges_out.append(unwrap([[(prev_shell[m][k], next_shell[k][j]) for j in range(len(next_shell[k]))] for k in range(len(prev_shell[m]))]))
  return edges_out

def find_path(glycan_a, glycan_b, libr = None, reducing_end = ['Glc', 'GlcNAc'],
              limit = 5):
  """find virtual node path between two glycans\n
  | Arguments:
  | :-
  | glycan_a (string): glycan in IUPAC-condensed format
  | glycan_b (string): glycan in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:['Glc','GlcNAc']
  | limit (int): maximum number of virtual nodes between observed nodes; default:5\n
  | Returns:
  | :-
  | (1) list of edges to connect glycan_a and glycan_b via virtual nodes
  | (2) dictionary of edge labels detailing difference between two connected nodes
  """
  if libr is None:
    libr = lib
  virtual_shells_t = []
  true_nodes = [glycan_a, glycan_b]
  smaller_glycan = true_nodes[np.argmin([len(j) for j in true_nodes])]
  larger_glycan = true_nodes[np.argmax([len(j) for j in true_nodes])]
  virtual_shells = [glycan_to_nxGraph(larger_glycan, libr = libr)]
  virtual_shells_t = [[[larger_glycan]]]
  virtuals, virtuals_t = propagate_virtuals([larger_glycan], libr = libr, reducing_end = reducing_end)
  virtual_shells.append(virtuals)
  virtual_shells_t.append(virtuals_t)
  county = 0
  while ((not any([fast_compare_glycans(glycan_to_nxGraph(smaller_glycan, libr = lib), k) for k in unwrap(virtuals)])) and (county < limit)):
    virtuals, virtuals_t = propagate_virtuals(unwrap(virtuals_t), libr = libr, reducing_end = reducing_end)
    virtual_shells.append(virtuals)
    virtual_shells_t.append(virtuals_t)
    county += 1
  virtual_edges = unwrap([unwrap(shells_to_edges(virtual_shells_t[k], virtual_shells_t[k+1])) for k in range(len(virtual_shells_t)-1)])
  edge_labels = {}
  for el in virtual_edges:
    try:
      edge_labels[el] = find_diff(el[0], el[1], libr = libr)
    except:
      pass
  return virtual_edges, edge_labels

def make_network_from_edges(edges, edge_labels = None):
  """converts edge list to network\n
  | Arguments:
  | :-
  | edges (list): list of edges
  | edge_labels (dict): dictionary of edge labels (optional)\n
  | Returns:
  | :-
  | Returns networkx object
  """
  network = nx.from_edgelist(edges)
  if edge_labels is not None:
    nx.set_edge_attributes(network, edge_labels, 'diffs')
  return network

def find_shortest_path(goal_glycan, glycan_list, libr = None, reducing_end = ['Glc','GlcNAc'],
                       limit = 5):
  """finds the glycan with the shortest path via virtual nodes to the goal glycan\n
  | Arguments:
  | :-
  | goal_glycan (string): glycan in IUPAC-condensed format
  | glycan_list (list): list of glycans in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:['Glc','GlcNAc']
  | limit (int): maximum number of virtual nodes between observed nodes; default:5\n
  | Returns:
  | :-
  | (1) list of edges of shortest path to connect goal_glycan and glycan via virtual nodes
  | (2) dictionary of edge labels detailing difference between two connected nodes in shortest path
  """
  if libr is None:
    libr = lib
  path_lengths = []
  for k in glycan_list:
    if subgraph_isomorphism(goal_glycan, k, libr = libr) and goal_glycan[-3:] == k[-3:]:
      virtual_edges, edge_labels = find_path(goal_glycan, k, libr = libr,
                                                              reducing_end = reducing_end,
                                                               limit = limit)
      network = make_network_from_edges(virtual_edges)
      if k in list(network.nodes()) and goal_glycan in list(network.nodes()):
        path_lengths.append(len(nx.algorithms.shortest_paths.generic.shortest_path(network, source = k, target = goal_glycan)))
      else:
        path_lengths.append(99)
    else:
      path_lengths.append(99)
  idx = np.argmin(path_lengths)
  virtual_edges, edge_labels = find_path(goal_glycan, glycan_list[idx], libr = libr,
                                                             reducing_end = reducing_end,
                                                           limit = limit)
  return virtual_edges, edge_labels

def get_unconnected_nodes(network, glycan_list):
  """find nodes that are currently unconnected\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network from construct_network
  | glycan_list (list): list of glycans in IUPAC-condensed format\n
  | Returns:
  | :-
  | Returns list of unconnected nodes
  """
  connected_nodes = list(network.edges())
  connected_nodes = list(sum(connected_nodes, ()))
  unconnected = [k for k in glycan_list if k not in connected_nodes]
  return unconnected

def construct_network(glycans, add_virtual_nodes = 'none', libr = None, reducing_end = ['Glc','GlcNAc'],
                 limit = 5):
  """visualize biosynthetic network\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | add_virtual_nodes (string): indicates whether no ('None'), proximal ('simple'), or all ('exhaustive') virtual nodes should be added; default:'none'
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:['Glc','GlcNAc']
  | limit (int): maximum number of virtual nodes between observed nodes; default:5\n
  | Returns:
  | :-
  | Returns either a networkx object of the network (and a list of virtual nodes if add_virtual_nodes either 'simple' or 'exhaustive')
  """
  if libr is None:
    libr = lib
  if add_virtual_nodes in ['simple', 'exhaustive']:
    virtuals = True
  else:
    virtuals = False
  adjacency_matrix, virtual_nodes = create_adjacency_matrix(glycans, libr = libr, virtual_nodes = virtuals,
                            reducing_end = reducing_end)
  network = adjacencyMatrix_to_network(adjacency_matrix)
  if add_virtual_nodes == 'exhaustive':
    unconnected_nodes = get_unconnected_nodes(network, list(network.nodes()))
    new_nodes = []
    new_edges = []
    new_edge_labels =  []
    for k in list(sorted(unconnected_nodes)):
      try:
        virtual_edges, edge_labels = find_shortest_path(k, [j for j in list(network.nodes()) if j != k], libr = libr,
                                                      reducing_end = reducing_end, limit = limit)
        total_nodes = list(set(list(sum(virtual_edges, ()))))
        new_nodes.append([j for j in total_nodes if j not in list(network.nodes())])
        new_edges.append(virtual_edges)
        new_edge_labels.append(edge_labels)
      except:
        pass
    network.add_edges_from(unwrap(new_edges), edge_labels = unwrap(new_edge_labels))
    virtual_nodes = virtual_nodes + list(set(unwrap(new_nodes)))
    for ed in (network.edges()):
      if ed[0] in virtual_nodes and ed[1] in virtual_nodes:
        larger_ed = np.argmax([len(e) for e in [ed[0], ed[1]]])
        if network.degree(ed[larger_ed]) == 1:
          network.remove_edge(ed)
  edge_labels = {}
  for el in list(network.edges()):
    edge_labels[el] = find_diff(el[0], el[1], libr = libr)
  nx.set_edge_attributes(network, edge_labels, 'diffs')
  if add_virtual_nodes in ['simple', 'exhaustive']:
    return network, virtual_nodes
  else:
    return network

def plot_network(network, virtual_nodes = None, plot_format = 'kamada_kawai'):
  """visualizes biosynthetic network\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | virtual_nodes (list): optional list of indicating which nodes are virtual nodes; default:None
  | plot_format (string): how to layout network, either 'kamada_kawai' or 'spring'; default:'kamada_kawai'\n
  """
  mpld3.enable_notebook()
  fig, ax = plt.subplots(figsize = (16,16))
  if plot_format == 'kamada_kawai':
    pos = nx.kamada_kawai_layout(network)
  elif plot_format == 'spring':
    pos = nx.spring_layout(network, k = 1/4)
  if virtual_nodes is None:
    scatter = nx.draw_networkx_nodes(network, pos, node_size = 50, alpha = 0.7, ax = ax)
  else:
    color_map = ['darkorange' if k in virtual_nodes else 'cornflowerblue' for k in (list(network.nodes()))]
    scatter = nx.draw_networkx_nodes(network, pos, node_size = 50, alpha = 0.7,
                                     node_color = color_map, ax = ax)
  labels = list(network.nodes())
  nx.draw_networkx_edges(network, pos, ax = ax, edge_color = 'cornflowerblue')
  nx.draw_networkx_edge_labels(network, pos, edge_labels = nx.get_edge_attributes(network, 'diffs'), ax = ax)
  [sp.set_visible(False) for sp in ax.spines.values()]
  ax.set_xticks([])
  ax.set_yticks([])
  tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels = labels)
  mpld3.plugins.connect(fig, tooltip)
