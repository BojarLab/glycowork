import re
import itertools
import mpld3
import copy
import pkg_resources
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glycowork.glycan_data.loader import lib, unwrap, linkages
from glycowork.motif.graph import fast_compare_glycans, compare_glycans, glycan_to_nxGraph, graph_to_string, subgraph_isomorphism
from glycowork.motif.processing import min_process_glycans
from glycowork.motif.tokenization import stemify_glycan

io = pkg_resources.resource_stream(__name__, "monolink_to_enzyme.csv")
df_enzyme = pd.read_csv(io, sep = '\t')

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
    return fast_compare_glycans(g1, g2, libr = libr)
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
  glycan_motif = [k for k in list(sorted(list(nx.get_node_attributes(subgraph, 'string_labels').values())))]
  glycan_motif = glycan_motif[0] + '(' + glycan_motif[1] + ')'
  return glycan_motif

def get_neighbors(glycan, glycans, libr = None, graphs = None,
                  min_size = 1):
  """find (observed) biosynthetic precursors of a glycan\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | glycans (list): list of glycans in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | graphs (list): list of glycans in df as graphs; optional if you call get_neighbors often with the same df and want to provide it precomputed
  | min_size (int): length of smallest root in biosynthetic network; default:1\n
  | Returns:
  | :-
  | (1) a list of direct glycan precursors in IUPAC-condensed
  | (2) a list of indices where each precursor from (1) can be found in glycans
  """
  if libr is None:
    libr = lib
  ggraph = glycan_to_nxGraph(glycan, libr = libr)
  if len(ggraph.nodes()) == 1:
    return ([], [])
  ggraph_nb = create_neighbors(ggraph, libr = libr, min_size = min_size)
  if graphs is None:
    temp = [glycan_to_nxGraph(k, libr = libr) for k in glycans]
  else:
    temp = graphs
  idx = [np.where([safe_compare(k, j, libr = libr) for k in temp])[0].tolist() for j in ggraph_nb]
  nb = [glycans[k[0]] for k in idx if len(k) > 0]
  return nb, idx

def create_adjacency_matrix(glycans, libr = None, virtual_nodes = False,
                            reducing_end = ['Glc-ol','GlcNAc-ol','Glc3S-ol',
                                            'GlcNAc6S-ol', 'GlcNAc6P-ol', 'GlcNAc1P-ol', 'Glc3P-ol', 'Glc6S-ol', 'GlcOS-ol'],
                            min_size = 1):
  """creates a biosynthetic adjacency matrix from a list of glycans\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | virtual_nodes (bool): whether to include virtual nodes in network; default:False
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | min_size (int): length of smallest root in biosynthetic network; default:1\n
  | Returns:
  | :-
  | (1) adjacency matrix (glycan X glycan) denoting whether two glycans are connected by one biosynthetic step
  | (2) list of which nodes are virtual nodes (empty list if virtual_nodes is False)
  """
  if libr is None:
    libr = lib
  df_out = pd.DataFrame(0, index = glycans, columns = glycans)
  graphs = [glycan_to_nxGraph(k, libr = libr) for k in glycans]
  neighbors_full = [get_neighbors(k, glycans, libr = libr, graphs = graphs,
                                  min_size = min_size) for k in glycans]
  neighbors = [k[0] for k in neighbors_full]
  idx = [k[1] for k in neighbors_full]
  for j in range(len(glycans)):
    if len(idx[j]) > 0:
      for i in range(len(idx[j])):
        if len(idx[j][i]) >= 1:
          inp = [idx[j][i], glycans.index(glycans[j])]
          df_out.iloc[inp[0], inp[1]] = 1
  new_nodes = []
  if virtual_nodes:
    virtual_edges = fill_with_virtuals(glycans, libr = libr, reducing_end = reducing_end,
                                       min_size = min_size)
    new_nodes = list(set([k[1] for k in virtual_edges]))
    new_nodes = [k for k in new_nodes if k not in df_out.columns.values.tolist()]
    idx = np.where([any([compare_glycans(k, j, libr = libr) for j in df_out.columns.values.tolist()]) for k in new_nodes])[0].tolist()
    if len(idx) > 0:
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
  if glycan_a == glycan_b:
    return ""
  glycan_a2 = [k for k in list(sorted(list(nx.get_node_attributes(glycan_to_nxGraph(glycan_a, libr = libr), 'string_labels').values())))]
  glycan_b2 = [k for k in list(sorted(list(nx.get_node_attributes(glycan_to_nxGraph(glycan_b, libr = libr), 'string_labels').values())))]
  lens = [len(glycan_a2), len(glycan_b2)]
  graphs = [glycan_a2, glycan_b2]
  ggraphs = [glycan_a, glycan_b]
  larger_graph = graphs[np.argmax(lens)]
  smaller_graph = graphs[np.argmin(lens)]
  for k in smaller_graph:
    try:
      larger_graph.remove(k)
    except:
      larger_graph = ['dis', 'regard']
  if subgraph_isomorphism(ggraphs[np.argmax(lens)], ggraphs[np.argmin(lens)], libr = libr):
    diff = (larger_graph[0] + "(" + larger_graph[1] + ")")
    if 'S(' in diff or 'P(' in diff:
      return 'disregard'
    else:
      return diff
  else:
    larger_graph = ['dis', 'regard']
    return "".join(larger_graph)

def construct_network(glycans, add_virtual_nodes = 'exhaustive', libr = None, reducing_end = ['Glc-ol','GlcNAc-ol','Glc3S-ol',
                                                                                        'GlcNAc6S-ol', 'GlcNAc6P-ol', 'GlcNAc1P-ol',
                                                                                        'Glc3P-ol', 'Glc6S-ol', 'GlcOS-ol'],
                 limit = 5, ptm = True, allowed_ptms = ['OS','3S','6S','1P','6P','OAc','4Ac'],
                 permitted_roots = ["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"],
                      directed = False, edge_type = 'monolink'):
  """visualize biosynthetic network\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | add_virtual_nodes (string): indicates whether no ('none'), proximal ('simple'), or all ('exhaustive') virtual nodes should be added; default:'exhaustive'
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | limit (int): maximum number of virtual nodes between observed nodes; default:5
  | ptm (bool): whether to consider post-translational modifications in the network construction; default:True
  | allowed_ptms (list): list of PTMs to consider
  | permitted_roots (list): which nodes should be considered as roots; default:["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]
  | directed (bool): whether to return a network with directed edges in the direction of biosynthesis; default:False
  | edge_type (string): indicates whether edges represent monosaccharides ('monosaccharide'), monosaccharide(linkage) ('monolink'), or enzyme catalyzing the reaction ('enzyme'); default:'monolink'\n
  | Returns:
  | :-
  | Returns a networkx object of the network
  """
  if libr is None:
    libr = lib
  if add_virtual_nodes in ['simple', 'exhaustive']:
    virtuals = True
  else:
    virtuals = False
  #generating graph from adjacency of observed glycans
  min_size = min([len(glycan_to_nxGraph(k, libr = libr).nodes()) for k in permitted_roots])
  adjacency_matrix, virtual_nodes = create_adjacency_matrix(glycans, libr = libr, virtual_nodes = virtuals,
                            reducing_end = reducing_end, min_size = min_size)
  network = adjacencyMatrix_to_network(adjacency_matrix)
  #connecting observed via virtual nodes
  if add_virtual_nodes == 'exhaustive':
    unconnected_nodes = get_unconnected_nodes(network, list(network.nodes()))
    new_nodes = []
    new_edges = []
    new_edge_labels =  []
    for k in list(sorted(unconnected_nodes)):
      try:
        virtual_edges, edge_labels = find_shortest_path(k, [j for j in list(network.nodes()) if j != k], libr = libr,
                                                      reducing_end = reducing_end, limit = limit, permitted_roots = permitted_roots)
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
          network.remove_edge(ed[0], ed[1])
  #create edge and node labels
  edge_labels = {}
  for el in list(network.edges()):
    edge_labels[el] = find_diff(el[0], el[1], libr = libr)
  nx.set_edge_attributes(network, edge_labels, 'diffs')
  virtual_labels = {k:(1 if k in virtual_nodes else 0) for k in list(network.nodes())}
  nx.set_node_attributes(network, virtual_labels, 'virtual')
  network = filter_disregard(network)
  #connect post-translational modifications
  if ptm:
    if '-ol' in ''.join(glycans):
      suffix = '-ol'
    elif '1Cer' in ''.join(glycans):
      suffix = '1Cer'
    else:
      suffix = ''
    ptm_links = process_ptm(glycans, allowed_ptms = allowed_ptms, libr = libr, suffix = suffix)
    if len(ptm_links) > 1:
      network = update_network(network, ptm_links[0], edge_labels = ptm_links[1])
  #find any remaining orphan nodes and connect them to the root(s)
  if add_virtual_nodes == 'simple':
    network = deorphanize_nodes(network, reducing_end = reducing_end,
                              permitted_roots = permitted_roots, libr = libr, limit = 1)
  elif add_virtual_nodes == 'exhaustive':
    network = deorphanize_nodes(network, reducing_end = reducing_end,
                              permitted_roots = permitted_roots, libr = libr, limit = limit)
  #final clean-up / condensation step
  if virtuals:
    nodeDict = dict(network.nodes(data = True))
    for node in list(sorted(list(network.nodes()), key = len, reverse = True)):
      if (network.degree[node] <= 1) and (nodeDict[node]['virtual'] == 1):
        network.remove_node(node)
    adj_matrix = create_adjacency_matrix(list(network.nodes()), libr = libr,
                                         reducing_end = reducing_end,
                                         min_size = min_size)
    filler_network = adjacencyMatrix_to_network(adj_matrix[0])
    network.add_edges_from(list(filler_network.edges()))
    network = deorphanize_edge_labels(network, libr = libr)
  #edge label specification
  if edge_type != 'monolink':
    ed = network.edges()
    for e in ed :
      elem = network.get_edge_data(e[0], e[1])
      edge = elem['diffs']
      for link in linkages:
        if link in edge:
          if edge_type == 'monosaccharide':
            elem['diffs'] = edge.split('(')[0]
          elif edge_type == 'enzyme':
            elem['diffs'] = monolink_to_glycoenzyme(edge, df_enzyme)
          else:
            pass
  #remove virtual nodes that are branch isomers of real nodes
  if virtuals:
    virtual_nodes = [x for x,y in network.nodes(data = True) if y['virtual'] == 1]
    real_nodes = [glycan_to_nxGraph(x, libr = libr) for x,y in network.nodes(data = True) if y['virtual'] == 0]
    to_cut = [v for v in virtual_nodes if any([fast_compare_glycans(glycan_to_nxGraph(v, libr = libr), r, libr = libr) for r in real_nodes])]
    network.remove_nodes_from(to_cut)
  #directed or undirected network
  if directed:
    network = make_network_directed(network)
    if virtuals:
      nodeDict = dict(network.nodes(data = True))
      for node in list(sorted(list(network.nodes()), key = len, reverse = True)):
        if (network.out_degree[node] < 1) and (nodeDict[node]['virtual'] == 1):
          network.remove_node(node)
  return network

def plot_network(network, plot_format = 'pydot2', edge_label_draw = True,
                 node_size = False, lfc_dict = None):
  """visualizes biosynthetic network\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | plot_format (string): how to layout network, either 'pydot2', 'kamada_kawai', or 'spring'; default:'pydot2'
  | edge_label_draw (bool): draws edge labels if True; default:True
  | node_size (bool): whether nodes should be sized by an "abundance" attribute in network; default:False
  | lfc_dict (dict): dictionary of enzyme:log2-fold-change to scale edge width; default:None\n
  """
  mpld3.enable_notebook()
  fig, ax = plt.subplots(figsize = (16,16))
  if node_size:
    node_sizes = list(nx.get_node_attributes(network, 'abundance').values())
  else:
    node_sizes = 50
  if plot_format == 'pydot2':
    try:
      pos = nx.nx_pydot.pydot_layout(network, prog = "dot")
    except:
      print("pydot2 threw an error (maybe you're using Windows?); we'll use kamada_kawai instead")
      pos = nx.kamada_kawai_layout(network)
  elif plot_format == 'kamada_kawai':
    pos = nx.kamada_kawai_layout(network)
  elif plot_format == 'spring':
    pos = nx.spring_layout(network, k = 1/4)
  if len(list(nx.get_node_attributes(network, 'origin').values())) > 0:
    node_origins = True
  else:
    node_origins = False
  if node_origins:
    alpha_map = [0.2 if k == 1 else 1 for k in list(nx.get_node_attributes(network, 'virtual').values())]
    scatter = nx.draw_networkx_nodes(network, pos, node_size = node_sizes, ax = ax,
                                       node_color = list(nx.get_node_attributes(network, 'origin').values()),
                                       alpha = alpha_map)
  else:
    color_map = ['darkorange' if k == 1 else 'cornflowerblue' if k == 0 else 'seagreen' for k in list(nx.get_node_attributes(network, 'virtual').values())]
    scatter = nx.draw_networkx_nodes(network, pos, node_size = node_sizes, alpha = 0.7,
                                     node_color = color_map, ax = ax)
  labels = list(network.nodes())
  if edge_label_draw:
    if lfc_dict is None:
      nx.draw_networkx_edges(network, pos, ax = ax, edge_color = 'cornflowerblue')
    else:
      diffs = list(nx.get_edge_attributes(network, 'diffs').values())
      for k in diffs:
        if k not in lfc_dict.keys() :
          lfc_dict[k] = 0
      c_list = ['red' if (lfc_dict[k] < 0) else 'green' if (lfc_dict[k] > 0) else 'cornflowerblue' for k in diffs]
      w_list = [abs(lfc_dict[k]) if abs(lfc_dict[k]) != 0 else 1 for k in diffs]
      nx.draw_networkx_edges(network, pos, ax = ax, edge_color = c_list, width = w_list)
    nx.draw_networkx_edge_labels(network, pos, edge_labels = nx.get_edge_attributes(network, 'diffs'), ax = ax,
                                 verticalalignment = 'baseline')
  else:
    diffs = list(nx.get_edge_attributes(network, 'diffs').values())
    c_list = ['cornflowerblue' if 'Glc' in k else 'yellow' if 'Gal' in k else 'red' if 'Fuc' in k else 'mediumorchid' if '5Ac' in k \
              else 'turquoise' if '5Gc' in k else 'silver' for k in diffs]
    w_list = [1 if 'b1' in k else 3 for k in diffs]
    nx.draw_networkx_edges(network, pos, ax = ax, edge_color = c_list, width = w_list)
  [sp.set_visible(False) for sp in ax.spines.values()]
  ax.set_xticks([])
  ax.set_yticks([])
  tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels = labels)
  mpld3.plugins.connect(fig, tooltip)

def find_shared_virtuals(glycan_a, glycan_b, libr = None, reducing_end = ['Glc-ol','GlcNAc-ol','Glc3S-ol',
                                                                          'GlcNAc6S-ol', 'GlcNAc6P-ol', 'GlcNAc1P-ol',
                                                                          'Glc3P-ol', 'Glc6S-ol', 'GlcOS-ol'],
                         min_size = 1):
  """finds virtual nodes that are shared between two glycans (i.e., that connect these two glycans)\n
  | Arguments:
  | :-
  | glycan_a (string): glycan in IUPAC-condensed format
  | glycan_b (string): glycan in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | min_size (int): length of smallest root in biosynthetic network; default:1\n
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
  if len(ggraph_nb_a) > 0:
    for k in range(len(ggraph_nb_a)):
      for j in range(len(ggraph_nb_b)):
        if len(ggraph_nb_a[k]) >= min_size:
          if fast_compare_glycans(ggraph_nb_a[k], ggraph_nb_b[j], libr = libr):
            if [glycan_a, glycans_a[k]] not in [list(m) for m in out]:
              out.append((glycan_a, glycans_a[k]))
            if [glycan_b, glycans_b[j]] not in [list(m) for m in out]:
              out.append((glycan_b, glycans_a[k]))
  return out

def fill_with_virtuals(glycans, libr = None, reducing_end = ['Glc-ol','GlcNAc-ol','Glc3S-ol',
                                                             'GlcNAc6S-ol', 'GlcNAc6P-ol', 'GlcNAc1P-ol',
                                                             'Glc3P-ol', 'Glc6S-ol', 'GlcOS-ol'],
                       min_size = 1):
  """for a list of glycans, identify virtual nodes connecting observed glycans and return their edges\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | min_size (int): length of smallest root in biosynthetic network; default:1\n
  | Returns:
  | :-
  | Returns list of edges that connect observed glycans to virtual nodes
  """
  if libr is None:
    libr = lib
  v_edges = [find_shared_virtuals(k[0], k[1], libr = libr, reducing_end = reducing_end,
                                  min_size = min_size) for k in list(itertools.combinations(glycans, 2))]
  v_edges = unwrap(v_edges)
  return v_edges

def create_neighbors(ggraph, libr = None, min_size = 1):
  """creates biosynthetic precursor glycans\n
  | Arguments:
  | :-
  | ggraph (networkx object): glycan graph from glycan_to_nxGraph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | min_size (int): length of smallest root in biosynthetic network; default:1\n
  | Returns:
  | :-
  | Returns biosynthetic precursor glycans
  """
  if libr is None:
    libr = lib
  if len(ggraph.nodes()) <= min_size:
    return []
  if len(ggraph.nodes()) == 3:
    ggraph_nb = [ggraph.subgraph([2])]
  else:
    ra = range(len(ggraph.nodes()))
    ggraph_nb = [ggraph.subgraph(k) for k in itertools.combinations(ra, len(ggraph.nodes())-2) if safe_max(np.diff(k)) in [1,3]]
    ggraph_nb = [k for k in ggraph_nb if nx.is_connected(k)]
  for k in range(len(ggraph_nb)):
    if list(ggraph_nb[k].nodes())[0] == 2:
      ggraph_nb[k] = nx.relabel_nodes(ggraph_nb[k], {j:j-2 for j in list(ggraph_nb[k].nodes())})
    if any([j not in list(ggraph_nb[k].nodes()) for j in range(len(list(ggraph_nb[k].nodes())))]):
      which = np.min(np.where([j not in list(ggraph_nb[k].nodes()) for j in range(len(list(ggraph_nb[k].nodes())))])[0].tolist())
      diff = len(list(ggraph_nb[k].nodes()))-which
      current_nodes = list(ggraph_nb[k].nodes())
      ggraph_nb[k] = nx.relabel_nodes(ggraph_nb[k], {current_nodes[m]:current_nodes[m]-diff for m in range(which, len(current_nodes))})
  #ggraph_nb = [j for j in ggraph_nb if sum([nx.is_isomorphic(j, i, node_match = nx.algorithms.isomorphism.categorical_node_match('labels', len(libr))) for i in ggraph_nb]) <= 1]
  return ggraph_nb

def get_virtual_nodes(glycan, libr = None, reducing_end = ['Glc-ol','GlcNAc-ol','Glc3S-ol',
                                                           'GlcNAc6S-ol', 'GlcNAc6P-ol', 'GlcNAc1P-ol',
                                                           'Glc3P-ol', 'Glc6S-ol', 'GlcOS-ol']):
  """find unobserved biosynthetic precursors of a glycan\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends\n
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
  if len(ggraph.nodes()) == 1:
    return ([], [])
  ggraph_nb = create_neighbors(ggraph, libr = libr)
  ggraph_nb_t = []
  for k in ggraph_nb:
    try:
      ggraph_nb_t.append(graph_to_string(k, libr = libr))
    except:
      pass
  ggraph_nb = []
  for k in list(set(ggraph_nb_t)):
    try:
      ggraph_nb.append(glycan_to_nxGraph(k, libr = libr))
    except:
      pass
  ggraph_nb_t = [graph_to_string(k, libr = libr) for k in ggraph_nb]
  ggraph_nb_t = [k if k[0] != '[' else k.replace('[','',1).replace(']','',1) for k in ggraph_nb_t]
  
  ggraph_nb_t = [k for k in ggraph_nb_t if any([k[-len(j):] == j for j in reducing_end])]
  ggraph_nb_t2 = []
  ggraph_nb = []
  for k in range(len(ggraph_nb_t)):
    try:
      ggraph_nb.append(glycan_to_nxGraph(ggraph_nb_t[k], libr = libr))
      ggraph_nb_t2.append(ggraph_nb_t[k])
    except:
      pass
  diffs = [find_diff(glycan, k, libr = libr) for k in ggraph_nb_t2]
  idx = [k for k in range(len(ggraph_nb_t2)) if ggraph_nb_t2[k][0] != '(']
  return [ggraph_nb[i] for i in idx], [ggraph_nb_t2[i] for i in idx]
  
def propagate_virtuals(glycans, libr = None, reducing_end = ['Glc-ol','GlcNAc-ol','Glc3S-ol',
                                                             'GlcNAc6S-ol', 'GlcNAc6P-ol', 'GlcNAc1P-ol',
                                                             'Glc3P-ol', 'Glc6S-ol', 'GlcOS-ol'],
                       permitted_roots = ["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]):
  """do one step of virtual node generation\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | permitted_roots (list): which nodes should be considered as roots; default:["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]\n
  | Returns:
  | :-
  | (1) list of virtual node graphs
  | (2) list of virtual nodes in IUPAC-condensed format
  """
  if libr is None:
    libr = lib
  linkage_count = min([k.count('(') for k in permitted_roots])
  virtuals = [get_virtual_nodes(k, libr = libr, reducing_end = reducing_end) for k in glycans if k.count('(') > linkage_count]
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


def find_path(glycan_a, glycan_b, libr = None, reducing_end = ['Glc-ol','GlcNAc-ol','Glc3S-ol',
                                                               'GlcNAc6S-ol', 'GlcNAc6P-ol', 'GlcNAc1P-ol',
                                                               'Glc3P-ol', 'Glc6S-ol', 'GlcOS-ol'],
              limit = 5, permitted_roots = ["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]):
  """find virtual node path between two glycans\n
  | Arguments:
  | :-
  | glycan_a (string): glycan in IUPAC-condensed format
  | glycan_b (string): glycan in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | limit (int): maximum number of virtual nodes between observed nodes; default:5
  | permitted_roots (list): which nodes should be considered as roots; default:["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]\n
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
  virtuals, virtuals_t = propagate_virtuals([larger_glycan], libr = libr, reducing_end = reducing_end,
                                            permitted_roots = permitted_roots)
  virtual_shells.append(virtuals)
  virtual_shells_t.append(virtuals_t)
  county = 0
  while ((not any([fast_compare_glycans(glycan_to_nxGraph(smaller_glycan, libr = libr), k) for k in unwrap(virtuals)])) and (county < limit)):
    virtuals, virtuals_t = propagate_virtuals(unwrap(virtuals_t), libr = libr, reducing_end = reducing_end,
                                              permitted_roots = permitted_roots)
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

def find_shortest_path(goal_glycan, glycan_list, libr = None, reducing_end = ['Glc-ol','GlcNAc-ol','Glc3S-ol',
                                                                              'GlcNAc6S-ol', 'GlcNAc6P-ol', 'GlcNAc1P-ol',
                                                                              'Glc3P-ol', 'Glc6S-ol', 'GlcOS-ol'],
                       limit = 5, permitted_roots = ["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]):
  """finds the glycan with the shortest path via virtual nodes to the goal glycan\n
  | Arguments:
  | :-
  | goal_glycan (string): glycan in IUPAC-condensed format
  | glycan_list (list): list of glycans in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | limit (int): maximum number of virtual nodes between observed nodes; default:5
  | permitted_roots (list): which nodes should be considered as roots; default:["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]\n
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
      try:
        virtual_edges, edge_labels = find_path(goal_glycan, k, libr = libr,
                                                              reducing_end = reducing_end,
                                                               limit = limit, permitted_roots = permitted_roots)
        network = make_network_from_edges(virtual_edges)
        if k in list(network.nodes()) and goal_glycan in list(network.nodes()):
          path_lengths.append(len(nx.algorithms.shortest_paths.generic.shortest_path(network, source = k, target = goal_glycan)))
        else:
          path_lengths.append(99)
      except:
        path_lengths.append(99)
    else:
      path_lengths.append(99)
  idx = np.argmin(path_lengths)
  virtual_edges, edge_labels = find_path(goal_glycan, glycan_list[idx], libr = libr,
                                                             reducing_end = reducing_end,
                                                           limit = limit, permitted_roots = permitted_roots)
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

def network_alignment(network_a, network_b):
  """combines two networks into a new network\n
  | Arguments:
  | :-
  | network_a (networkx object): biosynthetic network from construct_network
  | network_b (networkx object): biosynthetic network from construct_network\n
  | Returns:
  | :-
  | Returns combined network as a networkx object
  """
  U = nx.Graph()
  all_nodes = list(network_a.nodes(data = True)) + list(network_b.nodes(data = True))
  network_a_nodes = [k[0] for k in list(network_a.nodes(data = True))]
  network_b_nodes =  [k[0] for k in list(network_b.nodes(data = True))]
  node_origin = ['cornflowerblue' if (k[0] in network_a_nodes and k[0] not in network_b_nodes) \
            else 'darkorange' if (k[0] in network_b_nodes and k[0] not in network_a_nodes) else 'saddlebrown' for k in all_nodes]
  U.add_nodes_from(all_nodes)
  U.add_edges_from(list(network_a.edges(data = True))+list(network_b.edges(data = True)))
  nx.set_node_attributes(U, {all_nodes[k][0]:node_origin[k] for k in range(len(all_nodes))}, name = 'origin')
  if nx.is_directed(network_a):
    U = make_network_directed(U)
  return U

def infer_virtual_nodes(network_a, network_b, combined = None):
  """identifies virtual nodes that have been observed in other species\n
  | Arguments:
  | :-
  | network_a (networkx object): biosynthetic network from construct_network
  | network_b (networkx object): biosynthetic network from construct_network
  | combined (networkx object): merged network of network_a and network_b from network_alignment; default:None\n
  | Returns:
  | :-
  | (1) tuple of (virtual nodes of network_a observed in network_b, virtual nodes occurring in both network_a and network_b)
  | (2) tuple of (virtual nodes of network_b observed in network_a, virtual nodes occurring in both network_a and network_b)
  """
  if combined is None:
    combined = network_alignment(network_a, network_b)
  a_nodes = list(network_a.nodes())
  b_nodes = list(network_b.nodes())
  supported_a = [k for k in a_nodes if nx.get_node_attributes(network_a, 'virtual')[k] == 1 and k in list(network_b.nodes())]
  supported_a = [k for k in supported_a if nx.get_node_attributes(network_b, 'virtual')[k] == 1]
  supported_b = [k for k in b_nodes if nx.get_node_attributes(network_b, 'virtual')[k] == 1 and k in list(network_a.nodes())]
  supported_b = [k for k in supported_b if nx.get_node_attributes(network_a, 'virtual')[k] == 1]
  inferred_a = [k for k in a_nodes if nx.get_node_attributes(network_a, 'virtual')[k] == 1 and nx.get_node_attributes(combined, 'virtual')[k] == 0]
  inferred_b = [k for k in b_nodes if nx.get_node_attributes(network_b, 'virtual')[k] == 1 and nx.get_node_attributes(combined, 'virtual')[k] == 0]
  return (inferred_a, supported_a), (inferred_b, supported_b)

def filter_disregard(network):
  """filters out mistaken edges\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network from construct_network
  | Returns:
  | :-
  | Returns network without mistaken edges
  """
  for k in list(network.edges()):
    if nx.get_edge_attributes(network, 'diffs')[k] == 'disregard':
      network.remove_edge(k[0], k[1])
  return network

def infer_network(network, network_species, species_list, filepath = None, df = None,
                  add_virtual_nodes = 'exhaustive',
                  libr = None, reducing_end = ['Glc-ol','GlcNAc-ol','Glc3S-ol',
                                               'GlcNAc6S-ol', 'GlcNAc6P-ol', 'GlcNAc1P-ol',
                                               'Glc3P-ol', 'Glc6S-ol', 'GlcOS-ol'], limit = 5,
                  ptm = False, allowed_ptms = ['OS','3S','6S','1P','6P','OAc','4Ac'],
                 permitted_roots = ["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"], directed = False):
  """replaces virtual nodes if they are observed in other species\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network that should be inferred
  | network_species (string): species from which the network stems
  | species_list (list): list of species to compare network to
  | filepath (string): filepath to load biosynthetic networks from other species, if precalculated (def. recommended, as calculation will take ~1.5 hours); default:None
  | df (dataframe): dataframe containing species-specific glycans, only needed if filepath=None;default:None
  | add_virtual_nodes (string): indicates whether no ('None'), proximal ('simple'), or all ('exhaustive') virtual nodes should be added;only needed if filepath=None;default:'exhaustive'
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used;only needed if filepath=None
  | reducing_end (list): monosaccharides at the reducing end that are allowed;only needed if filepath=None;default:milk glycan reducing ends
  | limit (int): maximum number of virtual nodes between observed nodes;only needed if filepath=None;default:5
  | ptm (bool): whether to consider post-translational modifications in the network construction; default:False
  | allowed_ptms (list): list of PTMs to consider
  | permitted_roots (list): which nodes should be considered as roots; default:["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]
  | directed (bool): whether to return a network with directed edges in the direction of biosynthesis; default:False\n
  | Returns:
  | :-
  | Returns network with filled in virtual nodes
  """
  if libr is None:
    libr = lib
  inferences = []
  for k in species_list:
    if k != network_species:
      if filepath is None:
        temp_network = construct_network(df[df.Species == k].target.values.tolist(), add_virtual_nodes = add_virtual_nodes, libr = libr,
                                         reducing_end = reducing_end, limit = limit, ptm = ptm, allowed_ptms = allowed_ptms,
                                         permitted_roots = permitted_roots, directed = directed)
      else:
        temp_network = nx.read_gpickle(filepath + k + '_graph_exhaustive.pkl')
      temp_network = filter_disregard(temp_network)
      infer_network, infer_other = infer_virtual_nodes(network, temp_network)
      inferences.append(infer_network)
  network2 = network.copy()
  upd = {j:2 for j in list(set(unwrap([k[0] for k in inferences])))}
  nx.set_node_attributes(network2, upd, 'virtual')
  return network2

def retrieve_inferred_nodes(network, species = None):
  """returns the inferred virtual nodes of a network that has been used with infer_network\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network with inferred virtual nodes
  | species (string): species from which the network stems (only relevant if multiple species in network); default:None\n
  | Returns:
  | :-
  | Returns inferred nodes as list or dictionary (if species argument is used)
  """
  inferred_nodes = nx.get_node_attributes(network, 'virtual')
  inferred_nodes = [k for k in list(inferred_nodes.keys()) if inferred_nodes[k] == 2]
  if species is None:
    return inferred_nodes
  else:
    return {species:inferred_nodes}

def detect_ptm(glycans, allowed_ptms = ['OS','3S','6S','1P','6P','OAc','4Ac']):
  """identifies glycans that contain post-translational modifications\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | allowed_ptms (list): list of PTMs to consider\n
  | Returns:
  | :-
  | Returns glycans that contain PTMs
  """
  out = []
  for glycan in glycans:
    glycan_proc = 'x'.join(min_process_glycans([glycan])[0])
    if any([ptm in glycan_proc for ptm in allowed_ptms]):
      out.append(glycan)
  return out

def find_ptm(glycan, glycans, allowed_ptms = ['OS','3S','6S','1P','6P','OAc','4Ac'],
               libr = None, ggraphs = None, suffix = '-ol'):
  """identifies precursor glycans for a glycan with a PTM\n
  | Arguments:
  | :-
  | glycan (string): glycan with PTM in IUPAC-condensed format
  | glycans (list): list of glycans in IUPAC-condensed format
  | allowed_ptms (list): list of PTMs to consider
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | ggraphs (list): list of precomputed graphs of the glycan list
  | suffix (string): optional suffix to be added to the stemified glycan; default:'-ol'\n
  | Returns:
  | :-
  | (1) an edge tuple between input glycan and its biosynthetic precusor without PTM
  | (2) the PTM that is contained in the input glycan
  """
  if libr is None:
    libr = lib
  glycan_proc = min_process_glycans([glycan])[0]
  mod = [ptm for ptm in allowed_ptms if any([ptm in k for k in glycan_proc])][0]
  glycan_stem = stemify_glycan(glycan, libr = libr) + suffix
  if ('Sug' in glycan_stem) or ('Neu(' in glycan_stem):
    print("Stemification fail. This is just information, no need to act.")
    return 0
  try:
    g_stem = glycan_to_nxGraph(glycan_stem, libr = libr)
  except:
    raise IndexError("Okay, here's what's happened: you had a modified monosaccharide that was stemified into the unmodified monosaccharide that is absent from your input data, so it can't be indexed in libr; add the unmodified monosaccharide to your input libr.")
  if ggraphs is None:
    ggraphs = [glycan_to_nxGraph(k, libr = libr) for k in glycans]
  idx = np.where([safe_compare(k, g_stem, libr = libr) for k in ggraphs])[0].tolist()
  if len(idx) > 0:
    nb = [glycans[k] for k in idx][0]
    return ((glycan), (nb)), mod
  else:
    print("No neighbors found. This is just information, no need to act.")
    return 0

def process_ptm(glycans, allowed_ptms = ['OS','3S','6S','1P','6P','OAc','4Ac'],
               libr = None, suffix = '-ol', verbose = False):
  """identifies glycans that contain post-translational modifications and their biosynthetic precursor\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | allowed_ptms (list): list of PTMs to consider
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | suffix (string): optional suffix to be added to the stemified glycan; default:'-ol'
  | verbose (string): whether to write when no PTMs are presently unmatched; default:False\n
  | Returns:
  | :-
  | (1) list of edge tuples between input glycans and their biosynthetic precusor without PTM
  | (2) list of the PTMs that are contained in the input glycans
  """
  if libr is None:
    libr = lib
  ptm_glycans = detect_ptm(glycans, allowed_ptms = allowed_ptms)
  ggraphs = [glycan_to_nxGraph(k, libr = libr) for k in glycans]
  edges = [find_ptm(k, glycans, allowed_ptms = allowed_ptms,
                                 libr = libr, ggraphs = ggraphs, suffix = suffix) for k in ptm_glycans]
  if len([k for k in edges if k != 0]) > 0:
    edges, edge_labels = list(zip(*[k for k in edges if k != 0]))
    return list(edges), list(edge_labels)
  else:
    if verbose:
      print("No unmatched PTMs to match for this species.")
    return []

def update_network(network_in, edge_list, edge_labels = None, node_labels = None):
  """updates a network with new edges and their labels\n
  | Arguments:
  | :-
  | network (networkx object): network that should be modified
  | edge_list (list): list of edges as node tuples
  | edge_labels (list): list of edge labels as strings
  | node_labels (dict): dictionary of form node:0 or 1 depending on whether the node is observed or virtual\n
  | Returns:
  | :-
  | Returns network with added edges
  """
  network = network_in.copy()
  if edge_labels is None:
    network.add_edges_from(edge_list)
  else:
    network.add_edges_from(edge_list)
    nx.set_edge_attributes(network, {edge_list[k]:edge_labels[k] for k in range(len(edge_list))}, 'diffs')
  if node_labels is None:
    return network
  else:
    nx.set_node_attributes(network, node_labels, 'virtual')
    return network

def return_unconnected_to_root(network, permitted_roots = ["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]):
  """finds observed nodes that are not connected to the root nodes\n
  | Arguments:
  | :-
  | network (networkx object): network that should be analyzed
  | permitted_roots (list): which nodes should be considered as roots; default:["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]\n
  | Returns:
  | :-
  | Returns list of nodes that are not connected to the root nodes
  """
  unconnected = []
  nodeDict = dict(network.nodes(data = True))
  for node in list(network.nodes()):
    if nodeDict[node]['virtual'] == 0 and (node not in permitted_roots):
      path_tracing = [nx.algorithms.shortest_paths.generic.has_path(network, node, k) for k in permitted_roots if k in list(network.nodes())]
      if not any(path_tracing):
        unconnected.append(node)
  return unconnected

def deorphanize_nodes(network, reducing_end = ['Glc-ol','GlcNAc-ol','Glc3S-ol',
                                                'GlcNAc6S-ol', 'GlcNAc6P-ol', 'GlcNAc1P-ol',
                                                'Glc3P-ol', 'Glc6S-ol', 'GlcOS-ol'],
                      permitted_roots = ["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"], libr = None, limit = 5):
  """finds nodes unconnected to root nodes and tries to connect them\n
  | Arguments:
  | :-
  | network (networkx object): network that should be modified
  | reducing_end (list): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | permitted_roots (list): which nodes should be considered as roots; default:["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | limit (int): maximum number of virtual nodes between observed nodes; default:5\n
  | Returns:
  | :-
  | Returns network with deorphanized nodes (if they can be connected to root nodes)
  """
  if libr is None:
    libr = lib
  min_size = min([len(glycan_to_nxGraph(k, libr = libr).nodes()) for k in permitted_roots])
  permitted_roots = [k for k in permitted_roots if k in list(network.nodes())]
  if len(permitted_roots) > 0:
    unconnected_nodes = return_unconnected_to_root(network, permitted_roots = permitted_roots)
    temp_network = make_network_directed(network)
    pseudo_connected_nodes = [k[0] for k in list(temp_network.in_degree) if k[1] == 0]
    pseudo_connected_nodes = [k for k in pseudo_connected_nodes if k not in permitted_roots and k not in unconnected_nodes]
    unconnected_nodes = unconnected_nodes + pseudo_connected_nodes
    unconnected_nodes = [k for k in unconnected_nodes if len(glycan_to_nxGraph(k, libr = libr)) > min_size]
    nodeDict = dict(network.nodes(data = True))
    real_nodes = [node for node in list(network.nodes()) if nodeDict[node]['virtual'] == 0]
    real_nodes = [node for node in real_nodes if node not in unconnected_nodes]
    edges = []
    edge_labels = []
    for node in unconnected_nodes:
      try:
        e, el = find_shortest_path(node, real_nodes, reducing_end = reducing_end,
                                                       libr = libr, limit = limit,
                                   permitted_roots = permitted_roots)
        edges.append(e)
        edge_labels.append(el)
      except:
        pass
    edge_labels = unwrap([[edge_labels[k][edges[k][j]] for j in range(len(edges[k]))] for k in range(len(edge_labels))])
    edges = unwrap(edges)
    node_labels = {node:(nodeDict[node]['virtual'] if node in list(network.nodes()) else 1) for node in [i for sub in edges for i in sub]}
    network_out = update_network(network, edges, edge_labels = edge_labels, node_labels = node_labels)
    network_out = filter_disregard(network_out)
    return network_out
  else:
    return network

def make_network_directed(network):
  """converts a network with undirected edges to one with directed edges\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network\n
  | Returns:
  | :-
  | Returns a network with directed edges in the direction of biosynthesis
  """
  dnetwork = network.to_directed()
  dnetwork = prune_directed_edges(dnetwork)
  return dnetwork

def prune_directed_edges(network):
  """removes edges that go against the direction of biosynthesis\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network with directed edges\n
  | Returns:
  | :-
  | Returns a network with pruned edges that would go in the wrong direction
  """
  nodes = list(network.nodes())
  for k in nodes:
    for j in nodes:
      if network.has_edge(k,j) and (len(k) > len(j)):
        network.remove_edge(k,j)
  return network

def deorphanize_edge_labels(network, libr = None):
  """completes edge labels in newly added edges of a biosynthetic network\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns a network with completed edge labels
  """
  if libr is None:
    libr = lib
  edge_labels = nx.get_edge_attributes(network, 'diffs')
  for k in list(network.edges()):
    if k not in list(edge_labels.keys()):
      diff = find_diff(k[0], k[1], libr = libr)
      network.edges[k]['diffs'] = diff
  return network

def export_network(network, filepath):
  """converts NetworkX network into files usable by Gephi\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | filepath (string): should describe a valid path + file name prefix, will be appended by file description and type\n
  | Returns:
  | :-
  | (1) saves a .csv dataframe containing the edge list and edge labels
  | (2) saves a .csv dataframe containing node IDs and labels
  """
  edge_list = list(network.edges())
  edge_list = pd.DataFrame(edge_list)
  edge_list.columns = ['Source', 'Target']
  edge_labels = nx.get_edge_attributes(network, 'diffs')
  edge_list['Label'] = [edge_labels[k] for k in list(network.edges())]
  edge_list = edge_list.replace('z', '?', regex = True)
  edge_list.to_csv(filepath + 'edge_list.csv', index = False)

  node_labels = nx.get_node_attributes(network, 'virtual')
  node_labels = pd.DataFrame(node_labels, index = [0]).T.reset_index()
  node_labels.columns = ['Id', 'Virtual']
  node_labels = node_labels.replace('z', '?', regex = True)
  node_labels.to_csv(filepath + 'node_labels.csv', index = False)

def monolink_to_glycoenzyme(edge_label, df, enzyme_column = 'glycoenzyme',
                            monolink_column = 'monolink', mode = 'condensed') :
  """replaces monosaccharide(linkage) edge label by the name of the enzyme responsible for its synthesis\n
  | Arguments:
  | :-
  | edge_label (str): 'monolink' edge label
  | df (dataframe): dataframe containing the glycoenzymes and their corresponding 'monolink'
  | enzyme_column (str): name of the column in df that indicates the enzyme; default:'glycoenzyme'
  | monolink_column (str): name of the column in df that indicates the 'monolink'; default:'monolink'
  | mode (str): determine if all glycoenzymes must be represented (full) or if only glycoenzyme families are shown (condensed); default:'condensed'\n
  | Returns:
  | :-
  | Returns a new edge label corresponding to the enzyme that catalyzes this reaction
  """
  if mode == 'condensed' :
    enzyme_column = 'glycoclass'
  new_edge_label = df[enzyme_column].values[df[monolink_column] == edge_label]
  if str(new_edge_label) == 'None' or str(new_edge_label) == '[]' :
    return(edge_label)
  else :
    return(str(list(set(new_edge_label))).replace("'","").replace("[","").replace("]","").replace(", ","_"))

def infuse_network(network, node_df, node_abundance = True, glycan_col = 'target',
                   intensity_col = 'rel_intensity'):
  """add new data to an existing network, such as glycomics abundance data\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | node_df (dataframe): dataframe containing glycans and their relative intensity
  | node_abundance (bool): whether to add node abundance to the network as the node attribute "abundance"; default:True
  | glycan_col (string): column name of the glycans in node_df
  | intensity_col (string): column name of the relative intensities in node_df\n
  | Returns:
  | :-
  | Returns a network with added information
  """
  if node_abundance:
    for node in list(network.nodes()):
      if node in node_df[glycan_col].values.tolist():
        network.nodes[node]['abundance'] = node_df[intensity_col].values.tolist()[node_df[glycan_col].values.tolist().index(node)]*100
      else:
        network.nodes[node]['abundance'] = 50
  return network

def choose_path(source, target, species_list, filepath, libr = None,
                file_suffix = '_graph_exhaustive.pkl'):
  """given a diamond-shape in biosynthetic networks (A->B,A->C,B->D,C->D), which path is taken from A to D?\n
  | Arguments:
  | :-
  | source (string): glycan node that is the biosynthetic precursor
  | target (string): glycan node that is the biosynthetic product; has to two biosynthetic steps away from source
  | species_list (list): list of species to compare network to
  | filepath (string): filepath to load biosynthetic networks from other species; files need to be species name + file_suffix
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | file_suffix (string): generic end part of filename in filepath; default:'_graph_exhaustive.pkl'\n
  | Returns:
  | :-
  | Returns dictionary of each intermediary glycan and its proportion (0-1) of how often it has been experimentally observed in this path
  """
  if libr is None:
    libr = lib
  network = construct_network([source, target], libr = libr, add_virtual_nodes = 'exhaustive',
                              ptm = True, directed = True)
  alternatives = [path[1] for path in nx.all_simple_paths(network, source = source, target = target)]
  if len(alternatives) < 2:
    return {}
  alt_a = []
  alt_b = []
  for k in species_list:
    temp_network = nx.read_gpickle(filepath + k + file_suffix)
    if source in list(temp_network.nodes()) and target in list(temp_network.nodes()):
      if alternatives[0] in list(temp_network.nodes()):
        alt_a.append(temp_network.nodes[alternatives[0]]['virtual'])
      if alternatives[1] in list(temp_network.nodes()):
        alt_b.append(temp_network.nodes[alternatives[1]]['virtual'])
  alt_a = alt_a.count(0)/max([len(alt_a), 1])
  alt_b = alt_b.count(0)/max([len(alt_b), 1])
  if alt_a == alt_b == 0:
    alt_a = 0.01
    alt_b = 0.01
  inferences = {alternatives[0]:alt_a, alternatives[1]:alt_b}
  return inferences

def find_diamonds(network):
  """automatically extracts diamond-shaped motifs (A->B,A->C,B->D,C->D) from biosynthetic networks\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network\n
  | Returns:
  | :-
  | Returns list of dictionaries, with each dictionary a diamond motif in the form of position in the diamond : glycan
  """
  g1 = nx.DiGraph()
  nx.add_path(g1, [1,2,3])
  nx.add_path(g1, [1,4,3])
  graph_pair = nx.algorithms.isomorphism.GraphMatcher(network, g1)
  matchings_list = list(graph_pair.subgraph_isomorphisms_iter())
  unique_keys = list(set([tuple(list(sorted(list(k.keys())))) for k in matchings_list]))
  matchings_list = [[d for d in matchings_list if k == tuple(list(sorted(list(d.keys()))))][0] for k in unique_keys]
  matchings_list2 = []
  for d in matchings_list:
    d_rev = dict((v, k) for k, v in d.items())
    middle_nodes = [d_rev[2], d_rev[4]]
    virtual_state = [nx.get_node_attributes(network, 'virtual')[j] for j in middle_nodes]
    if any([j == 1 for j in virtual_state]):
      d = dict((v, k) for k, v in d.items())
      matchings_list2.append(d)
  return matchings_list2

def trace_diamonds(network, species_list, filepath, libr = None,
                   file_suffix = '_graph_exhaustive.pkl'):
  """extracts diamond-shape motifs from biosynthetic networks (A->B,A->C,B->D,C->D) and uses evolutionary information to determine which path is taken from A to D\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | species_list (list): list of species to compare network to
  | filepath (string): filepath to load biosynthetic networks from other species; files need to be species name + file_suffix
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | file_suffix (string): generic end part of filename in filepath; default:'_graph_exhaustive.pkl'\n
  | Returns:
  | :-
  | Returns dataframe of each intermediary glycan and its proportion (0-1) of how often it has been experimentally observed in this path
  """
  if libr is None:
    libr = lib
  matchings_list = find_diamonds(network)
  paths = [choose_path(d[1], d[3], species_list, filepath, libr = libr,
                       file_suffix = file_suffix) for d in matchings_list]
  df_out = pd.DataFrame(paths).T.mean(axis = 1).reset_index()
  df_out.columns = ['target', 'probability']
  return df_out

def prune_network(network, node_attr = 'abundance', threshold = 0.):
  """pruning a network by dismissing unlikely virtual paths\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network such as from construct_network
  | node_attr (string): which (numerical) node attribute to use for pruning; default:'abundance'
  | threshold (float): everything below or equal to that threshold will be cut; default:0.\n
  | Returns:
  | :-
  | Returns pruned network
  """
  network_out = copy.deepcopy(network)
  #prune nodes lower in attribute than threshold
  to_cut = [k for k in list(network_out.nodes()) if nx.get_node_attributes(network_out, node_attr)[k] <= threshold]
  #leave virtual nodes that are needed to retain a connected component
  to_cut = [k for k in to_cut if not any([network_out.in_degree[j] == 1 and len(j) > len(k) for j in network_out.neighbors(k)])]
  network_out.remove_nodes_from(to_cut)
  nodeDict = dict(network_out.nodes(data = True))
  for node in list(network_out.nodes()):
    if (network_out.degree[node] <= 1) and (nodeDict[node]['virtual'] == 1):
      network_out.remove_node(node)
  return network_out
