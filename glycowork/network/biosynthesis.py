import re
import itertools
import mpld3
import pickle
import copy
import os
import pkg_resources
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glycowork.glycan_data.loader import lib, unwrap, linkages
from glycowork.motif.graph import compare_glycans, glycan_to_nxGraph, graph_to_string, subgraph_isomorphism
from glycowork.motif.processing import min_process_glycans, choose_correct_isoform
from glycowork.motif.tokenization import get_stem_lib

io = pkg_resources.resource_stream(__name__, "monolink_to_enzyme.csv")
df_enzyme = pd.read_csv(io, sep = '\t')

this_dir, this_filename = os.path.split(__file__) 
data_path = os.path.join(this_dir, 'milk_networks_exhaustive.pkl')
net_dic = pickle.load(open(data_path, 'rb'))

reducing_end = {'Glc-ol','GlcNAc-ol','Glc3S-ol','GlcNAc6S-ol', 'GlcNAc6P-ol',
                'GlcNAc1P-ol','Glc3P-ol', 'Glc6S-ol', 'GlcOS-ol'}
permitted_roots = {"Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"}
allowed_ptms = {'OS','3S','6S','1P','3P','6P','OAc','4Ac'}

def safe_compare(g1, g2, libr = None):
  """compare_glycans with try/except error catch\n
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
    return compare_glycans(g1, g2, libr = libr)
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

def safe_index(glycan, graph_dic, libr = None):
  """retrieves glycan graph and if not present in graph_dic it will freshly calculate\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns a glycan graph, either from graph_dic or freshly generated
  """
  if libr is None:
    libr = lib
  try:
    return graph_dic[glycan]
  except:
    return glycan_to_nxGraph(glycan, libr = libr)

def get_neighbors(ggraph, glycans, libr = None, graphs = None,
                  min_size = 1):
  """find (observed) biosynthetic precursors of a glycan\n
  | Arguments:
  | :-
  | ggraph (networkx): glycan graph as networkx object
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
  #get graph; graphs of single monosaccharide can't have precursors
  if len(ggraph.nodes()) <= 1:
    return ([], [])
  #get biosynthetic precursors
  ggraph_nb = create_neighbors(ggraph, libr = libr, min_size = min_size)
  temp = graphs if graphs is not None else [glycan_to_nxGraph(k, libr = libr) for k in glycans]
  #find out whether any member in 'glycans' is a biosynthetic precursor of 'glycan'
  idx = [np.where([safe_compare(k, j, libr = libr) for k in temp])[0].tolist() for j in ggraph_nb]
  nb = [glycans[k[0]] for k in idx if len(k) > 0]
  return nb, idx

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
  #check whether glycan large enough to allow for precursors
  if len(ggraph.nodes()) <= min_size:
    return []
  if len(ggraph.nodes()) == 3:
    ggraph_nb = [ggraph.subgraph([2])]
  #generate all precursors by iteratively cleaving off the non-reducing-end monosaccharides
  else:
    terminal_nodes = [k for k in ggraph.nodes() if ggraph.degree(k) == 1 and k != max(list(ggraph.nodes()))]
    terminal_pairs = [[k,next(ggraph.neighbors(k))] for k in terminal_nodes]
    ggraph_nb = [copy.deepcopy(ggraph) for k in range(len(terminal_pairs))]
    for k in range(len(terminal_pairs)):
      ggraph_nb[k].remove_nodes_from(terminal_pairs[k])
  #cleaving off messes with the node labeling, so they have to be re-labeled
  for k in range(len(ggraph_nb)):
    node_list = list(ggraph_nb[k].nodes())
    ggraph_nb[k] = nx.relabel_nodes(ggraph_nb[k], {node_list[m]:m for m in range(len(node_list))})
  return ggraph_nb

def get_virtual_nodes(glycan, graph_dic, libr = None, reducing_end = reducing_end):
  """find unobserved biosynthetic precursors of a glycan\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (set): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends\n
  | Returns:
  | :-
  | (1) list of virtual node graphs
  | (2) list of virtual nodes in IUPAC-condensed format
  """
  if libr is None:
    libr = lib
  #get glycan graph
  ggraph = safe_index(glycan, graph_dic, libr = libr)
  if len(ggraph.nodes()) == 1:
    return ([], [])
  #get biosynthetic precursors
  ggraph_nb = create_neighbors(ggraph, libr = libr)
  ggraph_nb_t = [graph_to_string(k) for k in ggraph_nb]
  ggraph_nb_t = [k if k[0] != '[' else k.replace('[','',1).replace(']','',1) for k in ggraph_nb_t]

  #filter out reducing end precursors
  ggraph_nb_t = [k for k in ggraph_nb_t if any([k[-len(j):] == j for j in reducing_end])]
  ggraph_nb_t2 = []
  ggraph_nb = []
  #get both string and graph versions of the precursors
  for k in range(len(ggraph_nb_t)):
    try:
      ggraph_nb.append(safe_index(ggraph_nb_t[k], graph_dic, libr = libr))
      ggraph_nb_t2.append(ggraph_nb_t[k])
    except:
      pass
  #get the difference of glycan & precursor as a string
  idx = [k for k in range(len(ggraph_nb_t2)) if ggraph_nb_t2[k][0] != '(']
  return [ggraph_nb[i] for i in idx], [ggraph_nb_t2[i] for i in idx]

def find_diff(glycan_a, glycan_b, graph_dic, libr = None):
  """finds the subgraph that differs between glycans and returns it, will only work if the differing subgraph is connected\n
  | Arguments:
  | :-
  | glycan_a (networkx): glycan graph as networkx object
  | glycan_b (networkx): glycan graph as networkx object
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns difference between glycan_a and glycan_b in IUPAC-condensed
  """
  if libr is None:
    libr = lib
  #check whether the glycans are equal
  if glycan_a == glycan_b:
    return ""
  #convert to graphs and get sorted node labels
  glycan_a2 = list(sorted(list(nx.get_node_attributes(safe_index(glycan_a, graph_dic, libr = libr), 'string_labels').values())))
  glycan_b2 = list(sorted(list(nx.get_node_attributes(safe_index(glycan_b, graph_dic, libr = libr), 'string_labels').values())))
  graphs = [glycan_a2, glycan_b2]
  glycans = [glycan_a, glycan_b]
  larger_graph = max(graphs, key = len)
  smaller_graph = min(graphs, key = len)
  #iteratively subtract the smaller graph from the larger graph --> what remains is the difference
  for k in smaller_graph:
    try:
      larger_graph.remove(k)
    except:
      return 'disregard'
  if len(larger_graph) > 2:
    return 'disregard'
  #final check of whether the smaller glycan is a subgraph of the larger glycan
  if subgraph_isomorphism(max(glycans, key = len), min(glycans, key = len), libr = libr):
    linkage = [k for k in larger_graph if k in linkages][0]
    sugar = [k for k in larger_graph if k not in linkages][0]
    diff = (sugar + "(" + linkage + ")")
    if 'S(' in diff or 'P(' in diff:
      return 'disregard'
    elif diff is None:
      return 'disregard'
    else:
      return diff
  else:
    return 'disregard'

def find_shared_virtuals(glycan_a, glycan_b, graph_dic, libr = None, reducing_end = reducing_end,
                         min_size = 1):
  """finds virtual nodes that are shared between two glycans (i.e., that connect these two glycans)\n
  | Arguments:
  | :-
  | glycan_a (string): glycan in IUPAC-condensed format
  | glycan_b (string): glycan in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (set): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | min_size (int): length of smallest root in biosynthetic network; default:1\n
  | Returns:
  | :-
  | Returns list of edges between glycan and virtual node (if virtual node connects the two glycans)
  """
  if libr is None:
    libr = lib
  #get virtual nodes of both glycans
  ggraph_nb_a, glycans_a = get_virtual_nodes(glycan_a, graph_dic, libr = libr, reducing_end = reducing_end)                                
  ggraph_nb_b, glycans_b = get_virtual_nodes(glycan_b, graph_dic, libr = libr, reducing_end = reducing_end)
  out = []
  #check whether any of the nodes of glycan_a and glycan_b are the same
  if len(ggraph_nb_a) > 0:
    for k in range(len(ggraph_nb_a)):
      for j in range(len(ggraph_nb_b)):
        if len(ggraph_nb_a[k]) >= min_size:
          if compare_glycans(ggraph_nb_a[k], ggraph_nb_b[j], libr = libr):
            if [glycan_a, glycans_a[k]] not in [list(m) for m in out]:
              out.append((glycan_a, glycans_a[k]))
            if [glycan_b, glycans_b[j]] not in [list(m) for m in out]:
              out.append((glycan_b, glycans_a[k]))
  return out

def fill_with_virtuals(glycans, graph_dic, libr = None, reducing_end = reducing_end,
                       min_size = 1):
  """for a list of glycans, identify virtual nodes connecting observed glycans and return their edges\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (set): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | min_size (int): length of smallest root in biosynthetic network; default:1\n
  | Returns:
  | :-
  | Returns list of edges that connect observed glycans to virtual nodes
  """
  if libr is None:
    libr = lib
  #for each combination of glycans, check whether they have shared virtual nodes
  v_edges = [find_shared_virtuals(k[0], k[1], graph_dic, libr = libr, reducing_end = reducing_end,
                                  min_size = min_size) for k in list(itertools.combinations(glycans, 2))]
  v_edges = unwrap(v_edges)
  return v_edges

def create_adjacency_matrix(glycans, graph_dic, libr = None, virtual_nodes = False,
                            reducing_end = reducing_end, min_size = 1):
  """creates a biosynthetic adjacency matrix from a list of glycans\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | virtual_nodes (bool): whether to include virtual nodes in network; default:False
  | reducing_end (set): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | min_size (int): length of smallest root in biosynthetic network; default:1\n
  | Returns:
  | :-
  | (1) adjacency matrix (glycan X glycan) denoting whether two glycans are connected by one biosynthetic step
  | (2) list of which nodes are virtual nodes (empty list if virtual_nodes is False)
  """
  if libr is None:
    libr = lib
  #connect glycans with biosynthetic precursors
  df_out = pd.DataFrame(0, index = glycans, columns = glycans)
  graphs = [safe_index(k, graph_dic, libr = libr) for k in glycans]
  neighbors, idx = zip(*[get_neighbors(safe_index(k, graph_dic, libr = libr), glycans, libr = libr, graphs = graphs,
                                  min_size = min_size) for k in glycans])
  #fill adjacency matrix
  for j in range(len(glycans)):
    if len(idx[j]) > 0:
      for i in range(len(idx[j])):
        if len(idx[j][i]) >= 1:
          inp = [idx[j][i], glycans.index(glycans[j])]
          df_out.iloc[inp[0], inp[1]] = 1
  #find connections between virtual nodes that connect observed nodes
  new_nodes = []
  if virtual_nodes:
    virtual_edges = fill_with_virtuals(glycans, graph_dic, libr = libr, reducing_end = reducing_end,
                                       min_size = min_size)
    new_nodes = list(set([k[1] for k in virtual_edges]))
    new_nodes = [k for k in new_nodes if k not in df_out.columns.values.tolist()]
    new_nodes_g = [glycan_to_nxGraph(k, libr = libr) for k in new_nodes]
    idx = np.where([any([compare_glycans(k, safe_index(j, graph_dic, libr = libr), libr = libr) for j in df_out.columns.values.tolist()]) for k in new_nodes_g])[0].tolist()
    if len(idx) > 0:
      virtual_edges = [j for j in virtual_edges if j[1] not in [new_nodes[k] for k in idx]]
      new_nodes = [new_nodes[k] for k in range(len(new_nodes)) if k not in idx]
    #add virtual nodes
    for k in new_nodes:
      df_out[k] = 0
      df_out.loc[len(df_out)] = 0
      df_out.index = df_out.index.values.tolist()[:-1] + [k]
    #add virtual edges
    for k in virtual_edges:
      df_out.loc[k[0], k[1]] = 1
  return df_out.add(df_out.T, fill_value = 0), new_nodes

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

def propagate_virtuals(glycans, graph_dic, libr = None, reducing_end = reducing_end,
                       permitted_roots = permitted_roots):
  """do one step of virtual node generation\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (set): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | permitted_roots (set): which nodes should be considered as roots; default:["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]\n
  | Returns:
  | :-
  | (1) list of virtual node graphs
  | (2) list of virtual nodes in IUPAC-condensed format
  """
  if libr is None:
    libr = lib
  #while assuring that glycans don't become smaller than the root, find biosynthetic precursors of each glycan in glycans
  linkage_count = min([k.count('(') for k in permitted_roots])
  virtuals = [get_virtual_nodes(k, graph_dic, libr = libr, reducing_end = reducing_end) for k in glycans if k.count('(') > linkage_count]
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
  #find connections/edges between virtual nodes from propagate_virtuals run N and propagate_virtuals run N+1
  for m in range(len(prev_shell)):
    edges_out.append(unwrap([[(prev_shell[m][k], next_shell[k][j]) for j in range(len(next_shell[k]))] for k in range(len(prev_shell[m]))]))
  return edges_out


def find_path(glycan_a, glycan_b, graph_dic, libr = None, reducing_end = reducing_end,
              limit = 5, permitted_roots = permitted_roots):
  """find virtual node path between two glycans\n
  | Arguments:
  | :-
  | glycan_a (string): glycan in IUPAC-condensed format
  | glycan_b (string): glycan in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (set): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | limit (int): maximum number of virtual nodes between observed nodes; default:5
  | permitted_roots (set): which nodes should be considered as roots; default:["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]\n
  | Returns:
  | :-
  | (1) list of edges to connect glycan_a and glycan_b via virtual nodes
  | (2) dictionary of edge labels detailing difference between two connected nodes
  """
  if libr is None:
    libr = lib
  virtual_shells_t = []
  true_nodes = [glycan_a, glycan_b]
  smaller_glycan = safe_index(true_nodes[np.argmin([len(j) for j in true_nodes])], graph_dic, libr = libr)
  larger_glycan = true_nodes[np.argmax([len(j) for j in true_nodes])]
  #start with the larger glycan and do the first round of finding biosynthetic precursors
  virtual_shells = [safe_index(larger_glycan, graph_dic, libr = libr)]
  virtual_shells_t = [[[larger_glycan]]]
  virtuals, virtuals_t = propagate_virtuals([larger_glycan], graph_dic, libr = libr, reducing_end = reducing_end,
                                            permitted_roots = permitted_roots)
  virtual_shells.append(virtuals)
  virtual_shells_t.append(virtuals_t)
  county = 0
  #for as long as no *observed* biosynthetic precursor is spotted (and the limit hasn't been reached), continue generating biosynthetic precursors
  while ((not any([compare_glycans(smaller_glycan, k) for k in unwrap(virtuals)])) and (county < limit)):
    virtuals, virtuals_t = propagate_virtuals(unwrap(virtuals_t), graph_dic, libr = libr, reducing_end = reducing_end,
                                              permitted_roots = permitted_roots)
    virtual_shells.append(virtuals)
    virtual_shells_t.append(virtuals_t)
    county += 1
  #get the edges connecting the starting point to the experimentally observed node
  virtual_edges = unwrap([unwrap(shells_to_edges(virtual_shells_t[k], virtual_shells_t[k+1])) for k in range(len(virtual_shells_t)-1)])
  edge_labels = {}
  #find the edge labels / differences between precursors
  for el in virtual_edges:
    edge_labels[el] = find_diff(el[0], el[1], graph_dic, libr = libr)
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

def find_shortest_path(goal_glycan, glycan_list, graph_dic, libr = None, reducing_end = reducing_end,
                       limit = 5, permitted_roots = permitted_roots):
  """finds the glycan with the shortest path via virtual nodes to the goal glycan\n
  | Arguments:
  | :-
  | goal_glycan (string): glycan in IUPAC-condensed format
  | glycan_list (list): list of glycans in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (set): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | limit (int): maximum number of virtual nodes between observed nodes; default:5
  | permitted_roots (set): which nodes should be considered as roots; default:["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]\n
  | Returns:
  | :-
  | (1) list of edges of shortest path to connect goal_glycan and glycan via virtual nodes
  | (2) dictionary of edge labels detailing difference between two connected nodes in shortest path
  """
  if libr is None:
    libr = lib
  path_lengths = []
  for k in glycan_list:
    #for each glycan, check whether it could constitute a precursor (i.e., is it a sub-graph + does it stem from the correct root)
    if len(k) < len(goal_glycan) and goal_glycan[-6:] == k[-6:]:
      if subgraph_isomorphism(goal_glycan, k, libr = libr):
        try:
          #finding a path through shells of generated virtual nodes
          virtual_edges, edge_labels = find_path(goal_glycan, k, graph_dic, libr = libr,
                                                                reducing_end = reducing_end,
                                                                limit = limit, permitted_roots = permitted_roots)
          network = make_network_from_edges(virtual_edges)
          #if there are potential paths, get the shortest path(s)
          if k in network.nodes() and goal_glycan in network.nodes():
            path_lengths.append(len(nx.algorithms.shortest_paths.generic.shortest_path(network, source = k, target = goal_glycan)))
          else:
            path_lengths.append(99)
        except:
          path_lengths.append(99)
      else:
        path_lengths.append(99)
  idx = np.argmin(path_lengths)
  #construct the shortest path to add it to the network
  virtual_edges, edge_labels = find_path(goal_glycan, glycan_list[idx], graph_dic, libr = libr,
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
  connected_nodes = network.edges()
  connected_nodes = list(sum(connected_nodes, ()))
  #find nodes that have no edges
  unconnected = [k for k in glycan_list if k not in connected_nodes]
  return unconnected

def filter_disregard(network, attr = 'diffs', value = 'disregard'):
  """filters out mistaken edges\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network from construct_network
  | attr (string): edge attribute name; default:'diffs'
  | value (string): which value of attr to filter out; default:'disregard'\n
  | Returns:
  | :-
  | Returns network without mistaken edges
  """
  edges_to_remove = [(u, v) for u, v, data in network.edges(data = True) if data[attr] == value]
  network.remove_edges_from(edges_to_remove)
  return network

def detect_ptm(glycans, allowed_ptms = allowed_ptms):
  """identifies glycans that contain post-translational modifications\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | allowed_ptms (set): list of PTMs to consider\n
  | Returns:
  | :-
  | Returns glycans that contain PTMs
  """
  out = []
  #checks for presence of any of the potential PTMs
  for glycan in glycans:
    for ptm in allowed_ptms:
        if ptm in glycan:
            out.append(glycan)
            break
  return out

def stemify_glycan_fast(ggraph_in, stem_lib = None, libr = None):
  """stemifies a glycan graph\n
  | Arguments:
  | :-
  | ggraph_in (networkx): glycan graph as a networkx object
  | stem_lib (dictionary): dictionary of form modified_monosaccharide:core_monosaccharide; default:created from lib
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset; default:lib\n
  | Returns:
  | :-
  | (1) stemified glycan in IUPAC-condensed as a string
  | (2) stemified glycan graph as a networkx object
  """
  if libr is None:
    libr = lib
  if stem_lib is None:
    stem_lib = get_stem_lib(libr)
  ggraph = copy.deepcopy(ggraph_in)
  for k,v in nx.get_node_attributes(ggraph, "string_labels").items():
    ggraph.nodes[k]["string_labels"] = stem_lib[v]
    ggraph.nodes[k]["labels"] = libr.index(stem_lib[v])
  return graph_to_string(ggraph, libr = libr), ggraph

def find_ptm(glycan, glycans, graph_dic, allowed_ptms = allowed_ptms,
               libr = None, ggraphs = None, suffix = '-ol', stem_lib = None):
  """identifies precursor glycans for a glycan with a PTM\n
  | Arguments:
  | :-
  | glycan (string): glycan with PTM in IUPAC-condensed format
  | glycans (list): list of glycans in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | allowed_ptms (set): list of PTMs to consider
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | ggraphs (list): list of precomputed graphs of the glycan list
  | suffix (string): optional suffix to be added to the stemified glycan; default:'-ol'
  | stem_lib (dictionary): dictionary of form modified_monosaccharide:core_monosaccharide; default:created from lib\n
  | Returns:
  | :-
  | (1) an edge tuple between input glycan and its biosynthetic precusor without PTM
  | (2) the PTM that is contained in the input glycan
  """
  if libr is None:
    libr = lib
  if stem_lib is None:
    stem_lib = get_stem_lib(libr)
  #checks which PTM(s) are present
  mod = [ptm for ptm in allowed_ptms if ptm in glycan][0]
  #stemifying returns the unmodified glycan
  glycan_stem, g_stem = stemify_glycan_fast(safe_index(glycan, graph_dic, libr = libr),
                                            libr = libr, stem_lib = stem_lib)
  glycan_stem = glycan_stem + suffix
  if suffix == '-ol':
    g_stem.nodes[len(g_stem)-1]['string_labels'] = g_stem.nodes[len(g_stem)-1]['string_labels'] + suffix
    g_stem.nodes[len(g_stem)-1]['labels'] = libr.index(g_stem.nodes[len(g_stem)-1]['string_labels'])
  if ('Sug' in glycan_stem) or ('Neu(' in glycan_stem):
    return 0
  if ggraphs is None:
    ggraphs = [safe_index(k, graph_dic, libr = libr) for k in glycans]
  idx = np.where([safe_compare(k, g_stem, libr = libr) for k in ggraphs])[0].tolist()
  #if found, add a biosynthetic step of adding the PTM to the unmodified glycan
  if len(idx) > 0:
    nb = [glycans[k] for k in idx][0]
    return ((glycan), (nb)), mod
  #if not, just add a virtual node of the unmodified glycan+corresponding edge; will be connected to the main network later; will not properly work if glycan has multiple PTMs
  else:
    if ''.join(sorted(glycan)) == ''.join(sorted(glycan_stem + mod)):
      return ((glycan), (glycan_stem)), mod
    else:
      return 0

def process_ptm(glycans, graph_dic, allowed_ptms = allowed_ptms,
               libr = None, suffix = '-ol'):
  """identifies glycans that contain post-translational modifications and their biosynthetic precursor\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | allowed_ptms (set): list of PTMs to consider
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | suffix (string): optional suffix to be added to the stemified glycan; default:'-ol'\n
  | Returns:
  | :-
  | (1) list of edge tuples between input glycans and their biosynthetic precusor without PTM
  | (2) list of the PTMs that are contained in the input glycans
  """
  if libr is None:
    libr = lib
  stem_lib = get_stem_lib(libr)
  #get glycans with PTMs and convert them to graphs
  ptm_glycans = detect_ptm(glycans, allowed_ptms = allowed_ptms)
  ggraphs = [safe_index(k, graph_dic, libr = libr) for k in glycans]
  #connect modified glycans to their unmodified counterparts
  edges = [find_ptm(k, glycans, graph_dic, allowed_ptms = allowed_ptms,
                    libr = libr, ggraphs = ggraphs, suffix = suffix,
                    stem_lib = stem_lib) for k in ptm_glycans]
  if len([k for k in edges if k != 0]) > 0:
    edges, edge_labels = list(zip(*[k for k in edges if k != 0]))
    return list(edges), list(edge_labels)
  else:
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
    for node in network.nodes:
      try:
        temp = network.nodes[node]['virtual']
      except:
        network.nodes[node]['virtual'] = 1
    return network
  else:
    nx.set_node_attributes(network, node_labels, 'virtual')
    return network

def return_unconnected_to_root(network, permitted_roots = permitted_roots):
  """finds observed nodes that are not connected to the root nodes\n
  | Arguments:
  | :-
  | network (networkx object): network that should be analyzed
  | permitted_roots (set): which nodes should be considered as roots; default:{"Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"}\n
  | Returns:
  | :-
  | Returns list of nodes that are not connected to the root nodes
  """
  unconnected = []
  nodeDict = dict(network.nodes(data = True))
  #for each observed node, check whether it has a path to a root; if not, collect and return the node
  for node in network.nodes():
    if nodeDict[node]['virtual'] == 0 and (node not in permitted_roots):
      path_tracing = [nx.algorithms.shortest_paths.generic.has_path(network, node, k) for k in permitted_roots if k in network.nodes()]
      if not any(path_tracing):
        unconnected.append(node)
  return unconnected

def deorphanize_nodes(network, graph_dic, reducing_end = reducing_end,
                      permitted_roots = permitted_roots, libr = None, limit = 5):
  """finds nodes unconnected to root nodes and tries to connect them\n
  | Arguments:
  | :-
  | network (networkx object): network that should be modified
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | reducing_end (set): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | permitted_roots (set): which nodes should be considered as roots; default:["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | limit (int): maximum number of virtual nodes between observed nodes; default:5\n
  | Returns:
  | :-
  | Returns network with deorphanized nodes (if they can be connected to root nodes)
  """
  if libr is None:
    libr = lib
  min_size = min([len(k) for k in min_process_glycans(permitted_roots)])
  permitted_roots = [k for k in permitted_roots if k in network.nodes()]
  if len(permitted_roots) > 0:
    #get unconnected observed nodes
    unconnected_nodes = return_unconnected_to_root(network, permitted_roots = permitted_roots)
    #directed needed to detect nodes that appear to be connected (downstream) but are not connected to the root
    temp_network = make_network_directed(network)
    pseudo_connected_nodes = [k[0] for k in temp_network.in_degree if k[1] == 0]
    pseudo_connected_nodes = [k for k in pseudo_connected_nodes if k not in permitted_roots and k not in unconnected_nodes]
    unconnected_nodes = unconnected_nodes + pseudo_connected_nodes
    #only consider unconnected nodes if they are larger than the root (otherwise they should be pruned anyway)
    unconnected_nodes = [k for k in unconnected_nodes if len(safe_index(k, graph_dic, libr = libr)) > min_size]
    nodeDict = dict(network.nodes(data = True))
    #subset the observed nodes that are connected to the root
    real_nodes = [node for node in list(network.nodes()) if nodeDict[node]['virtual'] == 0 and node not in unconnected_nodes]
    #real_nodes = [node for node in real_nodes if node not in unconnected_nodes]
    edges = []
    edge_labels = []
    #for each unconnected node, find the shortest path to its root node
    for node in unconnected_nodes:
      p_root = [subgraph_isomorphism(safe_index(node, graph_dic, libr = libr), p, libr = libr) for p in permitted_roots]
      if sum(p_root) > 0:
        p_root = permitted_roots[np.where(p_root)[0][0]]
        e, el = find_path(node, p_root, graph_dic, reducing_end = reducing_end,
                                                        libr = libr, limit = 2*limit,
                                    permitted_roots = permitted_roots)
        edges.append(e)
        edge_labels.append(el)
    edge_labels = unwrap([[edge_labels[k][edges[k][j]] for j in range(len(edges[k]))] for k in range(len(edge_labels))])
    edges = unwrap(edges)
    node_labels = {node:(nodeDict[node]['virtual'] if node in network.nodes() else 1) for node in [i for sub in edges for i in sub]}
    network_out = update_network(network, edges, edge_labels = edge_labels, node_labels = node_labels)
    network_out = filter_disregard(network_out)
    return network_out
  else:
    return network

def prune_directed_edges(network):
  """removes edges that go against the direction of biosynthesis\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network with directed edges\n
  | Returns:
  | :-
  | Returns a network with pruned edges that would go in the wrong direction
  """
  nodes = network.nodes()
  for k in nodes:
    for j in nodes:
      if network.has_edge(k,j) and (len(k) > len(j)):
        network.remove_edge(k,j)
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

def deorphanize_edge_labels(network, graph_dic, libr = None):
  """completes edge labels in newly added edges of a biosynthetic network\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns a network with completed edge labels
  """
  if libr is None:
    libr = lib
  edge_labels = nx.get_edge_attributes(network, 'diffs')
  for k in network.edges():
    if k not in edge_labels.keys():
      diff = find_diff(k[0], k[1], graph_dic, libr = libr)
      network.edges[k]['diffs'] = diff
  return network

def construct_network(glycans, add_virtual_nodes = 'exhaustive', libr = None, reducing_end = reducing_end,
                 limit = 5, ptm = True, allowed_ptms = allowed_ptms,
                 permitted_roots = permitted_roots, directed = True, edge_type = 'monolink'):
  """construct a glycan biosynthetic network\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | add_virtual_nodes (string): indicates whether no ('none'), proximal ('simple'), or all ('exhaustive') virtual nodes should be added; default:'exhaustive'
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | reducing_end (set): monosaccharides at the reducing end that are allowed; default:milk glycan reducing ends
  | limit (int): maximum number of virtual nodes between observed nodes; default:5
  | ptm (bool): whether to consider post-translational modifications in the network construction; default:True
  | allowed_ptms (set): list of PTMs to consider
  | permitted_roots (set): which nodes should be considered as roots; default:{"Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"}
  | directed (bool): whether to return a network with directed edges in the direction of biosynthesis; default:True
  | edge_type (string): indicates whether edges represent monosaccharides ('monosaccharide'), monosaccharide(linkage) ('monolink'), or enzyme catalyzing the reaction ('enzyme'); default:'monolink'\n
  | Returns:
  | :-
  | Returns a networkx object of the network
  """
  if libr is None:
    libr = lib
  if add_virtual_nodes in {'simple', 'exhaustive'}:
    virtuals = True
  else:
    virtuals = False
  #generating graph from adjacency of observed glycans
  min_size = min([len(k) for k in min_process_glycans(permitted_roots)])
  add_to_virtuals = []
  for r in permitted_roots:
      if r not in glycans and any([r in g for g in glycans]):
        add_to_virtuals.append(r)
        glycans.append(r)
  graph_dic = {k:glycan_to_nxGraph(k, libr = libr) for k in glycans}
  adjacency_matrix, virtual_nodes = create_adjacency_matrix(glycans, graph_dic, libr = libr, virtual_nodes = virtuals,
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
        virtual_edges, edge_labels = find_shortest_path(k, [j for j in network.nodes() if j != k], graph_dic, libr = libr,
                                                      reducing_end = reducing_end, limit = limit, permitted_roots = permitted_roots)
        total_nodes = list(set(list(sum(virtual_edges, ()))))
        new_nodes.append([j for j in total_nodes if j not in network.nodes()])
        new_edges.append(virtual_edges)
        new_edge_labels.append(edge_labels)
      except:
        pass
    network.add_edges_from(unwrap(new_edges), edge_labels = unwrap(new_edge_labels))
    virtual_nodes = virtual_nodes + add_to_virtuals + list(set(unwrap(new_nodes)))
    for node in virtual_nodes:
      graph_dic[node] = glycan_to_nxGraph(node, libr = libr)
    for ed in network.edges():
      if ed[0] in virtual_nodes and ed[1] in virtual_nodes:
        larger_ed = np.argmax([len(e) for e in [ed[0], ed[1]]])
        if network.degree(ed[larger_ed]) == 1:
          network.remove_edge(ed[0], ed[1])
  #create edge and node labels
  edge_labels = {}
  for el in network.edges():
    edge_labels[el] = find_diff(el[0], el[1], graph_dic, libr = libr)
  nx.set_edge_attributes(network, edge_labels, 'diffs')
  virtual_labels = {k:(1 if k in virtual_nodes else 0) for k in network.nodes()}
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
    ptm_links = process_ptm(glycans, graph_dic, allowed_ptms = allowed_ptms, libr = libr, suffix = suffix)
    if len(ptm_links) > 1:
      network = update_network(network, ptm_links[0], edge_labels = ptm_links[1])
  #find any remaining orphan nodes and connect them to the root(s)
  if add_virtual_nodes == 'simple':
    network = deorphanize_nodes(network, graph_dic, reducing_end = reducing_end,
                              permitted_roots = permitted_roots, libr = libr, limit = 1)
  elif add_virtual_nodes == 'exhaustive':
    network = deorphanize_nodes(network, graph_dic, reducing_end = reducing_end,
                              permitted_roots = permitted_roots, libr = libr, limit = limit)
  #final clean-up / condensation step
  if virtuals:
    nodeDict = dict(network.nodes(data = True))
    for node in list(sorted(network.nodes(), key = len, reverse = True)):
      if (network.degree[node] <= 1) and (nodeDict[node]['virtual'] == 1) and (node not in permitted_roots):
        network.remove_node(node)
    for node in list(network.nodes()):
      if node not in graph_dic.keys():
        graph_dic[node] = glycan_to_nxGraph(node, libr = libr)
    adj_matrix = create_adjacency_matrix(list(network.nodes()), graph_dic, libr = libr,
                                         reducing_end = reducing_end,
                                         min_size = min_size)
    filler_network = adjacencyMatrix_to_network(adj_matrix[0])
    network.add_edges_from(list(filler_network.edges()))
    network = deorphanize_edge_labels(network, graph_dic, libr = libr)
  #edge label specification
  if edge_type != 'monolink':
    for e in network.edges():
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
    real_nodes = [safe_index(x, graph_dic, libr = libr) for x,y in network.nodes(data = True) if y['virtual'] == 0]
    to_cut = [v for v in virtual_nodes if any([compare_glycans(safe_index(v, graph_dic, libr = libr), r, libr = libr) for r in real_nodes])]
    network.remove_nodes_from(to_cut)
    virtual_nodes = [x for x,y in network.nodes(data = True) if y['virtual'] == 1]
    virtual_graphs = [safe_index(x, graph_dic, libr = libr) for x in virtual_nodes]
    isomeric_graphs = [k for k in list(itertools.combinations(virtual_graphs, 2)) if compare_glycans(k[0], k[1], libr = libr)]
    if len(isomeric_graphs) > 0:
      isomeric_nodes = [[virtual_nodes[virtual_graphs.index(k[0])],
                         virtual_nodes[virtual_graphs.index(k[1])]] for k in isomeric_graphs]
      to_cut = [choose_correct_isoform(k, reverse = True)[0] for k in isomeric_nodes]
      network.remove_nodes_from(to_cut)
  network.remove_edges_from(nx.selfloop_edges(network))
  #directed or undirected network
  if directed:
    network = make_network_directed(network)
    if virtuals:
      nodeDict = dict(network.nodes(data = True))
      for node in list(sorted(network.nodes(), key = len, reverse = False)):
        if (network.in_degree[node] < 1) and (nodeDict[node]['virtual'] == 1) and (node not in permitted_roots):
          network.remove_node(node)
      for node in list(sorted(network.nodes(), key = len, reverse = True)):
        if (network.out_degree[node] < 1) and (nodeDict[node]['virtual'] == 1):
          network.remove_node(node)
  return filter_disregard(network)

def plot_network(network, plot_format = 'pydot2', edge_label_draw = True,
                 lfc_dict = None):
  """visualizes biosynthetic network\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | plot_format (string): how to layout network, either 'pydot2', 'kamada_kawai', or 'spring'; default:'pydot2'
  | edge_label_draw (bool): draws edge labels if True; default:True
  | lfc_dict (dict): dictionary of enzyme:log2-fold-change to scale edge width; default:None\n
  """
  try:
    mpld3.enable_notebook()
  except:
    print("Not in a notebook environment --> mpld3 doesn't work, so we can't do fancy plotting. Reverting to just plotting the network. Recommended to run this in a notebook.")
    nx.draw(network)
    plt.show()
  fig, ax = plt.subplots(figsize = (16,16))
  #check whether are values to scale node size by
  if len(list(nx.get_node_attributes(network, 'abundance').values())) > 0:
    node_sizes = list(nx.get_node_attributes(network, 'abundance').values())
  else:
    node_sizes = 50
  #decide on network layout
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
  #check whether there are values to scale node color by
  if len(list(nx.get_node_attributes(network, 'origin').values())) > 0:
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
      #map log-fold changes of responsible enzymes onto biosynthetic steps
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
    #alternatively, color edge by type of monosaccharide that is added
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
  #get combined nodes and whether they occur in either network
  all_nodes = list(network_a.nodes(data = True)) + list(network_b.nodes(data = True))
  network_a_nodes = [k[0] for k in network_a.nodes(data = True)]
  network_b_nodes =  [k[0] for k in network_b.nodes(data = True)]
  node_origin = ['cornflowerblue' if (k[0] in network_a_nodes and k[0] not in network_b_nodes) \
            else 'darkorange' if (k[0] in network_b_nodes and k[0] not in network_a_nodes) else 'saddlebrown' for k in all_nodes]
  U.add_nodes_from(all_nodes)
  #get combined edges
  U.add_edges_from(list(network_a.edges(data = True)) + list(network_b.edges(data = True)))
  nx.set_node_attributes(U, {all_nodes[k][0]:node_origin[k] for k in range(len(all_nodes))}, name = 'origin')
  #if input is directed, make output directed
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
  #perform network alignment if not provided
  if combined is None:
    combined = network_alignment(network_a, network_b)
  a_nodes = network_a.nodes()
  b_nodes = network_b.nodes()
  #find virtual nodes in network_a that are observed in network_b and vice versa
  supported_a = [k for k in a_nodes if nx.get_node_attributes(network_a, 'virtual')[k] == 1 and k in b_nodes]
  supported_a = [k for k in supported_a if nx.get_node_attributes(network_b, 'virtual')[k] == 1]
  supported_b = [k for k in b_nodes if nx.get_node_attributes(network_b, 'virtual')[k] == 1 and k in a_nodes]
  supported_b = [k for k in supported_b if nx.get_node_attributes(network_a, 'virtual')[k] == 1]
  inferred_a = [k for k in a_nodes if nx.get_node_attributes(network_a, 'virtual')[k] == 1 and nx.get_node_attributes(combined, 'virtual')[k] == 0]
  inferred_b = [k for k in b_nodes if nx.get_node_attributes(network_b, 'virtual')[k] == 1 and nx.get_node_attributes(combined, 'virtual')[k] == 0]
  return (inferred_a, supported_a), (inferred_b, supported_b)

def infer_network(network, network_species, species_list, network_dic):
  """replaces virtual nodes if they are observed in other species\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network that should be inferred
  | network_species (string): species from which the network stems
  | species_list (list): list of species to compare network to
  | network_dic (dict): dictionary of form species name : biosynthetic network (gained from construct_network)\n
  | Returns:
  | :-
  | Returns network with filled in virtual nodes
  """
  inferences = []
  #for each species, try to identify observed nodes matching virtual nodes in input network
  for k in species_list:
    if k != network_species:
      temp_network = network_dic[k]
      #get virtual nodes observed in species k
      infer_network, infer_other = infer_virtual_nodes(network, temp_network)
      inferences.append(infer_network)
  #output a new network with formerly virtual nodes updated to a new node status --> inferred
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
  inferred_nodes = [k for k in inferred_nodes.keys() if inferred_nodes[k] == 2]
  if species is None:
    return inferred_nodes
  else:
    return {species:inferred_nodes}

def export_network(network, filepath, other_node_attributes = []):
  """converts NetworkX network into files usable, e.g., by Cytoscape or Gephi\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | filepath (string): should describe a valid path + file name prefix, will be appended by file description and type
  | other_node_attributes (list): string names of node attributes that should also be extracted; default:[]\n
  | Returns:
  | :-
  | (1) saves a .csv dataframe containing the edge list and edge labels
  | (2) saves a .csv dataframe containing node IDs and labels
  """
  #generate edge_list
  edge_list = network.edges()
  edge_list = pd.DataFrame(edge_list)
  if len(edge_list) > 0:
    edge_list.columns = ['Source', 'Target']
    edge_labels = nx.get_edge_attributes(network, 'diffs')
    edge_list['Label'] = [edge_labels[k] for k in network.edges()]
    edge_list.to_csv(filepath + '_edge_list.csv', index = False)
  else:
    edge_list = pd.DataFrame()
    edge_list = edge_list.reindex(columns = ['Source', 'Target'])
    edge_list.to_csv(filepath + '_edge_list.csv', index = False)

  #generate node_labels
  node_labels = nx.get_node_attributes(network, 'virtual')
  node_labels = pd.DataFrame(node_labels, index = [0]).T.reset_index()
  if len(other_node_attributes) > 0:
    other_node_labels = []
    for att in other_node_attributes:
      lab = nx.get_node_attributes(network, att)
      other_node_labels.append(pd.DataFrame(lab, index = [0]).T.reset_index().iloc[:,1])
    node_labels = pd.concat([node_labels] + other_node_labels, axis = 1)
  node_labels.columns = ['Id', 'Virtual'] + other_node_attributes
  node_labels.to_csv(filepath + '_node_labels.csv', index = False)

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

def choose_path(diamond, species_list, network_dic, libr = None, threshold = 0.,
                nb_intermediates = 2):
  """given a shape such as diamonds in biosynthetic networks (A->B,A->C,B->D,C->D), which path is taken from A to D?\n
  | Arguments:
  | :-
  | diamond (dict): dictionary of form position-in-diamond : glycan-string, describing the diamond
  | species_list (list): list of species to compare network to
  | network_dic (dict): dictionary of form species name : biosynthetic network (gained from construct_network)
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | threshold (float): everything below or equal to that threshold will be cut; default:0.
  | nb_intermediates (int): number of intermediate nodes expected in a network motif to extract; has to be a multiple of 2 (2: diamond, 4: hexagon,...)\n
  | Returns:
  | :-
  | Returns dictionary of each intermediary path and its experimental support (0-1) across species
  """
  if libr is None:
    libr = lib
  graph_dic = {k:glycan_to_nxGraph(k, libr = libr) for k in diamond.values()}
  #construct the diamond between source and target node
  adj, _ = create_adjacency_matrix(list(diamond.values()), graph_dic, libr = libr)
  network = adjacencyMatrix_to_network(adj)
  #get the intermediate nodes constituting the alternative paths
  alternatives = [path[1:int(((nb_intermediates/2)+1))] for path in nx.all_simple_paths(network, source = diamond[1], target = diamond[1+int(((nb_intermediates/2)+1))])]
  if len(alternatives) < 2:
    return {}
  alts = [[] for _ in range(len(alternatives))]
  #for each species, check whether an alternative has been observed
  for k in species_list:
    temp_network = network_dic[k]
    if diamond[1] in temp_network.nodes() and diamond[1+int((nb_intermediates/2)+1)] in temp_network.nodes():
      for a in range(len(alternatives)):
        if all([aa in temp_network.nodes() for aa in alternatives[a]]):
          alts[a].append(np.mean([temp_network.nodes[aa]['virtual'] for aa in alternatives[a]]))
        else:
          alts[a].append(1)
  alts = [1-np.mean(a) for a in alts]
  #if both alternatives aren't observed, add minimum value (because one of the paths *has* to be taken)
  inferences = {tuple(alternatives[a]):alts[a] for a in range(len(alternatives))}
  if sum(inferences.values()) == 0:
    inferences = {k:v+0.01+threshold for k,v in inferences.items()}
  #will be removed in the future
  if nb_intermediates == 2:
    inferences = {k[0]:v for k,v in inferences.items()}
  return inferences

def find_diamonds(network, libr = None, nb_intermediates = 2):
  """automatically extracts shapes such as diamonds (A->B,A->C,B->D,C->D) from biosynthetic networks\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | nb_intermediates (int): number of intermediate nodes expected in a network motif to extract; has to be a multiple of 2 (2: diamond, 4: hexagon,...)\n
  | Returns:
  | :-
  | Returns list of dictionaries, with each dictionary a network motif in the form of position in the motif : glycan
  """
  if libr is None:
    libr = lib
  if nb_intermediates % 2 > 0:
    print("nb_intermediates has to be a multiple of 2; please re-try.")
  #generate matching motif
  g1 = nx.DiGraph()
  nb_nodes = nb_intermediates + 2 #number of intermediates + 1 source node + 1 target node
  path_length = int(nb_intermediates/2 + 2) #the length of one of the two paths is half of the number of intermediates + 1 source node + 1 target node
  first_path = []
  second_path = []

  [first_path.append(x) for x in range(1, path_length+1)]
  [second_path.append(first_path[x]) if x in [0, len(first_path)-1] else second_path.append(first_path[x]+ len(first_path)-1) for x in range(0,len(first_path))]                                                                                                                     
  nx.add_path(g1, first_path)
  nx.add_path(g1, second_path)
    
  #find networks containing the matching motif
  graph_pair = nx.algorithms.isomorphism.GraphMatcher(network, g1)
  #find diamonds within those networks
  matchings_list = list(graph_pair.subgraph_isomorphisms_iter())
  unique_keys = list(set([tuple(list(sorted(k.keys()))) for k in matchings_list]))
  #map found diamonds to the same format
  matchings_list = [[d for d in matchings_list if k == tuple(list(sorted(d.keys())))][0] for k in unique_keys]
  matchings_list2 = []
  #for each diamond, only keep the diamond if at least one of the intermediate structures is virtual
  for d in matchings_list:
    d_rev = dict((v, k) for k, v in d.items())
    substrate_node = d_rev[1]
    product_node = d_rev[path_length]
    middle_nodes = []
    [middle_nodes.append(d_rev[key]) for key in range(2, len(d_rev.items())+1) if key != path_length] 
    
    #pattern matching sometimes identify discontinuous patterns containing nodes "out of the path" -> they are filtered here
    #For each middle node, test whether they are part of the same path as substrate node and product node (remove false positives)
    for middle_node in middle_nodes:
      try:
        are_connected = nx.bidirectional_dijkstra(network, substrate_node, middle_node)
        are_connected = nx.bidirectional_dijkstra(network, middle_node, product_node)
      except:
        break

    virtual_state = [nx.get_node_attributes(network, 'virtual')[j] for j in middle_nodes]
    if any([j == 1 for j in virtual_state]):
      d = dict((v, k) for k, v in d.items())
      #filter out non-diamond shapes with any cross-connections
      if nb_intermediates > 2:
        graph_dic = {k:glycan_to_nxGraph(k, libr = libr) for k in d.values()}
        if max((n for d,n in adjacencyMatrix_to_network(create_adjacency_matrix(list(d.values()), graph_dic, libr = libr)[0]).degree())) == 2:
          matchings_list2.append(d)
      else:
        matchings_list2.append(d)
  return matchings_list2

def trace_diamonds(network, species_list, network_dic, libr = None, threshold = 0.,
                   nb_intermediates = 2):
  """extracts diamond-shape motifs from biosynthetic networks (A->B,A->C,B->D,C->D) and uses evolutionary information to determine which path is taken from A to D\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | species_list (list): list of species to compare network to
  | network_dic (dict): dictionary of form species name : biosynthetic network (gained from construct_network)
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | threshold (float): everything below or equal to that threshold will be cut; default:0.
  | nb_intermediates (int): number of intermediate nodes expected in a network motif to extract; has to be a multiple of 2 (2: diamond, 4: hexagon,...)\n
  | Returns:
  | :-
  | Returns dataframe of each intermediary glycan and its proportion (0-1) of how often it has been experimentally observed in this path
  """
  if libr is None:
    libr = lib
  #get the diamonds
  matchings_list = find_diamonds(network, libr = libr, nb_intermediates = nb_intermediates)
  #calculate the path probabilities
  paths = [choose_path(d, species_list, network_dic, libr = libr,
                       threshold = threshold, nb_intermediates = nb_intermediates) for d in matchings_list]
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
  to_cut = [k for k in network_out.nodes() if nx.get_node_attributes(network_out, node_attr)[k] <= threshold]
  #leave virtual nodes that are needed to retain a connected component
  to_cut = [k for k in to_cut if not any([network_out.in_degree[j] == 1 and len(j) > len(k) for j in network_out.neighbors(k)])]
  network_out.remove_nodes_from(to_cut)
  nodeDict = dict(network_out.nodes(data = True))
  #remove virtual nodes with total degree of max 1 and an in-degree of at least 1
  for node in list(network_out.nodes()):
    if (network_out.degree[node] <= 1) and (nodeDict[node]['virtual'] == 1) and (network_out.in_degree[node] > 0):
      network_out.remove_node(node)
  return network_out

def evoprune_network(network, network_dic = None, species_list = None, libr = None,
                     node_attr = 'abundance', threshold = 0.01, nb_intermediates = 2):
  """given a biosynthetic network, this function uses evolutionary relationships to prune impossible paths\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | network_dic (dict): dictionary of form species name : biosynthetic network (gained from construct_network); default:pre-computed milk networks
  | species_list (list): list of species to compare network to; default:species from pre-computed milk networks
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | node_attr (string): which (numerical) node attribute to use for pruning; default:'abundance'
  | threshold (float): everything below or equal to that threshold will be cut; default:0.01
  | nb_intermediates (int): number of intermediate nodes expected in a network motif to extract; has to be a multiple of 2 (2: diamond, 4: hexagon,...)\n
  | Returns:
  | :-
  | Returns pruned network (with virtual node probability as a new node attribute)
  """
  if libr is None:
    libr = lib
  if network_dic is None:
    network_dic = net_dic
  if species_list is None:
    species_list = list(network_dic.keys())
  #calculate path probabilities of diamonds
  df_out = trace_diamonds(network, species_list, network_dic,
                          libr = libr, threshold = threshold, nb_intermediates = nb_intermediates)
  #scale virtual node size by path probability
  network_out = highlight_network(network, highlight = 'abundance', abundance_df = df_out,
                                  intensity_col = 'probability')
  #remove diamond paths below threshold
  network_out = prune_network(network_out, node_attr = node_attr, threshold = threshold*100)
  return network_out

def highlight_network(network, highlight, motif = None,
                      abundance_df = None, glycan_col = 'target',
                      intensity_col = 'rel_intensity', conservation_df = None,
                      network_dic = None, species = None, libr = None):
  """highlights a certain attribute in the network that will be visible when using plot_network\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | highlight (string): which attribute to highlight (choices are 'motif' for glycan motifs, 'abundance' for glycan abundances, 'conservation' for glycan conservation, 'species' for highlighting 1 species in multi-network)
  | motif (string): highlight=motif; which motif to highlight (absence/presence, in violet/green); default:None
  | abundance_df (dataframe): highlight=abundance; dataframe containing glycans and their relative intensity
  | glycan_col (string): highlight=abundance; column name of the glycans in abundance_df
  | intensity_col (string): highlight=abundance; column name of the relative intensities in abundance_df
  | conservation_df (dataframe): highlight=conservation; dataframe containing glycans from different species
  | network_dic (dict): highlight=conservation/species; dictionary of form species name : biosynthetic network (gained from construct_network); default:pre-computed milk networks
  | species (string): highlight=species; which species to highlight in a multi-species network
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns a network with the additional 'origin' (motif/species) or 'abundance' (abundance/conservation) node attribute storing the highlight
  """
  if libr is None:
    libr = lib
  if network_dic is None:
    network_dic = net_dic
  #determine highlight validity
  if highlight == 'motif' and motif is None:
    print("You have to provide a glycan motif to highlight")
  elif highlight == 'species' and species is None:
    print("You have to provide a species to highlight")
  elif highlight == 'abundance' and abundance_df is None:
    print("You have to provide a dataframe with glycan abundances to highlight")
  elif highlight == 'conservation' and conservation_df is None:
    print("You have to provide a dataframe with species-glycan associations to check conservation")
  network_out = copy.deepcopy(network)
  #color nodes whether they contain the motif (green) or not (violet)
  if highlight == 'motif':
    if motif[-1] == ')':
      motif_presence = {k:('limegreen' if motif in k else 'darkviolet') for k in network_out.nodes()}
    else:
      motif_presence = {k:('limegreen' if subgraph_isomorphism(k, motif, libr = libr) else 'darkviolet') for k in network_out.nodes()}
    nx.set_node_attributes(network_out, motif_presence, name = 'origin')
  #color nodes whether they are from a species (green) or not (violet)
  elif highlight == 'species':
    temp_net = network_dic[species]
    node_presence = {k:('limegreen' if k in temp_net.nodes() else 'darkviolet') for k in network_out.nodes()}
    nx.set_node_attributes(network_out, node_presence, name = 'origin')
  #add relative intensity values as 'abundance' node attribute used for node size scaling
  elif highlight == 'abundance':
    node_abundance = {node:(abundance_df[intensity_col].values.tolist()[abundance_df[glycan_col].values.tolist().index(node)]*100 if node in abundance_df[glycan_col].values.tolist() else 50) for node in network_out.nodes()}
    nx.set_node_attributes(network_out, node_abundance, name = 'abundance')
  #add the degree of evolutionary conervation as 'abundance' node attribute used for node size scaling
  elif highlight == 'conservation':
    spec_nodes = []
    specs = list(set(conservation_df.Species.values.tolist()))
    for k in specs:
      temp_net = network_dic[k]
      spec_nodes.append(list(temp_net.nodes()))
    specs = [[specs[k]]*len(spec_nodes[k]) for k in range(len(specs))]
    df_temp = pd.DataFrame(unwrap(specs), unwrap(spec_nodes)).reset_index()
    df_temp.columns = ['target', 'Species']
    node_conservation = {k:df_temp.target.values.tolist().count(k)*100 for k in network_out.nodes()}
    nx.set_node_attributes(network_out, node_conservation, name = 'abundance')
  return network_out
