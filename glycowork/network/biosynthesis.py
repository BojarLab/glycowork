import mpld3
import pickle
import re
from os import path
from copy import deepcopy
from itertools import combinations
from functools import lru_cache
from importlib import resources
from collections import defaultdict
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.stats.multitest import multipletests
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glycowork.glycan_data.loader import unwrap, linkages, lib
from glycowork.glycan_data.stats import cohen_d, get_alphaN
from glycowork.motif.graph import compare_glycans, glycan_to_nxGraph, graph_to_string, subgraph_isomorphism, get_possible_topologies
from glycowork.motif.processing import choose_correct_isoform, get_lib, rescue_glycans, in_lib, get_class
from glycowork.motif.tokenization import get_stem_lib
from glycowork.motif.regex import get_match
from glycowork.motif.annotate import link_find

this_dir, this_filename = path.split(__file__)
data_path = path.join(this_dir, 'milk_networks_exhaustive.pkl')

def __getattr__(name):
  if name == "net_dic":
    net_dic = pickle.load(open(data_path, 'rb'))
    globals()[name] = net_dic  # Cache it to avoid reloading
    return net_dic
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

permitted_roots = frozenset({"Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"})
allowed_ptms = frozenset({'OS', '3S', '6S', 'OP', '1P', '3P', '6P', 'OAc', '4Ac', '9Ac'})


@lru_cache(maxsize = None)
def safe_compare(g1, g2):
  """compare_glycans with try/except error catch\n
  | Arguments:
  | :-
  | g1 (networkx object): glycan graph from glycan_to_nxGraph
  | g2 (networkx object): glycan graph from glycan_to_nxGraph\n
  | Returns:
  | :-
  | Returns True if two glycans are the same and False if not; returns False if 'except' is triggered
  """
  try:
    return compare_glycans(g1, g2)
  except:
    return False


def safe_index(glycan, graph_dic):
  """retrieves glycan graph and if not present in graph_dic it will freshly calculate\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph\n
  | Returns:
  | :-
  | Returns a glycan graph, either from graph_dic or freshly generated
  """
  if glycan not in graph_dic:
    graph_dic[glycan] = glycan_to_nxGraph(glycan)
  return graph_dic[glycan]


def get_neighbors(ggraph, glycans, graphs, min_size = 1):
  """find (observed) biosynthetic precursors of a glycan\n
  | Arguments:
  | :-
  | ggraph (networkx): glycan graph as networkx object
  | glycans (list): list of glycans in IUPAC-condensed format
  | graphs (list): list of glycans in df as graphs
  | min_size (int): length of smallest root in biosynthetic network; default:1\n
  | Returns:
  | :-
  | (1) a list of direct glycan precursors in IUPAC-condensed
  | (2) a list of indices where each precursor from (1) can be found in glycans
  """
  # Get graph; graphs of single monosaccharide can't have precursors
  if len(ggraph.nodes()) <= 1:
    return [], []
  # Get biosynthetic precursors
  ggraph_nb = create_neighbors(ggraph, min_size = min_size)
  # Find out whether any member in 'glycans' is a biosynthetic precursor of 'glycan'
  idx = [np.where([safe_compare(k, j) for k in graphs])[0].tolist() for j in ggraph_nb]
  nb = [glycans[k[0]] for k in idx if k]
  return nb, idx


@lru_cache(maxsize = None)
def create_neighbors(ggraph, min_size = 1):
  """creates biosynthetic precursor glycans\n
  | Arguments:
  | :-
  | ggraph (networkx object): glycan graph from glycan_to_nxGraph
  | min_size (int): length of smallest root in biosynthetic network; default:1\n
  | Returns:
  | :-
  | Returns biosynthetic precursor glycans
  """
  num_nodes = len(ggraph.nodes())
  # Check whether glycan large enough to allow for precursors
  if num_nodes <= min_size:
    return []
  if num_nodes == 3:
    return [nx.relabel_nodes(ggraph.subgraph([2]), {2: 0})]
  # Generate all precursors by iteratively cleaving off the non-reducing-end monosaccharides
  max_node = max(ggraph.nodes())
  terminal_pairs = [frozenset({k, next(ggraph.neighbors(k))}) for k in ggraph.nodes() if ggraph.degree(k) == 1 and k != max_node]
  nodes = frozenset(ggraph.nodes())
  # Cleaving off messes with the node labeling, so they have to be re-labeled
  return [
        nx.relabel_nodes(ggraph.subgraph(nodes - pair), {m: i for i, m in enumerate(nodes - pair)})
        for pair in terminal_pairs
    ]


def get_virtual_nodes(glycan, graph_dic, min_size = 1):
  """find unobserved biosynthetic precursors of a glycan\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | min_size (int): length of smallest root in biosynthetic network; default:1\n
  | Returns:
  | :-
  | (1) list of virtual node graphs
  | (2) list of virtual nodes in IUPAC-condensed format
  """
  if glycan.count('(') + 1 <= min_size:
    return [], []
  # Get glycan graph
  ggraph = safe_index(glycan, graph_dic)
  # Get biosynthetic precursors
  ggraph_nb_t = set(map(graph_to_string, create_neighbors(ggraph, min_size = min_size)))
  # Get both string and graph versions of the precursors
  ggraph_nb, ggraph_nb_t2 = zip(*[(safe_index(k, graph_dic), k)
                                    for k in ggraph_nb_t if not k.startswith('(')])
  return ggraph_nb, ggraph_nb_t2


def find_diff(glycan_a, glycan_b, graph_dic, allowed_ptms = allowed_ptms):
  """finds the subgraph that differs between glycans and returns it, will only work if the differing subgraph is connected\n
  | Arguments:
  | :-
  | glycan_a (networkx): glycan graph as networkx object
  | glycan_b (networkx): glycan graph as networkx object
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | allowed_ptms (set): list of PTMs to consider\n
  | Returns:
  | :-
  | Returns difference between glycan_a and glycan_b in IUPAC-condensed
  """
  # Check whether the glycans are equal
  if glycan_a == glycan_b:
    return ""
  # Convert to graphs and get difference
  larger, smaller = max(glycan_a, glycan_b, key = len), min(glycan_a, glycan_b, key = len)
  graph_larger, graph_smaller = safe_index(larger, graph_dic), safe_index(smaller, graph_dic)
  matched = subgraph_isomorphism(graph_larger, graph_smaller, return_matches = True)
  if not isinstance(matched, bool) and matched[0]:
    diff_nodes = set(graph_larger.nodes()) - set(matched[1][0])
    diff_string = graph_to_string(graph_larger.subgraph(diff_nodes))
    return diff_string if not any(ptm in diff_string for ptm in allowed_ptms) else 'disregard'
  return 'disregard'


def find_shared_virtuals(glycan_a, glycan_b, graph_dic, min_size = 1):
  """finds virtual nodes that are shared between two glycans (i.e., that connect these two glycans)\n
  | Arguments:
  | :-
  | glycan_a (string): glycan in IUPAC-condensed format
  | glycan_b (string): glycan in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | min_size (int): length of smallest root in biosynthetic network; default:1\n
  | Returns:
  | :-
  | Returns list of edges between glycan and virtual node (if virtual node connects the two glycans)
  """
  if abs(glycan_a.count('(') - glycan_b.count('(')) != 2:
    return []
  # Get virtual nodes of both glycans
  ggraph_nb_a, glycans_a = get_virtual_nodes(glycan_a, graph_dic, min_size = min_size)
  if not ggraph_nb_a:
    return []
  ggraph_nb_b, _ = get_virtual_nodes(glycan_b, graph_dic, min_size = min_size)
  # Check whether any of the nodes of glycan_a and glycan_b are the same
  out = set((glycan_a, glycans_a[k]) for k, graph_a in enumerate(ggraph_nb_a)
              for graph_b in ggraph_nb_b if compare_glycans(graph_a, graph_b))
  out.update((glycan_b, g[1]) for g in out)
  return list(out)


def adjacencyMatrix_to_network(adjacency_matrix):
  """converts an adjacency matrix to a network\n
  | Arguments:
  | :-
  | adjacency_matrix (dataframe): denoting whether two glycans are connected by one biosynthetic step\n
  | Returns:
  | :-
  | Returns biosynthetic network as a networkx graph
  """
  network = nx.from_pandas_adjacency(adjacency_matrix)
  network = nx.relabel_nodes(network, {k: c for k, c in enumerate(adjacency_matrix.columns)})
  return network


def create_adjacency_matrix(glycans, graph_dic, min_size = 1):
  """creates a biosynthetic adjacency matrix from a list of glycans\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | min_size (int): length of smallest root in biosynthetic network; default:1\n
  | Returns:
  | :-
  | (1) adjacency matrix (glycan X glycan) denoting whether two glycans are connected by one biosynthetic step
  | (2) list of which nodes are virtual nodes (empty list if virtual_nodes is False)
  """
  # Connect glycans with biosynthetic precursors
  df_out = pd.DataFrame(0, index = glycans, columns = glycans)
  graphs = [safe_index(g, graph_dic) for g in glycans]
  results = [get_neighbors(g, glycans, graphs, min_size = min_size) for g in graphs]
  if all(not result[0] and not result[1] for result in results):
    idx = []
  else:
    _, idx = zip(*results)
  # Fill adjacency matrix
  for j, indices in enumerate(idx):
    for i in indices:
      if i:
        df_out.iat[i[0], j] = 1
  # Find connections between virtual nodes that connect observed nodes
  virtual_edges = set(unwrap([find_shared_virtuals(k[0], k[1], graph_dic,
                                               min_size = min_size) for k in combinations(glycans, 2)]))
  new_nodes = list(set([k[1] for k in virtual_edges if k[1] not in df_out.columns]))
  idx = [i for i, k in enumerate(new_nodes) if any(compare_glycans(safe_index(k, graph_dic), safe_index(j, graph_dic)) for j in df_out.columns)]
  if idx:
    sub_new = {new_nodes[k] for k in idx}
    virtual_edges = [j for j in virtual_edges if j[1] not in sub_new]
    new_nodes = [n for i, n in enumerate(new_nodes) if i not in idx]
  # Add virtual nodes
  for k in new_nodes:
    df_out[k] = 0
    df_out.loc[k] = 0
  # Add virtual edges
  for k in virtual_edges:
    df_out.at[k[0], k[1]] = 1
  return adjacencyMatrix_to_network(df_out.add(df_out.T, fill_value = 0)), new_nodes


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
  # Find connections/edges between virtual nodes from propagate_virtuals run N and propagate_virtuals run N+1
  return [unwrap([[(prev_shell[m][k], next_shell[k][j]) for j in range(len(next_shell[k]))] for k in range(len(prev_shell[m]))]) for m in range(len(prev_shell))]


def find_path(glycan_a, glycan_b, graph_dic,
              permitted_roots = permitted_roots, min_size = 1, allowed_ptms = allowed_ptms):
  """find virtual node path between two glycans\n
  | Arguments:
  | :-
  | glycan_a (string): glycan in IUPAC-condensed format
  | glycan_b (string): glycan in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | permitted_roots (set): which nodes should be considered as roots; default:["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]
  | min_size (int): length of smallest root in biosynthetic network; default:1
  | allowed_ptms (set): list of PTMs to consider\n
  | Returns:
  | :-
  | (1) list of edges to connect glycan_a and glycan_b via virtual nodes
  | (2) dictionary of edge labels detailing difference between two connected nodes
  """
  larger_glycan, smaller_glycan = max(glycan_a, glycan_b, key = len), min(glycan_a, glycan_b, key = len)
  # Bridge the distance between a and b by generating biosynthetic precursors
  dist = larger_glycan.count('(') - smaller_glycan.count('(')
  if dist not in range(1, 6):
    return [], {}
  virtuals_t = [[larger_glycan]]
  # While ensuring that glycans don't become smaller than the root, find biosynthetic precursors of each glycan in glycans
  virtual_shells_t = [[[larger_glycan]]] + [virtuals_t := [get_virtual_nodes(k, graph_dic, min_size = min_size)[1] for k in unwrap(virtuals_t) if k.count('(') > (min_size - 1)] for _ in range(dist)]
  # Get the edges connecting the starting point to the experimentally observed node
  virtual_edges = unwrap([unwrap(shells_to_edges(virtual_shells_t[k-1], virtual_shells_t[k])) for k in range(1, len(virtual_shells_t))])
  del virtual_shells_t
  # Find the edge labels / differences between precursors
  return virtual_edges, {el: find_diff(el[0], el[1], graph_dic, allowed_ptms = allowed_ptms) for el in virtual_edges}


def find_shortest_path(goal_glycan, glycan_list, graph_dic,
                       permitted_roots = permitted_roots, min_size = 1, allowed_ptms = allowed_ptms):
  """finds the glycan with the shortest path via virtual nodes to the goal glycan\n
  | Arguments:
  | :-
  | goal_glycan (string): glycan in IUPAC-condensed format
  | glycan_list (list): list of glycans in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | permitted_roots (set): which nodes should be considered as roots; default:["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]
  | min_size (int): length of smallest root in biosynthetic network; default:1
  | allowed_ptms (set): list of PTMs to consider\n
  | Returns:
  | :-
  | (1) list of edges of shortest path to connect goal_glycan and glycan via virtual nodes
  | (2) dictionary of edge labels detailing difference between two connected nodes in shortest path
  """
  ggraph = safe_index(goal_glycan, graph_dic)
  for glycan in sorted(glycan_list, key = len, reverse = True):
    # For each glycan, check whether it could constitute a precursor (i.e., is it a sub-graph + does it stem from the correct root)
    if len(glycan) < len(goal_glycan) and goal_glycan.endswith(glycan[-5:]) and subgraph_isomorphism(ggraph, safe_index(glycan, graph_dic)):
      try:
        # Finding a path through shells of generated virtual nodes
        virtual_edges, edge_labels = find_path(goal_glycan, glycan, graph_dic,
                                                 permitted_roots = permitted_roots, min_size = min_size, allowed_ptms = allowed_ptms)
        return virtual_edges, edge_labels
      except:
        continue
  return [], {}


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
  network.remove_edges_from([(u, v) for u, v, data in network.edges(data = True) if data.get(attr) == value])
  return network


def stemify_glycan_fast(ggraph_in, stem_lib):
  """stemifies a glycan graph\n
  | Arguments:
  | :-
  | ggraph_in (networkx): glycan graph as a networkx object
  | stem_lib (dictionary): dictionary of form modified_monosaccharide:core_monosaccharide\n
  | Returns:
  | :-
  | (1) stemified glycan in IUPAC-condensed as a string
  | (2) stemified glycan graph as a networkx object
  """
  ggraph = deepcopy(ggraph_in)
  nx.set_node_attributes(ggraph, {k: {"string_labels": stem_lib[v]} for k, v in ggraph.nodes(data = "string_labels")})
  return graph_to_string(ggraph), ggraph


def find_ptm(glycan, glycans, graph_dic, stem_lib, allowed_ptms = allowed_ptms,
             ggraphs = None, suffix = '-ol'):
  """identifies precursor glycans for a glycan with a PTM\n
  | Arguments:
  | :-
  | glycan (string): glycan with PTM in IUPAC-condensed format
  | glycans (list): list of glycans in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | stem_lib (dictionary): dictionary of form modified_monosaccharide:core_monosaccharide
  | allowed_ptms (set): list of PTMs to consider
  | ggraphs (list): list of precomputed graphs of the glycan list
  | suffix (string): optional suffix to be added to the stemified glycan; default:'-ol'\n
  | Returns:
  | :-
  | (1) an edge tuple between input glycan and its biosynthetic precusor without PTM
  | (2) the PTM that is contained in the input glycan
  """
  # Checks which PTM(s) are present
  mod = next((ptm for ptm in allowed_ptms if ptm in glycan), None)
  if mod is None:
    return 0
  # Stemifying returns the unmodified glycan
  glycan_stem, g_stem = stemify_glycan_fast(safe_index(glycan, graph_dic), stem_lib)
  glycan_stem += suffix
  if suffix == '-ol':
    last_node = max(g_stem.nodes())
    g_stem.nodes[last_node]['string_labels'] += suffix
  if ('Sug' in glycan_stem) or ('Neu(' in glycan_stem):
    return 0
  if ggraphs is None:
    ggraphs = [safe_index(k, graph_dic) for k in glycans]
  idx = next((i for i, k in enumerate(ggraphs) if safe_compare(k, g_stem)), None)
  # If found, add a biosynthetic step of adding the PTM to the unmodified glycan
  if idx is not None:
    return (glycan, glycans[idx]), mod
  # If not, just add a virtual node of the unmodified glycan+corresponding edge; will be connected to the main network later; will not properly work if glycan has multiple PTMs
  elif ''.join(sorted(glycan)) == ''.join(sorted(glycan_stem + mod)):
    return (glycan, glycan_stem), mod
  else:
    return 0


def process_ptm(glycans, graph_dic, stem_lib, allowed_ptms = allowed_ptms, suffix = '-ol'):
  """identifies glycans that contain post-translational modifications and their biosynthetic precursor\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | stem_lib (dictionary): dictionary of form modified_monosaccharide:core_monosaccharide
  | allowed_ptms (set): list of PTMs to consider
  | suffix (string): optional suffix to be added to the stemified glycan; default:'-ol'\n
  | Returns:
  | :-
  | (1) list of edge tuples between input glycans and their biosynthetic precusor without PTM
  | (2) list of the PTMs that are contained in the input glycans
  """
  # Get glycans with PTMs and convert them to graphs
  ptm_glycans = [glycan for glycan in glycans if any(ptm in glycan for ptm in allowed_ptms)]
  ggraphs = list(map(lambda k: safe_index(k, graph_dic), glycans))
  # Connect modified glycans to their unmodified counterparts
  edges = [find_ptm(k, glycans, graph_dic, stem_lib, allowed_ptms = allowed_ptms,
                    ggraphs = ggraphs, suffix = suffix) for k in ptm_glycans]
  valid_edges = [k for k in edges if k != 0]
  if valid_edges:
    return list(zip(*valid_edges))
  else:
    return [], []


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
  network.add_edges_from(edge_list)
  if edge_labels:
    nx.set_edge_attributes(network, dict(zip(edge_list, edge_labels)), 'diffs')
  if node_labels:
    nx.set_node_attributes(network, node_labels, 'virtual')
  else:
    nx.set_node_attributes(network, {node: 1 for node in network.nodes if 'virtual' not in network.nodes[node]}, 'virtual')
  return network


def return_unconnected_to_root(network, permitted_roots = permitted_roots):
  """finds observed nodes that are not connected to the root nodes\n
  | Arguments:
  | :-
  | network (networkx object): network that should be analyzed
  | permitted_roots (set): which nodes should be considered as roots; default:{"Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"}\n
  | Returns:
  | :-
  | Returns set of nodes that are not connected to the root nodes
  """
  # Pre-filter nodes that are observed and not in permitted_roots
  candidate_nodes = {node for node, attr in network.nodes(data = True) if attr.get('virtual', 1) == 0 and node not in permitted_roots}
  # For each observed node, check whether it has a path to a root; if not, collect and return the node
  return set(filter(lambda node: not any(nx.has_path(network, node, root) for root in permitted_roots), candidate_nodes))


def deorphanize_nodes(network, graph_dic, permitted_roots = permitted_roots, min_size = 1,
                                        allowed_ptms = allowed_ptms):
  """finds nodes unconnected to root nodes and tries to connect them\n
  | Arguments:
  | :-
  | network (networkx object): network that should be modified
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | permitted_roots (set): which nodes should be considered as roots; default:["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"]
  | min_size (int): length of smallest root in biosynthetic network; default:1
  | allowed_ptms (set): list of PTMs to consider\n
  | Returns:
  | :-
  | Returns network with deorphanized nodes (if they can be connected to root nodes)
  """
  permitted_roots = frozenset({root for root in permitted_roots if root in network.nodes()})
  if not permitted_roots:
    return network.copy()
  # Get unconnected observed nodes
  unconnected_nodes = return_unconnected_to_root(network, permitted_roots = permitted_roots)
  # Directed needed to detect nodes that appear to be connected (downstream) but are not connected to the root
  temp_network = prune_directed_edges(network.to_directed())
  pseudo_connected_nodes = {node for node, degree in temp_network.in_degree() if degree == 0} - (permitted_roots | unconnected_nodes)
  unconnected_nodes |= pseudo_connected_nodes
  # Only consider unconnected nodes if they are larger than the root (otherwise they should be pruned anyway)
  unconnected_nodes = {k for k in unconnected_nodes if len(safe_index(k, graph_dic)) > min_size}
  # Subset the observed nodes that are connected to the root and are at least the size of the smallest root
  real_nodes = sorted((node for node, attr in network.nodes(data = True)
                         if attr.get('virtual', 1) == 0 and node not in unconnected_nodes and
                         len(safe_index(node, graph_dic)) >= min_size), key = len, reverse = True)
  edges, edge_labels = [], []
  # For each unconnected node, find the shortest path to its root node
  for node in unconnected_nodes:
    g_node = safe_index(node, graph_dic)
    r_target = next((r for r in real_nodes if subgraph_isomorphism(g_node, safe_index(r, graph_dic))), '')
    if not r_target:
      continue
    e, el = find_path(node, r_target, graph_dic, permitted_roots = permitted_roots, min_size = min_size, allowed_ptms = allowed_ptms)
    edges.extend(e)
    edge_labels.extend(el[j] for j in e)
  new_nodes = set().union(*edges) - set(network.nodes())
  node_labels = {node: network.nodes[node].get('virtual', 1) for node in network.nodes()}
  node_labels.update({node: 1 for node in new_nodes})
  return update_network(network, edges, edge_labels = edge_labels, node_labels = node_labels)


def prune_directed_edges(network):
  """removes edges that go against the direction of biosynthesis\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network with directed edges\n
  | Returns:
  | :-
  | Returns a network with pruned edges that would go in the wrong direction
  """
  edges_to_remove = [(u, v) for u, v in network.edges() if len(u) > len(v)]
  network.remove_edges_from(edges_to_remove)
  return network


def deorphanize_edge_labels(network, graph_dic, allowed_ptms = allowed_ptms):
  """completes edge labels in newly added edges of a biosynthetic network\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | graph_dic (dict): dictionary of form glycan : glycan-graph
  | allowed_ptms (set): list of PTMs to consider\n
  | Returns:
  | :-
  | Returns a network with completed edge labels
  """
  edge_labels = nx.get_edge_attributes(network, 'diffs')
  unlabeled_edges = [edge for edge in network.edges() if edge not in edge_labels]
  for edge in unlabeled_edges:
    network.edges[edge]['diffs'] = find_diff(edge[0], edge[1], graph_dic, allowed_ptms = allowed_ptms)
  return network


@lru_cache(maxsize = None)
def infer_roots(glycans):
  """infers the correct permitted roots, given the glycan class\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format\n
  | Returns:
  | :-
  | Returns a set of permitted roots
  """
  # Free
  if any(k.endswith('-ol') for k in glycans):
    return frozenset({'Gal(b1-4)Glc-ol', 'Gal(b1-4)GlcNAc-ol'})
  # O-linked
  elif any(k.endswith('GalNAc') for k in glycans):
    return frozenset({'GalNAc', 'Fuc', 'Man'})
  # N-linked
  elif any(k.endswith('GlcNAc') for k in glycans):
    return frozenset({'Man(b1-4)GlcNAc(b1-4)GlcNAc'})
  # Glycolipid
  elif any(k.endswith('1Cer') or k.endswith('Ins') for k in glycans):
    return frozenset({'Glc1Cer', 'Gal1Cer', 'Ins'})
  else:
    print("Glycan class not detected; depending on the class, glycans should end in -ol, GalNAc, GlcNAc, or Glc")


def add_high_man_removal(network):
  """considers the process of cutting down high-mannose N-glycans for maturation\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network\n
  | Returns:
  | :-
  | Returns a network with edges going upstream to indicate removal of Man
  """
  edges_to_add = [(target, source, network[source][target])
                    for source, target in network.edges()
                    if target.count('Man') >= 5]
  network.add_edges_from(edges_to_add, diffs = lambda x: x[2])
  return network


@rescue_glycans
def construct_network(glycans, allowed_ptms = allowed_ptms,
                      edge_type = 'monolink', permitted_roots = None, abundances = []):
  """construct a glycan biosynthetic network\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format
  | allowed_ptms (set): list of PTMs to consider
  | edge_type (string): indicates whether edges represent monosaccharides ('monosaccharide'), monosaccharide(linkage) ('monolink'), or enzyme catalyzing the reaction ('enzyme'); default:'monolink'
  | permitted_roots (set): which nodes should be considered as roots; default:will be inferred
  | abundances (list): optional list of abundances, in the same order as glycans; default:empty list\n
  | Returns:
  | :-
  | Returns a networkx object of the network
  """
  stem_lib = get_stem_lib(get_lib(glycans))
  if permitted_roots is None:
    permitted_roots = infer_roots(frozenset(glycans))
  abundance_mapping = dict(zip(glycans, abundances)) if abundances else {}
  # Generating graph from adjacency of observed glycans
  min_size = min(k.count('(') for k in permitted_roots) + 1
  add_to_virtuals = [r for r in permitted_roots if r not in glycans and any(g.endswith(r) for g in glycans)]
  glycans.extend(add_to_virtuals)
  glycans.sort(key = len, reverse = True)
  graph_dic = {k: glycan_to_nxGraph(k) for k in glycans}
  network, virtual_nodes = create_adjacency_matrix(glycans, graph_dic, min_size = min_size)
  # Connecting observed via virtual nodes; Check whether the neighbor has a shorter string, indicating an "incoming" connection
  unconnected_nodes = {node for node in network.nodes() if not any(len(neighbor) < len(node) for neighbor in network.neighbors(node))}
  new_nodes, new_edges, new_edge_labels = set(), [], []
  for k in sorted(unconnected_nodes):
    try:
      virtual_edges, edge_labels = find_shortest_path(k, [j for j in network.nodes() if j != k], graph_dic,
                                                      permitted_roots = permitted_roots, min_size = min_size, allowed_ptms = allowed_ptms)
      new_nodes.update(set(sum(virtual_edges, ())) - set(network.nodes()))
      new_edges.extend(virtual_edges)
      new_edge_labels.extend(edge_labels)
    except:
      pass
  network.add_edges_from(new_edges, edge_labels = new_edge_labels)
  virtual_nodes = set(virtual_nodes + add_to_virtuals + list(new_nodes))
  graph_dic.update({node: glycan_to_nxGraph(node) for node in virtual_nodes if node not in graph_dic})
  network.remove_edges_from([(u, v) for u, v in network.edges if u in virtual_nodes and v in virtual_nodes and network.degree[u if len(u) > len(v) else v] == 1])
  # Create edge and node labels
  nx.set_edge_attributes(network, {el: find_diff(el[0], el[1], graph_dic, allowed_ptms = allowed_ptms) for el in network.edges()}, 'diffs')
  nx.set_node_attributes(network, {k: 1 if k in virtual_nodes else 0 for k in network.nodes()}, 'virtual')
  # Connect post-translational modifications
  suffix = '-ol' if '-ol' in ''.join(glycans) else '1Cer' if '1Cer' in ''.join(glycans) else ''
  virtual_nodes = [x for x, y in network.nodes(data = True) if y['virtual'] == 1]
  real_nodes = [safe_index(x, graph_dic) for x, y in network.nodes(data = True) if y['virtual'] == 0]
  network.remove_nodes_from([v for v in virtual_nodes if any(compare_glycans(safe_index(v, graph_dic), r) for r in real_nodes) or not in_lib(v, lib)])
  ptm_links = process_ptm(list(network.nodes()), graph_dic, stem_lib, allowed_ptms = allowed_ptms, suffix = suffix)
  if ptm_links:
    network = update_network(network, ptm_links[0], edge_labels = ptm_links[1])
  # Find any remaining orphan nodes and connect them to the root(s)
  network = deorphanize_nodes(network, graph_dic, permitted_roots = permitted_roots, min_size = min_size, allowed_ptms = allowed_ptms)
  # Final clean-up / condensation step
  to_remove_nodes = [node for node in sorted(network.nodes(), key = len, reverse = True) if
                       (network.degree[node] <= 1) and (network.nodes[node]['virtual'] == 1) and (node not in permitted_roots)]
  network.remove_nodes_from(to_remove_nodes)
  graph_dic.update({node: glycan_to_nxGraph(node) for node in network.nodes() if node not in graph_dic})
  filler_network, _ = create_adjacency_matrix(list(network.nodes()), graph_dic, min_size = min_size)
  network.add_edges_from(filler_network.edges())
  network = deorphanize_edge_labels(network, graph_dic, allowed_ptms = allowed_ptms)
  for node in network.nodes:
    network.nodes[node].setdefault('virtual', 1)
  # Edge label specification
  if edge_type != 'monolink':
    for e in network.edges():
      elem = network.get_edge_data(e[0], e[1])
      edge = elem['diffs']
      for link in linkages:
        if link in edge:
          if edge_type == 'monosaccharide':
            elem['diffs'] = edge.split('(')[0]
          elif edge_type == 'enzyme':
            with resources.open_text("glycowork.network", "monolink_to_enzyme.csv") as f:
              df_enzyme = pd.read_csv(f, sep = '\t')
            elem['diffs'] = monolink_to_glycoenzyme(edge, df_enzyme)
          else:
            pass
  # Remove virtual nodes that are branch isomers of real nodes
  virtual_nodes = [x for x, y in network.nodes(data = True) if y['virtual'] == 1]
  virtual_graphs = [safe_index(x, graph_dic) for x in virtual_nodes]
  isomeric_graphs = [k for k in combinations(virtual_graphs, 2) if compare_glycans(k[0], k[1])]
  if len(isomeric_graphs) > 0:
    isomeric_nodes = [choose_correct_isoform([virtual_nodes[virtual_graphs.index(k[0])], virtual_nodes[virtual_graphs.index(k[1])]], reverse = True)
                          for k in isomeric_graphs if len(set(k)) > 1]
    network.remove_nodes_from([n[0] for n in isomeric_nodes if n])
  network.remove_edges_from(nx.selfloop_edges(network))
  # Make network directed
  network = prune_directed_edges(network.to_directed())
  for node in sorted(network.nodes(), key = len, reverse = False):
    if (network.in_degree[node] < 1) and (network.nodes[node]['virtual'] == 1) and (node not in permitted_roots):
      network.remove_node(node)
  for node in sorted(network.nodes(), key = len, reverse = True):
    if (network.out_degree[node] < 1) and (network.nodes[node]['virtual'] == 1):
      network.remove_node(node)
  if 'GlcNAc' in ''.join(permitted_roots) and any(g.count('Man') >= 5 for g in network.nodes()):
    network = add_high_man_removal(network)
  if abundances:
    nx.set_node_attributes(network, {g: {'abundance': abundance_mapping.get(g, 0.0)} for g in network.nodes()})
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
    return
  fig, ax = plt.subplots(figsize = (16, 16))
  # Check whether are values to scale node size by
  node_attributes = nx.get_node_attributes(network, 'abundance')
  node_sizes = list(node_attributes.values()) if node_attributes else [50] * network.number_of_nodes()
  # Decide on network layout
  pos = {'pydot2': lambda: nx.nx_pydot.pydot_layout(network, prog = "dot"),
           'kamada_kawai': lambda: nx.kamada_kawai_layout(network),
           'spring': lambda: nx.spring_layout(network, k = 1/4)}.get(plot_format, lambda: nx.kamada_kawai_layout(network))()
  # Check whether there are values to scale node color by
  node_attributes = nx.get_node_attributes(network, 'origin')
  virtual_attrs = nx.get_node_attributes(network, 'virtual')
  alpha_map = [0.2 if virtual_attrs.get(node, 0) == 1 else 1 for node in network.nodes()]
  if node_attributes:
    scatter = nx.draw_networkx_nodes(network, pos, node_size = node_sizes, ax = ax,
                                     node_color = list(node_attributes.values()), alpha = alpha_map)
  else:
    color_map = ['darkorange' if virtual_attrs.get(node, 0) == 1 else 'cornflowerblue' if virtual_attrs.get(node, 0) == 0 else 'seagreen' for node in network.nodes()]
    scatter = nx.draw_networkx_nodes(network, pos, node_size = node_sizes, alpha = 0.7,
                                     node_color = color_map, ax = ax)
  edge_attributes = nx.get_edge_attributes(network, 'diffs')
  if edge_label_draw: 
    if lfc_dict:
      # Map log-fold changes of responsible enzymes onto biosynthetic steps
      c_list = ['red' if lfc_dict.get(attr, 0) < 0 else 'green' if lfc_dict.get(attr, 0) > 0 else 'cornflowerblue' for attr in edge_attributes.values()]
      w_list = [abs(lfc_dict.get(attr, 0)) or 1 for attr in edge_attributes.values()]
      nx.draw_networkx_edges(network, pos, ax = ax, edge_color = c_list, width = w_list)
    else:
      nx.draw_networkx_edges(network, pos, ax = ax, edge_color = 'cornflowerblue')
    nx.draw_networkx_edge_labels(network, pos, edge_labels = edge_attributes, ax = ax,
                                 verticalalignment = 'baseline')
  else:
    # Alternatively, color edge by type of monosaccharide that is added
    c_list = ['cornflowerblue' if 'Glc' in attr else 'yellow' if 'Gal' in attr else 'red' if 'Fuc' in attr else 'mediumorchid' if '5Ac' in attr
                  else 'turquoise' if '5Gc' in attr else 'silver' for attr in edge_attributes.values()]
    w_list = [1 if 'b1' in attr else 3 for attr in edge_attributes.values()]
    nx.draw_networkx_edges(network, pos, ax = ax, edge_color = c_list, width = w_list)
  ax.axis('off')
  tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels = list(network.nodes()))
  mpld3.plugins.connect(fig, tooltip)


def network_alignment(network_a, network_b):
  """combines two networks into a new network\n
  | Arguments:
  | :-
  | network_a (networkx object): biosynthetic network from construct_network
  | network_b (networkx object): biosynthetic network from construct_network\n
  | Returns:
  | :-
  | Returns combined network as a networkx object (brown: shared, blue: only in a, orange: only in b)
  """
  U = nx.Graph()
  network_a_nodes = set(network_a.nodes)
  network_b_nodes = set(network_b.nodes)
  # Get combined nodes and whether they occur in either network
  only_a = network_a_nodes - network_b_nodes
  only_b = network_b_nodes - network_a_nodes
  both = network_a_nodes & network_b_nodes
  node_origin = {node: 'cornflowerblue' for node in only_a}
  node_origin.update({node: 'darkorange' for node in only_b})
  node_origin.update({node: 'saddlebrown' for node in both})
  all_nodes = list(network_a.nodes(data = True)) + list(network_b.nodes(data = True))
  U.add_nodes_from(all_nodes)
  # Get combined edges
  all_edges = list(network_a.edges(data = True)) + list(network_b.edges(data = True))
  U.add_edges_from(all_edges)
  nx.set_node_attributes(U, node_origin, name = 'origin')
  # If input is directed, make output directed
  if nx.is_directed(network_a):
    U = prune_directed_edges(U.to_directed())
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
  # Perform network alignment if not provided
  if combined is None:
    combined = network_alignment(network_a, network_b)
  a_nodes = set(network_a.nodes())
  b_nodes = set(network_b.nodes())
  # Find virtual nodes in network_a that are observed in network_b and vice versa
  virtual_a = {k for k, v in network_a.nodes(data = True) if v.get('virtual', 0) == 1}
  virtual_b = {k for k, v in network_b.nodes(data = True) if v.get('virtual', 0) == 1}
  virtual_combined = {k for k, v in combined.nodes(data = True) if v.get('virtual', 0) == 1}
  supported_a = virtual_a & virtual_b
  supported_b = virtual_b & virtual_a
  inferred_a = virtual_a - virtual_combined
  inferred_b = virtual_b - virtual_combined
  return (list(inferred_a), list(supported_a)), (list(inferred_b), list(supported_b))


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
  inferences = set()
  # For each species, try to identify observed nodes matching virtual nodes in input network
  for k in species_list:
    if k == network_species:
      continue
    temp_network = network_dic[k]
    # Get virtual nodes observed in species k
    infer_network, _ = infer_virtual_nodes(network, temp_network)
    inferences.update(infer_network[0])
  # Output a new network with formerly virtual nodes updated to a new node status --> inferred
  network2 = network.copy()
  nx.set_node_attributes(network2, {j: 2 for j in inferences}, 'virtual')
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
  inferred_nodes = [node for node, value in nx.get_node_attributes(network, 'virtual').items() if value == 2]
  return {species: inferred_nodes} if species is not None else inferred_nodes


def export_network(network, filepath, other_node_attributes = None):
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
  # Generate edge_list
  if network.edges():
    edge_list = pd.DataFrame([(u, v, d.get('diffs', '')) for u, v, d in network.edges(data = True)], columns = ['Source', 'Target', 'Label'])
    edge_list.to_csv(f"{filepath}_edge_list.csv", index = False)
  else:
    pd.DataFrame(columns = ['Source', 'Target', 'Label']).to_csv(f"{filepath}_edge_list.csv", index = False)

  # Generate node_labels
  node_labels_dict = {'Id': list(network.nodes()), 'Virtual': [d.get('virtual', '') for _, d in network.nodes(data = True)]}
  if other_node_attributes is not None:
    for att in other_node_attributes:
      node_labels_dict[att] = [d.get(att, '') for _, d in network.nodes(data = True)]
  pd.DataFrame(node_labels_dict).to_csv(f"{filepath}_node_labels.csv", index = False)


def monolink_to_glycoenzyme(edge_label, df, enzyme_column = 'glycoenzyme',
                            monolink_column = 'monolink', mode = 'condensed'):
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
  if mode == 'condensed':
    enzyme_column = 'glycoclass'
  new_edge_label = df.loc[df[monolink_column] == edge_label, enzyme_column].unique()
  return '_'.join(new_edge_label) if new_edge_label.size else edge_label


def choose_path(diamond, species_list, network_dic, threshold = 0., nb_intermediates = 2, mode = 'presence'):
  """given a shape such as diamonds in biosynthetic networks (A->B,A->C,B->D,C->D), which path is taken from A to D?\n
  | Arguments:
  | :-
  | diamond (dict): dictionary of form position-in-diamond : glycan-string, describing the diamond
  | species_list (list): list of species to compare network to
  | network_dic (dict): dictionary of form species name : biosynthetic network (gained from construct_network)
  | threshold (float): everything below or equal to that threshold will be cut; default:0.
  | nb_intermediates (int): number of intermediate nodes expected in a network motif to extract; has to be a multiple of 2 (2: diamond, 4: hexagon,...)
  | mode (string): whether to analyze for "presence" or "abundance" of intermediates; default:"presence"\n
  | Returns:
  | :-
  | Returns dictionary of each intermediary path and its experimental support (0-1) across species
  """
  graph_dic = {k: glycan_to_nxGraph(k) for k in diamond.values()}
  # Construct the diamond between source and target node
  network, _ = create_adjacency_matrix(list(diamond.values()), graph_dic)
  source, target = diamond[1], diamond[1 + int((nb_intermediates/2) + 1)]
  # Get the intermediate nodes constituting the alternative paths
  alternatives = [tuple(path[1:int((nb_intermediates/2)+1)]) for path in nx.all_simple_paths(network, source = source, target = target)]
  if len(alternatives) < 2:
    return {}
  alts = {alt: [] for alt in alternatives}
  # For each species, check whether an alternative has been observed
  for species in species_list:
    temp_network = network_dic[species]
    temp_network_nodes = set(temp_network.nodes())
    if source in temp_network_nodes and target in temp_network_nodes:
      lookup_dic = nx.get_node_attributes(temp_network, 'virtual') if mode == 'presence' else nx.get_node_attributes(temp_network, 'abundance')
      for alt in alternatives:
        if set(alt).issubset(temp_network_nodes):
          alts[alt].append(np.mean([lookup_dic.get(node, 0) for node in alt]))
        else:
          alts[alt].append(1 if mode == 'presence' else 0)
  alts = {k: 1 - np.mean(v) if mode == 'presence' else np.mean(v) for k, v in alts.items()}
  # If both alternatives aren't observed, add minimum value (because one of the paths *has* to be taken)
  if sum(alts.values()) == 0:
    alts = {k: v + 0.01 + threshold for k, v in alts.items()}
  # Will be reworked in the future
  return {k[0]: v for k, v in alts.items()} if nb_intermediates == 2 else alts


def find_diamonds(network, nb_intermediates = 2, mode = 'presence'):
  """automatically extracts shapes such as diamonds (A->B,A->C,B->D,C->D) from biosynthetic networks\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | nb_intermediates (int): number of intermediate nodes expected in a network motif to extract; has to be a multiple of 2 (2: diamond, 4: hexagon,...)
  | mode (string): whether to analyze for "presence" or "abundance" of intermediates; default:"presence"\n
  | Returns:
  | :-
  | Returns list of dictionaries, with each dictionary a network motif in the form of position in the motif : glycan
  """
  if nb_intermediates % 2 > 0:
    print("nb_intermediates has to be a multiple of 2; please re-try.")
    return []
  # Generate matching motif
  path_length = nb_intermediates // 2 + 2 # The length of one of the two paths is half of the number of intermediates + 1 source node + 1 target node
  first_path = list(range(1, path_length + 1))
  second_path = [first_path[0]] + [x + path_length - 1 for x in first_path[1:-1]] + [first_path[-1]]
  g1 = nx.DiGraph()
  nx.add_path(g1, first_path)
  nx.add_path(g1, second_path)

  # Find networks containing the matching motif
  graph_pair = nx.algorithms.isomorphism.GraphMatcher(network, g1)
  # Find diamonds within those networks
  matchings_list = list(graph_pair.subgraph_isomorphisms_iter())
  unique_keys = {tuple(sorted(d.keys())) for d in matchings_list}
  # Map found diamonds to the same format
  matchings_list = [{v: k for k, v in next(d for d in matchings_list if tuple(sorted(d.keys())) == key).items()} for key in unique_keys]
  final_matchings = []
  # For each diamond, only keep the diamond if at least one of the intermediate structures is virtual
  virtual_attr = nx.get_node_attributes(network, 'virtual')
  for d in matchings_list:
    substrate_node, product_node = d[1], d[path_length]
    middle_nodes = [d[k] for k in range(2, path_length) if k != path_length]
    # For each middle node, test whether they are part of the same path as substrate node and product node (remove false positives)
    if all(nx.has_path(network, substrate_node, mn) and nx.has_path(network, mn, product_node) for mn in middle_nodes):
      virtual_states = [virtual_attr.get(mn, 0) for mn in middle_nodes]
      if any(vs == 1 for vs in virtual_states) or mode == 'abundance':
        # Filter out non-diamond shapes with any cross-connections
        if nb_intermediates > 2:
          graph_dic = {k: glycan_to_nxGraph(k) for k in d.values()}
          adj_matrix, _ = create_adjacency_matrix(list(d.values()), graph_dic)
          if all(deg == 2 for _, deg in adj_matrix.degree()):
            final_matchings.append(d)
        else:
          final_matchings.append(d)
  return final_matchings


def trace_diamonds(network, species_list, network_dic, threshold = 0., nb_intermediates = 2, mode = 'presence'):
  """extracts diamond-shape motifs from biosynthetic networks (A->B,A->C,B->D,C->D) and uses evolutionary information to determine which path is taken from A to D\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | species_list (list): list of species to compare network to
  | network_dic (dict): dictionary of form species name : biosynthetic network (gained from construct_network)
  | threshold (float): everything below or equal to that threshold will be cut; default:0.
  | nb_intermediates (int): number of intermediate nodes expected in a network motif to extract; has to be a multiple of 2 (2: diamond, 4: hexagon,...)
  | mode (string): whether to analyze for "presence" or "abundance" of intermediates; default:"presence"\n
  | Returns:
  | :-
  | Returns dataframe of each intermediary glycan and its proportion (0-1) of how often it has been experimentally observed in this path (or average abundance if mode = abundance)
  """
  # Get the diamonds
  matchings_list = find_diamonds(network, nb_intermediates = nb_intermediates, mode = mode)
  # Calculate the path probabilities
  paths = [choose_path(d, species_list, network_dic, mode = mode,
                       threshold = threshold, nb_intermediates = nb_intermediates) for d in matchings_list]
  df_out = pd.DataFrame(paths).T.mean(axis = 1).reset_index()
  df_out.columns = ['glycan', 'probability']
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
  network_out = deepcopy(network)
  node_attrs = nx.get_node_attributes(network_out, node_attr)
  # Prune nodes lower in attribute than threshold
  to_cut = {k for k, v in node_attrs.items() if v <= threshold}
  # Leave virtual nodes that are needed to retain a connected component
  to_cut = [k for k in to_cut if not any(network_out.in_degree[j] == 1 and len(j) > len(k) for j in network_out.neighbors(k))]
  network_out.remove_nodes_from(to_cut)
  # Remove virtual nodes with total degree of max 1 and an in-degree of at least 1
  nodes_to_remove = [node for node, attr in network_out.nodes(data = True)
                       if (network_out.degree[node] <= 1) and (attr.get('virtual') == 1) 
                       and (network_out.in_degree[node] > 0)]
  network_out.remove_nodes_from(nodes_to_remove)
  return network_out


def evoprune_network(network, network_dic = None, species_list = None, node_attr = 'abundance',
                     threshold = 0.01, nb_intermediates = 2, mode = 'presence'):
  """given a biosynthetic network, this function uses evolutionary relationships to prune impossible paths\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | network_dic (dict): dictionary of form species name : biosynthetic network (gained from construct_network); default:pre-computed milk networks
  | species_list (list): list of species to compare network to; default:species from pre-computed milk networks
  | node_attr (string): which (numerical) node attribute to use for pruning; default:'abundance'
  | threshold (float): everything below or equal to that threshold will be cut; default:0.01
  | nb_intermediates (int): number of intermediate nodes expected in a network motif to extract; has to be a multiple of 2 (2: diamond, 4: hexagon,...)
  | mode (string): whether to analyze for "presence" or "abundance" of intermediates; default:"presence"\n
  | Returns:
  | :-
  | Returns pruned network (with virtual node probability as a new node attribute)
  """
  if network_dic is None:
    network_dic = pickle.load(open(data_path, 'rb'))
  if species_list is None:
    species_list = list(network_dic.keys())
  # Calculate path probabilities of diamonds
  df_out = trace_diamonds(network, species_list, network_dic, threshold = threshold,
                          nb_intermediates = nb_intermediates, mode = mode)
  # Scale virtual node size by path probability
  network_out = highlight_network(network, highlight = 'abundance', abundance_df = df_out,
                                  intensity_col = 'probability')
  # Remove diamond paths below threshold
  return prune_network(network_out, node_attr = node_attr, threshold = threshold*100)


def highlight_network(network, highlight, motif = None,
                      abundance_df = None, glycan_col = 'glycan',
                      intensity_col = 'rel_intensity', conservation_df = None,
                      network_dic = None, species = None):
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
  | species (string): highlight=species; which species to highlight in a multi-species network\n
  | Returns:
  | :-
  | Returns a network with the additional 'origin' (motif/species) or 'abundance' (abundance/conservation) node attribute storing the highlight
  """
  if network_dic is None:
    network_dic = pickle.load(open(data_path, 'rb'))
  # Determine highlight validity
  if highlight not in ['motif', 'species', 'abundance', 'conservation']:
    print(f"Invalid highlight argument: {highlight}")
    return
  if highlight == 'motif' and motif is None:
    print("You have to provide a glycan motif to highlight")
    return
  elif highlight == 'species' and species is None:
    print("You have to provide a species to highlight")
    return
  elif highlight == 'abundance' and abundance_df is None:
    print("You have to provide a dataframe with glycan abundances to highlight")
    return
  elif highlight == 'conservation' and conservation_df is None:
    print("You have to provide a dataframe with species-glycan associations to check conservation")
    return
  network_out = deepcopy(network)
  # Color nodes as to whether they contain the motif (green) or not (violet)
  if highlight == 'motif':
    if motif[0] == 'r':
      motif_presence = {k: ('limegreen' if get_match(motif[1:], k) else 'darkviolet') for k in network_out.nodes()}
    elif motif[-1] == ')':
      motif_presence = {k: ('limegreen' if motif in k else 'darkviolet') for k in network_out.nodes()}
    else:
      motif_presence = {k: ('limegreen' if subgraph_isomorphism(k, motif) else 'darkviolet') for k in network_out.nodes()}
    nx.set_node_attributes(network_out, motif_presence, name = 'origin')
  # Color nodes as to whether they are from a species (green) or not (violet)
  elif highlight == 'species':
    temp_net_nodes = set(network_dic[species].nodes())
    node_presence = {k: ('limegreen' if k in temp_net_nodes else 'darkviolet') for k in network_out.nodes()}
    nx.set_node_attributes(network_out, node_presence, name = 'origin')
  # Add relative intensity values as 'abundance' node attribute used for node size scaling
  elif highlight == 'abundance':
    abundance_dict = dict(zip(abundance_df[glycan_col], abundance_df[intensity_col] * 100))
    node_abundance = {node: abundance_dict.get(node, 50) for node in network_out.nodes()}
    nx.set_node_attributes(network_out, node_abundance, name = 'abundance')
  # Add the degree of evolutionary conervation as 'abundance' node attribute used for node size scaling
  elif highlight == 'conservation':
    species_list = conservation_df['Species'].unique().tolist()
    spec_nodes = {k: set(network_dic[k].nodes()) for k in species_list}
    flat_spec_nodes = [node for nodes in spec_nodes.values() for node in nodes]
    node_conservation = {k: flat_spec_nodes.count(k) * 100 for k in network_out.nodes()}
    nx.set_node_attributes(network_out, node_conservation, name = 'abundance')
  return network_out


def get_edge_weight_by_abundance(network, root = "Gal(b1-4)Glc-ol", root_default = 10.0):
  """estimate reaction capacity by the relative abundance of source and sink\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | root (string): root node of network; default:"Gal(b1-4)Glc-ol"
  | root_default (float): dummy abundance value of root; default:10.0\n
  | Returns:
  | :-
  | Returns a network in which the edge attribute 'capacity' has been estimated
  """
  abundance_dict = nx.get_node_attributes(network, 'abundance')
  if abundance_dict.get(root, 1) < 0.1:
    abundance_dict[root] = root_default
  for u, v in network.edges():
    source_abundance = abundance_dict.get(u, 0.1)
    sink_abundance = abundance_dict.get(v, 0.1)
    network[u][v]['capacity'] = (source_abundance + sink_abundance) / 2
  return network


def estimate_weights(network, root = "Gal(b1-4)Glc-ol", root_default = 10, min_default = 0.001):
  """estimate reaction capacity of edges and missing relative abundances\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | root (string): root node of network; default:"Gal(b1-4)Glc-ol"
  | root_default (float): dummy abundance value of root; default:10.0
  | min_default (float): a small default value if no non-zero neighboring weights; default: 0.001\n
  | Returns:
  | :-
  | Returns a network in which the edge attribute 'capacity' and missing abundances have been estimated
  """
  net_estimated = get_edge_weight_by_abundance(network, root = root, root_default = root_default)
  # Function to estimate weight based on neighboring edges
  def estimate_weight(node):
    in_weights = [network[u][node]['capacity'] for u in network.predecessors(node) if network[u][node]['capacity'] != 0]
    out_weights = [network[node][v]['capacity'] for v in network.successors(node) if network[node][v]['capacity'] != 0]
    return np.mean(in_weights + out_weights) if in_weights or out_weights else min_default  # Small default value if no non-zero neighboring weights
  # Estimate weights for zero-weight intermediates
  zero_weight_nodes = [node for node in net_estimated.nodes if all(net_estimated[node][v]['capacity'] == 0 for v in net_estimated.successors(node))]
  for node in zero_weight_nodes:
    estimated_weight = estimate_weight(node)
    for v in net_estimated.successors(node):
      net_estimated[node][v]['capacity'] = estimated_weight
  return net_estimated


def get_maximum_flow(network, source = "Gal(b1-4)Glc-ol", sinks = None):
  """estimate maximum flow and flow paths between source and sinks\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | source (string): usually the root node of network; default:"Gal(b1-4)Glc-ol"
  | sinks (list of strings): specified sinks to estimate flow for; default:all terminal nodes\n
  | Returns:
  | :-
  | Returns a dictionary of type sink : {maximum flow value, flow path dictionary}
  """
  if sinks is None:
    sinks = [node for node, out_degree in network.out_degree() if out_degree == 0 and nx.has_path(network, source, node)]
  # Dictionary to store flow values and paths for each sink
  flow_results = {}
  for sink in sinks:
    path_length = nx.shortest_path_length(network, source = source, target = sink)
    try:
      flow_value, flow_dict = nx.maximum_flow(network, source, sink)
      flow_results[sink] = {
          'flow_value': flow_value * path_length,
          'flow_dict': flow_dict
          }
    except nx.NetworkXError:
      print(f"{sink} cannot be reached.")
  return flow_results


def get_max_flow_path(network, flow_dict, sink, source = "Gal(b1-4)Glc-ol"):
  """get the actual path between source and sink that gave rise to the maximum flow value\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | flow_dict (dict): dictionary of type source : {sink : flow} as returned by get_maximum_flow
  | sink (string): specified sink to retrieve maximum flow path
  | source (string): usually the root node of network; default:"Gal(b1-4)Glc-ol"\n
  | Returns:
  | :-
  | Returns a list of (source, sink) tuples describing the maximum flow path
  """
  path = []
  current_node = source
  abundance_dict = nx.get_node_attributes(network, 'abundance')
  while current_node != sink:
    max_metric = 0
    next_node = max(network.neighbors(current_node),
            key = lambda neighbor: flow_dict[current_node][neighbor] * max(abundance_dict.get(neighbor, 0), 0.1),
            default = None)
    if next_node is None:
      raise ValueError("No path found")
    path.append((current_node, next_node))
    current_node = next_node
  return path


def get_reaction_flow(network, res, aggregate = None):
  """get the aggregated flows for a type of reaction across entire network\n
  | Arguments:
  | :-
  | network (networkx object): biosynthetic network, returned from construct_network
  | res (dict): dictionary of type sink : {maximum flow value, flow path dictionary} as returned by get_maximum_flow
  | aggregate (string): if reaction flow values should be aggregated, options are "sum" and "mean"; default:None\n
  | Returns:
  | :-
  | Returns a dictionary of form reaction : flow(s)
  """
  # Aggregate flow values by reaction type
  reaction_flows = defaultdict(list)
  for sink_data in res.values():
    flow_dict = sink_data['flow_dict']
    for u, v, data in network.edges(data = True):
      reaction_flows[data['diffs']].append(flow_dict[u][v])
  if aggregate == "sum":
    reaction_flows = {k: sum(v) for k, v in reaction_flows.items()}
  elif aggregate == "mean":
    reaction_flows = {k: np.mean(v) for k, v in reaction_flows.items()}
  return reaction_flows


def get_differential_biosynthesis(df, group1, group2, analysis = "reaction", paired = False):
  """compares biosynthetic patterns between glycomes of two conditions\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing glycan sequences in first column and relative abundances in subsequent columns [alternative: filepath to .csv]
  | group1 (list): list of column indices or names for the first group of samples, usually the control
  | group2 (list): list of column indices or names for the second group of samples
  | analysis (string): what type of analysis to perform on networks, options are "reaction" for reaction type fluxes and "flow" for comparing flow to sinks; default:"reaction"
  | paired (bool): whether samples are paired or not (e.g., tumor & tumor-adjacent tissue from same patient); default:False\n
  | Returns:
  | :-
  | Returns a dataframe with:
  | (i) Differential flow features
  | (ii) Their mean abundance across all samples in group1 + group2
  | (iii) Log2-transformed fold change of group2 vs group1 (i.e., negative = lower in group2)
  | (iv) Uncorrected p-values (Welch's t-test) for difference in mean
  | (v) Corrected p-values (Welch's t-test with Benjamini-Hochberg correction) for difference in mean
  | (vi) Significance: True/False of whether the corrected p-value lies below the sample size-appropriate significance threshold
  | (vii) Effect size as Cohen's d
  """
  if paired:
    assert len(group1) == len(group2), "For paired samples, the size of group1 and group2 should be the same"
  if isinstance(df, str):
    df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
  if not isinstance(group1[0], str):
    columns_list = df.columns.tolist()
    group1 = [columns_list[k] for k in group1]
    group2 = [columns_list[k] for k in group2]
  all_groups = group1 + group2
  df = df.loc[:, [df.columns.tolist()[0]] + all_groups].fillna(0)
  # Sample-size aware alpha via Bayesian-Adaptive Alpha Adjustment
  alpha = get_alphaN(df.shape[1] - 1)
  df = df.set_index(df.columns.tolist()[0])
  df = df[df.any(axis = 1)]
  df = (df / df.sum(axis = 0)) * 100
  root = list(infer_roots(frozenset(df.index.tolist())))
  root = max(root, key = len) if '-ol' not in root[0] else min(root, key = len)
  min_default = 0.1 if root.endswith('GlcNAc') else 0.001
  core_net = construct_network(df.index.tolist())
  nets = {}
  for col in all_groups:
    temp = deepcopy(core_net)
    abundance_mapping = dict(zip(temp.nodes(), df[col].values.tolist()))
    nx.set_node_attributes(temp, {g: {'abundance': abundance_mapping.get(g, 0.0)} for g in temp.nodes()})
    nets[col] = estimate_weights(temp, root = root, min_default = min_default)
  res = {col: get_maximum_flow(nets[col], source = root) for col in all_groups}
  if analysis == "reaction":
    res2 = {col: get_reaction_flow(nets[col], res[col], aggregate = "sum") for col in all_groups}
    regex = re.compile(r"\(([ab])(\d)-(\d)\)")
    shadow_reactions = [regex.sub(r"(\1\2-?)", g) for g in res2[all_groups[0]]]
    shadow_reactions = {r for r in shadow_reactions if shadow_reactions.count(r) > 1}
    res2 = {k: {**v, **{r: np.mean([v2 for k2, v2 in v.items() if compare_glycans(k2, r)]) for r in shadow_reactions}} for k, v in res2.items()}
  elif analysis == "flow":
    res2 = {col: [res[col][sink]['flow_value'] for sink in res[col].keys()] for col in all_groups}
  else:
    raise ValueError("Only 'reaction' and 'flow' are currently supported analysis modes.")
  res2 = pd.DataFrame(res2).T
  res2 = res2.loc[:, res2.var(axis = 0) > 0.01]
  features = res2.columns.tolist() if analysis == "reaction" else list(res[all_groups[0]].keys())
  mean_abundance = res2.mean(axis = 0)
  df_a, df_b = res2.loc[group1, :].T, res2.loc[group2, :].T
  log2fc = np.log2((df_b.values + 1e-8) / (df_a.values + 1e-8)).mean(axis = 1) if paired else np.log2(df_b.mean(axis = 1) / df_a.mean(axis = 1))
  pvals = [ttest_rel(row_a, row_b)[1] if paired else ttest_ind(row_a, row_b, equal_var = False)[1] for row_a, row_b in zip(df_a.values, df_b.values)]
  corrpvals = multipletests(pvals, method = 'fdr_tsbh')[1] if pvals else []
  significance = [p < alpha for p in corrpvals] if pvals else []
  effect_sizes, _ = zip(*[cohen_d(row_b, row_a, paired = paired) for row_a, row_b in zip(df_a.values, df_b.values)])
  out = pd.DataFrame({'Feature': features, 'Mean abundance': mean_abundance, 'Log2FC': log2fc, 'p-val': pvals,
                      'corr p-val': corrpvals, 'significant': significance, 'Effect size': effect_sizes})
  out = out.set_index('Feature')
  return out.dropna().sort_values(by = 'p-val').sort_values(by = 'corr p-val')


def extend_glycans(glycans, reactions, allowed_disaccharides = None):
  """given a set of reactions, tries to extend observed glycans\n
  | Arguments:
  | :-
  | glycans (list or set): glycans in IUPAC-condensed format
  | reactions (list or set): reactions in the form of 'Fuc(a1-3)'
  | allowed_disaccharides (set): disaccharides that are permitted when creating possible glycans; default:not used\n
  | Returns:
  | :-
  | Returns set of new glycans that have been created with the provided reactions
  """
  new_glycans = set()
  for r in reactions:
    temp_glycans = [f'{{{r}}}{g}' for g in glycans]
    temp = (topology for g in temp_glycans for topology in get_possible_topologies(g, exhaustive = True, allowed_disaccharides = allowed_disaccharides))
    new_glycans.update(set(map(graph_to_string, temp)))
  return new_glycans


def edges_for_extension(leaf_glycans, new_glycans, graphs):
  """given leaf nodes and their extensions, creates the corresponding edges and edge labels\n
  | Arguments:
  | :-
  | leaf_glycans (set): set of glycans that are leaf nodes in network
  | new_glycans (set): set of glycans that have been produced beyond leaf nodes
  | graphs (dict): dictionary of glycan : glycan graph\n
  | Returns:
  | :-
  | Returns new edges and edge labels that connect leaf_glycans and new_glycans
  """
  new_edges, new_edge_labels = [], []
  new_glycans = list(new_glycans)
  leaf_graphs = {k: safe_index(k, graphs) for k in leaf_glycans}
  results = [get_neighbors(safe_index(g, graphs), list(leaf_glycans), list(leaf_graphs.values()), min_size = 1) for g in new_glycans]
  if all(not result[0] and not result[1] for result in results):
    return [], []
  all_neighbors, _ = zip(*results)
  for j, neighbors in enumerate(all_neighbors):
    for i in neighbors:
      if i:
        new_edges.append((i, new_glycans[j]))
        new_edge_labels.append(find_diff(i, new_glycans[j], graphs))
  return new_edges, new_edge_labels


def extend_network(network, steps = 1):
  """given a biosynthetic network, tries to extend it in a physiological manner\n
  | Arguments:
  | :-
  | network (networkx): glycan biosynthetic network as returned by construct_network
  | steps (int): how many biosynthetic steps to extend the network\n
  | Returns:
  | :-
  | Returns updated network and a list of added glycans
  """
  from glycowork.glycan_data.loader import df_species
  graphs = {}
  new_glycans = set()
  classy = get_class(list(network.nodes())[0])
  glycs = df_species[df_species.Class == "Mammalia"].glycan.drop_duplicates()
  glycs = glycs[glycs.apply(get_class) == classy]
  mammal_disac = set(unwrap(map(link_find, glycs)))
  reactions = set(nx.get_edge_attributes(network, "diffs").values())
  leaf_glycans = {x for x in network.nodes() if network.out_degree(x) == 0 and network.in_degree(x) > 0}
  for _ in range(steps):
    new_leaf_glycans = extend_glycans(leaf_glycans, reactions, allowed_disaccharides = mammal_disac)
    new_edges, new_edge_labels = edges_for_extension(leaf_glycans, new_leaf_glycans, graphs)
    network = update_network(network, new_edges, edge_labels = new_edge_labels, node_labels = {k: 1 for k in new_leaf_glycans})
    leaf_glycans = new_leaf_glycans
    new_glycans.update(leaf_glycans)
  return network, new_glycans
