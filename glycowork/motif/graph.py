import re
from copy import deepcopy
from typing import Dict, List, Tuple, Set, Union, Optional, Callable
from glycowork.glycan_data.loader import unwrap, modification_map
from glycowork.motif.processing import min_process_glycans, bracket_removal, get_possible_linkages, get_possible_monosaccharides, rescue_glycans, choose_correct_isoform
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import eigsh
from functools import lru_cache, wraps


PTM_REGEX = re.compile(r"(?<!Neu)(?<=\D)\d+(?=\D)")
NEGATION_REGEX = re.compile(r'(!\w+\([^)]+\))')
MONO_PATTERN = re.compile(r"^(Hex|HexOS|HexNAc|HexNAcOS|dHex|Sia|HexA|Pen|Monosaccharide)$")


def memoize_node_match(func):
  """Memoization decorator for narrow wildcard lists"""
  cache = {}
  @wraps(func)
  def wrapper(proc):
    key = frozenset(proc)
    if key not in cache:
      cache[key] = func(proc)
    return cache[key]
  return wrapper


@memoize_node_match
def build_wildcard_cache(proc: set) -> dict:
    """Precompute all possible wildcard expansions"""
    return {k: get_possible_linkages(k) for k in proc if '?' in k or '/' in k} | \
           {k: get_possible_monosaccharides(k) for k in proc if MONO_PATTERN.match(k) or k.startswith('!')}


@lru_cache(maxsize = 1024)
def evaluate_adjacency(glycan_part: str, # Residual part of a glycan from within glycan_to_graph
                      adjustment: int    # Number of characters to allow for extra length (consequence of tokenizing glycoletters)
                     ) -> bool: # True if adjacent, False if not
  "Checks whether two glycoletters are adjacent in the graph-to-be-constructed"
  last_char = glycan_part[-1]
  len_glycan_part = len(glycan_part)
  # Check whether (i) glycoletters are adjacent in the main chain or (ii) whether glycoletters are connected but separated by a branch delimiter
  return ((last_char in {'(', ')'} and len_glycan_part < 2+adjustment) or
          (last_char == ']' and glycan_part[-2] in {'(', ')'} and len_glycan_part-1 < 2+adjustment))


@lru_cache(maxsize = 128)
def glycan_to_graph(glycan: str  # IUPAC-condensed glycan sequence
                   ) -> Tuple[Dict[int, str], np.ndarray]: # (Dictionary of node:monosaccharide/linkage, Adjacency matrix)
  "Convert glycans into graphs"
  # Get glycoletters
  glycan_proc = tuple(min_process_glycans([glycan])[0])
  n = len(glycan_proc)
  # Map glycoletters to integers
  mask_dic = dict(enumerate(glycan_proc))
  for i, gl in mask_dic.items():
    glycan = glycan.replace(gl, str(i), 1)
  # Initialize adjacency matrix
  adj_matrix = np.zeros((n, n), dtype = np.uint8)
  # Caching index positions
  glycan_indices = {str(i): glycan.index(str(i)) for i in range(n)}
  # Loop through each pair of glycoletters
  for k in range(n):
    # Integers that are in place of glycoletters go up from 1 character (0-9) to 3 characters (>99)
    adjustment = 2 if k >= 100 else 1 if k >= 10 else 0
    adjustment2 = 2 + adjustment
    cache_first_part = glycan_indices[str(k)]+1
    for j in range(k+1, n):
      # Subset the part of the glycan that is bookended by k and j
      glycan_part = glycan[cache_first_part:glycan_indices[str(j)]]
      # Immediately adjacent residues
      if evaluate_adjacency(glycan_part, adjustment):
        adj_matrix[k, j] = 1
        continue
      # Adjacent residues separated by branches in the string
      res = bracket_removal(glycan_part)
      if len(res) <= adjustment2 and evaluate_adjacency(res, adjustment):
        adj_matrix[k, j] = 1
  return mask_dic, adj_matrix


def glycan_to_nxGraph_int(glycan: str, # Glycan in IUPAC-condensed format
                         libr: Optional[Dict[str, int]] = None, # Dictionary of form glycoletter:index
                         termini: str = 'ignore', # How to encode terminal/internal position; options: ignore, calc, provided
                         termini_list: Optional[List[str]] = None # List of positions from terminal/internal/flexible
                        ) -> nx.Graph: # NetworkX graph object of glycan
  "Convert glycans into networkx graphs"
  # This allows to make glycan graphs of motifs ending in a linkage
  cache = glycan.endswith(')')
  if cache:
    glycan += 'Hex'
  # Map glycan string to node labels and adjacency matrix
  node_dict, adj_matrix = glycan_to_graph(glycan)
  # Convert adjacency matrix to networkx graph
  if len(node_dict) > 1:
    g1 = nx.from_numpy_array(adj_matrix)
    # Needed for compatibility with monosaccharide-only graphs (size = 1)
    for _, _, d in g1.edges(data = True):
      d.pop('weight', None)
  else:
    g1 = nx.Graph()
    g1.add_node(0)
  # Remove the helper monosaccharide if used
  if cache and glycan.endswith('x'):
    node_to_remove = max(g1.nodes())
    node_dict.pop(node_to_remove, None)
    g1.remove_node(node_to_remove)
  # Add node labels
  if libr is None:
    node_attributes = {i: {'string_labels': k} for i, k in enumerate(node_dict.values())}
  else:
    node_attributes = {i: {'labels': libr[k], 'string_labels': k} for i, k in enumerate(node_dict.values())}
  nx.set_node_attributes(g1, node_attributes)
  if termini == 'calc':
    last_node = max(g1.nodes())
    nx.set_node_attributes(g1, {k: 'terminal' if g1.degree[k] == 1 or k == last_node else 'internal' for k in g1.nodes()}, 'termini')
  elif termini == 'provided':
    nx.set_node_attributes(g1, dict(zip(g1.nodes(), termini_list)), 'termini')
  return g1


@rescue_glycans
def glycan_to_nxGraph(glycan: str, # Glycan in IUPAC-condensed format
                      libr: Optional[Dict[str, int]] = None, # Dictionary of form glycoletter:index
                      termini: str = 'ignore', # How to encode terminal/internal position; options: ignore, calc, provided
                      termini_list: Optional[List[str]] = None # List of positions from terminal/internal/flexible
                     ) -> nx.Graph: # NetworkX graph object of glycan
  "Wrapper for converting glycans into networkx graphs; also works with floating substituents"
  if not glycan:
    return nx.Graph()
  if any([k in glycan for k in [';', '-D-', 'RES', '=']]):
    raise Exception
  termini_list = expand_termini_list(glycan, termini_list) if termini_list else None
  if '{' in glycan:
    parts = [glycan_to_nxGraph_int(k, libr = libr, termini = termini,
                                   termini_list = termini_list) for k in glycan.replace('}', '{').split('{') if k]
    len_org = len(parts[-1])
    for i, p in enumerate(parts[:-1]):
      parts[i] = nx.relabel_nodes(p, {pn: pn+len_org for pn in p.nodes()})
      len_org += len(p)
    g1 = nx.compose_all(parts)
  else:
    g1 = glycan_to_nxGraph_int(glycan, libr = libr, termini = termini,
                               termini_list = termini_list)
  return g1


def ensure_graph(glycan: Union[str, nx.Graph], # Glycan in IUPAC-condensed format or as networkx graph
                **kwargs # Keyword arguments passed to glycan_to_nxGraph
               ) -> nx.Graph: # NetworkX graph object of glycan
  "Ensures function compatibility with string glycans and graph glycans"
  return glycan_to_nxGraph(glycan, **kwargs) if isinstance(glycan, str) else glycan


def categorical_node_match_wildcard(attr: Union[str, Tuple[str, ...]], # Attribute or tuple of attributes to match
                                  default: str, # Default value if attribute is missing
                                  narrow_wildcard_list: Dict[str, Set[str]], # Dictionary mapping wildcards to possible matches
                                  attr2: str, # Secondary attribute to match
                                  default2: str # Default value for secondary attribute
                                 ) -> Callable[[dict, dict], bool]: # Function that matches node attributes
  "Match nodes while handling wildcards and flexible positions"
  if isinstance(attr, str):
    def match(data1, data2):
      data1_labels2, data2_labels2 = data1.get(attr2, default2), data2.get(attr2, default2)
      if data1_labels2 != data2_labels2 and 'flexible' not in {data1_labels2, data2_labels2}:
        return False
      data1_labels, data2_labels = data1.get(attr, default), data2.get(attr, default)
      comb_labels = data1_labels + data2_labels
      if "Monosaccharide" in comb_labels and  '-' not in comb_labels:
        return True
      if "?1-?" in comb_labels and comb_labels.count('-') == 2:
        return True
      if data2_labels.startswith('!') and data1_labels != data2_labels[1:] and '-' not in data1_labels:
        return True
      if data1_labels in narrow_wildcard_list and data2_labels in narrow_wildcard_list[data1_labels]:
        return True
      elif data2_labels in narrow_wildcard_list and data1_labels in narrow_wildcard_list[data2_labels]:
        return True
      return data1_labels == data2_labels
  else:
    def match(data1, data2):
      return all(data1.get(a, d) == data2.get(a, d) for a, d in zip(attr, default))
  return match


def ptm_wildcard_for_graph(graph: nx.Graph # Input graph
                         ) -> nx.Graph: # Modified graph with PTM wildcards
  "Standardize PTM wildcards in graph"
  for node in graph.nodes:
    graph.nodes[node]['string_labels'] = PTM_REGEX.sub('O', graph.nodes[node]['string_labels'])
  return graph


def compare_glycans(glycan_a: Union[str, nx.Graph], # First glycan to compare
                   glycan_b: Union[str, nx.Graph], # Second glycan to compare
                   return_matches: bool = False # Whether to return node mapping between glycans
                  ) -> bool: # True if glycans are same, False if not
  "Check whether two glycans are identical"
  if glycan_a == glycan_b:
    return (True, {i: i for i in range(len(glycan_a.nodes))} if return_matches else True) if isinstance(glycan_a, nx.Graph) else (True, None) if return_matches else True
  if isinstance(glycan_a, str) and isinstance(glycan_b, str):
    if glycan_a.count('(') != glycan_b.count('('):
      return (False, None) if return_matches else False
    proc = set(unwrap(min_process_glycans([glycan_a, glycan_b])))
    if 'O' in glycan_a + glycan_b:
      glycan_a, glycan_b = [PTM_REGEX.sub('O', glycan) for glycan in (glycan_a, glycan_b)]
    g1, g2 = map(glycan_to_nxGraph, (glycan_a, glycan_b))
  else:
    if len(glycan_a.nodes) != len(glycan_b.nodes):
      return (False, None) if return_matches else False
    proc = frozenset(nx.get_node_attributes(glycan_a, "string_labels").values()) | frozenset(nx.get_node_attributes(glycan_b, "string_labels").values())
    g1, g2 = (ptm_wildcard_for_graph(deepcopy(glycan_a)), ptm_wildcard_for_graph(deepcopy(glycan_b))) if 'O' in ''.join(proc) else (glycan_a, glycan_b)
  narrow_wildcard_list = build_wildcard_cache(proc)
  if narrow_wildcard_list:
    matcher = nx.isomorphism.GraphMatcher(g1, g2, categorical_node_match_wildcard('string_labels', 'unknown', narrow_wildcard_list, 'termini', 'flexible'))
  else:
    # First check whether components of both glycan graphs are identical, then check graph isomorphism (costly)
    if sorted(nx.get_node_attributes(g1, "string_labels").values()) == sorted(nx.get_node_attributes(g2, "string_labels").values()):
      matcher = nx.isomorphism.GraphMatcher(g1, g2, nx.algorithms.isomorphism.categorical_node_match('string_labels', 'unknown'))
    else:
      return (False, None) if return_matches else False
  for mapping in matcher.isomorphisms_iter():
    if all(mapping[u] < mapping[v] for u, v in g1.edges):
      return (True, mapping) if return_matches else True
  return (False, None) if return_matches else False


def expand_termini_list(motif: Union[str, nx.Graph], # Glycan motif sequence or graph
                       termini_list: List[str] # List of monosaccharide/linkage positions from terminal/internal/flexible
                      ) -> List[str]: # Expanded termini list including linkages
  "Convert monosaccharide-only termini list into full termini list"
  if len(termini_list[0]) < 2:
    mapping = {'t': 'terminal', 'i': 'internal', 'f': 'flexible'}
    termini_list = [mapping[t] for t in termini_list]
  num_linkages = motif.count('(') if isinstance(motif, str) else (len(motif) - 1) // 2
  result = ['flexible'] * (len(termini_list) + num_linkages)
  result[::2] = termini_list
  return result


def handle_negation(original_func: Callable # Function to wrap
                  ) -> Callable: # Wrapped function handling negation
  "Decorator for handling negation patterns in glycan matching functions"
  @wraps(original_func)
  def wrapper(glycan, motif, *args, **kwargs):
    if isinstance(motif, str) and '!' in motif:
      return subgraph_isomorphism_with_negation(glycan, motif, *args, **kwargs)
    elif hasattr(motif, 'nodes') and any('!' in data.get('string_labels', '') for _, data in motif.nodes(data = True)):
      return subgraph_isomorphism_with_negation(glycan, motif, *args, **kwargs)
    else:
      return original_func(glycan, motif, *args, **kwargs)
  return wrapper


@handle_negation
def subgraph_isomorphism(glycan: Union[str, nx.Graph], # Glycan sequence or graph
                        motif: Union[str, nx.Graph], # Glycan motif sequence or graph
                        termini_list: List = [], # List of monosaccharide positions from terminal/internal/flexible
                        count: bool = False, # Whether to return count instead of presence/absence
                        return_matches: bool = False # Whether to return matched subgraphs as node lists
                       ) -> Union[bool, int, Tuple[int, List[List[int]]]]: # Boolean presence, count, or (count, matches)
  "Check if motif exists as subgraph in glycan"
  if isinstance(glycan, str) and isinstance(motif, str):
    if motif.count('(') > glycan.count('('):
      return (0, []) if return_matches else 0 if count else False
    if not count and not return_matches and motif in glycan:
      return True
    motif_comp = min_process_glycans([motif, glycan])
    if 'O' in glycan + motif:
      glycan, motif = [PTM_REGEX.sub('O', g) for g in (glycan, motif)]
    g1 = glycan_to_nxGraph(glycan, termini = 'calc' if termini_list else None)
    g2 = glycan_to_nxGraph(motif, termini = 'provided' if termini_list else None, termini_list = termini_list)
  else:
    if len(glycan.nodes) < len(motif.nodes):
      return (0, []) if return_matches else 0 if count else False
    motif_comp = [nx.get_node_attributes(motif, "string_labels").values(), nx.get_node_attributes(glycan, "string_labels").values()]
    if 'O' in ''.join(unwrap(motif_comp)):
      g1, g2 = ptm_wildcard_for_graph(deepcopy(glycan)), ptm_wildcard_for_graph(deepcopy(motif))
    else:
      g1, g2 = glycan, motif
  narrow_wildcard_list = build_wildcard_cache(set(unwrap(motif_comp)))
  if termini_list or narrow_wildcard_list:
    graph_pair = nx.algorithms.isomorphism.GraphMatcher(g1, g2, node_match = categorical_node_match_wildcard('string_labels', 'unknown', narrow_wildcard_list,
                                                                                                             'termini', 'flexible'))
  else:
    g1_node_attr = set(nx.get_node_attributes(g1, "string_labels").values())
    if not set(motif_comp[0]).issubset(g1_node_attr):
      return (0, []) if return_matches else 0 if count else False
    graph_pair = nx.algorithms.isomorphism.GraphMatcher(g1, g2, node_match = nx.algorithms.isomorphism.categorical_node_match('string_labels', 'unknown'))
  # Count motif occurrence
  valid_mappings = []
  if graph_pair.subgraph_is_isomorphic():
    g2_edges = list(g2.edges())
    for mapping in graph_pair.subgraph_isomorphisms_iter():  # ensure directionality
      inverted_mapping = {v: k for k, v in mapping.items()}  # motif node -> glycan node
      if all(inverted_mapping[node] < inverted_mapping[neighbor] for node, neighbor in g2_edges):
        if not return_matches and not count:
          return True
        valid_mappings.append(list(mapping.keys()))
  if count:
    total = len(valid_mappings)
    return (total, valid_mappings) if return_matches else total
  elif return_matches:
    return (1 if valid_mappings else 0, valid_mappings)  # (count, matches)
  else:
    return bool(valid_mappings)


def subgraph_isomorphism_with_negation(glycan: Union[str, nx.Graph], # Glycan sequence or graph
                                     motif: Union[str, nx.Graph], # Glycan motif sequence or graph
                                     termini_list: List = [], # List of monosaccharide positions
                                     count: bool = False, # Whether to return count instead of presence/absence
                                     return_matches: bool = False # Whether to return matched subgraphs as node lists
                                    ) -> Union[bool, int, Tuple[int, List[List[int]]]]: # Boolean presence, count, or (count, matches)
  "Check if motif exists as subgraph in glycan, handling negation patterns"
  if isinstance(motif, str):
    negated_part = NEGATION_REGEX.search(motif).group(1)
    to_replace = '' if motif.startswith('!') or '[!' in motif else 'Monosaccharide(?1-?)'
    motif_stub = motif.replace(negated_part, to_replace).replace('[]', '')
    negated_part_clean = glycan_to_nxGraph(negated_part.replace('!', ''))
  else:
    motif_stub = motif.copy()
    negated_nodes = {n for n, data in motif.nodes(data = True) if '!' in data.get('string_labels', '')}
    negated_nodes.update({node + 1 for node in negated_nodes if node + 1 in motif_stub})
    motif_stub.remove_nodes_from(negated_nodes)
    negated_part_clean = nx.subgraph(motif, negated_nodes)
    for node in negated_part_clean.nodes():
      negated_part_clean.nodes[node]['string_labels'] = negated_part_clean.nodes[node]['string_labels'].replace('!', '')
  res = subgraph_isomorphism.__wrapped__(glycan, motif_stub, termini_list = termini_list, count = count, return_matches = True)
  if not res[0]:
    return (0, []) if return_matches else 0 if count else False
  valid_matches = []
  negated_len = len(negated_part_clean)
  for match_nodes in res[1]:
    ggraph = glycan_to_nxGraph(glycan) if isinstance(glycan, str) else glycan.copy()
    context_nodes = set(match_nodes)
    for _ in range(negated_len):
      for node in list(context_nodes):
        context_nodes.update(list(ggraph.neighbors(node)))
    context_subgraph = ggraph.subgraph(context_nodes)
    if not subgraph_isomorphism.__wrapped__(context_subgraph, negated_part_clean):
      valid_matches.append(match_nodes)
  if count:
    total = len(valid_matches)
    return (total, valid_matches) if return_matches else total
  elif return_matches:
    return (1 if valid_matches else 0, valid_matches)
  else:
    return bool(valid_matches)


def generate_graph_features(glycan: Union[str, nx.Graph], # Glycan sequence or network graph
                          glycan_graph: bool = True, # True if input is glycan, False if network
                          label: str = 'network' # Label for output dataframe if glycan_graph=False
                         ) -> pd.DataFrame: # Dataframe of graph features
    "Compute graph features of glycan or network"
    g = ensure_graph(glycan) if glycan_graph else glycan
    glycan = label if not glycan_graph else glycan
    nbr_node_types = len(set(nx.get_node_attributes(g, "labels") if glycan_graph else g.nodes()))
    # Adjacency matrix:
    A = nx.to_numpy_array(g)
    N = A.shape[0]
    directed = nx.is_directed(g)
    connected = nx.is_connected(g)
    if N == 1:
      deg = np.array([0])
      deg_to_leaves = np.array([0])
      centralities = {'betweeness': [0.0], 'eigen': [1.0], 'close': [0.0], 'load': [0.0],
                      'harm': [0.0], 'flow': [], 'flow_edge': [], 'secorder': []}
    else:
      deg = np.sum(A, axis = 1)
      deg_to_leaves = np.array([np.sum(A[:, deg == 1]) for i in range(N)])
      centralities = {'betweeness': list(nx.betweenness_centrality(g).values()), 'eigen': list(nx.katz_centrality_numpy(g).values()),
                      'close': list(nx.closeness_centrality(g).values()), 'load': list(nx.load_centrality(g).values()),
                      'harm': list(nx.harmonic_centrality(g).values()), 'flow': list(nx.current_flow_betweenness_centrality(g).values()) if not directed and connected else [],
                      'flow_edge': list(nx.edge_current_flow_betweenness_centrality(g).values()) if not directed and connected else [],
                      'secorder': list(nx.second_order_centrality(g).values()) if not directed and connected else []}
    features = {
      'diameter': np.nan if directed or not connected or N == 1 else nx.diameter(g),
      'branching': np.sum(deg > 2), 'nbrLeaves': np.sum(deg == 1),
      'avgDeg': np.mean(deg), 'varDeg': np.var(deg),
      'maxDeg': np.max(deg), 'nbrDeg4': np.sum(deg > 3),
      'max_deg_leaves': np.max(deg_to_leaves), 'mean_deg_leaves': np.mean(deg_to_leaves),
      'deg_assort': 0.0 if N == 1 else nx.degree_assortativity_coefficient(g), 'size_corona': 0 if N <= 1 else len(nx.k_corona(g, N).nodes()),
      'size_core': 0 if N <= 1 else len(nx.k_core(g, N).nodes()), 'nbr_node_types': nbr_node_types,
      'N': N, 'dens': np.sum(deg) / 2
      }
    for centr, vals in centralities.items():
      features.update({
        f"{centr}Max": np.nan if not vals else np.max(vals), f"{centr}Min": np.nan if not vals else np.min(vals),
        f"{centr}Avg": np.nan if not vals else np.mean(vals), f"{centr}Var": np.nan if not vals else np.var(vals)
        })
    if N == 1:
      features.update({'egap': 0.0, 'entropyStation': 0.0})
    else:
      M = ((A + np.diag(np.ones(N))).T / (deg + 1)).T
      eigval, vec = eigsh(M, 2, which = 'LM')
      distr = np.abs(vec[:, -1]) / sum(np.abs(vec[:, -1]))
      features.update({'egap': 1 - eigval[0], 'entropyStation': np.sum(distr * np.log(distr))})
    return pd.DataFrame(features, index = [glycan])


def neighbor_is_branchpoint(graph: nx.Graph, # Glycan graph
                          node: int # Node index to check
                         ) -> bool: # True if connected to downstream multi-branch node
  "Check if node is connected to downstream node with >2 branches"
  edges = graph.edges(node)
  edges = unwrap([e for e in edges if sum(e) > 2*node])
  edges = [graph.degree[e] for e in set(edges) if e != node]
  return max(edges, default = 0) > 3


def graph_to_string_int(graph: nx.Graph # Glycan graph
                      ) -> str: # IUPAC-condensed glycan string
  "Convert glycan graph back to IUPAC-condensed format"
  def assign_depth(G: nx.DiGraph, # Directed glycan graph
                idx: int # Node index
               ) -> float: # Depth of the node
    "Assign depth to each node in the graph recursively"
    min_depth = float("inf")
    children = list(G.neighbors(idx))
    if len(children) == 0:  # if it's a leaf node, the depth is zero
      G.nodes[idx]["depth"] = 0
      return 0
    for child in children:  # if it's not a leaf node, the depth is the minimum depth of its children + 1
      min_depth = min(assign_depth(G, child) + 1, min_depth)
    G.nodes[idx]["depth"] = min_depth
    return min_depth

  def dfs_to_string(G: nx.DiGraph, # DFS tree of glycan graph
                 idx: int # Node index
                ) -> str: # IUPAC-condensed glycan string
    "Convert the DFS tree of a glycan graph to IUPAC-condensed format recursively"
    output = graph.nodes[idx]["string_labels"]
    # put braces around the string describing linkages
    if (output[0] in "?abn" or output[0].isdigit()) and (output[-1] == "?" or output[-1].isdigit()):
      output = "(" + output + ")"
    # sort kids from shallow to deep to have deepest as main branch in the end
    children = list(sorted(G.neighbors(idx), key=lambda x: G.nodes[x]["depth"]))
    if len(children) == 0:
      return output
    for child in children[:-1]:  # iterate over all children except the last one and put them in branching brackets
      output = "[" + dfs_to_string(G, child) + "]" + output
    # put the last child in front of the output, without brackets
    output = dfs_to_string(G, children[-1]) + output
    return output

  # get the root node index, assuming the root node has the highest index
  root_idx = max(list(graph.nodes.keys()))
  # get the DFS tree of the graph
  dfs = nx.dfs_tree(graph, root_idx)
  # assign depth to each node
  assign_depth(dfs, root_idx)
  # convert the DFS tree to IUPAC-condensed format
  return dfs_to_string(dfs, root_idx)


def graph_to_string(graph: nx.Graph # Glycan graph (assumes root node is the one with the highest index)
                  ) -> str: # IUPAC-condensed glycan string
  "Convert glycan graph back to IUPAC-condensed format, handling disconnected components"
  if nx.number_connected_components(graph) > 1:
    parts = [graph.subgraph(sorted(c)) for c in nx.connected_components(graph)]
    len_org = len(parts[-1])
    for p in range(len(parts) - 1):
      H = nx.Graph()
      H.add_nodes_from(sorted(parts[p].nodes(data = True)))
      H.add_edges_from(parts[p].edges(data = True))
      parts[p] = nx.relabel_nodes(H, {pn: pn - len_org for pn in H.nodes()})
      len_org += len(H)
    parts = '}'.join(['{'+graph_to_string_int(p) for p in parts])
    return parts[:parts.rfind('{')] + parts[parts.rfind('{')+1:]
  else:
    return graph_to_string_int(graph)


def try_string_conversion(graph: nx.Graph # Glycan graph to validate
                        ) -> Optional[str]: # IUPAC string if valid, None if invalid
  "Check whether glycan graph describes a valid glycan"
  try:
    temp = graph_to_string(graph)
    temp = glycan_to_nxGraph(temp)
    return graph_to_string(temp)
  except (ValueError, IndexError):
    return None


def largest_subgraph(glycan_a: Union[str, nx.Graph], # First glycan
                    glycan_b: Union[str, nx.Graph] # Second glycan
                   ) -> str: # Largest common subgraph in IUPAC format
  "Find the largest common subgraph of two glycans"
  graph_a = ensure_graph(glycan_a)
  graph_b = ensure_graph(glycan_b)
  ismags = nx.isomorphism.ISMAGS(graph_a, graph_b,
                                 node_match = nx.algorithms.isomorphism.categorical_node_match('string_labels', 'unknown'))
  largest_common_subgraph = list(ismags.largest_common_subgraph())
  lgs = graph_a.subgraph(list(largest_common_subgraph[0].keys()))
  if nx.is_connected(lgs):
    min_num = min(lgs.nodes())
    node_dic = {k: k-min_num for k in lgs.nodes()}
    lgs = nx.relabel_nodes(lgs, node_dic)
    if lgs:
      return graph_to_string(lgs)
    else:
      return ""
  else:
    return ""


def get_possible_topologies(glycan: Union[str, nx.Graph], # Glycan with floating substituent
                          exhaustive: bool = False, # Whether to allow additions at internal positions
                          allowed_disaccharides: Optional[Set[str]] = None, # Permitted disaccharides when creating possible glycans
                          modification_map: Dict[str, Set[str]] = modification_map # Maps modifications to valid attachments
                         ) -> List[nx.Graph]: # List of possible topology graphs
  "Create possible glycan graphs given a floating substituent"
  ggraph = ensure_graph(glycan)
  parts = [ggraph.subgraph(c) for c in nx.connected_components(ggraph)]
  if len(parts) == 1:
    print("This glycan already has a defined topology; please don't use this function.")
    return [parts[0]]
  main_part, floating_part = parts[-1], parts[0]
  dangling_linkage = max(floating_part.nodes())
  is_modification = len(floating_part.nodes()) == 1
  if is_modification:
    modification = ggraph.nodes[dangling_linkage]['string_labels']
    dangling_carbon = modification[0]
  else:
    dangling_carbon = ggraph.nodes[dangling_linkage]['string_labels'][-1]
    floating_monosaccharide = dangling_linkage - 1
  topologies = []
  candidate_nodes = [k for k in list(main_part.nodes())[::2]
                     if exhaustive or (main_part.degree[k] == 1 and k != max(main_part.nodes()))]
  for k in candidate_nodes:
    neighbor_carbons = [ggraph.nodes[n]['string_labels'][-1] for n in ggraph.neighbors(k) if n < k]
    if dangling_carbon in neighbor_carbons:
      continue
    new_graph = deepcopy(ggraph)
    if is_modification:
      mono = new_graph.nodes[k]['string_labels']
      if modification_map and mono in modification_map.get(modification, set()):
        new_graph.nodes[k]['string_labels'] += modification
        new_graph.remove_node(dangling_linkage)
      else:
        continue
    else:
      new_graph.add_edge(dangling_linkage, k)
      if allowed_disaccharides is not None:
        disaccharide = (f"{new_graph.nodes[floating_monosaccharide]['string_labels']}"
                        f"({new_graph.nodes[dangling_linkage]['string_labels']})"
                        f"{new_graph.nodes[k]['string_labels']}")
        if disaccharide not in allowed_disaccharides:
          continue
    new_graph = nx.convert_node_labels_to_integers(new_graph)
    topologies.append(new_graph)
  return topologies


def possible_topology_check(glycan: Union[str, nx.Graph], # Glycan with floating substituent
                          glycans: List[Union[str, nx.Graph]], # List of glycans to check against
                          exhaustive: bool = False, # Whether to allow additions at internal positions
                          **kwargs # Arguments passed to compare_glycans
                         ) -> List[Union[str, nx.Graph]]: # List of matching glycans
  "Check whether glycan with floating substituent could match glycans from a list"
  topologies = get_possible_topologies(glycan, exhaustive = exhaustive)
  ggraphs = map(ensure_graph, glycans)
  return [g for g, ggraph in zip(glycans, ggraphs) if any(compare_glycans(t, ggraph, **kwargs) for t in topologies)]


def deduplicate_glycans(glycans: Union[List[str], Set[str]] # List/set of glycans to deduplicate
                      ) -> List[str]: # Deduplicated list of glycans
  "Remove duplicate glycans from a list/set, even if they have different strings"
  glycans = list(set(glycans) if isinstance(glycans, list) else glycans)
  if len(glycans) <= 1:
    return glycans
  ggraphs = list(map(glycan_to_nxGraph, glycans))
  n = len(glycans)
  keep = [True] * n
  for i in range(n):
    if not keep[i]:
      continue
    for j in range(i + 1, n):
      if not keep[j]:
        continue
      if compare_glycans(ggraphs[i], ggraphs[j]):
        correct_glycan = choose_correct_isoform(glycans[i], glycans[j])
        if correct_glycan == glycans[i]:
          keep[j] = False
        else:
          keep[i] = False
          break
  return [g for i, g in enumerate(glycans) if keep[i]]
