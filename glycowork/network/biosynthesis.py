import pickle
import re
from pathlib import Path
from copy import deepcopy
from functools import lru_cache
from importlib import resources
from collections import defaultdict, Counter
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import networkx as nx
import numpy as np
import pandas as pd
from glycowork.glycan_data.loader import unwrap, linkages, lib
from glycowork.glycan_data.stats import cohen_d, get_alphaN
from glycowork.motif.graph import compare_glycans, glycan_to_nxGraph, graph_to_string, graph_to_string_int, subgraph_isomorphism, get_possible_topologies
from glycowork.motif.processing import get_lib, rescue_glycans, in_lib, get_class, canonicalize_iupac
from glycowork.motif.tokenization import get_stem_lib, glycan_to_composition, map_to_basic
from glycowork.motif.regex import get_match
from glycowork.motif.annotate import get_k_saccharides

# Get the directory and filename of the current script and construct the data path
this_dir = Path(__file__).parent
this_filename = Path(__file__).name
data_path = this_dir / 'milk_networks_exhaustive.pkl'

def __getattr__(name):
  if name == "net_dic":
    net_dic = pickle.load(open(data_path, 'rb'))
    globals()[name] = net_dic  # Cache it to avoid reloading
    return net_dic
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

permitted_roots = frozenset({"Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"})
allowed_ptms = frozenset({'OS', '3S', '6S', 'OP', '1P', '3P', '6P', 'OAc', '4Ac', '9Ac'})


@lru_cache(maxsize = 1024)
def safe_compare(g1: nx.DiGraph, # First glycan graph
                g2: nx.DiGraph # Second glycan graph
               ) -> bool: # True if glycans are same
  "Compare glycans with error catch, returning False on exception"
  try:
    return compare_glycans(g1, g2)
  except Exception:
    return False


def safe_index(glycan: str, # Glycan in IUPAC-condensed format
              graph_dic: dict[str, nx.DiGraph] # Dictionary of glycan:graph mappings
             ) -> nx.Graph: # Glycan graph
  "Retrieve glycan graph from dictionary or calculate if not present"
  if glycan not in graph_dic:
    graph_dic[glycan] = glycan_to_nxGraph(glycan)
  return graph_dic[glycan]


@lru_cache(maxsize = 1024)
def create_neighbors(ggraph: nx.DiGraph, # Glycan graph
                    min_size: int = 1 # Minimum root size; default:1
                   ) -> list[nx.Graph]: # List of precursor graphs
  "Create biosynthetic precursor glycans"
  num_nodes = len(ggraph)
  # Check whether glycan large enough to allow for precursors
  if num_nodes <= min_size:
    return []
  if num_nodes == 3:
    return [nx.relabel_nodes(ggraph.subgraph([2]), {2: 0})]
  # Generate all precursors by iteratively cleaving off the non-reducing-end monosaccharides
  terminal_pairs = [frozenset({k, next(ggraph.predecessors(k))}) for k in ggraph.nodes() if ggraph.out_degree(k) == 0 and ggraph.in_degree(k) > 0]
  nodes = frozenset(ggraph)
  # Cleaving off messes with the node labeling, so they have to be re-labeled
  return [
        nx.relabel_nodes(ggraph.subgraph(nodes - pair), {m: i for i, m in enumerate(nodes - pair)})
        for pair in terminal_pairs
    ]


@lru_cache(maxsize = 2048)
def find_diff(glycan_a: str, # First glycan
             glycan_b: str, # Second glycan
             allowed_ptms: frozenset[str] = allowed_ptms # Set of allowed PTMs
            ) -> str: # Differing subgraph or 'disregard'
  "Find connected subgraph differing between glycans"
  # Check whether the glycans are equal
  if glycan_a == glycan_b:
    return ""
  # Convert to graphs and get difference
  larger, smaller = max(glycan_a, glycan_b, key = len), min(glycan_a, glycan_b, key = len)
  graph_larger, graph_smaller = glycan_to_nxGraph(larger), glycan_to_nxGraph(smaller)
  if len(graph_larger) - len(graph_smaller) == 2:
    smaller_canonical = graph_to_string(graph_smaller)
    for term_node in (n for n in graph_larger.nodes() if
                      graph_larger.out_degree(n) == 0 and graph_larger.in_degree(n) > 0):
      link_node = next(graph_larger.predecessors(term_node))
      remaining = set(graph_larger.nodes()) - {term_node, link_node}
      if graph_to_string(graph_larger.subgraph(remaining)) == smaller_canonical:
        diff_string = graph_to_string_int(graph_larger.subgraph({term_node, link_node}))
        return 'disregard' if any(ptm in diff_string for ptm in allowed_ptms) else diff_string
    # String comparison missed (wildcards/ambiguity), fall back to VF2
  matched = subgraph_isomorphism(graph_larger, graph_smaller, return_matches = True)
  if not isinstance(matched, bool) and matched[0]:
    diff_nodes = set(graph_larger) - set(matched[1][0])
    if not diff_nodes:
      return 'disregard'
    diff_string = graph_to_string_int(graph_larger.subgraph(diff_nodes))
    return 'disregard' if any(ptm in diff_string for ptm in allowed_ptms) else diff_string
  return 'disregard'


def filter_disregard(network: nx.DiGraph, # Biosynthetic network
                    attr: str = 'diffs', # Edge attribute name
                    value: str = 'disregard' # Value to filter
                   ) -> nx.Graph: # Filtered network
  "Filter out mistaken edges"
  network.remove_edges_from([(u, v) for u, v, data in network.edges(data = True) if data.get(attr) == value])
  return network


def stemify_glycan_fast(ggraph_in: nx.DiGraph, # Input glycan graph
                       stem_lib: dict[str, str] # Modified-to-core monosaccharide mapping
                      ) -> tuple[str, nx.DiGraph]: # (Stemified glycan, Stemified graph)
  "Stemify glycan graph quickly using precomputed mapping"
  ggraph = deepcopy(ggraph_in)
  nx.set_node_attributes(ggraph, {k: {"string_labels": stem_lib[v]} for k, v in ggraph.nodes(data = "string_labels")})
  return graph_to_string(ggraph), ggraph


def find_ptm(glycan: str, # Glycan with PTM
            glycans: list[str], # List of glycans
            graph_dic: dict[str, nx.DiGraph], # Dictionary of glycan:graph mappings
            stem_lib: dict[str, str], # Modified-to-core monosaccharide mapping
            allowed_ptms: frozenset[str] = allowed_ptms, # Set of allowed PTMs
            ggraphs: list[nx.DiGraph] | None = None, # Precomputed glycan graphs
            suffix: str = '-ol' # Glycan suffix
           ) -> tuple[tuple[str, str], str] | int: # Edge tuple (glycan, precursor) and PTM or 0
  "Identify precursor glycans for glycan with PTM"
  # Checks which PTM(s) are present
  mod = next((ptm for ptm in allowed_ptms if ptm in glycan), None)
  if mod is None:
    return 0
  # Stemifying returns the unmodified glycan
  glycan_stem, g_stem = stemify_glycan_fast(safe_index(glycan, graph_dic), stem_lib)
  glycan_stem += suffix
  if suffix == '-ol':
    last_node = max(g_stem)
    g_stem.nodes[last_node]['string_labels'] += suffix
  if ggraphs is None:
    ggraphs = [safe_index(k, graph_dic) for k in glycans]
  idx = next((i for i, k in enumerate(ggraphs) if safe_compare(k, g_stem)), None)
  # If found, add a biosynthetic step of adding the PTM to the unmodified glycan
  if idx is not None:
    return (glycan, glycans[idx]), mod
  # If not, just add a virtual node of the unmodified glycan+corresponding edge; will be connected to the main network later; will not properly work if glycan has multiple PTMs
  if ''.join(sorted(glycan)) == ''.join(sorted(glycan_stem + mod)):
    return (glycan, glycan_stem), mod
  return 0


def process_ptm(glycans: list[str], # List of glycans
               graph_dic: dict[str, nx.DiGraph], # Dictionary of glycan:graph mappings
               stem_lib: dict[str, str], # Modified-to-core monosaccharide mapping
               allowed_ptms: frozenset[str] = allowed_ptms, # Set of allowed PTMs
               suffix: str = '-ol' # Glycan suffix
              ) -> tuple[list, list]: # (Edge tuples (glycan, precursor), PTM labels)
  "Find PTM-containing glycans and their precursors"
  # Get glycans with PTMs and convert them to graphs
  ptm_glycans = [glycan for glycan in glycans if any(ptm in glycan for ptm in allowed_ptms)]
  ggraphs = [safe_index(k, graph_dic) for k in glycans]
  # Connect modified glycans to their unmodified counterparts
  edges = [find_ptm(k, glycans, graph_dic, stem_lib, allowed_ptms = allowed_ptms,
                    ggraphs = ggraphs, suffix = suffix) for k in ptm_glycans]
  valid_edges = [k for k in edges if k != 0]
  return list(zip(*valid_edges)) if valid_edges else ([], [])


def update_network(network_in: nx.DiGraph, # Input network
                  edge_list: list[tuple[str, str]], # List of edges to add
                  edge_labels: list[str] | None = None, # Labels for new edges
                  node_labels: dict[str, int] | None = None # Node virtual status (0: observed, 1: virtual)
                 ) -> nx.DiGraph: # Updated network
  "Update network with new edges and labels"
  network = network_in.copy()
  network.add_edges_from(edge_list)
  if edge_labels:
    nx.set_edge_attributes(network, dict(zip(edge_list, edge_labels)), 'diffs')
  if node_labels:
    nx.set_node_attributes(network, node_labels, 'virtual')
  else:
    nx.set_node_attributes(network, {node: 1 for node in network.nodes if 'virtual' not in network.nodes[node]}, 'virtual')
  return network


def prune_directed_edges(network: nx.DiGraph # Directed network
                       ) -> nx.DiGraph: # Pruned network
  "Remove edges going against biosynthetic direction"
  network.remove_edges_from([(u, v) for u, v in network.edges() if len(u) > len(v)])
  return network


def deorphanize_edge_labels(network: nx.DiGraph, # Biosynthetic network
                          graph_dic: dict[str, nx.DiGraph], # Dictionary of glycan:graph mappings
                          allowed_ptms: frozenset[str] = allowed_ptms # Set of allowed PTMs
                         ) -> nx.Graph: # Network with complete labels
  "Complete edge labels in newly added edges"
  edge_labels = nx.get_edge_attributes(network, 'diffs')
  for edge in (e for e in network.edges() if e not in edge_labels):
    network.edges[edge]['diffs'] = find_diff(edge[0], edge[1], allowed_ptms = allowed_ptms)
  return network


@lru_cache(maxsize = 128)
def infer_roots(glycans: frozenset[str] # Set of glycans
               ) -> frozenset[str]: # Set of permitted roots
  "Infer correct permitted roots for glycan class"
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
  elif any(k.endswith('Glc') for k in glycans):
    print("Are you working with free oligosaccharides or glycolipids? Append '-ol' or '1Cer' to your glycans, respectively. We'll pretend it's milk glycans for now")
    return frozenset({'Gal(b1-4)Glc', 'Gal(b1-4)GlcNAc'})
  print("Glycan class not detected; depending on the class, glycans should end in -ol, GalNAc, GlcNAc, or 1Cer")
  return frozenset()


def add_high_man_removal(network: nx.DiGraph # Biosynthetic network
                       ) -> nx.DiGraph: # Network with high-mannose removal
  "Add 'reverse' edges for high-mannose N-glycan maturation (i.e., cutting down)"
  edges_to_add = [(target, source, network[source][target]) for source, target in network.edges() if target.count('Man') >= 5]
  network.add_edges_from(edges_to_add, diffs = lambda x: x[2])
  return network


def _graph_fp(g: nx.DiGraph) -> tuple:
  "Wildcard-safe structural fingerprint for bucketing"
  leaves = branching = 0
  for _, d in g.out_degree():
    leaves += d == 0
    branching += d > 1
  return (len(g), leaves, branching)


def build_network_from_glycans(glycans: list[str], # Observed glycans
                               graph_dic: dict[str, nx.DiGraph], # Dictionary of glycan:graph mappings
                               min_size: int = 1, # Minimum root size; default:1
                               allowed_ptms: frozenset[str] = allowed_ptms # Set of allowed PTMs
                              ) -> tuple[nx.Graph, list[str], dict]: # (Network, Virtual nodes, Edge diffs)
  "Build biosynthetic network by BFS precursor expansion"
  network = nx.Graph()
  network.add_nodes_from(glycans)
  observed = set(glycans)
  virtual_nodes = set()
  queue = list(glycans)
  seen = set(glycans)
  observed_by_fp = defaultdict(list)
  edge_diffs = {}
  for g in glycans:
    observed_by_fp[_graph_fp(safe_index(g, graph_dic))].append(g)
  while queue:
    glycan = queue.pop()
    ggraph = safe_index(glycan, graph_dic)
    terminal_pairs = [(k, next(ggraph.predecessors(k))) for k in ggraph.nodes()
                      if ggraph.out_degree(k) == 0 and ggraph.in_degree(k) > 0]
    for prec_graph, (term_node, link_node) in zip(create_neighbors(ggraph, min_size = min_size), terminal_pairs):
      prec_str = graph_to_string(prec_graph)
      if prec_str.startswith('('):
        continue
      graph_dic.setdefault(prec_str, prec_graph)
      fp = _graph_fp(prec_graph)
      # Fast path: canonical string directly identifies observed glycan; fallback: graph isomorphism for wildcard cases
      if prec_str in observed:
        match = prec_str
      else:
        match = next((g for g in observed_by_fp.get(fp, []) if safe_compare(safe_index(g, graph_dic), prec_graph)),
                     prec_str)
      network.add_edge(glycan, match)
      diff = graph_to_string_int(ggraph.subgraph({term_node, link_node}))
      edge_diffs[(glycan, match)] = 'disregard' if any(ptm in diff for ptm in allowed_ptms) else diff
      if match not in seen:
        seen.add(match)
        virtual_nodes.add(match)
        observed_by_fp[fp].append(match)
        queue.append(match)
  return network, list(virtual_nodes), edge_diffs


@rescue_glycans
def construct_network(glycans: list[str], # List of glycans
                     allowed_ptms: frozenset[str] = allowed_ptms, # Set of allowed PTMs
                     edge_type: str = 'monolink', # Edge label type: monolink/monosaccharide/enzyme
                     permitted_roots: frozenset[str] | None = None, # Allowed root nodes
                     abundances: list[float] = [] # Glycan abundances in the same order as glycans; default:empty
                    ) -> nx.DiGraph: # Biosynthetic network
  "Construct glycan biosynthetic network"
  # Canonicalize all input strings upfront so string equality == graph isomorphism throughout
  glycans = list(set(canonicalize_iupac(g) for g in glycans))
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
  network, virtual_nodes, edge_diffs = build_network_from_glycans(glycans, graph_dic, min_size = min_size, allowed_ptms = allowed_ptms)
  # Create edge and node labels
  nx.set_edge_attributes(network, edge_diffs, 'diffs')
  nx.set_node_attributes(network, {k: 1 if k in virtual_nodes else 0 for k in network}, 'virtual')
  # Connect post-translational modifications
  suffix = '-ol' if '-ol' in ''.join(glycans) else '1Cer' if '1Cer' in ''.join(glycans) else ''
  virtual_nodes = [x for x, y in network.nodes(data = True) if y['virtual'] == 1]
  real_strings_by_size = defaultdict(set)
  for x, y in network.nodes(data = True):
    if y['virtual'] == 0:
      real_strings_by_size[len(safe_index(x, graph_dic))].add(x)
  to_remove_v = []
  for v in virtual_nodes:
    vg = safe_index(v, graph_dic)
    vsize = len(vg)
    # Fast path: node name is canonical string so direct set lookup suffices; fallback: graph isomorphism for wildcard cases
    if not in_lib(v, lib) or v in real_strings_by_size.get(vsize, set()) or \
            any(
              compare_glycans(vg, safe_index(r, graph_dic)) for r in real_strings_by_size.get(vsize, set()) if r != v):
      to_remove_v.append(v)
  network.remove_nodes_from(to_remove_v)
  ptm_links = process_ptm(list(network.nodes()), graph_dic, stem_lib, allowed_ptms = allowed_ptms, suffix = suffix)
  if ptm_links:
    network = update_network(network, ptm_links[0], edge_labels = ptm_links[1])
  # Final clean-up / condensation step
  to_remove_nodes = [node for node in sorted(network.nodes(), key = len, reverse = True) if
                     (network.degree[node] <= 1) and (network.nodes[node]['virtual'] == 1) and (
                               node not in permitted_roots)]
  network.remove_nodes_from(to_remove_nodes)
  size_buckets = defaultdict(list)
  for n in network.nodes():
    size_buckets[len(safe_index(n, graph_dic))].append(n)
  node_fps = {n: _graph_fp(safe_index(n, graph_dic)) for n in network.nodes()}
  for size, bigger in size_buckets.items():
    for n_big in bigger:
      fp_big = node_fps[n_big]
      candidates = [n_small for n_small in size_buckets.get(size - 2, [])
                    if not network.has_edge(n_big, n_small)
                    and node_fps[n_small][1] <= fp_big[1] and node_fps[n_small][2] <= fp_big[2]]
      if not candidates:
        continue
      # Lazily compute removal strings only when candidates exist; node names are canonical so direct string match suffices; fallback: graph isomorphism for wildcard cases
      g = safe_index(n_big, graph_dic)
      rs = set()
      for term_node in (nd for nd in g.nodes() if g.out_degree(nd) == 0 and g.in_degree(nd) > 0):
        link_node = next(g.predecessors(term_node))
        rs.add(graph_to_string(g.subgraph(set(g.nodes()) - {term_node, link_node})))
      for n_small in candidates:
        if n_small in rs or any(
                compare_glycans(safe_index(r, graph_dic), safe_index(n_small, graph_dic)) for r in rs):
          network.add_edge(n_big, n_small)
  network = deorphanize_edge_labels(network, graph_dic, allowed_ptms = allowed_ptms)
  for node in network:
    network.nodes[node].setdefault('virtual', 1)
  # Edge label specification
  if edge_type != 'monolink':
    df_enzyme = None
    if edge_type == 'enzyme':
      with resources.files("glycowork.network").joinpath("monolink_to_enzyme.csv").open() as f:
        df_enzyme = pd.read_csv(f, sep = '\t')
    for u, v in network.edges():
      elem = network[u][v]
      edge = elem['diffs']
      for link in linkages:
        if link in edge:
          if edge_type == 'monosaccharide':
            elem['diffs'] = edge.split('(')[0]
          elif edge_type == 'enzyme':
            elem['diffs'] = monolink_to_glycoenzyme(edge, df_enzyme)
  # Make network directed
  network = prune_directed_edges(network.to_directed())
  for node in sorted(network.nodes(), key = len):
    if (network.in_degree[node] < 1) and (network.nodes[node]['virtual'] == 1) and (node not in permitted_roots):
      network.remove_node(node)
  for node in sorted(network.nodes(), key = len, reverse = True):
    if (network.out_degree[node] < 1) and (network.nodes[node]['virtual'] == 1):
      network.remove_node(node)
  if 'GlcNAc' in ''.join(permitted_roots) and any(g.count('Man') >= 5 for g in network):
    network = add_high_man_removal(network)
  if abundances:
    nx.set_node_attributes(network, {g: {'abundance': abundance_mapping.get(g, 0.0)} for g in network.nodes()})
  return filter_disregard(network)


def plot_network(network: nx.DiGraph, # Biosynthetic network
                plot_format: str = 'hierarchical', # Layout type: hierarchical/pydot2/kamada_kawai/spring
                edge_label_draw: bool = True, # Whether to draw edge labels
                lfc_dict: dict[str, float] | None = None # Enzyme:log2FC mapping for edge width
               ) -> None: # Displays plot
  "Visualize biosynthetic network"
  from bokeh.plotting import figure, show
  from bokeh.io import output_notebook
  from bokeh.models import HoverTool, Arrow, NormalHead, LabelSet, ColumnDataSource
  try:
    output_notebook()
  except (ImportError, RuntimeError, Exception):
    pass
  if plot_format == 'hierarchical':
    roots = [n for n in network.nodes() if network.in_degree(n) == 0]
    levels = {}
    queue = [(r, 0) for r in roots]
    while queue:
      node, level = queue.pop(0)
      if node not in levels or levels[node] < level:
        levels[node] = level
        for s in network.successors(node):
          queue.append((s, level + 1))
    level_nodes = defaultdict(list)
    for node, level in levels.items():
      level_nodes[level].append(node)
    max_level = max(levels.values()) if levels else 0
    max_width = max(len(nodes) for nodes in level_nodes.values())
    pos = {}
    for level, nodes in level_nodes.items():
      for i, node in enumerate(nodes):
        pos[node] = ((i - (len(nodes)-1)/2) / max(max_width/2, 1), 1 - level/max(max_level, 1))
  else:
    pos = {'pydot2': lambda: nx.nx_pydot.pydot_layout(network, prog = "dot"),
           'kamada_kawai': lambda: nx.kamada_kawai_layout(network),
           'spring': lambda: nx.spring_layout(network, k = 0.5, seed = 42)}.get(plot_format, lambda: nx.kamada_kawai_layout(network))()
  node_attributes = nx.get_node_attributes(network, 'abundance')
  node_sizes = list(node_attributes.values()) if node_attributes else [50] * network.number_of_nodes()
  virtual_attrs = nx.get_node_attributes(network, 'virtual')
  origin_attrs = nx.get_node_attributes(network, 'origin')
  edge_attributes = nx.get_edge_attributes(network, 'diffs')
  # Prepare node data
  node_data = {'x': [], 'y': [], 'name': [], 'size': [], 'color': [], 'alpha': []}
  for i, node in enumerate(network.nodes()):
    x, y = pos[node]
    node_data['x'].append(x)
    node_data['y'].append(y)
    node_data['name'].append(str(node))
    node_data['size'].append(node_sizes[i]/2 if node_sizes else 25)
    if origin_attrs:
      node_data['color'].append(origin_attrs.get(node, 'cornflowerblue'))
    else:
      if virtual_attrs.get(node, 0) == 1:
        node_data['color'].append('darkorange')
      elif virtual_attrs.get(node, 0) == 0:
        node_data['color'].append('cornflowerblue')
      else:
        node_data['color'].append('seagreen')
    node_data['alpha'].append(0.2 if virtual_attrs.get(node, 0) == 1 else 1)
  node_source = ColumnDataSource(data = node_data)
  # Create figure
  p = figure(width = 900, height = 900, x_range = (-1.2, 1.2), y_range = (-1.2, 1.2),
             tools = "pan,wheel_zoom,box_zoom,reset,save", toolbar_location = "above",
             x_axis_type = None, y_axis_type = None, background_fill_color = "white")
  # Draw nodes with hover
  node_renderer = p.scatter('x', 'y', size = 'size', color = 'color', alpha = 'alpha', line_color = "#888", line_width = 1, source = node_source)
  p.add_tools(HoverTool(renderers = [node_renderer], tooltips = [("Node", "@name")]))
  # Draw edges and labels
  edge_data = {'xs': [], 'ys': [], 'label': [], 'color': [], 'width': []}
  for edge in network.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    attr = edge_attributes.get(edge, '')
    # Determine edge color and width
    if edge_label_draw and lfc_dict:
      if lfc_dict.get(attr, 0) < 0:
        edge_color = 'red'
      elif lfc_dict.get(attr, 0) > 0:
        edge_color = 'green'
      else:
        edge_color = 'cornflowerblue'
      width = abs(lfc_dict.get(attr, 0)) or 1
    elif not edge_label_draw:
      if 'Glc' in attr:
        edge_color = 'cornflowerblue'
      elif 'Gal' in attr:
        edge_color = 'yellow'
      elif 'Fuc' in attr:
        edge_color = 'red'
      elif '5Ac' in attr:
        edge_color = 'mediumorchid'
      elif '5Gc' in attr:
        edge_color = 'turquoise'
      else:
        edge_color = 'silver'
      width = 1 if 'b1' in attr else 3
    else:
      edge_color = 'cornflowerblue'
      width = 1
    edge_data['xs'].append([x0 + 0.15*(x1-x0), x1 - 0.15*(x1-x0)])
    edge_data['ys'].append([y0 + 0.15*(y1-y0), y1 - 0.15*(y1-y0)])
    edge_data['label'].append(attr)
    edge_data['color'].append(edge_color)
    edge_data['width'].append(width)
    # Add arrowhead
    arrow = Arrow(end = NormalHead(size = 8, fill_color = edge_color),
                 x_start = x0, y_start = y0, x_end = x1, y_end = y1, line_width = width, line_color = edge_color)
    p.add_layout(arrow)
  # Draw visible edges as segments; multi_line renderer (invisible) enables hover tooltips
  edge_source = ColumnDataSource(data = edge_data)
  edge_renderer = p.multi_line('xs', 'ys', color = 'color', line_width = 'width', source = edge_source, line_alpha = 0)
  p.add_tools(HoverTool(renderers = [edge_renderer], tooltips = [("Reaction", "@label")]))
  p.segment([x[0] for x in edge_data['xs']], [y[0] for y in edge_data['ys']],
            [x[1] for x in edge_data['xs']], [y[1] for y in edge_data['ys']],
            color = edge_data['color'], line_width = edge_data['width'])
  # Add all edge labels in one LabelSet
  if edge_label_draw:
    p.add_layout(LabelSet(x = 'x', y = 'y', text = 'text', text_color = 'black', text_font_size = '10pt',
                          x_offset = 0, y_offset = 0, text_align = 'center',
                          source = ColumnDataSource(data = dict(
                            x = [(pos[e[0]][0] + pos[e[1]][0]) / 2 for e in network.edges()],
                            y = [(pos[e[0]][1] + pos[e[1]][1]) / 2 for e in network.edges()],
                            text = [edge_attributes.get(e, '') for e in network.edges()]))))
  show(p)
  return p


def network_alignment(network_a: nx.DiGraph, # First network
                     network_b: nx.DiGraph # Second network
                    ) -> nx.DiGraph: # Combined network
  "Combine two networks into one with color coding (brown: shared, blue: only in a, orange: only in b)"""
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
  nx.set_node_attributes(U, {node: 1 if network_a.nodes[node]['virtual'] == 1 and network_b.nodes[node]['virtual'] == 1 else 0 for node in both}, 'virtual')
  # Get combined edges
  all_edges = list(network_a.edges(data = True)) + list(network_b.edges(data = True))
  U.add_edges_from(all_edges)
  nx.set_node_attributes(U, node_origin, name = 'origin')
  # If input is directed, make output directed
  if nx.is_directed(network_a):
    U = prune_directed_edges(U.to_directed())
  return U


def infer_virtual_nodes(network_a: nx.DiGraph, # First network
                       network_b: nx.DiGraph, # Second network
                       combined: nx.DiGraph | None = None # Pre-merged network from network_alignment
                      ) -> tuple[tuple[list[str], list[str]], tuple[list[str], list[str]]]: # ((a-virtuals in b, shared), (b-virtuals in a, shared))
  "Identify virtual nodes observed in other species"
  # Perform network alignment if not provided
  if combined is None:
    combined = network_alignment(network_a, network_b)
  # Find virtual nodes in network_a that are observed in network_b and vice versa
  virtual_a = {k for k, v in network_a.nodes(data = True) if v.get('virtual', 0) == 1}
  virtual_b = {k for k, v in network_b.nodes(data = True) if v.get('virtual', 0) == 1}
  virtual_combined = {k for k, v in combined.nodes(data = True) if v.get('virtual', 0) == 1}
  supported = virtual_a & virtual_b
  inferred_a = virtual_a - virtual_combined
  inferred_b = virtual_b - virtual_combined
  return (list(inferred_a), list(supported)), (list(inferred_b), list(supported))


def infer_network(network: nx.DiGraph, # Network to infer
                 network_species: str, # Source species
                 species_list: list[str], # Species to compare against
                 network_dic: dict[str, nx.DiGraph] # Species:network mapping
                ) -> nx.Graph: # Network with inferred nodes
  "Replace virtual nodes observed in other species"
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


def retrieve_inferred_nodes(network: nx.DiGraph, # Network with inferred nodes
                          species: str | None = None # Source species if multiple
                         ) -> list[str] | dict[str, list[str]]: # Inferred nodes list or dict
  "Get inferred virtual nodes from network"
  inferred_nodes = [node for node, value in nx.get_node_attributes(network, 'virtual').items() if value == 2]
  return {species: inferred_nodes} if species is not None else inferred_nodes


def export_network(network: nx.DiGraph, # Biosynthetic network
                  filepath: str, # Output path prefix, will be appended by file description and type
                  other_node_attributes: list[str] | None = None # Additional attributes for extraction
                 ) -> None: # Saves network files (edge list/labels + node IDs and labels)
  "Export network to Cytoscape/Gephi compatible files"
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


def monolink_to_glycoenzyme(edge_label: str, # Monolink edge label
                           df: pd.DataFrame, # Glycoenzyme mapping data
                           enzyme_column: str = 'glycoenzyme', # Enzyme column name
                           monolink_column: str = 'monolink', # Monolink column name
                           mode: str = 'condensed' # Output mode: condensed/full
                          ) -> str: # Enzyme label
  "Convert monosaccharide(linkage) edge label to enzyme name responsible for its synthesis"
  if mode == 'condensed':
    enzyme_column = 'glycoclass'
  new_edge_label = df.loc[df[monolink_column] == edge_label, enzyme_column].unique()
  return '_'.join(new_edge_label) if new_edge_label.size else edge_label


def choose_path(diamond: dict[int, str], # Diamond node positions mapping to glycans
               species_list: list[str], # Species to compare against
               network_dic: dict[str, nx.DiGraph], # Species:network mapping
               threshold: float = 0.0, # Cutoff threshold
               nb_intermediates: int = 2, # Number of intermediate nodes; has to be a multiple of 2
               mode: str = 'presence' # Analysis mode: presence/abundance
              ) -> dict[str, float]: # Path support scores
  "Determine preferred path through network motif across species/conditions"
  # Construct the diamond between source and target node
  path_length = nb_intermediates // 2 + 2
  source, target = diamond[1], diamond[path_length]
  alternatives = [tuple(diamond[k] for k in range(2, path_length)),
                  tuple(diamond[k] for k in range(path_length + 1, 2 * path_length - 1))]
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


def find_diamonds(network: nx.DiGraph, # Biosynthetic network
                 nb_intermediates: int = 2, # Number of intermediate nodes; has to be a multiple of 2
                 mode: str = 'presence' # Analysis mode: presence/abundance
                ) -> list[dict[int, str]]: # List of diamond motifs as diamond node positions mapping to glycans
  "Find diamond-like motifs (A->B,A->C,B->D,C->D) in biosynthetic network"
  if nb_intermediates % 2 > 0:
    raise ValueError("nb_intermediates has to be a multiple of 2; please re-try.")
  # Generate matching motif
  path_length = nb_intermediates // 2 + 2 # The length of one of the two paths is half of the number of intermediates + 1 source node + 1 target node
  first_path = list(range(1, path_length + 1))
  second_path = [first_path[0]] + [x + path_length - 1 for x in first_path[1:-1]] + [first_path[-1]]
  g1 = nx.DiGraph()
  nx.add_path(g1, first_path)
  nx.add_path(g1, second_path)
  # Find networks containing the matching motif
  graph_pair = nx.algorithms.isomorphism.DiGraphMatcher(network, g1)
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
      virtual_states = (virtual_attr.get(mn, 0) for mn in middle_nodes)
      if any(vs == 1 for vs in virtual_states) or mode == 'abundance':
        # Filter out non-diamond shapes with any cross-connections
        if nb_intermediates > 2:
          sub = network.subgraph(d.values()).to_undirected()
          if all(deg == 2 for _, deg in sub.degree()):
            final_matchings.append(d)
        else:
          final_matchings.append(d)
  return final_matchings


def trace_diamonds(network: nx.DiGraph, # Biosynthetic network
                  species_list: list[str], # Species to compare against
                  network_dic: dict[str, nx.DiGraph], # Species:network mapping
                  threshold: float = 0.0, # Cutoff threshold
                  nb_intermediates: int = 2, # Number of intermediate nodes; has to be a multiple of 2
                  mode: str = 'presence' # Analysis mode: presence/abundance
                 ) -> pd.DataFrame: # Path analysis results, with proportion (0-1) of how often glycan has been experimentally observed in this path (or average abundance)
  "Analyze diamond motif (A->B,A->C,B->D,C->D) path preferences using evolutionary data"
  # Get the diamonds
  matchings_list = find_diamonds(network, nb_intermediates = nb_intermediates, mode = mode)
  # Calculate the path probabilities
  paths = [choose_path(d, species_list, network_dic, mode = mode,
                       threshold = threshold, nb_intermediates = nb_intermediates) for d in matchings_list]
  df_out = pd.DataFrame(paths).T.mean(axis = 1).reset_index()
  df_out.columns = ['glycan', 'probability']
  return df_out


def prune_network(network: nx.DiGraph, # Biosynthetic network
                 node_attr: str = 'abundance', # Node attribute to use for pruning
                 threshold: float = 0.0 # Cutoff threshold
                ) -> nx.Graph: # Pruned network
  "Remove unlikely virtual paths from network"
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


def evoprune_network(network: nx.DiGraph, # Biosynthetic network
                    network_dic: dict[str, nx.DiGraph] | None = None, # Species:network mapping
                    species_list: list[str] | None = None, # Species to compare against
                    node_attr: str = 'abundance', # Node attribute to use for pruning
                    threshold: float = 0.01, # Cutoff threshold
                    nb_intermediates: int = 2, # Number of intermediate nodes; has to be a multiple of 2
                    mode: str = 'presence' # Analysis mode: presence/abundance
                   ) -> nx.DiGraph: # Evolutionarily pruned network (with virtual node probability as a new node attribute)
  "Prune network using evolutionary path preferences"
  if network_dic is None:
    network_dic = pickle.load(open(data_path, 'rb'))
  if species_list is None:
    species_list = list(network_dic.keys())
  # Calculate path probabilities of diamonds
  df_out = trace_diamonds(network, species_list, network_dic, threshold = threshold,
                          nb_intermediates = nb_intermediates, mode = mode)
  # Scale virtual node size by path probability
  network_out = highlight_network(network, highlight = 'abundance', abundance_df = df_out, intensity_col = 'probability')
  # Remove diamond paths below threshold
  return prune_network(network_out, node_attr = node_attr, threshold = threshold*100)


def highlight_network(network: nx.DiGraph, # Biosynthetic network
                     highlight: str, # What to highlight: motif/species/abundance/conservation
                     motif: str | None = None, # Motif to highlight; highlight=motif
                     abundance_df: pd.DataFrame | None = None, # Glycan abundance data; highlight=abundance
                     glycan_col: str = 'glycan', # Glycan column name; highlight=abundance
                     intensity_col: str = 'rel_intensity', # Intensity column name; highlight=abundance
                     conservation_df: pd.DataFrame | None = None, # Species-glycan data; highlight=conservation
                     network_dic: dict[str, nx.DiGraph] | None = None, # Species:network mapping; highlight=conservation/species
                     species: str | None = None # Species to highlight; highlight=species
                    ) -> nx.DiGraph: # Network with highlight attributes ('origin' (motif/species) or 'abundance' (abundance/conservation) node attribute)
  "Add visual highlighting to network nodes, to be used in plot_network"
  if network_dic is None:
    network_dic = pickle.load(open(data_path, 'rb'))
  # Determine highlight validity
  if highlight not in ['motif', 'species', 'abundance', 'conservation']:
    raise ValueError(f"Invalid highlight argument: {highlight}")
  if highlight == 'motif' and motif is None:
    raise ValueError("You have to provide a glycan motif to highlight")
  elif highlight == 'species' and species is None:
    raise ValueError("You have to provide a species to highlight")
  elif highlight == 'abundance' and abundance_df is None:
    raise ValueError("You have to provide a dataframe with glycan abundances to highlight")
  elif highlight == 'conservation' and conservation_df is None:
    raise ValueError("You have to provide a dataframe with species-glycan associations to check conservation")
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


def get_edge_weight_by_abundance(network: nx.DiGraph, # Biosynthetic network
                               root: str = "Gal(b1-4)Glc-ol", # Root node
                               root_default: float = 10.0 # Root abundance
                              ) -> nx.DiGraph: # Network with edge capacities
  "Estimate reaction capacity (edge attribute) from node abundances"
  abundance_dict = nx.get_node_attributes(network, 'abundance')
  if abundance_dict.get(root, 1) < 0.1:
    abundance_dict[root] = root_default
  for u, v in network.edges():
    source_abundance = abundance_dict.get(u, 0.1)
    sink_abundance = abundance_dict.get(v, 0.1)
    network[u][v]['capacity'] = (source_abundance + sink_abundance) / 2
  return network


def estimate_weights(network: nx.DiGraph, # Biosynthetic network
                    root: str = "Gal(b1-4)Glc-ol", # Root node
                    root_default: float = 10, # Root abundance
                    min_default: float = 0.001 # Minimum weight
                   ) -> nx.DiGraph: # Network with estimated weights
  "Estimate reaction capacity (edge attribute) and missing abundances"
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


def get_maximum_flow(network: nx.DiGraph, # Biosynthetic network
                    source: str = "Gal(b1-4)Glc-ol", # Source node
                    sinks: list[str] | None = None # Target nodes; default:all terminal nodes
                   ) -> dict[str, dict[str, float | dict[str, dict[str, float]]]]: # Flow results; sink: {maximum flow value, flow path dictionary}
  "Estimate maximum flow and flow paths between source and sinks"
  if sinks is None:
    sinks = [node for node, out_degree in network.out_degree() if out_degree == 0 and nx.has_path(network, source, node)]
  # Dictionary to store flow values and paths for each sink
  flow_results = {}
  for sink in sinks:
    try:
      path_length = nx.shortest_path_length(network, source = source, target = sink)
      try:
        flow_value, flow_dict = nx.maximum_flow(network, source, sink)
      except Exception:
        flow_value, flow_dict = nx.maximum_flow(network, source, sink, flow_func = nx.algorithms.flow.edmonds_karp)
      flow_results[sink] = {
          'flow_value': flow_value * path_length,
          'flow_dict': flow_dict
          }
    except (nx.NetworkXError, nx.NetworkXNoPath):
      print(f"{sink} cannot be reached.")
  return flow_results


def get_max_flow_path(network: nx.DiGraph, # Biosynthetic network
                     flow_dict: dict[str, dict[str, float]], # Flow dictionary as returned by get_maximum_flow
                     sink: str, # Target node
                     source: str = "Gal(b1-4)Glc-ol" # Source node
                    ) -> list[tuple[str, str]]: # Path edge list
  "Get path giving maximum flow value"
  path = []
  current_node = source
  abundance_dict = nx.get_node_attributes(network, 'abundance')
  while current_node != sink:
    next_node = max(network.neighbors(current_node),
            key = lambda neighbor: flow_dict[current_node][neighbor] * max(abundance_dict.get(neighbor, 0), 0.1), default = None)
    if next_node is None:
      raise ValueError("No path found")
    path.append((current_node, next_node))
    current_node = next_node
  return path


def get_reaction_flow(network: nx.DiGraph, # Biosynthetic network
                     res: dict[str, dict[str, float]], # Flow results as returned by get_maximum_flow
                     aggregate: str | None = None # Aggregation: sum/mean/None
                    ) -> dict[str, list[float]] | dict[str, float]: # Reaction flows (reaction: flow)
  "Get aggregated flows by reaction type"
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


def get_differential_biosynthesis(df: pd.DataFrame | str, # Glycan abundance data (first column: glycan sequences)
                                group1: list[str | int], # First group column indices/names (or time points in longitudinal analysis)
                                group2: list[str | int] | None = None, # Second group column indices/names (or time points in longitudinal analysis)
                                analysis: str = "reaction", # Type: reaction/flow
                                paired: bool = False, # Whether samples are paired
                                longitudinal: bool = False, # Whether to do perform longitudinal analysis
                                id_column: str = "ID" # Sample ID column for longitudinal analysis in the ID-style of participant_time_replicate
                               ) -> pd.DataFrame: # Differential analysis results (differential flow features and statistics OR reaction changes over time
  "Compare biosynthetic patterns between conditions/timepoints"
  if longitudinal:
    assert id_column is not None, "id_column must be specified for longitudinal analysis"
    assert group2 is None, "group2 should not be specified for longitudinal analysis"
    assert analysis == "reaction", "Longitudinal analysis of flow to sinks not yet supported"
    time_points = group1
  else:
    assert group2 is not None, "group2 must be specified for binary comparison"
    if paired:
      assert len(group1) == len(group2), "For paired samples, the size of group1 and group2 should be the same"
  # Handle input data
  if isinstance(df, str):
    df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
  if not longitudinal and not isinstance(group1[0], str):
    columns_list = df.columns.tolist()
    group1 = [columns_list[k] for k in group1]
    group2 = [columns_list[k] for k in group2]
  all_groups = group1 + (group2 or [])
  # Prepare data for analysis
  if longitudinal:
    df['participant'] = df[id_column].apply(lambda x: x.split('_')[0])
    df['time_point'] = df[id_column].apply(lambda x: x.split('_')[1])
    assert all(tp in df['time_point'].unique() for tp in time_points), "Not all specified time points found in the data"
    df = df.set_index(id_column)
    glycan_columns = [col for col in df.columns if col not in [id_column, 'participant', 'time_point']]
    df_analysis = df[df['time_point'].isin(time_points)].copy()
  else:
    glycan_columns = all_groups
    df_analysis = df.set_index(df.columns.tolist()[0])
  df_analysis = df_analysis.loc[:, glycan_columns].fillna(0)
  df_analysis = (df_analysis / df_analysis.sum(axis = 1).values[:, None]) * 100
  if not longitudinal:
    df_analysis = df_analysis[df_analysis.any(axis = 1)]
  else:
    df_analysis = df_analysis.T
  # Network analysis
  root = list(infer_roots(frozenset(df_analysis.index.tolist())))
  root = max(root, key = len) if '-ol' not in root[0] else min(root, key = len)
  min_default = 0.1 if root.endswith('GlcNAc') else 0.001
  core_net = construct_network(df_analysis.index.tolist())
  nets, features = {}, []
  for col in df_analysis.columns:
    temp = deepcopy(core_net)
    abundance_mapping = dict(zip(df_analysis.index.tolist(), df_analysis[col].values.tolist()))
    nx.set_node_attributes(temp, {g: {'abundance': abundance_mapping.get(g, 0.0)} for g in temp.nodes()})
    nets[col] = estimate_weights(temp, root = root, min_default = min_default)
  res = {col: get_maximum_flow(nets[col], source = root) for col in nets}
  # Perform reaction or flow analysis
  if analysis == "reaction":
    res2 = {col: get_reaction_flow(nets[col], res[col], aggregate = "sum") for col in nets}
    linkage_pat = re.compile(r'\(([ab?])([0-9?/]+)-([0-9?/]+)\)')
    backbone_groups = defaultdict(list)
    for g in res2[next(iter(res2))]:
      backbone_groups[linkage_pat.sub('(LINK)', g)].append(g)
    shadow_reactions = {}
    for backbone, variants in backbone_groups.items():
      if len(variants) < 2:
        continue
      anomer_set = {linkage_pat.search(v).group(1) for v in variants}
      starts = sorted({s for v in variants for s in linkage_pat.search(v).group(2).split('/') if s != '?'},
                      key = int) or ['?']
      ends = sorted({e for v in variants for e in linkage_pat.search(v).group(3).split('/') if e != '?'},
                    key = int) or ['?']
      anomer = next(iter(anomer_set)) if len(anomer_set) == 1 else '?'
      canonical = backbone.replace('(LINK)', f'({anomer}{"/".join(starts)}-{"/".join(ends)})')
      shadow_reactions[canonical] = variants
    res2 = {k: {**v, **{r: np.mean([v[k2] for k2 in shadow_reactions[r]]) for r in shadow_reactions}} for k, v in
            res2.items()}
  elif analysis == "flow":
    res2 = {col: [res[col][sink]['flow_value'] for sink in res[col].keys()] for col in nets}
    features = res[col].keys()
  else:
    raise ValueError("Only 'reaction' and 'flow' are currently supported analysis modes.")
  res2 = pd.DataFrame(res2).T
  if analysis == "reaction" and not longitudinal:
    res2 = res2.loc[:, res2.var(axis = 0) > 0.01]
  elif analysis == "flow" and not longitudinal:
    res2.columns = features
  features = res2.columns.tolist()
  # Perform statistical analysis
  if longitudinal:
    res_df = res2.reset_index()
    res_df = res_df.rename(columns = {'index': id_column})
    res_df = pd.merge(res_df, df[['participant', 'time_point']], on = id_column)
    results = []
    for reaction in features:
      reaction_data = res_df[[id_column, 'participant', 'time_point', reaction]]
      reaction_data = reaction_data.groupby(['participant', 'time_point'])[reaction].mean().reset_index()
      reaction_data['time_numeric'] = pd.Categorical(reaction_data['time_point']).codes
      # Calculate the average slope for each participant
      slopes = reaction_data.groupby('participant').apply(lambda x: np.polyfit(x['time_numeric'], x[reaction], 1)[0], include_groups = False)
      average_slope = slopes.mean()
      direction = "Increase" if average_slope > 0 else "Decrease"
      model = ols('Q("{0}") ~ C(time_point) + C(participant)'.format(reaction), data = reaction_data).fit()
      anova_table = sm.stats.anova_lm(model, typ = 2)
      f_value = anova_table.loc['C(time_point)', 'F']
      p_value = anova_table.loc['C(time_point)', 'PR(>F)']
      results.append({
          'Glycan': reaction,
          'F-statistic': f_value,
          'p-val': p_value,
          'Direction': direction,
          'Average Slope': average_slope
        })
    out = pd.DataFrame(results)
    out['corr p-val'] = multipletests(out['p-val'], method = 'fdr_bh')[1]
    out['significant'] = out['corr p-val'] < 0.05
  else:
    mean_abundance = res2.mean(axis = 0)
    df_a, df_b = res2.loc[group1, :].T, res2.loc[group2, :].T
    log2fc = np.log2((df_b.values + 1e-8) / (df_a.values + 1e-8)).mean(axis = 1) if paired else np.log2(df_b.mean(axis = 1) / df_a.mean(axis = 1))
    pvals = [ttest_rel(row_a, row_b)[1] if paired else ttest_ind(row_a, row_b, equal_var = False)[1] for row_a, row_b in zip(df_a.values, df_b.values)]
    pvals = [p if p > 0 else 1.0 for p in pvals]
    corrpvals = multipletests(pvals, method = 'fdr_tsbh')[1] if pvals else []
    alpha = get_alphaN(len(all_groups))
    significance = [p < alpha for p in corrpvals] if pvals else []
    effect_sizes, _ = zip(*[cohen_d(row_b, row_a, paired = paired) for row_a, row_b in zip(df_a.values, df_b.values)])
    out = pd.DataFrame({'Glycan': features, 'Mean abundance': mean_abundance, 'Log2FC': log2fc, 'p-val': pvals,
                        'corr p-val': corrpvals, 'significant': significance, 'Effect size': effect_sizes})
  out = out.set_index('Glycan')
  return out.dropna().sort_values(by = 'p-val')


def extend_glycans(glycans: list[str] | set[str], # Glycans to extend
                  reactions: list[str] | set[str], # Reactions to apply, in the form of 'Fuc(a1-3)'
                  allowed_disaccharides: set[str] | None = None # Valid disaccharides when creating possible glycans
                 ) -> set[str]: # New glycans
  "Extend glycans using provided reactions"
  new_glycans = set()
  for r in reactions:
    temp_glycans = [f'{{{r}}}{g}' for g in glycans]
    new_glycans.update(set(topology for g in temp_glycans for topology in get_possible_topologies(g, exhaustive = True, allowed_disaccharides = allowed_disaccharides)))
  return new_glycans


def edges_for_extension(leaf_glycans: set[str], # Terminal glycans in a network
                       new_glycans: set[str], # Extended glycans
                       graphs: dict[str, nx.DiGraph] # Glycan:graph mapping
                      ) -> tuple[list[tuple[str, str]], list[str]]: # (New edges, Labels)
  "Create edges connecting leaf and extended glycans"
  new_edges, new_edge_labels = [], []
  for new_g in new_glycans:
    for prec_graph in create_neighbors(safe_index(new_g, graphs), min_size = 1):
      match = next((leaf for leaf in leaf_glycans if safe_compare(safe_index(leaf, graphs), prec_graph)), None)
      if match:
        new_edges.append((match, new_g))
        new_edge_labels.append(find_diff(match, new_g))
  return new_edges, new_edge_labels


def choose_leaves_to_extend(leaf_glycans: set[str], # Terminal glycans in a network
                          target_composition: dict[str, int] # Target composition of type {"HexNAc": 2, "Hex": 9}
                         ) -> set[str]: # Leaf nodes to extend
  "Choose best leaf nodes to extend toward target"
  target_comp = Counter(target_composition)

  def score_glycan(glycan: str):
    comp = Counter(glycan_to_composition(glycan))
    if not set(comp.keys()).issubset(target_comp.keys()):
      return float('inf')
    return sum((target_comp - comp).values())

  scored_glycans = [(glycan, score_glycan(glycan)) for glycan in leaf_glycans]
  min_score = min(score for _, score in scored_glycans)
  if min_score == 0:
    print("Target composition found in leaf nodes")
  if min_score == float("inf"):
    print("Target composition cannot be reached from any leaf node")
    return {}
  return {glycan for glycan, score in scored_glycans if score == min_score}


def extend_network(network: nx.DiGraph, # Biosynthetic network
                  steps: int = 1, # Number of extension steps; default:1
                  to_extend: str | dict[str, int] | list[str] = "all", # Nodes to extend (all, specific leaf node, target composition)
                  strict_context: bool = False # Whether to use network only to derive allowed reaction products; default:False
                 ) -> tuple[nx.DiGraph, set[str]]: # (Extended network, New glycans)
  "Extend biosynthetic network physiologically"
  graphs = {}
  new_glycans = set()
  classy = get_class(next(iter(network.nodes())))
  if strict_context:
    glycs = list(network.nodes())
  else:
    from glycowork.glycan_data.loader import df_species
    glycs = df_species[df_species.Class == "Mammalia"].glycan.drop_duplicates()
    glycs = glycs[glycs.apply(get_class) == classy].tolist()
  mammal_disac = set(unwrap(get_k_saccharides(glycs, just_motifs = True)))
  reactions = {r for r in nx.get_edge_attributes(network, "diffs").values() if all(x not in r for x in ('?', 'Hex', 'O'))}
  leaf_glycans = {x for x in network.nodes() if network.out_degree(x) == 0 and network.in_degree(x) > 0}
  if isinstance(to_extend, dict):
    leaf_glycans = choose_leaves_to_extend(leaf_glycans, to_extend)
    reactions = {r for r in reactions if map_to_basic(r.split('(')[0]) in to_extend.keys()}
  if (isinstance(to_extend, str) and to_extend != "all") or isinstance(to_extend, list):
    leaf_glycans = {to_extend} if isinstance(to_extend, str) else set(to_extend)
  for _ in range(steps):
    new_leaf_glycans = extend_glycans(leaf_glycans, reactions, allowed_disaccharides = mammal_disac)
    new_edges, new_edge_labels = edges_for_extension(leaf_glycans, new_leaf_glycans, graphs)
    network = update_network(network, new_edges, edge_labels = new_edge_labels, node_labels = {k: 1 for k in new_leaf_glycans})
    leaf_glycans = new_leaf_glycans
    new_glycans.update(leaf_glycans)
    if isinstance(to_extend, dict):
      leaf_glycans = choose_leaves_to_extend(leaf_glycans, to_extend)
  return network, new_glycans
