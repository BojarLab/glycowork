import re
from pathlib import Path
from copy import deepcopy
from itertools import chain
from functools import lru_cache
from importlib import resources
from collections import defaultdict, Counter
import networkx as nx
import numpy as np
import pandas as pd
from glycowork.glycan_data.loader import unwrap, linkages, lib, GlycoList, GlycoDataFrame
from glycowork.glycan_data.stats import cohen_d, get_alphaN, correct_multiple_testing, moderated_variance, dag_neighbors, TST_grouped_benjamini_hochberg
from glycowork.motif.graph import compare_glycans, glycan_to_nxGraph, graph_to_string, graph_to_string_int, subgraph_isomorphism, get_possible_topologies
from glycowork.motif.processing import get_lib, rescue_glycans, in_lib, get_class, canonicalize_iupac, canonicalize_composition, is_composition
from glycowork.motif.tokenization import get_stem_lib, glycan_to_composition, map_to_basic
from glycowork.motif.annotate import get_k_saccharides

def __getattr__(name):
    if name == "net_dic":
        from glycowork.network.evolution import _load_net_dic
        net_dic = dict(_load_net_dic())  # hand out a copy of the one cached unpickle, as evolution does
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
    terminal_pairs = [frozenset({k, next(ggraph.predecessors(k))}) for k in ggraph.nodes() if
                      ggraph.out_degree(k) == 0 and ggraph.in_degree(k) > 0]
    comps = sorted(nx.weakly_connected_components(ggraph), key = min)
    if len(comps) > 1:
        # a floating bit has no fixed attachment point, so it cannot be ordered against a backbone addition; deferring it until the backbone is minimal stops the node count growing as 2^floaters x backbone precursors
        backbone = [p for p in terminal_pairs if p <= comps[0]]
        terminal_pairs = backbone or terminal_pairs
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
    if not ptm_glycans:
        return ([], [])
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
    counts = Counter(get_class(k) for k in glycans)
    # the root sets are mutually exclusive, so a handful of stray or misclassified sequences must not outvote the class the data actually belongs to
    classes = {c for c, n in counts.items() if c and n >= 0.1 * sum(counts.values())}
    if len([c for c in counts if c]) > 1:
        print(
            f"More than one glycan class detected ({dict(counts)}); the network will be rooted in the majority class, so check that the input is not mixed.")
    for net_class, roots in (('free', {'Gal(b1-4)Glc-ol', 'Gal(b1-4)GlcNAc-ol'}),
                             ('lipid', {'Glc1Cer', 'Gal1Cer', 'Ins'}),
                             ('N', {'Man(b1-4)GlcNAc(b1-4)GlcNAc'}), ('O', {'GalNAc', 'Fuc', 'Man'})):
        if net_class in classes:
            return frozenset(roots)
    if 'lipid/free' in classes or any(k.endswith('Glc') for k in glycans):
        print("Are you working with free oligosaccharides or glycolipids? Append '-ol' or '1Cer' to your glycans, respectively. We'll pretend it's milk glycans for now")
        return frozenset({'Gal(b1-4)Glc', 'Gal(b1-4)GlcNAc'})
    print("Glycan class not detected; depending on the class, glycans should end in -ol, GalNAc, GlcNAc, or 1Cer")
    return frozenset()


def add_high_man_removal(network: nx.DiGraph # Biosynthetic network
                         ) -> nx.DiGraph: # Network with high-mannose removal
    "Add 'reverse' edges for high-mannose N-glycan maturation (i.e., cutting down)"
    edges_to_add = [(target, source, network[source][target]) for source, target in network.edges() if target.count('Man') >= 5]
    network.add_edges_from(edges_to_add)
    return network


def _graph_fp(g: nx.DiGraph) -> tuple:
    "Wildcard-safe structural fingerprint for bucketing"
    leaves = branching = 0
    for _, d in g.out_degree():
        leaves += d == 0
        branching += d > 1
    memo = {}

    def enc(n):
        r = memo.get(n)
        if r is None:
            r = memo[n] = '(' + ''.join(sorted(enc(c) for c in g.successors(n))) + ')'
        return r

    return (len(g), leaves, branching, ''.join(sorted(enc(n) for n in g if g.in_degree(n) == 0)))


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
                      abundances: list[float] = [], # Glycan abundances in the same order as glycans; default:empty
                      constraints: bool | list[dict] | pd.DataFrame = True
                      # Apply established biochemical constraints on reaction order; False to disable, or pass custom rules
                      ) -> nx.DiGraph: # Biosynthetic network
    "Construct glycan biosynthetic network"
    # Canonicalize all input strings upfront so string equality == graph isomorphism throughout
    glycans = sorted(set(canonicalize_iupac(g) for g in glycans))
    stem_lib = get_stem_lib(get_lib(glycans))
    if permitted_roots is None:
        permitted_roots = infer_roots(frozenset(glycans))
        if not permitted_roots:
            raise ValueError(
                f"Could not detect the glycan class (e.g., from '{glycans[0] if glycans else ''}'), so no biosynthetic roots can be inferred; glycans should end in '-ol' (free), 'GalNAc' (O-linked), 'GlcNAc' (N-linked), or '1Cer'/'Ins' (glycolipid), or pass permitted_roots explicitly.")
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
    cmp_cache = {}
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
            rs = {}
            for term_node in (nd for nd in g.nodes() if g.out_degree(nd) == 0 and g.in_degree(nd) > 0):
                link_node = next(g.predecessors(term_node))
                sub = g.subgraph(set(g.nodes()) - {term_node, link_node})
                rs[graph_to_string(sub)] = _graph_fp(sub)
            for n_small in candidates:
                hit = n_small in rs
                if not hit:
                    fp_small = node_fps[n_small]
                    for r, fp_r in rs.items():
                        if fp_r != fp_small:
                            continue
                        res = cmp_cache.get((r, n_small))
                        if res is None:
                            res = cmp_cache[(r, n_small)] = compare_glycans(safe_index(r, graph_dic),
                                                                            safe_index(n_small, graph_dic))
                        if res:
                            hit = True
                            break
                if hit:
                    network.add_edge(n_big, n_small)
    network = deorphanize_edge_labels(network, graph_dic, allowed_ptms = allowed_ptms)
    for node in network:
        network.nodes[node].setdefault('virtual', 1)
    # Edge label specification
    if edge_type != 'monolink':
        df_enzyme, net_class = None, None
        if edge_type == 'enzyme':
            with resources.files("glycowork.network").joinpath("monolink_to_enzyme.csv").open() as f:
                df_enzyme = pd.read_csv(f, sep = ',')
            net_class = get_class(glycans[0])  # glycans sorted desc by len, [0] is a real observed glycan
        for u, v in network.edges():
            elem = network[u][v]
            edge = elem['diffs']
            for link in linkages:
                if link in edge:
                    if edge_type == 'monosaccharide':
                        elem['diffs'] = edge.split('(')[0]
                    elif edge_type == 'enzyme':
                        elem['diffs'] = monolink_to_glycoenzyme(edge, df_enzyme, glycan_class = net_class, product = v)
    # Make network directed
    network = prune_directed_edges(network.to_directed())
    if constraints is not False:
        network = apply_constraints(network, None if constraints is True else constraints)
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


@lru_cache(maxsize = None)
def _load_constraints() -> tuple: # Rule records from the shipped constraint table
    "Read the shipped table of established biochemical constraints on reaction order"
    with resources.files("glycowork.network").joinpath("biosynthetic_constraints.csv").open(encoding = 'utf-8-sig') as f:
        df = pd.read_csv(f)
    return tuple(df.to_dict('records'))


def apply_constraints(network: nx.DiGraph, # Biosynthetic network with 'diffs' edge labels
                      constraints: list[dict] | pd.DataFrame | None = None, # Rules of product/context/kind/glycan_class; default: the shipped table
                      prune: bool = True # Whether to remove violating edges instead of only labeling them
                      ) -> nx.DiGraph: # Network with a 'constraint' edge attribute on violating edges
    "Flag or remove reactions that established biochemistry forbids in their precursor context"
    rules = tuple(constraints.to_dict('records')) if isinstance(constraints, pd.DataFrame) else tuple(constraints) if constraints is not None else _load_constraints()
    net_class = get_class(max((n for n, v in network.nodes(data = 'virtual') if v == 0), key = len, default = ''))
    rules = [r for r in rules if not isinstance(r.get('glycan_class'), str) or net_class in str(r['glycan_class']).split('/')]
    if not rules:
        return network
    motifs = {m for r in rules for m in [r['product']] + str(r['context']).split('|')}
    # one subgraph search per (node, motif) instead of per edge, since every node takes part in several edges
    has = {m: {n: subgraph_isomorphism(n, m) for n in network.nodes()} for m in motifs}
    violations = {}
    for u, v in network.edges():
        for r in rules:
            # the rule only speaks to the edge that installs its product, not to every edge downstream of it
            if not has[r['product']].get(v, False) or has[r['product']].get(u, False):
                continue
            if any(has[c].get(u, False) for c in str(r['context']).split('|')) == (r['kind'] == 'forbids'):
                violations[(u, v)] = ': '.join(r[k] for k in ('enzyme', 'rationale') if isinstance(r.get(k),
                                                                                                   str)) or f"{r['product']} {r['kind']} {r['context']}"
    network = network.copy()
    nx.set_edge_attributes(network, violations, 'constraint')
    if prune and violations:
        flagged = {e: dict(network.edges[e]) for e in violations}
        network.remove_edges_from(violations)
        # an observed structure exists whatever the rule says, so its last route into the network is kept and stays flagged
        for node in sorted((n for n, v in network.nodes(data = 'virtual') if v == 0 and network.in_degree(n) == 0),
                           key = len):
            if restore := [e for e in violations if e[1] == node]:
                network.add_edges_from((u, v, flagged[(u, v)]) for u, v in restore)
    return network


def plot_network(network: nx.DiGraph, # Biosynthetic network
                 plot_format: str = 'hierarchical', # Layout type: hierarchical/pydot2/kamada_kawai/spring
                 edge_label_draw: bool = True, # Whether to draw edge labels
                 lfc_dict: dict[str, float] | None = None,  # Enzyme:log2FC mapping for edge width
                 draw_glycans: bool = False,  # Replace node labels with SNFG drawings, in a static figure
                 filepath: str | Path = '',  # Path to save the static figure (.svg/.pdf/.png) instead of showing the interactive plot; required for draw_glycans
                 compact: bool = False,  # Use compact SNFG style
                 glycan_size: str = 'small'  # Glycan size preset ('small', 'medium', 'large')
                 ) -> None:  # Displays plot
    "Visualize biosynthetic network"
    if draw_glycans and not filepath:
        raise ValueError(
            "draw_glycans = True draws the SNFG structures into a saved figure and therefore needs a filepath, e.g., filepath = 'network.svg'.")
    if plot_format == 'hierarchical':
        roots = [n for n in network.nodes() if network.in_degree(n) == 0]
        levels = {}
        queue = [(r, 0) for r in roots or list(network.nodes())[:1]]
        while queue:
            node, level = queue.pop(0)
            if node in levels:
                continue
            levels[node] = level
            for s in network.successors(node):
                queue.append((s, level + 1))
        levels.update({n: 0 for n in network.nodes() if n not in levels})
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
        node_data['size'].append(node_attributes.get(node, 50) / 2)
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
    if filepath:
        # Static rendering, so that annotate_figure can swap the node labels for SNFG drawings in the saved SVG
        import os
        import matplotlib.pyplot as plt
        from glycowork.motif.draw import annotate_figure
        # the SNFGs have a fixed size on the canvas, so the canvas has to grow with the layout instead of the other way around
        rows = Counter(round(y, 6) for y in node_data['y'])
        if (f := {'small': 1, 'medium': 2, 'large': 3}.get(glycan_size) if draw_glycans else 1) is None:
            raise ValueError(f"glycan_size has to be 'small', 'medium', or 'large' (got '{glycan_size}').")
        w, h = (1.5 * f * max(rows.values()), 1.8 * f * len(rows)) if max(rows.values()) > 1 else (1.5 * f * len(
            network) ** 0.5,) * 2
        fig, ax = plt.subplots(figsize = (min(60, max(12, w)), min(60, max(9, h))))
        for (x0, x1), (y0, y1), color, width in zip(edge_data['xs'], edge_data['ys'], edge_data['color'],
                                                    edge_data['width']):
            ax.annotate('', xy = (x1, y1), xytext = (x0, y0),
                        arrowprops = dict(arrowstyle = '-|>', color = color, linewidth = width, shrinkA = 0,
                                          shrinkB = 0))
        ax.scatter(node_data['x'], node_data['y'], s = [s ** 2 for s in node_data['size']], c = node_data['color'],
                   alpha = node_data['alpha'], edgecolors = '#888888', linewidths = 1)
        for x, y, name, s in zip(node_data['x'], node_data['y'], node_data['name'], node_data['size']):
            # anchored just right of the node, since annotate_figure grows the SNFG from the label's own origin
            ax.annotate(name, (x, y), textcoords = 'offset points', xytext = (s / 2 + 4, 0), fontsize = 8, ha = 'left',
                        va = 'center')
        if edge_label_draw:
            for (x0, x1), (y0, y1), label in zip(edge_data['xs'], edge_data['ys'], edge_data['label']):
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, fontsize = 8, ha = 'center', va = 'center')
        ax.margins(0.12)
        ax.axis('off')
        Path(filepath).parent.mkdir(parents = True, exist_ok = True)
        if not draw_glycans:
            plt.savefig(filepath, bbox_inches = 'tight')
            plt.close(fig)
            return
        svg_temp = str(Path(filepath).with_suffix('')) + '_temp.svg'
        plt.savefig(svg_temp, format = 'svg', bbox_inches = 'tight')
        plt.close(fig)
        try:
            annotate_figure(svg_temp, filepath = filepath, compact = compact, glycan_size = glycan_size)
        finally:
            os.remove(svg_temp)
        return
    from bokeh.plotting import figure, show
    from bokeh.io import output_notebook
    from bokeh.models import HoverTool, Arrow, NormalHead, LabelSet, ColumnDataSource
    try:
        output_notebook()
    except (ImportError, RuntimeError, Exception):
        pass
    p = figure(width = 900, height = 900, x_range = (-1.2, 1.2), y_range = (-1.2, 1.2),
               tools = "pan,wheel_zoom,box_zoom,reset,save", toolbar_location = "above",
               x_axis_type = None, y_axis_type = None, background_fill_color = "white")
    # Draw nodes with hover
    node_renderer = p.scatter('x', 'y', size = 'size', color = 'color', alpha = 'alpha', line_color = "#888",
                              line_width = 1, source = ColumnDataSource(data = node_data))
    p.add_tools(HoverTool(renderers = [node_renderer], tooltips = [("Node", "@name")]))
    for edge, color, width in zip(network.edges(), edge_data['color'], edge_data['width']):
        p.add_layout(
            Arrow(end = NormalHead(size = 8, fill_color = color), x_start = pos[edge[0]][0], y_start = pos[edge[0]][1],
                  x_end = pos[edge[1]][0], y_end = pos[edge[1]][1], line_width = width, line_color = color))
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
        inferred, _ = infer_virtual_nodes(network, temp_network)
        inferences.update(inferred[0])
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
                            mode: str = 'condensed',  # Output mode: condensed/full
                            glycan_class: str | None = None,  # Network glycan class to filter enzymes by
                            product: str | None = None  # Reaction product, used to resolve isozymes by acceptor context
                            ) -> str:  # Enzyme label
    "Convert monosaccharide(linkage) edge label to enzyme name responsible for its synthesis"
    if mode == 'condensed':
        enzyme_column = 'glycoclass'
    hits = df[df[monolink_column] == edge_label]
    if product is not None and 'acceptor' in df.columns and not hits.empty: # isozymes of one family differ by acceptor, not by linkage
        keep = hits['acceptor'].apply(
            lambda a: not isinstance(a, str) or any(subgraph_isomorphism(product, m) for m in a.split('|')))
        hits = hits[keep] if keep.any() else hits
    if glycan_class is not None and not hits.empty:
        net_cls = set(glycan_class.split('/'))
        keep = hits['glycan_class'].apply(lambda c: not isinstance(c, str) or bool(net_cls & set(c.split('/'))))  # blank rows apply to all classes
        hits = hits[keep] if keep.any() else hits  # fall back to class-agnostic if no class match
    new_edge_label = hits[enzyme_column].unique()
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
    if any(not alt for alt in alternatives) or alternatives[0] == alternatives[1]:
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
    # Map found diamonds to the same format
    seen_keys = {}
    for d in matchings_list:
        seen_keys.setdefault(tuple(sorted(d.keys())), {v: k for k, v in d.items()})
    matchings_list = list(seen_keys.values())
    final_matchings = []
    # For each diamond, only keep the diamond if at least one of the intermediate structures is virtual
    virtual_attr = nx.get_node_attributes(network, 'virtual')
    desc_cache = {}
    for d in matchings_list:
        substrate_node, product_node = d[1], d[path_length]
        middle_nodes = [d[k] for k in list(range(2, path_length)) + list(range(path_length + 1, 2 * path_length - 1))]
        # For each middle node, test whether they are part of the same path as substrate node and product node (remove false positives)
        for node in [substrate_node] + middle_nodes:
            if node not in desc_cache:
                desc_cache[node] = nx.descendants(network, node)
        if all(mn in desc_cache[substrate_node] and product_node in desc_cache[mn] for mn in middle_nodes):
            # 'presence' keeps only diamonds with an unobserved intermediate (network completion); 'abundance' keeps all (reaction order)
            if mode == 'abundance' or any(virtual_attr.get(mn, 0) == 1 for mn in middle_nodes):
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
    # An intermediate can dominate because its path is preferred or because nothing downstream consumes it, so the onward context travels with the preference
    caps, ctx = nx.get_edge_attributes(network, 'capacity'), []
    for g in df_out['glycan']:
        out_edges = list(network.out_edges(g)) if g in network else []
        ctx.append((len(out_edges), sum(caps.get(e, np.nan) for e in out_edges) if out_edges and caps else np.nan,
                    len(nx.descendants(network, g)) if g in network else 0))
    df_out[['out_degree', 'onward_capacity', 'n_descendants']] = pd.DataFrame(ctx, index = df_out.index,
                                                                              columns = ['out_degree',
                                                                                         'onward_capacity',
                                                                                         'n_descendants'])
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
        from glycowork.network.evolution import _load_net_dic
        network_dic = _load_net_dic()  # one cached unpickle of the bundle for the whole package, instead of one per call site
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
    # Determine highlight validity
    if highlight not in ['motif', 'species', 'abundance', 'conservation']:
        raise ValueError(f"Invalid highlight argument: {highlight}")
    if network_dic is None and highlight in {'species', 'conservation'}:
        from glycowork.network.evolution import _load_net_dic
        network_dic = _load_net_dic()
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
        from glycowork.glycan_data.loader import resolve_motif_name
        motif, termini = resolve_motif_name(motif) or (motif, [])
        motif_presence = {k: ('limegreen' if subgraph_isomorphism(k, motif, termini_list = termini) else 'darkviolet')
                          for k in network_out.nodes()}
        nx.set_node_attributes(network_out, motif_presence, name = 'origin')
    # Color nodes as to whether they are from a species (green) or not (violet)
    elif highlight == 'species':
        temp_net_nodes = set(network_dic[species].nodes())
        node_presence = {k: ('limegreen' if k in temp_net_nodes else 'darkviolet') for k in network_out.nodes()}
        nx.set_node_attributes(network_out, node_presence, name = 'origin')
    # Add relative intensity values as 'abundance' node attribute used for node size scaling
    elif highlight == 'abundance':
        abundance_dict = dict(zip(abundance_df[glycan_col], abundance_df[intensity_col] * 100))
        abundance_keys = GlycoList(list(abundance_dict))
        node_abundance = {
            node: abundance_dict[abundance_keys[abundance_keys.index(node)]] if node in abundance_keys else 50 for node
            in network_out.nodes()}
        nx.set_node_attributes(network_out, node_abundance, name = 'abundance')
    # Add the degree of evolutionary conervation as 'abundance' node attribute used for node size scaling
    elif highlight == 'conservation':
        species_list = conservation_df['Species'].unique().tolist()
        spec_nodes = {k: set(network_dic[k].nodes()) for k in species_list}
        node_counts = Counter(node for nodes in spec_nodes.values() for node in nodes)
        node_conservation = {k: node_counts[k] * 100 for k in network_out.nodes()}
        nx.set_node_attributes(network_out, node_conservation, name = 'abundance')
    return network_out


def get_edge_weight_by_abundance(network_in: nx.DiGraph, # Biosynthetic network
                                 root: str = "Gal(b1-4)Glc-ol", # Root node
                                 root_default: float = 10.0,  # Root abundance
                                 virtual_damping: float = 0.5  # Per-hop abundance decay for undetected intermediates
                                 ) -> nx.DiGraph:  # Network with edge capacities
    "Estimate reaction capacity (edge attribute) from node abundances"
    network = network_in.copy()
    abundance_dict = nx.get_node_attributes(network, 'abundance')
    if abundance_dict.get(root, 1) < 0.1:
        abundance_dict[root] = root_default
    detected = [a for a in abundance_dict.values() if a > 0.0]
    floor = min(detected) if detected else 0.1
    missing = sorted({n for n, v in network.nodes(data = 'virtual') if v == 1 or abundance_dict.get(n, 0.0) <= 0.0} - {root})
    for _ in range(len(missing)):
        updated = False
        for n in missing:  # a fixed list, since each estimate reads the neighbours updated before it and set order is not stable across processes
            known = [a for a in
                     (abundance_dict.get(m, 0.0) for m in chain(network.predecessors(n), network.successors(n))) if
                     a > 0.0]
            # Abundances are log-normal, so pool geometrically; a virtual node was not detected, so it cannot exceed the detection limit, and each further hop into unobserved territory decays
            if known:
                est = min(float(np.exp(np.mean(np.log(known)))) * virtual_damping, floor)
                if est > abundance_dict.get(n, 0.0):
                    abundance_dict[n] = est
                    updated = True
        if not updated:
            break
    for u, v in network.edges():
        network[u][v]['capacity'] = float(
            np.sqrt(max(abundance_dict.get(u, 0.1), 1e-6) * max(abundance_dict.get(v, 0.1), 1e-6)))
    return network


def get_maximum_flow(network: nx.DiGraph, # Biosynthetic network
                     source: str = "Gal(b1-4)Glc-ol", # Source node
                     sinks: list[str] | None = None # Target nodes; default:all terminal nodes
                     ) -> dict[str, dict[str, float | dict[str, dict[str, float]]]]: # Flow results; sink: {maximum flow value, flow path dictionary}
    "Estimate maximum flow and flow paths between source and sinks"
    path_lengths = nx.single_source_shortest_path_length(network, source)
    if sinks is None:
        sinks = [node for node, out_degree in network.out_degree() if out_degree == 0 and node in path_lengths]
    # Dictionary to store flow values and paths for each sink
    flow_results, unreachable = {}, []
    for sink in sinks:
        try:
            if sink not in path_lengths:
                raise nx.NetworkXNoPath(f"No path between {source} and {sink}.")
            try:
                flow_value, flow_dict = nx.maximum_flow(network, source, sink)
            except Exception:
                flow_value, flow_dict = nx.maximum_flow(network, source, sink,
                                                        flow_func = nx.algorithms.flow.edmonds_karp)
            flow_results[sink] = {
                'flow_value': flow_value,
                'flow_dict': flow_dict
            }
        except (nx.NetworkXError, nx.NetworkXNoPath):
            unreachable.append(sink)
    if unreachable:
        print(f"{len(unreachable)} of {len(sinks)} sinks could not be reached from {source}, e.g., {unreachable[0]}")
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
                                  group1: list[str | int] | None = None, # First group column indices/names (or time points in longitudinal analysis); default: from the frame's contrasts
                                  group2: list[str | int] | None = None, # Second group column indices/names (or time points in longitudinal analysis)
                                  analysis: str = "reaction",  # Type: reaction/flow/branchpoint
                                  paired: bool | None = None,  # Whether samples are paired; default: from the frame
                                  longitudinal: bool = False,  # Whether to do perform longitudinal analysis
                                  id_column: str = "ID", # Sample ID column for longitudinal analysis in the ID-style of participant_time_replicate
                                  edge_type: str = "monolink",  # Reaction resolution: monolink/monosaccharide/enzyme
                                  virtual_damping: float = 0.5  # Per-hop abundance decay for undetected intermediates
                                  ) -> pd.DataFrame: # Differential analysis results (differential flow features and statistics OR reaction changes over time
    "Compare biosynthetic patterns between conditions/timepoints"
    from scipy.stats import t as tdist
    from scipy.stats import f as f_dist
    if group1 is None and isinstance(df, GlycoDataFrame) and df._contrasts:
        group1, group2 = list(df.group1), list(df.group2)
    in_name, in_prov = getattr(df, '_glyco_name', ''), getattr(df, '_provenance', {})
    paired = df.paired if paired is None and isinstance(df, GlycoDataFrame) else bool(paired)
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
        df_analysis = df.set_index(GlycoDataFrame(df)._glycan_col or df.columns.tolist()[0])
    df_analysis = df_analysis.loc[:, glycan_columns].fillna(0)
    if longitudinal:
        df_analysis = (df_analysis / df_analysis.sum(axis = 1).values[:, None]) * 100
    else:
        df_analysis = (df_analysis / df_analysis.sum(axis = 0).values[None, :]) * 100
    if not longitudinal:
        df_analysis = df_analysis[df_analysis.any(axis = 1)]
    else:
        df_analysis = df_analysis.T
    # Network analysis
    root = sorted(infer_roots(frozenset(df_analysis.index.tolist())))
    if not root:
        raise ValueError(
            f"Could not detect the glycan class (e.g., from '{df_analysis.index[0]}'), so no biosynthetic root can be inferred; glycans should end in '-ol' (free), 'GalNAc' (O-linked), 'GlcNAc' (N-linked), or '1Cer'/'Ins' (glycolipid).")
    root = max(root, key = len) if '-ol' not in root[0] else min(root, key = len)
    core_net = construct_network(df_analysis.index.tolist(), edge_type = edge_type)
    nets, features = {}, []
    for col in df_analysis.columns:
        temp = deepcopy(core_net)
        abundance_mapping = dict(zip(df_analysis.index.tolist(), df_analysis[col].values.tolist()))
        nx.set_node_attributes(temp, {g: {'abundance': abundance_mapping.get(g, 0.0)} for g in temp.nodes()})
        nets[col] = get_edge_weight_by_abundance(temp, root = root, virtual_damping = virtual_damping)
    if analysis == "flow":
        res = {col: get_maximum_flow(nets[col], source = root) for col in nets}
    else:
        # One solve to a shared super-sink: adding up per-sink solutions counts a trunk edge once per downstream sink, and how many sinks stay reachable differs between samples, so that multiplicity leaks into every trunk feature
        res = {}
        for col, net in nets.items():
            aug = net.copy()
            aug.add_edges_from([(t, '__sink__') for t, d in net.out_degree() if d == 0])
            try:
                flow_value, flow_dict = nx.maximum_flow(aug, root, '__sink__')
            except Exception:
                flow_value, flow_dict = nx.maximum_flow(aug, root, '__sink__',
                                                        flow_func = nx.algorithms.flow.edmonds_karp)
            res[col] = {'__sink__': {'flow_value': flow_value, 'flow_dict': flow_dict}}
    # Perform reaction or flow analysis
    shadow_set = set()
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
        shadow_set = set(shadow_reactions)
    elif analysis == "branchpoint":
        # Competition at a branch point is the closest network-level proxy for relative enzyme activity, and as a ratio it survives global abundance shifts that swamp absolute flows
        res2 = {}
        for col, net in nets.items():
            merged = defaultdict(float)
            for sink_data in res[col].values():
                fd = sink_data['flow_dict']
                for u, v in net.edges():
                    merged[(u, v)] += fd[u][v]
            vals = {}
            for u in net.nodes():
                outs = [(v, merged[(u, v)]) for v in net.successors(u)]
                tot = sum(f for _, f in outs)
                if len(outs) < 2 or tot <= 0:
                    continue
                for v, f in outs:
                    vals[f"{u} -> {net[u][v]['diffs']}"] = f / tot
            res2[col] = vals
    elif analysis == "flow":
        sinks = sorted({sink for r in res.values() for sink in r})
        res2 = {col: [res[col][sink]['flow_value'] if sink in res[col] else 0.0 for sink in sinks] for col in nets}
        features = sinks
    else:
        raise ValueError("Only 'reaction', 'flow', and 'branchpoint' are currently supported analysis modes.")
    res2 = pd.DataFrame(res2).T.fillna(0.0)
    if analysis == "flow" and not longitudinal:
        res2.columns = features
    if not longitudinal:
        if analysis == "branchpoint":
            # Already a proportion, so logit instead of share-normalizing
            p = res2.clip(0.001, 0.999)
            res_lin, res2 = res2, np.log2(p / (1 - p))
            keep = res_lin.std(axis = 0) > 0
        else:
            # Flux shares strip the per-sample total-flow factor; the log puts trunk and peripheral reactions on a comparable spread so the variance filter stops being an abundance filter
            totals = res2.sum(axis = 1)
            res_lin = res2.div(totals.where(totals > 0, 1), axis = 0) * 100
            res2 = np.log2(res_lin + 0.01)
            keep = res_lin.mean(axis = 0) > 0.01
        res2, res_lin = res2.loc[:, keep], res_lin.loc[:, keep]
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
            y = reaction_data[reaction].values.astype(float)
            a = pd.get_dummies(reaction_data['time_point'], drop_first = True).values.astype(float)
            b = pd.get_dummies(reaction_data['participant'], drop_first = True).values.astype(float)
            X_full, X_red = np.column_stack([np.ones(len(y)), a, b]), np.column_stack([np.ones(len(y)), b])
            ss_full = ((y - X_full @ np.linalg.lstsq(X_full, y, rcond = None)[0]) ** 2).sum()
            ss_red = ((y - X_red @ np.linalg.lstsq(X_red, y, rcond = None)[0]) ** 2).sum()
            rk_full, rk_red = np.linalg.matrix_rank(X_full), np.linalg.matrix_rank(X_red)
            df_num, df_den = rk_full - rk_red, len(y) - rk_full
            f_value = ((ss_red - ss_full) / df_num) / (ss_full / df_den)
            p_value = f_dist.sf(f_value, df_num, df_den)
            results.append({
                'Glycan': reaction,
                'F-statistic': f_value,
                'p-val': p_value,
                'Direction': direction,
                'Average Slope': average_slope
            })
        out = pd.DataFrame(results)
        out['corr p-val'], out['significant'] = correct_multiple_testing(out['p-val'], get_alphaN(len(all_groups)), correction_method = "one-stage")
    else:
        mean_abundance = res_lin.mean(axis = 0)
        df_a, df_b = res2.loc[group1, :].T, res2.loc[group2, :].T
        log2fc = (df_b.values - df_a.values).mean(axis = 1) if paired else df_b.values.mean(
            axis = 1) - df_a.values.mean(axis = 1)
        # Sinks adjacent in the network share precursors, and the branches leaving one node are one flux split, so both make a sharper variance reference than unrelated features
        if analysis == "flow":
            neighbors = dag_neighbors(features, core_net)
        elif analysis == "branchpoint":
            by_node = defaultdict(list)
            for i, f in enumerate(features):
                by_node[f.rsplit(' -> ', 1)[0]].append(i)
            neighbors = [sorted(set(by_node[f.rsplit(' -> ', 1)[0]]) - {i}) for i, f in enumerate(features)]
        elif shadow_set:
            # A collapsed reaction is the mean of its own linkage variants, so that family is a sharper variance reference than the global prior
            fam, where = {}, {f: i for i, f in enumerate(features)}
            for shadow, variants in shadow_reactions.items():
                for f in [shadow] + variants:
                    fam.setdefault(f, set()).update([shadow] + variants)
            neighbors = [sorted(where[o] for o in fam.get(f, set()) - {f} if o in where) for f in features]
        else:
            neighbors = None
        if paired:
            D = df_b.values - df_a.values
            eff, dfree, scale = D.mean(axis = 1), D.shape[1] - 1, 1 / D.shape[1]
            s2 = D.var(axis = 1, ddof = 1)
        else:
            na, nb = df_a.shape[1], df_b.shape[1]
            eff, dfree, scale = df_b.values.mean(axis = 1) - df_a.values.mean(axis = 1), na + nb - 2, 1 / na + 1 / nb
            s2 = ((na - 1) * df_a.values.var(axis = 1, ddof = 1) + (nb - 1) * df_b.values.var(axis = 1,
                                                                                              ddof = 1)) / dfree
        s2_mod, df_post = moderated_variance(s2, dfree, neighbors)
        pvals = np.maximum(2 * tdist.sf(np.abs(eff / np.sqrt(s2_mod * scale)), df_post), np.finfo(float).tiny)
        # Shadow reactions are averages of their own variants, so testing both in one family nearly doubles it with redundant hypotheses; two-stage grouped BH gives each family its own pi0 instead of correcting them as unrelated runs
        alpha = get_alphaN(len(all_groups))
        fam = np.isin(features, list(shadow_set))
        grouped_f = {k: [f for f, m in zip(features, mask) if m] for k, mask in (('collapsed', fam), ('variant', ~fam))
                     if mask.any()}
        grouped_p = {k: [p for p, m in zip(pvals, mask) if m] for k, mask in (('collapsed', fam), ('variant', ~fam)) if
                     mask.any()}
        cp, sd = TST_grouped_benjamini_hochberg(grouped_f, grouped_p, alpha)
        corrpvals, significance = [cp[f] for f in features], [sd[f] for f in features]
        pvals = list(pvals)
        # the test statistic uses the moderated variance, so standardizing the effect by anything else makes the two columns disagree on the same row
        effect_sizes = eff / np.sqrt(s2_mod)
        out = pd.DataFrame({'Glycan': features, 'Mean abundance': mean_abundance, 'Log2FC': log2fc, 'p-val': pvals,
                            'corr p-val': corrpvals, 'significant': significance, 'Effect size': effect_sizes})
    out = out.set_index('Glycan')
    out.attrs.update({'alpha': get_alphaN(len(all_groups)), 'n': len(all_groups), 'test': 'moderated t-test',
                      'transform': None, 'paired': paired, 'dataset': in_name, 'provenance': in_prov})
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
    if not scored_glycans:
        return set()
    min_score = min(score for _, score in scored_glycans)
    if min_score == 0:
        print("Target composition found in leaf nodes")
    if min_score == float("inf"):
        print("Target composition cannot be reached from any leaf node")
        return {}
    return {glycan for glycan, score in scored_glycans if score == min_score}


@lru_cache(maxsize = 8)
def _reference_disaccharides(glycan_class: str # Glycan class the network belongs to
                            ) -> frozenset: # Disaccharides observed in mammalian glycans of that class
    "Reference disaccharides for physiological extension; cached because they depend only on the class"
    from glycowork.glycan_data.loader import df_species
    glycs = df_species.meta_filter(Class = "Mammalia").glycan.drop_duplicates()
    return frozenset(unwrap(get_k_saccharides(glycs[glycs.apply(get_class) == glycan_class].tolist(), just_motifs = True)))


def extend_network(network: nx.DiGraph, # Biosynthetic network
                   steps: int = 1, # Number of extension steps; default:1 (becomes max_steps when auto_steps is True)
                   to_extend: str | dict[str, int] | list[str] = "all", # Nodes to extend (all, specific leaf node, target composition)
                   strict_context: bool = False, # Whether to use network only to derive allowed reaction products; default:False
                   auto_steps: bool = False, # Infer minimum steps to reach target composition; converts steps into max_steps when to_extend is a composition
                   prioritize: bool = False,  # Rank candidates by the maximum flow reaching them; only informative if the input network carries 'abundance' node attributes
                   source: str = "leaves"  # Nodes to grow from: leaves (metabolic endpoints) or observed (every measured structure)
                   ) -> tuple[nx.DiGraph, set[str] | dict[str, float]]: # (Extended network, New glycans; a candidate:flow mapping sorted by descending flow when prioritize=True), optionally minimum number of steps from auto_steps
    "Extend biosynthetic network physiologically"
    graphs = {}
    new_glycans = set()
    classy = get_class(next(iter(network.nodes())))
    mammal_disac = set(unwrap(get_k_saccharides(list(network.nodes()), just_motifs = True))) if strict_context else set(
        _reference_disaccharides(classy))
    reactions = {r for r in nx.get_edge_attributes(network, "diffs").values() if all(x not in r for x in ('?', 'Hex', 'O', '/'))}
    # in a densely observed network a large glycan's precursors usually carry other observed children too, so restricting growth to leaves makes those targets unreachable rather than merely unlikely
    leaf_glycans = {x for x in network.nodes() if
                    network.out_degree(x) == 0 and network.in_degree(x) > 0} if source == "leaves" else {x for x, v in
                                                                                                         network.nodes(
                                                                                                             data = 'virtual')
                                                                                                         if
                                                                                                         v == 0 and network.in_degree(
                                                                                                             x) > 0}
    if isinstance(to_extend, str) and is_composition(to_extend):
        to_extend = canonicalize_composition(to_extend)
    if isinstance(to_extend, dict):
        existing = {g for g in network.nodes() if glycan_to_composition(g) == to_extend}
        if existing:
            print(
                "Target composition already present in network; skipping extension and instead returning matching structures from network.")
            return network, existing
        leaf_glycans = choose_leaves_to_extend(leaf_glycans, to_extend)
        reactions = {r for r in reactions if map_to_basic(r.split('(')[0]) in to_extend.keys()}
        if auto_steps:
            target_comp = Counter(to_extend)
            min_score = min(sum((target_comp - Counter(glycan_to_composition(g))).values()) for g in leaf_glycans)
            if min_score > steps:
                print(f"auto_steps: {min_score} step(s) needed but max steps is {steps}; aborting.")
                return network, set(), -1
            print(f"auto_steps: {min_score} step(s) needed to reach target composition.")
            steps = min_score
    if (isinstance(to_extend, str) and to_extend != "all") or isinstance(to_extend, list):
        leaf_glycans = {to_extend} if isinstance(to_extend, str) else set(to_extend)
    for _ in range(steps):
        new_leaf_glycans = extend_glycans(leaf_glycans, reactions, allowed_disaccharides = mammal_disac)
        if isinstance(to_extend, dict):
            new_leaf_glycans = choose_leaves_to_extend(new_leaf_glycans, to_extend)
        if not new_leaf_glycans:
            break
        new_edges, new_edge_labels = edges_for_extension(leaf_glycans, new_leaf_glycans, graphs)
        network = update_network(network, new_edges, edge_labels = new_edge_labels, node_labels = {k: 1 for k in new_leaf_glycans})
        leaf_glycans = new_leaf_glycans
        new_glycans.update(leaf_glycans)
    if prioritize and new_glycans:
        # candidates differ in how much of the observed glycome can physiologically reach them, which is the same flow criterion used to pick between isomers of one composition
        roots = sorted(r for r in infer_roots(frozenset(network.nodes())) if r in network)
        if not roots:
            raise ValueError(
                f"No biosynthetic root of this glycan class is present in the network (e.g., from '{next(iter(network.nodes()))}'), so candidates cannot be ranked by flow.")
        root = min(roots, key = len) if '-ol' in roots[0] else max(roots, key = len)
        weighted = get_edge_weight_by_abundance(network, root = root)
        caps = nx.get_edge_attributes(weighted, 'capacity')
        # a candidate reached through a single edge can carry no more than that edge or the flow arriving at its parent, so the solve is per parent, not per candidate
        simple = {g: next(iter(weighted.predecessors(g))) for g in new_glycans if weighted.in_degree(g) == 1}
        parent_flows = get_maximum_flow(weighted, source = root, sinks = sorted(set(simple.values())))
        flows = {g: {'flow_value': min(parent_flows[p]['flow_value'], caps.get((p, g), 0.0))} for g, p in simple.items()
                 if p in parent_flows}
        flows.update(get_maximum_flow(weighted, source = root, sinks = sorted(set(new_glycans) - set(simple))))
        new_glycans = dict(sorted(((g, v['flow_value']) for g, v in flows.items()), key = lambda kv: -kv[1]))
    return (network, new_glycans) if not auto_steps else (network, new_glycans, steps)


def get_biosynthetic_coherence(
        df: pd.DataFrame, # Glycan abundances (glycans as index or first column, samples as columns)
        group1: list[str] | None = None,  # First group column names; default: from the frame's contrasts
        group2: list[str] | None = None,  # Second group column names; default: from the frame's contrasts
        network: nx.DiGraph | None = None,  # Pre-built network; built from df if not provided
        paired: bool | None = None,  # Whether samples are paired; default: from the frame
        n_permutations: int = 20000,  # Label permutations for the exact shared-model null; 0 to skip
        random_state: int = 42  # Seed for permutation reproducibility
) -> tuple[
    pd.DataFrame, pd.DataFrame]:  # (Group-level results under both models, per-glycan coherence and its group difference)
    "Quantify how strongly each condition's glycome follows its own biosynthetic network, and which glycans decouple from it"
    from scipy.stats import ttest_ind, ttest_rel, t as tdist
    names = ('group1', 'group2')
    if group1 is None and isinstance(df, GlycoDataFrame) and df._contrasts:
        group1, group2 = list(df.group1), list(df.group2)
        names = (df.group1.name, df.group2.name)
    paired = df.paired if paired is None and isinstance(df, GlycoDataFrame) else bool(paired)
    paired = paired and len(group1) == len(group2)
    if not isinstance(df.index[0], str):
        df = df.set_index(df.columns[0])
    if network is None:
        network = construct_network(df.index.tolist())
    df = df.set_axis([canonicalize_iupac(g) for g in df.index])
    glycans = [g for g in df.index if network.nodes.get(g, {}).get('virtual', 1) == 0]
    gidx = {g: i for i, g in enumerate(glycans)}
    cols = list(group1) + list(group2)
    sub = df.loc[glycans, cols]
    col_sums = sub.sum(axis = 0)
    X_lin = ((sub / col_sums.where(col_sums > 0, 1)) * 100).values.astype(float)
    # Fractional conversion is multiplicative, so fit in log space; this also stops variance weighting from collapsing onto the few most abundant glycans
    X = np.log2(X_lin + 0.01)
    preds_map = {}
    for g in glycans:
        queue = list(network.predecessors(g))
        visited, preds = set(), []
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in gidx:
                preds.append(node)
            else:
                queue.extend(network.predecessors(node))
        preds_map[g] = preds
    n, n1 = X.shape[1], len(group1)
    idx_own = {t: [j for j in (range(n1) if t < n1 else range(n1, n)) if j != t] for t in range(n)}
    idx_shared = {t: [j for j in range(n) if j != t] for t in range(n)}
    own, shared, weights, dropped = {}, {}, {}, Counter()

    def _ridge(g, preds, idx):
        "Ridge precursor fit on a sample subset; coefficients are the fractional conversion precursor->product"
        center = np.array([X[gidx[p], idx].mean() for p in preds])
        Z = np.column_stack([X[gidx[p], idx] - center[i] for i, p in enumerate(preds)] + [np.ones(len(idx))])
        gram = Z.T @ Z
        gram[np.diag_indices(len(preds))] += max(0.1 * np.trace(gram[:-1, :-1]) / len(preds), 1e-8)
        return np.linalg.solve(gram, Z.T @ X[gidx[g], idx]), center

    def _fit_predict(g, preds, train, t):
        "Leave-one-out R2 of the precursor model, scored on held-out sample `t`"
        y = X[gidx[g]]
        coef, center = _ridge(g, preds, train)
        ss_base = (y[t] - y[train].mean()) ** 2
        pred = np.concatenate([np.array([X[gidx[p], t] for p in preds]) - center, [1.0]]) @ coef
        return float(np.clip(1.0 - float((y[t] - pred) ** 2) / ss_base, -3.0, 1.0)) if ss_base > 0 else np.nan

    # Two models, two questions: 'own' asks how tightly each condition follows its own network, 'shared' asks how
    # far each sample deviates from one common network and makes the label permutation below exact
    for g in glycans:
        preds = preds_map[g]
        var_g = float(np.var(X[gidx[g]]))
        if not preds:
            dropped['no observed precursor'] += 1
            continue
        if var_g < 1e-3:
            dropped['near-zero variance'] += 1
            continue
        rs = np.array([_fit_predict(g, preds, idx_shared[t], t) for t in range(n)])
        if np.isnan(rs).all():
            dropped['no scorable sample'] += 1
            continue
        own[g] = np.array([_fit_predict(g, preds, idx_own[t], t) if len(idx_own[t]) > 1 else np.nan for t in range(n)])
        shared[g], weights[g] = rs, var_g
    if not shared:
        raise ValueError("No glycan had both an observed precursor and enough variance to be scored")
    S = pd.DataFrame(shared, index = cols).T
    O = pd.DataFrame(own, index = cols).T.loc[S.index]
    w = np.array([weights[g] for g in S.index])[:, None]
    own_den = np.nansum(~np.isnan(O.values) * w,
                        axis = 0)  # zero when a group is too small to leave out a sample and still fit
    own_scores = np.divide(np.nansum(O.values * w, axis = 0), own_den, out = np.full(n, np.nan), where = own_den > 0)
    shared_scores = np.nansum(S.values * w, axis = 0) / np.nansum(~np.isnan(S.values) * w, axis = 0)
    o1, o2, s1, s2 = own_scores[:n1], own_scores[n1:], shared_scores[:n1], shared_scores[n1:]
    stat_o, p_o = ttest_rel(o2, o1, nan_policy = 'omit') if paired else ttest_ind(o2, o1, equal_var = False, nan_policy = 'omit')
    _, p_s = ttest_rel(s2, s1) if paired else ttest_ind(s2, s1, equal_var = False)
    effect, _ = cohen_d(o2, o1, paired = paired)
    rng = np.random.default_rng(random_state)
    obs, null, diffs = float(s2.mean() - s1.mean()), [], s2 - s1
    for _ in range(n_permutations):
        if paired:
            null.append(float(np.where(rng.random(len(diffs)) < 0.5, -diffs, diffs).mean()))
        else:
            perm = rng.permutation(shared_scores)
            null.append(float(perm[n1:].mean() - perm[:n1].mean()))
    p_perm = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_permutations + 1)) if n_permutations else np.nan
    rows, A, B = [], [], []
    for g in S.index:
        a, b = S.loc[g, group1].values.astype(float), S.loc[g, group2].values.astype(float)
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        # Below three samples a group the centered design is rank-deficient and the ridge floor, not the data, sets the coefficient
        dc = _ridge(g, preds_map[g], list(range(n1, n)))[0][:-1] - _ridge(g, preds_map[g], list(range(n1)))[0][
            :-1] if min(n1, n - n1) > 2 else None
        k = int(np.argmax(np.abs(dc))) if dc is not None else None
        A.append(a)
        B.append(b)
        rows.append({'glycan': g, 'top_changed_precursor': preds_map[g][k] if k is not None else None,
                     'precursor_coef_change': float(dc[k]) if k is not None else np.nan,
                     'group1_r2': float(a.mean()), 'group2_r2': float(b.mean()),
                     'difference': float(b.mean() - a.mean()), 'n_precursors': len(preds_map[g]),
                     'branches': g.count('['), 'n_Fuc': g.count('Fuc'), 'n_Sia': g.count('Neu5Ac') + g.count('Neu5Gc'),
                     'mean_abundance': float(X_lin[gidx[g]].mean()), 'variance_weight': weights[g]})
    per_glycan_df = pd.DataFrame(rows).set_index('glycan') if rows else pd.DataFrame(columns = ['glycan']).set_index('glycan')
    if len(per_glycan_df):
        A, B = np.array(A), np.array(B)
        if paired:
            D = B - A
            eff, dfree, scale, resid_var = D.mean(axis = 1), D.shape[1] - 1, 1 / D.shape[1], D.var(axis = 1, ddof = 1)
        else:
            na, nb = A.shape[1], B.shape[1]
            eff, dfree, scale = B.mean(axis = 1) - A.mean(axis = 1), na + nb - 2, 1 / na + 1 / nb
            resid_var = ((na - 1) * A.var(axis = 1, ddof = 1) + (nb - 1) * B.var(axis = 1, ddof = 1)) / dfree
        # A handful of R2 values per group leaves each glycan's variance dominated by noise; neighbors in the biosynthetic graph share precursors and so make a local prior
        s2_mod, df_post = moderated_variance(resid_var, dfree, dag_neighbors(list(per_glycan_df.index), network))
        per_glycan_df['p_val'] = np.maximum(2 * tdist.sf(np.abs(eff / np.sqrt(s2_mod * scale)), df_post),
                                            np.finfo(float).tiny)
        per_glycan_df['corr_p_val'], per_glycan_df['rewired'] = correct_multiple_testing(per_glycan_df.p_val,
                                                                                         get_alphaN(len(cols)))
        per_glycan_df = per_glycan_df.sort_values('corr_p_val')
    return pd.DataFrame([{
        'group1_mean': float(o1.mean()), 'group2_mean': float(o2.mean()), 'difference': float(o2.mean() - o1.mean()),
        't_statistic': float(stat_o), 'p_val': float(p_o), 'cohens_d': float(effect),
        'shared_model_difference': obs, 'shared_model_p_val': float(p_s), 'shared_model_p_val_permutation': p_perm,
        'null_sd': float(np.std(null)) if null else np.nan,
        'n_rewired_glycans': int(per_glycan_df.rewired.sum()) if len(per_glycan_df) else 0,
        'n_rewired_up': int((per_glycan_df.rewired & (per_glycan_df.difference > 0)).sum()) if len(
            per_glycan_df) else 0,
        'group1_name': names[0], 'group2_name': names[1],
        'n_glycans_scored': len(S), 'n_glycans_observed': len(glycans),
        'coverage': float(sum(weights.values()) / sum(float(np.var(X[gidx[g]])) for g in glycans)),
        'dropped': dict(dropped)
    }], index = ['global_r2_weighted']).assign(group1_scores = [o1.tolist()], group2_scores = [o2.tolist()],
                                               shared_group1_scores = [s1.tolist()],
                                               shared_group2_scores = [s2.tolist()]), per_glycan_df