import re
import sys
from copy import deepcopy
from typing import Callable
from glycowork.glycan_data.loader import unwrap, modification_map, HashableDict
from glycowork.motif.processing import min_process_glycans, get_possible_linkages, get_possible_monosaccharides, rescue_glycans, parse_floating_bit
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter, OrderedDict
from functools import lru_cache, wraps


PTM_REGEX = re.compile(r"(?<=[A-Za-z])(?<!Neu)(\d+/\d+|\d+)(?=\D)(?![^()]*\))")
NEGATION_REGEX = re.compile(r'(?<!\()(!\w+(?:\([^)]+\))?)')
MONO_PATTERN = re.compile(r"^(Hex|HexOS|HexNAc|HexNAcOS|dHex|Sia|HexA|Pen|Monosaccharide)$")
LINKAGE_PATTERN = re.compile(r'[ab\?][12]-(\d+|\?)')
LINKAGE_LABEL = re.compile(r'^[ab?][12]-')
from weakref import WeakKeyDictionary
_SL_CACHE = WeakKeyDictionary()
_NEG_CACHE = WeakKeyDictionary()
_PTM_CACHE = WeakKeyDictionary()
_HAS_O_CACHE = WeakKeyDictionary()

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


def _sl(g):  # cached string_labels, keyed on graph identity (subgraph views share g.graph, so it cannot live there)
    v = _SL_CACHE.get(g)
    if v is None:
        v = _SL_CACHE[g] = [d['string_labels'] for _, d in g.nodes(data = True)]
    return v


def _has_o(g):  # cached "needs PTM wildcarding" flag, keyed on graph identity
    v = _HAS_O_CACHE.get(g)
    if v is None:
        v = _HAS_O_CACHE[g] = any('O' in s for s in _sl(g))
    return v


def _ptm_wildcarded(g):  # cached PTM-wildcarded copy, keyed on graph identity
    v = _PTM_CACHE.get(g)
    if v is None:
        v = _PTM_CACHE[g] = ptm_wildcard_for_graph(g.copy())
    return v


@memoize_node_match
def build_wildcard_cache(proc: set) -> dict:
    """Precompute all possible wildcard expansions"""
    return {k: get_possible_linkages(k) for k in proc if '?' in k or ('/' in k and '-' in k)} | \
        {k: get_possible_monosaccharides(k) for k in proc if MONO_PATTERN.match(k) or k.startswith('!') or ('/' in k and '-' not in k)}


def resolve_anchor(ggraph: nx.DiGraph, # Glycan graph to search
                   anchor: str # Anchor motif with ^ marking the acceptor residue
                   ) -> set[int]: # Nodes of ggraph the marked residue could correspond to
    "Find all residues of a glycan graph that match the ^-marked residue of an anchor motif"
    idx = next(i for i, x in enumerate(min_process_glycans([anchor])[0]) if x.endswith('^'))
    g2 = glycan_to_nxGraph(anchor.replace('^', ''))
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(ggraph, g2, node_match = categorical_node_match_wildcard(
        'string_labels', 'unknown', build_wildcard_cache(set(_sl(ggraph)) | set(_sl(g2))), 'termini', 'flexible'))
    return {k for m in matcher.subgraph_isomorphisms_iter() for k, v in m.items() if v == idx}


def glycan_to_graph(glycan: str  # IUPAC-condensed glycan sequence
                    ) -> tuple[dict[int, str], np.ndarray]: # (Dictionary of node:monosaccharide/linkage, Adjacency matrix)
    "Convert glycans into graphs"
    # Get glycoletters
    glycan_proc = tuple(min_process_glycans([glycan])[0])
    n = len(glycan_proc)
    # Map glycoletters to integers
    mask_dic = dict(enumerate(glycan_proc))
    temp_glycan = glycan
    for i, gl in mask_dic.items():
        temp_glycan = temp_glycan.replace(gl, str(i), 1)
    # Initialize adjacency matrix
    adj_matrix = np.zeros((n, n), dtype = np.uint8)
    tokens = re.findall(r'\d+|\[|\]', temp_glycan)
    current_node = None
    stack = []
    for token in reversed(tokens):
        if token == ']':
            stack.append(current_node)
        elif token == '[':
            current_node = stack.pop()
        else:
            node_idx = int(token)
            if current_node is not None:
                adj_matrix[node_idx, current_node] = 1
            current_node = node_idx
    return mask_dic, adj_matrix


@lru_cache(maxsize = None)
def glycan_to_nxGraph_int(glycan: str, # Glycan in IUPAC-condensed format
                          libr: dict[str, int] | HashableDict[str, int] | None = None, # Dictionary of form glycoletter:index
                          termini: str = 'ignore', # How to encode terminal/internal position; options: ignore, calc, provided
                          termini_list: tuple[str] | None = None # List of positions from terminal/internal/flexible
                          ) -> nx.DiGraph: # NetworkX graph object of glycan
    "Convert glycans into networkx graphs"
    # This allows to make glycan graphs of motifs ending in a linkage
    appended_hex = glycan.endswith(')')
    if appended_hex:
        glycan += 'Hex'
    # Map glycan string to node labels and adjacency matrix
    node_dict, adj_matrix = glycan_to_graph(glycan)
    # Create directed graph directly from adjacency matrix
    edges = np.where(np.triu(adj_matrix, k = 1).T)  # k=1 excludes diagonal
    g1 = nx.DiGraph()
    g1.add_nodes_from(node_dict.keys())
    g1.add_edges_from(zip(edges[0].tolist(), edges[1].tolist()))
    # Remove the helper monosaccharide if used
    if appended_hex and glycan.endswith('x'):
        last_node = len(node_dict) - 1
        node_dict.pop(last_node, None)
        g1.remove_node(last_node)
    # Add node labels
    node_attributes = {
        i: {'string_labels': sys.intern(v)} if libr is None else {'labels': libr[v], 'string_labels': sys.intern(v)}
        for i, v in node_dict.items()}
    nx.set_node_attributes(g1, node_attributes)
    if termini == 'calc':
        last_node = max(g1.nodes())
        degrees = dict(g1.degree())
        nx.set_node_attributes(g1, {k: 'terminal' if (degrees[k] == 1 or k == last_node) else 'internal' for k in g1.nodes()}, 'termini')
    elif termini == 'provided':
        nx.set_node_attributes(g1, dict(zip(g1.nodes(), termini_list)), 'termini')
    return g1


@rescue_glycans
def glycan_to_nxGraph(glycan: str, # Glycan in IUPAC-condensed format
                      libr: HashableDict[str, int] | None = None, # Dictionary of form glycoletter:index
                      termini: str = 'ignore', # How to encode terminal/internal position; options: ignore, calc, provided
                      termini_list: tuple[str] | None = None # List of positions from terminal/internal/flexible
                      ) -> nx.DiGraph: # NetworkX graph object of glycan
    "Wrapper for converting glycans into networkx graphs; also works with floating substituents"
    if not glycan:
        return nx.DiGraph()
    if isinstance(libr, dict) and not isinstance(libr, HashableDict):
        libr = HashableDict(libr)
    if any(k in glycan for k in [';', 'β', 'α', 'RES', '=']):
        raise Exception
    termini_list = expand_termini_list(glycan, termini_list) if termini_list else None
    if '{' in glycan:
        chunks = [k for k in glycan.replace('}', '{').split('{') if k]
        chunks, anchor_specs = zip(
            *[(k, {}) if i == len(chunks) - 1 else parse_floating_bit(k) for i, k in enumerate(chunks)])
        parts = [glycan_to_nxGraph_int(k, libr = libr, termini = termini,
                                       termini_list = termini_list) for k in chunks]
        len_org = len(parts[-1])
        for i, p in enumerate(parts[:-1]):
            parts[i] = nx.relabel_nodes(p, {pn: pn + len_org for pn in p.nodes()})
            len_org += len(p)
            if anchor_specs[i]:
                parts[i].nodes[max(parts[i].nodes())]['anchors'] = anchor_specs[i]
        g1 = nx.compose_all(parts)
    else:
        g1 = glycan_to_nxGraph_int(glycan, libr = libr, termini = termini,
                                   termini_list = termini_list)
    return g1


def ensure_graph(glycan: str | nx.DiGraph, # Glycan in IUPAC-condensed format or as networkx graph
                 **kwargs # Keyword arguments passed to glycan_to_nxGraph
                 ) -> nx.DiGraph: # NetworkX graph object of glycan
    "Ensures function compatibility with string glycans and graph glycans"
    return glycan_to_nxGraph(glycan, **kwargs) if isinstance(glycan, str) else glycan


def categorical_node_match_wildcard(attr: str | tuple[str, ...], # Attribute or tuple of attributes to match
                                    default: str, # Default value if attribute is missing
                                    narrow_wildcard_list: dict[str, set[str]], # Dictionary mapping wildcards to possible matches
                                    attr2: str, # Secondary attribute to match
                                    default2: str # Default value for secondary attribute
                                    ) -> Callable[[dict, dict], bool]: # Function that matches node attributes
    "Match nodes while handling wildcards and flexible positions"
    def match(data1, data2):
        data1_labels2, data2_labels2 = data1.get(attr2, default2), data2.get(attr2, default2)
        if data1_labels2 != data2_labels2 and data1_labels2 != 'flexible' and data2_labels2 != 'flexible':
            return False
        data1_labels, data2_labels = data1.get(attr, default), data2.get(attr, default)
        if data1_labels == data2_labels:
            return True
        if (data1_labels == "Monosaccharide" or data2_labels == "Monosaccharide") and '-' not in data1_labels and '-' not in data2_labels:
            return True
        if data1_labels == "?1-?" or data2_labels == "?1-?":
            if data1_labels.count('-') == 1 and data2_labels.count('-') == 1:
                return True
        if data2_labels.startswith('!'):
            if data1_labels != data2_labels[1:] and '-' not in data1_labels:
                return True
        if data1_labels in narrow_wildcard_list and data2_labels in narrow_wildcard_list[data1_labels]:
            return True
        if data2_labels in narrow_wildcard_list and data1_labels in narrow_wildcard_list[data2_labels]:
            return True
        return False
    return match


def ptm_wildcard_for_graph(graph: nx.DiGraph # Input graph
                           ) -> nx.DiGraph: # Modified graph with PTM wildcards
    "Standardize PTM wildcards in graph"
    _SL_CACHE.pop(graph, None)
    _PTM_CACHE.pop(graph, None)
    _HAS_O_CACHE.pop(graph, None)
    for node in graph.nodes:
        if not LINKAGE_LABEL.match(label := graph.nodes[node]['string_labels']):
            graph.nodes[node]['string_labels'] = PTM_REGEX.sub('O', label)
    return graph


def _prefilter_labels(g1_labels: list, # G1 node labels
                      g2_labels: list, # G2 node labels
                      narrow_wildcard_list: dict # Wildcard expansion dict
                      ) -> bool: # True if label multisets are compatible
    "Conservative multiset pre-filter: each g2 label needs enough compatible g1 labels to cover its count"
    g1_counter = Counter(g1_labels)
    for l2, need in Counter(g2_labels).items():
        have = 0
        for l1, cnt in g1_counter.items():
            if (l1 == l2
                    or ((l1 == 'Monosaccharide' or l2 == 'Monosaccharide') and '-' not in l1 and '-' not in l2)
                    or ((l1 == '?1-?' or l2 == '?1-?') and '-' in l1 and '-' in l2)
                    or (l2.startswith('!') and l1 != l2[1:] and '-' not in l1)
                    or (l1.startswith('!') and l2 != l1[1:] and '-' not in l2)
                    or l2 in narrow_wildcard_list.get(l1, frozenset())
                    or l1 in narrow_wildcard_list.get(l2, frozenset())):
                have += cnt
                if have >= need:
                    break
        if have < need:
            return False
    return True


def compare_glycans(glycan_a: str | nx.DiGraph, # First glycan to compare
                    glycan_b: str | nx.DiGraph, # Second glycan to compare
                    return_matches: bool = False # Whether to return node mapping between glycans
                    ) -> bool: # True if glycans are same, False if not
    "Check whether two glycans are identical"
    if glycan_a == glycan_b:
        return ((True, {n: n for n in glycan_a.nodes}) if return_matches else True) if isinstance(glycan_a,
                                                                                                  nx.DiGraph) else (
            True, None) if return_matches else True
    anchored = lambda g: ('^' in g) if isinstance(g, str) else any('anchors' in d for _, d in g.nodes(data = True))
    if anchored(glycan_a) or anchored(glycan_b):
        topos = [get_possible_topologies(g, return_graphs = True) if anchored(g) else [ensure_graph(g)] for g in
                 (glycan_a, glycan_b)]
        same = len(topos[0]) == len(topos[1]) and all(
            any(compare_glycans(ta, tb) for tb in topos[1]) for ta in topos[0])
        return (same, None) if return_matches else same
    if isinstance(glycan_a, str) and isinstance(glycan_b, str):
        if glycan_a.count('(') != glycan_b.count('(') or glycan_a.count("[") != glycan_b.count("["):
            return (False, None) if return_matches else False
        proc = set(unwrap(min_process_glycans([glycan_a, glycan_b])))
        if 'O' in glycan_a or 'O' in glycan_b:
            glycan_a, glycan_b = PTM_REGEX.sub('O', glycan_a), PTM_REGEX.sub('O', glycan_b)
        g1, g2 = glycan_to_nxGraph(glycan_a), glycan_to_nxGraph(glycan_b)
        g1_sl, g2_sl = nx.get_node_attributes(g1, "string_labels"), nx.get_node_attributes(g2, "string_labels")
    else:
        glycan_a, glycan_b = ensure_graph(glycan_a), ensure_graph(glycan_b)
        if len(glycan_a.nodes) != len(glycan_b.nodes):
            return (False, None) if return_matches else False
        g1_sl, g2_sl = nx.get_node_attributes(glycan_a, "string_labels"), nx.get_node_attributes(glycan_b, "string_labels")
        proc = set(g1_sl.values()) | set(g2_sl.values())
        if any('O' in s for s in proc):
            g1, g2 = _ptm_wildcarded(glycan_a), _ptm_wildcarded(glycan_b)
            g1_sl = nx.get_node_attributes(g1, "string_labels")
            g2_sl = nx.get_node_attributes(g2, "string_labels")
        else:
            g1, g2 = glycan_a, glycan_b
    if sorted(d for _, d in g1.out_degree()) != sorted(d for _, d in g2.out_degree()):
        return (False, None) if return_matches else False
    narrow_wildcard_list = build_wildcard_cache(proc)
    if narrow_wildcard_list:
        g1_labels = list(g1_sl.values())
        g2_labels = list(g2_sl.values())
        if not _prefilter_labels(g1_labels, g2_labels, narrow_wildcard_list) or \
                not _prefilter_labels(g2_labels, g1_labels, narrow_wildcard_list):
            return (False, None) if return_matches else False
        matcher = nx.isomorphism.DiGraphMatcher(g1, g2, categorical_node_match_wildcard('string_labels', 'unknown', narrow_wildcard_list, 'termini', 'flexible'))
    else:
        # First check whether components of both glycan graphs are identical, then check graph isomorphism (costly)
        if sorted(g1_sl.values()) != sorted(g2_sl.values()):
            return (False, None) if return_matches else False
        if graph_to_string(g1) != graph_to_string(g2):
            return (False, None) if return_matches else False
        if not return_matches:
            return True
        matcher = nx.isomorphism.DiGraphMatcher(g1, g2, nx.algorithms.isomorphism.categorical_node_match('string_labels',
                                                                                                         'unknown'))
    for mapping in matcher.isomorphisms_iter():
        return (True, mapping) if return_matches else True
    return (False, None) if return_matches else False


def expand_termini_list(motif: str | nx.DiGraph, # Glycan motif sequence or graph
                        termini_list: list[str] # List of monosaccharide/linkage positions from terminal/internal/flexible
                        ) -> tuple[str]: # Expanded termini list including linkages
    "Convert monosaccharide-only termini list into full termini list"
    mapping = {'t': 'terminal', 'i': 'internal', 'f': 'flexible'}
    termini_list = [mapping.get(t, t) for t in termini_list]
    num_linkages = motif.count('(') if isinstance(motif, str) else len(motif) - len(termini_list)
    result = ['flexible'] * (len(termini_list) + num_linkages)
    result[::2] = termini_list
    return tuple(result)


def handle_negation(original_func: Callable # Function to wrap
                    ) -> Callable: # Wrapped function handling negation
    "Decorator for handling negation patterns in glycan matching functions"
    @wraps(original_func)
    def wrapper(glycan, motif, *args, **kwargs):
        if isinstance(motif, str) and '!' in motif:
            return subgraph_isomorphism_with_negation(glycan, motif, *args, **kwargs)
        elif hasattr(motif, 'nodes'):
            hit = _NEG_CACHE.get(motif)
            if hit is None:
                hit = _NEG_CACHE[motif] = any('!' in d.get('string_labels', '') for _, d in motif.nodes(data = True))
            return subgraph_isomorphism_with_negation(glycan, motif, *args, **kwargs) if hit else original_func(glycan,
                                                                                                                motif,
                                                                                                                *args,
                                                                                                                **kwargs)
        else:
            return original_func(glycan, motif, *args, **kwargs)
    return wrapper


@handle_negation
def subgraph_isomorphism(glycan: str | nx.DiGraph, # Glycan sequence or graph
                         motif: str | nx.DiGraph, # Glycan motif sequence or graph
                         termini_list: list = [], # List of monosaccharide positions from terminal/internal/flexible
                         count: bool = False, # Whether to return count instead of presence/absence
                         return_matches: bool = False # Whether to return matched subgraphs as node lists
                         ) -> bool | int | tuple[int, list[list[int]]]: # Boolean presence, count, or (count, matches)
    "Check if motif exists as subgraph in glycan"
    if isinstance(glycan, str) and isinstance(motif, str):
        if motif.count('(') > glycan.count('('):
            return (0, []) if return_matches else 0 if count else False
        if not count and not return_matches and not termini_list:
            m_len, i = len(motif), glycan.find(motif)
            while i != -1:
                if (i == 0 or glycan[i - 1] in '([)]') and (i + m_len == len(glycan) or glycan[i + m_len] in ')]'):
                    return True
                i = glycan.find(motif, i + 1)
        if 'O' in glycan or 'O' in motif:
            glycan, motif = PTM_REGEX.sub('O', glycan), PTM_REGEX.sub('O', motif)
        motif_comp = min_process_glycans([motif, glycan])
        g1 = glycan_to_nxGraph(glycan, termini = 'calc' if termini_list else 'ignore')
        g2 = glycan_to_nxGraph(motif, termini = 'provided' if termini_list else 'ignore', termini_list = termini_list)
    else:
        if len(glycan.nodes) < len(motif.nodes):
            return (0, []) if return_matches else 0 if count else False
        if termini_list and not nx.get_node_attributes(motif, 'termini'):
            motif = motif.copy()
            nx.set_node_attributes(motif, dict(zip(motif.nodes(), expand_termini_list(motif, termini_list) if len(termini_list) < len(motif) else termini_list)), 'termini')
        if _has_o(motif) or _has_o(glycan):
            g1, g2 = _ptm_wildcarded(glycan), _ptm_wildcarded(motif)
        else:
            g1, g2 = glycan, motif
        motif_comp = [_sl(g2), _sl(g1)]
    narrow_wildcard_list = build_wildcard_cache(set(unwrap(motif_comp)))
    if termini_list or narrow_wildcard_list:
        # no wildcards anywhere => node_match is plain label equality, so a label-subset check is sound
        if not narrow_wildcard_list and not set(_sl(g2)).issubset(set(_sl(g1))):
            return (0, []) if return_matches else 0 if count else False
        if narrow_wildcard_list and not _prefilter_labels(_sl(g1), _sl(g2), narrow_wildcard_list):
            return (0, []) if return_matches else 0 if count else False
        graph_pair = nx.algorithms.isomorphism.DiGraphMatcher(g1, g2, node_match = categorical_node_match_wildcard('string_labels', 'unknown', narrow_wildcard_list,
                                                                                                                   'termini', 'flexible'))
    else:
        g1_node_attr = set(_sl(g1))
        if not set(motif_comp[0]).issubset(g1_node_attr):
            return (0, []) if return_matches else 0 if count else False
        graph_pair = nx.algorithms.isomorphism.DiGraphMatcher(g1, g2, node_match = nx.algorithms.isomorphism.categorical_node_match('string_labels', 'unknown'))
    # Count motif occurrence
    valid_mappings, seen = [], set()
    for mapping in graph_pair.subgraph_isomorphisms_iter():
        if not return_matches and not count:
            return True
        key = frozenset(mapping.keys())
        if key not in seen:
            seen.add(key)
            valid_mappings.append(sorted(mapping.keys()))
    if count:
        total = len(valid_mappings)
        return (total, valid_mappings) if return_matches else total
    elif return_matches:
        return (1 if valid_mappings else 0, valid_mappings)  # (count, matches)
    else:
        return bool(valid_mappings)


def subgraph_isomorphism_with_negation(glycan: str | nx.DiGraph, # Glycan sequence or graph
                                       motif: str | nx.DiGraph, # Glycan motif sequence or graph
                                       termini_list: list = [], # List of monosaccharide positions
                                       count: bool = False, # Whether to return count instead of presence/absence
                                       return_matches: bool = False # Whether to return matched subgraphs as node lists
                                       ) -> bool | int | tuple[int, list[list[int]]]: # Boolean presence, count, or (count, matches)
    """Check if motif exists as subgraph in glycan, handling negation patterns
    Where the negated residue sits decides what the negation means:
    1. Leaf position, i.e., leading ('!Neu5Ac(a2-3)Gal(b1-4)GlcNAc') or a bracketed branch ('Gal(b1-3)[!GlcNAc(b1-6)]GalNAc'):
       the residue is dropped from the motif and the match requires its absence, so bare Gal(b1-4)GlcNAc matches the first
    2. Internal or root position ('Neu5Ac(a2-3)!Gal(b1-4)GlcNAc', 'Gal(b1-4)!GlcNAc'): dropping it would disconnect the motif,
       so it becomes a wildcard that must be filled by something other than the negated residue"""
    if isinstance(motif, str):
        if not (hit := NEGATION_REGEX.search(motif)) or hit.group(1) == motif:
            raise ValueError(
                f"Unsupported negation in motif '{motif}': negate a monosaccharide inside a larger motif (e.g. 'Gal(b1-4)[!Fuc(a1-3)]GlcNAc'), not a linkage or the whole motif")
        negated_part = hit.group(1)
        to_replace = '' if motif.startswith('!') or '[!' in motif else 'Monosaccharide(?1-?)' if negated_part.endswith(
            ')') else 'Monosaccharide'
        motif_stub = motif.replace(negated_part, to_replace).replace('[]', '')
        negated_part_clean = glycan_to_nxGraph(negated_part.replace('!', ''))
    else:
        motif_copy = deepcopy(motif)
        motif_stub = motif_copy.copy()
        negated_nodes = {n for n, data in motif.nodes(data = True) if '!' in data.get('string_labels', '')}
        negated_nodes.update({node + 1 for node in negated_nodes if node + 1 in motif_stub})
        motif_stub.remove_nodes_from(negated_nodes)
        negated_part_clean = motif_copy.subgraph(negated_nodes).copy()
        for node in negated_part_clean.nodes():
            negated_part_clean.nodes[node]['string_labels'] = negated_part_clean.nodes[node]['string_labels'].replace(
                '!', '')
        if termini_list:
            # The spec describes the full motif, so expand it there and keep only what the stub retained; the stub's node ids are non-contiguous, so a positional zip lands on the wrong nodes or fails to fit at all
            expanded = expand_termini_list(motif, termini_list) if len(termini_list) < len(motif) else termini_list
            termini_list = [expanded[n] for n in motif_stub.nodes()]
    res = subgraph_isomorphism.__wrapped__(glycan, motif_stub, termini_list = termini_list, count = count,
                                           return_matches = True)
    if not res[0]:
        return (0, []) if return_matches else 0 if count else False
    ggraph = glycan_to_nxGraph(glycan) if isinstance(glycan, str) else glycan
    if isinstance(motif, str):
        positive = glycan_to_nxGraph(motif.replace('!', ''))
    else:
        positive = motif_copy
        for node in positive.nodes():
            positive.nodes[node]['string_labels'] = positive.nodes[node]['string_labels'].replace('!', '')
    # A stub hit only dies to a hit of the fully positive motif that contains it, so the negated part has to sit exactly where the motif puts it instead of anywhere in the neighborhood
    extra = len(positive) - len(res[1][0])
    _, positive_matches = subgraph_isomorphism.__wrapped__(ggraph, positive, count = True, return_matches = True)
    valid_matches = []
    for match_nodes in res[1]:
        match_set = set(match_nodes)
        if not any(len(hit) - len(match_nodes) == extra and match_set.issubset(hit) and
                   (not extra or subgraph_isomorphism.__wrapped__(ggraph.subgraph(set(hit) - match_set),
                                                                  negated_part_clean))
                   for hit in positive_matches):
            valid_matches.append(match_nodes)
    if count:
        total = len(valid_matches)
        return (total, valid_matches) if return_matches else total
    elif return_matches:
        return (1 if valid_matches else 0, valid_matches)
    else:
        return bool(valid_matches)


def generate_graph_features(glycan: str | nx.DiGraph, # Glycan sequence or network graph
                            glycan_graph: bool = True, # True if input is glycan, False if network
                            label: str = 'network' # Label for output dataframe if glycan_graph=False
                            ) -> pd.DataFrame: # Dataframe of graph features
    "Compute graph features of glycan or network"
    g = ensure_graph(glycan) if glycan_graph else glycan
    glycan = label if not glycan_graph else glycan
    nbr_node_types = len(set(nx.get_node_attributes(g, "string_labels").values()) if glycan_graph else set(g.nodes()))
    # Adjacency matrix:
    A = nx.to_numpy_array(g)
    N = A.shape[0]
    connected = nx.is_weakly_connected(g)
    g_undir = g.to_undirected()
    if N == 1:
        deg = np.array([0])
        deg_to_leaves = np.array([0])
        centralities = {'betweeness': [0.0], 'eigen': [1.0], 'close': [0.0], 'load': [0.0],
                        'harm': [0.0], 'flow': [], 'flow_edge': [], 'secorder': []}
    else:
        deg = np.sum(A, axis = 1) + np.sum(A, axis = 0)
        deg_to_leaves = A[:, deg == 1].sum(axis = 1)
        centralities = {'betweeness': list(nx.betweenness_centrality(g).values()), 'eigen': list(nx.katz_centrality_numpy(g).values()),
                        'close': list(nx.closeness_centrality(g).values()), 'load': list(nx.load_centrality(g_undir).values()),
                        'harm': list(nx.harmonic_centrality(g).values()), 'flow': list(nx.current_flow_betweenness_centrality(g_undir).values()) if connected else [],
                        'flow_edge': list(nx.edge_current_flow_betweenness_centrality(g_undir).values()) if connected else [],
                        'secorder': list(nx.second_order_centrality(g_undir).values()) if connected else []}
    features = {
        'diameter': np.nan if not connected or N == 1 else nx.diameter(g_undir),
        'branching': np.sum(deg > 2), 'nbrLeaves': np.sum(deg == 1),
        'avgDeg': np.mean(deg), 'varDeg': np.var(deg),
        'maxDeg': np.max(deg), 'nbrDeg4': np.sum(deg > 3),
        'max_deg_leaves': np.max(deg_to_leaves), 'mean_deg_leaves': np.mean(deg_to_leaves),
        'deg_assort': 0.0 if N == 1 else nx.degree_assortativity_coefficient(g_undir), 'size_corona': 0 if N <= 1 else len(nx.k_corona(g_undir, max(nx.core_number(g_undir).values())).nodes()),
        'size_core': 0 if N <= 1 else len(nx.k_core(g_undir).nodes()), 'nbr_node_types': nbr_node_types,
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
        from scipy.sparse.linalg import eigsh
        eigval, vec = eigsh(M, 2, which = 'LM')
        distr = np.abs(vec[:, -1]) / sum(np.abs(vec[:, -1]))
        features.update({'egap': 1 - eigval[0], 'entropyStation': np.sum(distr * np.log(distr))})
    return pd.DataFrame(features, index = [glycan])


def get_linkage_number(node, graph):
    link_label = graph.nodes[node].get("string_labels", "")
    match = LINKAGE_PATTERN.search(link_label)
    if match:  # Extract the second number in the linkage (e.g., from "a1-3" get "3")
        linkage_num = match.group(1)
        if linkage_num == "?":
            return float('inf')  # Wildcard comes last
        return int(linkage_num) + 100 if '/' in link_label else int(linkage_num)  # Slash-wildcards rank after specified but before ?
    return float('inf')


def glycan_graph_memoize(maxsize: int = 128):
    cache = OrderedDict()
    def decorator(func):
        @wraps(func)
        def wrapper(graph, *args, **kwargs):
            if len(graph) < 4:
                return func(graph, *args, **kwargs)
            node_ids = sorted(graph.nodes())
            key = (tuple(graph.nodes[n]["string_labels"] for n in node_ids),
                   tuple(graph.out_degree(n) for n in node_ids),
                   tuple(sorted(graph.edges())), args, tuple(sorted(kwargs.items())))
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            if len(cache) >= maxsize:
                cache.popitem(last = False)
            cache[key] = func(graph, *args, **kwargs)
            return cache[key]
        return wrapper
    return decorator


@glycan_graph_memoize(maxsize = 128)
def graph_to_string_int(graph: nx.DiGraph, # Glycan graph
                        canonicalize: bool = True,  # Whether to output canonicalized IUPAC-condensed
                        order_by: str = "length"  # canonicalize by 'length' or 'linkage'
                        ) -> str: # IUPAC-condensed glycan string
    """Convert glycan graph back to IUPAC-condensed format
    Can canonicalize a glycan graph according to the following rules:
    1. Longest chain is the main chain
    2. Tie breakers:
       a. Lowest incoming linkage number
       b. Integer linkages preferred over wildcards
       c. Alphabetical ordering of leaf nodes"""
    if len(graph) == 3:
        nodes = sorted(list(graph.nodes()))
        node_labels = nx.get_node_attributes(graph, "string_labels")
        return f"{node_labels[nodes[0]]}({node_labels[nodes[1]]}){node_labels[nodes[2]]}"
    # Get the root node (highest index)
    root_idx = max(graph.nodes())
    # Build depths with a single traversal
    depths, leaf_labels, subtree_keys = {}, {}, {}

    def compute_metrics(node):
        if node in depths:
            return depths[node], leaf_labels[node]
        successors = list(graph.successors(node))
        label = graph.nodes[node].get("string_labels", "")
        if not successors:
            depths[node] = 0
            leaf_labels[node] = label
            subtree_keys[node] = label
        else:
            for child in successors:
                compute_metrics(child)
            depths[node] = 1 + max(depths[child] for child in successors)
            leaf_labels[node] = min(leaf_labels[child] for child in successors)
            subtree_keys[node] = label + ''.join(sorted(subtree_keys[child] for child in successors))
        return depths[node], leaf_labels[node]

    compute_metrics(root_idx)

    def is_special_branch(node):
        """Check if a 1-mono branch starting at node contains Gal/Man."""
        descendants = list(nx.descendants(graph, node))
        if len(descendants) != 1:
            return False
        tip_string = graph.nodes[descendants[0]].get("string_labels", "")
        if tip_string in {"Gal", "Man"}:
            return False
        if "GlcNAc" in tip_string and graph.nodes[node].get("string_labels", "") == 'b1-3':
            return False
        return True

    # Convert to string with a single traversal
    def node_to_string(node):
        output = graph.nodes[node].get("string_labels", "")
        # Handle formatting for linkage descriptions
        if (output[0] in "?ab" or output[0].isdigit()) and (output[-1] == "?" or output[-1].isdigit()):
            output = f"({output})"
        # Sort children by depth (shallow to deep)
        children = list(graph.successors(node))
        if not children:
            return output
        if canonicalize:
            # Combining the stable sorts: length-based and special branches use the same canonical tie-breakers
            if order_by == "length" or any(is_special_branch(child) for child in children):
                children.sort(key = lambda x: (-depths[x], get_linkage_number(x, graph), leaf_labels[x], subtree_keys[x]), reverse = True)
            else:
                # Standard linkage canonicalization relies on positive depth rather than negative
                children.sort(key = lambda x: (get_linkage_number(x, graph), depths[x], leaf_labels[x], subtree_keys[x]), reverse = True)
        else:
            # Sort children based on ordering mode
            if order_by == "length":
                # Sort by depth (shallow to deep)
                children.sort(key = lambda x: depths[x])
            else:  # order_by == "linkage"
                if any(is_special_branch(child) for child in children):
                    # Use length-based sorting if special single mono branches exist
                    children.sort(key = lambda x: (depths[x], -get_linkage_number(x, graph)))
                else:
                    # Use linkage-based sorting in all other cases
                    children.sort(key = lambda x: -get_linkage_number(x, graph))
        # Process all but the deepest child
        for child in children[:-1]:
            output = f"[{node_to_string(child)}]{output}"
        # Process the deepest child (last in sorted list)
        output = node_to_string(children[-1]) + output
        return output

    return node_to_string(root_idx)


def graph_to_string(graph: nx.DiGraph, # Glycan graph (assumes root node is the one with the highest index)
                    canonicalize: bool = True,  # Whether to output canonicalized IUPAC-condensed
                    order_by: str = "length"  # canonicalize by 'length' or 'linkage'
                    ) -> str: # IUPAC-condensed glycan string
    "Convert glycan graph back to IUPAC-condensed format, handling disconnected components"
    if isinstance(graph, str):
        return graph
    components = sorted(nx.weakly_connected_components(graph), key = min)
    if len(components) > 1:
        # glycan_to_nxGraph numbers the main part first and each floating bit after it, so the parts are emitted in that same order and each is put back on its own 0-based range
        parts = [graph.subgraph(sorted(c)) for c in components[1:] + components[:1]]
        parts = [nx.relabel_nodes(p.copy(), {pn: pn - min(p.nodes()) for pn in p.nodes()}) for p in parts]
        parts = '}'.join(['{' + graph_to_string_int(p, canonicalize = canonicalize, order_by = order_by) for p in parts])
        return parts[:parts.rfind('{')] + parts[parts.rfind('{') + 1:]
    else:
        return graph_to_string_int(graph, canonicalize = canonicalize, order_by = order_by)


def try_string_conversion(graph: nx.DiGraph # Glycan graph to validate
                          ) -> str | None: # IUPAC string if valid, None if invalid
    "Check whether glycan graph describes a valid glycan"
    try:
        temp = graph_to_string(graph)
        temp = glycan_to_nxGraph(temp)
        return graph_to_string(temp)
    except (ValueError, IndexError):
        return None


def largest_subgraph(glycan_a: str | nx.DiGraph, # First glycan
                     glycan_b: str | nx.DiGraph # Second glycan
                     ) -> str: # Largest common subgraph in IUPAC format
    "Find the largest common subgraph of two glycans"
    graph_a = ensure_graph(glycan_a)
    graph_b = ensure_graph(glycan_b)
    ismags = nx.isomorphism.ISMAGS(graph_a, graph_b,
                                   node_match = nx.algorithms.isomorphism.categorical_node_match('string_labels', 'unknown'))
    largest_common_subgraph = list(ismags.largest_common_subgraph(symmetry = False))
    if not largest_common_subgraph:
        return ''
    return graph_to_string(graph_a.subgraph(largest_common_subgraph[0].keys()))


def get_possible_topologies(glycan: str | nx.DiGraph, # Glycan with floating substituent
                            exhaustive: bool = False, # Whether to allow additions at internal positions
                            allowed_disaccharides: set[str] | None = None, # Permitted disaccharides when creating possible glycans
                            modification_map: dict[str, set[str]] = modification_map, # Maps modifications to valid attachments
                            return_graphs: bool = False # Whether to return glycan graphs (otherwise return converted strings)
                            ) -> list[str | nx.DiGraph]: # List of possible topology strings or graphs
    "Create possible glycan graphs given a floating substituent"
    ggraph = ensure_graph(glycan)
    parts = [ggraph.subgraph(c) for c in nx.weakly_connected_components(ggraph)]
    if len(parts) == 1:
        raise ValueError("This glycan already has a defined topology; please don't use this function.")
    main_part, floating_part = parts[-1], parts[0]
    dangling_linkage = max(floating_part.nodes())
    is_modification = len(floating_part.nodes()) == 1
    anchors = ggraph.nodes[dangling_linkage].get('anchors', {})
    if is_modification:
        modification = ggraph.nodes[dangling_linkage]['string_labels']
    else:
        floating_monosaccharide = dangling_linkage - 1
    topologies = []
    if anchors:
        candidates = [(n, link) for link, anchor in anchors.items() for n in resolve_anchor(main_part, anchor)]
    else:
        candidates = [(k, ggraph.nodes[dangling_linkage]['string_labels']) for i, k in enumerate(main_part.nodes()) if
                      i % 2 == 0 and (exhaustive or is_modification or main_part.out_degree[k] == 0)]
    for k, link in candidates:
        dangling_carbon = modification[0] if is_modification else link[-1]
        neighbor_carbons = [ggraph.nodes[n]['string_labels'][-1] for n in ggraph.neighbors(k) if n < k]
        if dangling_carbon in neighbor_carbons:
            continue
        new_graph = deepcopy(ggraph)
        new_graph.nodes[dangling_linkage].pop('anchors', None)
        if not is_modification:
            new_graph.nodes[dangling_linkage]['string_labels'] = link
        if is_modification:
            mono = new_graph.nodes[k]['string_labels']
            if modification_map and mono in modification_map.get(modification, set()):
                new_graph.nodes[k]['string_labels'] += modification
                new_graph.remove_node(dangling_linkage)
            else:
                continue
        else:
            new_graph.add_edge(k, dangling_linkage)
            if allowed_disaccharides is not None:
                disaccharide = (f"{new_graph.nodes[floating_monosaccharide]['string_labels']}"
                                f"({new_graph.nodes[dangling_linkage]['string_labels']})"
                                f"{new_graph.nodes[k]['string_labels']}")
                if disaccharide not in allowed_disaccharides:
                    continue
        new_graph = nx.convert_node_labels_to_integers(new_graph)
        topologies.append(new_graph)
    return topologies if return_graphs else [graph_to_string(t) for t in topologies]


def possible_topology_check(glycan: str | nx.DiGraph, # Glycan with floating substituent
                            glycans: list[str | nx.DiGraph], # List of glycans to check against
                            exhaustive: bool = False, # Whether to allow additions at internal positions
                            **kwargs # Arguments passed to compare_glycans
                            ) -> list[str | nx.DiGraph]: # List of matching glycans
    "Check whether glycan with floating substituent could match glycans from a list"
    topologies = get_possible_topologies(glycan, exhaustive = exhaustive, return_graphs = True)
    ggraphs = map(ensure_graph, glycans)
    return [g for g, ggraph in zip(glycans, ggraphs) if any(compare_glycans(t, ggraph, **kwargs) for t in topologies)]


def deduplicate_glycans(glycans: list[str] | set[str] # List/set of glycans to deduplicate
                        ) -> list[str]: # Deduplicated list of glycans
    "Remove duplicate glycans from a list/set, even if they have different strings"
    glycans = sorted(set(glycans))
    if len(glycans) <= 1:
        return glycans
    ggraphs = list(map(glycan_to_nxGraph, glycans))
    n = len(glycans)
    keep = [True] * n
    # graph_to_string is a total canonicalization, so it decides isomorphism outright once no token can match anything but itself; build_wildcard_cache is the predicate compare_glycans itself uses, so this gate cannot drift when MONO_PATTERN grows
    if build_wildcard_cache(set(unwrap(min_process_glycans(glycans)))) or any(c in g for g in glycans for c in 'O^{'):
        groups = [list(range(n))]
    else:
        buckets = {}
        for i in range(n):
            buckets.setdefault(graph_to_string(ggraphs[i]), []).append(i)
        groups = [b for b in buckets.values() if len(b) > 1]
    for group in groups:
        for gi, i in enumerate(group):
            if not keep[i]:
                continue
            correct_glycan = None
            for j in group[gi + 1:]:
                if not keep[j]:
                    continue
                if compare_glycans(ggraphs[i], ggraphs[j]):
                    if correct_glycan is None:
                        correct_glycan = graph_to_string(ggraphs[i])
                    if correct_glycan == glycans[i]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
    return [g for i, g in enumerate(glycans) if keep[i]]
