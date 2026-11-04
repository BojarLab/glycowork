from __future__ import annotations
import re
import sys
from typing import Any, Callable, Iterator
from glycowork.glycan_data.loader import unwrap, modification_map, HashableDict, pd, nx
from glycowork.motif.processing import min_process_glycans, get_possible_linkages, get_possible_monosaccharides, rescue_glycans, parse_floating_bit
import numpy as np
from collections import Counter, OrderedDict
from functools import lru_cache, wraps
from itertools import chain, permutations

PTM_REGEX = re.compile(r"(?<=[A-Za-z{])(?<!Neu)(\d+/\d+|\d+)(?=\D)(?![^()]*\))")
PTM_UNIT = re.compile(r"(?<=[A-Za-z{])(?<!Neu)(?:\d+/\d+|\d+)(?=\D)|O")  # in a residue label, one hit per 'O' of its PTM-wildcarded form: the stated position, or 'O' where none is stated
NEGATION_REGEX = re.compile(r'(?<!\()(!\w+(?:\([^)]+\))?)')
MONO_PATTERN = re.compile(r"^(Hex|HexOS|HexNAc|HexNAcOS|dHex|Sia|HexA|Pen|Monosaccharide)$")
LINKAGE_PATTERN = re.compile(r'[ab\?][12]-(\d+|\?)')
LINKAGE_LABEL = re.compile(r'^[ab?][12]-')
IS_LINKAGE = re.compile(r'^[ab?]?[0-9?/]+-[0-9?/]+$').match
TOKEN_PATTERN = re.compile(r'\d+|\[|\]')
from weakref import WeakKeyDictionary
_SLIM_CACHE = WeakKeyDictionary()  # networkx graph from outside -> its GlycanGraph
_WILDCARD_CACHE = {}
_ANY_ORDER = object()  # return_matches value of the internal callers that only read which node sets match, so tree embeddings can stand in for VF2
_ATTRS, _TOPOLOGIES, _NEIGHBORS = {}, {}, {}  # attribute dicts, (succ, pred) pairs per node order and edge list, and neighbor tuples, shared by all graphs


class GlycanGraph:
    "What the motif code needs of a glycan graph, without networkx: nodes maps a node to its attribute dict, succ and pred a node to a tuple of neighbors, all in insertion order. Attribute dicts are shared between graphs, so nothing is edited in place; with_attr and subgraph return new graphs. The other slots hold data derived once per graph"
    __slots__ = ('nodes', 'succ', 'pred', 'sl', 'has_o', 'roots', 'degs', 'ptm', 'negated', 'stubs', 'anchored', 'hits', '__weakref__')

    def __init__(self, nodes, succ, pred):
        self.nodes, self.succ, self.pred = nodes, succ, pred
        self.sl = self.has_o = self.degs = self.ptm = self.negated = self.stubs = self.anchored = self.hits = None
        self.roots = 0

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __contains__(self, node):
        return node in self.nodes

    def with_attr(self, attr, values):  # new graph with {node: value} written to attr
        nodes = dict(self.nodes)
        for n, v in values.items():
            nodes[n] = {**nodes[n], attr: v}
        return GlycanGraph(nodes, self.succ, self.pred)

    def subgraph(self, keep):  # new graph induced on the nodes in keep, in this graph's order
        return GlycanGraph({n: d for n, d in self.nodes.items() if n in keep}, {n: tuple(c for c in cs if c in keep) for n, cs in self.succ.items() if n in keep},
                           {n: tuple(c for c in cs if c in keep) for n, cs in self.pred.items() if n in keep})


def _attrs(label, termini = None):  # the one attribute dict of a label at a position
    if (hit := _ATTRS.get(key := (label, termini))) is None:
        hit = _ATTRS[key] = {'string_labels': sys.intern(label)} if termini is None else {'string_labels': sys.intern(label), 'termini': termini}
    return hit


def _graph(nodes, edges):  # GlycanGraph of {node: attributes} and (parent, child) edges, with neighbors in edge order; glycans of one topology share their succ and pred
    if (hit := _TOPOLOGIES.get(key := (tuple(nodes), tuple(chain.from_iterable(edges))))) is None:
        succ, pred = dict.fromkeys(nodes, ()), dict.fromkeys(nodes, ())
        for u, v in zip(key[1][::2], key[1][1::2]):
            succ[u] += (v,)
            pred[v] += (u,)
        hit = _TOPOLOGIES[key] = tuple({n: _NEIGHBORS.setdefault(t, t) for n, t in d.items()} for d in (succ, pred))
    return GlycanGraph(nodes, *hit)


def _is_graph(g):  # only an imported networkx can have made a networkx graph
    return isinstance(g, GlycanGraph) or ('networkx' in sys.modules and isinstance(g, nx.Graph))


def _slim(g):  # GlycanGraph of a graph from outside, kept while the networkx graph still holds the same, as it can legitimately change between calls
    if isinstance(g, GlycanGraph):
        return g
    _check_glycans(g)
    hit = _SLIM_CACHE.get(g)
    if hit is None or g._node != hit.nodes or list(g._node) != list(hit.nodes) or list(map(tuple, g._succ.values())) != list(hit.succ.values()):
        hit = _graph({n: _attrs(d['string_labels']) if len(d) == 1 and 'string_labels' in d else dict(d) for n, d in g.nodes(data = True)}, ((u, v) for u in g for v in g.succ[u]))
        if len(hit):
            _SLIM_CACHE[g] = hit
    return hit


def _to_nx(g, libr = None):  # networkx graph of a GlycanGraph, with attribute dicts of its own and, with libr, the 'labels' attribute
    h = nx.DiGraph()
    h.add_nodes_from((n, ({'labels': libr[d['string_labels']]} if libr is not None else {}) | d | ({'anchors': dict(d['anchors'])} if 'anchors' in d else {})) for n, d in g.nodes.items())
    h.add_edges_from((u, v) for u in g.succ for v in g.succ[u])
    return h


_for_vf2 = lru_cache(maxsize = 256)(_to_nx)  # VF2, the one thing still needing networkx here, only reads its graphs and runs per motif on the same glycan


def _label_match(data1, data2):  # node match on the written labels alone
    return data1.get('string_labels', 'unknown') == data2.get('string_labels', 'unknown')


def _component(g, start):  # nodes of the weakly connected part holding start
    seen, stack = {start}, [start]
    while stack:
        n = stack.pop()
        for m in g.succ[n] + g.pred[n]:
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return seen


def _components(g):  # weakly connected parts, in order of their first node
    seen, parts = set(), []
    for n in g.nodes:
        if n not in seen:
            parts.append(c := _component(g, n))
            seen |= c
    return parts


def _descendants(g, start):  # nodes below start
    seen, stack = set(), [start]
    while stack:
        for c in g.succ[stack.pop()]:
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def _sl(g):  # cached string_labels
    v = g.sl
    if v is None:
        v = g.sl = tuple(d['string_labels'] for d in g.nodes.values())
    return v


def _has_o(g):  # cached "needs PTM wildcarding" flag
    v = g.has_o
    if v is None:
        v = g.has_o = any('O' in s for s in _sl(g))
    return v


def _degs(g):  # cached sorted out-degree sequence
    v = g.degs
    if v is None:
        v = g.degs = tuple(sorted(map(len, g.succ.values())))
    return v


def _roots(g):  # cached in-degree-0 nodes if g is a forest of out-trees (as glycan_to_nxGraph always builds), else None
    v = g.roots
    if v == 0:
        # With every in-degree at most 1, an undirected cycle is a directed one, and a directed cycle is exactly what the roots cannot reach, so this is nx.is_branching without its generic forest check
        v = [n for n, p in g.pred.items() if not p] if len(g) and all(len(p) <= 1 for p in g.pred.values()) else None
        seen, stack = set(v or ()), list(v or ())
        while stack:
            for c in g.succ[stack.pop()]:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        v = g.roots = v if v is not None and len(seen) == len(g) else None
    return v


def _tree_embeddings(g1: GlycanGraph,  # Glycan graph
                     g2: GlycanGraph,  # Motif graph
                     node_match: Callable[[dict, dict], bool]  # Node matcher, called as node_match(g1 data, g2 data)
                     ) -> Iterator[set[frozenset[
    int]]] | None:  # Per g1 node, the distinct g1 node sets of induced copies of g2 rooted there; None if VF2 is needed
    "Induced subgraph isomorphism for the tree case, where it is exactly a rooted embedding: every motif child sits on a distinct child of its parent's image"
    if _roots(g1) is None or len(r2 := _roots(g2) or ()) != 1:
        return None
    embed = _embedder(g1, g2, node_match)
    return (embed(r2[0], v) for v in g1)


def _embedder(g1: GlycanGraph,  # Glycan graph
              g2: GlycanGraph,  # Motif graph
              node_match: Callable[[dict, dict], bool]  # Node matcher, called as node_match(g1 data, g2 data)
              ) -> Callable[[int, int], Any]:  # embed(u, v): the distinct g1 node sets that the g2 subtree below u can occupy with u on v
    "Memoized rooted embedding of out-tree subtrees, shared by _tree_embeddings and _forest_isomorphic"
    n1, n2, s1, s2, memo = g1.nodes, g2.nodes, g1.succ, g2.succ, {}

    def embed(u, v):
        if (u, v) not in memo:
            res = [frozenset([v])] if node_match(n1[v], n2[u]) else []
            for cu in s2[u]:
                if not res:
                    break
                res = {s | t for cv in s1[v] for t in embed(cu, cv) for s in res if cv not in s}
            memo[u, v] = res
        return memo[u, v]

    return embed


def _forest_isomorphic(g1: GlycanGraph,  # Glycan graph
                       g2: GlycanGraph,  # Glycan graph with the same out-degree multiset as g1
                       node_match: Callable[[dict, dict], bool]  # Node matcher, called as node_match(g1 data, g2 data)
                       ) -> bool | None:  # Whether g1 and g2 are isomorphic; None if VF2 is needed
    "Isomorphism of out-forests of equal size (floating parts included): every g2 tree lands root on root in its own g1 tree, and as the sizes add up, each then fills that tree"
    if (r1 := _roots(g1)) is None or (r2 := _roots(g2)) is None or len(r1) != len(r2):
        return None
    embed = _embedder(g1, g2, node_match)
    fits = [[j for j, v in enumerate(r1) if embed(u, v)] for u in r2]

    def pair(i, used):  # one g1 tree per g2 tree, by backtracking over the few trees a glycan has
        return i == len(fits) or any(j not in used and pair(i + 1, used | {j}) for j in fits[i])

    return pair(0, frozenset())


def _ptm_wildcarded(g):  # cached PTM-wildcarded version, with the written labels kept as 'ptm'
    v = g.ptm
    if v is None:
        nodes = dict(g.nodes)
        for n, d in nodes.items():
            if not LINKAGE_LABEL.match(label := d['string_labels']) and (wild := PTM_REGEX.sub('O', label)) != label:
                nodes[n] = {**d, 'string_labels': wild, 'ptm': label}
        v = g.ptm = GlycanGraph(nodes, g.succ, g.pred)
    return v


def _with_ptm(g, orig):  # version of g, built from the PTM-wildcarded string of orig, carrying orig's labels as 'ptm' wherever wildcarding changed them, as _ptm_wildcarded does
    return g.with_attr('ptm', {n: l for n, l in zip(orig, _sl(orig)) if l != g.nodes[n]['string_labels']})


def build_wildcard_cache(proc: set[str] # Node labels (monosaccharides and linkages) of the graphs to match
                         ) -> dict[str, frozenset[str]]: # Wildcard label : the labels it can stand for, for every wildcard in proc
    """Precompute all possible wildcard expansions, cached per label set"""
    if (hit := _WILDCARD_CACHE.get(key := frozenset(proc))) is None:
        hit = _WILDCARD_CACHE[key] = {k: get_possible_linkages(k) for k in proc if '?' in k or ('/' in k and '-' in k)} | {k: get_possible_monosaccharides(k) for k in proc if MONO_PATTERN.match(k) or k.startswith('!') or ('/' in k and '-' not in k)}
    return hit


def resolve_anchor(ggraph: GlycanGraph | nx.DiGraph, # Glycan graph to search
                   anchor: str # Anchor motif with ^ marking the acceptor residue
                   ) -> set[int]: # Nodes of ggraph the marked residue could correspond to
    "Find all residues of a glycan graph that match the ^-marked residue of an anchor motif"
    idx = next(i for i, x in enumerate(min_process_glycans([anchor])[0]) if x.endswith('^'))
    ggraph, g2 = _slim(ggraph), _glycan_graph(anchor.replace('^', ''))
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(_for_vf2(ggraph), _for_vf2(g2), node_match = categorical_node_match_wildcard(
        'string_labels', 'unknown', build_wildcard_cache(set(_sl(ggraph)) | set(_sl(g2))), 'termini', 'flexible'))
    return {k for m in matcher.subgraph_isomorphisms_iter() for k, v in m.items() if v == idx}


def glycan_to_edges(glycan: str  # IUPAC-condensed glycan sequence
                    ) -> tuple[dict[int, str], list[tuple[int, int]]]: # (Dictionary of node:monosaccharide/linkage, list of parent->child edges)
    "Convert glycans into a node dictionary and an edge list"
    glycan_proc = tuple(min_process_glycans([glycan])[0])
    mask_dic = dict(enumerate(glycan_proc))
    temp_glycan = glycan
    for i, gl in mask_dic.items():
        temp_glycan = temp_glycan.replace(gl, str(i), 1)
    edges, stack, current_node = [], [], None
    for token in reversed(TOKEN_PATTERN.findall(temp_glycan)):
        if token == ']':
            stack.append(current_node)
        elif token == '[':
            current_node = stack.pop()
        else:
            node_idx = int(token)
            if current_node is not None and node_idx < current_node:
                edges.append((current_node, node_idx))
            current_node = node_idx
    edges.sort()
    return mask_dic, edges


def glycan_to_graph(glycan: str  # IUPAC-condensed glycan sequence
                    ) -> tuple[dict[int, str], np.ndarray]: # (Dictionary of node:monosaccharide/linkage, Adjacency matrix)
    "Convert glycans into graphs"
    mask_dic, edges = glycan_to_edges(glycan)
    adj_matrix = np.zeros((len(mask_dic), len(mask_dic)), dtype = np.uint8)
    for parent, child in edges:
        adj_matrix[child, parent] = 1
    return mask_dic, adj_matrix


@lru_cache(maxsize = 50_000)
def _glycan_graph_int(glycan: str, # Glycan in IUPAC-condensed format
                      termini: str = 'ignore', # How to encode terminal/internal position; options: ignore, calc, provided
                      termini_list: tuple[str] | None = None # List of positions from terminal/internal/flexible
                      ) -> GlycanGraph: # Graph of the glycan; cached and shared between calls
    "Convert glycans into GlycanGraphs"
    # This allows to make glycan graphs of motifs ending in a linkage
    appended_hex = glycan.endswith(')')
    if appended_hex:
        glycan += 'Hex'
    # Map glycan string to node labels and adjacency matrix
    node_dict, edges = glycan_to_edges(glycan)
    if appended_hex:
        last_node = len(node_dict) - 1
        node_dict.pop(last_node, None)
        edges = [e for e in edges if last_node not in e]
    positions = {}
    if termini == 'calc':
        last_node = max(node_dict)
        degrees = Counter(n for e in edges for n in e)
        positions = {k: 'terminal' if (degrees[k] == 1 or k == last_node) else 'internal' for k in node_dict}
    elif termini == 'provided':
        positions = dict(zip(node_dict, termini_list))
    return _graph({i: _attrs(v, termini = positions.get(i)) for i, v in node_dict.items()}, edges)


@rescue_glycans
def _glycan_graph(glycan: str, # Glycan in IUPAC-condensed format
                  termini: str = 'ignore', # How to encode terminal/internal position; options: ignore, calc, provided
                  termini_list: tuple[str] | None = None  # List of positions from terminal/internal/flexible
                  ) -> GlycanGraph:  # Graph of the glycan; cached and shared between calls unless it has floating parts
    "Wrapper for converting glycans into GlycanGraphs; also works with floating substituents"
    if not glycan:
        return GlycanGraph({}, {}, {})
    if any(k in glycan for k in [';', 'β', 'α', 'RES', '=']):
        raise Exception
    if '{' in glycan:
        chunks = [k for k in glycan.replace('}', '{').split('{') if k]
        chunks, anchor_specs = zip(
            *[(k, {}) if i == len(chunks) - 1 else parse_floating_bit(k) for i, k in enumerate(chunks)])
        # the string writes floating bits first but the graph numbers them last, and an anchored bit spells out more linkages than it keeps, so expand over the parsed chunks and slice per chunk
        sizes = [len(min_process_glycans([k])[0]) for k in chunks] if termini_list else None
        termini_list = expand_termini_list(''.join(chunks), termini_list) if termini_list else None
        parts = [_glycan_graph_int(k, termini = termini, termini_list = None if termini_list is None else termini_list[sum(sizes[:i]):sum(sizes[:i + 1])]) for i, k in enumerate(chunks)]
        # floating bits are numbered after the main part but inserted first
        nodes, edges, len_org = {}, [], len(parts[-1])
        for i, part in enumerate(parts[:-1]):
            nodes.update((n + len_org, d) for n, d in part.nodes.items())
            edges.extend((u + len_org, v + len_org) for u in part.succ for v in part.succ[u])
            if anchor_specs[i]:
                nodes[max(part.nodes) + len_org] = {**part.nodes[max(part.nodes)], 'anchors': anchor_specs[i]}
            len_org += len(part)
        nodes.update(parts[-1].nodes)
        edges.extend((u, v) for u in parts[-1].succ for v in parts[-1].succ[u])
        return _graph(nodes, edges)
    return _glycan_graph_int(glycan, termini = termini, termini_list = expand_termini_list(glycan, termini_list) if termini_list else None)


@lru_cache(maxsize = 50_000)
def glycan_to_nxGraph_int(glycan: str, # Glycan in IUPAC-condensed format
                          libr: dict[str, int] | HashableDict[str, int] | None = None, # Dictionary of form glycoletter:index
                          termini: str = 'ignore', # How to encode terminal/internal position; options: ignore, calc, provided
                          termini_list: tuple[str] | None = None # List of positions from terminal/internal/flexible
                          ) -> nx.DiGraph: # NetworkX graph object of glycan; cached and shared between calls, so copy it before modifying
    "Convert glycans into networkx graphs"
    g = _glycan_graph_int(glycan, termini = termini, termini_list = termini_list)
    g1 = _to_nx(g, libr = libr)
    _SLIM_CACHE[g1] = g
    return g1


@rescue_glycans
def glycan_to_nxGraph(glycan: str, # Glycan in IUPAC-condensed format
                      libr: HashableDict[str, int] | None = None, # Dictionary of form glycoletter:index
                      termini: str = 'ignore', # How to encode terminal/internal position; options: ignore, calc, provided
                      termini_list: tuple[str] | None = None  # List of positions from terminal/internal/flexible
                      ) -> nx.DiGraph:  # NetworkX graph object of glycan; cached and shared between calls, so copy it before modifying (a changed label would change every later result for that glycan)
    "Wrapper for converting glycans into networkx graphs; also works with floating substituents"
    if not glycan:
        return nx.DiGraph()
    if isinstance(libr, dict) and not isinstance(libr, HashableDict):
        libr = HashableDict(libr)
    if '{' in glycan or any(k in glycan for k in [';', 'β', 'α', 'RES', '=']):
        return _to_nx(_glycan_graph.__wrapped__(glycan, termini = termini, termini_list = termini_list), libr = libr)
    return glycan_to_nxGraph_int(glycan, libr = libr, termini = termini,
                                 termini_list = expand_termini_list(glycan, termini_list) if termini_list else None)


def glycan_to_glycangraph(glycan: str, # Glycan in IUPAC-condensed format
                          termini: str = 'ignore', # How to encode terminal/internal position; options: ignore, calc, provided
                          termini_list: str | list[str] | None = None # List of positions from terminal/internal/flexible
                          ) -> GlycanGraph: # Read-only GlycanGraph: nodes maps a node to its attributes, succ and pred a node to a tuple of neighbors; cached and shared between calls, so never edit it
    "Convert glycans into the lightweight graph glycowork's motif code runs on, for code that only reads the graph: the nodes, attributes, and neighbor order of glycan_to_nxGraph at a tenth of its memory and without networkx; also works with floating substituents"
    return _glycan_graph(glycan, termini = termini, termini_list = termini_list)


def _check_glycans(*glycans: Any # Inputs that have to be single glycans
                   ) -> None:
    "Raises an actionable TypeError for anything that is no glycan string or graph, which otherwise failed deep inside with \"'dict' object has no attribute 'nodes'\""
    for glycan in glycans:
        if not isinstance(glycan, str) and not _is_graph(glycan):
            raise TypeError(f"Expected one glycan as an IUPAC-condensed string or networkx graph, got {type(glycan).__name__} {glycan!r:.80}; a composition has no structure to match, and several glycans are passed one at a time.")


def _as_graph(glycan: str | GlycanGraph | nx.DiGraph, # Glycan in IUPAC-condensed format or as a graph
              **kwargs # Keyword arguments passed to _glycan_graph
              ) -> GlycanGraph: # GlycanGraph of the glycan
    "ensure_graph for the motif code"
    if isinstance(glycan, str):
        return _glycan_graph(glycan, **kwargs)
    _check_glycans(glycan)
    return _slim(glycan)


def ensure_graph(glycan: str | nx.DiGraph, # Glycan in IUPAC-condensed format or as networkx graph
                 **kwargs # Keyword arguments passed to glycan_to_nxGraph
                 ) -> nx.DiGraph: # NetworkX graph object of glycan; for a string, the cached graph glycan_to_nxGraph returns
    "Ensures function compatibility with string glycans and graph glycans"
    if isinstance(glycan, str):
        return glycan_to_nxGraph(glycan, **kwargs)
    _check_glycans(glycan)
    return glycan


def categorical_node_match_wildcard(attr: str | tuple[str, ...], # Attribute or tuple of attributes to match
                                    default: str, # Default value if attribute is missing
                                    narrow_wildcard_list: dict[str, set[str]], # Dictionary mapping wildcards to possible matches
                                    attr2: str, # Secondary attribute to match
                                    default2: str # Default value for secondary attribute
                                    ) -> Callable[[dict, dict], bool]: # Function that matches node attributes
    "Match nodes while handling wildcards and flexible positions"
    def match(data1, data2):
        if data1 is data2:  # attribute dicts are shared per label and position, and a node always matches itself
            return True
        data1_labels2, data2_labels2 = data1.get(attr2, default2), data2.get(attr2, default2)
        if data1_labels2 != data2_labels2 and data1_labels2 != 'flexible' and data2_labels2 != 'flexible':
            return False
        # 'ptm' holds the label before PTM wildcarding, so two residues have to agree on every PTM position both of them state (Gal3S vs Gal6S, but also Glc3Ac4PGro vs Glc6AcOPGro), paired modification by modification
        if (p1 := data1.get('ptm')) and (p2 := data2.get('ptm')) and len(f1 := PTM_UNIT.findall(p1)) == len(f2 := PTM_UNIT.findall(p2)) and not all('O' in (x, y) or set(x.split('/')) & set(y.split('/')) for x, y in zip(f1, f2)):
            return False
        data1_labels, data2_labels = data1.get(attr, default), data2.get(attr, default)
        if data1_labels == data2_labels:
            return True
        if (data1_labels == "Monosaccharide" or data2_labels == "Monosaccharide") and not IS_LINKAGE(
                data1_labels) and not IS_LINKAGE(data2_labels):
            return True
        if data1_labels in ("?1-?", "?2-?") or data2_labels in ("?1-?", "?2-?"):
            if IS_LINKAGE(data1_labels) and IS_LINKAGE(data2_labels):
                return True
        if data2_labels.startswith('!'):
            if data1_labels != data2_labels[1:] and not IS_LINKAGE(data1_labels):
                return True
        if data1_labels in narrow_wildcard_list and data2_labels in narrow_wildcard_list[data1_labels]:
            return True
        if data2_labels in narrow_wildcard_list and data1_labels in narrow_wildcard_list[data2_labels]:
            return True
        return False
    return match


def ptm_wildcard_for_graph(graph: nx.DiGraph # Input graph, modified in place
                           ) -> nx.DiGraph: # The same graph with PTM wildcards, written labels kept as 'ptm' node attributes
    "Standardize PTM wildcards in graph, in place"
    for node in graph.nodes:
        if not LINKAGE_LABEL.match(label := graph.nodes[node]['string_labels']) and (wild := PTM_REGEX.sub('O', label)) != label:
            graph.nodes[node]['string_labels'], graph.nodes[node]['ptm'] = wild, label
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
                    or ((l1 == 'Monosaccharide' or l2 == 'Monosaccharide') and not IS_LINKAGE(l1) and not IS_LINKAGE(
                        l2))
                    or ((l1 in ('?1-?', '?2-?') or l2 in ('?1-?', '?2-?')) and IS_LINKAGE(l1) and IS_LINKAGE(l2))
                    or (l2.startswith('!') and l1 != l2[1:] and not IS_LINKAGE(l1))
                    or (l1.startswith('!') and l2 != l1[1:] and not IS_LINKAGE(l2))
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
                    return_matches: bool = False, # Whether to return node mapping between glycans
                    subsumes: bool = False # Whether to instead check that glycan_a covers every structure glycan_b can be, i.e., glycan_b is glycan_a or a more specific version of it (Gal(b1-4)GlcNAc for Gal(b1-3/4)GlcNAc, but not vice versa)
                    ) -> bool: # True if glycans are same (or glycan_a subsumes glycan_b), False if not
    "Check whether two glycans are identical, or whether glycan_a subsumes glycan_b"
    if glycan_a == glycan_b:
        return (True, None if isinstance(glycan_a, str) else {n: n for n in glycan_a}) if return_matches else True
    _check_glycans(glycan_a, glycan_b)
    glycan_a, glycan_b = glycan_a if isinstance(glycan_a, str) else _slim(glycan_a), glycan_b if isinstance(glycan_b, str) else _slim(glycan_b)
    if subsumes:
        expand = lambda l: (frozenset(k for k in get_possible_linkages(l) if '?' not in k and '/' not in k) if IS_LINKAGE(l) else get_possible_monosaccharides(l)) or frozenset([l])

        def covers(spec, gen):  # node_match(specific data, general data): everything spec can be, gen can be too
            if gen.get('anchors') not in (None, spec.get('anchors')):
                return False
            s, g = spec['string_labels'], gen['string_labels']
            if s == g or (g in ('?1-?', '?2-?') and IS_LINKAGE(s)) or (g == 'Monosaccharide' and not IS_LINKAGE(s)):
                return True
            return bool(IS_LINKAGE(s)) == bool(IS_LINKAGE(g)) and (expand(s) <= expand(g) or PTM_REGEX.sub('O', s) in expand(g))

        ga, gb = _as_graph(glycan_a), _as_graph(glycan_b)
        n_parts = len(_components(gb))
        main_b = gb.subgraph(_component(gb, 0))
        # place glycan_a's floating parts one at a time (as strings, so equal placements collapse into one lru-cached graph), keeping placements whose main part still embeds into glycan_b's at its root, until the layouts agree and every part of glycan_b sits on an equally or more general part of glycan_a
        frontier, same = [ga], False
        while frontier and not (same := any(_degs(g) == _degs(gb) and (iso if (iso := _forest_isomorphic(gb, g, covers)) is not None else nx.isomorphism.DiGraphMatcher(_for_vf2(gb), _for_vf2(g), covers).is_isomorphic()) for g in frontier if len(_components(g)) == n_parts)):
            placed = {}
            for t in frontier:
                for s in (map(graph_to_string, _topologies(t, exhaustive = True)) if len(_components(t)) > n_parts else []):
                    if s not in placed:
                        g = _glycan_graph(s)
                        hits = _tree_embeddings(main_b, g.subgraph(_component(g, 0)), covers)
                        placed[s] = g if hits is not None and any(e for v, e in zip(main_b, hits) if v == _roots(main_b)[0]) else None
            frontier = [g for g in placed.values() if g is not None]
        return (same, None) if return_matches else same

    def anchored(g):
        if isinstance(g, str):
            return '^' in g
        if g.anchored is None:
            g.anchored = any('anchors' in d for d in g.nodes.values())
        return g.anchored

    if anchored(glycan_a) or anchored(glycan_b):
        topos = [_topologies(_as_graph(g)) if anchored(g) else [_as_graph(g)] for g in (glycan_a, glycan_b)]
        same = len(topos[0]) == len(topos[1]) and all(
            any(compare_glycans(ta, tb) for tb in topos[1]) for ta in topos[0])
        return (same, None) if return_matches else same
    ptm = None
    if isinstance(glycan_a, str) and isinstance(glycan_b, str):
        if glycan_a.count('(') != glycan_b.count('('):
            return (False, None) if return_matches else False
        proc = set(unwrap(min_process_glycans([glycan_a, glycan_b])))
        if 'O' in glycan_a or 'O' in glycan_b:
            ptm = (_glycan_graph(glycan_a), _glycan_graph(glycan_b))
            glycan_a, glycan_b = PTM_REGEX.sub('O', glycan_a), PTM_REGEX.sub('O', glycan_b)
        g1, g2 = _glycan_graph(glycan_a), _glycan_graph(glycan_b)
        if ptm:
            g1, g2 = _with_ptm(g1, ptm[0]), _with_ptm(g2, ptm[1])
        g1_sl, g2_sl = _sl(g1), _sl(g2)
    else:
        glycan_a, glycan_b = _as_graph(glycan_a), _as_graph(glycan_b)
        if len(glycan_a) != len(glycan_b):
            return (False, None) if return_matches else False
        g1_sl, g2_sl = _sl(glycan_a), _sl(glycan_b)
        proc = set(g1_sl) | set(g2_sl)
        if ptm := any('O' in s for s in proc):
            g1, g2 = _ptm_wildcarded(glycan_a), _ptm_wildcarded(glycan_b)
            g1_sl, g2_sl = _sl(g1), _sl(g2)
        else:
            g1, g2 = glycan_a, glycan_b
    if _degs(g1) != _degs(g2):
        return (False, None) if return_matches else False
    narrow_wildcard_list = build_wildcard_cache(proc)
    if narrow_wildcard_list:
        if not _prefilter_labels(g1_sl, g2_sl, narrow_wildcard_list) or not _prefilter_labels(g2_sl, g1_sl,
                                                                                              narrow_wildcard_list):
            return (False, None) if return_matches else False
        node_match = categorical_node_match_wildcard('string_labels', 'unknown', narrow_wildcard_list, 'termini',
                                                     'flexible')
    else:
        # First check whether components of both glycan graphs are identical, then check graph isomorphism (costly)
        if sorted(g1_sl) != sorted(g2_sl):
            return (False, None) if return_matches else False
        # floating parts are written in numbering order, so the parts are compared as a multiset
        if sorted(graph_to_string(g1).split('}')) != sorted(graph_to_string(g2).split('}')):
            return (False, None) if return_matches else False
        if not return_matches and not ptm:
            return True
        # after PTM wildcarding, the wildcard matcher is the one that checks the stated PTM positions kept in 'ptm'
        node_match = categorical_node_match_wildcard('string_labels', 'unknown', {}, 'termini', 'flexible') if ptm else _label_match
    # equal out-degree multisets mean equal node counts, so any embedding of one forest into the other is already an isomorphism
    if not return_matches and (same := _forest_isomorphic(g1, g2, node_match)) is not None:
        return same
    matcher = nx.isomorphism.DiGraphMatcher(_for_vf2(g1), _for_vf2(g2), node_match)
    for mapping in matcher.isomorphisms_iter():
        return (True, mapping) if return_matches else True
    return (False, None) if return_matches else False


def expand_termini_list(motif: str | GlycanGraph | nx.DiGraph, # Glycan motif sequence or graph
                        termini_list: str | list[str] # Monosaccharide/linkage position(s) from terminal/internal/flexible
                        ) -> tuple[str]:  # Expanded termini list including linkages
    "Convert monosaccharide-only termini list into full termini list"
    mapping = {'t': 'terminal', 'i': 'internal', 'f': 'flexible'}
    termini_list = [mapping.get(t, t) for t in ([termini_list] if isinstance(termini_list, str) else termini_list)]
    if unknown := set(termini_list) - {'terminal', 'internal', 'flexible'}:
        raise ValueError(
            f"Unrecognized termini_list entries {sorted(unknown)}; use 'terminal', 'internal', or 'flexible'.")
    num_linkages = motif.count('(') if isinstance(motif, str) else len(motif) - len(termini_list)
    result = ['flexible'] * (len(termini_list) + num_linkages)
    result[::2] = termini_list
    return tuple(result)


def handle_negation(original_func: Callable # Function to wrap
                    ) -> Callable: # Wrapped function handling negation
    "Decorator for handling negation patterns in glycan matching functions"

    @wraps(original_func)
    def wrapper(glycan, motif, *args, **kwargs):
        if isinstance(motif, str) and motif.startswith('r'):
            from glycowork.motif.regex import get_match  # lazy, since regex.py is built on this module
            opts = dict(zip(('termini_list', 'count', 'return_matches'), args)) | kwargs
            if opts.get('return_matches'):
                raise ValueError(
                    f"Matches of the glyco-regular expression '{motif}' are sequences rather than node lists; use get_match for those")
            hits = get_match(motif, glycan)
            return len(hits) if opts.get('count') else bool(hits)
        if isinstance(motif, str) and '!' in motif:
            return subgraph_isomorphism_with_negation(glycan, motif, *args, **kwargs)
        elif _is_graph(motif):
            motif = _slim(motif)
            if motif.negated is None:
                motif.negated = any('!' in d.get('string_labels', '') for d in motif.nodes.values())
            return subgraph_isomorphism_with_negation(glycan, motif, *args, **kwargs) if motif.negated else original_func(glycan, motif, *args, **kwargs)
        else:
            return original_func(glycan, motif, *args, **kwargs)
    return wrapper


@handle_negation
def subgraph_isomorphism(glycan: str | nx.DiGraph, # Glycan sequence or graph
                         motif: str | nx.DiGraph, # Glycan motif sequence or graph
                         termini_list: str | list = [],  # Monosaccharide position(s) from terminal/internal/flexible
                         count: bool = False,  # Whether to return count instead of presence/absence
                         return_matches: bool = False  # Whether to return matched subgraphs as node lists
                         ) -> bool | int | tuple[int, list[list[int]]]:  # Boolean presence, count, or (count, matches)
    "Check if motif exists as subgraph in glycan"
    termini_list = [termini_list] if isinstance(termini_list, str) else termini_list
    if isinstance(glycan, str) and isinstance(motif, str):
        if motif.count('(') > glycan.count('('):
            return (0, []) if return_matches else 0 if count else False
        if not count and not return_matches and not termini_list:
            m_len, i = len(motif), glycan.find(motif)
            while i != -1:
                if (i == 0 or glycan[i - 1] in '([)]') and (i + m_len == len(glycan) or glycan[i + m_len] in ')]'):
                    return True
                i = glycan.find(motif, i + 1)
        if ptm := ('O' in glycan or 'O' in motif):
            orig = (_glycan_graph(glycan), _glycan_graph(motif))
            glycan, motif = PTM_REGEX.sub('O', glycan), PTM_REGEX.sub('O', motif)
        motif_comp = min_process_glycans([motif, glycan])
        g1 = _glycan_graph(glycan, termini = 'calc' if termini_list else 'ignore')
        g2 = _glycan_graph(motif, termini = 'provided' if termini_list else 'ignore', termini_list = termini_list)
        if ptm:
            g1, g2 = _with_ptm(g1, orig[0]), _with_ptm(g2, orig[1])
    else:
        _check_glycans(glycan, motif)
        glycan = _glycan_graph(glycan, termini = 'calc' if termini_list else 'ignore') if isinstance(glycan, str) else _slim(glycan)
        motif = _glycan_graph(motif, termini = 'provided' if termini_list else 'ignore', termini_list = termini_list) if isinstance(motif, str) else _slim(motif)
        if len(glycan) < len(motif):
            return (0, []) if return_matches else 0 if count else False
        if termini_list and all('termini' not in d for d in motif.nodes.values()):
            motif = motif.with_attr('termini', dict(zip(motif, expand_termini_list(motif, termini_list) if len(termini_list) < len(motif) else termini_list)))
        if termini_list and all('termini' not in d for d in glycan.nodes.values()):  # a glycan graph built without termini would match any position, so they are calculated as _glycan_graph(termini = 'calc') does, per part
            roots = {max(c) for c in _components(glycan)}
            glycan = glycan.with_attr('termini', {k: 'terminal' if len(glycan.succ[k]) + len(glycan.pred[k]) == 1 or k in roots else 'internal' for k in glycan})
        if ptm := (_has_o(motif) or _has_o(glycan)):
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
        node_match = categorical_node_match_wildcard('string_labels', 'unknown', narrow_wildcard_list, 'termini',
                                                     'flexible')
    else:
        g1_node_attr = set(_sl(g1))
        if not set(motif_comp[0]).issubset(g1_node_attr):
            return (0, []) if return_matches else 0 if count else False
        # after PTM wildcarding, the wildcard matcher is the one that checks the stated PTM positions kept in 'ptm'
        node_match = categorical_node_match_wildcard('string_labels', 'unknown', {}, 'termini', 'flexible') if ptm else _label_match
    if (not return_matches or return_matches is _ANY_ORDER) and (hits := _tree_embeddings(g1, g2, node_match)) is not None:  # only the order of returned matches still needs VF2
        if return_matches:
            return len(hits := set().union(*hits)), [sorted(h) for h in hits]
        return len(set().union(*hits)) if count else any(hits)
    graph_pair = nx.algorithms.isomorphism.DiGraphMatcher(_for_vf2(g1), _for_vf2(g2), node_match = node_match)
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
        negated_part_clean = _glycan_graph(negated_part.replace('!', ''))
        if termini_list and not to_replace:
            # The spec describes the full motif, so drop the entry of the residue the stub no longer has
            monos = [t for t in min_process_glycans([motif])[0] if not IS_LINKAGE(t)]
            termini_list = [t for i, t in enumerate(termini_list) if not monos[i].startswith('!')]
    elif (hit := ((motif := _slim(motif)).stubs or {}).get(key := tuple(termini_list))) is not None:  # the stub only depends on the motif and its spec, so a motif reused over many glycans is dissected once
        motif_stub, negated_part_clean, termini_list, positive = hit
    else:
        negated_nodes, labels = {n for n, data in motif.nodes.items() if '!' in data.get('string_labels', '')}, {}
        for n in sorted(negated_nodes):  # only a leaf can be dropped; removing an internal or root residue would disconnect the stub, so wildcard it instead, exactly as the string path does
            if motif.succ[n]:
                labels[n] = 'Monosaccharide'
                if n + 1 in motif:
                    labels[n + 1] = '?1-?'
                negated_nodes.discard(n)
        negated_nodes.update({node + 1 for node in negated_nodes if node + 1 in motif})
        motif_stub = motif.with_attr('string_labels', labels).subgraph(set(motif) - negated_nodes)
        positive = motif.with_attr('string_labels', {n: d['string_labels'].replace('!', '') for n, d in motif.nodes.items()})
        negated_part_clean = positive.subgraph(negated_nodes)
        if termini_list:
            # The spec describes the full motif, so expand it there and keep only what the stub retained; the stub's node ids are non-contiguous, so a positional zip lands on the wrong nodes or fails to fit at all
            expanded = expand_termini_list(motif, termini_list) if len(termini_list) < len(motif) else termini_list
            termini_list = [expanded[n] for n in motif_stub]
        motif.stubs = (motif.stubs or {}) | {key: (motif_stub, negated_part_clean, termini_list, positive)}
    res = subgraph_isomorphism.__wrapped__(glycan, motif_stub, termini_list = termini_list, count = count,
                                           return_matches = return_matches or _ANY_ORDER)
    if not res[0]:
        return (0, []) if return_matches else 0 if count else False
    ggraph = _as_graph(glycan)
    if isinstance(motif, str):
        positive = _glycan_graph(motif.replace('!', ''))
    # A stub hit only dies to a hit of the fully positive motif that contains it, so the negated part has to sit exactly where the motif puts it instead of anywhere in the neighborhood
    extra = len(positive) - len(res[1][0])
    _, positive_matches = subgraph_isomorphism.__wrapped__(ggraph, positive, count = True, return_matches = _ANY_ORDER)
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
        # The walk D^-1 (A + I) is not symmetric, so eigsh (which assumes it is) returned different, meaningless values on every call; on the undirected graph it is similar to the symmetric D^-1/2 (A + I) D^-1/2, whose spectrum eigvalsh gives exactly, and its stationary distribution is proportional to deg + 1
        d = deg + 1
        eigval = np.linalg.eigvalsh((A + A.T + np.eye(N)) / np.sqrt(np.outer(d, d)))
        distr = d / d.sum()
        features.update({'egap': 1 - np.sort(eigval[np.argsort(np.abs(eigval))[-2:]])[0], 'entropyStation': np.sum(distr * np.log(distr))})
    return pd.DataFrame(features, index = [glycan])


def get_linkage_number(node: int, # Linkage node
                       graph: nx.DiGraph # Glycan graph
                       ) -> float: # Acceptor position for ordering branches: the number, +100 for a slash wildcard (3/6), inf for '?' or a non-linkage node
    "Sort key of a linkage node by the acceptor position it links to"
    link_label = graph.nodes[node].get("string_labels", "")
    match = LINKAGE_PATTERN.search(link_label)
    if match:  # Extract the second number in the linkage (e.g., from "a1-3" get "3")
        linkage_num = match.group(1)
        if linkage_num == "?":
            return float('inf')  # Wildcard comes last
        return int(linkage_num) + 100 if '/' in link_label else int(linkage_num)  # Slash-wildcards rank after specified but before ?
    return float('inf')


def glycan_graph_memoize(maxsize: int = 128 # Maximum number of cached results
                         ) -> Callable: # Decorator for a function taking a glycan graph first
    "Memoizes a function of a glycan graph on its labels, out-degrees and edges, so equal graphs built separately share one result; graphs below 4 nodes are not cached"
    cache = OrderedDict()
    def decorator(func):
        @wraps(func)
        def wrapper(graph, *args, **kwargs):
            graph = _slim(graph)
            if len(graph) < 4:
                return func(graph, *args, **kwargs)
            node_ids = sorted(graph.nodes)
            # the sorted successors of every node in order are the out-degrees and the sorted edges
            key = (tuple(node_ids), tuple(graph.nodes[n]["string_labels"] for n in node_ids),
                   tuple(s if len(s := graph.succ[n]) < 2 else tuple(sorted(s)) for n in node_ids),
                   args, tuple(sorted(kwargs.items())))
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
def graph_to_string_int(graph: GlycanGraph | nx.DiGraph, # Glycan graph
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
        nodes = sorted(graph.nodes)
        return f"{graph.nodes[nodes[0]]['string_labels']}({graph.nodes[nodes[1]]['string_labels']}){graph.nodes[nodes[2]]['string_labels']}"
    # Get the root node (highest index)
    root_idx = max(graph.nodes)
    succ, labels = graph.succ, {n: d.get("string_labels", "") for n, d in graph.nodes.items()}
    # Build depths with a single traversal
    depths, leaf_labels, subtree_keys, tree_keys = {}, {}, {}, {}

    def compute_metrics(node):
        if node in depths:
            return
        successors, label = succ[node], labels[node]
        for child in successors:
            compute_metrics(child)
        # subtree_keys concatenate without delimiters, so two different subtrees can share one; the nested tree_keys spell out the topology and break such ties, which would otherwise leave the order to the input string
        if len(successors) == 1:  # linkages and chain residues, the vast majority of nodes
            child = successors[0]
            depths[node], leaf_labels[node], subtree_keys[node], tree_keys[node] = depths[child] + 1, leaf_labels[child], label + subtree_keys[child], (label, (tree_keys[child],))
        elif successors:
            depths[node] = 1 + max(map(depths.__getitem__, successors))
            leaf_labels[node] = min(map(leaf_labels.__getitem__, successors))
            subtree_keys[node] = label + ''.join(sorted(map(subtree_keys.__getitem__, successors)))
            tree_keys[node] = (label, tuple(sorted(map(tree_keys.__getitem__, successors))))
        else:
            depths[node], leaf_labels[node], subtree_keys[node], tree_keys[node] = 0, label, label, (label, ())

    compute_metrics(root_idx)

    def is_special_branch(node):
        """Check if a 1-mono branch starting at node contains Gal/Man."""
        descendants = succ[node]
        if len(descendants) != 1 or succ[descendants[0]]:
            return False
        tip_string = labels[descendants[0]]
        if tip_string in {"Gal", "Man"}:
            return False
        if "GlcNAc" in tip_string and labels[node] == 'b1-3':
            return False
        return True

    # Convert to string with a single traversal
    def node_to_string(node):
        output = labels[node]
        # Handle formatting for linkage descriptions
        if (output[0] in "?ab" or output[0].isdigit()) and (output[-1] == "?" or output[-1].isdigit()):
            output = f"({output})"
        children = succ[node]
        if not children:
            return output
        # Sort children by depth (shallow to deep); a single child needs no sorting
        if len(children) > 1:
            if canonicalize:
                # Combining the stable sorts: length-based and special branches use the same canonical tie-breakers
                if order_by == "length" or any(is_special_branch(child) for child in children):
                    children = sorted(children, key = lambda x: (-depths[x], get_linkage_number(x, graph), leaf_labels[x], subtree_keys[x], tree_keys[x]), reverse = True)
                else:
                    # Standard linkage canonicalization relies on positive depth rather than negative
                    children = sorted(children, key = lambda x: (get_linkage_number(x, graph), depths[x], leaf_labels[x], subtree_keys[x], tree_keys[x]), reverse = True)
            elif order_by == "length":
                # Sort by depth (shallow to deep)
                children = sorted(children, key = lambda x: depths[x])
            elif any(is_special_branch(child) for child in children):
                # Use length-based sorting if special single mono branches exist
                children = sorted(children, key = lambda x: (depths[x], -get_linkage_number(x, graph)))
            else:
                # Use linkage-based sorting in all other cases
                children = sorted(children, key = lambda x: -get_linkage_number(x, graph))
        # Process all but the deepest child
        for child in children[:-1]:
            output = f"[{node_to_string(child)}]{output}"
        # Process the deepest child (last in sorted list)
        output = node_to_string(children[-1]) + output
        return output

    return node_to_string(root_idx)


def graph_to_string(graph: GlycanGraph | nx.DiGraph, # Glycan graph (assumes root node is the one with the highest index)
                    canonicalize: bool = True,  # Whether to output canonicalized IUPAC-condensed
                    order_by: str = "length"  # canonicalize by 'length' or 'linkage'
                    ) -> str: # IUPAC-condensed glycan string
    "Convert glycan graph back to IUPAC-condensed format, handling disconnected components"
    if isinstance(graph, str):
        return graph
    graph = _slim(graph)
    if len(_roots(graph) or ()) == 1:  # the usual case, no floating parts
        return graph_to_string_int(graph, canonicalize = canonicalize, order_by = order_by)
    components = sorted(_components(graph), key = min)
    if len(components) > 1:
        # glycan_to_nxGraph numbers the main part first and each floating bit after it, so the parts are emitted in that same order and each is put back on its own 0-based range
        parts = []
        for c in components[1:] + components[:1]:
            off = min(c)
            parts.append(_graph({n - off: graph.nodes[n] for n in sorted(c)}, ((u - off, v - off) for u in sorted(c) for v in graph.succ[u])))
        out = []
        for p in parts:
            anchors = p.nodes[max(p.nodes)].get('anchors')
            if not anchors:
                out.append('{' + graph_to_string_int(p, canonicalize = canonicalize, order_by = order_by))
                continue
            # An anchored bit is written back by hanging its fragment off each recorded acceptor in turn, so the alternatives survive the round trip instead of collapsing into the merged linkage
            link, alts = max(p.nodes), []
            frag = sorted(set(p.nodes) - {link})
            nf, renumber = len(frag), {n: i for i, n in enumerate(frag)}
            for linkage, anchor in anchors.items():
                ga = _glycan_graph_int(anchor.replace('^', ''))
                nodes = {n + nf + 1: d for n, d in ga.nodes.items()} | {i: p.nodes[n] for i, n in enumerate(frag)} | {nf: _attrs(linkage)}
                nodes[nf - 1] = {**nodes[nf - 1], 'string_labels': nodes[nf - 1]['string_labels'] + '^'}
                edges = [(u + nf + 1, v + nf + 1) for u in ga.succ for v in ga.succ[u]] + [(renumber[u], renumber[v]) for u in frag for v in p.succ[u] if v != link]
                edges += [(next(i for i, x in enumerate(min_process_glycans([anchor])[0]) if x.endswith('^')) + nf + 1, nf), (nf, nf - 1)]
                alts.append(graph_to_string_int(_graph(nodes, edges), canonicalize = False, order_by = order_by))
            out.append('{' + '|'.join(alts))
        parts = '}'.join(out)
        return parts[:parts.rfind('{')] + parts[parts.rfind('{') + 1:]
    else:
        return graph_to_string_int(graph, canonicalize = canonicalize, order_by = order_by)


def try_string_conversion(graph: GlycanGraph | nx.DiGraph # Glycan graph to validate
                          ) -> str | None: # IUPAC string if valid, None if invalid
    "Check whether glycan graph describes a valid glycan"
    try:
        temp = graph_to_string(graph)
        temp = _glycan_graph(temp)
        return graph_to_string(temp)
    except (ValueError, IndexError):
        return None


def largest_subgraph(glycan_a: str | nx.DiGraph, # First glycan
                     glycan_b: str | nx.DiGraph # Second glycan
                     ) -> str: # Largest common subgraph in IUPAC format
    "Find the largest connected structure two glycans share, with wildcards and PTMs matching as in compare_glycans"
    graph_a, graph_b = _as_graph(glycan_a), _as_graph(glycan_b)
    wa, wb = (_ptm_wildcarded(graph_a), _ptm_wildcarded(graph_b)) if _has_o(graph_a) or _has_o(graph_b) else (graph_a, graph_b)
    node_match, memo = categorical_node_match_wildcard('string_labels', 'unknown', build_wildcard_cache(set(_sl(wa)) | set(_sl(wb))), 'termini', 'flexible'), {}

    def common(u, v):  # graph_a nodes of the largest shared subtree with u on v, its top node; children pair up one to one, and a linkage only counts with its donor, so nothing ends in a dangling linkage
        if (u, v) not in memo:
            memo[u, v] = None
            if node_match(wa.nodes[u], wb.nodes[v]):
                ca, cb = wa.succ[u], wb.succ[v]
                subs = [[common(x, y) for y in cb] for x in ca]
                w = subs if len(ca) <= len(cb) else [list(c) for c in zip(*subs)]
                picked = max(([s for i, j in enumerate(p) if (s := w[i][j])] for p in permutations(range(len(w[0])), len(w))), key = lambda ss: sum(map(len, ss))) if ca and cb else []
                if picked or not IS_LINKAGE(wa.nodes[u]['string_labels']):
                    memo[u, v] = frozenset([u]).union(*picked)
        return memo[u, v]

    best = max((s for u in wa for v in wb if (s := common(u, v))), key = len, default = None)
    return graph_to_string(graph_a.subgraph(best)) if best else ''


def _topologies(ggraph: GlycanGraph, # Glycan with floating substituent
                exhaustive: bool = False, # Whether to allow additions at internal positions
                allowed_disaccharides: set[str] | None = None, # Permitted disaccharides when creating possible glycans
                modification_map: dict[str, set[str]] = modification_map # Maps modifications to valid attachments
                ) -> list[GlycanGraph]: # List of possible topology graphs
    "get_possible_topologies for the motif code"
    parts = [ggraph.subgraph(c) for c in _components(ggraph)]
    if len(parts) == 1:
        raise ValueError("This glycan already has a defined topology; please don't use this function.")
    # the main part holds node 0; an anchored bit goes first, as its alternatives are what compare_glycans has to resolve
    main_part, floating = next(p for p in parts if 0 in p), [p for p in parts if 0 not in p]
    floating_part = next((p for p in floating if 'anchors' in ggraph.nodes[max(p)]), floating[0])
    dangling_linkage = max(floating_part)
    is_modification = len(floating_part) == 1
    anchors = ggraph.nodes[dangling_linkage].get('anchors', {})
    if is_modification:
        modification = ggraph.nodes[dangling_linkage]['string_labels']
    else:
        floating_monosaccharide = dangling_linkage - 1
    topologies = []
    if anchors:
        candidates = [(n, link) for link, anchor in anchors.items() for n in resolve_anchor(main_part, anchor)]
    else:
        candidates = [(k, ggraph.nodes[dangling_linkage]['string_labels']) for i, k in enumerate(main_part) if
                      i % 2 == 0 and (exhaustive or is_modification or not main_part.succ[k])]
    for k, link in candidates:
        # the bit cannot go on k only if every position it may take is already stated as taken there; '?' and '3/6' leave a position open
        options = set((modification[0] if is_modification else link.split('-')[-1]).split('/'))
        if all(o.isdigit() for o in options) and options <= {ggraph.nodes[n]['string_labels'].split('-')[-1] for n in ggraph.succ[k] if n < k}:
            continue
        nodes = dict(ggraph.nodes)
        nodes[dangling_linkage] = {a: v for a, v in nodes[dangling_linkage].items() if a != 'anchors'}
        if is_modification:
            mono = nodes[k]['string_labels']
            if not modification_map or mono not in modification_map.get(modification, set()):
                continue
            nodes[k] = {**nodes[k], 'string_labels': mono + modification}
            del nodes[dangling_linkage]
        else:
            nodes[dangling_linkage] = {**nodes[dangling_linkage], 'string_labels': link}
            if allowed_disaccharides is not None:
                disaccharide = (f"{nodes[floating_monosaccharide]['string_labels']}"
                                f"({nodes[dangling_linkage]['string_labels']})"
                                f"{nodes[k]['string_labels']}")
                if disaccharide not in allowed_disaccharides:
                    continue
        # renumbered into glycan_to_nxGraph's layout (main part first, the placed bit ahead of it so the root stays last; remaining floating parts numbered after, inserted before), so the output can be placed again
        main, rest = [n for n in nodes if n in main_part or n in floating_part], [n for n in nodes if n not in main_part and n not in floating_part]
        mapping = {n: i for i, n in enumerate(main + rest)}
        topologies.append(_graph({mapping[n]: nodes[n] for n in rest + main}, [(mapping[u], mapping[v]) for u in nodes for v in ggraph.succ[u] + ((dangling_linkage,) if u == k and not is_modification else ())]))
    return topologies


def get_possible_topologies(glycan: str | nx.DiGraph, # Glycan with floating substituent
                            exhaustive: bool = False, # Whether to allow additions at internal positions
                            allowed_disaccharides: set[str] | None = None, # Permitted disaccharides when creating possible glycans
                            modification_map: dict[str, set[str]] = modification_map, # Maps modifications to valid attachments
                            return_graphs: bool = False # Whether to return glycan graphs (otherwise return converted strings)
                            ) -> list[str | nx.DiGraph]: # List of possible topology strings or graphs
    "Create possible glycan graphs given a floating substituent"
    topologies = _topologies(_as_graph(glycan), exhaustive = exhaustive, allowed_disaccharides = allowed_disaccharides, modification_map = modification_map)
    return [_to_nx(t) for t in topologies] if return_graphs else [graph_to_string(t) for t in topologies]


def possible_topology_check(glycan: str | nx.DiGraph, # Glycan with floating substituent
                            glycans: list[str | nx.DiGraph], # List of glycans to check against
                            exhaustive: bool = False, # Whether to allow additions at internal positions
                            **kwargs # Arguments passed to compare_glycans
                            ) -> list[str | nx.DiGraph]: # List of matching glycans
    "Check whether glycan with floating substituent could match glycans from a list"
    topologies = _topologies(_as_graph(glycan), exhaustive = exhaustive)
    ggraphs = map(_as_graph, glycans)
    return [g for g, ggraph in zip(glycans, ggraphs) if any(compare_glycans(t, ggraph, **kwargs) for t in topologies)]


def deduplicate_glycans(glycans: list[str] | set[str] # List/set of glycans to deduplicate
                        ) -> list[str]: # Deduplicated list of glycans
    "Remove duplicate glycans from a list/set, even if they have different strings"
    glycans = sorted(set(glycans))
    if len(glycans) <= 1:
        return glycans
    ggraphs = list(map(_glycan_graph, glycans))
    n = len(glycans)
    keep = [True] * n
    # graph_to_string is a total canonicalization, so it decides isomorphism outright once no token can match anything but itself; build_wildcard_cache is the predicate compare_glycans itself uses, so this gate cannot drift when MONO_PATTERN grows
    if any('^' in g for g in glycans):
        groups = [list(range(n))]
    else:
        # Otherwise compare_glycans rejects any pair whose out-degree multisets (and with them node counts) differ before it looks at labels, so only glycans sharing one can collapse into each other
        exact = not build_wildcard_cache(set(unwrap(min_process_glycans(glycans)))) and not any(c in g for g in glycans for c in 'O{')
        buckets = {}
        for i in range(n):
            buckets.setdefault(graph_to_string(ggraphs[i]) if exact else _degs(ggraphs[i]), []).append(i)
        groups = [b for b in buckets.values() if len(b) > 1]
    for group in groups:
        for gi, i in enumerate(group):
            if not keep[i]:
                continue
            for j in group[gi + 1:]:
                if not keep[j] or not compare_glycans(ggraphs[i], ggraphs[j]):
                    continue
                # compare_glycans lets a wildcard match anything it could be, so only a glycan the other one covers is a duplicate, and the more specific one stays; two glycans that are each more specific somewhere both stay
                i_covers, j_covers = compare_glycans(ggraphs[i], ggraphs[j], subsumes = True), compare_glycans(ggraphs[j], ggraphs[i], subsumes = True)
                if not i_covers and not j_covers:
                    continue
                # the same glycan, spelled twice, keeps its canonical spelling
                drop = (j if graph_to_string(ggraphs[i]) == glycans[i] else i) if i_covers and j_covers else i if i_covers else j
                keep[drop] = False
                if drop == i:
                    break
    return [g for i, g in enumerate(glycans) if keep[i]]
