import ast
import networkx as nx
from pathlib import Path
import pandas as pd
import numpy as np
import re
from collections import Counter, deque, defaultdict
from functools import partial
from weakref import WeakKeyDictionary

from glycowork.glycan_data import loader
from glycowork.glycan_data.loader import lib, linkages, motif_list, unwrap, Hex, dHex, HexNAc, HexA, Pen, Sia, resolve_motif_name
from glycowork.motif.graph import subgraph_isomorphism, generate_graph_features, glycan_to_nxGraph, graph_to_string, ensure_graph, get_possible_topologies, compare_glycans, graph_to_string_int, expand_termini_list, build_wildcard_cache, _sl, _has_o, categorical_node_match_wildcard, PTM_REGEX, LINKAGE_LABEL
from glycowork.motif.processing import IUPAC_to_SMILES, get_lib, rescue_glycans, is_composition, canonicalize_composition, split_glycoform_id, GLYCOFORM_ID
from glycowork.motif.regex import get_match, compile_pattern, compile_component


LINKAGE_NODE_PATTERN = re.compile(r'^[ab?]?[0-9?/]+-[0-9?/]+$')
_WILDCARD_RE = re.compile(r'(?<![A-Za-z0-9])(?:Monosaccharide|HexNAcOS|HexNAcOP|HexNAc|HexAOS|HexOS|HexOP|HexNS|HexN|HexA|dHex|Hex|Sia|Pen)(?![A-Za-z0-9])')
_STRUCTURAL_ALDITOL = re.compile(r'(?:Thre|Ery|Rib|Gro|Ara)[A-Za-z0-9]*-ol$')
_REGEX_LOOKAROUND = re.compile(r'\(\?<?[=!][^()]*\)')
# Residue profile of each topology graph, keyed on the lru-cached feasible graphs so that a dataset with many floating glycans profiles each database glycan once
_RESIDUE_CACHE = WeakKeyDictionary()


def _motif_sequence(
        label: str # Motif label as produced by annotate_dataset/quantify_motifs
) -> tuple[str, list[str]]: # (IUPAC-condensed sequence, monosaccharide termini spec)
    "Resolves a motif label to the sequence and termini spec it was counted with"
    if (hit := resolve_motif_name(label)) is not None:
        return hit
    s = label[9:] if label.startswith('Terminal_') else label
    # Terminal_ columns were counted with a pinned non-reducing end, exhaustive ones position-agnostically
    return s, (['terminal'] if label.startswith('Terminal_') else ['flexible']) + ['flexible'] * (s.count('(') - (1 if s.endswith(')') else 0))


def _motif_ambiguity(
        label: str # Motif label as produced by annotate_dataset/quantify_motifs
) -> tuple[int, bool, int, int]: # Sort key; lower is more specific
    "Ranks a motif label by information content, since wildcards, ambiguous linkages, and unpinned termini convey less than the concrete forms they abstract and can be longer than them"
    s, sp = _motif_sequence(label)
    # A glyco-regex spells its own syntax with '?' and '/', neither of which is the linkage ambiguity this rank is about
    s = _REGEX_LOOKAROUND.sub('', s[1:]) if s.startswith('r') else s
    return (len(_WILDCARD_RE.findall(s)) + s.count('?') + s.count('/'), resolve_motif_name(label) is None,
            -sum(t != 'flexible' for t in sp), -len(s))


def _count_chains(
        ggraphs: list[nx.DiGraph],  # Glycan graphs, with termini wherever the motifs carry them
        motifs: list[nx.DiGraph]  # Motif graphs that are rootward chains, i.e., node i + 1 is the parent of node i
) -> list[list[int]]:  # Occurrences of each motif in each glycan
    "Counts chain motifs as subgraph isomorphism would: every node has at most one parent, so an occurrence is fixed by where the motif's first node lands and a walk up from each node finds them all, done once per distinct chain the glycans share"
    wild = lambda c: tuple((l if LINKAGE_LABEL.match(l) else PTM_REGEX.sub('O', l), t) for l, t in c)
    max_len = max(map(len, motifs), default = 0)
    # Chains keyed by their first node, as written for glycans without PTM notation and PTM-wildcarded otherwise, plus PTM-wildcarded for all glycans, since both sides are wildcarded as soon as the motif carries one; wildcarded chains keep the chain as written next to them, whose stated PTM positions still have to agree
    raw, ptm_o, ptm_all = (defaultdict(lambda: defaultdict(list)) for _ in range(3))
    for gi, g in enumerate(ggraphs):
        for v in g:
            p = [v]
            while len(p) < max_len and (u := next(iter(g.predecessors(p[-1])), None)) is not None:
                p.append(u)
            c = tuple((g.nodes[u]['string_labels'], g.nodes[u].get('termini', 'flexible')) for u in p)
            cp = wild(c)
            (ptm_o[cp[0]][cp, c] if _has_o(g) else raw[c[0]][c, c]).append(gi)
            ptm_all[cp[0]][cp, c].append(gi)
    mcs = [tuple((d['string_labels'], d.get('termini', 'flexible')) for _, d in sorted(m.nodes(data = True))) for m
           in motifs]
    nm = categorical_node_match_wildcard('string_labels', 'unknown', build_wildcard_cache(
        {l for idx in (raw, ptm_all) for cs in idx.values() for c, _ in cs for l, _ in c} | {l for c in mcs for l, _ in
                                                                                             c + wild(c)}), 'termini',
                                         'flexible')
    node = lambda l, o, t: {'string_labels': l, 'termini': t, 'ptm': o} if o != l else {'string_labels': l, 'termini': t}
    out = []
    for m, mc in zip(motifs, mcs):
        out.append(col := [0] * len(ggraphs))
        for idx, mm in ([(ptm_all, wild(mc))] if _has_o(m) else [(raw, mc), (ptm_o, wild(mc))]):
            mm = [node(l, o, t) for (l, t), (o, _) in zip(mm, mc)]
            for (l0, t0), cs in idx.items():
                if not nm({'string_labels': l0, 'termini': t0}, mm[0]):
                    continue
                for (c, co), gis in cs.items():
                    if len(c) >= len(mm) and all(nm(node(l, o, t), mm[i]) for i, ((l, t), (o, _)) in
                                                 enumerate(zip(c[:len(mm)], co))):
                        for gi in gis:
                            col[gi] += 1
    return out


def annotate_glycan(
        glycan: str | nx.DiGraph, # IUPAC-condensed glycan sequence or NetworkX graph
        motifs: pd.DataFrame | str | list[str] | None = None, # Motif dataframe (motif_name + motif columns), or motif sequence(s); defaults to motif_list
        termini_list: list = [], # Monosaccharide positions: 'terminal', 'internal', or 'flexible'
        gmotifs: list[nx.DiGraph] | None = None, # Precalculated motif graphs for speed
        condense: bool = False # Remove columns with only zeros
) -> pd.DataFrame: # DataFrame with motif counts for the glycan
    "Counts occurrences of known motifs in a glycan structure using subgraph isomorphism"
    if motifs is None:
        motifs = motif_list
    motifs = [motifs] if isinstance(motifs, str) else motifs
    # Check whether termini are specified
    if not termini_list and isinstance(motifs, pd.DataFrame):
        termini_list = list(map(ast.literal_eval, motifs.termini_spec)) if 'termini_spec' in motifs.columns else [
            ['flexible'] * (1 + m.count('(') - m.endswith(')')) for m in motifs.motif]
    if gmotifs is None:
        termini = 'provided' if termini_list else 'ignore'
        partial_glycan_to_nxGraph = partial(glycan_to_nxGraph, termini = termini)
        # r-prefixed rows are glyco-regular expressions, pre-compiled into chunk lists that the counting step tells apart from motif graphs
        gmotifs = [compile_pattern(mo[1:]) if mo.startswith('r') else partial_glycan_to_nxGraph(mo, termini_list = termini_list[i] if termini_list else None)
                   for i, mo in enumerate(motifs.motif if isinstance(motifs, pd.DataFrame) else motifs)]
    # Count the number of times each motif occurs in a glycan; as in annotate_dataset, a reducing-end alditol is its sugar, its position being what termini specs express
    ggraph = ensure_graph(glycan[:-3] if isinstance(glycan, str) and glycan.endswith('-ol') and not _STRUCTURAL_ALDITOL.search(glycan) else glycan, termini = 'calc' if termini_list else 'ignore')
    res = [len(get_match(g, ggraph)) if isinstance(g, list) else
           subgraph_isomorphism(ggraph, g, termini_list = termini_list[i] if termini_list else termini_list,
                                count = True)
           for i, g in enumerate(gmotifs)]
    out = pd.DataFrame(np.array([res], dtype = int),
                       columns = motifs.motif_name if isinstance(motifs, pd.DataFrame) else motifs,
                       index = [glycan] if isinstance(glycan, str) else [graph_to_string(glycan)])
    return out if not condense else out.loc[:, (out != 0).any(axis = 0)]


def annotate_glycan_topology_uncertainty(
        glycan: str | nx.DiGraph, # IUPAC-condensed glycan sequence or NetworkX graph
        feasibles: set[str] | None = None, # Set of potential full topology glycans; defaults to mammalian glycans
        motifs: pd.DataFrame | str | list[str] | None = None, # Motif dataframe (motif_name + motif columns), or motif sequence(s); defaults to motif_list
        termini_list: list = [], # Monosaccharide positions: 'terminal', 'internal', or 'flexible'
        gmotifs: list[nx.DiGraph] | None = None # Precalculated motif graphs for speed
) -> pd.DataFrame: # DataFrame with motif counts considering topology uncertainty
    "Annotates glycan motifs while handling uncertain topologies by comparing against physiological structures"
    if motifs is None:
        motifs = motif_list
    motifs = [motifs] if isinstance(motifs, str) else motifs
    if feasibles is None:
        feasibles = set(loader.df_species[loader.df_species.Class == "Mammalia"].glycan.values.tolist())
    # Check whether termini are specified
    if not termini_list and isinstance(motifs, pd.DataFrame):
        termini_list = list(map(ast.literal_eval, motifs.termini_spec)) if 'termini_spec' in motifs.columns else [
            ['flexible'] * (1 + m.count('(') - m.endswith(')')) for m in motifs.motif]
    if gmotifs is None:
        termini = 'provided' if termini_list else 'ignore'
        partial_glycan_to_nxGraph = partial(glycan_to_nxGraph, termini = termini)
        # r-prefixed rows are glyco-regular expressions, pre-compiled into chunk lists that the counting step tells apart from motif graphs
        gmotifs = [compile_pattern(mo[1:]) if mo.startswith('r') else partial_glycan_to_nxGraph(mo, termini_list = termini_list[i] if termini_list else None)
                   for i, mo in enumerate(motifs.motif if isinstance(motifs, pd.DataFrame) else motifs)]
    # Count the number of times each motif occurs in a glycan
    termini = 'calc' if termini_list else 'ignore'
    ggraph = ensure_graph(glycan, termini = termini)
    # round-trip through strings so termini are recalculated with the floating bit attached
    tgraphs = [glycan_to_nxGraph(t, termini = termini) for t in get_possible_topologies(ggraph, exhaustive = True)]
    if any('anchors' in d for _, d in ggraph.nodes(data = True)):
        # anchored bits already enumerate the complete, exact topology set; the feasibles database can only lose information here
        possibles = tgraphs
    else:
        sizes = {len(t) for t in tgraphs}
        # Residues map one to one onto compatible residues, and concrete ones (PTM-wildcarded, as compare_glycans does once either graph has a PTM) only onto equal ones,
        # so a pair in which the concrete residues of one graph do not fit into the residues of the other, if those are all concrete, is ruled out before compare_glycans

        def residues(g):
            if (hit := _RESIDUE_CACHE.get(g, False)) is not False:
                return hit
            if any('anchors' in d for _, d in g.nodes(data = True)):
                _RESIDUE_CACHE[g] = None
                return None
            r = [l if LINKAGE_LABEL.match(l) else PTM_REGEX.sub('O', l) for l in _sl(g) if
                 not LINKAGE_NODE_PATTERN.match(l)]
            w = build_wildcard_cache(set(r))
            _RESIDUE_CACHE[g] = hit = (Counter(l for l in r if l not in w), any(l in w for l in r))
            return hit

        tres = [residues(t) for t in tgraphs]
        possibles = [g for g in
                     (glycan_to_nxGraph(k, termini = 'calc') for k in feasibles if 2 * k.count('(') + 1 in sizes) for gr
                     in [residues(g)] if
                     any(compare_glycans(t, g) for t, tr in zip(tgraphs, tres) if tr is None or gr is None or
                         ((gr[1] or tr[0] <= gr[0]) and (tr[1] or gr[0] <= tr[0])))]
    # Rootward chains are counted by walking up the glycan and its possible topologies instead
    chains = [i for i, g in enumerate(gmotifs) if not isinstance(g, list) and not any('!' in l for l in _sl(g)) and
              all(g.has_edge(k + 1, k) for k in range(len(g) - 1)) and g.number_of_edges() == len(g) - 1]
    walked = dict(zip(chains, _count_chains([ggraph, *possibles], [gmotifs[i] for i in chains])))
    res = []
    for i, g in enumerate(gmotifs):
        spec = termini_list[i] if termini_list else termini_list
        count_in = (lambda gr: len(get_match(g, gr))) if isinstance(g, list) else (
            lambda gr: subgraph_isomorphism(gr, g, termini_list = spec, count = True))
        temp_res = walked[i][0] if i in walked else count_in(ggraph)
        if temp_res:
            res.append(float(temp_res))
            continue
        hits = walked[i][1:] if i in walked else [count_in(p) for p in possibles]
        res.append(float(np.mean(hits)) if hits else 0.0)
    return pd.DataFrame([res], columns = motifs.motif_name if isinstance(motifs, pd.DataFrame) else motifs,
                       index = [glycan] if isinstance(glycan, str) else [graph_to_string(glycan)], dtype = 'float')


def get_molecular_properties(
        glycan_list: str | list[str], # Glycan(s) in IUPAC-condensed nomenclature
        verbose: bool = False, # Print SMILES not found on PubChem
        placeholder: bool = False, # Return dummy values instead of dropping failed requests
        pubchem: bool = False # Additionally fetch xlogp and complexity, the only two descriptors that cannot be computed from the structure
) -> pd.DataFrame: # DataFrame with molecular parameters
    "Computes molecular properties of glycans from their own SMILES, optionally enriched with the two empirical descriptors only PubChem has"
    from glycowork.motif.smiles import glycan_to_molecule, GlycanSMILESError
    from glycowork.motif.tokenization import calculate_adduct_mass
    VALENCE = {'C': 4, 'N': 3, 'O': 2, 'S': 2, 'P': 3, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1}
    glycan_list = [glycan_list] if isinstance(glycan_list, str) else glycan_list
    # Repeats are computed once and expanded at the end, which keeps the index unique for the placeholder reindex and the PubChem join
    local, keep, unique = {}, [], list(dict.fromkeys(glycan_list))
    for g in unique:
        try:
            mol = glycan_to_molecule(g)
        except (GlycanSMILESError, ValueError, KeyError):
            continue
        graph, order, nb = nx.Graph(), defaultdict(int), defaultdict(list)
        graph.add_nodes_from(range(len(mol.atoms)))
        for i, j, o in mol.bonds:
            order[i] += o
            order[j] += o
            nb[i].append((j, o))
            nb[j].append((i, o))
            graph.add_edge(i, j, order = o)
        # Every atom is written without its hydrogens except the bracketed stereocentres, so the count follows from the standard valence
        hs = {k: max(VALENCE.get(el, 0) - order[k] + ch, 0) for k, (el, ch, _) in enumerate(mol.atoms)}
        elements = Counter(a[0] for a in mol.atoms)
        elements['H'] = sum(hs.values())
        formula = ''.join(f'{e}{elements[e] if elements[e] > 1 else ""}' for e in ('C', 'H', *sorted(elements.keys() - {'C', 'H'})) if elements[e])  # Hill order, so halogens of residues like Gal6F are kept
        carbonyl = {i for i, (el, _, _) in enumerate(mol.atoms) if el == 'C' and any(mol.atoms[j][0] == 'O' and o == 2 for j, o in nb[i])}
        amide = {i for i, (el, _, _) in enumerate(mol.atoms) if el == 'N' and any(j in carbonyl for j, _ in nb[i])}
        # A rotatable bond is an acyclic single bond between two non-terminal heavy atoms, which is exactly a bridge of the molecular graph, minus the amide bond
        rotatable = sum(1 for i, j in nx.bridges(graph) if graph[i][j]['order'] == 1 and graph.degree(i) > 1 and graph.degree(j) > 1
                        and not ({i, j} <= (amide | carbonyl) and (i in amide) != (j in amide)))
        tpsa = sum((23.06 if ch < 0 else 17.07 if any(o == 2 for _, o in nb[i]) else 20.23 if hs[i] else 9.23) if el == 'O'
                   else {0: 3.24, 1: 12.03, 2: 26.02}.get(hs[i], 3.24) if el == 'N' else 0.0
                   for i, (el, ch, _) in enumerate(mol.atoms))  # Ertl's topological polar surface area, over the O and N environments a glycan can present
        # Only the tagged centres are knowable without CIP perception, so the undefined ones an unspecified anomer leaves behind are not reported at all rather than reported as zero
        local[g] = {'molecular_formula': formula, 'molecular_weight': calculate_adduct_mass(formula, mass_value = 'average'),
                    'exact_mass': calculate_adduct_mass(formula), 'monoisotopic_mass': calculate_adduct_mass(formula),
                    'tpsa': round(tpsa, 2), 'h_bond_donor_count': sum(hs[i] for i, (el, _, _) in enumerate(mol.atoms) if el in 'NO'),
                    'h_bond_acceptor_count': sum(1 for i, (el, _, _) in enumerate(mol.atoms) if el == 'O' or (el == 'N' and i not in amide)),
                    'rotatable_bond_count': rotatable, 'charge': sum(a[1] for a in mol.atoms), 'heavy_atom_count': len(mol.atoms),
                    'ring_count': len(mol.rings), 'covalent_unit_count': 1, 'isotope_atom_count': 0,
                    'defined_atom_stereo_count': sum(bool(a[2]) for a in mol.atoms)}
        keep.append(g)
    df_local = pd.DataFrame([local[g] for g in keep], index = keep)
    if not pubchem:
        # Placeholder imputation takes the median over every row, repeats included
        return df_local.reindex([g for g in glycan_list if g in local]) if not placeholder else df_local.reindex(glycan_list).fillna(
            df_local.reindex([g for g in glycan_list if g in local]).median(numeric_only = True))
    try:
        import pubchempy as pcp
    except ImportError:
        raise ImportError("Fetching xlogp and complexity needs pubchempy ('pip install pubchempy'); every other property is already computed locally with pubchem = False.")
    if placeholder:
        dummy = IUPAC_to_SMILES(['Glc'])[0]
        dummy_compound = None
    compounds_list, succeeded_requests, failed_requests = [], [], []
    for s, g in zip(IUPAC_to_SMILES(unique), unique):
        try:
            c = pcp.get_compounds(s, 'smiles')[0]
            if c.cid is None:
                raise ValueError("no PubChem match")
            compounds_list.append(c)
            succeeded_requests.append(g)
        except Exception:
            failed_requests.append(s)
            if placeholder:  # the placeholder row belongs to this glycan, otherwise the index silently stops matching the input
                if dummy_compound is None:
                    dummy_compound = pcp.get_compounds(dummy, 'smiles')[0]
                compounds_list.append(dummy_compound)
                succeeded_requests.append(g)
    if verbose and len(failed_requests) >= 1:
        print('The following SMILES were not found on PubChem:\n')
        print(failed_requests)
    try:
        df = pcp.compounds_to_frame(compounds_list, properties = ['xlogp', 'complexity'])
        df = df.reset_index(drop = True)
        df.index = succeeded_requests
    except (KeyError, ValueError) as e:
        if verbose:
            print(f'PubChem API returned incomplete data: {e}')
        df = pd.DataFrame(index = succeeded_requests)
    # The locally derived columns are exact, so they win wherever both sources have an opinion, and they also carry the rows PubChem lost
    df = df.drop(columns = [c for c in df.columns if c in df_local.columns], errors = 'ignore')
    return df_local.join(df, how = 'outer' if placeholder else 'left').reindex(
        [g for g in glycan_list if g in df_local.index or g in df.index])


def get_size_branching_features(
        glycans: str | list[str],  # Glycan(s) in IUPAC-condensed nomenclature
        n_bins: int = 3  # Number of bins/features to create for size and branching level; 0 for the raw counts
) -> pd.DataFrame:  # Glycans x one-hot Size_<bin> and Branch_<bin> columns, or Size and Branching counts if n_bins = 0
    "Generate binned features for glycan size (parentheses count) and branching (bracket count)"
    glycans = [glycans] if isinstance(glycans, str) else glycans
    # Calculate size and branching for each glycan
    sizes = [glycan.count('(') + 1 for glycan in glycans]
    branchings = [glycan.count('[') for glycan in glycans]
    if n_bins == 0:
        return pd.DataFrame({'Size': sizes, 'Branching': branchings}, index = glycans)

    # Helper function to create bin edges and labels
    def create_bins(values: list[int], n_bins: int) -> tuple[list[float], list[str]]:
        min_val, max_val = min(values), max(values)
        # Never more bins than integers in range, so none is empty; a bin holds the integers from the ceiling of its left edge up to below its right edge, which is what np.digitize puts there
        edges = list(np.linspace(min_val - 0.5, max_val + 0.5, min(n_bins, max_val - min_val + 1) + 1))
        bounds = [(int(np.ceil(edges[i])), int(np.ceil(edges[i + 1])) - 1) for i in range(len(edges) - 1)]
        return edges, [f"{lo}" if lo == hi else f"{lo} - {hi}" for lo, hi in bounds]

    # Create bins for size and branching
    size_edges, size_labels = create_bins(sizes, n_bins)
    branch_edges, branch_labels = create_bins(branchings, n_bins)
    # Assign values to bins
    size_bins, branch_bins = np.digitize(sizes, size_edges) - 1, np.digitize(branchings, branch_edges) - 1
    rows = np.arange(len(glycans))
    size_arr, branch_arr = np.zeros((len(glycans), len(size_labels)), dtype = int), np.zeros((len(glycans), len(branch_labels)), dtype = int)
    ok_s, ok_b = size_bins < len(size_labels), branch_bins < len(branch_labels)
    size_arr[rows[ok_s], size_bins[ok_s]] = 1
    branch_arr[rows[ok_b], branch_bins[ok_b]] = 1
    # Combine size and branching features
    return pd.concat([pd.DataFrame(size_arr, index = glycans, columns = [f"Size_{label}" for label in size_labels]),
                      pd.DataFrame(branch_arr, index = glycans, columns = [f"Branch_{label}" for label in branch_labels])], axis = 1)


@rescue_glycans
def annotate_dataset(
        glycans: str | list[str],  # Glycan(s) in IUPAC-condensed nomenclature
        motifs: pd.DataFrame | None = None,  # Motif dataframe (name + sequence); defaults to motif_list
        feature_set: str | list[str] = ['known'], # Feature types to analyze: known, graph, exhaustive, terminal(1-3), custom, chemical, size_branch
        termini_list: list = [], # Monosaccharide positions: 'terminal', 'internal', or 'flexible'
        condense: bool = False, # Remove columns with only zeros
        custom_motifs: str | list[str] = [] # Custom motifs (sequences, r-prefixed glyco-regexes, or motif_list names) when using 'custom' feature set; columns keep these labels
) -> pd.DataFrame: # DataFrame mapping glycans to motif counts and other features
    "Comprehensive glycan annotation combining multiple feature types: structural motifs, graph properties, terminal sequences"
    glycans = [glycans] if isinstance(glycans, str) else glycans
    if glycans is None or isinstance(glycans, dict) or not len(glycans):
        # an empty list failed in pandas ('Shape of passed values is (0, 1)'), and a composition dict was annotated as the glycans 'Hex', 'HexNAc', ...
        raise ValueError(f"annotate_dataset needs one or more glycans as strings, got {glycans!r:.80}; a composition is passed as a string such as 'H5N4F1'.")
    if any(k in ''.join(glycans) for k in (';', 'β', 'α', 'RES', '=')):
        raise Exception
    # Repeated sequences, common in glycomics tables, have identical rows, so each is annotated once and the rows are expanded at the end
    all_glycans, unique = glycans, list(dict.fromkeys(glycans))
    # Reducing-end position is expressed through termini specs, not through the residue label; deduplicating first keeps a glycan and its alditol apart, whose chemical features differ
    glycans = [g[:-3] if g.endswith('-ol') and not _STRUCTURAL_ALDITOL.search(g) else g for g in unique]
    if isinstance(feature_set, str):
        feature_set = [feature_set]
    custom_motifs = ([custom_motifs] if isinstance(custom_motifs, str) else custom_motifs) if 'custom' in feature_set else []
    valid_features = {'known', 'graph', 'terminal', 'terminal1', 'terminal2', 'terminal3', 'custom', 'chemical',
                      'exhaustive', 'size_branch'}
    if invalid_features := set(feature_set) - valid_features:
        raise ValueError(
            f"Unrecognized feature(s) {', '.join(sorted(invalid_features))}; choose from {', '.join(sorted(valid_features))}.")
    if motifs is None:
        motifs = motif_list
    # Checks whether termini information is provided
    if not termini_list:
        termini_list = list(map(ast.literal_eval, motifs.termini_spec)) if 'termini_spec' in motifs.columns else [
            ['flexible'] * (1 + m.count('(') - m.endswith(')')) for m in motifs.motif]
    shopping_cart = []
    # Custom motifs are counted alongside the known ones, under the label they were given; a motif name is as valid an input here as it is in GlycoDraw or glyco_filter, and its termini spec is half of what the name means, so it is counted with it, while sequences stay position-agnostic
    resolved = [resolve_motif_name(m) or (m, []) for m in custom_motifs]
    if unknown := [m for m, (seq, _) in zip(custom_motifs, resolved) if '(' not in seq and not seq.startswith('r') and seq not in lib]:
        raise ValueError(f"Custom motif(s) {unknown} are neither motif_list names nor glycan sequences.")
    names = [*(motifs.motif_name if 'known' in feature_set else []), *custom_motifs]
    seqs = [*(motifs.motif if 'known' in feature_set else []), *(seq for seq, _ in resolved)]
    specs = [*(termini_list if 'known' in feature_set else []), *(list(sp) if sp or seq.startswith('r') else ['flexible'] * (1 + seq.count('(') - seq.endswith(')')) for seq, sp in resolved)]
    if names:
        # r-prefixed rows are glyco-regular expressions, pre-compiled into chunk lists that the counting step tells apart from motif graphs
        gmotifs = [compile_pattern(mo[1:]) if mo.startswith('r') else glycan_to_nxGraph(mo, termini = 'provided', termini_list = t) for mo, t in zip(seqs, specs)]
        if '{' in ''.join(glycans):
            feasibles = set(loader.df_species[loader.df_species.Class == "Mammalia"].glycan.values.tolist())
            partial_annotate_topology_uncertainty = partial(annotate_glycan_topology_uncertainty, feasibles = feasibles, motifs = names, termini_list = specs, gmotifs = gmotifs)
        # Every motif residue needs a glycan residue of compatible label and position, as written or PTM-wildcarded, so indexing which glycans carry which rules out most pairs before subgraph isomorphism;
        # a negated residue and its linkage may be missing from a match and are not required, and a glyco-regex requires the residues of every chunk it must match in a single form
        ggraphs = [None if g.count('{') == 1 else ensure_graph(g, termini = 'calc') for g in glycans]
        by_node, comp, cand = defaultdict(set), {}, []
        for gi, gg in enumerate(ggraphs):
            for _, d in (gg.nodes(data = True) if gg is not None else ()):
                by_node[d['string_labels'], d.get('termini', 'flexible')].add(gi)
        reqs = [({(l, 'flexible') for c in map(compile_component, m) if c['min'] and not c['absent'] and len(c['motifs']) == 1 and '!' not in c['motifs'][0]
                  for l in _sl(glycan_to_nxGraph(c['motifs'][0]))} if isinstance(m, list) else
                 {(d['string_labels'], d.get('termini', 'flexible')) for n, d in m.nodes(data = True) if '!' not in d['string_labels'] + (m.nodes[n - 1]['string_labels'] if n - 1 in m else '')}) or None
                for m in gmotifs]
        ptm = {l: l if LINKAGE_LABEL.match(l) else PTM_REGEX.sub('O', l) for l, _ in {*by_node, *(r for req in reqs if req for r in req)}}
        nm = categorical_node_match_wildcard('string_labels', 'unknown', build_wildcard_cache(set(ptm) | set(ptm.values())), 'termini', 'flexible')
        for req in reqs:
            for l2, t2 in (req or set()) - comp.keys():
                comp[l2, t2] = set().union(*(gis for (l, t), gis in by_node.items() if nm({'string_labels': l, 'termini': t}, {'string_labels': l2, 'termini': t2}) or
                                             nm({'string_labels': ptm[l], 'termini': t}, {'string_labels': ptm[l2], 'termini': t2})))
            cand.append(set.intersection(*(comp[r] for r in req)) if req else None)
        # Rootward chains, about half of the known motifs, are counted by walking up the glycans instead
        chains = [i for i, m in enumerate(gmotifs) if not isinstance(m, list) and not any('!' in l for l in _sl(m)) and
                  all(m.has_edge(k + 1, k) for k in range(len(m) - 1)) and m.number_of_edges() == len(m) - 1]
        walked = dict(zip(chains, _count_chains([nx.DiGraph() if gg is None else gg for gg in ggraphs], [gmotifs[i] for i in chains])))
        rows = [partial_annotate_topology_uncertainty(g).iloc[0].tolist() if gg is None else
                [walked[i][gi] if i in walked else 0 if c is not None and gi not in c else len(get_match(m, gg)) if isinstance(m, list) else
                 subgraph_isomorphism(gg, m, termini_list = specs[i], count = True) for i, (m, c) in enumerate(zip(gmotifs, cand))]
                for gi, (g, gg) in enumerate(zip(glycans, ggraphs))]
        # Averages over possible topologies make every column of a dataset with floating glycans fractional
        shopping_cart.append(pd.DataFrame(np.array(rows, dtype = float if any(gg is None for gg in ggraphs) else int), columns = pd.Index(names, name = 'motif_name'), index = glycans))
    if 'graph' in feature_set:
        # Calculates graph features of each glycan
        shopping_cart.append(pd.concat(list(map(generate_graph_features, glycans)), axis = 0))
    if 'exhaustive' in feature_set:
        # Counts disaccharides and monosaccharides in each glycan
        temp = get_k_saccharides(glycans, size = 2, up_to = True)
        temp.index = glycans
        shopping_cart.append(temp)
    if 'chemical' in feature_set:
        # Computed on the sequences as given, since an alditol carries two more hydrogens; placeholder imputation takes the median over every row, repeats included
        shopping_cart.append(get_molecular_properties(all_glycans, placeholder = True).select_dtypes('number')[~pd.Index(all_glycans).duplicated()].set_axis(glycans))
    if any(t in feature_set for t in ('terminal', 'terminal1', 'terminal2', 'terminal3')):
        bag1, bag2, bag3 = [], [], []
        if 'terminal' in feature_set or 'terminal1' in feature_set:
            bag1 = list(map(get_terminal_structures, glycans))
        if 'terminal2' in feature_set:
            bag2 = [get_terminal_structures(glycan, size = 2) for glycan in glycans]
        if 'terminal3' in feature_set:
            bag3 = [get_terminal_structures(glycan, size = 3) for glycan in glycans]
        bags = [b for b in [bag1, bag2, bag3] if b]
        bag = [sum(items, []) for items in zip(*bags)] if bags else []
        potentials = get_minimal_ksaccharide_ambiguity(glycans, motifs = list(set(unwrap(bag))))
        new_additions = sorted(set(list(potentials.keys()) + list(potentials.values())))
        # These were seeded from non-reducing ends, so demand that position when counting; without it Terminal_ columns silently count internal occurrences too
        specs = {m: ['terminal'] + ['flexible'] * (m.count('(') - (1 if m.endswith(')') else 0)) for m in new_additions}
        gmotifs_terminal = [glycan_to_nxGraph(m, termini = 'provided', termini_list = specs[m]) for m in new_additions]
        ggraphs = [glycan_to_nxGraph(g, termini = 'calc') for g in glycans]
        # Terminal structures are rootward chains, so they are counted by walking up the glycans rather than by a VF2 search per motif and glycan
        counts_dict = dict(zip(new_additions, _count_chains(ggraphs, gmotifs_terminal)))
        sia_motifs = {k for k in counts_dict if 'Sia' in k and 'Neu5Ac' not in k and 'Neu5Gc' not in k}
        specific_vals = {tuple(v) for k, v in counts_dict.items() if k not in sia_motifs}
        counts_dict = {k: v for k, v in counts_dict.items() if k not in sia_motifs or tuple(v) not in specific_vals}
        bag_out = pd.DataFrame(counts_dict).fillna(0).astype(int)
        bag_out.index = glycans
        bag_out.columns = ['Terminal_' + c for c in bag_out.columns]
        shopping_cart.append(bag_out)
    if 'size_branch' in feature_set:
        shopping_cart.append(get_size_branching_features(glycans))
    temp = pd.concat(shopping_cart, axis = 1)
    temp = temp.loc[:, ~temp.columns.duplicated()]
    if len(unique) < len(all_glycans):
        temp = temp.iloc[list(map({g: i for i, g in enumerate(unique)}.get, all_glycans))]
    temp.index = all_glycans  # rows were built from the de-reduced sequences but must come back keyed on what the caller passed in, or downstream index alignment silently drops everything
    return temp.loc[:, (temp != 0).any(axis = 0)] if condense else temp


def get_motif_dag(
        motifs: list[str], # Motif labels as produced by annotate_dataset/quantify_motifs, optionally per glycosite as protein_site_motif
        abundances: pd.DataFrame | None = None # Motifs x samples abundances, used as an exact prefilter for containment (not needed for per-glycosite motifs)
) -> nx.DiGraph: # Transitively reduced containment DAG; edge parent -> child means parent is a substructure of child
    "Builds the containment DAG of a motif set, in which a parent motif is a substructure of each of its children; per-glycosite motifs are only ordered within their glycosite"
    if any(GLYCOFORM_ID.fullmatch(str(m)) for m in motifs):
        # Containment is decided once between the distinct motifs and holds at every glycosite carrying both; the closure keeps a relation whose intermediate motif is missing at a site
        split = [split_glycoform_id(m) if GLYCOFORM_ID.fullmatch(str(m)) else ('', m) for m in motifs]
        closure = nx.transitive_closure_dag(get_motif_dag(list(dict.fromkeys(b for _, b in split))))
        by_site = {}
        for m, (site, b) in zip(motifs, split):
            if b in closure:
                by_site.setdefault(site, []).append((m, b))
        dag = nx.DiGraph()
        dag.add_nodes_from(m for members in by_site.values() for m, _ in members)
        dag.add_edges_from((p, c) for members in by_site.values() for p, bp in members for c, bc in members if closure.has_edge(bp, bc))
        tr = nx.transitive_reduction(dag)
        dag.remove_edges_from([e for e in dag.edges() if not tr.has_edge(*e)])
        return dag
    pat, tgt, spec, amb, regexy = {}, {}, {}, {}, set()
    for m in motifs:
        s, sp = _motif_sequence(m)
        if s.startswith('r'):
            # A glyco-regex has no motif graph, but every match must contain its mandatory single-alternative chunks, so the largest of those stands in for it
            cands = [c['motifs'][0] for c in map(compile_component, compile_pattern(s)) if
                     c['min'] and not c['absent'] and len(c['motifs']) == 1]
            if not cands:
                continue
            s = max(cands, key = lambda x: x.count('('))
            sp = ['flexible'] * (1 + s.count('(') - (1 if s.endswith(')') else 0))
            regexy.add(m)
        try:
            pat[m] = glycan_to_nxGraph(s, termini = 'provided', termini_list = sp)
        except Exception:
            continue  # non-structural features (graph/chemical/size_branch) have no place in the DAG
        sp = expand_termini_list(s, sp)
        t = glycan_to_nxGraph(s).copy()
        # A host may attach anything at c's open ends, so only what c itself pins down is provable: a node carrying something above it inside c is internal, an open end is what c declares or else genuinely unknown
        nx.set_node_attributes(t, {n: 'internal' if (t.out_degree(n) and t.in_degree(n)) else (sp[n] if sp[n] != 'flexible' else 'unknown') for n in t.nodes()}, 'termini')
        tgt[m], spec[m] = t, sp
        amb[m] = _motif_ambiguity(m)
    cols = list(pat)
    dag = nx.DiGraph()
    dag.add_nodes_from(cols)
    A = abundances.loc[cols].values if abundances is not None else None
    for i, p in enumerate(cols):
        if p in regexy:
            continue  # a glyco-regex only stands for the backbone every match must contain, which proves what contains it but never what it contains
        # p contains c implies count(p) >= count(c) in every glycan, hence abundance dominance is a necessary condition and an exact prefilter
        dom = (A <= A[i] + 1e-9).all(axis = 1) if A is not None else np.ones(len(cols), dtype = bool)
        for j, c in enumerate(cols):
            if i == j or not dom[j] or len(pat[p]) > len(tgt[c]) or not subgraph_isomorphism(tgt[c], pat[p], termini_list = spec[p]):
                continue
            if len(pat[p]) == len(tgt[c]) and c not in regexy and (amb[p], i) < (amb[c], j) and subgraph_isomorphism(tgt[p], pat[c],
                                                                                                 termini_list = spec[
                                                                                                     c]):
                continue  # mutually isomorphic labels: the wildcard form matches strictly more structures and so is the ancestor, leaving the specific form as the descendant; the positional fallback makes the order total and the DAG acyclic
            dag.add_edge(p, c)
    # networkx adds each node's reduced out-edges from a set, i.e., in string-hash order, which made every successor-ordered statistic downstream (e.g., Redistribution p-val) change between interpreter runs; pruning dag itself keeps its insertion order
    tr = nx.transitive_reduction(dag)
    dag.remove_edges_from([e for e in dag.edges() if not tr.has_edge(*e)])
    return dag


def get_composition_dag(
        compositions: list[str], # Composition or IUPAC-condensed labels, optionally prefixed as protein_site_composition
        abundances: pd.DataFrame | None = None # Glycoforms x samples abundances, used as an exact prefilter for containment
) -> nx.DiGraph: # Transitively reduced containment DAG; edge parent -> child means parent is contained in child
    "Builds the containment DAG of a glycoform set, ordering parts by substructure containment where sequences are given and by component-wise composition dominance otherwise"
    from glycowork.motif.tokenization import glycan_to_composition
    comp, site, struct = {}, {}, {}
    for c in compositions:
        pre, _, tail = str(c).rpartition('_')  # glycoproteomics indices carry a protein_site prefix
        if is_composition(tail):
            try:
                comp[c], site[c] = canonicalize_composition(tail), pre
            except Exception:
                continue  # labels that do not parse as a composition have no place in the DAG
        else:
            try:
                cc = glycan_to_composition(tail)  # canonicalize_composition does not raise on a sequence, it returns a nonsense residue vector, so sequences have to be routed here before it is ever called on them
                if not cc:
                    continue  # an unmappable glycoletter comes back as {}, not as an exception, and an all-zero row is dominated by every composition and would become the ancestor of the whole block
                comp[c], struct[c], site[c] = cc, glycan_to_nxGraph(tail), pre
            except Exception:
                continue
    cols = list(site)
    residues = sorted({r for d in comp.values() for r in d})
    V = np.array([[comp.get(c, {}).get(r, 0) for r in residues] for c in cols])
    A = abundances.loc[cols].values if abundances is not None else None
    dag = nx.DiGraph()
    dag.add_nodes_from(cols)
    blocks = {}
    for i, c in enumerate(cols):
        blocks.setdefault(site[c], []).append(i)
    for members in blocks.values():
        # glycoforms of different glycosites are separate compositional systems and never contain one another, so containment is decided one site block at a time and never across the full n^2
        m = np.array(members)
        # composition dominance is only a necessary condition for containment, so where sequences are known the real relation is used: it keeps Man5 out of the ancestry of complex glycoforms, whose alpha1-2 mannoses are trimmed rather than extended
        le = (V[m][:, None, :] <= V[m][None, :, :]).all(2)
        for i, j in zip(*np.nonzero(le)):
            # decided per pair rather than per block, so one composition-only label no longer costs every sequence at its glycosite the real containment relation
            if i != j and cols[m[i]] in struct and cols[m[j]] in struct:
                le[i, j] = subgraph_isomorphism(struct[cols[m[j]]], struct[cols[m[i]]])
        # equal parts are ordered by position, which makes the order total and the DAG acyclic
        M = le & (~le.T | (m[:, None] < m[None, :]))
        if A is not None:
            # p contained in c implies every glycoform counted toward c is counted toward p, hence abundance dominance is a necessary condition and an exact prefilter
            a = A[m]
            seen_both = ~np.isnan(a)[None, :, :] & ~np.isnan(a)[:, None, :]
            M &= np.where(seen_both, a[None, :, :] <= a[:, None, :] + 1e-9, True).all(
                2)  # a sample in which a part was never measured is no evidence against dominance; comparing against NaN silently deletes every edge at any glycosite that is not observed everywhere
        np.fill_diagonal(M, False)
        dag.add_edges_from((cols[m[i]], cols[m[j]]) for i, j in zip(*np.nonzero(M)))
    # networkx adds each node's reduced out-edges from a set, i.e., in string-hash order, which made every successor-ordered statistic downstream (e.g., Redistribution p-val) change between interpreter runs; pruning dag itself keeps its insertion order
    tr = nx.transitive_reduction(dag)
    dag.remove_edges_from([e for e in dag.edges() if not tr.has_edge(*e)])
    return dag


def deduplicate_motifs(
        df: pd.DataFrame # DataFrame with glycan motifs as rows, samples as columns
) -> pd.DataFrame: # DataFrame with redundant motifs removed
    "Removes redundant motif entries from glycan abundance data while preserving the most informative labels"
    # Keep the least ambiguous label per group of identical rows; wildcards are longer than the specifics they abstract, so length only ever breaks ties between equally specific labels
    ranks = [_motif_ambiguity(m) for m in df.index]
    # Grouped on positional column labels, since a repeated label (e.g., an annotated glycan list with duplicates) makes a label-based grouper ambiguous
    grouper = df.set_axis(range(df.shape[1]), axis = 1)
    grouper['_original_position'] = range(len(df))
    keep = [g['_original_position'].iloc[min(range(len(g)), key = lambda k: ranks[g['_original_position'].iloc[k]])]
            for _, g in grouper.groupby(list(grouper.columns[:-1]), sort = False, dropna = False)]
    return df.iloc[keep]


def quantify_motifs(
        df: pd.DataFrame | str | Path, # DataFrame, filepath, or shipped dataset name, with samples as columns, abundances as values
        glycans: list[str] | None = None, # List of IUPAC-condensed glycan sequences or compositions, or glycoproteomics glycoforms as protein_site_glycan; auto-detected from the first column, or the index, if None
        feature_set: str | list[str] = ['known', 'exhaustive'], # Feature types to analyze: known, graph, exhaustive, terminal(1-3), custom, chemical, size_branch
        custom_motifs: str | list[str] = [], # Custom motifs (sequences, r-prefixed glyco-regexes, or motif_list names) when using 'custom' feature set
        remove_redundant: bool = True  # Remove redundant motifs via deduplicate_motifs
) -> pd.DataFrame:  # DataFrame with motif abundances (motifs as rows, samples as columns); for glycoforms, one row per glycosite and motif as protein_site_motif
    "Extracts and quantifies motif abundances from glycan abundance data by weighting motif occurrences; glycoproteomics glycoforms are quantified per glycosite"
    if not isinstance(df, pd.DataFrame):
        from glycowork.glycan_data.data_entry import read_abundances
        df = read_abundances(df)
    if glycans is None:
        if pd.api.types.is_string_dtype(df.iloc[:, 0]):
            glycans = df.iloc[:, 0].tolist()
            df = df.iloc[:, 1:]
        elif len(df) and isinstance(df.index[0], str):
            glycans = df.index.tolist()  # glycans held in the index, as after set_index('glycan')
        else:
            raise ValueError("glycans must be provided if the first column is not glycan strings")
    # Glycoforms of one glycosite compete for that site only, so a motif is summed per site instead of over the whole run; before, protein_site_glycan labels were annotated as one unknown monosaccharide each and came back unchanged as 'motifs'
    sites = [split_glycoform_id(g) for g in glycans] if len(glycans) and all(GLYCOFORM_ID.fullmatch(str(g)) for g in glycans) else None
    # Motif extraction
    df_motif = annotate_dataset([g for _, g in sites] if sites else glycans, feature_set = feature_set,
                                condense = True, custom_motifs = custom_motifs)
    collect_dic = {}
    df = df.T
    log2 = (df.select_dtypes(include = 'number') < 0).any().any()
    if sites:
        X, W = df.to_numpy(dtype = float), df_motif.to_numpy(dtype = float)
        X = np.power(2, X) if log2 else X
        site_of = pd.Series([s for s, _ in sites])
        vals, labels, owner = [], [], []
        for site, idx in site_of.groupby(site_of, sort = False).indices.items():
            cols = np.flatnonzero((W[idx] > 0).any(axis = 0))
            v = np.nan_to_num(X[:, idx]) @ W[np.ix_(idx, cols)]
            seen = ~np.isnan(X[:, idx]).all(axis = 1, keepdims = True)  # per site, not per motif: a measured site without any carrier of the motif has it at 0
            vals.append(np.where(seen, np.log2(np.where(seen, v, 1)) if log2 else v, np.nan).T)  # a site never measured in a sample has no motif abundance there either, not a zero
            labels.extend(df_motif.columns[cols])
            owner.extend([site] * len(cols))
        out = pd.DataFrame(np.vstack(vals) if vals else np.empty((0, len(df))), index = labels, columns = df.index).assign(_site = owner)
        out = deduplicate_motifs(out) if remove_redundant else out  # the site column is part of the grouping key, so only identical motifs of the same site are merged
        return out.set_axis([f'{s}_{m}' for s, m in zip(out['_site'], out.index)]).drop(columns = '_site')
    # Motif quantification
    for col in df_motif.columns:
        indices = [i for i, x in enumerate(df_motif[col]) if x > 0]
        temp = df.iloc[:, indices]
        temp.columns = range(temp.columns.size)
        weights = df_motif[col].iloc[indices].reset_index(drop = True)
        if log2:
            linear_values = np.power(2, temp)
            collect_dic[col] = np.log2((linear_values * weights).sum(axis = 1))
        else:
            collect_dic[col] = (temp * weights).sum(axis = 1)
    df = pd.DataFrame(collect_dic).T
    return df if not remove_redundant else deduplicate_motifs(df)


def quantify_dag_steps(
        M: pd.DataFrame, # Motifs x samples abundances, as returned by quantify_motifs
        min_share: float = 1e-3, # Minimum median share of a sample's total motif signal; rarer motifs are dropped before the DAG is built
        verbose: bool = False # Print which motifs and features were dropped
) -> pd.DataFrame: # Features x samples; each row is a motif's log2 ratio to its tightest containment parent, or to the sample total for DAG roots
    "Re-expresses motif abundances as biosynthetic step yields along the motif containment DAG, which are non-redundant and scale-free and so need no log-ratio transformation"
    # Motifs at trace level carry six log2 units of detection noise once logged, which swamps real steps spanning tenths of a unit, so they are dropped before the DAG and their edges never exist
    share = M.div(M.sum(axis = 0), axis = 1)
    kept = share.index[share.median(axis = 1) >= min_share]
    if verbose and len(kept) < len(M):
        print(f"  {len(M) - len(kept)} trace motifs below {min_share:.0e} median share dropped: "
              f"{', '.join(sorted(set(M.index) - set(kept))[:6])}{' ...' if len(M) - len(kept) > 6 else ''}")
    M = M.loc[kept]
    # Non-structural features (graph, chemical, size_branch) have no containment relation, so get_motif_dag gives them no node and they drop out here
    dag = get_motif_dag(M.index.tolist(), abundances = M)
    # 0.65 x the smallest nonzero value: multiplicative replacement for continuous compositional data (Martin-Fernandez et al. 2003)
    A = M + 0.65 * float(M.values[M.values > 0].min())
    total = M.sum(axis = 0).values
    R = pd.DataFrame({m: np.log2(A.loc[m].values / A.loc[list(dag.predecessors(m))].min(axis = 0).values) if dag.in_degree(m) else np.log2(A.loc[m].values / total)
                      for m in dag.nodes()}, index = M.columns).T
    n_raw = len(R)
    # Relative, not absolute: an edge like Terminal_Neu5Ac(a2-3) -> Neu5Ac(a2-3)Gal is structurally empty (every terminal a2-3 sits on a Gal) but one sample deviating by 0.02 defeats a fixed floor
    sd = R.std(axis = 1)
    R = R.loc[sd >= 0.01 * sd.median()].drop_duplicates()
    if verbose:
        n_roots = sum(1 for m in dag.nodes() if not dag.in_degree(m))
        print(f"  {len(R)} DAG features from {len(M)} motifs "
              f"({len(dag) - n_roots} conditional yields + {n_roots} roots, {n_raw - len(R)} constant or duplicated)")
    return R


def count_unique_subgraphs_of_size_k(
        graph: nx.DiGraph, # NetworkX graph of a glycan
        size: int = 2, # Number of monosaccharides per subgraph
        terminal: bool = False # Only count terminal subgraphs
) -> dict[str, float]: # Dictionary mapping k-saccharide sequences to counts
    "Identifies and counts unique connected subgraphs of specified size from glycan structure"
    target_size = size * 2 - 1
    successor_dict = {v: list(graph.successors(v)) for v in graph.nodes}
    is_mono = {n: not LINKAGE_NODE_PATTERN.match(graph.nodes[n].get('string_labels', '')) for n in graph.nodes()}
    k_subgraphs, processed = [], set()
    terminal_lookup = {n: graph.out_degree(n) == 0 for n in graph.nodes()} if terminal else {}
    for node in sorted(graph.nodes(), reverse = True):
        queue = deque([(tuple([node]), 1 if is_mono[node] else 0, 1)])
        while queue:
            current_sg, mono_cnt, curr_size = queue.popleft()
            if current_sg in processed: continue
            processed.add(current_sg)
            if curr_size == target_size:
                if mono_cnt == size and (not terminal or any(terminal_lookup[n] for n in current_sg)):
                    k_subgraphs.append(graph_to_string_int(graph.subgraph(current_sg)))
                continue
            current_sg_set = set(current_sg)
            candidates = [s for n in current_sg_set for s in successor_dict[n] if s not in current_sg_set]
            for candidate in candidates:
                new_mono = mono_cnt + (1 if is_mono[candidate] else 0)
                if new_mono > size: continue
                queue.append((tuple(sorted(current_sg_set | {candidate})), new_mono, curr_size + 1))
    return dict(Counter(k_subgraphs))


def get_minimal_ksaccharide_ambiguity(
        glycans: str | list,  # Glycan(s) in IUPAC-condensed nomenclature
        size: int = 2,  # Number of monosaccharides per fragment
        motifs: list[str] | None = None  # Pre-computed motifs (for terminal structures); default: the glycans' k-saccharides
) -> dict:  # Dictionary of precise-k-saccharide : ideal wildcarded k-saccharide
    "Maps each k-saccharide to the form that merges every linkage variant of its backbone seen in the input, plus Sia forms where both Neu5Ac and Neu5Gc versions occur"
    glycans = [glycans] if isinstance(glycans, str) else glycans
    if motifs is None:
        ksaccharides = {}
        for g in glycans:
            graph = glycan_to_nxGraph(g)
            ksac_dict = count_unique_subgraphs_of_size_k(graph, size = size, terminal = False)
            for ksac_str in ksac_dict:
                ksaccharides[ksac_str] = ksaccharides.get(ksac_str, 0) + ksac_dict[ksac_str]
    else:
        ksaccharides = {m: 1 for m in motifs}
    linkage_pattern = re.compile(r'\(([ab?])([0-9?/]+)-([0-9?/]+)\)')
    backbone_linkages = defaultdict(lambda: defaultdict(lambda: [set(), set(), set()]))
    for ksac in ksaccharides:
        backbone = linkage_pattern.sub('(LINK)', ksac)
        linkages = linkage_pattern.findall(ksac)
        for i, (anomer, start, end) in enumerate(linkages):
            if anomer != '?':
                backbone_linkages[backbone][i][0].add(anomer)
            for s in start.split('/'):
                if s != '?':
                    backbone_linkages[backbone][i][1].add(s)
            for e in end.split('/'):
                if e != '?':
                    backbone_linkages[backbone][i][2].add(e)
    result = {}
    for ksac in ksaccharides:
        backbone = linkage_pattern.sub('(LINK)', ksac)
        linkages = linkage_pattern.findall(ksac)
        reconstructed = backbone
        for i in range(len(linkages)):
            anomers, starts, ends = backbone_linkages[backbone][i]
            anomer = list(anomers)[0] if len(anomers) == 1 else '?'
            starts_sorted = sorted(starts, key = int) if starts else ['?']
            ends_sorted = sorted(ends, key = int) if ends else ['?']
            start = '/'.join(starts_sorted) if len(starts_sorted) > 1 else starts_sorted[0]
            end = '/'.join(ends_sorted) if len(ends_sorted) > 1 else ends_sorted[0]
            reconstructed = reconstructed.replace('(LINK)', f'({anomer}{start}-{end})', 1)
        result[ksac] = reconstructed
    for sia_type, other_sia in [('Neu5Ac', 'Neu5Gc'), ('Neu5Gc', 'Neu5Ac')]:
        for ksac in list(ksaccharides.keys()):
            if sia_type not in ksac or ksac.replace(sia_type, other_sia) not in ksaccharides:
                continue
            sia_key = ksac.replace(sia_type, 'Sia')
            if sia_key not in result:
                result[sia_key] = result.get(ksac, ksac).replace(sia_type, 'Sia')
    return result


@rescue_glycans
def get_k_saccharides(
        glycans: str | list[str] | set[str],  # Glycan(s) in IUPAC-condensed nomenclature
        size: int = 2,  # Number of monosaccharides per fragment
        up_to: bool = False, # Include fragments up to size k (adds monosaccharides)
        just_motifs: bool = False, # Return nested list of motifs instead of count DataFrame
        terminal: bool = False # Only count terminal fragments
) -> pd.DataFrame | list[list[str]]: # DataFrame of k-saccharide counts or list of motifs per glycan
    "Extracts k-saccharide fragments from glycan sequences with options for different fragment sizes and positions"
    if isinstance(glycans, str):
        glycans = [glycans]
    elif not isinstance(glycans, (list, set)):
        raise TypeError("The input has to be a glycan string or a list or set of glycans")
    if any(k in ''.join(glycans) for k in (';', 'β', 'α', 'RES', '=')):
        raise Exception
    if not glycans or (not up_to and max(g.count('(') + 1 for g in glycans) < size):
        return [] if just_motifs else pd.DataFrame()
    ggraphs = [glycan_to_nxGraph(g) for g in glycans]
    if up_to:
        wga_letter_data = []
        actual_glycans = [g for g in glycans if not is_composition(g)]
        lib = get_lib(actual_glycans)
        add_sia = {'Neu5Ac', 'Neu5Gc'}.issubset(lib)
        monos = [i for i in lib if i not in linkages and not LINKAGE_NODE_PATTERN.match(i)]
        for g, gg in zip(glycans, ggraphs):
            if is_composition(g):
                comp_dict = canonicalize_composition(g)
                wga_letter_data.append(comp_dict)
            else:
                # Counted on graph nodes, since a substring search also counts the Fuc in D-Fuc or the Hep in LDManHep
                c = Counter(_sl(gg))
                d = {i: c[i] for i in monos}
                if add_sia:
                    d['Sia'] = d.get('Neu5Ac', 0) + d.get('Neu5Gc', 0)
                wga_letter_data.append(d)
        wga_letter = pd.DataFrame(wga_letter_data)
        _class_map = {'Hex': Hex - {'Hex'}, 'dHex': dHex - {'dHex'}, 'HexNAc': HexNAc - {'HexNAc'},
                      'HexA': HexA - {'HexA'}, 'Pen': Pen - {'Pen'}}
        if not add_sia:
            _class_map['Sia'] = Sia - {'Sia'}
        for cls, members in _class_map.items():
            if cls not in wga_letter.columns:
                continue
            member_cols = [m for m in members if m in wga_letter.columns]
            if member_cols:
                wga_letter[cls] = wga_letter[member_cols].fillna(0).sum(axis = 1) + wga_letter[cls].fillna(0)
    counts_dict = {}
    for s in (range(2, size + 1) if up_to else (size,)):
        frags = [count_unique_subgraphs_of_size_k(g, size = s, terminal = terminal) for g in ggraphs]
        vocab = sorted({f for d in frags for f in d})
        vgraphs = {f: glycan_to_nxGraph(f) for f in vocab}
        fuzzy = {i for i, f in enumerate(vocab) if build_wildcard_cache(set(_sl(vgraphs[f]))) or _has_o(vgraphs[f])}
        potentials = get_minimal_ksaccharide_ambiguity(glycans, size = s, motifs = vocab)
        mgraphs = {n: glycan_to_nxGraph(n) for n in sorted(set(potentials.keys()) | set(potentials.values()))}
        # A fragment can only contain a motif if it has a compatible label, as written or PTM-wildcarded, for each of the motif's labels, which rules out nearly every pair before subgraph isomorphism
        ptm, by_label, occ = {l: l if LINKAGE_LABEL.match(l) else PTM_REGEX.sub('O', l) for h in (*vgraphs.values(), *mgraphs.values()) for l in _sl(h)}, defaultdict(set), defaultdict(list)
        nm = categorical_node_match_wildcard('string_labels', 'unknown', build_wildcard_cache(set(ptm) | set(ptm.values())), 'termini', 'flexible')
        for i, f in enumerate(vocab):
            for l in _sl(vgraphs[f]):
                by_label[l].add(i)
        comp = {l2: set().union(*(ids for l, ids in by_label.items() if nm({'string_labels': l}, {'string_labels': l2}) or
                                  nm({'string_labels': ptm[l]}, {'string_labels': ptm[l2]}))) for l2 in {l for m in mgraphs.values() for l in _sl(m)}}
        for gi, d in enumerate(frags):
            for f, c in d.items():
                occ[f].append((gi, c))
        # Two rootward chains of equal length (every disaccharide, most larger fragments) can only map position by position, so those pairs need no VF2 search
        chain = {f: all(h.has_edge(i + 1, i) for i in range(len(h) - 1)) and h.number_of_edges() == len(h) - 1 for f, h in (*vgraphs.items(), *mgraphs.items())}
        for n, m in mgraphs.items():
            cand = sorted(set.intersection(*(comp[l] for l in _sl(m))))
            contains = lambda f: all(nm({'string_labels': ptm[a], 'ptm': a} if o and ptm[a] != a else {'string_labels': a}, {'string_labels': ptm[b], 'ptm': b} if o and ptm[b] != b else {'string_labels': b})
                                     for o in [_has_o(vgraphs[f]) or _has_o(m)] for a, b in zip(_sl(vgraphs[f]), _sl(m))) if chain[f] and chain[n] and len(vgraphs[f]) == len(m) else subgraph_isomorphism(vgraphs[f], m)
            hits = [vocab[i] for i in cand if contains(vocab[i])] if build_wildcard_cache(set(_sl(m))) or _has_o(m) else \
                ([n] if n in vgraphs else []) + [vocab[i] for i in cand if i in fuzzy and vocab[i] != n and contains(vocab[i])]
            counts_dict[n] = col = [0] * len(frags)
            for gi, c in (x for f in hits for x in occ[f]):
                col[gi] += c
    df_counts = pd.DataFrame(counts_dict)
    if up_to:
        combined_df = pd.concat([wga_letter, df_counts], axis = 1).fillna(0).astype(int)
        return combined_df.apply(lambda x: list(combined_df.columns[x > 0]), axis = 1).tolist() if just_motifs else combined_df
    return df_counts.apply(lambda x: list(df_counts.columns[x > 0]), axis = 1).tolist() if just_motifs else df_counts.fillna(0).astype(int)


def get_terminal_structures(
        glycan: str | nx.DiGraph, # IUPAC-condensed glycan sequence or NetworkX graph
        size: int = 1 # Number of monosaccharides in terminal fragment (1 or higher)
) -> list[str]: # List of terminal structures with linkages
    "Identifies terminal monosaccharide sequences from non-reducing ends of glycan structure"
    ggraph = ensure_graph(glycan)
    labels = nx.get_node_attributes(ggraph, 'string_labels')
    result = []
    for k in sorted(ggraph.nodes()):
        if ggraph.out_degree[k] or not ggraph.in_degree[k] or labels[k] in linkages:
            continue
        # Walk rootward through each residue's parent linkage instead of assuming it is the next node index, which breaks across the parts of a glycan with floating substituents
        structure, node, monos = '', k, 0
        while node is not None and monos < size:
            structure += labels[node]
            monos += 1
            up = next(iter(ggraph.predecessors(node)), None)
            if up is None:
                break
            structure += '(' + labels[up] + ')'
            node = next(iter(ggraph.predecessors(up)), None)
        if monos == size:
            result.append(structure)
    return result


def create_correlation_network(
        df: pd.DataFrame, # DataFrame with samples as rows, glycans/motifs as columns
        correlation_threshold: float # Minimum correlation for cluster inclusion
) -> list[set[str]]: # List of sets containing correlated glycans/motifs
    "Groups glycans or motifs into clusters based on abundance correlation patterns"
    # Calculate the correlation matrix
    correlation_matrix = df.corr()
    # Create an adjacency matrix based on the correlation threshold
    adjacency_matrix = correlation_matrix > correlation_threshold
    # Create a graph from the adjacency matrix
    graph = nx.from_numpy_array(adjacency_matrix.values)
    # Use the graph to find connected components (clusters of highly correlated glycans)
    clusters = list(nx.connected_components(graph))
    # Convert indices back to original labels
    return [set(df.columns[list(cluster)]) for cluster in clusters]


def group_glycans_core(
        glycans: list[str], # List of IUPAC-condensed O-glycan sequences
        p_values: list[float] # Statistical test p-values for each glycan
) -> tuple[dict[str, list[str]], dict[str, list[float]]]: # (group:glycans dict, group:p-values dict)
    "Groups O-glycans by core structure (core1, core2, other) for statistical analysis"
    temp = dict(zip(glycans, p_values))
    grouped_glycans, grouped_p_values = {}, {}
    for g in glycans:
        if any(subgraph_isomorphism(g, sub_g) for sub_g in ["GlcNAc(b1-6)GalNAc", "GlcNAcOS(b1-6)GalNAc"]):
            group = "core2"
        elif any(subgraph_isomorphism(g, sub_g) for sub_g in ["Gal(b1-3)GalNAc", "GalOS(b1-3)GalNAc"]):
            group = "core1"
        else:
            group = "rest"
        grouped_glycans.setdefault(group, []).append(g)
        grouped_p_values.setdefault(group, []).append(temp[g])
    return grouped_glycans, grouped_p_values


def group_glycans_sia_fuc(
        glycans: list[str], # List of IUPAC-condensed glycan sequences
        p_values: list[float] # Statistical test p-values for each glycan
) -> tuple[dict[str, list[str]], dict[str, list[float]]]: # (group:glycans dict, group:p-values dict)
    "Groups glycans by sialic acid and fucose content for statistical analysis"
    temp = dict(zip(glycans, p_values))
    grouped_glycans, grouped_p_values = {}, {}
    for g in glycans:
        has_sia = any(s in g for s in ("Neu", "Kdn", "Sia"))
        has_fuc = "Fuc" in g
        if has_sia and has_fuc:
            group = "SiaFuc"
        elif has_sia:
            group = "Sia"
        elif has_fuc:
            group = "Fuc"
        else:
            group = "rest"
        grouped_glycans.setdefault(group, []).append(g)
        grouped_p_values.setdefault(group, []).append(temp[g])
    return grouped_glycans, grouped_p_values


def group_glycans_N_glycan_type(
        glycans: list[str], # List of IUPAC-condensed N-glycan sequences
        p_values: list[float] # Statistical test p-values for each glycan
) -> tuple[dict[str, list[str]], dict[str, list[float]]]: # (group:glycans dict, group:p-values dict)
    "Groups N-glycans by type (complex, hybrid, high-mannose, other) for statistical analysis"
    temp = dict(zip(glycans, p_values))
    grouped_glycans, grouped_p_values = {}, {}
    # An antenna is a HexNAc on either core arm, whatever its linkage; complex carries one on both arms, hybrid on only one
    arms = [compile_pattern("HexNAc-Mana3-."), compile_pattern("HexNAc-Mana6-.")]
    for g in glycans:
        antennae = sum(bool(get_match(arm, g, return_matches = False)) for arm in arms)
        if antennae == 2:
            group = "complex"
        elif antennae == 1:
            group = "hybrid"
        elif g.count("Man") > 3:
            group = "high_man"
        else:
            group = "rest"
        grouped_glycans.setdefault(group, []).append(g)
        grouped_p_values.setdefault(group, []).append(temp[g])
    return grouped_glycans, grouped_p_values


class Lectin():
    def __init__(
            self,
            abbr: list[str], # List of lectin abbreviations
            name: list[str], # List of lectin names
            specificity: dict[str, dict[str, list[str]] | None] = {"primary": None, "secondary": None, "negative": None}, # Binding specificity details
            species: str = "", # Source species
            reference: str = "", # Literature reference
            notes: str = "" # Additional information
    ) -> None:
        "Lectin object storing binding specificity and annotation information"
        self.abbr = abbr
        self.name = name
        self.species = species
        self.reference = reference
        self.notes = notes
        self.specificity = specificity

    def get_all_binding_motifs_count(self) -> int: # Total number of binding motifs
        "Counts total number of primary and secondary binding motifs"
        return len(self.specificity["primary"]) + \
            (len(self.specificity["secondary"]) if self.specificity["secondary"] else 0)

    def get_all_binding_motifs(self) -> list[str]: # List of all binding motifs
        "Returns combined list of primary and secondary binding motifs"
        return list(self.specificity["primary"].keys()) + list(self.specificity["secondary"].keys() if self.specificity["secondary"] else [])

    def show_info(self) -> None:
        "Displays formatted lectin information including specificities"
        print(f"Name(s): {'; '.join(self.name)}\n" +
              f"Abbreviation(s): {'; '.join(self.abbr)}\n" +
              f"Species: {self.species}\n" +
              ''.join(f"Specificity ({key}): {'; '.join(value) if value else 'no information'}\n"
                      for key, value in self.specificity.items()) +
              f"Reference: {self.reference}\nnotes: {self.notes}\n")

    def check_binding(
            self,
            glycan: str # IUPAC-condensed glycan sequence
    ) -> int: # Binding level (0:none, 1:primary, 2:secondary)
        "Evaluates glycan binding to lectin based on motif recognition"
        # A negative motif only blocks the site it sits on, so a blocking feature on one antenna must not veto an intact epitope on another
        blocked = {n for motif, termini in (self.specificity.get("negative") or {}).items()
                   for hit in
                   subgraph_isomorphism(glycan, motif, termini_list = termini, count = True, return_matches = True)[1]
                   for n in hit}
        for key, level in (("primary", 1), ("secondary", 2)):
            for motif, termini in (self.specificity.get(key) or {}).items():
                hits = subgraph_isomorphism(glycan, motif, termini_list = termini, count = True, return_matches = True)[
                    1]
                if any(not blocked.intersection(hit) for hit in hits):
                    return level
        return 0


def load_lectin_lib() -> dict[int, Lectin]: # Dictionary mapping indices to Lectin objects
    "Loads curated lectin library with binding specificity information"
    from glycowork.glycan_data.loader import lectin_specificity
    lectin_lib = {}
    for i, row in enumerate(lectin_specificity.itertuples()):
        specificity_dict = {"primary": row.specificity_primary, "secondary": row.specificity_secondary, "negative": row.specificity_negative}
        lectin_lib[i] = Lectin(abbr = row.abbreviation.split(","), name = row.name.split(","), specificity = specificity_dict,
                               species = row.species, reference = row.reference, notes = row.notes)
    return lectin_lib


def create_lectin_and_motif_mappings(
        lectin_list: list[str], # List of lectin names/abbreviations
        lectin_lib: dict[int, Lectin] # Library of Lectin objects
) -> tuple[dict[str, int], dict[str, dict[str, int]]]: # (lectin:index dict, motif:{lectin:weight} dict)
    "Creates mappings between lectins and their recognized motifs with binding strength weights"
    useable_lectin_mapping = {}
    motif_mapping = {}
    lib_lectin_lookup = {i: set(abbr.lower() for abbr in lib_lectin.abbr) for i, lib_lectin in lectin_lib.items()}
    # Match user's lectin list (of abbreviations) to the library lectins.
    for lectin in lectin_list:
        lectin_low = lectin.split("_")[0].lower()
        for i, abbr_set in lib_lectin_lookup.items():
            if lectin_low in abbr_set:
                useable_lectin_mapping[lectin] = i
                # When an input lectin is matched to a library lectin, assign weight class 0 and 1 to its primary and secondary motif
                for motif in lectin_lib[i].specificity["primary"]:
                    if motif not in motif_mapping:
                        motif_mapping[motif] = {lectin: 0}
                    else:
                        motif_mapping[motif][lectin] = 0
                if lectin_lib[i].specificity["secondary"] is not None:
                    for motif in lectin_lib[i].specificity["secondary"]:
                        if motif not in motif_mapping:
                            motif_mapping[motif] = {lectin: 1}
                        else:
                            motif_mapping[motif][lectin] = 1
                break
        if lectin not in useable_lectin_mapping:
            print(f'Lectin "{lectin}" is not found in our annotated lectin library and is excluded from analysis.')
    # Now add lectins of weight class 2 and 3
    for motif, mapping in motif_mapping.items():
        for lectin, i in useable_lectin_mapping.items():
            if lectin not in mapping:
                binder = lectin_lib[i].check_binding(motif)
                if binder:
                    motif_mapping[motif][lectin] = 2 if binder == 1 else 3
        # sort lectins by weight class
        motif_mapping[motif] = dict(sorted(mapping.items(), key = lambda x: x[1]))
    return useable_lectin_mapping, motif_mapping


def lectin_motif_scoring(
        useable_lectin_mapping: dict[str, int], # Lectin to index mapping
        motif_mapping: dict[str, dict[str, int]], # Motif to {lectin:weight} mapping
        lectin_score_dict: dict[str, float], # Lectin effect sizes
        lectin_lib: dict[int, Lectin], # Library of Lectin objects
        idf: dict[str, float], # Lectin variance across groups
        class_weight_dict: dict[int, float] = {0: 1, 1: 0.5, 2: 0.5, 3: 0.25}, # Weights for binding classes
        specificity_weight_dict: dict[str, float] = {"<3": 1, "3-4": 0.75, ">4": 0.5}, # Weights for binding specificity
        duplicate_lectin_penalty_factor: float = 0.8 # Penalty for redundant lectins
) -> pd.DataFrame: # DataFrame with scored motifs and supporting evidence
    "Calculates weighted motif scores from lectin binding data incorporating specificity and redundancy factors"
    output = []
    _vc = Counter(useable_lectin_mapping.values())
    useable_lectin_count = {k: _vc[v] for k, v in useable_lectin_mapping.items()}
    for motif, lectins in motif_mapping.items():
        score = 0
        for lectin, weight_class in lectins.items():
            same_lectin_count = useable_lectin_count[lectin]
            specificity_count = lectin_lib[useable_lectin_mapping[lectin]].get_all_binding_motifs_count()
            tf = 1/specificity_count
            weight_final = class_weight_dict[weight_class] * (tf * idf[lectin])
            weight_final *= (duplicate_lectin_penalty_factor ** same_lectin_count) if same_lectin_count > 1 else 1
            score += lectin_score_dict[lectin] * weight_final
        output.append({"motif": motif, "lectin(s)": ", ".join(lectins),
                       "change": "down" if score < 0 else "up", "score": round(abs(score), ndigits = 2)})
    return pd.DataFrame(output)


def get_glycan_similarity(
        glycan1: str | nx.DiGraph, # IUPAC-condensed glycan sequence or NetworkX graph
        glycan2: str | nx.DiGraph, # IUPAC-condensed glycan sequence or NetworkX graph
        motifs: pd.DataFrame | None = None, # Motif dataframe (name + sequence); defaults to motif_list
        feature_set: str | list[str] = ['known', 'exhaustive', 'terminal'] # Feature types to analyze: known, graph, exhaustive, terminal(1-3), custom, chemical, size_branch
) -> float: # Cosine similarity between glycan1 and glycan2
    "Calculates cosine similarity between two glycans based on their motif count fingerprints"
    from scipy.spatial.distance import cosine
    fp = annotate_dataset([graph_to_string(glycan1), graph_to_string(glycan2)], motifs = motifs,
                          feature_set = feature_set)
    return 1 - cosine(fp.iloc[0].values, fp.iloc[1].values)
