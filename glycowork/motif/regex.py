import re
import warnings
import pandas as pd
import networkx as nx
from itertools import chain
from glycowork.glycan_data.loader import lib, unwrap
from glycowork.motif.processing import canonicalize_iupac, min_process_glycans
from glycowork.motif.graph import graph_to_string, subgraph_isomorphism, glycan_to_nxGraph, LINKAGE_LABEL

PREPROCESS_SPLIT = re.compile(r'(-?\s*(?:\((?:\?<=|\?<!)[^()]*\))?\s*\(?\[.*?\]\)?\s*(?:\{,?\d*,?\d*\}\?|\{,?\d*,?\d*\}|\*\?|\+\?|\?|\*|\+)\s*(?:\((?:\?=|\?!)[^()]*\))?\s*-?)')
LINKAGE_SHORTHAND = re.compile(r'[\d\?]\(|\d$')
LINKAGE_EXPAND = re.compile(r'([ab\?])(\d+/\d+|\d|\?)\(\?1-\?\)')
SIA_LINKAGE_FIX = re.compile(r'(5Ac|5Gc|Kdn|Sia)\([a\?]1')
CONTRACT_LINKAGE = re.compile(r'\((\w)1-(\d+/\d+|\d+|\?)\)')
COMPONENT_DASH = re.compile(r'(?<![DL])-(?![^(]*\))')
LOOKAROUND = re.compile(r'\((\?<=|\?<!|\?=|\?!)([^()]*)\)')
QUANTIFIER = re.compile(r'\{([^}]*)\}')
ALTERNATIVES = re.compile(r'\[([^\]]*)\]')


def preprocess_pattern(pattern: str # Glyco-regular expression like "Hex-HexNAc-([Hex|Fuc]){1,2}-HexNAc"
                       ) -> list[str]: # List of pattern chunks
    "Transform glyco-regular expression into chunks, rejecting malformed ones"
    if pattern.startswith('r'):
        pattern = pattern[1:]
    if pattern.count('[') != pattern.count(']') or pattern.count('(') != pattern.count(')'):
        raise ValueError(f"Unbalanced brackets in glyco-regular expression '{pattern}'")
    for body in QUANTIFIER.findall(pattern):
        if not re.fullmatch(r'\d+|\d*,\d*', body) or body in ('', ','):
            raise ValueError(f"'{{{body}}}' is not a valid occurrence range in glyco-regular expression '{pattern}'")
        edges = body.split(',')
        if len(edges) == 2 and edges[0] and edges[1] and int(edges[0]) > int(edges[1]):
            raise ValueError(f"Minimum occurrence exceeds maximum in '{{{body}}}' of glyco-regular expression '{pattern}'")
    if re.search(r'(?<![\]\)])(?:\{[^}]*\}|\*|\+)', LOOKAROUND.sub('', pattern)):
        raise ValueError(
            f"A quantifier in glyco-regular expression '{pattern}' is not attached to a bracketed group; write it as, e.g., '[Hex]{{1,2}}' rather than 'Hex{{1,2}}'")
    for body in ALTERNATIVES.findall(pattern):
        if not body or any(not alt for alt in body.split('|')):
            raise ValueError(f"'[{body}]' has an empty alternative in glyco-regular expression '{pattern}'")
    if '(?' in pattern and not LOOKAROUND.search(pattern):
        raise ValueError(f"Malformed lookahead/lookbehind in glyco-regular expression '{pattern}'")
    if any(not look.group(2).strip('-').strip() for look in LOOKAROUND.finditer(pattern)):
        raise ValueError(f"Empty lookahead/lookbehind in glyco-regular expression '{pattern}'")
    # Use regular expression to identify the conditional parts and keep other chunks together
    pattern = pattern.replace('.', 'Monosaccharide')
    components = PREPROCESS_SPLIT.split(pattern)
    # Remove any empty strings and trim whitespace
    components = [x.strip('-').strip() for x in components if x]
    unknown = {t for c in components for m in compile_component(c)['motifs'] + [k for _, ms in compile_component(c)['looks'] for k in ms]
               for tok in min_process_glycans([m.replace('!', '')])[0]
               for t in ([tok] if '-' in tok else tok.split('/')) if t not in lib}
    if unknown:
        warnings.warn(f"{sorted(unknown)} in glyco-regular expression '{pattern}' are not known glycoletters; "
                      f"such a chunk can only match a glycan that spells them the same way")
    return components


def specify_linkages(pattern_component: str # Chunk of glyco-regular expression
                     ) -> str: # Specified component with linkages
    "Convert expression linkages from shorthand to full notation, such as Mana6 to Man(a1-6) or Galb3/4 to Gal(b1-3/4)"
    if LINKAGE_SHORTHAND.search(pattern_component):
        pattern_component = LINKAGE_EXPAND.sub(lambda m: f'({m.group(1)}1-{m.group(2)})', pattern_component)
    return SIA_LINKAGE_FIX.sub(lambda m: m.group(0)[:-1] + '2', pattern_component)


def replace_patterns(s: str # String to process
                     ) -> str: # Processed string
    "Replace pattern strings with standardized forms"
    return SIA_LINKAGE_FIX.sub(lambda m: m.group(0)[:-1] + '2', COMPONENT_DASH.sub('(?1-?)', s))


def convert_pattern_component(pattern_component: str # Chunk of glyco-regular expression
                              ) -> str | dict[str, list[int]]: # String or dict mapping to occurrences
    "Process pattern component into string or occurrence dictionary"
    if not any(k in pattern_component for k in ('[', '{', '*', '+', '=', '<!', '?!')):
        if pattern_component[-1].isdigit() or pattern_component.endswith('?'):
            pattern_component += '-'
        return specify_linkages(replace_patterns(pattern_component))
    pattern, occurrence = None, None
    if '[' in pattern_component:
        prefix = pattern_component.split('[')[0].rstrip('-')
        alternatives = pattern_component.split('[')[1].split(']')[0].split('|')
        after_bracket = pattern_component.split(']', 1)[1] if ']' in pattern_component else ''
        if not any(q in after_bracket for q in ('{', '*', '+', '?', ')')) and (prefix or after_bracket.lstrip('-')):
            suffix = after_bracket.lstrip('-')
            if all('-' not in a and '(' not in a for a in alternatives):
                return specify_linkages(replace_patterns('-'.join(filter(None, [prefix, '/'.join(alternatives), suffix]))))
            pattern = ['-'.join(filter(None, [prefix, alt, suffix])) for alt in alternatives]
        else:
            pattern = alternatives
        pattern = [replace_patterns(p + '-' if p and (p[-1].isdigit() or p[-1] == '?') else p) for p in pattern]
    if '{' in pattern_component:
        occ = pattern_component.split('{')[1].split('}')[0].split(',')
        occurrence = [int(occ[0]), int(occ[0])] if len(occ) == 1 else [int(occ[0]) if occ[0] else 0, int(occ[1]) if occ[1] else 5]
        if '?' in pattern_component:
            occurrence = [occurrence[0], occurrence[0]]
    elif '*' in pattern_component or '+' in pattern_component:
        occurrence = list(range(8)) if '*' in pattern_component else list(range(1, 8))
        if '?' in pattern_component:
            occurrence = [occurrence[0], occurrence[0]]
    elif '?' in pattern_component:
        occurrence = [0, 1]
        part = pattern if pattern else pattern_component
        pattern = [replace_patterns(part)] if isinstance(part, str) else list(part)
    if pattern is None:
        # A quantifier or an optional marker can also sit on a bare motif, which still names exactly one motif; without the list the string below would be iterated character by character
        core = QUANTIFIER.sub('', pattern_component).rstrip('*+?')
        pattern = [replace_patterns(core + '-' if core and (core[-1].isdigit() or core[-1] == '?') else core)]
    if occurrence is None:
        occurrence = [1, 1]
    return {specify_linkages(p): occurrence for p in pattern}


def compile_component(pattern_component: str # Chunk of glyco-regular expression
                      ) -> dict: # Motifs, occurrence bounds, branch/location/lookaround metadata
    "Split a pattern chunk into its motifs and its modifiers"
    looks, core, cut = [], pattern_component, 0
    for look in LOOKAROUND.finditer(pattern_component):
        core = core[:look.start() - cut] + core[look.end() - cut:]
        cut += look.end() - look.start()
        look_core = look.group(2).strip('-').strip().replace('[^', '[!')
        look_motifs = convert_pattern_component(look_core) if look_core else []
        looks.append((look.group(1), [look_motifs] if isinstance(look_motifs, str) else list(look_motifs)))
    core = core.replace('[^', '[!').strip('-').strip()
    location = {k for s, k in (('^', 'start'), ('$', 'end'), ('%', 'internal')) if s in core} or None
    core = core.translate({ord(c): None for c in '^$%'})
    min_occur, max_occur = parse_pattern(core) if core else (0, 0)
    motifs = convert_pattern_component(core) if core else []
    motifs = [motifs] if isinstance(motifs, str) else list(motifs)
    absent = len(motifs) == 1 and motifs[0].startswith('!') and motifs[0].endswith(')') and motifs[0].count('(') == 1
    return {'motifs': [motifs[0][1:]] if absent else motifs, 'min': min_occur, 'max': max_occur,
            'lazy': any(k in core for k in ('}?', '*?', '+?')), 'absent': absent,
            'branch': core.startswith('('), 'location': location, 'looks': looks}


def filter_matches_by_location(matches: list[list[int]], # List of node index lists
                               ggraph: nx.DiGraph, # Glycan graph
                               match_location: str | set[str] | None # Location(s) to match: start/end/internal
                               ) -> list[list[int]]: # Filtered matches
    "Filter matches by location requirement"
    if matches and matches[0] and isinstance(matches[0][0], list):
        matches = unwrap(matches)
    if not match_location:
        return matches
    if isinstance(match_location, str):
        match_location = {match_location}
    # glycan_to_nxGraph numbers the main chain first and every floating bit above it, so the reducing end is the top of the component holding node 0, not the highest node id
    root = max(min(nx.weakly_connected_components(ggraph), key = min)) if len(ggraph) else -1
    if 'start' in match_location:
        matches = [m for m in matches if m and ggraph.out_degree[m[0]] == 0]
    if 'end' in match_location:
        matches = [m for m in matches if root in m]
    if 'internal' in match_location:
        matches = [m for m in matches if m and ggraph.out_degree[m[0]] > 0 and root not in m]
    return matches


def trace_matches(components: list[dict], # Compiled pattern chunks
                  ggraph: nx.DiGraph # Glycan graph
                  ) -> list[list[int]]: # Node lists of complete matches
    "Walk the glycan graph to chain pattern chunks into complete matches"
    parent = {c: p for p in ggraph.nodes() for c in ggraph.successors(p)}
    monos = set()
    for comp in nx.weakly_connected_components(ggraph):
        root = max(comp)
        # A motif can end in a dangling linkage, which puts a linkage rather than a monosaccharide at the root and flips the alternation
        offset = bool(LINKAGE_LABEL.match(ggraph.nodes[root].get('string_labels', '')))
        monos.update(n for n, d in nx.single_source_shortest_path_length(ggraph, root).items() if d % 2 == offset)

    def attach(nodes):  # monosaccharide directly rootward of a matched chunk, -1 if the chunk reaches the reducing end
        top = max(nodes)
        if top in monos:
            top = parent.get(top, -1)
            if top < 0:
                return -1
        return parent.get(top, -1)

    def candidates(motifs, location):
        out = []
        for m in motifs:
            if m.startswith('!') and '(' not in m:  # a lone negated residue has no positive motif graph to search for
                _, hits = subgraph_isomorphism(ggraph, glycan_to_nxGraph('Monosaccharide'), count = True, return_matches = True)
                _, excluded = subgraph_isomorphism(ggraph, glycan_to_nxGraph(m[1:]), count = True, return_matches = True)
                banned = {e[0] for e in excluded}
                hits = [h for h in hits if h[0] not in banned]
            else:
                g2 = glycan_to_nxGraph(m)
                if not len(g2):
                    continue
                _, hits = subgraph_isomorphism(ggraph, g2, count = True, return_matches = True)
            out.extend(h for h in hits if h)
        return sorted(filter_matches_by_location(out, ggraph, location), key = lambda m: (m[0], m[-1]))

    # Refute the chunk that is hardest to satisfy first, so a glycan without it costs one subgraph search instead of all of them
    for c in sorted(components, key = lambda c: -sum(len(m) for m in c['motifs'])):
        c['candidates'] = candidates(c['motifs'], c['location'])
        if not c['candidates'] and c['min'] and not c['absent']:
            return []
        c['look_candidates'] = [candidates(ms, None) for _, ms in c['looks']]
        if any(not hits and not kind.endswith('!') for (kind, _), hits in zip(c['looks'], c['look_candidates'])):
            return []

    def look_ok(c, segs):
        if not c['looks'] or not segs:
            return True
        for (kind, _), hits in zip(c['looks'], c['look_candidates']):
            if kind.startswith('?<'):
                found = any(attach(m) == min(segs[0]) for m in hits)
            else:
                spot = attach(segs[-1])
                found = spot >= 0 and any(min(m) == spot for m in hits)
            if found == kind.endswith('!'):
                return False
        return True

    def occurrences(c, n, anchor, used):
        if c['absent']:  # a negated chunk with a spelled-out linkage asserts absence instead of consuming residues
            if not any(attach(m) == anchor and not used.intersection(m) for m in c['candidates']):
                yield [], anchor
            return
        if n == 0:
            yield [], anchor
            return
        if c['branch'] and anchor is not None:
            pool = [m for m in c['candidates'] if attach(m) == anchor and not used.intersection(m)]

            def pick(k, start, taken):
                if k == 0:
                    yield []
                    return
                for i in range(start, len(pool)):
                    if not taken.intersection(pool[i]):
                        for rest in pick(k - 1, i + 1, taken.union(pool[i])):
                            yield [pool[i]] + rest

            for combo in pick(n, 0, used):
                yield combo, anchor
        else:

            def extend(k, cur, taken):
                if k == 0:
                    yield [], cur
                    return
                for m in c['candidates']:
                    if (cur is None or min(m) == cur) and not taken.intersection(m):
                        for rest, final in extend(k - 1, attach(m), taken.union(m)):
                            yield [m] + rest, final

            yield from extend(n, anchor, used)

    def place(i, anchor, used):
        if i == len(components):
            yield []
            return
        c = components[i]
        counts = (1,) if c['absent'] else range(c['min'], c['max'] + 1) if c['lazy'] else range(c['max'], c['min'] - 1, -1)
        for n in counts:
            for segs, new_anchor in occurrences(c, n, anchor, used):
                if not look_ok(c, segs):
                    continue
                for rest in place(i + 1, new_anchor, used.union(chain.from_iterable(segs))):
                    yield segs + rest

    traces, seen = [], set()
    for segs in place(0, None, frozenset()):
        nodes = set(chain.from_iterable(segs))
        if not nodes:
            continue
        for n in list(nodes):  # re-insert the unconstrained linkages that bridge two chunks
            up = parent.get(n)
            if up is not None and up not in nodes and parent.get(up) in nodes:
                nodes.add(up)
        nodes = {n for n in nodes if n in monos or parent.get(n) in nodes}  # drop a chunk-trailing linkage
        if not nodes or min(nodes) in seen:
            continue
        seen.add(min(nodes))
        traces.append(nodes)
    return [sorted(t) for t in traces if not any(t < o for o in traces)]


def parse_pattern(pattern: str # Pattern component from glyco-regular motif
                  ) -> tuple[int, int]: # (Minimum occurrence, Maximum occurrence)
    "Extract occurrence limits from glyco-regular motif pattern"
    if '{' in pattern:
        temp = pattern.split('{')[1].split('}')[0].split(',')
        min_occur = int(temp[0]) if temp[0] else 0
        max_occur = int(temp[-1]) if temp[-1] else 8
        if len(temp) == 1:
            max_occur = min_occur
    elif '*' in pattern:
        min_occur, max_occur = 0, 8
    elif '+' in pattern:
        min_occur, max_occur = 1, 8
    elif '?' in re.sub(r'\([^)]*\)|[ab]?\?-|[ab]\?', '', pattern) and '=' not in pattern and '!' not in pattern:
        min_occur, max_occur = 0, 1
    else:
        min_occur, max_occur = 1, 1
    if any(k in pattern for k in ('.?', '}?', '*?', '+?')):
        max_occur = min_occur
    return min_occur, max_occur


def format_retrieved_matches(lists: list[list[int]], # List of traces
                             ggraph: nx.DiGraph # Glycan graph
                             ) -> list[str]: # Matching glycan strings
    "Convert traces into glycan strings"
    out = [(graph_to_string(ggraph.subgraph(t)), t) for t in lists if nx.is_weakly_connected(ggraph.subgraph(t))]
    return [s for s, _ in sorted(out, key = lambda x: (-len(x[0]), min(x[1])))]


def compile_pattern(pattern: str # Glyco-regular expression, e.g., "Hex-HexNAc-([Hex|Fuc]){1,2}-HexNAc"
                    ) -> list[str]: # Pre-compiled pattern chunks
    "Pre-compile glyco-regular expression for faster processing"
    return preprocess_pattern(pattern)


def get_match(pattern: str | list[str], # Expression or pre-compiled pattern; e.g., "Hex-HexNAc-([Hex|Fuc]){1,2}-HexNAc"
              glycan: str | nx.DiGraph, # Glycan string or graph
              return_matches: bool = True # Whether to return matches vs boolean
              ) -> bool | list[str]: # Match results
    "Find matches for glyco-regular expression in glycan"
    if isinstance(glycan, str):
        if any(k in glycan for k in (';', '-D-', 'RES', '=', 'α', 'β')):
            glycan = canonicalize_iupac(glycan)
        ggraph = glycan_to_nxGraph(glycan)
    elif isinstance(glycan, nx.Graph):
        ggraph = glycan
    else:
        raise ValueError(
            f"get_match expects one glycan as a string or graph, got {type(glycan).__name__}; for several glycans at once use get_match_batch.")
    pattern_components = preprocess_pattern(pattern) if isinstance(pattern, str) else pattern
    if not pattern_components or not len(ggraph):
        return False if not return_matches else []
    traces = trace_matches([compile_component(p) for p in pattern_components], ggraph)
    if not traces:
        return False if not return_matches else []
    return True if not return_matches else format_retrieved_matches(traces, ggraph)


def explain_match(pattern: str, # Glyco-regular expression, e.g., "Hex-HexNAc-([Hex|Fuc]){1,2}-HexNAc"
                  glycan: str | nx.DiGraph # Glycan string or graph
                  ) -> pd.DataFrame: # One row per pattern chunk, with what it compiled to and how often it occurs
    "Show what each chunk of a glyco-regular expression means and where it does or does not occur in a glycan"
    ggraph = glycan_to_nxGraph(glycan) if isinstance(glycan, str) else glycan
    rows = []
    for chunk in preprocess_pattern(pattern):
        c = compile_component(chunk)
        hits = trace_matches([c], ggraph) if c['motifs'] and not c['absent'] else []
        rows.append({'chunk': chunk, 'matches': ' | '.join(c['motifs']), 'occurrences': f"{c['min']}-{c['max']}",
                     'branch': c['branch'], 'position': ','.join(sorted(c['location'])) if c['location'] else '',
                     'lookaround': ' '.join(k + ':' + ','.join(ms) for k, ms in c['looks']),
                     'hits_in_glycan': 'asserts absence' if c['absent'] else len(hits)})
    return pd.DataFrame(rows)


def get_match_batch(pattern: str | list[str], # Expression or pre-compiled pattern; e.g., "Hex-HexNAc-([Hex|Fuc]){1,2}-HexNAc"
                    glycan_list: list[str | nx.DiGraph], # List of glycans
                    return_matches: bool = True # Whether to return matches vs boolean
                    ) -> list[bool] | list[list[str]]: # Match results for each glycan
    "Find glyco-regular expression matches in list of glycans"
    pattern = compile_pattern(pattern) if isinstance(pattern, str) else pattern
    out = []
    for g in glycan_list:
        try:
            out.append(get_match(pattern, g, return_matches = return_matches))
        except Exception as e:
            raise ValueError(f"get_match failed on glycan '{g}': {e}") from e
    return out


def reformat_glycan_string(glycan: str # Glycan in IUPAC-condensed
                           ) -> str: # Reformatted pattern string
    "Convert glycan string to pattern format"
    # Contract linkages off the anomeric carbon; anything else has to stay spelled out to survive the round trip
    glycan = CONTRACT_LINKAGE.sub(r'\1\2-', glycan)
    # Format branches
    return glycan.replace("[", "([").replace("-]", "]){1}-").strip('-')


def motif_to_regex(motif: str # Glycan in IUPAC-condensed
                   ) -> str: # Regular expression
    "Convert glycan motif to regular expression pattern"
    motif = canonicalize_iupac(motif) if '!' not in motif else motif
    ggraph = glycan_to_nxGraph(motif)
    parent = {c: p for p in ggraph.nodes() for c in ggraph.successors(p)}
    root = max(ggraph.nodes())
    # A motif can end in a dangling linkage, which puts a linkage rather than a monosaccharide at the root and flips the alternation
    offset = bool(LINKAGE_LABEL.match(ggraph.nodes[root]['string_labels']))
    monos = {n for n, d in nx.single_source_shortest_path_length(ggraph, root).items() if d % 2 == offset}
    depth = {}
    for n in sorted(ggraph.nodes()):
        depth[n] = 1 + max([depth[c] for c in ggraph.successors(n)], default = -1)
    # Main chain: deepest path from the root, everything hanging off it becomes a branch group
    main, node = [], max(ggraph.nodes())
    while True:
        main.append(node)
        kids = [c for c in ggraph.successors(node)]
        if not kids:
            break
        node = max(kids, key = lambda c: (depth[c], -c))
    main = [n for n in main if n in monos]
    on_main, tokens = set(main), []
    for m in reversed(main):
        for link in sorted(ggraph.successors(m)):
            kid = next(iter(ggraph.successors(link)), None)
            if kid is None or kid in on_main:
                continue
            sub = {link} | nx.descendants(ggraph, link)
            if any(sum(1 for _ in ggraph.successors(n)) > 1 for n in sub):
                raise ValueError(
                    f"Motif '{motif}' has a branch that itself branches, which glyco-regular expressions cannot express.")
            tokens.append('([' + reformat_glycan_string(graph_to_string(ggraph.subgraph(sub))) + ']){1}-')
        tokens.append(CONTRACT_LINKAGE.sub(r'\1\2-', graph_to_string(ggraph.subgraph({m, parent[m]})))
                      if m in parent else ggraph.nodes[m]['string_labels'])
    pattern = ''.join(tokens).strip('-')
    control = motif
    if '!' in motif:  # the motif itself violates its own absence constraints, so validate on the positive skeleton
        negated = {n for n, d in ggraph.nodes(data = True) if d['string_labels'].startswith('!')}
        negated |= {parent[n] for n in negated if n in parent}
        control = graph_to_string(ggraph.subgraph(set(ggraph.nodes()) - negated))
    if not get_match(pattern, control, return_matches = False):
        raise ValueError(
            f"Could not build a working glyco-regular expression for motif '{motif}' (generated pattern: '{pattern}').")
    return pattern
