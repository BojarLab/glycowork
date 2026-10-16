#!/usr/bin/env python
"""Regenerate glycowork/motif/glytoucan_ids.json.xz, the GlyTouCan IDs that df_glycan does not carry.

Every value is either a sequence or another GlyTouCan ID whose sequence the key takes (a df_glycan ID, or an
ID in this file that holds a sequence). Sources are GlyCosmos' bulk downloads (CC BY 4.0), the full GlyTouCan
WURCS list and the archived-to-replacement ID table:

    python bin/build_glytoucan_ids.py
    python bin/build_glytoucan_ids.py glycosmos_glycans_list.csv glycosmos_glycans_archive_mapping.csv

Each WURCS goes through canonicalize_iupac and is then sensibly specified, i.e., only uncertainty that biology
leaves no room for is filled in, so that a GlyTouCan entry with, say, '?' in the chitobiose core resolves to the
glycan we would write instead of being carried as is or not supported at all:
* N-glycan core (GlcNAc root, or its alditol or aglycone form, with GlcNAc-Man on it): GlcNAc(b1-4)GlcNAc and
  Man(b1-4); the Man arms a1-3 and a1-6 when they are identical, else both a1-3/6 (one known position fixes the
  other); a GlcNAc on the core Man is the bisecting b1-4, a Xyl b1-2; a Fuc on the root a1-3/6; Man on Man is
  alpha anywhere in the glycan; floating Man/GlcNAc parts are alpha/beta.
* Everywhere else, an unknown anomer is filled when df_glycan glycans show one anomer in >= 99% of >= 30
  occurrences of (donor, carbon, position, acceptor), else (donor, carbon, acceptor), else the donor alone; the
  most specific context with enough occurrences decides, so an ambiguous context is never overruled by a coarser
  one. A missing position is filled the same way from (donor, anomer, acceptor). Generic residues (Hex, HexNAc,
  ...) are never filled.
A result that equals a df_glycan glycan (both canonicalized) becomes an alias to its ID; IDs sharing one new
sequence alias the one whose own structure is that sequence (else the first ID). Compositions, repeating units,
and cyclic or bridged structures are left out. Aliases into df_glycan already in the file (including the
hand-curated ones from the former backup_gids.json) are kept as they are.
"""
import re
import sys
import json
import lzma
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool
import networkx as nx
import pandas as pd

LINK = re.compile(r'^([ab?])([\d?])-([\d?/]+)$')
GID = re.compile(r'G\d{5}[A-Z]{2}')
GENERIC = {'Hex', 'HexNAc', 'HexN', 'HexA', 'dHex', '6dHex', 'Pen', 'Sia', 'Hep', 'Monosaccharide'}
OUT = Path(__file__).resolve().parents[1] / 'glycowork' / 'motif' / 'glytoucan_ids.json.xz'


def decide(counts, n = 30, frac = 0.99):
    "(the dominant value if it reaches frac, else None; whether there were n occurrences to decide on)"
    total = sum(counts.values()) if counts else 0
    if total < n:
        return None, False
    value, hits = counts.most_common(1)[0]
    return (value if hits / total >= frac else None), True


def set_link(lab, node, anomer, pos):
    "Sets a linkage label where it does not contradict what is already stated"
    a, c, p = LINK.match(lab[node]).groups()
    if a in ('?', anomer) and (p == '?' or pos in p.split('/') or set(pos.split('/')) >= set(p.split('/'))):
        lab[node] = f'{anomer}{c}-{pos}' if p == '?' or '/' in p else f'{anomer}{c}-{p}'


def resolve(wurcs):
    "WURCS to (canonical, sensibly specified) or (None, None)"
    warnings.filterwarnings('ignore')
    from glycowork.motif.processing import canonicalize_iupac
    from glycowork.motif.graph import glycan_to_nxGraph, graph_to_string
    try:
        canon = canonicalize_iupac(wurcs)
        g = glycan_to_nxGraph(canon)
        lab = dict(g.nodes(data = 'string_labels'))
        links = [n for n in g if LINK.match(lab[n] or '')]
        kids = lambda m: [(L, next(iter(g.successors(L)))) for L in g.successors(m) if L in links and g.out_degree(L)]
        root = max(n for n in g if g.in_degree(n) == 0 and n not in links)
        done = set()
        core = [(L1, L2, c2) for L1, c1 in kids(root) if lab[c1] == 'GlcNAc' for L2, c2 in kids(c1) if lab[c2] == 'Man']
        if re.fullmatch(r'GlcNAc(?:-ol|1[A-Z][A-Za-z]*)?', lab[root]) and len(core) == 1:
            L1, L2, cman = core[0]
            set_link(lab, L1, 'b', '4')
            set_link(lab, L2, 'b', '4')
            done |= {L1, L2}
            mans = [(L, c) for L, c in kids(cman) if lab[c] == 'Man']
            ps = [LINK.match(lab[L]).group(3) for L, _ in mans]
            if len(mans) == 2 and all(p in ('?', '3', '6', '3/6') for p in ps):
                fixed = [p for p in ps if p in ('3', '6')]
                if len(fixed) == 1:
                    ps = [p if p in ('3', '6') else '63'[fixed[0] == '6'] for p in ps]
                elif not fixed:
                    arms = [graph_to_string(g.subgraph(nx.descendants(g, c) | {c})) for _, c in mans]
                    ps = ['3', '6'] if arms[0] == arms[1] else ['3/6', '3/6']
                for (L, _), p in zip(mans, ps):
                    set_link(lab, L, 'a', p)
                    done.add(L)
            elif len(mans) == 1 and ps[0] in ('?', '3', '6', '3/6'):
                set_link(lab, mans[0][0], 'a', '3/6' if ps[0] == '?' else ps[0])
                done.add(mans[0][0])
            for L, c in kids(cman):
                if lab[c] in ('GlcNAc', 'Xyl') and (mans or lab[c] == 'Xyl'):
                    set_link(lab, L, 'b', '4' if lab[c] == 'GlcNAc' else '2')
                    done.add(L)
            for m in nx.descendants(g, cman):
                for L, c in (kids(m) if lab[m] == 'Man' else []):
                    if lab[c] == 'Man' and L not in done and lab[L][0] == '?':
                        lab[L] = 'a' + lab[L][1:]
            for L in links:
                dn = list(g.successors(L))
                if not g.in_degree(L) and len(dn) == 1 and lab[dn[0]] in ('Man', 'GlcNAc') and lab[L][0] == '?':
                    lab[L] = ('a' if lab[dn[0]] == 'Man' else 'b') + lab[L][1:]
            for L, c in kids(root):
                if lab[c] == 'Fuc':
                    set_link(lab, L, 'a', '3/6')
                    done.add(L)
        for L in links:
            dn, ac = list(g.successors(L)), list(g.predecessors(L))
            if L in done or '?' not in lab[L] or len(dn) != 1 or lab[dn[0]] in GENERIC:
                continue
            d, a = lab[dn[0]], (lab[ac[0]] if ac else None)
            x, c, p = LINK.match(lab[L]).groups()
            if x == '?':
                for key in ([(d, c, p, a)] if p.isdigit() and a else []) + ([(d, c, a)] if a else []) + [(d,)]:
                    value, enough = decide(ANOMERS.get(key))
                    if enough:
                        x = value or '?'
                        break
            if p == '?' and x != '?' and a:
                p = decide(POSITIONS.get((d, x + c, a)))[0] or '?'
            lab[L] = f'{x}{c}-{p}'
        nx.set_node_attributes(g, lab, 'string_labels')
        sens = canonicalize_iupac(graph_to_string(g))
        g = glycan_to_nxGraph(sens)
        valid = canonicalize_iupac(sens) == sens and '|' not in sens and all(
            l and l != '?' and (not LINK.match(l) or g.out_degree(n)) for n, l in g.nodes(data = 'string_labels'))
        return (canon, sens) if valid else (None, None)
    except Exception:  # unparseable, repeating, cyclic or otherwise inexpressible structures
        return None, None


def init(anomers, positions):
    "Hands the df_glycan linkage statistics to a worker"
    global ANOMERS, POSITIONS
    ANOMERS, POSITIONS = anomers, positions


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    from glycowork.glycan_data.loader import df_glycan
    from glycowork.motif.graph import glycan_to_nxGraph
    from glycowork.motif.processing import canonicalize_iupac
    anomers, positions = defaultdict(Counter), defaultdict(Counter)
    for glycan in df_glycan.glycan.unique():
        try:
            g = glycan_to_nxGraph(glycan)
        except Exception:
            continue
        lab = dict(g.nodes(data = 'string_labels'))
        for n, l in lab.items():
            m = LINK.match(l or '')
            dn, ac = list(g.successors(n)), list(g.predecessors(n))
            if not m or len(dn) != 1 or len(ac) != 1 or m.group(1) == '?':
                continue
            (x, c, p), d, a = m.groups(), lab[dn[0]], lab[ac[0]]
            for key in [(d, c, p, a), (d, c, a), (d,)]:
                anomers[key][x] += 1
            if p.isdigit():
                positions[(d, x + c, a)][p] += 1
    src = sys.argv[1:] or ['https://download.glycosmos.org/glycosmos_glycans_list.csv', 'https://download.glycosmos.org/glycosmos_glycans_archive_mapping.csv']
    gtc = pd.read_csv(src[0], usecols = ['Accession Number', 'WURCS', 'Subsumption'], keep_default_na = False)
    gtc = gtc[~gtc.Subsumption.isin(['Monosaccharide composition with linkage', 'Base composition with linkage'])]
    with Pool(initializer = init, initargs = (anomers, positions)) as pool:
        res = pool.map(resolve, gtc.WURCS.tolist(), chunksize = 500)
    has_id = df_glycan.dropna(subset = ['glytoucan_id'])
    df_ids = set(has_id.glytoucan_id)
    canon_to_id = defaultdict(set)
    for glycan, gid in zip(has_id.glycan, has_id.glytoucan_id):
        try:
            canon_to_id[canonicalize_iupac(glycan)].add(gid)
        except Exception:
            continue
    with lzma.open(OUT, 'rt') as f:
        out = {k: v for k, v in json.load(f).items() if v in df_ids and k not in df_ids}
    todo = [(gid, c, s) for gid, (c, s) in zip(gtc['Accession Number'], res) if s and gid not in df_ids and gid not in out
            and not (s.startswith('{') and '(' not in s[s.rindex('}') + 1:])]  # floating parts on a bare residue are a composition
    rep = {}
    for gid, c, s in sorted(todo, key = lambda t: (t[1] != t[2], t[0])):  # an ID whose own structure is the sequence holds it
        rep.setdefault(s, gid)
    out |= {gid: min(canon_to_id[s]) if s in canon_to_id else (s if rep[s] == gid else rep[s]) for gid, _, s in todo}
    for old, new in pd.read_csv(src[1], keep_default_na = False).values:
        new = new.rsplit('/', 1)[-1]
        if old not in df_ids and old not in out and (new in df_ids or new in out):
            out[old] = new if new in df_ids or not GID.fullmatch(out[new]) else out[new]
    with lzma.open(OUT, 'wt', preset = 9) as f:
        f.write(json.dumps(dict(sorted(out.items())), separators = (',', ':')))
    print(f'{len(out)} IDs: {sum(not GID.fullmatch(v) for v in out.values())} sequences, {sum(v in df_ids for v in out.values())} aliases into df_glycan')
