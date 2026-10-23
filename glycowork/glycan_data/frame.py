import pandas as pd
import networkx as nx
from glycowork.glycan_data.loader import GlycoList, lib, resolve_motif_name


class NamedGroup(list):
    def __init__(self, name, items):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return f"{self.name}: {list.__repr__(self)}"


class NamedGroups(list):
    def __init__(self, labels, mapping):
        super().__init__(labels)
        self.mapping = mapping

    def __repr__(self):
        return ' | '.join(f"{name} (n={len(s)})" for name, s in self.mapping.items())


class GlycoDataFrame(pd.DataFrame):
    _metadata = ['_contrasts', '_paired', '_glyco_name', '_provenance']
    _meta_groups = [('Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom', 'Domain', 'ref'),
                    ('disease_association', 'disease_id', 'disease_sample', 'disease_direction', 'disease_ref', 'disease_species'),
                    ('tissue_sample', 'tissue_id', 'tissue_ref', 'tissue_species')]

    @property
    def _constructor(self):
        return GlycoDataFrame

    def __finalize__(self, other, method = None, **kwargs):
        super().__finalize__(other, method = method, **kwargs)
        if method == "concat":  # pandas hands __finalize__ a _Concatenator here, not an NDFrame, so its _metadata branch never runs and the result would silently lose its contrasts
            src = next((o for o in getattr(other, 'objs', ()) if isinstance(o, GlycoDataFrame)), None)
            if src is not None:
                for name in self._metadata:
                    object.__setattr__(self, name, getattr(src, name, None))
        return self

    @property
    def _glycan_col(self):
        for name in ('glycan', 'Glycan', 'glycans', 'Glycans'):
            if name in self.columns:
                return name
        return None

    @property
    def glycans(self):
        col = self._glycan_col
        if col:
            return GlycoList(list(self[col]))
        if self.index.dtype != float and any(isinstance(v, str) and '(' in v for v in self.index[:3]):
            return GlycoList(list(self.index))
        first = [v for v in self.iloc[:, 0] if isinstance(v, str)]
        cols = [c for c in self.columns if isinstance(c, str) and '(' in c]
        if len(cols) > sum(1 for v in first if '(' in v):
            return GlycoList(cols)
        return GlycoList(list(self.iloc[:, 0]))

    @property
    def abundance(self):
        col = self._glycan_col
        if col:
            return self.drop(columns = col)
        if self.index.dtype != float and any(isinstance(v, str) and '(' in v for v in self.index[:3]):
            return self
        cols = [c for c in self.columns if isinstance(c, str) and '(' in c]
        if len(cols) > sum(1 for v in self.iloc[:, 0] if isinstance(v, str) and '(' in v):
            return self[cols].T  # the transposed layout .glycans now recognizes; every other branch hands back glycans as rows
        return self.iloc[:, 1:]

    @property
    def groups(self):
        """Flat group labels per sample; .mapping gives {group_name: [samples]}"""
        if not self._contrasts:
            return NamedGroups([], {})
        mapping = {}
        for col in self.columns:
            if col in self._contrasts:
                mapping.setdefault(self._contrasts[col], []).append(col)
        labels = [self._contrasts[col] for col in self.columns if col in self._contrasts]
        return NamedGroups(labels, mapping)

    @property
    def group1(self):
        """Sample columns for the first group, with .name for group label"""
        if not self._contrasts:
            return NamedGroup('', [])
        name = next(iter(dict.fromkeys(self._contrasts.values())))
        return NamedGroup(name, [col for col in self.columns if self._contrasts.get(col) == name])

    @property
    def group2(self):
        """Sample columns for the second group, with .name for group label"""
        if not self._contrasts:
            return NamedGroup('', [])
        names = list(dict.fromkeys(self._contrasts.values()))
        if len(names) < 2:
            return NamedGroup('', [])
        name = names[1]
        return NamedGroup(name, [col for col in self.columns if self._contrasts.get(col) == name])

    @property
    def paired(self):
        return self._paired

    @property
    def name(self):
        return self._glyco_name

    @name.setter
    def name(self,
             value):  # pandas assigns .name to every group handed to groupby.apply, so a read-only property made that raise on any GlycoDataFrame, e.g., df_species
        self._glyco_name = value

    @property
    def provenance(self):
        return self._provenance

    def __init__(self, *args, **kwargs):
        contrasts = kwargs.pop('contrasts', None)
        paired = kwargs.pop('paired', None)
        name = kwargs.pop('name', None)
        self._provenance = kwargs.pop('provenance', {})
        super().__init__(*args, **kwargs)
        if contrasts is not None:
            self._contrasts = contrasts
        elif not hasattr(self, '_contrasts'):
            self._contrasts = {}
        if paired is not None:
            self._paired = paired
        elif not hasattr(self, '_paired'):
            self._paired = False
        if name is not None:
            self._glyco_name = name
        elif not hasattr(self, '_glyco_name'):
            self._glyco_name = ''

    def glyco_filter(self, motif: str | nx.DiGraph,  # Motif sequence, motif name (e.g., 'Internal_LewisX'), glyco-regular expression, or graph
                     termini_list: str | list = [],  # Monosaccharide position(s) from terminal/internal/flexible; default: the motif name's termini spec
                     min_count: int | None = 1  # Minimum number of times motif needs to be present to pass
                     ) -> 'GlycoDataFrame':  # Records whose glycans contain the motif at least min_count times
        "Keeps only the records whose glycans contain the motif, by subgraph isomorphism rather than string matching"
        from glycowork.motif.graph import subgraph_isomorphism  # Lazy import to avoid circular dependencies
        if isinstance(motif, str) and (hit := resolve_motif_name(motif)) is not None:
            motif, termini_list = hit[0], termini_list or hit[1]
        elif isinstance(motif, str) and '(' not in motif and motif not in lib:
            raise ValueError(f"'{motif}' is neither a motif name from motif_list nor a glycan sequence.")
        glycans = list(self.glycans)
        indices = [i for i, g in enumerate(glycans) if
                   isinstance(g, str) and subgraph_isomorphism(g, motif, termini_list = termini_list, count = True) >= (
                       1 if min_count is None else min_count)]
        if not self._glycan_col and glycans == [c for c in self.columns if isinstance(c, str) and '(' in c]:
            return self[[glycans[i] for i in indices]]  # a transposed frame keeps its glycans in the columns, so .glycans enumerates that axis and slicing rows would return unrelated samples
        if not self._glycan_col and self.index.dtype != float and any(isinstance(v, str) and '(' in v for v in self.index[:3]):
            return self.iloc[indices, :]  # the glycans live in the index here, so resetting it would discard the very labels that were filtered on
        return self.iloc[indices, :].reset_index(drop = True)

    def meta_filter(self, narrow: bool = True,  # Restrict positionally aligned metadata lists (e.g., Species/Family/ref) to the matching records
                    match_all: bool = False,  # Require all values of a criterion to be present, instead of any of them
                    **criteria  # column = value, list of values (OR), or callable; multiple columns are combined with AND
                    ) -> 'GlycoDataFrame':  # Matching records, with aligned metadata lists narrowed if narrow
        "Keeps only the records matching all metadata criteria, e.g., df_species.meta_filter(Order = 'Fabales', Kingdom = 'Plantae')"
        norm = lambda v: v.strip().lower().replace(' ', '_') if isinstance(v, str) else v
        specs = {}
        for col, want in criteria.items():
            if col not in self.columns:
                raise KeyError(f"'{col}' is not a column of this DataFrame")
            if callable(want):
                specs[col] = (want, None)
            else:
                wanted = {norm(v) for v in (want if isinstance(want, (list, tuple, set)) else [want])}
                specs[col] = (lambda v, wanted = wanted: norm(v) in wanted, wanted)
        groups = [[c for c in g if c in self.columns] for g in self._meta_groups if
                  any(c in specs for c in g)] if narrow else []
        data = {c: self[c].tolist() for c in set(specs) | {c for g in groups for c in g}}
        keep, narrowed = [], []
        for i in range(len(self)):
            masks, ok = {}, True
            for col, (test, wanted) in specs.items():
                val = data[col][i]
                vals = val if isinstance(val, list) else ([] if val is None or val != val else [val])
                mask = [bool(test(v)) for v in vals]
                if not any(mask) or (
                        match_all and wanted is not None and len({norm(v) for v, m in zip(vals, mask) if m}) < len(
                        wanted)):
                    ok = False
                    break
                masks[col] = mask
            if not ok:
                continue
            sub = {}
            for group in groups:
                hits = [c for c in group if c in masks and isinstance(data[c][i], list)]
                if not hits:
                    continue
                n = len(data[hits[0]][i])
                idxs = [j for j in range(n) if all(masks[c][j] for c in hits)]
                if not idxs:
                    # criteria on one aligned group matched different records, so no single record satisfies all of them
                    ok = False
                    break
                if len(idxs) < n:
                    for c in group:
                        if isinstance(data[c][i], list) and len(data[c][i]) == n:
                            sub[c] = [data[c][i][j] for j in idxs]
            if not ok:
                continue
            keep.append(i)
            narrowed.append(sub)
        out = self.iloc[keep, :].reset_index(drop = True)
        for col in {c for sub in narrowed for c in sub}:
            out[col] = [sub.get(col, v) for sub, v in zip(narrowed, out[col])]
        return out

    def meta_values(self, column: str,  # Column whose (list or scalar) entries should be tallied
                    top: int | None = None  # Only return the n most frequent values
                    ) -> pd.Series:  # Value counts over all records, list entries counted element-wise, most frequent first
        "Tallies the values of a metadata column whose cells are lists or scalars, e.g., df_species.meta_values('Species', top = 10)"
        vals = [v.strip() if isinstance(v, str) else v for val in self[column] for v in
                (val if isinstance(val, list) else ([] if val is None or val != val else [val]))]
        return pd.Series(vals, dtype = object).value_counts().head(top)
