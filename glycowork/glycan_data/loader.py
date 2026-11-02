from __future__ import annotations  # annotations stay unevaluated, so a pd.DataFrame in a signature does not import pandas
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import re
import csv
import ast
import gc
import json
import lzma
import pickle
import networkx as nx
from pathlib import Path
from itertools import chain
from importlib import import_module, resources
from typing import Any


class LazyModule:
    "A module imported on its first attribute access: glycowork reaches pandas through one, since importing glycowork or drawing a glycan never needs it, and it adds 0.6 s and 31 MB"
    def __init__(self, name: str # module to import when first used, e.g., 'pandas'
                 ):
        self.__name = name

    def __getattr__(self, attr):
        module = import_module(self.__name)
        self.__dict__.update(vars(module))  # later lookups find the module's attributes directly and no longer come here
        return getattr(module, attr)


pd = LazyModule('pandas')
_MOTIF_IDX, _MOTIF_NORM_IDX, _MOTIFS = {}, {}, []
# Get the directory and filename of the current script
this_dir = Path(__file__).parent
this_filename = Path(__file__).name

# Construct the path to the data file and load it
data_path = this_dir / 'v12_lib.pkl'
with open(data_path, 'rb') as f:
    lib = pickle.load(f)


class GlycoList(list):

    def _compare(self, a, b):
        if a is None or b is None or (isinstance(a, float) and a != a) or (isinstance(b, float) and b != b):
            return False
        from glycowork.motif.graph import compare_glycans
        return compare_glycans(a, b)

    def index(self, value, start = 0, stop = None):
        if stop is None:
            stop = len(self)
        for i in range(start, stop):
            if self._compare(self[i], value):
                return i
        raise ValueError(f"{value} is not in list")

    def __contains__(self, value):
        if isinstance(value, str) and list.__contains__(self, value):
            return True
        return any(self._compare(item, value) for item in self)

    def count(self, value):
        return sum(1 for item in self if self._compare(item, value))

    def remove(self, value):
        for i, item in enumerate(self):
            if self._compare(item, value):
                del self[i]
                return
        raise ValueError("list.remove(x): x not in list")


def __getattr__(name):
    if name in ("GlycoDataFrame", "NamedGroup", "NamedGroups"):  # their own module, frame.py, since subclassing pd.DataFrame imports pandas
        from glycowork.glycan_data import frame
        return getattr(frame, name)
    elif name == "motif_list":
        with resources.files("glycowork.glycan_data").joinpath("glycan_motifs.csv").open(encoding = 'utf-8-sig') as f:
            motif_list = pd.read_csv(f)
        globals()[name] = motif_list  # Cache it to avoid reloading
        return motif_list
    elif name == "glycan_binding":
        with resources.files("glycowork.glycan_data").joinpath("v12_glycan_binding.csv").open(encoding = 'utf-8-sig') as f:
            glycan_binding = pd.read_csv(f)
        globals()[name] = glycan_binding  # Cache it to avoid reloading
        return glycan_binding
    elif name == "df_species":
        from glycowork.glycan_data.frame import GlycoDataFrame
        with resources.files("glycowork.glycan_data").joinpath("v12_df_species.csv").open(encoding = 'utf-8-sig') as f:
            df_species = GlycoDataFrame(pd.read_csv(f))
        globals()[name] = df_species  # Cache it to avoid reloading
        return df_species
    elif name == "df_glycan":
        from glycowork.glycan_data.frame import GlycoDataFrame
        data_path = this_dir / 'v12_sugarbase.json.xz'
        df_glycan = GlycoDataFrame(serializer.deserialize(data_path))
        globals()[name] = df_glycan  # Cache it to avoid reloading
        return df_glycan
    elif name == "lectin_specificity":
        data_path = this_dir / 'lectin_specificity.json'
        lectin_specificity = serializer.deserialize(data_path)
        globals()[name] = lectin_specificity  # Cache it to avoid reloading
        return lectin_specificity
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class LazyLoader:
    def __init__(self, package, directory, prefix = 'glycomics_'):
        self.package = package
        self.directory = directory
        self.prefix = prefix
        self._datasets = {}
        self._contrasts_map = None

    def _load_contrasts(self):
        if self._contrasts_map is None:
            try:
                with resources.files(f"{self.package}.{self.directory}").joinpath("contrasts.csv").open(
                        encoding = 'utf-8-sig') as f:
                    ct = pd.read_csv(f)
                    self._contrasts_map = {}
                    self._paired_map = {}
                    for _, row in ct.iterrows():
                        sample_map = self._contrasts_map.setdefault(row['dataset'], {})
                        for sample in row['samples'].split(','):
                            if sample_map.get(sample, row['group']) != row['group']:  # one group per sample, otherwise the last row would silently win
                                raise ValueError(f"{row['dataset']}: sample '{sample}' is in both '{sample_map[sample]}' and '{row['group']}' in contrasts.csv; give combined perturbations their own group")
                            sample_map[sample] = row['group']
                        self._paired_map[row['dataset']] = str(row['paired']).strip().lower() == 'true'
            except FileNotFoundError:
                self._contrasts_map = {}
                self._paired_map = {}
            try:
                with resources.files(f"{self.package}.{self.directory}").joinpath("datasets_metadata.csv").open(
                        encoding = 'utf-8-sig') as f:
                    self._provenance_map = pd.read_csv(f).set_index('dataset').to_dict(orient = 'index')
            except FileNotFoundError:
                self._provenance_map = {}
        return

    def __getattr__(self, name):
        if name.startswith('_'):  # a private lookup on a not-yet-populated instance (unpickling, copy) must not re-enter this method through self._datasets
            raise AttributeError(name)
        if name not in self._datasets:
            folder = resources.files(f"{self.package}.{self.directory}")
            from glycowork.glycan_data.frame import GlycoDataFrame
            filename = next((f"{self.prefix}{name}{ext}" for ext in ('.csv', '.csv.xz') if folder.joinpath(f"{self.prefix}{name}{ext}").is_file()), f"{self.prefix}{name}.csv")  # datasets above 1 MB ship as xz-compressed CSV
            try:
                with folder.joinpath(filename).open('rb') as f:
                    _df = pd.read_csv(f, encoding = 'utf-8-sig', compression = 'xz' if filename.endswith('.xz') else None)
                    cols = list(_df.columns)
                    cleaned = [re.sub(r'\.\d+$', '', c) for c in cols]
                    _df.columns = [cleaned[i] if cleaned[i] in cleaned[:i] + cleaned[i + 1:] else cols[i] for i in range(len(cols))]
                    self._load_contrasts()
                    dataset_key = f"{self.prefix}{name}"
                    contrasts = self._contrasts_map.get(dataset_key, {})
                    paired = self._paired_map.get(dataset_key, False)
                    # contrasts.csv keys datasets with the loader prefix, datasets_metadata.csv without it, so accept either spelling
                    self._datasets[name] = GlycoDataFrame(_df, contrasts = contrasts, paired = paired, name = name,
                                                          provenance = self._provenance_map.get(
                                                              dataset_key) or self._provenance_map.get(name, {}))
            except FileNotFoundError:
                raise AttributeError(
                    f"No dataset named {name} available under {self.directory} with prefix {self.prefix}")
        return self._datasets[name]

    def __dir__(self):
        files = resources.files(f"{self.package}.{self.directory}").iterdir()
        dataset_names = [re.sub(r'\.csv(\.xz)?$', '', file.name[len(self.prefix):], flags = re.I) for file in files if
                         file.name.startswith(self.prefix) and re.search(r'\.csv(\.xz)?$', file.name, flags = re.I)]
        return dataset_names

    def filter(self, **criteria  # column = value, list of values (OR), or callable; multiple columns are combined with AND
               ) -> list[str]:  # names of datasets whose metadata matches all criteria
        "Select datasets by their metadata, e.g., glycomics_data_loader.filter(glycan_class = 'O', source_type = ['primary tissue', 'body fluid'])"
        self._load_contrasts()
        norm = lambda v: v.strip().lower().replace(' ', '_') if isinstance(v, str) else v
        available, tests = set(self.__dir__()), {}
        for col, want in criteria.items():
            if not any(col in prov for prov in self._provenance_map.values()):
                raise KeyError(
                    f"'{col}' is not a metadata column of {self.directory}; available columns are {sorted({c for prov in self._provenance_map.values() for c in prov})}")
            tests[col] = want if callable(want) else (
                lambda v, w = {norm(x) for x in (want if isinstance(want, (list, tuple, set)) else [want])}: norm(
                    v) in w)
        # datasets_metadata.csv keys datasets without the loader prefix, contrasts.csv with it, so accept either spelling
        out = [n[len(self.prefix):] if n.startswith(self.prefix) else n for n in self._provenance_map]
        return sorted(n for n, prov in zip(out, self._provenance_map.values()) if
                      n in available and all(test(prov.get(col)) for col, test in tests.items()))


glycomics_data_loader = LazyLoader("glycowork", "glycan_data.datasets")
lectin_array_data_loader = LazyLoader("glycowork", "glycan_data.datasets", prefix = 'lectin_array_')
glycoproteomics_data_loader = LazyLoader("glycowork", "glycan_data.datasets", prefix = 'glycoproteomics_')


linkages = {
    '1-4', '1-6', 'a1-1', 'a1-2', 'a1-3', 'a1-4', 'a1-5', 'a1-6', 'a1-7', 'a1-8', 'a1-9', 'a1-11', 'a1-?', 'a2-1', 'a2-2', 'a2-3', 'a2-4', 'a2-5', 'a2-6', 'a2-7', 'a2-8', 'a2-9',
    'a2-11', 'a2-?', 'b1-1', 'b1-2', 'b1-3', 'b1-4', 'b1-5', 'b1-6', 'b1-7', 'b1-8', 'b1-9', 'b1-?', 'b2-1', 'b2-2', 'b2-3', 'b2-4', 'b2-5', 'b2-6', 'b2-7', 'b2-8', '?1-?',
    '?2-?', '?1-2', '?1-3', '?1-4', '?1-6', '?2-3', '?2-6', '?2-8'
}
Hex = {'Glc', 'Gal', 'Man', 'Ins', 'Galf', 'Hex'}
HexOS = {f"{h}OS" for h in Hex}
dHex = {'Fuc', 'Qui', 'Rha', 'dHex'}
HexA = {'GlcA', 'ManA', 'GalA', 'IdoA', 'HexA'}
HexN = {'GlcN', 'ManN', 'GalN', 'HexN'}
HexNAc = {'GlcNAc', 'GalNAc', 'ManNAc', 'HexNAc'}
HexNAcOS = {f"{h}OS" for h in HexNAc}
Pen = {'Ara', 'Xyl', 'Rib', 'Lyx', 'Pen'}
Sia = {'Neu5Ac', 'Neu5Gc', 'Kdn', 'Sia'}
modification_map = {'6S': {'GlcNAc', 'Gal'}, '3S': {'Gal'}, '4S': {'GalNAc'},
                    'OS': {'GlcNAc', 'Gal', 'GalNAc'}}


def resolve_motif_name(name: str # candidate motif_list name, e.g., 'Internal_LewisX', 'lewis x', or 'high_mannose'
                       ) -> tuple[str, list] | None: # (motif sequence, termini spec), or None if name is not a known motif
    "Maps a motif_list name to its sequence and termini spec, tolerating case/underscore/space/hyphen differences"
    if not _MOTIF_IDX:  # built on first use, so importing the package does not pay for it
        with resources.files("glycowork.glycan_data").joinpath("glycan_motifs.csv").open(encoding = 'utf-8-sig') as f:
            _MOTIFS.extend((r['motif_name'], r['motif'], r['termini_spec']) for r in csv.DictReader(f))
        names = [m[0] for m in _MOTIFS]
        _MOTIF_IDX.update({n: i for i, n in enumerate(names)})
        for i, n in enumerate(names):
            _MOTIF_NORM_IDX.setdefault(re.sub(r'[\s_-]', '', n.lower()), []).append(i)
    idx = _MOTIF_IDX.get(name)
    if idx is None:
        key = re.sub(r'[\s_-]', '', name.lower())
        hits = _MOTIF_NORM_IDX.get(key, [])
        if len(hits) > 1:
            raise ValueError(f"Motif name '{name}' is ambiguous between {[_MOTIFS[i][0] for i in hits]}; please use exact capitalization.")
        if not hits:
            generic = sorted(_MOTIF_NORM_IDX.get(f'terminal{key}', []) + _MOTIF_NORM_IDX.get(f'internal{key}', []))
            if not generic:
                return None
            # A position-less name (e.g., 'LewisX') means the motif wherever it sits, so take the variant without positional negations and relax its termini
            idx = min(generic, key = lambda i: _MOTIFS[i][1].count('!'))
            return _MOTIFS[idx][1], ['flexible'] * len(ast.literal_eval(_MOTIFS[idx][2]))
        idx = hits[0]
    return _MOTIFS[idx][1], ast.literal_eval(_MOTIFS[idx][2])


def unwrap(nested_list: list[Any] # list to be flattened
           ) -> list[Any]: # flattened list
    "converts a nested list into a flat list"
    return list(chain(*nested_list))


def find_nth(haystack: str, # string to search for motif
             needle: str, # motif
             n: int # n-th occurrence in string (not zero-indexed)
             ) -> int: # starting index of n-th occurrence
    "finds n-th instance of motif"
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start


def find_nth_reverse(string: str, # string to search
                     substring: str, # substring to find
                     n: int, # n-th occurrence from end
                     ignore_branches: bool = False # whether to ignore branches when counting
                     ) -> int: # position of n-th occurrence from end
    "finds n-th instance of motif from end of string"
    # Reverse the string and the substring
    reversed_string = string[::-1]
    reversed_substring = substring[::-1]
    # Initialize the start index for the search
    start_index = 0
    # Loop to find the n-th occurrence
    for i in range(n):
        # Find the next occurrence index
        idx = reversed_string.find(reversed_substring, start_index)
        # If the substring is not found, return -1
        if idx == -1:
            return -1
        # Update the start index
        start_index = idx + len(substring)
    # Calculate and return the original starting index
    original_start_index = len(string) - start_index
    if ignore_branches:
        # Check if there is an immediate branch preceding the match
        branch_end_idx = original_start_index - 1
        if branch_end_idx > 0 and string[branch_end_idx] == ']' and string[branch_end_idx - 1] != '[':
            # Find the start of the immediate branch
            bracket_count = 1
            for i in range(branch_end_idx - 1, -1, -1):
                if string[i] == ']':
                    bracket_count += 1
                elif string[i] == '[':
                    bracket_count -= 1
                if bracket_count == 0:
                    original_start_index = i
                    break
    return original_start_index


def remove_unmatched_brackets(s: str # glycan string in IUPAC-condensed
                              ) -> str: # glycan without unmatched brackets
    "Removes all unmatched brackets from the string s"
    while True:
        # Keep track of the indexes of the brackets
        stack = []
        unmatched_open = set()
        unmatched_close = set()
        for i, char in enumerate(s):
            if char == '[':
                stack.append(i)
            elif char == ']':
                if stack:
                    stack.pop()
                else:
                    unmatched_close.add(i)
        unmatched_open.update(stack)
        # If there are no unmatched brackets, break the loop
        if not unmatched_open and not unmatched_close:
            break
        # Build a new string without the unmatched brackets
        s = ''.join([char for i, char in enumerate(s) if i not in unmatched_open and i not in unmatched_close])
    return s


def reindex(df_new: pd.DataFrame, # dataframe with new row order
            df_old: pd.DataFrame, # dataframe with old row order
            out_col: str, # column name in df_old to reindex
            ind_col: str, # column name in df_old for index
            inp_col: str # column name in df_new for new order
            ) -> list: # out_col from df_old reordered to match inp_col in df_new
    "Returns columns values in order of new dataframe rows"
    if ind_col != inp_col:
        print("Mismatching column names for ind_col and inp_col. Doesn't mean it's wrong but pay attention.")
    out_vals = df_old[out_col].tolist()
    pos = {}
    for i, k in enumerate(df_old[ind_col].tolist()):
        pos.setdefault(k, i)
    new_keys = df_new[inp_col].tolist()
    if missing := [k for k in new_keys if k not in pos]:
        raise KeyError(
            f"{len(missing)} value(s) of df_new['{inp_col}'] have no match in df_old['{ind_col}'], e.g., {missing[:3]}")
    return [out_vals[pos[k]] for k in new_keys]


def stringify_dict(dicty: dict[Any, Any] # dictionary to convert
                   ) -> str: # string of type key:value for sorted items
    "Converts dictionary into a string"
    dicty = dict(sorted(dicty.items()))
    return ''.join(f"{key}{value}" for key, value in dicty.items())


def replace_every_second(string: str, # input string
                         old_char: str, # character to replace
                         new_char: str # character to replace with
                         ) -> str: # modified string
    "function to replace every second occurrence of old_char in string with new_char"
    count = 0
    result = []
    for char in string:
        if char == old_char:
            count += 1
            result.append(new_char if count % 2 == 0 else char)
        else:
            result.append(char)
    return ''.join(result)


def multireplace(string: str, # string to perform replacements on
                 remove_dic: dict[str, str] # dict of form to_replace:replace_with
                 ) -> str: # modified string
    "Replaces all occurences of items in a string with a given string"
    for k, v in remove_dic.items():
        string = string.replace(k, v)
    return string


def strip_suffixes(columns: list[Any] # column names
                   ) -> list[str]: # column names without numerical suffixes
    "Strip numerical suffixes like .1, .2, etc., from column names"
    return [re.sub(r"\.\d+$", "", str(name)) for name in columns]


def build_custom_df(df: pd.DataFrame, # df_glycan / sugarbase
                    kind: str = 'df_species' # whether to create 'df_species', 'df_tissue', or 'df_disease'
                    ) -> pd.DataFrame: # custom df with one glycan - species/tissue/disease association per row
    "creates custom df from df_glycan"
    kind_to_cols = {
        'df_species': ['glycan', 'Species', 'Genus', 'Family', 'Order', 'Class',
                       'Phylum', 'Kingdom', 'Domain', 'ref'],
        'df_tissue': ['glycan', 'tissue_sample', 'tissue_species', 'tissue_id', 'tissue_ref'],
        'df_disease': ['glycan', 'disease_association', 'disease_sample', 'disease_direction',
                       'disease_species', 'disease_id', 'disease_ref']
    }
    cols = kind_to_cols.get(kind, None)
    if cols is None:
        raise ValueError("Invalid value for 'kind' argument, only df_species, df_tissue, and df_disease are supported.")
    df = df.loc[df[cols[1]].str.len() > 0, cols]
    df = df.set_index('glycan')
    df = df.explode(cols[1:]).reset_index()
    df = df.sort_values([cols[1], 'glycan'], ascending = [True, True])
    df = df.reset_index(drop = True)
    from glycowork.glycan_data.frame import GlycoDataFrame
    return GlycoDataFrame(df)


def download_model(file_id: str # Filename in the HuggingFace repo
                   ) -> str:  # file path to cached model
    "Download the model weights file from HuggingFace Hub"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("<huggingface_hub missing; did you do 'pip install glycowork[ml]'?>")
    file_path = hf_hub_download(repo_id = "DBojar/glycowork_models", filename = file_id, etag_timeout = 30)
    print("Download completed.")
    return file_path


class DataFrameSerializer:
    """A utility class for serializing and deserializing pandas DataFrames with complex data types
    in a version-independent manner."""

    @staticmethod
    def _serialize_cell(value: Any) -> dict[str, Any]:
        """Convert a cell value to a serializable format with type information."""
        # Check if the value is a string representation of a list
        if isinstance(value, str) and value.strip().startswith('[') and value.strip().endswith(']'):
            try:
                # Try to convert the string to an actual list
                parsed_value = ast.literal_eval(value)
                if isinstance(parsed_value, list):
                    return {
                        'type': 'list',
                        'value': [str(item) for item in parsed_value]
                    }
            except (SyntaxError, ValueError):
                # If parsing fails, treat it as a regular string
                pass
        # Check if the value is a string representation of a dictionary
        if isinstance(value, str) and value.strip().startswith('{') and value.strip().endswith('}'):
            try:
                # Try to convert the string to an actual dictionary
                parsed_value = ast.literal_eval(value)
                if isinstance(parsed_value, dict):
                    # Use the appropriate dictionary type handler
                    if all(isinstance(k, str) and isinstance(v, list) for k, v in parsed_value.items()):
                        return {
                            'type': 'str_list_dict',
                            'value': {str(k): [str(item) for item in v] for k, v in parsed_value.items()}
                        }
                    elif all(isinstance(k, str) and isinstance(v, int) for k, v in parsed_value.items()):
                        return {
                            'type': 'str_int_dict',
                            'value': {str(k): int(v) for k, v in parsed_value.items()}
                        }
                    return {
                        'type': 'dict',
                        'value': {str(k): str(v) for k, v in parsed_value.items()}
                    }
            except (SyntaxError, ValueError):
                # If parsing fails, treat it as a regular string
                pass
        if isinstance(value, list):
            return {
                'type': 'list',
                'value': [str(item) for item in value]
            }
        elif isinstance(value, dict):
            # Check if it's a dictionary containing lists
            if all(isinstance(k, str) and isinstance(v, list) for k, v in value.items()):
                return {
                    'type': 'str_list_dict',
                    'value': {str(k): [str(item) for item in v] for k, v in value.items()}
                }
            # Check if it's a string:int dictionary
            elif all(isinstance(k, str) and isinstance(v, int) for k, v in value.items()):
                return {
                    'type': 'str_int_dict',
                    'value': {str(k): int(v) for k, v in value.items()}
                }
            # Fall back to string:string for other dictionaries
            return {
                'type': 'dict',
                'value': {str(k): str(v) for k, v in value.items()}
            }
        elif pd.isna(value):
            return {
                'type': 'null',
                'value': None
            }
        else:
            return {
                'type': 'primitive',
                'value': str(value)
            }

    @staticmethod
    def _deserialize_cell(cell_data: dict[str, Any]) -> Any:
        """Convert a serialized cell back to its original type."""
        if cell_data['type'] == 'list':
            return cell_data['value']
        elif cell_data['type'] == 'str_list_dict':
            return {str(k): list(v) for k, v in cell_data['value'].items()}
        elif cell_data['type'] == 'str_int_dict':
            return {str(k): int(v) for k, v in cell_data['value'].items()}
        elif cell_data['type'] == 'dict':
            return cell_data['value']
        elif cell_data['type'] == 'null':
            return None
        else:
            return cell_data['value']

    @classmethod
    def serialize(cls, df: pd.DataFrame, # DataFrame to serialize
                  path: str # file path to save serialized data
                  ) -> None:
        "Serialize a DataFrame to JSON with type information"
        data = {
            'columns': list(df.columns),
            'index': list(df.index),
            'data': []
        }
        for _, row in df.iterrows():
            serialized_row = [cls._serialize_cell(val) for val in row]
            data['data'].append(serialized_row)
        with (lzma.open if str(path).endswith('.xz') else open)(path, 'wt', encoding = 'utf-8') as f:
            json.dump(data, f)

    @classmethod
    def deserialize(cls, path: str # file path to load serialized data
                    ) -> pd.DataFrame: # DataFrame with restored data types
        "Deserialize a DataFrame from JSON"
        gc_was_enabled = gc.isenabled()
        gc.disable()  # the load allocates >1M small containers that can never form cycles, and the collector rescanning all of them over and over took ~80% of df_glycan's load time
        try:
            with (lzma.open if str(path).endswith('.xz') else open)(path, 'rt', encoding = 'utf-8') as f:
                data = json.load(f)
            deserialized_data = [[cls._deserialize_cell(cell) for cell in row] for row in data['data']]
            return pd.DataFrame(data = deserialized_data, columns = data['columns'], index = data['index'])
        finally:
            if gc_was_enabled:
                gc.enable()


serializer = DataFrameSerializer()


def count_nested_brackets(s: str, # Glycan in IUPAC-condensed nomenclature
                          length: bool = False # Whether to count the linkages ('(') two or more branch levels deep instead of the nested branches
                          ) -> int: # Number of branches opened inside another branch, or of deeply nested linkages if length
    "Counts how deeply a glycan string nests its branches"
    count = 0
    depth = 0
    nested_content = 0
    for c in s:
        if c == '[':
            if depth > 0:  # Only count if we're already inside brackets
                count += 1
            depth += 1
        elif c == ']':
            depth -= 1
        elif c == '(' and depth > 1:
            nested_content += 1
    return nested_content if length else count


def share_neighbor(edges: list[tuple], # Undirected edges as node pairs
                   node1: Any, # First node
                   node2: Any # Second node
                   ) -> bool: # Whether the two nodes have a common neighbor
    "Checks whether two nodes of an edge list share a neighbor"
    neighbors1 = set(v2 for v1, v2 in edges if v1 == node1) | set(v1 for v1, v2 in edges if v2 == node1)
    neighbors2 = set(v2 for v1, v2 in edges if v1 == node2) | set(v1 for v1, v2 in edges if v2 == node2)
    return bool(neighbors1 & neighbors2)


class HashableDict(dict):
    def __hash__(self):
        return hash(
            len(self))  # equal dicts always share their length and __eq__ settles the rest, so lru_cache lookups keyed on a 2,000-entry lib no longer sort it every call

    def __eq__(self, other):
        return isinstance(other, HashableDict) and dict.__eq__(self, other)


def parse_lines(excel_column_content: list | str # content of an Excel column pasted into a list and bookended by triple quotes
                ) -> list:  # proper list of strings
    "Splits a pasted Excel column into one string per line"
    inp = excel_column_content[0] if isinstance(excel_column_content, list) else excel_column_content
    return inp.strip().split('\n')
