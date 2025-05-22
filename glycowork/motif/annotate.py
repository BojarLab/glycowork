import networkx as nx
import pandas as pd
import numpy as np
import re
from collections import Counter, deque
from functools import partial
from typing import Dict, List, Optional, Set, Tuple, Union

from glycowork.glycan_data.loader import linkages, motif_list, unwrap, df_species
from glycowork.motif.graph import subgraph_isomorphism, generate_graph_features, glycan_to_nxGraph, graph_to_string, ensure_graph, possible_topology_check, graph_to_string_int
from glycowork.motif.processing import IUPAC_to_SMILES, get_lib, rescue_glycans
from glycowork.motif.regex import get_match


def annotate_glycan(
    glycan: Union[str, nx.DiGraph], # IUPAC-condensed glycan sequence or NetworkX graph
    motifs: Optional[pd.DataFrame] = None, # Motif dataframe (name + sequence); defaults to motif_list
    termini_list: List = [], # Monosaccharide positions: 'terminal', 'internal', or 'flexible'
    gmotifs: Optional[List[nx.DiGraph]] = None # Precalculated motif graphs for speed
    ) -> pd.DataFrame: # DataFrame with motif counts for the glycan
  "Counts occurrences of known motifs in a glycan structure using subgraph isomorphism"
  if motifs is None:
    motifs = motif_list
  # Check whether termini are specified
  if not termini_list and isinstance(motifs, pd.DataFrame):
    termini_list = list(map(eval, motifs.termini_spec))
  if gmotifs is None:
    termini = 'provided' if termini_list else 'ignore'
    partial_glycan_to_nxGraph = partial(glycan_to_nxGraph, termini = termini)
    gmotifs = list(map(lambda i_g: partial_glycan_to_nxGraph(i_g[1], termini_list = termini_list[i_g[0]]), enumerate(motifs.motif)))
  # Count the number of times each motif occurs in a glycan
  ggraph = ensure_graph(glycan, termini = 'calc' if termini_list else 'ignore')
  res = [subgraph_isomorphism(ggraph, g, termini_list = termini_list[i] if termini_list else termini_list,
                                count = True) for i, g in enumerate(gmotifs)]*1
  out = pd.DataFrame(columns = motifs.motif_name if isinstance(motifs, pd.DataFrame) else motifs)
  out.loc[0] = res
  out.loc[0] = out.loc[0].astype('int')
  out.index = [glycan] if isinstance(glycan, str) else [graph_to_string(glycan)]
  return out


def annotate_glycan_topology_uncertainty(
    glycan: Union[str, nx.DiGraph], # IUPAC-condensed glycan sequence or NetworkX graph
    feasibles: Optional[Set[str]] = None, # Set of potential full topology glycans; defaults to mammalian glycans
    motifs: Optional[pd.DataFrame] = None, # Motif dataframe (name + sequence); defaults to motif_list
    termini_list: List = [], # Monosaccharide positions: 'terminal', 'internal', or 'flexible'
    gmotifs: Optional[List[nx.DiGraph]] = None # Precalculated motif graphs for speed
    ) -> pd.DataFrame: # DataFrame with motif counts considering topology uncertainty
  "Annotates glycan motifs while handling uncertain topologies by comparing against physiological structures"
  if motifs is None:
    motifs = motif_list
  if feasibles is None:
    feasibles = set(df_species[df_species.Class == "Mammalia"].glycan.values.tolist())
  # Check whether termini are specified
  if not termini_list and isinstance(motifs, pd.DataFrame):
    termini_list = list(map(eval, motifs.termini_spec))
  if gmotifs is None:
    termini = 'provided' if termini_list else 'ignore'
    partial_glycan_to_nxGraph = partial(glycan_to_nxGraph, termini = termini)
    gmotifs = list(map(lambda i_g: partial_glycan_to_nxGraph(i_g[1], termini_list = termini_list[i_g[0]]), enumerate(motifs.motif)))
  # Count the number of times each motif occurs in a glycan
  ggraph = ensure_graph(glycan, termini = 'calc' if termini_list else 'ignore')
  glycan_len = glycan.count('(')
  feasibles = [glycan_to_nxGraph(g, termini = 'calc') for g in feasibles if g.count('(') == glycan_len]
  possibles = possible_topology_check(ggraph, feasibles, exhaustive = True)
  res = []
  for i, g in enumerate(gmotifs):
    temp_res = subgraph_isomorphism(ggraph, g, termini_list = termini_list[i] if termini_list else termini_list,
                                count = True)
    if temp_res:
      res.append(temp_res*1)
      continue
    temp_res = [subgraph_isomorphism(p, g, termini_list = termini_list[i] if termini_list else termini_list,
                                count = True) for p in possibles]
    if temp_res and np.mean(temp_res) > 0.5:
      res.append(1)
    else:
      res.append(0)
  out = pd.DataFrame(columns = motifs.motif_name if isinstance(motifs, pd.DataFrame) else motifs)
  out.loc[0] = res
  out.loc[0] = out.loc[0].astype('int')
  out.index = [glycan] if isinstance(glycan, str) else [graph_to_string(glycan)]
  return out


def get_molecular_properties(
    glycan_list: List[str], # List of IUPAC-condensed glycan sequences
    verbose: bool = False, # Print SMILES not found on PubChem
    placeholder: bool = False # Return dummy values instead of dropping failed requests
    ) -> pd.DataFrame: # DataFrame with molecular parameters from PubChem
  "Retrieves molecular properties from PubChem for a list of glycans using their SMILES representations"
  try:
    import pubchempy as pcp
  except ImportError:
    raise ImportError("You must install the 'chem' dependencies to use this feature. Try 'pip install glycowork[chem]'.")
  if placeholder:
    dummy = IUPAC_to_SMILES(['Glc'])[0]
  compounds_list,, succeeded_requests, failed_requests = [], [], []
  for s, g in zip(*[IUPAC_to_SMILES(glycan_list), glycan_list]):
    try:
      c = pcp.get_compounds(s, 'smiles')[0]
      if c.cid is None:
        compounds_list.append(pcp.get_compounds(dummy, 'smiles')[0]) if placeholder else failed_requests.append(s)
      else:
        compounds_list.append(c)
        succeeded_requests.append(g)
    except:
      failed_requests.append(s)
  if verbose and len(failed_requests) >= 1:
    print('The following SMILES were not found on PubChem:\n')
    print(failed_requests)
  df = pcp.compounds_to_frame(compounds_list, properties = ['molecular_weight', 'xlogp',
                                                            'charge', 'exact_mass', 'monoisotopic_mass', 'tpsa', 'complexity',
                                                            'h_bond_donor_count', 'h_bond_acceptor_count',
                                                            'rotatable_bond_count', 'heavy_atom_count', 'isotope_atom_count', 'atom_stereo_count',
                                                            'defined_atom_stereo_count', 'undefined_atom_stereo_count',
                                                            'bond_stereo_count', 'defined_bond_stereo_count',
                                                            'undefined_bond_stereo_count', 'covalent_unit_count'])
  df = df.reset_index(drop = True)
  df.index = succeeded_requests
  return df


def get_size_branching_features(
    glycans: List[str],  # List of IUPAC-condensed glycan sequences
    n_bins: int = 3  # Number of bins/features to create for size and branching level
    ) -> pd.DataFrame:
  "Generate binned features for glycan size (parentheses count) and branching (bracket count)"
  # Calculate size and branching for each glycan
  sizes = [glycan.count('(')+1 for glycan in glycans]
  branchings = [glycan.count('[') for glycan in glycans]
  if n_bins == 0:
    return pd.DataFrame({'Size': sizes, 'Branching': branchings}, index = glycans)
  # Helper function to create bin edges and labels
  def create_bins(values: List[int], n_bins: int) -> Tuple[List[int], List[str]]:
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
      return [min_val - 0.5, max_val + 0.5], [f"{min_val}"]
    edges = list(np.linspace(min_val - 0.5, max_val + 0.5, n_bins + 1))  # Create bin edges
    labels = [f"{int(edges[i]+0.5)}-{int(edges[i+1]-0.5)}" for i in range(len(edges)-1)]  # Create bin labels
    return edges, labels
  # Create bins for size and branching
  size_edges, size_labels = create_bins(sizes, n_bins)
  branch_edges, branch_labels = create_bins(branchings, n_bins)
  # Create DataFrames for size and branching distributions
  size_dist = pd.DataFrame(0, index = glycans, columns = [f"Size_{label}" for label in size_labels])
  branch_dist = pd.DataFrame(0, index = glycans, columns = [f"Branch_{label}" for label in branch_labels])
  # Assign values to bins
  for i, glycan in enumerate(glycans):
    size_bin = np.digitize([sizes[i]], size_edges)[0] - 1
    branch_bin = np.digitize([branchings[i]], branch_edges)[0] - 1
    if size_bin < len(size_labels):
      size_dist.iloc[i, size_bin] = 1
    if branch_bin < len(branch_labels):
      branch_dist.iloc[i, branch_bin] = 1
  # Combine size and branching features
  return pd.concat([size_dist, branch_dist], axis = 1)


@rescue_glycans
def annotate_dataset(
    glycans: List[str], # List of IUPAC-condensed glycan sequences
    motifs: Optional[pd.DataFrame] = None, # Motif dataframe (name + sequence); defaults to motif_list
    feature_set: List[str] = ['known'], # Feature types to analyze: known, graph, exhaustive, terminal(1-3), custom, chemical, size_branch
    termini_list: List = [], # Monosaccharide positions: 'terminal', 'internal', or 'flexible'
    condense: bool = False, # Remove columns with only zeros
    custom_motifs: List = [] # Custom motifs when using 'custom' feature set
    ) -> pd.DataFrame: # DataFrame mapping glycans to presence/absence of motifs
  "Comprehensive glycan annotation combining multiple feature types: structural motifs, graph properties, terminal sequences"
  if any([k in ''.join(glycans) for k in [';', 'β', 'α', 'RES', '=']]):
    raise Exception
  invalid_features = set(feature_set) - {'known', 'graph', 'terminal', 'terminal1', 'terminal2', 'terminal3', 'custom', 'chemical', 'exhaustive', 'size_branch'}
  if invalid_features:
    print(f"Warning: {', '.join(invalid_features)} not recognized as features.")
  if motifs is None:
    motifs = motif_list
  # Checks whether termini information is provided
  if not termini_list:
    termini_list = list(map(eval, motifs.termini_spec))
  shopping_cart = []
  if 'known' in feature_set:
    termini = 'provided' if termini_list else 'ignore'
    termini_list = termini_list if termini_list else ['ignore']*len(motifs)
    partial_glycan_to_nxGraph = partial(glycan_to_nxGraph, termini = termini)
    gmotifs = list(map(lambda i_g: partial_glycan_to_nxGraph(i_g[1], termini_list = termini_list[i_g[0]]), enumerate(motifs.motif)))
    # Counts literature-annotated motifs in each glycan
    partial_annotate = partial(annotate_glycan, motifs = motifs, termini_list = termini_list, gmotifs = gmotifs)
    if '{' in ''.join(glycans):
      feasibles = set(df_species[df_species.Class == "Mammalia"].glycan.values.tolist())
      partial_annotate_topology_uncertainty = partial(annotate_glycan_topology_uncertainty, feasibles = feasibles, motifs = motifs, termini_list = termini_list, gmotifs = gmotifs)
    else:
      partial_annotate_topology_uncertainty = partial(annotate_glycan, motifs = motifs, termini_list = termini_list, gmotifs = gmotifs)
    def annotate_switchboard(glycan):
      return partial_annotate_topology_uncertainty(glycan) if glycan.count('{') == 1 else partial_annotate(glycan)
    shopping_cart.append(pd.concat(list(map(annotate_switchboard, glycans)), axis = 0))
  if 'custom' in feature_set:
    normal_motifs = [m for m in custom_motifs if not m.startswith('r')]
    gmotifs = list(map(glycan_to_nxGraph, normal_motifs))
    partial_annotate = partial(annotate_glycan, motifs = normal_motifs, gmotifs = gmotifs)
    shopping_cart.append(pd.concat(list(map(partial_annotate, glycans)), axis = 0))
    regex_motifs = [m[1:] for m in custom_motifs if m.startswith('r')]
    shopping_cart.append(pd.concat([pd.DataFrame([len(get_match(p, k)) for p in regex_motifs], columns = regex_motifs, index = [k]) for k in glycans], axis = 0))
  if 'graph' in feature_set:
    # Calculates graph features of each glycan
    shopping_cart.append(pd.concat(list(map(generate_graph_features, glycans)), axis = 0))
  if 'exhaustive' in feature_set:
    # Counts disaccharides and monosaccharides in each glycan
    temp = get_k_saccharides(glycans, size = 2, up_to = True)
    temp.index = glycans
    shopping_cart.append(temp)
  if 'chemical' in feature_set:
    shopping_cart.append(get_molecular_properties(glycans, placeholder = True))
  if 'terminal' in feature_set or 'terminal1' in feature_set or 'terminal2' in feature_set:
    bag1, bag2 = [], []
    if 'terminal' in feature_set or 'terminal1' in feature_set:
      bag1 = list(map(get_terminal_structures, glycans))
    if 'terminal2' in feature_set:
      bag2 = [get_terminal_structures(glycan, size = 2) for glycan in glycans]
    bag = [a + b for a, b in zip(bag1, bag2)] if (bag1 and bag2) else bag1 + bag2
    repertoire = set(unwrap(bag))
    repertoire2 = [re.sub(r"\(([ab])(\d)-(\d)\)", r"(\1\2-?)", g) for g in repertoire]
    repertoire2 = set([k for k in repertoire2 if repertoire2.count(k) > 1 and k not in repertoire])
    repertoire.update(repertoire2)
    bag_out = pd.DataFrame([{i: j.count(i) for i in repertoire if '?' not in i} for j in bag])
    shadow_glycans = [[re.sub(r"\(([ab])(\d)-(\d)\)", r"(\1\2-?)", g) for g in b] for b in bag]
    shadow_bag = pd.DataFrame([{i: j.count(i) for i in repertoire if '?' in i} for j in shadow_glycans])
    bag_out = pd.concat([bag_out, shadow_bag], axis = 1).reset_index(drop = True)
    bag_out.index = glycans
    bag_out.columns = ['Terminal_' + c for c in bag_out.columns]
    shopping_cart.append(bag_out)
  if 'terminal3' in feature_set:
    temp = get_k_saccharides(glycans, size = 3, terminal = True)
    temp.index = glycans
    temp.columns = ['Terminal_' + c for c in temp.columns]
    shopping_cart.append(temp)
  if 'size_branch' in feature_set:
    shopping_cart.append(get_size_branching_features(glycans))
  if condense:
    # Remove motifs that never occur
    temp = pd.concat(shopping_cart, axis = 1)
    return temp.loc[:, (temp != 0).any(axis = 0)]
  else:
    return pd.concat(shopping_cart, axis = 1)


def deduplicate_motifs(
    df: pd.DataFrame # DataFrame with glycan motifs as rows, samples as columns
    ) -> pd.DataFrame: # DataFrame with redundant motifs removed
  "Removes redundant motif entries from glycan abundance data while preserving the most informative labels"
  motif_dic = dict(zip(motif_list.motif_name, motif_list.motif))
  original_index = df.index.copy()
  df.index = df.index.to_series().apply(lambda x: motif_dic[x] + ' ' * 20 if x in motif_dic else x)
  df['_original_position'] = range(len(df))
  # Group the DataFrame by identical rows
  grouped = df.groupby(list(df.columns[:-1]), sort = False)
  # Find the integer indices of rows with the longest string index within each group
  max_idx_positions = []
  for _, group in grouped:
    # Find the row with the longest string index
    longest_idx = group.index.to_series().str.len().idxmax()
    # Retrieve the original integer position of this row
    max_idx_positions.append(group.loc[longest_idx, '_original_position'])
  df.index = original_index
  return df.iloc[max_idx_positions].drop_duplicates().drop(['_original_position'], axis = 1)


def quantify_motifs(
   df: Union[str, pd.DataFrame], # DataFrame or filepath with samples as columns, abundances as values
   glycans: List[str], # List of IUPAC-condensed glycan sequences
   feature_set: List[str], # Feature types to analyze: known, graph, exhaustive, terminal(1-3), custom, chemical, size_branch
   custom_motifs: List = [], # Custom motifs when using 'custom' feature set
   remove_redundant: bool = True # Remove redundant motifs via deduplicate_motifs
   ) -> pd.DataFrame: # DataFrame with motif abundances (motifs as columns, samples as rows)
  "Extracts and quantifies motif abundances from glycan abundance data by weighting motif occurrences"
  if isinstance(df, str):
    df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
  # Motif extraction
  df_motif = annotate_dataset(glycans, feature_set = feature_set,
                              condense = True, custom_motifs = custom_motifs)
  collect_dic = {}
  df = df.T
  log2 = (df.values < 0).any()
  # Motif quantification
  for c, col in enumerate(df_motif.columns):
    indices = [i for i, x in enumerate(df_motif[col]) if x >= 1]
    temp = df.iloc[:, indices]
    temp.columns = range(temp.columns.size)
    if log2:
      linear_values = np.power(2, temp)
      weighted_values = (linear_values * df_motif.iloc[indices, c].reset_index(drop = True)).sum(axis = 1)
      collect_dic[col] = np.log2(weighted_values)
    else:
      collect_dic[col] = (temp * df_motif.iloc[indices, c].reset_index(drop = True)).sum(axis = 1)
  df = pd.DataFrame(collect_dic)
  return df if not remove_redundant else deduplicate_motifs(df.T)


def count_unique_subgraphs_of_size_k(
   graph: nx.DiGraph, # NetworkX graph of a glycan
   size: int = 2, # Number of monosaccharides per subgraph
   terminal: bool = False # Only count terminal subgraphs
   ) -> Dict[str, float]: # Dictionary mapping k-saccharide sequences to counts
  "Identifies and counts unique connected subgraphs of specified size from glycan structure"
  target_size = size * 2 - 1
  successor_dict = {v: list(graph.successors(v)) for v in graph.nodes}
  k_subgraphs, processed = [], set()
  terminal_lookup = {n: graph.out_degree(n) == 0 for n in graph.nodes()} if terminal else {}
  for node in sorted(graph.nodes(), reverse = True):
    queue = deque([(tuple([node]), 1 if node % 2 == 0 else 0, 1)])
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
        new_mono = mono_cnt + (1 if candidate % 2 == 0 else 0)
        if new_mono > size: continue
        queue.append((tuple(sorted(current_sg_set | {candidate})), new_mono, curr_size + 1))
  return dict(Counter(k_subgraphs))


@rescue_glycans
def get_k_saccharides(
   glycans: Union[List[str], Set[str]], # List or set of IUPAC-condensed glycan sequences
   size: int = 2, # Number of monosaccharides per fragment
   up_to: bool = False, # Include fragments up to size k (adds monosaccharides)
   just_motifs: bool = False, # Return nested list of motifs instead of count DataFrame
   terminal: bool = False # Only count terminal fragments
   ) -> Union[pd.DataFrame, List[List[str]]]: # DataFrame of k-saccharide counts or list of motifs per glycan
  "Extracts k-saccharide fragments from glycan sequences with options for different fragment sizes and positions"
  if not isinstance(glycans, (list, set)):
    raise TypeError("The input has to be a list or set of glycans")
  if any(k in ''.join(glycans) for k in [';', 'β', 'α', 'RES', '=']):
    raise Exception
  if not up_to and max(g.count('(')+1 for g in glycans) < size:
    return [] if just_motifs else pd.DataFrame()
  if up_to:
    wga_letter = pd.DataFrame([{i: len(re.findall(rf'{re.escape(i)}(?=\(|$)', g)) for i in get_lib(glycans) if i not in linkages} for g in glycans])
  regex = re.compile(r"([ab])(\d)-(\d)")
  ggraphs, shadow_graphs = [], []
  for g_str in glycans:
    g = glycan_to_nxGraph(g_str)
    ggraphs.append(g)
    sg = g.copy()
    labels = nx.get_node_attributes(sg, 'string_labels')
    nx.set_node_attributes(sg, {n: regex.sub(r"\1\2-?", labels[n]) for n in labels}, 'string_labels')
    shadow_graphs.append(sg)
  ggraphs = pd.DataFrame([count_unique_subgraphs_of_size_k(g, size = size, terminal = terminal) for g in ggraphs])
  ggraphs = ggraphs.drop(columns = [col for col in ggraphs.columns if '?' in col])
  shadow_glycans = pd.DataFrame([count_unique_subgraphs_of_size_k(g, size = size, terminal = terminal) for g in shadow_graphs])
  org_len = ggraphs.shape[1]
  out_matrix = pd.concat([ggraphs, shadow_glycans], axis = 1).reset_index(drop = True)
  out_matrix = out_matrix.loc[:, ~out_matrix.columns.duplicated()].copy()
  # Throw out redundant columns
  cols = out_matrix.columns.tolist()
  first_half_cols = cols[:org_len]
  second_half_cols = cols[org_len:]
  drop_columns = []
  regex = re.compile(r"(\([ab])(\d)-(\d)\)")
  col_sums = out_matrix.sum().to_dict()
  col_map = {regex.sub(r"\1\2-?)", col): col for col in first_half_cols}
  for col2 in second_half_cols:
    if col2 in col_map:
      col1 = col_map[col2]
      if col_sums[col1] == col_sums[col2]:
        drop_columns.append(col2)
  out_matrix = out_matrix.drop(drop_columns, axis = 1)
  out_matrix = out_matrix.T.groupby(level = 0).sum().T
  if up_to:
    combined_df= pd.concat([wga_letter, out_matrix], axis = 1).fillna(0).astype(int)
    return combined_df.apply(lambda x: list(combined_df.columns[x > 0]), axis = 1).tolist() if just_motifs else combined_df
  return out_matrix.apply(lambda x: list(out_matrix.columns[x > 0]), axis = 1).tolist() if just_motifs else out_matrix.fillna(0).astype(int)


def get_terminal_structures(
   glycan: Union[str, nx.DiGraph], # IUPAC-condensed glycan sequence or NetworkX graph
   size: int = 1 # Number of monosaccharides in terminal fragment (1 or 2)
   ) -> List[str]: # List of terminal structures with linkages
  "Identifies terminal monosaccharide sequences from non-reducing ends of glycan structure"
  if size > 2:
    raise ValueError("Please use get_k_saccharides with terminal = True for larger terminal structures")
  ggraph = ensure_graph(glycan)
  nodeDict = dict(ggraph.nodes(data = True))
  temp =  [nodeDict[k]['string_labels']+'('+nodeDict[k+1]['string_labels']+')' + \
   ''.join([nodeDict.get(k+1+j+i, {'string_labels': ''})['string_labels']+'('+nodeDict.get(k+2+j+i, {'string_labels': ''})['string_labels']+')' \
            for i, j in enumerate(range(1, size))]) for k in list(ggraph.nodes())[:-1] if \
          ggraph.out_degree[k] == 0 and k+1 in nodeDict.keys() and nodeDict[k]['string_labels'] not in linkages]
  return [g.replace('()', '') for g in temp]


def create_correlation_network(
   df: pd.DataFrame, # DataFrame with samples as rows, glycans/motifs as columns
   correlation_threshold: float # Minimum correlation for cluster inclusion
   ) -> List[Set[str]]: # List of sets containing correlated glycans/motifs
  "Groups glycans or motifs into clusters based on abundance correlation patterns"
  # Calculate the correlation matrix
  correlation_matrix = df.corr()
  # Create an adjacency matrix based on the correlation threshold
  adjacency_matrix = (correlation_matrix > correlation_threshold).astype(int)
  # Create a graph from the adjacency matrix
  graph = nx.from_numpy_array(adjacency_matrix.values)
  # Use the graph to find connected components (clusters of highly correlated glycans)
  clusters = list(nx.connected_components(graph))
  # Convert indices back to original labels
  clusters = [set(df.columns[list(cluster)]) for cluster in clusters]
  return clusters


def group_glycans_core(
   glycans: List[str], # List of IUPAC-condensed O-glycan sequences
   p_values: List[float] # Statistical test p-values for each glycan
   ) -> Tuple[Dict[str, List[str]], Dict[str, List[float]]]: # (group:glycans dict, group:p-values dict)
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
   glycans: List[str], # List of IUPAC-condensed glycan sequences
   p_values: List[float] # Statistical test p-values for each glycan
   ) -> Tuple[Dict[str, List[str]], Dict[str, List[float]]]: # (group:glycans dict, group:p-values dict)
  "Groups glycans by sialic acid and fucose content for statistical analysis"
  temp = dict(zip(glycans, p_values))
  grouped_glycans, grouped_p_values = {}, {}
  for g in glycans:
    has_sia = "Neu" in g
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
   glycans: List[str], # List of IUPAC-condensed N-glycan sequences
   p_values: List[float] # Statistical test p-values for each glycan
   ) -> Tuple[Dict[str, List[str]], Dict[str, List[float]]]: # (group:glycans dict, group:p-values dict)
  "Groups N-glycans by type (complex, hybrid, high-mannose, other) for statistical analysis"
  temp = dict(zip(glycans, p_values))
  grouped_glycans, grouped_p_values = {}, {}
  complex_patterns = ["GlcNAc(b1-2)Man(a1-6)", "GlcNAc(b1-2)Man(a1-?)[GlcNAc(b1-2)Man(a1-?)]Man"]
  hybrid_patterns = ["GlcNAc(b1-2)Man(a1-3)[Man(a1-?)Man(a1-6)]Man"]
  for g in glycans:
    if any(subgraph_isomorphism(g, pattern) for pattern in complex_patterns):
      group = "complex"
    elif any(subgraph_isomorphism(g, pattern) for pattern in hybrid_patterns):
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
       abbr: List[str], # List of lectin abbreviations
       name: List[str], # List of lectin names
       specificity: Dict[str, Optional[Dict[str, List[str]]]] = {"primary": None, "secondary": None, "negative": None}, # Binding specificity details
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

  def get_all_binding_motifs(self) -> List[str]: # List of all binding motifs
      "Returns combined list of primary and secondary binding motifs"
      return list(self.specificity["primary"].keys()) + list(self.specificity["secondary"].keys())

  def show_info(self) -> None:
      "Displays formatted lectin information including specificities"
      print(f"Name(s): {'; '.join(self.name)}\n"+\
      f"Abbreviation(s): {'; '.join(self.abbr)}\n"+\
      f"Species: {self.species}\n" + \
                       ''.join(f"Specificity ({key}): {'; '.join(value) if value else 'no information'}\n"
                               for key, value in self.specificity.items()) + \
                       f"Reference: {self.reference}\nnotes: {self.notes}\n")

  def check_binding(
       self,
       glycan: str # IUPAC-condensed glycan sequence
       ) -> int: # Binding level (0:none, 1:primary, 2:secondary)
    "Evaluates glycan binding to lectin based on motif recognition"
    for key in ["negative", "primary", "secondary"]:
      motifs = self.specificity.get(key, [])
      if motifs:
        for motif, termini in motifs.items():
          if subgraph_isomorphism(glycan, motif, termini_list = termini):
            return 0 if key == "negative" else (1 if key == "primary" else 2)
    return 0


def load_lectin_lib() -> Dict[int, Lectin]: # Dictionary mapping indices to Lectin objects
  "Loads curated lectin library with binding specificity information"
  from glycowork.glycan_data.loader import lectin_specificity
  lectin_lib = {}
  for i, row in enumerate(lectin_specificity.itertuples()):
    specificity_dict = {"primary": row.specificity_primary, "secondary": row.specificity_secondary, "negative": row.specificity_negative}
    lectin_lib[i] = Lectin(abbr = row.abbreviation.split(","), name = row.name.split(","), specificity = specificity_dict,
                           species = row.species, reference = row.reference, notes = row.notes)
  return lectin_lib


def create_lectin_and_motif_mappings(
   lectin_list: List[str], # List of lectin names/abbreviations
   lectin_lib: Dict[int, Lectin] # Library of Lectin objects
   ) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]: # (lectin:index dict, motif:{lectin:weight} dict)
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
    motif_mapping[motif] = dict(sorted(mapping.items(), key=lambda x:x[1]))
  return useable_lectin_mapping, motif_mapping


def lectin_motif_scoring(
   useable_lectin_mapping: Dict[str, int], # Lectin to index mapping
   motif_mapping: Dict[str, Dict[str, int]], # Motif to {lectin:weight} mapping
   lectin_score_dict: Dict[str, float], # Lectin effect sizes
   lectin_lib: Dict[int, Lectin], # Library of Lectin objects
   idf: Dict[str, float], # Lectin variance across groups
   class_weight_dict: Dict[int, float] = {0: 1, 1: 0.5, 2: 0.5, 3: 0.25}, # Weights for binding classes
   specificity_weight_dict: Dict[str, float] = {"<3": 1, "3-4": 0.75, ">4": 0.5}, # Weights for binding specificity
   duplicate_lectin_penalty_factor: float = 0.8 # Penalty for redundant lectins
   ) -> pd.DataFrame: # DataFrame with scored motifs and supporting evidence
  "Calculates weighted motif scores from lectin binding data incorporating specificity and redundancy factors"
  output = []
  useable_lectin_count = {k: list(useable_lectin_mapping.values()).count(v) for k, v in useable_lectin_mapping.items()}
  #max_motifs = max([lectin_lib[k].get_all_binding_motifs_count() for k in useable_lectin_mapping.values()]) + 1
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
