import networkx as nx
import pandas as pd
import re
from collections import defaultdict

from glycowork.glycan_data.loader import lib, linkages, motif_list, find_nth, unwrap, replace_every_second, remove_unmatched_brackets
from glycowork.motif.graph import subgraph_isomorphism, generate_graph_features, glycan_to_nxGraph, graph_to_string, ensure_graph
from glycowork.motif.processing import IUPAC_to_SMILES, get_lib, find_isomorphs, expand_lib, rescue_glycans


def link_find(glycan):
  """finds all disaccharide motifs in a glycan sequence using its isomorphs\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format\n
  | Returns:
  | :-
  | Returns list of unique disaccharides (strings) for a glycan in IUPAC-condensed
  """
  if '}' in glycan:
    glycan = glycan[glycan.rindex('}')+1:]
  # Get different string representations of the same glycan
  ss = find_isomorphs(glycan)
  coll = []
  # For each string representation, search and remove branches
  for iso in ss:
    if bool(re.search(r"\[[^\[\]]+\[[^\[\]]+\][^\]]+\]", iso)):
      b_re = re.sub(r"\[[^\[\]]+\[[^\[\]]+\][^\]]+\]", '', iso)
      b_re = re.sub(r"\[[^\]]+\]", '', b_re)
    else:
      b_re = re.sub(r"\[[^\]]+\]", '', iso)
    # From this linear glycan part, chunk away disaccharides and re-format them
    for i in [iso, b_re]:
      b = i.split('(')
      b = unwrap([k.split(')') for k in b])
      b = ['*'.join(b[i:i+3]) for i in range(0, len(b) - 2, 2)]
      b = [k for k in b if (re.search(r"\*\[", k) is None and re.search(r"\*\]\[", k) is None)]
      b = [k.strip('[').strip(']') for k in b]
      b = [k.replace('[', '').replace(']', '') for k in b]
      b = [k[:find_nth(k, '*', 1)] + '(' + k[find_nth(k, '*', 1)+1:] for k in b]
      coll += [k[:find_nth(k, '*', 1)] + ')' + k[find_nth(k, '*', 1)+1:] for k in b]
  return list(set(coll))


def annotate_glycan(glycan, motifs = None, libr = None,
                    termini_list = [], gmotifs = None):
  """searches for known motifs in glycan sequence\n
  | Arguments:
  | :-
  | glycan (string or networkx): glycan in IUPAC-condensed format (or as networkx graph) that has to contain a floating substituent
  | motifs (dataframe): dataframe of glycan motifs (name + sequence); default:motif_list
  | libr (dict): dictionary of form glycoletter:index
  | termini_list (list): list of monosaccharide positions (from 'terminal', 'internal', and 'flexible')
  | gmotifs (networkx): precalculated motif graphs for speed-up; default:None\n
  | Returns:
  | :-
  | Returns dataframe with counts of motifs in glycan
  """
  if motifs is None:
    motifs = motif_list
  # Check whether termini are specified
  if not termini_list:
    termini_list = [eval(k) for k in motifs.termini_spec]
  if libr is None:
    libr = lib
  if gmotifs is None:
    termini = 'provided' if termini_list else 'ignore'
    gmotifs = [glycan_to_nxGraph(g, libr = libr, termini = termini, termini_list = termini_list[i]) for i, g in enumerate(motifs.motif)]
  # Count the number of times each motif occurs in a glycan
  if termini_list:
    ggraph = ensure_graph(glycan, libr = libr, termini = 'calc')
    res = [subgraph_isomorphism(ggraph, gmotifs[k], libr = libr,
                                termini_list = termini_list[k],
                                count = True) for k in range(len(motifs))]*1
  else:
    ggraph = ensure_graph(glycan, libr = libr, termini = 'ignore')
    res = [subgraph_isomorphism(ggraph, gmotifs[k], libr = libr,
                                termini_list = termini_list,
                                count = True) for k in range(len(motifs))]*1
 
  out = pd.DataFrame(columns = motifs.motif_name)
  out.loc[0] = res
  out.loc[0] = out.loc[0].astype('int')
  if isinstance(glycan, str):
    out.index = [glycan]
  else:
    out.index = [graph_to_string(glycan)]
  return out


def get_molecular_properties(glycan_list, verbose = False, placeholder = False):
  """given a list of SMILES glycans, uses pubchempy to return various molecular parameters retrieved from PubChem\n
  | Arguments:
  | :-
  | glycan_list (list): list of glycans in IUPAC-condensed
  | verbose (bool): set True to print SMILES not found on PubChem; default:False
  | placeholder (bool): whether failed requests should return dummy values or be dropped; default:False\n
  | Returns:
  | :-
  | Returns a dataframe with all the molecular parameters retrieved from PubChem
  """
  try:
    import pubchempy as pcp
  except ImportError:
    raise ImportError("You must install the 'chem' dependencies to use this feature. Try 'pip install glycowork[chem]'.")
  smiles_list = IUPAC_to_SMILES(glycan_list)
  if placeholder:
    dummy = IUPAC_to_SMILES(['Glc'])[0]
  compounds_list = []
  failed_requests = []
  for s in smiles_list:
    try:
      c = pcp.get_compounds(s, 'smiles')[0]
      if c.cid is None:
        if placeholder:
          compounds_list.append(pcp.get_compounds(dummy, 'smiles')[0])
        else:
          failed_requests.append(s)
      else:
        compounds_list.append(c)
    except:
      failed_requests.append(s)
  if verbose and len(failed_requests) >= 1:
    print('The following SMILES were not found on PubChem:')
    for failed in failed_requests:
      print(failed)
  df = pcp.compounds_to_frame(compounds_list, properties = ['molecular_weight', 'xlogp',
                                                            'charge', 'exact_mass', 'monoisotopic_mass', 'tpsa', 'complexity',
                                                            'h_bond_donor_count', 'h_bond_acceptor_count',
                                                            'rotatable_bond_count', 'heavy_atom_count', 'isotope_atom_count', 'atom_stereo_count',
                                                            'defined_atom_stereo_count', 'undefined_atom_stereo_count',
                                                            'bond_stereo_count', 'defined_bond_stereo_count',
                                                            'undefined_bond_stereo_count', 'covalent_unit_count'])
  df = df.reset_index(drop = True)
  df.index = glycan_list
  return df


@rescue_glycans
def annotate_dataset(glycans, motifs = None,
                     feature_set = ['known'], termini_list = [], condense = False):
  """wrapper function to annotate motifs in list of glycans\n
  | Arguments:
  | :-
  | glycans (list): list of IUPAC-condensed glycan sequences as strings
  | motifs (dataframe): dataframe of glycan motifs (name + sequence); default:motif_list
  | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), and 'chemical' (molecular properties of glycan)
  | termini_list (list): list of monosaccharide/linkage positions (from 'terminal', 'internal', and 'flexible')
  | condense (bool): if True, throws away columns with only zeroes; default:False\n
  | Returns:
  | :-                      
  | Returns dataframe of glycans (rows) and presence/absence of known motifs (columns)
  """
  if any([k in glycans[0] for k in [';', '-D-', 'RES', '=']]):
    raise Exception
  if motifs is None:
    motifs = motif_list
  libr = get_lib(glycans + motifs.motif.values.tolist() + list(linkages))
  # Checks whether termini information is provided
  if not termini_list:
    termini_list = [eval(k) for k in motifs.termini_spec]
  shopping_cart = []
  if 'known' in feature_set:
    termini = 'provided' if termini_list else 'ignore'
    termini_list = termini_list if termini_list else ['ignore']*len(motifs)
    gmotifs = [glycan_to_nxGraph(g, libr = libr, termini = termini, termini_list = termini_list[i]) for i, g in enumerate(motifs.motif)]
    # Counts literature-annotated motifs in each glycan
    shopping_cart.append(pd.concat([annotate_glycan(k, motifs = motifs, libr = libr,
                                                    gmotifs = gmotifs, termini_list = termini_list) for k in glycans], axis = 0))
  if 'graph' in feature_set:
    # Calculates graph features of each glycan
    shopping_cart.append(pd.concat([generate_graph_features(k, libr = libr) for k in glycans], axis = 0))
  if 'exhaustive' in feature_set:
    # Counts disaccharides and monosaccharides in each glycan
    temp = get_k_saccharides(glycans, size = 2, up_to = True, libr = libr)
    temp.index = glycans
    shopping_cart.append(temp)
  if 'chemical' in feature_set:
    shopping_cart.append(get_molecular_properties(glycans, placeholder = True))
  if 'terminal' in feature_set:
    bag = [get_terminal_structures(glycan, libr = libr) for glycan in glycans]
    repertoire = set(unwrap(bag))
    repertoire2 = [re.sub(r"\(([ab])(\d)-(\d)\)", r"(\1\2-?)", g) for g in repertoire]
    repertoire2 = set([k for k in repertoire2 if repertoire2.count(k) > 1 and k not in repertoire])
    repertoire.update(repertoire2)
    bag_out = pd.DataFrame([{i: j.count(i) for i in repertoire if '?' not in i} for j in bag])
    shadow_glycans = [[re.sub(r"\(([ab])(\d)-(\d)\)", r"(\1\2-?)", g) for g in b] for b in bag]
    shadow_bag = pd.DataFrame([{i: j.count(i) for i in repertoire if '?' in i} for j in shadow_glycans])
    bag_out = pd.concat([bag_out, shadow_bag], axis = 1).reset_index(drop = True)
    bag_out.index = glycans
    shopping_cart.append(bag_out)
  if condense:
    # Remove motifs that never occur
    temp = pd.concat(shopping_cart, axis = 1)
    return temp.loc[:, (temp != 0).any(axis = 0)]
  else:
    return pd.concat(shopping_cart, axis = 1)


def quantify_motifs(df, glycans, feature_set):
    """Extracts and quantifies motifs for a dataset\n
    | Arguments:
    | :-
    | df (dataframe): dataframe containing relative abundances (each sample one column) [alternative: filepath to .csv]
    | glycans(list): glycans as IUPAC-condensed strings
    | feature_set (list): which feature set to use for annotations, add more to list to expand; default is ['exhaustive','known']; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), and 'chemical' (molecular properties of glycan)
    | Returns:
    | :-
    | Returns a pandas DataFrame with motifs as columns and samples as rows
    """
    if isinstance(df, str):
      df = pd.read_csv(df)
    # Motif extraction
    df_motif = annotate_dataset(glycans,
                                feature_set = feature_set,
                                condense = True)
    collect_dic = {}
    df = df.T
    # Motif quantification
    for c, col in enumerate(df_motif.columns):
      indices = [i for i, x in enumerate(df_motif[col]) if x >= 1]
      temp = df.iloc[:, indices]
      temp.columns = range(temp.columns.size)
      collect_dic[col] = (temp * df_motif.iloc[indices, c].reset_index(drop = True)).sum(axis = 1)
    df = pd.DataFrame(collect_dic)
    return df


def count_unique_subgraphs_of_size_k(graph, size = 2):
  """function to count unique, connected subgraphs of size k (default:disaccharides) occurring in a glycan graph\n
  | Arguments:
  | :-
  | graph (networkx): glycan graph as returned by glycan_to_nxGraph
  | size (int): number of monosaccharides per -saccharide, default:2 (for disaccharides)\n
  | Returns:
  | :-                   
  | Returns dictionary of type k-saccharide:count
  """
  size = size + (size-1)
  paths = nx.all_pairs_shortest_path(graph, cutoff = size-1)
  labels = nx.get_node_attributes(graph, 'string_labels')
  counts = defaultdict(int)
  for _, path_dict in paths:
    for path in path_dict.values():
      if len(path) == size:
        if labels[path[0]][0] not in {'a', 'b', '?', '('}:
          path_sorted = sorted(path)
          path_str = ""
          path_str_list = []
          for index, node in enumerate(path_sorted):
            prefix = ""
            # Check degree and add prefix accordingly
            degree = graph.degree[node]
            if degree == 1 and node > 0 and index > 0 and node != len(graph)-1:
              prefix = "["
            elif degree > 2 or (degree == 2 and node == len(graph)-1):
              prefix = "]"
            path_str_list.append(prefix + labels[node])
          path_str = '('.join(path_str_list)
          path_str = replace_every_second(path_str, '(', ')')
          path_str = path_str[:path_str.index('[')+1].replace('[', '') + path_str[path_str.index('['):] if '[' in path_str else path_str.replace(']', '')
          counts[path_str] += 0.5
  return counts


@rescue_glycans
def get_k_saccharides(glycans, size = 2, libr = None, up_to = False, just_motifs = False):
  """function to retrieve k-saccharides (default:disaccharides) occurring in a list of glycans\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed nomenclature
  | size (int): number of monosaccharides per -saccharide, default:2 (for disaccharides)
  | libr (dict): dictionary of form glycoletter:index
  | up_to (bool): in theory: include -saccharides up to size k; in practice: include monosaccharides; default:False
  | just_motifs (bool): if you only want the motifs as a nested list, no dataframe with counts; default:False\n
  | Returns:
  | :-                 
  | Returns dataframe with k-saccharide counts (columns) for each glycan (rows)
  """
  if any([k in glycans[0] for k in [';', '-D-', 'RES', '=']]):
    raise Exception
  if libr is None:
    libr = lib
  if up_to:
    wga_letter = pd.DataFrame([{i: len(re.findall(rf'{re.escape(i)}(?=\(|$)', g)) for i in get_lib(glycans) if i not in linkages} for g in glycans])
  regex = re.compile(r"\(([ab])(\d)-(\d)\)")
  shadow_glycans = [regex.sub(r"(\1\2-?)", g) for g in glycans]
  libr = expand_lib(libr, shadow_glycans)
  ggraphs = pd.DataFrame([count_unique_subgraphs_of_size_k(glycan_to_nxGraph(g, libr = libr), size = size) for g in glycans])
  ggraphs = ggraphs.drop(columns = [col for col in ggraphs.columns if '?' in col])
  shadow_glycans = pd.DataFrame([count_unique_subgraphs_of_size_k(glycan_to_nxGraph(g, libr = libr), size = size) for g in shadow_glycans])
  org_len = ggraphs.shape[1]
  out_matrix = pd.concat([ggraphs, shadow_glycans], axis = 1).reset_index(drop = True)
  out_matrix = out_matrix.loc[:, ~out_matrix.columns.duplicated()].copy()
  # Throw out redundant columns
  cols = out_matrix.columns.tolist()
  first_half_cols = cols[:org_len]
  second_half_cols = cols[org_len:]
  drop_columns = []
  col_sums = {col: out_matrix[col].sum() for col in cols}
  col_subs = {col: regex.sub(r"\1\2-?", col) for col in cols}
  for col1 in first_half_cols:
    for col2 in second_half_cols:
      if col_sums[col1] == col_sums[col2] and col_subs[col1] == col2:
        drop_columns.append(col2)
  out_matrix = out_matrix.drop(drop_columns, axis = 1)
  if size > 3:
    out_matrix.columns = [remove_unmatched_brackets(g) for g in out_matrix.columns]
  if up_to:
    combined_df= pd.concat([wga_letter, out_matrix], axis = 1).fillna(0).astype(int)
    if just_motifs:
      return combined_df.apply(lambda x: list(combined_df.columns[x > 0]), axis = 1).tolist()
    else:
      return combined_df
  else:
    if just_motifs:
      return out_matrix.apply(lambda x: list(out_matrix.columns[x > 0]), axis = 1).tolist()
    else:
      return out_matrix.fillna(0).astype(int)


def get_terminal_structures(glycan, libr = None):
  """returns terminal structures from all non-reducing ends (monosaccharide+linkage)\n
  | Arguments:
  | :-
  | glycan (string or networkx): glycan in IUPAC-condensed nomenclature or as networkx graph
  | libr (dict): dictionary of form glycoletter:index\n
  | Returns:
  | :-
  | Returns a list of terminal structures (strings)
  """
  if libr is None:
    libr = lib
  ggraph = ensure_graph(glycan, libr = libr)
  nodeDict = dict(ggraph.nodes(data = True))
  return [nodeDict[k]['string_labels']+'('+nodeDict[k+1]['string_labels']+')' for k in list(ggraph.nodes())[:-1] if ggraph.degree[k] == 1 and k+1 in nodeDict.keys() and nodeDict[k]['string_labels'] not in linkages]


def create_correlation_network(df, correlation_threshold):
  """finds clusters of motifs/glycans that correlate to at least the correlation_threshold\n
  | Arguments:
  | :-
  | df (dataframe): dataframe with samples as rows and glycans/motifs as columns
  | correlation_threshold (float): correlation value used as a threshold for clusters\n
  | Returns:
  | :-
  | Returns a list of sets, which correspond to the glycans/motifs that are put in a cluster
  """
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
