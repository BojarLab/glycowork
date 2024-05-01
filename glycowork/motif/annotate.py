import networkx as nx
import pandas as pd
import re
from collections import defaultdict

from glycowork.glycan_data.loader import linkages, motif_list, find_nth, unwrap, replace_every_second, remove_unmatched_brackets
from glycowork.motif.graph import subgraph_isomorphism, generate_graph_features, glycan_to_nxGraph, graph_to_string, ensure_graph
from glycowork.motif.processing import IUPAC_to_SMILES, get_lib, find_isomorphs, rescue_glycans
from glycowork.motif.regex import get_match


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


def annotate_glycan(glycan, motifs = None, termini_list = [], gmotifs = None):
  """searches for known motifs in glycan sequence\n
  | Arguments:
  | :-
  | glycan (string or networkx): glycan in IUPAC-condensed format (or as networkx graph) that has to contain a floating substituent
  | motifs (dataframe): dataframe of glycan motifs (name + sequence), can be used with a list of glycans too; default:motif_list
  | termini_list (list): list of monosaccharide positions (from 'terminal', 'internal', and 'flexible')
  | gmotifs (networkx): precalculated motif graphs for speed-up; default:None\n
  | Returns:
  | :-
  | Returns dataframe with counts of motifs in glycan
  """
  if motifs is None:
    motifs = motif_list
  # Check whether termini are specified
  if not termini_list and isinstance(motifs, pd.DataFrame):
    termini_list = list(map(eval, motifs.termini_spec))
  if gmotifs is None:
    termini = 'provided' if termini_list else 'ignore'
    gmotifs = [glycan_to_nxGraph(g, termini = termini, termini_list = termini_list[i]) for i, g in enumerate(motifs.motif)]
  # Count the number of times each motif occurs in a glycan
  if termini_list:
    ggraph = ensure_graph(glycan, termini = 'calc')
    res = [subgraph_isomorphism(ggraph, gmotifs[k], termini_list = termini_list[k],
                                count = True) for k in range(len(motifs))]*1
  else:
    ggraph = ensure_graph(glycan, termini = 'ignore')
    res = [subgraph_isomorphism(ggraph, gmotifs[k], termini_list = termini_list,
                                count = True) for k in range(len(motifs))]*1
 
  out = pd.DataFrame(columns = motifs.motif_name if isinstance(motifs, pd.DataFrame) else motifs)
  out.loc[0] = res
  out.loc[0] = out.loc[0].astype('int')
  out.index = [glycan] if isinstance(glycan, str) else [graph_to_string(glycan)]
  return out


def get_molecular_properties(glycan_list, verbose = False, placeholder = False):
  """given a list of glycans, uses pubchempy to return various molecular parameters retrieved from PubChem\n
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
def annotate_dataset(glycans, motifs = None, feature_set = ['known'],
                     termini_list = [], condense = False, custom_motifs = []):
  """wrapper function to annotate motifs in list of glycans\n
  | Arguments:
  | :-
  | glycans (list): list of IUPAC-condensed glycan sequences as strings
  | motifs (dataframe): dataframe of glycan motifs (name + sequence); default:motif_list
  | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), \
  |   'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs of size 1), \
  |   'terminal2' (non-reducing end motifs of size 2), 'terminal3' (non-reducing end motifs of size 3), 'custom' (specify your own motifs in custom_motifs), \
  |   and 'chemical' (molecular properties of glycan)
  | termini_list (list): list of monosaccharide/linkage positions (from 'terminal', 'internal', and 'flexible')
  | condense (bool): if True, throws away columns with only zeroes; default:False
  | custom_motifs (list): list of glycan motifs, used if feature_set includes 'custom'; default:empty\n
  | Returns:
  | :-                      
  | Returns dataframe of glycans (rows) and presence/absence of known motifs (columns)
  """
  if any([k in ''.join(glycans) for k in [';', '-D-', 'RES', '=']]):
    raise Exception
  if motifs is None:
    motifs = motif_list
  # Checks whether termini information is provided
  if not termini_list:
    termini_list = list(map(eval, motifs.termini_spec))
  shopping_cart = []
  if 'known' in feature_set:
    termini = 'provided' if termini_list else 'ignore'
    termini_list = termini_list if termini_list else ['ignore']*len(motifs)
    gmotifs = [glycan_to_nxGraph(g, termini = termini, termini_list = termini_list[i]) for i, g in enumerate(motifs.motif)]
    # Counts literature-annotated motifs in each glycan
    shopping_cart.append(pd.concat([annotate_glycan(k, motifs = motifs,
                                                    gmotifs = gmotifs, termini_list = termini_list) for k in glycans], axis = 0))
  if 'custom' in feature_set:
    normal_motifs = [m for m in custom_motifs if not m.startswith('r')]
    gmotifs = list(map(glycan_to_nxGraph, normal_motifs))
    shopping_cart.append(pd.concat([annotate_glycan(k, motifs = normal_motifs, gmotifs = gmotifs) for k in glycans], axis = 0))
    regex_motifs = [m[1:] for m in custom_motifs if m.startswith('r')]
    shopping_cart.append(pd.concat([pd.DataFrame([len(get_match(p, k)) for p in regex_motifs], columns = regex_motifs, index = [k]) for k in glycans], axis = 0))
  if 'graph' in feature_set:
    # Calculates graph features of each glycan
    shopping_cart.append(pd.concat([generate_graph_features(k) for k in glycans], axis = 0))
  if 'exhaustive' in feature_set:
    # Counts disaccharides and monosaccharides in each glycan
    temp = get_k_saccharides(glycans, size = 2, up_to = True)
    temp.index = glycans
    shopping_cart.append(temp)
  if 'chemical' in feature_set:
    shopping_cart.append(get_molecular_properties(glycans, placeholder = True))
  if 'terminal' or 'terminal2' in feature_set:
    bag1, bag2 = [], []
    if 'terminal' in feature_set:
      bag1 = list(map(get_terminal_structures, glycans))
    if 'terminal2' in feature_set:
      bag2 = [get_terminal_structures(glycan, size = 2) for glycan in glycans]
    bag = bag1 + bag2
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
  if 'terminal3' in feature_set:
    temp = get_k_saccharides(glycans, size = 3, terminal = True)
    temp.index = glycans
    shopping_cart.append(temp)
  if condense:
    # Remove motifs that never occur
    temp = pd.concat(shopping_cart, axis = 1)
    return temp.loc[:, (temp != 0).any(axis = 0)]
  else:
    return pd.concat(shopping_cart, axis = 1)


def quantify_motifs(df, glycans, feature_set, custom_motifs = []):
    """Extracts and quantifies motifs for a dataset\n
    | Arguments:
    | :-
    | df (dataframe): dataframe containing relative abundances (each sample one column) [alternative: filepath to .csv or .xlsx]
    | glycans(list): glycans as IUPAC-condensed strings
    | feature_set (list): which feature set to use for annotations, add more to list to expand; default is ['exhaustive','known']; options are: 'known' (hand-crafted glycan features), \
    |   'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), \
    |   'terminal2' (non-reducing end motifs of size 2), 'terminal3' (non-reducing end motifs of size 3), 'custom' (specify your own motifs in custom_motifs), and 'chemical' (molecular properties of glycan)
    | custom_motifs (list): list of glycan motifs, used if feature_set includes 'custom'; default:empty\n
    | Returns:
    | :-
    | Returns a pandas DataFrame with motifs as columns and samples as rows
    """
    if isinstance(df, str):
      df = pd.read_csv(df) if df.endswith(".csv") else pd.read_excel(df)
    # Motif extraction
    df_motif = annotate_dataset(glycans,
                                feature_set = feature_set,
                                condense = True, custom_motifs = custom_motifs)
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


def count_unique_subgraphs_of_size_k(graph, size = 2, terminal = False):
  """function to count unique, connected subgraphs of size k (default:disaccharides) occurring in a glycan graph\n
  | Arguments:
  | :-
  | graph (networkx): glycan graph as returned by glycan_to_nxGraph
  | size (int): number of monosaccharides per -saccharide, default:2 (for disaccharides)
  | terminal (bool): whether to only count terminal subgraphs; default:False\n
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
        if terminal:
          if graph.degree[min(path)] != 1:
            continue
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
def get_k_saccharides(glycans, size = 2, up_to = False, just_motifs = False, terminal = False):
  """function to retrieve k-saccharides (default:disaccharides) occurring in a list of glycans\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed nomenclature
  | size (int): number of monosaccharides per -saccharide, default:2 (for disaccharides)
  | up_to (bool): in theory: include -saccharides up to size k; in practice: include monosaccharides; default:False
  | just_motifs (bool): if you only want the motifs as a nested list, no dataframe with counts; default:False
  | terminal (bool): whether to only count terminal subgraphs; default:False\n
  | Returns:
  | :-                 
  | Returns dataframe with k-saccharide counts (columns) for each glycan (rows)
  """
  if not isinstance(glycans, list):
    raise TypeError("The input has to be a list of glycans")
  if any([k in ''.join(glycans) for k in [';', '-D-', 'RES', '=']]):
    raise Exception
  if up_to:
    wga_letter = pd.DataFrame([{i: len(re.findall(rf'{re.escape(i)}(?=\(|$)', g)) for i in get_lib(glycans) if i not in linkages} for g in glycans])
  regex = re.compile(r"\(([ab])(\d)-(\d)\)")
  shadow_glycans = [regex.sub(r"(\1\2-?)", g) for g in glycans]
  ggraphs = pd.DataFrame([count_unique_subgraphs_of_size_k(glycan_to_nxGraph(g), size = size, terminal = terminal) for g in glycans])
  ggraphs = ggraphs.drop(columns = [col for col in ggraphs.columns if '?' in col])
  shadow_glycans = pd.DataFrame([count_unique_subgraphs_of_size_k(glycan_to_nxGraph(g), size = size, terminal = terminal) for g in shadow_glycans])
  org_len = ggraphs.shape[1]
  out_matrix = pd.concat([ggraphs, shadow_glycans], axis = 1).reset_index(drop = True)
  out_matrix = out_matrix.loc[:, ~out_matrix.columns.duplicated()].copy()
  # Throw out redundant columns
  cols = out_matrix.columns.tolist()
  first_half_cols = cols[:org_len]
  second_half_cols = cols[org_len:]
  drop_columns = []
  regex = re.compile(r"(\([ab])(\d)-(\d)\)")
  col_sums = {col: out_matrix[col].sum() for col in cols}
  col_subs = {col: regex.sub(r"\1\2-?)", col) for col in cols}
  for col1 in first_half_cols:
    for col2 in second_half_cols:
      if col_sums[col1] == col_sums[col2] and col_subs[col1] == col2:
        drop_columns.append(col2)
  out_matrix = out_matrix.drop(drop_columns, axis = 1)
  out_matrix.columns = [remove_unmatched_brackets(g) for g in out_matrix.columns]
  out_matrix = out_matrix.groupby(by = out_matrix.columns, axis = 1).sum()
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


def get_terminal_structures(glycan, size = 1):
  """returns terminal structures from all non-reducing ends (monosaccharide+linkage)\n
  | Arguments:
  | :-
  | glycan (string or networkx): glycan in IUPAC-condensed nomenclature or as networkx graph
  | size (int): how large the extracted motif should be in terms of monosaccharides (for now 1 or 2 are supported; \
  |   if you want to go higher use get_k_saccharides with terminal = True); default:1\n
  | Returns:
  | :-
  | Returns a list of terminal structures (strings)
  """
  ggraph = ensure_graph(glycan)
  nodeDict = dict(ggraph.nodes(data = True))
  temp =  [nodeDict[k]['string_labels']+'('+nodeDict[k+1]['string_labels']+')' + \
   ''.join([nodeDict.get(k+1+j+i, {'string_labels': ''})['string_labels']+'('+nodeDict.get(k+2+j+i, {'string_labels': ''})['string_labels']+')' \
            for i, j in enumerate(range(1, size))]) for k in list(ggraph.nodes())[:-1] if \
          ggraph.degree[k] == 1 and k+1 in nodeDict.keys() and nodeDict[k]['string_labels'] not in linkages]
  return [g.replace('()', '') for g in temp]


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


def group_glycans_core(glycans, p_values):
  """group O-glycans based on core structure\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed nomenclature
  | p_values (list): list of associated p-values\n
  | Returns:
  | :-
  | Returns dictionaries of group : glycans and group : p-values
  """
  temp = {glycans[k]: p_values[k] for k in range(len(glycans))}
  grouped_glycans, grouped_p_values = {}, {}
  grouped_glycans["core2"] = [g for g in glycans if any([subgraph_isomorphism(g, sub_g) for sub_g in ["GlcNAc(b1-6)GalNAc", "GlcNAcOS(b1-6)GalNAc"]])]
  grouped_glycans["core1"] = [g for g in glycans if any([subgraph_isomorphism(g, sub_g) for sub_g in ["Gal(b1-3)GalNAc", "GalOS(b1-3)GalNAc"]]) and not g in grouped_glycans["core2"]]
  grouped_glycans["rest"] = [g for g in glycans if g not in grouped_glycans["core2"] and g not in grouped_glycans["core1"]]
  grouped_p_values["core2"] = [temp[g] for g in grouped_glycans["core2"]]
  grouped_p_values["core1"] = [temp[g] for g in grouped_glycans["core1"]]
  if len(grouped_glycans["rest"]) > 0:
    grouped_p_values["rest"] = [temp[g] for g in grouped_glycans["rest"]]
  else:
    del grouped_glycans["rest"]
  return grouped_glycans, grouped_p_values


def group_glycans_sia_fuc(glycans, p_values):
  """group glycans based on whether they contain sialic acid or fucose\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed nomenclature
  | p_values (list): list of associated p-values\n
  | Returns:
  | :-
  | Returns dictionaries of group : glycans and group : p-values
  """
  temp = {glycans[k]: p_values[k] for k in range(len(glycans))}
  grouped_glycans, grouped_p_values = {}, {}
  grouped_glycans["SiaFuc"] = [g for g in glycans if "Neu" in g and "Fuc" in g]
  grouped_glycans["Sia"] = [g for g in glycans if "Neu" in g and g not in grouped_glycans["SiaFuc"]]
  grouped_glycans["Fuc"] = [g for g in glycans if "Fuc" in g and g not in grouped_glycans["SiaFuc"]]
  grouped_glycans["rest"] = [g for g in glycans if g not in grouped_glycans["SiaFuc"] and g not in grouped_glycans["Sia"] and g not in grouped_glycans["Fuc"]]
  grouped_p_values["SiaFuc"] = [temp[g] for g in grouped_glycans["SiaFuc"]]
  grouped_p_values["Sia"] = [temp[g] for g in grouped_glycans["Sia"]]
  grouped_p_values["Fuc"] = [temp[g] for g in grouped_glycans["Fuc"]]
  grouped_p_values["rest"] = [temp[g] for g in grouped_glycans["rest"]]
  for grp in ["SiaFuc", "Sia", "Fuc", "rest"]:
    if not grouped_glycans[grp]:
      del grouped_glycans[grp]
      del grouped_p_values[grp]
  return grouped_glycans, grouped_p_values


def group_glycans_N_glycan_type(glycans, p_values):
  """group glycans based on complex/hybrid/high-mannose/rest for N-glycans\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed nomenclature
  | p_values (list): list of associated p-values\n
  | Returns:
  | :-
  | Returns dictionaries of group : glycans and group : p-values
  """
  temp = {glycans[k]: p_values[k] for k in range(len(glycans))}
  grouped_glycans, grouped_p_values = {}, {}
  grouped_glycans["complex"] = [g for g in glycans if any([subgraph_isomorphism(g, sub_g) for sub_g in ["GlcNAc(b1-2)Man(a1-6)", "GlcNAc(b1-2)Man(a1-?)[GlcNAc(b1-2)Man(a1-?)]Man"]])]
  grouped_glycans["hybrid"] = [g for g in glycans if any([subgraph_isomorphism(g, sub_g) for sub_g in ["GlcNAc(b1-2)Man(a1-3)[Man(a1-?)Man(a1-6)]Man"]]) and g not in grouped_glycans["complex"]]
  grouped_glycans["high_man"] = [g for g in glycans if g.count("Man") > 3 and g not in grouped_glycans["hybrid"] and g not in grouped_glycans["complex"]]
  grouped_glycans["rest"] = [g for g in glycans if g not in grouped_glycans["complex"] and g not in grouped_glycans["hybrid"] and g not in grouped_glycans["high_man"]]
  grouped_p_values["complex"] = [temp[g] for g in grouped_glycans["complex"]]
  grouped_p_values["hybrid"] = [temp[g] for g in grouped_glycans["hybrid"]]
  grouped_p_values["high_man"] = [temp[g] for g in grouped_glycans["high_man"]]
  grouped_p_values["rest"] = [temp[g] for g in grouped_glycans["rest"]]
  for grp in ["complex", "hybrid", "high_man", "rest"]:
    if not grouped_glycans[grp]:
      del grouped_glycans[grp]
      del grouped_p_values[grp]
  return grouped_glycans, grouped_p_values


def load_lectin_lib():
  """create dictionary of lectin specificities for get_lectin_array\n
  | Returns:
  | :-
  | Returns a dictionary of index: Lectin object, with its specificities
  """
  from glycowork.glycan_data.loader import lectin_specificity
  lectin_lib = {}
  for i, row in enumerate(lectin_specificity.itertuples()):
    specificity_dict = {"primary": row.specificity_primary, "secondary": row.specificity_secondary, "negative": row.specificity_negative}
    lectin_lib[i] = Lectin(abbr = row.abbreviation.split(","), name = row.name.split(","), specificity = specificity_dict,
                           species = row.species, reference = row.reference, notes = row.notes)
  return lectin_lib


class Lectin():
  def __init__(self, abbr: list, name: list, # A lectin may have multiple names and abbreviations
                 specificity: dict = {"primary": None, "secondary": None, "negative": None}, 
                 species: str = "", reference: str = "", notes: str = ""):
    self.abbr = abbr
    self.name = name
    self.species = species
    self.reference = reference
    self.notes = notes
    self.specificity = specificity

  def get_all_binding_motifs_count(self):
      return len(self.specificity["primary"]) + \
               (len(self.specificity["secondary"]) if self.specificity["secondary"] else 0)

  def get_all_binding_motifs(self):
      return list(self.specificity["primary"].keys()) + list(self.specificity["secondary"].keys())

  def show_info(self):
      print(f"Name(s): {'; '.join(self.name)}\n"+\
      f"Abbreviation(s): {'; '.join(self.abbr)}\n"+\
      f"Species: {self.species}\n" + \
                       ''.join(f"Specificity ({key}): {'; '.join(value) if value else 'no information'}\n" 
                               for key, value in self.specificity.items()) + \
                       f"Reference: {self.reference}\nnotes: {self.notes}\n")
    
  def check_binding(self, glycan: str):
    """Check whether a glycan binds to this lectin. 
    Returns an integer. 
    0: no binding; 
    1: binds to lectin's primary recognition motif;
    2: binds to lectin's seconary recognition motif;
    """
    for key in ["negative", "primary", "secondary"]:
      motifs = self.specificity.get(key, [])
      if motifs:
        for motif, termini in motifs.items():
          if subgraph_isomorphism(glycan, motif, termini_list = termini):
            return 0 if key == "negative" else (1 if key == "primary" else 2)
    return 0


def create_lectin_and_motif_mappings(lectin_list, lectin_lib):
  """Create a collection of lectins that can be used for motif scoring and all possible binding motifs.\n
  | Arguments:
  | :-
  | lectin_list (list): list of lectins used in this experiment
  | lectin_lib (dict): dictionary of type index : Lectin object\n
  | Returns:
  | :-
  | 1. A useable lectin mapping dictionary that maps a user input lectin (string) to the index of a lectin in library --- {lectin: index}.
  | 2. A motif mapping dictionary that maps a glycan motif to a dictionary of its corresponding lectins, each of which is assigned a weight class that indicates the level of evidence this lectin provides when scoring for this motif --- {motif: {lectin: weight_class}}.
  | The weight class depends on the match between the motif and the specificity profile of the lectin.
  | class 0: exact match between motif and primary specificity
  | class 1: exact match between motif and secondary specificity
  | class 2: motif encompasses primary specificity but not negative specificity
  | class 3: motif encompasses secondary specificity but not negative specificity
  """
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


def lectin_motif_scoring(useable_lectin_mapping, motif_mapping, lectin_score_dict, lectin_lib, idf,
                  class_weight_dict = {0: 1, 1: 0.5, 2: 0.5, 3: 0.25}, specificity_weight_dict = {"<3": 1, "3-4": 0.75, ">4": 0.5},
                  duplicate_lectin_penalty_factor = 0.8):
  """Scores motifs based on observed lectin binding\n
  | Arguments:
  | :-
  | useable_lectin_mapping (dict): returned from create_lectin_and_motif_mappings()
  | motif_mapping (dict): returned from create_lectin_and_motif_mappings()
  | lectin_score_dict (dict): dictionary of type lectin : effect_size
  | lectin_lib (dict): dictionary of type index : lectin
  | idf (float): variance of each lectin across groups
  | class_weight_dict (dict, optional): dictionary of type lectin class : score weighting factor
  | specificity_weight_dict (dict, optional): dictionary of type binding specificities : penalty weighting factor
  | duplicate_lectin_penalty_factor (float, optional): penalty for having several of the same lectin; default:0.8\n
  | Returns:
  | :-
  | Returns a dataframe with motifs, supporting lectins, observed change direction, and score
  """
  output = []
  useable_lectin_count = {k: list(useable_lectin_mapping.values()).count(v) for k, v in useable_lectin_mapping.items()}
  max_motifs = max([lectin_lib[k].get_all_binding_motifs_count() for k in useable_lectin_mapping.values()]) + 1
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
