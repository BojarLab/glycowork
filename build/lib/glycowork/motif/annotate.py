import pubchempy as pcp
import networkx as nx
import pandas as pd
import itertools
import re

from glycowork.glycan_data.loader import lib, linkages, motif_list, find_nth, unwrap
from glycowork.motif.graph import subgraph_isomorphism, generate_graph_features, glycan_to_nxGraph, graph_to_string, ensure_graph
from glycowork.motif.processing import IUPAC_to_SMILES, get_lib, find_isomorphs

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
  #get different string representations of the same glycan
  ss = find_isomorphs(glycan)
  coll = []
  #for each string representation, search and remove branches
  for iso in ss:
    if bool(re.search('\[[^\[\]]+\[[^\[\]]+\][^\]]+\]', iso)):
      b_re = re.sub('\[[^\[\]]+\[[^\[\]]+\][^\]]+\]', '', iso)
      b_re = re.sub('\[[^\]]+\]', '', b_re)
    else:
      b_re = re.sub('\[[^\]]+\]', '', iso)
    #from this linear glycan part, chunk away disaccharides and re-format them
    for i in [iso, b_re]:
      b = i.split('(')
      b = unwrap([k.split(')') for k in b])
      b = ['*'.join(b[i:i+3]) for i in range(0, len(b) - 2, 2)]
      b = [k for k in b if (re.search('\*\[', k) is None and re.search('\*\]\[', k) is None)]
      b = [k.strip('[').strip(']') for k in b]
      b = [k.replace('[', '').replace(']', '') for k in b]
      b = [k[:find_nth(k, '*', 1)] + '(' + k[find_nth(k, '*', 1)+1:] for k in b]
      coll += [k[:find_nth(k, '*', 1)] + ')' + k[find_nth(k, '*', 1)+1:] for k in b]
  return list(set(coll))

def motif_matrix(glycans):
  """generates dataframe with counted glycoletters and disaccharides in glycans\n
  | Arguments:
  | :-
  | glycans (list): list of IUPAC-condensed glycan sequences as strings\n
  | Returns:
  | :-
  | Returns dataframe with glycoletter + disaccharide counts (columns) for each glycan (rows)
  """
  shadow_glycans = [re.sub(r"\(([ab])(\d)-(\d)\)", r"(\1\2-?)", g) for g in glycans]
  out_matrix = []
  for c,gs in enumerate([glycans, shadow_glycans]):
    #get all disaccharides
    wga_di = [link_find(i) for i in gs]
    #collect disaccharide repertoire
    if c == 0:
      libr = {k for k in get_lib(gs) if '?' not in k}
      lib_di = {k for k in set(unwrap(wga_di)) if '?' not in k}
    else:
      libr = {k for k in get_lib(gs) if '?' in k}
      lib_di = set(unwrap(wga_di))
    #count each disaccharide in each glycan
    wga_di_out = pd.DataFrame([{i:j.count(i) if i in j else 0 for i in lib_di} for j in wga_di])
    #counts glycoletters in each glycan
    wga_letter = pd.DataFrame([{i:g.count(i) if i in g else 0 for i in libr} for g in gs])
    out_matrix.append(pd.concat([wga_letter, wga_di_out], axis = 1))
  org_len = out_matrix[0].shape[1]
  out_matrix = pd.concat(out_matrix, axis = 1).reset_index(drop = True)
  out_matrix = out_matrix.loc[:,~out_matrix.columns.duplicated()].copy()
  return out_matrix.drop([col2 for col1,col2 in itertools.product(out_matrix.columns.tolist()[:org_len], out_matrix.columns.tolist()[org_len:]) if out_matrix[col1].sum()==out_matrix[col2].sum() and re.sub(r"([ab])(\d)-(\d)", r"\1\2-?", col1) == col2], axis = 1)

def estimate_lower_bound(glycans, motifs):
  """searches for motifs which are present in at least one glycan; not 100% exact but useful for speedup\n
  | Arguments:
  | :-
  | glycans (string): list of IUPAC-condensed glycan sequences
  | motifs (dataframe): dataframe of glycan motifs (name + sequence)\n
  | Returns:
  | :-
  | Returns motif dataframe, only retaining motifs that are present in glycans
  """
  #pseudo-linearize glycans by removing branch boundaries + branches altogether
  glycans_a = [k.replace('[','').replace(']','') for k in glycans]
  glycans_b = [re.sub('\[[^[]\]', '', k) for k in glycans]
  glycans = glycans_a + glycans_b
  #convert everything to one giant string
  glycans = '_'.join(glycans)
  #pseudo-linearize motifs
  motif_ish = [k.replace('[','').replace(']','') for k in motifs.motif]
  #check if a motif occurs in giant string
  idx = [k for k, j in enumerate(motif_ish) if j in glycans]
  return motifs.iloc[idx, :].reset_index(drop = True)

def annotate_glycan(glycan, motifs = None, libr = None, extra = 'termini',
                    wildcard_list = [], termini_list = []):
  """searches for known motifs in glycan sequence\n
  | Arguments:
  | :-
  | glycan (string or networkx): glycan in IUPAC-condensed format (or as networkx graph) that has to contain a floating substituent
  | motifs (dataframe): dataframe of glycan motifs (name + sequence); default:motif_list
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset
  | extra (string): 'ignore' skips this, 'wildcards' allows for wildcard matching', and 'termini' allows for positional matching; will only be handled with string input; default:'termini'
  | wildcard_list (list): list of wildcard names (such as '?1-?', 'Hex', 'HexNAc', 'Sia')
  | termini_list (list): list of monosaccharide/linkage positions (from 'terminal', 'internal', and 'flexible')\n
  | Returns:
  | :-
  | Returns dataframe with counts of motifs in glycan
  """
  if motifs is None:
    motifs = motif_list
  #check whether termini are specified
  if extra == 'termini':
    if len(termini_list) < 1:
      termini_list = [eval(k) for k in motifs.termini_spec]
  if libr is None:
    libr = lib
  #count the number of times each motif occurs in a glycan
  if extra == 'termini':
    ggraph = ensure_graph(glycan, libr = libr, termini = 'calc')
    res = [subgraph_isomorphism(ggraph, motifs.motif[k], libr = libr,
                              extra = extra,
                              wildcard_list = wildcard_list,
                              termini_list = termini_list[k],
                                count = True) for k in range(len(motifs))]*1
  else:
    ggraph = ensure_graph(glycan, libr = libr, termini = 'ignore')
    res = [subgraph_isomorphism(ggraph, motifs.motif[k], libr = libr,
                              extra = extra,
                              wildcard_list = wildcard_list,
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

def get_molecular_properties(glycan_list, verbose = False, placeholder = False) :
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
  smiles_list = IUPAC_to_SMILES(glycan_list)
  if placeholder:
    dummy = IUPAC_to_SMILES(['Glc'])[0]
  compounds_list = []
  failed_requests = []
  for s in smiles_list:
    try:
      c = pcp.get_compounds(s, 'smiles')[0]
      if c.cid == None:
        if placeholder:
          compounds_list.append(pcp.get_compounds(dummy, 'smiles')[0])
        else:
          failed_requests.append(s)
      else:
        compounds_list.append(c)
    except:
      failed_requests.append(s)
  if verbose == True and len(failed_requests) >= 1:
    print('The following SMILES were not found on PubChem:')
    for failed in failed_requests:
      print(failed)
  df = pcp.compounds_to_frame(compounds_list, properties = ['molecular_weight','xlogp',
                                                            'charge','exact_mass','monoisotopic_mass','tpsa','complexity',
                                                            'h_bond_donor_count','h_bond_acceptor_count',
                                                            'rotatable_bond_count','heavy_atom_count','isotope_atom_count','atom_stereo_count',
                                                            'defined_atom_stereo_count','undefined_atom_stereo_count', 
                                                            'bond_stereo_count','defined_bond_stereo_count',
                                                            'undefined_bond_stereo_count', 'covalent_unit_count'])
  df = df.reset_index(drop = True)
  df.index = glycan_list
  return df

def annotate_dataset(glycans, motifs = None,
                     feature_set = ['known'], extra = 'termini',
                     wildcard_list = [], termini_list = [],
                     condense = False, estimate_speedup = False):
  """wrapper function to annotate motifs in list of glycans\n
  | Arguments:
  | :-
  | glycans (list): list of IUPAC-condensed glycan sequences as strings
  | motifs (dataframe): dataframe of glycan motifs (name + sequence); default:motif_list
  | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans), 'exhaustive' (all mono- and disaccharide features), 'terminal' (non-reducing end motifs), and 'chemical' (molecular properties of glycan)
  | extra (string): 'ignore' skips this, 'wildcards' allows for wildcard matching', and 'termini' allows for positional matching; default:'termini'
  | wildcard_list (list): list of wildcard names (such as '?1-?', 'Hex', 'HexNAc', 'Sia')
  | termini_list (list): list of monosaccharide/linkage positions (from 'terminal', 'internal', and 'flexible')
  | condense (bool): if True, throws away columns with only zeroes; default:False
  | estimate_speedup (bool): if True, pre-selects motifs for those which are present in glycans, not 100% exact; default:False\n
  | Returns:
  | :-                               
  | Returns dataframe of glycans (rows) and presence/absence of known motifs (columns)
  """
  if motifs is None:
    motifs = motif_list
  libr = get_lib(glycans + motifs.motif.values.tolist())
  #non-exhaustive speed-up that should only be used if necessary
  if estimate_speedup:
    motifs = estimate_lower_bound(glycans, motifs)
  #checks whether termini information is provided
  if extra == 'termini':
    if len(termini_list) < 1:
      termini_list = [eval(k) for k in motifs.termini_spec]
  shopping_cart = []
  if 'known' in feature_set:
    #counts literature-annotated motifs in each glycan
    shopping_cart.append(pd.concat([annotate_glycan(k, motifs = motifs, libr = libr,
                                                    extra = extra,
                                                    wildcard_list = wildcard_list,
                                                    termini_list = termini_list) for k in glycans], axis = 0))
  if 'graph' in feature_set:
    #calculates graph features of each glycan
    shopping_cart.append(pd.concat([generate_graph_features(k, libr = libr) for k in glycans], axis = 0))
  if 'exhaustive' in feature_set:
    #counts disaccharides and glycoletters in each glycan
    temp = motif_matrix(glycans)
    temp.index = glycans
    shopping_cart.append(temp)
  if 'chemical' in feature_set:
    shopping_cart.append(get_molecular_properties(glycans, placeholder = True))
  if 'terminal' in feature_set:
    bag = [get_terminal_structures(glycan, libr = libr) for glycan in glycans]
    repertoire = set(unwrap(bag))
    bag_out = pd.DataFrame([{i:j.count(i) for i in repertoire} for j in bag])
    if '?' in ''.join(repertoire):
      shadow_glycans = [[re.sub(r"\(([ab])(\d)-(\d)\)", r"(\1\2-?)", g) for g in b] for b in bag]
      shadow_bag = pd.DataFrame([{i:j.count(i) for i in repertoire if '?' in i} for j in shadow_glycans])
      bag_out = pd.concat([bag_out, shadow_bag], axis = 1).reset_index(drop = True)
    bag_out.index = glycans
    shopping_cart.append(bag_out)
  if condense:
    #remove motifs that never occur
    temp = pd.concat(shopping_cart, axis = 1)
    return temp.loc[:, (temp != 0).any(axis = 0)]
  else:
    return pd.concat(shopping_cart, axis = 1)

def get_k_saccharides(glycan, size = 3, libr = None):
  """function to retrieve k-saccharides (default:trisaccharides) occurring in a glycan\n
  | Arguments:
  | :-
  | glycan (string or networkx): glycan in IUPAC-condensed nomenclature or as networkx graph
  | size (int): number of monosaccharides per -saccharide, default:3 (for trisaccharides)
  | libr (list): sorted list of unique glycoletters observed in the glycans of our data; default:lib\n
  | Returns:
  | :-                               
  | Returns list of trisaccharides in glycan in IUPAC-condensed nomenclature
  """
  if libr is None:
    libr = lib
  size = size + (size-1)
  glycan = ensure_graph(glycan, libr = libr)
  if len(glycan.nodes()) < size:
    return []
  # initialize a list to store the subgraphs
  subgraphs = set()

  # define a recursive function to traverse the graph
  def traverse(node, path):
    # add the current node to the path
    path.append(node)

    # check if the path has the desired size
    if len(path) == size:
      # add the path as a subgraph to the list of subgraphs
      subgraph = glycan.subgraph(path).copy()
      subgraphs.add(subgraph)

    # iterate over the neighbors of the current node
    for neighbor in glycan[node]:
      # check if the neighbor is already in the path
      if neighbor not in path:
        # traverse the neighbor
        traverse(neighbor, path)

    # remove the current node from the path
    path.pop()

  # iterate over the nodes in the graph
  for node in glycan.nodes():
    # traverse the graph starting from the current node
    traverse(node, [])

  # return the list of subgraphs
  subgraphs = [nx.relabel_nodes(k, {list(k.nodes())[n]:n for n in range(len(k.nodes()))}) for k in subgraphs]
  subgraphs = [graph_to_string(k) for k in subgraphs]
  subgraphs = [k for k in subgraphs if k[0] not in ['a', 'b', '?']]
  return list(set(subgraphs))

def get_terminal_structures(glycan, libr = None):
  """returns terminal structures from all non-reducing ends (monosaccharide+linkage)\n
  | Arguments:
  | :-
  | glycan (string or networkx): glycan in IUPAC-condensed nomenclature or as networkx graph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns a list of terminal structures (strings)
  """
  if libr is None:
    libr = lib
  ggraph = ensure_graph(glycan, libr = libr)
  nodeDict = dict(ggraph.nodes(data = True))
  return [nodeDict[k]['string_labels']+'('+nodeDict[k+1]['string_labels']+')' for k in ggraph.nodes() if ggraph.degree[k] == 1 and k != max(list(ggraph.nodes()))]
