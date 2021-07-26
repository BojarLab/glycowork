import networkx as nx
import pandas as pd
import itertools
import re

from glycowork.glycan_data.loader import lib, motif_list, unwrap
from glycowork.motif.graph import subgraph_isomorphism, generate_graph_features, glycan_to_nxGraph, try_string_conversion
from glycowork.motif.tokenization import motif_matrix

def estimate_lower_bound(glycans, motifs):
  """searches for motifs which are present in at least glycan; not 100% exact but useful for speedup\n
  | Arguments:
  | :-
  | glycans (string): list of IUPAC-condensed glycan sequences
  | motifs (dataframe): dataframe of glycan motifs (name + sequence)\n
  | Returns:
  | :-
  | Returns motif dataframe, only retaining motifs that are present in glycans
  """
  glycans_a = [k.replace('[','').replace(']','') for k in glycans]
  glycans_b = [re.sub('\[[^[]\]', '', k) for k in glycans]
  glycans = glycans_a + glycans_b
  glycans = '_'.join(glycans)
  motif_ish = [k.replace('[','').replace(']','') for k in motifs.motif.values.tolist()]
  idx = [k for k, j in enumerate(motif_ish) if j in glycans]
  motifs = motifs.iloc[idx, :].reset_index(drop = True)
  return motifs

def annotate_glycan(glycan, motifs = None, libr = None, extra = 'termini',
                    wildcard_list = [], termini_list = []):
  """searches for known motifs in glycan sequence\n
  | Arguments:
  | :-
  | glycan (string): IUPAC-condensed glycan sequence
  | motifs (dataframe): dataframe of glycan motifs (name + sequence); default:motif_list
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset
  | extra (string): 'ignore' skips this, 'wildcards' allows for wildcard matching', and 'termini' allows for positional matching; default:'termini'
  | wildcard_list (list): list of wildcard names (such as 'bond', 'Hex', 'HexNAc', 'Sia')
  | termini_list (list): list of monosaccharide/linkage positions (from 'terminal','internal', and 'flexible')\n
  | Returns:
  | :-
  | Returns dataframe with absence/presence of motifs in glycan
  """
  if motifs is None:
    motifs = motif_list
  if extra == 'termini':
    if len(termini_list) < 1:
      termini_list = motifs.termini_spec.values.tolist()
      termini_list = [eval(k) for k in termini_list]
  if libr is None:
    libr = lib
  if extra == 'termini':
    res = [subgraph_isomorphism(glycan, motifs.motif.values.tolist()[k], libr = libr,
                              extra = extra,
                              wildcard_list = wildcard_list,
                              termini_list = termini_list[k]) for k in range(len(motifs))]*1
  else:
    res = [subgraph_isomorphism(glycan, motifs.motif.values.tolist()[k], libr = libr,
                              extra = extra,
                              wildcard_list = wildcard_list,
                              termini_list = termini_list) for k in range(len(motifs))]*1
      
  out = pd.DataFrame(columns = motifs.motif_name.values.tolist())
  out.loc[0] = res
  out.loc[0] = out.loc[0].astype('int')
  out.index = [glycan]
  return out

def annotate_dataset(glycans, motifs = None, libr = None,
                     feature_set = ['known'], extra = 'termini',
                     wildcard_list = [], termini_list = [],
                     condense = False, estimate_speedup = False):
  """wrapper function to annotate motifs in list of glycans\n
  | Arguments:
  | :-
  | glycans (list): list of IUPAC-condensed glycan sequences as strings
  | motifs (dataframe): dataframe of glycan motifs (name + sequence); default:motif_list
  | libr (list): sorted list of unique glycoletters observed in the glycans of our data; default:lib
  | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'known'; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans) and 'exhaustive' (all mono- and disaccharide features)
  | extra (string): 'ignore' skips this, 'wildcards' allows for wildcard matching', and 'termini' allows for positional matching; default:'termini'
  | wildcard_list (list): list of wildcard names (such as 'bond', 'Hex', 'HexNAc', 'Sia')
  | termini_list (list): list of monosaccharide/linkage positions (from 'terminal','internal', and 'flexible')
  | condense (bool): if True, throws away columns with only zeroes; default:False
  | estimate_speedup (bool): if True, pre-selects motifs for those which are present in glycans, not 100% exact; default:False\n
  | Returns:
  | :-                               
  | Returns dataframe of glycans (rows) and presence/absence of known motifs (columns)
  """
  if motifs is None:
    motifs = motif_list
  if libr is None:
    libr = lib
  if estimate_speedup:
    motifs = estimate_lower_bound(glycans, motifs)
  if extra == 'termini':
    if len(termini_list) < 1:
      termini_list = motifs.termini_spec.values.tolist()
      termini_list = [eval(k) for k in termini_list]
  shopping_cart = []
  if 'known' in feature_set:
    shopping_cart.append(pd.concat([annotate_glycan(k, motifs = motifs, libr = libr,
                                                    extra = extra,
                                                    wildcard_list = wildcard_list,
                                                    termini_list = termini_list) for k in glycans], axis = 0))
  if 'graph' in feature_set:
    shopping_cart.append(pd.concat([generate_graph_features(k, libr = libr) for k in glycans], axis = 0))
  if 'exhaustive' in feature_set:
    temp = motif_matrix(pd.DataFrame({'glycans':glycans, 'labels':range(len(glycans))}),
                                                   'glycans', 'labels', libr = libr)
    temp.index = glycans
    temp.drop(['labels'], axis = 1, inplace = True)
    shopping_cart.append(temp)
  if condense:
    temp = pd.concat(shopping_cart, axis = 1)
    return temp.loc[:, (temp != 0).any(axis = 0)]
  else:
    return pd.concat(shopping_cart, axis = 1)

def get_trisaccharides(glycan, libr = None):
  """function to retrieve trisaccharides occurring in a glycan (probably incomplete)\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed nomenclature
  | libr (list): sorted list of unique glycoletters observed in the glycans of our data; default:lib\n
  | Returns:
  | :-                               
  | Returns list of trisaccharides in glycan in IUPAC-condensed nomenclature
  """
  if libr is None:
    libr = lib
  ggraph = glycan_to_nxGraph(glycan, libr = libr)
  ra = list(ggraph.nodes())
  subgraphs = [ggraph.subgraph(k) for k in itertools.combinations(ra, 5)]
  subgraphs = [k for k in subgraphs if nx.is_connected(k)]
  forbidden_list = ['2(', '3(', '4(', '6(', ')(']
  for k in range(len(subgraphs)):
      if list(subgraphs[k].nodes())[0] == 2:
        subgraphs[k] = nx.relabel_nodes(subgraphs[k], {j:j-2 for j in list(subgraphs[k].nodes())})
  t_subgraphs = [try_string_conversion(k, libr = libr) for k in subgraphs]
  t_subgraphs = [k for k in t_subgraphs if k is not None]
  t_subgraphs = [k for k in t_subgraphs if k[0] != '(']
  t_subgraphs = [k if k[0] != '[' else k.replace('[','',1).replace(']','',1) for k in t_subgraphs]
  t_subgraphs = [k for k in t_subgraphs if not any([m in k for m in forbidden_list])]
  return t_subgraphs
