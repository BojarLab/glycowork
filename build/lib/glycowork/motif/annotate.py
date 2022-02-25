import networkx as nx
import pandas as pd
import itertools
import re

from glycowork.glycan_data.loader import lib, linkages, motif_list, find_nth, unwrap
from glycowork.motif.graph import subgraph_isomorphism, generate_graph_features, glycan_to_nxGraph, try_string_conversion, compare_glycans
from glycowork.motif.processing import small_motif_find


def convert_to_counts_glycoletter(glycan, libr = None):
  """counts the occurrence of glycoletters in glycan\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset\n
  | Returns:
  | :-
  | Returns dictionary with counts per glycoletter in a glycan
  """
  if libr is None:
    libr = lib
  letter_dict = dict.fromkeys(libr, 0)
  glycan = small_motif_find(glycan).split('*')
  for i in libr:
    letter_dict[i] = glycan.count(i)
  return letter_dict

def glycoletter_count_matrix(glycans, target_col, target_col_name, libr = None):
  """creates dataframe of counted glycoletters in glycan list\n
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed format as strings
  | target_col (list or pd.Series): label columns used for prediction
  | target_col_name (string): name for target_col
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset\n
  | Returns:
  | :-
  | Returns dataframe with glycoletter counts (columns) for every glycan (rows)
  """
  if libr is None:
    libr = lib
  counted_glycans = [convert_to_counts_glycoletter(i, libr) for i in glycans]
  out = pd.DataFrame(counted_glycans)
  out[target_col_name] = target_col
  return out

def find_isomorphs(glycan):
  """returns a set of isomorphic glycans by swapping branches etc.\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format\n
  | Returns:
  | :-
  | Returns list of unique glycan notations (strings) for a glycan in IUPAC-condensed
  """
  out_list = [glycan]
  #starting branch swapped with next side branch
  if '[' in glycan and glycan.index('[') > 0 and not bool(re.search('\[[^\]]+\[', glycan)):
    glycan2 = re.sub('^(.*?)\[(.*?)\]', r'\2[\1]', glycan, 1)
    out_list.append(glycan2)
  #double branch swap
  temp = []
  for k in out_list:
    if '][' in k:
      glycan3 = re.sub('(\[.*?\])(\[.*?\])', r'\2\1', k)
      temp.append(glycan3)
  #starting branch swapped with next side branch again to also include double branch swapped isomorphs
  temp2 = []
  for k in temp:
    if '[' in k and k.index('[') > 0 and not bool(re.search('\[[^\]]+\[', k)):
      glycan4 = re.sub('^(.*?)\[(.*?)\]', r'\2[\1]', k, 1)
      temp2.append(glycan4)
  return list(set(out_list + temp + temp2))

def link_find(glycan):
  """finds all disaccharide motifs in a glycan sequence using its isomorphs\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format\n
  | Returns:
  | :-
  | Returns list of unique disaccharides (strings) for a glycan in IUPAC-condensed
  """
  ss = find_isomorphs(glycan)
  coll = []
  for iso in ss:
    if bool(re.search('\[[^\[\]]+\[[^\[\]]+\][^\]]+\]', iso)):
      b_re = re.sub('\[[^\[\]]+\[[^\[\]]+\][^\]]+\]', '', iso)
      b_re = re.sub('\[[^\]]+\]', '', b_re)
    else:
      b_re = re.sub('\[[^\]]+\]', '', iso)
    for i in [iso, b_re]:
      b = i.split('(')
      b = [k.split(')') for k in b]
      b = [item for sublist in b for item in sublist]
      b = ['*'.join(b[i:i+3]) for i in range(0, len(b) - 2, 2)]
      b = [k for k in b if (re.search('\*\[', k) is None and re.search('\*\]\[', k) is None)]
      b = [k.strip('[') for k in b]
      b = [k.strip(']') for k in b]
      b = [k.replace('[', '') for k in b]
      b = [k.replace(']', '') for k in b]
      b = [k[:find_nth(k, '*', 1)] + '(' + k[find_nth(k, '*', 1)+1:] for k in b]
      b = [k[:find_nth(k, '*', 1)] + ')' + k[find_nth(k, '*', 1)+1:] for k in b]
      coll += b
  return list(set(coll))

def motif_matrix(df, glycan_col_name, label_col_name, libr = None):
  """generates dataframe with counted glycoletters and disaccharides in glycans\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing glycan sequences and labels
  | glycan_col_name (string): column name for glycan sequences
  | label_col_name (string): column name for labels; string
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset\n
  | Returns:
  | :-
  | Returns dataframe with glycoletter + disaccharide counts (columns) for each glycan (rows)
  """
  if libr is None:
    libr = lib
  matrix_list = []
  di_dics = []
  wga_di = [link_find(i) for i in df[glycan_col_name].values.tolist()]
  lib_di = list(sorted(list(set([item for sublist in wga_di for item in sublist]))))
  for j in wga_di:
    di_dict = dict.fromkeys(lib_di, 0)
    for i in list(di_dict.keys()):
      di_dict[i] = j.count(i)
    di_dics.append(di_dict)
  wga_di_out = pd.DataFrame(di_dics)
  wga_letter = glycoletter_count_matrix(df[glycan_col_name].values.tolist(),
                                          df[label_col_name].values.tolist(),
                                   label_col_name, libr)
  out_matrix = pd.concat([wga_letter, wga_di_out], axis = 1)
  out_matrix = out_matrix.loc[:,~out_matrix.columns.duplicated()]
  temp = out_matrix.pop(label_col_name)
  out_matrix[label_col_name] = temp
  out_matrix.reset_index(drop = True, inplace = True)
  return out_matrix

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
    ggraph = glycan_to_nxGraph(glycan, libr = libr, termini = 'calc')
    res = [subgraph_isomorphism(ggraph, motifs.motif.values.tolist()[k], libr = libr,
                              extra = extra,
                              wildcard_list = wildcard_list,
                              termini_list = termini_list[k],
                                count = True) for k in range(len(motifs))]*1
  else:
    ggraph = glycan_to_nxGraph(glycan, libr = libr, termini = 'ignore')
    res = [subgraph_isomorphism(ggraph, motifs.motif.values.tolist()[k], libr = libr,
                              extra = extra,
                              wildcard_list = wildcard_list,
                              termini_list = termini_list,
                                count = True) for k in range(len(motifs))]*1
      
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

def pretend_connect(glycoletter_list):
  """makes all permutations of the list of glycoletters\n
  | Arguments:
  | :-
  | glycoletter_list (list): list of monosaccharides or linkages in IUPAC-condensed\n
  | Returns:
  | :-                               
  | Returns a list of connected glycoletters
  """
  glycoletter_list = ['('+k+')' if k in linkages else k for k in glycoletter_list]
  pool = []
  for k in list(itertools.permutations(glycoletter_list)):
    glycan = "".join(k)
    if glycan[0]!='(' and glycan[-1]!=')':
      pool.append(glycan)
  return pool

def get_k_saccharides(glycan, libr = None, k = 3):
  """function to retrieve k-saccharides (default:trisaccharides) occurring in a glycan\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed nomenclature
  | libr (list): sorted list of unique glycoletters observed in the glycans of our data; default:lib
  | k (int): number of monosaccharides per -saccharide, default:3 (for trisaccharides)\n
  | Returns:
  | :-                               
  | Returns list of trisaccharides in glycan in IUPAC-condensed nomenclature
  """
  if libr is None:
    libr = lib
  actual_k = k + (k-1)
  ggraph = glycan_to_nxGraph(glycan, libr = libr)
  out = []
  for nodes in itertools.combinations(list(ggraph.nodes()), actual_k):
    sub_g = ggraph.subgraph(nodes)
    if nx.is_connected(sub_g):
      pool = pretend_connect(list(nx.get_node_attributes(sub_g, 'string_labels').values()))
      for p in pool:
        try:
          if subgraph_isomorphism(ggraph, p, libr = libr):
            if not any([compare_glycans(p, j, libr = libr) for j in out]):
              out.append(p)
        except:
          pass
  out = list(set(out))
  out = [k.replace('GlcNAc(b1-4)GlcNAc(a1-6)Fuc', 'GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc') for k in out]
  out = [k.replace('Man(a1-3)Man(a1-6)Man', 'Man(a1-3)[Man(a1-6)]Man') for k in out]
  return out
