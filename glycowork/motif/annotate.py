import pandas as pd

from glycowork.glycan_data.loader import lib, motif_list, unwrap
from glycowork.motif.graph import subgraph_isomorphism, generate_graph_features
from glycowork.motif.tokenization import motif_matrix

def annotate_glycan(glycan, motifs = None, libr = None, extra = 'termini',
                    wildcard_list = [], termini_list = []):
  """searches for known motifs in glycan sequence\n
  glycan -- IUPACcondensed glycan sequence (string)\n
  motifs -- dataframe of glycan motifs (name + sequence)\n
  libr -- sorted list of unique glycoletters observed in the glycans of our dataset\n
  extra -- 'ignore' skips this, 'wildcards' allows for wildcard matching',\n
           and 'termini' allows for positional matching; default:'termini'\n
  wildcard_list -- list of wildcard names (such as 'bond', 'Hex', 'HexNAc', 'Sia')\n
  termini_list -- list of monosaccharide/linkage positions (from 'terminal','internal', and 'flexible')\n

  returns dataframe with absence/presence of motifs in glycan
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
                     condense = False):
  """wrapper function to annotate motifs in list of glycans\n
  glycans -- list of IUPACcondensed glycan sequences (string)\n
  motifs -- dataframe of glycan motifs (name + sequence)\n
  libr -- sorted list of unique glycoletters observed in the glycans of our data\n
  feature_set -- which feature set to use for annotations, add more to list to expand; default is 'known'\n
                 options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans)
                               and 'exhaustive' (all mono- and disaccharide features)\n
  extra -- 'ignore' skips this, 'wildcards' allows for wildcard matching',\n
           and 'termini' allows for positional matching; default:'termini'\n
  wildcard_list -- list of wildcard names (such as 'bond', 'Hex', 'HexNAc', 'Sia')\n
  termini_list -- list of monosaccharide/linkage positions (from 'terminal','internal', and 'flexible')\n
  condense -- if True, throws away columns with only zeroes; default:False\n
                               
  returns dataframe of glycans (rows) and presence/absence of known motifs (columns)
  """
  if motifs is None:
    motifs = motif_list
  if extra == 'termini':
    if len(termini_list) < 1:
      termini_list = motifs.termini_spec.values.tolist()
      termini_list = [eval(k) for k in termini_list]
  if libr is None:
    libr = lib
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
