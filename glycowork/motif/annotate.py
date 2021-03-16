import pandas as pd

from glycowork.glycan_data.loader import lib, motif_list, unwrap
from glycowork.motif.graph import subgraph_isomorphism, generate_graph_features
from glycowork.motif.tokenization import motif_matrix

def annotate_glycan(glycan, motifs = motif_list, libr = lib):
  """searches for known motifs in glycan sequence
  glycan -- IUPACcondensed glycan sequence (string)
  motifs -- dataframe of glycan motifs (name + sequence)
  libr -- sorted list of unique glycoletters observed in the glycans of our dataset

  returns dataframe with absence/presence of motifs in glycan
  """
  res = [subgraph_isomorphism(glycan, k, libr = libr) for k in motifs.motif.values.tolist()]*1
  out = pd.DataFrame(columns = motifs.motif_name.values.tolist())
  out.loc[0] = res
  out.loc[0] = out.loc[0].astype('int')
  out.index = [glycan]
  return out

def annotate_dataset(glycans, motifs = motif_list, libr = lib,
                     feature_set = ['known']):
  """wrapper function to annotate motifs in list of glycans
  glycans -- list of IUPACcondensed glycan sequences (string)
  motifs -- dataframe of glycan motifs (name + sequence)
  libr -- sorted list of unique glycoletters observed in the glycans of our data
  feature_set -- which feature set to use for annotations, add more to list to expand; default is 'known'
                 options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans)
                               and 'exhaustive' (all mono- and disaccharide features)
                               
  returns dataframe of glycans (rows) and presence/absence of known motifs (columns)
  """
  shopping_cart = []
  if 'known' in feature_set:
    shopping_cart.append(pd.concat([annotate_glycan(k, motifs = motifs, libr = libr) for k in glycans], axis = 0))
  if 'graph' in feature_set:
    shopping_cart.append(pd.concat([generate_graph_features(k, libr = libr) for k in glycans], axis = 0))
  if 'exhaustive' in feature_set:
    temp = motif_matrix(pd.DataFrame({'glycans':glycans, 'labels':range(len(glycans))}),
                                                   'glycans', 'labels', libr = libr)
    temp.index = glycans
    temp.drop(['labels'], axis = 1, inplace = True)
    shopping_cart.append(temp)
  return pd.concat(shopping_cart, axis = 1)
