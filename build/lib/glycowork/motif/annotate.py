import pandas as pd

from glycowork.glycan_data.loader import lib, motif_list
from glycowork.motif.graph import subgraph_isomorphism

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

def annotate_dataset(glycans, motifs = motif_list, libr = lib):
  """wrapper function to annotate motifs in list of glycans
  glycans -- list of IUPACcondensed glycan sequences (string)
  motifs -- dataframe of glycan motifs (name + sequence)
  libr -- sorted list of unique glycoletters observed in the glycans of our data

  returns dataframe of glycans (rows) and presence/absence of known motifs (columns)
  """
  temp = [annotate_glycan(k, motifs = motifs, libr = libr) for k in glycans]
  return pd.concat(temp, axis = 0)
