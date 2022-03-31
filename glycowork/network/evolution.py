import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage
from glycowork.glycan_data.loader import lib
from glycowork.motif.graph import subgraph_isomorphism

def distance_from_embeddings(df, embeddings, cut_off = 10, rank = 'Species',
                             averaging = 'median'):
  """calculates a cosine distance matrix from learned embeddings\n
  | Arguments:
  | :-
  | df (dataframe): dataframe with glycans as rows and taxonomic information as columns
  | embeddings (dataframe): dataframe with glycans as rows and learned embeddings as columns (e.g., from glycans_to_emb)
  | cut_off (int): how many glycans a rank (e.g., species) needs to have at least to be included; default:10
  | rank (string): which taxonomic rank to use for grouping organisms; default:'Species'
  | averaging (string): how to average embeddings, by 'median' or 'mean'; default:'median'\n
  | Returns:
  | :-
  | Returns a rank x rank distance matrix
  """
  df_min = list(sorted([(df[rank].value_counts() >= cut_off).index.tolist()[k]
                           for k in range(len((df[rank].value_counts() >= cut_off).index.tolist()))
                           if (df[rank].value_counts() >= cut_off).values.tolist()[k]]))
  df_idx = [df.index[df[rank] == k].values.tolist() for k in df_min]
  if averaging == 'median':
    avgs = [np.median(embeddings.iloc[k,:], axis = 0) for k in df_idx]
  elif averaging == 'mean':
    avgs = [np.mean(embeddings.iloc[k,:], axis = 0) for k in df_idx]
  else:
    print("Only 'median' and 'mean' are permitted averaging choices.")
  dm = np.zeros((len(avgs), len(avgs)))
  dm = pd.DataFrame(dm, columns = df_min)
  for i in range(len(avgs)):
    for j in range(len(avgs)):
      dm.iloc[i,j] = cosine(avgs[i], avgs[j])
  return dm

def jaccard(list1, list2):
  """calculates Jaccard distance from two networks\n
  | Arguments:
  | :-
  | list1 (list or networkx graph): list containing objects to compare
  | list2 (list or networkx graph): list containing objects to compare\n
  | Returns:
  | :-
  | Returns Jaccard distance between list1 and list2
  """
  intersection = len(list(set(list1).intersection(list2)))
  union = (len(list1) + len(list2)) - intersection
  return 1- float(intersection) / union

def distance_from_metric(df, networks, metric = "Jaccard", cut_off = 10, rank = "Species"):
  """calculates a distance matrix of generated networks based on provided metric\n
  | Arguments:
  | :-
  | df (dataframe): dataframe with glycans as rows and taxonomic information as columns
  | networks (list): list of networks in networkx format
  | metric (string): which metric to use, available: 'Jaccard'; default:'Jaccard'
  | cut_off (int): how many glycans a rank (e.g., species) needs to have at least to be included; default:10
  | rank (string): which taxonomic rank to use for grouping organisms; default:'Species'\n
  | Returns:
  | :-
  | Returns a rank x rank distance matrix
  """
  if metric == "Jaccard":
    dist_func = jaccard
  else:
    print("Not a defined metric. At the moment, only 'Jaccard' is available as a metric.")
  specs = list(sorted(list(set(df[rank].values.tolist()))))
  idx_min = [k for k in range(len(specs)) if len(df[df[rank] == specs[k]]) >= cut_off]
  specs_min = [specs[k] for k in idx_min]
  networks_min = [networks[k] for k in idx_min]
  dm = np.zeros((len(networks_min), len(networks_min)))
  dm = pd.DataFrame(dm, columns = specs_min)
  for i in range(len(networks_min)):
    for j in range(len(networks_min)):
      dm.iloc[i,j] = dist_func(networks_min[i], networks_min[j])
  return dm

def dendrogram_from_distance(dm, ylabel = 'Mammalia', filepath = ''):
  """plots a dendrogram from distance matrix\n
  | Arguments:
  | :-
  | dm (dataframe): a rank x rank distance matrix (e.g., from distance_from_embeddings)
  | ylabel (string): how to label the y-axis of the dendrogram; default:'Mammalia'
  | filepath (string): absolute path including full filename allows for saving the plot\n
  """
  Z = linkage(dm)
  plt.figure(figsize = (10,10))
  plt.title('Hierarchical Clustering Dendrogram')
  plt.xlabel('Distance')
  plt.ylabel(ylabel)
  dendrogram(
      Z,
      truncate_mode = 'lastp',  # show only the last p merged clusters
      orientation = 'left',
      p = 300,  # show only the last p merged clusters
      show_leaf_counts = False,  # otherwise numbers in brackets are counts
      leaf_rotation = 0.,
      labels = dm.columns.values.tolist(),
      leaf_font_size = 11.,
      show_contracted = True,  # to get a distribution impression in truncated branches
      )
  if len(filepath) > 1:
    plt.savefig(filepath, format = filepath.split('.')[-1], dpi = 300,
                  bbox_inches = 'tight')

def check_conservation(glycan, df, libr = None, rank = 'Order', filepath = None, threshold = 5,
                       motif = False, file_suffix = '_graph_exhaustive.pkl'):
  """estimates evolutionary conservation of glycans and glycan motifs via biosynthetic networks\n
  | Arguments:
  | :-
  | glycan (string): full glycan or glycan motif in IUPAC-condensed nomenclature
  | df (dataframe): dataframe in the style of df_species, each row one glycan and columns are the taxonomic levels
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | rank (string): at which taxonomic level to assess conservation; default:Order
  | filepath (string): filepath to load biosynthetic networks from other species; files need to be species name + file_suffix
  | threshold (int): threshold of how many glycans a species needs to have to consider the species;default:5
  | motif (bool): whether glycan is a motif (True) or a full sequence (False); default:False
  | file_suffix (string): generic end part of filename in filepath; default:'_graph_exhaustive.pkl'\n
  | Returns:
  | :-
  | Returns a dictionary of taxonomic group : degree of conservation
  """
  if libr is None:
    libr = lib
  all_specs = list(sorted(list(set(df.Species.values.tolist()))))
  allowed = [k for k in all_specs if df.Species.values.tolist().count(k) >= threshold]
  df_freq = df[df.Species.isin(allowed)].reset_index(drop = True)
  all_specs = list(sorted(list(set(df_freq.Species.values.tolist()))))
  pool = list(set(df_freq[rank].values.tolist()))
  pool = [k for k in pool if len(list(set(df_freq[df_freq[rank] == k].Species.tolist()))) > 1]
  networks = {k:nx.read_gpickle(filepath + k + file_suffix) for k in all_specs}
  conserved = {}
  for k in pool:
    specs = list(set(df_freq[df_freq[rank] == k].Species.values.tolist()))
    nets = [networks[j] for j in specs]
    nets = [list(j.nodes()) for j in nets]
    if motif:
      if glycan[-1] == ')':
        conserved[k] = sum([glycan in "".join(j) for j in nets]) / len(nets)
      else:
        conserved[k] = sum([any([subgraph_isomorphism(m, glycan, libr = libr) for m in j]) for j in nets]) / len(nets)
    else:
      conserved[k] = sum([glycan in j for j in nets]) / len(nets)
  return conserved
