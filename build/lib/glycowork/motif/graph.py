import re
import networkx as nx
from glycowork.glycan_data.loader import lib, unwrap, find_nth
from glycowork.motif.processing import min_process_glycans
from glycowork.motif.tokenization import string_to_labels
import numpy as np
import pandas as pd
from scipy.sparse.linalg.eigen.arpack import eigsh


def glycan_to_graph(glycan, libr = None):
  """the monumental function for converting glycans into graphs\n
  | Arguments:
  | :-
  | glycan (string): IUPAC-condensed glycan sequence (string)
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset\n
  | Returns:
  | :-
  | (1) a list of labeled glycoletters from the glycan / node list
  | (2) two lists to indicate which glycoletters are connected in the glycan graph / edge list
  """
  if libr is None:
    libr = lib
  bracket_count = glycan.count('[')
  parts = []
  branchbranch = []
  branchbranch2 = []
  position_bb = []
  b_counts = []
  bb_count = 0
  #checks for branches-within-branches and handles them
  if bool(re.search('\[[^\]]+\[', glycan)):
    double_pos = [(k.start(),k.end()) for k in re.finditer('\[[^\]]+\[', glycan)]
    for spos, pos in double_pos:
      bracket_count -= 1
      glycan_part = glycan[spos+1:]
      glycan_part = glycan_part[glycan_part.find('['):]
      idx = [k.end() for k in re.finditer('\][^\(]+\(', glycan_part)][0]
      idx2 = [k.start() for k in re.finditer('\][^\(]+\(', glycan_part)][0]
      branchbranch.append(glycan_part[:idx-1].replace(']','').replace('[',''))
      branchbranch2.append(glycan[pos-1:])
      glycan_part = glycan[:pos-1]
      b_counts.append(glycan_part.count('[')-bb_count)
      glycan_part = glycan_part[glycan_part.rfind('[')+1:]
      position_bb.append(glycan_part.count('(')*2)
      bb_count += 1
    for b in branchbranch2:
      glycan =  glycan.replace(b, ']'.join(b.split(']')[1:]))
  main = re.sub("[\[].*?[\]]", "", glycan)
  position = []
  branch_points = [x.start() for x in re.finditer('\]', glycan)]
  for i in branch_points:
    glycan_part = glycan[:i+1]
    glycan_part = re.sub("[\[].*?[\]]", "", glycan_part)
    position.append(glycan_part.count('(')*2)
  parts.append(main)

  for k in range(1,bracket_count+1):
    start = find_nth(glycan, '[', k) + 1
    #checks whether glycan continues after branch
    if bool(re.search("[\]][^\[]+[\(]", glycan[start:])):
      #checks for double branches and removes second branch
      if bool(re.search('\]\[', glycan[start:])):
        glycan_part = re.sub("[\[].*?[\]]", "", glycan[start:])
        end = re.search("[\]].*?[\(]", glycan_part).span()[1] - 1
        parts.append(glycan_part[:end].replace(']',''))
      else:
        end = re.search("[\]].*?[\(]", glycan[start:]).span()[1] + start -1
        parts.append(glycan[start:end].replace(']',''))
    else:
      if bool(re.search('\]\[', glycan[start:])):
        glycan_part = re.sub("[\[].*?[\]]", "", glycan[start:])
        end = len(glycan_part)
        parts.append(glycan_part[:end].replace(']',''))
      else:
        end = len(glycan)
        parts.append(glycan[start:end].replace(']',''))
  
  try:
    for bb in branchbranch:
      parts.append(bb)
  except:
    pass

  parts = min_process_glycans(parts)
  parts_lengths = [len(j) for j in parts]
  parts_tokenized = [string_to_labels(k, libr) for k in parts]
  parts_tokenized = [parts_tokenized[0]] + [parts_tokenized[k][:-1] for k in range(1,len(parts_tokenized))]
  parts_tokenized = [item for sublist in parts_tokenized for item in sublist]
  
  range_list = list(range(len([item for sublist in parts for item in sublist])))
  init = 0
  parts_positions = []
  for k in parts_lengths:
    parts_positions.append(range_list[init:init+k])
    init += k

  for j in range(1,len(parts_positions)-len(branchbranch)):
    parts_positions[j][-1] = position[j-1]
  for j in range(1, len(parts_positions)):
    try:
      for z in range(j+1,len(parts_positions)):
        parts_positions[z][:-1] = [o-1 for o in parts_positions[z][:-1]]
    except:
      pass
  try:
    for i,j in enumerate(range(len(parts_positions)-len(branchbranch), len(parts_positions))):
      parts_positions[j][-1] = parts_positions[b_counts[i]][position_bb[i]]
  except:
    pass

  pairs = []
  for i in parts_positions:
    pairs.append([(i[m],i[m+1]) for m in range(0,len(i)-1)])
  pairs = list(zip(*[item for sublist in pairs for item in sublist]))
  return parts_tokenized, pairs

def categorical_node_match_wildcard(attr, default, wildcard_list):
  if isinstance(attr, str):
    def match(data1, data2):
      if data1['labels'] in wildcard_list:
        return True
      elif data2['labels'] in wildcard_list:
        return True
      else:
        return data1.get(attr, default) == data2.get(attr, default)
  else:
    attrs = list(zip(attr, default))
    def match(data1, data2):
      return all(data1.get(attr, d) == data2.get(attr, d) for attr, d in attrs)
  return match

def categorical_termini_match(attr1, attr2, default1, default2):
  if isinstance(attr1, str):
    def match(data1, data2):
      if data1[attr2] in ['flexible']:
        return all([data1.get(attr1, default1) == data2.get(attr1, default1), True])
      elif data2[attr2] in ['flexible']:
        return all([data1.get(attr1, default1) == data2.get(attr1, default1), True])
      else:
        return all([data1.get(attr1, default1) == data2.get(attr1, default1), data1.get(attr2, default2) == data2.get(attr2, default2)])
  else:
    attrs = list(zip(attr, default))
    def match(data1, data2):
      return all(data1.get(attr, d) == data2.get(attr, d) for attr, d in attrs)
  return match

def compare_glycans(glycan_a, glycan_b, libr = None,
                    wildcards = False, wildcard_list = []):
  """returns True if glycans are the same and False if not\n
  | Arguments:
  | :-
  | glycan_a (string): glycan in IUPAC-condensed format
  | glycan_b (string): glycan in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | wildcards (bool): set to True to allow wildcards (e.g., 'bond', 'monosaccharide'); default is False
  | wildcard_list (list): list of indices for wildcards in libr\n
  | Returns:
  | :-  
  | Returns True if two glycans are the same and False if not
  """
  if libr is None:
    libr = lib
  g1 = glycan_to_nxGraph(glycan_a, libr)
  g2 = glycan_to_nxGraph(glycan_b, libr)
  
  if wildcards:
    return nx.is_isomorphic(g1, g2, node_match = categorical_node_match_wildcard('labels', len(libr), wildcard_list))
  else:
    return nx.is_isomorphic(g1, g2, node_match = nx.algorithms.isomorphism.categorical_node_match('labels', len(libr)))

def fast_compare_glycans(g1, g2, libr = None,
                    wildcards = False, wildcard_list = []):
  """returns True if glycans are the same and False if not\n
  | Arguments:
  | :-
  | g1 (networkx object): glycan graph from glycan_to_nxGraph
  | g2 (networkx object): glycan graph from glycan_to_nxGraph
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | wildcards (bool): set to True to allow wildcards (e.g., 'bond', 'monosaccharide'); default is False
  | wildcard_list (list): list of indices for wildcards in libr\n
  | Returns:
  | :-  
  | Returns True if two glycans are the same and False if not
  """
  if libr is None:
    libr = lib
  if wildcards:
    return nx.is_isomorphic(g1, g2, node_match = categorical_node_match_wildcard('labels', len(libr), wildcard_list))
  else:
    return nx.is_isomorphic(g1, g2, node_match = nx.algorithms.isomorphism.categorical_node_match('labels', len(libr)))

def subgraph_isomorphism(glycan, motif, libr = None,
                         extra = 'ignore', wildcard_list = [],
                         termini_list = []):
  """returns True if motif is in glycan and False if not\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | motif (string): glycan motif in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | extra (string): 'ignore' skips this, 'wildcards' allows for wildcard matching', and 'termini' allows for positional matching; default:'ignore'
  | wildcard_list (list): list of wildcard names (such as 'bond', 'Hex', 'HexNAc', 'Sia')
  | termini_list (list): list of monosaccharide/linkage positions (from 'terminal','internal', and 'flexible')\n
  | Returns:
  | :-
  | Returns True if motif is in glycan and False if not
  """
  if libr is None:
    libr = lib
  if len(wildcard_list) >= 1:
    wildcard_list = [libr.index(k) for k in wildcard_list]
  if extra == 'termini':
    g1 = glycan_to_nxGraph(glycan, libr, termini = 'calc')
    g2 = glycan_to_nxGraph(motif, libr, termini = 'provided', termini_list = termini_list)
  else:
    g1 = glycan_to_nxGraph(glycan, libr)
    g2 = glycan_to_nxGraph(motif, libr)

  if extra == 'ignore':
    graph_pair = nx.algorithms.isomorphism.GraphMatcher(g1,g2,node_match = nx.algorithms.isomorphism.categorical_node_match('labels', len(libr)))
  elif extra == 'wildcards':
    graph_pair = nx.algorithms.isomorphism.GraphMatcher(g1,g2,node_match = categorical_node_match_wildcard('labels', len(libr), wildcard_list))
  elif extra == 'termini':
    graph_pair = nx.algorithms.isomorphism.GraphMatcher(g1,g2,node_match = categorical_termini_match('labels', 'termini', len(libr), 'flexible'))

  return graph_pair.subgraph_is_isomorphic()

def glycan_to_nxGraph(glycan, libr = None,
                      termini = 'ignore', termini_list = None):
  """converts glycans into networkx graphs\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | termini (string): whether to encode terminal/internal position of monosaccharides, 'ignore' for skipping, 'calc' for automatic annotation, or 'provided' if this information is provided in termini_list; default:'ignore'
  | termini_list (list): list of monosaccharide/linkage positions (from 'terminal','internal', and 'flexible')\n
  | Returns:
  | :-
  | Returns networkx graph object of glycan
  """
  if libr is None:
    libr = lib
  glycan_graph = glycan_to_graph(glycan, libr = libr)
  edgelist = list(zip(glycan_graph[1][0], glycan_graph[1][1]))
  g1 = nx.from_edgelist(edgelist)
  nx.set_node_attributes(g1, {k:glycan_graph[0][k] for k in range(len(glycan_graph[0]))},'labels')
  if termini == 'ignore':
    pass
  elif termini == 'calc':
    nx.set_node_attributes(g1, {k:'terminal' if g1.degree[k] == 1 else 'internal' for k in range(len(glycan_graph[0]))}, 'termini')
  elif termini == 'provided':
    nx.set_node_attributes(g1, {k:j for k,j in zip(range(len(glycan_graph[0])), termini_list)}, 'termini')
  return g1

def generate_graph_features(glycan, glycan_graph = True, libr = None, label = 'network'):
    """compute graph features of glycan\n
    | Arguments:
    | :-
    | glycan (string or networkx object): glycan in IUPAC-condensed format (or glycan network if glycan_graph=False)
    | glycan_graph (bool): True expects a glycan, False expects a network (from construct_network); default:True
    | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
    | label (string): Label to place in output dataframe if glycan_graph=False; default:'network'\n
    | Returns:
    | :-
    | Returns a pandas dataframe with different graph features as columns and glycan as row
    """
    if libr is None:
      libr = lib
    if glycan_graph:
      g = glycan_to_nxGraph(glycan, libr = libr)
      #nbr of different node features:
      nbr_node_types = len(set(nx.get_node_attributes(g, "labels")))
    else:
      g = glycan
      glycan = label
      nbr_node_types = len(set(list(g.nodes())))
    #adjacency matrix:
    A = nx.to_numpy_matrix(g)
    N = A.shape[0]
    if nx.is_connected(g):
      diameter = nx.algorithms.distance_measures.diameter(g)
    else:
      diameter = np.nan
    deg = np.array([np.sum(A[i,:]) for i in range(N)])
    dens = np.sum(deg)/2
    avgDeg = np.mean(deg)
    varDeg = np.var(deg)
    maxDeg = np.max(deg)
    nbrDeg4 = np.sum(deg > 3)
    branching = np.sum(deg > 2)
    nbrLeaves = np.sum(deg == 1)
    deg_to_leaves = np.array([np.sum(A[:,deg == 1]) for i in range(N)])
    max_deg_leaves = np.max(deg_to_leaves)
    mean_deg_leaves = np.mean(deg_to_leaves)
    deg_assort = nx.degree_assortativity_coefficient(g)
    betweeness_centr = np.array(pd.DataFrame(nx.betweenness_centrality(g), index = [0]).iloc[0,:])
    betweeness = np.mean(betweeness_centr)
    betwVar = np.var(betweeness_centr)
    betwMax = np.max(betweeness_centr)
    betwMin = np.min(betweeness_centr)
    eigen = np.array(pd.DataFrame(nx.katz_centrality_numpy(g), index = [0]).iloc[0,:])
    eigenMax = np.max(eigen)
    eigenMin = np.min(eigen)
    eigenAvg = np.mean(eigen)
    eigenVar = np.var(eigen)
    close = np.array(pd.DataFrame(nx.closeness_centrality(g), index = [0]).iloc[0,:])
    closeMax = np.max(close)
    closeMin = np.min(close)
    closeAvg = np.mean(close)
    closeVar = np.var(close)
    if nx.is_connected(g):
      flow = np.array(pd.DataFrame(nx.current_flow_betweenness_centrality(g), index = [0]).iloc[0,:])
      flowMax = np.max(flow)
      flowMin = np.min(flow)
      flowAvg = np.mean(flow)
      flowVar = np.var(flow)
      flow_edge = np.array(pd.DataFrame(nx.edge_current_flow_betweenness_centrality(g), index = [0]).iloc[0,:])
      flow_edgeMax = np.max(flow_edge)
      flow_edgeMin = np.min(flow_edge)
      flow_edgeAvg = np.mean(flow_edge)
      flow_edgeVar = np.var(flow_edge)
    else:
      flow = np.nan
      flowMax = np.nan
      flowMin = np.nan
      flowAvg = np.nan
      flowVar = np.nan
      flow_edge = np.nan
      flow_edgeMax = np.nan
      flow_edgeMin = np.nan
      flow_edgeAvg = np.nan
      flow_edgeVar = np.nan
    load = np.array(pd.DataFrame(nx.load_centrality(g), index = [0]).iloc[0,:])
    loadMax = np.max(load)
    loadMin = np.min(load)
    loadAvg = np.mean(load)
    loadVar = np.var(load)
    harm = np.array(pd.DataFrame(nx.harmonic_centrality(g), index = [0]).iloc[0,:])
    harmMax = np.max(harm)
    harmMin = np.min(harm)
    harmAvg = np.mean(harm)
    harmVar = np.var(harm)
    if nx.is_connected(g):
      secorder = np.array(pd.DataFrame(nx.second_order_centrality(g), index = [0]).iloc[0,:])
      secorderMax = np.max(secorder)
      secorderMin = np.min(secorder)
      secorderAvg = np.mean(secorder)
      secorderVar = np.var(secorder)
    else:
      secorder = np.nan
      secorderMax = np.nan
      secorderMin = np.nan
      secorderAvg = np.nan
      secorderVar = np.nan
    x = np.array([len(nx.k_corona(g,k).nodes()) for k in range(N)])
    size_corona = x[x > 0][-1]
    k_corona = np.where(x == x[x > 0][-1])[0][-1]
    x = np.array([len(nx.k_core(g,k).nodes()) for k in range(N)])
    size_core = x[x > 0][-1]
    k_core = np.where(x == x[x > 0][-1])[0][-1]
    M = ((A + np.diag(np.ones(N))).T/(deg + 1)).T
    eigval, vec = eigsh(M, 2, which = 'LM')
    egap = 1 - eigval[0]
    distr = np.abs(vec[:,-1])
    distr = distr/sum(distr)
    entropyStation = np.sum(distr*np.log(distr))
    features = np.array(
        [diameter, branching, nbrLeaves, avgDeg, varDeg, maxDeg, nbrDeg4, max_deg_leaves, mean_deg_leaves,
         deg_assort, betweeness, betwVar, betwMax, eigenMax, eigenMin, eigenAvg, eigenVar, closeMax, closeMin,
         closeAvg, closeVar, flowMax, flowAvg, flowVar,
         flow_edgeMax, flow_edgeMin, flow_edgeAvg, flow_edgeVar,
         loadMax, loadAvg, loadVar,
         harmMax, harmMin, harmAvg, harmVar,
         secorderMax, secorderMin, secorderAvg, secorderVar,
         size_corona, size_core, nbr_node_types,
         egap, entropyStation, N, dens
         ])
    col_names = ['diameter', 'branching', 'nbrLeaves', 'avgDeg', 'varDeg',
                 'maxDeg', 'nbrDeg4', 'max_deg_leaves', 'mean_deg_leaves',
                 'deg_assort', 'betweeness', 'betwVar', 'betwMax', 'eigenMax',
                 'eigenMin', 'eigenAvg', 'eigenVar', 'closeMax', 'closeMin',
                 'closeAvg', 'closeVar', 'flowMax', 'flowAvg', 'flowVar',
                 'flow_edgeMax', 'flow_edgeMin', 'flow_edgeAvg', 'flow_edgeVar',
                 'loadMax', 'loadAvg', 'loadVar', 'harmMax', 'harmMin', 'harmAvg',
                 'harmVar', 'secorderMax', 'secorderMin', 'secorderAvg', 'secorderVar',
                 'size_corona', 'size_core', 'nbr_node_types', 'egap', 'entropyStation',
                 'N', 'dens']
    feat_dic = {col_names[k]:features[k] for k in range(len(features))}
    return pd.DataFrame(feat_dic, index = [glycan])

def branch_crawl(starting_node, main_chain_connect, graph):
  """recursive function to ascertain branch length and collect all nodes\n
  | Arguments:
  | :-
  | starting_node (int): first node of the branch
  | main_chain_connect (int): main chain node to which branch is connected
  | graph (networkx object): glycan graph\n
  | Returns:
  | :-
  | Returns all nodes of the branch in the correct order
  """
  break_cond = True
  node_pool = [starting_node]
  node = starting_node
  while break_cond:
    try:
      temp = [k for k in list(graph.edges()) if (node in k) and (k != tuple(sorted([node,main_chain_connect])))]
      temp = [k for j in temp for k in j if k not in node_pool][0]
      node_pool.append(temp)
      node = temp
    except:
      break_cond = False
  return list(sorted(node_pool))

def graph_to_string(graph, libr = None):
  """converts glycan graph back to IUPAC-condensed format\n
  | Arguments:
  | :-
  | graph (networkx object): glycan graph, works with branched glycans
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns glycan in IUPAC-condensed format (string)
  """
  if libr is None:
    libr = lib
  diffs = [k[1]-k[0] for k in list(graph.edges())]
  if np.max(diffs)>1:
    branch_branch_pool = []
    branch_idx = np.where([j>1 for j in diffs])[0].tolist()
    branch_nodes = [list(graph.edges())[k][0] for k in branch_idx]
    branch = [list(graph.edges())[k][1] for k in branch_idx]
    branch_pool = [branch_crawl(branch[k], branch_nodes[k], graph) for k in range(len(branch))]
    branch_branch_nodes = [k for k in branch_nodes if k in unwrap(branch_pool)]
    if len(branch_branch_nodes)>0:
      branch_branch_start = []
      for j in branch_branch_nodes:
        branch_branch = [list(graph.edges())[k][1] for k in range(len(list(graph.edges()))) if j in list(graph.edges())[k]]
        branch_branch_start.append(np.max([k for k in branch_branch if k!=j]))
      branch_branch_pool = [branch_crawl(branch_branch_start[k], branch_branch_nodes[k], graph) for k in range(len(branch_branch_start))]
      branch_branch = [[list(nx.get_node_attributes(graph, 'labels').values())[j] for j in k] for k in branch_branch_pool]
      branch_branch = [[libr[j] for j in k] for k in branch_branch]
      branch_branch = ["".join([k[j] if (j % 2) == 0 else '('+k[j]+')' for j in range(len(k))]) for k in branch_branch]
    else:
      branch_branch_pool = []
    branch = [[list(nx.get_node_attributes(graph, 'labels').values())[j] for j in k] for k in branch_pool if k not in branch_branch_pool]
    branch = [[libr[j] for j in k] for k in branch]
    branch = ["".join([k[j] if (j % 2) == 0 else '('+k[j]+')' for j in range(len(k))]) for k in branch]
  else:
    branch_pool = []
    branch_branch_pool = []
  glycan_motif = [libr[list(nx.get_node_attributes(graph, 'labels').values())[k]] for k in range(len(graph.nodes)) if k not in unwrap(branch_pool)]
  main_len = len(glycan_motif)
  if len(branch_pool)>0:
    counter = 0
    for k in range(len(branch)):
      temp = min_process_glycans([branch[k]])[0]
      temp = [m for m in temp if len(m)>0]
      if branch[k].count('(')>1:
        temp = ['[' + temp[0]] + temp[1:-1] + [temp[-1]+']']
      else:
        temp = ['[' + temp[0], temp[-1]+']']
      glycan_motif = glycan_motif[:branch_nodes[k]+counter] + temp + glycan_motif[branch_nodes[k]+counter:]
      counter += len(temp)
  if len(branch_branch_pool)>0:
    start_idxs = []
    for k in range(len(branch_branch_nodes)):
      running_idx = main_len
      for i in range(sum(['[' in j for j in glycan_motif])):
        opened = np.where(['[' in j for j in glycan_motif])[0].tolist()[i]
        closed = np.where([']' in j for j in glycan_motif])[0].tolist()[i]
        prev_running_idx = running_idx
        running_idx += len(glycan_motif[opened:closed+1])
        if running_idx > branch_branch_nodes[k]:
          start_idx = i
          break
      start_idx = np.where(['[' in j for j in glycan_motif])[0].tolist()[start_idx]
      branch_progress = prev_running_idx-main_len
      start_idxs.append(start_idx+branch_branch_nodes[k]-(main_len+branch_progress))
    for k in range(len(start_idxs)):
      glycan_motif.insert(np.max([start_idxs[k]+k, 0]), '['+branch_branch[k]+']')
  glycan_motif = "".join([glycan_motif[k] if not ((glycan_motif[k].startswith('a')) or (glycan_motif[k].startswith('b')) or (re.match('^[0-9]+(-[0-9]+)+$', glycan_motif[k]))) \
                          else '('+glycan_motif[k]+')' for k in range(len(glycan_motif))])
  if '])' in glycan_motif:
    glycan_motif = glycan_motif.replace('])', ')]')
  return glycan_motif
