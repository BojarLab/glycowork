import re
import networkx as nx
from glycowork.glycan_data.loader import lib, unwrap, find_nth
from glycowork.motif.processing import min_process_glycans
from glycowork.motif.tokenization import string_to_labels

def glycan_to_graph(glycan, libr = lib):
  """the monumental function for converting glycans into graphs
  glycan -- IUPACcondensed glycan sequence (string)
  libr -- sorted list of unique glycoletters observed in the glycans of our dataset

  returns (1) a list of labeled glycoletters from the glycan / node list
          (2) two lists to indicate which glycoletters are connected in the glycan graph / edge list
  """
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

def compare_glycans(glycan_a, glycan_b, libr = lib,
                    wildcards = False, wildcard_list = []):
  """returns True if glycans are the same and False if not
  glycan_a -- glycan in string format (IUPACcondensed)
  glycan_b -- glycan in string format (IUPACcondensed)
  libr -- library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  wildcards -- set to True to allow wildcards (e.g., 'bond', 'monosaccharide'); default is False
  wildcard_list -- list of indices for wildcards in libr
  
  returns True if two glycans are the same and False if not
  """
  glycan_graph_a = glycan_to_graph(glycan_a, libr)
  edgelist = list(zip(glycan_graph_a[1][0], glycan_graph_a[1][1]))
  g1 = nx.from_edgelist(edgelist)
  nx.set_node_attributes(g1, {k:glycan_graph_a[0][k] for k in range(len(glycan_graph_a[0]))},'labels')
  
  glycan_graph_b = glycan_to_graph(glycan_b, libr)
  edgelist = list(zip(glycan_graph_b[1][0], glycan_graph_b[1][1]))
  g2 = nx.from_edgelist(edgelist)
  nx.set_node_attributes(g2, {k:glycan_graph_b[0][k] for k in range(len(glycan_graph_b[0]))},'labels')
  
  if wildcards:
    return nx.is_isomorphic(g1, g2, node_match = categorical_node_match_wildcard('labels', len(libr), wildcard_list))
  else:
    return nx.is_isomorphic(g1, g2, node_match = nx.algorithms.isomorphism.categorical_node_match('labels', len(libr)))

def fast_compare_glycans(g1, g2, libr = lib,
                    wildcards = False, wildcard_list = []):
  """returns True if glycans are the same and False if not
  g1 -- glycan graph from glycan_to_nxGraph
  g2 -- glycan graph from glycan_to_nxGraph
  libr -- library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  wildcards -- set to True to allow wildcards (e.g., 'bond', 'monosaccharide'); default is False
  wildcard_list -- list of indices for wildcards in libr
  
  returns True if two glycans are the same and False if not
  """
  if wildcards:
    return nx.is_isomorphic(g1, g2, node_match = categorical_node_match_wildcard('labels', len(libr), wildcard_list))
  else:
    return nx.is_isomorphic(g1, g2, node_match = nx.algorithms.isomorphism.categorical_node_match('labels', len(libr)))

def subgraph_isomorphism(glycan, motif, libr = lib,
                         wildcards = False, wildcard_list = []):
  """returns True if motif is in glycan and False if not
  glycan -- glycan in string format (IUPACcondensed)
  motif -- glycan motif in string format (IUPACcondensed)
  libr -- library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  wildcards -- set to True to allow wildcards (e.g., 'bond', 'monosaccharide'); default is False
  wildcard_list -- list of indices for wildcards in libr
  
  returns True if motif is in glycan and False if not
  """
  glycan_graph = glycan_to_graph(glycan, libr)
  edgelist = list(zip(glycan_graph[1][0], glycan_graph[1][1]))
  g1 = nx.from_edgelist(edgelist)
  nx.set_node_attributes(g1, {k:glycan_graph[0][k] for k in range(len(glycan_graph[0]))},'labels')
  
  motif_graph = glycan_to_graph(motif, libr)
  edgelist = list(zip(motif_graph[1][0], motif_graph[1][1]))
  g2 = nx.from_edgelist(edgelist)
  nx.set_node_attributes(g2, {k:motif_graph[0][k] for k in range(len(motif_graph[0]))},'labels')

  if wildcards:
    graph_pair = nx.algorithms.isomorphism.GraphMatcher(g1,g2,node_match = categorical_node_match_wildcard('labels', len(libr), wildcard_list))
  else:
    graph_pair = nx.algorithms.isomorphism.GraphMatcher(g1,g2,node_match = nx.algorithms.isomorphism.categorical_node_match('labels', len(libr)))

  return graph_pair.subgraph_is_isomorphic()

def glycan_to_nxGraph(glycan, libr = lib):
  """converts glycans into networkx graphs
  glycan -- glycan in string format (IUPACcondensed)
  libr -- library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used

  returns networkx graph object of glycan
  """
  glycan_graph = glycan_to_graph(glycan, libr = libr)
  edgelist = list(zip(glycan_graph[1][0], glycan_graph[1][1]))
  g1 = nx.from_edgelist(edgelist)
  nx.set_node_attributes(g1, {k:glycan_graph[0][k] for k in range(len(glycan_graph[0]))},'labels')
  return g1
