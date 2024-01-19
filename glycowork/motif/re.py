import re
import copy
import networkx as nx
from itertools import product, combinations, chain
from glycowork.glycan_data.loader import replace_every_second, lib, unwrap
from glycowork.motif.processing import min_process_glycans, bracket_removal, canonicalize_iupac
from glycowork.motif.graph import graph_to_string, subgraph_isomorphism, compare_glycans, glycan_to_nxGraph


def preprocess_pattern(pattern):
  """transforms a glyco-regular expression into chunks\n
  | Arguments:
  | :-
  | pattern (string): glyco-regular expression in the form of "Hex-HexNAc-([Hex|Fuc]){1,2}-HexNAc"\n
  | Returns:
  | :-
  | Returns list of chunks to be used by downstream functions
  """
  # Use regular expression to identify the conditional parts and keep other chunks together
  pattern = pattern.replace('.', 'Monosaccharide')
  components = re.split(r'(-?\s*\(?\[.*?\]\)?\s*(?:\{,?\d*,?\d*\}\?|\{,?\d*,?\d*\}|\*\?|\+\?|\?|\*|\+)\s*-?)', pattern)
  # Remove any empty strings and trim whitespace
  return [x.strip('-').strip() for x in components if x]


def specify_linkages(pattern_component):
  """allows for specification of exact linkages by converting expressions such as Mana6 to Man(a1-6)\n
  | Arguments:
  | :-
  | pattern_component (string): chunk of a glyco-regular expression\n
  | Returns:
  | :-
  | Returns specified chunk to be used by downstream functions
  """
  if re.search(r"[\d|\?]\(|\d$", pattern_component):
    pattern = re.compile(r"([ab\?])([2-9\?])\(\?1-\?\)")
    def replacer(match):
      letter, number = match.groups()
      return f'({letter}1-{number})'
    pattern_component = pattern.sub(replacer, pattern_component)
  return pattern_component.replace('5Ac(a1', '5Ac(a2').replace('5Gc(a1', '5Gc(a2').replace('Kdn(a1', 'Kdn(a2').replace('Sia(a1', 'Sia(a2')


def replace_patterns(s):
  return s.replace('-', '(?1-?)').replace('5Ac(?1', '5Ac(?2').replace('5Gc(?1', '5Gc(?2')


def process_occurrence(occ_str):
  """processes the minimum and maximum occurrence of a pattern component\n
  | Arguments:
  | :-
  | occ_str (string): content between {} of a pattern component\n
  | Returns:
  | :-
  | Returns list of minimum and maximum occurrence
  """
  occ = occ_str.split(',')
  if len(occ) == 1:
    return [int(occ[0]), int(occ[0])]
  return [int(occ[0]) if occ[0] else 0, int(occ[1]) if occ[1] else 5]


def process_question_mark(s, p):
  """processes pattern components with lookahead/-behind or just regular ? characters\n
  | Arguments:
  | :-
  | s (string): original pattern component
  | p (string): content between [] in pattern component\n
  | Returns:
  | :-
  | Returns list of minimum and maximum occurrence + cleaned up pattern component
  """
  if '?<' in s:
    occurrence = [1]
    part = s.split('=')[1].replace(')', '') if '=' in s else s.split(')')[1]
  elif '?=' in s or '?!' in s:
    occurrence = [1]
    part = s.replace('(?=', '').replace(')', '') if '=' in s else s.split('(')[0]
  else:
    occurrence = [0, 1]
    part = p if p else s
  pattern = [replace_patterns(part)] if isinstance(part, str) else [replace_patterns(ps) for ps in part]
  return occurrence, pattern


def expand_pattern(pattern_component):
  """allows for specification of ambiguous (but not wildcarded) linkages by converting expressions such as Galb3/4 to the two specified versions\n
  | Arguments:
  | :-
  | pattern_component (string): chunk of a glyco-regular expression\n
  | Returns:
  | :-
  | Returns a list of specified pattern components, ready for specify_linkages etc.
  """
  # Find the window of ambiguity (a segment with characters separated by slashes)
  match = re.findall(r'\d\/\d', pattern_component)
  if not match:
    return [pattern_component]  # Return the original pattern if no ambiguous window is found
  # Extract the ambiguous characters
  ambiguous_window = match[0]
  ambiguous_chars = ambiguous_window.split('/')
  # Duplicate the pattern for each ambiguous character and replace the window
  return [pattern_component.replace(ambiguous_window, char, 1) for char in ambiguous_chars]


def convert_pattern_component(pattern_component):
  """processes a pattern component into either a string or dict of form string : occurrence\n
  | Arguments:
  | :-
  | pattern_component (string): chunk of a glyco-regular expression\n
  | Returns:
  | :-
  | Returns a string for simple components and a dict of form string : occurrence for complex components
  """
  if not any([k in pattern_component for k in ['[', '{', '*', '+', '=', '!']]):
    return specify_linkages(replace_patterns(pattern_component))
  pattern, occurrence = None, None
  if '[' in pattern_component:
    pattern = pattern_component.split('[')[1].split(']')[0].split('|')
    pattern = [replace_patterns(p) for p in pattern]
  if '{' in pattern_component:
    occurrence = process_occurrence(pattern_component.split('{')[1].split('}')[0])
    if '?' in pattern_component:
      occurrence = [occurrence[0], occurrence[0]]
  elif '*' in pattern_component or '+' in pattern_component:
    occurrence = list(range(0, 5)) if '*' in pattern_component else list(range(1, 5))
    if '?' in pattern_component:
      occurrence = [occurrence[0], occurrence[0]]
  elif '?' in pattern_component:
    occurrence, pattern = process_question_mark(pattern_component, pattern)
  if pattern is None:
    pattern = replace_patterns(pattern_component)
  if any(['/' in p for p in pattern]):
    expanded_pattern = []
    for p in pattern:
      expanded_pattern.extend(expand_pattern(p) if '/' in p else [p])
    pattern = expanded_pattern
  elif '/' in pattern:
    pattern = expand_pattern(pattern)
  return {specify_linkages(p): occurrence for p in pattern}


def process_main_branch(glycan, glycan_parts):
  """extracts the main chain node indices of a glycan\n
  | Arguments:
  | :-
  | glycan (string): glycan sequence in IUPAC-condensed
  | glycan_parts (list): list of glycoletters as returned by min_process_glycans\n
  | Returns:
  | :-
  | Returns a list of node indices, in which node indices not belonging to the main chain have been replaced by empty strings
  """
  glycan_dic = {i:p for i,p in enumerate(glycan_parts)}
  glycan2 = bracket_removal(glycan)
  glycan_parts2 = min_process_glycans([glycan2])[0]
  i = 0
  for p in glycan_parts2:
    unfound = True
    while unfound:
      if glycan_dic[i] != p:
        glycan_dic[i] = ''
      else:
        unfound = False
      i += 1
  return list(glycan_dic.values())


def check_negative_look(matches, pattern, glycan):
  """filters the matches by whether they fulfill the negative lookahead/-behind condition\n
  | Arguments:
  | :-
  | matches (list): list of lists of node indices for each match
  | pattern (string): glyco-regular expression
  | glycan (string): glycan sequence in IUPAC-condensed\n
  | Returns:
  | :-
  | Returns a filtered list of matches
  """
  glycan_parts = min_process_glycans([glycan])[0]
  glycan_parts_main = process_main_branch(glycan, glycan_parts)
  part = convert_pattern_component(pattern.split('!')[1].split(')')[0])
  len_part = part.count('(')*2
  behind = '?<' in pattern
  if behind:
      starts = [min(m) for m in matches]
  else:
      part = ')'.join(part.split(')')[1:])
      starts = [max(m) for m in matches]
  def process_target_locs(starts, len_part, behind):
    target_locs = []
    for s in starts:
      segment = glycan_parts[s-len_part:s] if behind else glycan_parts[s+1:s+len_part+1]
      temp = glycan_parts_main[:s] if behind else glycan_parts_main[s+1:]
      segment_str = replace_every_second('('.join(segment), '(', ')')
      if segment_str not in glycan:
        temp_filtered = [k for k in temp if k]
        segment_slice = '('.join(temp_filtered[-len_part:]) if behind else '('+ '('.join(temp_filtered[:len_part])
        segment_str = replace_every_second(segment_slice, '(', ')')
      target_locs.append(segment_str if behind else ')'.join(segment_str.split(')')[1:]))
    return target_locs
  target_locs = process_target_locs(starts, len_part, behind)
  return [m for i, m in enumerate(matches) if not compare_glycans(target_locs[i], part)]


def filter_matches_by_location(matches, ggraph, match_location):
  """filters the matches by whether they fulfill the location condition (start/end)\n
  | Arguments:
  | :-
  | matches (list): list of lists of node indices for each match
  | ggraph (networkx): glycan graph as a networkx object
  | match_location (string): whether the match should have been at the "start" or "end" of the sequence\n
  | Returns:
  | :-
  | Returns a filtered list of matches
  """
  if match_location == 'start':
    degrees = {node: ggraph.degree[node] for node in ggraph}
    return [m for m in matches if degrees[m[0]] == 1]
  elif match_location == 'end':
    location_idx = len(ggraph) - 1
    return [m for m in matches if location_idx in m]
  return matches


def process_simple_pattern(p2, ggraph, libr, match_location):
  """for just a straight-up glycomotif, checks whether and where it can be found in glycan\n
  | Arguments:
  | :-
  | p2 (networkx): glycomotif graph as a networkx object
  | ggraph (networkx): glycan graph as a networkx object
  | libr (dict): dictionary of form glycoletter:index
  | match_location (string): whether the match should have been at the "start" or "end" of the sequence\n
  | Returns:
  | :-
  | Returns list of matches as list of node indices
  """
  res = subgraph_isomorphism(ggraph, p2, libr = libr, return_matches = True)
  if not res:
    return False
  matched, matches = res
  if not matched or not matches:
    return False
  if match_location:
    matches = filter_matches_by_location(matches, ggraph, match_location)
    if not matches:
      return False
  return [m for m in matches if all(x < y for x, y in zip(m, m[1:]))]


def calculate_len_matches_comb(len_matches):
  """takes a list of lengths of the matched sequences, returns a list of lengths considering the combination of the matches\n
  | Arguments:
  | :-
  | len_matches (list): list of match lengths\n
  | Returns:
  | :-
  | Returns list of lengths of possible combinations of matches
  """
  if not len_matches:
    return [0]
  if len(len_matches) == 1:
    return list(set(unwrap(len_matches))) + [0]
  cartesian_product = product(*len_matches)
  if len(len_matches) > 2:
    return list(set(unwrap(len_matches) + [sum(sum(sublist) for sublist in combination) for combination in cartesian_product]))
  else:
    return list(set(unwrap(len_matches) + [sum(combination) for combination in cartesian_product]))


def process_complex_pattern(p, p2, ggraph, glycan, libr, match_location):
  """for a glycomotif with regular expression modifiers, checks whether and where it can be found in glycan\n
  | Arguments:
  | :-
  | p (string): pattern component describing the glycomotif
  | p2 (dict): dictionary of form glycomotif : occurrence
  | ggraph (networkx): glycan graph as a networkx object
  | glycan (string): glycan sequence in IUPAC-condensed
  | libr (dict): dictionary of form glycoletter:index
  | match_location (string): whether the match should have been at the "start" or "end" of the sequence\n
  | Returns:
  | :-
  | Returns list of matches as list of node indices
  """
  counts_matches = [subgraph_isomorphism(ggraph, glycan_to_nxGraph(p_key, libr = libr),
                                         libr = libr, count = True, return_matches = True) for p_key in p2.keys()]
  if sum([k for k in counts_matches if isinstance(k, int)]) < 1 and isinstance(counts_matches[0], int):
    counts, matches = [], []
  else:
    counts, matches = zip(*[x for x in counts_matches if x])
  len_matches = [list(set([len(j) for j in k])) for k in matches]
  len_matches_comb = calculate_len_matches_comb(len_matches)
  len_motif = list(p2.keys())[0]
  len_motif = (len([l for l in len_motif.split('-') if l]) + len_motif.count('-'))
  len_motif = [v*len_motif for v in list(p2.values())[0]]
  if not any([l in len_motif for l in len_matches_comb]) and '{' in p:
    return False
  matches = list(matches) if not isinstance(matches, list) else matches
  if '=' in p or '!' in p:
    matches = unwrap(matches)
  if match_location:
    matches = filter_matches_by_location(matches, ggraph, match_location)
  matches = matches if (matches and matches[0] and isinstance(matches[0][0], int)) else unwrap(matches)
  matches = [m for m in matches if all(x < y for x, y in zip(m, m[1:]))]
  if '!' in p:
    matches = check_negative_look(matches, p, glycan)
  if '?<=' in p:
    len_look = p.split(')')[0].count('-') * 2
    matches = [m[len_look:] for m in matches]
  elif '?=' in p:
    len_look = p.split('=')[1]
    len_look = len([l for l in len_look.split('-') if l]) + len_look.count('-')
    if matches and not max([len(m) for m in matches]) > len_look:
      matches = [[m[1]] for m in matches if len(m) > 1]
      global lookahead_snuck_in
      lookahead_snuck_in = True
    else:
      matches = [m[:-len_look] for m in matches]
  return matches


def process_pattern(p, p2, ggraph, glycan, libr, match_location):
  """for a glycomotif, checks whether and where it can be found in glycan\n
  | Arguments:
  | :-
  | p (string): pattern component describing the glycomotif
  | p2 (dict): dictionary of form glycomotif : occurrence
  | ggraph (networkx): glycan graph as a networkx object
  | glycan (string): glycan sequence in IUPAC-condensed
  | libr (dict): dictionary of form glycoletter:index
  | match_location (string): whether the match should have been at the "start" or "end" of the sequence\n
  | Returns:
  | :-
  | Returns list of matches as list of node indices
  """
  if isinstance(p2, dict):
    return process_complex_pattern(p, p2, ggraph, glycan, libr, match_location)
  else:
    return process_simple_pattern(p2, ggraph, libr, match_location)


def match_it_up(pattern_components, glycan, ggraph, libr = None):
  """for a chunked glyco-regular expression, checks whether and where it can be found in glycan\n
  | Arguments:
  | :-
  | pattern_components (list): list of pattern components from glyco-regular expression
  | glycan (string): glycan sequence in IUPAC-condensed
  | ggraph (networkx): glycan graph as a networkx object
  | libr (dict): dictionary of form glycoletter:index; default:glycowork-internal libr\n
  | Returns:
  | :-
  | Returns list of tuples of form (pattern component, list of matches as node indices)
  """
  libr = libr if libr is not None else lib
  pattern_matches = []
  for p in pattern_components:
    p2 = convert_pattern_component(p)
    match_location = 'start' if '^' in p2 else 'end' if '$' in p2 else None
    p2 = glycan_to_nxGraph(p2.strip('^$'), libr = libr) if isinstance(p2, str) else p2
    res = process_pattern(p, p2, ggraph, glycan, libr, match_location)
    pattern_matches.append((p, res) if res else (p, []))
  return pattern_matches


def all_combinations(nested_list, min_len = 1, max_len = 2):
  """for a nested list of matches as node indices, creates possible combinations\n
  | Arguments:
  | :-
  | nested_list (list): list of matches as list of node indices
  | min_len (int): how long the combination has to be at least; default:1
  | max_len (int): how long the combination can be at most; default:2\n
  | Returns:
  | :-
  | Returns set of possible combinations of node indices for matches
  """
  # Flatten each sublist for intra-list combinations
  if isinstance(nested_list[0][0], int):
      nested_list = [nested_list]
  flat_sublists = [sorted(chain.from_iterable(sublist)) for sublist in nested_list]
  # Generate all combinations within each flattened sublist
  intra_list_combinations = set()
  for flat_list in flat_sublists:
      for i in range(min_len, max_len + 1):
          intra_list_combinations.update(combinations(flat_list, i))
  # Generate all combinations for inter-list combinations
  all_elements = sorted(chain.from_iterable(chain.from_iterable(nested_list)))
  inter_list_combinations = set(combinations(all_elements, i) for i in range(min_len, max_len + 1))
  # Combine intra-list and inter-list combinations, remove duplicates and sort
  return sorted(intra_list_combinations | inter_list_combinations)


def try_matching(current_trace, all_match_nodes, edges, min_occur = 1, max_occur = 1, branch = False):
  """tries to extend current trace to the matches of the next pattern component\n
  | Arguments:
  | :-
  | current_trace (list): list of node indices of current overall match
  | all_match_nodes (list): nested list of matches of next pattern component as lists of node indices
  | edges (list): list of edges as tuples of connected node indices in glycan graph
  | min_occur (int): how long the combination has to be at least; default:1
  | max_occur (int): how long the combination can be at most; default:1
  | branch (bool): whether next pattern component should be searched for in a different branch; default:False\n
  | Returns:
  | :-
  | Returns node indices from match that can be used to extend the trace
  """
  if max_occur == 0 and branch:
    return True
  if not all_match_nodes or max([len(k) for k in all_match_nodes]) < 1:
    return min_occur == 0
  last_trace_element = current_trace[-1]
  if max_occur > 1:
    all_match_nodes = all_combinations(all_match_nodes, min_len = min_occur, max_len = max_occur)
    #currently only working for branches of size 1
    idx = [all(last_trace_element - node == -2*(i+1) for i, node in enumerate(groupy)) for groupy in all_match_nodes]
  else:
    edges_set = set(edges)
    if all_match_nodes[0] and isinstance(all_match_nodes[0][0], list):
      all_match_nodes = unwrap(all_match_nodes)
    idx = [(last_trace_element - node[0] == -2 and not branch and (last_trace_element+1, node[0]) in edges_set) or \
     ((last_trace_element+1, node[0]) in edges_set) or \
      (last_trace_element - node[0] <= -2 and branch and not (last_trace_element+1, node[0]) in edges_set) and ((last_trace_element+1, node[-1]+2) in edges_set) or \
           (last_trace_element - node[0] == 2 and branch and (node[0]+1, last_trace_element+2) in edges_set)
           for node in all_match_nodes]
  matched_nodes = [node for i, node in enumerate(all_match_nodes) if (idx[i])]
  if branch and matched_nodes:
    matched_nodes = [nodes for nodes in matched_nodes if nodes and not (last_trace_element+1, nodes[0]) in edges_set]
  if not matched_nodes and min_occur == 0:
    return True
  return sorted(matched_nodes, key = lambda x: (len(x), x[-1]))


def parse_pattern(pattern):
  """extracts minimum/maximum occurrence from glyco-regular motif chunk\n
  | Arguments:
  | :-
  | pattern (string): pattern component of glyco-regular motif\n
  | Returns:
  | :-
  | Returns (minimum occurrence, maximum occurrence) as ints
  """
  temp = pattern.split('{')[1].split('}')[0].split(',') if '{' in pattern else None
  min_occur, max_occur = (int(temp[0]), int(temp[0])) if temp and len(temp) == 1 else \
                          (int(temp[0]) if temp[0] else 0, int(temp[1]) if temp[1] else 5) if temp else \
                          (0, 5) if '*' in pattern else \
                          (1, 5) if '+' in pattern else \
                          (0, 1) if '?' in pattern and '=' not in pattern and '!' not in pattern and '}' not in pattern else \
                          (1, 1)
  max_occur = min_occur if any([k in pattern for k in ['.?', '}?', '*?', '+?']]) else max_occur
  return min_occur, max_occur


def do_trace(start_pattern, idx, pattern_matches, optional_components, edges):
  all_traces, all_used_patterns = [], []
  for start_match in start_pattern[1]:
    if not start_match and optional_components.get(start_pattern[0], (99,99))[0] > 0:
      return [], []
    trace = copy.deepcopy(start_match)
    used_patterns = [start_pattern[0]]
    trace = trace[0] if isinstance(trace[0], list) else trace
    successful = True
    for component, component_matches in pattern_matches[idx+1:]:
      extended = False
      min_occur, max_occur = optional_components.get(component, (1, 1))
      branch = '(' in component and '(?' not in component
      to_extend = try_matching(trace, component_matches, edges, min_occur, max_occur, branch = branch)
      if to_extend:
        extend = to_extend[-1] if not isinstance(to_extend, bool) else []
        extend = list(extend) if isinstance(extend, tuple) else extend
        if len(extend) == 1 and extend[0] < trace[-1]:
          trace = trace[:-1] + extend + [trace[-1]]
        else:
          trace.extend(extend)
        if extend:
          used_patterns.append(component)
        extended = True
      if not extended:
        successful = False
    if successful:
      all_traces.append(trace)
      all_used_patterns.append(used_patterns)
  return all_traces, all_used_patterns


def trace_path(pattern_matches, ggraph):
  """given all matches to all pattern components, try to connect them in one trace\n
  | Arguments:
  | :-
  | pattern_matches (list): list of tuples of form (pattern component, list of matches as lists of node indices)
  | ggraph (networkx): glycan graph as a networkx object\n
  | Returns:
  | :-
  | i) nested list containing traces as lists of node indices
  | ii) nested list containing which pattern components are present in corresponding trace, as lists
  """
  all_traces, all_used_patterns = [], []
  patterns = [p[0] for p in pattern_matches]
  edges = list(ggraph.edges())
  optional_components = {p: parse_pattern(p) for p in patterns if any(x in p for x in ['{', '*', '+', '?'])}
  start_pattern = next(((p, m) for p, m in pattern_matches if (m and m[0] and not any([q in p for q in ['.?', '}?', '*?', '+?']])) or optional_components.get(p, (99,99))[0] > 0), patterns[0])
  idx = patterns.index(start_pattern[0])
  all_traces, all_used_patterns = do_trace(start_pattern, idx, pattern_matches, optional_components, edges)
  if not all_traces and optional_components.get(start_pattern[0], (99,99))[0] == 0:
    for p in range(len(patterns)-idx-1):
      if not all_traces and optional_components.get(start_pattern[0], (99,99))[0] == 0:
        idx += 1
        start_pattern = pattern_matches[idx]
        all_traces, all_used_patterns = do_trace(start_pattern, idx, pattern_matches, optional_components, edges)
        if all_traces:
          break
      else:
        break
  return all_traces, all_used_patterns


def fill_missing_in_list(lists):
  """Fills in the missing integers in a list of lists to make a full range\n
  | Arguments:
  | :-
  | lists (list of list of int): A list containing sublists of integers\n
  | Returns:
  | :-
  | Returns a list of list of int: The input list with missing integers filled in.
  """
  if lookahead_snuck_in:
    lists = [l[:-1] for l in lists]
  filled_lists = []
  for sublist in lists:
    if not sublist:
      filled_lists.append(sublist)
      continue
    filled_sublist = [sublist[0]]
    for i in range(1, len(sublist)):
      gap = sublist[i] - filled_sublist[-1]
      # Check whether the gap between current and previous element is exactly 2
      if gap == 2:
        filled_sublist.append(sublist[i] - 1)
      elif gap > 2 and gap % 2 == 0:
        filled_sublist.append(sublist[i-1] + 1)
      filled_sublist.append(sublist[i])
    filled_lists.append(filled_sublist)
  unique_tuples = set(tuple(lst) for lst in filled_lists)
  return [list(tpl) for tpl in unique_tuples]


def format_retrieved_matches(lists, ggraph):
  """transforms traces into glycan strings\n
  | Arguments:
  | :-
  | lists (list of list of int): A list of traces containing sublists of node indices
  | ggraph (networkx): glycan graph as a networkx object\n
  | Returns:
  | :-
  | Returns a list of glycan strings that match the glyco-regular expression
  """
  return sorted([graph_to_string(ggraph.subgraph(trace)) for trace in lists if nx.is_connected(ggraph.subgraph(trace))], key = len, reverse = True)


def compile(pattern):
  """pre-compiles glyco-regular expression for faster processing\n
  | Arguments:
  | :-
  | pattern (string): glyco-regular expression in the form of "Hex-HexNAc-([Hex|Fuc]){1,2}-HexNAc"\n
  | Returns:
  | :-
  | Returns a list of pattern components
  """
  pattern = pattern[1:] if pattern.startswith('r') else pattern
  return preprocess_pattern(pattern)


def get_match(pattern, glycan, libr = None, return_matches = True):
  """finds matches for a glyco-regular expression in a glycan\n
  | Arguments:
  | :-
  | pattern (string): glyco-regular expression in the form of "Hex-HexNAc-([Hex|Fuc]){1,2}-HexNAc"; accepts pre-compiled pattern
  | glycan (string): glycan sequence in IUPAC-condensed
  | libr (dict): dictionary of form glycoletter:index; default:glycowork-internal libr
  | return_matches (bool): whether to return True/False or return the matches as a list of strings; default:True\n
  | Returns:
  | :-
  | Returns either a boolean (return_matches = False) or a list of matches as strings (return_matches = True)
  """
  pattern = pattern[1:] if pattern.startswith('r') else pattern
  global lookahead_snuck_in
  lookahead_snuck_in = False
  if any([k in glycan for k in [';', '-D-', 'RES', '=']]):
    glycan = canonicalize_iupac(glycan)
  libr = libr if libr is not None else lib
  ggraph = glycan_to_nxGraph(glycan, libr = libr)
  pattern_components = preprocess_pattern(pattern) if isinstance(pattern, str) else pattern
  pattern_matches = match_it_up(pattern_components, glycan, ggraph, libr = libr)
  if pattern_matches:
    traces, used_patterns = trace_path(pattern_matches, ggraph)
    traces = fill_missing_in_list(traces)
    if traces:
      return True if not return_matches else format_retrieved_matches(traces, ggraph)
    else:
      return False if not return_matches else []
  return False if not return_matches else []
