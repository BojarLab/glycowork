import re
import copy
import networkx as nx
from itertools import product, combinations, chain
from typing import Dict, List, Union, Optional, Tuple
from glycowork.glycan_data.loader import replace_every_second, unwrap, share_neighbor
from glycowork.motif.processing import min_process_glycans, bracket_removal, canonicalize_iupac
from glycowork.motif.graph import graph_to_string, subgraph_isomorphism, compare_glycans, glycan_to_nxGraph


def preprocess_pattern(pattern: str # Glyco-regular expression like "Hex-HexNAc-([Hex|Fuc]){1,2}-HexNAc"
                     ) -> List[str]: # List of pattern chunks
  "Transform glyco-regular expression into chunks"
  # Use regular expression to identify the conditional parts and keep other chunks together
  pattern = pattern.replace('.', 'Monosaccharide')
  components = re.split(r'(-?\s*\(?\[.*?\]\)?\s*(?:\{,?\d*,?\d*\}\?|\{,?\d*,?\d*\}|\*\?|\+\?|\?|\*|\+)\s*-?)', pattern)
  # Remove any empty strings and trim whitespace
  return [x.strip('-').strip() for x in components if x]


def specify_linkages(pattern_component: str # Chunk of glyco-regular expression
                   ) -> str: # Specified component with linkages
  "Convert expression linkages from shorthand to full notation, such as Mana6 to Man(a1-6) or Galb3/4 to Gal(b1-3/4)"
  if re.search(r"[\d|\?]\(|\d$", pattern_component):
    pattern = re.compile(r"([ab\?])(\d+/\d+|\d|\?)\(\?1-\?\)")
    def replacer(match):
      letter, number = match.groups()
      return f'({letter}1-{number})'
    pattern_component = pattern.sub(replacer, pattern_component)
  return re.sub(r'(5Ac|5Gc|Kdn|Sia)\(a1', r'\1(a2', pattern_component)


def replace_patterns(s: str # String to process
                   ) -> str: # Processed string
  "Replace pattern strings with standardized forms"
  return s.replace('-', '(?1-?)').replace('5Ac(?1', '5Ac(?2').replace('5Gc(?1', '5Gc(?2')


def process_occurrence(occ_str: str # Content between {} of pattern component
                     ) -> List[int]: # [min_occurrence, max_occurrence]
  "Process min/max occurrence of pattern component"
  occ = occ_str.split(',')
  if len(occ) == 1:
    return [int(occ[0]), int(occ[0])]
  return [int(occ[0]) if occ[0] else 0, int(occ[1]) if occ[1] else 5]


def process_question_mark(s: str, # Original pattern component
                        p: str # Content between [] in pattern
                       ) -> Tuple[List[int], List[str]]: # ([occurrences], [patterns])
  "Process components with lookahead/behind or ? characters"
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


def convert_pattern_component(pattern_component: str # Chunk of glyco-regular expression
                           ) -> Union[str, Dict[str, List[int]]]: # String or dict mapping to occurrences
  "Process pattern component into string or occurrence dictionary"
  if not any([k in pattern_component for k in ['[', '{', '*', '+', '=', '<!', '?!']]):
    if pattern_component[-1].isdigit() or pattern_component.endswith('?'):
      pattern_component += '-'
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
    occurrence = list(range(0, 8)) if '*' in pattern_component else list(range(1, 8))
    if '?' in pattern_component:
      occurrence = [occurrence[0], occurrence[0]]
  elif '?' in pattern_component:
    occurrence, pattern = process_question_mark(pattern_component, pattern)
  if pattern is None:
    pattern = replace_patterns(pattern_component)
  return {specify_linkages(p): occurrence for p in pattern}


def process_main_branch(glycan: str, # Glycan in IUPAC-condensed
                       glycan_parts: List[str] # List of glycoletters as returned by min_process_glycans
                      ) -> List[str]: # Node index list with non-main chain nodes as empty strings
  "Extract main chain node indices of glycan"
  glycan_dic = {i: p for i,p in enumerate(glycan_parts)}
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


def check_negative_look(matches: List[List[int]], # List of node index lists for matches
                       pattern: str, # Glyco-regular expression
                       glycan: str # Glycan in IUPAC-condensed
                      ) -> List[List[int]]: # Filtered matches
  "Filter matches by negative lookahead/behind conditions"
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


def filter_matches_by_location(matches: List[List[int]], # List of node index lists
                             ggraph: nx.Graph, # Glycan graph
                             match_location: Optional[str] # Location to match: start/end/internal
                            ) -> List[List[int]]: # Filtered matches
  "Filter matches by location requirement"
  if matches and matches[0] and matches[0][0] and isinstance(matches[0][0], list):
    matches = unwrap(matches)
  if match_location == 'start':
    degrees = {node: ggraph.degree[node] for node in ggraph}
    return [m for m in matches if degrees[m[0]] == 1]
  elif match_location == 'end':
    location_idx = len(ggraph) - 1
    return [m for m in matches if location_idx in m]
  elif match_location == 'internal':
    return [m for m in matches if m[0] > 0 and m[-1] < len(ggraph) - 1]
  return matches


def process_simple_pattern(p2: nx.Graph, # Glycomotif graph
                         ggraph: nx.Graph, # Glycan graph
                         match_location: Optional[str] # Location to match: start/end/internal
                        ) -> Union[List[List[int]], bool]: # Match node indices or False
  "Check if simple glycomotif exists in glycan"
  matched, matches = subgraph_isomorphism(ggraph, p2, return_matches = True)
  if not matched or not matches:
    return False
  if match_location:
    matches = filter_matches_by_location(matches, ggraph, match_location)
    if not matches:
      return False
  if p2.number_of_nodes() % 2 == 0:
    matches = [m[:-1] for m in matches]
  return [m for m in matches if all(x < y for x, y in zip(m, m[1:]))]


def calculate_len_matches_comb(len_matches: List[List[int]] # List of match lengths
                            ) -> List[int]: # Combined match lengths
  "Calculate lengths considering combinations of matches"
  if not len_matches:
    return [0]
  if len(len_matches) == 1:
    return list(set(unwrap(len_matches))) + [0]
  cartesian_product = product(*len_matches)
  if len(len_matches) > 2:
    return list(set(unwrap(len_matches) + [sum(sum(sublist) for sublist in combination) for combination in cartesian_product]))
  else:
    return list(set(unwrap(len_matches) + [sum(combination) for combination in cartesian_product]))


def process_complex_pattern(p: str, # Pattern component
                          p2: Dict[str, List[int]], # Dict mapping pattern to occurrences
                          ggraph: nx.Graph, # Glycan graph
                          glycan: str, # Glycan in IUPAC-condensed
                          match_location: Optional[str] # Location to match: start/end/internal
                         ) -> Union[List[List[int]], bool]: # Match node indices or False
  "Check if complex glycomotif, containing regular expression modifiers, exists in glycan"
  if not any('-' in p_key for p_key in p2.keys()):
    p2_keys = [specify_linkages(replace_patterns(p_key + '-' if p_key[-1].isdigit() or p_key[-1] == '?' else p_key)) for p_key in p2.keys()]
  else:
    p2_keys = list(p2.keys())
  counts_matches = [subgraph_isomorphism(ggraph, glycan_to_nxGraph(p_key.strip('^$%')),
                                         count = True, return_matches = True) for p_key in p2_keys]
  if sum([k for k in counts_matches if isinstance(k, int)]) < 1 and isinstance(counts_matches[0], int):
    matches = []
  else:
    _, matches = zip(*[x for x in counts_matches if x])
  matches = [[n[:-1] for n in m] if p2_keys[i].endswith(')') else m for i, m in enumerate(matches)]
  len_matches = [list(set([len(j) for j in k])) for k in matches]
  len_matches_comb = calculate_len_matches_comb(len_matches)
  len_motif = list(p2.keys())[0]
  len_motif = (len([le for le in len_motif.split('-') if le]) + len_motif.count('-')) + len_motif[-1].isdigit() - p2_keys[0].endswith(')')
  len_motif = [v*len_motif for v in list(p2.values())[0]]
  if not any([le in len_motif for le in len_matches_comb]) and '{' in p:
    return False
  matches = list(matches) if not isinstance(matches, list) else matches
  if '=' in p or '<!' in p or '?!' in p:
    matches = unwrap(matches)
  if match_location:
    matches = filter_matches_by_location(matches, ggraph, match_location)
  matches = matches if (matches and matches[0] and isinstance(matches[0][0], int)) else unwrap(matches)
  matches = [m for m in matches if all(x < y for x, y in zip(m, m[1:]))]
  if '<!' in p or '?!' in p:
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


def match_it_up(pattern_components: List[str], # Pattern chunks
                glycan: str, # Glycan in IUPAC-condensed
                ggraph: nx.Graph # Glycan graph
               ) -> List[Tuple[str, List[List[int]]]]: # [(pattern, matches)]
  "Find pattern component matches in glycan"
  pattern_matches = []
  for p in pattern_components:
    p2 = convert_pattern_component(p)
    if isinstance(p2, dict):
      first_key = list(p2.keys())[0]
      match_location = 'start' if '^' in first_key else 'end' if '$' in first_key else 'internal' if '%' in first_key else None
    else:
      match_location = 'start' if '^' in p2 else 'end' if '$' in p2 else 'internal' if '%' in p2 else None
      p2 = glycan_to_nxGraph(p2.strip('^$%'))
    res = process_complex_pattern(p, p2, ggraph, glycan, match_location) if isinstance(p2, dict) else process_simple_pattern(p2, ggraph, match_location)
    res = sorted(res) if isinstance(res, list) and all(len(inner) == 1 for inner in res) else res
    pattern_matches.append((p, res) if res else (p, []))
  return pattern_matches


def all_combinations(nested_list: List[List[int]], # List of match indices
                    min_len: int = 1, # Minimum combination length
                    max_len: int = 2 # Maximum combination length
                   ) -> List[Tuple[int, ...]]: # Possible index combinations
  "Create possible combinations from nested list of matches"
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
  inter_list_combinations = set()
  for i in range(min_len, max_len + 1):
    inter_list_combinations.update(set(combinations(all_elements, i)))
  # Combine intra-list and inter-list combinations, remove duplicates and sort
  return sorted(intra_list_combinations | inter_list_combinations)


def try_matching(current_trace: List[int], # Current match indices
                all_match_nodes: List[List[int]], # Next component matches
                edges: List[Tuple[int, int]], # Graph edges
                min_occur: int = 1, # Minimum occurrences; default:1
                max_occur: int = 1, # Maximum occurrences; default:1
                branch: bool = False # Whether to search different branch for next pattern component; default:False
               ) -> Union[List[List[int]], bool]: # Extended trace or False
  "Try extending current trace to next pattern matches"
  if max_occur == 0 and branch:
    return True
  if not all_match_nodes or max([len(k) for k in all_match_nodes]) < 1:
    return min_occur == 0
  last_node = current_trace[-1]
  edges_set = set(edges)
  if max_occur > 1:
    all_match_nodes = all_combinations(all_match_nodes, min_len = min_occur, max_len = max_occur)
    #currently only working for branches of size 1
    idx = [all(last_node - node == -2*(i+1) for i, node in enumerate(groupy)) for groupy in all_match_nodes]
  else:
    if all_match_nodes[0] and isinstance(all_match_nodes[0][0], list):
      all_match_nodes = unwrap(all_match_nodes)
    idx = [node[0] > last_node and ((not branch and share_neighbor(edges_set, last_node, node[0]))
                                    or (branch and share_neighbor(edges_set, last_node+1, node[-1]+1))) for node in all_match_nodes]
  matched_nodes = [node for i, node in enumerate(all_match_nodes) if (idx[i]) and node]
  if branch and matched_nodes:
    matched_nodes = [nodes for nodes in matched_nodes if nodes and not (last_node+1, nodes[0]) in edges_set]
  if not matched_nodes and min_occur == 0:
    return True
  return sorted(matched_nodes, key = lambda x: (len(x), x[-1]))


def parse_pattern(pattern: str # Pattern component from glyco-regular motif
                ) -> Tuple[int, int]: # (Minimum occurrence, Maximum occurrence)
  "Extract occurrence limits from glyco-regular motif pattern"
  temp = pattern.split('{')[1].split('}')[0].split(',') if '{' in pattern else None
  min_occur, max_occur = (int(temp[0]), int(temp[0])) if temp and len(temp) == 1 else \
                          (int(temp[0]) if temp[0] else 0, int(temp[1]) if temp[1] else 8) if temp else \
                          (0, 8) if '*' in pattern else \
                          (1, 8) if '+' in pattern else \
                          (0, 1) if '?' in pattern and '=' not in pattern and '!' not in pattern and '}' not in pattern else \
                          (1, 1)
  max_occur = min_occur if any([k in pattern for k in ['.?', '}?', '*?', '+?']]) else max_occur
  return min_occur, max_occur


def do_trace(start_pattern: Tuple[str, List[List[int]]], # (Pattern, Match indices)
            idx: int, # Current pattern index
            pattern_matches: List[Tuple[str, List[List[int]]]], # List of (Pattern, Matches)
            optional_components: Dict[str, Tuple[int, int]], # Pattern to min/max occurrences
            edges: List[Tuple[int, int]] # Graph edges
           ) -> Tuple[List[List[int]], List[List[str]]]: # (Match traces, Used patterns)
  "Try to extend current trace through pattern component matches"
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


def trace_path(pattern_matches: List[Tuple[str, List[List[int]]]], # [(pattern, matches)]
              ggraph: nx.Graph # Glycan graph
             ) -> Tuple[List[List[int]], List[List[str]]]: # (traces, used patterns)
  "Connect pattern component matches into complete traces"
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


def fill_missing_in_list(lists: List[List[int]] # Lists of indices
                       ) -> List[List[int]]: # Lists with gaps filled
  "Fill missing integers in lists to make full ranges"
  if lookahead_snuck_in:
    lists = [le[:-1] for le in lists]
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


def format_retrieved_matches(lists: List[List[int]], # List of traces
                           ggraph: nx.Graph # Glycan graph
                          ) -> List[str]: # Matching glycan strings
  "Convert traces into glycan strings"
  return sorted([canonicalize_iupac(graph_to_string(ggraph.subgraph(trace))) for trace in lists if nx.is_connected(ggraph.subgraph(trace))], key = len, reverse = True)


def compile_pattern(pattern: str # Glyco-regular expression, e.g., "Hex-HexNAc-([Hex|Fuc]){1,2}-HexNAc"
                  ) -> List[str]: # Pre-compiled pattern chunks
  "Pre-compile glyco-regular expression for faster processing"
  pattern = pattern[1:] if pattern.startswith('r') else pattern
  return preprocess_pattern(pattern)


def get_match(pattern: Union[str, List[str]], # Expression or pre-compiled pattern; e.g., "Hex-HexNAc-([Hex|Fuc]){1,2}-HexNAc"
              glycan: Union[str, nx.Graph], # Glycan string or graph
              return_matches: bool = True # Whether to return matches vs boolean
             ) -> Union[bool, List[str]]: # Match results
  "Find matches for glyco-regular expression in glycan"
  pattern = pattern[1:] if isinstance(pattern, str) and pattern.startswith('r') else pattern
  global lookahead_snuck_in
  lookahead_snuck_in = False
  if any([k in glycan for k in [';', '-D-', 'RES', '=']]):
    glycan = canonicalize_iupac(glycan)
  if isinstance(glycan, str):
    ggraph = glycan_to_nxGraph(glycan)
  else:
    ggraph = glycan
    glycan = graph_to_string(ggraph)
  pattern_components = preprocess_pattern(pattern) if isinstance(pattern, str) else pattern
  pattern_matches = match_it_up(pattern_components, glycan, ggraph)
  if pattern_matches:
    traces, _ = trace_path(pattern_matches, ggraph)
    traces = fill_missing_in_list(traces)
    if traces:
      return True if not return_matches else format_retrieved_matches(traces, ggraph)
    else:
      return False if not return_matches else []
  return False if not return_matches else []


def get_match_batch(pattern: str, # Glyco-regular expression; e.g., "Hex-HexNAc-([Hex|Fuc]){1,2}-HexNAc"
                   glycan_list: List[Union[str, nx.Graph]], # List of glycans
                   return_matches: bool = True # Whether to return matches vs boolean
                  ) -> Union[List[bool], List[List[str]]]: # Match results for each glycan
  "Find glyco-regular expression matches in list of glycans"
  pattern = compile_pattern(pattern)
  return [get_match(pattern, g, return_matches = return_matches) for g in glycan_list]


def reformat_glycan_string(glycan: str # Glycan in IUPAC-condensed
                         ) -> str: # Reformatted pattern string
  "Convert glycan string to pattern format"
  # Contract linkages
  glycan = re.sub(r"\((\w)(\d+)-(\d+)\)", r"\1\3-", glycan)
  glycan = re.sub(r"\((\w)(\d+)-(\?)\)", r"\1?-", glycan)  # Handle cases with '?'
  # Format branches
  glycan = re.sub(r"\[", "([", glycan)
  glycan = re.sub(r"-\]", "]){1}-", glycan)
  return glycan.strip('-')


def motif_to_regex(motif: str # Glycan in IUPAC-condensed
                  ) -> str: # Regular expression
  "Convert glycan motif to regular expression pattern"
  motif = canonicalize_iupac(motif)
  pattern = reformat_glycan_string(motif)
  if not get_match(pattern, motif, return_matches = False):
    raise ValueError("Failed to make effective regular expression.")
  return pattern
