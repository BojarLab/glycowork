import pandas as pd
import numpy as np
import re
import copy
from itertools import combinations_with_replacement
from collections import Counter
from sklearn.cluster import DBSCAN

from glycowork.glycan_data.loader import lib, motif_list, find_nth, unwrap, df_species, Hex, dHex, HexNAc, Sia
from glycowork.motif.processing import small_motif_find, min_process_glycans
from glycowork.motif.graph import compare_glycans
from glycowork.motif.annotate import annotate_dataset

chars = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T',
     'V','W','Y', 'X', 'Z'] + ['z']

def constrain_prot(proteins, libr = None):
  """Ensures that no characters outside of libr are present in proteins\n
  | Arguments:
  | :-
  | proteins (list): list of proteins as strings
  | libr (list): sorted list of amino acids occurring in proteins\n
  | Returns:
  | :-
  | Returns list of proteins with only permitted amino acids
  """
  if libr is None:
    libr = chars
  mega_prot = list(set(list(''.join(proteins))))
  forbidden = [k for k in mega_prot if k not in libr]
  for k in forbidden:
    proteins = [j.replace(k,'z') for j in proteins]
  return proteins

def prot_to_coded(proteins, libr = None):
  """Encodes protein sequences to be used in LectinOracle-flex\n
  | Arguments:
  | :-
  | proteins (list): list of proteins as strings
  | libr (list): sorted list of amino acids occurring in proteins\n
  | Returns:
  | :-
  | Returns list of encoded proteins with only permitted amino acids
  """
  if libr is None:
    libr = chars
  prots = [k[:min(len(k), 1000)] for k in proteins]
  prots = constrain_prot(prots, libr = libr)
  prots = [pad_sequence(string_to_labels(str(k).upper(),libr = libr),
                        max_length = 1000,
                        pad_label = len(libr)-1) for k in prots]
  return prots

def character_to_label(character, libr = None):
  """tokenizes character by indexing passed library\n
  | Arguments:
  | :-
  | character (string): character to index
  | libr (list): list of library items\n
  | Returns:
  | :-
  | Returns index of character in library
  """
  if libr is None:
    libr = lib
  character_label = libr.index(character)
  return character_label

def string_to_labels(character_string, libr = None):
  """tokenizes word by indexing characters in passed library\n
  | Arguments:
  | :-
  | character_string (string): string of characters to index
  | libr (list): list of library items\n
  | Returns:
  | :-
  | Returns indexes of characters in library
  """
  if libr is None:
    libr = lib
  return list(map(lambda character: character_to_label(character, libr), character_string))

def pad_sequence(seq, max_length, pad_label = None, libr = None):
  """brings all sequences to same length by adding padding token\n
  | Arguments:
  | :-
  | seq (list): sequence to pad (from string_to_labels)
  | max_length (int): sequence length to pad to
  | pad_label (int): which padding label to use
  | libr (list): list of library items\n\n
  | Returns:
  | :-
  | Returns padded sequence
  """
  if libr is None:
    libr = lib
  if pad_label is None:
    pad_label = len(libr)
  seq += [pad_label for i in range(max_length - len(seq))]
  return seq

def get_core(sugar):
  """retrieves core monosaccharide from modified monosaccharide\n
  | Arguments:
  | :-
  | sugar (string): monosaccharide or linkage\n
  | Returns:
  | :-
  | Returns core monosaccharide as string
  """
  easy_cores = ['GlcNAc', 'GalNAc', 'ManNAc', 'FucNAc', 'QuiNAc', 'RhaNAc', 'GulNAc',
                'IdoNAc', 'MurNAc', 'HexNAc', '6dAltNAc', 'AcoNAc', 'GlcA', 'AltA',
                'GalA', 'ManA', 'Tyv', 'Yer', 'Abe', 'GlcfNAc', 'GalfNAc', 'ManfNAc',
                'FucfNAc', 'IdoA', 'GulA', 'LDManHep', 'DDManHep', 'DDGlcHep', 'LyxHep', 'ManHep',
                'DDAltHep', 'IdoHep', 'DLGlcHep', 'GalHep']
  next_cores = ['GlcN', 'GalN', 'ManN', 'FucN', 'QuiN', 'RhaN', 'AraN', 'IdoN' 'Glcf', 'Galf', 'Manf',
                'Fucf', 'Araf', 'Lyxf', 'Xylf', '6dAltf', 'Ribf', 'Fruf', 'Apif', 'Kdof', 'Sedf',
                '6dTal', 'AltNAc', '6dAlt']
  hard_cores = ['Glc', 'Gal', 'Man', 'Fuc', 'Qui', 'Rha', 'Ara', 'Oli', 'Kdn', 'Gul', 'Lyx',
                'Xyl', 'Dha', 'Rib', 'Kdo', 'Tal', 'All', 'Pse', 'Leg', 'Asc',
                'Fru', 'Hex', 'Alt', 'Xluf', 'Api', 'Ko', 'Pau', 'Fus', 'Erwiniose',
                'Aco', 'Bac', 'Dig', 'Thre-ol', 'Ery-ol']
  if bool([ele for ele in easy_cores if(ele in sugar)]):
    return [ele for ele in easy_cores if(ele in sugar)][0]
  elif bool([ele for ele in next_cores if(ele in sugar)]):
    return [ele for ele in next_cores if(ele in sugar)][0]
  elif bool([ele for ele in hard_cores if(ele in sugar)]):
    return [ele for ele in hard_cores if(ele in sugar)][0]
  elif (('Neu' in sugar) and ('5Ac' in sugar)):
    return 'Neu5Ac'
  elif (('Neu' in sugar) and ('5Gc' in sugar)):
    return 'Neu5Gc'
  elif 'Neu' in sugar:
    return 'Neu'
  elif ((sugar.startswith('a')) or sugar.startswith('b')):
    return sugar
  elif re.match('^[0-9]+(-[0-9]+)+$', sugar):
    return sugar
  else:
    return 'Sug'

def get_stem_lib(libr):
  """creates a mapping file to map modified monosaccharides to core monosaccharides\n
  | Arguments:
  | :-
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset\n
  | Returns:
  | :-
  | Returns dictionary of form modified_monosaccharide:core_monosaccharide
  """
  return {k:get_core(k) for k in libr}

def stemify_glycan(glycan, stem_lib):
  """removes modifications from all monosaccharides in a glycan\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | stem_lib (dictionary): dictionary of form modified_monosaccharide:core_monosaccharide\n
  | Returns:
  | :-
  | Returns stemmed glycan as string
  """
  clean_list = list(stem_lib.values())
  for k in list(stem_lib.keys())[::-1][:-1]:
    if ((k not in clean_list) and (k in glycan) and not (k.startswith(('a','b'))) and not (re.match('^[0-9]+(-[0-9]+)+$', k))):
      while ((k in glycan) and (sum(1 for s in clean_list if k in s) <= 1)):
        glycan_start = glycan[:glycan.rindex('(')]
        glycan_part = glycan_start
        if k in glycan_start:
          cut = glycan_start[glycan_start.index(k):]
          try:
            cut = cut[:cut.index('(')]
          except:
            pass
          if cut not in clean_list:
            glycan_part = glycan_start[:glycan_start.index(k)]
            glycan_part = glycan_part + stem_lib[k]
          else:
            glycan_part = glycan_start
        try:
          glycan_mid = glycan_start[glycan_start.index(k) + len(k):]
          if ((cut not in clean_list) and (len(glycan_mid)>0)):
            glycan_part = glycan_part + glycan_mid
        except:
          pass
        glycan_end = glycan[glycan.rindex('('):]
        if k in glycan_end:
          if ']' in glycan_end:
            filt = ']'
          else:
            filt = ')'
          cut = glycan_end[glycan_end.index(filt)+1:]
          if cut not in clean_list:
            glycan_end = glycan_end[:glycan_end.index(filt)+1] + stem_lib[k]
          else:
            pass
        glycan = glycan_part + glycan_end
  return glycan

def stemify_dataset(df, stem_lib = None, libr = None,
                    glycan_col_name = 'target', rarity_filter = 1):
  """stemifies all glycans in a dataset by removing monosaccharide modifications\n
  | Arguments:
  | :-
  | df (dataframe): dataframe with glycans in IUPAC-condensed format in column glycan_col_name
  | stem_lib (dictionary): dictionary of form modified_monosaccharide:core_monosaccharide; default:created from lib
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset; default:lib
  | glycan_col_name (string): column name under which glycans are stored; default:target
  | rarity_filter (int): how often monosaccharide modification has to occur to not get removed; default:1\n
  | Returns:
  | :-
  | Returns df with glycans stemified
  """
  if libr is None:
    libr = lib
  if stem_lib is None:
    stem_lib = get_stem_lib(libr)
  pool = unwrap(min_process_glycans(df[glycan_col_name].values.tolist()))
  pool_count = Counter(pool)
  for k in list(set(pool)):
    if pool_count[k] > rarity_filter:
      stem_lib[k] = k
  df_out = copy.deepcopy(df)
  df_out[glycan_col_name] = [stemify_glycan(k, stem_lib) for k in df_out[glycan_col_name].values.tolist()]
  return df_out

def match_composition(composition, group, level, df = None,
                      mode = "minimal", libr = None, glycans = None,
                      relaxed = False):
    """Given a monosaccharide composition, it returns all corresponding glycans\n
    | Arguments:
    | :-
    | composition (dict): a dictionary indicating the composition to match (for example {"Fuc":1, "Gal":1, "GlcNAc":1})
    | group (string): name of the Species, Genus, Family, Order, Class, Phylum, Kingdom, or Domain used to filter
    | level (string): Species, Genus, Family, Order, Class, Phylum, Kingdom, or Domain
    | df (dataframe): glycan dataframe for searching glycan structures; default:df_species
    | mode (string): can be "minimal" or "exact" to match glycans that contain at least the specified composition or glycans matching exactly the requirements
    | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset; default:lib
    | glycans (list): custom list of glycans to check the composition in; default:None
    | relaxed (bool): specify if "minimal" means exact counts (False) or _at least_ (True); default:False\n
    | Returns:
    | :-
    | Returns list of glycans matching composition in IUPAC-condensed
    """
    if df is None:
      df = df_species
    if libr is None:
      libr = lib
    filtered_df = df[df[level] == group]
        
    exact_composition = {}
    if mode == "minimal":
        for element in libr:
            if element in composition:
                exact_composition[element] = composition.get(element)
        if glycans is None:
          glycan_list = filtered_df.target.values.tolist()
        else:
          glycan_list = copy.deepcopy(glycans)
        to_remove = []
        output_list = glycan_list
        for glycan in glycan_list:
            for key in exact_composition:
                glycan_count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(key), glycan))
                if relaxed:
                  if exact_composition[key] > glycan_count:
                    to_remove.append(glycan)
                else:
                  if exact_composition[key] != glycan_count :
                    to_remove.append(glycan)
        for element in to_remove:
            try :
                output_list.remove(element)
            except :
                a = 1
        output_list = list(set(output_list))
        #print(str(len(output_list)) + " glycan structures match your composition.")
        #for element in output_list:
        #    print(element)
        
    if mode == "exact":
        for element in libr:
            if element in composition:
                exact_composition[element] = composition.get(element)
        if glycans is None:
          glycan_list = filtered_df.target.values.tolist()
        else:
          glycan_list = glycans
        to_remove = []
        output_list = glycan_list
        for glycan in glycan_list:
            count_sum = 0
            for key in exact_composition :
                glycan_count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(key), glycan))
                count_sum = count_sum + exact_composition[key]
                if exact_composition[key] != glycan_count:
                    to_remove.append(glycan)
            monosaccharide_number_in_glycan = glycan.count("(") + 1
            if monosaccharide_number_in_glycan != count_sum:
                to_remove.append(glycan)
        for element in to_remove:
            try :
                output_list.remove(element)
            except :
                a = 1
        output_list = list(set(output_list))
        #print(str(len(output_list)) + " glycan structures match your composition.")
        #for element in output_list:
        #    print(element)
            
    return output_list

def match_composition_relaxed(composition, group, level, df = None,
                      mode = "exact", libr = None, reducing_end = None):
    """Given a coarse-grained monosaccharide composition (Hex, HexNAc, etc.), it returns all corresponding glycans\n
    | Arguments:
    | :-
    | composition (dict): a dictionary indicating the composition to match (for example {"Fuc":1, "Gal":1, "GlcNAc":1})
    | group (string): name of the Species, Genus, Family, Order, Class, Phylum, Kingdom, or Domain used to filter
    | level (string): Species, Genus, Family, Order, Class, Phylum, Kingdom, or Domain
    | df (dataframe): glycan dataframe for searching glycan structures; default:df_species
    | mode (string): can be "minimal" or "exact" to match glycans that contain at least the specified composition or glycans matching exactly the requirements; default:"exact"
    | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset; default:lib
    | reducing_end (string): filters possible glycans by reducing end monosaccharide; default:None\n
    | Returns:
    | :-
    | Returns list of glycans matching composition in IUPAC-condensed
    """
    if df is None:
      df = df_species
    if reducing_end is not None:
      df = df[df.target.str.endswith(reducing_end)].reset_index(drop = True)
    if libr is None:
      libr = lib
    input_composition = copy.deepcopy(composition)
    input_composition2 = copy.deepcopy(composition)
    original_composition = copy.deepcopy(composition)
    output_list = df[df[level] == group].target.values.tolist()

    input_composition2.pop('Hex', None)
    input_composition2.pop('dHex', None)
    input_composition2.pop('HexNAc', None)
    if len(input_composition2)>0:
      output_list = match_composition(input_composition2, group, level, df = df,
                                     mode = 'minimal', libr = libr,
                                       glycans = output_list, relaxed = True)
    if 'Hex' in input_composition:
      if any([j in input_composition for j in Hex]):
        relaxed = True
      else:
        relaxed = False
      hex_pool = list(combinations_with_replacement(Hex, input_composition['Hex']))
      hex_pool = [Counter(k) for k in hex_pool]
      input_composition.pop('Hex')
      output_list = [match_composition(k, group, level, df = df,
                                     mode = 'minimal', libr = libr,
                                       glycans = output_list, relaxed = relaxed) for k in hex_pool]
      output_list = list(set(unwrap(output_list)))
    if 'dHex' in input_composition:
      if any([j in input_composition for j in dHex]):
        relaxed = True
      else:
        relaxed = False
      dhex_pool = list(combinations_with_replacement(dHex, input_composition['dHex']))
      dhex_pool = [Counter(k) for k in dhex_pool]
      input_composition.pop('dHex')
      temp = [match_composition(k, group, level, df = df,
                                     mode = 'minimal', libr = libr,
                                       glycans = output_list, relaxed = relaxed) for k in dhex_pool]
      output_list = list(set(unwrap(temp)))
    if 'HexNAc' in input_composition:
      if any([j in input_composition for j in HexNAc]):
        relaxed = True
      else:
        relaxed = False
      hexnac_pool = list(combinations_with_replacement(HexNAc, input_composition['HexNAc']))
      hexnac_pool = [Counter(k) for k in hexnac_pool]
      input_composition.pop('HexNAc')
      temp = [match_composition(k, group, level, df = df,
                                     mode = 'minimal', libr = libr,
                                       glycans = output_list, relaxed = relaxed) for k in hexnac_pool]
      output_list = list(set(unwrap(temp)))
      
    if mode == 'exact':
      monosaccharide_count = sum(original_composition.values())
      monosaccharide_types = list(original_composition.keys())
      if 'Hex' in original_composition:
        monosaccharide_types = monosaccharide_types + Hex
      if 'HexNAc' in original_composition:
        monosaccharide_types = monosaccharide_types + HexNAc
      if 'dHex' in original_composition:
        monosaccharide_types = monosaccharide_types + dHex
      output_list = [k for k in output_list if k.count('(') == monosaccharide_count-1]
      output_list = [k for k in output_list if not any([j not in monosaccharide_types for j in list(set(min_process_glycans([k])[0])) if j[0].isupper()])]
      if 'Hex' in original_composition and len(input_composition2)>0:
        output_list = [k for k in output_list if all([k.count(j) == original_composition[j] for j in HexNAc[:-1] if j in original_composition])]
        output_list = [k for k in output_list if all([k.count(j) == original_composition[j] for j in dHex[:-1] if j in original_composition])]
      elif 'dHex' in original_composition and len(input_composition2)>0:
        output_list = [k for k in output_list if all([k.count(j) == original_composition[j] for j in Hex[:-1] if j in original_composition])]
        output_list = [k for k in output_list if all([k.count(j) == original_composition[j] for j in HexNAc[:-1] if j in original_composition])]
      elif 'HexNAc' in original_composition and len(input_composition2)>0:
        output_list = [k for k in output_list if all([k.count(j) == original_composition[j] for j in Hex[:-1] if j in original_composition])]
        output_list = [k for k in output_list if all([k.count(j) == original_composition[j] for j in dHex[:-1] if j in original_composition])]
    return output_list

def condense_composition_matching(matched_composition, libr = None):
  """Given a list of glycans matching a composition, find the minimum number of glycans characterizing this set\n
  | Arguments:
  | :-
  | matched_composition (list): list of glycans matching to a composition
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset; default:lib\n
  | Returns:
  | :-
  | Returns minimal list of glycans that match a composition
  """
  if libr is None:
    libr = lib
  match_matrix = [[compare_glycans(k, j,libr = libr, wildcards = True,
                                  wildcard_list = [libr.index('bond')]) for j in matched_composition] for k in matched_composition]
  match_matrix = pd.DataFrame(match_matrix)
  match_matrix.columns = matched_composition
  clustering = DBSCAN(eps = 1, min_samples = 1).fit(match_matrix)
  cluster_labels = clustering.labels_
  num_clusters = len(list(set(cluster_labels)))
  sum_glycans = []
  for k in range(num_clusters):
    cluster_glycans = [matched_composition[j] for j in range(len(cluster_labels)) if cluster_labels[j] == k]
    #print(cluster_glycans)
    #idx = np.argmin([j.count('bond') for j in cluster_glycans])
    county = [j.count('bond') for j in cluster_glycans]
    idx = np.where(county == np.array(county).min())[0]
    if len(idx) == 1:
      sum_glycans.append(cluster_glycans[idx[0]])
    else:
      for j in idx:
        sum_glycans.append(cluster_glycans[j])
    #sum_glycans.append(cluster_glycans[idx])
  #print("This matching can be summarized by " + str(num_clusters) + " glycans.")
  return sum_glycans

def compositions_to_structures(composition_list, abundances, group, level,
                               df = None, libr = None, reducing_end = None,
                               verbose = False):
  """wrapper function to map compositions to structures, condense them, and match them with relative intensities\n
  | Arguments:
  | :-
  | composition_list (list): list of composition dictionaries of the form {'Hex': 1, 'HexNAc': 1}
  | abundances (dataframe): every row one glycan (matching composition_list in order), every column one sample; pd.DataFrame([range(len(composition_list))]*2).T if not applicable
  | group (string): name of the Species, Genus, Family, Order, Class, Phylum, Kingdom, or Domain used to filter
  | level (string): Species, Genus, Family, Order, Class, Phylum, Kingdom, or Domain
  | df (dataframe): glycan dataframe for searching glycan structures; default:df_species
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset; default:lib
  | reducing_end (string): filters possible glycans by reducing end monosaccharide; default:None
  | verbose (bool): whether to print any non-matching compositions; default:False\n
  | Returns:
  | :-
  | Returns dataframe of (matched structures) x (relative intensities)
  """
  if libr is None:
    libr = lib
  if df is None:
    df = df_species
  out_df = []
  not_matched = []
  for k in range(len(composition_list)):
    matched = match_composition_relaxed(composition_list[k], group, level,
                                reducing_end = reducing_end, df = df, libr = libr)
    if len(matched)>0:
        condensed = condense_composition_matching(matched, libr = libr)
        matched_data = [abundances.iloc[k,1:].values.tolist()]*len(condensed)
        for ele in range(len(condensed)):
            out_df.append([condensed[ele]] + matched_data[ele])
    else:
        not_matched.append(composition_list[k])
  df_out = pd.DataFrame(out_df)
  print(str(len(not_matched)) + " compositions could not be matched. Run with verbose = True to see which compositions.")
  if verbose:
    print(not_matched)
  return df_out

def structures_to_motifs(df, libr = None, feature_set = ['exhaustive'],
                         form = 'wide'):
  """function to convert relative intensities of glycan structures to those of glycan motifs\n
  | Arguments:
  | :-
  | df (dataframe): function expects glycans in the first column and rel. intensities of each sample in a new column
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset; default:lib
  | feature_set (list): which feature set to use for annotations, add more to list to expand; default is 'exhaustive'; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans) and 'exhaustive' (all mono- and disaccharide features)
  | form (string): whether to return 'wide' or 'long' dataframe; default:'wide'\n
  | Returns:
  | :-
  | Returns dataframe of motifs, relative intensities, and sample IDs
  """
  if libr is None:
    libr = lib
  annot = annotate_dataset(df.iloc[:,0].values.tolist(), libr = libr,
                           feature_set = feature_set, condense = True)
  annot2 = pd.concat([annot.reset_index(drop = True), df.iloc[:,1:]], axis = 1)
  out_tuples = []
  for k in range(len(annot2)):
    for j in range(annot.shape[1]):
      if annot2.iloc[k,j]>0:
          out_tuples.append([annot2.columns.values.tolist()[j]] + df.iloc[k, 1:].values.tolist())
  motif_df = pd.DataFrame(out_tuples)
  motif_df = motif_df.groupby(motif_df.columns.values.tolist()[0]).mean().reset_index()
  if form == 'wide':
    motif_df.columns = ['glycan'] + ['sample'+str(k) for k in range(1, motif_df.shape[1])]
    return motif_df
  elif form == 'long':
    motif_df.columns = ['glycan'] + ['rel_intensity' for k in range(1, motif_df.shape[1])]
    sample_dfs = [pd.concat([motif_df.iloc[:,0], motif_df.iloc[:,k]], axis = 1) for k in range(1, motif_df.shape[1])]
    out = pd.concat(sample_dfs, axis = 0, ignore_index = True)
    out['sample_id'] = unwrap([[k]*len(sample_dfs[k]) for k in range(len(sample_dfs))])
    return out
