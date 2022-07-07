import pandas as pd
import numpy as np
import networkx as nx
import re
import copy
import math
import pkg_resources
from itertools import combinations_with_replacement, product
from collections import Counter
from sklearn.cluster import DBSCAN

from glycowork.glycan_data.loader import lib, motif_list, unwrap, find_nth, df_species, Hex, dHex, HexA, HexN, HexNAc, Pen, Sia, linkages
from glycowork.motif.processing import small_motif_find, min_process_glycans, choose_correct_isoform
from glycowork.motif.graph import compare_glycans, glycan_to_nxGraph, graph_to_string
from glycowork.motif.annotate import annotate_dataset, find_isomorphs

chars = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T',
     'V','W','Y', 'X', 'Z'] + ['z']

io = pkg_resources.resource_stream(__name__, "mz_to_composition.csv")
mapping_file = pd.read_csv(io)

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
  #get list of unique characters
  mega_prot = list(set(list(''.join(proteins))))
  #check whether any character is not in libr and replace it with a 'z' placeholder character
  forbidden = [k for k in mega_prot if k not in libr]
  for k in forbidden:
    proteins = [j.replace(k,'z') for j in proteins]
  return proteins

def prot_to_coded(proteins, libr = None, pad_len = 1000):
  """Encodes protein sequences to be used in LectinOracle-flex\n
  | Arguments:
  | :-
  | proteins (list): list of proteins as strings
  | libr (list): sorted list of amino acids occurring in proteins
  | pad_len (int): length up to which the proteins are padded\n
  | Returns:
  | :-
  | Returns list of encoded proteins with only permitted amino acids
  """
  if libr is None:
    libr = chars
  #cut off protein sequence above pad_len
  prots = [k[:min(len(k), pad_len)] for k in proteins]
  #replace forbidden characters with 'z'
  prots = constrain_prot(prots, libr = libr)
  #pad up to a length of pad_len
  prots = [pad_sequence(string_to_labels(str(k).upper(),libr = libr),
                        max_length = pad_len,
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
                'IdoNAc', 'Ins', 'MurNAc', 'HexNAc', '6dAltNAc', 'AcoNAc', 'GlcA', 'AltA',
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
  elif sugar.startswith('a') or sugar.startswith('b') or sugar.startswith('z'):
    return sugar
  elif re.match('^[0-9]+(-[0-9]+)+$', sugar):
    return sugar
  else:
    return 'Monosaccharide'

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

def stemify_glycan(glycan, stem_lib = None, libr = None):
  """removes modifications from all monosaccharides in a glycan\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | stem_lib (dictionary): dictionary of form modified_monosaccharide:core_monosaccharide; default:created from lib
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset; default:lib\n
  | Returns:
  | :-
  | Returns stemmed glycan as string
  """
  if libr is None:
    libr = lib
  if stem_lib is None:
    stem_lib = get_stem_lib(libr)
  if '(' not in glycan:
    glycan = get_core(glycan)
    return glycan
  clean_list = list(stem_lib.values())
  for k in list(stem_lib.keys())[::-1][:-1]:
    #for each monosaccharide, check whether it's modified
    if ((k not in clean_list) and (k in glycan) and not (k.startswith(('a','b','z'))) and not (re.match('^[0-9]+(-[0-9]+)+$', k))):
      #go at it until all modifications are stemified
      while ((k in glycan) and (sum(1 for s in clean_list if k in s) <= 1)):
        glycan_start = glycan[:glycan.rindex('(')]
        glycan_part = glycan_start
        #narrow it down to the offending monosaccharide
        if k in glycan_start:
          cut = glycan_start[glycan_start.index(k):]
          try:
            cut = cut[:cut.index('(')]
          except:
            pass
          #replace offending monosaccharide with stemified monosaccharide
          if cut not in clean_list:
            glycan_part = glycan_start[:glycan_start.index(k)]
            glycan_part = glycan_part + stem_lib[k]
          else:
            glycan_part = glycan_start
        #check to see whether there is anything after the modification that should be appended
        try:
          glycan_mid = glycan_start[glycan_start.index(k) + len(k):]
          if ((cut not in clean_list) and (len(glycan_mid) > 0)):
            glycan_part = glycan_part + glycan_mid
        except:
          pass
        #handling the reducing end
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
  #get pool of monosaccharides, decide which one to stemify based on rarity
  pool = unwrap(min_process_glycans(df[glycan_col_name].values.tolist()))
  pool_count = Counter(pool)
  for k in list(set(pool)):
    if pool_count[k] > rarity_filter:
      stem_lib[k] = k
  #stemify all offending monosaccharides
  df_out = copy.deepcopy(df)
  df_out[glycan_col_name] = [stemify_glycan(k, stem_lib = stem_lib,
                                            libr = libr) for k in df_out[glycan_col_name].values.tolist()]
  return df_out

def mz_to_composition(mz_value, mode = 'positive', mass_value = 'monoisotopic',
                      sample_prep = 'underivatized', mass_tolerance = 0.2, human = True,
                      glycan_class = 'N', check_all_adducts = False, check_specific_adduct = None,
                      ptm = False):
  """mapping a m/z value to one or more matching monosaccharide compositions\n
  | Arguments:
  | :-
  | mz_value (float): the actual m/z value from mass spectrometry
  | mode (string): whether mz_value comes from MS in 'positive' or 'negative' mode; default:'positive'
  | mass_value (string): whether the expected mass is 'monoisotopic' or 'average'; default:'monoisotopic'
  | sample_prep (string): whether the glycans has been 'underivatized', 'permethylated', or 'peracetylated'; default:'underivatized'
  | mass_tolerance (float): how much deviation to tolerate for a match; default:0.2
  | human (bool): whether to only consider human monosaccharide types; default:True
  | glycan_class (string): which glycan class does the m/z value stem from, 'N' or 'O' linked glycans; default:'N'
  | check_all_adducts (bool): whether to also check for matches with ion adducts (depending on mode); default:False
  | check_specific_adduct (string): choose adduct from 'H+', 'Na+', 'K+', 'H', 'Acetate', 'Trifluoroacetic acid'; default:None
  | ptm (bool): whether to check for post-translational modification (sulfation, phosphorylation); default:False\n
  | Returns:
  | :-
  | Returns a list of matching compositions in dict form
  """
  #get correct masses depending on sample characteristics
  idx = sample_prep + '_' + mass_value
  #needed because of implied water loss in the residue masses
  free_reducing_end = 18.0105546

  #Hex,HexNAc,dHex,Neu5Ac,Neu5Gc,Pen,Kdn,HexA,S,P
  if glycan_class == 'N':
    ranges = [math.floor(mz_value/160),math.floor(mz_value/203),math.floor(mz_value/146),
              math.floor(mz_value/291)-1,math.floor(mz_value/307)-1,5,3,3,4,3]
  elif glycan_class == 'O':
    ranges = [math.floor(mz_value/160),math.floor(mz_value/203),math.floor(mz_value/146),
              math.floor(mz_value/291),math.floor(mz_value/307),4,3,3,7,7]
  else:
    print("Invalid glycan class; only N and O are allowed.")

  if human:
    pools = [list(range(k)) for k in ranges[:4]]
  else:
    pools = [list(range(k)) for k in ranges[:-2]]
  if ptm:
    ptm_pools = [list(range(k)) for k in ranges[8:]]
    pools = pools + ptm_pools
  #get combinations
  pools = list(product(*pools))

  compositions = []
  if mode == 'positive':
    adducts = ['H+', 'Na+', 'K+']
  elif mode == 'negative':
    adducts = ['H', 'Acetate', 'Trifluoroacetic acid']
  else:
    print("Only positive or negative mode allowed.")
  if not check_all_adducts:
    if check_specific_adduct is None:
      adducts = []
    else:
      adducts = [check_specific_adduct]
  mass_dict = dict(zip(mapping_file.composition, mapping_file[idx]))
  #for each combination calculate candidate precursor mass
  if human:
    precursors = [sum([mass_dict['Hex']*k[0], mass_dict['HexNAc']*k[1], mass_dict['dHex']*k[2],
                       mass_dict['Neu5Ac']*k[3], free_reducing_end]) for k in pools]
  else:
    precursors = [sum([mass_dict['Hex']*k[0], mass_dict['HexNAc']*k[1], mass_dict['dHex']*k[2],
                       mass_dict['Neu5Ac']*k[3], mass_dict['Neu5Gc']*k[4], mass_dict['Pen']*k[5],
                       mass_dict['Kdn']*k[6], mass_dict['HexA']*k[7], free_reducing_end]) for k in pools]
  if ptm:
    precursors = [precursors[k] + mass_dict['Sulphate']*pools[k][-2] + mass_dict['Phosphate']*pools[k][-1] for k in range(len(precursors))]

  #for each precursor candidate, check which lie within mass tolerance and format them
  for k in range(len(precursors)):
    precursor = precursors[k]
    pool = pools[k]
    if precursor < 1.2*mz_value:
      candidates = [precursor] + [precursor + mass_dict[j] for j in adducts]
      if any([abs(j - mz_value) < mass_tolerance for j in candidates]):
        if human:
          composition = {'Hex':pool[0], 'HexNAc':pool[1], 'dHex':pool[2],
                         'Neu5Ac':pool[3]}
        else:
          composition = {'Hex':pool[0], 'HexNAc':pool[1], 'dHex':pool[2],
                         'Neu5Ac':pool[3], 'Neu5Gc':pool[4],
                         'Pen':pool[5], 'Kdn':pool[6], 'HexA':pool[7]}
        if ptm:
          composition['S'] = pool[-2]
          composition['P'] = pool[-1]
        compositions.append(composition)

  #heuristics to rule out unphysiological glycan compositions        
  compositions = [dict(t) for t in {tuple(d.items()) for d in compositions}]
  compositions = [k for k in compositions if (k['Hex'] + k['HexNAc']) > 0]
  compositions = [k for k in compositions if (k['dHex'] + 1) <= (k['Hex'] + k['HexNAc'])]
  if glycan_class == 'N':
    compositions = [k for k in compositions if k['HexNAc'] >= 2]
    compositions = [k for k in compositions if any([(k['HexNAc'] > 2),
                                                    (k['HexNAc'] == 2 and k['Neu5Ac'] == 0)])]
  if glycan_class == 'O':
    compositions = [k for k in compositions if k['HexNAc'] >= 1]
  if ptm:
    compositions = [k for k in compositions if not all([(k['S'] > 0), (k['P'] > 0)])]
    compositions = [k for k in compositions if (k['S'] + k['P']) <= (k['Hex'] + k['HexNAc'])]
  return compositions

def match_composition_relaxed(composition, group, level, df = None,
                      libr = None, reducing_end = None):
    """Given a coarse-grained monosaccharide composition (Hex, HexNAc, etc.), it returns all corresponding glycans\n
    | Arguments:
    | :-
    | composition (dict): a dictionary indicating the composition to match (for example {"Fuc":1, "Gal":1, "GlcNAc":1})
    | group (string): name of the Species, Genus, Family, Order, Class, Phylum, Kingdom, or Domain used to filter
    | level (string): Species, Genus, Family, Order, Class, Phylum, Kingdom, or Domain
    | df (dataframe): glycan dataframe for searching glycan structures; default:df_species
    | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset; default:lib
    | reducing_end (string): filters possible glycans by reducing end monosaccharide; default:None\n
    | Returns:
    | :-
    | Returns list of glycans matching composition in IUPAC-condensed
    """
    if df is None:
      df = df_species
    if libr is None:
      libr = lib
    #subset for glycans with the right taxonomic group and reducing end 
    df = df[df[level] == group]
    if reducing_end is not None:
      df = df[df.target.str.endswith(reducing_end)].reset_index(drop = True)
    #subset for glycans with the right number of monosaccharides
    comp_count = sum(composition.values())
    len_distr = [len(k) - (len(k)-1)/2 for k in min_process_glycans(df.target.values.tolist())]
    idx = [k for k in range(len(df)) if len_distr[k] == comp_count]
    output_list = df.iloc[idx,:].target.values.tolist()
    output_compositions = [glycan_to_composition(k, libr = libr) for k in output_list]
    out = [output_list[k] for k in range(len(output_compositions)) if composition == output_compositions[k]]
    return out


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
  #define uncertainty wildcards
  wildcards = ['z1-z', 'z2-z', 'a2-z', 'a1-z', 'b1-z']
  #establish glycan equality given the wildcards
  match_matrix = [[compare_glycans(k, j, libr = libr, wildcards = True,
                                  wildcard_list = wildcards) for j in matched_composition] for k in matched_composition]
  match_matrix = pd.DataFrame(match_matrix)
  match_matrix.columns = matched_composition
  #cluster glycans by pairwise equality (given the wildcards)
  clustering = DBSCAN(eps = 1, min_samples = 1).fit(match_matrix)
  cluster_labels = clustering.labels_
  num_clusters = len(list(set(cluster_labels)))
  sum_glycans = []
  #for each cluster, get the most well-defined glycan and return it
  for k in range(num_clusters):
    cluster_glycans = [matched_composition[j] for j in range(len(cluster_labels)) if cluster_labels[j] == k]
    county = [sum([j.count(w) for w in wildcards]) for j in cluster_glycans]
    idx = np.where(county == np.array(county).min())[0]
    if len(idx) == 1:
      sum_glycans.append(cluster_glycans[idx[0]])
    else:
      for j in idx:
        sum_glycans.append(cluster_glycans[j])
  return sum_glycans

def compositions_to_structures(composition_list, group = 'Homo_sapiens', level = 'Species', abundances = None,
                               df = None, libr = None, reducing_end = None,
                               verbose = False):
  """wrapper function to map compositions to structures, condense them, and match them with relative intensities\n
  | Arguments:
  | :-
  | composition_list (list): list of composition dictionaries of the form {'Hex': 1, 'HexNAc': 1}
  | group (string): name of the Species, Genus, Family, Order, Class, Phylum, Kingdom, or Domain used to filter; default:Homo_sapiens
  | level (string): Species, Genus, Family, Order, Class, Phylum, Kingdom, or Domain; default:Species
  | abundances (dataframe): every row one composition (matching composition_list in order), every column one sample;default:pd.DataFrame([range(len(composition_list))]*2).T
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
  if abundances is None:
    abundances = pd.DataFrame([range(len(composition_list))]*2).T
  df_out = []
  not_matched = []
  for k in range(len(composition_list)):
    #for each composition, map it to potential structures
    matched = match_composition_relaxed(composition_list[k], group, level,
                                reducing_end = reducing_end, df = df, libr = libr)
    #if multiple structure matches, try to condense them by wildcard clustering
    if len(matched) > 0:
        condensed = condense_composition_matching(matched, libr = libr)
        matched_data = [abundances.iloc[k,1:].values.tolist()]*len(condensed)
        for ele in range(len(condensed)):
            df_out.append([condensed[ele]] + matched_data[ele])
    else:
        not_matched.append(composition_list[k])
  if len(df_out) > 0:
    df_out = pd.DataFrame(df_out)
    df_out.columns = ['glycan'] + ['abundance']*(abundances.shape[1]-1)
  print(str(len(not_matched)) + " compositions could not be matched. Run with verbose = True to see which compositions.")
  if verbose:
    print(not_matched)
  return df_out

def mz_to_structures(mz_list, reducing_end, group = 'Homo_sapiens', level = 'Species', abundances = None, mode = 'positive',
                     mass_value = 'monoisotopic', sample_prep = 'underivatized', mass_tolerance = 0.2,
                     check_all_adducts = False, check_specific_adduct = None,
                      ptm = False, df = None, libr = None, verbose = False):
  """wrapper function to map precursor masses to structures, condense them, and match them with relative intensities\n
  | Arguments:
  | :-
  | mz_list (list): list of precursor masses
  | reducing_end (string): filters possible glycans by reducing end monosaccharide
  | group (string): name of the Species, Genus, Family, Order, Class, Phylum, Kingdom, or Domain used to filter; default:Homo_sapiens
  | level (string): Species, Genus, Family, Order, Class, Phylum, Kingdom, or Domain; default:Species
  | abundances (dataframe): every row one composition (matching mz_list in order), every column one sample; default:pd.DataFrame([range(len(mz_list))]*2).T
  | mode (string): whether mz_value comes from MS in 'positive' or 'negative' mode; default:'positive'
  | mass_value (string): whether the expected mass is 'monoisotopic' or 'average'; default:'monoisotopic'
  | sample_prep (string): whether the glycans has been 'underivatized', 'permethylated', or 'peracetylated'; default:'underivatized'
  | mass_tolerance (float): how much deviation to tolerate for a match; default:0.2
  | check_all_adducts (bool): whether to also check for matches with ion adducts (depending on mode); default:False
  | check_specific_adduct (string): choose adduct from 'H+', 'Na+', 'K+', 'H', 'Acetate', 'Trifluoroacetic acid'; default:None
  | ptm (bool): whether to check for post-translational modification (sulfation, phosphorylation); default:False
  | df (dataframe): glycan dataframe for searching glycan structures; default:df_species
  | libr (list): sorted list of unique glycoletters observed in the glycans of our dataset; default:lib
  | verbose (bool): whether to print any non-matching compositions; default:False\n
  | Returns:
  | :-
  | Returns dataframe of (matched structures) x (relative intensities)
  """
  if libr is None:
    libr = lib
  if df is None:
    df = df_species
  if abundances is None:
    abundances = pd.DataFrame([range(len(mz_list))]*2).T
  if group == 'Homo_sapiens':
    human = True
  else:
    human = False
  #infer glycan class
  if 'GlcNAc' in reducing_end:
    glycan_class = 'N'
  elif 'GalNAc' in reducing_end:
    glycan_class = 'O'
  else:
    print("Not a valid class for mz_to_composition; currently N/O matching is supported. For everything else run composition_to_structures separately.")
  out_structures = []
  #map each m/z value to potential compositions
  compositions = [mz_to_composition(mz, mode = mode, mass_value = mass_value, sample_prep = sample_prep,
                                    mass_tolerance = mass_tolerance, human = human, glycan_class = glycan_class,
                                    check_all_adducts = check_all_adducts, check_specific_adduct = check_specific_adduct,
                                    ptm = ptm) for mz in mz_list]
  #map each of these potential compositions to potential structures
  for m in range(len(compositions)):
    structures = [compositions_to_structures([k], abundances = abundances.iloc[[m]], group = group, level = level, df = df,
                                             libr = libr, reducing_end = reducing_end, verbose = verbose) for k in compositions[m]]
    structures = [k for k in structures if not k.empty]
    #do not return matches if one m/z value matches multiple compositions that *each* match multiple structures, because of error propagation
    if len(structures) == 1:
      out_structures.append(structures[0])
    else:
      if verbose:
        print("m/z value " + str(mz_list[m]) + " with multiple matched compositions that each would have matching structures is filtered out.")
  out_structures = pd.concat(out_structures, axis = 0)
  return out_structures

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
  #find the motifs in the glycans
  annot = annotate_dataset(df.iloc[:,0].values.tolist(), libr = libr,
                           feature_set = feature_set, condense = True)
  annot2 = pd.concat([annot.reset_index(drop = True), df.iloc[:,1:]], axis = 1)
  out_tuples = []
  #reformat to record all present motifs in the provided structures
  for k in range(len(annot2)):
    for j in range(annot.shape[1]):
      if annot2.iloc[k,j]>0:
          out_tuples.append([annot2.columns.values.tolist()[j]] + df.iloc[k, 1:].values.tolist())
  #group abundances on a motif level
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

def mask_rare_glycoletters(glycans, thresh_monosaccharides = None, thresh_linkages = None):
  """masks rare monosaccharides and linkages in a list of glycans\n 
  | Arguments:
  | :-
  | glycans (list): list of glycans in IUPAC-condensed form
  | thresh_monosaccharides (int): threshold-value for monosaccharides seen as "rare"; default:(0.001*len(glycans))
  | thresh_linkages (int): threshold-value for linkages seen as "rare"; default:(0.03*len(glycans))\n
  | Returns:
  | :-
  | Returns list of glycans in IUPAC-condensed with masked rare monosaccharides and linkages
  """
  #get rarity thresholds
  if thresh_monosaccharides is None:
    thresh_monosaccharides = int(np.ceil(0.001*len(glycans)))
  if thresh_linkages is None:
    thresh_linkages = int(np.ceil(0.03*len(glycans)))
  rares = unwrap(min_process_glycans(glycans))
  rare_linkages, rare_monosaccharides = [], []
  #sort monosaccharides and linkages into different bins
  for x in rares:
    (rare_monosaccharides, rare_linkages)[x in linkages].append(x)
  rares = [rare_monosaccharides, rare_linkages]
  thresh = [thresh_monosaccharides, thresh_linkages]
  #establish which ones are considered to be rare
  rares = [list({x: count for x, count in Counter(rares[k]).items() if count <= thresh[k]}.keys()) for k in range(len(rares))]
  try:
    rares[0].remove('')
  except:
    pass
  out = []
  #for each glycan, check whether they have rare monosaccharides/linkages and mask them
  for k in glycans:
    for j in rares[0]:
      if (j in k) and ('-'+j not in k):
        k = k.replace(j+'(', 'Monosaccharide(')
        if k.endswith(j):
          k = re.sub(j+'$', 'Monosaccharide', k)
    for m in rares[1]:
      if m in k:
        if m[1] == '1':
          k = k.replace(m, 'z1-z')
        else:
          k = k.replace(m, 'z2-z')
    out.append(k)
  return out

def canonicalize_iupac(glycan):
  """converts a glycan from any IUPAC flavor into the exact IUPAC-condensed version that is optimized for glycowork\n
  | Arguments:
  | :-
  | glycan (string): glycan sequence in IUPAC; post-biosynthetic modifications can still be an issue\n
  | Returns:
  | :-
  | Returns glycan as a string in canonicalized IUPAC-condensed
  """
  #canonicalize usage of monosaccharides and linkages
  if 'NeuAc' in glycan:
    glycan = glycan.replace('NeuAc', 'Neu5Ac')
  if 'NeuGc' in glycan:
    glycan = glycan.replace('NeuGc', 'Neu5Gc')
  if "\u03B1" in glycan:
    glycan = glycan.replace("\u03B1", 'a')
  if "\u03B2" in glycan:
    glycan = glycan.replace("\u03B2", 'b')
  #canonicalize linkage uncertainty
  if '?' in glycan:
    glycan = glycan.replace('?', 'z')
  if bool(re.search(r'[a-z]\-[A-Z]', glycan)):
    glycan = re.sub(r'([a-z])\-([A-Z])', r'\1z1-z\2', glycan)
  if bool(re.search(r'[a-z][\(\)]', glycan)):
    glycan = re.sub(r'([a-z])([\(\)])', r'\1z1-z\2', glycan)
  if bool(re.search(r'[0-9]\-[\(\)]', glycan)):
    glycan = re.sub(r'([0-9])\-([\(\)])', r'\1-z\2', glycan)
  while '/' in glycan:
    glycan = glycan[:glycan.index('/')-1] + 'z' + glycan[glycan.index('/')+1:]
  #canonicalize usage of brackets and parentheses
  if bool(re.search(r'\([A-Z0-9]', glycan)):
    glycan = glycan.replace('(', '[')
    glycan = glycan.replace(')', ']')
  if '(' not in glycan and len(glycan) > 6:
    for k in range(1,glycan.count('-')+1):
      idx = find_nth(glycan, '-', k)
      if (glycan[idx-1].isnumeric()) and (glycan[idx+1].isnumeric() or glycan[idx+1]=='z'):
        glycan = glycan[:idx-2] + '(' + glycan[idx-2:idx+2] + ')' + glycan[idx+2:]
      elif (glycan[idx-1].isnumeric()) and bool(re.search(r'[A-Z]', glycan[idx+1])):
        glycan = glycan[:idx-2] + '(' + glycan[idx-2:idx+1] + 'z)' + glycan[idx+1:]
  #canonicalize reducing end
  if bool(re.search(r'[a-z]ol', glycan)):
    if 'Glcol' not in glycan:
      glycan = glycan[:-2]
    else:
      glycan = glycan[:-2] + '-ol'
  #handle modifications
  if bool(re.search(r'\[[0-9]?[SP]\][^\(]+', glycan)):
    glycan = re.sub(r'\[([0-9]?[SP])\]([^\(]+)', r'\2\1', glycan)
  if bool(re.search(r'\-ol[0-9]?[SP]', glycan)):
    glycan = re.sub(r'(\-ol)([0-9]?[SP])', r'\2\1', glycan)
  #canonicalize branch ordering
  if '[' in glycan:
    isos = find_isomorphs(glycan)
    glycan = choose_correct_isoform(isos)
  if '+' in glycan:
    print("Warning: There seems to be a floating substituent in the IUPAC string (curly brackets); we can't handle that in downstream functions.")
  return glycan

def check_nomenclature(glycan):
  """checks whether the proposed glycan has the correct nomenclature for glycowork\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | Returns:
  | :-
  | If salvageable, returns the re-formatted glycan; if not, prints the reason why it's not convertable
  """
  if not isinstance(glycan, str):
    print("You need to format your glycan sequences as strings.")
    return
  if '?' in glycan:
    print("You're likely using ? somewhere, to indicate linkage uncertainty. Glycowork uses 'z' to indicate linkage uncertainty")
    return canonicalize_iupac(glycan)
  if '=' in glycan:
    print("Could it be that you're using WURCS? Please convert to IUPACcondensed for using glycowork.")
  if 'RES' in glycan:
    print("Could it be that you're using GlycoCT? Please convert to IUPACcondensed for using glycowork.")
  print("Didn't spot an obvious error but this is not a guarantee that it will work.")
  return canonicalize_iupac(glycan)

def map_to_basic(glycoletter):
  """given a monosaccharide/linkage, try to map it to the corresponding base monosaccharide/linkage\n
  | Arguments:
  | :-
  | glycoletter (string): monosaccharide or linkage\n
  | Returns:
  | :-
  | Returns the base monosaccharide/linkage or the original glycoletter, if it cannot be mapped
  """
  if glycoletter in Hex:
    return 'Hex'
  elif glycoletter in dHex:
    return 'dHex'
  elif glycoletter in HexA:
    return 'HexA'
  elif glycoletter in HexN:
    return 'HexN'
  elif glycoletter in HexNAc:
    return 'HexNAc'
  elif glycoletter in Pen:
    return 'Pen'
  elif glycoletter in linkages:
    return 'z1-z'
  else:
    return glycoletter

def structure_to_basic(glycan, libr = None):
  """converts a monosaccharide- and linkage-defined glycan structure to the base topology\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed nomenclature
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns the glycan topology as a string
  """
  if libr is None:
    libr = lib
  if glycan[-3:] == '-ol':
    glycan = glycan[:-3]
  if '(' not in glycan:
    return map_to_basic(glycan)
  ggraph = glycan_to_nxGraph(glycan, libr = libr)
  nodeDict = dict(ggraph.nodes(data = True))
  temp = {k:map_to_basic(nodeDict[k]['string_labels']) for k in ggraph.nodes}
  nx.set_node_attributes(ggraph, temp, 'string_labels')
  return graph_to_string(ggraph)

def glycan_to_composition(glycan, libr = None):
  """maps glycan to its composition\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns a dictionary of form "monosaccharide" : count
  """
  if libr is None:
    libr = lib
  glycan2 = stemify_glycan(glycan, libr = libr)
  glycan2 = structure_to_basic(glycan2, libr = libr)
  composition = Counter(min_process_glycans([glycan2])[0])
  if 'S' in glycan:
    composition['S'] = glycan.count('S')
  if 'P' in glycan:
    composition['P'] = glycan.count('P')
  del composition['z1-z']
  return dict(composition)

def calculate_theoretical_mass(glycan, mass_value = 'monoisotopic', sample_prep = 'underivatized',
                               libr = None):
  """given a glycan, calculates it's theoretical mass; only allowed extra-modifications are sulfation, phosphorylation, and PCho\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed format
  | mass_value (string): whether the expected mass is 'monoisotopic' or 'average'; default:'monoisotopic'
  | sample_prep (string): whether the glycans has been 'underivatized', 'permethylated', or 'peracetylated'; default:'underivatized'
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns the theoretical mass of input glycan
  """
  if libr is None:
    libr = lib
  theoretical_mass = 0
  idx = sample_prep + '_' + mass_value
  mass_dict = dict(zip(mapping_file.composition, mapping_file[idx]))
  sulfate_count = glycan.count('S')
  phosphate_count = glycan.count('P')
  if 'PCho' in glycan:
    theoretical_mass += mass_dict['PCho'] * glycan.count('PCho') - mass_dict['Phosphate'] * glycan.count('PCho')
  glycan = stemify_glycan(glycan, libr = libr)
  glycan = structure_to_basic(glycan, libr = libr)
  theoretical_mass += sum([mass_dict[k] for k in min_process_glycans([glycan])[0] if k != 'z1-z'])+18.0105546
  if sulfate_count > 0:
    theoretical_mass += mass_dict['Sulphate'] * sulfate_count
  if phosphate_count > 0:
    theoretical_mass += mass_dict['Phosphate'] * phosphate_count
  return theoretical_mass
