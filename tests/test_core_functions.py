import pytest
import networkx as nx
import networkx.algorithms.isomorphism as iso
import pandas as pd
import numpy as np
from glycowork.motif.tokenization import (
    constrain_prot, prot_to_coded, string_to_labels, pad_sequence, mz_to_composition,
    get_core, get_modification, get_stem_lib, stemify_glycan, mask_rare_glycoletters,
    glycan_to_composition, calculate_adduct_mass, structure_to_basic, map_to_basic,
    compositions_to_structures, match_composition_relaxed, stemify_dataset, composition_to_mass,
    condense_composition_matching, get_unique_topologies, mz_to_structures, glycan_to_mass
)
from glycowork.motif.processing import (
    min_process_glycans, get_lib, expand_lib, get_possible_linkages,
    get_possible_monosaccharides, de_wildcard_glycoletter, canonicalize_iupac,
    glycoct_to_iupac, wurcs_to_iupac, oxford_to_iupac,
    canonicalize_composition, parse_glycoform, find_isomorphs,
    presence_to_matrix, process_for_glycoshift, linearcode_to_iupac, iupac_extended_to_condensed,
    in_lib, get_class, enforce_class, equal_repeats, find_matching_brackets_indices,
    choose_correct_isoform, bracket_removal, check_nomenclature, IUPAC_to_SMILES
)
from glycowork.glycan_data.loader import (
    unwrap, find_nth, find_nth_reverse, remove_unmatched_brackets,
    reindex, stringify_dict, replace_every_second, multireplace,
    strip_suffixes, build_custom_df, DataFrameSerializer, Hex, linkages
)
from glycowork.glycan_data.stats import (
    fast_two_sum, two_sum, expansion_sum, hlm, cohen_d,
    mahalanobis_distance, variance_stabilization, shannon_diversity_index,
    simpson_diversity_index, get_equivalence_test, sequence_richness,
    hotellings_t2, calculate_permanova_stat, omega_squared, anosim, alpha_biodiversity_stats, permanova_with_permutation,
    clr_transformation, alr_transformation, get_procrustes_scores,
    get_additive_logratio_transformation, get_BF, get_alphaN,
    pi0_tst, TST_grouped_benjamini_hochberg, compare_inter_vs_intra_group,
    correct_multiple_testing, partial_corr, estimate_technical_variance, jtkdist, jtkinit, jtkstat, jtkx, MissForest, impute_and_normalize,
    variance_based_filtering, get_glycoform_diff, get_glm, process_glm_results, replace_outliers_with_IQR_bounds,
    replace_outliers_winsorization, perform_tests_monte_carlo
)
from glycowork.motif.graph import (
    glycan_to_graph, glycan_to_nxGraph, evaluate_adjacency,
    compare_glycans, subgraph_isomorphism, generate_graph_features,
    graph_to_string, largest_subgraph, get_possible_topologies,
    deduplicate_glycans, neighbor_is_branchpoint, try_string_conversion,
    subgraph_isomorphism_with_negation, categorical_node_match_wildcard,
    expand_termini_list, ensure_graph, possible_topology_check
)
from glycowork.motif.annotate import (
    link_find, annotate_glycan, annotate_dataset, get_molecular_properties,
    get_k_saccharides, get_terminal_structures, create_correlation_network,
    group_glycans_core, group_glycans_sia_fuc, group_glycans_N_glycan_type,
    Lectin, load_lectin_lib, create_lectin_and_motif_mappings,
    lectin_motif_scoring, clean_up_heatmap, quantify_motifs,
    count_unique_subgraphs_of_size_k, annotate_glycan_topology_uncertainty
)
from glycowork.motif.regex import (preprocess_pattern, specify_linkages, process_occurrence,
                  expand_pattern, convert_pattern_component, reformat_glycan_string,
                  motif_to_regex, get_match, process_question_mark, calculate_len_matches_comb,
                  process_main_branch, check_negative_look,
                  filter_matches_by_location, parse_pattern)
from glycowork.motif.draw import (matches, process_bonds, split_monosaccharide_linkage,
                 scale_in_range, glycan_to_skeleton, process_per_residue,
                 get_hit_atoms_and_bonds, add_colours_to_map, unique)
from glycowork.motif.analysis import (preprocess_data, get_heatmap,
                     get_pvals_motifs, select_grouping, get_glycanova, get_differential_expression,
                     get_biodiversity, get_time_series, get_SparCC, get_roc,
                     get_representative_substructures, get_lectin_array)
from glycowork.network.biosynthesis import (safe_compare, safe_index, get_neighbors, create_neighbors,
                         find_diff, find_path, construct_network, prune_network, estimate_weights,
                         extend_glycans, highlight_network, infer_roots, deorphanize_nodes, get_edge_weight_by_abundance,
                         find_diamonds, trace_diamonds, get_maximum_flow, get_reaction_flow, process_ptm, get_differential_biosynthesis,
                         deorphanize_edge_labels)


@pytest.mark.parametrize("glycan", [
    # linear glycans
    "Rha(a1-2)Glc(b1-2)GlcAOBut",
    "ManNAc1PP(a1-5)Ribf1N",
    "Glc3Me(b1-3)Xyl(b1-4)Qui",
    "D-Fuc2Me4NAc(a1-4)GlcA",
    "Man(b1-3)Man(b1-3)Ery-onic",
    "Man(b1-3)Glc(a1-4)Rha(a1-4)Man(b1-3)Glc(a1-4)D-Rha",
    "Rha(?1-?)Glc(?1-?)Glc(?1-?)Gal",
    "GlcA(b1-3)Gal(b1-3)GalNAc(b1-4)GlcNAc6P(b1-3)Man(b1-4)Glc",
    "Neu5Gc(a2-6)Glc(?1-?)Neu5Gc(a2-?)Glc",
    "Neu5Ac(b2-6)GalNAc",

    # single branching
    "Gal(a1-3)Gal(b1-4)GlcNAc(b1-3)[Gal(a1-3)Gal(b1-4)GlcNAc(b1-6)]Gal(b1-?)GlcNAc(b1-3)Gal(b1-4)Glc",
    "Glc(b1-2)[Glc(b1-3)]Gal",
    "Fuc(a1-3)[Fuc(b1-6)]GlcNAc1N",
    "Neu5Ac(a2-?)GalNAc(b1-4)GlcNAc(b1-2)Man(a1-?)[Man(a1-?)Man(a1-?)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "GalNAc(a1-3)[Fuc(a1-2)]Gal(b1-4)Glc-ol",
    "QuiNAcOPGro(b1-3)[GlcNAcOAc(b1-2)]Gal(b1-3)GlcNAcOAc(b1-3)QuiNAcOPGro",
    "Neu5Ac(a2-?)GalNAc(b1-4)GlcNAc(b1-2)Man(a1-?)[Neu5Gc(a2-?)Gal(b1-4)GlcNAc(b1-2)Man(a1-?)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "Glc(a1-2)[Glc(b1-6)]Glc(a1-3)Glc(b1-6)Glc",
    "Glc(a1-6)Glc(a1-6)[Glc(a1-2)]Glc",
    "Neu5Ac(a2-6)Gal(b1-4)GlcNAc(b1-2)Man(a1-6)[Man(a1-3)]Man(b1-4)GlcNAc(b1-4)GlcNAc",

    # multiple branching
    "GalNAc(?1-?)[Neu5Ac(a2-?)]Gal(b1-?)GlcNAc(b1-?)Man(a1-?)[Neu5Ac(a2-?)Gal(b1-?)GlcNAc(b1-?)Man(a1-?)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-?)]GlcNAc",
    "Gal(?1-?)[Fuc(?1-?)]Gal(b1-?)GlcNAc(b1-?)Man(a1-?)[GlcNAc(b1-?)Man(a1-?)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-?)]GlcNAc",
    "Glc(b1-3)[Glc(b1-3)[Glc(b1-6)Glc(b1-6)]Glc(b1-6)]Glc(b1-6)Glc",
    "Neu5Gc(a2-?)Gal(b1-4)GlcNAc(b1-2)[Neu5Gc(a2-?)Gal(b1-4)GlcNAc(b1-4)]Man(a1-3)[Neu5Ac(a2-?)Gal(b1-4)GlcNAc(b1-2)[Neu5Ac(a2-?)Gal(b1-4)GlcNAc(b1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "Man(a1-3)[Gal3Me(b1-6)Gal3Me(b1-3)[Gal3Me(b1-6)]GlcNAc(b1-4)GlcNAc(b1-2)Man(a1-6)][Xyl(b1-2)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "Neu5Ac(a2-?)Hex(?1-?)HexNAc(?1-?)Hex(?1-?)[Neu5Ac(a2-?)Hex(?1-?)HexNAc(?1-?)Hex(?1-?)]Hex(?1-?)HexNAc(?1-?)[Fuc(a1-?)]HexNAc",
    "Neu5Ac(a2-?)Gal(b1-4)[Fuc(a1-3)]GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-2)[Neu5Ac(a2-?)Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-4)]Man(a1-3)[Man(a1-3)[Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "Fuc(a1-2)[GalNAc(a1-3)]Gal(b1-4)[Fuc(a1-3)]GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)Glc",
    "Gal(b1-4)GlcNAc(b1-2)[Gal(b1-4)GlcNAc(b1-6)]Man(a1-6)[Gal(b1-4)GlcNAc(b1-2)Man(a1-3)]Man",
    "GlcA(b1-2)Man(a1-4)[Araf(a1-3)]GlcA(b1-2)[Araf(a1-3)]Man(a1-4)GlcA",

    # double branching
    "Man(a1-3)[Man(a1-6)][Xylf(?1-?)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "GlcNAc(b1-2)Man(a1-3)[GlcNAc(b1-4)][GlcNAc(b1-2)Man(a1-6)]Man",
    "HexOMe(?1-?)[HexOMe(?1-?)]HexOMe(?1-?)[HexOMe(?1-?)]GalNAc(b1-4)GlcNAc(b1-2)Man(a1-3)[Xyl(b1-2)][Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "Neu5Ac(a2-8)Neu5Ac(a2-?)Gal(b1-4)[Fuc(a1-3)]GlcNAc(b1-2)Man(a1-3)[GlcNAc(b1-4)][GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "Neu5Ac(a2-?)Gal(b1-?)GlcNAc(b1-2)[Neu5Ac(a2-?)Gal(b1-?)GlcNAc(b1-4)]Man(a1-3)[Neu5Ac(a2-?)Gal(b1-?)GlcNAc(b1-2)Man(a1-6)][GlcNAc(b1-4)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "Gal3Me(b1-6)Gal3Me(b1-3)[Gal3Me(b1-6)]GalNAc(b1-4)GlcNAc(b1-2)Man(a1-6)[Xyl(b1-2)][Man(a1-3)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "Gal(b1-4)[Fuc(a1-3)]GlcNAc(b1-2)[GlcNAc(b1-4)]Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)[Fuc(a1-3)[Gal(b1-4)]GlcNAc(b1-6)]Man(a1-6)][GlcNAc(b1-4)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "Neu5Gc(a2-?)Gal(b1-4)GlcNAc(b1-2)[Gal(b1-4)GlcNAc(b1-4)]Man(a1-3)[GlcNAc(b1-4)][Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-2)[Gal(b1-4)GlcNAc(b1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "Neu5Ac(a2-?)Neu5Ac(a2-?)Gal(b1-?)GlcNAc(b1-?)Gal(b1-?)GlcNAc(b1-?)[Neu5Gc(a2-?)Gal(b1-?)[Fuc(a1-?)]GlcNAc(b1-?)Gal(b1-?)GlcNAc(b1-?)]Man(a1-?)[Man(a1-?)[Man(a1-?)]Man(a1-?)][GlcNAc(b1-4)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-2)[Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-4)]Man(a1-3)[GlcNAc(b1-4)][Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-2)[Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",

    # nested branching
    "Glc(a1-3)Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc-ol",
    "Man(a1-?)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc-ol",
    "Gal(?1-?)Gal(b1-?)GlcNAc(?1-?)[Fuc(a1-?)[Gal(b1-?)]GlcNAc(b1-?)]Man(a1-?)[Fuc(a1-?)[Gal(b1-?)]GlcNAc(b1-?)[Fuc(a1-?)[Gal(b1-?)]GlcNAc(b1-?)]Man(a1-?)][GlcNAc(b1-4)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-?)]GlcNAc",
    "Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-6)[Gal(b1-4)GlcNAc(b1-2)[Gal(b1-4)GlcNAc(b1-4)]Man(a1-3)]Man(b1-4)GlcNAc",
    "Man(a1-2)Man6P(a1-2)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc",
    "Neu5Ac(a2-6)Gal(b1-4)GlcNAc(b1-2)[Gal(b1-3)[Neu5Ac(a2-6)]GlcNAc(b1-4)]Man(a1-3)[Neu5Ac(a2-6)Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man",
    "Gal(b1-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-3)Gal(b1-4)GlcNAc(b1-2)[Gal(b1-3)Gal(b1-4)GlcNAc(b1-4)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "Neu5Ac(a2-?)Gal(b1-3)[Neu5Ac(a2-?)Gal(b1-?)[Fuc(a1-?)]GlcNAc(b1-6)]GalNAc",
    "Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)Gal(b1-4)[Fuc(a1-3)]GlcNAc(b1-2)Man(a1-6)][GlcNAc(b1-4)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)GlcNAc(b1-3)[Gal(b1-4)GlcNAc(b1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",

    # floating elements
    "{Fuc(a1-?)}Neu5Ac(a2-?)Gal(b1-4)GlcNAc(b1-2)Man(a1-?)[Gal(b1-4)GlcNAc(b1-2)Man(a1-?)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "{Fuc(a1-?)}Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)Man(a1-6)][GlcNAc(b1-4)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "{Gal(b1-4)GlcNAc(b1-?)}{Neu5Ac(a2-?)}{Neu5Ac(a2-?)}Gal(b1-4)GlcNAc(b1-2)[Gal(b1-4)GlcNAc(b1-?)]Man(a1-?)[Gal(b1-4)GlcNAc(b1-2)Man(a1-?)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "{Neu5Ac(a2-?)}Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Fuc(a1-3)[Gal(b1-4)]GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "{Fuc(a1-2)}Gal(b1-4)[Neu5Ac(a2-6)]GlcNAc(b1-?)[Gal(b1-4)GlcNAc(b1-?)]Gal(b1-4)Glc-ol",
    "{Neu5Ac(a2-?)}{Neu5Ac(a2-?)}Gal(b1-?)GlcNAc(b1-2)[Gal(b1-?)GlcNAc(b1-4)]Man(a1-3)[Gal(b1-?)GlcNAc(b1-2)[Gal(b1-?)GlcNAc(b1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "{Fuc(a1-?)}Neu5Ac(a2-?)Gal(b1-?)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-?)[Neu5Ac(a2-?)Gal(b1-4)GlcNAc(b1-?)]Man(a1-?)[Neu5Ac(a2-?)Gal(b1-4)GlcNAc(b1-2)[Neu5Ac(a2-?)Gal(b1-4)GlcNAc(b1-?)]Man(a1-?)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "{Fuc(a1-?)}{GlcNAc(b1-?)}{Hex(?1-?)}{Neu5Gc(a2-?)}GlcNAc(b1-2)Man(a1-3)[Man(a1-3)[Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "{Neu5Ac(a2-?)}Gal(b1-?)GlcNAc(b1-2)[Gal(b1-?)GlcNAc(b1-4)]Man(a1-3)[Gal(b1-?)GlcNAc(b1-2)Man(a1-6)][GlcNAc(b1-4)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "{Fuc(a1-?)}{Neu5Ac(a2-?)}Gal(b1-4)GlcNAc(b1-2)[Gal(b1-4)GlcNAc(b1-4)]Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)[Gal(b1-4)GlcNAc(b1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
])
def test_graph_to_string_int(glycan: str):
    """This test assumes that the function gylcan_to_graph_int is correct."""
    label = glycan_to_nxGraph(glycan)
    target = glycan_to_nxGraph(graph_to_string(label))

    assert nx.is_isomorphic(label, target, node_match=iso.categorical_node_match("string_labels", ""))


def test_constrain_prot():
    # Test replacing non-library characters with 'z'
    proteins = ['ABC', 'XYZ', 'A1C']
    chars = {'A': 1, 'B': 2, 'C': 3}
    result = constrain_prot(proteins, chars)
    assert result == ['ABC', 'zzz', 'AzC']


def test_prot_to_coded():
    proteins = ['ABC', 'DE']
    chars = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'z': 5}
    pad_len = 5
    result = prot_to_coded(proteins, chars, pad_len)
    # ABC should be [0,1,2,5,5] where len(chars)-1 is padding
    # DE should be [3,4,5,5,5]
    assert result[0] == [0, 1, 2, 5, 5]
    assert result[1] == [3, 4, 5, 5, 5]


def test_string_to_labels():
    chars = {'A': 1, 'B': 2, 'C': 3}
    assert string_to_labels('ABC', chars) == [1, 2, 3]
    assert string_to_labels('BAC', chars) == [2, 1, 3]
    assert string_to_labels('', chars) == []


def test_pad_sequence():
    seq = [1, 2, 3]
    max_length = 5
    pad_label = 0
    result = pad_sequence(seq, max_length, pad_label)
    assert result == [1, 2, 3, 0, 0]
    # Test when sequence is longer than max_length
    seq = [1, 2, 3, 4, 5, 6]
    max_length = 4
    result = pad_sequence(seq, max_length, pad_label)
    assert result == [1, 2, 3, 4, 5, 6]  # Should not truncate


def test_get_core():
    assert get_core('Neu5Ac') == 'Neu5Ac'
    assert get_core('GlcNAc') == 'GlcNAc'
    assert get_core('Gal6S') == 'Gal'
    assert get_core('Man3S') == 'Man'
    assert get_core('6dAlt') == '6dAlt'


def test_get_modification():
    assert get_modification('Neu5Ac9Ac') == '9Ac'
    assert get_modification('GlcNAc') == ''
    assert get_modification('Gal6S') == '6S'
    assert get_modification('Man3S') == '3S'
    assert get_modification('Gal3S6S') == '3S6S'


def test_get_stem_lib():
    lib = {'Neu5Ac': 1, 'Gal6S': 2, 'GlcNAc': 3, 'Neu5Gc8S': 4}
    result = get_stem_lib(lib)
    assert result == {'Neu5Ac': 'Neu5Ac', 'Gal6S': 'Gal', 'GlcNAc': 'GlcNAc', 'Neu5Gc8S': 'Neu5Gc'}


def test_stemify_glycan():
    # Test basic glycan
    glycan = "Neu5Ac9Ac(a2-3)Gal(b1-4)GlcNAc"
    result = stemify_glycan(glycan)
    assert result == "Neu5Ac(a2-3)Gal(b1-4)GlcNAc"
    # Test glycan with multiple modifications
    glycan = "Neu5Ac9Ac(a2-3)Gal6S(b1-4)GlcNAc"
    result = stemify_glycan(glycan)
    assert result == "Neu5Ac(a2-3)Gal(b1-4)GlcNAc"


def test_glycan_to_composition():
    # Test basic glycan
    glycan = "Neu5Ac(a2-3)Gal(b1-4)GlcNAc"
    result = glycan_to_composition(glycan)
    assert result == {'Neu5Ac': 1, 'Hex': 1, 'HexNAc': 1}
    # Test glycan with sulfation
    glycan = "Neu5Ac(a2-3)Gal6S(b1-4)GlcNAc"
    result = glycan_to_composition(glycan)
    assert result == {'Neu5Ac': 1, 'Hex': 1, 'HexNAc': 1, 'S': 1}


def test_calculate_adduct_mass():
    # Test simple molecules
    assert abs(calculate_adduct_mass('H2O') - 18.0106) < 0.001
    assert abs(calculate_adduct_mass('CH3COOH') - 60.0211) < 0.001
    # Test with different mass types
    h2o_mono = calculate_adduct_mass('H2O', mass_value='monoisotopic')
    h2o_avg = calculate_adduct_mass('H2O', mass_value='average')
    assert h2o_mono != h2o_avg


def test_structure_to_basic():
    # Basic glycan
    glycan = "Neu5Ac(a2-3)Gal(b1-4)GlcNAc"
    result = structure_to_basic(glycan)
    assert result == "Neu5Ac(?1-?)Hex(?1-?)HexNAc"
    # Glycan with sulfation
    glycan = "Gal6S(b1-4)GlcNAc"
    result = structure_to_basic(glycan)
    assert result == "HexOS(?1-?)HexNAc"
    # Glycan with branch
    glycan = "Neu5Ac(a2-3)Gal(b1-4)[Fuc(a1-3)]GlcNAc"
    result = structure_to_basic(glycan)
    assert result == "Neu5Ac(?1-?)Hex(?1-?)[dHex(?1-?)]HexNAc"


def test_map_to_basic():
    assert map_to_basic("Gal") == "Hex"
    assert map_to_basic("Man") == "Hex"
    assert map_to_basic("Glc") == "Hex"
    assert map_to_basic("GlcNAc") == "HexNAc"
    assert map_to_basic("Fuc") == "dHex"
    assert map_to_basic("GlcA") == "HexA"
    assert map_to_basic("Gal6S") == "HexOS"
    assert map_to_basic("GlcNAc6S") == "HexNAcOS"


def test_mask_rare_glycoletters():
    glycans = [
        "Neu5Ac(a2-3)Gal(b1-4)GlcNAc",
        "Neu5Ac(a2-3)Gal(b1-4)GlcNAc",
        "Neu5Ac(a2-6)Gal(b1-4)GlcNAc",
        "Gal(b1-4)GlcNAc",
        "Man6P(a1-2)Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
        "RareSugar(a1-2)GlcNAc"
    ]

    # Test with default thresholds
    result = mask_rare_glycoletters(glycans)
    assert "RareSugar" not in result[5]
    assert "Monosaccharide" in result[5]
    # Test with custom thresholds
    result = mask_rare_glycoletters(glycans, thresh_monosaccharides=2)
    assert "Neu5Ac" in result[0]  # Occurs twice, should remain
    assert "Man6P" not in result[4]  # Occurs once, should be masked


def test_mz_to_composition():
    # Test O-glycan mass
    result = mz_to_composition(
        675,
        mode='negative',
        mass_value='monoisotopic',
        glycan_class='O',
        mass_tolerance=0.5,
        reduced=True,
        filter_out = {'Kdn'}
    )
    expected = [{'Neu5Ac': 1, 'Hex': 1, 'HexNAc': 1}]
    assert result == expected


def test_compositions_to_structures():
    composition = {'Hex': 3, 'HexNAc': 2}
    result = compositions_to_structures(
        [composition],
        glycan_class='N',
        kingdom='Animalia'
    )
    # Should return DataFrame with N-glycan structures matching composition
    assert isinstance(result, pd.DataFrame)
    assert 'glycan' in result.columns
    # Verify returned glycans have correct composition
    for glycan in result['glycan']:
        comp = glycan_to_composition(glycan)
        assert comp['Hex'] == 3
        assert comp['HexNAc'] == 2


def test_match_composition_relaxed():
    composition = {'Hex': 3, 'HexNAc': 2}
    result = match_composition_relaxed(
        composition,
        glycan_class='N',
        kingdom='Animalia'
    )
    # Should return list of glycan strings
    assert isinstance(result, list)
    # Each glycan should match the composition
    for glycan in result:
        comp = glycan_to_composition(glycan)
        assert comp['Hex'] == 3
        assert comp['HexNAc'] == 2


def test_stemify_dataset():
    df = pd.DataFrame({
        'glycan': [
            "Neu5Ac(a2-3)Gal(b1-4)GlcNAc",
            "Neu5Ac9Ac(a2-3)Gal6S(b1-4)GlcNAc",
            "Man(a1-3)[Man(a1-6)]Man"
        ]
    })

    result = stemify_dataset(df, rarity_filter=1)

    # Check modifications are removed
    assert "Neu5Ac(a2-3)Gal(b1-4)GlcNAc" in result['glycan'].values
    assert "6S" not in result['glycan'].str.cat()
    assert "5Ac9Ac" not in result['glycan'].str.cat()


def test_composition_to_mass():
    # Test basic composition
    comp = {'Hex': 1, 'HexNAc': 1}
    mass = composition_to_mass(comp)
    expected_mass = 383.1322  # Theoretical mass for Hex-HexNAc
    assert abs(mass - expected_mass) < 0.1
    # Test with modifications
    comp = {'Hex': 1, 'HexNAc': 1, 'S': 1}
    mass = composition_to_mass(comp)
    assert mass > composition_to_mass({'Hex': 1, 'HexNAc': 1})


def test_condense_composition_matching():
    glycans = [
        "Gal(b1-4)GlcNAc",
        "Gal(b1-4)GlcNAc",  # Duplicate
        "Gal(?1-4)GlcNAc",  # Similar but with wildcard
        "Man(a1-3)GlcNAc"   # Different structure
    ]
    result = condense_composition_matching(glycans)
    # Should remove duplicates and similar structures with wildcards
    assert len(result) < len(glycans)
    assert "Gal(b1-4)GlcNAc" in result
    assert "Man(a1-3)GlcNAc" in result


def test_get_unique_topologies():
    composition = {'Hex': 3, 'HexNAc': 2}
    result = get_unique_topologies(
        composition,
        glycan_type='N',
        taxonomy_value='Animalia'
    )
    # Should return list of unique topology strings
    assert isinstance(result, list)
    # Each topology should match the original composition
    for topology in result:
        comp = glycan_to_composition(topology)
        assert comp['Hex'] == 3
        assert comp['HexNAc'] == 2
    # with universal replacer
    composition = {'Hex': 3, 'HexNAc': 2, 'dHex': 1}
    result = get_unique_topologies(
        composition,
        glycan_type='N',
        taxonomy_value='Animalia',
        universal_replacers={'dHex': 'Fuc'}
    )
    assert isinstance(result, list)
    for topology in result:
        comp = glycan_to_composition(topology)
        assert comp['Hex'] == 3
        assert comp['HexNAc'] == 2
        assert 'Fuc' in topology


def test_mz_to_structures():
    # Test single mass
    mz_values = [530]
    result = mz_to_structures(
        mz_values,
        glycan_class='O',
        mode='negative',
        mass_value='monoisotopic',
        mass_tolerance=0.5,
        reduced=True
    )
    assert isinstance(result, pd.DataFrame)
    # Verify returned structures match the mass
    for glycan in result['glycan']:
        mass = glycan_to_mass(glycan) + 1.0078
        assert abs(mass - mz_values[0]) < 0.5
    # Test multiple masses
    mz_values = [530, 675]
    abundances = pd.DataFrame({
        'mz': mz_values,
        'intensity': [100, 50]
    })
    result = mz_to_structures(
        mz_values,
        glycan_class='O',
        mode='negative',
        reduced=True,
        filter_out={'Kdn'},
        abundances=abundances
    )
    assert len(result) > 0
    assert 'abundance' in result.columns
    # Test filtering
    result = mz_to_structures(
        mz_values,
        glycan_class='O',
        reduced=True,
        filter_out={'Kdn', 'Neu5Ac'}
    )
    # Verify no structures contain Neu5Ac
    for glycan in result['glycan']:
        assert 'Neu5Ac' not in glycan


# Test the rescue_glycans decorator
def test_rescue_glycans():
    result = glycan_to_mass("Fuc(a1-2)Gal(b1-3)GalNAc") + 1.0078
    assert abs(result - 530) < 0.5
    result = glycan_to_mass("Fuc(a1-2Gal(b1-3)GalNAc") + 1.0078
    assert abs(result - 530) < 0.5
    result = glycan_to_mass("Fuca2Galb3GalNAc") + 1.0078
    assert abs(result - 530) < 0.5
    result = glycan_to_mass("Fuca2Galb3GalNac") + 1.0078
    assert abs(result - 530) < 0.5


# Test the rescue_compositions decorator
def test_rescue_compositions():
    result = composition_to_mass({'HexNAc': 1, 'Hex': 1, 'dHex': 1}) + 1.0078
    assert abs(result - 530) < 0.5
    result = composition_to_mass('Hex1HexNAc1Fuc1') + 1.0078
    assert abs(result - 530) < 0.5
    result = composition_to_mass('H1N1F1') + 1.0078
    assert abs(result - 530) < 0.5


def test_min_process_glycans():
    glycans = [
        "Neu5Ac(a2-3)Gal(b1-4)GlcNAc",
        "Gal(b1-4)[Fuc(a1-3)]GlcNAc"
    ]
    result = min_process_glycans(glycans)
    assert result[0] == ['Neu5Ac', 'a2-3', 'Gal', 'b1-4', 'GlcNAc']
    assert result[1] == ['Gal', 'b1-4', 'Fuc', 'a1-3', 'GlcNAc']


def test_get_lib():
    glycans = ["Neu5Ac(a2-3)Gal(b1-4)GlcNAc"]
    lib = get_lib(glycans)
    assert 'Neu5Ac' in lib
    assert 'Gal' in lib
    assert 'GlcNAc' in lib
    assert 'a2-3' in lib
    assert 'b1-4' in lib


def test_expand_lib():
    existing_lib = {'Gal': 0, 'GlcNAc': 1, 'b1-4': 2}
    new_glycans = ["Neu5Ac(a2-3)Gal(b1-4)GlcNAc"]
    expanded = expand_lib(existing_lib, new_glycans)
    assert 'Neu5Ac' in expanded
    assert 'a2-3' in expanded
    assert expanded['Gal'] == 0  # Should keep original indices
    assert expanded['GlcNAc'] == 1


def test_get_possible_linkages():
    # Test exact match
    assert 'a1-3' in get_possible_linkages('a1-3')
    # Test wildcard
    result = get_possible_linkages('a?-3')
    assert 'a1-3' in result
    assert 'a2-3' in result
    # Test multiple wildcards
    result = get_possible_linkages('?1-?')
    assert 'a1-3' in result
    assert 'b1-4' in result


def test_get_possible_monosaccharides():
    # Test Hex category
    hex_result = get_possible_monosaccharides('Hex')
    assert 'Glc' in hex_result
    assert 'Gal' in hex_result
    assert 'Man' in hex_result
    # Test HexNAc category
    hexnac_result = get_possible_monosaccharides('HexNAc')
    assert 'GlcNAc' in hexnac_result
    assert 'GalNAc' in hexnac_result


def test_de_wildcard_glycoletter():
    # Test linkage wildcard
    result = de_wildcard_glycoletter('?1-?')
    assert result in linkages
    # Test monosaccharide wildcard
    result = de_wildcard_glycoletter('Hex')
    assert result in Hex


def test_find_isomorphs():
    # Test branch swapping
    glycan = "Gal(b1-4)[Fuc(a1-3)]GlcNAc"
    isomorphs = find_isomorphs(glycan)
    assert "Fuc(a1-3)[Gal(b1-4)]GlcNAc" in isomorphs
    # Test complex branching
    glycan = "Neu5Ac(a2-3)Gal(b1-4)[Fuc(a1-3)]GlcNAc"
    isomorphs = find_isomorphs(glycan)
    assert len(isomorphs) > 1
    # Test with floating parts
    glycan = "{Neu5Ac(a2-3)}Gal(b1-4)[Fuc(a1-3)]GlcNAc"
    isomorphs = find_isomorphs(glycan)
    assert len(isomorphs) == 2
    assert all(iso.startswith("{Neu5Ac(a2-3)}") for iso in isomorphs)


def test_canonicalize_iupac():
    # Test basic cleanup
    assert canonicalize_iupac("Galb4GlcNAc") == "Gal(b1-4)GlcNAc"
    # Test linkage uncertainty
    assert canonicalize_iupac("Gal-GlcNAc") == "Gal(?1-?)GlcNAc"
    # Test modification handling
    assert canonicalize_iupac("6SGal(b1-4)GlcNAc") == "Gal6S(b1-4)GlcNAc"


def test_glycoct_to_iupac():
    glycoct = """RES
1b:x-dglc-HEX-1:5
2s:n-acetyl
3b:b-dglc-HEX-1:5
4s:n-acetyl
5b:b-dman-HEX-1:5
6b:a-dman-HEX-1:5
7b:x-dglc-HEX-1:5
8s:n-acetyl
9b:a-dman-HEX-1:5
10b:a-lgal-HEX-1:5|6:d
LIN
1:1d(2+1)2n
2:1o(4+1)3d
3:3d(2+1)4n
4:3o(4+1)5d
5:5o(3+1)6d
6:6o(-1+1)7d
7:7d(2+1)8n
8:5o(6+1)9d
9:1o(6+1)10d"""
    result = glycoct_to_iupac(glycoct)
    assert "Man" in result
    assert "GlcNAc" in result


def test_wurcs_to_iupac():
    wurcs = "WURCS=2.0/6,10,9/[a2122h-1x_1-5_2*NCC/3=O][a2122h-1b_1-5_2*NCC/3=O][a1122h-1b_1-5][a1122h-1a_1-5][a2112h-1b_1-5][Aad21122h-2a_2-6_5*NCC/3=O]/1-2-3-4-2-5-6-4-2-5/a4-b1_b4-c1_c3-d1_c6-h1_d2-e1_e4-f1_f6-g2_h2-i1_i4-j1"
    result = wurcs_to_iupac(wurcs)
    assert "GlcNAc" in result
    assert "Man" in result
    assert "Gal" in result
    assert "Neu5Ac" in result


def test_oxford_to_iupac():
    # Test basic N-glycan
    assert "GlcNAc" in oxford_to_iupac("M3")
    # Test complex glycan
    complex_result = oxford_to_iupac("FA2G2")
    assert "Gal" in complex_result
    assert "GlcNAc" in complex_result
    assert "Fuc" in complex_result


def test_canonicalize_composition():
    # Test H5N4F1A2 format
    result = canonicalize_composition("H5N4F1A2")
    assert result["Hex"] == 5
    assert result["HexNAc"] == 4
    assert result["dHex"] == 1
    assert result["Neu5Ac"] == 2
    # Test Hex5HexNAc4Fuc1Neu5Ac2 format
    result = canonicalize_composition("Hex5HexNAc4Fuc1Neu5Ac2")
    assert result["Hex"] == 5
    assert result["HexNAc"] == 4
    assert result["dHex"] == 1
    assert result["Neu5Ac"] == 2


def test_parse_glycoform():
    # Test string input
    result = parse_glycoform("H5N4F1A2")
    assert result["H"] == 5
    assert result["N"] == 4
    assert result["F"] == 1
    assert result["A"] == 2
    # Test dict input
    result = parse_glycoform({"Hex": 5, "HexNAc": 4, "dHex": 1, "Neu5Ac": 2})
    print(result)
    assert result["complex"] == 1
    assert result["high_Man"] == 0


def test_presence_to_matrix():
    df = pd.DataFrame({
        'glycan': ['Gal(b1-4)GlcNAc', 'Man(a1-3)Man', 'Gal(b1-4)GlcNAc'],
        'Species': ['Human', 'Mouse', 'Human']
    })
    result = presence_to_matrix(df)
    assert result.loc['Human', 'Gal(b1-4)GlcNAc'] == 2
    assert result.loc['Mouse', 'Man(a1-3)Man'] == 1


def test_process_for_glycoshift():
    df = pd.DataFrame(
        index=['PROT1_123_H5N4F1A2', 'PROT1_124_H3N3F1A1'],
        data={'abundance': [1.0, 2.0]}
    )
    result, features = process_for_glycoshift(df)
    assert 'Glycosite' in result.columns
    assert 'Glycoform' in result.columns
    assert 'HexNAc' in features
    assert 'Hex' in features


def test_linearcode_to_iupac():
    assert linearcode_to_iupac("Ma3(Ma6)Mb4GNb4GN;") == 'Mana3(Mana6)Manb4GlcNAcb4GlcNAc'


def test_iupac_extended_to_condensed():
    assert iupac_extended_to_condensed("β-D-Galp-(1→4)-β-D-GlcpNAc-(1→") == 'Galp(β1→4)GlcpNAc'
    assert iupac_extended_to_condensed("α-D-Neup5Ac-(2→3)-β-D-Galp-(1→4)-β-D-GlcpNAc-(1→") == 'Neup5Ac(α2→3)Galp(β1→4)GlcpNAc'


def test_get_class():
    assert get_class("GalNAc(b1-3)GalNAc") == "O"
    assert get_class("GlcNAc(b1-4)GlcNAc") == "N"
    assert get_class("Glc-ol") == "free"
    assert get_class("Glc1Cer") == "lipid"
    assert get_class("UnknownGlycan") == ""


def test_enforce_class():
    # Test N-glycan
    assert enforce_class("GlcNAc(b1-4)GlcNAc", "N") == True
    assert enforce_class("GalNAc(b1-3)GalNAc", "N") == False
    # Test O-glycan
    assert enforce_class("GalNAc(b1-3)GalNAc", "O") == True
    # Test with confidence override
    assert enforce_class("GalNAc(b1-3)GalNAc", "N", conf=0.9, extra_thresh=0.3) == True


def test_in_lib():
    library = {'Gal': 0, 'GlcNAc': 1, 'b1-4': 2, 'a2-3': 3, 'Neu5Ac': 4}
    assert in_lib("Gal(b1-4)GlcNAc", library) == True
    assert in_lib("Gal(b1-4)Unknown", library) == False
    assert in_lib("Neu5Ac(a2-3)Gal(b1-4)GlcNAc", library) == True


def test_equal_repeats():
    # Test identical repeats
    assert equal_repeats("Gal(b1-4)GlcNAc(b1-3)", "Gal(b1-4)GlcNAc(b1-3)") == True
    # Test shifted repeats
    assert equal_repeats("GlcNAc(b1-3)Gal(b1-3)", "Gal(b1-3)GlcNAc(b1-3)") == True
    # Test different repeats
    assert equal_repeats("Gal(b1-4)GlcNAc(b1-3)", "Man(a1-3)Man(a1-2)") == False


def test_find_matching_brackets_indices():
    # Test simple brackets
    assert find_matching_brackets_indices("[abc]") == [(0, 4)]
    # Test nested brackets
    assert find_matching_brackets_indices("a[b[c]d]e") == [(1, 7), (3, 5)]
    # Test multiple brackets
    result = find_matching_brackets_indices("[a][b][c]")
    assert len(result) == 3
    assert result[0] == (0, 2)
    # Test unmatched brackets
    assert find_matching_brackets_indices("[a[b") is None


def test_choose_correct_isoform():
    glycans = [
        "Gal(b1-4)[Fuc(a1-3)]GlcNAc",
        "Fuc(a1-3)[Gal(b1-4)]GlcNAc"
    ]
    # Should choose the form with the longest main chain
    result = choose_correct_isoform(glycans)
    assert result == "Fuc(a1-3)[Gal(b1-4)]GlcNAc"
    # Test with wildcards
    glycans_with_wildcards = [
        "Gal(b1-?)[Fuc(a1-3)]GlcNAc",
        "Gal(b1-4)[Fuc(a1-3)]GlcNAc"
    ]
    # Should choose the more specific form
    result = choose_correct_isoform(glycans_with_wildcards)
    assert result == "Gal(b1-4)[Fuc(a1-3)]GlcNAc"


def test_bracket_removal():
    # Test simple branch removal
    assert bracket_removal("Gal(b1-4)[Fuc(a1-3)]GlcNAc") == "Gal(b1-4)GlcNAc"
    # Test nested branch removal
    assert bracket_removal("Neu5Ac(a2-3)[Neu5Ac(a2-6)[Gal(b1-3)]Gal(b1-4)]GlcNAc") == "Neu5Ac(a2-3)GlcNAc"
    # Test multiple branches
    assert bracket_removal("Gal(b1-4)[Fuc(a1-3)][Man(a1-6)]GlcNAc") == "Gal(b1-4)GlcNAc"


def test_check_nomenclature():
    # Test SMILES format (should print warning)
    with pytest.raises(ValueError) as exc_info:
        check_nomenclature("C@C(=O)NC1C(O)OC(CO)C(O)C1O")
    assert "can only convert IUPAC-->SMILES" in str(exc_info.value)
    # Test non-string input
    with pytest.raises(TypeError) as exc_info:
        check_nomenclature(123)
    assert "must be formatted as strings" in str(exc_info.value)
    # Test valid IUPAC format
    assert check_nomenclature("Gal(b1-4)GlcNAc") is None


def test_IUPAC_to_SMILES():
    try:
        # Test basic conversion
        glycans = ["Gal(b1-4)GlcNAc"]
        smiles = IUPAC_to_SMILES(glycans)
        assert isinstance(smiles, list)
        assert len(smiles) == 1
        assert all('@' in s for s in smiles)  # SMILES should contain stereochemistry
        # Test multiple glycans
        glycans = ["Gal(b1-4)GlcNAc", "Man(a1-3)Man"]
        smiles = IUPAC_to_SMILES(glycans)
        assert len(smiles) == 2
        # Test invalid input type
        with pytest.raises(TypeError):
            IUPAC_to_SMILES("Gal(b1-4)GlcNAc")  # Should be list
    except ImportError:
        pytest.skip("glyles package not installed")


def test_unwrap():
    # Test nested list flattening
    nested = [[1, 2], [3, 4], [5, 6]]
    assert unwrap(nested) == [1, 2, 3, 4, 5, 6]
    # Test with strings
    nested_str = [['a', 'b'], ['c', 'd']]
    assert unwrap(nested_str) == ['a', 'b', 'c', 'd']
    # Test empty lists
    assert unwrap([[], []]) == []
    # Test single-level list
    assert unwrap([[1, 2, 3]]) == [1, 2, 3]


def test_find_nth():
    # Test basic string search
    assert find_nth("abcabc", "abc", 1) == 0
    assert find_nth("abcabc", "abc", 2) == 3
    # Test when pattern doesn't exist
    assert find_nth("abcabc", "xyz", 1) == -1
    # Test when n is too large
    assert find_nth("abcabc", "abc", 3) == -1
    # Test with glycan-like string
    glycan = "Gal(b1-4)Gal(b1-4)GlcNAc"
    assert find_nth(glycan, "Gal", 1) == 0
    assert find_nth(glycan, "Gal", 2) == 9


def test_find_nth_reverse():
    # Test basic reverse search
    assert find_nth_reverse("abcabc", "abc", 1) == 3
    assert find_nth_reverse("abcabc", "abc", 2) == 0
    # Test with ignore_branches=True
    glycan = "Gal(b1-4)[Fuc(a1-3)]GlcNAc"
    assert find_nth_reverse(glycan, "Gal", 1, ignore_branches=True) == 0
    # Test with nested branches
    glycan = "Neu5Ac(a2-3)[Neu5Ac(a2-6)[Gal(b1-3)]Gal(b1-4)]GlcNAc"
    result = find_nth_reverse(glycan, "Gal", 2, ignore_branches=True)
    assert result >= 0


def test_remove_unmatched_brackets():
    # Test balanced brackets
    assert remove_unmatched_brackets("[abc]") == "[abc]"
    # Test unmatched opening bracket
    assert remove_unmatched_brackets("[abc") == "abc"
    # Test unmatched closing bracket
    assert remove_unmatched_brackets("abc]") == "abc"
    # Test nested brackets
    assert remove_unmatched_brackets("[a[bc]]") == "[a[bc]]"
    # Test glycan-like string
    glycan = "Gal(b1-4)[Fuc(a1-3]GlcNAc"  # Missing closing bracket
    result = remove_unmatched_brackets(glycan)
    assert result.count('[') == result.count(']')


def test_reindex():
    # Create test dataframes
    df_old = pd.DataFrame({
        'id': ['a', 'b', 'c'],
        'value': [1, 2, 3]
    })
    df_new = pd.DataFrame({
        'id': ['c', 'a', 'b']
    })
    # Test reindexing
    result = reindex(df_new, df_old, 'value', 'id', 'id')
    assert result == [3, 1, 2]


def test_stringify_dict():
    # Test basic dictionary
    d = {'a': 1, 'b': 2, 'c': 3}
    assert stringify_dict(d) == 'a1b2c3'
    # Test empty dictionary
    assert stringify_dict({}) == ''
    # Test dictionary with various types
    d = {'Hex': 5, 'HexNAc': 4, 'Neu5Ac': 2}
    result = stringify_dict(d)
    assert isinstance(result, str)
    assert 'Hex5' in result


def test_replace_every_second():
    # Test basic replacement
    assert replace_every_second("aaa", "a", "b") == "aba"
    # Test with mixed characters
    assert replace_every_second("ababa", "a", "c") == "abcba"
    # Test empty string
    assert replace_every_second("", "a", "b") == ""
    # Test no matches
    assert replace_every_second("xyz", "a", "b") == "xyz"


def test_multireplace():
    # Test basic replacements
    replacements = {'a': 'x', 'b': 'y'}
    assert multireplace("abc", replacements) == "xyc"
    # Test glycan-specific replacements
    replacements = {'Nac': 'NAc', 'AC': 'Ac'}
    assert multireplace("GlcNac", replacements) == "GlcNAc"
    # Test empty string
    assert multireplace("", {'a': 'b'}) == ""
    # Test no matches
    assert multireplace("xyz", {'a': 'b'}) == "xyz"


def test_strip_suffixes():
    # Test numerical suffixes
    cols = ['name.1', 'value.2', 'data.10']
    assert strip_suffixes(cols) == ['name', 'value', 'data']
    # Test mixed columns
    cols = ['name.1', 'value', 'data.10']
    assert strip_suffixes(cols) == ['name', 'value', 'data']
    # Test no suffixes
    cols = ['name', 'value', 'data']
    assert strip_suffixes(cols) == ['name', 'value', 'data']


def test_build_custom_df():
    # Create test DataFrame
    df = pd.DataFrame({
        'glycan': ['Gal', 'GlcNAc'],
        'Species': [['Human', 'Mouse'], ['Mouse']],
        'Genus': [['Homo', 'Mus'], ['Mus']],
        'Family': [['Hominidae', 'Muridae'], ['Muridae']],
        'Order': [['Primates', 'Rodentia'], ['Rodentia']],
        'Class': [['Mammalia', 'Mammalia'], ['Mammalia']],
        'Phylum': [['Chordata', 'Chordata'], ['Chordata']],
        'Kingdom': [['Animalia', 'Animalia'], ['Animalia']],
        'Domain': [['Eukaryota', 'Eukaryota'], ['Eukaryota']],
        'ref': [['ref1', 'ref2'], ['ref3']]
    })
    # Test df_species creation
    result = build_custom_df(df, 'df_species')
    assert len(result) > len(df)  # Should expand due to explode
    assert 'glycan' in result.columns
    assert 'Species' in result.columns


def test_dataframe_serializer():
    # Create test DataFrame with various types
    df = pd.DataFrame({
        'strings': ['a', 'b', 'c'],
        'lists': [[1, 2], [3, 4], [5, 6]],
        'dicts': [{'x': 1}, {'y': 2}, {'z': 3}],
        'nulls': [None, pd.NA, None]
    })
    # Test serialization and deserialization
    serializer = DataFrameSerializer()
    # Test full cycle
    with pytest.raises(Exception):  # Should fail without proper file path
        serializer.serialize(df, "")
    # Test cell serialization
    cell = serializer._serialize_cell([1, 2, 3])
    assert cell['type'] == 'list'
    assert cell['value'] == ['1', '2', '3']
    # Test cell deserialization
    cell_data = {'type': 'list', 'value': ['1', '2', '3']}
    result = serializer._deserialize_cell(cell_data)
    assert result == ['1', '2', '3']


def test_fast_two_sum():
    # Test basic addition
    assert fast_two_sum(5, 3) == [8]
    # Test with floating points
    assert fast_two_sum(5.5, 3.3)[0] == 8.8
    # Test with negative numbers
    assert fast_two_sum(-5, -3) == [-8]


def test_two_sum():
    # Test basic addition
    assert two_sum(5, 3) == [8]
    # Test order doesn't matter
    assert two_sum(3, 5) == [8]
    # Test with floating points
    result = two_sum(5.5, 3.3)
    assert abs(result[0] - 8.8) < 1e-10


def test_expansion_sum():
    # Test multiple numbers
    assert expansion_sum(1, 2, 3) == 6
    # Test floating points
    result = expansion_sum(1.1, 2.2, 3.3)
    assert abs((result if isinstance(result, float) else sum(result[0])+result[1]) - 6.6) < 1e-10
    # Test negative numbers
    assert expansion_sum(-1, -2, -3) == -6


def test_hlm():
    # Test with simple array
    data = [1, 2, 3, 4, 5]
    result = hlm(data)
    assert 2.5 <= result <= 3.5  # Should be around 3
    # Test with negative numbers
    data = [-2, -1, 0, 1, 2]
    result = hlm(data)
    assert abs(result) < 1e-10  # Should be around 0


def test_cohen_d():
    # Test with clearly different groups
    group1 = [1, 2, 3, 4, 5]
    group2 = [6, 7, 8, 9, 10]
    d, _ = cohen_d(group1, group2)
    assert d < 0  # Effect size should be negative (group1 < group2)
    # Test with paired samples
    d, _ = cohen_d(group1, group2, paired=True)
    assert d < 0  # Should still be negative
    # Test with identical groups
    d, _ = cohen_d(group1, group1)
    assert abs(d) < 1e-10  # Effect size should be approximately 0


def test_mahalanobis_distance():
    # Test with 2D data
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[7, 8], [9, 10], [11, 12]])
    dist = mahalanobis_distance(x, y)
    assert dist > 0  # Distance should be positive
    # Test with paired data
    dist_paired = mahalanobis_distance(x, y, paired=True)
    assert dist_paired >= 0  # Should still be positive


def test_variance_stabilization():
    # Create test data
    data = pd.DataFrame({
        'sample1': [1, 2, 3],
        'sample2': [4, 5, 6],
        'sample3': [7, 8, 9]
    })
    # Test global normalization
    result = variance_stabilization(data)
    assert (result.std() - 1).abs().mean() < 0.1  # Should have ~unit variance
    # Test group-specific normalization
    groups = [['sample1', 'sample2'], ['sample3']]
    result = variance_stabilization(data, groups)
    assert result.shape == data.shape


def test_biodiversity_metrics():
    # Test data
    counts = np.array([10, 20, 30, 40])
    # Test sequence richness
    assert sequence_richness(counts) == 4
    assert sequence_richness(np.array([1, 0, 3, 0])) == 2
    # Test Shannon diversity
    shannon = shannon_diversity_index(counts)
    assert 0 <= shannon <= np.log(len(counts))
    # Test Simpson diversity
    simpson = simpson_diversity_index(counts)
    assert 0 <= simpson <= 1


def test_get_equivalence_test():
    # Test data
    group1 = np.array([1, 2, 3, 4, 5])
    group2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    # Test unpaired
    p_val = get_equivalence_test(group1, group2)
    assert 0 <= p_val <= 1
    # Test paired
    p_val_paired = get_equivalence_test(group1, group2, paired=True)
    assert 0 <= p_val_paired <= 1


def test_hotellings_t2():
    # Create multivariate test data
    group1 = np.array([[1, 2], [3, 4], [5, 6]])
    group2 = np.array([[7, 8], [9, 10], [11, 12]])
    # Test unpaired
    F_stat, p_val = hotellings_t2(group1, group2)
    assert F_stat >= 0
    assert 0 <= p_val <= 1
    # Test paired
    F_stat_paired, p_val_paired = hotellings_t2(group1, group2, paired=True)
    assert F_stat_paired >= 0
    assert 0 <= p_val_paired <= 1


def test_calculate_permanova_stat():
    # Create distance matrix
    dist_matrix = pd.DataFrame([
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [2, 2, 0, 1],
        [3, 3, 1, 0]
    ])
    group_labels = ['A', 'A', 'B', 'B']
    # Test calculation
    F_stat = calculate_permanova_stat(dist_matrix, group_labels)
    assert F_stat >= 0


def test_omega_squared():
    # Create test data
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    groups = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
    # Calculate effect size
    omega_sq = omega_squared(data, groups)
    assert -1 <= omega_sq <= 1  # Omega squared should be between -1 and 1


def test_anosim():
    # Create test distance matrix
    dist_matrix = pd.DataFrame([
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [2, 2, 0, 1],
        [3, 3, 1, 0]
    ])
    group_labels = ['A', 'A', 'B', 'B']
    # Test ANOSIM
    R, p_val = anosim(dist_matrix, group_labels)
    assert -1 <= R <= 1  # R should be between -1 and 1
    assert 0 <= p_val <= 1


def test_alpha_biodiversity_stats():
    # Create test data
    distances = pd.DataFrame([1.0, 2.0, 1.5, 2.5])
    group_labels = ['A', 'A', 'B', 'B']
    # Test statistics
    result = alpha_biodiversity_stats(distances, group_labels)
    if result:  # Will be None if groups have only 1 sample
        F_stat, p_val = result
        assert F_stat >= 0
        assert 0 <= p_val <= 1


def test_permanova_with_permutation():
    # Create test distance matrix
    dist_matrix = pd.DataFrame([
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [2, 2, 0, 1],
        [3, 3, 1, 0]
    ])
    group_labels = ['A', 'A', 'B', 'B']
    # Test PERMANOVA
    F_stat, p_val = permanova_with_permutation(dist_matrix, group_labels, permutations=99)
    assert F_stat >= 0
    assert 0 <= p_val <= 1


def test_clr_transformation():
    # Create test data
    data = pd.DataFrame({
        'sample1': [1, 2, 3],
        'sample2': [4, 5, 6],
        'sample3': [7, 8, 9]
    })
    group1 = ['sample1', 'sample2']
    group2 = ['sample3']
    # Test CLR transformation
    result = clr_transformation(data, group1, group2)
    assert result.shape == data.shape
    # Test with scale model
    result_scaled = clr_transformation(data, group1, group2, custom_scale=2.0)
    assert result_scaled.shape == data.shape


def test_alr_transformation():
    # Create test data
    data = pd.DataFrame({
        'sample1': [1, 2, 3],
        'sample2': [4, 5, 6],
        'sample3': [7, 8, 9]
    })
    group1 = ['sample1', 'sample2']
    group2 = ['sample3']
    # Test ALR transformation
    result = alr_transformation(data, 0, group1, group2)
    assert result.shape[0] == data.shape[0] - 1  # One fewer row due to reference
    # Test with scale model
    result_scaled = alr_transformation(data, 0, group1, group2, custom_scale=2.0)
    assert result_scaled.shape[0] == data.shape[0] - 1


def test_get_procrustes_scores():
    # Create test data
    data = pd.DataFrame({
        'glycan': ['Gal', 'GlcNAc', 'Man'],
        'sample1': [1, 2, 3],
        'sample2': [4, 5, 6],
        'sample3': [7, 8, 9],
        'sample4': [10, 11, 12]
    })
    group1 = ['sample1', 'sample2']
    group2 = ['sample3', 'sample4']
    # Test Procrustes analysis
    scores, corr, var = get_procrustes_scores(data, group1, group2)
    assert len(scores) == len(corr) == len(var) == data.shape[0]


def test_get_additive_logratio_transformation():
    # Create test data
    data = pd.DataFrame({
        'glycan': ['Gal', 'GlcNAc', 'Man'],
        'sample1': [1, 2, 3],
        'sample2': [4, 5, 6],
        'sample3': [7, 8, 9],
        'sample4': [10, 11, 12]
    })
    group1 = ['sample1', 'sample2']
    group2 = ['sample3', 'sample4']
    # Test ALR transformation with reference selection
    result = get_additive_logratio_transformation(data, group1, group2)
    assert 'glycan' in result.columns


def test_bayes_factor_functions():
    # Test get_BF
    bf = get_BF(n=20, p=0.05)
    assert bf > 0
    # Test get_alphaN
    alpha = get_alphaN(n=20)
    assert 0 < alpha < 1


def test_pi0_tst():
    # Create test p-values
    p_values = np.array([0.01, 0.02, 0.03, 0.5, 0.6, 0.7, 0.8, 0.9])
    # Test pi0 estimation
    pi0 = pi0_tst(p_values)
    assert 0 <= pi0 <= 1


def test_tst_grouped_benjamini_hochberg():
    # Create test data
    identifiers = {
        'group1': ['g1', 'g2'],
        'group2': ['g3', 'g4']
    }
    p_values = {
        'group1': [0.01, 0.02],
        'group2': [0.03, 0.04]
    }
    # Test correction
    adj_pvals, sig_dict = TST_grouped_benjamini_hochberg(identifiers, p_values, 0.05)
    assert len(adj_pvals) == len(sig_dict) == 4
    assert all(0 <= p <= 1 for p in adj_pvals.values())
    assert all(isinstance(s, bool) for s in sig_dict.values())


def test_compare_inter_vs_intra_group():
    # Create test data
    cohort_a = pd.DataFrame({
        'sample1': [1, 2, 3],
        'sample2': [4, 5, 6]
    })
    cohort_b = pd.DataFrame({
        'sample3': [7, 8, 9],
        'sample4': [10, 11, 12]
    })
    glycans = ['Gal', 'GlcNAc', 'Man']
    grouped_glycans = {'group1': ['Gal'], 'group2': ['GlcNAc', 'Man']}
    # Test correlation analysis
    intra, inter = compare_inter_vs_intra_group(cohort_b, cohort_a, glycans, grouped_glycans)
    assert -1 <= intra <= 1
    assert -1 <= inter <= 1


def test_correct_multiple_testing():
    # Create test p-values
    p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    # Test correction
    corrected_pvals, significance = correct_multiple_testing(p_values, 0.05)
    assert len(corrected_pvals) == len(significance) == len(p_values)
    assert all(0 <= p <= 1 for p in corrected_pvals)
    assert all(isinstance(s, bool) for s in significance)


def test_partial_corr():
    # Create test data
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    controls = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]).T
    # Test partial correlation
    corr, p_val = partial_corr(x, y, controls)
    assert -1 <= corr <= 1
    assert 0 <= p_val <= 1


def test_estimate_technical_variance():
    # Create test data
    data = pd.DataFrame({
        'sample1': [1, 2, 3],
        'sample2': [4, 5, 6],
        'sample3': [7, 8, 9]
    })
    group1 = ['sample1', 'sample2']
    group2 = ['sample3']
    # Test variance estimation
    result = estimate_technical_variance(data, group1, group2, num_instances=10)
    assert result.shape[1] == (len(group1) + len(group2)) * 10


def test_jtkdist():
    # Test JTK distribution calculation
    param_dic = {}
    result = jtkdist(timepoints=6, param_dic=param_dic)
    assert "GRP_SIZE" in result
    assert "NUM_GRPS" in result
    assert "NUM_VALS" in result
    assert "MAX" in result
    assert "DIMS" in result
    # Test with normal approximation
    result_normal = jtkdist(timepoints=6, param_dic={}, normal=True)
    assert "VAR" in result_normal
    assert "SDV" in result_normal
    assert "EXV" in result_normal
    assert "EXACT" in result_normal


def test_jtkinit():
    # Test JTK initialization
    periods = [24]  # 24-hour period
    param_dic = {
        'GRP_SIZE': [1] * 24,
        'INTERVAL': 1,
        'NUM_GRPS': 24,  # Match period length
        'NUM_VALS': 24,
        'DIMS': [276, 1],  # (24 * 23) / 2 = 276
        'PERFACTOR': np.ones(24),
        'CGOOSV': [np.zeros([276, 24])]
    }
    result = jtkinit(periods, param_dic, interval=1, replicates=1)
    assert "INTERVAL" in result
    assert "PERIODS" in result
    assert "PERFACTOR" in result
    assert "SIGNCOS" in result
    assert "CGOOSV" in result


def test_jtkstat():
    # Test JTK statistics calculation
    # Create mock expression data
    z = pd.DataFrame([1, 2, 1, 2, 1, 2])  # Cycling data
    param_dic = {
        "MAX": 15,
        "PERIODS": [6],
        "DIMS": [15, 1],
        "CGOOSV": [[np.zeros(15) for _ in range(6)]]
    }
    result = jtkstat(z, param_dic)
    assert "CJTK" in result


def test_jtkx():
    # Test JTK deployent
    z = pd.Series([1, 2, 1, 2, 1, 2], index=['t1', 't2', 't3', 't4', 't5', 't6'])  # Cycling data
    param_dic = {"GRP_SIZE": [], "NUM_GRPS": [], "MAX": [], "DIMS": [], "EXACT": bool(True),
                 "VAR": [], "EXV": [], "SDV": [], "CGOOSV": []}
    param_dic = jtkdist(6, param_dic, reps=1)
    param_dic = jtkinit([6], param_dic, interval=1, replicates=1)
    result = jtkx(z, param_dic)
    assert len(result) == 4  # Should return p-value, period, lag, and amplitude


def test_missforest():
    # Test MissForest imputation
    data = pd.DataFrame({
        'A': [1, np.nan, 3, 4],
        'B': [np.nan, 2, 3, 4],
        'C': [1, 2, np.nan, 4]
    })
    mf = MissForest()
    imputed = mf.fit_transform(data)
    assert not imputed.isnull().any().any()
    assert imputed.shape == data.shape


def test_impute_and_normalize():
    # Test imputation and normalization
    data = pd.DataFrame({
        'glycan': ['Gal', 'GlcNAc', 'Man'],
        'sample1': [1, 0, 3],
        'sample2': [0, 2, 0],
        'sample3': [3, 0, 3]
    })
    groups = [['sample1', 'sample2'], ['sample3']]
    result = impute_and_normalize(data, groups)
    assert result.shape == data.shape
    assert not (result.iloc[:, 1:] == 0).any().any()
    assert abs(result.iloc[:, 1:].sum().sum() - 300) < 1e-5  # Check normalization to 100%


def test_variance_based_filtering():
    # Test variance-based filtering
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [1.1, 1.9, 3.1],
        'C': [2, 2, 2]  # Low variance
    }).T
    filtered, discarded = variance_based_filtering(data, min_feature_variance=0.1)
    assert len(filtered) + len(discarded) == len(data)
    assert len(discarded) == 1  # The constant feature should be discarded


def test_get_glycoform_diff():
    # Test glycoform differential expression
    result_df = pd.DataFrame({
        'Glycosite': ['Prot1_123_H5N4', 'Prot1_123_H6N5', 'Prot2_456_H4N3'],
        'corr p-val': [0.01, 0.02, 0.03],
        'Effect size': [0.5, 0.6, 0.7]
    })
    result = get_glycoform_diff(result_df, alpha=0.05, level='peptide')
    assert 'Glycosite' in result.columns
    assert 'corr p-val' in result.columns
    assert 'significant' in result.columns
    assert 'Effect size' in result.columns


def test_get_glm():
    # Test GLM fitting
    data = pd.DataFrame({
        'Glycosite': ['Site1'] * 8 + ['Site2'] * 8,
        'Abundance': [1.0, 1.2, 0.8, 1.1, 1.3, 0.9, 1.4, 0.7,  # Site1 abundances
                     2.0, 2.2, 1.8, 2.1, 2.3, 1.9, 2.4, 1.7],  # Site2 abundances
        'Condition': [0, 0, 0, 0, 1, 1, 1, 1] * 2,
        'H': [1, 2, 1, 2, 1, 2, 1, 2] * 2,
        'N': [2, 3, 2, 3, 2, 3, 2, 3] * 2
    })
    model, variables = get_glm(data)
    if not isinstance(model, str):  # If model fitting succeeded
        assert hasattr(model, 'params')
        assert hasattr(model, 'pvalues')
        assert len(variables) > 0


def test_process_glm_results():
    # Test GLM results processing
    data = pd.DataFrame({
        'Glycosite': ['Site1'] * 8 + ['Site2'] * 8,
        'Abundance': [1.0, 1.2, 0.8, 1.1, 1.3, 0.9, 1.4, 0.7,  # Site1 abundances
                     2.0, 2.2, 1.8, 2.1, 2.3, 1.9, 2.4, 1.7],  # Site2 abundances
        'Condition': [0, 0, 0, 0, 1, 1, 1, 1] * 2,
        'H': [1, 2, 1, 2, 1, 2, 1, 2] * 2,
        'N': [2, 3, 2, 3, 2, 3, 2, 3] * 2
    })
    result = process_glm_results(data, 0.05, ['H', 'N'])
    assert 'Condition_coefficient' in result.columns
    assert 'Condition_corr_pval' in result.columns
    assert 'Condition_significant' in result.columns
    assert result.shape[0] == len(['Site1', 'Site2'])


def test_replace_outliers_with_iqr_bounds():
    # Test IQR-based outlier replacement
    data = pd.Series([1, 2, 3, 10, 2, 3, 1])  # 10 is an outlier
    result = replace_outliers_with_IQR_bounds(data)
    assert max(result) < 10  # Outlier should be replaced
    assert len(result) == len(data)


def test_replace_outliers_winsorization():
    # Test Winsorization-based outlier replacement
    data = pd.Series([1, 2, 3, 10, 2, 3, 1, 2, 3, 1])  # 10 is an outlier
    result = replace_outliers_winsorization(data)
    assert max(result) < 10  # Outlier should be replaced
    assert len(result) == len(data)


def test_perform_tests_monte_carlo():
    # Test Monte Carlo testing
    group_a = pd.DataFrame(np.random.normal(0, 1, (5, 100)))
    group_b = pd.DataFrame(np.random.normal(0.5, 1, (5, 100)))
    raw_p, adj_p, effect = perform_tests_monte_carlo(
        group_a, group_b, num_instances=10
    )
    assert len(raw_p) == len(adj_p) == len(effect) == 5
    assert all(0 <= p <= 1 for p in raw_p)
    assert all(0 <= p <= 1 for p in adj_p)


def test_evaluate_adjacency():
    assert evaluate_adjacency("(", 0)
    assert evaluate_adjacency(")", 0)
    assert not evaluate_adjacency("(1)[", 0)
    assert not evaluate_adjacency(")[", 0)


def test_glycan_to_graph():
    glycan = "Gal(b1-4)GlcNAc"
    node_dict, adj_matrix = glycan_to_graph(glycan)
    assert len(node_dict) == 3 # Gal, b1-4, GlcNAc
    assert adj_matrix.shape == (3, 3)
    assert adj_matrix[0, 1] == 1  # Gal connected to linkage
    assert adj_matrix[1, 2] == 1  # Linkage connected to GlcNAc


def test_glycan_to_nxGraph():
    glycan = "Gal(b1-4)GlcNAc"
    graph = glycan_to_nxGraph(glycan)
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    assert all('string_labels' in data for _, data in graph.nodes(data=True))


def test_compare_glycans():
    # Test identical glycans
    assert compare_glycans("Gal(b1-4)GlcNAc", "Gal(b1-4)GlcNAc")
    # Test order-specificity of comparison
    assert not compare_glycans("Gal(b1-4)GlcNAc", "GlcNAc(b1-4)Gal")
    # Test non-equivalent glycans
    assert not compare_glycans("Gal(b1-4)GlcNAc", "Man(a1-3)GlcNAc")
    # Test with PTM wildcards
    assert compare_glycans("Gal6S(b1-4)GlcNAc", "GalOS(b1-4)GlcNAc")


def test_subgraph_isomorphism():
    # Test motif presence
    assert subgraph_isomorphism("Gal(b1-4)GlcNAc", "GlcNAc")
    # Test motif count
    assert subgraph_isomorphism("Gal(b1-4)GlcNAc(b1-4)GlcNAc", "GlcNAc", count=True) == 2
    # Test with negation
    assert subgraph_isomorphism("Gal(b1-4)GlcNAc", "!Man(a1-3)GlcNAc")
    # Test with termini constraints
    assert subgraph_isomorphism("Gal(b1-4)GlcNAc", "GlcNAc", termini_list=['terminal'])


def test_generate_graph_features():
    glycan = "Gal(b1-4)GlcNAc"
    features = generate_graph_features(glycan)
    assert isinstance(features, pd.DataFrame)
    assert len(features) == 1
    assert 'diameter' in features.columns
    assert 'branching' in features.columns
    assert 'avgDeg' in features.columns


def test_largest_subgraph():
    glycan1 = "Gal(b1-4)GlcNAc(b1-4)GlcNAc"
    glycan2 = "Man(a1-3)GlcNAc(b1-4)GlcNAc"
    result = largest_subgraph(glycan1, glycan2)
    assert "GlcNAc(b1-4)GlcNAc" in result


def test_get_possible_topologies():
    glycan = "{Neu5Ac(a2-?)}Gal(b1-3)GalNAc"
    topologies = get_possible_topologies(glycan)
    assert len(topologies) > 0
    assert all(isinstance(g, nx.Graph) for g in topologies)


def test_deduplicate_glycans():
    glycans = [
        "Gal(b1-4)[Fuc(a1-3)]GlcNAc",
        "Fuc(a1-3)[Gal(b1-4)]GlcNAc",
        "Man(a1-3)GlcNAc"
    ]
    result = deduplicate_glycans(glycans)
    assert len(result) == 2


def test_neighbor_is_branchpoint():
    # Create test graph
    G = nx.Graph()
    G.add_nodes_from(range(6))
    G.add_edges_from([(0,1), (1,2), (2,3), (2,4), (2,5)])
    assert neighbor_is_branchpoint(G, 1)  # Node 1 connected to branch point 2
    assert not neighbor_is_branchpoint(G, 3)  # Node 3 is leaf


def test_subgraph_isomorphism_with_negation():
    glycan = "Gal(b1-4)GlcNAc"
    motif = "!Man(a1-3)GlcNAc"
    # Should match since glycan doesn't contain Man(a1-3)GlcNAc
    assert subgraph_isomorphism_with_negation(glycan, motif)
    # Test with counting
    result, matches = subgraph_isomorphism_with_negation(glycan, motif, count=True, return_matches=True)
    assert isinstance(result, int)
    assert isinstance(matches, list)


def test_categorical_node_match_wildcard():
    narrow_wildcard_list = {"Hex": ["Gal", "Glc", "Man"]}
    matcher = categorical_node_match_wildcard(
        'string_labels', 'unknown', narrow_wildcard_list, 'termini', 'flexible'
    )
    # Test wildcard matching
    assert matcher(
        {'string_labels': 'Hex', 'termini': 'terminal'},
        {'string_labels': 'Gal', 'termini': 'terminal'}
    )
    # Test exact matching
    assert matcher(
        {'string_labels': 'GlcNAc', 'termini': 'terminal'},
        {'string_labels': 'GlcNAc', 'termini': 'terminal'}
    )
    # Test flexible position
    assert matcher(
        {'string_labels': 'GlcNAc', 'termini': 'flexible'},
        {'string_labels': 'GlcNAc', 'termini': 'terminal'}
    )


def test_expand_termini_list():
    motif = "Gal(b1-4)GlcNAc"
    termini_list = ['t', 'i']  # terminal, internal
    result = expand_termini_list(motif, termini_list)
    assert len(result) == 3  # Two monosaccharides + one linkage
    assert 'terminal' in result
    assert 'internal' in result
    assert 'flexible' in result


def test_ensure_graph():
    # Test with string input
    glycan = "Gal(b1-4)GlcNAc"
    result = ensure_graph(glycan)
    assert isinstance(result, nx.Graph)
    # Test with graph input
    G = nx.Graph()
    G.add_node(0, string_labels="Gal")
    result = ensure_graph(G)
    assert result is G


def test_possible_topology_check():
    float_glycan = "{Neu5Ac(a2-?)}Gal(b1-4)GlcNAc"
    glycans = [
        "Neu5Ac(a2-3)Gal(b1-4)GlcNAc",
        "Neu5Ac(a2-6)Gal(b1-4)GlcNAc",
        "Gal(b1-4)GlcNAc"
    ]
    matches = possible_topology_check(float_glycan, glycans)
    assert len(matches) == 2  # Should match first two glycans
    assert "Gal(b1-4)GlcNAc" not in matches


def test_link_find():
    # Simple glycan
    glycan = "Gal(b1-4)Gal(b1-4)GlcNAc"
    result = link_find(glycan)
    assert "Gal(b1-4)GlcNAc" in result
    # Branched glycan
    glycan = "Gal(b1-4)[Fuc(a1-3)]GlcNAc"
    result = link_find(glycan)
    assert "Gal(b1-4)GlcNAc" in result
    assert "Fuc(a1-3)GlcNAc" in result


def test_annotate_glycan():
    glycan = "Gal(b1-4)GlcNAc"
    result = annotate_glycan(glycan)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.index[0] == glycan


def test_annotate_dataset():
    glycans = ["Gal(b1-4)GlcNAc", "Man(a1-3)GlcNAc"]
    # Test known motifs
    result = annotate_dataset(glycans, feature_set=['known'])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    # Test graph features
    result = annotate_dataset(glycans, feature_set=['graph'])
    assert 'diameter' in result.columns
    assert 'branching' in result.columns


def test_get_molecular_properties():
    try:
        glycans = ["Gal(b1-4)GlcNAc"]
        result = get_molecular_properties(glycans, verbose=True, placeholder=True)
        assert isinstance(result, pd.DataFrame)
        assert 'molecular_weight' in result.columns
        assert 'xlogp' in result.columns
    except ImportError:
        pytest.skip("Skipping test due to missing dependencies")


def test_get_k_saccharides():
    glycans = ["Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc"]
    # Test basic functionality
    result = get_k_saccharides(glycans, size=2)
    assert isinstance(result, pd.DataFrame)
    # Test with up_to=True
    result = get_k_saccharides(glycans, size=2, up_to=True)
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) > 0


def test_get_terminal_structures():
    glycan = "Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc"
    # Test size=1
    result = get_terminal_structures(glycan, size=1)
    assert isinstance(result, list)
    assert "Man(b1-4)" in result[0]
    # Test size=2
    result = get_terminal_structures(glycan, size=2)
    assert isinstance(result, list)
    assert "Man(b1-4)GlcNAc(b1-4)" in result


def test_create_correlation_network():
    data = pd.DataFrame({
        'glycan1': [1, 2, 3],
        'glycan2': [1, 2, 3],
        'glycan3': [-1, -2, -3]
    })
    clusters = create_correlation_network(data, 0.9)
    assert isinstance(clusters, list)
    assert isinstance(clusters[0], set)
    assert len(clusters) > 0


def test_group_glycans_core():
    glycans = ["GlcNAc(b1-6)GalNAc", "Gal(b1-3)GalNAc"]
    p_values = [0.01, 0.02]
    glycan_groups, _ = group_glycans_core(glycans, p_values)
    assert "core2" in glycan_groups
    assert "core1" in glycan_groups
    assert len(glycan_groups["core2"]) == 1
    assert len(glycan_groups["core1"]) == 1


def test_group_glycans_sia_fuc():
    glycans = ["Neu5Ac(a2-3)Gal", "Fuc(a1-3)GlcNAc", "Gal(b1-4)GlcNAc"]
    p_values = [0.01, 0.02, 0.03]
    glycan_groups, _ = group_glycans_sia_fuc(glycans, p_values)
    assert "Sia" in glycan_groups
    assert "Fuc" in glycan_groups
    assert "rest" in glycan_groups


def test_group_glycans_N_glycan_type():
    glycans = [
        "Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",  # complex
        "Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",  # high mannose
        "Man(b1-4)GlcNAc(b1-4)GlcNAc"  # other
    ]
    p_values = [0.01, 0.02, 0.03]
    glycan_groups, _ = group_glycans_N_glycan_type(glycans, p_values)
    assert "complex" in glycan_groups
    assert "high_man" in glycan_groups
    assert "rest" in glycan_groups


def test_Lectin():
    lectin = Lectin(
        abbr=["WGA"],
        name=["Wheat Germ Agglutinin"],
        specificity={
            "primary": {"GlcNAc": []},
            "secondary": None,
            "negative": None
        }
    )

    # Test basic attributes
    assert lectin.abbr == ["WGA"]
    assert lectin.name == ["Wheat Germ Agglutinin"]
    # Test binding check
    result = lectin.check_binding("GlcNAc(b1-4)GlcNAc")
    assert result in [0, 1, 2]


def test_load_lectin_lib():
    lectin_lib = load_lectin_lib()
    assert isinstance(lectin_lib, dict)
    assert all(isinstance(v, Lectin) for v in lectin_lib.values())


def test_create_lectin_and_motif_mappings():
    lectin_lib = load_lectin_lib()
    lectin_list = ["WGA", "ConA"]
    lectin_mapping, motif_mapping = create_lectin_and_motif_mappings(
        lectin_list, lectin_lib
    )
    assert isinstance(lectin_mapping, dict)
    assert isinstance(motif_mapping, dict)

def test_lectin_motif_scoring():
    lectin_lib = load_lectin_lib()
    lectin_mapping = {"WGA": 0, "ConA": 1}
    motif_mapping = {"GlcNAc": {"WGA": 0, "ConA": 1}}
    lectin_scores = {"WGA": 1.0, "ConA": -0.5}
    idf = {"WGA": 1.0, "ConA": 1.0}
    result = lectin_motif_scoring(
        lectin_mapping, motif_mapping, lectin_scores,
        lectin_lib, idf
    )
    assert isinstance(result, pd.DataFrame)
    assert "motif" in result.columns
    assert "score" in result.columns


def test_clean_up_heatmap():
    data = pd.DataFrame({
        'sample1': [1, 1],
        'sample2': [2, 2]
    }, index=['motif1', 'motif2'])
    result = clean_up_heatmap(data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= len(data)


def test_quantify_motifs():
    df = pd.DataFrame({
        'sample1': [1, 2],
        'sample2': [2, 3]
    })
    glycans = ["Gal(b1-4)GlcNAc", "Man(a1-3)GlcNAc"]
    result = quantify_motifs(
        df, glycans, feature_set=['exhaustive'],
        remove_redundant=True
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) > 0


def test_count_unique_subgraphs_of_size_k():
    glycan = "Man(a1-2)Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-3)[Man(a1-2)Man(a1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc"
    graph = glycan_to_nxGraph(glycan)
    # Test size 2
    result = count_unique_subgraphs_of_size_k(graph, size=2)
    assert isinstance(result, dict)
    assert len(result) > 0
    # Test terminal only
    result = count_unique_subgraphs_of_size_k(graph, size=2, terminal=True)
    assert isinstance(result, dict)


def test_annotate_glycan_topology_uncertainty():
    glycan = "{Neu5Ac(a2-?)}Fuc(a1-3)[Gal(b1-4)]GlcNAc"
    result = annotate_glycan_topology_uncertainty(glycan)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.index[0] == glycan


def test_preprocess_pattern():
    # Test basic pattern splitting
    assert preprocess_pattern("Hex-HexNAc") == ["Hex-HexNAc"]
    # Test pattern with quantifiers
    assert preprocess_pattern("Hex-[HexNAc|Fuc]{1,2}-Gal") == ["Hex", "[HexNAc|Fuc]{1,2}", "Gal"]
    # Test wildcard replacement
    assert preprocess_pattern("Hex-.-Gal") == ['Hex-Monosaccharide-Gal']
    # Test empty pattern
    assert preprocess_pattern("") == []


def test_specify_linkages():
    # Test basic linkage conversion
    assert specify_linkages("Gala6(?1-?)Gal") == "Gal(a1-6)Gal"
    assert specify_linkages("Manb4(?1-?)GlcNAc") == "Man(b1-4)GlcNAc"
    # Test sialic acid special cases
    assert specify_linkages("Neu5Ac(a1-?)Gal") == "Neu5Ac(a2-?)Gal"
    assert specify_linkages("Sia(a1-?)Gal") == "Sia(a2-?)Gal"
    # Test no modification needed
    assert specify_linkages("Gal(b1-4)GlcNAc") == "Gal(b1-4)GlcNAc"
    assert specify_linkages("Man(a1-6)Man") == "Man(a1-6)Man"


def test_process_occurrence():
    # Test single number
    assert process_occurrence("2") == [2, 2]
    # Test range
    assert process_occurrence("1,3") == [1, 3]
    # Test open-ended range
    assert process_occurrence(",3") == [0, 3]
    assert process_occurrence("2,") == [2, 5]


def test_expand_pattern():
    # Test pattern with ambiguous linkage
    assert expand_pattern("Galb3/4GlcNAc") == ["Galb3GlcNAc", "Galb4GlcNAc"]
    # Test pattern without ambiguous linkage
    assert expand_pattern("Galb4GlcNAc") == ["Galb4GlcNAc"]
    # Test multiple ambiguous linkages (if supported)
    pattern = "Galb3/4GlcNAca3/6"
    result = expand_pattern(pattern)
    assert len(result) == 2  # Should handle first ambiguity


def test_convert_pattern_component():
    # Test simple component
    result = convert_pattern_component("Hex")
    assert isinstance(result, str)
    # Test component with alternatives
    result = convert_pattern_component("[Hex|Fuc]{1}")
    assert isinstance(result, dict)
    assert all(isinstance(k, str) for k in result.keys())
    assert all(isinstance(v, list) for v in result.values())
    # Test component with quantifier
    result = convert_pattern_component("[Hex|Fuc]{1,2}")
    assert isinstance(result, dict)
    assert list(result.values())[0] == [1, 2]


def test_reformat_glycan_string():
    # Test basic reformatting
    assert reformat_glycan_string("Gal(b1-4)GlcNAc") == "Galb4-GlcNAc"
    # Test with branch
    input_glycan = "Gal(b1-4)GlcNAc(b1-2)[Fuc(a1-6)]Man"
    result = reformat_glycan_string(input_glycan)
    assert "([" in result and "]){1}" in result
    # Test with uncertain linkage
    assert reformat_glycan_string("Gal(b1-?)GlcNAc") == "Galb?-GlcNAc"


def test_get_match():
    # Test exact match
    pattern = "Hex-HexNAc"
    glycan = "Gal(b1-4)GlcNAc"
    assert get_match(pattern, glycan, return_matches=False) is True
    # Test no match
    pattern = "Hex-Hex-Hex"
    glycan = "Gal(b1-4)GlcNAc"
    assert get_match(pattern, glycan, return_matches=False) is False
    # Test pattern with quantifier
    pattern = "Hex-[HexNAc]{1,2}"
    glycan = "Gal(b1-4)GlcNAc(b1-4)GlcNAc"
    assert get_match(pattern, glycan, return_matches=False) is True


def test_motif_to_regex():
    # Test simple motif conversion
    motif = "Gal(b1-4)GlcNAc"
    pattern = motif_to_regex(motif)
    assert isinstance(pattern, str)
    assert get_match(pattern, motif, return_matches=False) is True
    # Test motif with branch
    motif = "Gal(b1-4)GlcNAc(b1-2)[Fuc(a1-6)]Man"
    pattern = motif_to_regex(motif)
    assert isinstance(pattern, str)
    assert get_match(pattern, motif, return_matches=False) is True


def test_process_question_mark():
    # Test basic lookahead
    result, pattern = process_question_mark("(?=Hex)", "Hex")
    assert result == [1]
    assert pattern == ["Hex"]
    # Test lookbehind
    result, pattern = process_question_mark("(?<=Hex)", "Hex")
    assert result == [1]
    assert pattern == ["Hex"]
    # Test simple question mark
    result, pattern = process_question_mark("Hex?", "Hex")
    assert result == [0, 1]
    assert pattern == ["Hex"]


def test_calculate_len_matches_comb():
    # Test single list of matches
    len_matches = [[2, 3]]
    result = calculate_len_matches_comb(len_matches)
    assert 0 in result
    assert 2 in result
    assert 3 in result
    # Test multiple lists
    len_matches = [[2], [3]]
    result = calculate_len_matches_comb(len_matches)
    assert 2 in result
    assert 3 in result
    assert 5 in result  # Combined length


def test_process_main_branch():
    glycan = "Gal(b1-4)GlcNAc(b1-2)[GlcNAc(b1-4)]Man"
    glycan_parts = min_process_glycans(["Gal(b1-4)GlcNAc(b1-2)Man"])[0]
    result = process_main_branch(glycan, glycan_parts)
    assert len(result) == len(glycan_parts)
    assert all(isinstance(x, str) for x in result)


def test_check_negative_look():
    # Test negative lookahead
    matches = [[0, 2, 4]]
    pattern = "(?!Hex)"
    glycan = "Gal(b1-4)GlcNAc(b1-2)Man"
    result = check_negative_look(matches, pattern, glycan)
    assert isinstance(result, list)
    # Test negative lookbehind
    pattern = "(?<!Hex)"
    result = check_negative_look(matches, pattern, glycan)
    assert isinstance(result, list)


def test_filter_matches_by_location():
    # Create a simple test graph
    ggraph = glycan_to_nxGraph("Fuc(a1-2)Gal(b1-4)Glc")
    # Test start location
    matches = [[0, 1], [1, 2]]
    result = filter_matches_by_location(matches, ggraph, "start")
    assert len(result) == 1
    assert result[0] == [0, 1]
    # Test end location
    result = filter_matches_by_location(matches, ggraph, "end")
    assert len(result) == 0  # None of the matches end at the last node
    # Test internal location
    matches = [[1, 2]]
    result = filter_matches_by_location(matches, ggraph, "internal")
    assert len(result) == 1
    assert result[0] == [1, 2]


def test_parse_pattern():
    # Test exact occurrence
    pattern = "Hex{2}"
    min_occur, max_occur = parse_pattern(pattern)
    assert min_occur == 2
    assert max_occur == 2
    # Test range occurrence
    pattern = "Hex{1,3}"
    min_occur, max_occur = parse_pattern(pattern)
    assert min_occur == 1
    assert max_occur == 3
    # Test wildcard
    pattern = "Hex*"
    min_occur, max_occur = parse_pattern(pattern)
    assert min_occur == 0
    assert max_occur == 8
    # Test plus
    pattern = "Hex+"
    min_occur, max_occur = parse_pattern(pattern)
    assert min_occur == 1
    assert max_occur == 8
    # Test question mark
    pattern = "Hex?"
    min_occur, max_occur = parse_pattern(pattern)
    assert min_occur == 0
    assert max_occur == 1


def test_matches():
    # Test nested parentheses matching
    result = list(matches("(a(b)c)", "(", ")"))
    assert len(result) == 2
    assert result[0][2] == 1  # Nested level
    assert result[1][2] == 0  # Outer level
    # Test square brackets
    result = list(matches("[a[b]c]", "[", "]"))
    assert len(result) == 2
    assert result[0][2] == 1  # Nested level
    assert result[1][2] == 0  # Outer level
    # Test empty string
    result = list(matches("", "(", ")"))
    assert len(result) == 0


def test_process_bonds():
    # Test basic linkages
    assert process_bonds(["a1-2"]) == ["α 2"]
    assert process_bonds(["b1-4"]) == ["β 4"]
    # Test uncertain linkages
    assert process_bonds(["?1-?"]) == ["?"]
    assert process_bonds(["b1-2"]) == ["β 2"]
    assert process_bonds(["a1-?"]) == ["α"]
    # Test multiple bonds
    result = process_bonds([["a1-2", "b1-4"]])
    assert result == [["α 2", "β 4"]]


def test_split_monosaccharide_linkage():
    # Test basic splitting
    sugars, mods, bonds = split_monosaccharide_linkage(["GlcNAc", "b1-2", "Man"])
    assert sorted(sugars) == sorted(["Man", "GlcNAc"])
    assert all(m == "" for m in mods)
    assert bonds == ["b1-2"]
    # Test with modifications
    sugars, mods, bonds = split_monosaccharide_linkage(["Gal6S", "b1-4", "Glc"])
    assert sorted(sugars) == sorted(["Gal", "Glc"])
    assert "6S" in mods
    assert bonds == ["b1-4"]
    # Test multiple entries
    sugars, mods, bonds = split_monosaccharide_linkage([["GlcNAc", "b1-2", "Man"], ["Gal6S", "b1-4", "Glc"]])
    assert len(sugars) == 2
    assert all(isinstance(s, list) for s in sugars)


def test_scale_in_range():
    # Test basic scaling
    result = scale_in_range([1, 2, 3], 0, 1)
    assert result[0] == 0  # Min value
    assert result[-1] == 1  # Max value
    assert 0 < result[1] < 1  # Middle value
    # Test with negative numbers
    result = scale_in_range([-2, 0, 2], 0, 10)
    assert result[0] == 0
    assert result[1] == 5
    assert result[2] == 10
    # Test single value
    result = scale_in_range([5, 5, 5], 0, 1)
    assert all(x == 0 for x in result)  # All values map to min when input range is 0


def test_glycan_to_skeleton():
    # Test simple glycan
    result = glycan_to_skeleton("Gal(b1-4)GlcNAc")
    assert "0-1" in result
    # Test branched glycan
    result = glycan_to_skeleton("Fuc(a1-3)[Gal(b1-4)]GlcNAc(b1-2)Man")
    assert "[" in result and "]" in result
    # Test empty string
    result = glycan_to_skeleton("")
    assert result == ""


def test_process_per_residue():
    # Test linear glycan
    values = [1, 2, 3]
    main, side, branched = process_per_residue("Gal(b1-4)GlcNAc(b1-2)Man", values)
    assert len(main) == 3
    assert len(side) == 0
    assert len(branched) == 0
    # Test branched glycan
    values = [1, 2, 3, 4]
    main, side, branched = process_per_residue("Fuc(a1-3)[Gal(b1-4)]GlcNAc(b1-2)Man", values)
    assert len(main) > 0
    assert len(side) > 0
    assert isinstance(side[0], list)


def test_unique():
    # Test basic deduplication
    result = unique([1, 2, 2, 3, 3, 1])
    assert result == [1, 2, 3]
    # Test order preservation
    result = unique([3, 1, 2, 1, 2, 3])
    assert result == [3, 1, 2]
    # Test with strings
    result = unique(["a", "b", "a", "c"])
    assert result == ["a", "b", "c"]


try:
    import rdkit
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
def test_get_hit_atoms_and_bonds():
    from rdkit import Chem
    # Create a simple molecule
    mol = Chem.MolFromSmiles("CC(=O)O")
    atoms, bonds = get_hit_atoms_and_bonds(mol, "CC(=O)O")
    assert isinstance(atoms, list)
    assert isinstance(bonds, list)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
def test_add_colours_to_map():
    # Test color assignment
    cols = {}
    els = [1, 2, 3]
    add_colours_to_map(els, cols, 0, alpha=True)
    assert len(cols) == 3
    assert all(isinstance(v, list) for v in cols.values())


def test_preprocess_data():
    # Create test dataframe
    df = pd.DataFrame({
        'glycan': ['Gal(b1-4)GlcNAc', 'Man(a1-6)Man'],
        'sample1': [10, 20],
        'sample2': [15, 25],
        'sample3': [12, 22],
        'sample4': [18, 28]
    })
    group1 = ['sample1', 'sample2']
    group2 = ['sample3', 'sample4']
    # Test basic preprocessing
    df_trans, df_org, g1, g2 = preprocess_data(df, group1, group2, impute=False)
    assert isinstance(df_trans, pd.DataFrame)
    assert isinstance(df_org, pd.DataFrame)
    assert g1 == group1
    assert g2 == group2
    # Test with motifs
    df_trans, df_org, g1, g2 = preprocess_data(df, group1, group2, motifs=True)
    assert 'Terminal_LacNAc_type2' in df_trans.index
    assert 'Man' in df_trans.index


def test_get_pvals_motifs():
    # Create test data with multiple glycans and varied z-scores
    glycans = [
        'Gal(b1-4)GlcNAc',
        'Man(a1-6)Man',
        'Neu5Ac(a2-6)Gal(b1-4)GlcNAc',
        'Fuc(a1-3)GlcNAc',
        'GalNAc(b1-4)GlcNAc',
        'Man(a1-3)[Man(a1-6)]Man',
        'Gal(b1-4)GlcNAc(b1-2)Man',
        'Fuc(a1-2)Gal(b1-4)GlcNAc'
    ]
    # Create z-scores with clear separation for testing
    np.random.seed(42)  # For reproducibility
    n_samples = len(glycans) * 4  # Total number of rows needed
    n_high = n_samples // 2
    n_low = n_samples - n_high
    high_scores = np.random.normal(2.5, 0.5, n_high)  # Above threshold
    low_scores = np.random.normal(-2.5, 0.5, n_low)   # Below threshold
    # Create multiple samples per glycan
    df = pd.DataFrame({
        'glycan': glycans * 4,  # Replicate each glycan 4 times
        'target': np.concatenate([high_scores, low_scores])  # Mix high and low z-scores
    })
    # Test basic functionality with default parameters
    results = get_pvals_motifs(df, zscores=True)
    # Check results structure
    assert isinstance(results, pd.DataFrame)
    assert 'motif' in results.columns
    assert 'pval' in results.columns
    assert 'corr_pval' in results.columns
    assert 'effect_size' in results.columns
    # Should find some significant motifs
    assert len(results) > 0
    assert any(results['corr_pval'] < 0.05)
    # Effect sizes should be non-zero
    assert not all(results['effect_size'] == 0)
    # Test with different feature sets
    results_known = get_pvals_motifs(df, feature_set=['known'])
    assert len(results_known) > 0
    results_terminal = get_pvals_motifs(df, feature_set=['terminal1'])
    assert len(results_terminal) > 0
    # Test with different threshold
    results_thresh = get_pvals_motifs(df, thresh=1.0)
    assert isinstance(results_thresh, pd.DataFrame)
    assert len(results_thresh) > 0


def test_get_differential_expression():
    df = pd.DataFrame({
        'glycan': [
            'Gal(b1-4)GlcNAc',
            'Man(a1-6)Man',
            'Neu5Ac(a2-6)Gal(b1-4)GlcNAc',
            'Fuc(a1-3)GlcNAc'
        ],
        'sample1': [100, 200, 150, 300],
        'sample2': [110, 220, 140, 280],
        'sample3': [120, 210, 160, 290],  # Group 1
        'sample4': [500, 200, 140, 700],
        'sample5': [480, 205, 155, 680],
        'sample6': [520, 195, 160, 720]   # Group 2 - clearly different
    })
    group1 = ['sample1', 'sample2', 'sample3']
    group2 = ['sample4', 'sample5', 'sample6']
    # Test basic functionality
    results = get_differential_expression(df, group1, group2, impute=False)
    assert isinstance(results, pd.DataFrame)
    assert 'Glycan' in results.columns
    assert 'p-val' in results.columns
    assert 'Effect size' in results.columns
    assert 'corr p-val' in results.columns
    assert 'significant' in results.columns
    # At least some results should be significant
    assert any(results['significant'])
    # Effect sizes should be reasonable
    assert all(abs(x) > 0.5 for x in results['Effect size'])
    # Test with motifs
    results_motifs = get_differential_expression(df, group1, group2, motifs=True, impute=False)
    assert isinstance(results_motifs, pd.DataFrame)
    assert len(results_motifs) > 0
    assert any(results_motifs['significant'])  # Should find some significant motifs
    # Test with paired samples
    group1_paired = ['sample1', 'sample2']
    group2_paired = ['sample4', 'sample5']
    results_paired = get_differential_expression(df[['glycan'] + group1_paired + group2_paired],
                                              group1_paired, group2_paired,
                                              paired=True, impute=False)
    assert isinstance(results_paired, pd.DataFrame)


def test_get_glycanova():
    # Create test data with THREE groups
    df = pd.DataFrame({
        'glycan': ['Gal(b1-4)GlcNAc', 'Man(a1-6)Man', 'Fuc(a1-2)Gal', 'Neu5Ac(a2-6)Gal'],
        'sample1': [100, 200, 150, 300],
        'sample2': [150, 250, 120, 280],  # Group 1
        'sample3': [120, 220, 140, 290],
        'sample4': [180, 280, 200, 350],
        'sample5': [200, 300, 180, 330],  # Group 2
        'sample6': [190, 290, 190, 340],
        'sample7': [220, 310, 210, 360],
        'sample8': [240, 330, 230, 380],  # Group 3
        'sample9': [230, 320, 220, 370]
    })
    # Three groups with three samples each
    groups = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    # Test basic functionality
    results, posthoc = get_glycanova(df, groups, impute=False)
    # Check results structure
    assert isinstance(results, pd.DataFrame)
    assert 'Glycan' in results.columns
    assert 'F statistic' in results.columns
    assert 'p-val' in results.columns
    assert 'corr p-val' in results.columns
    assert 'significant' in results.columns
    assert 'Effect size' in results.columns
    # Check posthoc results for all pairwise comparisons
    assert isinstance(posthoc, dict)
    if len(posthoc) > 0:  # If any significant differences found
        for glycan, result in posthoc.items():
            assert isinstance(result, pd.DataFrame)
            # Should have pairwise comparisons between groups
            assert len(result) == 3  # Number of pairwise comparisons for 3 groups = 3
    # Test with motifs enabled
    results_motifs, posthoc_motifs = get_glycanova(df, groups, motifs=True, impute=False)
    assert isinstance(results_motifs, pd.DataFrame)
    assert len(results_motifs) > 0  # Should find some motifs
    # Test without posthoc
    results_no_posthoc, posthoc_empty = get_glycanova(df, groups, posthoc=False, impute=False)
    assert isinstance(results_no_posthoc, pd.DataFrame)
    assert posthoc_empty == {}  # Should be empty dict when posthoc=False


def test_select_grouping():
    # Create test data
    cohort_a = pd.DataFrame({
        'sample1': [10, 20],
        'sample2': [15, 25]
    })
    cohort_b = pd.DataFrame({
        'sample3': [12, 22],
        'sample4': [18, 28]
    })
    glycans = ['Gal(b1-4)GlcNAc', 'Man(a1-6)Man']
    p_values = [0.04, 0.02]
    # Test without grouped BH
    glycan_groups, pval_groups = select_grouping(cohort_b, cohort_a, glycans, p_values, grouped_BH=False)
    assert len(glycan_groups) == 1
    assert "group1" in glycan_groups
    assert len(glycan_groups["group1"]) == len(glycans)
    # Test with grouped BH
    glycan_groups, pval_groups = select_grouping(cohort_b, cohort_a, glycans, p_values, grouped_BH=True)
    assert isinstance(glycan_groups, dict)
    assert isinstance(pval_groups, dict)


def test_get_biodiversity():
    # Create more realistic glycan compositions based on biological patterns
    np.random.seed(42)  # For reproducibility
    # Helper function to generate realistic glycan proportions
    def generate_group_data(base_proportions, n_samples=3, noise_scale=0.15):
        samples = []
        for _ in range(n_samples):
            # Add multiplicative noise to preserve positivity
            noise = np.random.normal(1, noise_scale, len(base_proportions))
            noisy = base_proportions * noise
            # Normalize to sum to 100
            normalized = 100 * noisy / noisy.sum()
            samples.append(normalized)
        return np.array(samples)
    # Define realistic base proportions for two distinct glycan profiles
    # Based on patterns seen in real data where some glycans are dominant
    group1_base = np.array([35, 25, 15, 10, 8, 7])  # More even distribution
    group2_base = np.array([45, 30, 10, 8, 4, 3])   # More dominated by top glycans
    # Generate sample data
    group1_data = generate_group_data(group1_base, n_samples=3)
    group2_data = generate_group_data(group2_base, n_samples=3)
    # Create DataFrame with realistic glycan names
    df = pd.DataFrame({
        'glycan': [
            'Neu5Ac(a2-3)Gal(b1-3)GalNAc',
            'Neu5Ac(a2-3)Gal(b1-3)[Neu5Ac(a2-6)]GalNAc',
            'Gal(b1-4)GlcNAc(b1-6)[Gal(b1-3)]GalNAc',
            'Neu5Ac(a2-3)Gal(b1-3)[Gal(b1-4)GlcNAc(b1-6)]GalNAc',
            'Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-6)[Gal(b1-3)]GalNAc',
            'Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-6)[Neu5Ac(a2-3)Gal(b1-3)]GalNAc'
        ]
    })
    # Add sample columns
    for i in range(3):
        df[f'sample1_{i+1}'] = group1_data[i]
        df[f'sample2_{i+1}'] = group2_data[i]
    # Define groups for testing
    group1 = [f'sample1_{i+1}' for i in range(3)]
    group2 = [f'sample2_{i+1}' for i in range(3)]
    # Run biodiversity analysis
    results = get_biodiversity(df, group1, group2, metrics=['alpha', 'beta'])
    # Basic assertions
    assert isinstance(results, pd.DataFrame), "Results should be a DataFrame"
    assert 'Metric' in results.columns, "Results should have a Metric column"
    assert 'p-val' in results.columns, "Results should have a p-val column"
    # Additional assertions to verify realistic results
    assert len(results) >= 2, "Should have at least alpha and beta diversity results"
    assert all(0 <= p <= 1 for p in results['p-val']), "p-values should be between 0 and 1"
    # Optional: Check if the differences are detectable
    # The groups are designed to be different, so p-values should be < 0.05
    assert any(p < 0.05 for p in results['p-val']), "Should detect differences between groups"


def test_get_time_series():
    df = pd.DataFrame({
        'glycan': ['Gal(b1-4)GlcNAc', 'Man(a1-6)Man', 'Neu5Ac(a2-6)Gal'],
        'T1_h0_r1': [100, 200, 150],
        'T1_h0_r2': [110, 220, 140],
        'T1_h4_r1': [120, 210, 160],
        'T1_h4_r2': [500, 600, 450],
        'T1_h8_r1': [480, 580, 470],
        'T1_h8_r2': [520, 620, 430]
    })
    results = get_time_series(df, impute=False)
    assert isinstance(results, pd.DataFrame)
    assert 'Glycan' in results.columns
    assert 'p-val' in results.columns


def test_get_SparCC():
    df1 = pd.DataFrame({
        'glycan': ['Gal(b1-4)GlcNAc', 'Man(a1-6)Man', 'Neu5Ac(a2-6)Gal'],
        'sample1': [100, 200, 150],
        'sample2': [110, 220, 140],
        'sample3': [120, 210, 160]
    })
    df2 = pd.DataFrame({
        'glycan': ['Gal(b1-4)GlcNAc', 'Man(a1-6)Man', 'Neu5Ac(a2-6)Gal'],
        'sample1': [500, 600, 450],
        'sample2': [480, 580, 470],
        'sample3': [520, 620, 430]
    })
    corr, pvals = get_SparCC(df1, df2)
    assert isinstance(corr, pd.DataFrame)
    assert isinstance(pvals, pd.DataFrame)


def test_get_roc():
    df = pd.DataFrame({
        'glycan': ['Gal(b1-4)GlcNAc', 'Man(a1-6)Man', 'Neu5Ac(a2-6)Gal', 'Fuc(a1-3)GlcNAc'],
        'sample1': [100, 200, 150, 300],
        'sample2': [110, 220, 140, 280],
        'sample3': [120, 210, 160, 290],
        'sample4': [500, 600, 450, 700],
        'sample5': [480, 580, 470, 680],
        'sample6': [520, 620, 430, 720]
    })
    group1 = ['sample1', 'sample2', 'sample3']
    group2 = ['sample4', 'sample5', 'sample6']
    results = get_roc(df, group1, group2)
    assert isinstance(results, list)
    assert all(isinstance(x, tuple) for x in results)


def test_get_representative_substructures():
    # Create test data with both monosaccharides and disaccharides
    enrichment_df = pd.DataFrame({
        'motif': [
            'Gal',  # Monosaccharides
            'GlcNAc',
            'Neu5Ac',
            'Gal(b1-3)GalNAc',  # Disaccharides
            'Gal(b1-4)GlcNAc',
            'Neu5Ac(a2-6)Gal'
        ],
        'pval': [0.01, 0.015, 0.02, 0.03, 0.04, 0.045],
        'corr_pval': [0.02, 0.025, 0.03, 0.04, 0.045, 0.049],
        'effect_size': [2.0, 1.8, 1.5, 1.3, 1.2, 1.1]
    })
    results = get_representative_substructures(enrichment_df)
    assert isinstance(results, list)
    assert all(isinstance(x, str) for x in results)
    assert len(results) <= 10  # Function should return up to 10 structures


def test_get_lectin_array():
    df = pd.DataFrame({
        'sample_id': ['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6'],
        'ConA': [100, 110, 120, 500, 480, 520],
        'WGA': [200, 220, 210, 600, 580, 620],
        'SNA': [150, 140, 160, 450, 470, 430]
    })
    group1 = ['sample1', 'sample2', 'sample3']
    group2 = ['sample4', 'sample5', 'sample6']
    results = get_lectin_array(df, group1, group2)
    assert isinstance(results, pd.DataFrame)
    assert 'score' in results.columns


@pytest.fixture
def simple_glycans():
    return [
        "Gal(b1-4)Glc-ol",
        "Gal(b1-4)GlcNAc-ol",
        "Neu5Ac(a2-3)[Neu5Ac(a2-6)]Gal(b1-4)Glc-ol",
        "Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)Glc-ol"
    ]


@pytest.fixture
def simple_network():
    # Create a simple directed network
    G = nx.DiGraph()
    G.add_edge("Gal(b1-4)Glc-ol", "GlcNAc(b1-3)Gal(b1-4)Glc-ol", diffs="GlcNAc(b1-3)")
    G.add_edge("GlcNAc(b1-3)Gal(b1-4)GlcNAc-ol", "Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)Glc-ol", diffs="Gal(b1-4)")
    nx.set_node_attributes(G, {
        "Gal(b1-4)Glc-ol": {"virtual": 0},
        "Gal(b1-4)GlcNAc-ol": {"virtual": 0},
        "Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)Glc-ol": {"virtual": 0}
    })
    return G


def test_safe_compare():
    g1 = glycan_to_nxGraph("Gal(b1-4)Glc-ol")
    g2 = glycan_to_nxGraph("Gal(b1-4)Glc-ol")
    g3 = glycan_to_nxGraph("Gal(b1-4)GlcNAc-ol")
    assert safe_compare(g1, g2) == True
    assert safe_compare(g1, g3) == False
    assert safe_compare(g1, None) == False


def test_safe_index():
    glycan = "Gal(b1-4)Glc-ol"
    graph_dic = {}
    result = safe_index(glycan, graph_dic)
    assert isinstance(result, nx.Graph)
    assert glycan in graph_dic
    assert graph_dic[glycan] == result


def test_get_neighbors(simple_glycans):
    ggraph = glycan_to_nxGraph("GlcNAc(b1-3)Gal(b1-4)Glc-ol")
    graphs = [glycan_to_nxGraph(g) for g in simple_glycans]
    neighbors, indices = get_neighbors(ggraph, simple_glycans, graphs)
    assert len(neighbors) > 0
    assert all(isinstance(n, str) for n in neighbors)
    assert all(isinstance(idx, list) for idx in indices)


def test_create_neighbors():
    glycan = "Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)Glc-ol"
    ggraph = glycan_to_nxGraph(glycan)
    neighbors = create_neighbors(ggraph)
    assert len(neighbors) > 0
    assert all(isinstance(n, nx.Graph) for n in neighbors)
    assert len(neighbors[0].nodes()) < len(ggraph.nodes())


def test_find_diff():
    graph_dic = {}
    glycan_a = "Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)Glc-ol"
    glycan_b = "Gal(b1-4)Glc-ol"
    diff = find_diff(glycan_a, glycan_b, graph_dic)
    assert isinstance(diff, str)
    assert "GlcNAc" in diff
    assert find_diff(glycan_a, glycan_a, graph_dic) == ""


def test_find_path():
    graph_dic = {}
    glycan_a = "Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)Glc-ol"
    glycan_b = "Gal(b1-4)Glc-ol"
    edges, labels = find_path(glycan_a, glycan_b, graph_dic)
    assert isinstance(edges, list)
    assert isinstance(labels, dict)
    assert len(edges) > 0
    assert all(isinstance(e, tuple) for e in edges)


def test_construct_network(simple_glycans):
    network = construct_network(simple_glycans)
    assert isinstance(network, nx.DiGraph)
    assert len(network.nodes()) >= len(simple_glycans)
    assert len(network.edges()) > 0
    assert all("diffs" in data for _, _, data in network.edges(data=True))


def test_prune_network(simple_glycans):
    net = construct_network(simple_glycans)
    nx.set_node_attributes(net, {
        "Gal(b1-4)Glc-ol": {"abundance": 0.5},
        "Neu5Ac(a2-6)Gal(b1-4)Glc-ol": {"abundance": 0.0},
        "Neu5Ac(a2-3)Gal(b1-4)Glc-ol": {"abundance": 0.5},
        "Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)Glc-ol": {"abundance": 0.8}
    })
    pruned = prune_network(net, threshold=0.1)
    assert isinstance(pruned, nx.DiGraph)
    assert len(pruned.nodes()) < len(net.nodes())
    assert "Neu5Ac(a2-6)Gal(b1-4)Glc-ol" not in pruned.nodes()


def test_extend_glycans():
    glycans = {"Gal(b1-4)Glc-ol"}
    reactions = {"Fuc(a1-3)", "GlcNAc(b1-3)"}
    new_glycans = extend_glycans(glycans, reactions)
    assert isinstance(new_glycans, set)
    assert len(new_glycans) > 0
    assert all(isinstance(g, str) for g in new_glycans)
    assert all(g.endswith('Glc-ol') for g in new_glycans)


def test_highlight_network_motif(simple_network):
    motif = "Gal(b1-4)Glc"
    highlighted = highlight_network(simple_network, highlight="motif", motif=motif)
    assert isinstance(highlighted, nx.DiGraph)
    assert all("origin" in data for _, data in highlighted.nodes(data=True))
    assert all(data["origin"] in ["limegreen", "darkviolet"]
              for _, data in highlighted.nodes(data=True))


def test_highlight_network_abundance(simple_network):
    abundance_df = pd.DataFrame({
        "glycan": list(simple_network.nodes()),
        "rel_intensity": [0.5] * len(simple_network)
    })
    highlighted = highlight_network(simple_network, highlight="abundance",
                                  abundance_df=abundance_df)
    assert isinstance(highlighted, nx.DiGraph)
    assert all("abundance" in data for _, data in highlighted.nodes(data=True))
    assert all(isinstance(data["abundance"], (int, float))
              for _, data in highlighted.nodes(data=True))


@pytest.fixture
def sample_network():
    G = nx.DiGraph()
    # Create a diamond-like structure
    G.add_edge("Gal(b1-4)Glc-ol", "Neu5Ac(a2-3)Gal(b1-4)Glc-ol", diffs="Neu5Ac(a2-3)")
    G.add_edge("Gal(b1-4)Glc-ol", "Neu5Ac(a2-6)Gal(b1-4)Glc-ol", diffs="Neu5Ac(a2-6)")
    G.add_edge("Neu5Ac(a2-3)Gal(b1-4)Glc-ol", "Neu5Ac(a2-3)[Neu5Ac(a2-6)]Gal(b1-4)Glc-ol", diffs="Neu5Ac(a2-6)")
    G.add_edge("Neu5Ac(a2-6)Gal(b1-4)Glc-ol", "Neu5Ac(a2-3)[Neu5Ac(a2-6)]Gal(b1-4)Glc-ol", diffs="Neu5Ac(a2-3)")
    G.add_edge("Gal(b1-4)Glc-ol", "Fuc(a1-2)Gal(b1-4)Glc-ol", diffs="Fuc(a1-2)")
    G.add_edge("Fuc(a1-2)Gal(b1-4)Glc-ol", "Fuc(a1-2)Gal(b1-4)[Fuc(a1-3)]Glc-ol", diffs="Fuc(a1-3)")
    # Set node attributes
    nx.set_node_attributes(G, {
        "Gal(b1-4)Glc-ol": {"virtual": 0, "abundance": 10.0},
        "Neu5Ac(a2-3)Gal(b1-4)Glc-ol": {"virtual": 0, "abundance": 5.0},
        "Neu5Ac(a2-6)Gal(b1-4)Glc-ol": {"virtual": 1, "abundance": 3.0},
        "Neu5Ac(a2-3)[Neu5Ac(a2-6)]Gal(b1-4)Glc-ol": {"virtual": 0, "abundance": 7.0},
        "Fuc(a1-2)Gal(b1-4)Glc-ol": {"virtual": 0, "abundance": 1.0},
        "Fuc(a1-2)Gal(b1-4)[Fuc(a1-3)]Glc-ol": {"virtual": 0, "abundance": 0.5}
    })
    return G


@pytest.fixture
def abundance_data():
    return pd.DataFrame({
        'glycan': ["Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)Glc-ol",
                  "Fuc(a1-2)Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)Glc-ol"],
        'sample1': [10.0, 5.0, 7.0],
        'sample2': [8.0, 6.0, 8.0],
        'sample3': [9.0, 4.0, 6.0],
        'sample4': [11.0, 7.0, 9.0]
    })


def test_infer_roots():
    # Test O-linked glycans
    o_glycans = {"GalNAc(b1-3)Gal", "Fuc(a1-2)GalNAc"}
    roots = infer_roots(frozenset(o_glycans))
    assert "GalNAc" in roots
    assert "Fuc" in roots
    # Test free glycans
    free_glycans = {"Gal(b1-4)Glc-ol", "Gal(b1-4)GlcNAc-ol"}
    roots = infer_roots(frozenset(free_glycans))
    assert "Gal(b1-4)Glc-ol" in roots
    assert "Gal(b1-4)GlcNAc-ol" in roots


def test_deorphanize_nodes(sample_network):
    # Add an orphaned node
    sample_network.add_node("GlcNAc(b1-3)Gal(b1-4)Glc-ol", virtual=0)
    deorphanized = deorphanize_nodes(sample_network, {})
    assert isinstance(deorphanized, nx.DiGraph)
    assert len(deorphanized.edges()) > len(sample_network.edges())
    # Check if orphaned node is now connected
    assert deorphanized.degree("GlcNAc(b1-3)Gal(b1-4)Glc-ol") > 0


def test_get_edge_weight_by_abundance(sample_network):
    weighted = get_edge_weight_by_abundance(sample_network)
    assert isinstance(weighted, nx.DiGraph)
    assert all("capacity" in data for _, _, data in weighted.edges(data=True))
    assert all(isinstance(data["capacity"], (int, float))
              for _, _, data in weighted.edges(data=True))


def test_find_diamonds(simple_glycans):
    diamonds = find_diamonds(construct_network(simple_glycans))
    assert isinstance(diamonds, list)
    assert len(diamonds) > 0
    assert all(isinstance(d, dict) for d in diamonds)
    assert all(len(d) >= 4 for d in diamonds)  # Diamond should have at least 4 nodes


def test_trace_diamonds(simple_glycans):
    net = construct_network(simple_glycans)
    species_list = ["species1"]
    network_dic = {"species1": net}
    results = trace_diamonds(net, species_list, network_dic)
    assert isinstance(results, pd.DataFrame)
    assert "probability" in results.columns
    assert len(results) > 0


def test_get_maximum_flow(sample_network):
    sample_network = estimate_weights(sample_network, root = "Gal(b1-4)Glc-ol", min_default = 0.1)
    flow_results = get_maximum_flow(sample_network,
                                  source="Gal(b1-4)Glc-ol")
    assert isinstance(flow_results, dict)
    assert all("flow_value" in data for data in flow_results.values())
    assert all("flow_dict" in data for data in flow_results.values())
    assert all(isinstance(data["flow_value"], (int, float))
              for data in flow_results.values())


def test_get_reaction_flow(sample_network):
    sample_network = estimate_weights(sample_network, root = "Gal(b1-4)Glc-ol", min_default = 0.1)
    flow_results = get_maximum_flow(sample_network,
                                  source="Gal(b1-4)Glc-ol")
    reaction_flows = get_reaction_flow(sample_network, flow_results,
                                     aggregate="sum")
    assert isinstance(reaction_flows, dict)
    assert len(reaction_flows) > 0
    assert all(isinstance(v, (int, float)) for v in reaction_flows.values())


def test_process_ptm():
    glycans = ["Gal(b1-4)Glc3S-ol", "Gal(b1-4)Glc-ol"]
    stem_lib = get_stem_lib(get_lib(glycans))
    edges, labels = process_ptm(glycans, {}, stem_lib)
    if edges:  # If PTMs were found
        assert len(edges) == len(labels)
        assert all(isinstance(e, tuple) for e in edges)
        assert all(isinstance(l, str) for l in labels)


def test_get_differential_biosynthesis(abundance_data):
    results = get_differential_biosynthesis(
        abundance_data,
        group1=['sample1', 'sample2'],
        group2=['sample3', 'sample4'],
        analysis="reaction"
    )
    assert isinstance(results, pd.DataFrame)
    assert "Log2FC" in results.columns
    assert "p-val" in results.columns
    assert "significant" in results.columns
    assert len(results) > 0


def test_deorphanize_edge_labels(sample_network):
    # Remove some edge labels
    edge_list = list(sample_network.edges())[0:2]
    for edge in edge_list:
        del sample_network.edges[edge]["diffs"]
    labeled = deorphanize_edge_labels(sample_network, {})
    assert isinstance(labeled, nx.DiGraph)
    assert all("diffs" in data for _, _, data in labeled.edges(data=True))


def test_get_differential_biosynthesis_longitudinal(abundance_data):
    # Add ID column for longitudinal analysis
    abundance_data = abundance_data.copy().set_index('glycan').T
    abundance_data['ID'] = ['p1_t1_1', 'p1_t2_1', 'p2_t1_1', 'p2_t2_1']
    results = get_differential_biosynthesis(
        abundance_data,
        group1=['t1', 't2'],
        longitudinal=True,
        id_column='ID'
    )
    assert isinstance(results, pd.DataFrame)
    assert "F-statistic" in results.columns
    assert "Direction" in results.columns
    assert "Average Slope" in results.columns
    assert len(results) > 0


def test_process_ptm_with_multiple_ptms():
    glycans = ["Gal(b1-4)Glc3S6S-ol", "Gal(b1-4)Glc-ol", "Gal(b1-4)Glc6S-ol"]
    stem_lib = get_stem_lib(get_lib(glycans))
    edges, labels = process_ptm(glycans, {}, stem_lib)
    if edges:
        assert len(edges) > 1  # Should find multiple PTM edges
        assert all(isinstance(e, tuple) for e in edges)
        assert all(isinstance(l, str) for l in labels)


def test_get_maximum_flow_with_no_path(sample_network):
    # Add isolated node with no path from source
    sample_network.add_node("IsolatedGlycan", virtual=0, abundance=1.0)
    sample_network = estimate_weights(sample_network, root = "Gal(b1-4)Glc-ol", min_default = 0.1)
    flow_results = get_maximum_flow(sample_network,
                                  source="Gal(b1-4)Glc-ol",
                                  sinks=["IsolatedGlycan"])
    assert isinstance(flow_results, dict)
    assert len(flow_results) == 0  # Should return empty dict for unreachable sink
