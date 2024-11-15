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
    choose_correct_isoform, bracket_removal
)
from glycowork.glycan_data.loader import (Hex, linkages)

from glycowork.motif.graph import glycan_to_nxGraph, graph_to_string


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
