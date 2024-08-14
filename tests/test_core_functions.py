import pytest
import networkx as nx
import networkx.algorithms.isomorphism as iso

from glycowork.motif.graph import glycan_to_nxGraph, graph_to_string_int, graph_to_string


@pytest.mark.parametrize("glycan", [
    "Gal(b1-4)GlcNAc(b1-6)GlcOMe",
    "3,6-Anhydro-Gal(a1-3)Gal4S(b1-4)3,6-Anhydro-Gal2S(a1-3)Gal4S(b1-4)3,6-Anhydro-Gal(a1-3)Gal4S",
    "GlcNAc(a1-2)Rha(a1-2)Rha(a1-3)Rha(a1-3)GlcNAc",
    "Gal(?1-?)Gal3S(?1-?)GlcNAc(?1-?)Fuc",
    "Tyv1PP(a1-5)Ribf1N",
    "GlcNAc(a1-4)Gal(a1-4)Gal(b1-3)GlcNAc(b1-2)D-FucNAcOAc(b1-3)GlcNAc",
    "RhaNAc3NAc(?1-?)D-Ara-ol",
    "Gal(b1-3)GlcNAc1Me",
    "Xyl(a1-3)Xyl(b1-4)Xyl",
    "Fuc(a1-?)Gal(b1-4)Glc",
    "Neu5Gc(a2-8)Neu5Ac(a2-6)[GalNAc(a1-3)]GalNAc",
    "Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)[Neu5Ac(a2-6)]GalNAc",
    "Neu5Ac(a2-3)Gal(b1-3)[Neu5Ac(a2-6)]GlcNAc(b1-3)Gal",
    "Glc(b1-2)[Xyl(b1-3)]Gal",
    "GalNAc6Ac(a1-4)[Glc(b1-3)]GalNAc(a1-3)GlcNAc(a1-4)GalNAc6Ac",
    "Fruf(b2-1)[Gal6ulo(a1-6)]Glc",
    "Gal(b1-3)[Neu(a2-6)]GalNAc",
    "Xyl(b1-4)Xyl(b1-4)Xyl(b1-4)Xyl3Ac(b1-4)Xyl(b1-4)Xyl(b1-4)[GlcA(a1-2)]Xyl(b1-4)Xyl",
    "HexNAc(?1-4)HexNAc(?1-?)HexA(?1-?)[Fuc(a1-?)]GalNAc",
    "Man(a1-6)Man(a1-5)[Gal(b1-8)Kdo(a2-4)]Kdo(a2-6)GlcN4P(b1-6)GlcN1P",
    "Neu5Ac(a2-?)Gal(?1-?)[Neu5Ac(a2-?)]GlcNAc(?1-?)[Neu5Ac(a2-?)Gal(b1-?)GlcNAc(b1-?)]Man(a1-?)[Neu5Ac(a2-?)Gal(?1-?)[Neu5Ac(a2-?)]GlcNAc(?1-?)Man(?1-?)]Man(?1-?)GalNAc(?1-?)GlcNAc",
    "Gal(b1-?)GlcNAc(b1-?)Man(a1-?)[Gal(b1-?)GlcNAc(b1-?)Man(a1-?)][GlcNAc(b1-?)]Man(a1-?)GlcNAc",
    "Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)Man3Me(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "Gal(b1-4)[Fuc(a1-3)]GlcNAc(b1-2)[GlcNAc(b1-4)]Man(a1-3)[Fuc(a1-3)[Gal(b1-4)]GlcNAc(b1-2)[Fuc(a1-3)[Gal(b1-4)]GlcNAc(b1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "GalNAc(b1-4)GlcNAc(b1-2)Man(a1-?)[GlcNAc(b1-2)Man(a1-?)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]Man-ol",
    "Man(b1-4)Man(b1-4)[Gal(a1-6)]Man(b1-4)[Gal(a1-6)]Man(b1-4)[Gal(a1-6)]Man(b1-4)Man(b1-4)Man(b1-4)Man(b1-4)Man(b1-4)Man",
    "Gal(b1-?)GlcNAc(b1-?)Man(a1-?)[Gal(b1-?)GlcNAc(b1-?)Man(a1-?)][GlcNAc(b1-?)]Man(a1-?)GlcNAc(?1-?)[D-Fuc(a1-?)]GlcNAc",
    "Qui(b1-2)[Oli3NAc(b1-3)][Oli3NAc(b1-3)Qui(b1-4)]Qui",
    "Neu5Ac(a2-?)Gal6S(?1-?)[Fuc(a1-?)]GlcNAc(?1-?)[Neu5Ac(a2-?)Gal(?1-?)]GalNAc",
    "Neu5Gc(a2-3)Gal(b1-4)GlcNAc(b1-2)[Gal(b1-4)GlcNAc(b1-4)]Man(a1-3)[Kdn(a2-3)Gal(b1-4)GlcNAc(b1-2)[Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "[Fruf(b2-2)][Fruf(b2-3)]GalOPGro(a1-4)GalOPGro",
    "Gal(b1-?)GlcNAc(b1-2)Man(a1-3)[Man(a1-3)[Man(a1-6)]Man(a1-6)][GlcNAc(b1-4)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "Neu5Ac(a2-?)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[GlcNAc(b1-4)][Neu5Ac(a2-?)Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "Neu5Gc(a2-?)Gal(b1-4)[Fuc(a1-3)]GlcNAc(b1-2)[Neu5Gc(a2-?)Gal(b1-4)[Fuc(a1-3)]GlcNAc(b1-4)]Man(a1-3)[GlcNAc(b1-4)][Neu5Gc(a2-?)Gal(b1-4)[Fuc(a1-3)]GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-2)[Neu5Ac(a2-?)Gal(b1-4)GlcNAc(b1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "GlcNAc(?1-?)Man(a1-3)[GlcNAc(b1-4)][Fuc(a1-?)[Gal(b1-?)]GlcNAc(b1-?)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "Neu5Gc(a2-?)Gal(b1-?)[Fuc(a1-?)]GlcNAc(b1-?)Gal(b1-?)GlcNAc(b1-?)[Fuc(a1-?)[Gal(b1-?)]GlcNAc(b1-?)]Man(a1-?)[Fuc(a1-?)[Gal(b1-?)]GlcNAc(b1-?)[Fuc(a1-?)[Gal(b1-?)]GlcNAc(b1-?)]Man(a1-?)][GlcNAc(b1-4)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-?)]GlcNAc",
    "Fuc(a1-2)[Gal(a1-3)][Fuc(a1-6)]Gal(a1-2)Gal(a1-3)Gal(b1-6)Man(a1-?)Ins",
    "Man(a1-?)[Man(a1-?)]Man(a1-?)[GlcNAc(b1-?)Man(a1-?)][GlcNAc(b1-?)]Man(a1-?)GlcNAc",
    "Neu5Ac(a2-?)Gal(b1-?)[Fuc(a1-?)]GlcNAc(b1-?)[Fuc(a1-?)[Gal(b1-?)]GlcNAc(b1-?)]Man(a1-?)[Neu5Ac(a2-?)Gal(b1-?)[Fuc(a1-?)]GlcNAc(b1-?)[Fuc(a1-?)[Gal(b1-?)]GlcNAc(b1-?)]Man(a1-?)][GlcNAc(b1-4)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "Kdo(?2-?)[Glc(b1-3)[GlcNAc(a1-2)Glc(b1-4)][Glc(a1-2)Glc(b1-6)]Glc(a1-5)]Kdo",
    "{Man(a1-?)}Man(a1-?)Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "{Gal(b1-?)GlcNAc(b1-?)}Gal(b1-?)GlcNAc(b1-2)Man(a1-3)[Gal(b1-?)GlcNAc(b1-2)Man(a1-6)][GlcNAc(b1-4)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "{Fuc(a1-?)}{Neu5Ac(a2-?)}Neu5Ac(a2-3)[GalNAc(b1-4)]Gal(b1-4)GlcNAc(b1-3)Gal(b1-3)[GlcNAc(b1-6)]GalNAc",
    "{Man(a1-2)}{Man(a1-2)}Man(a1-3)[Man(a1-6)]Man(a1-6)[Man(a1-3)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "{Gal(b1-?)GlcNAc(b1-3)}{Gal(b1-?)GlcNAc(b1-3)}{Gal(b1-?)GlcNAc(b1-3)}Gal(b1-?)Gal(b1-4)GlcNAc(b1-2)[Gal(b1-?)Gal(b1-4)GlcNAc(b1-?)]Man(a1-?)[Gal(b1-4)GlcNAc(b1-2)[Gal(b1-4)GlcNAc(b1-?)]Man(a1-?)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "{Fuc(a1-3)}{Fuc(a1-3)}{Neu5Ac(a2-?)}Gal(b1-4)GlcNAc(b1-2)[Gal(b1-4)GlcNAc(b1-?)]Man(a1-?)[Gal(b1-4)GlcNAc(b1-2)[Gal(b1-4)GlcNAc(b1-?)]Man(a1-?)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "{Neu5Ac(a2-?)}Gal(b1-3)[GlcNAc(b1-6)]GalNAc",
    "{Gal(b1-4)GlcNAc(b1-?)}{Gal(b1-4)GlcNAc(b1-?)}{Neu5Ac(a2-?)}{Neu5Ac(a2-?)}Gal(b1-4)GlcNAc(b1-2)[Gal(b1-4)GlcNAc(b1-4)]Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)[Gal(b1-4)GlcNAc(b1-6)]Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
    "{Fuc(a1-?)}{Fuc(a1-?)}Gal(b1-4)GlcNAc(b1-2)Man(a1-?)[Gal(b1-4)GlcNAc(b1-2)[Gal(b1-4)GlcNAc(b1-?)]Man(a1-?)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc",
    "{Gal(b1-4)GlcNAc(b1-?)}Gal(b1-4)GlcNAc(b1-?)Gal(b1-4)Glc-ol",
])
def test_graph_to_string_int(glycan: str):
    """This test assumes that the function gylcan_to_graph_int is correct."""
    label = glycan_to_nxGraph(glycan)
    target = glycan_to_nxGraph(graph_to_string(label))

    assert nx.is_isomorphic(label, target, node_match=iso.categorical_node_match("string_labels", ""))
