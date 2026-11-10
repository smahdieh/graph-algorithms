"""Unit tests for the Graph class and the DIMACS loader."""

import pytest

from graph_algorithms.graph import Graph, load_dimacs, parse_dimacs


def test_empty_graph():
    graph = Graph(0)
    assert graph.num_vertices == 0
    assert graph.num_edges == 0
    assert list(graph.edges()) == []
    assert graph.max_degree() == 0


def test_add_edge_is_undirected():
    graph = Graph(3)
    graph.add_edge(0, 2)
    assert graph.has_edge(0, 2) and graph.has_edge(2, 0)
    assert graph.neighbors(0) == {2}
    assert graph.neighbors(2) == {0}
    assert graph.degree(1) == 0


def test_duplicate_edge_counted_once():
    graph = Graph(2)
    graph.add_edge(0, 1)
    graph.add_edge(1, 0)
    assert graph.num_edges == 1


def test_self_loop_rejected():
    with pytest.raises(ValueError):
        Graph(2).add_edge(1, 1)


@pytest.mark.parametrize("u, v", [(-1, 0), (0, 3)])
def test_out_of_range_vertex_rejected(u, v):
    with pytest.raises(ValueError):
        Graph(3).add_edge(u, v)


def test_negative_size_rejected():
    with pytest.raises(ValueError):
        Graph(-1)


def test_from_edges_and_edges_listed_once():
    graph = Graph.from_edges(4, [(0, 1), (2, 1), (3, 0)])
    assert sorted(graph.edges()) == [(0, 1), (0, 3), (1, 2)]
    assert graph.max_degree() == 2


def test_parse_dimacs_renumbers_from_zero():
    text = """c a tiny triangle plus an isolated vertex
p edge 4 3
e 1 2
e 2 3
e 1 3
"""
    graph = parse_dimacs(text.splitlines())
    assert graph.num_vertices == 4
    assert sorted(graph.edges()) == [(0, 1), (0, 2), (1, 2)]
    assert graph.degree(3) == 0


def test_parse_dimacs_accepts_col_header():
    graph = parse_dimacs(["p col 2 1", "e 1 2"])
    assert graph.num_edges == 1


@pytest.mark.parametrize(
    "lines",
    [
        [],                      # no header
        ["e 1 2"],               # edge before header
        ["p edge"],              # header too short
        ["p edge 2 1", "e 1"],   # edge too short
        ["p edge 2 1", "x 1 2"], # unknown line type
    ],
)
def test_parse_dimacs_rejects_bad_input(lines):
    with pytest.raises(ValueError):
        parse_dimacs(lines)


def test_load_dimacs_from_file(tmp_path):
    path = tmp_path / "path3.clq"
    path.write_text("p edge 3 2\ne 1 2\ne 2 3\n")
    graph = load_dimacs(path)
    assert sorted(graph.edges()) == [(0, 1), (1, 2)]
