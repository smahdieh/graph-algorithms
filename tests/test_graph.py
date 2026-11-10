"""Unit tests for the Graph class and the DIMACS loader."""

import pytest

from graph_algorithms.graph import Graph, load_dimacs, parse_dimacs


def test_empty_graph():
    """A graph with no vertices has no edges and maximum degree 0."""
    graph = Graph(0)
    assert graph.num_vertices == 0
    assert graph.num_edges == 0
    assert list(graph.edges()) == []
    assert graph.max_degree() == 0


def test_add_edge_is_undirected():
    """An added edge is visible from both endpoints."""
    graph = Graph(3)
    graph.add_edge(0, 2)
    assert graph.has_edge(0, 2) and graph.has_edge(2, 0)
    assert graph.neighbors(0) == {2}
    assert graph.neighbors(2) == {0}
    assert graph.degree(1) == 0


def test_duplicate_edge_counted_once():
    """Adding the same edge twice (either direction) counts it once."""
    graph = Graph(2)
    graph.add_edge(0, 1)
    graph.add_edge(1, 0)
    assert graph.num_edges == 1


def test_self_loop_rejected():
    """An edge from a vertex to itself raises ValueError."""
    with pytest.raises(ValueError):
        Graph(2).add_edge(1, 1)


@pytest.mark.parametrize("u, v", [(-1, 0), (0, 3)])
def test_out_of_range_vertex_rejected(u, v):
    """A vertex outside 0 .. n-1 raises ValueError."""
    with pytest.raises(ValueError):
        Graph(3).add_edge(u, v)


def test_negative_size_rejected():
    """A negative number of vertices raises ValueError."""
    with pytest.raises(ValueError):
        Graph(-1)


def test_from_edges_and_edges_listed_once():
    """from_edges builds the graph, and edges() lists each edge once as (u, v), u < v."""
    graph = Graph.from_edges(4, [(0, 1), (2, 1), (3, 0)])
    assert sorted(graph.edges()) == [(0, 1), (0, 3), (1, 2)]
    assert graph.max_degree() == 2


def test_parse_dimacs_renumbers_from_zero():
    """DIMACS vertices 1 .. N become 0 .. N-1, and isolated vertices are kept."""
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
    """Files that use the 'p col' header are read too."""
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
    """Missing headers, short lines and unknown line types raise ValueError."""
    with pytest.raises(ValueError):
        parse_dimacs(lines)


def test_load_dimacs_from_file(tmp_path):
    """load_dimacs reads a DIMACS file from disk."""
    path = tmp_path / "path3.clq"
    path.write_text("p edge 3 2\ne 1 2\ne 2 3\n")
    graph = load_dimacs(path)
    assert sorted(graph.edges()) == [(0, 1), (1, 2)]
