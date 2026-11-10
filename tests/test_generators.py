"""Unit tests for the graph generators."""

import pytest

from graph_algorithms.generators import (
    complete,
    crown,
    cycle,
    path,
    random_graph,
    random_sparse_graph,
    random_tree,
    star,
)


@pytest.mark.parametrize(
    "graph, n, m",
    [(star(5), 6, 5), (path(6), 6, 5), (cycle(6), 6, 6), (complete(6), 6, 15), (crown(4), 8, 12)],
    ids=["star", "path", "cycle", "complete", "crown"],
)
def test_small_graph_sizes(graph, n, m):
    """Each fixed graph has the expected number of vertices and edges."""
    assert (graph.num_vertices, graph.num_edges) == (n, m)


def test_crown_is_bipartite_by_parity():
    """Crown edges only join an even vertex (side a) to an odd one (side b)."""
    assert all((u + v) % 2 == 1 for u, v in crown(5).edges())


def test_random_tree_is_a_tree():
    """A random tree has n - 1 edges and every vertex is reachable from 0."""
    graph = random_tree(50, seed=3)
    assert graph.num_edges == 49
    seen, stack = {0}, [0]
    while stack:
        for u in graph.neighbors(stack.pop()):
            if u not in seen:
                seen.add(u)
                stack.append(u)
    assert len(seen) == 50


def test_random_graph_same_seed_same_graph():
    """G(n, p) with the same seed gives the same edges; another seed differs."""
    first = sorted(random_graph(30, 0.3, seed=1).edges())
    assert first == sorted(random_graph(30, 0.3, seed=1).edges())
    assert first != sorted(random_graph(30, 0.3, seed=2).edges())


def test_random_graph_extreme_probabilities():
    """p = 0 gives no edges and p = 1 gives the complete graph."""
    assert random_graph(10, 0.0, seed=0).num_edges == 0
    assert random_graph(10, 1.0, seed=0).num_edges == 45


@pytest.mark.parametrize("n, avg_degree", [(100, 4), (1000, 10)])
def test_random_sparse_graph_edge_count(n, avg_degree):
    """G(n, m) has exactly n * avg_degree / 2 edges and is reproducible."""
    graph = random_sparse_graph(n, avg_degree, seed=n)
    assert graph.num_edges == n * avg_degree // 2
    assert sorted(graph.edges()) == sorted(random_sparse_graph(n, avg_degree, seed=n).edges())
